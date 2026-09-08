#!/usr/bin/env python3
"""L3b Phase A -- read-only worklist report, NOT the online-learning autopilot.

D-qNg64-plan-1's L3b design doc proposed a full "online-learning loop" (live detection ->
automatic real-kernel sweep -> promotion_writeback.py -> QWEN_MOE_PROMOTION_FILE_NQ). An Opus
adversarial review of that design (2026-09-08) found the automation steps (3/4/5 in the
original proposal) each individually unsafe today:
  - fetch_prior_points_by_event()'s sorted(set(...)) + dict() collapse lets a simulated PASS
    silently overwrite a real-kernel FAIL on a same-(n) collision (verified by execution).
  - quant_search_n.py's "live" oracle only ever drives the SIMULATED F32-override path
    (QWEN_MOE_ATTRIB_SIM_PATH), never the real qNg64 kernel (QWEN_MOE_ATTRIB_SIM_QN) --
    despite the name, "live" means "real engine invocation", not "real kernel".
  - req numbering is ambiguous across manifest chunks (already caused two silent, non-erroring
    provenance corruptions in this project, D-d5-27 and its repeat) -- there is currently no
    way to go from a live-serving (req,pos) to a safely-resolvable sweep manifest without a
    manual verification step this design would automate away.
  - promotion_writeback.py's --out write TRUNCATES the promotion file to only the targets
    passed in a given run, rather than merging -- re-running it for one (role,layer) would
    silently un-promote every other already-promoted target.
See .claude/history/2026-09-08_l3b-design.md and this session's own conversation history for the
full review. None of that is fixed here. This script builds ONLY what the review recommended as
the single safe, high-value thing to build first ("Phase A"): a read-only report of what live
serving has actually been detecting, so the still-open benefit-metric question (does qNg64
promotion actually help broadly, D-qNg64-4/7's "preview, not proof") has real, ranked data to
work from -- with NO engine runs, NO writes to Supabase, NO changes to QWEN_MOE_PROMOTION_FILE_NQ,
NO changes to qwen_infer.c or the SQL schema.

Usage:
    python3 tools/promotion_controller.py --report <jsonl_path> [<jsonl_path> ...] [--json out.json]

Reads local JSONL attribution logs (the same file format d4_supabase_push.py already tracks --
this script does NOT touch that script's .pushed_offset cursor, and does not use a cursor of its
own: this is a cheap, read-only, idempotent report, and re-scanning whole files every run is
simpler and safer than adding new cursor-state to maintain for something this inexpensive. In
this project's actual practice (checked directly, 2026-09-08), there is no single canonical
QWEN_MOE_NEARTIE_EVENTS_LOG path in continuous use -- every real test round pointed at its own
scratch file (/private/tmp/step6/*.jsonl, step7/*.jsonl, step9/*.jsonl, etc.) -- so this script
takes an explicit list/glob of paths rather than assuming one file.

Parses only "kind":"attribution" lines directly -- an earlier draft of this design assumed
attribution lines needed to be matched to a preceding "kind":"event" line to recover req/pos,
but the Opus review found that assumption backwards on two counts: (a) moe_neartie_maybe_correct()
is called BEFORE moe_neartie_maybe_log() in both emit loops (qwen_infer.c:7053/7073), so
attribution lines precede rather than follow their event line; (b) the attribution line already
carries req/pos/corrected_argmax directly (qwen_infer.c:6272-6277) -- no reconstruction needed at
all, event-line adjacency was a red herring.
"""
import argparse
import glob as globmod
import json
import subprocess
import sys
from collections import defaultdict

# Events independently confirmed by this project's own manual reproduction-check discipline
# (RESULTS.md's own "confirmed single-flip" language) -- see report_manifest_status() below for
# exactly what confidence level each entry gets and why. Keep this list SMALL and conservative;
# it is not this script's job to guess at manifest resolution (that's explicitly Phase B/C
# territory per the Opus review) -- when in doubt, report "unknown", never a guess.
KNOWN_VERIFIED_EVENTS = {
    # (model, pos): (confidence, note)
    # p60 -- this session's own D-qNg64-1/2/3 rounds used this exact (req=0,pos=14) event
    # repeatedly, cross-checked against a known-good manifest each time. Full confidence.
    ("deepseek-v2-lite", 14): ("VERIFIED (req=0 confirmed by direct reuse across D-qNg64-1/2/3)",
                                "p60"),
    # p105/p155 -- peer session's "redo flip hunt, single-flip discipline" round (commits
    # c08567d/dde405e) verified these as real single-flip events via the mandatory
    # reproduction check, but this script has not independently re-derived which req value (or
    # manifest chunk) they resolve to -- report as peer-verified-but-req-unconfirmed, a real,
    # distinct confidence tier from "VERIFIED", not silently upgraded to it.
    ("deepseek-v2-lite", 13): ("PEER-VERIFIED, req not independently confirmed by this script",
                                "p105"),
    ("deepseek-v2-lite", 12): ("PEER-VERIFIED, req not independently confirmed by this script",
                                "p155"),
}


def load_attributions(paths):
    """Parse every kind=="attribution" line from the given JSONL files. Skips unparseable lines
    with a warning rather than crashing (the Opus review flagged this project's existing
    d4_supabase_push.py/quant_sim_analyze.py precedents as NOT doing this -- an uncaught
    json.loads exception on a live-writer's trailing partial line is a real, documented hazard
    for anything that keeps state across runs; this script keeps no state, so the only cost of
    a bad line here is one skipped row, not a corrupted cursor)."""
    rows = []
    for pattern in paths:
        for path in sorted(globmod.glob(pattern)) or [pattern]:
            try:
                with open(path) as f:
                    for lineno, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"WARN: {path}:{lineno}: unparseable line, skipped", file=sys.stderr)
                            continue
                        if row.get("kind") == "attribution":
                            row["_source_file"] = path
                            rows.append(row)
            except FileNotFoundError:
                print(f"WARN: {path}: not found, skipped", file=sys.stderr)
    return rows


def group_by_triple(rows):
    """{(model,corpus,role,layer): {"count": int, "events": {(req,pos): [corrected_argmax,...]}}}"""
    out = defaultdict(lambda: {"count": 0, "events": defaultdict(list)})
    for r in rows:
        key = (r["model"], r["corpus"], r["role"], r["layer"])
        out[key]["count"] += 1
        out[key]["events"][(r["req"], r["pos"])].append(r.get("corrected_argmax"))
    return out


def query_sweep_rows_exist(pat, model, role, layer):
    """Read-only SELECT via the Supabase Management API (see
    .claude/memory/reference_supabase_management_api_access.md). Returns row count for this
    (model,role,layer) in moe_quant_sweep_results, or None on any query failure (report
    "unknown" rather than assuming zero on a transient error)."""
    ref = "btdjbfgqzglucifcnuoc"
    q = (f"select count(*) as n from moe_quant_sweep_results "
         f"where model='{model}' and role='{role}' and layer={layer};")
    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST",
             f"https://api.supabase.com/v1/projects/{ref}/database/query",
             "-H", f"Authorization: Bearer {pat}",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"query": q})],
            capture_output=True, text=True, timeout=20,
        )
        data = json.loads(result.stdout)
        return data[0]["n"]
    except Exception as e:
        print(f"WARN: sweep-row query failed for {model}/{role}/L{layer}: {e}", file=sys.stderr)
        return None


def manifest_status(model, pos):
    if (model, pos) in KNOWN_VERIFIED_EVENTS:
        note, label = KNOWN_VERIFIED_EVENTS[(model, pos)]
        return f"{note} [{label}]"
    return "unknown / needs manual verification before any real sweep"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true",
                    help="explicit marker: this is the only mode this script has (Phase A, read-only). "
                         "Present so a future --apply (Phase B/C, not built here) reads as a deliberate "
                         "opt-in, not a typo away from the safe default.")
    ap.add_argument("paths", nargs="+", help="JSONL log file(s) or glob pattern(s)")
    ap.add_argument("--json", help="also write the worklist to this JSON path")
    ap.add_argument("--pat-env", default=None,
                     help="shell var name already holding the Management API PAT (skip live Supabase check if unset)")
    args = ap.parse_args()

    rows = load_attributions(args.paths)
    if not rows:
        print("No attribution rows found in the given path(s). Nothing to report.", file=sys.stderr)
        sys.exit(1)

    grouped = group_by_triple(rows)

    import os
    pat = os.environ.get(args.pat_env) if args.pat_env else None

    worklist = []
    for (model, corpus, role, layer), info in grouped.items():
        distinct_events = sorted(info["events"].keys())
        n_sweep_rows = query_sweep_rows_exist(pat, model, role, layer) if pat else None
        manifests = {f"req={r} pos={p}": manifest_status(model, p) for (r, p) in distinct_events}
        worklist.append({
            "model": model, "corpus": corpus, "role": role, "layer": layer,
            "attribution_count": info["count"],
            "distinct_events": [f"req={r} pos={p}" for (r, p) in distinct_events],
            "n_distinct_events": len(distinct_events),
            "sweep_rows_exist": (
                "unknown (no --pat-env given)" if n_sweep_rows is None and pat is None else
                "query failed" if n_sweep_rows is None else
                f"{n_sweep_rows} rows exist, provenance unknown (table predates a source column)"
                if n_sweep_rows > 0 else "0 rows -- never swept"
            ),
            "manifest_status": manifests,
        })

    worklist.sort(key=lambda w: w["attribution_count"], reverse=True)

    print(f"{'role':<24} {'layer':>5} {'attrib_count':>12} {'events':>7}  sweep_rows")
    print("-" * 80)
    for w in worklist:
        print(f"{w['role']:<24} {w['layer']:>5} {w['attribution_count']:>12} "
              f"{w['n_distinct_events']:>7}  {w['sweep_rows_exist']}")

    print(f"\n{len(worklist)} distinct (model,corpus,role,layer) triples, "
          f"{len(rows)} total attribution rows from {len(args.paths)} path pattern(s)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(worklist, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
