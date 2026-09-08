#!/usr/bin/env python3
"""D-qNg64-3 (L3a): the "small script" qwen_infer.c:6250-6260's own comment anticipated --
"today a human [writes QWEN_MOE_PROMOTION_FILE], eventually a small script reading
moe_role_precision_state." This is that script's qNg64 counterpart: aggregates ALL known events
for a (model,role,layer) target into one safe n, across ALL corpora, and writes a
QWEN_MOE_PROMOTION_FILE_NQ-format file ("<role> <layer> <n>" per line) the engine's
moe_promotion_nq_init() reads at startup.

Not a new decision algorithm -- calls quant_search_n.py's already-validated
fetch_prior_points_by_event()/suffix_closed_knee() and aggregates their output. Two-level max:

  1. Within one corpus: a target's safe n for that corpus is the MAX suffix-closed knee across
     EVERY event recorded for it in that corpus (L2 gap #3 -- "adaptive" means safe against every
     known event, not just the one it happened to be tested against; taking the max, not picking
     one, is what makes a single n safe for all of them simultaneously).
  2. Across corpora: the final n is the MAX across every corpus's per-corpus safe n (plan point 6
     -- the promotion file is global, it can't know at load time which corpus a live request will
     resemble, so it has to be safe for all of them; suffix-closure means anything at or above a
     corpus's own knee is safe for that corpus, so the max across corpora is a valid, if
     conservative, upper bound that's safe everywhere at once).

If ANY event/corpus has no safe n within the tested ladder (suffix_closed_knee returns None), the
whole target is UNSAFE to promote and is skipped with a clear reason -- never silently dropped from
the aggregation, since dropping it would mean promoting a "safe" n that was never actually checked
against that event.

Usage:
    python3 tools/promotion_writeback.py --model deepseek-v2-lite --targets role1:layer1,role2:layer2 --out /path/to/promotion_nq_file.txt
    python3 tools/promotion_writeback.py --model deepseek-v2-lite --targets-file targets.txt --out ...

Requires QWEN_SUPABASE_URL/QWEN_SUPABASE_KEY (same as quant_search_n.py --live) -- this script does
not itself drive new engine runs, it only aggregates ALREADY-COLLECTED moe_quant_sweep_results rows.
Collecting fresh real-kernel data for a target is quant_search_n.py --live's job (via
QWEN_MOE_ATTRIB_SIM_QN, D-qNg64-2), run separately before this script has anything to aggregate for
a new target.
"""
import argparse
import sys

from quant_search_n import fetch_prior_points_by_event, suffix_closed_knee


def target_safe_n(model, role, layer):
    """Returns (n, detail) where n is the aggregated safe n (None if unsafe/no data) and detail
    is a dict with the per-corpus/per-event breakdown, for logging/audit -- never hide how a
    number was derived."""
    events_by_corpus = fetch_prior_points_by_event(model, role, layer)
    if not events_by_corpus:
        return None, {"reason": "no prior sweep data"}

    per_corpus = {}
    for corpus, events in events_by_corpus.items():
        per_event_knees = {}
        for ev, pts in events.items():
            known = dict(pts)
            per_event_knees[ev] = suffix_closed_knee(known)
        if any(k is None for k in per_event_knees.values()):
            unsafe_events = [ev for ev, k in per_event_knees.items() if k is None]
            per_corpus[corpus] = {"safe_n": None, "per_event": per_event_knees, "unsafe_events": unsafe_events}
        else:
            per_corpus[corpus] = {"safe_n": max(per_event_knees.values()), "per_event": per_event_knees}

    if any(c["safe_n"] is None for c in per_corpus.values()):
        return None, {"reason": "at least one corpus has an event with no safe n in the tested ladder", "per_corpus": per_corpus}

    final_n = max(c["safe_n"] for c in per_corpus.values())
    return final_n, {"per_corpus": per_corpus, "final_n": final_n}


def parse_targets(args):
    targets = []
    if args.targets:
        for t in args.targets.split(","):
            role, layer = t.split(":")
            targets.append((role, int(layer)))
    if args.targets_file:
        with open(args.targets_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                role, layer = line.split()
                targets.append((role, int(layer)))
    return targets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", help='comma-separated "role:layer" pairs')
    ap.add_argument("--targets-file", help='file of "role layer" lines, one per target')
    ap.add_argument("--out", required=True, help="output QWEN_MOE_PROMOTION_FILE_NQ path")
    ap.add_argument("--dry-run", action="store_true", help="print what would be written, don't write --out")
    args = ap.parse_args()

    targets = parse_targets(args)
    if not targets:
        print("no targets given -- pass --targets or --targets-file", file=sys.stderr)
        sys.exit(1)

    lines = []
    for role, layer in targets:
        n, detail = target_safe_n(args.model, role, layer)
        if n is None:
            print(f"SKIP {role}/L{layer}: {detail.get('reason', 'unsafe')}", file=sys.stderr)
            continue
        print(f"{role}/L{layer}: n={n}  ({detail})")
        lines.append(f"{role} {layer} {n}")

    if not lines:
        print("no targets produced a safe n -- nothing to write", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("--- dry run, would write ---")
        print("\n".join(lines))
        return

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} promotion(s) to {args.out}")


if __name__ == "__main__":
    main()
