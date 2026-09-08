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
import shlex
import subprocess
import sys
import time

from quant_search_n import fetch_prior_points_by_event, suffix_closed_knee, event_source_contradiction

# D-qNg64-12 (second Opus review of L3b Phase C follow-up): the file moe_promotion_nq_init()
# actually reads is on the ENGINE's host (bob), not wherever this script runs. The original
# --out wrote a LOCAL file and truncated it to only the targets passed in one run -- fixing "it
# truncates" locally was fixing the bug in a place that doesn't matter; re-running for one target
# would still silently un-promote every other already-promoted target on the real, remote file.
# This is the actual fix: read bob's live file, upsert only the targets in THIS run's scope
# (replaced if safe, REMOVED if now unsafe -- never left stale), leave every other line
# untouched, write back atomically (remote tempfile + mv, since the engine fscanf()s this file
# at startup and a torn write would silently truncate the whole promotion set).
DEFAULT_PROMOTION_FILE = "/private/tmp/qng64_ctl/promotion_nq_live.txt"

# D-qNg64-11: the only bit-widths the qNg64 bit-plane decoder actually supports for a real
# deployment promotion (matches qwen_infer.c's moe_promotion_nq_init()/moe_register_hi_role()
# hard allowlist, commit c36aadc). suffix_closed_knee() must be called with this explicit ladder
# for real-sourced data -- NOT the sim data's historical n=2..16 range, which includes values
# (4, 8, 9-16) this path cannot decode at all.
REAL_LADDER = (5, 6, 7)

TRUSTED_SOURCE = "qng64_real"


def target_safe_n(model, role, layer):
    """Returns (n, detail) where n is the aggregated safe n (None if unsafe/no data) and detail
    is a dict with the per-corpus/per-event breakdown, for logging/audit -- never hide how a
    number was derived.

    D-qNg64-11 (second Opus review of L3b Phase C, 2026-09-08): only source='qng64_real' rows
    are trusted for an actual deployment decision -- source='sim' data is never mixed in, at any
    level. This isn't a preference, it's this project's own established conclusion
    (D-qNg64-plan-1/2/5/6: "F32-override 경유는 전부 배선 스모크테스트일 뿐 배포 결정 근거로
    쓰지 않는다") finally enforced in code rather than only in memory. A target with real data
    for some events but not others is refused, not partially trusted -- see per-corpus loop
    below. Real, concrete consequence verified against this repo's own numbers: the OLD
    (source-blind) version of this function emitted `kv_b_proj/L9 n=4` and
    `kv_a_proj_with_mqa/L3 n=11` from mixed real+sim data -- n=4 is a measured real-kernel FAIL
    for kv_b_proj/L9 (D-qNg64-2), and n=11 is not decodable by the qNg64 kernel at all (outside
    REAL_LADDER, would have hit qwen_infer.c's new c36aadc FATAL if ever actually promoted). This
    version cannot reproduce either output."""
    events_by_corpus = fetch_prior_points_by_event(model, role, layer)
    if not events_by_corpus:
        return None, {"reason": "no prior sweep data"}

    per_corpus = {}
    any_real_data_anywhere = False
    for corpus, events in events_by_corpus.items():
        per_event_knees = {}
        unsafe_events = []
        no_real_data_events = []
        for ev, by_source in events.items():
            real_pts = by_source.get(TRUSTED_SOURCE, [])
            if not real_pts:
                no_real_data_events.append(ev)
                continue
            any_real_data_anywhere = True
            contradictions = event_source_contradiction(real_pts)
            if contradictions:
                # A retried/duplicate push landed two different results for the same n within
                # the TRUSTED source -- never resolve this by sort order, treat as unsafe and
                # say exactly why (this table has no unique constraint; a duplicate push with a
                # different outcome is a real, if rare, possible event, not just theoretical).
                per_event_knees[ev] = None
                unsafe_events.append(f"{ev}: contradictory real-kernel results at n={sorted(contradictions)}")
                continue
            per_event_knees[ev] = suffix_closed_knee(dict(real_pts), ladder=REAL_LADDER)
            if per_event_knees[ev] is None:
                unsafe_events.append(f"{ev}: no safe n in {REAL_LADDER} (real-kernel data)")

        if no_real_data_events:
            # A target where SOME events have real-kernel data and others don't is refused
            # wholesale, not partially trusted -- promoting on a subset of known events isn't
            # "safe against every known event", it's safe against the ones that happened to be
            # swept. D-qNg64-plan-1's write-back principle (max across ALL known events) only
            # holds if "all known events" actually means all of them.
            per_corpus[corpus] = {
                "safe_n": None, "per_event": per_event_knees,
                "reason": f"{len(no_real_data_events)} event(s) have no source='{TRUSTED_SOURCE}' data yet: {no_real_data_events}",
            }
            continue

        if unsafe_events or any(k is None for k in per_event_knees.values()):
            per_corpus[corpus] = {"safe_n": None, "per_event": per_event_knees, "unsafe_events": unsafe_events}
        else:
            per_corpus[corpus] = {"safe_n": max(per_event_knees.values()), "per_event": per_event_knees}

    if not any_real_data_anywhere:
        return None, {"reason": f"no source='{TRUSTED_SOURCE}' rows exist yet for this target -- "
                                 f"sim-only data is never used for a deployment decision (D-qNg64-plan-1/2/5/6)"}

    if any(c["safe_n"] is None for c in per_corpus.values()):
        return None, {"reason": "at least one corpus is not fully real-kernel-verified or has an unsafe event", "per_corpus": per_corpus}

    final_n = max(c["safe_n"] for c in per_corpus.values())
    return final_n, {"per_corpus": per_corpus, "final_n": final_n}


def read_remote_promotion_file(ssh_host, remote_path):
    """Returns {(role, layer): n} from the LIVE file on ssh_host, or {} if it doesn't exist yet
    (a fresh install -- not an error). Raises RuntimeError on any other failure (a transient ssh
    hiccup must never be silently treated as "file is empty, safe to overwrite" -- that would
    itself reintroduce the truncation bug this function exists to prevent)."""
    q_path = shlex.quote(remote_path)
    check = subprocess.run(["ssh", ssh_host, f"test -f {q_path}"])
    if check.returncode != 0:
        return {}
    cat = subprocess.run(["ssh", ssh_host, f"cat {q_path}"], capture_output=True, text=True, timeout=30)
    if cat.returncode != 0:
        raise RuntimeError(f"could not read existing promotion file on {ssh_host}:{remote_path}: {cat.stderr}")
    existing = {}
    # moe_promotion_nq_init() parses this with fscanf(f, "%63s %d %d", ...) -- a plain
    # whitespace-delimited token stream, NOT one-entry-per-line with '#' comments (a '#'-led
    # token would be read as a role name, fail moe_attrib_role_from_name(), and FATAL the
    # engine). Match that exactly: split on all whitespace, group in 3s, no comment handling.
    tokens = cat.stdout.split()
    if len(tokens) % 3 != 0:
        raise RuntimeError(f"existing promotion file on {ssh_host}:{remote_path} has {len(tokens)} "
                            f"tokens, not a multiple of 3 -- malformed, refusing to guess how to parse it")
    for i in range(0, len(tokens), 3):
        role, layer_s, n_s = tokens[i], tokens[i + 1], tokens[i + 2]
        existing[(role, int(layer_s))] = int(n_s)
    return existing


def write_remote_promotion_file_atomic(ssh_host, remote_path, merged):
    """Writes {(role, layer): n} to ssh_host:remote_path atomically -- a remote tempfile in the
    SAME directory (so the final `mv` is a same-filesystem rename, not a cross-device copy) then
    `mv` over the real path. moe_promotion_nq_init() fscanf()s this file at process startup; a
    direct (non-atomic) overwrite leaves a real window where a process starting mid-write sees a
    truncated or empty file and silently promotes nothing. Sorted output for a stable, diffable
    file across runs."""
    lines = [f"{role} {layer} {n}" for (role, layer), n in sorted(merged.items())]
    content = ("\n".join(lines) + "\n") if lines else ""
    remote_dir = remote_path.rsplit("/", 1)[0]
    tmp_path = f"{remote_path}.tmp.{int(time.time() * 1000)}"
    q_dir, q_tmp, q_final = shlex.quote(remote_dir), shlex.quote(tmp_path), shlex.quote(remote_path)
    write_cmd = (
        f"mkdir -p {q_dir} && cat > {q_tmp} && mv {q_tmp} {q_final}"
    )
    result = subprocess.run(["ssh", ssh_host, write_cmd], input=content, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"failed to write promotion file on {ssh_host}:{remote_path}: {result.stderr}")
    return len(lines)


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
    ap.add_argument("--ssh-host", default="bob", help="host that runs the engine and owns the live promotion file")
    ap.add_argument("--promotion-file", default=DEFAULT_PROMOTION_FILE,
                     help=f"REMOTE path on --ssh-host of the live QWEN_MOE_PROMOTION_FILE_NQ "
                          f"(default: {DEFAULT_PROMOTION_FILE})")
    ap.add_argument("--out", help="optional: also write a LOCAL copy of the full merged file here, for audit/diff")
    ap.add_argument("--dry-run", action="store_true", help="print the upsert diff, touch nothing on --ssh-host")
    args = ap.parse_args()

    targets = parse_targets(args)
    if not targets:
        print("no targets given -- pass --targets or --targets-file", file=sys.stderr)
        sys.exit(1)

    # D-qNg64-12: read the LIVE remote file first -- every target named on this run's CLI is "in
    # scope" and gets its line replaced (safe) or REMOVED (now unsafe); every (role,layer) NOT
    # named on this run's CLI is untouched, whatever its current value. This is what makes
    # re-running for one target safe to do repeatedly without silently un-promoting the rest.
    try:
        existing = read_remote_promotion_file(args.ssh_host, args.promotion_file)
    except RuntimeError as e:
        print(f"FATAL: cannot safely proceed without reading the existing live file first: {e}", file=sys.stderr)
        sys.exit(1)

    merged = dict(existing)
    removed, updated, unchanged, added = [], [], [], []
    for role, layer in targets:
        n, detail = target_safe_n(args.model, role, layer)
        key = (role, layer)
        old_n = existing.get(key)
        if n is None:
            print(f"SKIP {role}/L{layer}: {detail.get('reason', 'unsafe')}", file=sys.stderr)
            if key in merged:
                del merged[key]
                removed.append((role, layer, old_n))
            continue
        print(f"{role}/L{layer}: n={n}  ({detail})")
        merged[key] = n
        if old_n is None:
            added.append((role, layer, n))
        elif old_n != n:
            updated.append((role, layer, old_n, n))
        else:
            unchanged.append((role, layer, n))

    print(f"\n--- upsert summary (scope: {len(targets)} target(s) this run, "
          f"{len(existing)} pre-existing line(s) on {args.ssh_host}:{args.promotion_file}) ---")
    for role, layer, n in added:
        print(f"  ADD    {role}/L{layer} -> n={n}")
    for role, layer, old_n, n in updated:
        print(f"  UPDATE {role}/L{layer}: n={old_n} -> n={n}")
    for role, layer, n in unchanged:
        print(f"  SAME   {role}/L{layer}: n={n} (no change)")
    for role, layer, old_n in removed:
        print(f"  REMOVE {role}/L{layer} (was n={old_n}, now unsafe -- see SKIP reason above)")
    out_of_scope = len(merged) - len(added) - len(updated) - len(unchanged)
    print(f"  ({out_of_scope} out-of-scope line(s) preserved untouched)")

    if not added and not updated and not removed:
        print("no changes -- not writing", file=sys.stderr)
        return

    if args.dry_run:
        print("\n--- dry run: NOT writing to the live file ---")
        return

    n_written = write_remote_promotion_file_atomic(args.ssh_host, args.promotion_file, merged)
    print(f"\nwrote {n_written} total promotion(s) to {args.ssh_host}:{args.promotion_file} (atomic)")

    if args.out:
        lines = [f"{role} {layer} {n}" for (role, layer), n in sorted(merged.items())]
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"also wrote a local audit copy to {args.out}")


if __name__ == "__main__":
    main()
