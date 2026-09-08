#!/usr/bin/env python3
"""ROI-G Phase 2: classification-gated arbitrary-n search.

The Opus plan's original bidirectional/ablation-priority bisection (its own
Phase 2, section 3.3) assumed monotonicity in n universally. This session's
real 6-target sweep found 4/6 targets violate it (all attention-family) --
bisecting blindly would have silently produced wrong "minimal n" answers
for those 4. This driver decides per-target, from REAL prior sweep data
(not an assumption), whether bisection is safe:

  - "exhaustive_required": some already-tested corpus showed a violation
    for this exact (model,role,layer) -- bisection is unsafe, full scan
    only (matches the Opus plan's own Sec 3.5 fallback).
  - "bisection_candidate": exactly one corpus tested, and it was clean --
    plausibly safe, but unconfirmed across corpora yet. Used with a
    warning, not blind trust.
  - "bisection": 2+ corpora tested, ALL clean -- confirmed safe, full
    bisection speed.
  - "unknown": no prior data at all -- exhaustive (safe default).

Two "test oracle" backends:
  - historical: looks up an already-completed sweep's real results (no new
    engine calls) -- used to VALIDATE this driver against the 6 known
    WikiText-2 targets before ever trusting it on new data.
  - live: SSHes to a worker host, reuses a pre-built override safetensors
    file per n (from quant_sim_n.py) plus a single-request isolation
    manifest and a HI_COMBOS restriction file -- same mechanism Step 6's
    own sweep scripts used -- and parses the real engine stdout for a
    'hit req=.. pos=.. role=.. layer=..' line. classify() is fed from a
    live Supabase query (fetch_prior_points_by_event), same real-data-only
    discipline as historical validation.

  First real --live round-trip (2026-09-07, q_proj/L1, wikitext-2-fullext,
  req=0/pos=8, reusing /private/tmp/step6's already-built sim override
  files): reproduced the exact known ground truth (fail only at n=4, pass
  everywhere else) via a genuinely fresh engine run, not a cached read --
  confirms the live oracle is wired correctly end-to-end. Two real bugs
  were found and fixed getting there, both worth recording since either
  one would have silently produced a wrong "knee": (1) the remote command
  needs `cd` into the engine's working directory first (it loads
  `weights_moe/arch_config_moe.txt` via a relative path -- omitting the
  cd doesn't error, it just fails to find real near-tie events, so every
  n silently reads as fail); (2) picking the wrong same-req manifest file
  -- `manifest_wt2_req32.txt` (chunk_aa, wrong) vs `manifest_wt2_req32_ac.
  txt` (chunk_ac, correct) -- reproduces the exact "two wrong guesses
  before the right one" trap D-d5-27 already documented for this same
  req32/pos8 event; the failure mode isn't a crash, it's a real-looking
  but wrong near-tie at a different pos, so the sim override never gets
  exercised. Neither failure mode looks like an error in the tool's own
  output -- both need checking the actual engine log for the expected
  hit line, not just trusting a clean exit code.

  As of this check, every (model,role,layer) with real sweep data in
  Supabase classifies exhaustive_required (a violation has been observed
  in every tested corpus, for every tested target so far) -- so in
  practice --live currently always runs the full 15-value scan. That is
  the correct, conservative behavior given the data, not a bug: bisection
  mode is wired and will activate automatically the first time some target
  is ever found clean across 2+ corpora, but that has not happened yet.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

GROUP = 64
LADDER = list(range(2, 17))


def eff_bpw(n):
    return n + 32.0 / (GROUP * n)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _event_monotonic(pts):
    """pts: sorted [(n, pass_bool), ...] for ONE (corpus,req,pos) event.
    True iff no pass->fail regression as n increases."""
    seen_pass = False
    for n, p in sorted(pts):
        if p:
            seen_pass = True
        elif seen_pass:
            return False
    return True


def suffix_closed_knee(known_results, ladder=None):
    """known_results: {n: pass_bool} for ONE event. Returns the smallest n such
    that EVERY tested n' >= n also passes (Opus B1 fix -- 'first n that passes'
    is only equal to this under monotonicity, which is violated 58-83% of the
    time per this project's own Step 6 / ROI-G findings; a real counterexample
    -- kv_b_proj L8, passes at n=3, fails at n=4, recovers n=5+ -- is recorded
    in RESULTS.md. Returning the first-pass n there would deploy n=3, BELOW the
    int4 production base, on the strength of a rounding coincidence.
    Returns None if no n in the tested range satisfies this (a real, valid
    outcome -- report it, don't force a fallback)."""
    tested = sorted(known_results.keys()) if ladder is None else sorted(n for n in ladder if n in known_results)
    for n in tested:
        if all(known_results.get(n2, False) for n2 in tested if n2 >= n):
            return n
    return None


def classify(events_by_corpus):
    """events_by_corpus: {corpus: {(req,pos): [(n, pass_bool), ...]}} for ONE
    (model,role,layer) -- event-scoped (Opus B3 fix: the old
    {corpus: [(n,pass),...]} shape merged every event tested against this
    target into one pass/fail set. Two individually-monotone events with
    different knees can manufacture a 'violation' neither one actually
    exhibits -- e.g. event A passes n>=4, event B passes n>=8: merged and
    de-duped, (4,False) from B can sort before (4,True) from A and the merged
    sequence reads as violating even though both events are individually
    clean. A real instance of this exact failure mode -- one override
    perturbing a second nearby event's own resolution -- was found and
    retracted in this repo's own sweep data (commit c3bcf81, 'multi-flip
    contamination'). Classify per event, then aggregate.
    Returns (mode, reason)."""
    if not events_by_corpus:
        return "unknown", "no prior sweep data for this target"

    clean_corpora = []
    violating_corpora = []
    for corpus, events in events_by_corpus.items():
        violated_events = [ev for ev, pts in events.items() if not _event_monotonic(pts)]
        (violating_corpora if violated_events else clean_corpora).append(
            corpus if not violated_events else f"{corpus}(events={sorted(violated_events)})")

    if violating_corpora:
        return "exhaustive_required", f"violation observed in corpus/corpora: {violating_corpora}"
    if len(clean_corpora) >= 2:
        return "bisection", f"{len(clean_corpora)} corpora all monotonic: {clean_corpora}"
    return "bisection_candidate", f"only 1 corpus tested so far ({clean_corpora[0]}), clean but unconfirmed cross-corpus"


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------

def bisection_search(test, n_min=2, n_src=16, budget=20):
    """Opus plan Sec 3.3 pseudocode verbatim: ablation frontier prioritized
    (larger stride, tested first each round), growth frontier fills in.
    Returns (knee, tests_run) where tests_run is [(n, pass_bool), ...].

    NOT DEPLOYMENT-SAFE (Opus B1 finding, 2026-09-08): a partial/bisecting scan
    never observes the full suffix above its reported knee, so it cannot verify
    suffix-closure even when classify() says "bisection" (2+ corpora clean) --
    clean-so-far is not proof the untested gaps stay clean. Use only for
    diagnostics/speed comparison against exhaustive_search(); every deployment
    decision must go through exhaustive_search()'s suffix-closed knee."""
    lo = n_min - 1   # highest proven-insufficient
    hi = n_src        # lowest proven-sufficient (n_src sufficient by definition)
    g = n_min          # growth frontier, walks up
    a = n_src - 1      # ablation frontier, walks down
    tests_run = []

    while lo + 1 < hi and budget > 0:
        w = hi - lo

        if a > lo:
            r = test(a)
            tests_run.append((a, r))
            budget -= 1
            if r:
                hi = min(hi, a)
                a = a - max(1, w // 2)
            else:
                lo = max(lo, a)
                a = a + 1
        if lo + 1 >= hi or budget == 0:
            break

        if g < hi:
            r = test(g)
            tests_run.append((g, r))
            budget -= 1
            if r:
                hi = min(hi, g)
            else:
                lo = max(lo, g)
                g = g + 1

        g = max(g, lo + 1)
        a = min(a, hi - 1)

    return hi, tests_run


def exhaustive_search(test, ladder=LADDER):
    """Runs the full ladder and returns the suffix-closed knee (Opus B1 fix --
    NOT the first n that happens to pass; see suffix_closed_knee()'s own
    docstring). This is the only search mode this project currently treats as
    deployment-safe (see bisection_search()'s docstring)."""
    tests_run = []
    results = {}
    for n in ladder:
        r = test(n)
        tests_run.append((n, r))
        results[n] = r
    knee = suffix_closed_knee(results, ladder)
    return knee, tests_run


# ---------------------------------------------------------------------------
# Historical oracle (validation mode -- no new engine calls)
# ---------------------------------------------------------------------------

def make_historical_oracle(known_results):
    """known_results: {n: pass_bool} for one target. Returns a test(n) callable
    that looks up the REAL already-collected result instead of calling the
    engine -- used to validate the search algorithm against ground truth."""
    def test(n):
        if n not in known_results:
            raise KeyError(f"no historical data for n={n} -- sweep didn't cover this value")
        return known_results[n]
    return test


# ---------------------------------------------------------------------------
# Supabase lookup (real prior sweep data, for live-mode classification)
# ---------------------------------------------------------------------------

def fetch_prior_points_by_event(model, role, layer):
    """Query moe_quant_sweep_results for every (n, pass) row already recorded
    for this exact (model, role, layer), grouped by corpus THEN by (req,pos)
    event (Opus B3 fix, 2026-09-08 -- the old fetch_prior_points_by_corpus()
    merged every event into one flat per-corpus list, which classify() could
    misread as a violation no single event exhibited; req/pos are real columns
    on this table, supabase_schema_d_roadmap4.sql:376-377, just previously
    unused here). Returns {corpus: {(req,pos): [(n, pass_bool), ...]}}."""
    url = os.environ.get("QWEN_SUPABASE_URL")
    key = os.environ.get("QWEN_SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY must be set for live mode")

    qs = (f"model=eq.{model}&role=eq.{role}&layer=eq.{layer}"
          f"&select=corpus,req,pos,n,pass")
    req_obj = urllib.request.Request(
        f"{url}/rest/v1/moe_quant_sweep_results?{qs}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req_obj, timeout=30) as resp:
        rows = json.loads(resp.read())

    events_by_corpus = {}
    for row in rows:
        ev = (row["req"], row["pos"])
        events_by_corpus.setdefault(row["corpus"], {}).setdefault(ev, []).append((row["n"], bool(row["pass"])))
    # de-dup identical (n, pass) pairs within one event (the table has no unique
    # constraint -- the same real event can be pushed more than once)
    for corpus in events_by_corpus:
        for ev in events_by_corpus[corpus]:
            events_by_corpus[corpus][ev] = sorted(set(events_by_corpus[corpus][ev]))
    return events_by_corpus


# ---------------------------------------------------------------------------
# Live oracle -- real engine calls via bob, same mechanism Step 6's sweep
# scripts already use (QWEN_MOE_ATTRIB_SIM_* override + a single-request
# isolation manifest + a HI_COMBOS file restricting attribution replay to
# just the target combo). Requires the per-n override safetensors files to
# already exist under sim_dir (built ahead of time by quant_sim_n.py, same
# as every prior round this session) -- this function does not generate
# them, it only drives the real inference + parses the real result.
# ---------------------------------------------------------------------------

def make_live_oracle(ssh_host, moe_base, bin_path, manifest, combo_path, sim_dir,
                      model, corpus, role, layer, req, pos,
                      events_log="/dev/null", safetensors_index=None, max_pos=19,
                      timeout=180, cwd="/Users/bob/vdsp_m4_bench"):
    """Returns test(n) -> bool, driving ONE real engine invocation per call
    over `ssh ssh_host`. pass/fail is read from the real stdout, not assumed:
    a 'hit req=<req> pos=<pos> role=<role> layer=<layer>' line means the
    correction fired at this n (pass); its absence means it didn't (fail).
    Raises if the sim override file for this n is missing -- no silent
    fallback to some other n's file."""
    def test(n):
        sim_path = f"{sim_dir}/sim_n{n}.safetensors"
        check = subprocess.run(["ssh", ssh_host, f"test -f {sim_path}"])
        if check.returncode != 0:
            raise FileNotFoundError(f"{ssh_host}:{sim_path} missing -- build it with quant_sim_n.py first")

        env = {
            "BOB_LOAD_OK": "1",
            "QWEN_MOE_BASE": moe_base,
            "QWEN_MOE_CBATCH": "1",
            "QWEN_MOE_CB_ONLINE": "1",
            "QWEN_MOE_CB_PROMPT_MANIFEST": manifest,
            "QWEN_MOE_CB_REQS": "1",
            "QWEN_MOE_NEARTIE_CORRECT": "1",
            "QWEN_MOE_NEARTIE_LOG": "1",
            "QWEN_MOE_NEARTIE_MODEL": model,
            "QWEN_MOE_NEARTIE_CORPUS": corpus,
            "QWEN_MOE_NEARTIE_EVENTS_LOG": events_log,
            "QWEN_MOE_ATTRIB": "1",
            "QWEN_MOE_ATTRIB_MAX_POS": str(max_pos),
            "QWEN_MOE_NEARTIE_HI_COMBOS": combo_path,
            "QWEN_MOE_ATTRIB_SIM_ROLE": role,
            "QWEN_MOE_ATTRIB_SIM_LAYER": str(layer),
            "QWEN_MOE_ATTRIB_SIM_PATH": sim_path,
        }
        if safetensors_index:
            env["QWEN_MOE_NEARTIE_CORRECT_SAFETENSORS"] = safetensors_index

        env_str = " ".join(f"{k}={v}" for k, v in env.items())
        remote_cmd = f"cd {cwd} && {env_str} {bin_path}"
        result = subprocess.run(
            ["ssh", ssh_host, remote_cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        out = result.stdout + result.stderr
        needle = f"hit req={req} pos={pos} role={role} layer={layer}"
        return needle in out
    return test


# ---------------------------------------------------------------------------
# Validation driver
# ---------------------------------------------------------------------------

def run_validation(sweep_tsv_path, model, corpus):
    rows = []
    with open(sweep_tsv_path) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            row["n"] = int(row["n"])
            row["pass"] = bool(int(row["pass"]))
            rows.append(row)

    targets = {}
    for row in rows:
        key = (row["role"], int(row["layer"]))
        targets.setdefault(key, {})[row["n"]] = row["pass"]

    print(f"{'target':<28} {'true_knee':>10} {'mode':>20} {'exhaustive':>10} {'bisect':>7} {'bisect_wrong':>12}")
    all_correct = True
    bisect_wrong_count = 0
    for (role, layer), known_results in sorted(targets.items()):
        # Opus B1: ground truth is the suffix-closed knee, not "first n that
        # passes" -- the two differ exactly on the targets B1 warns about
        # (e.g. kv_b_proj L8: passes n=3, fails n=4, recovers n=5+ -- "first
        # pass" reports 3, suffix-closed correctly reports 5).
        true_knee = suffix_closed_knee(known_results)

        # This TSV predates req/pos tracking and was built one event per
        # target by construction (the original 6-target WikiText-2 sweep) --
        # a single synthetic event key (0,0) reproduces classify()'s intended
        # per-corpus behavior for this file without inventing req/pos data
        # that was never collected.
        events_by_corpus = {corpus: {(0, 0): list(known_results.items())}}
        mode, reason = classify(events_by_corpus)

        oracle = make_historical_oracle(known_results)
        # Deployment-authoritative: always exhaustive, regardless of mode
        # (Opus B1 -- this project doesn't trust bisection for a deploy
        # decision even when classify() reports it as safe).
        exh_knee, exh_tests = exhaustive_search(oracle)
        correct = (exh_knee == true_knee)
        all_correct = all_correct and correct

        # Diagnostic only: what bisection WOULD have found, and whether it
        # would have been wrong -- quantifies why it's excluded from
        # deployment above, doesn't drive any decision here.
        bisect_knee, bisect_tests = bisection_search(oracle)
        bisect_wrong = (bisect_knee != true_knee)
        if bisect_wrong:
            bisect_wrong_count += 1

        target_name = f"{role}/L{layer}"
        print(f"{target_name:<28} {str(true_knee):>10} {mode:>20} {str(exh_knee):>10} {str(bisect_knee):>7} {str(bisect_wrong):>12}")
        if not correct:
            print(f"  -> MISMATCH (exhaustive): found {exh_knee}, true suffix-closed knee is {true_knee}. reason: {reason}")
        if bisect_wrong:
            print(f"  -> bisection would have found {bisect_knee} (WRONG vs true {true_knee}) -- diagnostic only, not used for deployment")

    print()
    print("ALL CORRECT (exhaustive, deployment-authoritative)" if all_correct else "SOME MISMATCHES (exhaustive) -- see above")
    print(f"bisection would have been wrong on {bisect_wrong_count}/{len(targets)} targets -- this is why B1 excludes it from deployment")
    return all_correct


def run_live(args):
    role, layer = args.role, args.layer
    events_by_corpus = fetch_prior_points_by_event(args.model, role, layer)
    mode, reason = classify(events_by_corpus)
    print(f"target={role}/L{layer}  mode={mode}  reason={reason}  (informational only, see below)")

    oracle = make_live_oracle(
        ssh_host=args.ssh_host, moe_base=args.moe_base, bin_path=args.bin,
        manifest=args.manifest, combo_path=args.combo, sim_dir=args.sim_dir,
        model=args.model, corpus=args.corpus, role=role, layer=layer,
        req=args.req, pos=args.pos, events_log=args.events_log,
        safetensors_index=args.safetensors_index, max_pos=args.max_pos, cwd=args.cwd,
    )
    # Opus B1: always exhaustive for the deployed knee, regardless of classify()'s
    # mode -- bisection cannot verify suffix-closure even when classify() reports
    # "bisection" (2+ corpora clean so far). Only search mode this project
    # currently trusts for a deployment decision.
    knee, tests_run = exhaustive_search(oracle)

    for n, p in tests_run:
        print(f"  n={n:2d}  pass={int(p)}")
    print(f"knee={knee} (suffix-closed)  n_tests={len(tests_run)} (naive exhaustive would be {len(LADDER)})")
    if knee is not None:
        print(f"  NOTE: verify knee={knee} against this role's actual production base-bits before promoting -- "
              f"a pass at low n is not itself evidence of adequate precision (see B1); L3a enforces n>=base_bits.")
    return knee, tests_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", metavar="TSV", help="validate against an already-completed sweep TSV (historical oracle, no new engine calls)")
    ap.add_argument("--live", action="store_true", help="run real engine calls via SSH (needs --role/--layer/--req/--pos/--manifest/--combo/--sim-dir/--bin/--moe-base)")
    ap.add_argument("--model", default="deepseek-v2-lite")
    ap.add_argument("--corpus", default="deepseek-moe4a-builtin-corpus")
    ap.add_argument("--role", help="live mode: target role, e.g. kv_b_proj")
    ap.add_argument("--layer", type=int, help="live mode: target layer")
    ap.add_argument("--req", type=int, default=0, help="live mode: isolated request id in the manifest (always 0 after isolation)")
    ap.add_argument("--pos", type=int, help="live mode: token position of the real near-tie event")
    ap.add_argument("--ssh-host", default="bob")
    ap.add_argument("--moe-base", default="/Users/bob/moe_base_deepseek")
    ap.add_argument("--bin", default="/tmp/qwen_quantsim3_bin")
    ap.add_argument("--cwd", default="/Users/bob/vdsp_m4_bench", help="live mode: remote working dir the binary must run from (relative asset paths like weights_moe/)")
    ap.add_argument("--manifest", help="live mode: path (on ssh-host) to the single-request isolation manifest")
    ap.add_argument("--combo", help="live mode: path (on ssh-host) to the QWEN_MOE_NEARTIE_HI_COMBOS file")
    ap.add_argument("--sim-dir", help="live mode: path (on ssh-host) to the dir with sim_n<N>.safetensors override files")
    ap.add_argument("--events-log", default="/dev/null")
    ap.add_argument("--safetensors-index", default="/Volumes/D50/deepseek_v2lite_bf16_safetensors/model.safetensors.index.json")
    ap.add_argument("--max-pos", type=int, default=19)
    args = ap.parse_args()

    if args.validate:
        ok = run_validation(args.validate, args.model, args.corpus)
        sys.exit(0 if ok else 1)
    elif args.live:
        missing = [f for f in ("role", "layer", "pos", "manifest", "combo", "sim_dir") if getattr(args, f) is None]
        if missing:
            print(f"--live requires: {', '.join('--' + m.replace('_','-') for m in missing)}", file=sys.stderr)
            sys.exit(1)
        run_live(args)
    else:
        print("Nothing to do -- pass --validate <tsv> or --live <target args>.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
