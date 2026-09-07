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
    live Supabase query (fetch_prior_points_by_corpus), same real-data-only
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

def classify(points_by_corpus):
    """points_by_corpus: {corpus: [(n, pass_bool), ...]} for ONE (model,role,layer).
    Returns (mode, reason)."""
    if not points_by_corpus:
        return "unknown", "no prior sweep data for this target"

    clean_corpora = []
    violating_corpora = []
    for corpus, points in points_by_corpus.items():
        pts = sorted(points)
        seen_pass = False
        violated = False
        for n, p in pts:
            if p:
                seen_pass = True
            elif seen_pass:
                violated = True
        (violating_corpora if violated else clean_corpora).append(corpus)

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
    Returns (knee, tests_run) where tests_run is [(n, pass_bool), ...]."""
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
    tests_run = []
    knee = None
    for n in ladder:
        r = test(n)
        tests_run.append((n, r))
        if r and knee is None:
            knee = n
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

def fetch_prior_points_by_corpus(model, role, layer):
    """Query moe_quant_sweep_results for every (n, pass) row already recorded
    for this exact (model, role, layer), grouped by corpus. Used to feed
    classify() in live mode -- same real-data-only discipline as historical
    validation, just read from Supabase instead of a local TSV."""
    url = os.environ.get("QWEN_SUPABASE_URL")
    key = os.environ.get("QWEN_SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY must be set for live mode")

    qs = (f"model=eq.{model}&role=eq.{role}&layer=eq.{layer}"
          f"&select=corpus,n,pass")
    req = urllib.request.Request(
        f"{url}/rest/v1/moe_quant_sweep_results?{qs}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        rows = json.loads(resp.read())

    points_by_corpus = {}
    for row in rows:
        points_by_corpus.setdefault(row["corpus"], []).append((row["n"], bool(row["pass"])))
    # de-dup identical (n, pass) pairs within a corpus (the table has no unique
    # constraint -- the same real event can be pushed more than once, e.g.
    # shared_down_proj/L26 pos=9 was; classify() only needs the distinct set)
    for corpus in points_by_corpus:
        points_by_corpus[corpus] = sorted(set(points_by_corpus[corpus]))
    return points_by_corpus


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

    print(f"{'target':<28} {'true_knee':>10} {'mode':>20} {'found_knee':>11} {'n_tests':>8} {'correct':>8}")
    all_correct = True
    for (role, layer), known_results in sorted(targets.items()):
        true_knee = min((n for n, p in known_results.items() if p), default=None)

        # classification: only this ONE corpus's data exists yet (pre-WikiText-103),
        # so every target should classify as bisection_candidate (clean) or
        # exhaustive_required (violated) -- never full "bisection" until a 2nd
        # corpus confirms. This IS the expected, correct behavior right now.
        points_by_corpus = {corpus: list(known_results.items())}
        mode, reason = classify(points_by_corpus)

        oracle = make_historical_oracle(known_results)
        if mode == "exhaustive_required" or mode == "unknown":
            found_knee, tests_run = exhaustive_search(oracle)
        else:  # bisection or bisection_candidate -- try bisection, but this is
               # exactly the risky case; validation checks whether it actually
               # gets the right answer despite the (candidate, not confirmed) label
            found_knee, tests_run = bisection_search(oracle)

        correct = (found_knee == true_knee)
        all_correct = all_correct and correct
        target_name = f"{role}/L{layer}"
        print(f"{target_name:<28} {str(true_knee):>10} {mode:>20} {str(found_knee):>11} {len(tests_run):>8} {str(correct):>8}")
        if not correct:
            print(f"  -> MISMATCH: bisection found {found_knee}, true knee (from exhaustive n=2..16) is {true_knee}. reason: {reason}")

    print()
    print("ALL CORRECT" if all_correct else "SOME MISMATCHES -- see above")
    return all_correct


def run_live(args):
    role, layer = args.role, args.layer
    points_by_corpus = fetch_prior_points_by_corpus(args.model, role, layer)
    mode, reason = classify(points_by_corpus)
    print(f"target={role}/L{layer}  mode={mode}  reason={reason}")

    oracle = make_live_oracle(
        ssh_host=args.ssh_host, moe_base=args.moe_base, bin_path=args.bin,
        manifest=args.manifest, combo_path=args.combo, sim_dir=args.sim_dir,
        model=args.model, corpus=args.corpus, role=role, layer=layer,
        req=args.req, pos=args.pos, events_log=args.events_log,
        safetensors_index=args.safetensors_index, max_pos=args.max_pos, cwd=args.cwd,
    )
    if mode in ("exhaustive_required", "unknown"):
        knee, tests_run = exhaustive_search(oracle)
    else:
        knee, tests_run = bisection_search(oracle)

    for n, p in tests_run:
        print(f"  n={n:2d}  pass={int(p)}")
    print(f"knee={knee}  n_tests={len(tests_run)} (naive exhaustive would be {len(LADDER)})")
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
