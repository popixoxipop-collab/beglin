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
import re
import time

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
    for this exact (model, role, layer), grouped by corpus, THEN by (req,pos)
    event, THEN by source (D-qNg64-11, 2026-09-08 -- second Opus review of the
    L3b Phase C design found the original per-event dict(pts) collapse lets a
    source='sim' PASS silently overwrite a source='qng64_real' FAIL for the
    same n on a same-event collision, sorted() putting False before True --
    verified by execution against real kv_b_proj/L9 data. Grouping by source
    here, one level deeper than the event, is what lets a caller keep sim and
    real data from ever being blended into one suffix_closed_knee() call).
    Returns {corpus: {(req,pos): {source: [(n, pass_bool), ...]}}}."""
    url = os.environ.get("QWEN_SUPABASE_URL")
    key = os.environ.get("QWEN_SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY must be set for live mode")

    qs = (f"model=eq.{model}&role=eq.{role}&layer=eq.{layer}"
          f"&select=corpus,req,pos,n,pass,source")
    req_obj = urllib.request.Request(
        f"{url}/rest/v1/moe_quant_sweep_results?{qs}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req_obj, timeout=30) as resp:
        rows = json.loads(resp.read())

    events_by_corpus = {}
    for row in rows:
        ev = (row["req"], row["pos"])
        src = row.get("source", "sim")  # pre-D-qNg64-9 rows predate the column; treat as sim (correct: they are)
        (events_by_corpus.setdefault(row["corpus"], {})
                         .setdefault(ev, {})
                         .setdefault(src, [])
                         .append((row["n"], bool(row["pass"]))))
    # de-dup identical (n, pass) pairs within one (event, source) -- the table has no unique
    # constraint, the same real event can be pushed more than once
    for corpus in events_by_corpus:
        for ev in events_by_corpus[corpus]:
            for src in events_by_corpus[corpus][ev]:
                events_by_corpus[corpus][ev][src] = sorted(set(events_by_corpus[corpus][ev][src]))
    return events_by_corpus


def event_source_contradiction(pts_by_n_and_flag):
    """pts_by_n_and_flag: [(n, pass_bool), ...] for ONE (event, source). Returns the set of n
    values that appear with BOTH True and False -- a real contradiction within a single trusted
    source (e.g. a retried push landed twice with different results), which must never be
    silently resolved by sort order (D-qNg64-11's whole point). Empty set means clean."""
    by_n = {}
    for n, p in pts_by_n_and_flag:
        by_n.setdefault(n, set()).add(p)
    return {n for n, flags in by_n.items() if len(flags) > 1}


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

        # D-qNg64-3 (L3a point 9): positive control -- this module's own docstring records TWO
        # real silent-failure modes that both look exactly like a clean "no hit" (missing cd:
        # every n reads fail; wrong same-req manifest: a real-looking near-tie at the WRONG pos).
        # Neither raises, neither has a distinct error signature -- absence of the hit line was
        # being trusted as a genuine fail with no way to tell it apart from "the run never
        # actually reached this req/pos at all". Require positive evidence the engine reached
        # and evaluated this exact (req,pos) -- any diagnostic line naming both -- before trusting
        # a "no hit" as a real fail rather than a silently broken run.
        reached = any((f"req={req} " in line or f"req={req},"  in line or line.endswith(f"req={req}"))
                      and (f"pos={pos} " in line or f"pos={pos},"  in line or line.endswith(f"pos={pos}"))
                      for line in out.splitlines())
        if not reached:
            raise RuntimeError(
                f"live oracle positive control failed: no diagnostic line mentions req={req} pos={pos} "
                f"at n={n} -- the run likely never reached this position (wrong cd, wrong manifest, "
                f"or a crash) rather than genuinely testing it. Full output:\n{out[-2000:]}"
            )

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


# ---------------------------------------------------------------------------
# D-qNg64-11 (second Opus review of L3b Phase C, 2026-09-08): derive an isolated
# single-request manifest from a live attribution event's own (manifest, req), and verify
# the derivation via a real baseline replay BEFORE trusting it for anything -- this is the
# "biggest structural gap" the review found: a live attribution's req is only meaningful
# relative to the FULL manifest it fired against (qwen_infer.c:6973's `sp = r % MCN`), and
# getting this wrong has already caused two silent, non-erroring provenance corruptions in
# this project (D-d5-27 and its documented repeat). This code makes the derivation explicit
# and the Step-0 gate is what actually catches a wrong derivation -- the `test -f` checks
# below are a cheap pre-filter, NOT the safety property.
# ---------------------------------------------------------------------------
import shlex


def derive_isolated_manifest(ssh_host, manifest_path, req, derived_dir, run_id):
    """Returns (derived_manifest_path_on_host, mf_n, selected_line) or raises RuntimeError with
    a specific reason. Mirrors qwen_infer.c's own `sp = r % MCN` (:6973) exactly -- req indexes
    manifest line (req % mf_n), NOT req itself, since manifest entries repeat cyclically across
    requests. Does NOT trust that the derivation is correct -- that's the caller's job (run the
    Step-0 gate below), this function only performs the mechanical derivation + a cheap
    existence pre-filter."""
    q_manifest = shlex.quote(manifest_path)
    check = subprocess.run(["ssh", ssh_host, f"test -f {q_manifest}"])
    if check.returncode != 0:
        raise RuntimeError(f"manifest not found on {ssh_host}: {manifest_path}")

    cat = subprocess.run(["ssh", ssh_host, f"cat {q_manifest}"], capture_output=True, text=True, timeout=30)
    if cat.returncode != 0:
        raise RuntimeError(f"could not read manifest on {ssh_host}: {manifest_path}: {cat.stderr}")

    lines = [ln.strip() for ln in cat.stdout.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    mf_n = len(lines)
    if mf_n == 0:
        raise RuntimeError(f"manifest has 0 usable (non-blank, non-#) lines: {manifest_path}")

    idx = req % mf_n
    selected = lines[idx]
    parts = selected.split()
    if len(parts) < 2:
        raise RuntimeError(f"manifest line {idx} malformed (expected '<token_file> <max_new_tokens>'): {selected!r}")
    token_file = parts[0]

    # Recursive check (Opus review point 4.3): the manifest existing doesn't mean the prompt
    # file it references still does -- load_ids() FATALs if it's gone, which would otherwise
    # look like an eligible-but-actually-crashing sweep target.
    q_token_file = shlex.quote(token_file)
    tf_check = subprocess.run(["ssh", ssh_host, f"test -f {q_token_file}"])
    if tf_check.returncode != 0:
        raise RuntimeError(f"manifest line {idx} references a token file that no longer exists on {ssh_host}: {token_file}")

    derived_path = f"{derived_dir}/{run_id}_iso.txt"
    write_cmd = f"mkdir -p {shlex.quote(derived_dir)} && printf '%s\n' {shlex.quote(selected)} > {shlex.quote(derived_path)}"
    w = subprocess.run(["ssh", ssh_host, write_cmd], capture_output=True, text=True, timeout=30)
    if w.returncode != 0:
        raise RuntimeError(f"failed to write derived manifest on {ssh_host}: {w.stderr}")

    return derived_path, mf_n, selected


def step0_baseline_gate(ssh_host, moe_base, bin_path, cwd, derived_manifest, combo_path,
                         model, corpus, recorded_orig_argmax, recorded_corrected_argmax,
                         recorded_pos, selflog_dir, run_id, safetensors_index=None,
                         max_pos=19, timeout_s=180):
    """Runs the UNMODIFIED hi mirror (no QWEN_MOE_ATTRIB_SIM_QN/SIM_PATH) against the derived
    single-request manifest at req=0 (isolation means req is always 0 after derivation -- the
    derived manifest has exactly one entry). Returns (ok: bool, detail: dict). ok=True ONLY if
    the real run reproduces BOTH recorded_orig_argmax AND recorded_corrected_argmax exactly at
    the recorded pos -- this is the actual defense against a wrong derivation silently producing
    confident-but-wrong data (Opus review 3a/3b/6.1), not a per-n check.

    Distinguishes no-signal (nothing at all mentions req=0 pos=<recorded_pos>, or a BASELINE WARN
    line appears -- the run itself is not evidence of anything) from wrong-signal (the position
    was reached but produced different orig/corrected values than recorded -- a real, informative
    mismatch, e.g. this genuinely is the wrong manifest/event). Self-logs to a controller-owned
    directory (never /dev/null, never the production log this same run's own worklist might read
    from -- the caller must ensure selflog_dir is excluded from any report/worklist scan)."""
    self_log_path = f"{selflog_dir}/{run_id}_step0.jsonl"
    env = {
        "QWEN_MOE_BASE": moe_base,
        "QWEN_MOE_CBATCH": "1", "QWEN_MOE_CB_ONLINE": "1",
        "QWEN_MOE_CB_PROMPT_MANIFEST": derived_manifest, "QWEN_MOE_CB_REQS": "1",
        "QWEN_MOE_NEARTIE_CORRECT": "1", "QWEN_MOE_NEARTIE_LOG": "1",
        "QWEN_MOE_NEARTIE_MODEL": model, "QWEN_MOE_NEARTIE_CORPUS": corpus,
        "QWEN_MOE_NEARTIE_EVENTS_LOG": self_log_path,
        "QWEN_MOE_ATTRIB": "1", "QWEN_MOE_NEARTIE_HI_COMBOS": combo_path,
        "QWEN_MOE_ATTRIB_MAX_POS": str(max_pos),
    }
    if safetensors_index:
        env["QWEN_MOE_NEARTIE_CORRECT_SAFETENSORS"] = safetensors_index

    env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    # Sanitize inherited production-serving state (Opus review point 4, "unmentioned and
    # important"): a sweep must never silently inherit bob's own live QWEN_MOE_PROMOTION_FILE_NQ/
    # QWEN_MOE_PROMOTION_FILE from the shell environment -- that would measure the target on top
    # of whatever else is already promoted, incomparable to the recorded attribution.
    unset_str = "env -u QWEN_MOE_PROMOTION_FILE_NQ -u QWEN_MOE_PROMOTION_FILE -u QWEN_MOE_NEARTIE_HI_EXPERT_LAYERS"
    # Kills the REMOTE process on expiry (Opus review point finding 5/8) -- `subprocess.run(...,
    # timeout=T)` alone only kills the local ssh client, leaving an orphaned multi-GB engine
    # process running on the remote host. Originally written as `timeout -k 30 {T}s <bin>`
    # (GNU coreutils) -- discovered by ACTUALLY RUNNING this against bob (a Mac, BSD userland)
    # that neither `timeout` nor `gtimeout` exist there and coreutils isn't installed; rather
    # than install a new system package on a shared machine without asking, this uses a portable
    # POSIX background-process-plus-watcher pattern instead (background the real command, a
    # sibling `sleep T && kill -9` watcher races it, `wait` on the real command's PID, then kill
    # the watcher so it doesn't linger).
    # D-qNg64-13: selflog_dir (SELFLOG_DIR/runs) is never created remotely before this --
    # QWEN_MOE_NEARTIE_EVENTS_LOG fopen()s it eagerly at startup and FATALs immediately if the
    # directory doesn't exist, before loading anything -- which this function's own output
    # parsing correctly, but misleadingly, reports as "no-signal" (nothing WAS logged, but not
    # because the position wasn't reached -- the engine never got that far). Found only by
    # actually running this end-to-end: a hand-reconstructed version of this exact command
    # against a directory that already existed reproduced the real flip perfectly, which is what
    # exposed the gap. Same mkdir-first pattern derive_isolated_manifest() already uses.
    selflog_parent = shlex.quote(os.path.dirname(self_log_path))
    bg_cmd = (
        f"mkdir -p {selflog_parent} && "
        f"cd {shlex.quote(cwd)} && {unset_str} {env_str} {shlex.quote(bin_path)} & "
        f"CMD_PID=$!; "
        f"( sleep {timeout_s} && kill -9 $CMD_PID 2>/dev/null ) & WATCHER_PID=$!; "
        f"wait $CMD_PID; CMD_EXIT=$?; kill $WATCHER_PID 2>/dev/null; exit $CMD_EXIT"
    )
    remote_cmd = bg_cmd
    result = subprocess.run(["ssh", ssh_host, remote_cmd], capture_output=True, text=True, timeout=timeout_s + 60)
    out = result.stdout + result.stderr

    detail = {"stdout_tail": out[-2000:], "derived_manifest": derived_manifest, "self_log": self_log_path}

    if "[moe promotion nq]" in out:
        detail["reason"] = "sweep inherited a live promotion despite env sanitization -- environment leak, treat as no-signal"
        return False, detail

    warn_marker = f"pos={recorded_pos}"
    if f"BASELINE WARN" in out and warn_marker in out:
        detail["reason"] = "BASELINE WARN present at the recorded pos -- structurally invalid run, not evidence"
        return False, detail

    correct_marker = f"correct req=0 pos={recorded_pos}"
    if correct_marker not in out:
        detail["reason"] = f"no-signal: nothing in the run mentions '{correct_marker}' -- the position was never reached or correction never fired at all"
        return False, detail

    flip_marker = f"REAL FLIP orig={recorded_orig_argmax} corrected={recorded_corrected_argmax}"
    if flip_marker in out:
        detail["reason"] = "Step-0 PASS: derivation reproduces the exact recorded flip"
        return True, detail

    import re
    m = re.search(rf"REAL FLIP orig=(\d+) corrected=(\d+)", out)
    if m:
        detail["reason"] = f"wrong-signal: real flip found but orig={m.group(1)} corrected={m.group(2)} != recorded orig={recorded_orig_argmax} corrected={recorded_corrected_argmax}"
    else:
        detail["reason"] = f"position reached, correction fired, but no REAL FLIP line at all (recorded event says there should be one) -- possible wrong derivation or non-deterministic divergence"
    return False, detail


# D-qNg64-12 (L3b Phase C, continued): the real per-n sweep loop -- this is what actually
# tests a triple across the deployable ladder {5,6,7}, gated on step0_baseline_gate() already
# having passed for this exact (manifest, event). Reuses the same portable timeout-kill pattern
# step0 established (no GNU coreutils on bob), same env-sanitization, same self-log discipline.
REAL_LADDER = (5, 6, 7)


def sweep_one_n(ssh_host, moe_base, bin_path, cwd, derived_manifest, combo_path,
                 model, corpus, role, layer, n,
                 recorded_corrected_argmax, recorded_pos,
                 selflog_dir, run_id, safetensors_index=None, max_pos=19, timeout_s=180):
    """Runs ONE real engine invocation with QWEN_MOE_ATTRIB_SIM_QN=n overriding role/layer's hi
    mirror (moe_register_hi_role(), qwen_infer.c:14440+ -- real qNg64 kernel, not a simulated
    F32 override). Classifies into 5 outcomes per the Opus review (do NOT collapse "flipped to
    the wrong token" into PASS -- that was the original design's actual bug):

      "no_signal"   -- BASELINE WARN at this pos, or nothing mentions it at all -- the run is
                       not evidence of anything, caller should abort the WHOLE triple, not just
                       this n (if the position can't even be reached under one n, treating other
                       n's results as trustworthy is unfounded).
      "fail_noflip" -- position reached, correction fired, but no REAL FLIP -- this n does not
                       recover the token.
      "fail_wrong"  -- REAL FLIP fired but to a DIFFERENT token than recorded -- this n recovers
                       something, but not the right thing. A real, informative FAIL, not a PASS.
      "pass"        -- REAL FLIP matches the recorded corrected_argmax exactly.

    Returns (outcome: str, detail: dict)."""
    self_log_path = f"{selflog_dir}/{run_id}_n{n}.jsonl"
    env = {
        "QWEN_MOE_BASE": moe_base,
        "QWEN_MOE_CBATCH": "1", "QWEN_MOE_CB_ONLINE": "1",
        "QWEN_MOE_CB_PROMPT_MANIFEST": derived_manifest, "QWEN_MOE_CB_REQS": "1",
        "QWEN_MOE_NEARTIE_CORRECT": "1", "QWEN_MOE_NEARTIE_LOG": "1",
        "QWEN_MOE_NEARTIE_MODEL": model, "QWEN_MOE_NEARTIE_CORPUS": corpus,
        "QWEN_MOE_NEARTIE_EVENTS_LOG": self_log_path,
        "QWEN_MOE_ATTRIB": "1", "QWEN_MOE_NEARTIE_HI_COMBOS": combo_path,
        "QWEN_MOE_ATTRIB_MAX_POS": str(max_pos),
        # The three env vars moe_register_hi_role() requires TOGETHER to take the real-kernel
        # SIM_QN branch (qwen_infer.c:14444-14453) rather than falling through to the F32-sim
        # path or the unmodified bits=16 hi mirror.
        "QWEN_MOE_ATTRIB_SIM_ROLE": role, "QWEN_MOE_ATTRIB_SIM_LAYER": str(layer),
        "QWEN_MOE_ATTRIB_SIM_QN": str(n),
    }
    if safetensors_index:
        env["QWEN_MOE_NEARTIE_CORRECT_SAFETENSORS"] = safetensors_index

    env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    unset_str = "env -u QWEN_MOE_PROMOTION_FILE_NQ -u QWEN_MOE_PROMOTION_FILE -u QWEN_MOE_NEARTIE_HI_EXPERT_LAYERS"
    # D-qNg64-13: same missing-remote-mkdir bug as step0_baseline_gate (see its own comment) --
    # fixed here identically.
    selflog_parent = shlex.quote(os.path.dirname(self_log_path))
    bg_cmd = (
        f"mkdir -p {selflog_parent} && "
        f"cd {shlex.quote(cwd)} && {unset_str} {env_str} {shlex.quote(bin_path)} & "
        f"CMD_PID=$!; "
        f"( sleep {timeout_s} && kill -9 $CMD_PID 2>/dev/null ) & WATCHER_PID=$!; "
        f"wait $CMD_PID; CMD_EXIT=$?; kill $WATCHER_PID 2>/dev/null; exit $CMD_EXIT"
    )
    result = subprocess.run(["ssh", ssh_host, bg_cmd], capture_output=True, text=True, timeout=timeout_s + 60)
    out = result.stdout + result.stderr
    detail = {"n": n, "stdout_tail": out[-2000:], "self_log": self_log_path}

    if "[moe promotion nq]" in out:
        detail["reason"] = "environment leak: sweep inherited a live promotion despite env sanitization"
        return "no_signal", detail

    if "BASELINE WARN" in out and f"pos={recorded_pos}" in out:
        detail["reason"] = "BASELINE WARN at the recorded pos -- structurally invalid run"
        return "no_signal", detail

    correct_marker = f"correct req=0 pos={recorded_pos}"
    if correct_marker not in out:
        detail["reason"] = f"no-signal: nothing mentions '{correct_marker}' -- position never reached or correction never fired"
        return "no_signal", detail

    flip_marker = f"REAL FLIP orig="
    if flip_marker not in out:
        detail["reason"] = f"n={n} does not recover the token (correction fired, no flip)"
        return "fail_noflip", detail

    match = re.search(r"REAL FLIP orig=(\d+) corrected=(\d+)", out)
    if not match:
        detail["reason"] = f"n={n}: 'REAL FLIP orig=' present but line didn't parse -- treat as no-signal, don't guess"
        return "no_signal", detail

    corrected = int(match.group(2))
    if corrected != recorded_corrected_argmax:
        detail["reason"] = f"n={n} flips to token {corrected}, NOT the recorded {recorded_corrected_argmax} -- wrong recovery"
        return "fail_wrong", detail

    detail["reason"] = f"n={n} PASS: recovers the exact recorded token {recorded_corrected_argmax}"
    return "pass", detail


def sweep_triple(ssh_host, moe_base, bin_path, cwd, derived_manifest, combo_path,
                  model, corpus, role, layer,
                  recorded_orig_argmax, recorded_corrected_argmax, recorded_pos,
                  selflog_dir, run_id, safetensors_index=None, max_pos=19, timeout_s=180):
    """Runs Step-0 then the full REAL_LADDER sweep for one triple. Returns
    (results: {n: (outcome, detail)} or None, abort_reason: str or None) -- results is None iff
    Step-0 failed OR any n produced "no_signal" (the whole triple is aborted, per the review:
    a triple with ANY no-signal n is not safe to push partial results for)."""
    ok, step0_detail = step0_baseline_gate(
        ssh_host, moe_base, bin_path, cwd, derived_manifest, combo_path, model, corpus,
        recorded_orig_argmax, recorded_corrected_argmax, recorded_pos, selflog_dir, run_id,
        safetensors_index=safetensors_index, max_pos=max_pos, timeout_s=timeout_s,
    )
    if not ok:
        return None, f"Step-0 baseline gate failed: {step0_detail.get('reason')}"

    results = {}
    for n in REAL_LADDER:
        outcome, detail = sweep_one_n(
            ssh_host, moe_base, bin_path, cwd, derived_manifest, combo_path,
            model, corpus, role, layer, n, recorded_corrected_argmax, recorded_pos,
            selflog_dir, run_id, safetensors_index=safetensors_index, max_pos=max_pos, timeout_s=timeout_s,
        )
        results[n] = (outcome, detail)
        if outcome == "no_signal":
            return None, f"n={n}: {detail.get('reason')} -- aborting whole triple, no-signal is not evidence"

    return results, None


def push_sweep_results_atomic(model, corpus, role, layer, req, pos, results):
    """results: {n: (outcome, detail)} from sweep_triple(), ALL n present (never called on a
    partial/aborted triple -- sweep_triple() returns None for `results` in that case, caller
    must not call this with that). Pushes all len(REAL_LADDER) rows in ONE array POST (Postgres
    commits a JSON array in a single transaction via PostgREST -- this IS the atomicity, per-n
    posting would silently lose it), source='qng64_real' EXPLICITLY asserted present before
    serializing (a push that forgets it would be indistinguishable from real data mislabeled as
    simulated -- the single worst outcome this whole effort exists to prevent). Then re-SELECTs
    to confirm exactly len(REAL_LADDER) rows landed with that exact (model,corpus,role,layer,req,
    pos,source) -- `Prefer: return=minimal` on the POST means nothing about the write result is
    otherwise observable. Raises RuntimeError on ANY unverified state -- caller must not proceed
    to promotion_writeback on an unverified push."""
    url = os.environ.get("QWEN_SUPABASE_URL")
    key = os.environ.get("QWEN_SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY must be set to push results")

    rows = []
    for n in REAL_LADDER:
        outcome, detail = results[n]
        row = {
            "model": model, "corpus": corpus, "role": role, "layer": layer,
            "req": req, "pos": pos, "n": n, "pass": (outcome == "pass"),
            "eff_bpw": eff_bpw(n), "source": "qng64_real",
        }
        assert row["source"] == "qng64_real", "refusing to push a row without explicit real-source tag"
        rows.append(row)

    post_req = urllib.request.Request(
        f"{url}/rest/v1/moe_quant_sweep_results",
        data=json.dumps(rows).encode(),
        headers={"apikey": key, "Authorization": f"Bearer {key}",
                 "Content-Type": "application/json", "Prefer": "return=minimal"},
        method="POST",
    )
    try:
        urllib.request.urlopen(post_req, timeout=30)
    except Exception as e:
        raise RuntimeError(f"push POST failed for {role}/L{layer} req={req} pos={pos}: {e}")

    verify_qs = (f"model=eq.{model}&corpus=eq.{corpus}&role=eq.{role}&layer=eq.{layer}"
                 f"&req=eq.{req}&pos=eq.{pos}&source=eq.qng64_real&select=n,pass")
    verify_req = urllib.request.Request(
        f"{url}/rest/v1/moe_quant_sweep_results?{verify_qs}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(verify_req, timeout=30) as resp:
        landed = json.loads(resp.read())
    landed_ns = sorted(r["n"] for r in landed)
    if landed_ns != sorted(REAL_LADDER):
        raise RuntimeError(f"PUSH_UNVERIFIED: expected rows for n={sorted(REAL_LADDER)}, "
                            f"post-push SELECT found n={landed_ns} -- do NOT proceed to "
                            f"promotion_writeback on this triple until this is resolved")
    return landed


# ---------------------------------------------------------------------------
# D-qNg64-12: backoff ledger -- prevents one slow/stuck/failing triple from permanently
# blocking the whole worklist (a real livelock the Opus review found: deterministic ranking +
# no state means the same top-N triples get re-selected forever if they never succeed).
# ---------------------------------------------------------------------------

def _ledger_key(model, corpus, role, layer, req, pos, manifest):
    return json.dumps([model, corpus, role, layer, req, pos, manifest], sort_keys=True)


def load_ledger(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_ledger(path, ledger):
    # D-qNg64-13: parent dir (DEFAULT_LEDGER=/private/tmp/qng64_ctl/...) is never created
    # anywhere else -- first real --run crashed here with FileNotFoundError, masking whatever
    # the actual triple outcome was (this fires from every ledger_record() call site, including
    # the except-block ones, so the crash pre-empted the real error being visible at all).
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(ledger, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def ledger_eligible(ledger, key, now=None):
    now = now if now is not None else time.time()
    entry = ledger.get(key)
    if entry is None:
        return True
    return now >= entry.get("next_eligible_ts", 0)


def ledger_record(ledger, key, outcome, duration_s):
    """outcome: 'success' | 'timeout' | 'no_signal' | 'gate_fail' | 'push_unverified'.
    Exponential backoff (hours) on anything but success; success resets attempts to 0 and sets
    next_eligible_ts far in the future (an audit record, not "never touch again" -- a real n>=8
    knee case or new corpus data could still warrant a future re-sweep, just not automatically)."""
    now = time.time()
    entry = ledger.get(key, {"attempts": 0})
    if outcome == "success":
        entry["attempts"] = 0
        entry["next_eligible_ts"] = now + 365 * 86400
    else:
        entry["attempts"] = entry.get("attempts", 0) + 1
        backoff_hours = 2 ** min(entry["attempts"], 6)
        entry["next_eligible_ts"] = now + backoff_hours * 3600
    entry["last_outcome"] = outcome
    entry["last_duration_s"] = duration_s
    entry["last_attempt_ts"] = now
    ledger[key] = entry
    return entry


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
