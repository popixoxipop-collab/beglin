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
  - live: not implemented in this pass (would SSH to bob, build override
    files via quant_sim_n.py, and run the real engine per test -- same
    mechanism the sweep scripts already use). --validate is the intended
    first real usage; live wiring is a natural follow-up once the search
    logic itself is trusted.
"""
import argparse
import csv
import json
import os
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", metavar="TSV", help="validate against an already-completed sweep TSV (historical oracle, no new engine calls)")
    ap.add_argument("--model", default="deepseek-v2-lite")
    ap.add_argument("--corpus", default="deepseek-moe4a-builtin-corpus")
    args = ap.parse_args()

    if args.validate:
        ok = run_validation(args.validate, args.model, args.corpus)
        sys.exit(0 if ok else 1)
    else:
        print("Nothing to do -- pass --validate <tsv> to check against known-good sweep data.", file=sys.stderr)
        print("Live mode (real engine calls via bob) is not wired up in this pass -- see module docstring.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
