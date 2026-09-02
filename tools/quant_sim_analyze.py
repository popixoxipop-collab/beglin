#!/usr/bin/env python3
# Parses quantsim_sweep_summary.tsv (target/role/layer/n/pass/logline) + the
# per-tensor summary.json rel_l2 curves, finds each target's pass/fail knee
# (lowest n where pass==1, with a monotonicity check across all higher n),
# and reports both axes (container n, effective bpw).
import csv
import json
import sys

GROUP = 64


def eff_bpw(n):
    return n + 32.0 / (GROUP * n)


def load_summary(tsv_path):
    rows = []
    with open(tsv_path) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            row["n"] = int(row["n"])
            row["pass"] = int(row["pass"])
            rows.append(row)
    return rows


def load_rel_l2(summary_json_path):
    with open(summary_json_path) as f:
        d = json.load(f)
    return {r["n"]: r["rel_l2"] for r in d["results"]}


def analyze(rows, rel_l2_by_target):
    targets = {}
    for row in rows:
        targets.setdefault(row["target"], []).append(row)

    for target, trows in targets.items():
        trows.sort(key=lambda r: r["n"])
        role = trows[0]["role"]
        layer = trows[0]["layer"]
        print(f"\n=== {target} ({role} layer={layer}) ===")
        print(f"{'n':>3} {'eff_bpw':>8} {'pass':>5} {'rel_l2':>12}")
        knee = None
        violations = []
        seen_pass = False
        for row in trows:
            n = row["n"]
            p = row["pass"]
            rl2 = rel_l2_by_target.get(target, {}).get(n, float("nan"))
            print(f"{n:>3} {eff_bpw(n):>8.3f} {p:>5} {rl2:>12.6e}")
            if p == 1 and knee is None:
                knee = n
            if p == 1:
                seen_pass = True
            elif seen_pass and p == 0:
                violations.append(n)  # passed at a lower n, failed at a higher n -- non-monotone
        print(f"knee (lowest passing n): {knee}")
        if knee is not None:
            print(f"knee effective bpw: {eff_bpw(knee):.3f}")
            is_tier = knee in (4, 8, 16)
            print(f"knee is a standard tier (4/8/16): {is_tier}")
        if violations:
            print(f"MONOTONICITY VIOLATION -- failed at n={violations} after passing at a lower n")
        else:
            print("monotonicity: OK (once passing, stays passing at every higher n tested)")


if __name__ == "__main__":
    tsv_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/quantsim_sweep_summary.tsv"
    rows = load_summary(tsv_path)
    rel_l2_by_target = {}
    for name, json_path in [
        ("q_proj_l5", sys.argv[2] if len(sys.argv) > 2 else None),
        ("kv_a_l13", sys.argv[3] if len(sys.argv) > 3 else None),
        ("shared_gate_l25", sys.argv[4] if len(sys.argv) > 4 else None),
    ]:
        if json_path:
            rel_l2_by_target[name] = load_rel_l2(json_path)
    analyze(rows, rel_l2_by_target)
