#!/usr/bin/env python3
# Parses a quant_sim n-sweep summary TSV (target/role/layer/n/pass/logline,
# produced by run_quant_sweep*.sh on bob) + the per-tensor summary.json
# rel_l2 curves (produced by tools/quant_sim_n.py), finds each target's
# pass/fail knee (lowest n where pass==1) with a monotonicity check across
# all higher n, and reports both axes (container n, effective bpw).
#
# D-quant-mono-1 (2026-09-02): also supports --push, writing every raw row
# to Supabase's moe_quant_sweep_results (model/corpus/role/layer/req/pos/n/
# eff_bpw/pass/rel_l2) -- the durable, corpus-tagged home for this data, so
# a later WikiText-103 sweep can be compared against WikiText-2's without
# guessing which local TSV file is authoritative. stdlib-only (urllib),
# same env-var-only credential convention as d4_supabase_push.py
# (QWEN_SUPABASE_URL/QWEN_SUPABASE_KEY, never hardcoded).
import argparse
import csv
import glob
import json
import os
import re
import sys
import urllib.error
import urllib.request

GROUP = 64
LOGLINE_RE = re.compile(r"req=(\d+)\s+pos=(\d+)")


def eff_bpw(n):
    return n + 32.0 / (GROUP * n)


def load_summary(tsv_path, req_fallback=None, pos_fallback=None):
    # A "fail" row's logline can legitimately be empty: moe_neartie_attribute()
    # only ever prints "req=X pos=Y done: ..." when moe_neartie_maybe_correct()
    # first flags a REAL flip -- at a low enough n the selective-restricted
    # correction can fail to move the argmax away from baseline at all, so no
    # flip is detected and nothing is printed, even though this n's pass=0 is
    # itself real, correct data. req/pos fallback covers exactly that case
    # (every row in one sweep run always shares the same real (req,pos) the
    # driver script targeted -- found live: 17/90 of today's rows needed this).
    rows = []
    with open(tsv_path) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            row["n"] = int(row["n"])
            row["pass"] = int(row["pass"])
            m = LOGLINE_RE.search(row.get("logline", "") or "")
            row["req"] = int(m.group(1)) if m else req_fallback
            row["pos"] = int(m.group(2)) if m else pos_fallback
            rows.append(row)
    return rows


def load_rel_l2(summary_json_path):
    with open(summary_json_path) as f:
        d = json.load(f)
    return {r["n"]: r["rel_l2"] for r in d["results"]}


def discover_rel_l2(summary_dir, targets):
    """Auto-discover <summary_dir>/sim_<target>/summary.json for each target
    seen in the TSV -- avoids hardcoding a fixed positional-arg list that
    breaks the moment a sweep covers more than 3 targets (Phase 1's own
    original CLI shape, outgrown as soon as the 3-target extension ran)."""
    out = {}
    if not summary_dir:
        return out
    for target in targets:
        p = os.path.join(summary_dir, f"sim_{target}", "summary.json")
        if os.path.exists(p):
            out[target] = load_rel_l2(p)
    return out


def analyze(rows, rel_l2_by_target):
    targets = {}
    for row in rows:
        targets.setdefault(row["target"], []).append(row)

    results = {}
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
        is_tier = None
        if knee is not None:
            print(f"knee effective bpw: {eff_bpw(knee):.3f}")
            is_tier = knee in (4, 8, 16)
            print(f"knee is a standard tier (4/8/16): {is_tier}")
        monotonic = not violations
        if violations:
            print(f"MONOTONICITY VIOLATION -- failed at n={violations} after passing at a lower n")
        else:
            print("monotonicity: OK (once passing, stays passing at every higher n tested)")
        results[target] = {"role": role, "layer": layer, "knee": knee, "is_tier": is_tier, "monotonic": monotonic}
    return results


def push_rows(url, key, model, corpus, rows):
    url = url.rstrip("/")
    payload = []
    skipped = 0
    for row in rows:
        if row["req"] is None or row["pos"] is None:
            skipped += 1
            continue
        payload.append({
            "model": model,
            "corpus": corpus,
            "role": row["role"],
            "layer": int(row["layer"]),
            "req": row["req"],
            "pos": row["pos"],
            "n": row["n"],
            "eff_bpw": eff_bpw(row["n"]),
            "pass": bool(row["pass"]),
            "rel_l2": row.get("rel_l2"),
        })
    if skipped:
        print(f"[quant_sim_analyze] WARN {skipped} rows skipped (no req/pos parsed from logline)", file=sys.stderr)
    if not payload:
        print("[quant_sim_analyze] nothing to push", file=sys.stderr)
        return
    req = urllib.request.Request(
        url + "/rest/v1/moe_quant_sweep_results",
        data=json.dumps(payload).encode(),
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=20)
        print(f"[quant_sim_analyze] pushed {len(payload)} rows to moe_quant_sweep_results (model={model}, corpus={corpus})")
    except urllib.error.HTTPError as e:
        print(f"[quant_sim_analyze] WARN HTTP {e.code} pushing {len(payload)} rows: {e.read()[:400]}", file=sys.stderr)
    except Exception as e:
        print(f"[quant_sim_analyze] WARN {type(e).__name__} pushing rows: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tsv", nargs="?", default="/tmp/quantsim_sweep_summary.tsv")
    ap.add_argument("--summary-dir", help="directory containing sim_<target>/summary.json subdirs for rel_l2 curves")
    ap.add_argument("--push", action="store_true", help="also push every raw row to Supabase moe_quant_sweep_results")
    ap.add_argument("--model", help="required with --push")
    ap.add_argument("--corpus", help="required with --push")
    ap.add_argument("--req", type=int, help="fallback req when a row's logline has none (fail rows with no detected flip)")
    ap.add_argument("--pos", type=int, help="fallback pos when a row's logline has none")
    args = ap.parse_args()

    rows = load_summary(args.tsv, req_fallback=args.req, pos_fallback=args.pos)
    targets = sorted(set(r["target"] for r in rows))
    rel_l2_by_target = discover_rel_l2(args.summary_dir, targets)
    # merge per-n rel_l2 back onto each row for pushing
    for row in rows:
        row["rel_l2"] = rel_l2_by_target.get(row["target"], {}).get(row["n"])

    analyze(rows, rel_l2_by_target)

    if args.push:
        url = os.environ.get("QWEN_SUPABASE_URL", "")
        key = os.environ.get("QWEN_SUPABASE_KEY", "")
        if not url or not key:
            print("FATAL: --push requires QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY", file=sys.stderr)
            sys.exit(1)
        if not args.model or not args.corpus:
            print("FATAL: --push requires --model and --corpus", file=sys.stderr)
            sys.exit(1)
        push_rows(url, key, args.model, args.corpus, rows)


if __name__ == "__main__":
    main()
