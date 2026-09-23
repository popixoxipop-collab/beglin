#!/usr/bin/env python3
"""Run one real qNg64 ladder directly from durable attribution provenance.

This never edits the live promotion file. It only:
provenance -> isolated manifest -> Step-0 -> qNg64 {5,6,7} -> atomic DB push.
A later P5 plan/live-preflight decides whether anything may be promoted.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import attribution_provenance as prov
import autopilot_guarded as guarded
import quant_search_n as qsn
import promotion_writeback as pwb


def _write_combo(host, path, role, layer):
    directory = path.rsplit("/", 1)[0]
    cmd = (
        f"mkdir -p {shlex.quote(directory)} && "
        f"printf '%s\n' {shlex.quote(f'{role} {int(layer)}')} > "
        f"{shlex.quote(path)}"
    )
    p = subprocess.run(["ssh", host, cmd], capture_output=True, text=True, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(f"cannot write combo: {p.stderr}")


def _candidate_from_plan(path):
    plan = guarded._load_plan(path)
    items = plan.get("real_sweep_candidates", [])
    if not items:
        raise RuntimeError("plan has no real_sweep_candidates")
    return items[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="deepseek-v2-lite")
    ap.add_argument("--role")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--plan")
    ap.add_argument("--ssh-host", default="bob")
    ap.add_argument("--moe-base", default="/Users/bob/moe_base_deepseek")
    ap.add_argument("--bin", default="/Users/bob/vdsp_m4_bench/qwen_infer_prod")
    ap.add_argument("--cwd", default="/Users/bob/vdsp_m4_bench")
    ap.add_argument(
        "--safetensors-index",
        default="/Volumes/D50/deepseek_v2lite_bf16_safetensors/model.safetensors.index.json",
    )
    ap.add_argument("--work-dir", default="/private/tmp/qng64_ctl/provenance_sweeps")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--max-pos", type=int, default=19)
    args = ap.parse_args()

    if args.plan:
        c = _candidate_from_plan(args.plan)
        role, layer = c["role"], int(c["layer"])
        model = c.get("model") or args.model
        row = c.get("provenance") or prov.fetch_best(model, role, layer)
    else:
        if not args.role or args.layer is None:
            ap.error("provide --plan or both --role and --layer")
        model, role, layer = args.model, args.role, args.layer
        row = prov.fetch_best(model, role, layer)

    if not row:
        raise RuntimeError(f"no replayable provenance for {model}/{role}/L{layer}")

    stamp = int(time.time())
    run_id = f"prov_{role}_L{layer}_{row['id']}_{stamp}"
    combo = f"{args.work_dir}/{run_id}.combo.txt"
    _write_combo(args.ssh_host, combo, role, layer)
    derived, _, _ = qsn.derive_isolated_manifest(
        args.ssh_host, row["manifest"], int(row["req"]),
        f"{args.work_dir}/derived", run_id,
    )
    max_pos = max(int(args.max_pos), int(row["pos"]))
    results, abort = qsn.sweep_triple(
        args.ssh_host, args.moe_base, args.bin, args.cwd,
        derived, combo, model, row["corpus"], role, layer,
        int(row["orig_argmax"]), int(row["corrected_argmax"]), int(row["pos"]),
        f"{args.work_dir}/runs", run_id,
        safetensors_index=args.safetensors_index,
        max_pos=max_pos, timeout_s=args.timeout,
    )
    if results is None:
        raise RuntimeError(f"real sweep aborted: {abort}")

    print("RESULTS", {n: outcome for n, (outcome, _) in results.items()})
    landed = qsn.push_sweep_results_atomic(
        model, row["corpus"], role, layer, int(row["req"]), int(row["pos"]), results
    )
    print("LANDED_ROWS", len(landed))
    safe_n, detail = pwb.target_safe_n(model, role, layer)
    print("TARGET_SAFE_N", safe_n)
    print("DETAIL", json.dumps(detail, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
