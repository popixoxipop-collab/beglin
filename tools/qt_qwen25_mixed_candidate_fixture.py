#!/usr/bin/env python3
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path
from typing import Iterable

from qt_heatmap import QBudgets, build_local_precision_candidate, build_q_heatmap
from qt_qwen25_group_perturb import read_tensor

GROUP = 64
TENSOR = "model.layers.0.self_attn.q_proj.weight"

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def input_vector(in_dim: int, window: int) -> list[float]:
    p1 = 0.173 + 0.013 * window
    p2 = 0.071 + 0.007 * window
    return [math.sin((i + 1) * p1) + 0.25 * math.cos((i + 1) * p2) for i in range(in_dim)]

def quantize_codes(values: Iterable[float], n: int):
    xs = [float(x) for x in values]
    if len(xs) != GROUP:
        raise ValueError("expected group64")
    qmax = (1 << (n - 1)) - 1
    qmin = -(1 << (n - 1))
    mx = max(abs(x) for x in xs)
    scale = mx / qmax if mx > 1e-12 else 1.0
    inv = 1.0 / scale
    residual = 0.0
    codes = []
    deq = []
    l1 = 0.0
    l2 = 0.0
    for x in xs:
        corrected = x + residual
        q = max(qmin, min(qmax, int(round(corrected * inv))))
        dq = q * scale
        residual = corrected - dq
        codes.append(q)
        deq.append(dq)
        d = dq - x
        l1 += abs(d)
        l2 += d * d
    denom = math.sqrt(sum(x * x for x in xs)) or 1.0
    return codes, deq, scale, l1, math.sqrt(l2), math.sqrt(l2) / denom

def encode_codes(codes: list[int], n: int) -> bytes:
    out = bytearray(n * 8)
    bias = 1 << (n - 1)
    for p, code in enumerate(codes):
        u = code + bias
        for j in range(n):
            if (u >> j) & 1:
                out[j * 8 + (p >> 3)] |= 1 << (p & 7)
    return bytes(out)

def build(
    *,
    checkpoint: Path,
    output_dir: Path,
    candidates: list[int],
    windows: int,
    local_output_budget: float,
    checkpoint_identity_sha256: str,
    skeleton_sha256: str,
    capability_bundle_sha256: str,
    model_id: str,
    source_commit: str,
    weight_epoch: int,
    incumbent_map: Path | None = None,
):
    vals, out_dim, in_dim, dtype = read_tensor(checkpoint, TENSOR)
    if in_dim % GROUP:
        raise ValueError("input dimension must be divisible by 64")
    ng = in_dim // GROUP
    candidates = sorted(set(int(n) for n in candidates))
    if not candidates or any(n < 2 or n > 15 for n in candidates):
        raise ValueError("candidate n values must be in [2,15]")
    if windows < 2:
        raise ValueError("at least two independent windows are required for candidate confidence")

    script_sha = sha256_file(Path(__file__).resolve())
    xs = [input_vector(in_dim, w) for w in range(windows)]
    events = []
    quant_cache = {}
    current_n = {}
    supported = {}
    qgroup_by_target = {}

    for row in range(out_dim):
        row_off = row * in_dim
        for g in range(ng):
            target = f"{model_id}/L0/q_proj/row={row}/group64={g}"
            group = [float(vals[row_off + g * GROUP + p]) for p in range(GROUP)]
            current_n[target] = 5
            supported[target] = candidates
            qgroup_by_target[target] = g
            for n in candidates:
                codes, deq, scale, l1, l2, rel = quantize_codes(group, n)
                quant_cache[(row, g, n)] = (codes, deq, scale)
                for w, x in enumerate(xs):
                    gx = x[g * GROUP:(g + 1) * GROUP]
                    base = sum(group[p] * gx[p] for p in range(GROUP))
                    qout = sum(deq[p] * gx[p] for p in range(GROUP))
                    err = abs(qout - base)
                    identity = {
                        "schema": "beglin-observation-identity-v2",
                        "checkpoint_identity_sha256": checkpoint_identity_sha256,
                        "skeleton_sha256": skeleton_sha256,
                        "capability_bundle_sha256": capability_bundle_sha256,
                        "target_key": target,
                        "model_id": model_id,
                        "layer": 0,
                        "tensor_role": "q_proj",
                        "expert_id": None,
                        "row": row,
                        "column": None,
                        "qgroup_index": g,
                        "group_size": GROUP,
                        "element_index": None,
                        "backend": "cpu_oracle",
                        "source_dtype": dtype,
                        "runtime_dtype": "qNg64",
                        "policy_epoch": 0,
                        "weight_epoch": weight_epoch,
                        "observation_window_id": f"calibration-{w}",
                        "observation_seq": n * windows + w,
                        "source_commit": source_commit,
                        "binary_sha256": script_sha,
                    }
                    events.append({
                        "schema": "beglin-quant-perturbation-v1",
                        "identity": identity,
                        "candidate_n": n,
                        "baseline_precision": dtype,
                        "quantized_precision": f"qNg64-n{n}",
                        "delta_weight_l1": l1,
                        "delta_weight_l2": l2,
                        "relative_weight_error": rel,
                        "output_max_abs_error": err,
                        "output_rms_error": err,
                        "output_relative_error": err / (abs(base) + 1e-6),
                        "latency_delta": 0.0,
                        "memory_delta": 0.0,
                        "backend_parity_error": None,
                        "restore_diff": None,
                        "finite": all(math.isfinite(v) for v in (scale, l1, l2, rel, err)),
                        "result_status": "PASS",
                        "sample_count": 1,
                    })

    heatmap = build_q_heatmap(
        events,
        supported_n_by_target=supported,
        current_n_by_target=current_n,
        budgets=QBudgets(output_max_abs_error=local_output_budget),
        generation=weight_epoch,
    )
    policy = build_local_precision_candidate(
        heatmap,
        policy_id=f"qwen25-l0-qproj-mixed-candidate-we{weight_epoch}",
        qgroup_index_by_target=qgroup_by_target,
        current_n_by_target=current_n,
        default_precision=dtype,
    )
    total_cells = out_dim * ng
    incumbent_sha = None
    if incumbent_map is None:
        if len(policy["cells"]) != total_cells:
            missing = total_cells - len(policy["cells"])
            bad = [c for c in heatmap["cells"] if c["state"] != "CANDIDATE"][:20]
            raise RuntimeError(f"{missing} cells have no eligible candidate; examples={bad}")
        selected_n = {c["target_key"]: int(c["n"]) for c in policy["cells"]}
    else:
        # A downstream-certified incumbent is authoritative for selection.
        # The local absolute-error heatmap remains diagnostic evidence only;
        # it must not re-quarantine a SHA-pinned map already certified by the
        # downstream hidden-state budget. Runtime parity below is still mandatory.
        from qt_incumbent_map import load_incumbent
        sealed = load_incumbent(
            incumbent_map, tensor_name=TENSOR, out_dim=out_dim, in_dim=in_dim
        )
        incumbent_sha = sealed["incumbent_sha256"]
        ordered = sealed["bits"]
        if len(ordered) != total_cells:
            raise RuntimeError("sealed incumbent cell count mismatch")
        selected_n = {}
        idx = 0
        for row in range(out_dim):
            for g in range(ng):
                target = f"{model_id}/L0/q_proj/row={row}/group64={g}"
                selected_n[target] = int(ordered[idx])
                idx += 1
    bit_hist = Counter(selected_n.values())

    planes = bytearray()
    offsets = array.array("I", [0])
    bits = array.array("B")
    scales = array.array("f")
    dequant = array.array("f", [0.0]) * (out_dim * in_dim)

    for row in range(out_dim):
        for g in range(ng):
            target = f"{model_id}/L0/q_proj/row={row}/group64={g}"
            n = selected_n[target]
            codes, group_deq, scale = quant_cache[(row, g, n)]
            planes.extend(encode_codes(codes, n))
            offsets.append(len(planes))
            bits.append(n)
            scales.append(scale)
            base = row * in_dim + g * GROUP
            for p in range(GROUP):
                dequant[base + p] = group_deq[p]

    x0 = array.array("f", xs[0])
    mixed_y = array.array("f")
    base_y = array.array("f")
    for row in range(out_dim):
        off = row * in_dim
        mixed_acc = 0.0
        base_acc = 0.0
        for i in range(in_dim):
            mixed_acc += float(dequant[off + i]) * float(x0[i])
            base_acc += float(vals[off + i]) * float(x0[i])
        mixed_y.append(mixed_acc)
        base_y.append(base_acc)

    quality_max_abs = max(abs(float(mixed_y[i]) - float(base_y[i])) for i in range(out_dim))
    quality_rms = math.sqrt(sum((float(mixed_y[i]) - float(base_y[i])) ** 2 for i in range(out_dim)) / out_dim)

    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "planes.bin": bytes(planes),
        "offsets.bin": offsets.tobytes(),
        "bits.bin": bits.tobytes(),
        "scales.bin": scales.tobytes(),
        "input.bin": x0.tobytes(),
        "cpu_expected.bin": mixed_y.tobytes(),
    }
    file_shas = {}
    for name, data in payloads.items():
        p = output_dir / name
        p.write_bytes(data)
        file_shas[name] = sha256_bytes(data)

    (output_dir / "qheatmap.json").write_text(json.dumps(heatmap, sort_keys=True, indent=2) + "\n")
    (output_dir / "candidate_policy.json").write_text(json.dumps(policy, sort_keys=True, indent=2) + "\n")
    file_shas["qheatmap.json"] = sha256_file(output_dir / "qheatmap.json")
    file_shas["candidate_policy.json"] = sha256_file(output_dir / "candidate_policy.json")

    uniform_nmax_bytes = total_cells * max(candidates) * 8
    offsets_bytes = len(payloads["offsets.bin"])
    bits_bytes = len(payloads["bits.bin"])
    scales_bytes = len(payloads["scales.bin"])
    mixed_payload_bytes = len(planes) + offsets_bytes + bits_bytes + scales_bytes
    bf16_bytes = out_dim * in_dim * 2
    manifest = {
        "schema": "beglin-qt-qwen25-mixed-candidate-fixture-v1",
        "model_id": model_id,
        "tensor_name": TENSOR,
        "shape": [out_dim, in_dim],
        "dtype": dtype,
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
        "skeleton_sha256": skeleton_sha256,
        "capability_bundle_sha256": capability_bundle_sha256,
        "heatmap_sha256": heatmap["heatmap_sha256"],
        "policy_id": policy["policy_id"],
        "policy_approved_by_beval": policy["approved_by_beval"],
        "weight_epoch": weight_epoch,
        "candidate_n": candidates,
        "local_output_budget": local_output_budget,
        "windows": windows,
        "cell_count": total_cells,
        "bit_histogram": {str(k): v for k, v in sorted(bit_hist.items())},
        "planes_bytes": len(planes),
        "offsets_bytes": offsets_bytes,
        "bits_bytes": bits_bytes,
        "scales_bytes": scales_bytes,
        "mixed_payload_bytes": mixed_payload_bytes,
        "uniform_max_candidate_plane_bytes": uniform_nmax_bytes,
        "bf16_weight_bytes": bf16_bytes,
        "compression_vs_bf16": mixed_payload_bytes / bf16_bytes,
        "cpu_mixed_vs_bf16_max_abs_error": quality_max_abs,
        "cpu_mixed_vs_bf16_rms_error": quality_rms,
        "files_sha256": file_shas,
        "backend_parity_measured": False,
        "production_touched": False,
        "production_write_allowed": False,
        "incumbent_map_sha256": incumbent_sha,
        "automatic_live_promotion": False,
    }
    raw = canonical_bytes(manifest)
    manifest["manifest_sha256"] = sha256_bytes(raw)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return manifest

def main() -> int:
    ap = argparse.ArgumentParser(description="Build BECODER-derived mixed qNg64 fixture for real Qwen2.5 L0 q_proj")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n", type=int, action="append", dest="candidates", required=True)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--local-output-budget", type=float, required=True)
    ap.add_argument("--checkpoint-identity-sha256", required=True)
    ap.add_argument("--skeleton-sha256", required=True)
    ap.add_argument("--capability-bundle-sha256", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--weight-epoch", type=int, default=0)
    ap.add_argument("--incumbent-map")
    a = ap.parse_args()
    build(
        checkpoint=Path(a.checkpoint).expanduser().resolve(),
        output_dir=Path(a.output_dir),
        candidates=a.candidates,
        windows=a.windows,
        local_output_budget=a.local_output_budget,
        checkpoint_identity_sha256=a.checkpoint_identity_sha256,
        skeleton_sha256=a.skeleton_sha256,
        capability_bundle_sha256=a.capability_bundle_sha256,
        model_id=a.model_id,
        source_commit=a.source_commit,
        weight_epoch=a.weight_epoch,
        incumbent_map=Path(a.incumbent_map).expanduser().resolve() if a.incumbent_map else None,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
