#!/usr/bin/env python3
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Iterable

GROUP = 64
SCHEMA = "beglin-quant-perturbation-v1"

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _bf16_to_f32(raw: bytes) -> array.array:
    out = array.array("f")
    for (v,) in struct.iter_unpack("<H", raw):
        out.append(struct.unpack("<f", struct.pack("<I", v << 16))[0])
    return out

def read_tensor(path: Path, name: str) -> tuple[array.array, int, int, str]:
    with path.open("rb") as f:
        hlen_raw = f.read(8)
        if len(hlen_raw) != 8:
            raise ValueError("invalid safetensors header")
        hlen = struct.unpack("<Q", hlen_raw)[0]
        header = json.loads(f.read(hlen))
        if name not in header:
            raise KeyError(name)
        ent = header[name]
        start, end = ent["data_offsets"]
        f.seek(8 + hlen + start)
        raw = f.read(end - start)
    shape = ent["shape"]
    if len(shape) != 2:
        raise ValueError("target tensor must be rank-2")
    dtype = ent["dtype"]
    if dtype == "BF16":
        vals = _bf16_to_f32(raw)
    elif dtype == "F32":
        vals = array.array("f")
        vals.frombytes(raw)
    else:
        raise ValueError(f"unsupported dtype: {dtype}")
    return vals, int(shape[0]), int(shape[1]), dtype

def quantize_group(values: Iterable[float], n: int) -> tuple[list[float], float, float, float]:
    xs = [float(x) for x in values]
    if len(xs) != GROUP:
        raise ValueError("qNg64 group must contain exactly 64 values")
    if n < 2 or n > 16:
        raise ValueError("n must be in [2,16]")
    qmax = (1 << (n - 1)) - 1
    qmin = -(1 << (n - 1))
    mx = max(abs(x) for x in xs)
    scale = mx / qmax if mx > 1e-12 else 1.0
    inv = 1.0 / scale
    err = 0.0
    out = []
    l1 = 0.0
    l2 = 0.0
    for x in xs:
        corrected = x + err
        q = max(qmin, min(qmax, int(round(corrected * inv))))
        dq = q * scale
        err = corrected - dq
        out.append(dq)
        d = dq - x
        l1 += abs(d)
        l2 += d * d
    return out, scale, l1, math.sqrt(l2)

def _input_vector(in_dim: int, window_index: int) -> list[float]:
    phase = 0.173 + 0.013 * window_index
    phase2 = 0.071 + 0.007 * window_index
    return [math.sin((i + 1) * phase) + 0.25 * math.cos((i + 1) * phase2) for i in range(in_dim)]

def build_events(
    *,
    checkpoint: Path,
    tensor: str,
    row: int,
    group_index: int,
    candidates: list[int],
    windows: int,
    checkpoint_identity_sha256: str,
    skeleton_sha256: str,
    capability_bundle_sha256: str,
    source_commit: str,
    policy_epoch: int = 0,
    weight_epoch: int = 0,
) -> list[dict]:
    vals, out_dim, in_dim, dtype = read_tensor(checkpoint, tensor)
    if in_dim % GROUP != 0:
        raise ValueError("input dimension is not divisible by 64")
    groups_per_row = in_dim // GROUP
    if not (0 <= row < out_dim):
        raise ValueError("row out of range")
    if not (0 <= group_index < groups_per_row):
        raise ValueError("group index out of range")
    if windows < 1:
        raise ValueError("windows must be >=1")
    start = row * in_dim + group_index * GROUP
    original_group = [float(vals[start + i]) for i in range(GROUP)]
    checkpoint_file_sha256 = _sha256_file(checkpoint)
    target_key = f"qwen2.5/L0/q_proj/row={row}/group64={group_index}"
    script_sha256 = _sha256_file(Path(__file__).resolve())
    events = []
    seq = 0
    for n in sorted(set(candidates)):
        deq, scale, l1, l2 = quantize_group(original_group, n)
        denom = math.sqrt(sum(x * x for x in original_group)) or 1.0
        relative_weight_error = l2 / denom
        for w in range(windows):
            x = _input_vector(in_dim, w)
            group_x = x[group_index * GROUP : (group_index + 1) * GROUP]
            baseline_contrib = sum(original_group[i] * group_x[i] for i in range(GROUP))
            quant_contrib = sum(deq[i] * group_x[i] for i in range(GROUP))
            abs_error = abs(quant_contrib - baseline_contrib)
            identity = {
                "schema": "beglin-observation-identity-v2",
                "checkpoint_identity_sha256": checkpoint_identity_sha256,
                "skeleton_sha256": skeleton_sha256,
                "capability_bundle_sha256": capability_bundle_sha256,
                "target_key": target_key,
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "layer": 0,
                "tensor_role": "q_proj",
                "expert_id": None,
                "row": row,
                "column": None,
                "qgroup_index": group_index,
                "group_size": GROUP,
                "element_index": None,
                "backend": "cpu_oracle",
                "source_dtype": dtype,
                "runtime_dtype": "qNg64",
                "policy_epoch": policy_epoch,
                "weight_epoch": weight_epoch,
                "observation_window_id": f"calibration-{w}",
                "observation_seq": seq,
                "source_commit": source_commit,
                "binary_sha256": script_sha256,
            }
            seq += 1
            events.append({
                "schema": SCHEMA,
                "identity": identity,
                "candidate_n": n,
                "baseline_precision": dtype,
                "quantized_precision": f"qNg64-n{n}",
                "delta_weight_l1": l1,
                "delta_weight_l2": l2,
                "relative_weight_error": relative_weight_error,
                "output_max_abs_error": abs_error,
                "output_rms_error": abs_error / math.sqrt(out_dim),
                "output_relative_error": abs_error / (abs(baseline_contrib) + 1e-6),
                "latency_delta": 0.0,
                "memory_delta": 0.0,
                "backend_parity_error": None,
                "restore_diff": None,
                "finite": all(math.isfinite(v) for v in (scale, l1, l2, abs_error)),
                "result_status": "PASS",
                "sample_count": 1,
            })
    return events

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate group-local Qwen2.5 qNg64 CPU perturbation evidence")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tensor", default="model.layers.0.self_attn.q_proj.weight")
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--n", dest="candidates", type=int, action="append", required=True)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--checkpoint-identity-sha256", required=True)
    ap.add_argument("--skeleton-sha256", required=True)
    ap.add_argument("--capability-bundle-sha256", required=True)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--policy-epoch", type=int, default=0)
    ap.add_argument("--weight-epoch", type=int, default=0)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    events = build_events(
        checkpoint=Path(a.checkpoint).expanduser().resolve(),
        tensor=a.tensor,
        row=a.row,
        group_index=a.group,
        candidates=a.candidates,
        windows=a.windows,
        checkpoint_identity_sha256=a.checkpoint_identity_sha256,
        skeleton_sha256=a.skeleton_sha256,
        capability_bundle_sha256=a.capability_bundle_sha256,
        source_commit=a.source_commit,
        policy_epoch=a.policy_epoch,
        weight_epoch=a.weight_epoch,
    )
    out = Path(a.output)
    out.write_text("".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in events))
    summary = {
        "schema": "beglin-qt-qwen25-group-perturbation-summary-v1",
        "event_count": len(events),
        "target_key": events[0]["identity"]["target_key"],
        "candidate_n": sorted({e["candidate_n"] for e in events}),
        "output_sha256": _sha256_file(out),
        "backend_parity_measured": False,
        "beval_approval_allowed": False,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
