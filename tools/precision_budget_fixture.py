#!/usr/bin/env python3
"""Deterministic SYNTHETIC BF16 group-64 calibration fixture.

This does not use model weights or measure GPU kernels. Error is a synthetic
quantize/dequantize reconstruction proxy, *never* real task quality.
"""
import argparse
import array
import hashlib
import json
import random
import struct
from pathlib import Path

from qng64_storage import check_backend_n
from precision_budget_search import SCHEMA


def bf16_truncate(x):
    raw = struct.pack("<f", x)
    # The original XOX quant_sim_n.py widens BF16 by shifting 16 bits; no BF16
    # RNE is implied here. Match the stated truncation semantics.
    bits = int.from_bytes(raw, "little") & 0xFFFF0000
    return struct.unpack("<f", bits.to_bytes(4, "little"))[0]


def quant_sim_error(weights, n):
    """Reference sequential RTN + error-feedback, 64 weights per scale."""
    qmax = (1 << (n - 1)) - 1
    qmin = -(1 << (n - 1))
    err2, weight2 = 0.0, 0.0
    for offset in range(0, len(weights), 64):
        group = weights[offset : offset + 64]
        peak = max(abs(v) for v in group)
        scale = peak / qmax if peak > 1e-12 else 1.0
        inv = 1.0 / scale
        residual = 0.0
        for v in group:
            x = v + residual
            code = min(qmax, max(qmin, round(x * inv)))
            deq = code * scale
            err2 += (v - deq) ** 2
            weight2 += v * v
            residual = x - deq
    return (err2 / weight2) ** 0.5 if weight2 else 0.0


def generate(backend="mlx_metal", seed=20261017, budget_frac=0.5):
    roles = (
        ("q_proj", 0, 3.0), ("kv_a_proj_with_mqa", 0, 2.0),
        ("shared_up_proj", 0, 1.0), ("shared_down_proj", 0, 3.0),
        ("q_proj", 1, 2.0), ("shared_down_proj", 1, 1.0),
    )
    widths = (2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    bits = [n for n in widths if not any_exception(backend, n)]
    rng = random.Random(seed)
    tensors = []
    h = hashlib.sha256()
    for i, (role, layer, importance) in enumerate(roles):
        sigma = 0.07 + i * 0.008
        weights = array.array("f", (
            bf16_truncate(rng.gauss(0, sigma) * (3.0 if k % 173 == 0 else 1.0))
            for k in range(128 * 16)))
        for value in weights:
            h.update(struct.pack("<f", value))
        tensors.append({"name": f"synthetic.layers.{layer}.{role}.weight",
                        "role": role, "layer": layer, "elements": len(weights),
                        "importance": importance, "candidates": [
                            {"n": n, "rel_l2": quant_sim_error(weights, n)} for n in bits
                        ]})
    raw_bf16 = sum(2 * t["elements"] for t in tensors)
    return {
        "schema": SCHEMA, "model_id": "SYNTHETIC-deepseek-role-pattern-NOT-MODEL",
        "checkpoint_sha256": h.hexdigest(), "backend": backend,
        "budget_bytes": int(raw_bf16 * budget_frac), "tensors": tensors,
    }


def any_exception(backend, n):
    try:
        check_backend_n(backend, n)
        return False
    except ValueError:
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--backend", choices=["cpu", "mlx_metal"], default="mlx_metal")
    p.add_argument("--budget-frac", type=float, default=0.5)
    args = p.parse_args()
    if not 0.25 <= args.budget_frac <= 1.0:
        p.error("--budget-frac must be 0.25..1.0")
    args.output.write_text(json.dumps(generate(args.backend, budget_frac=args.budget_frac), indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
