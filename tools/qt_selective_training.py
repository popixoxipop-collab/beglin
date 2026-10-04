#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
import re
from array import array
from dataclasses import dataclass
from typing import Any, Iterable

GROUP = 64
_TARGET_RE = re.compile(r"/row=(\d+)/group64=(\d+)$")

@dataclass(frozen=True)
class UpdateStats:
    changed_targets: tuple[str, ...]
    changed_group_count: int
    selected_weight_count: int
    frozen_weight_count: int
    selected_update_max_abs: float
    selected_update_mean_abs: float
    frozen_leakage_max_abs: float

def parse_group_target(target_key: str) -> tuple[int, int]:
    m = _TARGET_RE.search(target_key)
    if not m:
        raise ValueError(f"target key lacks row/group64 suffix: {target_key}")
    return int(m.group(1)), int(m.group(2))

def _check_shape(values: Iterable[float], out_dim: int, in_dim: int) -> list[float]:
    vals = [float(x) for x in values]
    if out_dim <= 0 or in_dim <= 0 or in_dim % GROUP:
        raise ValueError("invalid qNg64 matrix shape")
    if len(vals) != out_dim * in_dim:
        raise ValueError("matrix length does not match shape")
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("matrix contains non-finite values")
    return vals

def apply_selective_sgd(
    master: Iterable[float],
    gradient: Iterable[float],
    *,
    out_dim: int,
    in_dim: int,
    policy: dict[str, Any],
    base_lr: float,
) -> tuple[array, UpdateStats]:
    if policy.get("schema") != "beglin-selective-training-policy-v1":
        raise ValueError("unexpected selective-training policy schema")
    if policy.get("approved_by_beval") is True:
        # An approved policy is legal too; this guard only prevents missing/foreign schema.
        pass
    if not math.isfinite(base_lr) or base_lr < 0.0:
        raise ValueError("base_lr must be finite and non-negative")

    before = _check_shape(master, out_dim, in_dim)
    grad = _check_shape(gradient, out_dim, in_dim)
    after = array("f", before)
    seen: set[tuple[int, int]] = set()
    changed_targets: list[str] = []
    selected_weight_count = 0
    selected_update_sum = 0.0
    selected_update_max = 0.0
    frozen_weight_count = 0
    frozen_leakage = 0.0

    cells = policy.get("cells")
    if not isinstance(cells, list):
        raise ValueError("policy.cells must be a list")

    for cell in cells:
        target = str(cell["target_key"])
        row, group = parse_group_target(target)
        if not (0 <= row < out_dim and 0 <= group < in_dim // GROUP):
            raise ValueError(f"policy cell out of range: {target}")
        key = (row, group)
        if key in seen:
            raise ValueError(f"duplicate policy cell: {target}")
        seen.add(key)
        trainable = bool(cell["trainable"])
        lr_scale = float(cell["lr_scale"])
        if not math.isfinite(lr_scale) or lr_scale < 0.0:
            raise ValueError(f"invalid lr_scale: {target}")
        if not trainable and lr_scale != 0.0:
            raise ValueError(f"frozen cell has non-zero lr_scale: {target}")

        start = row * in_dim + group * GROUP
        if trainable:
            changed = False
            for p in range(GROUP):
                idx = start + p
                delta = -base_lr * lr_scale * grad[idx]
                if not math.isfinite(delta):
                    raise ValueError(f"non-finite update: {target}")
                after[idx] = before[idx] + delta
                a = abs(delta)
                selected_update_sum += a
                selected_update_max = max(selected_update_max, a)
                selected_weight_count += 1
                changed = changed or a != 0.0
            if changed:
                changed_targets.append(target)
        else:
            for p in range(GROUP):
                idx = start + p
                frozen_leakage = max(frozen_leakage, abs(float(after[idx]) - before[idx]))
                frozen_weight_count += 1

    expected_cells = out_dim * (in_dim // GROUP)
    if len(seen) != expected_cells:
        raise ValueError(f"policy coverage incomplete: {len(seen)}/{expected_cells} cells")

    mean_update = selected_update_sum / selected_weight_count if selected_weight_count else 0.0
    return after, UpdateStats(
        changed_targets=tuple(sorted(changed_targets)),
        changed_group_count=len(changed_targets),
        selected_weight_count=selected_weight_count,
        frozen_weight_count=frozen_weight_count,
        selected_update_max_abs=selected_update_max,
        selected_update_mean_abs=mean_update,
        frozen_leakage_max_abs=frozen_leakage,
    )

def qng64_encode_group(values: Iterable[float], n: int) -> tuple[bytes, float]:
    xs = [float(x) for x in values]
    if len(xs) != GROUP:
        raise ValueError("qNg64 encode requires 64 values")
    if n < 2 or n > 15:
        raise ValueError("qNg64 n must be in [2,15]")
    qmax = (1 << (n - 1)) - 1
    qmin = -(1 << (n - 1))
    mx = max(abs(x) for x in xs)
    scale = mx / qmax if mx > 1e-12 else 1.0
    inv = 1.0 / scale
    residual = 0.0
    codes: list[int] = []
    for x in xs:
        corrected = x + residual
        q = max(qmin, min(qmax, int(round(corrected * inv))))
        dq = q * scale
        residual = corrected - dq
        codes.append(q)

    planes = bytearray(n * 8)
    bias = 1 << (n - 1)
    for p, q in enumerate(codes):
        u = q + bias
        for j in range(n):
            if (u >> j) & 1:
                planes[j * 8 + (p >> 3)] |= 1 << (p & 7)
    return bytes(planes), scale

def requantize_changed_groups(
    master: Iterable[float],
    *,
    out_dim: int,
    in_dim: int,
    changed_targets: Iterable[str],
    n_by_target: dict[str, int] | None = None,
    default_n: int = 5,
) -> dict[str, Any]:
    vals = _check_shape(master, out_dim, in_dim)
    n_by_target = n_by_target or {}
    rows = []
    seen = set()
    total_bytes = 0
    for target in sorted(changed_targets):
        row, group = parse_group_target(target)
        if not (0 <= row < out_dim and 0 <= group < in_dim // GROUP):
            raise ValueError(f"changed target out of range: {target}")
        if (row, group) in seen:
            raise ValueError(f"duplicate changed target: {target}")
        seen.add((row, group))
        n = int(n_by_target.get(target, default_n))
        start = row * in_dim + group * GROUP
        planes, scale = qng64_encode_group(vals[start:start + GROUP], n)
        total_bytes += len(planes) + 4
        rows.append({
            "target_key": target,
            "n": n,
            "scale": scale,
            "planes_sha256": hashlib.sha256(planes).hexdigest(),
            "planes_bytes": len(planes),
        })
    canonical = repr([(r["target_key"], r["n"], r["scale"], r["planes_sha256"]) for r in rows]).encode()
    return {
        "schema": "beglin-selective-requantization-v1",
        "group_size": GROUP,
        "changed_group_count": len(rows),
        "payload_bytes": total_bytes,
        "groups": rows,
        "requantization_sha256": hashlib.sha256(canonical).hexdigest(),
    }

def build_training_sensitivity_event(
    *,
    identity: dict[str, Any],
    group_weights: Iterable[float],
    group_gradients: Iterable[float],
    n: int,
    optimizer_step: int,
    learning_rate: float,
    loss_before: float,
) -> dict[str, Any]:
    w = [float(x) for x in group_weights]
    g = [float(x) for x in group_gradients]
    if len(w) != GROUP or len(g) != GROUP:
        raise ValueError("training sensitivity is group64-scoped")
    planes, scale = qng64_encode_group(w, n)
    del planes
    qmax = (1 << (n - 1)) - 1
    qmin = -(1 << (n - 1))
    inv = 1.0 / scale
    residual = 0.0
    qerr = []
    for x in w:
        corrected = x + residual
        q = max(qmin, min(qmax, int(round(corrected * inv))))
        dq = q * scale
        residual = corrected - dq
        qerr.append(abs(dq - x))
    grad_abs = [abs(x) for x in g]
    grad_rms = math.sqrt(sum(x * x for x in g) / GROUP)
    grad_mean = sum(grad_abs) / GROUP
    grad_max = max(grad_abs)
    grad_norm = math.sqrt(sum(x * x for x in g))
    qe_mean = sum(qerr) / GROUP
    gtqe = sum(abs(g[i]) * qerr[i] for i in range(GROUP)) / GROUP
    influence = grad_rms * qe_mean
    return {
        "schema": "beglin-training-sensitivity-v1",
        "identity": identity,
        "loss_before": float(loss_before),
        "loss_after": None,
        "grad_abs_mean": grad_mean,
        "grad_rms": grad_rms,
        "grad_max_abs": grad_max,
        "grad_norm": grad_norm,
        "grad_variance_ema": 0.0,
        "quant_error_abs": qe_mean,
        "gradient_times_quant_error": gtqe,
        "fisher_diag_ema": sum(x * x for x in g) / GROUP,
        "influence_proxy": influence,
        "optimizer_step": int(optimizer_step),
        "learning_rate": float(learning_rate),
        "sample_count": 1,
        "finite": all(math.isfinite(x) for x in (loss_before, grad_mean, grad_rms, grad_max, qe_mean, gtqe, influence)),
    }
