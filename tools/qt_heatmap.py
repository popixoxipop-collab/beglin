#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

Q_SCHEMA = "beglin-qheatmap-v2"
T_SCHEMA = "beglin-theatmap-v2"

def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha256_obj(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value

def _identity(event: dict[str, Any]) -> dict[str, Any]:
    identity = event.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("event.identity must be an object")
    required = (
        "checkpoint_identity_sha256",
        "skeleton_sha256",
        "capability_bundle_sha256",
        "target_key",
        "group_size",
        "backend",
        "weight_epoch",
        "observation_window_id",
        "observation_seq",
    )
    missing = [k for k in required if k not in identity]
    if missing:
        raise ValueError(f"identity missing fields: {missing}")
    if int(identity["group_size"]) != 64:
        raise ValueError("QT v1 only permits qNg64 group_size=64")
    if identity.get("qgroup_index") is None:
        raise ValueError("QT v1 requires qgroup_index")
    return identity

def _boundary(identity: dict[str, Any]) -> tuple[Any, ...]:
    return (
        identity["checkpoint_identity_sha256"],
        identity["skeleton_sha256"],
        identity["capability_bundle_sha256"],
        identity["target_key"],
        int(identity["group_size"]),
        int(identity["weight_epoch"]),
    )

def _global_generation(events: list[dict[str, Any]]) -> tuple[str, str, int]:
    if not events:
        raise ValueError("at least one event is required")
    identities = [_identity(e) for e in events]
    triples = {
        (
            i["checkpoint_identity_sha256"],
            i["skeleton_sha256"],
            int(i["weight_epoch"]),
        )
        for i in identities
    }
    if len(triples) != 1:
        raise ValueError("events cross checkpoint/skeleton/weight_epoch generations")
    return next(iter(triples))

def _sorted_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized = list(events)
    return sorted(
        materialized,
        key=lambda e: (
            _identity(e)["target_key"],
            int(_identity(e)["qgroup_index"]),
            _identity(e)["backend"],
            _identity(e)["observation_window_id"],
            int(_identity(e)["observation_seq"]),
            canonical_json(e),
        ),
    )

def _check_duplicate_sequence(events: list[dict[str, Any]]) -> None:
    seen: set[tuple[Any, ...]] = set()
    for event in events:
        i = _identity(event)
        key = (
            *_boundary(i),
            i["backend"],
            i["observation_window_id"],
            int(i["observation_seq"]),
            event.get("schema"),
            event.get("candidate_n"),
        )
        if key in seen:
            raise ValueError(f"duplicate observation sequence: {key}")
        seen.add(key)

def _weighted_mean(rows: list[dict[str, Any]], field: str) -> float:
    total = 0.0
    weight = 0
    for row in rows:
        if field not in row:
            continue
        n = int(row.get("sample_count", 1))
        total += _require_finite(field, row[field]) * n
        weight += n
    return total / weight if weight else 0.0

def _weighted_variance(rows: list[dict[str, Any]], field: str, mean: float) -> float:
    total = 0.0
    weight = 0
    for row in rows:
        if field not in row:
            continue
        n = int(row.get("sample_count", 1))
        x = _require_finite(field, row[field])
        total += ((x - mean) ** 2) * n
        weight += n
    return total / weight if weight else 0.0

def _confidence(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    windows = {str(_identity(r)["observation_window_id"]) for r in rows}
    finite_ratio = sum(1 for r in rows if r.get("finite", True)) / len(rows)
    pass_ratio = sum(1 for r in rows if r.get("result_status", "PASS") == "PASS") / len(rows)
    window_score = min(1.0, len(windows) / 2.0)
    coverage_score = min(1.0, sum(int(r.get("sample_count", 1)) for r in rows) / 2.0)
    return round(window_score * coverage_score * finite_ratio * pass_ratio, 9)

@dataclass(frozen=True)
class QBudgets:
    output_max_abs_error: float = 5e-4
    backend_parity_error: float = 5e-4
    restore_diff: float = 0.0

def build_q_heatmap(
    events: Iterable[dict[str, Any]],
    supported_n_by_target: dict[str, list[int]] | None = None,
    current_n_by_target: dict[str, int] | None = None,
    budgets: QBudgets = QBudgets(),
    generation: int = 0,
) -> dict[str, Any]:
    rows = _sorted_events(events)
    _check_duplicate_sequence(rows)
    checkpoint, skeleton, weight_epoch = _global_generation(rows)
    supported_n_by_target = supported_n_by_target or {}
    current_n_by_target = current_n_by_target or {}

    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    boundaries: dict[str, tuple[Any, ...]] = {}
    for row in rows:
        if row.get("schema") != "beglin-quant-perturbation-v1":
            raise ValueError("Q heatmap accepts only quant perturbation events")
        i = _identity(row)
        target = str(i["target_key"])
        boundary = _boundary(i)
        if target in boundaries and boundaries[target] != boundary:
            raise ValueError(f"identity contamination for target {target}")
        boundaries[target] = boundary
        n = int(row["candidate_n"])
        grouped[target][n].append(row)

    cells = []
    for target in sorted(grouped):
        by_n = grouped[target]
        observed_n = sorted(by_n)
        supported = sorted(set(int(n) for n in supported_n_by_target.get(target, observed_n)))
        errors: dict[str, float] = {}
        parity: dict[str, float] = {}
        safe: list[int] = []
        all_rows: list[dict[str, Any]] = []
        for n in supported:
            nrows = by_n.get(n, [])
            if not nrows:
                continue
            all_rows.extend(nrows)
            max_error = max(_require_finite("output_max_abs_error", r["output_max_abs_error"]) for r in nrows)
            parity_rows = [r for r in nrows if r.get("backend_parity_error") is not None and r.get("restore_diff") is not None]
            errors[str(n)] = max_error
            if parity_rows:
                parity[str(n)] = max(_require_finite("backend_parity_error", r["backend_parity_error"]) for r in parity_rows)
            if (
                all(bool(r.get("finite")) for r in nrows)
                and all(r.get("result_status") == "PASS" for r in nrows)
                and max_error <= budgets.output_max_abs_error
            ):
                safe.append(n)

        recommended = min(safe) if safe else None
        conf = _confidence(all_rows)
        state = "CANDIDATE" if recommended is not None and conf >= 0.75 else ("OBSERVED" if all_rows else "COLD")
        if recommended is None and all_rows:
            state = "QUARANTINED"
        current_n = current_n_by_target.get(target)
        hysteresis = "STABLE"
        if recommended is not None and current_n is not None:
            if recommended < current_n:
                hysteresis = "DOWNGRADE_ARMED"
            elif recommended > current_n:
                hysteresis = "UPGRADE_ARMED"
        backend_verified_by_n = {}
        for n in supported:
            nrows = by_n.get(n, [])
            verified = [
                r for r in nrows
                if r.get("backend_parity_error") is not None
                and r.get("restore_diff") is not None
                and bool(r.get("finite"))
                and r.get("result_status") == "PASS"
                and _require_finite("backend_parity_error", r["backend_parity_error"]) <= budgets.backend_parity_error
                and _require_finite("restore_diff", r["restore_diff"]) <= budgets.restore_diff
            ]
            backend_verified_by_n[str(n)] = bool(verified)
        cells.append(
            {
                "target_key": target,
                "current_n": current_n,
                "supported_n": supported,
                "error_by_n": errors,
                "backend_parity_by_n": parity,
                "backend_verified_by_n": backend_verified_by_n,
                "recommended_n": recommended,
                "minimum_safe_n": recommended,
                "confidence": conf,
                "hysteresis_state": hysteresis,
                "state": state,
                "evidence_refs": [sha256_obj(r) for r in all_rows],
            }
        )

    body = {
        "schema": Q_SCHEMA,
        "checkpoint_identity_sha256": checkpoint,
        "skeleton_sha256": skeleton,
        "weight_epoch": weight_epoch,
        "generation": int(generation),
        "cells": cells,
    }
    body["heatmap_sha256"] = sha256_obj(body)
    return body

def build_t_heatmap(
    events: Iterable[dict[str, Any]],
    generation: int = 0,
) -> dict[str, Any]:
    rows = _sorted_events(events)
    _check_duplicate_sequence(rows)
    checkpoint, skeleton, weight_epoch = _global_generation(rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    boundaries: dict[str, tuple[Any, ...]] = {}
    for row in rows:
        if row.get("schema") != "beglin-training-sensitivity-v1":
            raise ValueError("T heatmap accepts only training sensitivity events")
        i = _identity(row)
        target = str(i["target_key"])
        boundary = _boundary(i)
        if target in boundaries and boundaries[target] != boundary:
            raise ValueError(f"identity contamination for target {target}")
        boundaries[target] = boundary
        grouped[target].append(row)

    metrics = []
    for target in sorted(grouped):
        trows = grouped[target]
        grad_rms = _weighted_mean(trows, "grad_rms")
        grad_var = _weighted_variance(trows, "grad_rms", grad_rms)
        quant_error = _weighted_mean(trows, "quant_error_abs")
        gtqe = _weighted_mean(trows, "gradient_times_quant_error")
        influence = _weighted_mean(trows, "influence_proxy")
        conf = _confidence(trows)
        stability = 1.0 / (1.0 + math.sqrt(max(0.0, grad_var)))
        score = max(0.0, gtqe) * stability * conf
        metrics.append((target, score, grad_rms, grad_var, quant_error, gtqe, influence, conf, trows))

    ranked = sorted(metrics, key=lambda x: (-x[1], x[0]))
    rank_by_target = {row[0]: idx for idx, row in enumerate(ranked)}
    n_targets = max(1, len(ranked))

    cells = []
    for target, score, grad_rms, grad_var, quant_error, gtqe, influence, conf, trows in metrics:
        percentile_from_top = rank_by_target[target] / n_targets
        if conf < 0.75 or not all(bool(r.get("finite")) for r in trows):
            trainable, lr_scale, state = False, 0.0, "OBSERVED"
        elif percentile_from_top < 0.01:
            trainable, lr_scale, state = True, 1.0, "CANDIDATE"
        elif percentile_from_top < 0.05:
            trainable, lr_scale, state = True, 0.7, "CANDIDATE"
        elif percentile_from_top < 0.20:
            trainable, lr_scale, state = True, 0.3, "CANDIDATE"
        elif percentile_from_top < 0.50:
            trainable, lr_scale, state = True, 0.1, "CANDIDATE"
        else:
            trainable, lr_scale, state = False, 0.0, "CANDIDATE"
        cells.append(
            {
                "target_key": target,
                "grad_rms_ema": grad_rms,
                "grad_variance": grad_var,
                "quant_error": quant_error,
                "gradient_times_quant_error": gtqe,
                "influence_proxy": influence,
                "recommended_trainable": trainable,
                "lr_scale": lr_scale,
                "weight_decay_scale": 1.0 if trainable else 0.0,
                "confidence": conf,
                "state": state,
                "evidence_refs": [sha256_obj(r) for r in trows],
            }
        )

    cells.sort(key=lambda c: c["target_key"])
    body = {
        "schema": T_SCHEMA,
        "checkpoint_identity_sha256": checkpoint,
        "skeleton_sha256": skeleton,
        "weight_epoch": weight_epoch,
        "generation": int(generation),
        "cells": cells,
    }
    body["heatmap_sha256"] = sha256_obj(body)
    return body

def build_local_precision_candidate(
    qheatmap: dict[str, Any],
    policy_id: str,
    qgroup_index_by_target: dict[str, int],
    current_n_by_target: dict[str, int],
    default_precision: str = "BF16",
) -> dict[str, Any]:
    cells = []
    for cell in qheatmap["cells"]:
        target = cell["target_key"]
        recommended = cell.get("recommended_n")
        if recommended is None or cell.get("state") != "CANDIDATE":
            continue
        if target not in qgroup_index_by_target:
            raise ValueError(f"missing qgroup index for {target}")
        cells.append(
            {
                "target_key": target,
                "qgroup_index": int(qgroup_index_by_target[target]),
                "group_size": 64,
                "n": int(recommended),
                "confidence": float(cell["confidence"]),
            }
        )
    return {
        "schema": "beglin-local-precision-policy-v1",
        "policy_id": policy_id,
        "checkpoint_identity_sha256": qheatmap["checkpoint_identity_sha256"],
        "skeleton_sha256": qheatmap["skeleton_sha256"],
        "heatmap_sha256": qheatmap["heatmap_sha256"],
        "weight_epoch": qheatmap["weight_epoch"],
        "generation": qheatmap["generation"],
        "default_precision": default_precision,
        "approved_by_beval": False,
        "cells": sorted(cells, key=lambda c: (c["target_key"], c["qgroup_index"])),
    }

def build_selective_training_candidate(
    theatmap: dict[str, Any],
    policy_id: str,
    master_precision: str = "BF16",
) -> dict[str, Any]:
    cells = []
    for cell in theatmap["cells"]:
        cells.append(
            {
                "target_key": cell["target_key"],
                "trainable": bool(cell["recommended_trainable"]),
                "lr_scale": float(cell["lr_scale"]),
                "weight_decay_scale": float(cell["weight_decay_scale"]),
                "master_precision": master_precision,
                "confidence": float(cell["confidence"]),
            }
        )
    return {
        "schema": "beglin-selective-training-policy-v1",
        "policy_id": policy_id,
        "checkpoint_identity_sha256": theatmap["checkpoint_identity_sha256"],
        "skeleton_sha256": theatmap["skeleton_sha256"],
        "heatmap_sha256": theatmap["heatmap_sha256"],
        "weight_epoch": theatmap["weight_epoch"],
        "generation": theatmap["generation"],
        "approved_by_beval": False,
        "cells": sorted(cells, key=lambda c: c["target_key"]),
    }

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return rows

def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic Beglin Q/T heatmaps from JSONL evidence")
    parser.add_argument("kind", choices=("q", "t"))
    parser.add_argument("events", type=Path)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = _load_jsonl(args.events)
    result = build_q_heatmap(rows, generation=args.generation) if args.kind == "q" else build_t_heatmap(rows, generation=args.generation)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
