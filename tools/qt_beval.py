#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Iterable

@dataclass(frozen=True)
class QBevalLimits:
    output_max_abs_error: float = 5e-4
    backend_parity_error: float = 5e-4
    restore_max_abs_diff: float = 0.0
    task_metric_regression_budget: float = 0.0

@dataclass(frozen=True)
class TBevalLimits:
    frozen_leakage_abs: float = 0.0
    holdout_metric_regression_budget: float = 0.0

def _finite_number(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value

def _check_common(policy: dict[str, Any], heatmap: dict[str, Any]) -> None:
    if policy.get("checkpoint_identity_sha256") != heatmap.get("checkpoint_identity_sha256"):
        raise ValueError("checkpoint identity mismatch")
    if policy.get("skeleton_sha256") != heatmap.get("skeleton_sha256"):
        raise ValueError("skeleton identity mismatch")
    if policy.get("heatmap_sha256") != heatmap.get("heatmap_sha256"):
        raise ValueError("heatmap identity mismatch")
    if int(policy.get("weight_epoch", -1)) != int(heatmap.get("weight_epoch", -2)):
        raise ValueError("weight_epoch mismatch")
    if policy.get("approved_by_beval") is True:
        raise ValueError("input policy is already approved")

def _evidence_map(
    evidence: Iterable[dict[str, Any]],
    *,
    kind: str,
    policy: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    out = {}
    for row in evidence:
        if row.get("schema") != "beglin-qt-beval-evidence-v1":
            raise ValueError("unexpected evidence schema")
        if row.get("kind") != kind:
            continue
        if row.get("checkpoint_identity_sha256") != policy.get("checkpoint_identity_sha256"):
            raise ValueError("evidence checkpoint mismatch")
        if row.get("skeleton_sha256") != policy.get("skeleton_sha256"):
            raise ValueError("evidence skeleton mismatch")
        if row.get("heatmap_sha256") != policy.get("heatmap_sha256"):
            raise ValueError("evidence heatmap mismatch")
        if int(row.get("weight_epoch", -1)) != int(policy.get("weight_epoch", -2)):
            raise ValueError("evidence weight_epoch mismatch")
        target = str(row.get("target_key", ""))
        if not target:
            raise ValueError("evidence target missing")
        if target in out:
            raise ValueError(f"duplicate BEVAL evidence for {target}")
        out[target] = row
    return out

def approve_q_policy(
    policy: dict[str, Any],
    qheatmap: dict[str, Any],
    evidence: Iterable[dict[str, Any]],
    limits: QBevalLimits = QBevalLimits(),
) -> dict[str, Any]:
    if policy.get("schema") != "beglin-local-precision-policy-v1":
        raise ValueError("unexpected Q policy schema")
    if qheatmap.get("schema") != "beglin-qheatmap-v2":
        raise ValueError("unexpected Q heatmap schema")
    _check_common(policy, qheatmap)
    emap = _evidence_map(evidence, kind="Q", policy=policy)
    heatmap_cells = {c["target_key"]: c for c in qheatmap.get("cells", [])}

    for cell in policy.get("cells", []):
        target = cell["target_key"]
        hcell = heatmap_cells.get(target)
        if hcell is None:
            raise ValueError(f"Q target absent from heatmap: {target}")
        if hcell.get("state") not in ("CANDIDATE", "APPROVED"):
            raise ValueError(f"Q target not eligible: {target}")
        if int(cell["n"]) != int(hcell.get("recommended_n")):
            raise ValueError(f"Q precision differs from heatmap recommendation: {target}")
        row = emap.get(target)
        if row is None:
            raise ValueError(f"Q BEVAL evidence missing: {target}")
        if row.get("status") != "PASS":
            raise ValueError(f"Q BEVAL did not pass: {target} status={row.get('status')}")
        m = row.get("metrics") or {}
        if int(m.get("n", -1)) != int(cell["n"]):
            raise ValueError(f"Q BEVAL n mismatch: {target}")
        if _finite_number(m.get("output_max_abs_error"), "output_max_abs_error") > limits.output_max_abs_error:
            raise ValueError(f"Q output error budget exceeded: {target}")
        if _finite_number(m.get("backend_parity_error"), "backend_parity_error") > limits.backend_parity_error:
            raise ValueError(f"Q backend parity budget exceeded: {target}")
        if _finite_number(m.get("restore_max_abs_diff"), "restore_max_abs_diff") > limits.restore_max_abs_diff:
            raise ValueError(f"Q restore budget exceeded: {target}")
        if _finite_number(m.get("task_metric_delta", 0.0), "task_metric_delta") < -limits.task_metric_regression_budget:
            raise ValueError(f"Q task metric regression: {target}")
        if not bool(m.get("finite", False)):
            raise ValueError(f"Q non-finite evidence: {target}")

    approved = copy.deepcopy(policy)
    approved["approved_by_beval"] = True
    approved["beval_evidence_ids"] = sorted(emap[t["target_key"]]["evidence_id"] for t in policy.get("cells", []))
    return approved

def approve_t_policy(
    policy: dict[str, Any],
    theatmap: dict[str, Any],
    evidence: Iterable[dict[str, Any]],
    limits: TBevalLimits = TBevalLimits(),
) -> dict[str, Any]:
    if policy.get("schema") != "beglin-selective-training-policy-v1":
        raise ValueError("unexpected T policy schema")
    if theatmap.get("schema") != "beglin-theatmap-v2":
        raise ValueError("unexpected T heatmap schema")
    _check_common(policy, theatmap)
    emap = _evidence_map(evidence, kind="T", policy=policy)
    heatmap_cells = {c["target_key"]: c for c in theatmap.get("cells", [])}

    for cell in policy.get("cells", []):
        target = cell["target_key"]
        hcell = heatmap_cells.get(target)
        if hcell is None:
            raise ValueError(f"T target absent from heatmap: {target}")
        if bool(cell["trainable"]) != bool(hcell.get("recommended_trainable")):
            raise ValueError(f"T trainability differs from heatmap recommendation: {target}")
        row = emap.get(target)
        if row is None:
            raise ValueError(f"T BEVAL evidence missing: {target}")
        if row.get("status") != "PASS":
            raise ValueError(f"T BEVAL did not pass: {target} status={row.get('status')}")
        m = row.get("metrics") or {}
        frozen_leakage = _finite_number(m.get("frozen_leakage_abs", 0.0), "frozen_leakage_abs")
        holdout_delta = _finite_number(m.get("holdout_metric_delta", 0.0), "holdout_metric_delta")
        selected_update = _finite_number(m.get("selected_update_abs", 0.0), "selected_update_abs")
        if frozen_leakage > limits.frozen_leakage_abs:
            raise ValueError(f"T frozen-mask leakage: {target}")
        if holdout_delta < -limits.holdout_metric_regression_budget:
            raise ValueError(f"T holdout regression: {target}")
        if bool(cell["trainable"]) and selected_update <= 0.0:
            raise ValueError(f"T selected cell did not update: {target}")
        if not bool(m.get("post_q_pass", False)):
            raise ValueError(f"T post-training Q gate missing: {target}")
        if not bool(m.get("lineage_complete", False)):
            raise ValueError(f"T checkpoint lineage incomplete: {target}")
        if not bool(m.get("finite", False)):
            raise ValueError(f"T non-finite evidence: {target}")

    approved = copy.deepcopy(policy)
    approved["approved_by_beval"] = True
    approved["beval_evidence_ids"] = sorted(emap[t["target_key"]]["evidence_id"] for t in policy.get("cells", []))
    return approved
