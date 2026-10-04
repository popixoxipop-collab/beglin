#!/usr/bin/env python3
"""P12 capability gate for the existing P8-P11 precision pipeline.

The bridge is pure/read-only.  It binds one exact ModelCapabilityBundle and
target/backend capability to downstream artifacts without enabling production
mutation or automatic promotion.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

import model_capability as mc


class P12PipelineBridgeError(RuntimeError):
    pass


CANARY_BY_MUTATION_MODE = {
    "HOT_REBIND_SINGLE": "SAME_WORKER_PRECISION_EPOCH",
    "HOT_REBIND_MULTI": "SAME_WORKER_PRECISION_EPOCH",
    "RESTART_REQUIRED": "ISOLATED_RESTART_CANARY",
    "IMMUTABLE": "DENIED",
}


def validate_bundle(bundle: Mapping[str, Any]) -> dict:
    value = copy.deepcopy(dict(bundle))
    if value.get("schema") != "beglin-model-capability-bundle-v1":
        raise P12PipelineBridgeError("unsupported ModelCapabilityBundle schema")
    try:
        mc.verify_identity(value, "bundle_sha256")
    except Exception as exc:
        raise P12PipelineBridgeError("ModelCapabilityBundle identity mismatch") from exc
    if value.get("p8_p11_eligibility") not in {"FULL", "PARTIAL", "DENIED"}:
        raise P12PipelineBridgeError("invalid P8-P11 eligibility")
    return value


def _target_cell(bundle: Mapping[str, Any], *, target_key: str, backend: str) -> dict:
    matches = [
        dict(cell)
        for cell in bundle.get("backend_matrix", [])
        if cell.get("target_key") == target_key and cell.get("backend") == backend
    ]
    if len(matches) != 1:
        raise P12PipelineBridgeError(
            f"expected exactly one backend capability cell for {target_key}/{backend}"
        )
    return matches[0]


def build_p8_capability_gate(
    *,
    model_capability_bundle: Mapping[str, Any],
    target_key: str,
    backend: str,
    requested_n: int,
) -> dict:
    bundle = validate_bundle(model_capability_bundle)
    if bundle["p8_p11_eligibility"] == "DENIED":
        raise P12PipelineBridgeError("ModelCapabilityBundle denies P8-P11 pipeline")

    target_key = str(target_key)
    backend = str(backend)
    cell = _target_cell(bundle, target_key=target_key, backend=backend)
    if cell.get("inference_status") != "VERIFIED":
        raise P12PipelineBridgeError("target backend capability is not VERIFIED")
    if target_key not in set(bundle.get("precision_search_targets", [])):
        raise P12PipelineBridgeError("target is outside precision search universe")
    if int(requested_n) not in {int(v) for v in cell.get("supported_n", [])}:
        raise P12PipelineBridgeError("requested precision is unsupported for target/backend")

    mutation_mode = str(cell["mutation_mode"])
    canary_mode = CANARY_BY_MUTATION_MODE.get(mutation_mode)
    if canary_mode is None or canary_mode == "DENIED":
        raise P12PipelineBridgeError(
            f"target mutation mode cannot enter canary pipeline: {mutation_mode}"
        )
    gate = {
        "schema": "beglin-p12-pipeline-capability-gate-v1",
        "status": "READY_FOR_P8",
        "model_id": bundle["model_id"],
        "model_capability_bundle_sha256": bundle["bundle_sha256"],
        "checkpoint_identity": bundle["checkpoint_identity"],
        "skeleton_sha256": bundle["skeleton_sha256"],
        "target_key": target_key,
        "backend": backend,
        "requested_n": int(requested_n),
        "mutation_mode": mutation_mode,
        "canary_mode": canary_mode,
        "production_write_allowed": False,
        "automatic_live_promotion": False,
    }
    gate["gate_sha256"] = mc.sha256_json(gate)
    return gate


def bind_p9_certification(
    *,
    certification_bundle: Mapping[str, Any],
    capability_gate: Mapping[str, Any],
) -> dict:
    if capability_gate.get("schema") != "beglin-p12-pipeline-capability-gate-v1":
        raise P12PipelineBridgeError("invalid capability gate schema")
    if capability_gate.get("status") != "READY_FOR_P8":
        raise P12PipelineBridgeError("capability gate is not READY_FOR_P8")
    if certification_bundle.get("status") != "MANUAL_REVIEW_CANDIDATE":
        raise P12PipelineBridgeError("P9 certification is not manual-review candidate")
    if certification_bundle.get("production_write_allowed") is not False:
        raise P12PipelineBridgeError("P9 certification unexpectedly permits production write")

    out = copy.deepcopy(dict(certification_bundle))
    out["model_capability_bundle_sha256"] = capability_gate[
        "model_capability_bundle_sha256"
    ]
    out["p12_capability_gate_sha256"] = capability_gate["gate_sha256"]
    out["p12_target_key"] = capability_gate["target_key"]
    out["p12_backend"] = capability_gate["backend"]
    out["p12_mutation_mode"] = capability_gate["mutation_mode"]
    out["p12_canary_mode"] = capability_gate["canary_mode"]
    out["production_write_allowed"] = False
    out["automatic_live_promotion"] = False
    out["p12_bound_certification_sha256"] = mc.sha256_json(out)
    return out


def select_p10_canary(
    *,
    p9_bundle: Mapping[str, Any],
    model_capability_bundle: Mapping[str, Any],
) -> dict:
    bundle = validate_bundle(model_capability_bundle)
    expected = p9_bundle.get("model_capability_bundle_sha256")
    if expected != bundle["bundle_sha256"]:
        raise P12PipelineBridgeError("P9 capability bundle binding is stale/missing")
    target_key = str(p9_bundle.get("p12_target_key") or "")
    backend = str(p9_bundle.get("p12_backend") or "")
    cell = _target_cell(bundle, target_key=target_key, backend=backend)
    if cell.get("inference_status") != "VERIFIED":
        raise P12PipelineBridgeError("P10 target capability lost VERIFIED status")
    mutation_mode = str(cell["mutation_mode"])
    mode = CANARY_BY_MUTATION_MODE.get(mutation_mode, "DENIED")
    if mode == "DENIED":
        raise P12PipelineBridgeError("P10 canary denied by current mutation capability")
    out = {
        "schema": "beglin-p12-p10-canary-selection-v1",
        "status": "READY_FOR_CANARY",
        "model_capability_bundle_sha256": bundle["bundle_sha256"],
        "p12_bound_p9_sha256": mc.sha256_json(dict(p9_bundle)),
        "target_key": target_key,
        "backend": backend,
        "mutation_mode": mutation_mode,
        "canary_mode": mode,
        "production_write_allowed": False,
        "automatic_live_promotion": False,
    }
    out["selection_sha256"] = mc.sha256_json(out)
    return out


def build_p11_capability_binding(
    *,
    p10_final_gate: Mapping[str, Any],
    p10_selection: Mapping[str, Any],
    model_capability_bundle: Mapping[str, Any],
) -> dict:
    bundle = validate_bundle(model_capability_bundle)
    if p10_final_gate.get("status") != "AWAITING_TRUSTED_PRODUCTION_APPROVAL":
        raise P12PipelineBridgeError("P10 final gate is not approval-ready")
    if p10_final_gate.get("production_cutover_allowed") is not False:
        raise P12PipelineBridgeError("P10 final gate already permits cutover")
    if p10_selection.get("model_capability_bundle_sha256") != bundle["bundle_sha256"]:
        raise P12PipelineBridgeError("P10 selection capability binding mismatch")

    binding = {
        "schema": "beglin-p12-p11-capability-binding-v1",
        "status": "AWAITING_TRUSTED_PRODUCTION_APPROVAL",
        "model_id": bundle["model_id"],
        "model_capability_bundle_sha256": bundle["bundle_sha256"],
        "checkpoint_identity": bundle["checkpoint_identity"],
        "skeleton_sha256": bundle["skeleton_sha256"],
        "target_key": p10_selection["target_key"],
        "backend": p10_selection["backend"],
        "mutation_mode": p10_selection["mutation_mode"],
        "canary_mode": p10_selection["canary_mode"],
        "p10_final_gate_sha256": mc.sha256_json(dict(p10_final_gate)),
        "p10_selection_sha256": p10_selection["selection_sha256"],
        "production_cutover_allowed": False,
        "production_write_allowed": False,
        "automatic_live_promotion": False,
    }
    binding["binding_sha256"] = mc.sha256_json(binding)
    return binding
