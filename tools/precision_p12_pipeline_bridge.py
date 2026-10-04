#!/usr/bin/env python3
"""P12 capability gate for the existing P8-P11 precision pipeline.

The bridge is pure/read-only. It binds one exact ModelCapabilityBundle,
target/backend capability and requested precision to downstream artifacts
without enabling production mutation or automatic promotion.
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

_RUNTIME_ROLE_TO_SEMANTIC = {
    "shared_gate_proj": "shared_gate",
    "shared_up_proj": "shared_up",
    "shared_down_proj": "shared_down",
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "o_proj",
    "q_a_proj": "q_a_proj",
    "q_b_proj": "q_b_proj",
    "kv_a_proj_with_mqa": "kv_a_proj",
    "kv_b_proj": "kv_b_proj",
    "gate_proj": "dense_gate",
    "up_proj": "dense_up",
    "down_proj": "dense_down",
}


def _verify_self_hash(value: Mapping[str, Any], field: str) -> None:
    expected = str(value.get(field) or "")
    try:
        mc.require_sha(field, expected)
    except Exception as exc:
        raise P12PipelineBridgeError(f"{field} is missing/invalid") from exc
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    actual = mc.sha256_json(payload)
    if actual != expected:
        raise P12PipelineBridgeError(
            f"{field} mismatch: expected={expected} actual={actual}"
        )


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


def _require_verified_target(
    bundle: Mapping[str, Any],
    *,
    target_key: str,
    backend: str,
    requested_n: int,
) -> dict:
    cell = _target_cell(bundle, target_key=target_key, backend=backend)
    if cell.get("inference_status") != "VERIFIED":
        raise P12PipelineBridgeError("target backend capability is not VERIFIED")
    if target_key not in set(bundle.get("precision_search_targets", [])):
        raise P12PipelineBridgeError("target is outside precision search universe")
    if int(requested_n) not in {int(v) for v in cell.get("supported_n", [])}:
        raise P12PipelineBridgeError(
            "requested precision is unsupported for target/backend"
        )
    mutation_mode = str(cell.get("mutation_mode"))
    canary_mode = CANARY_BY_MUTATION_MODE.get(mutation_mode)
    if canary_mode is None or canary_mode == "DENIED":
        raise P12PipelineBridgeError(
            f"target mutation mode cannot enter canary pipeline: {mutation_mode}"
        )
    return cell


def _semantic_suffix_from_p9_target(target: Mapping[str, Any]) -> tuple[str, int]:
    try:
        runtime_role = str(target["role"])
        layer = int(target["layer"])
        new_n = int(target["new_n"])
    except (KeyError, TypeError, ValueError) as exc:
        raise P12PipelineBridgeError(
            "P9 target must include role/layer/new_n"
        ) from exc
    semantic = _RUNTIME_ROLE_TO_SEMANTIC.get(runtime_role)
    if semantic is None:
        raise P12PipelineBridgeError(
            f"P9 runtime role has no P12 semantic mapping: {runtime_role}"
        )
    return f"/L{layer}/{semantic}", new_n


def validate_capability_gate(gate: Mapping[str, Any]) -> dict:
    value = copy.deepcopy(dict(gate))
    if value.get("schema") != "beglin-p12-pipeline-capability-gate-v1":
        raise P12PipelineBridgeError("invalid capability gate schema")
    if value.get("status") != "READY_FOR_P8":
        raise P12PipelineBridgeError("capability gate is not READY_FOR_P8")
    _verify_self_hash(value, "gate_sha256")
    if value.get("production_write_allowed") is not False:
        raise P12PipelineBridgeError("capability gate permits production write")
    if value.get("automatic_live_promotion") is not False:
        raise P12PipelineBridgeError("capability gate permits automatic promotion")
    return value


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
    requested_n = int(requested_n)
    cell = _require_verified_target(
        bundle,
        target_key=target_key,
        backend=backend,
        requested_n=requested_n,
    )
    mutation_mode = str(cell["mutation_mode"])
    canary_mode = CANARY_BY_MUTATION_MODE[mutation_mode]
    gate = {
        "schema": "beglin-p12-pipeline-capability-gate-v1",
        "status": "READY_FOR_P8",
        "model_id": bundle["model_id"],
        "model_capability_bundle_sha256": bundle["bundle_sha256"],
        "checkpoint_identity": bundle["checkpoint_identity"],
        "skeleton_sha256": bundle["skeleton_sha256"],
        "target_key": target_key,
        "backend": backend,
        "requested_n": requested_n,
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
    gate = validate_capability_gate(capability_gate)
    if certification_bundle.get("status") != "MANUAL_REVIEW_CANDIDATE":
        raise P12PipelineBridgeError(
            "P9 certification is not manual-review candidate"
        )
    if certification_bundle.get("production_write_allowed") is not False:
        raise P12PipelineBridgeError(
            "P9 certification unexpectedly permits production write"
        )
    if certification_bundle.get("automatic_live_promotion") not in {None, False}:
        raise P12PipelineBridgeError(
            "P9 certification unexpectedly permits automatic promotion"
        )

    target = certification_bundle.get("target")
    if not isinstance(target, Mapping):
        raise P12PipelineBridgeError("P9 certification target is missing")
    suffix, p9_new_n = _semantic_suffix_from_p9_target(target)
    if not str(gate["target_key"]).endswith(suffix):
        raise P12PipelineBridgeError(
            "P9 certification target does not match capability gate target"
        )
    if p9_new_n != int(gate["requested_n"]):
        raise P12PipelineBridgeError(
            "P9 certification new_n does not match requested precision"
        )

    out = copy.deepcopy(dict(certification_bundle))
    out["model_capability_bundle_sha256"] = gate[
        "model_capability_bundle_sha256"
    ]
    out["p12_capability_gate_sha256"] = gate["gate_sha256"]
    out["p12_target_key"] = gate["target_key"]
    out["p12_backend"] = gate["backend"]
    out["p12_requested_n"] = int(gate["requested_n"])
    out["p12_mutation_mode"] = gate["mutation_mode"]
    out["p12_canary_mode"] = gate["canary_mode"]
    out["production_write_allowed"] = False
    out["automatic_live_promotion"] = False
    out["p12_bound_certification_sha256"] = mc.sha256_json(out)
    return out


def validate_bound_p9(p9_bundle: Mapping[str, Any]) -> dict:
    value = copy.deepcopy(dict(p9_bundle))
    _verify_self_hash(value, "p12_bound_certification_sha256")
    if value.get("status") != "MANUAL_REVIEW_CANDIDATE":
        raise P12PipelineBridgeError("bound P9 is not manual-review candidate")
    if value.get("production_write_allowed") is not False:
        raise P12PipelineBridgeError("bound P9 permits production write")
    if value.get("automatic_live_promotion") is not False:
        raise P12PipelineBridgeError("bound P9 permits automatic promotion")
    return value


def select_p10_canary(
    *,
    p9_bundle: Mapping[str, Any],
    model_capability_bundle: Mapping[str, Any],
) -> dict:
    p9 = validate_bound_p9(p9_bundle)
    bundle = validate_bundle(model_capability_bundle)
    expected = p9.get("model_capability_bundle_sha256")
    if expected != bundle["bundle_sha256"]:
        raise P12PipelineBridgeError(
            "P9 capability bundle binding is stale/missing"
        )
    target_key = str(p9.get("p12_target_key") or "")
    backend = str(p9.get("p12_backend") or "")
    requested_n = int(p9.get("p12_requested_n"))
    cell = _require_verified_target(
        bundle,
        target_key=target_key,
        backend=backend,
        requested_n=requested_n,
    )
    mutation_mode = str(cell["mutation_mode"])
    mode = CANARY_BY_MUTATION_MODE[mutation_mode]
    if mutation_mode != p9.get("p12_mutation_mode"):
        raise P12PipelineBridgeError("P9 mutation mode is stale")
    if mode != p9.get("p12_canary_mode"):
        raise P12PipelineBridgeError("P9 canary mode is stale")

    out = {
        "schema": "beglin-p12-p10-canary-selection-v1",
        "status": "READY_FOR_CANARY",
        "model_capability_bundle_sha256": bundle["bundle_sha256"],
        "p12_bound_p9_sha256": p9["p12_bound_certification_sha256"],
        "target_key": target_key,
        "backend": backend,
        "requested_n": requested_n,
        "mutation_mode": mutation_mode,
        "canary_mode": mode,
        "production_write_allowed": False,
        "automatic_live_promotion": False,
    }
    out["selection_sha256"] = mc.sha256_json(out)
    return out


def validate_p10_selection(selection: Mapping[str, Any]) -> dict:
    value = copy.deepcopy(dict(selection))
    if value.get("schema") != "beglin-p12-p10-canary-selection-v1":
        raise P12PipelineBridgeError("invalid P10 selection schema")
    if value.get("status") != "READY_FOR_CANARY":
        raise P12PipelineBridgeError("P10 selection is not READY_FOR_CANARY")
    _verify_self_hash(value, "selection_sha256")
    if value.get("production_write_allowed") is not False:
        raise P12PipelineBridgeError("P10 selection permits production write")
    if value.get("automatic_live_promotion") is not False:
        raise P12PipelineBridgeError("P10 selection permits automatic promotion")
    return value


def build_p11_capability_binding(
    *,
    p10_final_gate: Mapping[str, Any],
    p10_selection: Mapping[str, Any],
    model_capability_bundle: Mapping[str, Any],
) -> dict:
    selection = validate_p10_selection(p10_selection)
    bundle = validate_bundle(model_capability_bundle)
    if p10_final_gate.get("status") != "AWAITING_TRUSTED_PRODUCTION_APPROVAL":
        raise P12PipelineBridgeError(
            "P10 final gate is not approval-ready"
        )
    if p10_final_gate.get("production_cutover_allowed") is not False:
        raise P12PipelineBridgeError("P10 final gate already permits cutover")
    if p10_final_gate.get("production_write_allowed") not in {None, False}:
        raise P12PipelineBridgeError("P10 final gate permits production write")
    if selection.get("model_capability_bundle_sha256") != bundle["bundle_sha256"]:
        raise P12PipelineBridgeError(
            "P10 selection capability binding mismatch"
        )

    cell = _require_verified_target(
        bundle,
        target_key=str(selection["target_key"]),
        backend=str(selection["backend"]),
        requested_n=int(selection["requested_n"]),
    )
    current_mutation = str(cell["mutation_mode"])
    current_canary = CANARY_BY_MUTATION_MODE[current_mutation]
    if current_mutation != selection["mutation_mode"]:
        raise P12PipelineBridgeError(
            "P10 selection mutation capability is stale"
        )
    if current_canary != selection["canary_mode"]:
        raise P12PipelineBridgeError(
            "P10 selection canary capability is stale"
        )

    binding = {
        "schema": "beglin-p12-p11-capability-binding-v1",
        "status": "AWAITING_TRUSTED_PRODUCTION_APPROVAL",
        "model_id": bundle["model_id"],
        "model_capability_bundle_sha256": bundle["bundle_sha256"],
        "checkpoint_identity": bundle["checkpoint_identity"],
        "skeleton_sha256": bundle["skeleton_sha256"],
        "target_key": selection["target_key"],
        "backend": selection["backend"],
        "requested_n": int(selection["requested_n"]),
        "mutation_mode": selection["mutation_mode"],
        "canary_mode": selection["canary_mode"],
        "p10_final_gate_sha256": mc.sha256_json(dict(p10_final_gate)),
        "p10_selection_sha256": selection["selection_sha256"],
        "production_cutover_allowed": False,
        "production_write_allowed": False,
        "automatic_live_promotion": False,
    }
    binding["binding_sha256"] = mc.sha256_json(binding)
    return binding
