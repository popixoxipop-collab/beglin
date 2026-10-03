#!/usr/bin/env python3
"""P12 capability-bound front door for the proven P8->P11 precision pipeline.

This module is additive and read-only. It does not mutate a worker, route,
policy, or production state. Its job is to bind a candidate to one exact
ModelCapabilityBundle before the existing P8/P9/P10/P11 machinery is invoked.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

import model_capability as mc


HEX64 = re.compile(r"^[0-9a-f]{64}$")


class PipelineCapabilityError(RuntimeError):
    pass


def _require_sha(name: str, value: Any) -> str:
    out = str(value or "").lower()
    if not HEX64.fullmatch(out):
        raise PipelineCapabilityError(f"{name} must be sha256 hex")
    return out


def require_bundle(bundle: Mapping[str, Any]) -> dict:
    value = dict(bundle)
    if value.get("schema") != "beglin-model-capability-bundle-v1":
        raise PipelineCapabilityError("unsupported model capability bundle schema")
    got = _require_sha("bundle_sha256", value.get("bundle_sha256"))
    actual = mc.stable_identity_sha256(value)
    if got != actual:
        raise PipelineCapabilityError(
            f"model capability bundle hash mismatch: expected={got} actual={actual}"
        )
    return value


def _eligibility(bundle: Mapping[str, Any], phase: str) -> dict:
    eligibility = dict(bundle.get("p8_p11_eligibility") or {})
    key = f"{phase.lower()}_allowed"
    if eligibility.get(key) is not True:
        raise PipelineCapabilityError(
            f"{phase} denied by model capability bundle: "
            + ",".join(eligibility.get("reasons") or [])
        )
    return eligibility


def _search_target(
    bundle: Mapping[str, Any],
    *,
    target_key: str,
    backend: str,
) -> dict:
    matches = [
        dict(row)
        for row in bundle.get("precision_search_targets") or []
        if row.get("target_key") == target_key and row.get("backend") == backend
    ]
    if len(matches) != 1:
        raise PipelineCapabilityError(
            f"precision target is not uniquely eligible: backend={backend} target={target_key}"
        )
    return matches[0]


def _backend_row(bundle: Mapping[str, Any], *, target_key: str, backend: str) -> dict:
    matches = [
        dict(row)
        for row in (bundle.get("backend_capability_matrix") or {}).get("rows", [])
        if row.get("target_key") == target_key and row.get("backend") == backend
    ]
    if len(matches) != 1:
        raise PipelineCapabilityError(
            f"backend capability row missing/ambiguous: backend={backend} target={target_key}"
        )
    return matches[0]


def _mutation_row(bundle: Mapping[str, Any], *, target_key: str, backend: str) -> dict:
    matches = [
        dict(row)
        for row in (bundle.get("runtime_mutation_matrix") or {}).get("rows", [])
        if row.get("target_key") == target_key and row.get("backend") == backend
    ]
    if len(matches) != 1:
        raise PipelineCapabilityError(
            f"runtime mutation row missing/ambiguous: backend={backend} target={target_key}"
        )
    return matches[0]


def _binding_digest(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return mc.stable_identity_sha256(payload)


def _self_hash(value: dict, field: str) -> dict:
    out = dict(value)
    out[field] = _binding_digest(out, field)
    return out


def _verify_binding_hash(value: Mapping[str, Any], field: str) -> str:
    got = _require_sha(field, value.get(field))
    actual = _binding_digest(value, field)
    if got != actual:
        raise PipelineCapabilityError(
            f"{field} mismatch: expected={got} actual={actual}"
        )
    return got


def bind_p8_target(
    bundle: Mapping[str, Any],
    *,
    target_key: str,
    backend: str,
    requested_n: int | None = None,
) -> dict:
    """Bind provenance capture to one exact capability bundle + target."""
    bundle = require_bundle(bundle)
    _eligibility(bundle, "P8")
    target = _search_target(bundle, target_key=target_key, backend=backend)
    supported_n = [int(x) for x in target.get("supported_n") or []]
    if requested_n is not None and int(requested_n) not in supported_n:
        raise PipelineCapabilityError(
            f"requested n={requested_n} unsupported for {backend}:{target_key}"
        )
    backend_cap = _backend_row(bundle, target_key=target_key, backend=backend)
    binding = {
        "schema": "beglin-p12-p8-capability-binding-v1",
        "status": "READY_FOR_P8_PROVENANCE",
        "model_id": bundle["model_id"],
        "checkpoint_identity_sha256": bundle["checkpoint_identity_sha256"],
        "model_capability_bundle_sha256": bundle["bundle_sha256"],
        "model_skeleton_sha256": bundle["model_skeleton"]["skeleton_sha256"],
        "target_key": target_key,
        "backend": backend,
        "supported_n": supported_n,
        "requested_n": int(requested_n) if requested_n is not None else None,
        "mutation_mode": target["mutation_mode"],
        "backend_inference_status": backend_cap["inference_status"],
        "qng64_status": backend_cap["qng64_status"],
        "requires_validation": bool(target.get("requires_validation")),
        "automatic_live_promotion": False,
        "production_write_allowed": False,
    }
    return _self_hash(binding, "p8_binding_sha256")


def bind_p9_certification(
    bundle: Mapping[str, Any],
    *,
    p8_binding: Mapping[str, Any],
    provenance_evidence_sha256: str,
) -> dict:
    """Carry the exact P12 capability identity into P9 certification input."""
    bundle = require_bundle(bundle)
    _eligibility(bundle, "P9")
    p8 = dict(p8_binding)
    if p8.get("schema") != "beglin-p12-p8-capability-binding-v1":
        raise PipelineCapabilityError("unsupported P8 capability binding")
    _verify_binding_hash(p8, "p8_binding_sha256")
    if p8.get("model_capability_bundle_sha256") != bundle["bundle_sha256"]:
        raise PipelineCapabilityError("P8 binding uses stale model capability bundle")
    if p8.get("checkpoint_identity_sha256") != bundle["checkpoint_identity_sha256"]:
        raise PipelineCapabilityError("P8 binding checkpoint identity mismatch")
    provenance = _require_sha("provenance_evidence_sha256", provenance_evidence_sha256)
    binding = {
        "schema": "beglin-p12-p9-capability-binding-v1",
        "status": "READY_FOR_P9_CERTIFICATION",
        "model_id": bundle["model_id"],
        "checkpoint_identity_sha256": bundle["checkpoint_identity_sha256"],
        "model_capability_bundle_sha256": bundle["bundle_sha256"],
        "model_skeleton_sha256": bundle["model_skeleton"]["skeleton_sha256"],
        "p8_binding_sha256": _require_sha(
            "p8_binding_sha256", p8.get("p8_binding_sha256")
        ),
        "provenance_evidence_sha256": provenance,
        "target_key": p8["target_key"],
        "backend": p8["backend"],
        "requested_n": p8.get("requested_n"),
        "automatic_live_promotion": False,
        "production_write_allowed": False,
    }
    return _self_hash(binding, "p9_binding_sha256")


def select_p10_canary(
    bundle: Mapping[str, Any],
    *,
    p9_binding: Mapping[str, Any],
) -> dict:
    """Choose canary strategy only from the runtime mutation capability matrix."""
    bundle = require_bundle(bundle)
    _eligibility(bundle, "P10")
    p9 = dict(p9_binding)
    if p9.get("schema") != "beglin-p12-p9-capability-binding-v1":
        raise PipelineCapabilityError("unsupported P9 capability binding")
    _verify_binding_hash(p9, "p9_binding_sha256")
    if p9.get("model_capability_bundle_sha256") != bundle["bundle_sha256"]:
        raise PipelineCapabilityError("P9 binding uses stale model capability bundle")
    target_key = str(p9["target_key"])
    backend = str(p9["backend"])
    mutation = _mutation_row(bundle, target_key=target_key, backend=backend)
    backend_cap = _backend_row(bundle, target_key=target_key, backend=backend)
    mode = str(mutation["mutation_mode"])

    if mode in {"HOT_REBIND_SINGLE", "HOT_REBIND_MULTI"}:
        if (
            backend_cap.get("inference_status") == "VERIFIED"
            and backend_cap.get("qng64_status") == "VERIFIED"
        ):
            strategy = "SAME_WORKER_PRECISION_CANARY"
            state = "READY_FOR_P10_CANARY"
        else:
            strategy = "ISOLATED_RUNTIME_VALIDATION"
            state = "VALIDATION_REQUIRED_BEFORE_P10_CANARY"
    elif mode == "RESTART_REQUIRED":
        strategy = "ISOLATED_RESTART_CANARY"
        state = "READY_FOR_P10_CANARY"
    elif mode == "IMPLEMENTED_UNVERIFIED":
        strategy = "ISOLATED_RUNTIME_VALIDATION"
        state = "VALIDATION_REQUIRED_BEFORE_P10_CANARY"
    else:
        raise PipelineCapabilityError(
            f"P10 canary unavailable for mutation_mode={mode}"
        )

    binding = {
        "schema": "beglin-p12-p10-capability-binding-v1",
        "status": state,
        "model_id": bundle["model_id"],
        "checkpoint_identity_sha256": bundle["checkpoint_identity_sha256"],
        "model_capability_bundle_sha256": bundle["bundle_sha256"],
        "model_skeleton_sha256": bundle["model_skeleton"]["skeleton_sha256"],
        "p9_binding_sha256": _require_sha(
            "p9_binding_sha256", p9.get("p9_binding_sha256")
        ),
        "target_key": target_key,
        "backend": backend,
        "mutation_mode": mode,
        "canary_strategy": strategy,
        "backend_inference_status": backend_cap["inference_status"],
        "qng64_status": backend_cap["qng64_status"],
        "automatic_live_promotion": False,
        "production_write_allowed": False,
    }
    return _self_hash(binding, "p10_binding_sha256")


def build_p11_capability_preimage(
    bundle: Mapping[str, Any],
    *,
    p10_binding: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
) -> dict:
    """Bind P11 eligibility to an exact FULL capability bundle and runtime state.

    This does not grant production permission. It only proves that a later
    trusted approval request is about the same verified capability state.
    """
    bundle = require_bundle(bundle)
    _eligibility(bundle, "P11")
    p10 = dict(p10_binding)
    if p10.get("schema") != "beglin-p12-p10-capability-binding-v1":
        raise PipelineCapabilityError("unsupported P10 capability binding")
    _verify_binding_hash(p10, "p10_binding_sha256")
    if p10.get("model_capability_bundle_sha256") != bundle["bundle_sha256"]:
        raise PipelineCapabilityError("P10 binding uses stale model capability bundle")
    if p10.get("status") != "READY_FOR_P10_CANARY":
        raise PipelineCapabilityError("P10 capability binding still requires validation")

    target_key = str(p10["target_key"])
    backend = str(p10["backend"])
    backend_cap = _backend_row(bundle, target_key=target_key, backend=backend)
    if backend_cap.get("inference_status") != "VERIFIED":
        raise PipelineCapabilityError("P11 requires VERIFIED backend inference")
    if backend_cap.get("qng64_status") != "VERIFIED":
        raise PipelineCapabilityError("P11 requires VERIFIED qNg64 capability")

    runtime_bundle = _require_sha(
        "runtime model capability bundle",
        runtime_state.get("model_capability_bundle_sha256"),
    )
    if runtime_bundle != bundle["bundle_sha256"]:
        raise PipelineCapabilityError("runtime uses stale model capability bundle")
    checkpoint = _require_sha(
        "runtime checkpoint identity",
        runtime_state.get("checkpoint_identity_sha256"),
    )
    if checkpoint != bundle["checkpoint_identity_sha256"]:
        raise PipelineCapabilityError("runtime checkpoint identity mismatch")
    if str(runtime_state.get("backend")) != backend:
        raise PipelineCapabilityError("runtime backend mismatch")
    if str(runtime_state.get("target_key")) != target_key:
        raise PipelineCapabilityError("runtime target mismatch")

    preimage = {
        "schema": "beglin-p12-p11-capability-preimage-v1",
        "status": "AWAITING_TRUSTED_PRODUCTION_APPROVAL",
        "model_id": bundle["model_id"],
        "checkpoint_identity_sha256": bundle["checkpoint_identity_sha256"],
        "model_capability_bundle_sha256": bundle["bundle_sha256"],
        "model_skeleton_sha256": bundle["model_skeleton"]["skeleton_sha256"],
        "p10_binding_sha256": _require_sha(
            "p10_binding_sha256", p10.get("p10_binding_sha256")
        ),
        "target_key": target_key,
        "backend": backend,
        "runtime_identity_sha256": mc.stable_identity_sha256(dict(runtime_state)),
        "trusted_production_approval_present": False,
        "production_cutover_allowed": False,
        "automatic_live_promotion": False,
        "production_write_allowed": False,
    }
    return _self_hash(preimage, "p11_capability_preimage_sha256")
