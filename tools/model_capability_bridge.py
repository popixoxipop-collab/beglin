#!/usr/bin/env python3
"""P12 capability binding helpers for the existing P8-P11 precision pipeline.

This module is intentionally read-only.  It validates that one allocator
target exists in a ModelCapabilityBundle and is admissible for the requested
backend/precision before the legacy P8 pipeline is allowed to carry the bundle
identity forward.
"""
from __future__ import annotations

from typing import Any, Mapping

import model_capability as mc


class CapabilityBridgeError(RuntimeError):
    pass


ROLE_ALIASES = {
    "q_proj": "Q_PROJ",
    "k_proj": "K_PROJ",
    "v_proj": "V_PROJ",
    "o_proj": "O_PROJ",
    "q_a_proj": "Q_A_PROJ",
    "q_b_proj": "Q_B_PROJ",
    "kv_a_proj": "KV_A_PROJ",
    "kv_a_proj_with_mqa": "KV_A_PROJ",
    "kv_b_proj": "KV_B_PROJ",
    "gate_proj": "DENSE_GATE",
    "up_proj": "DENSE_UP",
    "down_proj": "DENSE_DOWN",
    "shared_gate_proj": "SHARED_GATE",
    "shared_up_proj": "SHARED_UP",
    "shared_down_proj": "SHARED_DOWN",
    "expert_gate_proj": "EXPERT_GATE",
    "expert_up_proj": "EXPERT_UP",
    "expert_down_proj": "EXPERT_DOWN",
}


def _hex64(name: str, value: Any) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise CapabilityBridgeError(f"{name} must be a sha256 hex string")
    return text


def validate_bundle(bundle: Mapping[str, Any]) -> dict:
    if not isinstance(bundle, Mapping):
        raise CapabilityBridgeError("model capability bundle must be an object")
    if bundle.get("schema") != "beglin-model-capability-bundle-v1":
        raise CapabilityBridgeError("unexpected model capability bundle schema")
    if bundle.get("production_write_allowed") is not False:
        raise CapabilityBridgeError("capability bundle unexpectedly permits production write")
    expected = _hex64("bundle_sha256", bundle.get("bundle_sha256"))
    actual = mc.stable_identity_sha256(dict(bundle))
    if actual != expected:
        raise CapabilityBridgeError(
            f"model capability bundle SHA mismatch: expected={expected} actual={actual}"
        )
    return dict(bundle)


def canonical_role(role: str) -> str:
    raw = str(role).strip()
    if not raw:
        raise CapabilityBridgeError("empty target role")
    if raw in mc.PRECISION_ROLES:
        return raw
    lowered = raw.lower()
    if lowered in ROLE_ALIASES:
        return ROLE_ALIASES[lowered]
    upper = raw.upper()
    if upper in mc.PRECISION_ROLES:
        return upper
    raise CapabilityBridgeError(f"unknown precision target role: {role!r}")


def validate_p8_target(
    *,
    bundle: Mapping[str, Any],
    role: str,
    layer: int,
    target_n: int,
    backend: str = "mlx_metal",
    expert_id: int | None = None,
) -> dict:
    value = validate_bundle(bundle)
    eligibility = value.get("p8_p11_eligibility") or {}
    if eligibility.get("p8_allowed") is not True:
        raise CapabilityBridgeError(
            f"capability bundle is not P8-eligible: {eligibility.get('status')}"
        )

    canonical = canonical_role(role)
    layer = int(layer)
    target_n = int(target_n)
    nodes = [
        row
        for row in (value.get("tensor_role_graph") or {}).get("nodes", [])
        if row.get("role") == canonical
        and row.get("layer") == layer
        and (
            expert_id is None
            or row.get("expert_id") == int(expert_id)
        )
    ]
    if expert_id is None:
        non_expert = [row for row in nodes if row.get("expert_id") is None]
        if non_expert:
            nodes = non_expert
    if len(nodes) != 1:
        raise CapabilityBridgeError(
            f"target resolution must produce exactly one tensor role node: "
            f"role={canonical} layer={layer} expert_id={expert_id} matches={len(nodes)}"
        )
    node = nodes[0]
    target_key = str(node["canonical_target_key"])

    cap_rows = [
        row
        for row in (value.get("backend_capability_matrix") or {}).get("rows", [])
        if row.get("target_key") == target_key and row.get("backend") == backend
    ]
    if len(cap_rows) != 1:
        raise CapabilityBridgeError(
            f"backend capability missing/ambiguous for target={target_key} backend={backend}"
        )
    cap = cap_rows[0]
    if str(cap.get("inference_status", "")).startswith("UNSUPPORTED"):
        raise CapabilityBridgeError(
            f"backend inference unsupported for target={target_key} backend={backend}"
        )
    if str(cap.get("qng64_status", "")).startswith("UNSUPPORTED"):
        raise CapabilityBridgeError(
            f"qNg64 unsupported for target={target_key} backend={backend}"
        )
    supported_n = [int(x) for x in cap.get("supported_n", [])]
    if target_n not in supported_n:
        raise CapabilityBridgeError(
            f"precision n={target_n} unsupported for target={target_key} "
            f"backend={backend}; supported={supported_n}"
        )

    mut_rows = [
        row
        for row in (value.get("runtime_mutation_matrix") or {}).get("rows", [])
        if row.get("target_key") == target_key and row.get("backend") == backend
    ]
    if len(mut_rows) != 1:
        raise CapabilityBridgeError(
            f"runtime mutation capability missing/ambiguous for {target_key}/{backend}"
        )

    return {
        "schema": "beglin-p12-p8-capability-binding-v1",
        "model_capability_bundle_sha256": value["bundle_sha256"],
        "model_id": value.get("model_id"),
        "checkpoint_identity_sha256": value.get("checkpoint_identity_sha256"),
        "capability_target_key": target_key,
        "canonical_role": canonical,
        "layer": layer,
        "expert_id": node.get("expert_id"),
        "backend": backend,
        "target_n": target_n,
        "inference_status": cap.get("inference_status"),
        "qng64_status": cap.get("qng64_status"),
        "mutation_mode": mut_rows[0].get("mutation_mode"),
        "requires_validation": bool(cap.get("validation_required")),
        "production_write_allowed": False,
        "automatic_live_promotion": False,
    }


def optional_lineage(value: Mapping[str, Any]) -> dict:
    """Return only additive P12 lineage fields when they are present."""
    out = {}
    for key in (
        "model_capability_bundle_sha256",
        "capability_target_key",
        "capability_backend",
    ):
        if value.get(key) is not None:
            out[key] = value[key]
    return out
