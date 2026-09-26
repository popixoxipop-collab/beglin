#!/usr/bin/env python3
"""Production-intent sealing for Agent E.

This module can prove that proposal, runtime preimage, candidate policy and a
verified human signature agree. It deliberately cannot execute production
mutation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import manual_canary_contract as mc


class ProductionIntentError(RuntimeError):
    pass


class ProductionBridgeDisabled(ProductionIntentError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _policy_map(rows: list[dict]) -> dict[tuple[str, int], int]:
    out = {}
    for row in rows:
        try:
            key = (str(row["role"]), int(row["layer"]))
            n = int(row["n"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProductionIntentError("invalid policy row") from exc
        if n <= 0:
            raise ProductionIntentError("policy n must be positive")
        if key in out:
            raise ProductionIntentError("duplicate role/layer in policy")
        out[key] = n
    return out


def seal_production_intent(
    *,
    proposal: Mapping[str, Any],
    verified_signature: Mapping[str, Any],
    runtime_preimage: Mapping[str, Any],
    candidate_policy: list[dict],
) -> dict:
    p = mc.normalize_proposal(proposal)
    sig = dict(verified_signature)
    if sig.get("schema") != "manual-canary-human-signature-verification-v1":
        raise ProductionIntentError("unsupported signature verification schema")
    if sig.get("status") != "VERIFIED":
        raise ProductionIntentError("human signature is not VERIFIED")
    if sig.get("production_write_allowed") is not False:
        raise ProductionIntentError("signature verification must remain production-write disabled")
    digest = mc.sha256_json(p)
    if sig.get("proposal_digest") != digest:
        raise ProductionIntentError("signature proposal_digest mismatch")

    try:
        active_policy = list(runtime_preimage["active_policy"])
        active_policy_hash = str(runtime_preimage["active_policy_hash"])
        weight_epoch = int(runtime_preimage["weight_epoch"])
        ack_sha256 = str(runtime_preimage["ack_sha256"])
        worker_instance_id = str(runtime_preimage["worker_instance_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProductionIntentError("runtime preimage is incomplete") from exc

    if active_policy_hash != p["baseline_policy_hash"]:
        raise ProductionIntentError("runtime baseline policy hash mismatch")
    if sha256_json(active_policy) != p["baseline_policy_hash"]:
        raise ProductionIntentError("runtime active_policy does not hash to baseline_policy_hash")
    if weight_epoch != p["expected_epoch"]:
        raise ProductionIntentError("runtime epoch mismatch")
    if not ack_sha256 or not worker_instance_id:
        raise ProductionIntentError("runtime ACK/worker identity must be non-empty")
    if worker_instance_id != p["restart_instance_id"]:
        raise ProductionIntentError("restart instance mismatch")

    before = _policy_map(active_policy)
    after = _policy_map(candidate_policy)
    target = p["single_target"]
    key = (target["role"], int(target["layer"]))
    if before.get(key) != int(target["before_n"]):
        raise ProductionIntentError("runtime before_n mismatch")
    if after.get(key) != int(target["after_n"]):
        raise ProductionIntentError("candidate after_n mismatch")
    differing = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    if differing != {key}:
        raise ProductionIntentError("candidate policy must change exactly one target")
    if sha256_json(candidate_policy) != p["candidate_policy_hash"]:
        raise ProductionIntentError("candidate policy hash mismatch")

    intent = {
        "schema": "manual-canary-production-intent-v1",
        "status": "READY_FOR_EXTERNAL_REVIEW",
        "execution_enabled": False,
        "production_write_allowed": False,
        "proposal_digest": digest,
        "human_signature": {
            "approval_id": sig["approval_id"],
            "principal": sig["principal"],
            "signed_payload_sha256": sig["signed_payload_sha256"],
            "signature_sha256": sig["signature_sha256"],
            "allowed_signers_sha256": sig["allowed_signers_sha256"],
        },
        "runtime_preimage": {
            "active_policy_hash": active_policy_hash,
            "weight_epoch": weight_epoch,
            "ack_sha256": ack_sha256,
            "worker_instance_id": worker_instance_id,
        },
        "candidate_policy_hash": p["candidate_policy_hash"],
        "target": p["single_target"],
        "budget": p["budget"],
        "evidence_refs": p["evidence_refs"],
        "rollback_plan": p["rollback_plan"],
    }
    intent["intent_sha256"] = sha256_json(intent)
    return intent


def execute_production_intent(*args, **kwargs):
    raise ProductionBridgeDisabled(
        "production adapter execution is intentionally disabled; "
        "a separate reviewed bridge is required"
    )
