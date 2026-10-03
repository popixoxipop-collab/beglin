#!/usr/bin/env python3
"""P11 read-only production preimage and approval-request builder."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import precision_context as pc

PREIMAGE_SCHEMA = "beglin-precision-p11-production-preimage-v1"
PLAN_SCHEMA = "beglin-precision-p11-production-cutover-plan-v1"
REQUEST_SCHEMA = "beglin-precision-p11-production-approval-request-v1"

P10_RESULT_SHA256 = "9c9c809097d18d735855f09bd85016272a319087501d160873f52985669f2fff"
ACTIVE_ROUTE_ID = "beglin-candidate-shared-up-l3-n6"
ROUTER_ID = "beglin-local-http-persistent-supervisor-v1"

BASELINE_POLICY = pc.normalize_policy([
    {"role": "shared_up_proj", "layer": 3, "n": 6},
    {"role": "shared_down_proj", "layer": 26, "n": 5},
])
TARGET_POLICY = pc.normalize_policy([
    {"role": "shared_up_proj", "layer": 3, "n": 5},
    {"role": "shared_down_proj", "layer": 26, "n": 5},
])
BASELINE_POLICY_HASH = pc.policy_hash(BASELINE_POLICY)
TARGET_POLICY_HASH = pc.policy_hash(TARGET_POLICY)


class P11Error(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def require_sha(name: str, value: Any) -> str:
    out = str(value or "").lower()
    if len(out) != 64 or any(ch not in "0123456789abcdef" for ch in out):
        raise P11Error(f"{name} must be sha256 hex")
    return out


def validate_p10(result: Mapping[str, Any], result_sha256: str) -> dict:
    if require_sha("p10 result", result_sha256) != P10_RESULT_SHA256:
        raise P11Error("unexpected P10 result sha256")
    if result.get("schema") != "beglin-p10-first-real-canary-v1":
        raise P11Error("unexpected P10 schema")
    if result.get("status") != "PASS" or result.get("production_touched") is not False:
        raise P11Error("P10 is not a clean PASS")
    if result.get("production_before") != result.get("production_after"):
        raise P11Error("P10 changed production")
    final = result.get("final_gate") or {}
    if final.get("status") != "AWAITING_TRUSTED_PRODUCTION_APPROVAL":
        raise P11Error("P10 final gate is not waiting for approval")
    for key in ("production_write_allowed", "automatic_live_promotion", "production_cutover_allowed"):
        if final.get(key) is not False:
            raise P11Error(f"P10 boundary changed: {key}")
    candidate = result.get("candidate_canary") or {}
    rollback = result.get("rollback_drill") or {}
    if candidate.get("candidate_policy_hash") != TARGET_POLICY_HASH:
        raise P11Error("P10 candidate policy differs from P11 target")
    if candidate.get("responses") != [[55222, 1]]:
        raise P11Error("P10 candidate reference response mismatch")
    if rollback.get("restored_policy_hash") != BASELINE_POLICY_HASH:
        raise P11Error("P10 rollback baseline mismatch")
    if rollback.get("responses") != [[55222, 372]]:
        raise P11Error("P10 rollback reference response mismatch")
    return {
        "p10_result_sha256": result_sha256,
        "p9_bundle_sha256": require_sha("p9 bundle", result.get("p9_bundle_sha256")),
        "canary_pass_sha256": require_sha("canary", final.get("canary_pass_sha256")),
        "rollback_drill_sha256": require_sha("rollback", final.get("rollback_drill_sha256")),
        "raw_token_sha256": require_sha("raw token", result.get("raw_token_sha256")),
    }


def build_preimage(
    *,
    captured_at: str,
    health: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
    ack: Mapping[str, Any],
    ack_file_sha256: str,
    txn_text: str,
    txn_file_sha256: str,
    persistent_binary_sha256: str,
    p10_evidence: Mapping[str, Any],
) -> dict:
    if health.get("schema") != "beglin-supervisor-persistent-health-v1":
        raise P11Error("unexpected health schema")
    if health.get("status") != "ok" or health.get("router_id") != ROUTER_ID:
        raise P11Error("production supervisor is not healthy")
    if health.get("listen_host") != "127.0.0.1":
        raise P11Error("supervisor is not localhost-only")
    if health.get("external_network_exposed") is not False:
        raise P11Error("external exposure changed")
    if health.get("production_write_allowed") is not False:
        raise P11Error("production write flag changed")
    if health.get("auto_promotion_enabled") is not False:
        raise P11Error("automatic promotion changed")

    generation = int(manifest.get("generation"))
    logical_manifest_sha = require_sha("manifest", manifest.get("manifest_sha256"))
    if int(health.get("route_generation")) != generation:
        raise P11Error("route generation mismatch")
    if health.get("route_manifest_sha256") != logical_manifest_sha:
        raise P11Error("route manifest sha mismatch")
    active = dict(manifest.get("active_route") or {})
    if active.get("route_id") != ACTIVE_ROUTE_ID:
        raise P11Error("unexpected active route")
    if health.get("active_route") != active:
        raise P11Error("health active route mismatch")

    worker = (health.get("workers") or {}).get(ACTIVE_ROUTE_ID) or {}
    if worker.get("alive") is not True:
        raise P11Error("candidate worker is not alive")
    runtime_policy = pc.normalize_policy(worker.get("runtime_policy") or [])
    if runtime_policy != BASELINE_POLICY:
        raise P11Error("candidate worker policy is not baseline")
    if worker.get("runtime_policy_hash") != BASELINE_POLICY_HASH:
        raise P11Error("candidate worker policy hash is not baseline")

    ack_policy = pc.normalize_policy(ack.get("active_policy") or [])
    if ack_policy != BASELINE_POLICY or ack.get("active_policy_hash") != BASELINE_POLICY_HASH:
        raise P11Error("runtime ACK policy mismatch")
    epoch = int(ack.get("weight_epoch"))
    if epoch != int(worker.get("weight_epoch")):
        raise P11Error("runtime epoch mismatch")
    ack_sha = require_sha("ack", ack.get("ack_sha256"))
    if worker.get("ack_sha256") != ack_sha:
        raise P11Error("runtime ACK identity mismatch")

    txn = str(txn_text).strip()
    parts = txn.split()
    if len(parts) < 2 or str(ack.get("txn_id") or "") != parts[1]:
        raise P11Error("last transaction does not match terminal ACK")

    preimage = {
        "schema": PREIMAGE_SCHEMA,
        "status": "READY_FOR_APPROVAL_BINDING",
        "captured_at": str(captured_at),
        "router_id": ROUTER_ID,
        "route_generation": generation,
        "route_manifest_sha256": logical_manifest_sha,
        "route_manifest_file_sha256": require_sha("manifest file", manifest_file_sha256),
        "active_route": active,
        "persistent_binary_sha256": require_sha("persistent binary", persistent_binary_sha256),
        "external_network_exposed": False,
        "production_write_allowed": False,
        "automatic_live_promotion": False,
        "worker": {
            "route_id": ACTIVE_ROUTE_ID,
            "pid": int(worker.get("pid")),
            "weight_epoch": epoch,
            "runtime_policy": runtime_policy,
            "runtime_policy_hash": BASELINE_POLICY_HASH,
            "ack_sha256": ack_sha,
            "ack_file_sha256": require_sha("ack file", ack_file_sha256),
            "last_txn_id": parts[1],
            "last_txn_file_sha256": require_sha("txn file", txn_file_sha256),
        },
        "p10_evidence": dict(p10_evidence),
    }
    preimage["preimage_sha256"] = sha256_json(preimage)
    return preimage


def build_plan(preimage: Mapping[str, Any], *, executor_source_sha256: str) -> dict:
    if preimage.get("schema") != PREIMAGE_SCHEMA:
        raise P11Error("unexpected preimage schema")
    check = dict(preimage)
    expected_preimage_sha = check.pop("preimage_sha256", None)
    if require_sha("preimage", expected_preimage_sha) != sha256_json(check):
        raise P11Error("preimage self-hash mismatch")
    worker = preimage["worker"]
    p10 = preimage["p10_evidence"]
    executor_source_sha256 = require_sha("executor source", executor_source_sha256)
    expected_epoch = int(worker["weight_epoch"])
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "AWAITING_TRUSTED_PRODUCTION_APPROVAL",
        "production_cutover_allowed": False,
        "production_write_allowed": False,
        "automatic_live_promotion": False,
        "external_network_exposed": False,
        "preimage_sha256": expected_preimage_sha,
        "p10_result_sha256": p10["p10_result_sha256"],
        "executor_source_sha256": executor_source_sha256,
        "evidence": dict(p10),
        "expected_live_preimage": {
            "route_generation": int(preimage["route_generation"]),
            "route_manifest_sha256": preimage["route_manifest_sha256"],
            "active_route_id": ACTIVE_ROUTE_ID,
            "persistent_binary_sha256": preimage["persistent_binary_sha256"],
            "worker_pid": int(worker["pid"]),
            "weight_epoch": expected_epoch,
            "runtime_policy_hash": BASELINE_POLICY_HASH,
            "ack_sha256": worker["ack_sha256"],
            "last_txn_id": worker["last_txn_id"],
        },
        "target": {
            "runtime_policy": TARGET_POLICY,
            "runtime_policy_hash": TARGET_POLICY_HASH,
            "expected_after_epoch": expected_epoch + 1,
            "changes": [{
                "role": "shared_up_proj",
                "layer": 3,
                "expected_n": 6,
                "target_n": 5,
            }],
        },
        "execution_contract": {
            "kind": "single_worker_precision_epoch_rebind",
            "trusted_approval_required": True,
            "recapture_exact_preimage_before_any_write": True,
            "quiescent_boundary_required": True,
            "route_manifest_change_required": False,
            "startup_route_identity_remains": ACTIVE_ROUTE_ID,
            "expected_reference_after_cutover": [55222, 1],
            "rollback_target_policy_hash": BASELINE_POLICY_HASH,
            "expected_reference_after_rollback": [55222, 372],
        },
    }
    plan["cutover_plan_sha256"] = sha256_json(plan)
    return plan


def build_approval_request(plan: Mapping[str, Any]) -> dict:
    check = dict(plan)
    plan_sha = check.pop("cutover_plan_sha256", None)
    if require_sha("plan", plan_sha) != sha256_json(check):
        raise P11Error("plan self-hash mismatch")
    expected = plan["expected_live_preimage"]
    target = plan["target"]
    req = {
        "schema": REQUEST_SCHEMA,
        "status": "AWAITING_TRUSTED_PRODUCTION_APPROVAL",
        "trusted_production_approval_present": False,
        "production_cutover_allowed": False,
        "automatic_live_promotion": False,
        "cutover_plan_sha256": plan_sha,
        "preimage_sha256": plan["preimage_sha256"],
        "p10_result_sha256": plan["p10_result_sha256"],
        "executor_source_sha256": plan["executor_source_sha256"],
        "route_generation": int(expected["route_generation"]),
        "route_manifest_sha256": expected["route_manifest_sha256"],
        "worker_pid": int(expected["worker_pid"]),
        "weight_epoch": int(expected["weight_epoch"]),
        "ack_sha256": expected["ack_sha256"],
        "candidate_policy_hash": target["runtime_policy_hash"],
        "requested_capability": "ONE_SHOT_PRECISION_EPOCH_CUTOVER",
    }
    req["approval_request_sha256"] = sha256_json(req)
    return req
