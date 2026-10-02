#!/usr/bin/env python3
"""Atomic production-routing cutover gate for Beglin.

The inference engine already supports continuous batching and isolated candidate
workers, but the repository does not currently contain a production ingress
router. This module therefore provides the reviewed *routing contract* and an
atomic file-backed route manifest that a serving supervisor can integrate.

It never discovers or invents a network endpoint. Live traffic cutover is
permitted only when an explicit router-capability document proves that the
supervisor implements atomic CAS, health observation and rollback.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


CUTOVER_PLAN_SCHEMA = "beglin-production-routing-cutover-plan/1"
ROUTE_SCHEMA = "beglin-serving-route-v1"
ROUTE_MANIFEST_SCHEMA = "beglin-serving-route-manifest/1"
ROUTER_CAPABILITY_SCHEMA = "beglin-serving-router-capability/1"
CUTOVER_APPROVAL_SCHEMA = "beglin-production-routing-cutover-approval-v1"


class RoutingCutoverError(RuntimeError):
    pass


class StaleRoute(RoutingCutoverError):
    pass


class ServingRouterUnavailable(RoutingCutoverError):
    pass


class CutoverAuthorizationError(RoutingCutoverError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _hex64(name: str, value: Any) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise RoutingCutoverError(f"{name} must be 64 lowercase hex chars")
    return value


def normalize_route(value: Mapping[str, Any]) -> dict:
    route = {
        "schema": ROUTE_SCHEMA,
        "route_id": str(value.get("route_id", "")),
        "worker_instance_id": str(value.get("worker_instance_id", "")),
        "endpoint": str(value.get("endpoint", "")),
        "source_commit": str(value.get("source_commit", "")),
        "binary_sha256": _hex64("binary_sha256", value.get("binary_sha256", "")),
        "checkpoint_sha256": _hex64("checkpoint_sha256", value.get("checkpoint_sha256", "")),
        "policy_hash": _hex64("policy_hash", value.get("policy_hash", "")),
        "role": str(value.get("role", "")),
        "layer": int(value.get("layer")),
        "n": value.get("n"),
    }
    for key in ("route_id", "worker_instance_id", "endpoint", "source_commit"):
        if not route[key]:
            raise RoutingCutoverError(f"{key} must be non-empty")
    if route["layer"] < 0:
        raise RoutingCutoverError("layer must be non-negative")
    if route["n"] is not None:
        route["n"] = int(route["n"])
        if route["n"] <= 0:
            raise RoutingCutoverError("n must be positive when present")
    return route


def normalize_router_capability(value: Mapping[str, Any]) -> dict:
    cap = {
        "schema": str(value.get("schema", "")),
        "status": str(value.get("status", "")),
        "router_id": str(value.get("router_id", "")),
        "implementation_verified": value.get("implementation_verified"),
        "atomic_cas": value.get("atomic_cas"),
        "health_observation": value.get("health_observation"),
        "rollback": value.get("rollback"),
        "request_drain": value.get("request_drain"),
        "route_store_kind": str(value.get("route_store_kind", "")),
    }
    if cap["schema"] != ROUTER_CAPABILITY_SCHEMA:
        raise RoutingCutoverError("unsupported router capability schema")
    return cap


def router_is_verified(capability: Mapping[str, Any]) -> bool:
    cap = normalize_router_capability(capability)
    return (
        cap["status"] == "VERIFIED"
        and cap["implementation_verified"] is True
        and cap["atomic_cas"] is True
        and cap["health_observation"] is True
        and cap["rollback"] is True
        and cap["request_drain"] is True
        and bool(cap["router_id"])
        and bool(cap["route_store_kind"])
    )


def build_cutover_plan(
    *,
    bridge_status: Mapping[str, Any],
    baseline_route: Mapping[str, Any],
    candidate_route: Mapping[str, Any],
    router_capability: Mapping[str, Any],
) -> dict:
    bridge = dict(bridge_status)
    if bridge.get("schema") != "beglin-production-adapter-bridge-execution/1":
        raise RoutingCutoverError("unsupported production-adapter bridge evidence")
    verdict = bridge.get("verdict")
    result = bridge.get("result")
    if not isinstance(verdict, dict) or not isinstance(result, dict):
        raise RoutingCutoverError("bridge evidence is incomplete")
    if verdict.get("status") != "CANARY_PASS_REVIEW_REQUIRED":
        raise RoutingCutoverError("isolated bridge canary has not passed")
    if verdict.get("decision") != "NO_AUTO_PROMOTION":
        raise RoutingCutoverError("bridge verdict does not preserve manual review")
    if verdict.get("production_routing_switch_allowed") is not False:
        raise RoutingCutoverError("bridge verdict unexpectedly enabled routing")
    if result.get("production_routing_switched") is not False:
        raise RoutingCutoverError("bridge execution already changed production routing")
    if result.get("baseline_worker_mutated") is not False:
        raise RoutingCutoverError("bridge execution mutated baseline worker")
    if result.get("candidate_worker_exited") is not True:
        raise RoutingCutoverError("bridge candidate worker ownership is unresolved")
    if result.get("reference_token_match") is not True or result.get("finite_logits") is not True:
        raise RoutingCutoverError("bridge result lacks correctness proof")

    baseline = normalize_route(baseline_route)
    candidate = normalize_route(candidate_route)
    if baseline["route_id"] == candidate["route_id"]:
        raise RoutingCutoverError("baseline and candidate route ids must differ")
    if baseline["source_commit"] != result.get("source_commit"):
        raise RoutingCutoverError("baseline source identity mismatch")
    if candidate["source_commit"] != result.get("source_commit"):
        raise RoutingCutoverError("candidate source identity mismatch")
    if baseline["binary_sha256"] != result.get("binary_sha256"):
        raise RoutingCutoverError("baseline binary identity mismatch")
    if candidate["binary_sha256"] != result.get("binary_sha256"):
        raise RoutingCutoverError("candidate binary identity mismatch")
    if baseline["checkpoint_sha256"] != result.get("checkpoint_sha256"):
        raise RoutingCutoverError("baseline checkpoint identity mismatch")
    if candidate["checkpoint_sha256"] != result.get("checkpoint_sha256"):
        raise RoutingCutoverError("candidate checkpoint identity mismatch")
    if candidate["policy_hash"] != result.get("candidate_policy_hash"):
        raise RoutingCutoverError("candidate policy hash mismatch")
    if candidate["role"] != "shared_up_proj" or candidate["layer"] != 3 or candidate["n"] != 6:
        raise RoutingCutoverError("candidate route is outside reviewed target scope")

    cap = normalize_router_capability(router_capability)
    verified = router_is_verified(cap)
    plan = {
        "schema": CUTOVER_PLAN_SCHEMA,
        "status": (
            "READY_FOR_EXPLICIT_CUTOVER_APPROVAL"
            if verified
            else "BLOCKED_NO_VERIFIED_SERVING_ROUTER"
        ),
        "bridge_plan_sha256": str(bridge["plan"]["plan_sha256"]),
        "bridge_result_sha256": str(result["result_sha256"]),
        "bridge_verdict_sha256": str(verdict["verdict_sha256"]),
        "baseline_route": baseline,
        "candidate_route": candidate,
        "router_capability": cap,
        "router_verified": verified,
        "cutover_authorization_required": True,
        "automatic_cutover_allowed": False,
        "production_routing_switch_allowed": False,
        "rollback_required_on_any_health_failure": True,
        "rollback_route_id": baseline["route_id"],
        "candidate_route_id": candidate["route_id"],
        "health_window": {
            "min_requests": 12,
            "max_error_rate": 0.0,
            "require_reference_parity": True,
            "max_duration_ms": 30000,
        },
    }
    plan["plan_sha256"] = sha256_json(plan)
    return plan


def validate_cutover_approval(
    approval: Mapping[str, Any],
    *,
    cutover_plan: Mapping[str, Any],
) -> dict:
    a = dict(approval)
    if a.get("schema") != CUTOVER_APPROVAL_SCHEMA:
        raise CutoverAuthorizationError("unsupported cutover approval schema")
    if a.get("status") != "VERIFIED_GITHUB_OIDC":
        raise CutoverAuthorizationError("cutover approval is not VERIFIED_GITHUB_OIDC")
    if a.get("cutover_plan_sha256") != cutover_plan.get("plan_sha256"):
        raise CutoverAuthorizationError("cutover approval plan hash mismatch")
    if a.get("candidate_route_id") != cutover_plan.get("candidate_route_id"):
        raise CutoverAuthorizationError("cutover approval candidate route mismatch")
    if a.get("baseline_route_id") != cutover_plan.get("rollback_route_id"):
        raise CutoverAuthorizationError("cutover approval baseline route mismatch")
    if a.get("automatic_cutover_allowed") is not False:
        raise CutoverAuthorizationError("automatic cutover must remain false")
    if a.get("production_routing_switch_allowed") is not True:
        raise CutoverAuthorizationError("approval does not authorize routing switch")
    return a


def _fsync_parent(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    fd = os.open(str(path.parent), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_parent(path)


class AtomicRouteManifestStore:
    """A compare-and-swap route manifest for a future serving supervisor."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read(self) -> dict:
        if not self.path.is_file():
            raise RoutingCutoverError(f"route manifest is missing: {self.path}")
        with open(self.path) as handle:
            value = json.load(handle)
        if value.get("schema") != ROUTE_MANIFEST_SCHEMA:
            raise RoutingCutoverError("unsupported route manifest schema")
        return value

    def initialize(self, route: Mapping[str, Any]) -> dict:
        if self.path.exists():
            raise RoutingCutoverError("route manifest already exists")
        normalized = normalize_route(route)
        value = {
            "schema": ROUTE_MANIFEST_SCHEMA,
            "generation": 1,
            "active_route": normalized,
            "previous_route": None,
            "last_reason": "initialize",
            "last_authorization_digest": None,
        }
        value["manifest_sha256"] = sha256_json(value)
        _atomic_json(self.path, value)
        return value

    def compare_and_swap(
        self,
        *,
        expected_generation: int,
        expected_route_id: str,
        new_route: Mapping[str, Any],
        reason: str,
        authorization_digest: str,
    ) -> dict:
        current = self.read()
        if int(current["generation"]) != int(expected_generation):
            raise StaleRoute(
                f"route generation changed: expected={expected_generation} "
                f"actual={current['generation']}"
            )
        if current["active_route"]["route_id"] != str(expected_route_id):
            raise StaleRoute(
                f"active route changed: expected={expected_route_id!r} "
                f"actual={current['active_route']['route_id']!r}"
            )
        new = normalize_route(new_route)
        value = {
            "schema": ROUTE_MANIFEST_SCHEMA,
            "generation": int(current["generation"]) + 1,
            "active_route": new,
            "previous_route": copy.deepcopy(current["active_route"]),
            "last_reason": str(reason),
            "last_authorization_digest": _hex64(
                "authorization_digest", authorization_digest
            ),
        }
        value["manifest_sha256"] = sha256_json(value)
        _atomic_json(self.path, value)
        return value


def execute_cutover(
    *,
    store: AtomicRouteManifestStore,
    cutover_plan: Mapping[str, Any],
    approval: Mapping[str, Any],
) -> dict:
    plan = dict(cutover_plan)
    if plan.get("status") != "READY_FOR_EXPLICIT_CUTOVER_APPROVAL":
        raise ServingRouterUnavailable(
            "no verified serving router is available for production cutover"
        )
    if plan.get("router_verified") is not True:
        raise ServingRouterUnavailable("router capability is not verified")
    auth = validate_cutover_approval(approval, cutover_plan=plan)
    current = store.read()
    if current["active_route"] != plan["baseline_route"]:
        raise StaleRoute("current active route no longer matches approved baseline")
    updated = store.compare_and_swap(
        expected_generation=int(current["generation"]),
        expected_route_id=plan["rollback_route_id"],
        new_route=plan["candidate_route"],
        reason="explicit reviewed production cutover",
        authorization_digest=sha256_json(auth),
    )
    return {
        "schema": "beglin-production-routing-cutover-result/1",
        "status": "CUTOVER_COMMITTED_AWAITING_HEALTH",
        "generation": updated["generation"],
        "active_route": updated["active_route"],
        "previous_route": updated["previous_route"],
        "production_routing_switched": True,
        "automatic_cutover_used": False,
        "health_review_required": True,
        "manifest_sha256": updated["manifest_sha256"],
    }


def evaluate_health(
    *,
    cutover_plan: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict:
    window = cutover_plan["health_window"]
    try:
        requests = int(metrics["requests"])
        errors = int(metrics["errors"])
        duration_ms = int(metrics["duration_ms"])
        reference_parity = bool(metrics["reference_parity"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RoutingCutoverError("health metrics are incomplete") from exc
    if requests <= 0 or errors < 0 or duration_ms < 0:
        raise RoutingCutoverError("health metrics are out of range")
    error_rate = errors / requests
    reasons = []
    if requests < int(window["min_requests"]):
        reasons.append("insufficient_requests")
    if error_rate > float(window["max_error_rate"]):
        reasons.append("error_rate")
    if duration_ms > int(window["max_duration_ms"]):
        reasons.append("duration")
    if window["require_reference_parity"] and not reference_parity:
        reasons.append("reference_parity")
    if reasons:
        return {
            "schema": "beglin-production-routing-health-verdict/1",
            "status": "ROLLBACK_REQUIRED",
            "reasons": reasons,
            "requests": requests,
            "errors": errors,
            "error_rate": error_rate,
            "duration_ms": duration_ms,
            "reference_parity": reference_parity,
        }
    return {
        "schema": "beglin-production-routing-health-verdict/1",
        "status": "CUTOVER_HEALTH_PASS_REVIEW_REQUIRED",
        "reasons": [],
        "requests": requests,
        "errors": errors,
        "error_rate": error_rate,
        "duration_ms": duration_ms,
        "reference_parity": reference_parity,
    }


def rollback(
    *,
    store: AtomicRouteManifestStore,
    cutover_plan: Mapping[str, Any],
    cutover_result: Mapping[str, Any],
    reason: str,
) -> dict:
    current = store.read()
    candidate = cutover_plan["candidate_route"]
    baseline = cutover_plan["baseline_route"]
    if current["active_route"] != candidate:
        raise StaleRoute("rollback precondition: candidate is not the active route")
    auth_digest = str(current.get("last_authorization_digest") or "")
    updated = store.compare_and_swap(
        expected_generation=int(current["generation"]),
        expected_route_id=candidate["route_id"],
        new_route=baseline,
        reason=str(reason),
        authorization_digest=auth_digest,
    )
    return {
        "schema": "beglin-production-routing-rollback-result/1",
        "status": "ROLLBACK_COMMITTED",
        "generation": updated["generation"],
        "active_route": updated["active_route"],
        "previous_route": updated["previous_route"],
        "production_routing_switched": True,
        "restored_baseline": True,
        "manifest_sha256": updated["manifest_sha256"],
    }
