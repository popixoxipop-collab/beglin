#!/usr/bin/env python3
"""Fail-closed G6 restart-canary controller for MLX/Metal precision changes."""
from __future__ import annotations

import precision_context as pc
from autopilot_observer_v3 import ObservationEvidence, evaluate as evaluate_observer


SCHEMA = "gpu-restart-canary-v1"
BACKEND = "mlx_metal"


class GpuCanaryError(RuntimeError):
    pass


def _mapping(policy):
    return {
        (r["role"], int(r["layer"])): int(r["n"])
        for r in pc.normalize_policy(policy)
    }


def one_target_delta(baseline_policy, candidate_policy) -> dict:
    before = _mapping(baseline_policy)
    after = _mapping(candidate_policy)
    keys = sorted(set(before) | set(after))
    changed = [
        key for key in keys
        if before.get(key) != after.get(key)
    ]
    if len(changed) != 1:
        raise GpuCanaryError(
            f"restart canary requires exactly one target delta, got {len(changed)}"
        )
    role, layer = changed[0]
    if after.get((role, layer)) is None:
        raise GpuCanaryError("restart canary cannot be a removal-only policy change")
    return {
        "role": role,
        "layer": int(layer),
        "from_n": before.get((role, layer)),
        "to_n": int(after[(role, layer)]),
    }


def build_restart_canary(
    admission: dict,
    *,
    baseline_policy,
    min_requests: int = 50,
    max_requests: int = 100,
    min_effective_checks: int = 1,
) -> dict:
    if admission.get("action") != "ADMIT_ONE_TARGET_RESTART_CANARY":
        raise GpuCanaryError(
            f"planner did not admit restart canary: {admission.get('action')!r}"
        )
    if admission.get("backend") != BACKEND:
        raise GpuCanaryError("restart canary requires mlx_metal backend")
    if admission.get("evidence_mode") != "isolated_restart":
        raise GpuCanaryError("restart canary requires isolated_restart evidence")

    baseline = pc.normalize_policy(baseline_policy)
    candidate = pc.normalize_policy(admission.get("candidate_policy") or [])
    baseline_hash = pc.policy_hash(baseline)
    candidate_hash = pc.policy_hash(candidate)
    if admission.get("baseline_policy_hash") != baseline_hash:
        raise GpuCanaryError("planner baseline hash no longer matches canary baseline")
    if admission.get("candidate_policy_hash") != candidate_hash:
        raise GpuCanaryError("planner candidate hash no longer matches canary candidate")

    min_requests = int(min_requests)
    max_requests = int(max_requests)
    min_effective_checks = int(min_effective_checks)
    if min_requests <= 0 or max_requests < min_requests:
        raise GpuCanaryError("invalid request budget")
    if min_effective_checks <= 0:
        raise GpuCanaryError("min_effective_checks must be positive")

    target = one_target_delta(baseline, candidate)
    return {
        "schema": SCHEMA,
        "mode": "restart",
        "backend": BACKEND,
        "context_hash": admission["context_hash"],
        "binary_sha256": admission["binary_sha256"],
        "expected_live_epoch": int(admission["expected_epoch"]),
        "baseline_policy": baseline,
        "baseline_policy_hash": baseline_hash,
        "candidate_policy": candidate,
        "candidate_policy_hash": candidate_hash,
        "target": target,
        "limits": {
            "min_requests": min_requests,
            "max_requests": max_requests,
            "min_effective_checks": min_effective_checks,
            "auto_expand": False,
        },
        "rollback_policy": baseline,
        "rollback_policy_hash": baseline_hash,
    }


def _bind_observation(plan, evidence: ObservationEvidence, expected_policy_hash: str, label: str):
    if evidence.context_hash != plan["context_hash"]:
        raise GpuCanaryError(f"{label} context hash mismatch")
    if evidence.backend != BACKEND:
        raise GpuCanaryError(f"{label} backend mismatch")
    if evidence.policy_hash != expected_policy_hash:
        raise GpuCanaryError(f"{label} policy hash mismatch")


def evaluate_restart_canary(
    plan: dict,
    baseline: ObservationEvidence,
    post: ObservationEvidence,
) -> dict:
    if plan.get("schema") != SCHEMA or plan.get("mode") != "restart":
        raise GpuCanaryError("invalid restart-canary plan")
    _bind_observation(plan, baseline, plan["baseline_policy_hash"], "PRE")
    _bind_observation(plan, post, plan["candidate_policy_hash"], "POST")

    limits = plan["limits"]
    if int(post.requests_completed) > int(limits["max_requests"]):
        return {
            "status": "CANARY_BUDGET_EXCEEDED",
            "decision": "ROLLBACK_REQUIRED",
            "rollback_required": True,
            "auto_expand": False,
            "reason": (
                f"POST requests {post.requests_completed} exceeded "
                f"max_requests={limits['max_requests']}"
            ),
        }

    verdict = evaluate_observer(
        baseline,
        post,
        min_requests=int(limits["min_requests"]),
        min_effective_checks=int(limits["min_effective_checks"]),
        transition_mode="restart",
    )
    status = verdict["status"]
    if status == "CANARY_PASS":
        decision = "CANARY_PASS_NO_AUTO_EXPANSION"
        rollback_required = False
    elif status == "REGRESSION_DETECTED":
        decision = "ROLLBACK_REQUIRED"
        rollback_required = True
    else:
        # Inconclusive is not permission to leave a new GPU policy promoted.
        decision = "ROLLBACK_REQUIRED_INCONCLUSIVE"
        rollback_required = True
    return {
        "status": status,
        "decision": decision,
        "rollback_required": rollback_required,
        "auto_expand": False,
        "target": dict(plan["target"]),
        "rollback_policy_hash": plan["rollback_policy_hash"],
        "observer": verdict,
    }
