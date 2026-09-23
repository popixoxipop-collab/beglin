#!/usr/bin/env python3
"""Backend/context-scoped admission gate for Precision Evidence Contract v3."""
from __future__ import annotations

import precision_context as pc


def build_candidate_policy(current_policy, role, layer, n):
    mapping = {
        (row["role"], int(row["layer"])): int(row["n"])
        for row in pc.normalize_policy(current_policy)
    }
    mapping[(str(role), int(layer))] = int(n)
    return pc.normalize_policy(mapping)


def evaluate_candidate(
    *,
    context: pc.ExecutionContext,
    current_policy,
    current_epoch: int,
    role: str,
    layer: int,
    n: int,
    preflight_evidence: dict | None,
):
    baseline_hash = pc.policy_hash(current_policy)
    candidate_policy = build_candidate_policy(current_policy, role, layer, n)
    candidate_hash = pc.policy_hash(candidate_policy)

    if preflight_evidence is None:
        return {
            "action": "NEEDS_BACKEND_PREFLIGHT",
            "baseline_policy_hash": baseline_hash,
            "candidate_policy_hash": candidate_hash,
            "candidate_policy": candidate_policy,
        }

    if preflight_evidence.get("context_hash") != context.context_hash:
        return {
            "action": "EVIDENCE_CONTEXT_MISMATCH",
            "reason": "preflight context hash differs from current execution context",
        }
    if preflight_evidence.get("baseline_policy_hash") != baseline_hash:
        return {
            "action": "STALE_PREFLIGHT_PREIMAGE",
            "reason": "preflight baseline policy is not the currently applied policy",
        }
    if preflight_evidence.get("requested_policy_hash") != candidate_hash:
        return {
            "action": "EVIDENCE_CANDIDATE_MISMATCH",
            "reason": "preflight candidate policy hash differs from requested candidate",
        }
    if int(preflight_evidence.get("expected_epoch", -1)) != int(current_epoch):
        return {
            "action": "STALE_PREFLIGHT_EPOCH",
            "reason": "preflight expected epoch differs from current epoch",
        }
    if (
        preflight_evidence.get("status") != "passed"
        or preflight_evidence.get("pass") is not True
    ):
        return {
            "action": "BLOCKED_BACKEND_EVIDENCE",
            "reason": preflight_evidence.get("reason") or "backend preflight did not pass",
        }
    if preflight_evidence.get("applied_policy_hash") != candidate_hash:
        return {
            "action": "UNVERIFIED_APPLIED_POLICY",
            "reason": "preflight did not prove the requested policy was actually applied",
        }
    observed_epoch = preflight_evidence.get("observed_epoch")
    if observed_epoch is None or int(observed_epoch) <= int(current_epoch):
        return {
            "action": "UNVERIFIED_EPOCH_TRANSITION",
            "reason": "preflight did not prove a candidate epoch transition",
        }

    return {
        "action": "ADMIT_ONE_TARGET",
        "baseline_policy_hash": baseline_hash,
        "candidate_policy_hash": candidate_hash,
        "candidate_policy": candidate_policy,
        "context_hash": context.context_hash,
        "backend": context.backend,
        "expected_epoch": int(current_epoch),
    }
