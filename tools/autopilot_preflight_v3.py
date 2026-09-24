#!/usr/bin/env python3
"""Backend-local A/B/R preflight contract for Precision Evidence v3."""
from __future__ import annotations

import copy

import precision_context as pc
from backend_adapters import BackendAdapter


class PreflightV3Error(RuntimeError):
    pass


def _validate_result(result, *, context_hash, backend, policy_hash, label):
    if result.get("context_hash") != context_hash:
        raise PreflightV3Error(f"{label} context hash mismatch")
    if result.get("backend") != backend:
        raise PreflightV3Error(f"{label} backend mismatch")
    if result.get("applied_policy_hash") != policy_hash:
        raise PreflightV3Error(f"{label} applied policy hash mismatch")
    if result.get("correction_mode") != "off":
        raise PreflightV3Error(f"{label} correction must be OFF")
    if result.get("finite_logits") is not True:
        raise PreflightV3Error(f"{label} logits are not finite")
    if result.get("run_error"):
        raise PreflightV3Error(f"{label} backend run error: {result['run_error']}")


def run_preflight(
    adapter: BackendAdapter,
    *,
    expected_context_hash: str,
    expected_epoch: int,
    baseline_policy,
    candidate_policy,
    event: dict,
    reference: dict,
):
    """Run baseline and candidate on one isolated backend worker.

    reference is immutable external oracle evidence. It must carry
    emitted_token, run_id, and context_hash. Its context may differ from
    the candidate backend because it is reference scope, not admission evidence.
    """
    context = adapter.collect_context()
    pc.require_same_context(expected_context_hash, context)
    if context.backend != adapter.name:
        raise PreflightV3Error("adapter/backend context mismatch")

    baseline_policy = pc.normalize_policy(baseline_policy)
    candidate_policy = pc.normalize_policy(candidate_policy)
    baseline_hash = pc.policy_hash(baseline_policy)
    candidate_hash = pc.policy_hash(candidate_policy)

    state = adapter.query_applied_state()
    if state.epoch != int(expected_epoch):
        raise PreflightV3Error(
            f"stale worker epoch: expected={expected_epoch} actual={state.epoch}"
        )
    if state.policy_hash != baseline_hash:
        raise PreflightV3Error(
            f"baseline policy mismatch: expected={baseline_hash} "
            f"actual={state.policy_hash}"
        )

    for key in ("orig_token", "corrected_token", "req", "pos"):
        if key not in event:
            raise PreflightV3Error(f"event missing {key}")
    for key in ("emitted_token", "run_id", "context_hash"):
        if key not in reference:
            raise PreflightV3Error(f"reference missing {key}")

    baseline = adapter.run_validation({
        "validation_key": "baseline",
        "event": copy.deepcopy(event),
        "correction_mode": "off",
    })
    _validate_result(
        baseline,
        context_hash=context.context_hash,
        backend=adapter.name,
        policy_hash=baseline_hash,
        label="baseline",
    )
    if int(baseline.get("emitted_token")) != int(event["orig_token"]):
        raise PreflightV3Error(
            "baseline no longer reproduces the recorded original token"
        )

    snapshot = None
    candidate = None
    restore_state = None
    try:
        adapter.pause_admission()
        adapter.drain()
        adapter.synchronize()
        snapshot = adapter.snapshot()

        applied = adapter.apply_policy(candidate_policy)
        if applied.policy_hash != candidate_hash:
            raise PreflightV3Error("candidate apply returned wrong policy hash")
        adapter.synchronize()

        candidate = adapter.run_validation({
            "validation_key": "candidate",
            "event": copy.deepcopy(event),
            "correction_mode": "off",
        })
        _validate_result(
            candidate,
            context_hash=context.context_hash,
            backend=adapter.name,
            policy_hash=candidate_hash,
            label="candidate",
        )

        if int(candidate.get("emitted_token")) != int(reference["emitted_token"]):
            raise PreflightV3Error(
                "candidate token does not match immutable reference token"
            )
        if int(reference["emitted_token"]) != int(event["corrected_token"]):
            raise PreflightV3Error(
                "reference token does not match recorded corrected token"
            )

        adapter.synchronize()
        restore_state = adapter.restore(snapshot)
        adapter.verify_policy(baseline_hash)
        adapter.resume_admission()
    except Exception:
        if snapshot is not None:
            try:
                adapter.synchronize()
                adapter.restore(snapshot)
                adapter.verify_policy(baseline_hash)
                adapter.resume_admission()
            except Exception as rollback_exc:
                raise PreflightV3Error(
                    f"preflight failed and isolated rollback failed: {rollback_exc}"
                )
        raise

    final = adapter.query_applied_state()
    if final.policy_hash != baseline_hash:
        raise PreflightV3Error("isolated worker did not restore baseline policy")

    return {
        "status": "passed",
        "backend": adapter.name,
        "context_hash": context.context_hash,
        "baseline_policy_hash": baseline_hash,
        "candidate_policy_hash": candidate_hash,
        "baseline_epoch": int(expected_epoch),
        "candidate_epoch": candidate.get("weight_epoch"),
        "restored_epoch": restore_state.epoch if restore_state else None,
        "event": copy.deepcopy(event),
        "reference": copy.deepcopy(reference),
        "baseline": baseline,
        "candidate": candidate,
    }
