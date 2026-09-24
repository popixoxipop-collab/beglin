#!/usr/bin/env python3
"""G5 GPU observer -> durable rollback control bridge.

Important invariant: writing a DEMOTE command is not rollback success.
Regression handling is deliberately split into:

  1. persist desired baseline + quarantine, then request one-target demotion
  2. verify a later ROLLBACK_APPLIED runtime ACK before updating applied_state

That split survives controller/worker restarts without silently re-admitting a
failed candidate.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

import autopilot_observer_v3 as observer
import gpu_runtime_control as grc
import precision_context as pc


class GpuObserverControlError(RuntimeError):
    pass


REQUIRED_METRICS = (
    "requests_completed",
    "tokens_evaluated",
    "near_tie_events",
    "reference_checks_attempted",
    "effective_attribution_checks",
    "attribution_hits",
)


def _int_metric(metrics: Mapping, key: str) -> int:
    if key not in metrics:
        raise GpuObserverControlError(f"observer metrics missing {key}")
    try:
        value = int(metrics[key])
    except (TypeError, ValueError) as exc:
        raise GpuObserverControlError(
            f"observer metric {key} must be an integer"
        ) from exc
    if value < 0:
        raise GpuObserverControlError(
            f"observer metric {key} must be non-negative"
        )
    return value


def evidence_from_runtime_ack(
    *,
    context_hash: str,
    runtime_ack: dict,
    metrics: Mapping,
    target_replay_pass: bool | None,
) -> observer.ObservationEvidence:
    """Build scope-matched G5 evidence from runtime-applied state + counters."""
    ack = grc.normalize_runtime_ack(runtime_ack)
    if ack["backend"] != "mlx_metal":
        raise GpuObserverControlError("G5 GPU evidence must be mlx_metal")
    if ack["correction_mode"] != "off":
        raise GpuObserverControlError(
            "G5 GPU observation must run with correction OFF"
        )

    values = {key: _int_metric(metrics, key) for key in REQUIRED_METRICS}
    run_errors = _int_metric(metrics, "run_errors") if "run_errors" in metrics else 0
    if values["attribution_hits"] > values["effective_attribution_checks"]:
        raise GpuObserverControlError(
            "attribution_hits exceeds effective_attribution_checks"
        )
    if (
        values["effective_attribution_checks"]
        > values["reference_checks_attempted"]
    ):
        raise GpuObserverControlError(
            "effective_attribution_checks exceeds reference_checks_attempted"
        )

    margin = metrics.get("median_margin")
    if margin is not None:
        try:
            margin = float(margin)
        except (TypeError, ValueError) as exc:
            raise GpuObserverControlError(
                "median_margin must be numeric or null"
            ) from exc

    return observer.ObservationEvidence(
        context_hash=str(context_hash),
        backend="mlx_metal",
        policy_hash=ack["active_policy_hash"],
        weight_epoch=int(ack["weight_epoch"]),
        requests_completed=values["requests_completed"],
        tokens_evaluated=values["tokens_evaluated"],
        near_tie_events=values["near_tie_events"],
        reference_checks_attempted=values["reference_checks_attempted"],
        effective_attribution_checks=values["effective_attribution_checks"],
        attribution_hits=values["attribution_hits"],
        run_errors=run_errors,
        median_margin=margin,
        target_replay_pass=target_replay_pass,
    )


def _policy_map(policy) -> dict[tuple[str, int], int]:
    return {
        (row["role"], int(row["layer"])): int(row["n"])
        for row in pc.normalize_policy(policy)
    }


def require_single_target_rollback(
    *,
    baseline_policy,
    failed_policy,
    role: str,
    layer: int,
    n: int,
) -> None:
    """Require one-target-at-a-time rollback against the exact failed preimage."""
    before = _policy_map(baseline_policy)
    after = _policy_map(failed_policy)
    key = (str(role), int(layer))
    if after.get(key) != int(n):
        raise GpuObserverControlError(
            f"failed policy does not contain target {role}/L{layer} n={n}"
        )
    differing = {
        k for k in set(before) | set(after)
        if before.get(k) != after.get(k)
    }
    if differing != {key}:
        raise GpuObserverControlError(
            "rollback must change exactly one role/layer from the observed preimage"
        )


def request_regression_rollback(
    *,
    adapter,
    store,
    baseline: observer.ObservationEvidence,
    post: observer.ObservationEvidence,
    baseline_policy,
    failed_policy,
    role: str,
    layer: int,
    n: int,
    txn_id: str,
    evidence_id=None,
    min_requests: int = 50,
    min_effective_checks: int = 1,
) -> dict:
    """Evaluate POST and, only on regression, durably request one-target rollback."""
    verdict = observer.evaluate(
        baseline,
        post,
        min_requests=min_requests,
        min_effective_checks=min_effective_checks,
    )
    result = {
        "verdict": verdict,
        "action": "NONE",
        "rollback_complete": False,
    }
    if verdict["status"] != "REGRESSION_DETECTED":
        return result

    if adapter.name != "mlx_metal" or store.backend != "mlx_metal":
        raise GpuObserverControlError(
            "GPU observer rollback requires mlx_metal adapter/store"
        )
    require_single_target_rollback(
        baseline_policy=baseline_policy,
        failed_policy=failed_policy,
        role=role,
        layer=layer,
        n=n,
    )
    baseline_rows = pc.normalize_policy(baseline_policy)
    failed_rows = pc.normalize_policy(failed_policy)
    baseline_hash = pc.policy_hash(baseline_rows)
    failed_hash = pc.policy_hash(failed_rows)
    if baseline.policy_hash != baseline_hash:
        raise GpuObserverControlError(
            "baseline evidence policy hash does not match rollback baseline"
        )
    if post.policy_hash != failed_hash:
        raise GpuObserverControlError(
            "POST evidence policy hash does not match failed runtime policy"
        )

    actual = adapter.query_applied_state()
    if actual.epoch != post.weight_epoch or actual.policy_hash != post.policy_hash:
        raise GpuObserverControlError(
            "runtime applied state changed after POST observation; rollback command is stale"
        )

    quarantine = store.quarantine_target(
        context_hash=post.context_hash,
        role=role,
        layer=layer,
        n=n,
        reason=verdict["reason"],
        evidence_id=evidence_id,
    )
    desired = store.set_desired(
        context_hash=post.context_hash,
        policy=baseline_rows,
        reason=f"observer rollback {txn_id}: {verdict['reason']}",
        source="gpu_observer_v3",
    )

    request = adapter.request_demote(
        txn_id=txn_id,
        expected_epoch=post.weight_epoch,
        expected_policy_hash=post.policy_hash,
        role=role,
        layer=int(layer),
        expected_n=int(n),
    )
    return {
        **result,
        "action": "ROLLBACK_REQUESTED",
        "txn_id": txn_id,
        "request": request,
        "desired": desired,
        "quarantine": quarantine,
        "expected_baseline_policy_hash": baseline_hash,
    }


def complete_regression_rollback(
    *,
    adapter,
    store,
    txn_id: str,
    context_hash: str,
    baseline_policy,
    failed_epoch: int,
) -> dict:
    """Consume the later runtime ACK; never infer success from command existence."""
    baseline_rows = pc.normalize_policy(baseline_policy)
    baseline_hash = pc.policy_hash(baseline_rows)
    ack = adapter.verify_runtime_txn(txn_id)
    if ack.get("status") != "ROLLBACK_APPLIED":
        return {
            "status": "ROLLBACK_NOT_APPLIED",
            "txn_id": txn_id,
            "runtime_status": ack.get("status"),
            "rollback_complete": False,
        }
    if ack["active_policy_hash"] != baseline_hash:
        raise GpuObserverControlError(
            "rollback ACK policy does not match desired baseline"
        )
    if int(ack["weight_epoch"]) <= int(failed_epoch):
        raise GpuObserverControlError(
            "rollback ACK did not advance weight epoch"
        )
    if ack.get("correction_mode") != "off":
        raise GpuObserverControlError(
            "rollback ACK reports correction mode other than OFF"
        )

    applied = store.set_applied(
        context_hash=context_hash,
        policy=ack["active_policy"],
        epoch=ack["weight_epoch"],
        txn_id=txn_id,
        ack_sha256=ack["ack_sha256"],
    )
    reconcile = store.reconcile()
    if reconcile["status"] != "IN_SYNC":
        raise GpuObserverControlError(
            f"durable control state not in sync after rollback ACK: {reconcile['status']}"
        )
    return {
        "status": "ROLLBACK_APPLIED",
        "txn_id": txn_id,
        "rollback_complete": True,
        "ack": ack,
        "applied": applied,
        "reconcile": reconcile,
    }


def evidence_payload(evidence: observer.ObservationEvidence) -> dict:
    """JSON-friendly helper for observer_state/verdict bundles."""
    value = asdict(evidence)
    value["attribution_rate"] = evidence.attribution_rate
    return value
