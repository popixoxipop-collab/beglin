#!/usr/bin/env python3
"""Backend/context-scoped P4 observer semantics for Precision Evidence v3."""
from __future__ import annotations

from dataclasses import dataclass


TERMINAL = {
    "CANARY_PASS",
    "INCONCLUSIVE",
    "INCONCLUSIVE_NO_OBSERVABILITY",
    "INCONCLUSIVE_CONTEXT_MISMATCH",
    "REGRESSION_DETECTED",
}


@dataclass(frozen=True)
class ObservationEvidence:
    context_hash: str
    backend: str
    policy_hash: str
    weight_epoch: int
    requests_completed: int
    tokens_evaluated: int
    near_tie_events: int
    reference_checks_attempted: int
    effective_attribution_checks: int
    attribution_hits: int
    run_errors: int = 0
    median_margin: float | None = None
    target_replay_pass: bool | None = None

    @property
    def attribution_rate(self):
        if self.effective_attribution_checks <= 0:
            return None
        return self.attribution_hits / self.effective_attribution_checks


def evaluate(
    baseline: ObservationEvidence,
    post: ObservationEvidence,
    *,
    min_requests: int = 50,
    min_effective_checks: int = 1,
):
    if baseline.context_hash != post.context_hash or baseline.backend != post.backend:
        return {
            "status": "INCONCLUSIVE_CONTEXT_MISMATCH",
            "reason": "PRE/POST execution context or backend differs",
        }
    if post.weight_epoch <= baseline.weight_epoch:
        return {
            "status": "INCONCLUSIVE_CONTEXT_MISMATCH",
            "reason": "POST weight epoch did not advance",
        }
    if baseline.policy_hash == post.policy_hash:
        return {
            "status": "INCONCLUSIVE_CONTEXT_MISMATCH",
            "reason": "PRE/POST policy hash is identical",
        }
    if post.run_errors:
        return {
            "status": "REGRESSION_DETECTED",
            "reason": f"POST run_errors={post.run_errors}",
        }
    if post.effective_attribution_checks < min_effective_checks:
        return {
            "status": "INCONCLUSIVE_NO_OBSERVABILITY",
            "reason": "no effective attribution checks in POST window",
        }
    if post.requests_completed < min_requests:
        return {
            "status": "INCONCLUSIVE",
            "reason": (
                f"POST requests {post.requests_completed} < required {min_requests}"
            ),
        }
    if post.target_replay_pass is False:
        return {
            "status": "REGRESSION_DETECTED",
            "reason": "matched target replay failed after promotion",
        }

    before_rate = baseline.attribution_rate
    after_rate = post.attribution_rate
    if before_rate is None:
        return {
            "status": "INCONCLUSIVE_NO_OBSERVABILITY",
            "reason": "baseline had no effective attribution checks",
        }
    if after_rate is None:
        return {
            "status": "INCONCLUSIVE_NO_OBSERVABILITY",
            "reason": "POST attribution rate unavailable",
        }

    fewer = after_rate < before_rate
    margin_ok = (
        post.attribution_hits == 0
        or (
            post.median_margin is not None
            and baseline.median_margin is not None
            and post.median_margin > baseline.median_margin
        )
    )
    if fewer and margin_ok and post.target_replay_pass is not False:
        return {
            "status": "CANARY_PASS",
            "reason": "attribution rate decreased and remaining margin did not regress",
            "baseline_rate": before_rate,
            "post_rate": after_rate,
        }
    return {
        "status": "REGRESSION_DETECTED",
        "reason": "attribution rate/margin criteria not satisfied",
        "baseline_rate": before_rate,
        "post_rate": after_rate,
    }
