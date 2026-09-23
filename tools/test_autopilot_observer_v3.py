#!/usr/bin/env python3
import unittest

from autopilot_observer_v3 import ObservationEvidence, evaluate


def sample(**kw):
    base = dict(
        context_hash="ctx",
        backend="mlx_metal",
        policy_hash="pre",
        weight_epoch=1,
        requests_completed=50,
        tokens_evaluated=500,
        near_tie_events=5,
        reference_checks_attempted=5,
        effective_attribution_checks=5,
        attribution_hits=2,
        run_errors=0,
        median_margin=0.02,
        target_replay_pass=True,
    )
    base.update(kw)
    return ObservationEvidence(**base)


class ObserverV3Tests(unittest.TestCase):
    def test_context_mismatch_is_inconclusive(self):
        got = evaluate(sample(), sample(context_hash="other", weight_epoch=2, policy_hash="post"))
        self.assertEqual(got["status"], "INCONCLUSIVE_CONTEXT_MISMATCH")

    def test_zero_observability_never_passes(self):
        got = evaluate(
            sample(),
            sample(
                policy_hash="post", weight_epoch=2,
                effective_attribution_checks=0, attribution_hits=0,
            ),
        )
        self.assertEqual(got["status"], "INCONCLUSIVE_NO_OBSERVABILITY")

    def test_insufficient_requests_is_inconclusive(self):
        got = evaluate(
            sample(),
            sample(policy_hash="post", weight_epoch=2, requests_completed=10),
        )
        self.assertEqual(got["status"], "INCONCLUSIVE")

    def test_run_error_is_regression(self):
        got = evaluate(
            sample(),
            sample(policy_hash="post", weight_epoch=2, run_errors=1),
        )
        self.assertEqual(got["status"], "REGRESSION_DETECTED")

    def test_lower_rate_and_zero_hits_passes(self):
        got = evaluate(
            sample(),
            sample(
                policy_hash="post", weight_epoch=2,
                effective_attribution_checks=10, attribution_hits=0,
                median_margin=None,
            ),
        )
        self.assertEqual(got["status"], "CANARY_PASS")

    def test_replay_failure_overrides_rate_improvement(self):
        got = evaluate(
            sample(),
            sample(
                policy_hash="post", weight_epoch=2,
                effective_attribution_checks=10, attribution_hits=0,
                target_replay_pass=False,
            ),
        )
        self.assertEqual(got["status"], "REGRESSION_DETECTED")


if __name__ == "__main__":
    unittest.main()
