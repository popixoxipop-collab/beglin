#!/usr/bin/env python3
import unittest

import precision_context as pc
from autopilot_observer_v3 import ObservationEvidence
import gpu_restart_canary as gc


BASE = [{"role": "shared_up_proj", "layer": 3, "n": 5}]
CAND = BASE + [{"role": "shared_down_proj", "layer": 4, "n": 6}]


def admission(candidate=CAND):
    return {
        "action": "ADMIT_ONE_TARGET_RESTART_CANARY",
        "baseline_policy_hash": pc.policy_hash(BASE),
        "candidate_policy_hash": pc.policy_hash(candidate),
        "candidate_policy": pc.normalize_policy(candidate),
        "context_hash": "c" * 64,
        "backend": "mlx_metal",
        "expected_epoch": 7,
        "evidence_mode": "isolated_restart",
        "binary_sha256": "a" * 64,
    }


def obs(**kw):
    value = dict(
        context_hash="c" * 64,
        backend="mlx_metal",
        policy_hash=pc.policy_hash(BASE),
        weight_epoch=7,
        requests_completed=60,
        tokens_evaluated=600,
        near_tie_events=5,
        reference_checks_attempted=5,
        effective_attribution_checks=5,
        attribution_hits=2,
        run_errors=0,
        median_margin=0.02,
        target_replay_pass=True,
        worker_instance_id="pre",
    )
    value.update(kw)
    return ObservationEvidence(**value)


class RestartCanaryTests(unittest.TestCase):
    def test_build_requires_exactly_one_target_delta(self):
        plan = gc.build_restart_canary(admission(), baseline_policy=BASE)
        self.assertEqual(plan["target"]["role"], "shared_down_proj")
        self.assertEqual(plan["target"]["layer"], 4)
        self.assertIsNone(plan["target"]["from_n"])
        self.assertEqual(plan["target"]["to_n"], 6)
        self.assertFalse(plan["limits"]["auto_expand"])

        two = CAND + [{"role": "shared_gate_proj", "layer": 14, "n": 6}]
        with self.assertRaises(gc.GpuCanaryError):
            gc.build_restart_canary(admission(two), baseline_policy=BASE)

    def test_pass_never_auto_expands(self):
        plan = gc.build_restart_canary(admission(), baseline_policy=BASE)
        result = gc.evaluate_restart_canary(
            plan,
            obs(),
            obs(
                policy_hash=pc.policy_hash(CAND),
                weight_epoch=1,
                worker_instance_id="post",
                effective_attribution_checks=10,
                attribution_hits=0,
                median_margin=None,
            ),
        )
        self.assertEqual(result["status"], "CANARY_PASS")
        self.assertEqual(result["decision"], "CANARY_PASS_NO_AUTO_EXPANSION")
        self.assertFalse(result["rollback_required"])
        self.assertFalse(result["auto_expand"])

    def test_regression_requires_rollback(self):
        plan = gc.build_restart_canary(admission(), baseline_policy=BASE)
        result = gc.evaluate_restart_canary(
            plan,
            obs(),
            obs(
                policy_hash=pc.policy_hash(CAND),
                weight_epoch=1,
                worker_instance_id="post",
                run_errors=1,
            ),
        )
        self.assertEqual(result["status"], "REGRESSION_DETECTED")
        self.assertTrue(result["rollback_required"])
        self.assertEqual(result["decision"], "ROLLBACK_REQUIRED")

    def test_inconclusive_requires_rollback(self):
        plan = gc.build_restart_canary(
            admission(), baseline_policy=BASE, min_requests=50
        )
        result = gc.evaluate_restart_canary(
            plan,
            obs(),
            obs(
                policy_hash=pc.policy_hash(CAND),
                weight_epoch=1,
                worker_instance_id="post",
                requests_completed=10,
            ),
        )
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertTrue(result["rollback_required"])
        self.assertEqual(result["decision"], "ROLLBACK_REQUIRED_INCONCLUSIVE")

    def test_budget_exceeded_requires_rollback_before_observer_pass(self):
        plan = gc.build_restart_canary(
            admission(), baseline_policy=BASE, min_requests=50, max_requests=80
        )
        result = gc.evaluate_restart_canary(
            plan,
            obs(),
            obs(
                policy_hash=pc.policy_hash(CAND),
                weight_epoch=1,
                worker_instance_id="post",
                requests_completed=81,
                effective_attribution_checks=20,
                attribution_hits=0,
            ),
        )
        self.assertEqual(result["status"], "CANARY_BUDGET_EXCEEDED")
        self.assertTrue(result["rollback_required"])

    def test_observation_binding_mismatch_fails_closed(self):
        plan = gc.build_restart_canary(admission(), baseline_policy=BASE)
        with self.assertRaises(gc.GpuCanaryError):
            gc.evaluate_restart_canary(
                plan,
                obs(),
                obs(
                    context_hash="d" * 64,
                    policy_hash=pc.policy_hash(CAND),
                    worker_instance_id="post",
                ),
            )


if __name__ == "__main__":
    unittest.main()
