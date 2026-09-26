#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest

import gpu_coverage_f as f
from autopilot_observer_v3 import ObservationEvidence


def obs(**kw):
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


class GpuCoverageFTests(unittest.TestCase):
    def test_plan_is_bounded_and_real_gpu_waits_for_b(self):
        got = f.bounded_plan(base_sha="abc")
        self.assertFalse(got["real_gpu_ready"])
        self.assertEqual(got["budgets"]["max_candidates_per_target"], 3)
        self.assertEqual(got["budgets"]["max_prompts"], 3)
        self.assertEqual(got["budgets"]["max_requests_per_run"], 100)
        self.assertEqual(got["budgets"]["max_restarts"], 12)
        kv = got["planned_targets"][0]
        self.assertEqual(kv["bad_candidate_budget"], [5, 6, 7])

    def test_verified_b_without_lease_stays_blocked(self):
        got = f.bounded_plan(
            base_sha="abc",
            checkpoint_handoff={
                "status": "VERIFIED",
                "checkpoint_sha256": "a" * 64,
                "manifest_path": "/scratch/checkpoint_identity.json",
            },
        )
        self.assertFalse(got["real_gpu_ready"])
        self.assertIn("GPU lease", got["real_gpu_block_reason"])

    def test_verified_b_and_a0_lease_enable_real_gpu_gate(self):
        got = f.bounded_plan(
            base_sha="abc",
            checkpoint_handoff={
                "status": "VERIFIED",
                "checkpoint_sha256": "a" * 64,
                "manifest_path": "/scratch/checkpoint_identity.json",
            },
            lease_handoff={
                "state": "GRANTED",
                "owner": "F",
                "resource": "xox_gpu_model_io",
                "lease_id": "lease-1",
            },
        )
        self.assertTrue(got["real_gpu_ready"])
        self.assertEqual(got["gpu_lease_id"], "lease-1")
        self.assertFalse(got["resource_contract"]["production_mutation_allowed"])
        self.assertFalse(got["resource_contract"]["shadow_history_mutation_allowed"])

    def test_wrong_owner_or_resource_does_not_open_gate(self):
        for lease in (
            {"state": "GRANTED", "owner": "B", "resource": "xox_gpu_model_io"},
            {"state": "ACTIVE", "owner": "F", "resource": "eoe_cpu_build"},
            {"state": "REQUESTED", "owner": "F", "resource": "xox_gpu_model_io"},
        ):
            got = f.bounded_plan(
                base_sha="abc",
                checkpoint_handoff={
                    "status": "VERIFIED",
                    "checkpoint_sha256": "a" * 64,
                    "manifest_path": "/scratch/checkpoint_identity.json",
                },
                lease_handoff=lease,
            )
            self.assertFalse(got["real_gpu_ready"])

    def test_budget_bounds_fail_closed(self):
        with self.assertRaises(f.CoverageError):
            f.bounded_plan(base_sha="x", max_candidates=4)
        with self.assertRaises(f.CoverageError):
            f.bounded_plan(base_sha="x", max_prompts=4)
        with self.assertRaises(f.CoverageError):
            f.bounded_plan(base_sha="x", max_requests_per_run=101)
        with self.assertRaises(f.CoverageError):
            f.bounded_plan(base_sha="x", max_restarts=13)

    def test_observation_denominator_validation(self):
        with self.assertRaises(f.CoverageError):
            f.observation_from_json({
                "context_hash": "c",
                "backend": "mlx_metal",
                "policy_hash": "p",
                "weight_epoch": 1,
                "requests_completed": 1,
                "tokens_evaluated": 1,
                "near_tie_events": 1,
                "reference_checks_attempted": 1,
                "effective_attribution_checks": 1,
                "attribution_hits": 2,
            })

    def test_positive_budget_and_inconclusive_are_decision_only(self):
        pre = obs()
        post = obs(
            policy_hash="post",
            weight_epoch=2,
            requests_completed=80,
            effective_attribution_checks=10,
            attribution_hits=0,
            median_margin=None,
        )
        rows = f.decision_quadrants(baseline=pre, passing_post=post)
        self.assertEqual(
            [r["case_kind"] for r in rows],
            ["positive", "budget_exceeded", "inconclusive"],
        )
        self.assertEqual(rows[0]["observer_status"], "CANARY_PASS")
        self.assertEqual(rows[0]["expected_action"], "CANARY_PASS")
        self.assertEqual(rows[1]["observer_status"], "INCONCLUSIVE")
        self.assertEqual(rows[1]["expected_action"], "ROLLBACK_REQUIRED_BUDGET")
        self.assertEqual(rows[2]["observer_status"], "INCONCLUSIVE_NO_OBSERVABILITY")
        self.assertEqual(
            rows[2]["expected_action"],
            "ROLLBACK_REQUIRED_INCONCLUSIVE",
        )
        self.assertTrue(all(r["evidence_level"] == "decision_only" for r in rows))
        self.assertTrue(all(r["reused_hardware_run"] for r in rows))

    def test_regression_requires_explicit_failing_evidence(self):
        pre = obs()
        failed = obs(
            policy_hash="post",
            weight_epoch=2,
            requests_completed=80,
            effective_attribution_checks=10,
            attribution_hits=0,
            target_replay_pass=False,
        )
        row = f.evaluate_case(
            baseline=pre,
            post=failed,
            case_kind="regression",
            evidence_level="simulated",
            reused_hardware_run=False,
        )
        self.assertEqual(row["observer_status"], "REGRESSION_DETECTED")
        self.assertEqual(row["expected_action"], "ROLLBACK_REQUIRED")
        self.assertEqual(row["evidence_level"], "simulated")

    def test_real_status_cannot_be_claimed_by_decision_only_row(self):
        with self.assertRaises(f.CoverageError):
            f.coverage_row(
                role="kv_a_proj_with_mqa",
                layer=11,
                n=7,
                test_kind="g6_positive",
                status="PASS",
                evidence_level="decision_only",
                historical=False,
                new_run=True,
                case_id="c1",
                run_id="r1",
                reused_hardware_run=True,
            )

    def test_matrix_rejects_duplicate_and_fake_real_gpu(self):
        rows = [
            {
                "case_id": "same",
                "evidence_level": "real_gpu",
                "historical": False,
                "run_id": None,
                "reused_hardware_run": False,
                "status": "PASS",
            },
            {
                "case_id": "same",
                "evidence_level": "decision_only",
                "historical": False,
                "run_id": "r2",
                "reused_hardware_run": True,
                "status": "DECISION_ROLLBACK",
            },
        ]
        got = f.validate_matrix(rows)
        self.assertFalse(got["ok"])
        self.assertTrue(any("duplicate case_id" in x for x in got["errors"]))
        self.assertTrue(any("missing run_id" in x for x in got["errors"]))

    def test_cli_plan_writes_json(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "plan.json"
            value = f.bounded_plan(base_sha="abc")
            f.atomic_json(out, value)
            loaded = json.loads(out.read_text())
            self.assertEqual(loaded["schema"], f.SCHEMA)
            self.assertFalse(loaded["real_gpu_ready"])


if __name__ == "__main__":
    unittest.main()
