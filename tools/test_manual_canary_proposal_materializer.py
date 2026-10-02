#!/usr/bin/env python3
import unittest

import manual_canary_contract as mc
import manual_canary_proposal_materializer as mat


def h(ch):
    return ch * 64


TEMPLATE = {
    "mode": "dry_run",
    "production_write_allowed": False,
    "proposal_id": "p-materialized",
    "proposer": "planner-agent",
    "environment_id": "xox-vdsp-gpu-precision",
    "model_revision": "deepseek-v2-lite",
    "backend": "mlx_metal",
    "architecture": "mla",
    "source_commit": "330954b27f146b8a17db2cb353c3e620968bad5e",
    "binary_sha256": "6e6258d6d4e402033c303c8c4b622db5a088354a0de79e687eb0fad4190c4392",
    "checkpoint_sha256": "1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898",
    "baseline_policy_hash": None,
    "candidate_policy_hash": None,
    "single_target": {"role": "shared_up_proj", "layer": 3, "before_n": None, "after_n": 6},
    "evidence_refs": [
        {"kind": "G4_A_B_R", "run_id": "f-real", "sha256": h("a")},
        {"kind": "G6_RESTART_CANARY", "run_id": "f-real", "sha256": h("b")},
    ],
    "budget": {
        "max_requests": None,
        "max_tokens": None,
        "max_duration_ms": None,
        "max_memory_bytes": None,
    },
    "expected_epoch": None,
    "restart_instance_id": None,
    "kill_switch_scope": "single-target",
    "rollback_plan": "restore exact runtime baseline preimage",
}


class BudgetTests(unittest.TestCase):
    def test_budget_uses_observed_metrics_with_headroom(self):
        got = mat.derive_budget(
            {"requests": 12, "tokens": 120, "duration_ms": 4000, "memory_bytes": 8_000_000_000}
        )
        self.assertEqual(got["max_requests"], 18)
        self.assertEqual(got["max_tokens"], 256)
        self.assertEqual(got["max_duration_ms"], 30000)
        self.assertEqual(got["max_memory_bytes"], 10_000_000_000)

    def test_budget_respects_physical_memory_cap(self):
        got = mat.derive_budget(
            {"requests": 12, "tokens": 120, "duration_ms": 4000, "memory_bytes": 9_000},
            physical_memory_bytes=10_500,
        )
        self.assertEqual(got["max_memory_bytes"], 9450)

    def test_budget_fails_if_observed_memory_reaches_physical(self):
        with self.assertRaises(mat.ProposalMaterializerError):
            mat.derive_budget(
                {"requests": 1, "tokens": 1, "duration_ms": 1, "memory_bytes": 100},
                physical_memory_bytes=100,
            )


class MaterializerTests(unittest.TestCase):
    def test_absent_baseline_materializes_null_before_n(self):
        runtime = {
            "active_policy": [],
            "active_policy_hash": mc.sha256_json([]),
            "weight_epoch": 0,
            "ack_sha256": h("c"),
            "worker_instance_id": "pid-123",
        }
        candidate = [{"role": "shared_up_proj", "layer": 3, "n": 6}]
        got = mat.materialize_proposal(
            template=TEMPLATE,
            runtime_preimage=runtime,
            candidate_policy=candidate,
            observed_metrics={
                "requests": 12, "tokens": 120, "duration_ms": 4000, "memory_bytes": 8_000_000_000
            },
        )
        self.assertEqual(got["status"], "READY_FOR_HUMAN_SIGNATURE")
        self.assertIsNone(got["proposal"]["single_target"]["before_n"])
        self.assertEqual(got["proposal"]["expected_epoch"], 0)
        self.assertEqual(got["proposal"]["restart_instance_id"], "pid-123")
        self.assertEqual(len(got["proposal_digest"]), 64)

    def test_numeric_baseline_is_preserved(self):
        baseline = [{"role": "shared_up_proj", "layer": 3, "n": 5}]
        runtime = {
            "active_policy": baseline,
            "active_policy_hash": mc.sha256_json(baseline),
            "weight_epoch": 7,
            "ack_sha256": h("c"),
            "worker_instance_id": "worker-7",
        }
        candidate = [{"role": "shared_up_proj", "layer": 3, "n": 6}]
        got = mat.materialize_proposal(
            template=TEMPLATE,
            runtime_preimage=runtime,
            candidate_policy=candidate,
            observed_metrics={
                "requests": 12, "tokens": 120, "duration_ms": 4000, "memory_bytes": 8_000_000_000
            },
        )
        self.assertEqual(got["proposal"]["single_target"]["before_n"], 5)

    def test_runtime_hash_mismatch_is_rejected(self):
        runtime = {
            "active_policy": [],
            "active_policy_hash": h("f"),
            "weight_epoch": 0,
            "ack_sha256": h("c"),
            "worker_instance_id": "pid-123",
        }
        with self.assertRaises(mat.ProposalMaterializerError):
            mat.materialize_proposal(
                template=TEMPLATE,
                runtime_preimage=runtime,
                candidate_policy=[{"role": "shared_up_proj", "layer": 3, "n": 6}],
                observed_metrics={
                    "requests": 12, "tokens": 120, "duration_ms": 4000, "memory_bytes": 8_000_000_000
                },
            )

    def test_multitarget_candidate_is_rejected(self):
        runtime = {
            "active_policy": [],
            "active_policy_hash": mc.sha256_json([]),
            "weight_epoch": 0,
            "ack_sha256": h("c"),
            "worker_instance_id": "pid-123",
        }
        candidate = [
            {"role": "shared_up_proj", "layer": 3, "n": 6},
            {"role": "shared_down_proj", "layer": 4, "n": 6},
        ]
        with self.assertRaises(mat.ProposalMaterializerError):
            mat.materialize_proposal(
                template=TEMPLATE,
                runtime_preimage=runtime,
                candidate_policy=candidate,
                observed_metrics={
                    "requests": 12, "tokens": 120, "duration_ms": 4000, "memory_bytes": 8_000_000_000
                },
            )


if __name__ == "__main__":
    unittest.main()
