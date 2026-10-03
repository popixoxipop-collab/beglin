#!/usr/bin/env python3
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gpu_autopilot as ga


def h(ch):
    return ch * 64


class SupabaseV3AutopilotTests(unittest.TestCase):
    def context(self):
        return ga.build_context(
            binary_sha256=h("1"),
            checkpoint_sha256=h("2"),
            model_id="deepseek-v2-lite",
            architecture="mla",
            kernel_revision="test",
        )

    def test_build_context_uses_v3_schema(self):
        ctx = self.context()
        self.assertEqual(ctx.schema, "precision-context-v3")
        self.assertEqual(ctx.backend, "mlx_metal")

    def test_explicit_identity_is_preserved_for_v3(self):
        ctx = ga.build_context(
            binary_sha256=h("1"),
            checkpoint_sha256=h("2"),
            tokenizer_sha256=h("3"),
            base_artifact_sha256=h("4"),
            runtime_config_sha256=h("5"),
            device_fingerprint="xox-apple-silicon",
            kernel_revision="rev-1",
        )
        self.assertEqual(ctx.tokenizer_sha256, h("3"))
        self.assertEqual(ctx.base_artifact_sha256, h("4"))
        self.assertEqual(ctx.runtime_config_sha256, h("5"))
        self.assertEqual(ctx.device_fingerprint, "xox-apple-silicon")
        self.assertEqual(ctx.kernel_revision, "rev-1")

    @patch.object(ga.ev3, "configured", return_value=True)
    def test_persistence_requires_explicit_immutable_identity(self, _configured):
        args = SimpleNamespace(
            tokenizer_sha256=None,
            base_artifact_sha256=None,
            runtime_config_sha256=None,
            device_fingerprint=None,
            kernel_revision=None,
        )
        with self.assertRaises(ga.AutopilotError):
            ga._require_v3_identity_for_persistence(args)

    @patch.object(ga.ev3, "upsert_execution_context")
    @patch.object(ga.ev3, "configured", return_value=True)
    def test_context_persistence_verifies_hash(self, _configured, upsert):
        ctx = self.context()
        upsert.return_value = {"context_hash": ctx.context_hash}
        self.assertTrue(ga._persist_context_v3(ctx))
        upsert.assert_called_once_with(ctx)

    @patch.object(ga.time, "time_ns", return_value=201)
    @patch.object(ga.ev3, "insert_validation_run")
    @patch.object(ga.ev3, "configured", return_value=True)
    def test_g6_verdict_persists_observer_evidence(
        self, _configured, insert_validation, _time_ns
    ):
        ctx = self.context()
        args = SimpleNamespace(
            model="deepseek-v2-lite",
            role="shared_up_proj",
            layer=3,
            n=6,
        )
        pre = SimpleNamespace(
            worker_instance_id="pid-10",
            requests_completed=12,
            attribution_hits=12,
        )
        post = SimpleNamespace(
            worker_instance_id="pid-11",
            requests_completed=12,
            attribution_hits=0,
            target_replay_pass=True,
        )
        canary = {
            "status": "CANARY_PASS",
            "decision": "CANARY_PASS_NO_AUTO_EXPANSION",
            "rollback_required": False,
            "auto_expand": False,
            "observer": {
                "reason": "attribution rate decreased",
                "baseline_rate": 1.0,
                "post_rate": 0.0,
            },
        }
        insert_validation.side_effect = lambda row: row
        got = ga._persist_g6_v3(
            args=args,
            context=ctx,
            pre_evidence=pre,
            post_evidence=post,
            pre_ack={"active_policy_hash": h("8")},
            post_ack={"active_policy_hash": h("9"), "weight_epoch": 1},
            canary_result=canary,
        )
        self.assertEqual(got["run_kind"], "observer")
        self.assertTrue(got["pass"])
        self.assertEqual(got["status"], "CANARY_PASS")
        self.assertEqual(got["metrics"]["baseline_worker_instance_id"], "pid-10")
        self.assertEqual(got["metrics"]["post_worker_instance_id"], "pid-11")
        self.assertEqual(got["metrics"]["post_attribution_hits"], 0)

    @patch.object(ga.time, "time_ns", side_effect=[101, 102])
    @patch.object(ga.ev3, "insert_live_preflight")
    @patch.object(ga.ev3, "insert_validation_run")
    @patch.object(ga.ev3, "configured", return_value=True)
    def test_g4_persistence_links_context_runs_and_preflight(
        self, _configured, insert_validation, insert_preflight, _time_ns
    ):
        ctx = self.context()
        args = SimpleNamespace(
            model="deepseek-v2-lite",
            role="shared_up_proj",
            layer=3,
            n=6,
        )
        baseline = {
            "weight_epoch": 0,
            "manifest_sha256": h("3"),
            "worker_log_path": "/tmp/base.log",
            "worker_log_sha256": h("4"),
            "ack_sha256": h("5"),
            "returncode": 0,
        }
        candidate = {
            "weight_epoch": 1,
            "manifest_sha256": h("3"),
            "worker_log_path": "/tmp/candidate.log",
            "worker_log_sha256": h("6"),
            "ack_sha256": h("7"),
            "returncode": 0,
        }
        g4 = {
            "baseline": baseline,
            "candidate": candidate,
            "baseline_policy_hash": h("8"),
            "candidate_policy_hash": h("9"),
            "baseline_epoch": 0,
            "candidate_epoch": 1,
            "baseline_emitted_token": 3268,
            "candidate_emitted_token": 1224,
            "reference_emitted_token": 1224,
            "evidence_bundle": {
                "verdict_payload_sha256": h("a"),
            },
        }
        insert_validation.side_effect = lambda row: row
        insert_preflight.return_value = {"id": 55, "status": "PASS"}

        got = ga._persist_g4_v3(
            args=args,
            context=ctx,
            g4_result=g4,
            event={"req": 0, "pos": 16},
            reference={"emitted_token": 1224},
        )

        self.assertEqual(insert_validation.call_count, 2)
        baseline_row = insert_validation.call_args_list[0].args[0]
        candidate_row = insert_validation.call_args_list[1].args[0]
        preflight_row = insert_preflight.call_args.args[0]

        self.assertEqual(baseline_row["context_hash"], ctx.context_hash)
        self.assertEqual(candidate_row["context_hash"], ctx.context_hash)
        self.assertEqual(candidate_row["n"], 6)
        self.assertEqual(preflight_row["baseline_run_id"], baseline_row["run_id"])
        self.assertEqual(preflight_row["candidate_run_id"], candidate_row["run_id"])
        self.assertEqual(preflight_row["emitted_token"], 1224)
        self.assertEqual(preflight_row["reference_token"], 1224)
        self.assertEqual(got["preflight_id"], 55)


if __name__ == "__main__":
    unittest.main()
