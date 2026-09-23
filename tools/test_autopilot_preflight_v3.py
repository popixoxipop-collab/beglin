#!/usr/bin/env python3
import unittest

import precision_context as pc
from autopilot_preflight_v3 import PreflightV3Error, run_preflight
from backend_adapters import MockBackendAdapter


def h(ch):
    return ch * 64


def context():
    return pc.ExecutionContext(
        schema="precision-context-v3",
        model_id="deepseek-v2-lite",
        architecture="deepseek_v2_mla",
        checkpoint_sha256=h("1"),
        tokenizer_sha256=h("2"),
        base_artifact_sha256=h("3"),
        backend="mlx_metal",
        device_fingerprint="apple-m4-test",
        binary_sha256=h("4"),
        build_manifest_sha256=h("5"),
        kernel_revision="mlx-test",
        execution_mode="isolated-preflight",
        runtime_config_sha256=h("6"),
        quant_format="qng64_g64_ef_v1",
        group_size=64,
        correction_mode="off",
    )


BASE = [{"role": "shared_up_proj", "layer": 3, "n": 5}]
CANDIDATE = BASE + [{"role": "shared_down_proj", "layer": 4, "n": 6}]
EVENT = {"req": 3, "pos": 16, "orig_token": 3268, "corrected_token": 1224}
REFERENCE = {"run_id": "ref-1", "context_hash": "oracle", "emitted_token": 1224}


class PreflightV3Tests(unittest.TestCase):
    def setUp(self):
        self.ctx = context()
        self.adapter = MockBackendAdapter(
            backend="mlx_metal",
            policy=BASE,
            epoch=4,
            context=self.ctx,
        )
        self.adapter.set_validation_result("baseline", {
            "emitted_token": 3268,
            "correction_mode": "off",
            "finite_logits": True,
        })
        self.adapter.set_validation_result("candidate", {
            "emitted_token": 1224,
            "correction_mode": "off",
            "finite_logits": True,
        })

    def run(self, **kwargs):
        return run_preflight(
            self.adapter,
            expected_context_hash=kwargs.get("context_hash", self.ctx.context_hash),
            expected_epoch=kwargs.get("epoch", 4),
            baseline_policy=BASE,
            candidate_policy=CANDIDATE,
            event=EVENT,
            reference=kwargs.get("reference", REFERENCE),
        )

    def test_pass_restores_isolated_worker_baseline(self):
        got = self.run()
        self.assertEqual(got["status"], "passed")
        self.assertEqual(got["candidate"]["emitted_token"], 1224)
        state = self.adapter.query_applied_state()
        self.assertEqual(state.policy_hash, pc.policy_hash(BASE))
        self.assertFalse(state.admission_paused)

    def test_candidate_correction_on_is_rejected_and_restored(self):
        self.adapter.set_validation_result("candidate", {
            "emitted_token": 1224,
            "correction_mode": "on",
            "finite_logits": True,
        })
        with self.assertRaisesRegex(PreflightV3Error, "correction must be OFF"):
            self.run()
        state = self.adapter.query_applied_state()
        self.assertEqual(state.policy_hash, pc.policy_hash(BASE))
        self.assertFalse(state.admission_paused)

    def test_candidate_wrong_token_is_rejected_and_restored(self):
        self.adapter.set_validation_result("candidate", {
            "emitted_token": 999,
            "correction_mode": "off",
            "finite_logits": True,
        })
        with self.assertRaisesRegex(PreflightV3Error, "reference token"):
            self.run()
        self.assertEqual(
            self.adapter.query_applied_state().policy_hash,
            pc.policy_hash(BASE),
        )

    def test_candidate_wrong_applied_policy_is_rejected(self):
        self.adapter.set_validation_result("candidate", {
            "emitted_token": 1224,
            "correction_mode": "off",
            "finite_logits": True,
            "applied_policy_hash": "wrong",
        })
        with self.assertRaisesRegex(PreflightV3Error, "applied policy hash mismatch"):
            self.run()

    def test_context_mismatch_is_rejected_before_mutation(self):
        with self.assertRaisesRegex(RuntimeError, "execution context mismatch"):
            self.run(context_hash=h("f"))
        state = self.adapter.query_applied_state()
        self.assertEqual(state.epoch, 4)
        self.assertEqual(state.policy_hash, pc.policy_hash(BASE))

    def test_stale_epoch_is_rejected_before_mutation(self):
        with self.assertRaisesRegex(PreflightV3Error, "stale worker epoch"):
            self.run(epoch=3)
        self.assertEqual(self.adapter.query_applied_state().epoch, 4)

    def test_reference_must_match_recorded_corrected_token(self):
        bad = {"run_id": "ref-x", "context_hash": "oracle", "emitted_token": 777}
        with self.assertRaisesRegex(PreflightV3Error, "reference token"):
            self.run(reference=bad)


if __name__ == "__main__":
    unittest.main()
