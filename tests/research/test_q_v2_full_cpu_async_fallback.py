#!/usr/bin/env python3
"""Independent non-promotable BEGLIN Qv2 full decoder CPU async audit tests.

Six original and five selective stage native receipts, three same-executable
timing controls, original 131 public-prompt positions. Does not touch a
running service or model, change original test files, or run new inference.
"""
from __future__ import annotations
import json
import pathlib
import subprocess
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v2_full_cpu_async_fallback_audit as audit


class QV2ActualFullCPUPthreadQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results=audit.evaluate()

    def test_native_origins_and_exact_mask(self):
        old,hybrid,quality=audit.verify_original_source_and_mask()
        self.assertEqual(old["concurrency_worker_count"],3)
        self.assertEqual(old["selected_role_mask_SHA256"],audit.ORIGINAL_MASK_SHA)
        self.assertEqual(hybrid["role_policy_SHA256"],audit.ORIGINAL_MASK_SHA)
        self.assertEqual(quality["selected_tensors"]["bits8"],28)
        self.assertEqual(quality["selected_tensors"]["bits4"],168)

    def test_four_full_model_modes_are_bitwise_equal_to_original_logits(self):
        outputs=audit.compare_logit_files()
        self.assertEqual(set(outputs),{"serial","qkv_and_ffn_async","qkv_only","ffn_only"})
        for case,data in outputs.items():
            with self.subTest(case=case):
                self.assertTrue(data["bitwise_equal_to_original"])
                self.assertEqual(data["top1_id"],9707)
                self.assertEqual(data["logit_sha256"],audit.PRIOR_LLM_OUTPUT_SHA)

    def test_actual_worker_barriers_are_real(self):
        evidence=audit.audit_receipts()
        self.assertEqual(evidence["real_QKV_worker_joins"],{"dump":280,"bench":1596,"ppl":3892})
        self.assertEqual(evidence["real_FFN_only_worker_joins"],{"dump":280,"bench":1596})
        self.assertEqual(evidence["PPL_131_serial_and_both_parallel"],807.363)

    def test_frozen_public_input_same_in_all_modes(self):
        a,hybrid,_=audit.verify_original_source_and_mask()
        original=audit.private_bytes(audit.ORIGINAL/"v2_async_activation/eval/ppl_work/w0.i32")
        for case in ("serial","async"):
            self.assertEqual(original,audit.private_bytes(audit.ACTUAL/case/"eval/ppl_work/w0.i32"))
        for case in ("qkv_only","ffn_only","serial_control","qkv_repeat","both_control"):
            self.assertEqual(original,audit.private_bytes(audit.HYBRID/case/"eval/ppl_work/w0.i32"))

    def test_one_off_short_qkv_gain_cannot_count_as_reproducible(self):
        evidence=audit.audit_receipts()
        first=evidence["actual_initial_native_bench_best_of_three_ms"]
        repeat=evidence["same_binary_repeat_bench_best_of_three_ms"]
        self.assertLess(first["qkv_only"],first["serial"])
        self.assertGreater(repeat["qkv_repeat"],repeat["serial_control"])
        self.assertGreater(repeat["both_control"],repeat["serial_control"])

    def test_ppl_unchanged_but_quality_failed(self):
        result=self.results
        self.assertEqual(result["PPL_131_evaluation_dev_positions"],807.363)
        self.assertLess(result["PPL_131_dev_allQ4_reference"],807.363)
        self.assertLess(result["PPL_131_dev_matched_random_reference"],807.363)
        self.assertFalse(result["quality_gain_established"])
        self.assertFalse(result["whole_decoder_async_speed_gain_established"])

    def test_serial_default_and_live_guards(self):
        result=self.results
        self.assertEqual(result["status"],"NOT_READY_DEFAULT_SERIAL")
        self.assertEqual(result["default_runtime_decision"],"SERIAL")
        self.assertFalse(result["production_changed"])
        self.assertFalse(result["main_changed"])
        self.assertFalse(result["tailnet_immutable_release_changed"])
        self.assertFalse(result["prior_security_blocked_per_tensor_causal_logits_intervention_retried"])

    def test_audit_receipt_immutable_and_private(self):
        self.assertEqual(audit.AUDIT_OUT.stat().st_mode&0o777,0o600)
        receipt=json.loads(audit.private_bytes(audit.AUDIT_OUT))
        self.assertEqual(receipt,self.results)
        self.assertEqual(len(receipt["promotion_blocking_reasons"]),8)

    def test_require_ready_exits_two(self):
        cmd=[sys.executable,str(ROOT/"research/oml_acs/q_v2_full_cpu_async_fallback_audit.py"),"--verify-only","--require-ready"]
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=25)
        self.assertEqual(result.returncode,2,result.stderr)
        self.assertIn("NOT_READY_DEFAULT_SERIAL",result.stdout)


if __name__=="__main__":
    unittest.main(verbosity=2)
