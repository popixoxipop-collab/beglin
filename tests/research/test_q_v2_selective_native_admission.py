#!/usr/bin/env python3
"""Fail-closed source/receipt tests for real BEGLIN Qv2 selective pthread."""
import pathlib,sys,unittest
from unittest.mock import patch

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"research/oml_acs"))
import q_v2_selective_native_admission as gate

class SelectiveNativeAdmissionQATest(unittest.TestCase):
    def test_actual_native_two_sweep_gate_keeps_async_off(self):
        r=gate.decide()
        self.assertEqual(r["status"],"PASS_RESEARCH_SAFETY_OFF")
        self.assertFalse(r["research_candidate_eligible_for_additional_fresh_stress"])
        self.assertEqual(r["default_BEGLIN_QV2_NATIVE_ASYNC"],"0")
        self.assertEqual(r["default_BEGLIN_QV2_NATIVE_STAGE_POLICY"],"serial")
    def test_source_hash_drift_fails_closed(self):
        with patch.object(gate,"AUDIT_SHA","0"*64):
            with self.assertRaises(RuntimeError):gate.decide()
    def test_headroom_receipt_drift_fails_closed(self):
        with patch.object(gate,"PROFILE_SHA","f"*64):
            with self.assertRaises(RuntimeError):gate.decide()
    def test_first_vocab_and_131_position_ppl_parity_preserved(self):
        r=gate.decide()
        self.assertTrue(r["exact_logits_and_131_token_PPL_parity"])
        self.assertAlmostEqual(r["native_2_sweep_speed_ratio"],1.00240397749,places=7)
    def test_no_implicit_prod_toggle_and_no_overwrite(self):
        r=gate.decide()
        self.assertFalse(r["available_auto_runtime_promotion"])
        self.assertTrue(r["main_and_immutable_release_untouched"])
        self.assertTrue(r["old_V6_24_records_sealed"])
        self.assertEqual(gate.OUT.stat().st_mode&0o777,0o600)
        with self.assertRaises(FileExistsError):gate.main()
if __name__=="__main__":
    unittest.main(verbosity=2)
