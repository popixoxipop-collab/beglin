"""Read-only BEGLIN Qv2 actual native async ON/OFF fail-closed proof tests."""
import json,math,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"research/oml_acs"))
import q_v2_actual_async_eligibility as gate

class Qv2ActualAsyncGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=gate.audit()
    def test_independent_original_binary_mask_and_eval_sha(self):
        self.assertEqual(cls_sha:=self.report["native_binary_sha256"],gate.EXPECTED_BINARY)
        self.assertEqual(self.report["selected_role_mask_sha256"],gate.EXPECTED_POLICY)
        self.assertTrue(self.report["complete_first_prompt_full_logits_byte_equal"])
    def test_exact_native_ABBA_clock_interleaving(self):
        self.assertEqual(sorted(self.report["timed_native_OFF_single_token_ms"]),[90.61,92.04,98.21])
        self.assertEqual(sorted(self.report["timed_native_ON_single_token_ms"]),[92.66,93.98,94.33])
        self.assertGreater(self.report["measured_overall_ratio_OFF_div_ON"],.98)
        self.assertLess(self.report["measured_overall_ratio_OFF_div_ON"],1.02)
    def test_real_pthread_qkv_and_gateup_joined(self):
        self.assertTrue(self.report["observed_full_model_native_Qwen_CPU_async"])
        self.assertTrue(all(x["qkv_joined"]>0 and x["gateup_joined"]>0 for x in gate.inputs()["ABBA"]["individual_runs"] if x["mode"]=="on"))
        self.assertTrue(self.report["same_131_token_PPL"])
    def test_source_public_PPL_full_native_on_off_match(self):
        self.assertEqual(self.report["fixed_public_eval_token_positions"],131)
        self.assertEqual(self.report["fixed_public_PPL_ON_OFF"],{"on":827.4999,"off":827.4999})
    def test_precision_policy_quality_below_controls(self):
        self.assertGreater(self.report["fixed_public_PPL_ON_OFF"]["on"],
                           self.report["native_development_all_Q4_PPL"])
        self.assertGreater(self.report["fixed_public_PPL_ON_OFF"]["on"],
                           self.report["native_development_seed73_random_PPL"])
        self.assertFalse(self.report["quantized_source_quality_gate_pass"])
    def test_not_statistically_or_engineering_significant_async_speed(self):
        self.assertFalse(self.report["async_speed_gate_pass"])
        self.assertLess(self.report["measured_overall_ratio_OFF_div_ON"],
                        self.report["conservative_engineering_async_min_speedup_ratio"])
    def test_fail_closed_no_production_or_live_toggle(self):
        self.assertEqual(self.report["recommended_default_research_env_BEGLIN_QV2_NATIVE_ASYNC"],"0")
        self.assertFalse(self.report["approval_to_modify_production_or_main"])
        self.assertFalse(self.report["eligible_for_auto_precision_policy_promotion"])
        self.assertEqual(self.report["quality_hypothesis_verdict"],"NO_VERDICT")
    def test_source_evidence_private_and_immutable(self):
        self.assertEqual(gate.OUT.stat().st_mode&0o777,0o600)
        with self.assertRaises(FileExistsError):
            gate.run()
if __name__=="__main__":
    unittest.main(verbosity=2)
