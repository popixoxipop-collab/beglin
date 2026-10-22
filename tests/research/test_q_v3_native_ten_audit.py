import json,hashlib,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"research/oml_acs"))
import q_v3_native_ten_independent_audit as auditor

class QwenV3FrozenTenPublicNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,cls.original,cls.outcomes=auditor.check_actual()
    def test_v3_source_policy_frozen_before_new_evaluation(self):
        source=auditor.verify_source()
        self.assertEqual(source["prompt_count"],10)
        self.assertEqual(source["teacher_forced_prediction_positions_expected_sum_input_lengths_minus_10"],187)
        self.assertEqual(self.manifest["source_V3_28_n8_mask_SHA256"],auditor.MASK_SHA)
    def test_exact_five_source_frozen_native_modes(self):
        self.assertEqual(set(self.original["all_native_mode_results"]),set(auditor.MODES))
        self.assertEqual(len(self.outcomes),5)
    def test_full_native_28layer_projection_counts(self):
        for label in auditor.MODES:
            with self.subTest(label=label):
                items=self.original["all_native_mode_results"][label]
                for phase in ("dump","ppl","bench"):
                    self.assertEqual(items[phase]["model_total_decoder_layers"],28)
                    self.assertEqual(items[phase]["projection_counts"],auditor.EXPECTED_COUNTS[label])
    def test_each_native_policy_uses_all_10_same_public_inputs(self):
        for label in auditor.MODES:
            self.assertEqual(self.original["all_native_mode_results"][label]["ppl"]["windows"],10)
            self.assertEqual(self.original["all_native_mode_results"][label]["ppl"]["teacher_forced_prediction_positions"],187)
    def test_no_nonfinite_native_logits_and_true_numeric_KL(self):
        self.assertEqual(self.outcomes["f32"]["KL_F32_to_candidate"],0.0)
        for label,r in self.outcomes.items():
            with self.subTest(label=label):
                self.assertGreater(r["ppl"],0)
                self.assertGreater(r["best_of_three_ms"],0)
                self.assertGreater(r["peak_RSS_bytes"],0)
                self.assertGreaterEqual(r["KL_F32_to_candidate"],-1e-10)
    def test_same_native_model_binary_for_three_mixed_policies(self):
        files=self.manifest["model_workloads"]
        a=files["role_random"]["binary_sha256"]
        self.assertEqual(a,files["v2_diag"]["binary_sha256"])
        self.assertEqual(a,files["v3_cov"]["binary_sha256"])
    def test_no_actual_parallel_promotion_or_final_Q(self):
        self.assertFalse(self.original["real_fullmodel_async_scheduling_used"])
        self.assertTrue(self.original["previous_v2_actual_parallel_failclosed_preserved"])
        self.assertEqual(self.original["scientific_Q_verdict"],"NO_VERDICT")
        self.assertFalse(self.original["production_or_main_modified"])
    def test_private_receipts_immutable(self):
        self.assertEqual(auditor.ORIGINAL.stat().st_mode&0o777,0o600)
        self.assertEqual(auditor.OUTPUT.stat().st_mode&0o777,0o600)
        self.assertEqual(json.loads(auditor.OUTPUT.read_text())["scientific_Q_conclusion"],"NO_VERDICT")
        with self.assertRaises(FileExistsError):
            auditor.run()
if __name__=="__main__":
    unittest.main(verbosity=2)
