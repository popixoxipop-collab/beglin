import json,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"research"/"oml_acs"))
import qng64_n8_repacked_independent_audit as a

class NativeRepackedQ8Tests(unittest.TestCase):
    def test_private_source_receipts_and_binaries(self):
        parts=a.receipts()
        self.assertEqual(parts["build"]["bf16_source_rev"],a.SNAPSHOT)
        self.assertTrue(parts["build"]["main_untouched"])
        self.assertFalse(parts["build"]["production_touched"])
    def test_196_tensor_full_model_logits_exact_against_prior_neon(self):
        r=a.actual_logits(a.LAB/"n8/results/c_logits_last.f32",
                          a.OLD/"n8/results/c_logits_last.f32")
        self.assertTrue(r["bitwise_equal"])
        self.assertEqual(r["top1"],9707)
    def test_frozen_mixed_random_workload_all_phases(self):
        r=a.validate_mixed("random",a.receipts())
        self.assertTrue(r["exact_same_code_as_old_NEON"])
        self.assertTrue(r["same_PPL_original"])
        self.assertEqual(r["actual_repacked_GEMV_calls"],{"dump":280,"bench":1596,"ppl":644})
    def test_frozen_mixed_proxy_workload_all_phases(self):
        r=a.validate_mixed("proxy",a.receipts())
        self.assertTrue(r["exact_same_code_as_old_NEON"])
        self.assertTrue(r["same_PPL_original"])
        self.assertEqual(r["actual_repacked_GEMV_calls"],{"dump":280,"bench":1596,"ppl":644})
    def test_previous_21_position_ppl_unchanged(self):
        for name,expected in [("random",3411.2283),("proxy",4747.6983)]:
            r=a.validate_mixed(name,a.receipts())
            self.assertEqual(r["original_PPL"],expected)
    def test_four_real_shapes_n8_code_storage_not_increased(self):
        parts=a.receipts()
        seen={tuple(r["shape"]) for r in parts["multi"]["rows"]}
        self.assertEqual(seen,{(256,1536),(1536,1536),(8960,1536),(1536,8960)})
        for row in parts["multi"]["rows"]:
            self.assertEqual(row["native_bitplane_weight_bytes"],row["repacked_code_weight_bytes"])
            self.assertLess(row["max_abs_error"],.0005)
            self.assertGreater(row["speedup"],8)
    def test_m3_batched_and_other_n_scalar_fallback(self):
        result=a.receipts()["batch"]
        self.assertEqual(result["batch_rows"],3)
        self.assertLess(result["n8_original_scalar_vs_repacked_maxabs"],.0005)
        self.assertEqual(set(result["other_bitwidth_original_fallback"]),{"3","5","6","7"})
        for item in result["other_bitwidth_original_fallback"].values():
            self.assertTrue(item["bitwise_output_identical"])
    def test_independent_evidence_frozen_and_not_final_Q(self):
        self.assertEqual(a.OUT.stat().st_mode&0o777,0o600)
        r=json.loads(a.OUT.read_text())
        self.assertFalse(r["real_task_accuracy_evaluated"])
        self.assertTrue(r["V6_24_holdout_still_sealed"])
        self.assertTrue(r["kernel_format_discriminant_not_integrated_into_upstream_runtime"])
        with self.assertRaises(FileExistsError):a.run()
if __name__=="__main__":unittest.main(verbosity=2)
