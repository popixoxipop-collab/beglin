"""BEGLIN Qv2 BF16 source-activation and real pthread CPU correctness tests."""
import json,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"research/oml_acs"))
import q_v2_activation_async_independent_audit as a

class ActivationAsyncNativeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.src=a.verify_source_allocator()
  cls.eval=a.verify_native_eval()
  cls.native=a.verify_actual_pthreads()
 def test_calibration_from_original_bf16_196_source_projections(self):
  self.assertEqual(self.src["private_source_CAL_196_stats_sha256"],"ee7b177b4dde984a5d83e0cdb5983d2ae00c34ee752fef9113a79568a1d12663")
  self.assertTrue(self.src["source_only_exact_in_topk_family"])
  self.assertTrue(self.src["freeze_before_native_eval"])
 def test_eight_QKV_and_four_GateUp_real_CPU_stage_costs(self):
  self.assertEqual(self.src["async_stage_profile_sha256"],"380670d42ccd2f12c4a015600be3d921f672c570a3d3a4a1652c61e217791808")
 def test_source_only_n8_budget_and_kernel_bounds(self):
  for name,item in self.src["role_source_allocator_choices"].items():
   self.assertEqual(item["n8"],28)
   self.assertEqual(item["n4"],168)
   self.assertLessEqual(item["packed_projection_bytes"],830570496)
   self.assertLessEqual(item["serial_predicted_ms"],100)
   if name=="parallel80":self.assertLessEqual(item["only_hypothetical_async_predicted_ms"],80)
 def test_role_masks_change_when_async_latency_bound_tightens(self):
  a0=self.src["role_source_allocator_choices"]["serial100"]
  a1=self.src["role_source_allocator_choices"]["parallel80"]
  self.assertNotEqual(a0["role_counts"],a1["role_counts"])
  self.assertNotEqual(a0["policy_sha256"],a1["policy_sha256"])
 def test_prospective_windows_are_identical_across_independent_runs(self):
  self.assertTrue(self.eval["same_public_inputs_across_five_and_original_global_MILP"])
  self.assertTrue(self.eval["public_eight_prompts_131_positions"])
 def test_quality_proxy_losses_are_not_hidden(self):
  x=self.eval["native_quality_PPL_by_condition"]
  self.assertEqual(x["widened_f32"],348.3639)
  self.assertEqual(x["prior_random_mixed"],552.702)
  self.assertGreater(x["activation_async80_candidate"],x["full_q4"])
  self.assertGreater(x["activation_serial100"],x["activation_async80_candidate"])
  self.assertEqual(self.eval["existing_global_MILP_source_frozen_PPL"],827.4999)
 def test_true_native_QKV_GU_pthreads_and_vocab_parity(self):
  self.assertTrue(self.native["actual_both_join_types_executed"])
  self.assertTrue(self.native["output_logits_identical_to_former_serial"])
  self.assertEqual(self.native["actual_async_QKV_join_count"],3892)
  self.assertEqual(self.native["actual_async_gateup_join_count"],3892)
  self.assertEqual(self.native["serial_new_public_PPL"],self.native["actual_async_new_public_PPL"])
 def test_actual_native_parallel_is_slightly_slower_no_promotion(self):
  self.assertGreater(self.native["full_model_actual_async_ms"],self.native["full_model_actual_serial_ms"])
  self.assertFalse(self.native["actual_async_performance_superiority"])
  self.assertLess(self.native["actual_async_speedup"],1.0)
  self.assertEqual(self.native["serial_new_public_PPL"],827.4999)
 def test_audit_is_private_and_fail_closed_to_posthoc_edits(self):
  self.assertEqual(a.OUTPUT.stat().st_mode&0o777,0o600)
  obj=json.loads(a.OUTPUT.read_text())
  self.assertEqual(obj["scientific_Q_final_verdict"],"NO_VERDICT")
  self.assertTrue(obj["no_blocked_output_sensitivity_retry"])
  with self.assertRaises(FileExistsError):a.main()
if __name__=="__main__":
 unittest.main(verbosity=2)
