import json,math,pathlib,sys,unittest
from collections import Counter
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"research/oml_acs"))
import q_kernel_aware_independent_audit as audit
import q_kernel_aware_allocator as alloc

class KernelAwareAllocatorQATest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.private=json.loads(audit.secure(audit.OUTPUT))
  cls.src=cls.private["source_only_exact_integer_policy"]
  cls.actual=cls.private["native_BEGLIN_Qwen_28_layer_model_measurement"]
 def test_frozen_196_roles_exact_28_promotions(self):
  self.assertEqual(self.src["role_counts"],[6,10,9,0,0,0,3])
  self.assertEqual(sum(self.src["role_counts"]),28)
  self.assertEqual(self.actual["packed_n8_repacked_projection_count"],28)
  self.assertEqual(self.actual["packed_n4_projection_count"],168)
 def test_native_cpu_kernel_cost_and_memory_budget(self):
  self.assertTrue(self.src["actual_kernel_profile_used"])
  self.assertLessEqual(self.src["selected_projection_bytes"],self.src["memory_limit_bytes"])
  self.assertLessEqual(self.src["selected_predicted_single_token_ms"],self.src["kernel_latency_ceiling_ms"])
  self.assertEqual(self.src["selected_projection_bytes"],768442368)
 def test_integer_source_solution_audited_and_not_target_tuned(self):
  self.assertTrue(self.src["exact_source_objective_optimality_recomputed"])
  self.assertTrue(self.src["model_source_prompts_not_used_for_selection"])
  self.assertEqual(self.src["policy_manifest_sha256"],alloc.sha(audit.secure(alloc.TARGET/"kernel_aware_source_only_selection.json")))
 def test_kernel_time_constraint_consequential_at_80ms(self):
  self.assertFalse(self.src["kernel_time_constraint_100ms_changed_memory_only_optimum"])
  self.assertEqual(self.src["kernel_latency_sensitivity_80ms_role_counts"],[7,10,11,0,0,0,0])
  self.assertLessEqual(self.src["kernel_latency_sensitivity_80ms_predicted_ms"],80.0)
 def test_real_model_logit_vs_source_F32_qualifies_negative(self):
  vals=self.actual["logits_and_PPL_by_condition"]
  self.assertGreater(vals["new_kernel_aware"]["KL_widened_F32_to_candidate_nats"],
                     vals["all_q4"]["KL_widened_F32_to_candidate_nats"])
  self.assertEqual(vals["new_kernel_aware"]["top1_id"],9707)
 def test_native_model_PPL_fail_to_beat_prior_random(self):
  vals=self.actual["logits_and_PPL_by_condition"]
  self.assertGreater(vals["new_kernel_aware"]["NLL_public_21_token_windows"],
                     vals["prior_balanced_random"]["NLL_public_21_token_windows"])
  self.assertLess(vals["new_kernel_aware"]["NLL_public_21_token_windows"],
                  vals["all_q4"]["NLL_public_21_token_windows"])
  self.assertLess(self.actual["frozen_actual_best_of_3_single_token_ms"],100.)
 def test_integrity_private_and_source_only_release_guard(self):
  self.assertEqual(audit.OUTPUT.stat().st_mode&0o777,0o600)
  self.assertEqual(self.private["scientific_Q_final_verdict"],"NO_VERDICT")
  self.assertFalse(self.private["no_live_release_or_main_change"] is False)
  self.assertTrue(self.private["no_blocked_output_sensitivity_retry"])
  self.assertFalse(self.private["condition_not_selected_on_native_logits_or_PPL"] is False)
 def test_fail_closed_no_overwrite(self):
  with self.assertRaises(FileExistsError):
   audit.run()
if __name__=="__main__":
 unittest.main(verbosity=2)
