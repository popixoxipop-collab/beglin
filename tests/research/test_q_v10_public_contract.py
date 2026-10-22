"""Portable (no model weights) Qv10 original source, policy & RACE contracts."""
import hashlib,json,pathlib,math,unittest
R=pathlib.Path(__file__).resolve().parents[2]
D=R/"docs/research";C=R/"configs/research"
P=C/"BEGLIN_QV10_TOPK_OTHER_TRUST_PRELOCK_20261023.json"
FREEZE=D/"BEGLIN_QV10_RACE_INPUT_PREMODEL_FREEZE_20261023.json"
SRC=D/"BEGLIN_QV10_TOP2OTHER_TRAIN24_PUBLIC_20261023.json"
POL=D/"BEGLIN_QV10_TOPK_OTHER_TRUST_SOURCE_POLICY_20261023.json"
MASK=C/"BEGLIN_QV10_TOP2OTHER_TRUST_28N8_SOURCE_POLICY_20261023.txt"
DIAG=D/"BEGLIN_QV10_TOP2OTHER_DIAGNOSTIC24_PUBLIC_20261023.json"
SOURCED=D/"BEGLIN_QV10_48SOURCE_DIAGNOSTIC_INDEPENDENT_PUBLIC_20261023.json"
NATIVE=D/"BEGLIN_QV10_RACE48_NATIVE_3ARM_INDEPENDENT_PUBLIC_20261023.json"
BF16=D/"BEGLIN_QV10_ORIGINAL_BF16_RACE48_REFERENCE_PUBLIC_20261023.json"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Qv10PublicContract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.pre=json.loads(P.read_text())
  cls.freeze=json.loads(FREEZE.read_text())
  cls.src=json.loads(SRC.read_text())
  cls.policy=json.loads(POL.read_text())
 def test_original_qwen_pinned_prelock(self):
  self.assertEqual(sha(P),"4ef4d31a4b9d0be2e015ffaac84a8ee86feb86047a8a8aae4a50318144d5d207")
  self.assertEqual(self.pre["status"],"FROZEN_PLAN_BEFORE_FULLVOCAB_TOPK_GRADIENT_OR_RACE_VALIDATION")
  self.assertEqual(self.pre["model_revision"],"989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
 def test_race_48_distinct_original_passages(self):
  self.assertEqual(self.freeze["status"],"ALL_3_COHORTS_SHA_LOCKED_BEFORE_TOPK_PLUS_OTHER_MODEL_EXECUTION")
  self.assertEqual(self.freeze["eval_48_distinct_article_count"],48)
  self.assertEqual(self.freeze["original_48_RACE_revision"],"2fec9fd81f1dc971569a9b729c43f2f0e6436637")
  self.assertTrue(self.freeze["test_model_performance_unseen"])
 def test_prior_train_diag_nonpristine_disclosed(self):
  self.assertTrue(self.freeze["source_train_24_and_diag_24_have_prior_Qv9_analyses"])
  self.assertTrue(self.policy["normal_source_not_independent_across_Qv9_prior_work"])
 def test_source_original_signed_actual_C(self):
  self.assertEqual(self.src["status"],"SOURCE_ONLY_ACTUAL_ORIGINAL_BF16_24x196x6_TOP2_OTHER_SIGNED_DONE")
  self.assertEqual(self.src["num_cases"],24)
  self.assertEqual(self.src["num_projected_tensors"],196)
  self.assertEqual(self.src["num_7group_nonanchor"],6)
  self.assertTrue(self.src["numeric_group_probability_mass_original_fullvocab_exact"])
 def test_fullvocab_top2_other_probability_mass_reasonable(self):
  vals=[self.src["mean_mass_letters"],self.src["mean_mass_top2"],self.src["mean_mass_OTHER"]]
  self.assertAlmostEqual(sum(vals),1,places=4)
  self.assertGreaterEqual(min(vals),0)
  self.assertGreater(self.src["mean_mass_letters"],.95)
 def test_topk_OTHER_not_full_vocab_Hessian(self):
  self.assertTrue(self.src["not_fullvocab_Hessian_due_to_collapsed_OTHER"])
  self.assertTrue(self.policy["full_vocabulary_Hessian_not_exact_due_to_OTHER_collapsing"])
  self.assertTrue(self.policy["grouped_exact_KL_is_only_of_linearly_predicted_logits"])
 def test_policy_28_unique_original_tensors(self):
  role={"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
  rows=[tuple(l.split()) for l in MASK.read_text().splitlines()]
  self.assertEqual(len(rows),28)
  self.assertEqual(len(set(rows)),28)
  self.assertTrue(all(r in role and t.isdigit() and 0<=int(t)<28 for r,t in rows))
  self.assertEqual(self.policy["mask_sha256"],sha(MASK))
 def test_trust_and_ridge_precommitted(self):
  self.assertEqual(self.policy["fixed_trust_relative_logit_threshold"],2)
  self.assertEqual(self.policy["fixed_trust_regularization_weight"],.05)
  self.assertEqual(self.policy["fixed_isolated_per_tensor_grouped_GGN_weight"],.1)
  self.assertTrue(self.policy["source_fold_worst_even_odd_12_optimization"])
 def test_cost_constraints_and_no_lag_claim(self):
  k=self.policy["original_Apple_NEON_kernel_resources"]
  self.assertEqual(self.policy["n8_count"],28)
  self.assertEqual(self.policy["q4_count"],168)
  self.assertLessEqual(k["memory_bytes"],830570496)
  self.assertLessEqual(k["serial_estimate_ms"],100)
  self.assertLessEqual(k["isolated_async_stage_ms"],50.00001)
 def test_source_objective_not_native_NLL(self):
  self.assertLess(self.policy["selected_source_train_objective"]["objective"],
                  self.policy["prior_mask_source_only_objectives"]["all_Q4"]["objective"])
  self.assertTrue(self.policy["source_only_single_swap_local_no_global_optimum_proof"])
 def test_extra_diagnostic_original_BF16_if_present(self):
  if not DIAG.exists():self.skipTest("separate original BF16 source diagnostic not finished")
  x=json.loads(DIAG.read_text())
  self.assertEqual(x["stage"],"diagnostic")
  self.assertTrue(x["not_fullvocab_Hessian_due_to_collapsed_OTHER"])
 def test_independent_source_audit_if_present(self):
  if not SOURCED.exists():self.skipTest("independent 7-group source diagnostic under execution")
  x=json.loads(SOURCED.read_text())
  self.assertEqual(x["status"],"PASS_SOURCE_24_AND_DIAG_24_BF16_7GROUP_SIGNED_NATIVE_RESIDUAL_RECALC")
  self.assertTrue(x["no_RACE_eval_outputs_used"])
  self.assertFalse(x["no_live_promotion_allowed"]==False)
 def test_original_BF16_RACE_reference_if_present(self):
  if not BF16.exists():self.skipTest("external original BF16 arithmetic reference unavailable")
  x=json.loads(BF16.read_text())
  self.assertEqual(x["total"],48)
  self.assertEqual(x["source_RACE48_SHA256"],self.freeze["RACE48_private_SHA256"])
 def test_real_three_arm_audit_if_present(self):
  if not NATIVE.exists():self.skipTest("three-arm RACE48 native source benchmark not yet complete")
  x=json.loads(NATIVE.read_text())
  self.assertEqual(x["status"],"PASS_144_REAL_ORIGINAL_BEGLIN_C_RACE_NEXTTOKEN_LOGS_AND_BF16_SOURCE_AUDIT")
  self.assertEqual(set(x["native_summary"]),{"q4","fisher64","qv10_top2other_trust"})
  self.assertTrue(x["native_generalization_and_speed_superiority_unproved"])
 def test_autopromotion_off(self):
  self.assertFalse(self.pre["no_automatic_live_or_GPU_promotion"]==False)
  self.assertTrue(self.policy["no_automatic_precision_promotion"])
  self.assertTrue(self.policy["new_RACE48_logits_and_labels_not_used_to_tune"])
if __name__=="__main__":unittest.main(verbosity=2)
