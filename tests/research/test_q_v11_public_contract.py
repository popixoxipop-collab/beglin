"""No native model download: audit Qv11 prereg, source high/low, isolated-C contract."""
import json,hashlib,pathlib,unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
C=ROOT/"configs/research";D=ROOT/"docs/research"
PRE=C/"BEGLIN_QV11_REAL_ONE_N8_INTERVENTION_PRELOCK_20261023.json"
FROZEN=D/"BEGLIN_QV11_RACE_TEST16_PREMODEL_INPUTS_20261023.json"
BUILD=D/"BEGLIN_QV11_ORIGINAL_NATIVE_1N8_BUILD_PUBLIC_20261023.json"
Q6=D/"BEGLIN_QV6_FISHER_SCALING_196_SOURCE_AGGREGATE_20261022.json"
BF16=D/"BEGLIN_QV11_ORIGINAL_BF16_RACE_TEST16_PUBLIC_20261023.json"
FINAL=D/"BEGLIN_QV11_REAL_ONE_TENSOR_NATIVE_CAUSAL_AUDIT_PUBLIC_20261023.json"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class OriginalNativeOneN8Contracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.pre=json.loads(PRE.read_text())
  cls.frozen=json.loads(FROZEN.read_text())
  cls.build=json.loads(BUILD.read_text())
 def test_preregistered_source_before_any_native_trial(self):
  self.assertEqual(sha(PRE),"2cd480dc4c72b5383014a1325d350e51a0b47b1335ec0013a6d069773b546a39")
  self.assertEqual(self.pre["status"],"PRELOCK_BEFORE_ACTUAL_1N8_RACE_NATIVE_INTERVENTION")
  self.assertEqual(self.pre["model_HF_revision"],"989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
 def test_original_test16_independent_articles_frozen(self):
  self.assertEqual(self.frozen["status"],"FROZEN_SOURCE_SIGNED_SCORE_6_SINGLETON_C_PROBES_AND_RACE_TEST16")
  self.assertEqual(self.frozen["RACE_middletest_question_count"],16)
  self.assertEqual(self.frozen["prompt_min_max_token_length"][1]<=208,True)
  self.assertTrue(self.frozen["no_native_output_seen"])
 def test_original_native_singleton_not_full_model_new_training(self):
  self.assertTrue(self.build["no_original_BF16_checkpoint_changed"])
  self.assertTrue(self.build["no_original_quant_Kernel_source_changed"])
  self.assertEqual(self.build["original_Qwen_layers"],28)
  self.assertEqual(self.build["exact_projection_role_count"],196)
  self.assertTrue(self.build["production_untouched"])
 def test_c_base_vs_patched_binaries_distinct_and_pinned(self):
  self.assertEqual(self.build["original_all_Q4_base_binary_sha256"],"408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c")
  self.assertEqual(self.build["patched_native_one_n8_binary_sha256"],"a054e68789fd6bc20a4bd815da5108b63361b14227b1fc4ab9e68a3b7b9a2711")
  self.assertNotEqual(self.build["patched_native_one_n8_binary_sha256"],self.build["original_all_Q4_base_binary_sha256"])
  self.assertEqual(self.build["diagnostic_log_text_updates"],2)
 def test_source_128_score_extrema_fixed_before_evaluation(self):
  obj=json.loads(Q6.read_text())["quality_upgrade_by_role_layer_per_size"]["128"]
  for role,x in self.pre["role_extremes"].items():
   vals=obj[role]
   self.assertEqual(len(vals),28)
   self.assertEqual(max(range(28),key=lambda i:vals[i]),x["high_layer"])
   self.assertEqual(min(range(28),key=lambda i:vals[i]),x["low_layer"])
   self.assertGreater(x["high_gain"],x["low_gain"])
 def test_6_single_native_role_policies(self):
  arms=self.pre["study_arms"]
  self.assertEqual(arms[0],"allQ4")
  self.assertEqual(len(arms),7)
  self.assertEqual(len(set(arms)),7)
  self.assertEqual(self.frozen["selected_source_extremes"],self.pre["role_extremes"])
 def test_single_tensor_experiment_not_28bit_budget(self):
  self.assertIn("Never merge to main",self.pre["operational_policy"])
  self.assertIn("Singleton n8 masks violate 28-n8 target budget",self.pre["evaluation_limitations"])
  self.assertIn("allQ4",self.pre["native_counterfactual"])
  self.assertIn("1 QNg64-n8 projection",self.pre["native_counterfactual"])
 def test_original_BF16_if_published(self):
  if not BF16.exists():self.skipTest("external BF16 not yet computed")
  x=json.loads(BF16.read_text())
  self.assertEqual(x["status"],"TRUE_ORIGINAL_BF16_QWEN_SOURCE_RACE16_MPS_REFERENCE_PASS")
  self.assertEqual(x["n"],16)
  self.assertTrue(x["independent_MPS_arithmetic_not_native_C_exact_parity"])
 def test_full_native_audit_if_published(self):
  if not FINAL.exists():self.skipTest("one-N8 real C arm execution not completed")
  x=json.loads(FINAL.read_text())
  self.assertEqual(x["status"],"REAL_SINGLE_TENSOR_1N8_VS_ALLQ4_16RACE_7ARMS_INDEPENDENT_AUDIT_PASS")
  self.assertEqual(len(x["measured_per_tensor_fullmodel_actual_NLL_effect"]),6)
  self.assertEqual(len(x["within_role_high_vs_low_source_proxy_recovery_alignment"]),3)
  self.assertTrue(x["model_single_tensor_native_intervention_proven"])
  self.assertFalse(x["automated_live_precision_promotion"])
 def test_postnative_role_pair_bootstrap_evidence(self):
  p=D/"BEGLIN_QV11_ROLE_RANK_PAIRED_BOOTSTRAP_PUBLIC_20261023.json"
  a=json.loads(p.read_text())
  self.assertEqual(a["status"],"PAIRWISE_TRUE_C_16ARTICLE_6INTERVENTION_BOOTSTRAP_COMPLETE")
  self.assertEqual(a["paired_source_bootstrap_n"],5000)
  self.assertEqual(a["sample_size_original_test_articles"],16)
  self.assertEqual(a["pair_direction_concordance"],2)
  self.assertEqual(a["pair_direction_total"],3)
  self.assertAlmostEqual(a["six_source_Fisher_vs_actual_C_NLL_gain_Spearman"],.6)
  self.assertFalse(a["promotion_denied"]==False)
  self.assertTrue(a["source_Fisher_global_ranking_not_established"])
  self.assertLess(a["six_probe_rank_Spearman_question_bootstrap_95CI"][0],0)
  self.assertGreater(a["six_probe_rank_Spearman_question_bootstrap_95CI"][1],0)
  self.assertFalse(a["within_role_high_vs_low_all_paired"]["down_proj"]["rank_direction_correct"])
 def test_final_handoff_preserves_non_generalization(self):
  p=D/"BEGLIN_QV11_REAL_SINGLE_TENSOR_NATIVE_HANDOFF_20261023.md"
  data=p.read_text()
  self.assertIn("16문항",data)
  self.assertIn("2/3",data)
  self.assertIn("112개",data)
  self.assertIn("운영",data)
  self.assertNotIn("범용 성능 입증",data)
if __name__=="__main__":unittest.main(verbosity=2)
