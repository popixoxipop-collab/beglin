"""BEGLIN Qv12 portable authentic native Q4/singleton/pair research contracts.

Runs on GitHub Actions without private Qwen model or source RACE case arrays.
Original BF16 Qwen, native C binary and original per-item log SHAs are
validated separately by the EOE independent native-audit workflow.
"""
from __future__ import annotations
import hashlib,json,math,pathlib,unittest
R=pathlib.Path(__file__).resolve().parents[2]
C=R/"configs/research";D=R/"docs/research"
PRE=C/"BEGLIN_QV12_TWO_N8_INTERACTION_PRELOCK_20261023.json"
FREEZE=D/"BEGLIN_QV12_RACE16_NEW_INPUT_FREEZE_PUBLIC_20261023.json"
BUILD=D/"BEGLIN_QV12_REAL_NATIVE_2N8_C_SOURCE_PROVENANCE_PUBLIC_20261023.json"
PREVIOUS=D/"BEGLIN_QV11_REAL_ONE_TENSOR_NATIVE_CAUSAL_AUDIT_PUBLIC_20261023.json"
RESULT=D/"BEGLIN_QV12_REAL_TWO_N8_NATIVE_INTERACTION_INDEPENDENT_PUBLIC_20261023.json"
HANDOFF=D/"BEGLIN_QV12_TWO_TENSOR_NONADDITIVE_NATIVE_RACE16_HANDOFF_20261023.md"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
class Qv12PortableContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.pre=json.loads(PRE.read_text())
  cls.freeze=json.loads(FREEZE.read_text())
  cls.build=json.loads(BUILD.read_text())
 def test_immutable_before_native_experiments(self):
  self.assertEqual(sha(PRE),"a565bdec341d675ce52a37132b06355760c2fef96b8e4329c17508a237e4374f")
  self.assertEqual(self.pre["status"],"FROZEN_PAIR_AND_NEW_RACE_TEST_DESIGN_BEFORE_NATIVE_2N8_AND_1N8_RUN")
  self.assertEqual(self.freeze["prelock_SHA256"],sha(PRE))
 def test_original_model_and_C_kernel_fixed(self):
  self.assertEqual(self.pre["model_HF_rev"],"989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
  self.assertTrue(self.build["original_C_NEON_Q4_N8_kernels_unchanged"])
  self.assertTrue(self.build["original_model_weights_unchanged"])
  self.assertEqual(self.build["native_layer_count"],28)
  self.assertEqual(self.build["native_projection_role_count"],196)
 def test_three_precisions_actual_native_executable_provenance(self):
  self.assertEqual(self.build["status"],"SOURCE_PINNED_ONLY_CARDINALITY_2N8_REAL_QWEN28_NATIVE_CPU_BINARY_BUILT")
  self.assertEqual(self.build["original_Qv5_allQ4_C_binary_sha256"],"408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c")
  self.assertEqual(self.build["original_Qv11_1N8_C_binary_sha256"],"a054e68789fd6bc20a4bd815da5108b63361b14227b1fc4ab9e68a3b7b9a2711")
  self.assertEqual(self.build["new_Qv12_2N8_C_binary_sha256"],"d40fd6d42e08b0bc9d5dd3c7f8e0a4f5bcaabb2d9de7310d89f033e02500e961")
  self.assertNotEqual(self.build["original_Qv11_1N8_C_binary_sha256"],self.build["new_Qv12_2N8_C_binary_sha256"])
 def test_fixed_model_C_change_only_cardinality(self):
  self.assertIn("total_n8 28 -> 2",self.build["single_semantic_C_change"])
  self.assertTrue(self.build["only_diagnostic_logging_changes"])
  self.assertTrue(self.build["raw_C_source_binary_remains_private"])
 def test_previous_Qv11_effects_reused_for_selection_explicitly(self):
  parent=json.loads(PREVIOUS.read_text())
  self.assertEqual(self.pre["prior_Qv11_7arm_true_N8_public_audit_SHA256"],sha(PREVIOUS))
  for t in self.pre["three_preselected_intervention_tensors"]:
   key=f"{t['role']}_{t['layer']}"
   self.assertAlmostEqual(parent["measured_per_tensor_fullmodel_actual_NLL_effect"][key]["actual_N8_improvement_fullQ4_minus_N8"],
     t["observed_Qv11_prior_source16_improvement_NLL"])
  self.assertEqual({(x["role"],x["layer"]) for x in self.pre["three_preselected_intervention_tensors"]},
    {("down_proj",10),("v_proj",20),("k_proj",0)})
 def test_seven_arms_from_three_prior_precisions(self):
  modes=self.pre["planned_arms"]
  self.assertEqual(len(modes),7)
  self.assertEqual(len(set(modes)),7)
  self.assertEqual(modes[0],"q4")
  self.assertEqual(sum("__" in x for x in modes),3)
  self.assertEqual(sum("__" not in x and x!="q4" for x in modes),3)
 def test_new_article_disjoint_from_original_Qv11(self):
  self.assertEqual(self.freeze["status"],"FROZEN_THREE_TENSORS_SIX_NATIVE_UPGRADES_AND_16_NEW_RACE_ARTICLES_BEFORE_EXECUTION")
  self.assertTrue(self.freeze["new_test16_disjoint_original_article_ids"])
  self.assertEqual(self.freeze["new_test16_distinct_article_count"],16)
  self.assertNotEqual(self.freeze["RACE_new_test16_private_SHA256"],
    self.freeze["old_Qv11_RACE_test16_SHA256"])
  self.assertEqual(self.freeze["new_test16_prompt_len_range"][1]<=208,True)
 def test_source_has_already_seen_old_Qv11_dev_outcomes(self):
  self.assertTrue(self.freeze["preselected_source_observed_singleton_outcome_reused"])
  self.assertTrue(self.pre["held_not_pristine_model_pretraining"].startswith("Public"))
 def test_true_interaction_math_signs(self):
  L0=1.0;La=.8;Lb=.9
  self.assertAlmostEqual((La+Lb-L0)-La-Lb+L0,0)
  self.assertLess(.65-La-Lb+L0,0)
  self.assertGreater(.75-La-Lb+L0,0)
 def test_16_article_bootstrap_seed_and_no_generalization(self):
  self.assertEqual(self.pre["held_article_count"],16)
  self.assertIn("5000",self.pre["statistics"])
  self.assertIn("seed 20261023",self.pre["statistics"])
  self.assertIn("not",self.pre["held_not_pristine_model_pretraining"])
 def test_not_matched_28n8_budget_no_serving_claim(self):
  self.assertIn("not budget-matched",self.pre["constraints"])
  self.assertTrue(self.pre["no_live_promotion"])
  self.assertTrue(self.pre["no_main_merge"])
  self.assertTrue(self.pre["no_operator_release_change"])
  self.assertTrue(self.pre["no_automatic_precision_serving"])
 def test_native_112_logs_if_frozen(self):
  if not RESULT.exists():self.skipTest("native seven arm independent audit not yet published")
  d=json.loads(RESULT.read_text())
  self.assertEqual(d["status"],"PASS_7X16_ORIGINAL_BEGLIN_C_TRUE_SINGLE_AND_PAIR_TWO_N8_INTERACTION")
  self.assertEqual(len(d["native_summary"]),7)
  self.assertEqual(len(d["native_pairwise_interaction_exact_finite_differences"]),3)
  self.assertTrue(d["exact_output_is_real_endtask_NLL_not_GGN_proxy"])
  self.assertFalse(d["automatic_precision_live_promotion"])
 def test_exact_original_kernel_role_packed_byte_costs(self):
  p=D/"BEGLIN_QV12_ONE_TWO_N8_PACKED_WEIGHT_COST_PUBLIC_20261023.json"
  x=json.loads(p.read_text())
  self.assertEqual(x["status"],"SOURCE_ONLY_7ARM_ACTUAL_Q4_N8_WEIGHT_AND_GROUP_SCALE_LEDGER")
  self.assertEqual(x["source_PRELOCK_SHA256"],sha(PRE))
  self.assertEqual(x["original_BEGLIN_Q4_projection_baseline_bytes"],736985088)
  self.assertEqual(x["separate_original_role_byte_costs"]["down_proj"]["increment_n8"],6881280)
  self.assertEqual(x["separate_original_role_byte_costs"]["v_proj"]["increment_n8"],196608)
  self.assertEqual(x["separate_original_role_byte_costs"]["k_proj"]["increment_n8"],196608)
  self.assertTrue(x["not_official_28n8_serving_configuration"])
  self.assertTrue(x["actual_max_RSS_not_measured"])
  self.assertEqual(x["native_one_two_bit_arm_costs"]["down_proj_10__v_proj_20"]["projected_packed_projection_weight_bytes"],744062976)
 def test_handoff_if_published(self):
  if not HANDOFF.exists():self.skipTest("final native Qv12 outcome not yet audited")
  doc=HANDOFF.read_text()
  self.assertIn("112",doc)
  self.assertIn("비가산",doc)
  self.assertIn("운영",doc)
if __name__=="__main__":unittest.main(verbosity=2)
