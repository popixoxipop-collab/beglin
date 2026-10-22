"""Qv9 portable research evidence contracts, standard library only."""
import hashlib,json,math,pathlib,unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
C=ROOT/"configs/research";D=ROOT/"docs/research"
PRE=C/"BEGLIN_QV9_GGN_SOURCE_PRELOCK_20261023.json"
LOCK=D/"BEGLIN_QV9_INPUT_PREMODEL_FREEZE_20261023.json"
SRC=D/"BEGLIN_QV9_TRAIN24_CONDITIONAL_GGN_PUBLIC_20261023.json"
POL=D/"BEGLIN_QV9_ROBUST_CONDITIONAL_GGN_SOURCE_POLICY_20261023.json"
MASK=C/"BEGLIN_QV9_GGN_28N8_FROZEN_MASK_20261023.txt"
DIAG=D/"BEGLIN_QV9_DIAGNOSTIC24_CONDITIONAL_GGN_PUBLIC_20261023.json"
SOURCE_AUDIT=D/"BEGLIN_QV9_INDEPENDENT_OUTPUT_GGN_TRAIN_DIAGNOSTIC_20261023.json"
RESULT=D/"BEGLIN_QV9_MMLU48_NATIVE_FIVE_ARM_INDEPENDENT_PUBLIC_20261023.json"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
class Qv9PortableContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.pre=json.loads(PRE.read_text())
  cls.lock=json.loads(LOCK.read_text())
  cls.src=json.loads(SRC.read_text())
  cls.pol=json.loads(POL.read_text())
 def test_prereg_and_model_revision(self):
  self.assertEqual(sha(PRE),"73dc2829da79637a97d68f903f72fed753eace2b154caec111c1fe60a7f4f983")
  self.assertEqual(self.pre["original_model_rev"],self.pol["original_checkpoint_rev"])
  self.assertEqual(self.lock["prelock_sha256"],sha(PRE))
 def test_mmlu_subjects_are_frozen_before_model(self):
  self.assertEqual(self.lock["status"],"ALL_THREE_COHORTS_FROZEN_BEFORE_BF16_GGN_BACKWARD_AND_NATIVE")
  self.assertEqual(self.lock["eval_count"],48)
  self.assertEqual(len(self.lock["source_subjects"]),2)
  self.assertFalse(self.lock["model_logit_or_accuracy_accessed"])
 def test_source_train24_and_diagnostic24_sha_distinct(self):
  self.assertNotEqual(self.lock["source_train24_private_SHA256"],self.lock["source_diagnostic24_private_SHA256"])
  self.assertEqual(self.src["n_source_examples"],24)
  self.assertEqual(self.src["n_contrastive_logits_backprop"],3)
  self.assertTrue(self.pol["source_diagnostic_24_unopened"])
 def test_true_original_quant_error_jacobians(self):
  self.assertEqual(self.src["source_group"],"train")
  self.assertEqual(self.src["model_projection_tensors"],196)
  self.assertTrue(self.src["GGN_conditional_4choice_0p5_var_correct"])
  self.assertTrue(self.src["no_single_tensor_downstream_output_intervention"])
 def test_choice_output_GGN_not_true_full_vocab(self):
  self.assertTrue(self.pol["source_only_choice_GGN_not_full_vocab"])
  self.assertTrue(self.pol["conditional_GGN_does_not_include_unselected_other_vocab_logits"])
  self.assertFalse(self.pol["global_optimality_proved"])
 def test_frozen_bit_policy_has_28_unique(self):
  rows=[tuple(r.split()) for r in MASK.read_text().splitlines()]
  self.assertEqual(len(rows),28)
  self.assertEqual(len(set(rows)),28)
  self.assertEqual(sha(MASK),self.pol["mask_sha256"])
 def test_original_kernel_costs_frozen(self):
  k=self.pol["final_Apple_native_source_costs"]
  self.assertLessEqual(k["memory_bytes"],830570496)
  self.assertLessEqual(k["serial_estimate_ms"],100)
  self.assertLessEqual(k["isolated_async_stage_ms"],50.00001)
 def test_positive_semidefinite_nonnegative_GGN(self):
  self.assertGreater(self.src["source_mean_Q4_conditional_GGN"],0)
  self.assertGreaterEqual(self.pol["selected_components"]["GGN_mean"],0)
  self.assertGreaterEqual(self.pol["selected_components"]["ridge_diagonal_GGN"],0)
 def test_source_train_proxy_decreased_not_fullmodel_NLL(self):
  self.assertLess(self.pol["selected_regularized_GGN_risk"],self.pol["prior_mask_source_objectives"]["allQ4"]["objective"])
  self.assertTrue(self.pol["source_only_choice_GGN_not_full_vocab"])
 def test_source_diagnostics_if_present(self):
  if not DIAG.exists():self.skipTest("independent pre-frozen source diagnostic in progress")
  a=json.loads(DIAG.read_text())
  self.assertEqual(a["source_group"],"diagnostic")
  self.assertEqual(a["n_contrastive_logits_backprop"],3)
 def test_independent_source_replay_if_present(self):
  if not SOURCE_AUDIT.exists():self.skipTest("independent original BF16 GGN replay in progress")
  s=json.loads(SOURCE_AUDIT.read_text())
  self.assertTrue(s["only_source24_train_selects_mask"])
  self.assertFalse(s["auto_promotion"])
 def test_native_5arm_if_present(self):
  if not RESULT.exists():self.skipTest("real 5-condition MMLU CPU trials incomplete")
  s=json.loads(RESULT.read_text())
  self.assertEqual(s["status"],"PASS_SOURCE_AND_ORIGINAL_MMLU48_FIVE_NATIVE_POLICY_AUDIT")
  self.assertEqual(len(s["native_summary"]),5)
  self.assertFalse(s["generalization_established"])
 def test_no_production_promotion(self):
  self.assertFalse(self.pol["production_promotion"])
  self.assertTrue(self.pol["MMLU_eval_48_logits_unopened"])
  self.assertTrue(self.pre["no_live_release_or_main_changes"])
if __name__=="__main__":unittest.main(verbosity=2)
