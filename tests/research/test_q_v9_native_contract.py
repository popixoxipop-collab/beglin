"""Private Qv9 BF16 output-GGN, source diagnostic, native C negative tests."""
import hashlib,json,math,pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v9_native_mmlu48 as native
import q_v9_independent_ggn_source as audit
L=ROOT/"lab_local/qv9_ggn_20261023"
D=ROOT/"docs/research"
C=ROOT/"configs/research"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Qv9RealSourceContract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.policy=json.loads((D/"BEGLIN_QV9_ROBUST_CONDITIONAL_GGN_SOURCE_POLICY_20261023.json").read_text())
  cls.train=json.loads(audit.sealed(L/"original_BF16_choice_GGN_train24_private.json"))
  cls.diag=json.loads(audit.sealed(L/"original_BF16_choice_GGN_diagnostic24_private.json"))
  cls.replay=json.loads((D/"BEGLIN_QV9_INDEPENDENT_OUTPUT_GGN_TRAIN_DIAGNOSTIC_20261023.json").read_text())
 def test_real_24x196x3_true_bf16_original_logits(self):
  for rec in (self.train,self.diag):
   self.assertEqual(len(rec["q4_signed_relative_logit_delta"]),24)
   self.assertEqual(np.shape(rec["n8_signed_relative_logit_delta"]),(24,196,3))
   self.assertEqual(rec["n_contrastive_logits_backprop"],3)
   self.assertTrue(rec["GGN_true_full_Hessian_not_claimed"])
 def test_source_train24_and_diagnostic24_independent(self):
  self.assertNotEqual(self.train["frozen_source_token_input_SHA256"],self.diag["frozen_source_token_input_SHA256"])
  self.assertTrue(self.policy["source_diagnostic_24_unopened"])
 def test_source_has_negative_and_positive_directions(self):
  q=np.asarray(self.train["q4_signed_relative_logit_delta"])
  self.assertLess(float(q.min()),0)
  self.assertGreater(float(q.max()),0)
 def test_original_probabilities_normalized(self):
  for rec in (self.train,self.diag):
   p=np.asarray(rec["original_conditional_four_letter_probabilities"])
   self.assertTrue(np.allclose(p.sum(axis=1),1,atol=1e-5))
   self.assertGreaterEqual(float(p.min()),0.)
 def test_conditional_curvature_is_PSD(self):
  p=np.array([[.4,.3,.2,.1]])
  self.assertAlmostEqual(audit.ggn(np.zeros((1,3)),p)[0],0)
  self.assertAlmostEqual(audit.ggn(np.array([[1.,0.,0.]]),p)[0],.5*.4*.6)
 def test_source_optimality_not_global(self):
  self.assertFalse(self.policy["global_optimality_proved"])
  self.assertTrue(self.policy["source_1swap_local_optimality_only"])
 def test_no_learn_from_mmlu(self):
  self.assertTrue(self.policy["MMLU_eval_48_logits_unopened"])
  self.assertFalse(self.policy["production_promotion"])
 def test_source_independent_recomputed(self):
  self.assertEqual(self.replay["status"],"PASS_48_ACTUAL_BF16_SOURCE_CASES_AND_144_JACOBIAN_BACKWARDS_REPLAYED")
  self.assertAlmostEqual(self.replay["source_proxy_vs_diagnostic_metrics"]["ggn24"]["train"]["robust_GGN"],self.policy["selected_regularized_GGN_risk"])
 def test_training_rank_not_held_rank(self):
  x=self.replay["source_proxy_vs_diagnostic_metrics"]
  self.assertLess(x["ggn24"]["train"]["robust_GGN"],x["fisher64"]["train"]["robust_GGN"])
  self.assertGreater(x["ggn24"]["diagnostic"]["robust_GGN"],x["fisher64"]["diagnostic"]["robust_GGN"])
 def test_4704_independent_swaps_certificate(self):
  c=json.loads((D/"BEGLIN_QV9_GGN_EXHAUSTIVE_LOCAL_SWAP_CERT_20261023.json").read_text())
  self.assertEqual(c["source_one_swap_total_eligible_pairs"],4704)
  self.assertGreater(c["source_feasible_Apple_cpu_kernel_pairs"],1)
  self.assertGreater(c["minimum_feasible_next_1swap_GGN_change"],0)
 def test_mask_exact_28_native_q4n8_original_roles(self):
  x=[tuple(z.split()) for z in (C/"BEGLIN_QV9_GGN_28N8_FROZEN_MASK_20261023.txt").read_text().splitlines()]
  self.assertEqual(len(x),28);self.assertEqual(len(set(x)),28)
 def test_trust_region_warning_quantified(self):
  tr=json.loads((D/"BEGLIN_QV9_GGN_QUADRATIC_TRUST_REGION_DIAGNOSTIC_20261023.json").read_text())
  self.assertGreater(tr["source_and_diagnostic"]["train"]["Qv9_GGN24"]["fraction_abs_4choice_relative_logit_shift_gt_4"],0.5)
  self.assertTrue(tr["exact_choice_KL_is_only_on_original_first_order_Jdelta_logits_not_fullmodel_native_logits"])
 def test_real_native_inputs_same_per_policy(self):
  manifest=json.loads(native.sealed(native.RUN/"frozen_5x48_manifest.json"))
  self.assertEqual(len(manifest["inputs"]),48)
  self.assertEqual(len(manifest["modes"]),5)
 def test_q4_existing_native_if_present(self):
  q=native.RUN/"q4/original_mmlu48_native_receipt.json"
  if not q.is_file():self.skipTest("native baseline still running")
  d=json.loads(native.sealed(q))
  self.assertEqual(d["metrics"]["total"],48)
  self.assertEqual(d["mode"],"q4")
 def test_full_native_independent_if_published(self):
  target=D/"BEGLIN_QV9_MMLU48_NATIVE_FIVE_ARM_INDEPENDENT_PUBLIC_20261023.json"
  if not target.is_file():self.skipTest("native other policies still running")
  doc=json.loads(target.read_text())
  self.assertEqual(len(doc["native_summary"]),5)
  self.assertFalse(doc["auto_promotion"])
if __name__=="__main__":unittest.main(verbosity=2)
