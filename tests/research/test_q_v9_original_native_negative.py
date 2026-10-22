"""Qv9 negative native C evidence and four-choice GGN mathematical contracts.

All mutations are in-memory via unittest.mock. Never alter native BF16 source,
original C trace, private held inputs, existing receipts or model checkpoints.
"""
import hashlib,json,math,pathlib,sys,unittest
from unittest.mock import patch
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v9_independent_native_mmlu as native
def sha(b):return hashlib.sha256(b).hexdigest()

class Qv9NegativeOriginalNativeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.real_read=staticmethod(native.read)
  cls.items=json.loads(cls.real_read(native.V))["items"]
  cls.manifest=json.loads(cls.real_read(native.N/"frozen_5x48_manifest.json"))
  cls.folder=native.N/"q4"
  cls.log_path=cls.folder/"real_mmlu48_native_C_log.txt"
  cls.receipt_path=cls.folder/"original_mmlu48_native_receipt.json"
  cls.real_log=cls.real_read(cls.log_path)
  cls.orig_receipt=json.loads(cls.real_read(cls.receipt_path))
 def fake_run(self, newlog, retarget_sha=False, question=None, manifest=None):
  blob=newlog if isinstance(newlog,bytes) else newlog.encode()
  orig=self.real_read
  rec=dict(self.orig_receipt)
  if retarget_sha:rec["native_log_SHA256"]=sha(blob)
  def retrieve(path):
   if path==self.log_path:return blob
   if path==self.receipt_path:return (json.dumps(rec)+"\n").encode()
   return orig(path)
  with patch.object(native,"read",side_effect=retrieve):
   return native.reparse("q4",self.items if question is None else question,
      self.manifest if manifest is None else manifest)
 def test_original_C_q4_runs_48_entire_model(self):
  scores=self.fake_run(self.real_log)
  self.assertEqual(scores["correct"],37)
  self.assertEqual(scores["n"],48)
  self.assertAlmostEqual(scores["mean_fullvocab_correct_nexttoken_NLL"],.8203794649745332)
 def test_native_q4_rejects_logged_output_tamper(self):
  altered=self.real_log.decode().replace("[qv5-choice] window 0 NLL ","[qv5-choice] window 0 NLL 99",1)
  with self.assertRaises(RuntimeError):self.fake_run(altered,False)
 def test_native_q4_rejects_wrong_precision_even_with_forged_sha(self):
  corrupted=self.real_log.decode().replace("projection_role_tensors: f32=0 n8=0 q4=196",
         "projection_role_tensors: f32=0 n8=28 q4=168",1)
  with self.assertRaises(RuntimeError):self.fake_run(corrupted,True)
 def test_native_q4_rejects_missing_case_even_with_forged_sha(self):
  lines=self.real_log.decode().splitlines()
  lines=[line for line in lines if not line.startswith("[qv5-choice] window 3 ")]
  with self.assertRaises(RuntimeError):self.fake_run("\n".join(lines)+"\n",True)
 def test_native_q4_rejects_duplicate_case_index_even_with_forged_sha(self):
  src=self.real_log.decode()
  altered=src.replace("[qv5-choice] window 5 ", "[qv5-choice] window 4 ",1)
  self.assertNotEqual(src,altered)
  with self.assertRaises(RuntimeError):self.fake_run(altered,True)
 def test_native_q4_rejects_nonfinite_nll_even_with_forged_sha(self):
  import re
  source=self.real_log.decode()
  altered=re.sub(r"(\[qv5-choice\] window 2 NLL )([0-9.eE+-]+)",r"\1nan",source,count=1)
  self.assertNotEqual(source,altered)
  with self.assertRaises(RuntimeError):self.fake_run(altered,True)
 def test_native_q4_rejects_forged_prefix_length(self):
  import re
  source=self.real_log.decode()
  altered=re.sub(r"(\[qv5-choice\] window 0 NLL [^\n]*? prefix )(\d+)",
    lambda m:m.group(1)+str(int(m.group(2))+2),source,count=1)
  self.assertNotEqual(source,altered)
  with self.assertRaises(RuntimeError):self.fake_run(altered,True)
 def test_native_q4_rejects_forged_answer_choice(self):
  import re
  source=self.real_log.decode()
  altered=re.sub(r"(\[qv5-choice\] window 1 NLL [^\n]*? expected )(\d+)",
    lambda m:m.group(1)+str((int(m.group(2))+1)%4),source,count=1)
  self.assertNotEqual(source,altered)
  with self.assertRaises(RuntimeError):self.fake_run(altered,True)
 def test_native_q4_rejects_swapped_frozen_question_ids(self):
  cases=list(self.items)
  cases[0],cases[1]=cases[1],cases[0]
  with self.assertRaises(RuntimeError):self.fake_run(self.real_log,False,question=cases)
 def test_native_q4_rejects_tampered_input_manifest(self):
  frozen=json.loads(json.dumps(self.manifest))
  frozen["inputs"][0]["prompt_SHA256"]="0"*64
  with self.assertRaises(RuntimeError):self.fake_run(self.real_log,False,manifest=frozen)
 def test_native_q4_rejects_fake_sum_even_with_forged_sha(self):
  source=self.real_log.decode()
  self.assertIn("qv5_choice NLL ",source)
  altered=source.replace("qv5_choice NLL ", "qv5_choice NLL 999",1)
  with self.assertRaises(RuntimeError):self.fake_run(altered,True)
 def test_original_five_condition_bit_contract(self):
  self.assertEqual(self.manifest["modes"],list(native.MODES))
  self.assertEqual(len(self.manifest["inputs"]),48)
  for label,path in native.MASK.items():
   tokens={tuple(line.split()) for line in path.read_text().splitlines()}
   self.assertEqual(len(tokens),28,label)
 def test_gauss_newton_softmax_small_signal_Taylor(self):
  p=np.array([.19,.27,.16,.38],dtype=np.float64)
  pert=np.array([.2,-.32,.45,.0],dtype=np.float64)
  eps=1e-4
  delta=eps*pert
  # Exact D_KL(p || softmax(log(p)+delta)) = log(E_p e^delta)-E_p delta.
  exact=float(np.log(np.sum(p*np.exp(delta)))-np.sum(p*delta))
  approx=.5*(np.sum(p*delta**2)-(np.sum(p*delta))**2)
  self.assertGreater(exact,0)
  self.assertAlmostEqual(exact,approx,delta=3e-10)
 def test_ggn_is_not_full_vocab_Hessian(self):
  t=json.loads((ROOT/"docs/research/BEGLIN_QV9_GGN_QUADRATIC_TRUST_REGION_DIAGNOSTIC_20261023.json").read_text())
  self.assertTrue(t["exact_choice_KL_is_only_on_original_first_order_Jdelta_logits_not_fullmodel_native_logits"])
  diag=t["source_and_diagnostic"]["diagnostic"]["Qv9_GGN24"]
  self.assertGreater(diag["median_max_abs_relative_logit_shift"],1)
  self.assertGreater(diag["mean_linearized_4choice_exact_KL"],0)
 def test_original_source_proxy_overfits_diagnostic(self):
  d=json.loads((ROOT/"docs/research/BEGLIN_QV9_INDEPENDENT_OUTPUT_GGN_TRAIN_DIAGNOSTIC_20261023.json").read_text())
  rows=d["source_proxy_vs_diagnostic_metrics"]
  self.assertLess(rows["ggn24"]["train"]["robust_GGN"],rows["fisher64"]["train"]["robust_GGN"])
  self.assertGreater(rows["ggn24"]["diagnostic"]["robust_GGN"],rows["fisher64"]["diagnostic"]["robust_GGN"])
  self.assertFalse(d["auto_promotion"])
if __name__=="__main__": unittest.main(verbosity=2)
