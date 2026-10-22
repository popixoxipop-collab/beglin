"""Fail-closed negative controls against authentic BEGLIN RACE48 C logs.

In-memory monkeypatch for tampered native C trace/receipt, original held
source prompt, true 4-option answer and bit precision count. Files remain
unchanged and the original 48-question trial is not rerun.
"""
import hashlib,json,pathlib,sys,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v10_independent_native_race as audit
import q_v10_grouped_kl_optimizer as grouped
import numpy as np

def sha(b):return hashlib.sha256(b).hexdigest()
class Qv10NativeNegative(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls._read=staticmethod(audit.private)
  cls.samples=json.loads(cls._read(audit.RACE))["questions"]
  cls.manifest=json.loads(cls._read(audit.DIR/"RACE48_native_original_preexecution_lock.json"))
  cls.orig=cls._read(audit.DIR/"q4"/"original_Qwen_C_RACE48_native_log.txt")
  cls.receipt=json.loads(cls._read(audit.DIR/"q4"/"original_RACE48_native_receipt.json"))
  cls.logfile=audit.DIR/"q4"/"original_Qwen_C_RACE48_native_log.txt"
  cls.recfile=audit.DIR/"q4"/"original_RACE48_native_receipt.json"
 def run_changed(self,log=None,update_digest=True,cases=None,manifest=None):
  raw=self.orig if log is None else (log.encode() if isinstance(log,str) else log)
  rec=dict(self.receipt)
  if update_digest:rec["real_original_48_window_C_log_SHA256"]=sha(raw)
  orig=self._read
  def get(path):
   if path==self.logfile:return raw
   if path==self.recfile:return (json.dumps(rec)+"\n").encode()
   return orig(path)
  with patch.object(audit,"private",side_effect=get):
   return audit.native("q4",self.manifest if manifest is None else manifest,
       self.samples if cases is None else cases)
 def test_real_original_q4_scores_present(self):
  a=self.run_changed(self.orig)
  self.assertEqual(a["correct"],32)
  self.assertEqual(a["n"],48)
  self.assertAlmostEqual(a["correct_letter_mean_fullvocab_NLL"],1.183725255948745)
 def test_native_log_checksum_tamper_without_receipt_rejected(self):
  bad=self.orig.replace(b"[qv5-choice] window 0 NLL ",b"[qv5-choice] window 0 NLL 55",1)
  with self.assertRaises(RuntimeError):self.run_changed(bad,False)
 def test_duplicate_native_index_rejected_even_forged_sha(self):
  bad=self.orig.replace(b"[qv5-choice] window 3 ",b"[qv5-choice] window 2 ",1)
  with self.assertRaises(RuntimeError):self.run_changed(bad)
 def test_removed_original_question_rejected(self):
  txt=self.orig.decode().splitlines()
  bad="\n".join(s for s in txt if not s.startswith("[qv5-choice] window 4 "))+"\n"
  with self.assertRaises(RuntimeError):self.run_changed(bad)
 def test_forged_q4_n8_mix_rejected(self):
  bad=self.orig.replace(b"projection_role_tensors: f32=0 n8=0 q4=196",
     b"projection_role_tensors: f32=0 n8=28 q4=168",1)
  with self.assertRaises(RuntimeError):self.run_changed(bad)
 def test_corrupted_correct_letter_rejected(self):
  import re
  txt=self.orig.decode()
  bad=re.sub(r"(\[qv5-choice\] window 0 NLL [^\n]*? expected )(\d+)",
      lambda m:m.group(1)+str((int(m.group(2))+1)%4),txt,count=1)
  self.assertNotEqual(txt,bad)
  with self.assertRaises(RuntimeError):self.run_changed(bad)
 def test_changed_prompt_length_rejected(self):
  import re
  txt=self.orig.decode()
  bad=re.sub(r"(\[qv5-choice\] window 0 NLL [^\n]*? prefix )(\d+)",
     lambda m:m.group(1)+str(int(m.group(2))+2),txt,count=1)
  self.assertNotEqual(txt,bad)
  with self.assertRaises(RuntimeError):self.run_changed(bad)
 def test_original_source_article_swapped_rejected(self):
  altered=list(self.samples)
  altered[0],altered[1]=altered[1],altered[0]
  with self.assertRaises(RuntimeError):self.run_changed(self.orig,cases=altered)
 def test_original_frozen_manifest_sha_corrupted_rejected(self):
  bad=json.loads(json.dumps(self.manifest))
  bad["all_48_inputs"][1]["prefix_sha256"]="0"*64
  with self.assertRaises(RuntimeError):self.run_changed(self.orig,manifest=bad)
 def test_native_nonfinite_loss_rejected(self):
  txt=self.orig.decode()
  import re
  bad=re.sub(r"(\[qv5-choice\] window 1 NLL )([0-9.+\-eE]+)",r"\1nan",txt,count=1)
  self.assertNotEqual(txt,bad)
  with self.assertRaises(RuntimeError):self.run_changed(bad)
 def test_modified_original_total_rejected(self):
  bad=self.orig.replace(b"qv5_choice NLL ",b"qv5_choice NLL 500",1)
  with self.assertRaises(RuntimeError):self.run_changed(bad)
 def test_gaussnewton_topk_KL_small_signal_positive(self):
  p=np.array([[.15,.05,.23,.22,.18,.07,.10]],dtype=np.float64)
  delta=np.array([[.08,-.06,.04,-.15,.03,.05]],dtype=np.float64)
  kl=grouped.grouped_KL(delta,p)
  self.assertGreaterEqual(float(kl[0]),0)
  u7=np.r_[delta[0],0.]
  second=.5*(np.sum(p[0]*u7*u7)-(np.sum(p[0]*u7))**2)
  self.assertAlmostEqual(float(kl[0]),float(second),delta=2e-4)
 def test_fullvocab_mass_other_collapsed_not_full_hessian(self):
  doc=json.loads((ROOT/"docs/research/BEGLIN_QV10_TOP2OTHER_TRAIN24_PUBLIC_20261023.json").read_text())
  self.assertLess(doc["mean_mass_OTHER"],.005)
  self.assertTrue(doc["not_fullvocab_Hessian_due_to_collapsed_OTHER"])
if __name__=="__main__":unittest.main(verbosity=2)
