"""Qv11 non-destructive adversarial audit checks for actual native 1-N8 C trace.

This file intentionally uses private EOE research evidence; GitHub public CI
runs only test_q_v11_public_contract.py. All invalid cases are in-memory,
via mock, and never alter the genuine private model, masks or original logs.
"""
import hashlib,json,pathlib,sys,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v11_independent_true_native_audit as a
def sha(b):return hashlib.sha256(b).hexdigest()
class Qv11NativeSingletonNegative(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.read=staticmethod(a.sealed)
  cls.questions=json.loads(cls.read(a.FREEZE))["real_question_source"]
  cls.manifest=json.loads(cls.read(a.RUN/"Qv11_true_one_tensor_native_all_7arms_prelock.json"))
  cls.logs={x:cls.read(a.RUN/x/"true_28layer_RACE16_original_C_process.log") for x in ("q4","down_proj_2")}
  cls.rec={x:json.loads(cls.read(a.RUN/x/"original_singleton_1n8_RACE16_native_result.json")) for x in ("q4","down_proj_2")}
 def run_altered(self,mode,newlog=None,retarget_receipt=True,new_source=None,new_manifest=None):
  directory=a.RUN/mode
  p=directory/"true_28layer_RACE16_original_C_process.log"
  receiptp=directory/"original_singleton_1n8_RACE16_native_result.json"
  raw=self.logs[mode] if newlog is None else (newlog.encode() if isinstance(newlog,str) else newlog)
  record=dict(self.rec[mode])
  if retarget_receipt:record["original_native_CPU_process_log_SHA256"]=sha(raw)
  base_read=self.read
  def load(path):
   if path==p:return raw
   if path==receiptp:return (json.dumps(record)+"\n").encode()
   return base_read(path)
  with patch.object(a,"sealed",side_effect=load):
   return a.parse(mode,self.questions if new_source is None else new_source,
                  self.manifest if new_manifest is None else new_manifest)
 def test_allQ4_original_true_C_16_scores(self):
  x=self.run_altered("q4")
  self.assertEqual(x["correct"],8)
  self.assertAlmostEqual(x["actual_mean_correct_token_NLL"],2.4325947732200754)
 def test_true_one_n8_195_Q4_original_C(self):
  x=self.run_altered("down_proj_2")
  self.assertEqual(x["correct"],7)
  self.assertAlmostEqual(x["actual_mean_correct_token_NLL"],2.164659843781045)
 def test_tampered_log_rejected_at_sha(self):
  log=self.logs["q4"].replace(b"[qv5-choice] window 0 NLL ",b"[qv5-choice] window 0 NLL 9",1)
  with self.assertRaises(RuntimeError):self.run_altered("q4",log,False)
 def test_wrong_quantized_role_count_rejected_forged_sha(self):
  old=b"projection_role_tensors: f32=0 n8=1 q4=195"
  log=self.logs["down_proj_2"].replace(old,b"projection_role_tensors: f32=0 n8=2 q4=194",1)
  self.assertNotEqual(log,self.logs["down_proj_2"])
  with self.assertRaises(RuntimeError):self.run_altered("down_proj_2",log)
 def test_missing_window_with_forged_sha(self):
  items=self.logs["q4"].decode().splitlines()
  log="\n".join(x for x in items if not x.startswith("[qv5-choice] window 5 "))+"\n"
  with self.assertRaises(RuntimeError):self.run_altered("q4",log)
 def test_duplicate_window_id_forged_sha(self):
  log=self.logs["q4"].replace(b"[qv5-choice] window 3 ",b"[qv5-choice] window 2 ",1)
  with self.assertRaises(RuntimeError):self.run_altered("q4",log)
 def test_changed_source_article_order_rejected(self):
  new=list(self.questions)
  new[0],new[1]=new[1],new[0]
  with self.assertRaises(RuntimeError):self.run_altered("q4",new_source=new)
 def test_changed_candidate_checksum_rejected(self):
  obj=json.loads(json.dumps(self.manifest))
  obj["same_question_SHA_and_candidate_SHA_across_7arms"][0]["answer_option_sha"]="0"*64
  with self.assertRaises(RuntimeError):self.run_altered("q4",new_manifest=obj)
 def test_wrong_original_precision_mask_sha_rejected(self):
  obj=json.loads(json.dumps(self.manifest))
  obj["actual_source_role_mask_SHA256"]["down_proj_2"]="0"*64
  with self.assertRaises(RuntimeError):self.run_altered("down_proj_2",new_manifest=obj)
 def test_bad_total_rejected_forged_sha(self):
  txt=self.logs["q4"].replace(b"qv5_choice NLL ",b"qv5_choice NLL 499",1)
  with self.assertRaises(RuntimeError):self.run_altered("q4",txt)
if __name__=="__main__":unittest.main(verbosity=2)
