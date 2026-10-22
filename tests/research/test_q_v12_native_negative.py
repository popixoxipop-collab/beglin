"""Qv12 adversarial original native C 0/1/2 n8 trace verification.

All tampering is in-memory via unittest.mock. Original EOE private RACE
token arrays, model weights, original C logs and quantized binaries remain
unchanged. GitHub CI executes portable public contracts, not this private
real-Qwen evidence suite.
"""
import hashlib,json,pathlib,sys,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v12_independent_native_pair_audit as a
def sha(b):return hashlib.sha256(b).hexdigest()
class OriginalQv12NativeNegative(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.read=staticmethod(a.sealed)
  cls.source=json.loads(cls.read(a.SOURCE))["full_frozen_questions"]
  cls.manifest=json.loads(cls.read(a.RUN/"seven_arm_original_inputs_and_masks_preexec_sha_manifest.json"))
  cls.logs={name:cls.read(a.RUN/name/"actual_original_Qwen28_C_pair_single_RACEtest16.log")
    for name in ("q4","down_proj_10","down_proj_10__v_proj_20")}
  cls.receipts={name:json.loads(cls.read(a.RUN/name/"original_C_RACE16_arm_receipt.json"))
    for name in cls.logs}
 def run_changed(self,mode,newlog=None,fix_sha=True,new_source=None,new_manifest=None,newmask=None):
  base=a.RUN/mode
  logfile=base/"actual_original_Qwen28_C_pair_single_RACEtest16.log"
  rpath=base/"original_C_RACE16_arm_receipt.json"
  pathmask=base/"native_n8_role_mask.txt"
  blob=self.logs[mode] if newlog is None else (newlog if isinstance(newlog,bytes) else newlog.encode())
  receipt=dict(self.receipts[mode])
  if fix_sha:receipt["actual_original_C_process_log_sha256"]=sha(blob)
  original_read=self.read
  def fake(p):
   if p==logfile:return blob
   if p==rpath:return (json.dumps(receipt)+"\n").encode()
   if newmask is not None and p==pathmask:return newmask
   return original_read(p)
  with patch.object(a,"sealed",side_effect=fake):
   return a.parse_arm(mode,self.source if new_source is None else new_source,
      self.manifest if new_manifest is None else new_manifest)
 def test_real_authentic_allQ4_16_new_articles(self):
  x=self.run_changed("q4")
  self.assertEqual(x["n"],16)
  self.assertEqual(x["correct"],7)
  self.assertAlmostEqual(x["mean_fullvocab_correct_token_NLL"],1.691991822618448)
 def test_real_single_1n8_original_C_source(self):
  x=self.run_changed("down_proj_10")
  self.assertEqual(x["n8_count"],1)
  self.assertEqual(x["q4_count"],195)
  self.assertAlmostEqual(x["mean_fullvocab_correct_token_NLL"],1.5543751034674018)
 def test_real_doubleton_2n8_C_kernel(self):
  x=self.run_changed("down_proj_10__v_proj_20")
  self.assertEqual(x["n8_count"],2)
  self.assertEqual(x["q4_count"],194)
  self.assertEqual(x["n"],16)
 def test_not_same_original_C_0_1_2n8_binaries(self):
  self.assertEqual(len({sha(p.read_bytes()) for p in a.BINARY.values()}),3)
  a.check_binary_source()
 def test_tampered_native_process_sha_rejected(self):
  blob=self.logs["q4"].replace(b"[qv5-choice] window 0 NLL ",b"[qv5-choice] window 0 NLL 999",1)
  with self.assertRaises(RuntimeError):self.run_changed("q4",blob,False)
 def test_quantized_pair_has_wrong_role_count_forged_sha(self):
  before=b"projection_role_tensors: f32=0 n8=2 q4=194"
  after=b"projection_role_tensors: f32=0 n8=1 q4=195"
  blob=self.logs["down_proj_10__v_proj_20"].replace(before,after,1)
  self.assertNotEqual(blob,self.logs["down_proj_10__v_proj_20"])
  with self.assertRaises(RuntimeError):self.run_changed("down_proj_10__v_proj_20",blob)
 def test_quantized_pair_wrong_mask_tensors_rejected(self):
  with self.assertRaises(RuntimeError):
   self.run_changed("down_proj_10__v_proj_20",newmask=b"down_proj 2\nv_proj 20\n")
 def test_quantized_pair_wrong_mask_digest_rejected(self):
  obj=json.loads(json.dumps(self.manifest))
  obj["true_native_role_masks_SHA256"]["down_proj_10__v_proj_20"]="0"*64
  with self.assertRaises(RuntimeError):
   self.run_changed("down_proj_10__v_proj_20",new_manifest=obj)
 def test_missing_one_real_article_window_rejected(self):
  rows=self.logs["q4"].decode().splitlines()
  changed="\n".join(x for x in rows if not x.startswith("[qv5-choice] window 5 "))+"\n"
  with self.assertRaises(RuntimeError):self.run_changed("q4",changed)
 def test_duplicate_window_rejected(self):
  blob=self.logs["q4"].replace(b"[qv5-choice] window 2 ",b"[qv5-choice] window 1 ",1)
  with self.assertRaises(RuntimeError):self.run_changed("q4",blob)
 def test_wrong_ground_truth_letter_rejected(self):
  import re
  text=self.logs["q4"].decode()
  changed=re.sub(r"(\[qv5-choice\] window 0 NLL [^\n]*? expected )(\d+)",
    lambda m:m.group(1)+str((int(m.group(2))+1)%4),text,count=1)
  self.assertNotEqual(text,changed)
  with self.assertRaises(RuntimeError):self.run_changed("q4",changed)
 def test_wrong_original_prompt_length_rejected(self):
  import re
  text=self.logs["q4"].decode()
  changed=re.sub(r"(\[qv5-choice\] window 0 NLL [^\n]*? prefix )(\d+)",
    lambda m:m.group(1)+str(int(m.group(2))+2),text,count=1)
  self.assertNotEqual(text,changed)
  with self.assertRaises(RuntimeError):self.run_changed("q4",changed)
 def test_NLL_nan_rejected_even_forged_sha(self):
  import re
  text=self.logs["q4"].decode()
  changed=re.sub(r"(\[qv5-choice\] window 0 NLL )([0-9.eE+-]+)",r"\1nan",text,count=1)
  with self.assertRaises(RuntimeError):self.run_changed("q4",changed)
 def test_negative_native_NLL_rejected_even_forged_sha(self):
  import re
  text=self.logs["q4"].decode()
  changed=re.sub(r"(\[qv5-choice\] window 1 NLL )([0-9.eE+-]+)",r"\1-0.123",text,count=1)
  with self.assertRaises(RuntimeError):self.run_changed("q4",changed)
 def test_wrong_native_16_article_total_rejected(self):
  blob=self.logs["q4"].replace(b"qv5_choice NLL ",b"qv5_choice NLL 100",1)
  with self.assertRaises(RuntimeError):self.run_changed("q4",blob)
 def test_source_article_order_tampered_rejected(self):
  source=list(self.source);source[0],source[1]=source[1],source[0]
  with self.assertRaises(RuntimeError):self.run_changed("q4",new_source=source)
 def test_preexisting_16_articles_not_reused(self):
  old=json.loads(self.read(a.PRIOR))["real_question_source"]
  self.assertFalse({q["article_original_id"] for q in old}&{q["article_original_id"] for q in self.source})
 def test_interaction_zero_if_two_N8_effects_add_exactly(self):
  q4=1.0;i=.8;j=.9
  self.assertAlmostEqual(.7-i-j+q4,0)
  self.assertLess(.65-i-j+q4,0)
  self.assertGreater(.75-i-j+q4,0)
if __name__=="__main__":unittest.main(verbosity=2)
