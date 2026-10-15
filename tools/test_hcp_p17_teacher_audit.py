#!/usr/bin/env python3
"""P17 genuine 12-whole-model BF16 teacher full-logit negative evidence tests."""
from __future__ import annotations
import json,pathlib,shutil,tempfile,unittest
import numpy as np
from hcp_p17_teacher_audit import audit
REPO=pathlib.Path(__file__).resolve().parents[1]
ORIG=REPO.parent/"private_original_teacher"
FROZEN=REPO/"experiments/hcp_p17/P17_FROZEN_TOKENS.json"
class P17TeacherFullEvidenceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not (ORIG/"p17-00.bf16_logits.npy").is_file() or not (ORIG/"p17-11.json").is_file():
   raise RuntimeError("P17 genuine 12-teacher original sources missing")
 def setUp(self):
  self.t=tempfile.TemporaryDirectory()
  self.root=pathlib.Path(self.t.name)
  self.docs=self.root/"documents";shutil.copytree(ORIG,self.docs)
  self.frozen=self.root/"FROZEN_TOKENS.json";shutil.copyfile(FROZEN,self.frozen)
 def tearDown(self):self.t.cleanup()
 def validate(self):return audit(self.frozen,self.docs)
 def reject(self):
  with self.assertRaises((ValueError,KeyError,RuntimeError,IndexError,TypeError,AssertionError)):self.validate()
 def edit(self,name,change):
  p=self.docs/name;x=json.loads(p.read_text());change(x)
  p.write_text(json.dumps(x,sort_keys=True)+"\n")
 def test_01_real_genuine_original_BF16_12_passes(self):
  z=self.validate()
  self.assertEqual(z["source_tokens_gold"],31)
  self.assertEqual(z["question_count"],12)
  self.assertTrue(z["all_12_source_checkpoint_tensor_value_hashes_equal"])
  self.assertFalse(z["noninferiority_certified"])
 def test_02_missing_original_whole_model_report_refused(self):
  (self.docs/"p17-11.json").unlink();self.reject()
 def test_03_forged_26_of_27_layers_refused(self):
  self.edit("p17-02.json",lambda x:x.update(decoder_layers_forwarded=26));self.reject()
 def test_04_missing_5291_source_tensors_refused(self):
  self.edit("p17-02.json",lambda x:x.update(original_source_tensors_exactly_once=5290));self.reject()
 def test_05_wrong_full_vocabulary_shape_refused(self):
  self.edit("p17-00.json",lambda x:x.update(logits_shape=[2,3000]));self.reject()
 def test_06_logit_file_byte_tampered_refused(self):
  p=self.docs/"p17-01.bf16_logits.npy";b=np.load(p,allow_pickle=False);b[0,0]+=1
  with p.open("wb") as f:np.save(f,b,allow_pickle=False)
  self.reject()
 def test_07_original_full_logit_nonfinite_refused(self):
  p=self.docs/"p17-03.bf16_logits.npy";b=np.load(p,allow_pickle=False);b[0,1]=np.nan
  with p.open("wb") as f:np.save(f,b,allow_pickle=False)
  self.reject()
 def test_08_forged_teacher_NLL_rejected(self):
  self.edit("p17-04.json",lambda x:x["gold_nll_nats"].__setitem__(0,33.0));self.reject()
 def test_09_original_model_source_index_not_unchanged_refused(self):
  self.edit("p17-05.json",lambda x:x.update(source_checkpoint_index_sha256="a"*64));self.reject()
 def test_10_wrong_original_gold_answer_token_sha_refused(self):
  self.edit("p17-07.json",lambda x:x.update(gold_answer_token_sha256="b"*64));self.reject()
 def test_11_missing_5291_hash_manifest_refused(self):
  (self.docs/"p17-09.source_hashes.json").unlink();self.reject()
 def test_12_modified_original_5291_tensor_sha_refused(self):
  p=self.docs/"p17-06.source_hashes.json";z=json.loads(p.read_text());key=next(iter(z));z[key]="c"*64
  p.write_text(json.dumps(z)+"\n");self.reject()
 def test_13_untrusted_input_bank_amendment_refused(self):
  p=self.frozen;j=json.loads(p.read_text());j["items"][0]["gold_answer_ids"].append(5)
  p.write_text(json.dumps(j)+"\n");self.reject()
 def test_14_fake_B_policy_noninferiority_claim_refused(self):
  self.edit("p17-10.json",lambda x:x.update(end_to_end_noninferiority_certified=True));self.reject()
 def test_15_original_GPU_used_claim_refused(self):
  self.edit("p17-08.json",lambda x:x.update(GPU_used=True));self.reject()
 def test_16_spoofed_live_production_PID_refused(self):
  self.edit("p17-11.json",lambda x:x["production_after"].update(pids=[0]));self.reject()
if __name__=="__main__":unittest.main(verbosity=2)
