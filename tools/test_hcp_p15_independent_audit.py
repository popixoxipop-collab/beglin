#!/usr/bin/env python3
"""P15 independent original full-logits forensic audit: controlled tamper suite.

Requires EOE-private genuine P15 BF16 teacher artifacts; never ships raw logits
or 5,291 source-tensor hashes in GitHub. No native worker, network, or model load.
"""
import hashlib,json,pathlib,shutil,tempfile,unittest
import numpy as np
from hcp_p15_independent_audit import audit

REPO=pathlib.Path(__file__).resolve().parents[1]
PRIVATE=REPO.parent/"private_full_teacher_evidence"
LABEL="prompt0_bf16"
INPUTS=(LABEL+".json",LABEL+".logits.npy",LABEL+".source_hashes.json",
 "SELFTEST.json","SELFTEST.log","FROZEN_INPUTS.json","SOURCE_MODEL_INDEX.json")

class P15IndependentTamperTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not PRIVATE.is_dir() or any(not (PRIVATE/f).is_file() for f in INPUTS):
   raise unittest.SkipTest("original private XOX full-logits witness is not on this host")
 def setUp(self):
  self.t=tempfile.TemporaryDirectory()
  self.root=pathlib.Path(self.t.name)
  for name in INPUTS:shutil.copyfile(PRIVATE/name,self.root/name)
 def tearDown(self):self.t.cleanup()
 def attempt(self):return audit(self.root,LABEL)
 def rejects(self):
  with self.assertRaises((ValueError,RuntimeError,KeyError,TypeError,AssertionError)):self.attempt()
 def edit(self,name,fn):
  p=self.root/name;j=json.loads(p.read_text());fn(j);p.write_text(json.dumps(j,sort_keys=True)+"\n")
 def test_01_actual_complete_original_full_vocabulary_passes(self):
  r=self.attempt()
  self.assertEqual(r["layers"],27)
  self.assertEqual(r["source_tensor_count"],5291)
  self.assertEqual(r["vocabulary_logits"],102400)
  self.assertFalse(r["promotion_allowed"])
 def test_02_false_model_completed(self):
  self.edit(LABEL+".json",lambda x:x.update(full_model_forward=False));self.rejects()
 def test_03_26_instead_of_27_layers(self):
  self.edit(LABEL+".json",lambda x:x.update(all_original_decoder_layers=26));self.rejects()
 def test_04_out_of_order_layer_records(self):
  self.edit(LABEL+".json",lambda x:x["per_layer"].reverse());self.rejects()
 def test_05_missing_last_checkpoint_parameter(self):
  self.edit(LABEL+".source_hashes.json",lambda x:x.pop(next(iter(x))));self.rejects()
 def test_06_wrong_number_of_checkpoint_tensors(self):
  self.edit(LABEL+".json",lambda x:x.update(all_original_checkpoint_tensors_read=5290));self.rejects()
 def test_07_exported_logits_file_tampered(self):
  p=self.root/(LABEL+".logits.npy");a=np.load(p,allow_pickle=False)
  a[0,0]+=1
  with p.open("wb") as f:np.save(f,a,allow_pickle=False)
  self.rejects()
 def test_08_logits_not_finite(self):
  p=self.root/(LABEL+".logits.npy");a=np.load(p,allow_pickle=False)
  a[0,1]=np.nan
  with p.open("wb") as f:np.save(f,a,allow_pickle=False)
  self.rejects()
 def test_09_wrong_logits_shape(self):
  self.edit(LABEL+".json",lambda x:x.update(logits_shape=[102400]));self.rejects()
 def test_10_fake_vocabulary_size(self):
  self.edit(LABEL+".json",lambda x:x.update(full_vocabulary_size=8192));self.rejects()
 def test_11_source_implementation_changed(self):
  self.edit(LABEL+".json",lambda x:x.update(original_modeling_source_sha256="0"*64));self.rejects()
 def test_12_wrong_model_index(self):
  p=self.root/"SOURCE_MODEL_INDEX.json"
  p.write_bytes(p.read_bytes()+b"\n");self.rejects()
 def test_13_wrong_frozen_inputs(self):
  self.edit("FROZEN_INPUTS.json",lambda x:x["entries"][0].update(text="forged"));self.rejects()
 def test_14_bad_source_tensor_hash_receipt(self):
  self.edit(LABEL+".json",lambda x:x.update(source_tensor_hash_manifest_sha256="a"*64));self.rejects()
 def test_15_fake_quantized_A_B_accuracy_proven(self):
  self.edit(LABEL+".json",lambda x:x.update(quality_noninferiority_certified=True));self.rejects()
 def test_16_fake_promotion_approved(self):
  self.edit(LABEL+".json",lambda x:x.update(automatic_live_promotion=True));self.rejects()
 def test_17_fake_GPU_executed(self):
  self.edit(LABEL+".json",lambda x:x.update(GPU_used=True));self.rejects()
 def test_18_wrong_native_production_route(self):
  self.edit(LABEL+".json",lambda x:x["production_after"].update(route_sha256="f"*64));self.rejects()
 def test_19_forged_argmax_token(self):
  self.edit(LABEL+".json",lambda x:x.update(next_token_id=101999));self.rejects()
 def test_20_forged_reported_logits_norm(self):
  self.edit(LABEL+".json",lambda x:x.update(logits_l2=0.0));self.rejects()
 def test_21_resource_limit_exceeded(self):
  self.edit(LABEL+".json",lambda x:x.update(process_rss_peak_bytes=9*1024**3));self.rejects()
 def test_22_selftest_failed(self):
  self.edit("SELFTEST.json",lambda x:x.update(successful=False));self.rejects()
 def test_23_unsupported_compute_dtype(self):
  self.edit(LABEL+".json",lambda x:x.update(execution_arithmetic="INT4"));self.rejects()
 def test_24_token_manifest_hash_mismatch(self):
  self.edit(LABEL+".json",lambda x:x.update(token_sha256="f"*64));self.rejects()

 def test_25_false_resident_full_model_parity_claim(self):
  self.edit(LABEL+".json",lambda x:x.update(original_resident_full_model_bitwise_comparison_performed=True));self.rejects()
 def test_26_original_parameters_modified_flag(self):
  self.edit(LABEL+".json",lambda x:x.update(source_parameters_modified=True));self.rejects()
 def test_27_original_source_checkpoint_modified_flag(self):
  self.edit(LABEL+".json",lambda x:x.update(source_checkpoint_metadata_unchanged=False));self.rejects()
 def test_28_native_KV_cache_wrongly_enabled(self):
  self.edit(LABEL+".json",lambda x:x.update(native_KV_cache_enabled=True));self.rejects()

if __name__=="__main__":unittest.main(verbosity=2)
