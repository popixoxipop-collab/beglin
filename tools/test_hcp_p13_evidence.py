#!/usr/bin/env python3
"""P13 independent native bytes / BF16 source evidence tamper regression."""
from __future__ import annotations
import hashlib,json,pathlib,shutil,sys,tempfile,unittest
from hcp_p13_independent_audit import audit
from hcp_p13_pack_sha_patch import patch,strip,SOURCE_SHA
from hcp_p5_scratch_capacity128 import patch as cb128

REPO=pathlib.Path(__file__).resolve().parents[1]
E=REPO/"experiments/hcp_p13/evidence/native_xox_20261014"
FROZEN=REPO/"experiments/hcp_p13/P13_REFERENCE_CODES.json"
QUALITY=REPO/"experiments/hcp_p6/evidence/quality.json"
class P13TraceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.original=cb128((REPO/"qwen_infer.c").read_text()).encode()
  cls.patched=patch(cls.original)
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory()
  self.root=pathlib.Path(self.tmp.name)
  for a,b in ((E/"real_native_pack_sha_v2.json","result.json"),
              (E/"preflight.json","preflight.json"),
              (FROZEN,"ref.json"),
              (QUALITY,"quality.json")):
   shutil.copyfile(a,self.root/b)
 def tearDown(self):self.tmp.cleanup()
 def attempt(self):
  return audit(self.root/"result.json",self.root/"ref.json",self.root/"preflight.json",self.root/"quality.json")
 def mutate(self,name,fn):
  p=self.root/name
  j=json.loads(p.read_text());fn(j)
  p.write_text(json.dumps(j,sort_keys=True)+"\n")
 def rejected(self):
  with self.assertRaises((ValueError,KeyError,AssertionError,RuntimeError)):
   self.attempt()
 def test_01_exact_cb128_source_only(self):
  self.assertEqual(hashlib.sha256(self.original).hexdigest(),SOURCE_SHA)
 def test_02_C_SHA_annotation_perfectly_reversible(self):
  self.assertEqual(strip(self.patched),self.original)
 def test_03_double_source_annotation_denied(self):
  with self.assertRaises(ValueError):patch(self.patched)
 def test_04_annotation_default_off(self):
  text=self.patched.decode()
  self.assertIn('strcmp(enabled,"1")!=0',text)
  self.assertIn('"P13_NATIVE_PACKED_SHA ',text)
  self.assertEqual(text.count("CC_SHA256(data,"),1)
 def test_05_full_native_paired_BF16_output_valid(self):
  d=self.attempt()
  self.assertTrue(d["frozen_original_BF16_4_representations_byte_equal"])
  self.assertEqual(d["independent_original_P6_generated_SHA_matches"],48)
 def test_06_wrong_packed_code_SHA_denied(self):
  self.mutate("result.json",
   lambda d:d["native_hash_log_events"][0].update(packed_SHA256="a"*64))
  self.rejected()
 def test_07_wrong_scale_code_SHA_denied(self):
  self.mutate("result.json",
   lambda d:d["native_hash_log_events"][1].update(scales_SHA256="b"*64))
  self.rejected()
 def test_08_missing_native_trace_denied(self):
  self.mutate("result.json",
   lambda d:d["native_hash_log_events"].pop())
  self.rejected()
 def test_09_generated_sequence_changed_denied(self):
  self.mutate("result.json",
   lambda d:d["generation_rows"][0].update(generated_sequence_sha256="c"*64))
  self.rejected()
 def test_10_fake_terminal_ACK_denied(self):
  self.mutate("result.json",
   lambda d:d["generation_rows"][2].update(native_ACK_verified=False))
  self.rejected()
 def test_11_wrong_binary_SHA_denied(self):
  self.mutate("result.json",lambda d:d.update(native_traced_binary_SHA256="d"*64))
  self.rejected()
 def test_12_frozen_reference_mutation_denied(self):
  self.mutate("ref.json",
   lambda d:d["replacements"][0].update(packed_SHA256="e"*64))
  self.rejected()
 def test_13_fake_full_BF16_teacher_claim_denied(self):
  self.mutate("result.json",
   lambda d:d.update(bf16_full_model_logits_not_measured=False))
  self.rejected()
 def test_14_production_stage_changed_denied(self):
  self.mutate("result.json",
   lambda d:d["production_after"].update(stage_sha256="f"*64))
  self.rejected()
 def test_15_missing_mode_admission_denied(self):
  self.mutate("result.json",lambda d:d["generation_rows"].pop())
  self.rejected()
 def test_16_input_domain_mismatch_denied(self):
  self.mutate("result.json",
   lambda d:d["generation_rows"][0].update(domain="untrusted"))
  self.rejected()

if __name__=="__main__":unittest.main()
