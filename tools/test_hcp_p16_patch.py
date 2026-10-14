#!/usr/bin/env python3
"""P16 C observer source integrity and research-only first prefill capture guards."""
from __future__ import annotations
import hashlib,pathlib,unittest
from hcp_p5_scratch_capacity128 import patch as p5
from hcp_p16_native_logit_patch import (
 patch,strip,HELPER,NEW,INCLUDES,ANCHOR,START,END,C128_SHA,
 CAPTURE_DIR,PROMPT0_SHA,PROMPT1_SHA,MODEL_VOCAB
)
REPO=pathlib.Path(__file__).resolve().parents[1]
class P16ObserverTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cb128=p5((REPO/"qwen_infer.c").read_text()).encode()
  cls.changed=patch(cls.cb128)
 def test_01_exact_original_p5_CB128(self):self.assertEqual(hashlib.sha256(self.cb128).hexdigest(),C128_SHA)
 def test_02_full_byte_reversibility(self):self.assertEqual(strip(self.changed),self.cb128)
 def test_03_reject_duplicate_patch(self):
  with self.assertRaises(ValueError):patch(self.changed)
 def test_04_reject_unpinned_original_C32(self):
  with self.assertRaises(ValueError):patch((REPO/"qwen_infer.c").read_bytes())
 def test_05_only_matched_last_prefill_logits_source(self):
  t=self.changed.decode();body=t[t.index(START):t.index(END,t.index(START))]
  self.assertEqual(body.count('p16_write_native_logits('),1)
  self.assertLess(body.index('p16_write_native_logits('),body.index('rq_out[r][rq_nout[r]++] = am; rq_t_first'))
 def test_06_require_original_prompt_token_sha(self):
  self.assertIn(PROMPT0_SHA,HELPER);self.assertIn(PROMPT1_SHA,HELPER)
  self.assertIn('P16_UNFROZEN_TOKENS',HELPER)
 def test_07_explicit_env_optin(self):
  self.assertIn('getenv("BEGLIN_P16_LOGITS_DIR")',NEW)
  self.assertIn('strcmp(dir,allowed_dir)!=0',HELPER)
  self.assertIn(CAPTURE_DIR,HELPER)
 def test_08_no_warmup_capture(self):
  self.assertIn('pass == 1 && persistent_mode',NEW)
 def test_09_exact_one_request_one_new_token(self):
  for part in ('p16_capture_count != 0','R != 1','rq_maxnew[r] != 1','rq_nout[r] != 0'):
   self.assertIn(part,NEW)
 def test_10_exact_last_position(self):
  self.assertIn('spos_arr[m] != rq_plen[r] - 1',NEW)
  self.assertIn('plen-1',HELPER)
 def test_11_exclusive_nofollow_output(self):
  self.assertIn('O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW',HELPER)
  self.assertIn('fsync(fd)',HELPER)
 def test_12_checksum_real_float32_and_token_ascii(self):
  self.assertIn('CC_SHA256(data',HELPER)
  self.assertIn('size=(size_t)MOE_VOCAB*sizeof(float)',HELPER)
  self.assertIn('for(int i=0;i<MOE_VOCAB;i++)',HELPER)
  self.assertEqual(MODEL_VOCAB,102400)
 def test_13_bytewise_math_unchanged(self):
  self.assertEqual(strip(self.changed),self.cb128)
  self.assertIn('mlx_gpu_cbatch_forward_finalize(w_finalnorm, gpu_logits)',self.changed.decode())
 def test_14_reject_nonfinite(self):self.assertIn('P16_LOGIT_NONFINITE',HELPER)
 def test_15_claims_invalid_scope_die(self):
  self.assertIn('P16_CAPTURE_SCOPE_INVALID',HELPER)
  self.assertIn('_exit(117)',HELPER)

if __name__=="__main__":unittest.main(verbosity=2)
