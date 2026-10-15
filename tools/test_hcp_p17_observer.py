#!/usr/bin/env python3
"""P17 exact 31-prefix source-instrumentation mathematical integrity tests."""
import hashlib,json,pathlib,unittest
from hcp_p5_scratch_capacity128 import patch as p5
from hcp_p17_observer_patch import patch,strip,allowlist,FROZEN_SHA,C128_SHA,DIR,NEW_LOG
R=pathlib.Path(__file__).resolve().parents[1]
FROZEN=R/"experiments/hcp_p17/P17_FROZEN_TOKENS.json"
class P17PatchedSource(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.raw=p5((R/"qwen_infer.c").read_text()).encode()
  cls.instrumented=patch(cls.raw,FROZEN)
 def test_01_exact_original_C128(self):
  self.assertEqual(hashlib.sha256(self.raw).hexdigest(),C128_SHA)
 def test_02_exact_inverse_to_C128(self):
  self.assertEqual(strip(self.instrumented,FROZEN),self.raw)
 def test_03_unapproved_C32_rejected(self):
  with self.assertRaises(ValueError):patch((R/"qwen_infer.c").read_bytes(),FROZEN)
 def test_04_double_instrumentation_refused(self):
  with self.assertRaises(ValueError):patch(self.instrumented,FROZEN)
 def test_05_freeze_sha_preexisting(self):
  self.assertEqual(hashlib.sha256(FROZEN.read_bytes()).hexdigest(),FROZEN_SHA)
 def test_06_exact_31_frozen_prefix_allowlist(self):
  j=json.loads(FROZEN.read_text())
  a=allowlist(FROZEN)
  self.assertEqual(len(j["items"]),12)
  self.assertEqual(sum(len(x["forced_prefixes"]) for x in j["items"]),31)
  self.assertEqual(a.count('"'+j["items"][0]["forced_prefixes"][0]["prefix_token_sha256"]+'"'),1)
 def test_07_only_bounded_KV128_and_generated_single_token(self):
  src=self.instrumented.decode()
  self.assertIn('rq_plen[r] < 20 || rq_plen[r] > 112',src)
  self.assertIn('rq_maxnew[r] != 1',src)
  self.assertIn('spos_arr[m] != rq_plen[r] - 1',src)
 def test_08_failclosed_unfrozen_token_sha(self):
  self.assertIn("p17_prefix_allowed(tok_hash)",self.instrumented.decode())
 def test_09_optin_only_and_no_worker_math_change(self):
  src=self.instrumented.decode()
  self.assertIn("BEGLIN_P17_LOGITS_DIR",src)
  self.assertIn("P17_NATIVE_GOLD_PREFIX_LOGITS",src)
  self.assertIn(DIR,src)
  self.assertEqual(strip(self.instrumented,FROZEN),self.raw)
 def test_10_exclusive_capture_and_fsync_preserved(self):
  s=self.instrumented.decode()
  self.assertIn('O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW',s)
  self.assertIn('fsync(fd)',s)
 def test_11_original_native_vocab_size(self):
  self.assertIn("MOE_VOCAB!=102400",self.instrumented.decode())
 def test_12_do_not_capture_warmup(self):
  self.assertIn("pass == 1 && persistent_mode",self.instrumented.decode())
if __name__=="__main__":unittest.main(verbosity=2)
