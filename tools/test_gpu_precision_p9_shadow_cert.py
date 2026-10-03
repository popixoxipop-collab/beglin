#!/usr/bin/env python3
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import precision_context as pc
import precision_p9_shadow_cert as p9

def proposal():
 return {"schema":"beglin-precision-policy-refresh-v1","status":"PROPOSAL_ONLY","production_write_allowed":False,
 "automatic_live_promotion":False,"proposal_id":"abc123","lineage_admissions":600,
 "selected_shadow_target":{"role":"shared_up_proj","layer":3,"old_n":6,"new_n":5,
 "allocator_target_sha256":"a"*64},"proposed_policy":[{"role":"shared_up_proj","layer":3,"n":5}],
 "proposed_policy_hash":pc.policy_hash([{"role":"shared_up_proj","layer":3,"n":5}])}

class P9Tests(unittest.TestCase):
 def test_missing_provenance_waits(self):
  got=p9.queue_item(proposal=proposal(),provenance=None)
  self.assertEqual(got["status"],"WAITING_FOR_REPLAY_PROVENANCE")
  self.assertFalse(got["automatic_live_promotion"])

 def test_provenance_sha_mismatch_fails(self):
  with tempfile.TemporaryDirectory() as td:
   raw=Path(td)/"p.i32"; raw.write_bytes(b"1234"); man=Path(td)/"m"; man.write_text(f"{raw} 10\n")
   with self.assertRaises(p9.P9Error):
    p9.queue_item(proposal=proposal(),provenance={"schema":"beglin-replay-provenance-v1",
     "production_write_allowed":False,"raw_token_file":str(raw),"manifest":str(man),
     "raw_token_sha256":"0"*64,"max_new_tokens":10,"prompt_len":1})

 def test_unanimous_shadow_only_certifies_manual_review(self):
  with tempfile.TemporaryDirectory() as td:
   raw=Path(td)/"p.i32"; raw.write_bytes(b"1234")
   import hashlib; sha=hashlib.sha256(raw.read_bytes()).hexdigest()
   man=Path(td)/"m"; man.write_text(f"{raw} 10\n")
   q=p9.queue_item(proposal=proposal(),provenance={"schema":"beglin-replay-provenance-v1",
    "production_write_allowed":False,"raw_token_file":str(raw),"manifest":str(man),
    "raw_token_sha256":sha,"max_new_tokens":10,"prompt_len":1})
   spec={"candidate_id":"p8-abc123","role":"shared_up_proj","layer":3,"n":5}
   with patch.object(p9.shadow,"run_shadow",side_effect=[
    {"shadow_status":"SHADOW_ADMITTED","production_write_allowed":False,"result_sha256":"1"*64},
    {"shadow_status":"SHADOW_ADMITTED","production_write_allowed":False,"result_sha256":"2"*64},
    {"shadow_status":"SHADOW_ADMITTED","production_write_allowed":False,"result_sha256":"3"*64},
   ]):
    repeated=p9.run_repeated_shadow(queue=q,spec=spec,shadow_root="/tmp/s",autopilot="/tmp/a",repeats=3)
   cert=p9.certification_bundle(proposal=proposal(),repeated_shadow=repeated)
   self.assertEqual(cert["status"],"MANUAL_REVIEW_CANDIDATE")
   self.assertFalse(cert["automatic_live_promotion"])
   self.assertEqual(cert["p9_repeats"],3)

 def test_one_reject_blocks_certification(self):
  with tempfile.TemporaryDirectory() as td:
   raw=Path(td)/"p.i32"; raw.write_bytes(b"1234")
   import hashlib; sha=hashlib.sha256(raw.read_bytes()).hexdigest()
   man=Path(td)/"m"; man.write_text(f"{raw} 10\n")
   q=p9.queue_item(proposal=proposal(),provenance={"schema":"beglin-replay-provenance-v1",
    "production_write_allowed":False,"raw_token_file":str(raw),"manifest":str(man),
    "raw_token_sha256":sha,"max_new_tokens":10,"prompt_len":1})
   with patch.object(p9.shadow,"run_shadow",side_effect=[
    {"shadow_status":"SHADOW_ADMITTED","production_write_allowed":False},
    {"shadow_status":"SHADOW_REJECTED","production_write_allowed":False},
   ]):
    repeated=p9.run_repeated_shadow(queue=q,spec={},shadow_root="/tmp/s",autopilot="/tmp/a",repeats=2)
   with self.assertRaises(p9.P9Error): p9.certification_bundle(proposal=proposal(),repeated_shadow=repeated)

if __name__=="__main__": unittest.main()
