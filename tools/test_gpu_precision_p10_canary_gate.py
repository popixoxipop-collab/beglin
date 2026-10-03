#!/usr/bin/env python3
import unittest
import precision_context as pc
import precision_p10_canary_gate as p10
A=[{"role":"shared_up_proj","layer":3,"n":6},{"role":"shared_down_proj","layer":26,"n":5}]
B=[{"role":"shared_up_proj","layer":3,"n":5},{"role":"shared_down_proj","layer":26,"n":5}]
def bundle():
 return {"status":"MANUAL_REVIEW_CANDIDATE","automatic_live_promotion":False,"production_write_allowed":False,
 "proposal_id":"abc","target":{"role":"shared_up_proj","layer":3,"old_n":6,"new_n":5},
 "proposed_policy":B,"proposed_policy_hash":pc.policy_hash(B)}
class T(unittest.TestCase):
 def test_materialize_stays_dry_run(self):
  got=p10.materialize_canary_template(p9_bundle=bundle(),baseline_policy=A,source_commit="s",
   binary_sha256="a"*64,checkpoint_sha256="b"*64,g4_run_id="g4",g4_sha256="c"*64,g6_run_id="g6",g6_sha256="d"*64)
  self.assertEqual(got["status"],"READY_FOR_RUNTIME_PREIMAGE_MATERIALIZATION")
  self.assertFalse(got["production_cutover_allowed"])
  self.assertEqual(got["manual_canary_template"]["single_target"]["before_n"],6)
 def test_auto_promotion_bundle_rejected(self):
  b=bundle(); b["automatic_live_promotion"]=True
  with self.assertRaises(p10.P10Error):
   p10.materialize_canary_template(p9_bundle=b,baseline_policy=A,source_commit="s",binary_sha256="a"*64,
    checkpoint_sha256="b"*64,g4_run_id="g4",g4_sha256="c"*64,g6_run_id="g6",g6_sha256="d"*64)
 def test_final_gate_requires_both_pass_and_rollback_drill(self):
  got=p10.final_gate(p9_bundle=bundle(),canary_pass={"state":"CANARY_PASS_REVIEW_REQUIRED"},
                     rollback_drill={"state":"ROLLBACK_VERIFIED"})
  self.assertEqual(got["status"],"AWAITING_TRUSTED_PRODUCTION_APPROVAL")
  self.assertFalse(got["production_cutover_allowed"])
  with self.assertRaises(p10.P10Error):
   p10.final_gate(p9_bundle=bundle(),canary_pass={"state":"CANARY_PASS_REVIEW_REQUIRED"},
                  rollback_drill={"state":"ROLLBACK_REQUIRED"})
if __name__=="__main__": unittest.main()
