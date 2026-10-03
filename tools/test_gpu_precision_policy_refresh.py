#!/usr/bin/env python3
import unittest
import precision_policy_refresh as p8

CURRENT=[{"role":"shared_up_proj","layer":3,"n":6},{"role":"shared_down_proj","layer":26,"n":5}]
def c(role,layer,n,feasible=True):
 return {"role":role,"layer":layer,"n":n,"feasible":feasible,"conditionally_feasible":False,
 "conditional_recoveries":[],"blocked_reasons":[] if feasible else ["FAIL"],"real_pass_events":10,
 "real_fail_events":0,"real_event_count":10,"real_pass_event_keys":[],"real_fail_event_keys":[],
 "g4_context_hash":f"g4-{role}-{layer}-{n}","g6_context_hash":f"g6-{role}-{layer}-{n}",
 "tensor_count":1,"numel":1000,"effective_bpw":float(n),"estimated_bytes":float(n*1000),
 "persistent_p50_ms":None,"persistent_p95_ms":None,"persistent_rss_bytes":None,"benchmark_pid":None}
CANDS=[c("shared_up_proj",3,5),c("shared_up_proj",3,6),c("shared_down_proj",26,5),c("shared_down_proj",26,6)]
SUMMARY={"admissions":600,"finite_logits_rate":1.0,"lineage_head_sha256":"a"*64}
P7={"status":"PASS","production_touched":False,"result_sha256":"b"*64}
class T(unittest.TestCase):
 def test_proposal_is_one_shadow_target_and_no_live_write(self):
  x=p8.propose(candidates=CANDS,current_policy=CURRENT,lineage_summary=SUMMARY,p7_certification=P7)
  self.assertEqual(x["status"],"PROPOSAL_ONLY"); self.assertFalse(x["production_write_allowed"])
  self.assertFalse(x["automatic_live_promotion"]); self.assertTrue(x["manual_review_required"])
  self.assertEqual(x["selected_shadow_target"]["role"],"shared_up_proj")
  self.assertEqual(x["selected_shadow_target"]["new_n"],5)
 def test_insufficient_soak_fails(self):
  with self.assertRaises(p8.PolicyRefreshError):
   p8.propose(candidates=CANDS,current_policy=CURRENT,lineage_summary={**SUMMARY,"admissions":60},p7_certification=P7)
 def test_shadow_rejection_cannot_certify(self):
  x=p8.propose(candidates=CANDS,current_policy=CURRENT,lineage_summary=SUMMARY,p7_certification=P7)
  with self.assertRaises(p8.PolicyRefreshError):
   p8.certification_candidate(proposal=x,shadow_result={"shadow_status":"SHADOW_REJECTED","production_write_allowed":False})
 def test_admitted_shadow_becomes_manual_review_only(self):
  x=p8.propose(candidates=CANDS,current_policy=CURRENT,lineage_summary=SUMMARY,p7_certification=P7)
  t=x["selected_shadow_target"]
  y=p8.certification_candidate(proposal=x,shadow_result={"shadow_status":"SHADOW_ADMITTED","production_write_allowed":False,
      "candidate":{"role":t["role"],"layer":t["layer"],"n":t["new_n"]},"result_sha256":"c"*64})
  self.assertEqual(y["status"],"MANUAL_REVIEW_CANDIDATE"); self.assertFalse(y["automatic_live_promotion"])
if __name__=="__main__": unittest.main()
