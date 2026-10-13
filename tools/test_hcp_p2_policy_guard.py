#!/usr/bin/env python3
"""P2 predicts proxy risk only; original Beglin evidence gates own action authority."""
from __future__ import annotations
import copy,unittest
import precision_context as pc
from hcp_p2_policy_guard import propose

BASE=[{"role":"shared_down_proj","layer":26,"n":5},
      {"role":"shared_up_proj","layer":3,"n":6}]
ALT=[{"role":"shared_down_proj","layer":26,"n":6},
     {"role":"shared_up_proj","layer":3,"n":5}]
CONTEXT="a"*64
FORECAST={"source":"REAL_QUANTIZED_AF_H2_TOP6_ROUTING","forecast_risk_proxy":.95}

def bundle():
    allocation={"schema":"beglin-precision-allocator-v1",
        "status":"PROPOSAL_ONLY","production_write_allowed":False,
        "current_policy":BASE,"proposed_policy":ALT,
        "pairwise_evidence_required_before_combined_production":True,
        "targets":[
            {"role":"shared_down_proj","layer":26,"status":"PROPOSED","selected_n":6},
            {"role":"shared_up_proj","layer":3,"status":"PROPOSED","selected_n":5},
        ]}
    selector={"schema":"beglin-dynamic-precision-selector-v1",
        "status":"DECISION_ONLY","production_write_allowed":False,
        "nonmonotonic_precision":True,"signal":{"active_triggers":[]},
        "targets":[
            {"role":"shared_down_proj","layer":26,"selected_n":6,"status":"BASE_LOW_COST"},
            {"role":"shared_up_proj","layer":3,"selected_n":5,"status":"BASE_LOW_COST"},
        ]}
    certified={"policy":ALT,"policy_hash":pc.policy_hash(ALT),
        "context_hash":CONTEXT,"evidence_sha256":"b"*64,
        "status":"PASS","pass":True,"production_touched":False}
    return allocation,selector,[certified]

class PolicyEvidenceSafetyTests(unittest.TestCase):
    def call(self,**overrides):
        a,s,c=bundle()
        vals={"current_policy":BASE,"context_hash":CONTEXT,
            "learned_prediction":FORECAST,
            "allocation":a,"selection":s,"combined_evidence":c}
        vals.update(overrides)
        return propose(**vals)
    def test_no_authoritative_evidence_holds_even_high_forecast(self):
        v=self.call(combined_evidence=[])
        self.assertEqual(v["status"],"HOLD_UNCERTIFIED")
        self.assertEqual(v["selected_policy_hash"],pc.policy_hash(BASE))
        self.assertFalse(v["native_policy_change_allowed"])
    def test_wrong_execution_context_cannot_reuse_old_cert(self):
        v=self.call(context_hash="c"*64)
        self.assertEqual(v["status"],"HOLD_UNCERTIFIED")
    def test_context_bound_certificate_makes_shadow_proposal_only(self):
        v=self.call()
        self.assertEqual(v["status"],"SHADOW_CERTIFIED_PROPOSAL_ONLY")
        self.assertEqual(v["selected_policy_hash"],pc.policy_hash(ALT))
        self.assertFalse(v["production_write_allowed"])
        self.assertFalse(v["native_policy_change_allowed"])
    def test_wrong_surrogate_source_cannot_propose(self):
        v=self.call(learned_prediction={"source":"P7_OPERATOR_INJECTED","forecast_risk_proxy":1.})
        self.assertEqual(v["status"],"HOLD_UNCERTIFIED")
    def test_triggered_change_requires_exact_trigger_evidence(self):
        a,s,c=bundle()
        # Real trigger alternates must differ from their allocator's base n.
        a["proposed_policy"]=BASE
        a["targets"][0]["selected_n"]=5
        a["targets"][0]["dynamic_escalation"]={"candidate_alternates":[{"n":6}]}
        a["targets"][1]["selected_n"]=6
        s["targets"][1]["selected_n"]=6
        target=[{"role":"shared_down_proj","layer":26,"n":6},
                {"role":"shared_up_proj","layer":3,"n":6}]
        c[0]["policy"]=target
        c[0]["policy_hash"]=pc.policy_hash(target)
        s["signal"]["active_triggers"]=["low_margin"]
        s["targets"][0]["status"]="TRIGGER_CONDITIONED_ALTERNATE"
        s["targets"][0]["evidence"]={"status":"PASS","pass":True,
            "context_hash":"c"*64,"trigger_type":"low_margin"}
        v=self.call(allocation=a,selection=s,combined_evidence=c)
        self.assertEqual(v["reason"],"MISSING_EXACT_CONTEXT_TRIGGER_EVIDENCE")
        s["targets"][0]["evidence"]["context_hash"]=CONTEXT
        v=self.call(allocation=a,selection=s,combined_evidence=c)
        self.assertEqual(v["status"],"SHADOW_CERTIFIED_PROPOSAL_ONLY")
    def test_wrong_policy_shape_or_nonmonotonic_contract_holds(self):
        a,s,c=bundle()
        s["nonmonotonic_precision"]=False
        v=self.call(allocation=a,selection=s,combined_evidence=c)
        self.assertEqual(v["status"],"HOLD_UNCERTIFIED")
        a,s,c=bundle()
        a["proposed_policy"]=a["proposed_policy"][:-1]
        v=self.call(allocation=a,selection=s,combined_evidence=c)
        self.assertEqual(v["status"],"HOLD_UNCERTIFIED")

if __name__=="__main__":unittest.main()
