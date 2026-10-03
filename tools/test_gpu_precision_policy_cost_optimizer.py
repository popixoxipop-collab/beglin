#!/usr/bin/env python3
import unittest

import precision_context as pc
import precision_policy_cost_optimizer as pco

A=[
    {"role":"shared_down_proj","layer":26,"n":5},
    {"role":"shared_up_proj","layer":3,"n":6},
]
B=[
    {"role":"shared_down_proj","layer":26,"n":6},
    {"role":"shared_up_proj","layer":3,"n":5},
]
AH=pc.policy_hash(A); BH=pc.policy_hash(B)


def target(role,layer,base,alt):
    return {
        "role":role,"layer":layer,"status":"PROPOSED","selected_n":base,
        "selected":{"n":base,"estimated_bytes":100.0},
        "pareto_frontier":[
            {"n":base,"estimated_bytes":100.0},
            {"n":alt,"estimated_bytes":120.0},
        ],
        "dynamic_escalation":{"candidate_alternates":[
            {"n":alt,"real_pass_events":2}
        ]},
    }


ALLOCATION={
    "targets":[
        target("shared_down_proj",26,5,6),
        target("shared_up_proj",3,5,6),
    ]
}


def accepted():
    return [
        {"policy":A,"policy_hash":AH,"status":"PASS","pass":True,
         "production_touched":False,"evidence_sha256":"a"*64},
        {"policy":B,"policy_hash":BH,"status":"PASS","pass":True,
         "production_touched":False,"evidence_sha256":"b"*64},
    ]


def costs():
    return {
        "schema":"beglin-precision-e2e-cost-v1",
        "target_profiles":[
            {"role":"shared_down_proj","layer":26,"n":5,
             "steady_engine_p50_ms":10,"steady_roundtrip_p50_ms":12,
             "rss_bytes":1000,"transitions":[]},
        ],
        "accepted_policy_profiles":[
            {"policy":A,"policy_hash":AH,"steady_engine_p50_ms":10,
             "steady_roundtrip_p50_ms":12,"expected_inference_passes":1,
             "transitions":[
                 {"from_policy_hash":BH,"cache_state":"warm",
                  "p50_transition_ms":5,"cache_bytes_added":0}
             ]},
            {"policy":B,"policy_hash":BH,"steady_engine_p50_ms":11,
             "steady_roundtrip_p50_ms":13,"expected_inference_passes":1,
             "transitions":[
                 {"from_policy_hash":AH,"cache_state":"cold",
                  "p50_transition_ms":50,"cache_bytes_added":200},
                 {"from_policy_hash":AH,"cache_state":"warm",
                  "p50_transition_ms":4,"cache_bytes_added":0},
             ]},
        ],
    }
class PolicyCostTests(unittest.TestCase):
    def runtime(self,warm_b=False):
        rows=[
            {"role":"shared_down_proj","layer":26,"n":5,"bytes":100},
            {"role":"shared_up_proj","layer":3,"n":6,"bytes":100},
        ]
        if warm_b:
            rows += [
                {"role":"shared_down_proj","layer":26,"n":6,"bytes":100},
                {"role":"shared_up_proj","layer":3,"n":5,"bytes":100},
            ]
        return {"qng64_cache":rows,"resident_qng64_cache_bytes":len(rows)*100}

    def test_no_trigger_prefers_current_accepted_policy(self):
        selection={
            "signal":{"active_triggers":[]},
            "targets":[
                {"role":"shared_down_proj","layer":26,"selected_n":5,"status":"BASE_LOW_COST"},
                {"role":"shared_up_proj","layer":3,"selected_n":5,"status":"BASE_LOW_COST"},
            ],
        }
        got=pco.optimize_policy(
            current_policy=A,allocation=ALLOCATION,selection=selection,
            combined_policy_evidence=accepted(),cost_evidence=costs(),
            runtime_state=self.runtime(),e2e_weight=1.0,
        )
        self.assertEqual(got["selected_policy_hash"],AH)
        self.assertEqual(got["selected_cost"]["transition_p50_ms"],0.0)

    def test_active_trigger_pins_selector_precision(self):
        selection={
            "signal":{"active_triggers":["low_margin"]},
            "targets":[
                {"role":"shared_down_proj","layer":26,"selected_n":6,
                 "status":"TRIGGER_CONDITIONED_ALTERNATE"},
                {"role":"shared_up_proj","layer":3,"selected_n":5,
                 "status":"HOLD_BASE_NO_TRIGGER_EVIDENCE"},
            ],
        }
        got=pco.optimize_policy(
            current_policy=A,allocation=ALLOCATION,selection=selection,
            combined_policy_evidence=accepted(),cost_evidence=costs(),
            runtime_state=self.runtime(),e2e_weight=1.0,
        )
        self.assertEqual(got["selected_policy_hash"],BH)
        self.assertEqual(got["selected_cost"]["cache_state"],"cold")
        self.assertEqual(got["selected_cost"]["expected_e2e_ms"],63.0)
    def test_warm_cache_reduces_policy_transition_cost(self):
        selection={
            "signal":{"active_triggers":["low_margin"]},
            "targets":[
                {"role":"shared_down_proj","layer":26,"selected_n":6,
                 "status":"TRIGGER_CONDITIONED_ALTERNATE"},
                {"role":"shared_up_proj","layer":3,"selected_n":5,
                 "status":"HOLD_BASE_NO_TRIGGER_EVIDENCE"},
            ],
        }
        got=pco.optimize_policy(
            current_policy=A,allocation=ALLOCATION,selection=selection,
            combined_policy_evidence=accepted(),cost_evidence=costs(),
            runtime_state=self.runtime(warm_b=True),e2e_weight=1.0,
        )
        self.assertEqual(got["selected_cost"]["cache_state"],"warm")
        self.assertEqual(got["selected_cost"]["transition_p50_ms"],4.0)
        self.assertEqual(got["selected_cost"]["expected_e2e_ms"],17.0)

    def test_missing_exact_policy_fails_closed(self):
        selection={
            "signal":{"active_triggers":["low_margin"]},
            "targets":[
                {"role":"shared_down_proj","layer":26,"selected_n":6,
                 "status":"TRIGGER_CONDITIONED_ALTERNATE"},
                {"role":"shared_up_proj","layer":3,"selected_n":5,
                 "status":"HOLD_BASE_NO_TRIGGER_EVIDENCE"},
            ],
        }
        with self.assertRaises(pco.PolicyCostError):
            pco.optimize_policy(
                current_policy=A,allocation=ALLOCATION,selection=selection,
                combined_policy_evidence=accepted()[:1],cost_evidence=costs(),
                runtime_state=self.runtime(),e2e_weight=1.0,
            )


if __name__=="__main__":
    unittest.main()
