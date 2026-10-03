#!/usr/bin/env python3
import unittest

import precision_e2e_cost as pec


CANDIDATES=[
    {"role":"shared_up_proj","layer":3,"n":5,"feasible":True,
     "estimated_bytes":500.0,"real_pass_events":4},
    {"role":"shared_up_proj","layer":3,"n":6,"feasible":True,
     "estimated_bytes":600.0,"real_pass_events":4},
]
CURRENT=[{"role":"shared_up_proj","layer":3,"n":6}]


def evidence(passes5=1):
    return {
        "schema":"beglin-precision-e2e-cost-v1",
        "target_profiles":[
            {
                "role":"shared_up_proj","layer":3,"n":5,
                "steady_engine_p50_ms":10.0,
                "steady_roundtrip_p50_ms":12.0,
                "rss_bytes":1000,
                "expected_inference_passes":passes5,
                "transitions":[
                    {"from_n":6,"cache_state":"cold","p50_transition_ms":8.0,
                     "cache_hits":0,"cache_misses":1,"cache_bytes_added":200},
                    {"from_n":6,"cache_state":"warm","p50_transition_ms":1.0,
                     "cache_hits":1,"cache_misses":0,"cache_bytes_added":0},
                ],
            },
            {
                "role":"shared_up_proj","layer":3,"n":6,
                "steady_engine_p50_ms":11.0,
                "steady_roundtrip_p50_ms":13.0,
                "rss_bytes":1100,
                "expected_inference_passes":1,
                "transitions":[],
            },
        ],
    }
class CostTests(unittest.TestCase):
    def test_cold_candidate_prices_cache_miss_and_bytes(self):
        got=pec.enrich_candidates(
            CANDIDATES,current_policy=CURRENT,
            runtime_state={"qng64_cache":[
                {"role":"shared_up_proj","layer":3,"n":6,"bytes":300}
            ],"resident_qng64_cache_bytes":300},
            cost_evidence=evidence(),
        )
        by_n={r["n"]:r for r in got}
        self.assertEqual(by_n[5]["transition_cache_state"],"cold")
        self.assertEqual(by_n[5]["transition_cache_misses"],1)
        self.assertEqual(by_n[5]["transition_cache_bytes_added"],200)
        self.assertEqual(by_n[5]["resident_cache_bytes_after"],500)
        self.assertEqual(by_n[5]["expected_e2e_ms"],20.0)
        self.assertEqual(by_n[6]["expected_e2e_ms"],13.0)

    def test_warm_candidate_uses_hit_cost(self):
        got=pec.enrich_candidates(
            CANDIDATES,current_policy=CURRENT,
            runtime_state={"qng64_cache":[
                {"role":"shared_up_proj","layer":3,"n":6,"bytes":300},
                {"role":"shared_up_proj","layer":3,"n":5,"bytes":200},
            ],"resident_qng64_cache_bytes":500},
            cost_evidence=evidence(),
        )
        row={r["n"]:r for r in got}[5]
        self.assertEqual(row["transition_cache_state"],"warm")
        self.assertEqual(row["transition_cache_hits"],1)
        self.assertEqual(row["transition_cache_bytes_added"],0)
        self.assertEqual(row["expected_e2e_ms"],13.0)

    def test_extra_inference_passes_are_in_e2e_cost(self):
        got=pec.enrich_candidates(
            CANDIDATES,current_policy=CURRENT,
            runtime_state={"qng64_cache":[],"resident_qng64_cache_bytes":0},
            cost_evidence=evidence(passes5=2),
        )
        row={r["n"]:r for r in got}[5]
        self.assertEqual(row["expected_inference_passes"],2)
        self.assertEqual(row["expected_e2e_ms"],32.0)
    def test_missing_transition_marks_cost_incomplete(self):
        ev=evidence()
        ev["target_profiles"][0]["transitions"]=[]
        got=pec.enrich_candidates(
            CANDIDATES,current_policy=CURRENT,
            runtime_state={"qng64_cache":[],"resident_qng64_cache_bytes":0},
            cost_evidence=ev,
        )
        row={r["n"]:r for r in got}[5]
        self.assertFalse(row["cost_evidence_complete"])
        self.assertEqual(row["cost_missing_transition"]["cache_state"],"cold")

    def test_invalid_negative_runtime_cache_is_rejected(self):
        with self.assertRaises(pec.PrecisionCostError):
            pec.enrich_candidates(
                CANDIDATES,current_policy=CURRENT,
                runtime_state={"qng64_cache":[],"resident_qng64_cache_bytes":-1},
                cost_evidence=evidence(),
            )


if __name__=="__main__":
    unittest.main()
