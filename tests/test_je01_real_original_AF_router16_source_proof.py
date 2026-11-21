#!/usr/bin/env python3
"""Recompute original native BEGLIN real route Top6 16x26 provenance."""
from collections import Counter
import hashlib,json,math,pathlib,unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
PATH=ROOT/"docs/research/JE01_ACTUAL_XOX_ORIGINAL_AF_ROUTER26_16CAL_LOGIT_PARITY_20261118.json"
SHA="bfacc0c0257d9916ef2aac3ffea840f9228ca4ee1b488797034daec9f5e7e223"

def open_sealed():
    data=json.loads(PATH.read_text())
    seal=data.pop("receipt_sha256")
    actual=hashlib.sha256(json.dumps(data,sort_keys=True,
                     separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if seal!=SHA or actual!=SHA:raise AssertionError("JE01_ORIGINAL_REAL_ROUTER16_SEAL_DRIFT")
    return data

class NativeOriginalRouterReal512Calibration16(unittest.TestCase):
    def test_real_original_26layer_gpu_evidence_and_full_logit_parity(self):
        d=open_sealed()
        self.assertEqual(d["schema"],"beglin-je01-original-AF-real-router26-cal16-logit-parity/1")
        self.assertEqual(d["status"],"PASS_REAL_ORIGINAL_GPU_26_LAYER_16_CAL_ROUTER_PARITY")
        self.assertEqual(d["real_route_events"],416)
        self.assertEqual(d["real_expert_decisions"],2496)
        self.assertEqual(d["real_router_scores"],26624)
        self.assertEqual(d["raw_102400_full_logits_sha256"],
                         "428d2f985d1eca80ab299a7ab8cbf3d1330391efe46791f87d013a670f5d387a")
        self.assertEqual(d["source_previous_full102400_receipt_sha256"],
                         "34da06b2441bd1cfa8c8593ff20e8c2d3e40850e1c0ef6d6c8a127d70cb75d14")
        self.assertEqual(d["source_research_binary_sha256"],
                         "37eccd3870e4b95be493ae18773497b8c6ab1ba54fff37d317477c0c7cfc78ad")
        self.assertEqual(d["real_calibration_only_unlabeled_router"],True)
        self.assertTrue(d["all_event_sums_valid"])
        self.assertTrue(d["all_native_top6_boundaries_nonnegative"])
        self.assertEqual(d["actual_bayesian_expert_NLL_effect"],"NOT_MEASURED")
        self.assertEqual(d["original_256_calibration_slot_freeze"],"NOT_PERFORMED")
        self.assertFalse(d["source_original_weights_mutated"])
        self.assertFalse(d["production_route_mutated"])
        self.assertFalse(d["production_promotion_allowed"])
        self.assertEqual(d["production_before"],d["production_after"])
        self.assertEqual(set(d["production_before"].values()),{35481,35493})

    def test_all_416_real_native_selected_routes_recompute_top8(self):
        d=open_sealed()
        self.assertEqual(len(d["events"]),16)
        counts=Counter()
        minmargin=float("inf")
        for request,event in enumerate(d["events"]):
            self.assertEqual(event["request"],request)
            self.assertGreaterEqual(event["native_final_prefill_position"],1)
            self.assertEqual(len(event["per_layer_top6"]),26)
            for l,row in enumerate(event["per_layer_top6"],start=1):
                self.assertEqual(row["layer"],l)
                ids=row["expert_ids"]
                self.assertEqual(len(ids),6)
                self.assertEqual(len(set(ids)),6)
                self.assertTrue(all(type(e) is int and 0<=e<64 for e in ids))
                self.assertTrue(math.isfinite(row["min_6_vs_7_probability_gap"]))
                self.assertGreaterEqual(row["min_6_vs_7_probability_gap"],0)
                minmargin=min(minmargin,row["min_6_vs_7_probability_gap"])
                counts.update((l,e) for e in ids)
            self.assertGreaterEqual(event["min_top6_margin"],minmargin if request==0 else 0)
        self.assertEqual(sum(counts.values()),2496)
        ranks=sorted([{"layer":l,"expert":e,"calls":counts[(l,e)]}
                      for l in range(1,27) for e in range(64)],
                      key=lambda r:(-r["calls"],r["layer"],r["expert"]))
        self.assertEqual(d["real_pilot_top8_occupancy_only_not_frozen_for_256"],ranks[:8])
        self.assertGreater(minmargin,0.0)
        self.assertLess(minmargin,1e-4)

if __name__=="__main__":
    unittest.main(verbosity=2)
