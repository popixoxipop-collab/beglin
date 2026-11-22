#!/usr/bin/env python3
"""Independent arithmetic and evidence-contract audit of actual JE01 joint FFN receipt.

Verifies the pinned REAL native source replay ledger, not a second execution
of the original 27-layer model. The reported benefits are TRAIN240 in-sample
BF16 reconstruction, NOT gold NLL, Fisher/Hessian or heldout generalization.
"""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/research/JE01_CAL240_JOINT_NONLINEAR_FFN_NATIVE_REPLAY_20261122.json"
SOURCE_SHA256 = "454d42987feaacf5032e92d17a764c850c5eec16078a43db0754328a0089a8a0"
SIGNED_RECEIPT_SHA256 = "bb365f1a39e46271317075e672600dd41faf65003d2b429a5997e44461b51d04"
CAL240_SOURCE_RECEIPT_SHA = "35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb"
ORIGINAL_AF4_CHECKPOINT_SHA = "9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
BUDGET = 43_253_760
EXPERTS = {(12,13),(13,22),(13,28),(14,63),(20,17),(21,61),(24,26),(24,60)}


def load():
    raw = SOURCE.read_bytes()
    if len(raw) != 89672 or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise AssertionError("JE01_NATIVE_JOINT_LEDGER_BYTES_OR_SHA_CHANGED")
    d = json.loads(raw)
    receipt = d.get("receipt_sha256")
    canonical = hashlib.sha256(json.dumps(
        {k:v for k,v in d.items() if k != "receipt_sha256"},
        separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode()).hexdigest()
    if receipt != SIGNED_RECEIPT_SHA256 or canonical != receipt:
        raise AssertionError("JE01_NATIVE_JOINT_LEDGER_SIGNED_RECEIPT_CHANGED")
    return d


class TestJE01ActualJointFFNIndependentLedger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load()
        cls.experts = cls.d["per_expert_original_activations_full_joint_FFN"]
        cls.cases = cls.d["paired_training_prompt_layer_metrics"]

    def test_only_clean_training_frozen_scope_and_source_contract(self):
        d=self.d
        self.assertEqual(d["status"], "JE01_CAL240_FROZEN_SSE_DP_JOINT_NONLINEAR_FFN_REPLAY_COMPLETE")
        self.assertEqual(d["clean_training_prompts"], 240)
        self.assertEqual(d["clean_routed_expert_records"], 637)
        self.assertTrue(d["training_first16_prior_pilot_excluded"])
        self.assertTrue(d["native_source_original_BF16_verified"])
        self.assertEqual(d["FFN_roles"], 24)
        self.assertEqual((d["frozen_Q4_roles"], d["frozen_Q5_roles"], d["frozen_original_AF4_roles"]), (6,6,12))
        self.assertEqual(d["physical_selected_budget_bytes"], BUDGET)
        self.assertEqual(d["original_AF4_checkpoint_sha256"], ORIGINAL_AF4_CHECKPOINT_SHA)
        self.assertEqual(d["original_cal240_rolewise_DP_receipt_sha256"], CAL240_SOURCE_RECEIPT_SHA)
        self.assertIs(d["production_modified"], False)
        self.assertEqual(d["untouched_validation_OOD_gold"], "NOT_USED")
        self.assertEqual(d["real_full_native_27layer_NLL"], "NOT_RUN")
        self.assertEqual(d["true_loss_Fisher"], "NOT_MEASURED")
        self.assertEqual(d["true_loss_Hessian"], "NOT_MEASURED")

    def test_original_swiglu_parity_and_all_eight_experts(self):
        e=self.experts
        self.assertEqual(len(e), 8)
        self.assertEqual({(x["layer"],x["expert"]) for x in e}, EXPERTS)
        self.assertEqual(sum(x["training_clean_prompt_events"] for x in e), 637)
        self.assertTrue(all(0 <= x["actual_original_swiglu_replay_parity_maxabs"] < 0.0003 for x in e))
        self.assertTrue(all(0 <= x["actual_original_swiglu_replay_parity_relative_rmse"] < 1.3e-6 for x in e))
        self.assertEqual(
            sum(v=="Q4" for x in e for v in x["tiers"].values()), 6
        )
        self.assertEqual(
            sum(v=="Q5" for x in e for v in x["tiers"].values()), 6
        )

    def test_real_native_full_swiglu_weighted_sse_sums_and_change(self):
        e=self.experts;d=self.d
        base=sum(x["full_swiglu_AF4_baseline_weighted_SSE"] for x in e)
        cand=sum(x["full_swiglu_frozen_SSE_policy_weighted_SSE"] for x in e)
        delta=sum(x["full_swiglu_candidate_minus_original_SSE"] for x in e)
        self.assertAlmostEqual(base, 10899.437801231934, places=8)
        self.assertAlmostEqual(cand, 8424.407911040542, places=8)
        self.assertAlmostEqual(delta, cand-base, places=8)
        self.assertAlmostEqual(base, d["native_full_FFN_weighted_SSE_AF4_baseline"], places=8)
        self.assertAlmostEqual(cand, d["native_full_FFN_weighted_SSE_SSE_DP_candidate"], places=8)
        self.assertAlmostEqual(cand/base - 1, d["native_full_FFN_weighted_SSE_relative_change"], places=12)

    def test_real_corouted_case_layer_sums_and_no_pilot_leak(self):
        p=self.cases; d=self.d
        self.assertEqual(len(p), 577)
        self.assertEqual(len({(v["case"],v["layer"]) for v in p}), 577)
        self.assertTrue(all(16 <= v["case"] < 256 and v["layer"] in {12,13,14,20,21,24} for v in p))
        self.assertTrue(all(v["baseline_squared"]>=0 and v["candidate_squared"]>=0 for v in p))
        baseline=sum(x["baseline_squared"] for x in p)
        candidate=sum(x["candidate_squared"] for x in p)
        self.assertEqual(d["joint_layer_prompt_pairs"], 577)
        self.assertAlmostEqual(baseline, 1263.6102780048805, places=8)
        self.assertAlmostEqual(candidate, 946.6335188621481, places=8)
        self.assertAlmostEqual(baseline, d["native_same_layer_corouted_experts_squared_error_baseline"], places=8)
        self.assertAlmostEqual(candidate, d["native_same_layer_corouted_experts_squared_error_candidate"], places=8)
        self.assertAlmostEqual(candidate/baseline-1,
            d["native_same_layer_corouted_experts_squared_error_relative_change"], places=12)
        clusters=defaultdict(list)
        for x in p:clusters[x["case"]].append(x)
        self.assertLessEqual(len(clusters), 240)

    def test_regression_risk_nonuniform_and_one_late_expert_dominance(self):
        x={(e["layer"],e["expert"]):e for e in self.experts}
        self.assertTrue(all(x[(layer,expert)]["full_swiglu_candidate_minus_original_SSE"]>0
                            for layer,expert in ((12,13),(13,22),(13,28))))
        self.assertTrue(all(x[(layer,expert)]["full_swiglu_candidate_minus_original_SSE"]<0
                            for layer,expert in ((20,17),(21,61),(24,26),(24,60))))
        last=x[(24,26)]
        other_base=self.d["native_full_FFN_weighted_SSE_AF4_baseline"]-last["full_swiglu_AF4_baseline_weighted_SSE"]
        other_cand=self.d["native_full_FFN_weighted_SSE_SSE_DP_candidate"]-last["full_swiglu_frozen_SSE_policy_weighted_SSE"]
        self.assertAlmostEqual(100*(1-other_cand/other_base),5.049287557607151,places=7)
        self.assertGreater(last["full_swiglu_AF4_baseline_weighted_SSE"]-
                           last["full_swiglu_frozen_SSE_policy_weighted_SSE"], 2000)


if __name__ == "__main__":
    unittest.main()
