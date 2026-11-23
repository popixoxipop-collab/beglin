#!/usr/bin/env python3
"""JE01 real CAL240 24-role signed evidence, independent stdlib-only replay.

This is a pure offline audit of the SHA-pinned output-error *ledger* (not
raw GEMV re-execution, loss Fisher/Hessian, or native 27-layer NLL).
The 16 previously explored TRAIN prompts are excluded from policy selection.
"""
from __future__ import annotations
import hashlib
import json
import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import je01_cal240_real_dp_independent_audit_20261122 as audit

SRC = ROOT / "docs" / "research" / "JE01_ACTUAL_CAL240_FFN_H2_SWIGLU_NATIVE_OUTPUT_DP_20261122.json"
FILE_SHA = "479c8548f74de0a8efab51547559be7db5fcd0e34e3e2722192c67c10c8f1dbd"
RECEIPT = "35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb"


def read():
    content = SRC.read_bytes()
    if hashlib.sha256(content).hexdigest() != FILE_SHA:
        raise ValueError("JE01_REAL_CAL240_FILE_SHA_DRIFT")
    return json.loads(content)


class JE01RealCalibration240PinnedEvidence(unittest.TestCase):
    def test_full_role_source_file_and_signed_receipt(self):
        data = read()
        self.assertEqual(data["receipt_sha256"], RECEIPT)
        self.assertEqual(audit.canonical({k:v for k,v in data.items()
                                          if k != "receipt_sha256"}), RECEIPT)
        self.assertFalse(data["source_heldout_OOD_labels_used"])
        self.assertFalse(data["production_modified"])
        self.assertTrue(data["training_first16_prior_pilot_excluded_from_precision_selection"])
        self.assertEqual(data["training_unseen240_prompt_count"], 240)
        self.assertEqual(data["observed_total_original256_routed_activation_records"], 682)
        self.assertEqual(data["full_native_27layer_new_NLL"], "NOT_RUN")
        self.assertEqual(data["true_logprob_gradients"], "NOT_MEASURED")

    def test_independent_dp_and_all24_role_contracts(self):
        data = read()
        result = audit.audit(data, expected_receipt=RECEIPT)
        self.assertEqual(result["status"], "JE01_CAL240_LEDGER_AND_INDEPENDENT_DP_REPLAY_PASS")
        self.assertEqual(result["clean_actual_route_events"], 637)
        self.assertEqual(result["role_count"], 24)
        self.assertEqual(result["budget_bytes"], 43_253_760)
        self.assertEqual(result["results"]["route_weighted_output_SSE"]["best_q4_roles"], 6)
        self.assertEqual(result["results"]["route_weighted_output_SSE"]["best_q5_roles"], 6)
        self.assertEqual(result["results"]["route_weighted_output_relative_MSE"]["best_q4_roles"], 0)
        self.assertEqual(result["results"]["route_weighted_output_relative_MSE"]["best_q5_roles"], 0)

    def test_absolute_sse_reduction_and_role_decomposition(self):
        data = read()
        rows = data["per_role"]
        chosen = data["separate_exactbyte_dp_objective_results"]["route_weighted_output_SSE"]["best_overall"]["source_assignment"]
        policies = {(l,e,r):tier for l,e,r,tier in chosen}
        self.assertEqual(len(policies), 24)
        baseline = sum(r["route_weighted_output_SSE_AF4"] for r in rows)
        candidate = sum(r["route_weighted_output_SSE_" + policies[(r["layer"],r["expert"],r["role"])]] for r in rows)
        self.assertAlmostEqual(baseline, 7192.091944662442, places=7)
        self.assertAlmostEqual(candidate, 5584.880989209061, places=7)
        self.assertAlmostEqual((baseline-candidate)/baseline * 100, 22.34691891899631, places=8)
        by_role = {}
        for row in rows:
            role=row["role"]
            orig=row["route_weighted_output_SSE_AF4"]
            new=row["route_weighted_output_SSE_" + policies[(row["layer"],row["expert"],role)]]
            acc=by_role.setdefault(role, [0.,0.])
            acc[0]+=orig;acc[1]+=new
        self.assertGreater(by_role["gate_proj"][1], by_role["gate_proj"][0])
        self.assertGreater(by_role["up_proj"][1], by_role["up_proj"][0])
        self.assertLess(by_role["down_proj"][1], by_role["down_proj"][0])

    def test_role_risk_concentration_and_no_independent_nll_claim(self):
        data = read()
        rows=data["per_role"]
        assignment = {tuple(x[:3]):x[3] for x in data["separate_exactbyte_dp_objective_results"]["route_weighted_output_SSE"]["best_overall"]["source_assignment"]}
        down_q5 = [r for r in rows if r["role"] == "down_proj" and
                   assignment[(r["layer"], r["expert"], r["role"])] == "Q5"]
        self.assertEqual(len(down_q5), 4)
        l24e26 = next(r for r in down_q5 if (r["layer"], r["expert"]) == (24,26))
        benefit = (l24e26["route_weighted_output_SSE_AF4"]-
                   l24e26["route_weighted_output_SSE_Q5"])
        self.assertAlmostEqual(benefit, 1115.1181033829303, places=7)
        self.assertGreater(benefit / 1607.2109554533818, .69)
        self.assertEqual(data["full_native_27layer_new_NLL"], "NOT_RUN")
        self.assertEqual(data["scientific_scope"],
                         "ACTUAL_TRAINED_ORIGINAL_ROUTE_WEIGHTED_GEMV_DIAGONAL_OUTPUT_METRIC_NOT_TRUE_LOSS_FISHER_OR_HESSIAN")

if __name__ == "__main__":
    unittest.main()
