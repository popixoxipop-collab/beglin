#!/usr/bin/env python3
"""Independently verify the committed REAL TRAIN240 24-role signed JE01 ledger.

This verifies source receipt digest, 24-role records, count and budget contract
and mathematically replays both DPs from real measured per-role numbers.
Raw EOE hidden/binary weights are not available to GitHub CI; this is NOT a
second FFN GEMV measurement, Fisher/Hessian or full NLL.
"""
from __future__ import annotations
import copy
import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "research" / "JE01_ACTUAL_CAL240_FFN_H2_SWIGLU_NATIVE_OUTPUT_DP_20261122.json"
SOURCE = ROOT / "tools" / "je01_cal240_real_dp_independent_audit_20261122.py"
spec = importlib.util.spec_from_file_location("je01_cal240_signed_dp_real_ledger", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class JE01RealSignedCal240SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(LEDGER.read_text(encoding="utf-8"))

    def test_full_actual_signed_source_and_all_role_dp_replay(self):
        out = audit.audit(self.data)
        self.assertEqual(out["status"], "JE01_CAL240_LEDGER_AND_INDEPENDENT_DP_REPLAY_PASS")
        self.assertEqual(out["receipt_sha256"], audit.EXPECTED_RECEIPT)
        self.assertEqual(out["role_count"], 24)
        self.assertEqual(out["clean_actual_route_events"], 637)
        self.assertEqual(out["budget_bytes"], 43_253_760)
        m = out["results"]
        self.assertAlmostEqual(
            m["route_weighted_output_SSE"]["best_risk_delta"],
            -1607.2109554533818, places=6)
        self.assertEqual(m["route_weighted_output_SSE"]["best_q4_roles"], 6)
        self.assertEqual(m["route_weighted_output_SSE"]["best_q5_roles"], 6)
        self.assertEqual(m["route_weighted_output_relative_MSE"]["best_risk_delta"], 0.0)
        self.assertAlmostEqual(m["route_weighted_output_relative_MSE"]["nonbaseline_risk_delta"],
                               0.0024733855809832266, places=12)

    def test_actual_sse_relative_mse_normalization_and_role_tradeoff(self):
        roles = self.data["per_role"]
        tiers = {(l, e, role): tier for l, e, role, tier in
                 self.data["separate_exactbyte_dp_objective_results"][
                     "route_weighted_output_SSE"]["best_overall"]["source_assignment"]}
        baseline_sse = sum(r["route_weighted_output_SSE_AF4"] for r in roles)
        candidate_sse = sum(r["route_weighted_output_SSE_" +
                            tiers[(r["layer"], r["expert"], r["role"])]] for r in roles)
        baseline_relative = sum(r["route_weighted_output_relative_MSE_AF4"] for r in roles)
        candidate_relative = sum(r["route_weighted_output_relative_MSE_" +
                                 tiers[(r["layer"], r["expert"], r["role"])]] for r in roles)
        self.assertAlmostEqual(baseline_sse, 7192.091944662442, places=7)
        self.assertAlmostEqual(candidate_sse, 5584.880989209061, places=7)
        self.assertAlmostEqual((baseline_sse - candidate_sse)/baseline_sse,
                               0.2234691891899631, places=12)
        self.assertAlmostEqual(baseline_relative, 0.134837209449249, places=12)
        self.assertAlmostEqual(candidate_relative, 0.1664768825277238, places=12)
        self.assertGreater(candidate_relative, baseline_relative)

    def test_mutating_record_with_stale_receipt_must_fail(self):
        corrupt = copy.deepcopy(self.data)
        corrupt["per_role"][0]["route_weighted_output_SSE_Q4"] += 1.0
        with self.assertRaisesRegex(ValueError, "SIGNED_CAL240_RECEIPT_SHA_MISMATCH"):
            audit.audit(corrupt)

    def test_removing_expert_role_with_recomputed_receipt_must_fail(self):
        corrupt = copy.deepcopy(self.data)
        corrupt["per_role"].pop()
        corrupt["receipt_sha256"] = audit.canonical({k: v for k,v in corrupt.items()
                                                      if k != "receipt_sha256"})
        with self.assertRaisesRegex(ValueError, "ROLE_COUNT_INVALID"):
            audit.audit(corrupt, expected_receipt=corrupt["receipt_sha256"])


if __name__ == "__main__":
    unittest.main()
