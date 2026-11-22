#!/usr/bin/env python3
"""JE01 exploratory real TRAIN16/45 h2 objective-disagreement audit.

Uses previously SHA-pinned original27 BF16-vs-native AF4/Q4/Q5
gate/up GEMV evidence. Verifies output SSE against the independent
covariance receipt. This is output reconstruction only, not Fisher,
Hessian, full-model NLL, a clean-240 policy, or a promotion gate.
"""
from __future__ import annotations
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIR = ROOT / "docs" / "research"
MAIN = DIR / "JE01_TRUE_ORIGINAL_ROUTED_H2_GATEUP_ACTIVATION_WEIGHTED_GEMV_20261122.json"
COV = DIR / "JE01_TRUE_ORIGINAL_H2_DIAGONAL_VS_FULL_COV_GATEUP_20261122.json"
RECEIPT = "d55f4b506076a3a30f73c582ede270c38d41ac12b06190e959ac5a7383157a89"
AF4_SHA = "9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
COST = {"AF4": 1_802_240, "Q4": 1_622_016, "Q5": 1_982_464}
FULL_BUDGET = 43_253_760
ROLES = {"gate_proj", "up_proj"}


def sources():
    a = json.loads(MAIN.read_text(encoding="utf-8"))
    b = json.loads(COV.read_text(encoding="utf-8"))
    assert a["receipt_sha256"] == RECEIPT
    assert b["source_16_role_actual_gemv_receipt_SHA256"] == RECEIPT
    assert a["actual_original_AF4_checkpoint_sha256"] == AF4_SHA
    assert not a["independent_heldout_OOD_accessed"]
    assert not b["independent_heldout_labels_used"]
    assert a["source_training_prompts"] == 16
    assert a["source_real_routed_h2_observations"] == 45
    assert len(a["per_role"]) == len(b["per_role"]) == 16
    assert {(r["layer"], r["expert"], r["role"]) for r in a["per_role"]} == {
        (r["layer"], r["expert"], r["role"]) for r in b["per_role"]}
    assert all(r["role"] in ROLES for r in a["per_role"])
    return a, b


def optimal(rows, objective):
    """Independent exact-byte 16-role DP, with all 8 down roles frozen at AF4."""
    states = {(0, False): (0., ())}
    for idx, r in enumerate(rows):
        weight = r["true_BF16_AF4_symmetric_output_energy"] if objective == "SSE" else 1.
        base = r["actual_activation_output_relative_MSE_AF4_vs_BF16"] * weight
        nxt = {}
        for (balance, changed), (risk, assignment) in states.items():
            for tier, shift in (("AF4", 0), ("Q4", -1), ("Q5", 1)):
                score = r[f"actual_activation_output_relative_MSE_{tier}_vs_BF16"] * weight
                key = (balance + shift, changed or tier != "AF4")
                candidate = (risk + score - base, assignment + (tier,))
                if key not in nxt or candidate[0] < nxt[key][0] - 1e-12:
                    nxt[key] = candidate
        states = nxt
    return states[(0, False)], states[(0, True)]


class TestJE01ExploratoryObjectiveDisagreement(unittest.TestCase):
    def test_true_source_is_exploratory_and_tightly_scoped(self):
        a, b = sources()
        self.assertEqual(a["summary"]["actual_real_FFN_gate_up_role_sensitivity_count"], 16)
        self.assertIn("true_model_loss_Fisher", a["summary"]["not_measured"])
        self.assertIn("real_down_proj_swiglu_activation", a["summary"]["not_measured"])
        self.assertTrue(b["actual_uncentered_covariance_not_empirical_Fisher"])
        self.assertFalse(a["production_modified"])
        self.assertFalse(b["production_modified"])

    def test_independent_exact_sse_matches_source(self):
        a, b = sources()
        indexed = {(r["layer"], r["expert"], r["role"]): r for r in b["per_role"]}
        worst = 0.
        for r in a["per_role"]:
            c = indexed[(r["layer"], r["expert"], r["role"])]
            for tier in COST:
                expected = (r[f"actual_activation_output_relative_MSE_{tier}_vs_BF16"] *
                            r["true_BF16_AF4_symmetric_output_energy"])
                actual = c["tier_diagnostics"][tier]["actual_exact_weighted_output_sse"]
                worst = max(worst, abs(expected - actual))
        self.assertLess(worst, 2e-5)

    def test_relative_mse_keeps_original_af4(self):
        a, _ = sources()
        baseline, changed = optimal(a["per_role"], "relative_MSE")
        self.assertAlmostEqual(baseline[0], 0., places=12)
        self.assertGreater(changed[0], 0.)
        self.assertAlmostEqual(changed[0], .0036970805603191944, places=9)

    def test_energy_weighted_sse_changes_some_roles_but_not_nll(self):
        a, _ = sources()
        original, changed = optimal(a["per_role"], "SSE")
        self.assertAlmostEqual(original[0], 0., places=12)
        self.assertAlmostEqual(changed[0], -8.795737023600797, places=5)
        self.assertLess(changed[0], original[0])
        assignments = changed[1]
        self.assertEqual(assignments.count("Q4"), 4)
        self.assertEqual(assignments.count("Q5"), 4)
        self.assertEqual(assignments.count("AF4"), 8)
        all_bytes = sum(COST[x] for x in assignments) + 8 * COST["AF4"]
        self.assertEqual(all_bytes, FULL_BUDGET)
        original_SSE = sum(
            r["actual_activation_output_relative_MSE_AF4_vs_BF16"] *
            r["true_BF16_AF4_symmetric_output_energy"] for r in a["per_role"])
        self.assertAlmostEqual(original_SSE, 144.6805760521923, places=5)
        self.assertAlmostEqual(-100 * changed[0] / original_SSE, 6.079418028048083, places=4)
        # This test does not claim model quality: selected using prior TRAIN16 pilot.


if __name__ == "__main__":
    unittest.main()
