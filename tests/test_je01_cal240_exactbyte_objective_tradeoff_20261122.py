#!/usr/bin/env python3
"""JE01 signed CAL240 *training-only* objective-sensitivity audit.

Independently recomputes 24-role equal-byte DP from the actual committed
native FFN role-level source ledger. No pretrained model, NLL, Fisher/Hessian,
new GPU forward, validation labels, or model-promotion claims are involved.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

LEDGER_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/research/JE01_ACTUAL_CAL240_FFN_H2_SWIGLU_NATIVE_OUTPUT_DP_20261122.json"
)
LEDGER_SHA256 = "479c8548f74de0a8efab51547559be7db5fcd0e34e3e2722192c67c10c8f1dbd"
SIGNED_SHA256 = "35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb"
COST = {"AF4": 1802240, "Q4": 1622016, "Q5": 1982464}
TIERS = ("AF4", "Q4", "Q5")
BUDGET = 43253760


def source():
    raw = LEDGER_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LEDGER_SHA256:
        raise ValueError("JE01_CAL240_TRADEOFF_SOURCE_SHA_DRIFT")
    doc = json.loads(raw)
    if doc.get("receipt_sha256") != SIGNED_SHA256:
        raise ValueError("JE01_CAL240_TRADEOFF_SOURCE_RECEIPT_DRIFT")
    if (doc.get("role_count") != 24
            or doc.get("training_unseen240_prompt_count") != 240
            or doc.get("training_first16_prior_pilot_excluded_from_precision_selection") is not True
            or doc.get("source_heldout_OOD_labels_used") is not False
            or doc.get("production_modified") is not False):
        raise ValueError("JE01_CAL240_TRADEOFF_SOURCE_SCOPE_DRIFT")
    roles = doc["per_role"]
    if len(roles) != 24 or len({(x["layer"], x["expert"], x["role"]) for x in roles}) != 24:
        raise ValueError("JE01_CAL240_TRADEOFF_ROLE_COUNT_DRIFT")
    return doc, roles


def total(roles, allocation, suffix):
    return sum(
        x[f"route_weighted_output_{suffix}_{tier}"]
        for x, tier in zip(roles, allocation)
    )


def minimize_exact_byte(roles, lam, down_only=False):
    """DP state = number of Q5 roles minus number of Q4 roles."""
    assert 0 <= lam <= 1
    baseline_sse = total(roles, ["AF4"] * len(roles), "SSE")
    baseline_rel = total(roles, ["AF4"] * len(roles), "relative_MSE")
    states = {0: (0.0, ())}
    for role in roles:
        nxt = {}
        choices = ("AF4",) if down_only and role["role"] != "down_proj" else TIERS
        for balance, (score, policy) in states.items():
            for tier in choices:
                b = balance + (tier == "Q5") - (tier == "Q4")
                item_score = (
                    (1 - lam) * role[f"route_weighted_output_SSE_{tier}"] / baseline_sse
                    + lam * role[f"route_weighted_output_relative_MSE_{tier}"] / baseline_rel
                )
                candidate = score + item_score
                if b not in nxt or candidate < nxt[b][0] - 1e-13:
                    nxt[b] = (candidate, policy + (tier,))
        states = nxt
    objective, allocation = states[0]
    assert len(allocation) == 24
    assert allocation.count("Q4") == allocation.count("Q5")
    assert sum(COST[t] for t in allocation) == BUDGET
    sse = total(roles, allocation, "SSE")
    rel = total(roles, allocation, "relative_MSE")
    return {
        "allocation": allocation,
        "objective": objective,
        "sse_improvement_pct": 100 * (1 - sse / baseline_sse),
        "relative_mse_change_pct": 100 * (rel / baseline_rel - 1),
    }


class TestJE01Cal240ExactByteSensitivity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc, cls.roles = source()

    def test_baseline_and_24_provenance_constraints(self):
        self.assertAlmostEqual(total(self.roles, ["AF4"] * 24, "SSE"),
                               7192.091944662442, places=8)
        self.assertAlmostEqual(total(self.roles, ["AF4"] * 24, "relative_MSE"),
                               0.134837209449249, places=12)
        self.assertEqual(24 * COST["AF4"], BUDGET)

    def test_frozen_sse_policy_reproduced_not_a_new_policy(self):
        frozen = self.doc["separate_exactbyte_dp_objective_results"]["route_weighted_output_SSE"]["best_overall"]
        allocation = tuple(x[3] for x in frozen["source_assignment"])
        # The source assignment is ordered by the source's 24-role rows.
        source_keys = [(x["layer"], x["expert"], x["role"]) for x in self.roles]
        assignment_keys = [tuple(x[:3]) for x in frozen["source_assignment"]]
        self.assertEqual(source_keys, assignment_keys)
        optimum = minimize_exact_byte(self.roles, 0)
        self.assertEqual(optimum["allocation"], allocation)
        self.assertEqual(allocation.count("Q4"), 6)
        self.assertEqual(allocation.count("Q5"), 6)
        self.assertAlmostEqual(optimum["sse_improvement_pct"],
                               22.34691891899631, places=8)
        self.assertAlmostEqual(optimum["relative_mse_change_pct"],
                               23.465090391375654, places=8)

    def test_half_normalized_sensitivity_prefers_only_two_changed_roles(self):
        result = minimize_exact_byte(self.roles, 0.5)
        self.assertEqual(result["allocation"].count("Q4"), 1)
        self.assertEqual(result["allocation"].count("Q5"), 1)
        edited = {
            (r["layer"], r["expert"], r["role"], tier)
            for r, tier in zip(self.roles, result["allocation"]) if tier != "AF4"
        }
        self.assertEqual(edited, {
            (14, 63, "gate_proj", "Q4"),
            (24, 26, "down_proj", "Q5"),
        })
        self.assertAlmostEqual(result["sse_improvement_pct"],
                               14.156141043968217, places=8)
        self.assertAlmostEqual(result["relative_mse_change_pct"],
                               2.350746440382334, places=8)

    def test_down_only_preserving_all_gate_up_is_still_proxy_only(self):
        full_down = minimize_exact_byte(self.roles, 0, down_only=True)
        self.assertEqual(full_down["allocation"].count("Q4"), 4)
        self.assertEqual(full_down["allocation"].count("Q5"), 4)
        self.assertAlmostEqual(full_down["sse_improvement_pct"],
                               16.75655822947004, places=8)
        self.assertAlmostEqual(full_down["relative_mse_change_pct"],
                               34.014090039064236, places=8)
        for role, tier in zip(self.roles, full_down["allocation"]):
            if role["role"] != "down_proj":
                self.assertEqual(tier, "AF4")

    def test_pure_relative_mse_returns_original_af4(self):
        optimum = minimize_exact_byte(self.roles, 1)
        self.assertEqual(optimum["allocation"], ("AF4",) * 24)
        self.assertAlmostEqual(optimum["sse_improvement_pct"], 0, places=8)
        self.assertAlmostEqual(optimum["relative_mse_change_pct"], 0, places=8)


if __name__ == "__main__":
    unittest.main()
