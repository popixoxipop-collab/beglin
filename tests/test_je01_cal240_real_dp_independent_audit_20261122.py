#!/usr/bin/env python3
"""Synthetic-only negative controls for the independent JE01 CAL240 DP receipt audit."""
from __future__ import annotations
import copy
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "je01_cal240_real_dp_independent_audit_20261122.py"
spec = importlib.util.spec_from_file_location("je01_cal240_signed_audit", SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def sample_rows():
    roles = []
    for l, e in mod.FROZEN:
        for role in mod.ROLE_ORDER:
            row = {"layer": l, "expert": e, "role": role,
                   "actual_clean_unseen_training_router_observations": mod.CLEAN_COUNT[(l, e)],
                   "routed_probability_mass": 4.5,
                   "actual_input_dimension": 1408 if role == "down_proj" else 2048,
                   "actual_output_dimension": 2048 if role == "down_proj" else 1408,
                   "actual_original_BF16_tensor_sha256": "a" * 64,
                   "actual_original_AF4_packed_sha256": "b" * 64,
                   "original_BF16_activation_output_energy": 100.}
            for tier, value in {"AF4": 10., "Q4": 13., "Q5": 9.}.items():
                row["route_weighted_output_SSE_" + tier] = value
                row["route_weighted_output_relative_MSE_" + tier] = value / 100
            roles.append(row)
    return roles


def sample_receipt():
    rows = sample_rows()
    d = {
        "status": "JE01_ACTUAL_ORIGINAL27_CAL240_UNSEEN_FFN_H2_SWIGLU_24ROLE_ACTIVATION_DP_COMPLETE",
        "source_actual_training_original27_256_h2_sw_trace_receipt_sha256": mod.TRACE_RECEIPT,
        "source_original_router256_receipt_sha256": mod.ROUTER_SHA,
        "source_full_original_AF4_sha256": mod.AF_SHA,
        "true_original_Chat_BF16_four_shard_SHA_reverified": True,
        "actual_original_256_training_prompts_all_capture": True,
        "training_first16_prior_pilot_excluded_from_precision_selection": True,
        "training_unseen240_prompt_count": 240,
        "observed_total_original256_routed_activation_records": 682,
        "role_count": 24,
        "selected_frozen_expert_count": 8,
        "observed_FFN_input_dimensions": {"gate_up": 2048, "down_SwiGLU": 1408},
        "observed_source_input_moment_npz_sha256": "c" * 64,
        "per_role": rows,
        "separate_exactbyte_dp_objective_results": {
            name: mod.independent_dp(rows, name) for name in mod.METRICS},
        "scientific_scope": "ACTUAL_TRAINED_ORIGINAL_ROUTE_WEIGHTED_GEMV_DIAGONAL_OUTPUT_METRIC_NOT_TRUE_LOSS_FISHER_OR_HESSIAN",
        "true_logprob_gradients": "NOT_MEASURED",
        "full_native_27layer_new_NLL": "NOT_RUN",
        "source_heldout_OOD_labels_used": False,
        "production_modified": False,
    }
    d["receipt_sha256"] = mod.canonical(d)
    return d


class JE01RealCal240SignedAuditTests(unittest.TestCase):
    def test_independent_dp_no_profitable_swap_stays_af4(self):
        for metric in mod.METRICS:
            out = mod.independent_dp(sample_rows(), metric)
            self.assertEqual(out["best_overall"]["risk_difference"], 0.)
            self.assertEqual(out["best_overall"]["unchanged_AF4_roles"], 24)
            self.assertEqual(out["best_nonbaseline_exactbyte"]["q4_roles"], 1)
            self.assertEqual(out["best_nonbaseline_exactbyte"]["q5_roles"], 1)

    def test_independent_dp_profitable_exactbyte_trade(self):
        rows = sample_rows()
        for metric in mod.METRICS:
            rows[0][metric + "_Q4"] = rows[0][metric + "_AF4"] + 0.0001
            rows[1][metric + "_Q5"] = rows[1][metric + "_AF4"] - 1.0
            out = mod.independent_dp(rows, metric)
            self.assertEqual(out["best_overall"]["q4_roles"], 1)
            self.assertEqual(out["best_overall"]["q5_roles"], 1)
            self.assertLess(out["best_overall"]["risk_difference"], 0)
            self.assertEqual(sum(mod.COST[x[3]] for x in
                                 out["best_overall"]["source_assignment"]), mod.BUDGET)

    def test_source_receipt_digest_tamper_is_rejected(self):
        d = sample_receipt()
        d["per_role"][0]["route_weighted_output_SSE_Q4"] = -1
        with self.assertRaisesRegex(ValueError, "SIGNED_CAL240_RECEIPT_SHA_MISMATCH"):
            mod.audit(d, expected_receipt=d["receipt_sha256"])

    def test_recomputed_signature_cannot_hide_wrong_routing_count(self):
        d = sample_receipt()
        d["per_role"][0]["actual_clean_unseen_training_router_observations"] -= 1
        d["receipt_sha256"] = mod.canonical({k: v for k, v in d.items()
                                              if k != "receipt_sha256"})
        with self.assertRaisesRegex(ValueError, "CLEAN_ROUTING_COUNT_DRIFT"):
            mod.audit(d, expected_receipt=d["receipt_sha256"])

    def test_correct_signed_synthetic_ledger_not_misrepresented_as_real_run(self):
        d = sample_receipt()
        with self.assertRaisesRegex(ValueError, "SERVED_RESULT_DIFFERS_FROM_FINISHED_JOB_RECEIPT"):
            mod.audit(d, expected_receipt=d["receipt_sha256"])

    def test_frozen_model_and_h2_source_are_eight_experts_637_clean_events(self):
        self.assertEqual(sum(mod.CLEAN_COUNT.values()), 637)
        self.assertEqual(len(mod.FROZEN), 8)
        self.assertEqual(len(mod.ROLE_ORDER) * len(mod.FROZEN), 24)
        self.assertEqual(24 * mod.COST["AF4"], mod.BUDGET)


if __name__ == "__main__":
    unittest.main()
