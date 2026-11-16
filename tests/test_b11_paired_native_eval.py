#!/usr/bin/env python3
"""Synthetic-only B11 paired native evaluator contract, no original C running."""
import copy
import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import b11_paired_native_eval as ev
import test_b11_bayesian_veto as fx


def fixture(n=20, *, predicted_delta=-0.35, random_nll=0.65, source_nll=1.0,
            probability=0.01):
    out = []
    for i in range(n):
        out.append({"sample_id": "paired_%03d" % i,
                    "prefix_len": 44 if i % 2 == 0 else 124,
                    "predicted_delta": predicted_delta,
                    "predicted_severe_probability": probability,
                    "nll_source": source_nll, "nll_random": random_nll})
    return out


class B11PairContracts(unittest.TestCase):
    def setUp(self):
        _, self.frozen = fx.model()

    def evaluate(self, rows=None):
        return ev.score_paired_rows(self.frozen, fixture() if rows is None else rows,
                                    split="in_domain", policy_sha256=fx.POLICY,
                                    synthetic_fixture=True, bootstrap_reps=80)

    def test_paired_metrics_and_controls(self):
        result = self.evaluate()
        self.assertEqual(result["n"], 20)
        self.assertEqual(result["data_status"], "SYNTHETIC_FIXTURE")
        self.assertFalse(result["production_promotion_authorized"])
        self.assertFalse(result["native_provenance_independently_verified_here"])
        self.assertAlmostEqual(result["nll_means"]["source"], 1.0)
        self.assertAlmostEqual(result["nll_means"]["random"], 0.65)
        self.assertEqual(result["selection_harm"]["qv21"]["n_selected"], 20)
        self.assertIn("pooled", result["raw_controls"])
        self.assertIn("stratified", result["raw_controls"])
        self.assertEqual(result["paired_nll_gain_vs_best_static"]["paired_bootstrap_reps"], 80)

    def test_evaluation_does_not_change_frozen_state(self):
        before = copy.deepcopy(self.frozen)
        a = self.evaluate()
        b = self.evaluate()
        self.assertEqual(a, b)
        self.assertEqual(self.frozen, before)

    def test_veto_only_when_original_qv21_refuses(self):
        result = self.evaluate(fixture(predicted_delta=-0.19))
        self.assertEqual(result["selection_harm"]["qv21"]["n_selected"], 0)
        self.assertEqual(result["selection_harm"]["b11_student_t"]["n_selected"], 0)
        self.assertFalse(result["candidate_science_gates_all_pass"])

    def test_student_t_veto_on_large_residual_shift(self):
        _, frozen = fx.model(fit_shift=1.0, cal_shift=1.0)
        self.frozen = frozen
        report = self.evaluate()
        self.assertEqual(report["selection_harm"]["qv21"]["n_selected"], 20)
        self.assertEqual(report["selection_harm"]["b11_student_t"]["n_selected"], 0)
        self.assertFalse(report["candidate_science_gates_all_pass"])

    def test_missing_native_question_pair_denied(self):
        bad = fixture()
        del bad[-1]["nll_random"]
        with self.assertRaisesRegex(ValueError, "FIELDS"):
            self.evaluate(bad)

    def test_bad_nll_and_outcome_in_predictor_denied(self):
        for loss in (float("nan"), float("inf"), -0.01):
            bad = fixture()
            bad[4]["nll_source"] = loss
            with self.assertRaises(ValueError):
                self.evaluate(bad)

    def test_duplicate_ids_denied(self):
        bad = fixture()
        bad[-1]["sample_id"] = bad[0]["sample_id"]
        with self.assertRaisesRegex(ValueError, "DUPLICATE"):
            self.evaluate(bad)

    def test_real_split_count_held(self):
        with self.assertRaisesRegex(ValueError, "REAL_SPLIT_COUNT"):
            ev.score_paired_rows(self.frozen, fixture(), split="in_domain",
                                 policy_sha256=fx.POLICY,
                                 synthetic_fixture=False, bootstrap_reps=ev.REPS)

    def test_real_bootstrap_20k_held(self):
        with self.assertRaisesRegex(ValueError, "REAL_BOOTSTRAP_CONTRACT"):
            ev.score_paired_rows(self.frozen, fixture(), split="ood",
                                 policy_sha256=fx.POLICY, synthetic_fixture=False,
                                 bootstrap_reps=80)

    def test_paired_ci_exact_and_signed(self):
        z = ev.paired_gain([1.0] * 10, [0.8] * 10, reps=80)
        self.assertAlmostEqual(z["mean_nll_gain"], 0.2)
        self.assertAlmostEqual(z["paired_bootstrap_95"][0], 0.2)
        self.assertAlmostEqual(z["paired_bootstrap_95"][1], 0.2)

    def test_provenance_not_self_certifying(self):
        result = self.evaluate()
        self.assertNotIn("NATIVE_C_AUDITED_PASS", result.values())
        self.assertFalse(result["production_promotion_authorized"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
