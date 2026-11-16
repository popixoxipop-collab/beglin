#!/usr/bin/env python3
"""B11 synthetic contract tests. NO model runs or access to Qv21 heldout data."""
import copy
import math
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tools"))
import b11_bayesian_veto as b

POLICY = "a" * 64
MANIFEST = "b" * 64


def data(prefix, n=25, shift=0.0):
    rows = []
    for group, plen in (("short", 44), ("long", 124)):
        for i in range(n):
            rows.append({"sample_id": f"{prefix}-{group}-{i}",
                         "prefix_len": plen,
                         "predicted_delta": -0.35,
                         "observed_delta": -0.35 + shift + (i % 5 - 2) * 0.01})
    return rows


def model(fit_shift=0.0, cal_shift=0.0):
    trained = b.fit(data("fit", shift=fit_shift),
                    policy_sha256=POLICY, cohort_manifest_sha256=MANIFEST)
    frozen = b.calibrate(trained, data("cal", shift=cal_shift))
    return trained, frozen


def row(sample="eval-1", plen=44, delta=-0.35, risk=0.01):
    return {"sample_id": sample, "prefix_len": plen,
            "predicted_delta": delta, "predicted_severe_probability": risk}


class B11Contracts(unittest.TestCase):
    def test_schema_pins_preregistered_qv21(self):
        self.assertEqual(b.FROZEN_QV21_SPEC_SHA256,
                         "408c661e66259784480b18eefc15f75d3bc7d034124194baaf3a218637e2f2d0")
        self.assertEqual(b.ALPHA, 0.05)
        self.assertEqual(b.PRED_DELTA_LT, -0.20)

    def test_student_t_posterior_strata(self):
        trained, frozen = model()
        for group in ("short", "long"):
            p = trained["posteriors"][group]
            self.assertEqual(p["n"], 25)
            self.assertEqual(p["df"], 29)
            self.assertGreater(p["predictive_scale"], 0)
            self.assertTrue(math.isfinite(b.student_t_nll(-0.4, -0.3, p)))
            self.assertEqual(frozen["calibration"][group]["order_statistic"], 25)

    def test_reproducibility(self):
        x = model()
        y = model()
        self.assertEqual(x, y)
        self.assertNotEqual(x[0]["state_sha256"], x[1]["state_sha256"])

    def test_fit_cal_overlap_denied(self):
        fitted = b.fit(data("same"), policy_sha256=POLICY, cohort_manifest_sha256=MANIFEST)
        with self.assertRaisesRegex(ValueError, "OVERLAP"):
            b.calibrate(fitted, data("same"))

    def test_forbidden_source_ids_denied(self):
        with self.assertRaisesRegex(ValueError, "FORBIDDEN"):
            b.fit(data("fit"), policy_sha256=POLICY, cohort_manifest_sha256=MANIFEST,
                  forbidden_ids=["fit-short-1"])
        fitted = model()[0]
        with self.assertRaisesRegex(ValueError, "FORBIDDEN"):
            b.calibrate(fitted, data("cal"), forbidden_ids=["cal-long-2"])

    def test_no_real_receipts_no_shadow_policy(self):
        with self.assertRaises(ValueError):
            b.fit([], policy_sha256=POLICY, cohort_manifest_sha256=MANIFEST)
        with self.assertRaisesRegex(ValueError, "STRATUM"):
            b.fit(data("few", n=19), policy_sha256=POLICY, cohort_manifest_sha256=MANIFEST)

    def test_strict_prediction_label_firewall(self):
        _, frozen = model()
        sample = row()
        sample["observed_delta"] = -0.5
        with self.assertRaisesRegex(ValueError, "FIELDS"):
            b.predict(frozen, sample, policy_sha256=POLICY)
        sample.pop("observed_delta")
        sample["correct"] = "A"
        with self.assertRaisesRegex(ValueError, "FIELDS"):
            b.predict(frozen, sample, policy_sha256=POLICY)

    def test_id_reuse_and_policy_drift_denied(self):
        _, frozen = model()
        with self.assertRaisesRegex(ValueError, "TRAIN_ID"):
            b.predict(frozen, row("fit-short-1"), policy_sha256=POLICY)
        with self.assertRaisesRegex(ValueError, "POLICY_BINDING"):
            b.predict(frozen, row(), policy_sha256="c" * 64)

    def test_tampered_frozen_denied(self):
        _, frozen = model()
        bad = copy.deepcopy(frozen)
        bad["calibration"]["short"]["normalized_upper_q"] -= 100.0
        with self.assertRaisesRegex(ValueError, "HASH"):
            b.predict(bad, row(), policy_sha256=POLICY)

    def test_veto_only_not_more_random_choices(self):
        _, frozen = model()
        for d, p in ((-0.35, 0.01), (-0.19, 0.01), (-0.35, 0.15), (-0.35, 0.95)):
            result = b.predict(frozen, row(delta=d, risk=p), policy_sha256=POLICY)
            if result["b11_shadow_select_random73"]:
                self.assertTrue(result["baseline_qv21_select_random73"])
        self.assertFalse(b.predict(frozen, row(delta=-0.19), policy_sha256=POLICY)
                         ["b11_shadow_select_random73"])

    def test_shifted_residual_vetoes_baseline(self):
        _, frozen = model(fit_shift=1.0, cal_shift=1.0)
        r = b.predict(frozen, row(), policy_sha256=POLICY)
        self.assertTrue(r["baseline_qv21_select_random73"])
        self.assertTrue(r["vetoed"])
        self.assertFalse(r["b11_shadow_select_random73"])

    def test_small_residual_allows_baseline(self):
        _, frozen = model()
        r = b.predict(frozen, row(), policy_sha256=POLICY)
        self.assertTrue(r["baseline_qv21_select_random73"])
        self.assertTrue(r["b11_shadow_select_random73"])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            b.fit(data("fit"), policy_sha256="not-a-sha", cohort_manifest_sha256=MANIFEST)
        bad = data("fit")
        bad[0]["observed_delta"] = float("nan")
        with self.assertRaises(ValueError):
            b.fit(bad, policy_sha256=POLICY, cohort_manifest_sha256=MANIFEST)
        _, frozen = model()
        with self.assertRaises(ValueError):
            b.predict(frozen, row(risk=math.inf), policy_sha256=POLICY)

    def test_prediction_and_evaluation_are_separate(self):
        _, frozen = model()
        before = copy.deepcopy(frozen)
        entries = [row(f"new-short-{i}", delta=-0.35, risk=0.01)
                   for i in range(10)]
        truth = {f"new-short-{i}": -0.35 + (i % 5 - 2) * 0.01
                 for i in range(10)}
        report = b.score_sealed_holdout(frozen, entries, truth, policy_sha256=POLICY)
        self.assertEqual(report["group_coverage"]["short"]["n"], 10)
        self.assertGreaterEqual(report["group_coverage"]["short"]["covered"], 9)
        self.assertEqual(frozen, before)

    def test_holdout_pairing_rejected(self):
        _, frozen = model()
        with self.assertRaises(ValueError):
            b.score_sealed_holdout(frozen, [row("a")], {}, policy_sha256=POLICY)

    def test_original_source_unchanged_contract(self):
        self.assertIn("Qv21", b.__doc__)
        self.assertEqual(b.PRED_SEVERE_LT, 0.15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
