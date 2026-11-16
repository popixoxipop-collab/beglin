#!/usr/bin/env python3
"""B11 synthetic contract tests. NO model runs or access to Qv21 heldout data."""
import copy
import hashlib
import json
import math
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tools"))
import b11_bayesian_veto as b

def _synthetic_qv21_policy_pair(admitted=True):
    """Fixture-only, never the native Qv21 receipt."""
    private = {
        "schema": "BEGLIN-Qv21-tailrisk-prefix-only-regularized-ridge-logistic-frozen/1",
        "study_spec_SHA256": b.FROZEN_QV21_SPEC_SHA256,
        "source_preexec_public_sha256": "c" * 64,
        "calibration_only_actual_original_C_verified": True,
        "calibration_n": 96,
        "qv20_holdout_opened_at_fit": False,
        "qv21_evaluation_opened_at_fit": False,
        "candidate_projection_packed_bytes_each": b.FROZEN_PROJECTION_BYTES_EACH,
        "mask_sha256": b.FROZEN_MASK_SHA256,
        "native_original_C_sha256": b.FROZEN_NATIVE_C_SHA256,
        "static_fallback_index": 0,
        "crossfit": {"accepted": admitted, "best_static_index": 0,
                     "observed_crossfit_gain": 0.05 if admitted else 0.0,
                     "bootstrap_95_lower_bound": 0.001 if admitted else 0.0,
                     "selected_random73_count": 12 if admitted else 0,
                     "selected_severe_rate": 0.0 if admitted else 1.0,
                     "checks": {
            "minimum_crossfit_NLL_gain": admitted,
            "positive_bootstrap_lower_bound": admitted,
            "minimum_random73_uses": admitted,
            "bounded_selected_tail_event_frequency": admitted}},
        "predictor": {"means": [0.0] * 13, "scales": [1.0] * 13},
        "policy_dynamic_admitted": admitted,
        "selected_active_mode": (
            "CALIBRATED_TAIL_RISK_ROUTER" if admitted else "CALIBRATION_BEST_STATIC"
        ),
    }
    raw = (json.dumps(private, indent=2, sort_keys=True) + "\n").encode()
    public = {k: v for k, v in private.items() if k != "predictor"}
    public["private_policy_sha256"] = hashlib.sha256(raw).hexdigest()
    public["evaluation_unopened_when_frozen"] = True
    public_raw = (json.dumps(public, indent=2, sort_keys=True) + "\n").encode()
    return raw, public_raw


PRIVATE, PUBLIC = _synthetic_qv21_policy_pair()
POLICY = hashlib.sha256(PRIVATE).hexdigest()
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
                    policy_sha256=POLICY, cohort_manifest_sha256=MANIFEST,
                    qv21_private_policy_bytes=PRIVATE, qv21_public_policy_bytes=PUBLIC)
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
            self.assertGreater(b._student_predict(p, 44 if group == "short" else 124)[1], 0)
            self.assertTrue(math.isfinite(b.student_t_nll(-0.4, -0.3, p, prefix_len=44)))
            self.assertEqual(frozen["calibration"][group]["order_statistic"], 25)

    def test_reproducibility(self):
        x = model()
        y = model()
        self.assertEqual(x, y)
        self.assertNotEqual(x[0]["state_sha256"], x[1]["state_sha256"])

    def test_fit_cal_overlap_denied(self):
        fitted = b.fit(data("same"), policy_sha256=POLICY, cohort_manifest_sha256=MANIFEST,
                    qv21_private_policy_bytes=PRIVATE, qv21_public_policy_bytes=PUBLIC)
        with self.assertRaisesRegex(ValueError, "OVERLAP"):
            b.calibrate(fitted, data("same"))

    def test_forbidden_source_ids_denied(self):
        with self.assertRaisesRegex(ValueError, "FORBIDDEN"):
            b.fit(data("fit"), policy_sha256=POLICY, cohort_manifest_sha256=MANIFEST,
                    qv21_private_policy_bytes=PRIVATE, qv21_public_policy_bytes=PUBLIC,
                  forbidden_ids=["fit-short-1"])
        fitted = model()[0]
        with self.assertRaisesRegex(ValueError, "FORBIDDEN"):
            b.calibrate(fitted, data("cal"), forbidden_ids=["cal-long-2"])

    def test_no_real_receipts_no_shadow_policy(self):
        with self.assertRaises(ValueError):
            b.fit([], policy_sha256=POLICY, cohort_manifest_sha256=MANIFEST,
                    qv21_private_policy_bytes=PRIVATE, qv21_public_policy_bytes=PUBLIC)
        with self.assertRaisesRegex(ValueError, "STRATUM"):
            b.fit(data("few", n=19), policy_sha256=POLICY, cohort_manifest_sha256=MANIFEST,
                    qv21_private_policy_bytes=PRIVATE, qv21_public_policy_bytes=PUBLIC)

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
            b.fit(bad, policy_sha256=POLICY, cohort_manifest_sha256=MANIFEST,
                    qv21_private_policy_bytes=PRIVATE, qv21_public_policy_bytes=PUBLIC)
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

    def test_length_dependent_student_t_width_non_cancelling(self):
        # Same frozen calibration, same Qv21 prediction, different length => different bound.
        _, frozen = model()
        low = b.predict(frozen, row("novel-a", plen=44), policy_sha256=POLICY)
        edge = b.predict(frozen, row("novel-b", plen=84), policy_sha256=POLICY)
        self.assertNotAlmostEqual(low["student_t_predictive_scale"],
                                  edge["student_t_predictive_scale"], places=9)
        self.assertNotAlmostEqual(low["upper_delta"], edge["upper_delta"], places=9)

    def test_duplicate_holdout_ids_cannot_replace_unscored_sample(self):
        _, frozen = model()
        bad = [row("duplicate"), row("duplicate")]
        truth = {"duplicate": -0.3, "missing": 0.75}
        with self.assertRaisesRegex(ValueError, "DUPLICATE"):
            b.score_sealed_holdout(frozen, bad, truth, policy_sha256=POLICY)

    def test_raw_pooled_and_stratified_controls_are_frozen(self):
        _, frozen = model()
        self.assertEqual(frozen["calibration"]["raw_pooled"]["n"], 50)
        self.assertEqual(frozen["calibration"]["short"]["n"], 25)
        self.assertEqual(frozen["calibration"]["long"]["n"], 25)
        for plen, group in ((44, "short"), (124, "long")):
            qv21 = row("new-raw-" + group, plen=plen, delta=-0.35, risk=0.01)
            result = b.predict(frozen, qv21, policy_sha256=POLICY)
            self.assertAlmostEqual(result["raw_pooled_upper_delta"],
                                   -0.35 + frozen["calibration"]["raw_pooled"]["raw_upper_q"])
            self.assertAlmostEqual(result["raw_stratified_upper_delta"],
                                   -0.35 + frozen["calibration"][group]["raw_upper_q"])
            self.assertTrue(result["baseline_qv21_select_random73"])
            for key in ("raw_pooled_shadow_select_random73",
                        "raw_stratified_shadow_select_random73",
                        "b11_shadow_select_random73"):
                self.assertIn(result[key], (True, False))

    def test_raw_controls_never_create_selection_when_qv21_refuses(self):
        _, frozen = model()
        for plen in (18, 44, 84, 85, 124, 208):
            for delta, prob in ((-0.19, 0.01), (-0.30, 0.15),
                                (-0.10, 0.99)):
                out = b.predict(frozen, row("raw-%d-%s-%s" %
                                            (plen, delta, prob), plen=plen,
                                            delta=delta, risk=prob),
                                policy_sha256=POLICY)
                self.assertFalse(out["baseline_qv21_select_random73"])
                self.assertFalse(out["raw_pooled_shadow_select_random73"])
                self.assertFalse(out["raw_stratified_shadow_select_random73"])
                self.assertFalse(out["b11_shadow_select_random73"])

    def test_raw_holdout_ablation_report_is_immutable(self):
        _, frozen = model()
        before = copy.deepcopy(frozen)
        entries = [row("holdout-%d" % i, plen=44 if i % 2 == 0 else 124)
                   for i in range(20)]
        truth = {r["sample_id"]: -0.35 for r in entries}
        result = b.score_sealed_holdout(frozen, entries, truth, policy_sha256=POLICY)
        self.assertEqual(result["raw_controls"]["pooled"]["group_coverage"]["short"]["n"], 10)
        self.assertEqual(result["raw_controls"]["stratified"]["group_coverage"]["long"]["n"], 10)
        self.assertEqual(frozen, before)

    def test_raw_controls_detect_stratum_shift(self):
        fitted = b.fit(data("train"), policy_sha256=POLICY, cohort_manifest_sha256=MANIFEST,
                    qv21_private_policy_bytes=PRIVATE, qv21_public_policy_bytes=PUBLIC)
        calibration = data("recal")
        for r in calibration:
            if r["prefix_len"] < 85:
                r["observed_delta"] += 0.6
        frozen = b.calibrate(fitted, calibration)
        self.assertNotAlmostEqual(
            frozen["calibration"]["short"]["raw_upper_q"],
            frozen["calibration"]["long"]["raw_upper_q"])
        self.assertTrue(math.isfinite(frozen["calibration"]["raw_pooled"]["raw_upper_q"]))

    def test_admission_pair_rejected_when_qv21_fail_closed(self):
        rejected_priv, rejected_pub = _synthetic_qv21_policy_pair(admitted=False)
        with self.assertRaisesRegex(ValueError, "NATIVE_ADMISSION_REQUIRED"):
            b.verify_qv21_admission(rejected_priv, rejected_pub)

    def test_missing_native_policy_bytes_denied_at_fit(self):
        with self.assertRaisesRegex(ValueError, "POLICY_BYTES_REQUIRED"):
            b.fit(data("missing"), policy_sha256=POLICY,
                  cohort_manifest_sha256=MANIFEST)

    def test_public_private_policy_tamper_rejected(self):
        tampered_pub = json.loads(PUBLIC)
        tampered_pub["private_policy_sha256"] = "0" * 64
        malformed = (json.dumps(tampered_pub, sort_keys=True) + "\n").encode()
        with self.assertRaisesRegex(ValueError, "PUBLIC_PRIVATE_MISMATCH"):
            b.verify_qv21_admission(PRIVATE, malformed)

    def test_accepted_policy_must_bind_exact_private_sha(self):
        wrong = "d" * 64
        with self.assertRaisesRegex(ValueError, "PROOF_SHA_MISMATCH"):
            b.fit(data("mismatch"), policy_sha256=wrong,
                  cohort_manifest_sha256=MANIFEST,
                  qv21_private_policy_bytes=PRIVATE,
                  qv21_public_policy_bytes=PUBLIC)

    def test_admission_evidence_survives_freeze(self):
        _, state = model()
        self.assertTrue(state["qv21_admission"]["qv21_native_calibration_admitted"])
        self.assertEqual(state["qv21_admission"]["private_policy_sha256"], POLICY)
        self.assertEqual(state["qv21_admission"]["original_c_sha256"],
                         b.FROZEN_NATIVE_C_SHA256)
        self.assertTrue(b.predict(state, row(), policy_sha256=POLICY))

    def test_original_source_unchanged_contract(self):
        self.assertIn("Qv21", b.__doc__)
        self.assertEqual(b.PRED_SEVERE_LT, 0.15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
