#!/usr/bin/env python3
"""B11 paired native C metrics, computed only from auditable records.

This module cannot verify original-native execution by itself. It never
trains, retunes, or mutates the frozen B11/Qv21 policies. Output is a
candidate score, NOT sufficient evidence for model/operating promotion.
"""
from __future__ import annotations

import math
import random

import b11_bayesian_veto as b

FIELDS = {
    "sample_id", "prefix_len", "predicted_delta", "predicted_severe_probability",
    "nll_source", "nll_random",
}
ARMS = (
    "source", "random", "best_static", "qv21",
    "b11_student_t", "b11_raw_pooled", "b11_raw_stratified",
)
EXPECTED_SPLITS = {"in_domain": {"n": 80, "short": 40, "long": 40},
                   "ood": {"n": 120, "short": 60, "long": 60}}
REPS = 20000
SEED = 20261117


def _percentile(x, p):
    s = sorted(x)
    if not s:
        raise ValueError("B11_BOOTSTRAP_EMPTY")
    q = (len(s) - 1) * p
    lo, hi = int(math.floor(q)), int(math.ceil(q))
    return s[lo] + (s[hi] - s[lo]) * (q - lo)


def paired_gain(ref, candidate, *, reps=REPS, seed=SEED):
    if not isinstance(ref, list) or not isinstance(candidate, list):
        raise ValueError("B11_PAIRED_LOSS_TYPE")
    if not ref or len(ref) != len(candidate):
        raise ValueError("B11_PAIRED_LOSS_LENGTH")
    diffs = [b._finite(a, "B11_PAIRED_REF") -
             b._finite(v, "B11_PAIRED_CHALLENGER") for a, v in zip(ref, candidate)]
    n = len(diffs)
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        draws.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    return {"mean_nll_gain": sum(diffs) / n,
            "paired_bootstrap_95": [_percentile(draws, .025), _percentile(draws, .975)],
            "paired_bootstrap_reps": reps}


def score_paired_rows(frozen, rows, *, split, policy_sha256,
                      synthetic_fixture=False, bootstrap_reps=REPS):
    """Pair-wise actual NLL evaluation; thresholds remain frozen.

    Real score contract: exactly 80/120 IDs with 40/40 or 60/60 strata.
    Synthetic fixtures may use shorter inputs and fewer bootstrap reps;
    never report their metrics as real native measurements.
    """
    b._check_state(frozen, "frozen")
    if split not in EXPECTED_SPLITS:
        raise ValueError("B11_SPLIT_NAME")
    if type(synthetic_fixture) is not bool:
        raise ValueError("B11_SYNTHETIC_FLAG")
    if not isinstance(rows, list) or not rows:
        raise ValueError("B11_PAIRED_ROWS_EMPTY")
    if type(bootstrap_reps) is not int or bootstrap_reps < 50:
        raise ValueError("B11_BOOTSTRAP_REPS")
    if not synthetic_fixture and bootstrap_reps != REPS:
        raise ValueError("B11_REAL_BOOTSTRAP_CONTRACT")
    if not synthetic_fixture and len(rows) != EXPECTED_SPLITS[split]["n"]:
        raise ValueError("B11_REAL_SPLIT_COUNT")
    base_policy = frozen["qv21_admission"]
    if base_policy["qv21_native_calibration_admitted"] is not True:
        raise RuntimeError("B11_QV21_ADMISSION_NOT_VERIFIED")
    baseline_static = base_policy["qv21_static_fallback_index"]
    if baseline_static not in (0, 1) or type(baseline_static) is not int:
        raise ValueError("B11_STATIC_FALLBACK_INVALID")

    by_arm = {arm: [] for arm in ARMS}
    group_counts = {"short": 0, "long": 0}
    group_includes = {"short": 0, "long": 0}
    selected_counts = {arm: 0 for arm in ARMS if arm not in ("source", "random")}
    harm_counts = {arm: 0 for arm in selected_counts}
    seen = set()
    predictions = []
    observed_by_id = {}

    for record in rows:
        if not isinstance(record, dict) or set(record) != FIELDS:
            raise ValueError("B11_PAIRED_ROW_FIELDS")
        ident = record.get("sample_id")
        if ident in seen:
            raise ValueError("B11_PAIRED_DUPLICATE_ID")
        seen.add(ident)
        a = b._finite(record["nll_source"], "B11_NATIVE_NLL_SOURCE")
        c = b._finite(record["nll_random"], "B11_NATIVE_NLL_RANDOM")
        if a < 0 or c < 0:
            raise ValueError("B11_NATIVE_NLL_NEGATIVE")
        predictor_input = {key: record[key] for key in
                           ("sample_id", "prefix_len", "predicted_delta",
                            "predicted_severe_probability")}
        pred = b.predict(frozen, predictor_input, policy_sha256=policy_sha256)
        delta = c - a
        group = pred["group"]
        group_counts[group] += 1
        group_includes[group] += int(delta <= pred["upper_delta"])
        predicted_flags = {
            "best_static": baseline_static == 1,
            "qv21": pred["baseline_qv21_select_random73"],
            "b11_student_t": pred["b11_shadow_select_random73"],
            "b11_raw_pooled": pred["raw_pooled_shadow_select_random73"],
            "b11_raw_stratified": pred["raw_stratified_shadow_select_random73"],
        }
        by_arm["source"].append(a)
        by_arm["random"].append(c)
        for arm, selected in predicted_flags.items():
            by_arm[arm].append(c if selected else a)
            selected_counts[arm] += int(selected)
            harm_counts[arm] += int(selected and delta >= 0.75)
        predictions.append(predictor_input)
        observed_by_id[ident] = delta

    if not synthetic_fixture:
        for group in ("short", "long"):
            if group_counts[group] != EXPECTED_SPLITS[split][group]:
                raise ValueError("B11_REAL_STRATUM_COUNT")

    base_report = b.score_sealed_holdout(
        frozen, predictions, observed_by_id, policy_sha256=policy_sha256)
    loss_means = {arm: sum(vals) / len(vals) for arm, vals in by_arm.items()}
    select_rate = {
        arm: {"n_selected": selected_counts[arm],
              "severe_count": harm_counts[arm],
              "severe_rate": harm_counts[arm] / selected_counts[arm]
              if selected_counts[arm] else None}
        for arm in selected_counts
    }
    gain_vs_qv21 = paired_gain(by_arm["qv21"], by_arm["b11_student_t"],
                               reps=bootstrap_reps, seed=SEED)
    gain_vs_static = paired_gain(by_arm["best_static"], by_arm["b11_student_t"],
                                reps=bootstrap_reps, seed=SEED + 1)
    gates = {
        "qv21_native_admitted": True,
        "b11_veto_only": selected_counts["b11_student_t"] <= selected_counts["qv21"],
        "minimum_random73_uses": selected_counts["b11_student_t"] >= 12,
        "selected_severe_harm_bounded": (
            select_rate["b11_student_t"]["severe_rate"] is not None and
            select_rate["b11_student_t"]["severe_rate"] <= 0.10),
        "each_stratum_coverage95": all(
            group_counts[g] > 0 and group_includes[g] / group_counts[g] >= 0.95
            for g in ("short", "long")),
        "no_regression_vs_frozen_qv21": gain_vs_qv21["mean_nll_gain"] >= 0,
        "positive_best_static_bootstrap_lower": gain_vs_static["paired_bootstrap_95"][0] > 0,
    }
    return {
        "schema": "BEGLIN-B11-Qv21-paired-native-c-shadow-evaluation/1",
        "split": split, "n": len(rows),
        "frozen_state_sha256": frozen["state_sha256"],
        "qv21_policy_sha256": policy_sha256,
        "data_status": "SYNTHETIC_FIXTURE" if synthetic_fixture else
                       "UNVERIFIED_INPUT_UNTIL_EXTERNAL_NATIVE_RECEIPT_AUDIT",
        "production_promotion_authorized": False,
        "native_provenance_independently_verified_here": False,
        "candidate_science_gates": gates,
        "candidate_science_gates_all_pass": all(gates.values()),
        "group_coverage": base_report["group_coverage"],
        "raw_controls": base_report["raw_controls"],
        "nll_means": loss_means,
        "selection_harm": select_rate,
        "paired_nll_gain_vs_qv21": gain_vs_qv21,
        "paired_nll_gain_vs_best_static": gain_vs_static,
    }
