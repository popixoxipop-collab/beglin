#!/usr/bin/env python3
"""B11 offline, veto-only Bayesian risk bridge for the frozen BEGLIN Qv21 selector.

NO native inference, production hot-path changes, Qv20/Qv21 heldout reads,
weight/Router changes, or implicit reuse of B9/B10 image-model calibration.
Fit, calibration and evaluation records must come from separately audited,
disjoint real C receipts; all in this module is pure and side-effect free.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Mapping

SCHEMA = "BEGLIN-B11-Qv21-StudentT-OneSided-Conformal-Veto/1"
FROZEN_QV21_SPEC_SHA256 = "408c661e66259784480b18eefc15f75d3bc7d034124194baaf3a218637e2f2d0"
ALPHA = 0.05
MIN_PER_STRATUM = 20
PRED_DELTA_LT = -0.20
PRED_SEVERE_LT = 0.15
MAX_PREFIX_LEN = 208
STRATUM_BOUNDARY = 85
PRIOR = {"mu0": 0.0, "kappa0": 1.0, "alpha0": 2.0, "beta0": 1.0}
HEX_SHA = re.compile(r"^[a-f0-9]{64}$")
ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or not HEX_SHA.fullmatch(value):
        raise ValueError(name + "_INVALID")
    return value


def _finite(value, name):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(name + "_INVALID")
    return float(value)


def _check_record(row, *, observed):
    if not isinstance(row, Mapping):
        raise ValueError("B11_RECORD_TYPE")
    expected = {"sample_id", "prefix_len", "predicted_delta"}
    if observed:
        expected.add("observed_delta")
    else:
        expected.add("predicted_severe_probability")
    if set(row) != expected:
        raise ValueError("B11_RECORD_FIELDS_FAIL_CLOSED")
    if not isinstance(row["sample_id"], str) or not ID_PATTERN.fullmatch(row["sample_id"]):
        raise ValueError("B11_SAMPLE_ID_INVALID")
    n = row["prefix_len"]
    if type(n) is not int or not 18 <= n <= MAX_PREFIX_LEN:
        raise ValueError("B11_PREFIX_LEN_INVALID")
    _finite(row["predicted_delta"], "B11_PREDICTION")
    if observed:
        _finite(row["observed_delta"], "B11_OBSERVED_DELTA")
    else:
        p = _finite(row["predicted_severe_probability"], "B11_RISK")
        if not 0 <= p <= 1:
            raise ValueError("B11_RISK_RANGE")
    return "short" if n < STRATUM_BOUNDARY else "long"


def _validated_rows(rows, *, observed):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("B11_ROWS_EMPTY")
    groups = {"short": [], "long": []}
    ids = set()
    for row in rows:
        group = _check_record(row, observed=observed)
        if row["sample_id"] in ids:
            raise ValueError("B11_DUPLICATE_SAMPLE_ID")
        ids.add(row["sample_id"])
        groups[group].append(row)
    return groups, ids


def _log_length(prefix_len):
    return math.log(prefix_len / STRATUM_BOUNDARY)


def _posterior(rows):
    """2x2 Bayesian residual regression with a Student-t predictive.

    Prior: coefficients | variance ~ N(0,variance*I2); variance ~ IG(2,1).
    Fixed basis (1,log(prefix_len/85)); no expert-to-expert covariance.
    """
    n = len(rows)
    a00, a01, a11 = 1.0 + n, 0.0, 1.0
    b0, b1, y2 = 0.0, 0.0, 0.0
    for row in rows:
        z = _log_length(row["prefix_len"])
        residual = row["observed_delta"] - row["predicted_delta"]
        a01 += z
        a11 += z*z
        b0 += residual
        b1 += z*residual
        y2 += residual*residual
    determinant = a00*a11 - a01*a01
    if determinant <= 0 or not math.isfinite(determinant):
        raise ValueError("B11_POSTERIOR_PRECISION")
    v00, v01, v11 = a11/determinant, -a01/determinant, a00/determinant
    c0, c1 = v00*b0+v01*b1, v01*b0+v11*b1
    unexplained = y2 - (b0*c0+b1*c1)
    if unexplained < -1e-8:
        raise ValueError("B11_NEGATIVE_RESIDUAL_ENERGY")
    alpha = PRIOR["alpha0"]+n/2
    beta = PRIOR["beta0"]+max(0.0, unexplained)/2
    return {"n":n, "coefficients":[c0,c1], "covariance":[v00,v01,v11],
            "alpha":alpha, "beta":beta, "df":2*alpha}


def _student_predict(posterior, prefix_len):
    z = _log_length(prefix_len)
    c0,c1 = posterior["coefficients"]
    v00,v01,v11 = posterior["covariance"]
    mean = c0+c1*z
    variance_multiplier = 1.0 + v00 + 2.0*v01*z + v11*z*z
    scale = math.sqrt(posterior["beta"]/posterior["alpha"] * variance_multiplier)
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("B11_STUDENT_PREDICTIVE_INVALID")
    return mean, scale


def _check_audit_ids(rows_ids, forbidden):
    blocked = set(forbidden)
    if rows_ids.intersection(blocked):
        raise ValueError("B11_FORBIDDEN_ID_REUSE")


def fit(fit_rows, *, policy_sha256, cohort_manifest_sha256, forbidden_ids=()):
    """Fit only on NEW B11 source data; no Qv21 calibration/evaluation data."""
    _sha(policy_sha256, "B11_QV21_POLICY_SHA")
    _sha(cohort_manifest_sha256, "B11_COHORT_SHA")
    groups, ids = _validated_rows(fit_rows, observed=True)
    _check_audit_ids(ids, forbidden_ids)
    if any(len(rows) < MIN_PER_STRATUM for rows in groups.values()):
        raise ValueError("B11_FIT_STRATUM_TOO_SMALL")
    state = {
        "schema": SCHEMA, "phase": "fit",
        "qv21_spec_sha256": FROZEN_QV21_SPEC_SHA256,
        "qv21_policy_sha256": policy_sha256,
        "cohort_manifest_sha256": cohort_manifest_sha256,
        "fit_ids": sorted(ids),
        "fit_ids_sha256": _digest(sorted(ids)),
        "posteriors": {
            name: _posterior(rs)
            for name, rs in groups.items()
        },
    }
    state["state_sha256"] = _digest(state)
    return state


def _check_state(state, phase):
    if not isinstance(state, Mapping) or state.get("schema") != SCHEMA or state.get("phase") != phase:
        raise ValueError("B11_STATE_PHASE_OR_SCHEMA")
    if state.get("qv21_spec_sha256") != FROZEN_QV21_SPEC_SHA256:
        raise ValueError("B11_QV21_PREREG_DRIFT")
    _sha(state.get("qv21_policy_sha256"), "B11_QV21_POLICY_SHA")
    clean = dict(state)
    signature = clean.pop("state_sha256", None)
    if signature != _digest(clean):
        raise ValueError("B11_STATE_HASH_MISMATCH")
    return state


def calibrate(fitted, calibration_rows, *, forbidden_ids=()):
    """Lock split-conformal upper residual ranks on disjoint calibration IDs."""
    _check_state(fitted, "fit")
    groups, ids = _validated_rows(calibration_rows, observed=True)
    if set(fitted["fit_ids"]).intersection(ids):
        raise ValueError("B11_FIT_CAL_OVERLAP")
    _check_audit_ids(ids, forbidden_ids)
    calibration = {}
    for name, rs in groups.items():
        n = len(rs)
        if n < MIN_PER_STRATUM:
            raise ValueError("B11_CAL_STRATUM_TOO_SMALL")
        post = fitted["posteriors"][name]
        scores = []
        for r in rs:
            mean, scale = _student_predict(post, r["prefix_len"])
            scores.append((r["observed_delta"] - r["predicted_delta"] - mean) / scale)
        scores.sort()
        k = math.ceil((n + 1) * (1 - ALPHA))
        if k > n:
            raise ValueError("B11_CAL_INFINITE_QUANTILE")
        calibration[name] = {"n": n, "order_statistic": k, "normalized_upper_q": scores[k - 1]}
    frozen = {key: val for key, val in fitted.items() if key != "state_sha256"}
    frozen.update(phase="frozen", calibration_ids=sorted(ids),
                  calibration_ids_sha256=_digest(sorted(ids)), calibration=calibration,
                  confidence=1 - ALPHA)
    frozen["state_sha256"] = _digest(frozen)
    return frozen


def predict(frozen, row, *, policy_sha256):
    """No gold, correct, observed_delta, or runtime label accepted."""
    _check_state(frozen, "frozen")
    if _sha(policy_sha256, "B11_QV21_POLICY_SHA") != frozen["qv21_policy_sha256"]:
        raise ValueError("B11_POLICY_BINDING_DRIFT")
    group = _check_record(row, observed=False)
    if row["sample_id"] in frozen["fit_ids"] or row["sample_id"] in frozen["calibration_ids"]:
        raise ValueError("B11_TRAIN_ID_AS_PREDICTION")
    p, delta = row["predicted_severe_probability"], row["predicted_delta"]
    prior_selection = delta < PRED_DELTA_LT and p < PRED_SEVERE_LT
    post = frozen["posteriors"][group]
    q = frozen["calibration"][group]["normalized_upper_q"]
    mean, scale = _student_predict(post, row["prefix_len"])
    upper = delta + mean + q * scale
    safe = bool(prior_selection and upper < PRED_DELTA_LT)
    return {"sample_id": row["sample_id"], "group": group,
            "baseline_qv21_select_random73": bool(prior_selection),
            "b11_shadow_select_random73": safe, "upper_delta": upper,
            "student_t_df": post["df"], "student_t_predictive_scale": scale,
            "vetoed": bool(prior_selection and not safe),
            "frozen_sha256": frozen["state_sha256"]}


def student_t_nll(observed_delta, predicted_delta, posterior, *, prefix_len):
    """Descriptive NLL, using sample-dependent 2x2 posterior predictive."""
    residual = _finite(observed_delta, "B11_OBSERVED") - _finite(predicted_delta, "B11_PREDICTED")
    if type(prefix_len) is not int or not 18 <= prefix_len <= MAX_PREFIX_LEN:
        raise ValueError("B11_PREFIX_LEN_INVALID")
    mean, scale = _student_predict(posterior, prefix_len)
    df = posterior["df"]
    z = (residual - mean) / scale
    return (math.lgamma(df / 2) - math.lgamma((df + 1) / 2)
            + 0.5 * math.log(df * math.pi) + math.log(scale)
            + (df + 1) / 2 * math.log1p(z * z / df))


def score_sealed_holdout(frozen, predictions, observed_by_id, *, policy_sha256):
    """Read-only evaluation; returns metrics, never changes thresholds or state."""
    if not isinstance(observed_by_id, dict) or len(predictions) != len(observed_by_id):
        raise ValueError("B11_HOLDOUT_PAIRING")
    counts = {"short": [0, 0], "long": [0, 0]}
    total_veto = total_base = total_new = severe_new = 0
    evaluated_ids = set()
    for row in predictions:
        result = predict(frozen, row, policy_sha256=policy_sha256)
        ident = result["sample_id"]
        if ident in evaluated_ids:
            raise ValueError("B11_HOLDOUT_DUPLICATE_ID")
        evaluated_ids.add(ident)
        if ident not in observed_by_id:
            raise ValueError("B11_HOLDOUT_ID_MISSING")
        actual = _finite(observed_by_id[ident], "B11_HOLDOUT_TRUTH")
        count = counts[result["group"]]
        count[1] += 1
        count[0] += int(actual <= result["upper_delta"])
        total_base += int(result["baseline_qv21_select_random73"])
        total_new += int(result["b11_shadow_select_random73"])
        total_veto += int(result["vetoed"])
        severe_new += int(result["b11_shadow_select_random73"] and actual >= 0.75)
    if evaluated_ids != set(observed_by_id) or sum(x[1] for x in counts.values()) != len(observed_by_id):
        raise RuntimeError("B11_HOLDOUT_TOTAL_MISMATCH")
    return {"schema": SCHEMA, "frozen_sha256": frozen["state_sha256"],
            "group_coverage": {g: {"covered": c[0], "n": c[1],
                                   "fraction": c[0] / c[1] if c[1] else None}
                               for g, c in counts.items()},
            "baseline_random73_count": total_base,
            "shadow_random73_count": total_new, "veto_count": total_veto,
            "shadow_severe_event_count": severe_new,
            "shadow_severe_event_fraction": severe_new / total_new if total_new else None,
            "no_real_model_claim": True}
