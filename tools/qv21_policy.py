#!/usr/bin/env python3
"""BEGLIN Qv21 precommitted prefix-only risk-aware selector.

Pure feature extraction and prediction. No Qv20 heldout I/O, model inference,
network operations, or mutable deployment side effects.
"""
from __future__ import annotations
import hashlib
import json
import math
import pathlib
import numpy as np
from sklearn.linear_model import LogisticRegression

SPEC_SHA256 = '408c661e66259784480b18eefc15f75d3bc7d034124194baaf3a218637e2f2d0'
SPEC_PATH = pathlib.Path(__file__).resolve().parent.parent / 'qv21_spec.json'
FEATURES = (
    'log_prefix_over85', 'unique_fraction', 'high100k_fraction',
    'mean_token_id_fraction', 'adjacent_repeat_fraction', 'low256_fraction',
    'mid256_2048_fraction', 'normalized_token_std', 'bucket16_entropy',
    'half_means_gap', 'large_adjacent_jump_fraction', 'distance2_repeat_fraction',
    'longest_run_fraction',
)


def read_spec():
    raw = SPEC_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPEC_SHA256:
        raise RuntimeError('QV21_SPEC_SHA_DRIFT')
    obj = json.loads(raw)
    if obj.get('schema') != 'BEGLIN-Qv21-preregistered-tail-risk-two-fixed-Q4N8-masks/1':
        raise RuntimeError('QV21_SPEC_SCHEMA_DRIFT')
    return obj


def feature_vector(ids):
    """13 finite, dimensionless values from user-visible prefix IDs only."""
    n = len(ids)
    if not 18 <= n <= 208:
        raise ValueError('QV21_PREFIX_LENGTH_INVALID')
    if any(type(t) is not int or not 0 <= t < 151936 for t in ids):
        raise ValueError('QV21_TOKEN_RANGE_INVALID')
    a = np.asarray(ids, dtype=np.float64)
    pairs = a[1:] - a[:-1]
    hist = np.bincount(np.minimum(15, (a * 16 // 151936).astype(int)), minlength=16)
    p = hist[hist > 0] / n
    entropy = float(-(p * np.log(p)).sum() / math.log(16))
    longest = cur = 1
    for i in range(1, n):
        cur = cur + 1 if ids[i] == ids[i-1] else 1
        longest = max(longest, cur)
    mid = n // 2
    return np.array([
        math.log(n / 85), len(set(ids)) / n, float(np.mean(a >= 100000)),
        float(np.mean(a) / 151936), float(np.mean(pairs == 0)),
        float(np.mean(a < 256)), float(np.mean((a >= 256) & (a < 2048))),
        float(np.std(a) / 151936), entropy,
        float(abs(np.mean(a[:mid]) - np.mean(a[mid:])) / 151936),
        float(np.mean(np.abs(pairs) >= 32768)),
        float(np.mean(a[2:] == a[:-2])), longest / n,
    ], dtype=float)


def features(rows):
    x = np.stack([feature_vector(row['prefix_ids']) for row in rows])
    if x.shape != (len(rows), 13) or not np.all(np.isfinite(x)):
        raise RuntimeError('QV21_FEATURES_SHAPE_OR_FINITE')
    return x


def fold_assignments(rows, folds=5):
    """Freeze selects short/long alternately; order within stratum is label-free SHA."""
    if len(rows) != 96 or folds != 5:
        raise ValueError('QV21_CV_SHAPE_INVALID')
    lengths = [r['prefix_len'] for r in rows]
    if any((v < 85) != (i % 2 == 0) for i, v in enumerate(lengths)):
        raise ValueError('QV21_CV_NONINTERLEAVED_STRATA')
    a = np.array([(i // 2) % folds for i in range(len(rows))], dtype=int)
    if any(np.count_nonzero(a == j) < 16 for j in range(folds)):
        raise RuntimeError('QV21_CV_FOLD_TOO_SMALL')
    return a


def fit_models(x, delta, spec):
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or len(x) != len(delta) or len(x) < 32:
        raise ValueError('QV21_FIT_SHAPE')
    if not np.isfinite(x).all() or not np.isfinite(delta).all():
        raise ValueError('QV21_FIT_NONFINITE')
    mu = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), spec['std_floor'])
    z = (x - mu) / scale
    y = np.clip(delta, *spec['ridge_label_clip'])
    z1 = np.column_stack([np.ones(len(x)), z])
    penalty = np.eye(z1.shape[1]) * spec['ridge_alpha']
    penalty[0, 0] = 0
    coef = np.linalg.solve(z1.T @ z1 + penalty, z1.T @ y)
    severe = (delta >= spec['catastrophic_delta_ge']).astype(int)
    if len(set(severe)) == 1:
        risk = {'kind': 'constant', 'probability': (int(severe.sum()) + 1) / (len(severe) + 2)}
    else:
        cls = LogisticRegression(C=spec['logistic_C'], solver=spec['logistic_solver'],
                                 max_iter=spec['logistic_max_iter'], random_state=20261025)
        cls.fit(z, severe)
        if list(cls.classes_) != [0, 1]:
            raise RuntimeError('QV21_CLASS_ORDER_DRIFT')
        risk = {'kind': 'logistic', 'intercept': float(cls.intercept_[0]),
                'coefficients': cls.coef_[0].astype(float).tolist()}
    return {'means': mu.tolist(), 'scales': scale.tolist(), 'ridge_coefficients': coef.tolist(),
            'risk_model': risk, 'features': list(FEATURES)}


def predict_risk(x, policy):
    mu = np.asarray(policy['means']); scale = np.asarray(policy['scales'])
    z = (x - mu) / scale
    if z.ndim == 1:
        z = z.reshape(1, -1)
    delta = policy['ridge_coefficients'][0] + z @ np.asarray(policy['ridge_coefficients'][1:])
    risk = policy['risk_model']
    if risk['kind'] == 'constant':
        prob = np.full(len(z), risk['probability'], dtype=float)
    else:
        logits = risk['intercept'] + z @ np.asarray(risk['coefficients'])
        prob = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
    if not np.isfinite(delta).all() or not np.isfinite(prob).all():
        raise RuntimeError('QV21_PREDICTION_INVALID')
    return np.asarray(delta), np.asarray(prob)


def choose(x, policy, spec):
    pred_delta, pred_risk = predict_risk(x, policy)
    return (pred_delta < spec['switch_predicted_delta_lt']) & (
        pred_risk < spec['switch_predicted_severe_probability_lt'])


def evaluate_gate(a, b, selected_random, spec):
    """Inputs are calibration-only paired real C scores; never use evaluation rows."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    chosen = np.asarray(selected_random, bool)
    if a.shape != b.shape or a.shape != chosen.shape or a.shape != (96,):
        raise ValueError('QV21_GATE_96_EXACT')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any():
        raise ValueError('QV21_GATE_LOSS_INVALID')
    best_index = int(b.mean() < a.mean())
    best_static = b if best_index == 1 else a
    router = np.where(chosen, b, a)
    diff = best_static - router
    gain = float(diff.mean())
    rng = np.random.default_rng(spec['crossfit_bootstrap_seed'])
    draws = rng.integers(0, 96, (spec['crossfit_bootstrap_reps'], 96))
    lower = float(np.percentile(diff[draws].mean(axis=1), spec['crossfit_bootstrap_lower_percentile']))
    random_count = int(chosen.sum())
    severe_count = int(((b - a >= spec['catastrophic_delta_ge']) & chosen).sum())
    frac = severe_count / random_count if random_count else 1.0
    checks = {
        'minimum_crossfit_NLL_gain': gain >= spec['crossfit_min_nll_gain_vs_better_static'],
        'positive_bootstrap_lower_bound': lower > 0,
        'minimum_random73_uses': random_count >= spec['crossfit_random_selection_min'],
        'bounded_selected_tail_event_frequency': frac <= spec['crossfit_severe_harm_selected_max_fraction'],
    }
    return {
        'accepted': all(checks.values()), 'checks': checks,
        'best_static_index': best_index, 'observed_crossfit_gain': gain,
        'bootstrap_95_lower_bound': lower, 'selected_random73_count': random_count,
        'selected_severe_count': severe_count, 'selected_severe_rate': frac,
        'mean_source_NLL': float(a.mean()), 'mean_random_NLL': float(b.mean()),
        'mean_crossfit_NLL': float(router.mean()),
    }


def crossfit(rows, a, b, spec):
    x = features(rows)
    folds = fold_assignments(rows, spec['fold_count'])
    delta = np.asarray(b, float) - np.asarray(a, float)
    selected = np.zeros(len(rows), dtype=bool)
    for fold in range(5):
        tr = folds != fold
        te = ~tr
        model = fit_models(x[tr], delta[tr], spec)
        selected[te] = choose(x[te], model, spec)
    result = evaluate_gate(a, b, selected, spec)
    final = fit_models(x, delta, spec)
    return result, final, selected
