#!/usr/bin/env python3
"""Run once AFTER both 96-case original-C calibration arms audit PASS.

Never reads Qv21 evaluation cohorts, Qv20 validation, or external data.
Accept only preregistered fivefold training. Freeze policy with create-only writes.
"""
import hashlib
import json
import os
import pathlib
import numpy as np
from qv21_calibration_gate import audit
from qv21_policy import crossfit, choose, features, read_spec, SPEC_SHA256
from qv21_native_static import ROOT, filehash


def canonical(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode()


def save_new(path, content):
    with path.open('xb') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    path.chmod(0o600)


def fit_once():
    out = ROOT / 'policy'
    if out.exists() or out.is_symlink():
        raise FileExistsError('QV21_POLICY_ALREADY_FROZEN')
    spec = read_spec()
    evidence = audit()  # Complete independent hash-bound native logs and all 96 cases per arm.
    if evidence['status'] != 'BOTH_QV21_ORIGINAL_C_CALIBRATION_ARMS_VERIFIED':
        raise RuntimeError('QV21_CALIBRATION_NOT_VERIFIED')
    rows = json.loads((ROOT / 'cal_arc_train96/frozen_inputs_private.json').read_text('utf8'))['selected_rows']
    scores = {}
    for name in ('static_source', 'static_random'):
        path = ROOT / 'native_original_C_static/cal_arc_train96' / name / 'PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json'
        scores[name] = np.asarray(json.loads(path.read_text('utf8'))['original_full_vocab_gold_nexttoken_NLL'])
    a, b = scores['static_source'], scores['static_random']
    gate, predictor, fold_selection = crossfit(rows, a, b, spec)
    predicted_full = choose(features(rows), predictor, spec)
    freeze = {
        'schema': 'BEGLIN-Qv21-tailrisk-prefix-only-regularized-ridge-logistic-frozen/1',
        'study_spec_SHA256': SPEC_SHA256,
        'source_preexec_public_sha256': evidence['public_preexec_sha256'],
        'calibration_only_actual_original_C_verified': True,
        'calibration_n': 96,
        'qv20_holdout_opened_at_fit': False,
        'qv21_evaluation_opened_at_fit': False,
        'candidate_projection_packed_bytes_each': spec['projection_packed_bytes_each'],
        'mask_sha256': spec['static_masks'],
        'native_original_C_sha256': spec['original_native_c_sha256'],
        'crossfit': gate,
        'predictor': predictor,
        'fitted_full_model_random73_count_on_calibration_inputs': int(predicted_full.sum()),
        'policy_dynamic_admitted': bool(gate['accepted']),
        'selected_active_mode': 'CALIBRATED_TAIL_RISK_ROUTER' if gate['accepted'] else 'CALIBRATION_BEST_STATIC',
        'static_fallback_index': gate['best_static_index'],
        'no_retraining_or_hyperparameter_tuning_from_evaluation': True,
        'no_Metal_or_total_RSS_or_crossfamily_claim': True,
    }
    os.umask(0o077)
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    private = out / 'calibration_policy_private.json'
    save_new(private, canonical(freeze))
    public = {k: v for k, v in freeze.items() if k not in ('predictor',)}
    public['private_policy_sha256'] = filehash(private)
    public['evaluation_unopened_when_frozen'] = True
    publicfile = out / 'Qv21_FROZEN_POLICY_PUBLIC.json'
    save_new(publicfile, canonical(public))
    print('QV21_CALIBRATION_POLICY_FROZEN', json.dumps({
        'admitted': gate['accepted'], 'selected_mode': freeze['selected_active_mode'],
        'crossfit_NLL_gain': gate['observed_crossfit_gain'],
        'bootstrap_lower': gate['bootstrap_95_lower_bound'],
        'selected_random': gate['selected_random73_count'],
        'severe_rate': gate['selected_severe_rate'],
        'private_policy_sha256': filehash(private),
        'public_policy_sha256': filehash(publicfile),
    }, sort_keys=True), flush=True)


if __name__ == '__main__':
    fit_once()
