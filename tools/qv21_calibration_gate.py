#!/usr/bin/env python3
"""Independent Qv21 original C calibration audit; read-only and fail-closed.

Checks 96 per-case gold-token NLLs, role masks, C binary, input SHA,
log-to-receipt alignment, and private/public binding for BOTH static arms.
Never opens Qv21 evaluation inputs or Qv20 validation inputs.
"""
import hashlib
import json
import math
import pathlib
import re
from qv21_policy import read_spec
from qv21_native_static import EXE, ROOT, R_ROW, R_SUM, filehash

SCHEMA = 'BEGLIN-Qv21-actual-original-Qwen28-C-single-static-mask-case-scores/1'


def require(condition, reason):
    if not condition:
        raise RuntimeError('QV21_CALIBRATION_AUDIT_FAIL_CLOSED_' + reason)


def audit():
    spec = read_spec()
    pub_path = ROOT / 'Qv21_PREEXEC_PUBLIC.json'
    public = json.loads(pub_path.read_text('utf8'))
    require(public.get('state') == 'FROZEN_BEFORE_ANY_QV21_NATIVE_SCORING', 'PREEXEC_NOT_FROZEN')
    require(filehash(EXE) == spec['original_native_c_sha256'], 'ORIGINAL_C_DRIFT')
    manifest = ROOT / 'cal_arc_train96/frozen_inputs_private.json'
    require(public['frozen_private_input_SHA256']['cal_arc_train96'] == filehash(manifest), 'PRIVATE_CAL_SHA')
    require(public['study_spec_SHA256'] == hashlib.sha256(
        (pathlib.Path(__file__).resolve().parent.parent / 'qv21_spec.json').read_bytes()).hexdigest(),
        'STUDY_SPEC_PIN')
    rows = json.loads(manifest.read_text('utf8'))['selected_rows']
    require(len(rows) == 96, 'CAL_ROWS_96')
    outcome = {}
    for arm in ('static_source', 'static_random'):
        require(filehash(ROOT / 'masks' / (arm + '.txt')) == spec['static_masks'][arm], 'MASK_DRIFT_' + arm)
        folder = ROOT / 'native_original_C_static/cal_arc_train96' / arm
        privpath = folder / 'PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json'
        pubpath = folder / 'PUBLIC_AGGREGATE.json'
        logpath = folder / 'native_original_C_Qwen28.log'
        d = json.loads(privpath.read_text('utf8'))
        require(d.get('schema') == SCHEMA and
                d.get('state') == 'ACTUAL_ORIGINAL_QWEN28_C_STATIC_' + arm.upper() + '_PASS' and
                d.get('split') == 'cal_arc_train96' and d.get('arm') == arm and
                type(d.get('n')) is int and d['n'] == 96 and
                d.get('original_Qwen28_C_executable_sha256') == spec['original_native_c_sha256'] and
                d.get('source_frozen_private_SHA256') == filehash(manifest) and
                d.get('source_public_preexec_SHA256') == filehash(pub_path) and
                d.get('mask_sha256') == spec['static_masks'][arm] and
                d.get('Q4_THREADS') == 1 and
                d.get('real_native_Q4_N8_weights_active') is True,
                'NATIVE_RECEIPT_CONTRACT_' + arm)
        require(d.get('real_native_original_C_log_SHA256') == filehash(logpath), 'ORIGINAL_C_LOG_SHA_' + arm)
        doc_public = {k: v for k, v in d.items() if k not in
                      ('original_full_vocab_gold_nexttoken_NLL', 'original_choice_correct_each')}
        doc_public['private_scores_receipt_SHA256'] = filehash(privpath)
        require(json.loads(pubpath.read_text('utf8')) == doc_public, 'PRIVATE_PUBLIC_BINDING_' + arm)
        losses = d['original_full_vocab_gold_nexttoken_NLL']
        correct = d['original_choice_correct_each']
        require(len(losses) == len(correct) == 96 and
                all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in losses) and
                all(type(v) is int and v in (0, 1) for v in correct),
                'ORIGINAL_CASE_SCORES_VALID_' + arm)
        text = logpath.read_text('utf8')
        entries = R_ROW.findall(text); summary = R_SUM.findall(text)
        require(len(entries) == 96 and len(summary) == 1 and
                text.count('projection_role_tensors: f32=0 n8=28 q4=168') == 1,
                'ORIGINAL_C_LOG_SHAPE_' + arm)
        for i, (idx, nll, gold, prediction, prefix_len) in enumerate(entries):
            row = rows[i]
            require(int(idx) == i and int(gold) == 'ABCD'.index(row['correct']) and
                    int(prefix_len) == row['prefix_len'] and
                    abs(float(nll) - float(losses[i])) < 1e-8 and
                    int(int(gold) == int(prediction)) == correct[i],
                    f'ORIGINAL_C_CASE_{arm}_{i}')
        total, count, won, mean, ppl = map(float, summary[0])
        require(int(count) == 96 and int(won) == sum(correct) and
                abs(total - sum(losses)) < 1e-7 and
                abs(mean - sum(losses) / 96) < 1e-8 and
                abs(d['mean_NLL'] - mean) < 1e-8 and
                d['correct'] == sum(correct) and math.isfinite(ppl) and ppl > 0,
                'ORIGINAL_C_AGGREGATE_' + arm)
        outcome[arm] = {
            'n': 96, 'correct': sum(correct), 'mean_NLL': float(mean),
            'private_sha256': filehash(privpath), 'original_log_sha256': filehash(logpath),
        }
    return {'status': 'BOTH_QV21_ORIGINAL_C_CALIBRATION_ARMS_VERIFIED',
            'schema': 'BEGLIN-Qv21-independent-train-only-native-receipt-audit/1',
            'public_preexec_sha256': filehash(pub_path),
            'calibration_only': True, 'eval_scores_accessed': False,
            'qv20_holdout_accessed': False, 'aggregates': outcome}


if __name__ == '__main__':
    try:
        print(json.dumps(audit(), sort_keys=True))
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'NOT_READY', 'failure_type': type(exc).__name__,
                          'no_policy_fit': True, 'no_eval_access': True}, sort_keys=True))
        raise SystemExit(5)
