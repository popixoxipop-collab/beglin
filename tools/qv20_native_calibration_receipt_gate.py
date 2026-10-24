#!/usr/bin/env python3
"""BEGLIN Qv20 read-only audit of both genuine C train32 calibration arms.

Validates immutable input & mask SHA, C executable provenance, original native
logs and their private/public receipts before running the previously committed
policy fitter. Never reads heldout splits, trains a policy, or writes any files.
"""
import argparse
import hashlib
import json
import math
import pathlib
import re
import sys

DEFAULT_ROOT = pathlib.Path('/Volumes/D50/TailnetCommander/work/beglin-qv20-prompt-aware-20261024')
DEFAULT_EXE = pathlib.Path('/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_mixed')
PINS = {
    'binary': 'f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221',
    'public': '2aae8540fef6bce9c15e4c71becd873f62d69da556056583a41bae810ec5da1a',
    'cal': '6bedf97854f140939b248001a491b6f7ae38a078b502a8af9a85d3a47cc99f28',
    'static_source': '9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922',
    'static_random': '5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde',
}
SCHEMA = 'BEGLIN-Qv20-actual-original-Qwen28-C-single-static-mask-case-scores/1'
R_ROW = re.compile(r'^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$', re.M)
R_SUM = re.compile(r'^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$', re.M)


class AuditFailure(RuntimeError):
    pass


def require(condition, label):
    if not condition:
        raise AuditFailure('QV20_CALIBRATION_RECEIPT_GATE_FAILED_' + label)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as src:
        for chunk in iter(lambda: src.read(262144), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(root=DEFAULT_ROOT, original_exe=DEFAULT_EXE, pins=None):
    """Pass only with both legitimate log-bound train32 arms (aggregate result only)."""
    pins = dict(PINS if pins is None else pins)
    require(sha256(original_exe) == pins['binary'], 'original_C_executable_SHA256')
    require(sha256(root / 'Qv20_PREEXEC_PUBLIC.json') == pins['public'], 'public_preexecution_SHA256')
    cohort_path = root / 'cal_arc_train/frozen_inputs_private.json'
    require(sha256(cohort_path) == pins['cal'], 'calibration_train_input_SHA256')
    rows = read_json(cohort_path)['selected_rows']
    require(len(rows) == 32, 'calibration_size')
    for mode in ('static_source', 'static_random'):
        require(sha256(root / 'masks' / (mode + '.txt')) == pins[mode], 'mask_SHA256_' + mode)

    aggregate = {}
    for mode in ('static_source', 'static_random'):
        folder = root / 'native_original_C_static/cal_arc_train' / mode
        private_path = folder / 'PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json'
        public_path = folder / 'PUBLIC_AGGREGATE.json'
        logfile = folder / 'native_original_C_Qwen28.log'
        receipt = read_json(private_path)
        require(
            receipt.get('schema') == SCHEMA
            and receipt.get('state') == 'ACTUAL_ORIGINAL_QWEN28_C_STATIC_' + mode.upper() + '_PASS'
            and receipt.get('cohort') == 'cal_arc_train'
            and receipt.get('arm') == mode
            and type(receipt.get('n')) is int and receipt['n'] == 32
            and receipt.get('original_Qwen28_C_executable_sha256') == pins['binary']
            and receipt.get('source_frozen_private_SHA256') == pins['cal']
            and receipt.get('source_public_preexec_SHA256') == pins['public']
            and receipt.get('real_native_Q4_N8_weights_active') is True
            and receipt.get('no_model_kernel_or_operator_release_changes') is True,
            'native_receipt_contract_' + mode,
        )
        require(receipt.get('real_native_original_C_log_SHA256') == sha256(logfile),
                'original_C_log_SHA256_' + mode)
        public_expected = {k: v for k, v in receipt.items()
                           if k not in ('original_full_vocab_gold_nexttoken_NLL', 'original_choice_correct_each')}
        public_expected['private_scores_receipt_SHA256'] = sha256(private_path)
        require(read_json(public_path) == public_expected, 'public_private_binding_' + mode)
        losses = receipt['original_full_vocab_gold_nexttoken_NLL']
        correct = receipt['original_choice_correct_each']
        require(len(losses) == len(correct) == 32
                and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in losses)
                and all(type(c) is int and c in (0, 1) for c in correct),
                'case_score_shape_' + mode)

        original_log = logfile.read_text(encoding='utf-8')
        lines = R_ROW.findall(original_log)
        summaries = R_SUM.findall(original_log)
        require(len(lines) == 32 and len(summaries) == 1
                and original_log.count('projection_role_tensors: f32=0 n8=28 q4=168') == 1,
                'original_C_log_structure_' + mode)
        for idx, (window, nll, expected, predicted, prefix_len) in enumerate(lines):
            row = rows[idx]
            require(int(window) == idx
                    and int(expected) == 'ABCD'.index(row['correct'])
                    and int(prefix_len) == row['prefix_len']
                    and abs(float(nll) - float(losses[idx])) <= 1e-8
                    and int(int(expected) == int(predicted)) == correct[idx],
                    'native_log_case_alignment_' + mode + '_' + str(idx))
        log_sum, count, wins, log_mean, ppl = map(float, summaries[0])
        mean = sum(float(v) for v in losses) / 32
        require(int(count) == 32 and int(wins) == sum(correct)
                and abs(log_sum - sum(losses)) < 1e-7
                and abs(log_mean - mean) < 1e-8
                and abs(float(receipt['mean_NLL']) - mean) < 1e-8
                and type(receipt.get('correct')) is int and receipt['correct'] == sum(correct)
                and math.isfinite(ppl) and ppl > 0,
                'native_log_aggregate_alignment_' + mode)
        aggregate[mode] = {'n': 32, 'correct': sum(correct),
                           'mean_gold_full_vocab_NLL': mean,
                           'original_C_log_SHA256': sha256(logfile),
                           'private_receipt_SHA256': sha256(private_path)}

    return {
        'schema': 'BEGLIN-Qv20-train-only-original-native-calibration-receipt-audit/1',
        'status': 'BOTH_ORIGINAL_NATIVE_TRAIN_ARMS_VERIFIED',
        'calibration_only': True, 'heldout_data_opened': False,
        'inference_launched': False, 'frozen_policy_trained': False,
        'per_case_scores_disclosed': False, 'aggregate': aggregate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument('--exe', type=pathlib.Path, default=DEFAULT_EXE)
    args = parser.parse_args()
    try:
        result = audit(root=args.root, original_exe=args.exe)
    except (AuditFailure, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'NOT_READY', 'reason': type(exc).__name__,
                          'real_native_train32_random73_not_verified': True,
                          'do_not_run_policy_fit': True}, sort_keys=True))
        return 5
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
