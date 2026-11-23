#!/usr/bin/env python3
"""Independently replay the JE01 all48 v2 recovery with one inherited OFF.

Only Python's standard library is used. No experiment code, prior auditor,
MLX, model forward, policy search, TRAIN240, validation, or OOD is imported or
executed. Raw binary gold/JVP records are the numerical source of truth.
The six new and three inherited FD checkpoints cover nine columns, not 48.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys

WORK = Path('/Volumes/D50/TailnetCommander/work')
ROOT = WORK / 'je01_native_role_jvp_all48_train16_20261123_v2'
V1_ROOT = WORK / 'je01_native_all48_train16_20261123_v1'
V1_RUNNER = '7451c867ebb810cff9fe854322e00937bf8b1807c2d0475006d6703be32dd4de'
V1_PLAN = '60fdced19fad43cbf2b73801dd68effbbd75c186f8a5f39de9274cad39453419'
V1_PLAN_FILE = 'dc4cf87f557ef806d44c02cd9bf66a60dbec893303d74aa89cbf9ef0c5a3454f'
V1_OFF = 'f044aa2e2ee1a22ab305f1854d2cf6c7280a5d72cded594859d786fcc5a0caa9'
V1_OFF_FILE = '98d68ca9a5984502bd3858bbb0523653814589abf1114ec5ad28eff39e6c6f6d'
V1_FAILED_FD = '51b3795007b69b783d94e22eae5f62fc96d8c73f8b377d3a1d18e5ee7e561129'
V1_FAILED_FD_FILE = '40daee88c55a53c478a17c34b584a7736c54b76fd06142cdd0656bcdc5e74579'
V1_WORKFLOW = '694f066501ddf1f1e31d94a746c56fc10239081e5c072311118c2c89c6e13b53'
V1_WORKFLOW_FILE = '1f32a3461859f8c90f485fd8e6fff95553cb872b9723bd59ba3c988f5f63cf0e'
V1_INTERRUPTED_AUDIT = 'b21f0e037e27ce30c47bb5ec4ef8b17a44d09c3b38f18389e33ae2fb62df6d3f'
V1_INTERRUPTED_AUDIT_FILE = '66fd0128598c2705ea735ea0876006b8effa2602aa25f455fec9e4e4e90a6e35'
PARENT = WORK / 'je01_native_role_jvp_followup_20261123_v1'
ORDINARY = WORK / 'je01_role_directional_20261123_v1'
FIXED = WORK / 'je01_role_directional_fixed_20261123_v1'
BANK = WORK / 'je01_role_direction_bank_20261123_v1'
MODEL = WORK / 'hcp-p18-external-multimetric-20261015/native_moe_base_source_copy/weights_moe/deepseek_moe_af.bin'
PARENT_AUDIT = '5ce09188e65d5cfcc0f956447b2918fe9c80bf81f14e8486ba97864f7137b87b'
PARENT_AUDIT_FILE = '9d82c2d727d4d20eb799e9a5b7625474311c954aaf7694063793bfaf09f18fcc'
PARENT_AUDITOR = 'c9e366c879715fdfbb406175b85f613860534da978e632245bede6c505069445'
PARENT_SUMMARY = '1448de4fdfb34082c7bd6976676d8a5404f078d649d730adbaf331fb233857df'
PARENT_BUILD = '2cbeb4dc1a7e1ab5642a4abb22395498c10dde43bee051eec3c303c7e53fa008'
PARENT_RUNNER = '9be8372fa3858577011eb5ab1d4ad654f746e151a95012aaf23e515fd8a7eb90'
PARENT_PATCH = '68c8cd6541657312bca6604dcbe74dcdc8156ff4da80de9c8fd672a1ad08d755'
PIN = {
    'model': '9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4',
    'binary': '110b1978d97bcd7c6bdfd6d42142569e56e9dacffa4ff462dc0786b3128e49f4',
    'fixed_binary': '46ea1f61fdd7483767ebc20900f4c97f9663a51148c74e566d2f66ccc848e688',
    'gold': 'b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c',
    'routes': '7e5811d13953f99db5df55169bf465f039acb9224f626336d247168e0f570711',
    'meta': 'be0dab12624efb4fae17538b4c5e6ef76bd09aebf4c09b1b428fc1d8f0fff406',
    'manifest': 'fab455198ef37d443b20af8e3029df3ddf4f4ba0f6d584eceeb262d26d6a146d',
    'dataset': 'a0091a208adaa559788a942149fe28f12f1de4e03dfbbafe840a5acce50f3e2a',
    'bank': '3b18257d22caddd0916292bef692110f9eeaf0edf83ad3fd4d5417fdf3443bb8',
    'bank_tsv': 'fdc801f59f9c6438dd60bb33fab9a1332cb19c6e15eb15bef7425b7fbd522e4f',
    'bank_receipt': '4994ec762b3a9b8a44015bad18f4b5bc7ea26505208384146238e7def28464d4',
    'ordinary_off': '85d8e28d51fbc0e2d319f0e60e5f9e6ac3d2f9050b2e386d81a969f6ad8f672a',
    'role_order': 'd232ca637eb17a1b40bf29f75e0590d8f63ecbc9aa950fb838a34b6fa90259a1',
    'adapter': '9ae9d98b921334bae321e583908e71bdc9c08178cbf3ca5d7a1d4d4e552a09ec',
    'mlx_library': 'afb1939edc96dc10e452132e4e665276fe0c48bf8cdc1a69286feb8a14de1da0',
    'cpp': '6055ca3b261075fe5fcf76e5829f0fb20b8ecdd52599248f4b9b1032cfbeb49f',
    'c': '5e2a6ea76c22c1ec0ff9f2fb6d093b7d3a471679adbe439ed1c4b898a56a52a6',
    'cmake': '1479cf49e18da616e6fcfdde2ed9433b468701a3d5a4482fa7e9ee398683da19',
    'old_audit': 'b7f20261c382f065ef5f6dc70ec04ed41450fa7ea495507a7014c385778e3cac',
    'old_audit_file': '71c304b6b69b8a07e4911c336352d142c12acf86c6522cd054cd2c48744d009e',
    'old_auditor': '7da57610f20e1c837e31528e001b91dc614bad5786a0d3c47cb2515f67d628a9',
    'fixed_audit': '87f239d900d9aff3921ab99d478024b13002e80b3b3d65683d4a055e4147598c',
    'fixed_audit_file': 'a64d13644a47396c8d514d60f6b5eee2f5683ef91975bc7433ab0260066208b9',
}
PAIRS = ((12, 13), (13, 22), (13, 28), (14, 63),
         (20, 17), (21, 61), (24, 26), (24, 60))
ROLE_NAMES = ('gate_proj', 'up_proj', 'down_proj')
ROLE_ORDER = [[layer, expert, role] for layer, expert in PAIRS for role in ROLE_NAMES]
COORDINATES = [key + [tier] for key in ROLE_ORDER for tier in ('Q4', 'Q5')]
NEW_FD = [(0, 'Q5'), (1, 'Q5'), (2, 'Q5'), (3, 'Q4'), (13, 'Q5'), (23, 'Q4')]
PRIOR_FD = [(0, 'Q4'), (1, 'Q4'), (2, 'Q4')]
ALPHAS = ('0', '+0.05', '-0.05', '+0.025', '-0.025')
COUNTERS = {'tokens': 4334, 'cases': 32, 'layers': 117018, 'gold': 656}
META_MAGIC = b'JE01_ORIGINAL_FULL_GOLD_TARGETS_META_V1\n'
GOLD_MAGIC = b'JE01_ORIGINAL_AF_FULL_GOLD_NLL_V1\n'
ROUTE_MAGIC = b'JE01_ROLE_DIRECTION_ROUTES_V1\n'
JVP_MAGIC = b'JE01_NATIVE_ONE_COLUMN_JVP_GOLD_V1\n'
SCOPE = 'PREVIOUSLY_SEEN_TRAIN16_FIXED_HARD_ROUTE_48_DIRECTION_DIAGNOSTIC_ONLY'
NO_ACTION = {'full_parameter_fisher': None, 'full48_empirical_fisher': None,
             'Hessian': 'NOT_MEASURED', 'train240_allocation': 'NOT_RUN',
             'allocation_selected': False, 'independent_validation_ood_accessed': False,
             'production_modified': False}
HELPERS = {
    'je01_native_all48_train16_runner.py': V1_RUNNER,
    'je01_native_role_jvp_runner.py': PARENT_RUNNER,
    'je01_native_one_column_jvp_runner.py': 'ffd632d79d948119ad261e2720ebbcd7c5076af21070aeaf60bf0ca8e7d79bd6',
    'je01_role_directional_native_runner.py': '5607c0afef3bd4b1a61bcab6622267acc6b6137fc65ba0e79873e92e31d7e9e6',
    'je01_role_directional_fixed_native.py': 'cffe2e1dd8298681f3a134caace742ddc9fea77209c58683cd8d2f48fe6284e7',
    'je01_role_directional_fixed_routes_patch.py': '233937b59293bec11300b66450b819dc48bb09b8d28eb6e98ca81aac7f511019',
    'je01_role_directional_bank.py': '045fd22d86e4ce18f967d66830a1f71b0c61ad6ab5c5889d953cb9a8d05bce33',
}


def need(condition, label):
    if not condition:
        raise ValueError('JE01_ALL48_INDEPENDENT_AUDIT:' + label)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'duplicate_json_key:' + str(key))
        result[key] = value
    return result


def strict_json(raw):
    def reject(value):
        raise ValueError('nonfinite_json_constant:' + value)
    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject)


def sealed_object(value, expected=None):
    need(isinstance(value, dict) and valid_sha(value.get('receipt_sha256')), 'receipt_type_or_sha')
    body = {key: item for key, item in value.items() if key != 'receipt_sha256'}
    need(canonical(body) == value['receipt_sha256'], 'receipt_canonical_integrity')
    need(expected is None or value['receipt_sha256'] == expected, 'receipt_external_pin')
    return value


def safe_path(path, *, directory=False):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink(), 'path_link:' + str(path))
    need(path.is_dir() if directory else path.is_file(), 'path_type:' + str(path))
    return path


class Evidence:
    """Record actual bytes; do not trust a receipt's own numerical claims."""
    def __init__(self):
        self.files = {}

    def read(self, path, expected=None, *, collect=True):
        path = safe_path(path)
        before = path.stat()
        digest, blocks = hashlib.sha256(), [] if collect else None
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(2 << 20), b''):
                digest.update(block)
                if blocks is not None:
                    blocks.append(block)
        after = path.stat()
        need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'file_changed:' + str(path))
        actual = digest.hexdigest()
        need(expected is None or actual == expected, 'file_sha:' + str(path))
        self.files[str(path)] = {'sha256': actual, 'size_bytes': after.st_size}
        return b''.join(blocks) if blocks is not None else actual

    def receipt(self, path, expected=None, file_sha=None):
        return sealed_object(strict_json(self.read(path, file_sha)), expected)


def exact(actual, expected, label):
    """Type-aware exact structural equality, including double representation."""
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), 'keys:' + label)
        for key in expected:
            exact(actual[key], expected[key], label + '.' + key)
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), 'length:' + label)
        for index, (left, right) in enumerate(zip(actual, expected)):
            exact(left, right, label + '[' + str(index) + ']')
    elif type(expected) is float:
        need(type(actual) in (int, float) and math.isfinite(actual) and
             struct.pack('<d', actual) == struct.pack('<d', expected), 'double:' + label)
    else:
        need(type(actual) is type(expected) and actual == expected, 'value:' + label)


def approximate(actual, expected, label):
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), 'metric_keys:' + label)
        for key in expected:
            approximate(actual[key], expected[key], label + '.' + key)
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), 'metric_length:' + label)
        for index, (left, right) in enumerate(zip(actual, expected)):
            approximate(left, right, label + '[' + str(index) + ']')
    elif type(expected) is float:
        need(type(actual) in (float, int) and math.isfinite(actual) and math.isfinite(expected)
             and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), 'metric:' + label)
    else:
        exact(actual, expected, label)


def flat(rows):
    return [value for row in rows for value in row]


def average(values):
    need(len(values) > 0, 'empty_average')
    return math.fsum(values) / len(values)


def metadata(raw):
    need(raw[:len(META_MAGIC)] == META_MAGIC and len(raw) == len(META_MAGIC) + 16 * 1544,
         'metadata_magic_length')
    starts, targets = [], []
    for case in range(16):
        offset = len(META_MAGIC) + case * 1544
        start, count = struct.unpack_from('<II', raw, offset)
        ids = struct.unpack_from('<384i', raw, offset + 8)
        need(start >= 2 and 1 <= count and start + count <= 384, 'metadata_bounds')
        need(all(0 <= value < 102400 for value in ids[:count]) and
             all(value == -1 for value in ids[count:]), 'metadata_ids_padding')
        starts.append(start)
        targets.append(list(ids[:count]))
    need(sum(map(len, targets)) == 656, 'metadata_total_tokens')
    return starts, targets


def token_keys(starts, targets):
    return [{'case_id': case, 'token_index': index, 'target_token_id': token,
             'position': starts[case] + index - 1}
            for case, row in enumerate(targets) for index, token in enumerate(row)]


def token_partition(targets):
    offset, result = 0, []
    for case, row in enumerate(targets):
        result.append({'case_id': case, 'token_offset_start': offset,
                       'token_offset_end': offset + len(row), 'gold_tokens': len(row)})
        offset += len(row)
    need(offset == 656 and len(result) == 16, 'sentence_partition_total')
    return result


def gold(raw, starts, targets):
    need(raw[:len(GOLD_MAGIC)] == GOLD_MAGIC and len(raw) == len(GOLD_MAGIC) + 32 * 656,
         'gold_magic_length')
    values, keys = [], token_keys(starts, targets)
    for record, key in zip(struct.iter_unpack('<4IdIf', raw[len(GOLD_MAGIC):]), keys):
        case, index, token, position, nll, top1, logit = record
        need((case, index, token, position) == tuple(key.values()), 'gold_token_alignment')
        need(math.isfinite(nll) and nll >= 0 and math.isfinite(logit) and 0 <= top1 < 102400,
             'gold_nonfinite_or_domain')
        values.append(nll)
    return [values[p['token_offset_start']:p['token_offset_end']] for p in token_partition(targets)]


def routes(raw):
    need(raw[:len(ROUTE_MAGIC)] == ROUTE_MAGIC and len(raw) == len(ROUTE_MAGIC) + 40 * 112684,
         'routes_magic_length')
    last_position = None
    for index, record in enumerate(struct.iter_unpack('<10i', raw[len(ROUTE_MAGIC):])):
        event, layer, slot, position, *experts = record
        need(event == index and layer == 1 + index % 26 and slot == 0 and 0 <= position < 384,
             'route_event_layer_slot')
        need(len(set(experts)) == 6 and all(0 <= expert < 64 for expert in experts), 'route_experts')
        need(index % 26 == 0 or last_position == position, 'route_position_layers')
        last_position = position
    need(112684 == COUNTERS['tokens'] * 26 and COUNTERS['layers'] == COUNTERS['tokens'] * 27,
         'route_completion_counter_relation')


def jvp(raw, starts, targets, baseline):
    need(raw[:len(JVP_MAGIC)] == JVP_MAGIC and len(raw) == len(JVP_MAGIC) + 48 * 656,
         'jvp_magic_length')
    keys, original = token_keys(starts, targets), flat(baseline)
    derivative, target_logits, logits_rms = [], [], []
    records = struct.iter_unpack('<4I4d', raw[len(JVP_MAGIC):])
    for record, key, expected_nll in zip(records, keys, original):
        case, index, token, position, nll, dnll, dtarget, drms = record
        need((case, index, token, position) == tuple(key.values()), 'jvp_token_alignment')
        need(all(math.isfinite(v) for v in (nll, dnll, dtarget, drms)) and nll >= 0 and drms >= 0,
             'jvp_nonfinite_or_domain')
        need(struct.pack('<d', nll) == struct.pack('<d', expected_nll), 'jvp_original_primal_bit_parity')
        derivative.append(dnll)
        target_logits.append(dtarget)
        logits_rms.append(drms)
    partition = token_partition(targets)
    regroup = lambda values: [values[p['token_offset_start']:p['token_offset_end']] for p in partition]
    return {'target_token_ids': targets, 'per_token_dnll': regroup(derivative),
            'per_token_dtargetlogit': regroup(target_logits), 'per_token_dlogits_rms': regroup(logits_rms)}


def finite_difference(plus, minus, half_step):
    need(len(plus) == len(minus) == 656 and half_step in (.05, .025), 'fd_grid_alignment')
    values = [(a - b) / (2 * half_step) for a, b in zip(plus, minus)]
    need(all(math.isfinite(value) for value in values), 'fd_nonfinite')
    return values


def comparison(actual, reference):
    need(len(actual) == len(reference) == 656 and
         all(type(v) in (int, float) and math.isfinite(v) for v in actual + reference), 'comparison_values')
    delta = [left - right for left, right in zip(actual, reference)]
    reference_square = average([v * v for v in reference])
    actual_square = average([v * v for v in actual])
    reference_rms, error_rms = math.sqrt(reference_square), math.sqrt(average([v * v for v in delta]))
    threshold = 1e-4 + .05 * reference_rms
    norm = math.sqrt(math.fsum(v * v for v in actual) * math.fsum(v * v for v in reference))
    values = {'tokens': 656, 'mean_actual': average(actual), 'mean_reference': average(reference),
              'rms_disagreement': error_rms, 'max_absolute_disagreement': max(map(abs, delta)),
              'rms_reference': reference_rms, 'rms_acceptance_threshold': threshold,
              'absolute_tolerance': 1e-4, 'relative_tolerance': .05, 'passed': error_rms <= threshold,
              'cosine_similarity': math.fsum(a * b for a, b in zip(actual, reference)) / norm if norm else None,
              'actual_square_mean': actual_square, 'reference_square_mean': reference_square}
    need(all(v is None or isinstance(v, bool) or math.isfinite(v) for v in values.values()), 'comparison_overflow')
    return values


def gram_statistics(matrix, targets):
    need(len(matrix) == 656 and all(isinstance(row, list) and len(row) == 48 for row in matrix),
         'matrix_shape')
    need(all(type(v) in (int, float) and math.isfinite(v) for row in matrix for v in row), 'matrix_finite')
    columns = [list(column) for column in zip(*matrix)]
    gradient = [average(column) for column in columns]
    gram = [[math.fsum(a * b for a, b in zip(left, right)) / 656 for right in columns] for left in columns]
    need(all(math.isfinite(v) for row in gram for v in row), 'gram_finite')
    sentences = []
    for group in token_partition(targets):
        begin, end = group['token_offset_start'], group['token_offset_end']
        sums = [math.fsum(column[begin:end]) for column in columns]
        sentences.append({**group, 'sum_dnll': sums, 'mean_dnll': [v / group['gold_tokens'] for v in sums]})
    aggregate = [math.fsum(group['sum_dnll'][j] for group in sentences) / 656 for j in range(48)]
    approximate(aggregate, gradient, 'sentence_weighted_gradient_identity')
    return gradient, gram, sentences


def spectral_check(matrix):
    """Cyclic Jacobi, independently implemented; report the final residual.

    The residual Gershgorin bound avoids treating unfinished diagonalization as
    a PSD certificate. This is only a numerical Gram check, not a Hessian test.
    """
    n = len(matrix)
    need(n > 0 and all(len(row) == n for row in matrix) and
         all(type(v) in (int, float) and math.isfinite(v) for row in matrix for v in row), 'spectral_shape_finite')
    asymmetry = max(abs(matrix[i][j] - matrix[j][i]) for i in range(n) for j in range(n))
    scale = max(max(abs(v) for v in row) for row in matrix)
    need(asymmetry <= 1e-12 * max(1., scale), 'gram_asymmetry')
    work = [[(matrix[i][j] + matrix[j][i]) / 2 for j in range(n)] for i in range(n)]
    original_trace = math.fsum(work[i][i] for i in range(n))
    tolerance = max(scale, 1e-300) * 1e-14
    rotations = 0
    for sweep in range(80):
        off = max((abs(work[i][j]) for i in range(n) for j in range(i)), default=0.)
        if off <= tolerance:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                pq = work[p][q]
                if abs(pq) <= tolerance:
                    continue
                theta = (work[q][q] - work[p][p]) / (2 * pq)
                t = math.copysign(1., theta) / (abs(theta) + math.hypot(1., theta))
                c = 1. / math.sqrt(1. + t * t)
                s = t * c
                pp, qq = work[p][p], work[q][q]
                work[p][p], work[q][q] = pp - t * pq, qq + t * pq
                work[p][q] = work[q][p] = 0.
                for k in range(n):
                    if k in (p, q):
                        continue
                    kp, kq = work[k][p], work[k][q]
                    work[k][p] = work[p][k] = c * kp - s * kq
                    work[k][q] = work[q][k] = s * kp + c * kq
                rotations += 1
    else:
        raise ValueError('JE01_ALL48_INDEPENDENT_AUDIT:jacobi_nonconvergence')
    residual = max(math.fsum(abs(work[i][j]) for j in range(n) if j != i) for i in range(n))
    eigenvalues = sorted(work[i][i] for i in range(n))
    need(all(math.isfinite(v) for v in eigenvalues) and math.isfinite(residual), 'jacobi_nonfinite')
    need(abs(math.fsum(eigenvalues) - original_trace) <= 1e-12 * max(1., abs(original_trace)),
         'jacobi_trace_residual')
    bound = min(work[i][i] - math.fsum(abs(work[i][j]) for j in range(n) if j != i) for i in range(n))
    acceptance = 1e-8 * max(1., original_trace)
    return {'method': 'INDEPENDENT_STDLIB_CYCLIC_JACOBI_WITH_RESIDUAL_GERSHGORIN_BOUND',
            'dimension': n, 'max_absolute_asymmetry': asymmetry, 'trace': original_trace,
            'eigenvalues_ascending': eigenvalues, 'minimum_eigenvalue': eigenvalues[0],
            'maximum_eigenvalue': eigenvalues[-1], 'offdiagonal_residual_bound': residual,
            'minimum_eigenvalue_lower_bound': bound, 'negative_tolerance': acceptance,
            'sweeps': sweep, 'rotations': rotations, 'converged': True,
            'passed': bound >= -acceptance,
            'interpretation': 'GRAM_PSD_IS_ALGEBRAIC_NOT_INDEPENDENT_DERIVATIVE_VALIDATION'}


def native_log(raw, kind, role, tier, alpha):
    text = raw.decode('utf-8', errors='strict')
    need('FATAL:' not in text and 'RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16' in text,
         'native_fatal_or_incomplete')
    need(('JE01_ROLE_DIRECTION_REGISTERED' in text) is (kind != 'OFF'), 'native_overlay_marker')
    need(('JE01_FIXED_BASELINE_ROUTES_ENABLED records=112684' in text) is (kind != 'OFF'), 'native_replay_marker')
    configs = re.findall(r'^JE01_ROLE_DIRECTION_CONFIG role=(\d+) layer=(\d+) expert=(\d+) tier=(Q4|Q5) alpha=(\S+)\s*$',
                        text, flags=re.MULTILINE)
    if kind == 'OFF':
        need(not configs and 'JE01_ROLE_DIRECTION_CONFIG' not in text, 'off_overlay_configuration')
    else:
        need(type(role) is int and 0 <= role < 24 and tier in ('Q4', 'Q5') and alpha in ALPHAS,
             'native_coordinate_domain')
        layer, expert, _ = ROLE_ORDER[role]
        need(configs and set(configs) == {(str(role), str(layer), str(expert), tier, alpha)} and
             text.count('JE01_ROLE_DIRECTION_CONFIG') == len(configs),
             'native_role_tier_alpha_configuration')
    complete = re.findall(r'^JE01_ONE_COLUMN_JVP_PREFIX_COMPLETE tokens=(\d+) cases=(\d+) layers=(\d+) gold=(\d+)\s*$',
                         text, flags=re.MULTILINE)
    if kind == 'JVP':
        need(complete == [tuple(map(str, COUNTERS.values()))] and
             text.count('JE01_ONE_COLUMN_JVP_PREFIX_COMPLETE') == 1, 'native_jvp_prefix_completion')
    else:
        need(not complete and 'JE01_ONE_COLUMN_JVP_' not in text, 'nonjvp_tangent_marker')
    return text


def verify_parent(evidence, expected_build):
    need(expected_build == PARENT_BUILD, 'external_parent_build_pin')
    prior = evidence.receipt(PARENT / 'JE01_THREE_ROLE_INDEPENDENT_AUDIT.json',
                             PARENT_AUDIT, PARENT_AUDIT_FILE)
    required = {'status': 'INDEPENDENT_AUDIT_CONFIRMED_THREE_ROLE_PASS',
                'evidence_integrity': 'PASS', 'experiment_numerical_verdict': 'PASS',
                'all_three_directions_pass': True, 'gold_tokens': 656,
                'case_ids': list(range(16)), 'native_cells_raw_replayed': 14,
                'auditor_source_sha256': PARENT_AUDITOR,
                'build_receipt_sha256': PARENT_BUILD, 'source_report_receipt_sha256': PARENT_SUMMARY,
                'original_af4_full_file_sha_verified_after_run': PIN['model'],
                'prior_independent_audit_receipt_sha256': PIN['old_audit'],
                'prior_independent_audit_file_sha256': PIN['old_audit_file'],
                'pure_auditor_utilities_sha256': PIN['old_auditor'],
                'full48_empirical_fisher': None, 'allocation_selected': False,
                'independent_validation_ood_accessed': False, 'production_modified': False}
    for key, value in required.items():
        exact(prior.get(key), value, 'parent_audit.' + key)
    files = prior['verified_files']
    need(isinstance(files, dict) and len(files) == 132 and str(MODEL) in files, 'parent_audit_file_coverage')
    # The original 9.814 GB model is fully streamed once, after all new raw reads.
    for path, claim in files.items():
        need(isinstance(claim, dict) and set(claim) == {'sha256', 'size_bytes'} and
             valid_sha(claim['sha256']) and type(claim['size_bytes']) is int, 'parent_file_claim')
        if Path(path) != MODEL:
            evidence.read(path, claim['sha256'], collect=False)
            need(evidence.files[path]['size_bytes'] == claim['size_bytes'], 'parent_file_size:' + path)
        else:
            need(claim == {'sha256': PIN['model'], 'size_bytes': 9814343680}, 'parent_model_pin')
    evidence.read(PARENT / 'tools/je01_native_three_role_independent_audit.py', PARENT_AUDITOR, collect=False)
    evidence.read(PARENT / 'tools/je01_native_one_column_independent_audit.py', PIN['old_auditor'], collect=False)
    old = evidence.receipt(PARENT / 'JE01_ONE_COLUMN_INDEPENDENT_AUDIT.json',
                          PIN['old_audit'], PIN['old_audit_file'])
    need(old['status'] == 'PASS_INDEPENDENT_RAW_NATIVE_JVP_FIXED_FD', 'parent_previous_audit_status')
    summary = evidence.receipt(PARENT / 'JE01_THREE_ROLE_JVP_DIAGNOSTIC_attempt1.json', PARENT_SUMMARY)
    need(summary['status'] == 'THREE_ROLE_NATIVE_JVP_FIXED_FD_PASS', 'parent_summary_status')
    build = evidence.receipt(PARENT / 'NATIVE_BUILD_RECEIPT.json', PARENT_BUILD)
    config = build['configuration']
    need(build['status'] == 'ROLE_JVP_NATIVE_COMPILE_PASS' and build['binary_sha256'] == PIN['binary'] and
         build['model_original_af4_sha256'] == PIN['model'] and build['runner_source_sha256'] == PARENT_RUNNER and
         build['cmake_sha256'] == PIN['cmake'] and build['mlx_library_sha256'] == PIN['mlx_library'] and
         build['bank_receipt_sha256'] == PIN['bank_receipt'] and build['dataset_receipt_sha256'] == PIN['dataset'],
         'parent_build_pin_map')
    need(config['patched_cpp_sha256'] == PIN['cpp'] and config['patched_c_sha256'] == PIN['c'] and
         config['patch_source_sha256'] == PARENT_PATCH and config['adapter_sha256'] == PIN['adapter'] and
         config['runtime_mode'] == 'ROLE_DIRECTION_FIXED_Q4Q5', 'parent_source_configuration')
    for path, pin in {'bin/je01_native_role_jvp': PIN['binary'], 'source/mlx_moe.cpp': PIN['cpp'],
                      'source/qwen_infer.c': PIN['c'], 'source/CMakeLists.txt': PIN['cmake'],
                      'source/je01_mlx_native_jvp_adapters.h': PIN['adapter'],
                      'tools/je01_native_role_jvp_runner.py': PARENT_RUNNER,
                      'tools/je01_native_role_jvp_patch.py': PARENT_PATCH}.items():
        need(files[str(PARENT / path)]['sha256'] == pin, 'parent_specific_file:' + path)
    return prior, build, summary


def verify_dataset(evidence):
    dataset = evidence.receipt(ORDINARY / 'datasets/pilot16.json', PIN['dataset'])
    need(dataset['scope'] == 'TRAIN16_DIAGNOSTIC' and dataset['case_ids'] == list(range(16)) and
         dataset['independent_validation_ood_accessed'] is False, 'dataset_seen_train16_only')
    starts, targets = metadata(evidence.read(dataset['gold_metadata'], PIN['meta']))
    exact(dataset['target_token_ids'], targets, 'dataset_target_ids')
    lines = evidence.read(dataset['manifest'], PIN['manifest']).decode().splitlines()
    need(len(lines) == len(dataset['input_sha256']) == 16, 'input_manifest_cardinality')
    input_tokens = 0
    for case, line in enumerate(lines):
        name, repeat = line.rsplit(' ', 1)
        need(repeat == '1', 'input_repeat')
        raw = evidence.read(name, dataset['input_sha256'][case])
        count = starts[case] + len(targets[case]) - 1
        need(len(raw) == count * 4, 'input_full_prefix_length')
        ids = struct.unpack('<' + 'i' * count, raw)
        need(ids[0] == 100000 and all(0 <= token < 102400 for token in ids) and
             list(ids[starts[case]:]) == targets[case][:-1], 'input_teacher_forcing')
        input_tokens += count
    need(2 * input_tokens == COUNTERS['tokens'], 'input_warmup_prefix_counter')
    off_root = ORDINARY / 'runs/pilot16/off_attempt1'
    original = evidence.receipt(off_root / 'RUN_RECEIPT.json', PIN['ordinary_off'])
    raw_gold = evidence.read(off_root / 'gold_native.bin', PIN['gold'])
    baseline = gold(raw_gold, starts, targets)
    exact(original['per_token_nll'], baseline, 'original_baseline_raw')
    raw_routes = evidence.read(off_root / 'routes.bin', PIN['routes'])
    routes(raw_routes)
    return {'dataset': dataset, 'starts': starts, 'targets': targets, 'baseline': baseline,
            'raw_gold': raw_gold, 'raw_routes': raw_routes, 'token_rows': token_keys(starts, targets),
            'sentence_groups': token_partition(targets)}


def verify_bank(evidence):
    record = evidence.receipt(BANK / 'JE01_ROLE_DIRECTION_BANK_RECEIPT.json', PIN['bank_receipt'])
    need(record['schema'] == 'beglin-je01-original27-native-role-direction-bank/1' and
         record['status'] == 'JE01_NATIVE_AF4_Q4_Q5_72_ENTRY_DIRECTION_BANK_BUILT' and
         record['source_original_AF4_sha256'] == PIN['model'] and
         record['source_original_AF4_full_rehash'] is True and
         record['source_original_AF4_file_bytes'] == 9814343680 and
         record['source_native_builder_git_blob_sha1'] == '89f05ad396b3face37da585b024f23e31ab706e6' and
         record['source_cal240_ledger_receipt_sha256'] == '35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb' and
         record['source_cal240_python_sha256'] == '863fb9c1d5a17fc810564610739bd1c53f727addee99a7966f65404b8725a414' and
         record['builder_file_sha256'] == '045fd22d86e4ce18f967d66830a1f71b0c61ad6ab5c5889d953cb9a8d05bce33',
         'bank_source_attestation')
    need(record['physical_file_sha256'] == PIN['bank'] and record['role_manifest_sha256'] == PIN['bank_tsv'] and
         record['physical_file_size_bytes'] == 129761280 and record['role_count'] == 24 and
         record['catalog_entry_count'] == 72 and record['counts'] == {'AF4': 24, 'Q4': 24, 'Q5': 24} and
         record['catalog_is_same_budget_candidate'] is False and record['all_choices_built_without_loss_based_selection'] is True and
         record['heldout_validation_OOD_accessed'] is False and record['original_model_weights_modified'] is False and
         record['production_modified'] is False, 'bank_scope_or_physical_pins')
    path = BANK / 'JE01_ROLE_DIRECTION_BANK.bin'
    evidence.read(path, PIN['bank'], collect=False)
    need(evidence.files[str(path)]['size_bytes'] == 129761280, 'bank_byte_count')
    manifest = evidence.read(BANK / 'JE01_ROLE_DIRECTION_BANK.tsv', PIN['bank_tsv']).decode()
    rows = record['per_entry']
    need(isinstance(rows, list) and len(rows) == 72, 'bank_rows')
    cursor, lines, mappings = 0, [], []
    before = path.stat()
    with path.open('rb') as stream:
        for index, row in enumerate(rows):
            role_index, tier_index = divmod(index, 3)
            layer, expert, role = ROLE_ORDER[role_index]
            tier = ('AF4', 'Q4', 'Q5')[tier_index]
            out, inp = (2048, 1408) if role == 'down_proj' else (1408, 2048)
            bits = 5 if tier == 'Q5' else 4
            for key, expected in {'role_index': role_index, 'layer': layer, 'expert': expert, 'role': role,
                                  'tier': tier, 'output': out, 'input': inp, 'bits': bits}.items():
                exact(row.get(key), expected, 'bank_row.' + key)
            sizes = {'packed': out * inp * bits // 8, 'scale': out * (inp // 64) * 4}
            if tier == 'AF4':
                sizes['bias'] = sizes['scale']
            need(set(row['regions']) == set(row['part_sha256']) == set(sizes) and
                 row['physical_bytes'] == sum(sizes.values()), 'bank_parts_shape')
            full = hashlib.sha256()
            for part, size in sizes.items():
                exact(row['regions'][part], [cursor, cursor + size], 'bank_contiguous_span')
                block = stream.read(size)
                need(len(block) == size and sha_bytes(block) == row['part_sha256'][part], 'bank_part_raw_sha')
                full.update(block)
                cursor += size
            need(full.hexdigest() == row['payload_sha256'], 'bank_entry_payload_sha')
            af4 = rows[3 * role_index]
            need(all(valid_sha(row.get(key)) and row[key] == af4[key]
                     for key in ('source_original_bf16_tensor_sha256', 'source_original_affine4_packed_sha256')),
                 'bank_shared_source_tensor_binding')
            need(af4['part_sha256']['packed'] == row['source_original_affine4_packed_sha256'],
                 'bank_exact_af4_source_packed')
            fields = [layer, expert, role, bits, out, inp, *row['regions']['packed'], *row['regions']['scale'],
                      *row['regions'].get('bias', [0, 0]), tier]
            lines.append('\t'.join(map(str, fields)))
            mappings.append({'bank_entry_index': index, 'role_index': role_index, 'role_key': ROLE_ORDER[role_index],
                             'tier': tier, 'output': out, 'input': inp, 'payload_sha256': row['payload_sha256'],
                             'source_original_bf16_tensor_sha256': row['source_original_bf16_tensor_sha256'],
                             'source_original_affine4_packed_sha256': row['source_original_affine4_packed_sha256']})
        need(not stream.read(1), 'bank_trailing_bytes')
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns),
         'bank_changed_during_region_read')
    need(cursor == 129761280 and manifest == '\n'.join(lines) + '\n', 'bank_layout_manifest_exact')
    return mappings


def inherited_fd(evidence, context, parent):
    starts, targets, baseline = context['starts'], context['targets'], context['baseline']
    result = {}
    for role in range(3):
        values, receipts = {}, {}
        for alpha in ALPHAS[1:]:
            suffix = alpha.replace('+', 'p').replace('-', 'm')
            folder = (FIXED / 'runs/pilot16' / f'r00_Q4_a{suffix}_attempt1' if role == 0 else
                      PARENT / 'runs/pilot16' / f'fd_r{role:02d}_Q4_a{suffix}_attempt1')
            record = evidence.receipt(folder / 'RUN_RECEIPT.json')
            need(record['status'] == 'SUCCEEDED_EXIT0' and record['role_index'] == role and
                 record['role_key'] == ROLE_ORDER[role] and record['tier'] == 'Q4' and
                 record['alpha'] == float(alpha) and record['fixed_route_replay_enabled'] is True,
                 'prior_fd_cell_identity')
            parsed = gold(evidence.read(folder / 'gold_native.bin', record['output_sha256']), starts, targets)
            exact(record['per_token_nll'], parsed, 'prior_fd_raw_replay')
            need(evidence.read(folder / 'routes.bin', PIN['routes']) == context['raw_routes'], 'prior_fd_raw_routes')
            values[alpha], receipts[alpha] = flat(parsed), record['receipt_sha256']
        coarse = finite_difference(values['+0.05'], values['-0.05'], .05)
        fine = finite_difference(values['+0.025'], values['-0.025'], .025)
        jvp_folder = PARENT / 'runs/pilot16' / f'jvp_r{role:02d}_Q4_a0_attempt1'
        jvp_record = evidence.receipt(jvp_folder / 'RUN_RECEIPT.json')
        raw = evidence.read(jvp_folder / 'jvp_gold.bin', jvp_record['gradient_output_sha256'])
        old_column = flat(jvp(raw, starts, targets, baseline)['per_token_dnll'])
        exact(parent['raw_directional_derivatives'][str(role)],
              {'fixed_coarse_fd': coarse, 'fixed_fine_fd': fine, 'native_jvp': old_column},
              'prior_independent_array_binding')
        need(comparison(coarse, fine)['passed'] and comparison(old_column, fine)['passed'], 'prior_fd_numerical_pass')
        result[(role, 'Q4')] = {'coarse': coarse, 'fine': fine, 'prior_native': old_column,
                                 'run_receipt_sha256': receipts,
                                 'prior_jvp_output_sha256': jvp_record['gradient_output_sha256']}
    return result


def coordinate_order():
    return [{'column_index': index, 'role_index': index // 2,
             'role_key': ROLE_ORDER[index // 2], 'tier': ('Q4', 'Q5')[index % 2]} for index in range(48)]


def sequence():
    directions = [(0, 'Q5'), (1, 'Q5'), (2, 'Q5')]
    directions += [(r, t) for r in range(24) for t in ('Q4', 'Q5') if (r, t) not in directions]
    cells = [{'kind': 'OFF', 'role_index': None, 'tier': None, 'alpha_literal': '0', 'column_index': None}]
    for role, tier in directions:
        index = 2 * role + int(tier == 'Q5')
        if (role, tier) in NEW_FD:
            cells.extend({'kind': 'FD', 'role_index': role, 'tier': tier, 'alpha_literal': alpha,
                          'column_index': index} for alpha in ALPHAS)
        cells.append({'kind': 'JVP', 'role_index': role, 'tier': tier, 'alpha_literal': '0', 'column_index': index})
    need(len(cells) == 79, 'internal_preregistered_sequence')
    return [{'cell_index': index, **cell} for index, cell in enumerate(cells)]


def cell_label(cell):
    if cell['kind'] == 'OFF':
        label = 'off'
    else:
        alpha = cell['alpha_literal'].replace('+', 'p').replace('-', 'm')
        label = f"{cell['kind'].lower()}_r{cell['role_index']:02d}_{cell['tier']}_a{alpha}"
    return f"cell_{cell['cell_index']:03d}_{label}"


def native_output_path_contract(source, root):
    """Inspect the real SHA-pinned C guard, rather than relying on root naming."""
    source = source.decode('utf-8', errors='strict') if isinstance(source, bytes) else source
    anchor = 'const char *je01_jvp_mode=getenv("JE01_NATIVE_JVP_MODE");'
    open_call = 'int fd=open(je01_jvp_path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);'
    need(source.count(anchor) == 1 and source.count(open_call) == 1, 'native_c_open_anchors')
    begin, end = source.index(anchor), source.index(open_call)
    need(begin < end, 'native_c_open_order')
    block = source[begin:end]
    matches = re.findall(r'const char prefix\[\]="([^"]+)";', block)
    need(matches == ['/Volumes/D50/TailnetCommander/work/je01_native_role_jvp_'] and
         'strcmp(je01_jvp_mode,"ROLE_DIRECTION_FIXED_Q4Q5")' in block and
         'strncmp(je01_jvp_path,prefix,sizeof(prefix)-1)' in block and
         '!je01_jvp_path' in block and 'exit(1)' in block, 'native_c_open_guard')
    prefix = matches[0]
    root = Path(root)
    need(root.is_absolute() and str(root).startswith(prefix), 'native_c_open_root_prefix_rejected')
    paths = [root / 'runs/train16' / (cell_label(cell) + '_attempt1') / 'jvp_gold.bin'
             for cell in sequence() if cell['kind'] == 'JVP']
    need(len(paths) == 48 and all(str(path).startswith(prefix) and path.is_relative_to(root) for path in paths),
         'native_c_open_all48_output_paths')
    return {'pinned_c_source_sha256': PIN['c'], 'native_jvp_output_prefix': prefix,
            'checked_root': str(root), 'all48_jvp_output_paths_conform': True,
            'native_path_guard_modified': False}


def recovery_contract():
    return {'reason': 'SOURCE_ENFORCED_JVP_OUTPUT_ROOT_MISMATCH_FOUND_BEFORE_FIRST_JVP',
            'change': 'FRESH_NATIVE_COMPATIBLE_ROOT_ONLY_NO_NATIVE_BINARY_OR_MATH_CHANGE',
            'v1_root': str(V1_ROOT), 'v1_plan_receipt_sha256': V1_PLAN,
            'v1_runner_source_sha256': V1_RUNNER,
            'v1_off_receipt_sha256': V1_OFF, 'v1_off_file_sha256': V1_OFF_FILE,
            'v1_interrupted_fd_receipt_sha256': V1_FAILED_FD,
            'v1_interrupted_fd_file_sha256': V1_FAILED_FD_FILE,
            'v1_workflow_receipt_sha256': V1_WORKFLOW, 'v1_workflow_file_sha256': V1_WORKFLOW_FILE,
            'v1_independent_confirmation_receipt_sha256': V1_INTERRUPTED_AUDIT,
            'v1_independent_confirmation_file_sha256': V1_INTERRUPTED_AUDIT_FILE,
            'v1_independent_auditor_source_sha256': '270fc991cd0e4378457fe80cfa4f7f924ab69d44d77656ecfb95bb11449730a9',
            'inherited_successful_cell_indices': [0], 'v1_jvp_columns_measured': 0,
            'off_native_reexecution_forbidden': True,
            'interrupted_fd_was_not_a_completed_numerical_result': True,
            'v1_evidence_is_preserved': True}


def scope(record):
    exact(record.get('scope'), SCOPE, 'scope')
    for key, value in NO_ACTION.items():
        need(key in record, 'missing_scope_field:' + key)
        exact(record.get(key), value, 'no_action.' + key)


def verify_plan(args, evidence, context):
    root = safe_path(args.root, directory=True)
    need(root == ROOT, 'exact_preregistered_root')
    need(valid_sha(args.expected_plan_receipt) and valid_sha(args.expected_runner_sha), 'external_plan_runner_pins_required')
    plan = evidence.receipt(root / 'PLAN_RECEIPT.json', args.expected_plan_receipt)
    scope(plan)
    evidence.read(root / 'tools/je01_native_all48_train16_runner_v2.py', args.expected_runner_sha, collect=False)
    for name, pin in HELPERS.items():
        evidence.read(root / 'tools' / name, pin, collect=False)
    evidence.read(root / 'tools/je01_native_role_jvp_patch.py', PARENT_PATCH, collect=False)
    parent_plan_pin = '12ae8c5eb7a3d6409eea2aa79337de9da9c56453bac33ec084c4e4a64aec8a14'
    parent_plan = evidence.receipt(PARENT / 'PLAN_RECEIPT.json', parent_plan_pin)
    identity = {**parent_plan['source_identity'],
                'derivative_definition': 'ALL48_NATIVE_ALPHA_ZERO_FIXED_HARD_ROUTE_FULL_PREFIX_DIRECTIONS',
                'parent_three_role_report_sha256': PARENT_SUMMARY,
                'parent_three_role_independent_audit_sha256': PARENT_AUDIT}
    runtime = plan['runtime']
    need(isinstance(runtime, dict), 'runtime_object')
    library = safe_path(runtime['library_path'], directory=True)
    guard = native_output_path_contract(evidence.read(PARENT / 'source/qwen_infer.c', PIN['c']), root)
    context['native_output_path_contract'] = guard
    runtime_expected = {
        'library_path': str(library), 'libmlx_path': str(library / 'libmlx.dylib'),
        'libmlx_sha256': PIN['mlx_library'], 'manifest_path': context['dataset']['manifest'],
        'manifest_sha256': PIN['manifest'], 'metadata_path': context['dataset']['gold_metadata'],
        'metadata_sha256': PIN['meta'], 'native_executable_path': str(PARENT / 'bin/je01_native_role_jvp'),
        'fixed_fd_executable_path': str(FIXED / 'bin/je01_role_directional_fixed_native'),
        'fixed_fd_executable_sha256': PIN['fixed_binary'],
        'baseline_routes_path': str(ORDINARY / 'runs/pilot16/off_attempt1/routes.bin'),
        'baseline_routes_sha256': PIN['routes'], 'baseline_gold_sha256': PIN['gold'],
        'bank_path': str(BANK / 'JE01_ROLE_DIRECTION_BANK.bin'),
        'bank_receipt_sha256': PIN['bank_receipt'], 'bank_sha256': PIN['bank'],
    }
    estimates = {'OFF': 73.8836, 'JVP': 154.9563, 'FD': 71.4456}
    expected = {
        'schema': 'beglin-je01-native-all48-train16-plan/2', 'scope': SCOPE, 'root': str(root),
        'source_identity': identity, 'runner_source_sha256': args.expected_runner_sha,
        'helper_source_sha256': HELPERS, 'patch_source_sha256': PARENT_PATCH,
        'proven_native_build_receipt_sha256': PARENT_BUILD, 'native_binary_sha256': PIN['binary'],
        'parent_root': str(PARENT), 'parent_report_receipt_sha256': PARENT_SUMMARY,
        'parent_report_file_sha256': '22f929b8e92eff027358fa7c2104044cda7967ab46503199793c42618b5ae2b5',
        'parent_audit_receipt_sha256': PARENT_AUDIT, 'parent_audit_file_sha256': PARENT_AUDIT_FILE,
        'parent_plan_receipt_sha256': parent_plan_pin, 'runtime': runtime_expected,
        'coordinate_order': coordinate_order(), 'column_count': 48,
        'coordinate_definition': 'ALTERNATIVE_MINUS_ORIGINAL_NATIVE_WEIGHT_ACTION_AT_ALPHA_ZERO',
        'evidence_execution_sequence': sequence(), 'evidence_cell_count': 79,
        'native_execution_sequence': sequence()[1:], 'native_cell_count': 78, 'inherited_native_cell_count': 1,
        'native_output_path_contract': {'source_c_sha256': PIN['c'],
                                        'required_absolute_prefix': guard['native_jvp_output_prefix'],
                                        'output_root': str(root), 'validated_before_any_native_execution': True},
        'recovery': recovery_contract(),
        'new_fd_checkpoints': [{'role_index': r, 'tier': t, 'column_index': 2 * r + int(t == 'Q5')} for r, t in NEW_FD],
        'reused_fd_checkpoints': [{'role_index': r, 'tier': t, 'column_index': 2 * r} for r, t in PRIOR_FD],
        'new_fd_cell_count': 30, 'fd_verified_coordinate_count': 9,
        'fd_coverage_interpretation': 'NINE_PREDECLARED_CHECKPOINTS_NOT_ALL_48_INDIVIDUAL_FD_VALIDATIONS',
        'alpha_grid': list(ALPHAS), 'jvp_alpha': 0, 'case_ids': list(range(16)), 'gold_tokens': 656,
        'target_token_ids': context['targets'], 'full_prefix_completion_counters': COUNTERS,
        'fd_convergence_rule': {'coarse_h': .05, 'fine_h': .025, 'rms_absolute_tolerance': 1e-4,
                                'rms_relative_tolerance': .05, 'reference': 'ALL_656_FINE_FD_TOKEN_DERIVATIVES'},
        'jvp_agreement_rule': {'rms_absolute_tolerance': 1e-4, 'rms_relative_tolerance': .05,
                               'reference': 'ALL_656_FINE_FD_TOKEN_DERIVATIVES'},
        'prior_q4_jvp_requires_exact_gradient_output_sha': True,
        'every_zero_and_jvp_primal_requires_original_gold_sha': PIN['gold'],
        'every_fixed_route_output_requires_baseline_sha': PIN['routes'],
        'natural_routing_stability': 'NOT_ESTABLISHED_BY_FORCED_ROUTE_REPLAY',
        'matrix_rules': {'raw_shape': [656, 48], 'g_definition': 'MEAN_TOKEN_DNLL',
                         'score_definition': 'NEGATIVE_DNLL', 'gram_definition': 'MEAN_TOKEN_DNLL_OUTER_PRODUCT',
                         'centering': 'NONE', 'sentence_partition': '16_FROZEN_CASE_IDS_IN_METADATA_ORDER',
                         'psd_min_eigenvalue_lower_bound': '-1e-8 * max(1, trace(Gram))',
                         'eigen_residual_tolerance_times_max_1_abs_trace': 1e-12,
                         'publish_only_if_all_48_raw_and_9_checkpoints_and_psd_pass': True},
        'execution_policy': {'sequential': True, 'default_max_new_cells': 12, 'maximum_max_new_cells': 79,
                             'native_cell_timeout_seconds_max': 900, 'initial_memory_free_percent_min': 55,
                             'memory_free_percent_floor': 35, 'rss_limit_gib': 18,
                             'retry': 'SEALED_EXECUTION_OR_INTEGRITY_FAILURE_ONLY',
                             'successful_cells_never_reexecuted': True,
                             'numerical_failure': 'SEAL_HOLD_AND_STOP_FURTHER_NATIVE_CELLS',
                             'plan_or_threshold_changes_after_first_cell': False},
        'timing_estimate': {'source': 'MEASURED_PRIOR_THREE_ROLE_NATIVE_CELL_RECEIPTS',
                            'seconds_per_cell': estimates, 'observed_up_jvp_seconds': 151.2950,
                            'estimated_native_seconds': math.fsum(estimates[cell['kind']] for cell in sequence()[1:]),
                            'inherited_off_elapsed_seconds': context['v1_original_off']['execution']['elapsed_seconds'],
                            'excludes_preflight_hashing_and_audit': True, 'estimate_is_not_a_runtime_guarantee': True},
        'result_adaptive_selection': False, 'independent_audit_required_before_train240': True,
        **NO_ACTION,
    }
    exact({k: v for k, v in plan.items() if k != 'receipt_sha256'}, expected, 'sealed_plan')
    evidence.read(runtime['libmlx_path'], PIN['mlx_library'], collect=False)
    evidence.read(runtime['native_executable_path'], PIN['binary'], collect=False)
    evidence.read(runtime['fixed_fd_executable_path'], PIN['fixed_binary'], collect=False)
    patch = evidence.receipt(PARENT / 'SOURCE_PATCH_RECEIPT.json',
                             '01d6b42e588ee79dd1f58f5065b1a3e2a5239f855f319eaba019ba8c459c0f6a')
    need(patch['supported_role_indices'] == list(range(24)) and patch['supported_tiers'] == ['Q4', 'Q5'] and
         patch['alpha'] == 0 and patch['one_direction_per_run'] is True and
         patch['runtime_mode'] == 'ROLE_DIRECTION_FIXED_Q4Q5', 'source_all48_selector_support')
    return plan, {**identity, 'plan_receipt_sha256': plan['receipt_sha256']}


def verify_source_ledger(evidence, root, mappings):
    name = 'JE01_ACTUAL_CAL240_FFN_H2_SWIGLU_NATIVE_OUTPUT_DP_20261122.json'
    ledger_pin = '35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb'
    origin = evidence.receipt(PARENT / 'docs/research' / name, ledger_pin)
    record = evidence.receipt(root / 'docs/research' / name, ledger_pin)
    exact(record, origin, 'immutable_source_ledger')
    need(record['source_full_original_AF4_sha256'] == PIN['model'] and len(record['per_role']) == 24,
         'source_ledger_model_roles')
    source_roles = {(r['layer'], r['expert'], r['role']): r for r in record['per_role']}
    need(set(source_roles) == {tuple(r) for r in ROLE_ORDER}, 'source_ledger_role_uniqueness')
    for mapping in mappings:
        row = source_roles[tuple(mapping['role_key'])]
        need(mapping['source_original_bf16_tensor_sha256'] == row['actual_original_BF16_tensor_sha256'] and
             mapping['source_original_affine4_packed_sha256'] == row['actual_original_AF4_packed_sha256'],
             'source_tensor_role_tier_mapping')


def verify_cell(evidence, context, folder, cell, attempt, failure_pins, control_pin, prior, *, original=False):
    record = evidence.receipt(folder / 'RUN_RECEIPT.json')
    scope(record)
    role, kind, tier, alpha = cell['role_index'], cell['kind'], cell['tier'], cell['alpha_literal']
    contract = {'schema': 'beglin-je01-native-all48-train16-cell/' + ('1' if original else '2'), 'status': 'SUCCEEDED_EXIT0',
                **cell, 'role_key': ROLE_ORDER[role] if role is not None else None,
                'alpha': float(alpha), 'attempt': attempt, 'source_identity': context['identity'],
                'plan_receipt_sha256': context['plan']['receipt_sha256'], 'runner_source_sha256': context['runner_sha'],
                'case_ids': list(range(16)), 'executed_binary_sha256': PIN['fixed_binary'] if kind == 'FD' else PIN['binary'],
                'new_off_receipt_sha256': control_pin, 'prior_failure_receipt_sha256': failure_pins,
                'native_jvp_enabled': kind == 'JVP', 'native_overlay_enabled': kind != 'OFF',
                'fixed_route_replay_enabled': kind != 'OFF', 'full_prompt_and_KV_recomputed': True,
                'router_scores_and_selected_weights_dynamic': True, 'full_prefix_tangent_carried': kind == 'JVP',
                'route_equality_interpretation': 'REPLAY_CONFORMANCE_NOT_NATURAL_ROUTING_STABILITY'}
    if not original:
        contract.update({'native_executed_in_this_plan': kind != 'OFF',
                         'executed_runner_source_sha256': V1_RUNNER if kind == 'OFF' else context['runner_sha'],
                         'reused_from_v1_plan_receipt_sha256': V1_PLAN if kind == 'OFF' else None,
                         'original_run_receipt_sha256': V1_OFF if kind == 'OFF' else None,
                         'original_run_file_sha256': V1_OFF_FILE if kind == 'OFF' else None,
                         'original_run_folder': str(V1_ROOT / 'runs/train16/cell_000_off_attempt1') if kind == 'OFF' else None,
                         'v1_independent_confirmation_receipt_sha256': V1_INTERRUPTED_AUDIT if kind == 'OFF' else None,
                         'recovered_v1_interrupted_cell_receipt_sha256': V1_FAILED_FD if cell['cell_index'] == 1 else None})
    for key, value in contract.items():
        exact(record.get(key), value, 'cell.' + key)
    telemetry = record['execution']
    need(telemetry['native_exit_code'] == 0 and telemetry['stop_reason'] is None and telemetry['native_started'] is True and
         type(telemetry['elapsed_seconds']) in (int, float) and math.isfinite(telemetry['elapsed_seconds']) and
         telemetry['elapsed_seconds'] >= 0 and type(telemetry['timeout_seconds']) is int and
         1 <= telemetry['timeout_seconds'] <= 900 and telemetry['rss_limit_gib'] == 18 and
         telemetry['memory_free_floor_pct'] == 35 and 35 <= telemetry['min_memory_free_pct'] <= 100 and
         0 <= telemetry['max_observed_rss_kib'] <= 18 * 1024 * 1024, 'successful_native_telemetry')
    raw = evidence.read(folder / 'gold_native.bin', record['output_sha256'])
    values = gold(raw, context['starts'], context['targets'])
    exact(record['target_token_ids'], context['targets'], 'cell_gold_ids')
    exact(record['per_token_nll'], values, 'cell_raw_nll')
    approximate(record['mean_gold_nll'], average(flat(values)), 'cell_mean_nll')
    if alpha == '0':
        need(raw == context['raw_gold'], 'alpha_zero_entire_gold_byte_parity')
    route_raw = evidence.read(folder / 'routes.bin', PIN['routes'])
    need(route_raw == context['raw_routes'] and record['route_output_sha256'] == PIN['routes'] and
         record['route_output_bytes'] == len(route_raw) and record['routed_token_layer_events'] == 112684,
         'cell_frozen_route_bytes')
    evidence.read(folder / 'stdout.txt', record['stdout_sha256'])
    native_log(evidence.read(folder / 'stderr.txt', record['stderr_sha256']), kind, role, tier, alpha)
    gradient = None
    if kind == 'JVP':
        raw_jvp = evidence.read(folder / 'jvp_gold.bin', record['gradient_output_sha256'])
        gradient = jvp(raw_jvp, context['starts'], context['targets'], context['baseline'])
        exact(record['gradients'], gradient, 'cell_raw_gradients')
        exact(record['native_prefix_completion'], COUNTERS, 'cell_native_prefix')
        need(record['gradient_output_bytes'] == len(raw_jvp), 'cell_jvp_byte_length')
        if (role, tier) in PRIOR_FD:
            reference = prior[(role, tier)]
            need(sha_bytes(raw_jvp) == reference['prior_jvp_output_sha256'], 'prior_q4_full_jvp_byte_parity')
            exact(flat(gradient['per_token_dnll']), reference['prior_native'], 'prior_q4_derivative_parity')
    else:
        need(not (folder / 'jvp_gold.bin').exists() and not (folder / 'jvp_gold.bin').is_symlink(),
             'nonjvp_output_forbidden')
        for key in ('gradients', 'native_prefix_completion', 'gradient_output_sha256', 'gradient_output_bytes'):
            need(record[key] is None, 'nonjvp_gradient_claim:' + key)
    if kind == 'OFF' and not original:
        need(attempt == 1 and not failure_pins, 'inherited_off_cannot_be_retried')
        copied = evidence.receipt(folder / 'INHERITED_ORIGINAL_RUN_RECEIPT.json', V1_OFF, V1_OFF_FILE)
        origin = context['v1_original_off']
        exact(copied, {key: value for key, value in origin.items() if not key.startswith('_')},
              'inherited_original_receipt_verbatim')
        exact(record['execution'], copied['execution'], 'inherited_off_original_execution_telemetry')
        for name, key in (('gold_native.bin', 'output_sha256'), ('routes.bin', 'route_output_sha256'),
                          ('stdout.txt', 'stdout_sha256'), ('stderr.txt', 'stderr_sha256')):
            need(evidence.files[str(folder / name)]['sha256'] == copied[key], 'inherited_off_raw_file:' + name)
    return {**record, '_folder': folder, '_nll': values, '_gradient': gradient}


def verify_v1_recovery_origin(evidence, context):
    """Only the already successful OFF is reusable; interrupted FD is not data."""
    plan = evidence.receipt(V1_ROOT / 'PLAN_RECEIPT.json', V1_PLAN, V1_PLAN_FILE)
    need(plan['schema'] == 'beglin-je01-native-all48-train16-plan/1' and plan['root'] == str(V1_ROOT) and
         plan['runner_source_sha256'] == V1_RUNNER and plan['proven_native_build_receipt_sha256'] == PARENT_BUILD and
         plan['native_binary_sha256'] == PIN['binary'], 'v1_origin_plan')
    evidence.read(V1_ROOT / 'tools/je01_native_all48_train16_runner.py', V1_RUNNER, collect=False)
    evidence.read(V1_ROOT / 'tools/je01_native_all48_train16_independent_audit.py',
                  '270fc991cd0e4378457fe80cfa4f7f924ab69d44d77656ecfb95bb11449730a9', collect=False)
    workflow = evidence.receipt(V1_ROOT / 'WORKFLOW_RECEIPT_0001.json', V1_WORKFLOW, V1_WORKFLOW_FILE)
    need(workflow['status'] == 'FAILED_OR_INTERRUPTED_NO_COMPLETION_CLAIM' and
         workflow['plan_receipt_sha256'] == V1_PLAN and workflow['runner_source_sha256'] == V1_RUNNER and
         workflow['analysis_receipt_sha256'] is None and workflow['independent_audit_receipt_sha256'] is None and
         workflow['independent_validation_ood_accessed'] is False and workflow['production_modified'] is False and
         workflow['allocation_selected'] is False, 'v1_cancelled_workflow_no_result')
    off_folder = V1_ROOT / 'runs/train16/cell_000_off_attempt1'
    failed_folder = V1_ROOT / 'runs/train16/cell_001_fd_r00_Q5_a0_attempt1'
    directory = safe_path(V1_ROOT / 'runs/train16', directory=True)
    need({p.name for p in directory.iterdir()} == {off_folder.name, failed_folder.name},
         'v1_only_off_and_interrupted_fd_no_jvp')
    original_receipt = evidence.receipt(off_folder / 'RUN_RECEIPT.json', V1_OFF, V1_OFF_FILE)
    need(not (off_folder / 'FAILED_RUN_RECEIPT.json').exists() and
         not (failed_folder / 'RUN_RECEIPT.json').exists(), 'v1_success_and_failure_unambiguous')
    failure = evidence.receipt(failed_folder / 'FAILED_RUN_RECEIPT.json', V1_FAILED_FD, V1_FAILED_FD_FILE)
    expected_failure = {'schema': 'beglin-je01-native-all48-train16-cell/1',
                        'status': 'EXECUTION_OR_INTEGRITY_FAILED', 'cell': sequence()[1], 'attempt': 1,
                        'plan_receipt_sha256': V1_PLAN, 'runner_source_sha256': V1_RUNNER,
                        'prior_failure_receipt_sha256': []}
    for key, expected in expected_failure.items():
        exact(failure.get(key), expected, 'v1_interrupted_fd.' + key)
    need(failure['execution']['native_exit_code'] == -15 and failure['execution']['native_started'] is True and
         failure['execution']['stop_reason'] == 'MONITOR_OR_INTERRUPTION_ERROR' and
         'InterruptedError' in failure['error'], 'v1_interruption_not_numerical_rejection')
    scope(failure)
    old_context = {**context, 'root': V1_ROOT, 'plan': plan, 'runner_sha': V1_RUNNER,
                   'identity': {**plan['source_identity'], 'plan_receipt_sha256': V1_PLAN}}
    origin = verify_cell(evidence, old_context, off_folder, sequence()[0], 1, [], None, {}, original=True)
    need(origin['receipt_sha256'] == original_receipt['receipt_sha256'] == V1_OFF,
         'v1_off_receipt_raw_replay')
    confirmation = evidence.receipt(V1_ROOT / 'JE01_ALL48_TRAIN16_INDEPENDENT_AUDIT_V1_INTERRUPTED.json',
                                    V1_INTERRUPTED_AUDIT, V1_INTERRUPTED_AUDIT_FILE)
    confirmation_contract = {
        'status': 'INDEPENDENT_AUDIT_CONFIRMED_PARTIAL_NO_PASS', 'evidence_integrity': 'PASS',
        'experiment_numerical_verdict': 'NOT_COMPLETE', 'completed_native_cell_count': 1,
        'native_jvp_columns': 0, 'fd_verified_coordinates': 0, 'g': None, 'observed_label_directional_gram': None,
        'plan_receipt_sha256': V1_PLAN, 'runner_source_sha256': V1_RUNNER,
        'auditor_source_sha256': '270fc991cd0e4378457fe80cfa4f7f924ab69d44d77656ecfb95bb11449730a9',
        'source_report_receipt_sha256': None, 'all_48_raw_directions_verified': False,
        'all_nine_fd_checkpoints_pass': False,
    }
    for key, expected in confirmation_contract.items():
        exact(confirmation.get(key), expected, 'v1_independent_partial_confirmation.' + key)
    scope(confirmation)
    need(confirmation['verified_files'][str(off_folder / 'RUN_RECEIPT.json')]['sha256'] == V1_OFF_FILE and
         len(confirmation['failed_execution_or_integrity_attempts']) == 1 and
         confirmation['failed_execution_or_integrity_attempts'][0]['receipt_sha256'] == V1_FAILED_FD,
         'v1_independent_confirmation_exact_off_and_failed_fd')
    return origin


def legacy_metric_shape(values):
    """Map independently calculated quantities to the runner's historical field names."""
    return {'tokens': values['tokens'], 'mean_jvp': values['mean_actual'],
            'mean_fixed_fine_fd': values['mean_reference'], 'rms_disagreement': values['rms_disagreement'],
            'max_absolute_disagreement': values['max_absolute_disagreement'],
            'rms_fixed_fine_fd': values['rms_reference'], 'rms_acceptance_threshold': values['rms_acceptance_threshold'],
            'absolute_tolerance': 1e-4, 'relative_tolerance': .05, 'passed': values['passed'],
            'cosine_similarity': values['cosine_similarity'],
            'jvp_observed_label_one_direction_square_mean': values['actual_square_mean'],
            'fixed_fine_observed_label_one_direction_square_mean': values['reference_square_mean'],
            'interpretation': 'ONE_FIXED_HARD_BRANCH_COMPUTATIONAL_DIRECTION_ON_SEEN_TRAIN16',
            'full48_empirical_fisher': None, 'allocation_selected': False}


def verify_checkpoint(evidence, context, measured, records, prior, parent):
    role, tier = measured['role_index'], measured['tier']
    if (role, tier) in PRIOR_FD:
        reference = prior[(role, tier)]
        coarse, fine = reference['coarse'], reference['fine']
        refs = {'parent_audit_receipt_sha256': PARENT_AUDIT,
                'parent_fd_reference_receipt_sha256': parent['roles'][role]['fd_reference_receipt_sha256']}
        kind = 'REUSED_INDEPENDENTLY_AUDITED_Q4_FD'
    else:
        cells = {r['alpha_literal']: r for r in records.values()
                 if r['kind'] == 'FD' and (r['role_index'], r['tier']) == (role, tier)}
        need(set(cells) == set(ALPHAS), 'five_new_fd_cells_required')
        coarse = finite_difference(flat(cells['+0.05']['_nll']), flat(cells['-0.05']['_nll']), .05)
        fine = finite_difference(flat(cells['+0.025']['_nll']), flat(cells['-0.025']['_nll']), .025)
        refs = {'fd_run_receipt_sha256': {alpha: cells[alpha]['receipt_sha256'] for alpha in ALPHAS}}
        kind = 'NEW_FIVE_CELL_FIXED_FD_CHECKPOINT'
    convergence = comparison(coarse, fine)
    agreement = comparison(flat(measured['_gradient']['per_token_dnll']), fine)
    passed = convergence['passed'] and agreement['passed']
    groups = context['sentence_groups']
    regroup = lambda values: [values[p['token_offset_start']:p['token_offset_end']] for p in groups]
    expected = {'schema': 'beglin-je01-native-all48-train16-checkpoint/2', 'scope': SCOPE,
                'status': 'FD_CONVERGENCE_AND_JVP_AGREEMENT_PASS' if passed else 'NUMERICAL_REJECTED',
                'passed': passed, 'kind': kind, 'role_index': role, 'role_key': ROLE_ORDER[role],
                'tier': tier, 'column_index': 2 * role + int(tier == 'Q5'),
                'plan_receipt_sha256': context['plan']['receipt_sha256'], 'source_identity': context['identity'],
                'jvp_run_receipt_sha256': measured['receipt_sha256'], 'jvp_output_sha256': measured['gradient_output_sha256'],
                'fd_convergence': legacy_metric_shape(convergence), 'jvp_vs_fine_fd': legacy_metric_shape(agreement),
                'per_token_coarse_fd': regroup(coarse), 'per_token_fine_fd': regroup(fine), **refs, **NO_ACTION}
    path = context['root'] / f'CHECKPOINT_r{role:02d}_{tier}.json'
    saved = evidence.receipt(path)
    approximate({key: value for key, value in saved.items() if key != 'receipt_sha256'}, expected, 'checkpoint')
    exact(saved['per_token_coarse_fd'], regroup(coarse), 'checkpoint_raw_coarse')
    exact(saved['per_token_fine_fd'], regroup(fine), 'checkpoint_raw_fine')
    return {**saved, '_path': path, '_independent_convergence': convergence, '_independent_agreement': agreement}


def verify_prefix(evidence, context, prior, parent):
    base = context['root'] / 'runs/train16'
    planned = sequence()
    by_label = {cell_label(cell): cell for cell in planned}
    attempts = {index: {} for index in range(79)}
    if base.exists():
        safe_path(base, directory=True)
        need({path.name for path in base.parent.iterdir()} == {'train16'}, 'unplanned_dataset_run_directory')
        for path in base.iterdir():
            match = re.fullmatch(r'(.+)_attempt([1-9]\d*)', path.name)
            need(match is not None and match.group(1) in by_label and 1 <= int(match.group(2)) <= 100,
                 'unplanned_cell_or_attempt')
            safe_path(path, directory=True)
            attempts[by_label[match.group(1)]['cell_index']][int(match.group(2))] = path
    records, checkpoints, failed_records = {}, {}, []
    gap, rejected = False, False
    for cell in planned:
        index = cell['cell_index']
        found = attempts[index]
        need(not ((gap or rejected) and found), 'native_attempt_after_gap_or_numerical_rejection')
        need(sorted(found) == list(range(1, len(found) + 1)), 'attempt_sequence_gap')
        failure_pins, saved = [], None
        for attempt, folder in sorted(found.items()):
            need(saved is None, 'attempt_after_success')
            good, bad = folder / 'RUN_RECEIPT.json', folder / 'FAILED_RUN_RECEIPT.json'
            need(good.exists() != bad.exists() and not good.is_symlink() and not bad.is_symlink(),
                 'ambiguous_or_incomplete_attempt')
            if good.exists():
                control = records[0]['receipt_sha256'] if index else None
                saved = verify_cell(evidence, context, folder, cell, attempt, failure_pins, control, prior)
            else:
                need(cell['kind'] != 'OFF', 'inherited_off_cannot_be_failed_native_attempt')
                failed = evidence.receipt(bad)
                scope(failed)
                expected = {'schema': 'beglin-je01-native-all48-train16-cell/2',
                            'status': 'EXECUTION_OR_INTEGRITY_FAILED', 'cell': cell, 'attempt': attempt,
                            'source_identity': context['identity'], 'plan_receipt_sha256': context['plan']['receipt_sha256'],
                            'runner_source_sha256': context['runner_sha'], 'prior_failure_receipt_sha256': failure_pins}
                for key, value in expected.items():
                    exact(failed.get(key), value, 'failed_attempt.' + key)
                need(isinstance(failed['error'], str) and failed['error'], 'failure_reason')
                failure_pins.append(failed['receipt_sha256'])
                failed_records.append({'cell_index': index, 'attempt': attempt,
                                       'receipt_sha256': failed['receipt_sha256'], 'error': failed['error']})
        if saved is None:
            gap = True
        else:
            records[index] = saved
            pair = (cell['role_index'], cell['tier'])
            if cell['kind'] == 'JVP' and pair in PRIOR_FD + NEW_FD:
                checkpoint = verify_checkpoint(evidence, context, saved, records, prior, parent)
                checkpoints[pair] = checkpoint
                rejected = not checkpoint['passed']
    expected_checks = {f'CHECKPOINT_r{r:02d}_{t}.json' for r, t in checkpoints}
    need({p.name for p in context['root'].glob('CHECKPOINT_*.json')} == expected_checks, 'checkpoint_inventory')
    return records, checkpoints, failed_records


def verify_psd_claim(claim, independent):
    need(isinstance(claim, dict) and set(claim) == {
        'minimum_eigenvalue', 'maximum_eigenvalue', 'trace', 'eigenvalues_ascending',
        'offdiagonal_residual_bound', 'acceptance_lower_bound', 'passed', 'method'}, 'psd_claim_fields')
    need(claim['method'] == 'REAL_SYMMETRIC_CYCLIC_JACOBI_WITH_RESIDUAL_BOUND', 'psd_claim_method')
    approximate(claim['trace'], independent['trace'], 'psd_trace')
    approximate(claim['minimum_eigenvalue'], independent['minimum_eigenvalue'], 'psd_minimum')
    approximate(claim['maximum_eigenvalue'], independent['maximum_eigenvalue'], 'psd_maximum')
    approximate(claim['eigenvalues_ascending'], independent['eigenvalues_ascending'], 'psd_spectrum')
    exact(claim['eigenvalues_ascending'], sorted(claim['eigenvalues_ascending']), 'psd_sorted')
    approximate(claim['minimum_eigenvalue'], claim['eigenvalues_ascending'][0], 'psd_minimum_consistency')
    approximate(claim['maximum_eigenvalue'], claim['eigenvalues_ascending'][-1], 'psd_maximum_consistency')
    approximate(claim['acceptance_lower_bound'], -independent['negative_tolerance'], 'psd_fixed_tolerance')
    residual = claim['offdiagonal_residual_bound']
    need(type(residual) in (int, float) and math.isfinite(residual) and
         0 <= residual <= 1e-12 * max(1., independent['trace']), 'psd_reported_residual')
    passed = claim['minimum_eigenvalue'] - residual >= claim['acceptance_lower_bound']
    exact(claim['passed'], passed, 'psd_reported_decision')
    exact(claim['passed'], independent['passed'], 'independent_psd_decision')


def verify_matrix_artifact(evidence, context, measured, matrix):
    path = context['root'] / 'RAW_DERIVATIVE_MATRIX.json'
    saved = evidence.receipt(path)
    expected = {'schema': 'beglin-je01-native-all48-train16-raw-matrix/2', 'scope': SCOPE,
                'plan_receipt_sha256': context['plan']['receipt_sha256'], 'runner_source_sha256': context['runner_sha'],
                'shape': [656, 48], 'coordinate_order': coordinate_order(), 'token_rows': context['token_rows'],
                'per_token_dnll': matrix, 'sentence_groups': context['sentence_groups'],
                'jvp_receipt_sha256': [measured[i]['receipt_sha256'] for i in range(48)],
                'jvp_output_sha256': [measured[i]['gradient_output_sha256'] for i in range(48)], **NO_ACTION}
    exact({key: value for key, value in saved.items() if key != 'receipt_sha256'}, expected,
          'entire_raw_matrix_reconstruction')
    return saved


def verify_analysis(evidence, context, records, checkpoints, measured, matrix_receipt, statistics):
    gradient, gram, sentences, psd = statistics
    root = context['root']
    saved = evidence.receipt(root / 'JE01_ALL48_TRAIN16_ANALYSIS.json')
    scope(saved)
    directions = []
    for coordinate in coordinate_order():
        pair = (coordinate['role_index'], coordinate['tier'])
        directions.append({**coordinate, 'raw_native_validation': 'PRIMAL_ROUTE_TARGET_FINITE_FULL_PREFIX_PASS',
                           'fd_validation': 'CHECKPOINT_PASS' if pair in checkpoints else 'NOT_INDIVIDUALLY_FD_CHECKED',
                           'checkpoint_receipt_sha256': checkpoints[pair]['receipt_sha256'] if pair in checkpoints else None,
                           'jvp_receipt_sha256': measured[coordinate['column_index']]['receipt_sha256']})
    elapsed = {kind: [r['execution']['elapsed_seconds'] for r in records.values() if r['kind'] == kind]
               for kind in ('OFF', 'FD', 'JVP')}
    matrix_path = root / 'RAW_DERIVATIVE_MATRIX.json'
    expected = {
        'schema': 'beglin-je01-native-all48-train16-analysis/2', 'scope': SCOPE,
        'status': 'ALL48_NATIVE_COMPLETE_CHECKPOINTS_AND_PSD_PASS', 'evidence_integrity': 'PASS',
        'checkpoint_numerical_verdict': 'PASS', 'source_identity': context['identity'],
        'plan_receipt_sha256': context['plan']['receipt_sha256'], 'runner_source_sha256': context['runner_sha'],
        'case_ids': list(range(16)), 'gold_tokens': 656, 'completed_native_cell_count': 79,
        'completed_evidence_cell_count': 79, 'native_cells_executed_in_this_plan': 78, 'inherited_native_cell_count': 1,
        'native_jvp_columns': 48, 'fd_verified_coordinates': 9,
        'fd_coverage_interpretation': 'NINE_PREDECLARED_CHECKPOINTS_NOT_ALL_48_INDIVIDUAL_FD_VALIDATIONS',
        'coordinate_order': coordinate_order(), 'directions': directions,
        'cell_receipts': [{'cell_index': index, 'attempt': records[index]['attempt'],
                           'path': str(records[index]['_folder'] / 'RUN_RECEIPT.json'),
                           'receipt_sha256': records[index]['receipt_sha256']} for index in range(79)],
        'checkpoint_receipts': [{'role_index': r, 'tier': t, 'receipt_sha256': checkpoints[(r, t)]['receipt_sha256']}
                               for r, t in PRIOR_FD + NEW_FD],
        'raw_derivative_matrix': {'path': str(matrix_path), 'shape': [656, 48],
                                  'receipt_sha256': matrix_receipt['receipt_sha256'],
                                  'file_sha256': evidence.files[str(matrix_path)]['sha256']},
        'g': gradient, 'g_definition': 'MEAN_TOKEN_DNLL', 'score_definition': 'NEGATIVE_DNLL',
        'observed_label_directional_gram': {'definition': 'MEAN_TOKEN_DNLL_OUTER_PRODUCT', 'matrix': gram,
            'centering': 'NONE', 'tokens': 656, 'coordinate_order': coordinate_order(),
            'interpretation': 'EXPLORATORY_48_DIRECTION_FIXED_BRANCH_GRAM_ON_SEEN_TRAIN16',
            'used_for_policy_selection': False},
        'sentence_statistics': sentences,
        'measured_timing': {'seconds_by_kind': elapsed,
                            'total_native_elapsed_seconds': math.fsum(v for values in elapsed.values() for v in values),
                            'native_elapsed_seconds_this_plan': math.fsum(r['execution']['elapsed_seconds'] for r in records.values()
                                                                         if r['native_executed_in_this_plan']),
                            'inherited_native_elapsed_seconds': records[0]['execution']['elapsed_seconds']},
        'post_run_original_af4_sha256': PIN['model'],
        'natural_routing_stability': 'NOT_ESTABLISHED_BY_FORCED_ROUTE_REPLAY',
        'independent_audit_required_before_train240': True, **NO_ACTION,
    }
    approximate({key: value for key, value in saved.items() if key not in ('receipt_sha256', 'psd')},
                expected, 'analysis_raw_recomputation')
    verify_psd_claim(saved['psd'], psd)
    return saved


def verify_hold(evidence, context, records, checkpoints, psd):
    root = context['root']
    path = root / 'NUMERICAL_HOLD_RECEIPT.json'
    rejected = [checkpoint for checkpoint in checkpoints.values() if not checkpoint['passed']]
    expected_base = {'schema': 'beglin-je01-native-all48-train16-numerical-hold/2', 'scope': SCOPE,
                     'status': 'NUMERICAL_HOLD_NO_FURTHER_NATIVE_CELLS',
                     'plan_receipt_sha256': context['plan']['receipt_sha256'],
                     'completed_native_cell_count': len(records), 'completed_cell_indices': list(range(len(records))),
                     'completed_evidence_cell_count': len(records), 'inherited_native_cell_count': int(bool(records)),
                     'native_cells_executed_in_this_plan': max(0, len(records) - 1),
                     'g': None, 'observed_label_directional_gram': None, **NO_ACTION}
    if rejected:
        need(len(rejected) == 1, 'multiple_failed_checkpoints_despite_stop_rule')
        checkpoint = rejected[0]
        saved = evidence.receipt(path)
        expected = {**expected_base, 'checkpoint_receipt_sha256': checkpoint['receipt_sha256'],
                    'checkpoint_path': str(checkpoint['_path'])}
        exact({k: v for k, v in saved.items() if k != 'receipt_sha256'}, expected, 'checkpoint_hold')
        need(not (root / 'JE01_ALL48_TRAIN16_ANALYSIS.json').exists(), 'complete_analysis_after_checkpoint_failure')
    elif psd is not None and not psd['passed']:
        need(len(records) == 79 and len(checkpoints) == 9, 'psd_hold_requires_complete_raw_design')
        saved = evidence.receipt(path)
        expected = {**expected_base, 'reason': 'GRAM_PSD_REJECTED', 'post_run_original_af4_sha256': PIN['model']}
        exact({k: v for k, v in saved.items() if k not in ('receipt_sha256', 'psd')}, expected, 'psd_hold')
        verify_psd_claim(saved['psd'], psd)
        analysis = evidence.receipt(root / 'JE01_ALL48_TRAIN16_ANALYSIS.json')
        expected_report = {'schema': 'beglin-je01-native-all48-train16-analysis/2', 'scope': SCOPE,
                           'status': 'NUMERICAL_HOLD_NO_FURTHER_NATIVE_CELLS',
                           'plan_receipt_sha256': context['plan']['receipt_sha256'], 'reason': 'GRAM_PSD_REJECTED',
                           'evidence_integrity': 'PASS', 'checkpoint_numerical_verdict': 'PASS', 'psd_verdict': 'FAIL',
                           'completed_native_cell_count': 79, 'native_jvp_columns': 48, 'fd_verified_coordinates': 9,
                           'completed_evidence_cell_count': 79, 'native_cells_executed_in_this_plan': 78,
                           'inherited_native_cell_count': 1,
                           'hold_receipt_sha256': saved['receipt_sha256'], 'g': None,
                           'observed_label_directional_gram': None, **NO_ACTION}
        exact({k: v for k, v in analysis.items() if k not in ('receipt_sha256', 'psd')}, expected_report, 'psd_hold_analysis')
        verify_psd_claim(analysis['psd'], psd)
    else:
        need(not path.exists() and not path.is_symlink(), 'unexplained_numerical_hold')
        return None
    need(not (root / 'RAW_DERIVATIVE_MATRIX.json').exists() and not (root / 'RAW_DERIVATIVE_MATRIX.json').is_symlink(),
         'matrix_published_on_numerical_hold')
    return saved


def verify_progress(evidence, context, records, checkpoints, analysis, hold):
    root = context['root']
    chunks = sorted(root.glob('CHUNK_RECEIPT_*.json'))
    need([p.name for p in chunks] == [f'CHUNK_RECEIPT_{i:04d}.json' for i in range(1, len(chunks) + 1)],
         'chunk_sequence_gap')
    prior_after = 0
    for path in chunks:
        chunk = evidence.receipt(path)
        scope(chunk)
        need(chunk['schema'] == 'beglin-je01-native-all48-train16-chunk/2' and
             chunk['plan_receipt_sha256'] == context['plan']['receipt_sha256'] and
             chunk['runner_source_sha256'] == context['runner_sha'], 'chunk_plan_source')
        before, after = chunk['completed_cells_before'], chunk['completed_cells_after']
        need(type(before) is int and type(after) is int and 1 <= before and prior_after <= before <= after <= len(records) and
             chunk['new_native_cells'] == after - before and type(chunk['requested_max_new_cells']) is int and
             1 <= chunk['requested_max_new_cells'] <= 79 and after - before <= chunk['requested_max_new_cells'] and
             type(chunk['requested_max_cell_attempt']) is int and 1 <= chunk['requested_max_cell_attempt'] <= 100,
             'chunk_progress_counts')
        exact(chunk['inherited_native_cell_count'], 1, 'chunk_inherited_off_count')
        exact(chunk['native_cells_executed_in_this_plan'], after - 1, 'chunk_new_native_count')
        exact(chunk['completed_cell_receipt_sha256'], [records[i]['receipt_sha256'] for i in range(after)],
              'chunk_successful_prefix_binding')
        prior_after = after
        rejected_here = any(not checkpoint['passed'] and
                            any(records[i]['receipt_sha256'] == checkpoint['jvp_run_receipt_sha256'] for i in range(after))
                            for checkpoint in checkpoints.values())
        terminal_psd = bool(hold and hold.get('reason') == 'GRAM_PSD_REJECTED' and after == 79)
        terminal = rejected_here or terminal_psd
        complete = after == 79 and not terminal
        status = ('NUMERICAL_HOLD_NO_FURTHER_NATIVE_CELLS' if terminal else
                  'ALL48_NATIVE_COMPLETE_CHECKPOINTS_AND_PSD_PASS' if complete else 'PARTIAL_NO_MATRIX_NO_PASS')
        exact(chunk['status'], status, 'chunk_status')
        exact(chunk['next_cell_index'], None if terminal or complete else after, 'chunk_next_cell')
        exact(chunk['more_native_cells_required'], not terminal and not complete, 'chunk_continuation')
        if complete:
            need(analysis is not None and chunk['analysis_receipt_sha256'] == analysis['receipt_sha256'],
                 'complete_chunk_analysis_binding')
        elif terminal_psd:
            report = evidence.receipt(root / 'JE01_ALL48_TRAIN16_ANALYSIS.json')
            exact(chunk['analysis_receipt_sha256'], report['receipt_sha256'], 'psd_hold_chunk_analysis_binding')
        else:
            need(chunk['analysis_receipt_sha256'] is None, 'partial_chunk_no_analysis_claim')
    for path in sorted(root.glob('PARTIAL_ANALYSIS_*.json')):
        partial = evidence.receipt(path)
        scope(partial)
        need(partial['schema'] == 'beglin-je01-native-all48-train16-analysis/2' and
             partial['plan_receipt_sha256'] == context['plan']['receipt_sha256'] and
             partial['g'] is None and partial['observed_label_directional_gram'] is None and
             partial['status'] in ('PARTIAL_NO_MATRIX_NO_PASS', 'NUMERICAL_HOLD_NO_FURTHER_NATIVE_CELLS'),
             'partial_report_cannot_claim_pass')
        count = partial['completed_native_cell_count']
        need(type(count) is int and 0 <= count <= len(records), 'partial_report_count')
        exact(partial['completed_evidence_cell_count'], count, 'partial_evidence_count')
        exact(partial['inherited_native_cell_count'], int(count > 0), 'partial_inherited_count')
        exact(partial['native_cells_executed_in_this_plan'], max(0, count - 1), 'partial_new_native_count')
        exact(partial['native_jvp_columns'], sum(records[i]['kind'] == 'JVP' for i in range(count)),
              'partial_report_jvp_count')
        seen = [checkpoint for checkpoint in checkpoints.values()
                if any(records[i]['receipt_sha256'] == checkpoint['jvp_run_receipt_sha256'] for i in range(count))]
        exact(partial['checkpoint_receipt_sha256'], [c['receipt_sha256'] for c in seen], 'partial_checkpoint_binding')
        expected_status = ('NUMERICAL_HOLD_NO_FURTHER_NATIVE_CELLS' if any(not c['passed'] for c in seen)
                           else 'PARTIAL_NO_MATRIX_NO_PASS')
        exact(partial['status'], expected_status, 'partial_scientific_status')
    return [evidence.files[str(path)]['sha256'] for path in chunks]


def audit(args):
    root = safe_path(args.root, directory=True)
    need(root == ROOT and args.expected_build_receipt == PARENT_BUILD, 'external_root_build')
    evidence = Evidence()
    parent, build, _ = verify_parent(evidence, args.expected_build_receipt)
    context = verify_dataset(evidence)
    mappings = verify_bank(evidence)
    context['v1_original_off'] = verify_v1_recovery_origin(evidence, context)
    plan, identity = verify_plan(args, evidence, context)
    verify_source_ledger(evidence, root, mappings)
    prior = inherited_fd(evidence, context, parent)
    context.update({'root': root, 'identity': identity, 'plan': plan, 'runner_sha': args.expected_runner_sha})
    records, checkpoints, failures = verify_prefix(evidence, context, prior, parent)
    measured = {r['column_index']: r for r in records.values() if r['kind'] == 'JVP'}
    need(len(measured) == sum(r['kind'] == 'JVP' for r in records.values()), 'unique_completed_jvp_columns')
    complete = len(records) == 79
    matrix = gradient = gram = sentences = psd = matrix_receipt = analysis = None
    if complete:
        need(set(measured) == set(range(48)) and set(checkpoints) == set(PRIOR_FD + NEW_FD) and
             all(c['passed'] for c in checkpoints.values()), 'complete_all48_nine_checkpoint_gate')
        columns = [flat(measured[index]['_gradient']['per_token_dnll']) for index in range(48)]
        matrix = [list(row) for row in zip(*columns)]
        gradient, gram, sentences = gram_statistics(matrix, context['targets'])
        psd = spectral_check(gram)
    hold = verify_hold(evidence, context, records, checkpoints, psd)
    if complete and not hold:
        matrix_receipt = verify_matrix_artifact(evidence, context, measured, matrix)
        analysis = verify_analysis(evidence, context, records, checkpoints, measured, matrix_receipt,
                                   (gradient, gram, sentences, psd))
    elif not complete and hold is None:
        need(not (root / 'RAW_DERIVATIVE_MATRIX.json').exists() and
             not (root / 'JE01_ALL48_TRAIN16_ANALYSIS.json').exists(), 'aggregate_published_before_complete_gates')
    chunks = verify_progress(evidence, context, records, checkpoints, analysis, hold)
    need(not complete or hold is not None or chunks, 'complete_result_has_no_execution_chunk_receipt')
    # Full-file post-run identity: no inference, sparse hash, or receipt-only substitute.
    evidence.read(MODEL, PIN['model'], collect=False)
    need(evidence.files[str(MODEL)]['size_bytes'] == 9814343680, 'original_model_full_length')
    passed = complete and hold is None
    status = ('INDEPENDENT_AUDIT_CONFIRMED_ALL48_TRAIN16_RAW_AND_NINE_CHECKPOINTS_PASS' if passed else
              'INDEPENDENT_AUDIT_CONFIRMED_NUMERICAL_HOLD' if hold else
              'INDEPENDENT_AUDIT_CONFIRMED_PARTIAL_NO_PASS')
    summaries = []
    for pair in PRIOR_FD + NEW_FD:
        if pair in checkpoints:
            checked = checkpoints[pair]
            summaries.append({'role_index': pair[0], 'role_key': ROLE_ORDER[pair[0]], 'tier': pair[1],
                              'column_index': 2 * pair[0] + int(pair[1] == 'Q5'), 'passed': checked['passed'],
                              'fd_convergence': checked['_independent_convergence'],
                              'jvp_vs_fine_fd': checked['_independent_agreement'],
                              'checkpoint_receipt_sha256': checked['receipt_sha256']})
    result = {
        'schema': 'beglin-je01-native-all48-train16-independent-raw-audit/2', 'status': status,
        'observed_at': datetime.now(timezone.utc).isoformat(), 'root': str(root), 'scope': SCOPE,
        'evidence_integrity': 'PASS', 'experiment_numerical_verdict': 'PASS' if passed else 'FAIL' if hold else 'NOT_COMPLETE',
        'completed_native_cell_count': len(records), 'completed_evidence_cell_count': len(records),
        'planned_evidence_cell_count': 79, 'planned_native_cell_count': 78,
        'native_cells_executed_in_this_plan': max(0, len(records) - 1),
        'inherited_native_cell_count': int(bool(records)), 'planned_inherited_native_cell_count': 1,
        'native_jvp_columns': len(measured), 'planned_native_jvp_columns': 48,
        'fd_verified_coordinates': sum(c['passed'] for c in checkpoints.values()), 'planned_fd_checkpoint_count': 9,
        'fd_coverage_interpretation': 'NINE_PREDECLARED_CHECKPOINTS_NOT_ALL_48_INDIVIDUAL_FD_VALIDATIONS',
        'case_ids': list(range(16)), 'gold_tokens': 656, 'sentence_groups_count': 16,
        'baseline_nll': average(flat(context['baseline'])), 'coordinate_order': coordinate_order(),
        'checkpoints': summaries, 'failed_execution_or_integrity_attempts': failures,
        'plan_receipt_sha256': plan['receipt_sha256'], 'external_plan_pin': args.expected_plan_receipt,
        'runner_source_sha256': args.expected_runner_sha, 'proven_native_build_receipt_sha256': build['receipt_sha256'],
        'native_output_path_contract': context['native_output_path_contract'],
        'recovery': recovery_contract(), 'v1_original_off_receipt_sha256': V1_OFF,
        'v1_independent_partial_confirmation_receipt_sha256': V1_INTERRUPTED_AUDIT,
        'parent_three_role_independent_audit_sha256': PARENT_AUDIT,
        'parent_three_role_independent_audit_file_sha256': PARENT_AUDIT_FILE,
        'source_report_receipt_sha256': analysis['receipt_sha256'] if analysis else None,
        'numerical_hold_receipt_sha256': hold['receipt_sha256'] if hold else None,
        'raw_matrix_receipt_sha256': matrix_receipt['receipt_sha256'] if matrix_receipt else None,
        'reconstructed_matrix_values_sha256': canonical(matrix) if passed else None,
        'all_48_raw_directions_verified': complete,
        'all_nine_fd_checkpoints_pass': len(checkpoints) == 9 and all(c['passed'] for c in checkpoints.values()),
        'g': gradient if passed else None, 'g_definition': 'MEAN_TOKEN_DNLL', 'score_definition': 'NEGATIVE_DNLL',
        'observed_label_directional_gram': gram if passed else None,
        'sentence_groups': context['sentence_groups'], 'sentence_statistics': sentences if passed else None,
        'independent_psd_check': psd,
        'bank_role_tier_mapping': mappings, 'original_af4_full_file_sha_verified_after_run': PIN['model'],
        'native_inference_rerun': False, 'experiment_runner_or_helper_imported': False,
        'natural_routing_stability': 'NOT_ESTABLISHED_BY_FORCED_ROUTE_REPLAY',
        'scientific_limits': [
            'Previously seen TRAIN16: 16 sentence clusters and 656 correlated token targets, not an independent test.',
            'Only nine predeclared columns have finite-difference convergence and JVP-agreement checks.',
            'The 48 by 48 matrix is an uncentered observed-label score second moment in fixed-route computational directions.',
            'Gram PSD follows algebraically; its numerical check is not independent evidence that all 48 derivatives are correct.',
            'A fixed-route local surrogate may guide explicitly labelled TRAIN240 research but does not establish finite policy loss.',
            'Any discrete same-byte allocation still requires finite native natural-route NLL testing; no allocation is selected here.',
            'No parameter Fisher, expected Fisher, NLL Hessian, prospective generalization, or heldout result is claimed.',
            'Original BF16 source role hashes are inherited from the exact pinned source ledger; this audit rehashes the full AF4 and direction bank.',
        ],
        **NO_ACTION,
        'auditor_source_sha256': sha_bytes(Path(__file__).read_bytes()), 'verified_files': evidence.files,
    }
    result['receipt_sha256'] = canonical(result)
    return result


def write_output(path, root, result):
    path, root = Path(path), Path(root)
    audit_name = (re.fullmatch(r'JE01_ALL48_TRAIN16_INDEPENDENT_AUDIT(?:_[A-Za-z0-9_-]+)?\.json', path.name) is not None
                  or path.name == 'ALL48_PREFLIGHT_INDEPENDENT_AUDIT.json')
    need(path.is_absolute() and path.parent == root and path.parent.resolve() == path.parent and
         audit_name and
         not path.exists() and not path.is_symlink(), 'fresh_audit_only_output')
    # No rename/replace/link; O_EXCL forbids overwriting any prior sealed artifact.
    with path.open('x') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def self_test():
    equal = comparison([.25] * 656, [.25] * 656)
    rejected = comparison([.5] * 656, [.25] * 656)
    need(equal['passed'] and not rejected['passed'], 'selftest_fd_decision')
    spectrum = spectral_check([[2., 1.], [1., 2.]])
    approximate(spectrum['eigenvalues_ascending'], [1., 3.], 'selftest_known_spectrum')
    need(spectrum['passed'] and not spectral_check([[1., 2.], [2., 1.]])['passed'], 'selftest_psd_rejection')
    need(len(sequence()) == 79 and len({cell_label(cell) for cell in sequence()}) == 79 and
         sum(cell['kind'] == 'JVP' for cell in sequence()) == 48 and sum(cell['kind'] == 'FD' for cell in sequence()) == 30,
         'selftest_complete_design')
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
        try:
            strict_json(raw)
        except ValueError:
            pass
        else:
            raise AssertionError('selftest invalid JSON accepted')
    return {'status': 'PASS_ALL48_INDEPENDENT_AUDITOR_SELF_TEST_ONLY', 'actual_model_evidence': False,
            'numerical_rejection_and_psd_controls': 'PASS', 'json_corruption_controls': 'PASS',
            'planned_evidence_cells': 79, 'planned_native_cells': 78, 'inherited_off_cells': 1,
            'jvp_columns': 48, 'new_fd_cells': 30, 'fd_checkpoints': 9}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--expected-build-receipt')
    parser.add_argument('--expected-plan-receipt')
    parser.add_argument('--expected-runner-sha')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True, allow_nan=False))
        return 0
    need(args.root is not None and args.expected_build_receipt is not None and
         args.expected_plan_receipt is not None and args.expected_runner_sha is not None,
         'root_external_build_plan_runner_arguments_required')
    result = audit(args)
    if args.output is not None:
        write_output(args.output, args.root, result)
        print(json.dumps({key: result[key] for key in ('status', 'evidence_integrity', 'experiment_numerical_verdict',
                                                       'completed_native_cell_count', 'native_jvp_columns',
                                                       'fd_verified_coordinates', 'receipt_sha256')},
                         sort_keys=True, allow_nan=False))
    else:
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result['experiment_numerical_verdict'] == 'PASS' else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({'status': 'FAIL_ALL48_EVIDENCE_INTEGRITY', 'evidence_integrity': 'FAIL',
                          'error': type(error).__name__ + ': ' + str(error)}, sort_keys=True), file=sys.stderr)
        sys.exit(1)

