#!/usr/bin/env python3
"""Independently audit the sealed JE01 one-column native JVP experiment.

Uses only the Python standard library. Does not import the experiment runner,
execute inference, select a bit allocation, or open validation/OOD data.
All numerical quantities are recomputed from original native binary arrays.
The default audit also streams the complete 9.814 GB original AF4 file through
SHA-256. Run through normal resource admission; --self-test needs no model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

WORK = Path('/Volumes/D50/TailnetCommander/work')
DEFAULT_ROOT = WORK / 'je01_native_one_column_jvp_20261123_v1'
ORDINARY = WORK / 'je01_role_directional_20261123_v1'
FIXED = WORK / 'je01_role_directional_fixed_20261123_v1'
BANK = WORK / 'je01_role_direction_bank_20261123_v1'
MODEL = WORK / 'hcp-p18-external-multimetric-20261015/native_moe_base_source_copy/weights_moe/deepseek_moe_af.bin'
PIN = {
    'model': '9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4',
    'gold': 'b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c',
    'routes': '7e5811d13953f99db5df55169bf465f039acb9224f626336d247168e0f570711',
    'meta': 'be0dab12624efb4fae17538b4c5e6ef76bd09aebf4c09b1b428fc1d8f0fff406',
    'manifest': 'fab455198ef37d443b20af8e3029df3ddf4f4ba0f6d584eceeb262d26d6a146d',
    'dataset': 'a0091a208adaa559788a942149fe28f12f1de4e03dfbbafe840a5acce50f3e2a',
    'bank': '3b18257d22caddd0916292bef692110f9eeaf0edf83ad3fd4d5417fdf3443bb8',
    'bank_tsv': 'fdc801f59f9c6438dd60bb33fab9a1332cb19c6e15eb15bef7425b7fbd522e4f',
    'bank_receipt': '4994ec762b3a9b8a44015bad18f4b5bc7ea26505208384146238e7def28464d4',
    'ordinary_off': '85d8e28d51fbc0e2d319f0e60e5f9e6ac3d2f9050b2e386d81a969f6ad8f672a',
    'build': '811605094dd704b46b3453c01199fa44946653e5acee05083a5f2b0d3ffe0ff3',
    'binary': '559ed0aae0181c03e3ca617fa83ff8f727f3cfb57bde09b3cf3c2ea2dda4c3e6',
    'patch_receipt': '1ffe363507b75e14f920640ed96c374e34a41f37ee1a9f040635c4d0e7d8cf61',
    'cpp': '56301599354a4c3081428ca822c1d2768ff0ce6b9b8589331ecb8d6f153077b5',
    'c': '6b30ab9997df17af23fc8af2adf5ba2c8fea64489378427e008f97ce7d904067',
    'cmake': '6775ad35ecb67d45000955a7a2efbabd523f55b23c8787c9967959fbffb018e8',
    'adapter': '9ae9d98b921334bae321e583908e71bdc9c08178cbf3ca5d7a1d4d4e552a09ec',
    'runner': 'ffd632d79d948119ad261e2720ebbcd7c5076af21070aeaf60bf0ca8e7d79bd6',
    'patch': '518c65b7ca22c1df2a257f1f30c4e71fb12255f9d657793b961f4feed6e387bf',
    'smoke': 'b0825ab7a048f7d95b155592760e10d73e3df2b112230adbf7b348f388ca1595',
    'fixed_binary': '46ea1f61fdd7483767ebc20900f4c97f9663a51148c74e566d2f66ccc848e688',
    'fixed_audit': '87f239d900d9aff3921ab99d478024b13002e80b3b3d65683d4a055e4147598c',
    'fixed_audit_file': 'a64d13644a47396c8d514d60f6b5eee2f5683ef91975bc7433ab0260066208b9',
    'off': '5fd7ce1cbe98c91283bb7fb4f042e4eb23e1ceedc7992c32acc0bc0244603ce4',
    'jvp': '81c743cff8079464c8b4a83d4307249b6239ad54108a2f17c13d5bd35a262315',
    'comparison': '55b6ea73d6ac65f8d561951d19bb4f15365dafba8751a34089356d7edb2f4f20',
    'role_order': 'd232ca637eb17a1b40bf29f75e0590d8f63ecbc9aa950fb838a34b6fa90259a1',
}
HELPERS = {
    'je01_role_directional_native_runner.py': '5607c0afef3bd4b1a61bcab6622267acc6b6137fc65ba0e79873e92e31d7e9e6',
    'je01_role_directional_fixed_native.py': 'cffe2e1dd8298681f3a134caace742ddc9fea77209c58683cd8d2f48fe6284e7',
    'je01_role_directional_fixed_routes_patch.py': '233937b59293bec11300b66450b819dc48bb09b8d28eb6e98ca81aac7f511019',
    'je01_role_directional_bank.py': '045fd22d86e4ce18f967d66830a1f71b0c61ad6ab5c5889d953cb9a8d05bce33',
}
META_MAGIC = b'JE01_ORIGINAL_FULL_GOLD_TARGETS_META_V1\n'
GOLD_MAGIC = b'JE01_ORIGINAL_AF_FULL_GOLD_NLL_V1\n'
ROUTE_MAGIC = b'JE01_ROLE_DIRECTION_ROUTES_V1\n'
JVP_MAGIC = b'JE01_NATIVE_ONE_COLUMN_JVP_GOLD_V1\n'
ROLE = [12, 13, 'gate_proj']
CASE_IDS = list(range(16))
COMPLETION = {'tokens': 4334, 'cases': 32, 'layers': 117018, 'gold': 656}
FILES = {}


def require(test, label):
    if not test:
        raise ValueError('JE01_INDEPENDENT_JVP_AUDIT:' + label)


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def verify_object(obj, pin=None):
    require(isinstance(obj, dict), 'receipt_object')
    digest = obj.get('receipt_sha256')
    require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'receipt_hash')
    require(canonical({k: v for k, v in obj.items() if k != 'receipt_sha256'}) == digest,
            'receipt_canonical_content')
    require(pin is None or digest == pin, 'receipt_external_pin')
    return obj


def safe_file(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file()
            and not path.is_symlink(), 'file_or_parent_link:' + str(path))
    return path


def inspect(path, expected=None, *, binary=False):
    path = safe_file(path)
    before = path.stat()
    digest = hashlib.sha256()
    parts = [] if binary else None
    with path.open('rb') as stream:
        while True:
            block = stream.read(2 << 20)
            if not block:
                break
            digest.update(block)
            if parts is not None:
                parts.append(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'file_changed_during_read:' + str(path))
    actual = digest.hexdigest()
    require(expected is None or actual == expected, 'file_sha:' + str(path))
    FILES[str(path)] = {'sha256': actual, 'size_bytes': after.st_size}
    return b''.join(parts) if binary else actual


def receipt(path, expected=None, file_sha=None):
    return verify_object(json.loads(inspect(path, file_sha, binary=True)), expected)


def close(actual, expected, label):
    require(type(actual) in (int, float) and type(expected) in (int, float)
            and math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), 'metric:' + label)


def flatten(rows):
    return [item for row in rows for item in row]


def mean(values):
    return math.fsum(values) / len(values)


def decode_meta(raw):
    require(raw.startswith(META_MAGIC) and len(raw) == len(META_MAGIC) + 16 * 1544,
            'metadata_magic_or_size')
    starts, targets = [], []
    for i in range(16):
        offset = len(META_MAGIC) + i * 1544
        start, count = struct.unpack_from('<II', raw, offset)
        ids = struct.unpack_from('<384i', raw, offset + 8)
        require(start >= 2 and count >= 1 and start + count <= 384, 'metadata_bounds')
        require(all(0 <= v < 102400 for v in ids[:count]) and
                all(v == -1 for v in ids[count:]), 'metadata_targets_or_padding')
        starts.append(start)
        targets.append(list(ids[:count]))
    require(sum(map(len, targets)) == 656, 'metadata_gold_count')
    return starts, targets


def decode_gold(raw, starts, targets):
    require(raw.startswith(GOLD_MAGIC) and len(raw) == len(GOLD_MAGIC) + 656 * 32,
            'gold_magic_or_size')
    rows, offset = [], len(GOLD_MAGIC)
    for case, target in enumerate(targets):
        row = []
        for index, token in enumerate(target):
            rr, ii, tt, pos, nll, top1, logit = struct.unpack_from('<4IdIf', raw, offset)
            offset += 32
            require((rr, ii, tt, pos) == (case, index, token, starts[case] + index - 1),
                    'gold_alignment')
            require(math.isfinite(nll) and nll >= 0 and math.isfinite(logit)
                    and 0 <= top1 < 102400, 'gold_value')
            row.append(nll)
        rows.append(row)
    return rows


def decode_routes(raw):
    require(raw.startswith(ROUTE_MAGIC) and len(raw) == len(ROUTE_MAGIC) + 112684 * 40,
            'route_magic_or_size')
    previous_position = None
    for i, row in enumerate(struct.iter_unpack('<10i', raw[len(ROUTE_MAGIC):])):
        event, layer, slot, pos, *ids = row
        require(event == i and layer == 1 + i % 26 and slot == 0 and 0 <= pos < 384
                and len(set(ids)) == 6 and all(0 <= x < 64 for x in ids), 'route_coordinate')
        require(i % 26 == 0 or pos == previous_position, 'route_token_layer_position')
        previous_position = pos
    return 112684


def decode_jvp(raw, starts, targets, original):
    require(raw.startswith(JVP_MAGIC) and len(raw) == len(JVP_MAGIC) + 656 * 48,
            'jvp_magic_or_size')
    output = {'target_token_ids': targets, 'per_token_dnll': [],
              'per_token_dtargetlogit': [], 'per_token_dlogits_rms': []}
    offset = len(JVP_MAGIC)
    for case, target in enumerate(targets):
        gradients, logits, rms = [], [], []
        for index, token in enumerate(target):
            rr, ii, tt, pos, nll, dnll, dtarget, drms = struct.unpack_from('<4I4d', raw, offset)
            offset += 48
            require((rr, ii, tt, pos) == (case, index, token, starts[case] + index - 1),
                    'jvp_alignment')
            require(all(math.isfinite(v) for v in (nll, dnll, dtarget, drms))
                    and nll >= 0 and drms >= 0, 'jvp_value')
            require(struct.pack('<d', nll) == struct.pack('<d', original[case][index]),
                    'jvp_original_nll_bitidentity')
            gradients.append(dnll)
            logits.append(dtarget)
            rms.append(drms)
        output['per_token_dnll'].append(gradients)
        output['per_token_dtargetlogit'].append(logits)
        output['per_token_dlogits_rms'].append(rms)
    return output


def metrics(actual, reference):
    require(len(actual) == len(reference) == 656 and
            all(math.isfinite(v) for v in actual + reference), 'metric_length_or_finite')
    reference_rms = math.sqrt(mean([v * v for v in reference]))
    differences = [a - b for a, b in zip(actual, reference)]
    rms = math.sqrt(mean([v * v for v in differences]))
    norm = math.sqrt(math.fsum(v * v for v in actual) * math.fsum(v * v for v in reference))
    require(norm > 0, 'zero_direction_for_cosine')
    return {'tokens': 656, 'mean_jvp': mean(actual), 'mean_fixed_fine_fd': mean(reference),
            'rms_disagreement': rms, 'max_absolute_disagreement': max(map(abs, differences)),
            'rms_fixed_fine_fd': reference_rms, 'rms_acceptance_threshold': 1e-4 + .05 * reference_rms,
            'absolute_tolerance': 1e-4, 'relative_tolerance': .05,
            'passed': rms <= 1e-4 + .05 * reference_rms,
            'cosine_similarity': math.fsum(a * b for a, b in zip(actual, reference)) / norm,
            'jvp_observed_label_one_direction_square_mean': mean([v * v for v in actual]),
            'fixed_fine_observed_label_one_direction_square_mean': mean([v * v for v in reference])}


def verify_metrics(actual, claimed):
    for key, value in actual.items():
        if type(value) is bool:
            require(claimed.get(key) is value, 'metric_bool:' + key)
        else:
            close(value, claimed.get(key), key)


def constraints(record):
    require(record.get('scope') == 'TRAIN16_DIAGNOSTIC' and record.get('case_ids') == CASE_IDS,
            'seen_train16_scope')
    require(record.get('production_modified') is False and
            record.get('independent_validation_ood_accessed') is False and
            record.get('allocation_selected') is False and record.get('full48_empirical_fisher') is None,
            'no_promotion_or_full_fisher_claim')


def verify_stderr(folder, jvp=False, fixed=False):
    text = inspect(folder / 'stderr.txt', binary=True).decode('utf-8', errors='strict')
    require('FATAL:' not in text and
            'RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16' in text,
            'native_complete_or_fatal')
    require(('JE01_ROLE_DIRECTION_REGISTERED' in text) is (jvp or fixed), 'overlay_marker')
    require(('JE01_FIXED_BASELINE_ROUTES_ENABLED records=112684' in text) is (jvp or fixed),
            'fixed_route_marker')
    found = re.findall(r'^JE01_ONE_COLUMN_JVP_PREFIX_COMPLETE tokens=(\d+) cases=(\d+) layers=(\d+) gold=(\d+)\s*$',
                       text, flags=re.MULTILINE)
    require(found == ([tuple(str(v) for v in COMPLETION.values())] if jvp else []), 'full_prefix_counts')


def audit(args):
    root = Path(args.root)
    require(root == DEFAULT_ROOT and root.resolve() == root and not root.is_symlink(),
            'sealed_original_experiment_root')
    build = receipt(root / 'NATIVE_BUILD_RECEIPT.json', PIN['build'])
    patch = receipt(root / 'SOURCE_PATCH_RECEIPT.json', PIN['patch_receipt'])
    summary = receipt(root / 'JE01_ONE_COLUMN_JVP_COMPARISON_attempt1.json', PIN['comparison'])
    require(build['schema'] == 'beglin-je01-native-one-column-jvp-build/1' and
            build['status'] == 'ONE_COLUMN_JVP_NATIVE_COMPILE_PASS' and
            build['model_original_af4_sha256'] == PIN['model'] and
            build['adapter_smoke_receipt_sha256'] == PIN['smoke'], 'build_contract')
    require(patch['role_key'] == ROLE and patch['tier'] == 'Q4' and patch['alpha'] == 0 and
            patch['production_modified'] is False, 'patch_contract')
    for relative, key in [('bin/je01_native_one_column_jvp', 'binary'),
                          ('source/mlx_moe.cpp', 'cpp'), ('source/qwen_infer.c', 'c'),
                          ('source/CMakeLists.txt', 'cmake'), ('source/je01_mlx_native_jvp_adapters.h', 'adapter'),
                          ('tools/je01_native_one_column_jvp_runner.py', 'runner'),
                          ('tools/je01_native_one_column_jvp_patch.py', 'patch')]:
        inspect(root / relative, PIN[key])
    for name, pin in HELPERS.items():
        inspect(root / 'tools' / name, pin)
    require(MODEL.stat().st_size == 9814343680, 'original_model_byte_count')
    inspect(MODEL, PIN['model'])
    bank = receipt(BANK / 'JE01_ROLE_DIRECTION_BANK_RECEIPT.json', PIN['bank_receipt'])
    require(bank['physical_file_sha256'] == PIN['bank'] and bank['role_manifest_sha256'] == PIN['bank_tsv'],
            'bank_receipt_source')
    inspect(BANK / 'JE01_ROLE_DIRECTION_BANK.bin', PIN['bank'])
    inspect(BANK / 'JE01_ROLE_DIRECTION_BANK.tsv', PIN['bank_tsv'])
    dataset = receipt(ORDINARY / 'datasets/pilot16.json', PIN['dataset'])
    require(dataset['scope'] == 'TRAIN16_DIAGNOSTIC' and dataset['case_ids'] == CASE_IDS and
            dataset['independent_validation_ood_accessed'] is False, 'dataset_train_scope')
    meta = inspect(Path(dataset['gold_metadata']), PIN['meta'], binary=True)
    starts, targets = decode_meta(meta)
    require(targets == dataset['target_token_ids'], 'dataset_targets')
    manifest = inspect(Path(dataset['manifest']), PIN['manifest'], binary=True).decode().splitlines()
    require(len(manifest) == len(dataset['input_sha256']) == 16, 'manifest_count')
    for i, line in enumerate(manifest):
        name, count = line.rsplit(' ', 1)
        require(count == '1', 'manifest_repetition')
        raw = inspect(Path(name), dataset['input_sha256'][i], binary=True)
        require(len(raw) == 4 * (starts[i] + len(targets[i]) - 1), 'input_full_prefix_length')
        tokens = list(struct.unpack('<' + 'i' * (len(raw) // 4), raw))
        require(tokens[0] == 100000 and all(0 <= t < 102400 for t in tokens) and
                tokens[starts[i]:] == targets[i][:-1], 'teacher_forced_prefix')
    original_folder = ORDINARY / 'runs/pilot16/off_attempt1'
    original = receipt(original_folder / 'RUN_RECEIPT.json', PIN['ordinary_off'])
    gold = inspect(original_folder / 'gold_native.bin', PIN['gold'], binary=True)
    original_nll = decode_gold(gold, starts, targets)
    require(original_nll == original['per_token_nll'] and targets == original['target_token_ids'],
            'original_raw_gold_receipt')
    routes = inspect(original_folder / 'routes.bin', PIN['routes'], binary=True)
    decode_routes(routes)
    fixed_audit = receipt(args.fixed_audit, PIN['fixed_audit'], PIN['fixed_audit_file'])
    require(fixed_audit['status'] == 'PASS_INDEPENDENT_RAW_FIXED_BRANCH_REPLAY' and
            fixed_audit['diagnostic_status'] == 'FIXED_BRANCH_CONVERGED' and
            fixed_audit['dataset_receipt'] == dataset, 'fixed_reference_contract')
    inspect(FIXED / 'bin/je01_role_directional_fixed_native', PIN['fixed_binary'])
    cell_names = ['ordinary_off_attempt1', 'r00_Q4_a0_attempt1', 'r00_Q4_ap0.05_attempt1',
                  'r00_Q4_am0.05_attempt1', 'r00_Q4_ap0.025_attempt1', 'r00_Q4_am0.025_attempt1']
    require(set(fixed_audit['native_run_receipts']) == set(cell_names), 'fixed_six_cell_coverage')
    fixed_rows = []
    for i, (name, alpha) in enumerate(zip(cell_names, [0, 0, .05, -.05, .025, -.025])):
        folder = FIXED / 'runs/pilot16' / name
        record = receipt(folder / 'RUN_RECEIPT.json')
        require(record == fixed_audit['native_run_receipts'][name], 'fixed_raw_receipt_binding')
        require(record['status'] == 'SUCCEEDED_EXIT0' and record['native_exit_code'] == 0 and
                record['scope'] == 'TRAIN16_DIAGNOSTIC' and record['case_ids'] == CASE_IDS and
                record['target_token_ids'] == targets and record['alpha'] == alpha and
                record['full_prompt_and_KV_recomputed'] is True and
                record['router_scores_and_selected_weights_dynamic'] is True and
                record['fixed_route_replay_enabled'] is (i != 0) and
                record['role_key'] == (ROLE if i else None) and
                record['tier'] == ('Q4' if i else None) and
                record['production_modified'] is False and
                record['independent_validation_ood_accessed'] is False, 'fixed_run_contract')
        values = decode_gold(inspect(folder / 'gold_native.bin', record['output_sha256'], binary=True), starts, targets)
        require(values == record['per_token_nll'], 'fixed_raw_gold_values')
        require(inspect(folder / 'routes.bin', PIN['routes'], binary=True) == routes, 'fixed_route_byteidentity')
        verify_stderr(folder, fixed=i != 0)
        if i < 2:
            require(values == original_nll and record['output_sha256'] == PIN['gold'], 'fixed_zero_gold_identity')
        fixed_rows.append(flatten(values))
    coarse = [(p - m) / .1 for p, m in zip(fixed_rows[2], fixed_rows[3])]
    fine = [(p - m) / .05 for p, m in zip(fixed_rows[4], fixed_rows[5])]
    fd_convergence = metrics(coarse, fine)
    require(fd_convergence['passed'] and fine == fixed_audit['statistics']['per_token_fine_derivative'],
            'raw_fd_convergence')
    runs = []
    for jvp, name, key in [(False, 'off_attempt1', 'off'), (True, 'jvp_r00_Q4_a0_attempt1', 'jvp')]:
        folder = root / 'runs/pilot16' / name
        record = receipt(folder / 'RUN_RECEIPT.json', PIN[key])
        constraints(record)
        require(record['status'] == 'SUCCEEDED_EXIT0' and record['schema'] == 'beglin-je01-native-one-column-jvp-run/1'
                and record['execution']['native_exit_code'] == 0 and record['execution']['stop_reason'] is None
                and record['target_token_ids'] == targets and record['native_jvp_enabled'] is jvp
                and record['native_overlay_enabled'] is jvp and record['fixed_route_replay_enabled'] is jvp
                and record['full_prefix_tangent_carried'] is jvp and record['full_prompt_and_KV_recomputed'] is True
                and record['router_scores_and_selected_weights_dynamic'] is True
                and record['role_key'] == (ROLE if jvp else None)
                and record['tier'] == ('Q4' if jvp else None) and record['alpha'] == 0
                and record['runner_source_sha256'] == PIN['runner'], 'jvp_run_contract')
        identity = record['source_identity']
        expected = {'original_af4_sha256': PIN['model'], 'native_executable_sha256': PIN['binary'],
                    'new_build_receipt_sha256': PIN['build'], 'bank_sha256': PIN['bank'],
                    'gold_metadata_sha256': PIN['meta'], 'input_manifest_sha256': PIN['manifest'],
                    'adapter_header_sha256': PIN['adapter'], 'adapter_smoke_receipt_sha256': PIN['smoke'],
                    'baseline_route_sha256': PIN['routes'], 'fixed_reference_audit_sha256': PIN['fixed_audit'],
                    'original_off_receipt_sha256': PIN['ordinary_off'], 'role_order_sha256': PIN['role_order']}
        require(all(identity.get(k) == v for k, v in expected.items()), 'jvp_source_identity')
        require(inspect(folder / 'gold_native.bin', PIN['gold'], binary=True) == gold and
                inspect(folder / 'routes.bin', PIN['routes'], binary=True) == routes, 'jvp_gold_routes_byteidentity')
        require(record['per_token_nll'] == original_nll, 'jvp_primal_nll_claim')
        close(mean(flatten(original_nll)), record['mean_gold_nll'], 'baseline_mean')
        verify_stderr(folder, jvp=jvp)
        if jvp:
            require(record['native_prefix_completion'] == COMPLETION and record['new_off_receipt_sha256'] == PIN['off'],
                    'jvp_prefix_or_off_binding')
            raw = inspect(folder / 'jvp_gold.bin', record['gradient_output_sha256'], binary=True)
            require(len(raw) == record['gradient_output_bytes'], 'jvp_gradient_size_claim')
            gradient = decode_jvp(raw, starts, targets, original_nll)
            require(gradient == record['gradients'], 'jvp_gradient_raw_replay')
            actual = flatten(gradient['per_token_dnll'])
            recomputed = metrics(actual, fine)
            verify_metrics(recomputed, record['comparison_with_fixed_fine_fd'])
        else:
            require(not (folder / 'jvp_gold.bin').exists() and record['gradients'] is None,
                    'off_has_no_gradient')
        runs.append(record)
    constraints(summary)
    require(summary['off_receipt_sha256'] == runs[0]['receipt_sha256'] and
            summary['jvp_receipt_sha256'] == runs[1]['receipt_sha256'] and
            summary['source_identity'] == runs[1]['source_identity'] == runs[0]['source_identity'] and
            summary['role_key'] == ROLE and summary['tier'] == 'Q4' and summary['alpha'] == 0,
            'summary_bindings')
    verify_metrics(recomputed, summary['comparison'])
    require(recomputed['passed'], 'jvp_fd_disagreement')
    result = {
        'schema': 'beglin-je01-native-one-column-independent-raw-audit/1',
        'status': 'PASS_INDEPENDENT_RAW_NATIVE_JVP_FIXED_FD',
        'observed_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'PREVIOUSLY_SEEN_TRAIN16_FIXED_HARD_BRANCH_ONE_COMPUTATIONAL_DIRECTION',
        'root': str(root), 'role_key': ROLE, 'tier': 'Q4', 'alpha': 0,
        'case_ids': CASE_IDS, 'gold_tokens': 656, 'baseline_nll': mean(flatten(original_nll)),
        'original_af4_full_file_sha_verified': True,
        'off_and_jvp_original_gold_and_routes_byteidentical': True,
        'fixed_fd_six_cells_raw_replayed': True, 'native_prefix_completion': COMPLETION,
        'fd_convergence_recomputed': fd_convergence, 'jvp_vs_fixed_fd_recomputed': recomputed,
        'per_token_native_jvp_dnll': actual, 'per_token_fixed_fine_fd_dnll': fine,
        'single_direction_second_moment_interpretation':
            'Observed-label score square mean for one fixed-route computational direction; '
            'not the 48-direction or full parameter Fisher, expected Fisher, or Hessian.',
        'natural_finite_step_routing_stability': 'NOT_ESTABLISHED_BY_FIXED_ROUTE_REPLAY',
        'allocation_selected': False, 'full48_empirical_fisher': None,
        'independent_validation128_accessed': False, 'independent_ood128_accessed': False,
        'production_modified': False, 'native_inference_rerun': False,
        'external_pins': PIN, 'verified_files': FILES,
        'auditor_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    result['receipt_sha256'] = canonical(result)
    return result


def self_test():
    starts, targets = [2] * 16, [list(range(100, 141)) for _ in range(16)]
    meta = META_MAGIC + b''.join(struct.pack('<II384i', 2, 41, *(row + [-1] * 343)) for row in targets)
    require(decode_meta(meta) == (starts, targets), 'selftest_metadata')
    gold = GOLD_MAGIC + b''.join(struct.pack('<4IdIf', c, i, t, 1 + i, 2.5, 7, .25)
                                for c, row in enumerate(targets) for i, t in enumerate(row))
    original = decode_gold(gold, starts, targets)
    jvp = JVP_MAGIC + b''.join(struct.pack('<4I4d', c, i, t, 1 + i, 2.5, .125, .25, .5)
                              for c, row in enumerate(targets) for i, t in enumerate(row))
    decoded = decode_jvp(jvp, starts, targets, original)
    require(flatten(decoded['per_token_dnll']) == [.125] * 656, 'selftest_jvp')
    require(metrics([.125] * 656, [.125] * 656)['rms_disagreement'] == 0, 'selftest_metrics')
    negative = []
    def rejected(label, fn):
        try:
            fn()
        except (ValueError, struct.error):
            negative.append(label)
        else:
            raise AssertionError('accepted corrupted input:' + label)
    rejected('metadata_trailing', lambda: decode_meta(meta + b'X'))
    rejected('gold_truncation', lambda: decode_gold(gold[:-1], starts, targets))
    rejected('jvp_trailing', lambda: decode_jvp(jvp + b'X', starts, targets, original))
    changed = bytearray(jvp)
    struct.pack_into('<I', changed, len(JVP_MAGIC) + 8, 999)
    rejected('jvp_target_misalignment', lambda: decode_jvp(bytes(changed), starts, targets, original))
    changed = bytearray(jvp)
    struct.pack_into('<d', changed, len(JVP_MAGIC) + 16, 2.50000000000001)
    rejected('jvp_primal_bit_drift', lambda: decode_jvp(bytes(changed), starts, targets, original))
    changed = bytearray(jvp)
    struct.pack_into('<d', changed, len(JVP_MAGIC) + 24, float('nan'))
    rejected('nonfinite_gradient', lambda: decode_jvp(bytes(changed), starts, targets, original))
    rejected('canonical_digest_drift', lambda: verify_object({'x': 1, 'receipt_sha256': '0' * 64}))
    rejected('metric_claim_drift', lambda: verify_metrics(metrics([.125] * 656, [.125] * 656), {}))
    return {'status': 'PASS_PARSER_SELF_TEST_ONLY', 'positive_checks': 4,
            'corruption_controls_rejected': negative, 'real_model_evidence': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--fixed-audit', type=Path,
                        default=FIXED / 'JE01_FIXED_ROUTE_GATE_INDEPENDENT_AUDIT.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True, allow_nan=False))
        return
    require(args.fixed_audit is not None, 'fixed_audit_path_required')
    result = audit(args)
    if args.output is not None:
        path = args.output
        require(path.is_absolute() and path.parent.resolve() == path.parent and
                not path.exists() and not path.is_symlink() and
                not any(path.is_relative_to(parent) for parent in
                        (DEFAULT_ROOT, FIXED, ORDINARY, BANK, MODEL.parent)),
                'new_output_outside_source_experiments_required')
        with path.open('x') as stream:
            stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'status': result['status'], 'output': str(path),
                          'receipt_sha256': result['receipt_sha256'],
                          'comparison': result['jvp_vs_fixed_fd_recomputed']}, sort_keys=True))
    else:
        print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'FAIL_INDEPENDENT_JVP_AUDIT',
                          'error': type(error).__name__ + ': ' + str(error)}, sort_keys=True), file=sys.stderr)
        raise
