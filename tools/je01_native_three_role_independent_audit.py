#!/usr/bin/env python3
"""Raw, independent audit of the fixed JE01 three-direction diagnostic.

No experiment runner/helper or MLX is imported. The separately SHA-pinned pure
one-column auditor supplies only file, binary-format and canonical-JSON readers.
This audit never runs inference or selects a policy. A numerical rejection is
reported as a rejection even when every evidence-integrity check succeeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys

UTIL_SHA = '7da57610f20e1c837e31528e001b91dc614bad5786a0d3c47cb2515f67d628a9'
PRIOR_FILE_SHA = '71c304b6b69b8a07e4911c336352d142c12acf86c6522cd054cd2c48744d009e'
PRIOR_RECEIPT = 'b7f20261c382f065ef5f6dc70ec04ed41450fa7ea495507a7014c385778e3cac'
RUNNER_SHA = '9be8372fa3858577011eb5ab1d4ad654f746e151a95012aaf23e515fd8a7eb90'
PATCH_SHA = '68c8cd6541657312bca6604dcbe74dcdc8156ff4da80de9c8fd672a1ad08d755'
BUILD_PIN = '2cbeb4dc1a7e1ab5642a4abb22395498c10dde43bee051eec3c303c7e53fa008'
BINARY_SHA = '110b1978d97bcd7c6bdfd6d42142569e56e9dacffa4ff462dc0786b3128e49f4'
ROOT = Path('/Volumes/D50/TailnetCommander/work/je01_native_role_jvp_followup_20261123_v1')
SCOPE = 'PREVIOUSLY_SEEN_TRAIN16_DIAGNOSTIC_ONLY'
MODE = 'ROLE_DIRECTION_FIXED_Q4Q5'
ROLE_KEYS = [[12, 13, role] for role in ('gate_proj', 'up_proj', 'down_proj')]
ALPHAS = ('0', '+0.05', '-0.05', '+0.025', '-0.025')


def require(value, label):
    if not value:
        raise ValueError('JE01_THREE_ROLE_INDEPENDENT_AUDIT:' + label)


def load_utils(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file()
            and not path.is_symlink(), 'pure_auditor_path')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == UTIL_SHA, 'pure_auditor_sha')
    spec = importlib.util.spec_from_file_location('je01_pinned_pure_independent_audit_utils', path)
    require(spec is not None and spec.loader is not None, 'pure_auditor_loader')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(actual, expected, label):
    require(type(actual) in (float, int) and type(expected) in (float, int)
            and math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), 'number:' + label)


def same_values(actual, claimed, label):
    if isinstance(actual, dict):
        require(isinstance(claimed, dict) and set(claimed) == set(actual), 'keys:' + label)
        for key, value in actual.items():
            same_values(value, claimed[key], label + '.' + key)
    elif isinstance(actual, list):
        require(isinstance(claimed, list) and len(claimed) == len(actual), 'list:' + label)
        for i, value in enumerate(actual):
            same_values(value, claimed[i], label + '[' + str(i) + ']')
    elif type(actual) in (int, float):
        close(actual, claimed, label)
    else:
        require(type(claimed) is type(actual) and claimed == actual, 'value:' + label)


def mean(values):
    return math.fsum(values) / len(values)


def direction_metrics(actual, reference):
    require(len(actual) == len(reference) == 656
            and all(type(v) in (int, float) and math.isfinite(v) for v in actual + reference),
            'direction_cardinality_or_nonfinite')
    delta = [a - b for a, b in zip(actual, reference)]
    ref_square, actual_square = mean([v * v for v in reference]), mean([v * v for v in actual])
    ref_rms, error_rms = math.sqrt(ref_square), math.sqrt(mean([v * v for v in delta]))
    norm = math.sqrt(math.fsum(v * v for v in actual) * math.fsum(v * v for v in reference))
    return {'tokens': 656, 'mean_jvp': mean(actual), 'mean_fixed_fine_fd': mean(reference),
            'rms_disagreement': error_rms, 'max_absolute_disagreement': max(map(abs, delta)),
            'rms_fixed_fine_fd': ref_rms, 'rms_acceptance_threshold': 1e-4 + .05 * ref_rms,
            'absolute_tolerance': 1e-4, 'relative_tolerance': .05,
            'passed': error_rms <= 1e-4 + .05 * ref_rms,
            'cosine_similarity': math.fsum(a * b for a, b in zip(actual, reference)) / norm if norm else None,
            'jvp_observed_label_one_direction_square_mean': actual_square,
            'fixed_fine_observed_label_one_direction_square_mean': ref_square,
            'interpretation': 'ONE_FIXED_HARD_BRANCH_COMPUTATIONAL_DIRECTION_ON_SEEN_TRAIN16',
            'full48_empirical_fisher': None, 'allocation_selected': False}


def convergence_metrics(coarse, fine):
    values = direction_metrics(coarse, fine)
    return {'coarse_h': .05, 'fine_h': .025, 'absolute_tolerance': 1e-4, 'relative_tolerance': .05,
            'rms_fine_derivative': values['rms_fixed_fine_fd'],
            'rms_disagreement': values['rms_disagreement'],
            'rms_acceptance_threshold': values['rms_acceptance_threshold'],
            'max_absolute_disagreement': values['max_absolute_disagreement'],
            'cosine_similarity': values['cosine_similarity'], 'passed': values['passed']}


def derivative(plus, minus, denominator):
    require(len(plus) == len(minus) == 16 and
            all(len(p) == len(m) > 0 for p, m in zip(plus, minus)), 'fd_row_alignment')
    result = [[(p - m) / denominator for p, m in zip(pr, mr)] for pr, mr in zip(plus, minus)]
    require(sum(map(len, result)) == 656, 'fd_token_count')
    return result


def execution_sequence():
    result = [{'kind': 'OFF', 'role_index': None, 'alpha': '0'},
              {'kind': 'JVP', 'role_index': 0, 'alpha': '0'}]
    for role in (1, 2):
        result.extend({'kind': 'FD', 'role_index': role, 'alpha': alpha} for alpha in ALPHAS)
        result.append({'kind': 'JVP', 'role_index': role, 'alpha': '0'})
    return result


def folder_name(kind, role, alpha):
    name = 'off' if kind == 'OFF' else f'{kind.lower()}_r{role:02d}_Q4_a{alpha.replace("+", "p").replace("-", "m")}'
    return name + '_attempt1'


def scope_flags(record):
    require(record.get('scope') == SCOPE and record.get('full48_empirical_fisher') is None
            and record.get('allocation_selected') is False
            and record.get('independent_validation_ood_accessed') is False
            and record.get('production_modified') is False, 'diagnostic_scope_flags')


def expected_identity(u, build_hash, binary_hash):
    p = u.PIN
    return {'original_af4_sha256': p['model'], 'native_executable_sha256': binary_hash,
            'bank_sha256': p['bank'], 'input_manifest_sha256': p['manifest'],
            'gold_metadata_sha256': p['meta'], 'role_order_sha256': p['role_order'],
            'fixed_fd_executable_sha256': p['fixed_binary'], 'new_build_receipt_sha256': build_hash,
            'adapter_header_sha256': p['adapter'], 'adapter_smoke_receipt_sha256': p['smoke'],
            'baseline_route_sha256': p['routes'], 'original_off_receipt_sha256': p['ordinary_off'],
            'gate_fixed_reference_audit_sha256': p['fixed_audit'],
            'derivative_definition': 'THREE_NATIVE_ROLE_Q4_ALPHA_ZERO_FIXED_HARD_ROUTE_FULL_PREFIX_DIRECTIONS'}


def validate_plan(u, plan, identity, build_hash):
    scope_flags(plan)
    required = {
        'schema': 'beglin-je01-native-three-role-jvp-prerun-plan/1',
        'source_identity': identity, 'new_build_receipt_sha256': build_hash,
        'runner_source_sha256': RUNNER_SHA, 'patch_source_sha256': PATCH_SHA,
        'diagnostic_role_indices': [0, 1, 2], 'diagnostic_role_keys': ROLE_KEYS,
        'tier': 'Q4', 'alpha_grid': list(ALPHAS), 'jvp_alpha': 0,
        'native_execution_sequence': execution_sequence(), 'native_cell_count': 14,
        'case_ids': list(range(16)), 'gold_tokens': 656,
        'full_prefix_completion_counters': u.COMPLETION,
        'gate_fd_reference': {'canonical_sha256': u.PIN['fixed_audit'],
                              'file_sha256': u.PIN['fixed_audit_file'],
                              'status': 'PREVIOUSLY_CONVERGED_INDEPENDENT_FIXED_ROUTE_AUDIT'},
        'fd_convergence_rule': {'coarse_h': .05, 'fine_h': .025, 'rms_absolute_tolerance': 1e-4,
                                'rms_relative_tolerance': .05,
                                'reference': 'RMS_OF_FINE_FD_OVER_ALL_656_GOLD_TOKENS'},
        'jvp_agreement_rule': {'rms_absolute_tolerance': 1e-4, 'rms_relative_tolerance': .05,
                               'reference': 'RMS_OF_FINE_FD_OVER_ALL_656_GOLD_TOKENS'},
        'pass_requires_fd_convergence_and_jvp_agreement': True,
        'every_zero_and_jvp_primal_requires_original_gold_sha': u.PIN['gold'],
        'every_fixed_route_output_requires_baseline_sha': u.PIN['routes'],
        'all_predeclared_roles_reported_regardless_of_numerical_sign_or_pass': True,
        'result_adaptive_thresholds': False, 'result_adaptive_roles': False,
        'result_adaptive_attempts': False,
        'retry_policy': 'ONLY_EXECUTION_OR_INTEGRITY_FAILURE_WITH_NEW_ATTEMPT',
        'observed_label_gram_rule': 'ONLY_AFTER_ALL_THREE_DIRECTIONS_PASS; EXPLORATORY_NOT_ALLOCATION',
        'train240_allocation': 'NOT_RUN',
    }
    for key, value in required.items():
        require(key in plan, 'plan_missing:' + key)
        same_values(value, plan[key], 'plan.' + key)
    require(set(plan) == set(required) | {'scope', 'full48_empirical_fisher', 'allocation_selected',
                                       'independent_validation_ood_accessed', 'production_modified', 'receipt_sha256'},
            'plan_unknown_fields')


def audit(args, u):
    root = Path(args.root)
    require(root == ROOT and root.resolve() == root and not root.is_symlink(), 'exact_new_experiment_root')
    require(args.expected_build_receipt == BUILD_PIN, 'external_exact_build_pin_required')
    prior = u.receipt(args.prior_audit, PRIOR_RECEIPT, PRIOR_FILE_SHA)
    require(prior['status'] == 'PASS_INDEPENDENT_RAW_NATIVE_JVP_FIXED_FD' and
            prior['auditor_source_sha256'] == UTIL_SHA and prior['external_pins'] == u.PIN and
            prior['full48_empirical_fisher'] is None and prior['allocation_selected'] is False,
            'prior_independent_audit_scope')
    build = u.receipt(root / 'NATIVE_BUILD_RECEIPT.json', args.expected_build_receipt)
    scope_flags(build)
    require(build['schema'] == 'beglin-je01-native-role-jvp-build/1' and
            build['status'] == 'ROLE_JVP_NATIVE_COMPILE_PASS' and build['runner_source_sha256'] == RUNNER_SHA and
            build['binary_sha256'] == BINARY_SHA and
            build['model_original_af4_sha256'] == u.PIN['model'] and
            build['adapter_smoke_receipt_sha256'] == u.PIN['smoke'], 'build_contract')
    config = build['configuration']
    require(config['patch_source_sha256'] == PATCH_SHA and config['adapter_sha256'] == u.PIN['adapter'] and
            config['smoke_canonical_sha256'] == u.PIN['smoke'] and config['runtime_mode'] == MODE and
            config['diagnostic_role_indices'] == [0, 1, 2] and config['diagnostic_tier'] == 'Q4' and
            config['diagnostic_scope'] == SCOPE, 'configuration_contract')
    patch = u.receipt(root / 'SOURCE_PATCH_RECEIPT.json', build['source_patch_receipt_sha256'])
    require(patch['schema'] == 'beglin-je01-native-role-jvp-source-patch/1' and patch['runtime_mode'] == MODE and
            patch['adapter_sha256'] == u.PIN['adapter'] and patch['production_modified'] is False and
            patch['patched_cpp_sha256'] == config['patched_cpp_sha256'] and
            patch['patched_c_sha256'] == config['patched_c_sha256'], 'patch_contract')
    source_files = {
        'bin/je01_native_role_jvp': build['binary_sha256'],
        'source/mlx_moe.cpp': config['patched_cpp_sha256'],
        'source/qwen_infer.c': config['patched_c_sha256'],
        'source/CMakeLists.txt': build['cmake_sha256'],
        'source/je01_mlx_native_jvp_adapters.h': u.PIN['adapter'],
        'tools/je01_native_role_jvp_runner.py': RUNNER_SHA,
        'tools/je01_native_role_jvp_patch.py': PATCH_SHA,
    }
    helper_pins = {**u.HELPERS, 'je01_native_one_column_jvp_runner.py': u.PIN['runner']}
    require(build['helper_source_sha256'] == helper_pins, 'helper_pin_map')
    source_files.update({'tools/' + key: value for key, value in helper_pins.items()})
    for name, pin in source_files.items():
        u.inspect(root / name, pin)
    require((root / 'bin/je01_native_role_jvp').stat().st_size == build['binary_bytes'], 'binary_size')
    u.inspect(u.FIXED / 'bin/je01_role_directional_fixed_native', u.PIN['fixed_binary'])
    u.inspect(u.BANK / 'JE01_ROLE_DIRECTION_BANK.bin', u.PIN['bank'])
    u.inspect(u.BANK / 'JE01_ROLE_DIRECTION_BANK.tsv', u.PIN['bank_tsv'])
    u.receipt(u.BANK / 'JE01_ROLE_DIRECTION_BANK_RECEIPT.json', u.PIN['bank_receipt'])
    dataset = u.receipt(u.ORDINARY / 'datasets/pilot16.json', u.PIN['dataset'])
    require(dataset['scope'] == 'TRAIN16_DIAGNOSTIC' and dataset['case_ids'] == list(range(16)) and
            dataset['independent_validation_ood_accessed'] is False, 'dataset_scope')
    starts, targets = u.decode_meta(u.inspect(Path(dataset['gold_metadata']), u.PIN['meta'], binary=True))
    require(targets == dataset['target_token_ids'], 'dataset_targets')
    manifest = u.inspect(Path(dataset['manifest']), u.PIN['manifest'], binary=True).decode().splitlines()
    require(len(manifest) == len(dataset['input_sha256']) == 16, 'input_manifest_count')
    for case, line in enumerate(manifest):
        name, count = line.rsplit(' ', 1)
        require(count == '1', 'input_repetitions')
        raw = u.inspect(Path(name), dataset['input_sha256'][case], binary=True)
        require(len(raw) == 4 * (starts[case] + len(targets[case]) - 1), 'full_prefix_bytes')
        ids = list(struct.unpack('<' + 'i' * (len(raw) // 4), raw))
        require(ids[0] == 100000 and all(0 <= token < 102400 for token in ids) and
                ids[starts[case]:] == targets[case][:-1], 'teacher_forced_input')
    original_folder = u.ORDINARY / 'runs/pilot16/off_attempt1'
    original = u.receipt(original_folder / 'RUN_RECEIPT.json', u.PIN['ordinary_off'])
    raw_gold = u.inspect(original_folder / 'gold_native.bin', u.PIN['gold'], binary=True)
    baseline = u.decode_gold(raw_gold, starts, targets)
    require(baseline == original['per_token_nll'], 'baseline_raw_nll')
    raw_routes = u.inspect(original_folder / 'routes.bin', u.PIN['routes'], binary=True)
    u.decode_routes(raw_routes)
    identity = expected_identity(u, build['receipt_sha256'], build['binary_sha256'])
    plan = u.receipt(root / 'PLAN_RECEIPT.json', args.expected_plan_receipt)
    validate_plan(u, plan, identity, build['receipt_sha256'])
    bound_identity = {**identity, 'plan_receipt_sha256': plan['receipt_sha256']}
    summary = u.receipt(root / 'JE01_THREE_ROLE_JVP_DIAGNOSTIC_attempt1.json')
    scope_flags(summary)
    require(summary['schema'] == 'beglin-je01-native-three-role-jvp-fixed-fd-diagnostic/1' and
            summary['source_identity'] == bound_identity and summary['plan_receipt_sha256'] == plan['receipt_sha256'] and
            summary['case_ids'] == list(range(16)) and summary['gold_tokens'] == 656 and
            summary['runner_source_sha256'] == RUNNER_SHA and summary['train240_allocation'] == 'NOT_RUN',
            'summary_contract')
    expected_names = {folder_name(cell['kind'], cell['role_index'], cell['alpha']) for cell in execution_sequence()}
    run_root = root / 'runs/pilot16'
    require({p.name for p in run_root.iterdir()} == expected_names, 'complete_exact14_cells_no_extra_attempts')
    runs = {}
    for cell in execution_sequence():
        kind, role, alpha = cell['kind'], cell['role_index'], cell['alpha']
        folder = run_root / folder_name(kind, role, alpha)
        require(folder.resolve() == folder and not folder.is_symlink() and
                not (folder / 'FAILED_RUN_RECEIPT.json').exists(), 'successful_unique_cell')
        record = u.receipt(folder / 'RUN_RECEIPT.json')
        scope_flags(record)
        expected = {'schema': 'beglin-je01-native-role-jvp-fd-run/1', 'status': 'SUCCEEDED_EXIT0',
                    'kind': kind, 'source_identity': bound_identity, 'plan_receipt_sha256': plan['receipt_sha256'],
                    'role_index': role, 'role_key': ROLE_KEYS[role] if role is not None else None,
                    'tier': 'Q4' if role is not None else None, 'alpha': float(alpha), 'alpha_literal': alpha,
                    'attempt': 1, 'case_ids': list(range(16)), 'target_token_ids': targets,
                    'runner_source_sha256': RUNNER_SHA, 'original_off_receipt_sha256': u.PIN['ordinary_off'],
                    'executed_binary_sha256': u.PIN['fixed_binary'] if kind == 'FD' else build['binary_sha256'],
                    'native_jvp_enabled': kind == 'JVP', 'native_overlay_enabled': kind != 'OFF',
                    'fixed_route_replay_enabled': kind != 'OFF', 'full_prompt_and_KV_recomputed': True,
                    'router_scores_and_selected_weights_dynamic': True, 'full_prefix_tangent_carried': kind == 'JVP',
                    'route_equality_interpretation': 'REPLAY_CONFORMANCE_NOT_NATURAL_ROUTING_STABILITY'}
        for key, value in expected.items():
            same_values(value, record.get(key), 'cell.' + key)
        execution = record['execution']
        require(execution['native_exit_code'] == 0 and execution['stop_reason'] is None and
                execution['native_started'] is True, 'native_exit_or_start')
        gold_bytes = u.inspect(folder / 'gold_native.bin', record['output_sha256'], binary=True)
        nll = u.decode_gold(gold_bytes, starts, targets)
        require(nll == record['per_token_nll'], 'cell_raw_gold_replay')
        close(mean(u.flatten(nll)), record['mean_gold_nll'], 'cell_mean_nll')
        require(u.inspect(folder / 'routes.bin', u.PIN['routes'], binary=True) == raw_routes and
                record['route_output_sha256'] == u.PIN['routes'] and
                record['route_output_bytes'] == len(raw_routes) and record['routed_token_layer_events'] == 112684,
                'cell_raw_routes')
        if alpha == '0':
            require(gold_bytes == raw_gold and nll == baseline, 'cell_exact_alpha_zero_primal')
        u.inspect(folder / 'stdout.txt', record['stdout_sha256'])
        u.inspect(folder / 'stderr.txt', record['stderr_sha256'])
        u.verify_stderr(folder, jvp=kind == 'JVP', fixed=kind == 'FD')
        native_log = (folder / 'stderr.txt').read_text(errors='strict')
        configurations = re.findall(r'^JE01_ROLE_DIRECTION_CONFIG role=(\d+) layer=(\d+) expert=(\d+) tier=(Q4|Q5) alpha=(\S+)\s*$',
                                    native_log, flags=re.MULTILINE)
        if kind == 'OFF':
            require(not configurations, 'off_has_no_native_role_configuration')
        else:
            require(configurations and set(configurations) == {(str(role), '12', '13', 'Q4', alpha)},
                    'native_executed_role_tier_alpha_configuration')
        if kind == 'JVP':
            grad_bytes = u.inspect(folder / 'jvp_gold.bin', record['gradient_output_sha256'], binary=True)
            gradient = u.decode_jvp(grad_bytes, starts, targets, baseline)
            require(gradient == record['gradients'] and len(grad_bytes) == record['gradient_output_bytes'] and
                    record['native_prefix_completion'] == u.COMPLETION, 'cell_raw_gradient')
        else:
            require(not (folder / 'jvp_gold.bin').exists() and not (folder / 'jvp_gold.bin').is_symlink() and
                    record['gradients'] is None and record['native_prefix_completion'] is None and
                    record['gradient_output_sha256'] is None and record['gradient_output_bytes'] is None,
                    'non_jvp_has_no_gradient')
        runs[(kind, role, alpha)] = {**record, '_raw_nll': nll}
        if kind != 'OFF':
            require(record['new_off_receipt_sha256'] == runs[('OFF', None, '0')]['receipt_sha256'], 'off_receipt_binding')
    require(summary['off_receipt_sha256'] == runs[('OFF', None, '0')]['receipt_sha256'], 'summary_off_binding')
    fixed_audit = u.receipt(args.fixed_audit, u.PIN['fixed_audit'], u.PIN['fixed_audit_file'])
    require(fixed_audit['dataset_receipt'] == dataset and
            fixed_audit['status'] == 'PASS_INDEPENDENT_RAW_FIXED_BRANCH_REPLAY', 'gate_reference_binding')
    gate_cells = {}
    for alpha in ALPHAS[1:]:
        name = 'r00_Q4_a' + alpha.replace('+', 'p').replace('-', 'm') + '_attempt1'
        folder = u.FIXED / 'runs/pilot16' / name
        record = u.receipt(folder / 'RUN_RECEIPT.json')
        require(record == fixed_audit['native_run_receipts'][name], 'gate_reference_receipt')
        raw = u.inspect(folder / 'gold_native.bin', record['output_sha256'], binary=True)
        values = u.decode_gold(raw, starts, targets)
        require(values == record['per_token_nll'] and
                u.inspect(folder / 'routes.bin', u.PIN['routes'], binary=True) == raw_routes,
                'gate_reference_raw_replay')
        gate_cells[alpha] = values
    raw_derivatives, results, columns = {}, [], []
    require(len(summary['roles']) == 3, 'three_role_report_count')
    for role in range(3):
        cells = gate_cells if role == 0 else {alpha: runs[('FD', role, alpha)]['_raw_nll'] for alpha in ALPHAS}
        coarse = derivative(cells['+0.05'], cells['-0.05'], .1)
        fine = derivative(cells['+0.025'], cells['-0.025'], .05)
        convergence = convergence_metrics(u.flatten(coarse), u.flatten(fine))
        if role == 0:
            require(u.flatten(fine) == prior['per_token_fixed_fine_fd_dnll'] ==
                    fixed_audit['statistics']['per_token_fine_derivative'], 'gate_prior_raw_fine_binding')
            same_values(convergence, fixed_audit['statistics']['convergence'], 'gate_convergence')
            reference_pin = u.PIN['fixed_audit']
        else:
            reference = u.receipt(root / f'FD_REFERENCE_r{role:02d}_Q4_attempt1.json')
            scope_flags(reference)
            require(reference['schema'] == 'beglin-je01-native-role-fixed-fd-reference/1' and
                    reference['kind'] == 'NEW_PINNED_NATIVE_FIXED_ROUTE_FD_REFERENCE' and
                    reference['role_index'] == role and reference['role_key'] == ROLE_KEYS[role] and
                    reference['tier'] == 'Q4' and reference['source_identity'] == bound_identity and
                    reference['case_ids'] == list(range(16)) and reference['target_token_ids'] == targets and
                    reference['run_receipt_sha256'] ==
                    {alpha: runs[('FD', role, alpha)]['receipt_sha256'] for alpha in ALPHAS}, 'new_fd_reference_binding')
            same_values(convergence, reference['convergence'], 'fd_convergence')
            require(reference['per_token_coarse_derivative'] == coarse and
                    reference['per_token_fine_derivative'] == fine and reference['status'] ==
                    ('FIXED_BRANCH_CONVERGED' if convergence['passed'] else 'FIXED_BRANCH_NUMERICAL_REJECTED'),
                    'fd_raw_derivatives_or_status')
            reference_pin = reference['receipt_sha256']
        measured = runs[('JVP', role, '0')]
        column = u.flatten(measured['gradients']['per_token_dnll'])
        comparison = direction_metrics(column, u.flatten(fine))
        passed = convergence['passed'] and comparison['passed']
        result = {'role_index': role, 'role_key': ROLE_KEYS[role], 'tier': 'Q4',
                  'status': 'FIXED_FD_AND_JVP_AGREE' if passed else 'NUMERICAL_REJECTED',
                  'passed': passed, 'fd_convergence': convergence, 'jvp_vs_fine_fd': comparison,
                  'fd_reference_receipt_sha256': reference_pin,
                  'jvp_receipt_sha256': measured['receipt_sha256'],
                  'jvp_output_sha256': measured['gradient_output_sha256']}
        same_values(result, summary['roles'][role], 'role_report')
        columns.append(column)
        results.append(result)
        raw_derivatives[str(role)] = {'native_jvp': column, 'fixed_coarse_fd': u.flatten(coarse),
                                      'fixed_fine_fd': u.flatten(fine)}
    all_pass = all(row['passed'] for row in results)
    gram = None
    if all_pass:
        gram = {'definition': 'MEAN_OBSERVED_LABEL_DIRECTIONAL_GRADIENT_OUTER_PRODUCT_ON_SEEN_TRAIN16',
                'coordinate_order': [key + ['Q4'] for key in ROLE_KEYS],
                'matrix': [[math.fsum(a * b for a, b in zip(left, right)) / 656 for right in columns]
                           for left in columns],
                'signed_mean_gradient': [mean(column) for column in columns], 'tokens': 656,
                'interpretation': 'EXPLORATORY_FIXED_BRANCH_THREE_DIRECTION_GRAM_NOT_FULL_PARAMETER_FISHER',
                'used_for_policy_selection': False}
    same_values(gram, summary['observed_label_three_direction_gram'], 'gram')
    experiment_status = 'THREE_ROLE_NATIVE_JVP_FIXED_FD_PASS' if all_pass else 'THREE_ROLE_NUMERICAL_REJECTED'
    require(summary['status'] == experiment_status and summary['post_run_original_af4_sha256'] == u.PIN['model'],
            'summary_status_or_model_pin')
    require(u.MODEL.stat().st_size == 9814343680, 'model_byte_count')
    u.inspect(u.MODEL, u.PIN['model'])
    result = {
        'schema': 'beglin-je01-native-three-role-independent-raw-audit/1',
        'status': 'INDEPENDENT_AUDIT_CONFIRMED_THREE_ROLE_PASS' if all_pass else
                  'INDEPENDENT_AUDIT_CONFIRMED_NUMERICAL_REJECTION',
        'evidence_integrity': 'PASS', 'experiment_numerical_verdict': 'PASS' if all_pass else 'FAIL',
        'experiment_status': experiment_status, 'all_three_directions_pass': all_pass,
        'observed_at': datetime.now(timezone.utc).isoformat(), 'root': str(root), 'scope': SCOPE,
        'case_ids': list(range(16)), 'gold_tokens': 656, 'native_cells_raw_replayed': 14,
        'baseline_nll': mean(u.flatten(baseline)), 'roles': results,
        'observed_label_three_direction_gram': gram, 'raw_directional_derivatives': raw_derivatives,
        'plan_receipt_sha256': plan['receipt_sha256'], 'external_plan_pin': args.expected_plan_receipt,
        'build_receipt_sha256': build['receipt_sha256'], 'source_report_receipt_sha256': summary['receipt_sha256'],
        'prior_independent_audit_receipt_sha256': PRIOR_RECEIPT,
        'prior_independent_audit_file_sha256': PRIOR_FILE_SHA,
        'original_af4_full_file_sha_verified_after_run': u.PIN['model'],
        'native_inference_rerun': False, 'natural_routing_stability': 'NOT_ESTABLISHED_BY_FORCED_ROUTE_REPLAY',
        'full48_empirical_fisher': None, 'Hessian': 'NOT_MEASURED', 'train240_allocation': 'NOT_RUN',
        'allocation_selected': False, 'independent_validation_ood_accessed': False, 'production_modified': False,
        'auditor_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pure_auditor_utilities_sha256': UTIL_SHA, 'verified_files': u.FILES,
    }
    result['receipt_sha256'] = u.canonical(result)
    return result


def self_test(u):
    prior = u.self_test()
    equal = direction_metrics([.25] * 656, [.25] * 656)
    require(equal['passed'] and equal['rms_disagreement'] == 0, 'selftest_metric_equality')
    require(not direction_metrics([.5] * 656, [.25] * 656)['passed'], 'selftest_numerical_rejection')
    zero = direction_metrics([0.] * 656, [0.] * 656)
    require(zero['passed'] and zero['cosine_similarity'] is None, 'selftest_zero_direction')
    require(len(execution_sequence()) == 14 and len({folder_name(c['kind'], c['role_index'], c['alpha'])
                                                 for c in execution_sequence()}) == 14, 'selftest14cells')
    rejected = []
    for label, actual, altered in [('false_to_true', False, True), ('missing_metric', equal, {}),
                                   ('gram_on_rejection', None, [[0.] * 3] * 3)]:
        try:
            same_values(actual, altered, label)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError('selftest accepted ' + label)
    return {'status': 'PASS_THREE_ROLE_AUDITOR_SELF_TEST_ONLY', 'pure_parser_tests': prior,
            'additional_positive_checks': 4, 'additional_corruptions_rejected': rejected,
            'actual_model_evidence': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--expected-build-receipt')
    parser.add_argument('--expected-plan-receipt')
    parser.add_argument('--prior-audit', type=Path)
    parser.add_argument('--fixed-audit', type=Path,
                        default=Path('/Volumes/D50/TailnetCommander/work/je01_role_directional_fixed_20261123_v1/JE01_FIXED_ROUTE_GATE_INDEPENDENT_AUDIT.json'))
    parser.add_argument('--audit-utils', type=Path,
                        default=Path(__file__).resolve().with_name('je01_native_one_column_independent_audit.py'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    u = load_utils(args.audit_utils)
    if args.self_test:
        print(json.dumps(self_test(u), sort_keys=True, allow_nan=False))
        return
    require(args.prior_audit is not None, 'prior_audit_path_required')
    result = audit(args, u)
    if args.output is None:
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    else:
        path = args.output
        require(path.is_absolute() and path.parent.resolve() == path.parent and not path.exists()
                and not path.is_symlink(), 'new_audit_output_only')
        with path.open('x') as stream:
            stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'status': result['status'], 'experiment_numerical_verdict': result['experiment_numerical_verdict'],
                          'receipt_sha256': result['receipt_sha256'], 'output': str(path), 'roles': result['roles']},
                         sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'FAIL_THREE_ROLE_EVIDENCE_INTEGRITY',
                          'error': type(error).__name__ + ': ' + str(error)}, sort_keys=True), file=sys.stderr)
        raise
