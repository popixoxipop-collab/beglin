#!/usr/bin/env python3
"""Pinned TRAIN16 Gate/Up/Down Q4 native-JVP and fixed-FD diagnostic.

Build a fresh isolated native executable from the signed fixed-route parent.
Measure only the three L12/E13 Q4 directions on previously observed TRAIN16.
Gate's converged independent FD audit is reused; Up/Down get new alpha-zero
and +/-0.05,+/-0.025 fixed-route runs. Every JVP uses exact AF4 primal gold
bytes and carries hidden/K/V tangents through the entire original prefix.
No policy is selected, no holdout opened, and no full48 Fisher is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys


TOOL_DIR = Path(__file__).resolve().parent
SOURCE_PINS = {
    'je01_native_one_column_jvp_runner.py': 'ffd632d79d948119ad261e2720ebbcd7c5076af21070aeaf60bf0ca8e7d79bd6',
    'je01_role_directional_native_runner.py': '5607c0afef3bd4b1a61bcab6622267acc6b6137fc65ba0e79873e92e31d7e9e6',
    'je01_role_directional_fixed_native.py': 'cffe2e1dd8298681f3a134caace742ddc9fea77209c58683cd8d2f48fe6284e7',
    'je01_role_directional_fixed_routes_patch.py': '233937b59293bec11300b66450b819dc48bb09b8d28eb6e98ca81aac7f511019',
    'je01_role_directional_bank.py': '045fd22d86e4ce18f967d66830a1f71b0c61ad6ab5c5889d953cb9a8d05bce33',
}
PATCH_SHA = '68c8cd6541657312bca6604dcbe74dcdc8156ff4da80de9c8fd672a1ad08d755'
ADAPTER_SHA = '9ae9d98b921334bae321e583908e71bdc9c08178cbf3ca5d7a1d4d4e552a09ec'
SMOKE_SHA = 'b0825ab7a048f7d95b155592760e10d73e3df2b112230adbf7b348f388ca1595'
WORK = Path('/Volumes/D50/TailnetCommander/work')
PREFIX = 'je01_native_role_jvp_'
MODE = 'ROLE_DIRECTION_FIXED_Q4Q5'
BUILD_SCHEMA = 'beglin-je01-native-role-jvp-build/1'
PATCH_SCHEMA = 'beglin-je01-native-role-jvp-source-patch/1'
RUN_SCHEMA = 'beglin-je01-native-role-jvp-fd-run/1'
REPORT_SCHEMA = 'beglin-je01-native-three-role-jvp-fixed-fd-diagnostic/1'
FD_SCHEMA = 'beglin-je01-native-role-fixed-fd-reference/1'
PLAN_SCHEMA = 'beglin-je01-native-three-role-jvp-prerun-plan/1'
DIAGNOSTIC_ROLES = (0, 1, 2)
TIER = 'Q4'
FD_ALPHAS = ('+0.05', '-0.05', '+0.025', '-0.025')
SCOPE = 'PREVIOUSLY_SEEN_TRAIN16_DIAGNOSTIC_ONLY'


def need(value, message):
    if not value:
        raise ValueError('JE01_ROLE_JVP_' + message)


def verify_import_sources():
    for name, expected in SOURCE_PINS.items():
        source = TOOL_DIR / name
        need(source.is_file() and not source.is_symlink()
             and hashlib.sha256(source.read_bytes()).hexdigest() == expected,
             'IMMUTABLE_HELPER_PIN:' + name)


# Check bytes before importing code. Imports remain at their original paths;
# no helper globals, restrictions, or receipt pins are changed.
verify_import_sources()
_spec = importlib.util.spec_from_file_location('je01_role_jvp_pinned_legacy',
                                               TOOL_DIR / 'je01_native_one_column_jvp_runner.py')
need(_spec is not None and _spec.loader is not None, 'LEGACY_LOADER')
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
legacy.verify_helpers()
common, fixed = legacy.common, legacy.fixed
sha, sealed, is_sha = legacy.sha, legacy.sealed, legacy.is_sha
flatten, mean = legacy.flatten, legacy.mean
ADAPTER_NAME = legacy.ADAPTER_NAME


def require_root(value):
    path = Path(value)
    need(path.is_absolute() and path.parent == WORK and path.name.startswith(PREFIX)
         and path.name != PREFIX and path.resolve() == path and not path.is_symlink(),
         'NEW_SEPARATE_D50_ROOT_REQUIRED')
    return path


def output_path(root, path):
    root, path = require_root(root), Path(path)
    need(path.is_absolute() and path != root and path.is_relative_to(root)
         and path.resolve() == path and not path.is_symlink(), 'OUTPUT_ESCAPE_OR_LINK')
    return path


def validate_cell(kind, role, alpha, attempt):
    need(kind in ('OFF', 'FD', 'JVP') and type(attempt) is int and 1 <= attempt <= 100,
         'CELL_MODE_OR_ATTEMPT')
    need(type(alpha) is str and alpha in ('0',) + FD_ALPHAS, 'ALPHA_GRID')
    if kind == 'OFF':
        need(role is None and alpha == '0', 'OFF_MUST_HAVE_NO_ROLE_AND_ZERO_ALPHA')
    else:
        need(type(role) is int and role in DIAGNOSTIC_ROLES, 'THREE_PREDECLARED_Q4_ROLES_ONLY')
        need(kind != 'JVP' or alpha == '0', 'JVP_BASELINE_ALPHA_ZERO_ONLY')


def checked_configuration(args):
    root = require_root(args.root)
    need(Path(args.patch).resolve() == TOOL_DIR / 'je01_native_role_jvp_patch.py'
         and args.expected_patch_sha == PATCH_SHA and sha(args.patch) == PATCH_SHA,
         'FINAL_ROLE_PATCH_PIN')
    need(args.expected_adapter_sha == ADAPTER_SHA and args.expected_smoke_receipt == SMOKE_SHA,
         'GUARDED_ADAPTER_AND_SMOKE_PINS')
    for key in ('expected_patched_cpp_sha', 'expected_patched_c_sha'):
        need(is_sha(getattr(args, key, None)), 'EXTERNAL_PATCHED_SOURCE_PIN:' + key)
    verify_import_sources()
    legacy.verify_helpers()
    return {'patch_source_sha256': PATCH_SHA, 'patched_cpp_sha256': args.expected_patched_cpp_sha,
            'patched_c_sha256': args.expected_patched_c_sha, 'adapter_sha256': ADAPTER_SHA,
            'smoke_canonical_sha256': SMOKE_SHA, 'runtime_mode': MODE,
            'diagnostic_role_indices': list(DIAGNOSTIC_ROLES), 'diagnostic_tier': TIER,
            'diagnostic_scope': SCOPE}


def write_or_compare(path, value, *, resume):
    path = Path(path)
    need('receipt_sha256' not in value, 'DO_NOT_HASH_CHECKSUM_INTO_ITSELF')
    if path.exists() or path.is_symlink():
        need(resume, 'REPORT_EXISTS_USE_EXPLICIT_RESUME')
        previous = sealed(path)
        need({k: v for k, v in previous.items() if k != 'receipt_sha256'} == value,
             'RESUMED_REPORT_RAW_RECOMPUTATION_DRIFT')
        return previous
    return common.write_seal(path, value)


def build(args):
    configuration = checked_configuration(args)
    smoke = legacy.verify_adapter_gate(args)
    parent_source, parent = legacy.verify_parent_source()
    root = require_root(args.root)
    need(not any((root / name).exists() or (root / name).is_symlink()
                 for name in ('source', 'build', 'bin', 'NATIVE_BUILD_RECEIPT.json', 'SOURCE_PATCH_RECEIPT.json')),
         'FRESH_BUILD_REQUIRED')
    root.mkdir(mode=0o700, exist_ok=True)
    ledger = 'JE01_ACTUAL_CAL240_FFN_H2_SWIGLU_NATIVE_OUTPUT_DP_20261122.json'
    ledger_source = parent_source / 'docs/research' / ledger
    sealed(ledger_source, '35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb')
    target = output_path(root, root / 'docs/research' / ledger)
    if target.exists():
        need(sha(target) == sha(ledger_source), 'PRESERVED_LEDGER_DRIFT')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ledger_source, target)
    original = legacy.verified_original_runtime()
    legacy.runtime_matches_smoke(original['library'], smoke)
    copied = output_path(root, root / 'source')
    shutil.copytree(parent_source, copied, symlinks=True)
    patch = legacy.pinned_module(args.patch, PATCH_SHA, 'je01_role_jvp_patch_for_build')
    applied = patch.stage(copied)
    need(applied.get('schema') == PATCH_SCHEMA
         and applied.get('status') == 'SEPARATE_SCRATCH_SOURCE_PATCHED_NOT_EXECUTED'
         and applied.get('base_cpp_sha256') == legacy.BASE_CPP_SHA
         and applied.get('base_c_sha256') == legacy.BASE_C_SHA
         and applied.get('patched_cpp_sha256') == args.expected_patched_cpp_sha
         and applied.get('patched_c_sha256') == args.expected_patched_c_sha
         and applied.get('adapter_sha256') == ADAPTER_SHA and applied.get('runtime_mode') == MODE
         and applied.get('supported_role_indices') == list(range(24))
         and applied.get('supported_tiers') == ['Q4', 'Q5'] and applied.get('one_direction_per_run') is True
         and applied.get('alpha') == 0 and applied.get('production_modified') is False,
         'SOURCE_PATCH_CONTRACT')
    need(sha(copied / 'mlx_moe.cpp') == args.expected_patched_cpp_sha
         and sha(copied / 'qwen_infer.c') == args.expected_patched_c_sha
         and sha(copied / ADAPTER_NAME) == ADAPTER_SHA, 'PATCHED_BYTES')
    patch_receipt = common.write_seal(root / 'SOURCE_PATCH_RECEIPT.json',
                                     {k: v for k, v in applied.items() if k != 'receipt_sha256'})
    cmake_file = copied / 'CMakeLists.txt'
    previous = cmake_file.read_text()
    anchor = 'OUTPUT_NAME je01_role_directional_fixed_native'
    need(previous.count(anchor) == 1, 'CMAKE_OUTPUT_ANCHOR')
    cmake_file.write_text(previous.replace(anchor, 'OUTPUT_NAME je01_native_role_jvp', 1))
    cmake = fixed.BC_BUILD / 'venv/bin/cmake'
    mlx_cmake = original['library'].parent / 'share/cmake'
    need(cmake.is_file() and mlx_cmake.is_dir(), 'ORIGINAL_TOOLCHAIN_REQUIRED')
    commands = [[str(cmake), '-S', str(copied), '-B', str(root / 'build'), '-G', 'Ninja',
                 '-DCMAKE_BUILD_TYPE=Release', '-DMLX_CMAKE_DIR=' + str(mlx_cmake)],
                [str(cmake), '--build', str(root / 'build'), '--target', 'qwen_infer_gpu', '--parallel', '2']]
    for index, command in enumerate(commands):
        with (root / f'compile{index}.stdout.txt').open('xb') as out, (root / f'compile{index}.stderr.txt').open('xb') as err:
            subprocess.run(command, stdout=out, stderr=err, check=True, timeout=300)
    (root / 'bin').mkdir(mode=0o700)
    executable = root / 'bin/je01_native_role_jvp'
    shutil.copy2(root / 'build/je01_native_role_jvp', executable)
    executable.chmod(0o700)
    need(executable.stat().st_size > 100000, 'NATIVE_BINARY_TOO_SMALL')
    receipt = {'schema': BUILD_SCHEMA, 'status': 'ROLE_JVP_NATIVE_COMPILE_PASS',
               'configuration': configuration, 'fixed_build_receipt_sha256': parent['receipt_sha256'],
               'fixed_native_sha256': legacy.FIXED_NATIVE_SHA,
               'base_cpp_sha256': legacy.BASE_CPP_SHA, 'base_c_sha256': legacy.BASE_C_SHA,
               'source_patch_receipt_sha256': patch_receipt['receipt_sha256'],
               'binary_sha256': sha(executable), 'binary_bytes': executable.stat().st_size,
               'cmake_sha256': sha(cmake_file), 'mlx_library_sha256': sha(original['library'] / 'libmlx.dylib'),
               'runner_source_sha256': sha(Path(__file__)), 'helper_source_sha256': SOURCE_PINS,
               'adapter_smoke_receipt_sha256': smoke['canonical_sha256'],
               'adapter_smoke_source_sha256': smoke['source_sha256'],
               'model_original_af4_sha256': common.AF_SHA, 'dataset_receipt_sha256': fixed.DATASET_RECEIPT,
               'bank_receipt_sha256': fixed.BANK_RECEIPT, 'scope': SCOPE,
               'new_role_runtime_validation': 'NOT_RUN', 'full48_empirical_fisher': None,
               'allocation_selected': False, 'independent_validation_ood_accessed': False, 'production_modified': False}
    common.write_seal(root / 'NATIVE_BUILD_RECEIPT.json', receipt)
    print(json.dumps({'event': 'ROLE_JVP_BUILD_DONE', 'receipt_sha256': receipt['receipt_sha256'],
                      'binary_sha256': receipt['binary_sha256']}, sort_keys=True), flush=True)
    return receipt


def preflight(args):
    configuration = checked_configuration(args)
    need(is_sha(args.expected_build_receipt), 'EXTERNAL_BUILD_RECEIPT_REQUIRED')
    smoke = legacy.verify_adapter_gate(args)
    original = legacy.verified_original_runtime()
    legacy.runtime_matches_smoke(original['library'], smoke)
    legacy.verify_parent_source()
    root = require_root(args.root)
    receipt = sealed(root / 'NATIVE_BUILD_RECEIPT.json', args.expected_build_receipt)
    executable = root / 'bin/je01_native_role_jvp'
    need(receipt.get('schema') == BUILD_SCHEMA and receipt.get('status') == 'ROLE_JVP_NATIVE_COMPILE_PASS'
         and receipt.get('configuration') == configuration
         and receipt.get('fixed_build_receipt_sha256') == legacy.FIXED_BUILD_RECEIPT
         and receipt.get('fixed_native_sha256') == legacy.FIXED_NATIVE_SHA
         and receipt.get('adapter_smoke_receipt_sha256') == SMOKE_SHA
         and receipt.get('runner_source_sha256') == sha(Path(__file__))
         and receipt.get('helper_source_sha256') == SOURCE_PINS
         and receipt.get('production_modified') is False, 'BUILD_RECEIPT_CONTRACT')
    need(sha(executable) == receipt.get('binary_sha256')
         and sha(root / 'source/mlx_moe.cpp') == args.expected_patched_cpp_sha
         and sha(root / 'source/qwen_infer.c') == args.expected_patched_c_sha
         and sha(root / 'source' / ADAPTER_NAME) == ADAPTER_SHA
         and sha(root / 'source/CMakeLists.txt') == receipt.get('cmake_sha256')
         and sha(original['library'] / 'libmlx.dylib') == receipt.get('mlx_library_sha256'),
         'BUILD_SOURCE_OR_RUNTIME_DRIFT')
    sealed(root / 'SOURCE_PATCH_RECEIPT.json', receipt['source_patch_receipt_sha256'])
    audit, reference = legacy.fixed_reference(args.fixed_audit, original['dataset'])
    identity = {**original['original_identity'], 'native_executable_sha256': receipt['binary_sha256'],
                'fixed_fd_executable_sha256': legacy.FIXED_NATIVE_SHA,
                'new_build_receipt_sha256': receipt['receipt_sha256'],
                'adapter_header_sha256': ADAPTER_SHA, 'adapter_smoke_receipt_sha256': SMOKE_SHA,
                'baseline_route_sha256': fixed.ROUTE_SHA, 'original_off_receipt_sha256': fixed.OFF_RECEIPT,
                'gate_fixed_reference_audit_sha256': legacy.FIXED_AUDIT_SHA,
                'derivative_definition': 'THREE_NATIVE_ROLE_Q4_ALPHA_ZERO_FIXED_HARD_ROUTE_FULL_PREFIX_DIRECTIONS'}
    return {**original, 'build': receipt, 'identity': identity, 'executable': executable,
            'fd_executable': legacy.FIXED_ROOT / 'bin/je01_role_directional_fixed_native',
            'gate_reference_audit': audit, 'gate_reference_derivatives': reference}


def cell_folder(root, kind, role, alpha, attempt):
    validate_cell(kind, role, alpha, attempt)
    label = 'off' if kind == 'OFF' else f'{kind.lower()}_r{role:02d}_Q4_a{alpha.replace("+", "p").replace("-", "m")}'
    return output_path(root, Path(root) / 'runs/pilot16' / f'{label}_attempt{attempt}')


def establish_plan(args, context):
    """Seal the whole design before the first native cell; never adapt to results."""
    root = require_root(args.root)
    path = output_path(root, root / 'PLAN_RECEIPT.json')
    sequence = [{'kind': 'OFF', 'role_index': None, 'alpha': '0'},
                {'kind': 'JVP', 'role_index': 0, 'alpha': '0'}]
    for role in (1, 2):
        sequence.extend({'kind': 'FD', 'role_index': role, 'alpha': alpha}
                        for alpha in ('0',) + FD_ALPHAS)
        sequence.append({'kind': 'JVP', 'role_index': role, 'alpha': '0'})
    plan = {'schema': PLAN_SCHEMA, 'scope': SCOPE,
            'source_identity': context['identity'], 'new_build_receipt_sha256': context['build']['receipt_sha256'],
            'runner_source_sha256': sha(Path(__file__)), 'patch_source_sha256': PATCH_SHA,
            'diagnostic_role_indices': list(DIAGNOSTIC_ROLES),
            'diagnostic_role_keys': [common.ROLE_ORDER[role] for role in DIAGNOSTIC_ROLES],
            'tier': TIER, 'alpha_grid': ['0'] + list(FD_ALPHAS), 'jvp_alpha': 0,
            'native_execution_sequence': sequence, 'native_cell_count': 14,
            'case_ids': list(range(16)), 'gold_tokens': 656,
            'full_prefix_completion_counters': dict(legacy.COMPLETION),
            'gate_fd_reference': {'canonical_sha256': legacy.FIXED_AUDIT_SHA,
                                  'file_sha256': sha(args.fixed_audit),
                                  'status': 'PREVIOUSLY_CONVERGED_INDEPENDENT_FIXED_ROUTE_AUDIT'},
            'fd_convergence_rule': {'coarse_h': .05, 'fine_h': .025,
                                    'rms_absolute_tolerance': 1e-4, 'rms_relative_tolerance': .05,
                                    'reference': 'RMS_OF_FINE_FD_OVER_ALL_656_GOLD_TOKENS'},
            'jvp_agreement_rule': {'rms_absolute_tolerance': 1e-4, 'rms_relative_tolerance': .05,
                                   'reference': 'RMS_OF_FINE_FD_OVER_ALL_656_GOLD_TOKENS'},
            'pass_requires_fd_convergence_and_jvp_agreement': True,
            'every_zero_and_jvp_primal_requires_original_gold_sha': common.BASE_SHA,
            'every_fixed_route_output_requires_baseline_sha': fixed.ROUTE_SHA,
            'all_predeclared_roles_reported_regardless_of_numerical_sign_or_pass': True,
            'result_adaptive_thresholds': False, 'result_adaptive_roles': False,
            'result_adaptive_attempts': False, 'retry_policy': 'ONLY_EXECUTION_OR_INTEGRITY_FAILURE_WITH_NEW_ATTEMPT',
            'observed_label_gram_rule': 'ONLY_AFTER_ALL_THREE_DIRECTIONS_PASS; EXPLORATORY_NOT_ALLOCATION',
            'full48_empirical_fisher': None, 'train240_allocation': 'NOT_RUN',
            'allocation_selected': False, 'independent_validation_ood_accessed': False, 'production_modified': False}
    if not path.exists() and not path.is_symlink():
        need(not (root / 'runs').exists() and not (root / 'runs').is_symlink(),
             'PLAN_MUST_BE_SEALED_BEFORE_ANY_NATIVE_CELL')
    sealed_plan = write_or_compare(path, plan, resume=True)
    if args.attempt > 1:
        for previous_attempt in range(1, args.attempt):
            need(not (root / f'JE01_THREE_ROLE_JVP_DIAGNOSTIC_attempt{previous_attempt}.json').exists(),
                 'COMPLETED_DIAGNOSTIC_CANNOT_BE_RETRIED_FOR_A_DIFFERENT_RESULT')
        failures = []
        for failure_path in (root / 'runs/pilot16').glob('*_attempt*/FAILED_RUN_RECEIPT.json'):
            failed = sealed(failure_path)
            if (failed.get('schema') == RUN_SCHEMA and failed.get('status') == 'FAILED_NO_VERIFIED_DIRECTION'
                    and type(failed.get('attempt')) is int and 1 <= failed['attempt'] < args.attempt
                    and failed.get('plan_receipt_sha256') == sealed_plan['receipt_sha256']):
                failures.append(failed['receipt_sha256'])
        need(failures, 'RETRY_REQUIRES_VERIFIED_PRIOR_EXECUTION_OR_INTEGRITY_FAILURE')
    print(json.dumps({'event': 'ROLE_JVP_PLAN_SEALED', 'receipt_sha256': sealed_plan['receipt_sha256'],
                      'native_cell_count': 14, 'role_indices': list(DIAGNOSTIC_ROLES)}, sort_keys=True), flush=True)
    return sealed_plan


def cell_environment(folder, context, kind, role, alpha):
    env = legacy.environment(folder, context['manifest'], context['metadata'], context['library'], jvp=False)
    if kind != 'OFF':
        env.update({'JE01_REAL_Q35_BUNDLE': str(common.BANK / 'JE01_ROLE_DIRECTION_BANK.bin'),
                    'JE01_REAL_Q35_TSV': str(common.BANK / 'JE01_ROLE_DIRECTION_BANK.tsv'),
                    'JE01_DIRECTION_ROLE': str(role), 'JE01_DIRECTION_TIER': TIER,
                    'JE01_DIRECTION_ALPHA': alpha,
                    'JE01_DIRECTION_BASELINE_ROUTES': str(fixed.OFF_FOLDER / 'routes.bin')})
    if kind == 'JVP':
        env.update({'JE01_NATIVE_JVP_MODE': MODE, 'JE01_NATIVE_JVP_GOLD_OUTPUT': str(folder / 'jvp_gold.bin')})
    return env


def validate_completion(stderr, kind):
    if kind in ('OFF', 'JVP'):
        return legacy.validate_completion(stderr, jvp=kind == 'JVP')
    need('FATAL:' not in stderr and 'RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16' in stderr
         and common.MARKER in stderr
         and 'JE01_FIXED_BASELINE_ROUTES_ENABLED records=' + str(fixed.ROUTE_RECORDS) in stderr
         and legacy.JVP_MARKER not in stderr, 'FIXED_FD_NATIVE_COMPLETION')
    return None


def parse_raw_cell(folder, context, kind, role, alpha):
    gold, route = folder / 'gold_native.bin', folder / 'routes.bin'
    gold_sha, route_sha = sha(gold), sha(route)
    need(route_sha == fixed.ROUTE_SHA and route.stat().st_size == fixed.ROUTE_BYTES,
         'FIXED_ROUTE_BYTE_PARITY')
    values, targets = common.read_gold(gold.read_bytes(), context['metadata'].read_bytes())
    need(targets == context['dataset']['target_token_ids'] and len(values) == 16
         and sum(map(len, values)) == 656 and common.read_routes(route.read_bytes()) == context['routes'],
         'RAW_TARGET_OR_ROUTE_ALIGNMENT')
    if alpha == '0':
        need(gold_sha == common.BASE_SHA and values == context['off']['per_token_nll'],
             'ZERO_ALPHA_OR_JVP_EXACT_AF4_PRIMAL')
    completion = validate_completion((folder / 'stderr.txt').read_text(errors='replace'), kind)
    gradients = None
    if kind == 'JVP':
        gradients = legacy.read_jvp((folder / 'jvp_gold.bin').read_bytes(), context['metadata'].read_bytes(), values)
    else:
        need(not (folder / 'jvp_gold.bin').exists() and not (folder / 'jvp_gold.bin').is_symlink(),
             'NON_JVP_GRADIENT_FILE_FORBIDDEN')
    return {'target_token_ids': targets, 'per_token_nll': values, 'mean_gold_nll': mean(flatten(values)),
            'output_sha256': gold_sha, 'route_output_sha256': route_sha,
            'route_output_bytes': route.stat().st_size, 'routed_token_layer_events': fixed.ROUTE_RECORDS,
            'gradients': gradients, 'native_prefix_completion': completion,
            'gradient_output_sha256': sha(folder / 'jvp_gold.bin') if kind == 'JVP' else None,
            'gradient_output_bytes': (folder / 'jvp_gold.bin').stat().st_size if kind == 'JVP' else None,
            'stdout_sha256': sha(folder / 'stdout.txt'), 'stderr_sha256': sha(folder / 'stderr.txt')}


def validate_saved_contract(receipt, context, kind, role, alpha, attempt):
    validate_cell(kind, role, alpha, attempt)
    need(receipt.get('schema') == RUN_SCHEMA and receipt.get('status') == 'SUCCEEDED_EXIT0'
         and receipt.get('scope') == SCOPE and receipt.get('kind') == kind
         and receipt.get('source_identity') == context['identity']
         and receipt.get('plan_receipt_sha256') == context['plan']['receipt_sha256']
         and receipt.get('role_index') == role
         and receipt.get('role_key') == (common.ROLE_ORDER[role] if role is not None else None)
         and receipt.get('tier') == (TIER if role is not None else None)
         and receipt.get('alpha') == common.ALPHAS[alpha] and receipt.get('alpha_literal') == alpha
         and receipt.get('attempt') == attempt and receipt.get('case_ids') == list(range(16))
         and receipt.get('target_token_ids') == context['dataset']['target_token_ids']
         and receipt.get('runner_source_sha256') == sha(Path(__file__))
         and receipt.get('execution', {}).get('native_exit_code') == 0
         and receipt.get('execution', {}).get('stop_reason') is None
         and receipt.get('native_jvp_enabled') is (kind == 'JVP')
         and receipt.get('native_overlay_enabled') is (kind != 'OFF')
         and receipt.get('fixed_route_replay_enabled') is (kind != 'OFF')
         and receipt.get('full_prompt_and_KV_recomputed') is True
         and receipt.get('router_scores_and_selected_weights_dynamic') is True
         and receipt.get('full_prefix_tangent_carried') is (kind == 'JVP')
         and receipt.get('full48_empirical_fisher') is None and receipt.get('allocation_selected') is False
         and receipt.get('independent_validation_ood_accessed') is False
         and receipt.get('production_modified') is False, 'RESUME_CELL_CONTRACT')
    executable_sha = context['build']['binary_sha256'] if kind != 'FD' else legacy.FIXED_NATIVE_SHA
    need(receipt.get('executed_binary_sha256') == executable_sha, 'CELL_EXECUTABLE_PIN')


def verify_saved(args, context, kind, role, alpha):
    folder = cell_folder(args.root, kind, role, alpha, args.attempt)
    receipt = sealed(folder / 'RUN_RECEIPT.json')
    validate_saved_contract(receipt, context, kind, role, alpha, args.attempt)
    actual = parse_raw_cell(folder, context, kind, role, alpha)
    need(all(receipt.get(key) == value for key, value in actual.items()), 'RESUME_RAW_REPLAY_DRIFT')
    if kind != 'OFF':
        control = verify_saved(args, context, 'OFF', None, '0')
        need(receipt.get('new_off_receipt_sha256') == control['receipt_sha256'], 'RESUME_OFF_RECEIPT_BINDING')
    return receipt


def run_cell(args, context, kind, role=None, alpha='0'):
    validate_cell(kind, role, alpha, args.attempt)
    folder = cell_folder(args.root, kind, role, alpha, args.attempt)
    if folder.exists() or folder.is_symlink():
        need(args.resume, 'CELL_EXISTS_USE_EXPLICIT_RESUME_OR_FRESH_ATTEMPT')
        return verify_saved(args, context, kind, role, alpha)
    control = verify_saved(args, context, 'OFF', None, '0') if kind != 'OFF' else None
    if kind == 'FD' and alpha != '0':
        verify_saved(args, context, 'FD', role, '0')
    folder.mkdir(parents=True, mode=0o700)
    (folder / 'tmp').mkdir(mode=0o700)
    executable = context['fd_executable'] if kind == 'FD' else context['executable']
    expected_binary = legacy.FIXED_NATIVE_SHA if kind == 'FD' else context['build']['binary_sha256']
    need(sha(executable) == expected_binary, 'PRE_RUN_BINARY_HASH')
    env = cell_environment(folder, context, kind, role, alpha)
    print(json.dumps({'event': 'ROLE_JVP_CELL_START', 'kind': kind, 'role': role,
                      'alpha': alpha, 'attempt': args.attempt, 'root': str(args.root)}, sort_keys=True), flush=True)
    execution = None
    try:
        execution = legacy.bounded_native(executable, folder, env, args.timeout_seconds)
        need(execution['native_exit_code'] == 0 and execution['stop_reason'] is None, 'NATIVE_PROCESS_FAILED')
        raw = parse_raw_cell(folder, context, kind, role, alpha)
        receipt = {'schema': RUN_SCHEMA, 'status': 'SUCCEEDED_EXIT0', 'scope': SCOPE,
                   'kind': kind, 'source_identity': context['identity'], 'case_ids': list(range(16)),
                   'plan_receipt_sha256': context['plan']['receipt_sha256'],
                   'role_index': role, 'role_key': common.ROLE_ORDER[role] if role is not None else None,
                   'tier': TIER if role is not None else None, 'alpha': common.ALPHAS[alpha],
                   'alpha_literal': alpha, 'attempt': args.attempt,
                   'executed_binary_sha256': expected_binary,
                   'new_off_receipt_sha256': control['receipt_sha256'] if control else None,
                   'original_off_receipt_sha256': fixed.OFF_RECEIPT,
                   'native_jvp_enabled': kind == 'JVP', 'native_overlay_enabled': kind != 'OFF',
                   'fixed_route_replay_enabled': kind != 'OFF',
                   'full_prompt_and_KV_recomputed': True, 'router_scores_and_selected_weights_dynamic': True,
                   'full_prefix_tangent_carried': kind == 'JVP',
                   'route_equality_interpretation': 'REPLAY_CONFORMANCE_NOT_NATURAL_ROUTING_STABILITY',
                   'execution': execution, 'runner_source_sha256': sha(Path(__file__)),
                   'full48_empirical_fisher': None, 'allocation_selected': False,
                   'independent_validation_ood_accessed': False, 'production_modified': False, **raw}
        common.write_seal(folder / 'RUN_RECEIPT.json', receipt)
    except BaseException as error:
        if isinstance(error, legacy.NativeExecutionError):
            execution = error.execution
        common.write_seal(folder / 'FAILED_RUN_RECEIPT.json', {
            'schema': RUN_SCHEMA, 'status': 'FAILED_NO_VERIFIED_DIRECTION', 'scope': SCOPE,
            'source_identity': context['identity'], 'kind': kind, 'role_index': role,
            'plan_receipt_sha256': context['plan']['receipt_sha256'],
            'alpha_literal': alpha, 'attempt': args.attempt, 'execution': execution,
            'error': f'{type(error).__name__}: {error}', 'full48_empirical_fisher': None,
            'allocation_selected': False, 'independent_validation_ood_accessed': False, 'production_modified': False})
        raise
    print(json.dumps({'event': 'ROLE_JVP_CELL_DONE', 'kind': kind, 'role': role,
                      'alpha': alpha, 'receipt_sha256': receipt['receipt_sha256'],
                      'mean_gold_nll': receipt['mean_gold_nll']}, sort_keys=True), flush=True)
    return receipt


def derivative_rows(positive, negative, denominator):
    need(isinstance(positive, list) and isinstance(negative, list) and len(positive) == len(negative) == 16
         and type(denominator) in (int, float) and math.isfinite(denominator) and denominator > 0,
         'FD_ARRAY_OR_DENOMINATOR')
    need(all(isinstance(a, list) and isinstance(b, list) and len(a) == len(b) > 0
             for a, b in zip(positive, negative)) and sum(map(len, positive)) == 656, 'FD_TOKEN_ALIGNMENT')
    need(all(type(value) in (int, float) and math.isfinite(value)
             for value in flatten(positive) + flatten(negative)), 'FD_NONFINITE')
    return [[(a - b) / denominator for a, b in zip(pa, na)] for pa, na in zip(positive, negative)]


def fd_reference(args, context, role):
    need(role in DIAGNOSTIC_ROLES and type(role) is int, 'FD_REFERENCE_ROLE')
    if role == 0:
        audit, fine = legacy.fixed_reference(args.fixed_audit, context['dataset'])
        return {'schema': FD_SCHEMA, 'kind': 'REUSED_PINNED_INDEPENDENT_GATE_AUDIT',
                'role_index': 0, 'role_key': common.ROLE_ORDER[0], 'tier': TIER,
                'scope': SCOPE, 'reference_receipt_sha256': audit['receipt_sha256'],
                'reference_file_sha256': sha(args.fixed_audit), 'convergence': audit['statistics']['convergence'],
                'per_token_fine_derivative': fine, 'full48_empirical_fisher': None,
                'allocation_selected': False, 'independent_validation_ood_accessed': False, 'production_modified': False}
    cells = {alpha: run_cell(args, context, 'FD', role, alpha) for alpha in ('0',) + FD_ALPHAS}
    # Reopen and hash-check every raw input, including when these cells were
    # just generated. The summary is never derived from a status label alone.
    cells = {alpha: verify_saved(args, context, 'FD', role, alpha) for alpha in ('0',) + FD_ALPHAS}
    coarse = derivative_rows(cells['+0.05']['per_token_nll'], cells['-0.05']['per_token_nll'], .1)
    fine = derivative_rows(cells['+0.025']['per_token_nll'], cells['-0.025']['per_token_nll'], .05)
    comparison = legacy.compare_derivatives(coarse, fine)
    convergence = {'coarse_h': .05, 'fine_h': .025, 'absolute_tolerance': 1e-4, 'relative_tolerance': .05,
                   'rms_fine_derivative': comparison['rms_fixed_fine_fd'],
                   'rms_disagreement': comparison['rms_disagreement'],
                   'rms_acceptance_threshold': comparison['rms_acceptance_threshold'],
                   'max_absolute_disagreement': comparison['max_absolute_disagreement'],
                   'cosine_similarity': comparison['cosine_similarity'], 'passed': comparison['passed']}
    report = {'schema': FD_SCHEMA, 'kind': 'NEW_PINNED_NATIVE_FIXED_ROUTE_FD_REFERENCE',
              'status': 'FIXED_BRANCH_CONVERGED' if convergence['passed'] else 'FIXED_BRANCH_NUMERICAL_REJECTED',
              'role_index': role, 'role_key': common.ROLE_ORDER[role], 'tier': TIER,
              'scope': SCOPE, 'source_identity': context['identity'], 'case_ids': list(range(16)),
              'target_token_ids': context['dataset']['target_token_ids'],
              'run_receipt_sha256': {key: value['receipt_sha256'] for key, value in cells.items()},
              'convergence': convergence, 'per_token_coarse_derivative': coarse, 'per_token_fine_derivative': fine,
              'full48_empirical_fisher': None, 'allocation_selected': False,
              'independent_validation_ood_accessed': False, 'production_modified': False}
    return write_or_compare(output_path(args.root, args.root / f'FD_REFERENCE_r{role:02d}_Q4_attempt{args.attempt}.json'),
                            report, resume=args.resume)


def diagnostic(args):
    context = preflight(args)
    plan = establish_plan(args, context)
    context = {**context, 'plan': plan,
               'identity': {**context['identity'], 'plan_receipt_sha256': plan['receipt_sha256']}}
    control = run_cell(args, context, 'OFF')
    reports, columns = [], []
    for role in DIAGNOSTIC_ROLES:
        reference = fd_reference(args, context, role)
        measured = run_cell(args, context, 'JVP', role)
        measured = verify_saved(args, context, 'JVP', role, '0')
        comparison = legacy.compare_derivatives(measured['gradients']['per_token_dnll'],
                                                reference['per_token_fine_derivative'])
        passed = reference['convergence']['passed'] is True and comparison['passed'] is True
        result = {'role_index': role, 'role_key': common.ROLE_ORDER[role], 'tier': TIER,
                  'status': 'FIXED_FD_AND_JVP_AGREE' if passed else 'NUMERICAL_REJECTED',
                  'passed': passed, 'fd_convergence': reference['convergence'], 'jvp_vs_fine_fd': comparison,
                  'fd_reference_receipt_sha256': reference.get('receipt_sha256', reference.get('reference_receipt_sha256')),
                  'jvp_receipt_sha256': measured['receipt_sha256'],
                  'jvp_output_sha256': measured['gradient_output_sha256']}
        reports.append(result)
        columns.append(flatten(measured['gradients']['per_token_dnll']))
        print(json.dumps({'event': 'ROLE_JVP_DIRECTION_DONE', **result}, sort_keys=True), flush=True)
    passed = all(row['passed'] for row in reports)
    original = common.MODEL / 'weights_moe/deepseek_moe_af.bin'
    need(sha(original) == common.AF_SHA, 'POST_RUN_ORIGINAL_AF4_DRIFT')
    gram = None
    if passed:
        gram = {'definition': 'MEAN_OBSERVED_LABEL_DIRECTIONAL_GRADIENT_OUTER_PRODUCT_ON_SEEN_TRAIN16',
                'coordinate_order': [common.ROLE_ORDER[role] + [TIER] for role in DIAGNOSTIC_ROLES],
                'matrix': [[math.fsum(a * b for a, b in zip(left, right)) / 656
                            for right in columns] for left in columns],
                'signed_mean_gradient': [mean(column) for column in columns], 'tokens': 656,
                'interpretation': 'EXPLORATORY_FIXED_BRANCH_THREE_DIRECTION_GRAM_NOT_FULL_PARAMETER_FISHER',
                'used_for_policy_selection': False}
    report = {'schema': REPORT_SCHEMA,
              'status': 'THREE_ROLE_NATIVE_JVP_FIXED_FD_PASS' if passed else 'THREE_ROLE_NUMERICAL_REJECTED',
              'scope': SCOPE, 'source_identity': context['identity'], 'case_ids': list(range(16)),
              'plan_receipt_sha256': plan['receipt_sha256'],
              'gold_tokens': 656, 'off_receipt_sha256': control['receipt_sha256'],
              'roles': reports, 'observed_label_three_direction_gram': gram,
              'post_run_original_af4_sha256': common.AF_SHA,
              'full48_empirical_fisher': None, 'train240_allocation': 'NOT_RUN',
              'allocation_selected': False, 'independent_validation_ood_accessed': False,
              'production_modified': False, 'runner_source_sha256': sha(Path(__file__))}
    report = write_or_compare(output_path(args.root, args.root / f'JE01_THREE_ROLE_JVP_DIAGNOSTIC_attempt{args.attempt}.json'),
                              report, resume=args.resume)
    print(json.dumps({'event': 'ROLE_JVP_DIAGNOSTIC_DONE', 'status': report['status'],
                      'receipt_sha256': report['receipt_sha256'], 'roles': reports}, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'diagnostic'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke-receipt', type=Path, required=True)
    parser.add_argument('--expected-smoke-receipt', required=True)
    parser.add_argument('--expected-adapter-sha', required=True)
    parser.add_argument('--patch', type=Path, default=TOOL_DIR / 'je01_native_role_jvp_patch.py')
    parser.add_argument('--expected-patch-sha', required=True)
    parser.add_argument('--expected-patched-cpp-sha', required=True)
    parser.add_argument('--expected-patched-c-sha', required=True)
    parser.add_argument('--expected-build-receipt')
    parser.add_argument('--fixed-audit', type=Path)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--timeout-seconds', type=int, default=900)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    validate_cell('OFF', None, '0', args.attempt)
    legacy.memory_stop_reason(0, 100, 0, args.timeout_seconds)
    if args.action == 'build':
        need(not args.resume, 'BUILD_IS_FRESH_ONLY')
        build(args)
    else:
        need(args.fixed_audit is not None, 'FIXED_GATE_AUDIT_REQUIRED')
        diagnostic(args)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError(f'JE01_ROLE_JVP_SIGNAL:{signum}')
    signal.signal(signal.SIGTERM, interrupted)
    main()
