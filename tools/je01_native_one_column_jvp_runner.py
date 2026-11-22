#!/usr/bin/env python3
"""Receipt-gated full-prefix native JVP for one frozen TRAIN16 direction.

This lane has its own source tree and executable. It first proves new-binary
OFF gold/routes byte parity, then executes L12/E13 gate/Q4 at alpha zero with
fixed baseline hard indices and dynamic router scores/weights. Full-prefix JVP
records are compared with the separately converged fixed-branch finite
difference. It does not construct a full48 Fisher, choose an allocation, read
holdouts, modify MLX, or perform fleet admission; admission belongs to the caller.

Build/run require the final guarded adapter smoke's external canonical pin and
exact source hashes. No provisional adapter or default fabricated pin can pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import time

import je01_role_directional_native_runner as common
import je01_role_directional_fixed_native as fixed


WORK = Path('/Volumes/D50/TailnetCommander/work')
PREFIX = 'je01_native_one_column_jvp_'
FIXED_ROOT = WORK / 'je01_role_directional_fixed_20261123_v1'
FIXED_BUILD_RECEIPT = 'dbcdd1beb38a3863a5cbb1ac4bb697dddcc6e263448bea80d03385b7db8c2fc7'
FIXED_NATIVE_SHA = '46ea1f61fdd7483767ebc20900f4c97f9663a51148c74e566d2f66ccc848e688'
BASE_CPP_SHA = '390fb6b136bdd9f4978e6482b7a18b11951333e453752a15c3fd094a7d3b07f3'
BASE_C_SHA = '8d1a8fce7d578319ceb45d17948ebe0349758acb2ae0b786944c8119ea795c3d'
HELPER_PINS = {
    'je01_role_directional_native_runner.py': '5607c0afef3bd4b1a61bcab6622267acc6b6137fc65ba0e79873e92e31d7e9e6',
    'je01_role_directional_fixed_native.py': 'cffe2e1dd8298681f3a134caace742ddc9fea77209c58683cd8d2f48fe6284e7',
    'je01_role_directional_fixed_routes_patch.py': '233937b59293bec11300b66450b819dc48bb09b8d28eb6e98ca81aac7f511019',
    'je01_role_directional_bank.py': '045fd22d86e4ce18f967d66830a1f71b0c61ad6ab5c5889d953cb9a8d05bce33',
}
FIXED_AUDIT_SHA = '87f239d900d9aff3921ab99d478024b13002e80b3b3d65683d4a055e4147598c'
FIXED_AUDIT_FILE_SHA = 'a64d13644a47396c8d514d60f6b5eee2f5683ef91975bc7433ab0260066208b9'
FIXED_AUDIT_SCHEMA = 'beglin-je01-independent-fixed-route-six-cell-audit/1'
BUILD_SCHEMA = 'beglin-je01-native-one-column-jvp-build/1'
RUN_SCHEMA = 'beglin-je01-native-one-column-jvp-run/1'
REPORT_SCHEMA = 'beglin-je01-native-one-column-jvp-comparison/1'
PATCH_SCHEMA = 'beglin-je01-native-one-column-jvp-source-patch/1'
SMOKE_SCHEMA = 'beglin-je01-mlx-native-custom-jvp-smoke-receipt/2'
ADAPTER_NAME = 'je01_mlx_native_jvp_adapters.h'
SMOKE_CPP_NAME = 'je01_mlx_native_jvp_smoke.cpp'
SMOKE_VALIDATOR_NAME = 'je01_mlx_native_jvp_smoke.py'
SMOKE_NAMES = {ADAPTER_NAME, SMOKE_CPP_NAME, SMOKE_VALIDATOR_NAME}
GUARDS = {
    'gather_qmm_vjp': 'JE01_JVP_ADAPTER_REVERSE_MODE_NOT_SUPPORTED',
    'gather_qmm_direct_vmap': 'JE01_JVP_ADAPTER_DIRECT_VMAP_NOT_SUPPORTED',
    'scatter_vjp': 'JE01_JVP_ADAPTER_REVERSE_MODE_NOT_SUPPORTED',
    'scatter_direct_vmap': 'JE01_JVP_ADAPTER_DIRECT_VMAP_NOT_SUPPORTED',
}
SMOKE_CASES = {'gather_gate_q4', 'gather_down_q4', 'gather_gate_q5', 'gather_down_q5',
               'scatter_both', 'scatter_cache_only', 'scatter_update_only', 'kv_two_step'}
JVP_MAGIC = b'JE01_NATIVE_ONE_COLUMN_JVP_GOLD_V1\n'
JVP_MODE = 'ONE_L12_E13_GATE_Q4'
COMPLETION = {'tokens': 4334, 'cases': 32, 'layers': 117018, 'gold': 656}
JVP_MARKER = 'JE01_ONE_COLUMN_JVP_PREFIX_COMPLETE'
DERIVATIVE_KIND = 'NATIVE_FULL_PREFIX_ONE_COLUMN_JVP_FIXED_HARD_ROUTE_COMPUTATIONAL_DIRECTION'
ROLE_KEY = [12, 13, 'gate_proj']


def need(condition, message):
    if not condition:
        raise ValueError('JE01_ONE_COLUMN_JVP_' + message)


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def sha(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'FILE_MISSING_OR_LINK:' + str(path))
    return common.sha(path)


def verify_seal(value, expected=None, field='receipt_sha256'):
    need(isinstance(value, dict) and is_sha(value.get(field)), 'RECEIPT_MISSING')
    need(value[field] == common.canonical({k: v for k, v in value.items() if k != field}),
         'RECEIPT_CONTENT_DRIFT')
    need(expected is None or (is_sha(expected) and value[field] == expected), 'RECEIPT_PIN_DRIFT')
    return value


def sealed(path, expected=None, field='receipt_sha256'):
    sha(path)
    return verify_seal(json.loads(Path(path).read_bytes()), expected, field)


def number(value):
    need(type(value) in (int, float) and math.isfinite(value), 'NONFINITE_NUMBER')
    return value


def close(value, expected, label):
    need(math.isclose(number(value), number(expected), rel_tol=1e-10, abs_tol=1e-12),
         'NUMERIC_REPLAY:' + label)


def flatten(rows):
    return [value for row in rows for value in row]


def mean(values):
    return math.fsum(values) / len(values)


def require_root(path):
    path = Path(path)
    need(path.is_absolute() and path.parent == WORK and path.name.startswith(PREFIX)
         and path.name != PREFIX and path.resolve() == path and not path.is_symlink(),
         'SEPARATE_D50_RESEARCH_ROOT_REQUIRED')
    return path


def output_path(root, path):
    root = require_root(root)
    path = Path(path)
    need(path.is_absolute() and path != root and path.is_relative_to(root)
         and path.resolve() == path and not path.is_symlink(), 'OUTPUT_ESCAPE_OR_LINK')
    return path


def pinned_module(path, expected, name):
    need(is_sha(expected) and sha(path) == expected, 'MODULE_SOURCE_PIN:' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, 'MODULE_LOADER')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_helpers():
    tool_dir = Path(__file__).resolve().parent
    for name, expected in HELPER_PINS.items():
        need(sha(tool_dir / name) == expected, 'HELPER_SOURCE_PIN:' + name)
    need(Path(common.__file__).resolve() == tool_dir / 'je01_role_directional_native_runner.py'
         and Path(fixed.__file__).resolve() == tool_dir / 'je01_role_directional_fixed_native.py',
         'IMPORTED_HELPER_PATH_DRIFT')


def validate_smoke_contract(receipt, expected_adapter):
    need(receipt.get('schema') == SMOKE_SCHEMA and receipt.get('status') == 'PASS',
         'FINAL_GUARDED_V2_SMOKE_REQUIRED')
    need(receipt.get('proof_scope') == 'operator-smoke-only'
         and receipt.get('model_loaded') is False and receipt.get('installed_mlx_modified') is False
         and receipt.get('full_model_fisher_proven') is False, 'SMOKE_SCOPE_OVERCLAIM')
    sources = receipt.get('source_sha256')
    need(isinstance(sources, dict) and set(sources) == SMOKE_NAMES
         and all(is_sha(value) for value in sources.values()) and is_sha(expected_adapter)
         and sources[ADAPTER_NAME] == expected_adapter, 'FINAL_ADAPTER_SOURCE_PIN_REQUIRED')
    verified = receipt.get('verified', {})
    errors = verified.get('executed_transform_refusal_controls', {})
    need(set(errors) == set(GUARDS) and all(isinstance(errors[key], str) and text in errors[key]
                                           for key, text in GUARDS.items()),
         'ALL_FOUR_EXECUTED_TRANSFORM_GUARDS_REQUIRED')
    cases = verified.get('cases', [])
    need(isinstance(cases, list) and len(cases) == 8
         and {row.get('name') for row in cases} == SMOKE_CASES
         and all(row.get('native_primal_exact') is True and row.get('jvp_primal_exact') is True
                 for row in cases), 'ALL_EIGHT_OPERATOR_CASES_REQUIRED')
    pairs = verified.get('native_primal_sha_pairs', [])
    need(len(pairs) == 8 and {row.get('case') for row in pairs} == SMOKE_CASES
         and all(is_sha(row.get('native_sha256')) and
                 row['native_sha256'] == row.get('wrapped_sha256') == row.get('jvp_primal_sha256')
                 for row in pairs), 'SMOKE_EXACT_PRIMAL_SHA_PAIRS_REQUIRED')
    need(receipt.get('runtime', {}).get('mlx_version') == '0.32.1'
         and receipt.get('compile', {}).get('returncode') == 0
         and receipt.get('run', {}).get('returncode') == 0, 'SMOKE_NATIVE_RUNTIME_OR_EXIT')


def verify_adapter_gate(args):
    need(is_sha(args.expected_smoke_receipt) and is_sha(args.expected_adapter_sha),
         'EXTERNAL_FINAL_SMOKE_AND_ADAPTER_PINS_REQUIRED')
    receipt_path = Path(args.smoke_receipt)
    receipt = sealed(receipt_path, args.expected_smoke_receipt, 'canonical_sha256')
    validate_smoke_contract(receipt, args.expected_adapter_sha)
    smoke_root = receipt_path.parent
    need(smoke_root.is_absolute() and smoke_root.resolve() == smoke_root
         and receipt.get('output_root') == str(smoke_root), 'SMOKE_ORIGINAL_ROOT_REQUIRED')
    source = smoke_root / 'source'
    for name in SMOKE_NAMES:
        need(sha(source / name) == receipt['source_sha256'][name], 'SMOKE_SOURCE_FILE_DRIFT:' + name)
    need(sha(smoke_root / 'je01_mlx_native_jvp_smoke') == receipt.get('executable_sha256')
         and sha(smoke_root / 'run.stdout.log') == receipt.get('native_report_sha256'),
         'SMOKE_EXECUTABLE_OR_REPORT_DRIFT')
    validator = pinned_module(source / SMOKE_VALIDATOR_NAME, receipt['source_sha256'][SMOKE_VALIDATOR_NAME],
                              'je01_one_column_final_guarded_smoke_validator')
    regenerated = validator.validate_smoke_report(json.loads((smoke_root / 'run.stdout.log').read_bytes()),
                                                  smoke_root / 'arrays')
    need(regenerated == receipt['verified'], 'SMOKE_RAW_ARRAY_REPLAY_DRIFT')
    return receipt


def compare_derivatives(actual_rows, reference_rows):
    need(isinstance(actual_rows, list) and isinstance(reference_rows, list)
         and len(actual_rows) == len(reference_rows) > 0
         and all(isinstance(a, list) and isinstance(b, list) and len(a) == len(b) > 0
                 for a, b in zip(actual_rows, reference_rows)), 'DERIVATIVE_ARRAY_ALIGNMENT')
    actual, reference = flatten(actual_rows), flatten(reference_rows)
    for value in actual + reference:
        number(value)
    differences = [a - b for a, b in zip(actual, reference)]
    reference_rms = math.sqrt(mean([v * v for v in reference]))
    rms = math.sqrt(mean([v * v for v in differences]))
    threshold = 1e-4 + .05 * reference_rms
    norm = math.sqrt(math.fsum(v * v for v in actual) * math.fsum(v * v for v in reference))
    return {'tokens': len(actual), 'mean_jvp': mean(actual), 'mean_fixed_fine_fd': mean(reference),
            'rms_disagreement': rms, 'max_absolute_disagreement': max(map(abs, differences)),
            'rms_fixed_fine_fd': reference_rms, 'rms_acceptance_threshold': threshold,
            'absolute_tolerance': 1e-4, 'relative_tolerance': .05, 'passed': rms <= threshold,
            'cosine_similarity': math.fsum(a * b for a, b in zip(actual, reference)) / norm if norm else None,
            'jvp_observed_label_one_direction_square_mean': mean([v * v for v in actual]),
            'fixed_fine_observed_label_one_direction_square_mean': mean([v * v for v in reference]),
            'interpretation': 'ONE_FIXED_HARD_BRANCH_COMPUTATIONAL_DIRECTION_ON_SEEN_TRAIN16',
            'full48_empirical_fisher': None, 'allocation_selected': False}


def read_jvp(raw, metadata, original_nll):
    starts, targets = common.read_meta(metadata)
    need(raw.startswith(JVP_MAGIC) and len(raw) == len(JVP_MAGIC) + 48 * sum(map(len, targets)),
         'GRADIENT_MAGIC_OR_EXACT_BYTE_COUNT')
    need(len(targets) == len(original_nll) == 16 and sum(map(len, targets)) == 656
         and all(len(a) == len(b) for a, b in zip(original_nll, targets)), 'PILOT_GRADIENT_SHAPE')
    offset, gradients, target_logits, logits_rms = len(JVP_MAGIC), [], [], []
    for case, ids in enumerate(targets):
        g, t, r = [], [], []
        for index, target in enumerate(ids):
            row, token, gold, position, nll, dnll, dtarget, drms = struct.unpack_from('<4I4d', raw, offset)
            offset += 48
            need((row, token, gold, position) == (case, index, target, starts[case] + index - 1),
                 'GRADIENT_TARGET_OR_POSITION_ALIGNMENT')
            need(all(math.isfinite(v) for v in (nll, dnll, dtarget, drms)) and nll >= 0 and drms >= 0,
                 'GRADIENT_NONFINITE_OR_INVALID_VALUE')
            need(struct.pack('<d', nll) == struct.pack('<d', original_nll[case][index]),
                 'GRADIENT_NLL_NOT_EXACT_GOLD_DOUBLE')
            g.append(dnll); t.append(dtarget); r.append(drms)
        gradients.append(g); target_logits.append(t); logits_rms.append(r)
    return {'target_token_ids': targets, 'per_token_dnll': gradients,
            'per_token_dtargetlogit': target_logits, 'per_token_dlogits_rms': logits_rms}


def validate_completion(stderr, *, jvp):
    need('FATAL:' not in stderr
         and 'RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16' in stderr,
         'NATIVE_COMPLETION_OR_FATAL')
    need((common.MARKER in stderr) is jvp
         and (('JE01_FIXED_BASELINE_ROUTES_ENABLED records=' + str(fixed.ROUTE_RECORDS)) in stderr) is jvp,
         'OFF_OR_JVP_RUNTIME_MODE_MARKERS')
    matches = re.findall(r'^JE01_ONE_COLUMN_JVP_PREFIX_COMPLETE tokens=(\d+) cases=(\d+) layers=(\d+) gold=(\d+)\s*$',
                         stderr, flags=re.MULTILINE)
    if not jvp:
        need(JVP_MARKER not in stderr, 'OFF_MUST_NOT_RUN_JVP')
        return None
    need(len(matches) == 1 and stderr.count(JVP_MARKER) == 1
         and tuple(map(int, matches[0])) == tuple(COMPLETION.values()), 'FULL_PREFIX_COUNTERS_REQUIRED')
    return dict(COMPLETION)


def fixed_reference(path, dataset=None):
    need(sha(path) == FIXED_AUDIT_FILE_SHA, 'FIXED_REFERENCE_EXTERNAL_FILE_SHA')
    audit = sealed(path, FIXED_AUDIT_SHA)
    need(audit.get('schema') == FIXED_AUDIT_SCHEMA
         and audit.get('status') == 'PASS_INDEPENDENT_RAW_FIXED_BRANCH_REPLAY'
         and audit.get('diagnostic_status') == 'FIXED_BRANCH_CONVERGED'
         and audit.get('scope') == 'PREVIOUSLY_SEEN_TRAIN16_DIAGNOSTIC_ONLY', 'FIXED_REFERENCE_NOT_CONVERGED')
    need(all(audit.get(key) is False for key in ('allocation_selected', 'production_modified',
                                               'independent_validation128_accessed', 'independent_ood128_accessed'))
         and audit.get('full48_empirical_fisher') is None, 'FIXED_REFERENCE_SCOPE')
    data = verify_seal(audit['dataset_receipt'], fixed.DATASET_RECEIPT)
    if dataset is not None:
        need(data == dataset, 'FIXED_REFERENCE_DATASET_DRIFT')
    targets = data['target_token_ids']
    need(data['case_ids'] == list(range(16)) and sum(map(len, targets)) == 656, 'FIXED_REFERENCE_GOLD_SHAPE')
    identity = audit['source_identity']
    need(all(identity.get(key) == value for key, value in fixed.EXPECTED_ORDINARY_IDENTITY.items()
             if key != 'native_executable_sha256')
         and identity.get('native_executable_sha256') == FIXED_NATIVE_SHA
         and identity.get('ordinary_off_receipt_sha256') == fixed.OFF_RECEIPT
         and identity.get('baseline_routes_sha256') == fixed.ROUTE_SHA, 'FIXED_REFERENCE_SOURCE_IDENTITY')
    names = ['ordinary_off_attempt1', 'r00_Q4_a0_attempt1', 'r00_Q4_ap0.05_attempt1',
             'r00_Q4_am0.05_attempt1', 'r00_Q4_ap0.025_attempt1', 'r00_Q4_am0.025_attempt1']
    records = audit['native_run_receipts']
    need(set(records) == set(names), 'FIXED_REFERENCE_SIX_CELLS')
    for name, alpha in zip(names, (0, 0, .05, -.05, .025, -.025)):
        run = verify_seal(records[name])
        control = name == names[0]
        need(run.get('schema') == fixed.SCHEMA and run.get('status') == 'SUCCEEDED_EXIT0'
             and run.get('native_exit_code') == 0 and run.get('source_identity') == identity
             and run.get('scope') == 'TRAIN16_DIAGNOSTIC' and run.get('case_ids') == list(range(16))
             and run.get('target_token_ids') == targets and run.get('alpha') == alpha
             and run.get('role_key') == (None if control else ROLE_KEY)
             and run.get('tier') == (None if control else 'Q4')
             and run.get('derivative_kind') == 'FINITE_DIFFERENCE_FIXED_HARD_BRANCH_NATIVE_ROLE_DIRECTION'
             and run.get('fixed_route_replay_enabled') is (not control)
             and run.get('router_scores_and_selected_weights_dynamic') is True
             and run.get('full_prompt_and_KV_recomputed') is True
             and run.get('independent_validation_ood_accessed') is False
             and run.get('production_modified') is False, 'FIXED_REFERENCE_RUN_CONTRACT')
    def derivative(plus, minus, denominator):
        return [[(a - b) / denominator for a, b in zip(pa, ma)]
                for pa, ma in zip(records[plus]['per_token_nll'], records[minus]['per_token_nll'])]
    fine = derivative(names[4], names[5], .05)
    coarse = derivative(names[2], names[3], .1)
    convergence = compare_derivatives(coarse, fine)
    statistics = audit['statistics']
    need(convergence['passed'] is True and statistics['convergence']['passed'] is True
         and flatten(fine) == statistics['per_token_fine_derivative'], 'FIXED_FINE_REPLAY_OR_CONVERGENCE')
    close(convergence['rms_disagreement'], statistics['convergence']['rms_disagreement'], 'FIXED_RMS')
    close(convergence['rms_acceptance_threshold'], statistics['convergence']['rms_acceptance_threshold'], 'FIXED_THRESHOLD')
    close(mean(flatten(fine)), statistics['mean_fine_directional_nll_derivative'], 'FIXED_MEAN')
    return audit, fine


def validate_request(role=0, tier='Q4', alpha=0, attempt=1):
    need(type(role) is int and role == 0 and tier == 'Q4' and type(alpha) in (int, float) and alpha == 0
         and type(attempt) is int and 1 <= attempt <= 100, 'ONLY_L12_E13_GATE_Q4_ALPHA_ZERO')


def environment(folder, manifest, metadata, library, *, jvp):
    env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin',
           'TMPDIR': str(folder / 'tmp'), 'DYLD_LIBRARY_PATH': str(library),
           'QWEN_MOE_BASE': str(common.MODEL), 'QWEN_MOE_GPU_CBATCH_ONLINE': '1',
           'QWEN_MOE_CB_SLOTS': '1', 'QWEN_MOE_CB_REQS': '16', 'QWEN_MOE_CB_PREFILL_BUDGET': '1',
           'QWEN_MOE_CB_PROMPT_MANIFEST': str(manifest), 'QWEN_MOE_GPU_SORT_THRESHOLD': '64',
           'QWEN_MOE_GPU_CBATCH_BUDGET': '0', 'JE01_ORIGINAL_GOLD_META': str(metadata),
           'JE01_ORIGINAL_GOLD_OUTPUT': str(folder / 'gold_native.bin'),
           'JE01_DIRECTION_ROUTE_OUTPUT': str(folder / 'routes.bin')}
    if os.environ.get('HOME'):
        env['HOME'] = os.environ['HOME']
    if jvp:
        env.update({'JE01_REAL_Q35_BUNDLE': str(common.BANK / 'JE01_ROLE_DIRECTION_BANK.bin'),
                    'JE01_REAL_Q35_TSV': str(common.BANK / 'JE01_ROLE_DIRECTION_BANK.tsv'),
                    'JE01_DIRECTION_ROLE': '0', 'JE01_DIRECTION_TIER': 'Q4', 'JE01_DIRECTION_ALPHA': '0',
                    'JE01_DIRECTION_BASELINE_ROUTES': str(fixed.OFF_FOLDER / 'routes.bin'),
                    'JE01_NATIVE_JVP_MODE': JVP_MODE,
                    'JE01_NATIVE_JVP_GOLD_OUTPUT': str(folder / 'jvp_gold.bin')})
    return env


def memory_stop_reason(elapsed, free_percent, rss_kib, timeout):
    need(type(timeout) is int and 1 <= timeout <= 900, 'TIMEOUT_MUST_BE_AT_MOST900')
    need(type(free_percent) is int and 0 <= free_percent <= 100, 'MEMORY_SENSOR_INVALID')
    need(type(rss_kib) is int and rss_kib >= 0, 'RSS_SENSOR_INVALID')
    if elapsed > timeout:
        return 'TIMEOUT'
    if free_percent < 35:
        return 'MEMORY_FREE_BELOW35'
    if rss_kib > 18 * 1024 * 1024:
        return 'RSS_OVER18GIB'
    return None


def terminate_group(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=6)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=6)


class NativeExecutionError(RuntimeError):
    def __init__(self, message, execution):
        super().__init__(message)
        self.execution = execution


def bounded_native(executable, folder, env, timeout=900):
    memory_stop_reason(0, 100, 0, timeout)
    initial = common.memory_free()
    started, lowest, maximum_rss, stop = time.monotonic(), initial, 0, None
    def telemetry(code, reason, native_started):
        return {'native_exit_code': code, 'stop_reason': reason, 'native_started': native_started,
                'elapsed_seconds': time.monotonic() - started, 'min_memory_free_pct': lowest,
                'max_observed_rss_kib': maximum_rss, 'timeout_seconds': timeout,
                'rss_limit_gib': 18, 'memory_free_floor_pct': 35,
                'memory_measurement': 'PROCESS_RSS_AND_SYSTEM_FREE; GPU_ALLOCATION_NOT_SEPARATELY_MEASURED'}
    if initial < 55:
        raise NativeExecutionError('JE01_ONE_COLUMN_JVP_INITIAL_MEMORY_FREE_BELOW55',
                                   telemetry(None, 'INITIAL_MEMORY_FREE_BELOW55', False))
    with (folder / 'stdout.txt').open('xb') as out, (folder / 'stderr.txt').open('xb') as err:
        proc = subprocess.Popen([str(executable)], cwd=folder, env=env, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, start_new_session=True)
        try:
            while proc.poll() is None:
                free = common.memory_free()
                lowest = min(lowest, free)
                measured = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(proc.pid)],
                                           capture_output=True, text=True, timeout=5).stdout.strip()
                if proc.poll() is not None:
                    break
                need(measured.isdigit(), 'RSS_SENSOR_MISSING')
                rss = int(measured)
                maximum_rss = max(maximum_rss, rss)
                stop = memory_stop_reason(time.monotonic() - started, free, rss, timeout)
                if stop:
                    terminate_group(proc)
                    break
                time.sleep(.8)
        except BaseException as error:
            terminate_group(proc)
            raise NativeExecutionError(f'{type(error).__name__}: {error}',
                                       telemetry(proc.poll(), 'MONITOR_OR_INTERRUPTION_ERROR', True)) from error
        code = proc.wait(timeout=6)
    return telemetry(code, stop, True)


def verify_parent_source():
    receipt = sealed(FIXED_ROOT / 'NATIVE_BUILD_RECEIPT.json', FIXED_BUILD_RECEIPT)
    source = FIXED_ROOT / 'source'
    need(receipt.get('schema') == 'beglin-je01-native-fixed-route-build/1'
         and receipt.get('status') == 'JE01_FIXED_ROUTE_NATIVE_COMPILE_PASS'
         and receipt.get('binary_sha256') == FIXED_NATIVE_SHA
         and receipt.get('patched_cpp_sha256') == BASE_CPP_SHA
         and receipt.get('unchanged_c_sha256') == BASE_C_SHA
         and receipt.get('production_modified') is False, 'PARENT_FIXED_BUILD_CONTRACT')
    need(sha(FIXED_ROOT / 'bin/je01_role_directional_fixed_native') == FIXED_NATIVE_SHA
         and sha(source / 'mlx_moe.cpp') == BASE_CPP_SHA and sha(source / 'qwen_infer.c') == BASE_C_SHA,
         'PARENT_FIXED_SOURCE_OR_BINARY_HASH')
    need(source.is_dir() and not source.is_symlink() and not any(p.is_symlink() for p in source.rglob('*')),
         'PARENT_SOURCE_LINK_FORBIDDEN')
    return source, receipt


def checked_configuration(args):
    require_root(args.root)
    for key in ('expected_patch_sha', 'expected_patched_cpp_sha', 'expected_patched_c_sha', 'expected_adapter_sha'):
        need(is_sha(getattr(args, key, None)), 'EXACT_FINAL_SOURCE_PIN_REQUIRED:' + key)
    need(sha(args.patch) == args.expected_patch_sha, 'PATCH_SOURCE_HASH')
    return {'patch_source_sha256': args.expected_patch_sha,
            'patched_cpp_sha256': args.expected_patched_cpp_sha, 'patched_c_sha256': args.expected_patched_c_sha,
            'adapter_sha256': args.expected_adapter_sha, 'smoke_canonical_sha256': args.expected_smoke_receipt}


def runtime_matches_smoke(library, smoke):
    need(sha(library / 'libmlx.dylib') == smoke['runtime']['dylib_sha256'], 'SMOKE_MLX_LIBRARY_DRIFT')
    for name, expected in smoke['runtime']['header_sha256'].items():
        need(name in ('array.h', 'ops.h', 'transforms.h')
             and sha(library.parent / 'include/mlx' / name) == expected, 'SMOKE_MLX_HEADER_DRIFT')
    need(set(smoke['runtime']['header_sha256']) == {'array.h', 'ops.h', 'transforms.h'}, 'SMOKE_HEADER_COVERAGE')


def verified_original_runtime():
    verify_helpers()
    off, dataset, routes = fixed.verified_baseline()
    data, manifest, metadata, ordinary_exe, library, identity = common.preflight(fixed.DATASET)
    need(data == dataset and identity == fixed.EXPECTED_ORDINARY_IDENTITY
         and sha(ordinary_exe) == fixed.ORDINARY_EXE_SHA, 'ORIGINAL_RUNTIME_IDENTITY')
    bank = sealed(common.BANK / 'JE01_ROLE_DIRECTION_BANK_RECEIPT.json', fixed.BANK_RECEIPT)
    need(bank.get('physical_file_sha256') == fixed.BANK_SHA
         and bank.get('role_manifest_sha256') == fixed.BANK_TSV_SHA, 'BANK_RECEIPT_PIN')
    return {'off': off, 'dataset': dataset, 'routes': routes, 'manifest': manifest,
            'metadata': metadata, 'library': library, 'original_identity': identity}


def build(args):
    configuration = checked_configuration(args)
    verify_helpers()
    smoke = verify_adapter_gate(args)
    source, parent = verify_parent_source()
    root = require_root(args.root)
    need(not any((root / name).exists() or (root / name).is_symlink()
                 for name in ('source', 'build', 'bin', 'NATIVE_BUILD_RECEIPT.json', 'SOURCE_PATCH_RECEIPT.json')),
         'FRESH_BUILD_DIRECTORIES_REQUIRED')
    root.mkdir(mode=0o700, exist_ok=True)
    # The immutable bank checker resolves its ledger beside staged tools.
    # Preserve its exact bytes in the new root; do not redirect the checker.
    ledger_name = 'JE01_ACTUAL_CAL240_FFN_H2_SWIGLU_NATIVE_OUTPUT_DP_20261122.json'
    ledger_source = source / 'docs/research' / ledger_name
    sealed(ledger_source, '35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb')
    ledger_target = output_path(root, root / 'docs/research' / ledger_name)
    if ledger_target.exists():
        need(sha(ledger_target) == sha(ledger_source), 'STAGED_LEDGER_DRIFT')
    else:
        ledger_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ledger_source, ledger_target)
    original = verified_original_runtime()
    runtime_matches_smoke(original['library'], smoke)
    copied = output_path(root, root / 'source')
    shutil.copytree(source, copied, symlinks=True)
    patch = pinned_module(args.patch, args.expected_patch_sha, 'je01_one_column_jvp_source_patch')
    applied = patch.stage(copied)
    need(applied.get('schema') == PATCH_SCHEMA
         and applied.get('status') == 'SEPARATE_SCRATCH_SOURCE_PATCHED_NOT_EXECUTED'
         and applied.get('base_cpp_sha256') == BASE_CPP_SHA and applied.get('base_c_sha256') == BASE_C_SHA
         and applied.get('patched_cpp_sha256') == args.expected_patched_cpp_sha
         and applied.get('patched_c_sha256') == args.expected_patched_c_sha
         and applied.get('adapter_sha256') == args.expected_adapter_sha
         and applied.get('role_key') == ROLE_KEY and applied.get('tier') == 'Q4' and applied.get('alpha') == 0
         and applied.get('production_modified') is False, 'SOURCE_PATCH_CONTRACT')
    need(sha(copied / 'mlx_moe.cpp') == args.expected_patched_cpp_sha
         and sha(copied / 'qwen_infer.c') == args.expected_patched_c_sha
         and sha(copied / ADAPTER_NAME) == args.expected_adapter_sha, 'PATCHED_SOURCE_FILE_SHA')
    # Stage receipts may already be sealed; do not hash a checksum into itself.
    patch_receipt = common.write_seal(root / 'SOURCE_PATCH_RECEIPT.json',
                                     {k: v for k, v in applied.items() if k != 'receipt_sha256'})
    cmake_file = copied / 'CMakeLists.txt'
    previous = cmake_file.read_text()
    anchor = 'OUTPUT_NAME je01_role_directional_fixed_native'
    need(previous.count(anchor) == 1, 'CMAKE_UNIQUE_OUTPUT_NAME_REQUIRED')
    cmake_file.write_text(previous.replace(anchor, 'OUTPUT_NAME je01_native_one_column_jvp', 1))
    cmake = fixed.BC_BUILD / 'venv/bin/cmake'
    mlx_cmake = original['library'].parent / 'share/cmake'
    need(cmake.is_file() and mlx_cmake.is_dir(), 'PINNED_EXISTING_TOOLCHAIN_MISSING')
    commands = [[str(cmake), '-S', str(copied), '-B', str(root / 'build'), '-G', 'Ninja',
                 '-DCMAKE_BUILD_TYPE=Release', '-DMLX_CMAKE_DIR=' + str(mlx_cmake)],
                [str(cmake), '--build', str(root / 'build'), '--target', 'qwen_infer_gpu', '--parallel', '2']]
    for index, command in enumerate(commands):
        with (root / f'compile{index}.stdout.txt').open('xb') as out, (root / f'compile{index}.stderr.txt').open('xb') as err:
            subprocess.run(command, stdout=out, stderr=err, check=True, timeout=300)
    (root / 'bin').mkdir(mode=0o700)
    executable = root / 'bin/je01_native_one_column_jvp'
    shutil.copy2(root / 'build/je01_native_one_column_jvp', executable)
    executable.chmod(0o700)
    need(executable.stat().st_size > 100000, 'COMPILED_BINARY_TOO_SMALL')
    receipt = {'schema': BUILD_SCHEMA, 'status': 'ONE_COLUMN_JVP_NATIVE_COMPILE_PASS',
               'configuration': configuration, 'fixed_build_receipt_sha256': parent['receipt_sha256'],
               'fixed_native_sha256': FIXED_NATIVE_SHA, 'base_cpp_sha256': BASE_CPP_SHA, 'base_c_sha256': BASE_C_SHA,
               'source_patch_receipt_sha256': patch_receipt['receipt_sha256'],
               'binary_sha256': sha(executable), 'binary_bytes': executable.stat().st_size,
               'cmake_sha256': sha(cmake_file), 'mlx_library_sha256': sha(original['library'] / 'libmlx.dylib'),
               'runner_source_sha256': sha(Path(__file__)), 'helper_source_sha256': HELPER_PINS,
               'adapter_smoke_receipt_sha256': smoke['canonical_sha256'],
               'adapter_smoke_source_sha256': smoke['source_sha256'],
               'model_original_af4_sha256': common.AF_SHA, 'dataset_receipt_sha256': fixed.DATASET_RECEIPT,
               'bank_receipt_sha256': fixed.BANK_RECEIPT, 'scope': 'TRAIN16_DIAGNOSTIC',
               'full_prefix_jvp': 'NOT_EXECUTED', 'off_parity': 'NOT_RUN', 'full48_empirical_fisher': None,
               'allocation_selected': False, 'independent_validation_ood_accessed': False, 'production_modified': False}
    common.write_seal(root / 'NATIVE_BUILD_RECEIPT.json', receipt)
    print(json.dumps({'event': 'ONE_COLUMN_JVP_BUILD_DONE', 'receipt_sha256': receipt['receipt_sha256'],
                      'binary_sha256': receipt['binary_sha256']}, sort_keys=True), flush=True)
    return receipt


def preflight(args):
    configuration = checked_configuration(args)
    need(is_sha(args.expected_build_receipt), 'EXTERNAL_BUILD_RECEIPT_PIN_REQUIRED')
    smoke = verify_adapter_gate(args)
    original = verified_original_runtime()
    runtime_matches_smoke(original['library'], smoke)
    root = require_root(args.root)
    receipt = sealed(root / 'NATIVE_BUILD_RECEIPT.json', args.expected_build_receipt)
    executable = root / 'bin/je01_native_one_column_jvp'
    need(receipt.get('schema') == BUILD_SCHEMA and receipt.get('status') == 'ONE_COLUMN_JVP_NATIVE_COMPILE_PASS'
         and receipt.get('configuration') == configuration
         and receipt.get('fixed_build_receipt_sha256') == FIXED_BUILD_RECEIPT
         and receipt.get('adapter_smoke_receipt_sha256') == smoke['canonical_sha256']
         and receipt.get('runner_source_sha256') == sha(Path(__file__))
         and receipt.get('helper_source_sha256') == HELPER_PINS
         and receipt.get('production_modified') is False, 'NEW_BINARY_BUILD_CONTRACT')
    need(sha(executable) == receipt.get('binary_sha256')
         and sha(root / 'source/mlx_moe.cpp') == args.expected_patched_cpp_sha
         and sha(root / 'source/qwen_infer.c') == args.expected_patched_c_sha
         and sha(root / 'source' / ADAPTER_NAME) == args.expected_adapter_sha
         and sha(root / 'source/CMakeLists.txt') == receipt.get('cmake_sha256'), 'NEW_BINARY_OR_SOURCE_DRIFT')
    audit, reference = fixed_reference(args.fixed_audit, original['dataset'])
    identity = {**original['original_identity'], 'native_executable_sha256': receipt['binary_sha256'],
                'new_build_receipt_sha256': receipt['receipt_sha256'],
                'adapter_header_sha256': args.expected_adapter_sha,
                'adapter_smoke_receipt_sha256': smoke['canonical_sha256'],
                'baseline_route_sha256': fixed.ROUTE_SHA, 'original_off_receipt_sha256': fixed.OFF_RECEIPT,
                'fixed_reference_audit_sha256': FIXED_AUDIT_SHA, 'derivative_kind': DERIVATIVE_KIND}
    return {**original, 'build': receipt, 'identity': identity, 'executable': executable,
            'reference_audit': audit, 'reference_derivatives': reference}


def cell_folder(root, *, jvp, attempt):
    validate_request(attempt=attempt)
    label = 'jvp_r00_Q4_a0' if jvp else 'off'
    return output_path(root, Path(root) / 'runs/pilot16' / f'{label}_attempt{attempt}')


def verify_off_control(root, context, attempt):
    folder = cell_folder(root, jvp=False, attempt=attempt)
    receipt = sealed(folder / 'RUN_RECEIPT.json')
    need(receipt.get('schema') == RUN_SCHEMA and receipt.get('status') == 'SUCCEEDED_EXIT0'
         and receipt.get('mode') == 'OFF_CONTROL' and receipt.get('source_identity') == context['identity']
         and receipt.get('case_ids') == list(range(16)) and receipt.get('scope') == 'TRAIN16_DIAGNOSTIC'
         and receipt.get('native_jvp_enabled') is False and receipt.get('native_overlay_enabled') is False
         and receipt.get('fixed_route_replay_enabled') is False
         and receipt.get('full_prompt_and_KV_recomputed') is True
         and receipt.get('target_token_ids') == context['dataset']['target_token_ids']
         and receipt.get('runner_source_sha256') == sha(Path(__file__))
         and receipt.get('execution', {}).get('native_exit_code') == 0
         and receipt.get('execution', {}).get('stop_reason') is None
         and receipt.get('production_modified') is False and receipt.get('independent_validation_ood_accessed') is False,
         'NEW_BINARY_OFF_CONTROL_REQUIRED')
    need(sha(folder / 'gold_native.bin') == receipt.get('output_sha256') == common.BASE_SHA
         and sha(folder / 'routes.bin') == receipt.get('route_output_sha256') == fixed.ROUTE_SHA,
         'NEW_BINARY_OFF_RAW_PARITY')
    validate_completion((folder / 'stderr.txt').read_text(errors='replace'), jvp=False)
    return receipt


def run_cell(args, context, *, jvp):
    validate_request(attempt=args.attempt)
    root = require_root(args.root)
    folder = cell_folder(root, jvp=jvp, attempt=args.attempt)
    need(not folder.exists() and not folder.is_symlink(), 'CELL_ALREADY_ATTEMPTED_USE_FRESH_ATTEMPT')
    control = verify_off_control(root, context, args.attempt) if jvp else None
    folder.mkdir(parents=True, mode=0o700)
    (folder / 'tmp').mkdir(mode=0o700)
    env = environment(folder, context['manifest'], context['metadata'], context['library'], jvp=jvp)
    print(json.dumps({'event': 'ONE_COLUMN_JVP_CELL_START', 'mode': 'JVP_FIXED_REPLAY' if jvp else 'OFF_CONTROL',
                      'attempt': args.attempt, 'root': str(root)}, sort_keys=True), flush=True)
    execution = None
    try:
        execution = bounded_native(context['executable'], folder, env, args.timeout_seconds)
        need(execution['native_exit_code'] == 0 and execution['stop_reason'] is None, 'NATIVE_PROCESS_FAILED_OR_STOPPED')
        completion = validate_completion((folder / 'stderr.txt').read_text(errors='replace'), jvp=jvp)
        gold, route = folder / 'gold_native.bin', folder / 'routes.bin'
        need(sha(gold) == common.BASE_SHA and sha(route) == fixed.ROUTE_SHA
             and route.stat().st_size == fixed.ROUTE_BYTES, 'NEW_BINARY_GOLD_OR_ROUTE_BYTE_PARITY')
        values, targets = common.read_gold(gold.read_bytes(), context['metadata'].read_bytes())
        need(values == context['off']['per_token_nll'] and targets == context['dataset']['target_token_ids']
             and common.read_routes(route.read_bytes()) == context['routes'], 'NATIVE_RAW_TARGET_OR_ROUTE_REPLAY')
        gradients, comparison = None, None
        if jvp:
            gradients = read_jvp((folder / 'jvp_gold.bin').read_bytes(), context['metadata'].read_bytes(), values)
            comparison = compare_derivatives(gradients['per_token_dnll'], context['reference_derivatives'])
        else:
            need(not (folder / 'jvp_gold.bin').exists(), 'OFF_GRADIENT_FILE_FORBIDDEN')
        receipt = {'schema': RUN_SCHEMA, 'status': 'SUCCEEDED_EXIT0', 'scope': 'TRAIN16_DIAGNOSTIC',
                   'mode': 'JVP_FIXED_REPLAY' if jvp else 'OFF_CONTROL', 'source_identity': context['identity'],
                   'case_ids': list(range(16)), 'target_token_ids': targets, 'per_token_nll': values,
                   'mean_gold_nll': mean(flatten(values)), 'output_sha256': sha(gold),
                   'route_output_sha256': sha(route), 'routed_token_layer_events': fixed.ROUTE_RECORDS,
                   'native_jvp_enabled': jvp, 'native_overlay_enabled': jvp, 'fixed_route_replay_enabled': jvp,
                   'role_index': 0 if jvp else None, 'role_key': ROLE_KEY if jvp else None,
                   'tier': 'Q4' if jvp else None, 'alpha': 0,
                   'new_off_receipt_sha256': control['receipt_sha256'] if control else None,
                   'original_off_receipt_sha256': fixed.OFF_RECEIPT,
                   'full_prompt_and_KV_recomputed': True, 'router_scores_and_selected_weights_dynamic': True,
                   'full_prefix_tangent_carried': jvp, 'native_prefix_completion': completion,
                   'derivative_kind': DERIVATIVE_KIND if jvp else 'OFF_CONTROL_NO_DERIVATIVE',
                   'gradient_output_sha256': sha(folder / 'jvp_gold.bin') if jvp else None,
                   'gradient_output_bytes': (folder / 'jvp_gold.bin').stat().st_size if jvp else None,
                   'gradients': gradients, 'comparison_with_fixed_fine_fd': comparison,
                   'diagnostic_status': ('ONE_COLUMN_JVP_AGREES_WITH_FIXED_FD' if comparison['passed'] else
                                         'ONE_COLUMN_JVP_NUMERICAL_REJECTED') if jvp else 'EXACT_ORIGINAL_OFF_PARITY',
                   'execution': execution, 'runner_source_sha256': sha(Path(__file__)),
                   'full48_empirical_fisher': None, 'allocation_selected': False,
                   'independent_validation_ood_accessed': False, 'production_modified': False}
        common.write_seal(folder / 'RUN_RECEIPT.json', receipt)
    except BaseException as error:
        if isinstance(error, NativeExecutionError):
            execution = error.execution
        common.write_seal(folder / 'FAILED_RUN_RECEIPT.json', {
            'schema': RUN_SCHEMA, 'status': 'FAILED_NO_VERIFIED_DIRECTION', 'scope': 'TRAIN16_DIAGNOSTIC',
            'source_identity': context['identity'], 'native_jvp_requested': jvp, 'execution': execution,
            'error': f'{type(error).__name__}: {error}', 'full48_empirical_fisher': None,
            'allocation_selected': False, 'independent_validation_ood_accessed': False, 'production_modified': False})
        raise
    print(json.dumps({'event': 'ONE_COLUMN_JVP_CELL_DONE', 'mode': receipt['mode'],
                      'diagnostic_status': receipt['diagnostic_status'], 'receipt_sha256': receipt['receipt_sha256']},
                     sort_keys=True), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--build', action='store_true')
    action.add_argument('--diagnostic', action='store_true')
    action.add_argument('--run-off', action='store_true')
    action.add_argument('--run-jvp', action='store_true')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke-receipt', type=Path, required=True)
    parser.add_argument('--expected-smoke-receipt', required=True)
    parser.add_argument('--expected-adapter-sha', required=True)
    parser.add_argument('--patch', type=Path, default=Path(__file__).with_name('je01_native_one_column_jvp_patch.py'))
    parser.add_argument('--expected-patch-sha', required=True)
    parser.add_argument('--expected-patched-cpp-sha', required=True)
    parser.add_argument('--expected-patched-c-sha', required=True)
    parser.add_argument('--expected-build-receipt')
    parser.add_argument('--fixed-audit', type=Path)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--timeout-seconds', type=int, default=900)
    args = parser.parse_args()
    memory_stop_reason(0, 100, 0, args.timeout_seconds)
    validate_request(attempt=args.attempt)
    if args.build:
        build(args)
        return
    need(args.fixed_audit is not None, 'FIXED_AUDIT_REQUIRED_FOR_RUNTIME')
    context = preflight(args)
    if args.run_off:
        run_cell(args, context, jvp=False)
    elif args.run_jvp:
        run_cell(args, context, jvp=True)
    else:
        control = run_cell(args, context, jvp=False)
        measured = run_cell(args, context, jvp=True)
        comparison = measured['comparison_with_fixed_fine_fd']
        report = {'schema': REPORT_SCHEMA, 'status': measured['diagnostic_status'], 'scope': 'TRAIN16_DIAGNOSTIC',
                  'source_identity': context['identity'], 'role_key': ROLE_KEY, 'tier': 'Q4', 'alpha': 0,
                  'off_receipt_sha256': control['receipt_sha256'], 'jvp_receipt_sha256': measured['receipt_sha256'],
                  'fixed_reference_audit_sha256': FIXED_AUDIT_SHA, 'case_ids': list(range(16)), 'gold_tokens': 656,
                  'comparison': comparison, 'full48_empirical_fisher': None, 'allocation_selected': False,
                  'independent_validation_ood_accessed': False, 'production_modified': False}
        name = f'JE01_ONE_COLUMN_JVP_COMPARISON_attempt{args.attempt}.json'
        common.write_seal(output_path(args.root, args.root / name), report)
        print(json.dumps({'event': 'ONE_COLUMN_JVP_DIAGNOSTIC_DONE', 'status': report['status'],
                          'receipt_sha256': report['receipt_sha256'], 'comparison': comparison}, sort_keys=True), flush=True)


if __name__ == '__main__':
    # Let the same cleanup/failure-receipt path handle a parent's TERM signal.
    def interrupted(signum, frame):
        raise InterruptedError(f'JE01_ONE_COLUMN_JVP_SIGNAL:{signum}')
    signal.signal(signal.SIGTERM, interrupted)
    main()
