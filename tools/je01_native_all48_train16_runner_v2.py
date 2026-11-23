#!/usr/bin/env python3
"""Recovery v2 of the preregistered 48-column JVP study on seen TRAIN16.

Reuse the exact proven role-JVP executable; do not rebuild or alter its source.
The evidence design has one inherited, verified OFF and 78 new native cells:
48 alpha-zero JVPs and five fixed FD cells at each of six new checkpoints. Three earlier Q4 FD checkpoints are
independently pinned and reused. Every non-OFF cell replays the original hard routes.
No holdout, allocation search, training, or production mutation is permitted.

plan (or --plan-only): validate source/runtime evidence and seal the whole plan.
run: consume the next cells in order, with verified append-only resume.
analyze: reopen all raw outputs and publish g/Gram only after all gates pass.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import time
from types import SimpleNamespace


TOOL_DIR = Path(__file__).resolve().parent
V1_RUNNER_SHA = '7451c867ebb810cff9fe854322e00937bf8b1807c2d0475006d6703be32dd4de'
PARENT_RUNNER_SHA = '9be8372fa3858577011eb5ab1d4ad654f746e151a95012aaf23e515fd8a7eb90'
IMPORT_PINS = {
    'je01_native_all48_train16_runner.py': V1_RUNNER_SHA,
    'je01_native_role_jvp_runner.py': PARENT_RUNNER_SHA,
    'je01_native_one_column_jvp_runner.py': 'ffd632d79d948119ad261e2720ebbcd7c5076af21070aeaf60bf0ca8e7d79bd6',
    'je01_role_directional_native_runner.py': '5607c0afef3bd4b1a61bcab6622267acc6b6137fc65ba0e79873e92e31d7e9e6',
    'je01_role_directional_fixed_native.py': 'cffe2e1dd8298681f3a134caace742ddc9fea77209c58683cd8d2f48fe6284e7',
    'je01_role_directional_fixed_routes_patch.py': '233937b59293bec11300b66450b819dc48bb09b8d28eb6e98ca81aac7f511019',
    'je01_role_directional_bank.py': '045fd22d86e4ce18f967d66830a1f71b0c61ad6ab5c5889d953cb9a8d05bce33',
}


def need(value, message):
    if not value:
        raise ValueError('JE01_ALL48_' + message)


def verify_import_sources():
    for name, expected in IMPORT_PINS.items():
        path = TOOL_DIR / name
        need(path.is_file() and not path.is_symlink()
             and hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'HELPER_SOURCE_PIN:' + name)


verify_import_sources()
_spec = importlib.util.spec_from_file_location('je01_all48_pinned_three_role',
                                               TOOL_DIR / 'je01_native_role_jvp_runner.py')
need(_spec is not None and _spec.loader is not None, 'PARENT_IMPORT_SPEC')
parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parent)
legacy, common, fixed = parent.legacy, parent.common, parent.fixed
sha, sealed = legacy.sha, legacy.sealed
v1 = legacy.pinned_module(TOOL_DIR / 'je01_native_all48_train16_runner.py', V1_RUNNER_SHA,
                          'je01_all48_immutable_v1_recovery_parser')

WORK = Path('/Volumes/D50/TailnetCommander/work')
ROOT_PREFIX = 'je01_native_role_jvp_all48_train16_'
V1_ROOT = WORK / 'je01_native_all48_train16_20261123_v1'
V1_PLAN_PIN = '60fdced19fad43cbf2b73801dd68effbbd75c186f8a5f39de9274cad39453419'
V1_OFF_PIN = 'f044aa2e2ee1a22ab305f1854d2cf6c7280a5d72cded594859d786fcc5a0caa9'
V1_OFF_FILE_PIN = '98d68ca9a5984502bd3858bbb0523653814589abf1114ec5ad28eff39e6c6f6d'
V1_FAILED_FD_PIN = '51b3795007b69b783d94e22eae5f62fc96d8c73f8b377d3a1d18e5ee7e561129'
V1_FAILED_FD_FILE_PIN = '40daee88c55a53c478a17c34b584a7736c54b76fd06142cdd0656bcdc5e74579'
V1_WORKFLOW_PIN = '694f066501ddf1f1e31d94a746c56fc10239081e5c072311118c2c89c6e13b53'
V1_WORKFLOW_FILE_PIN = '1f32a3461859f8c90f485fd8e6fff95553cb872b9723bd59ba3c988f5f63cf0e'
V1_AUDIT_PIN = 'b21f0e037e27ce30c47bb5ec4ef8b17a44d09c3b38f18389e33ae2fb62df6d3f'
V1_AUDIT_FILE_PIN = '66fd0128598c2705ea735ea0876006b8effa2602aa25f455fec9e4e4e90a6e35'
V1_AUDITOR_SHA = '270fc991cd0e4378457fe80cfa4f7f924ab69d44d77656ecfb95bb11449730a9'
NATIVE_OUTPUT_PREFIX = '/Volumes/D50/TailnetCommander/work/je01_native_role_jvp_'
PARENT_ROOT = WORK / 'je01_native_role_jvp_followup_20261123_v1'
SMOKE_RECEIPT = WORK / 'je01_native_jvp_smoke_20261123_v4/JE01_NATIVE_JVP_SMOKE_RECEIPT.json'
FIXED_AUDIT = legacy.FIXED_ROOT / 'JE01_FIXED_ROUTE_GATE_INDEPENDENT_AUDIT.json'
BUILD_PIN = '2cbeb4dc1a7e1ab5642a4abb22395498c10dde43bee051eec3c303c7e53fa008'
BINARY_PIN = '110b1978d97bcd7c6bdfd6d42142569e56e9dacffa4ff462dc0786b3128e49f4'
CPP_PIN = '6055ca3b261075fe5fcf76e5829f0fb20b8ecdd52599248f4b9b1032cfbeb49f'
C_PIN = '5e2a6ea76c22c1ec0ff9f2fb6d093b7d3a471679adbe439ed1c4b898a56a52a6'
PARENT_PLAN_PIN = '12ae8c5eb7a3d6409eea2aa79337de9da9c56453bac33ec084c4e4a64aec8a14'
PARENT_REPORT_PIN = '1448de4fdfb34082c7bd6976676d8a5404f078d649d730adbaf331fb233857df'
PARENT_REPORT_FILE_PIN = '22f929b8e92eff027358fa7c2104044cda7967ab46503199793c42618b5ae2b5'
PARENT_AUDIT_PIN = '5ce09188e65d5cfcc0f956447b2918fe9c80bf81f14e8486ba97864f7137b87b'
PARENT_AUDIT_FILE_PIN = '9d82c2d727d4d20eb799e9a5b7625474311c954aaf7694063793bfaf09f18fcc'
LIBMLX_PIN = 'afb1939edc96dc10e452132e4e665276fe0c48bf8cdc1a69286feb8a14de1da0'
LEDGER_NAME = 'JE01_ACTUAL_CAL240_FFN_H2_SWIGLU_NATIVE_OUTPUT_DP_20261122.json'
LEDGER_PIN = '35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb'
SCOPE = 'PREVIOUSLY_SEEN_TRAIN16_FIXED_HARD_ROUTE_48_DIRECTION_DIAGNOSTIC_ONLY'
PLAN_SCHEMA = 'beglin-je01-native-all48-train16-plan/2'
CELL_SCHEMA = 'beglin-je01-native-all48-train16-cell/2'
CHECK_SCHEMA = 'beglin-je01-native-all48-train16-checkpoint/2'
MATRIX_SCHEMA = 'beglin-je01-native-all48-train16-raw-matrix/2'
ANALYSIS_SCHEMA = 'beglin-je01-native-all48-train16-analysis/2'
CHUNK_SCHEMA = 'beglin-je01-native-all48-train16-chunk/2'
HOLD_SCHEMA = 'beglin-je01-native-all48-train16-numerical-hold/2'
TIERS = ('Q4', 'Q5')
ALPHAS = ('0', '+0.05', '-0.05', '+0.025', '-0.025')
NEW_CHECKPOINTS = ((0, 'Q5'), (1, 'Q5'), (2, 'Q5'), (3, 'Q4'), (13, 'Q5'), (23, 'Q4'))
PRIOR_CHECKPOINTS = ((0, 'Q4'), (1, 'Q4'), (2, 'Q4'))
ALL_CHECKPOINTS = PRIOR_CHECKPOINTS + NEW_CHECKPOINTS
COMPLETE_STATUS = 'ALL48_NATIVE_COMPLETE_CHECKPOINTS_AND_PSD_PASS'
NO_ACTION = {'full_parameter_fisher': None, 'full48_empirical_fisher': None,
             'Hessian': 'NOT_MEASURED', 'train240_allocation': 'NOT_RUN',
             'allocation_selected': False, 'independent_validation_ood_accessed': False,
             'production_modified': False}


def require_root(value):
    path = Path(value)
    need(path.is_absolute() and path.parent == WORK and path.name.startswith(ROOT_PREFIX)
         and path.name != ROOT_PREFIX and path.resolve() == path and not path.is_symlink(),
         'NEW_SEPARATE_D50_ALL48_ROOT_REQUIRED')
    return path


def output_path(root, path):
    root, path = require_root(root), Path(path)
    need(path.is_absolute() and path != root and path.is_relative_to(root)
         and path.resolve() == path and not path.is_symlink(), 'OUTPUT_ESCAPE_OR_SYMLINK')
    return path


def seal_receipt(path, payload, *, resume=False):
    """Append-only publication under the root lock; partial writes fail sealing.

    Do not use hard links: an earlier EOE environment hit a hard-link probe
    hang. Exclusive creation plus fsync preserves earlier artifacts. A killed
    write is deliberately an integrity block, never a valid partial receipt.
    """
    path = Path(path)
    need('receipt_sha256' not in payload and path.resolve() == path and not path.is_symlink(),
         'SEAL_PATH_OR_SELF_CHECKSUM')
    if path.exists():
        need(resume, 'APPEND_ONLY_RECEIPT_EXISTS')
        previous = sealed(path)
        need({k: v for k, v in previous.items() if k != 'receipt_sha256'} == payload,
             'RESUME_RECOMPUTATION_DRIFT')
        return previous
    result = {**payload, 'receipt_sha256': common.canonical(payload)}
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return result


@contextmanager
def exclusive_root(root):
    path = output_path(root, Path(root) / 'EXECUTION_LOCK')
    with path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('JE01_ALL48_CONCURRENT_EXECUTION_FORBIDDEN') from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def coordinate_order():
    return [{'column_index': 2 * role + offset, 'role_index': role,
             'role_key': common.ROLE_ORDER[role], 'tier': tier}
            for role in range(24) for offset, tier in enumerate(TIERS)]


def column_index(role, tier):
    need(type(role) is int and 0 <= role < 24 and tier in TIERS, 'ROLE_OR_TIER')
    return 2 * role + TIERS.index(tier)


def execution_sequence():
    cells = [{'kind': 'OFF', 'role_index': None, 'tier': None, 'alpha_literal': '0', 'column_index': None}]

    def add_direction(role, tier):
        if (role, tier) in NEW_CHECKPOINTS:
            cells.extend({'kind': 'FD', 'role_index': role, 'tier': tier, 'alpha_literal': alpha,
                          'column_index': column_index(role, tier)} for alpha in ALPHAS)
        cells.append({'kind': 'JVP', 'role_index': role, 'tier': tier, 'alpha_literal': '0',
                      'column_index': column_index(role, tier)})

    for role in (0, 1, 2):
        add_direction(role, 'Q5')
    for coordinate in coordinate_order():
        pair = (coordinate['role_index'], coordinate['tier'])
        if pair not in NEW_CHECKPOINTS[:3]:
            add_direction(*pair)
    return [{'cell_index': index, **cell} for index, cell in enumerate(cells)]


def validate_cell(cell):
    index = cell.get('cell_index') if isinstance(cell, dict) else None
    need(type(index) is int and 0 <= index < 79 and cell == execution_sequence()[index],
         'CELL_OUTSIDE_SEALED_79_SEQUENCE')
    need(type(cell.get('kind')) is str and type(cell.get('alpha_literal')) is str
         and (cell.get('role_index') is None and cell.get('column_index') is None and cell.get('tier') is None
              if cell['kind'] == 'OFF' else
              type(cell.get('role_index')) is int and type(cell.get('column_index')) is int
              and type(cell.get('tier')) is str), 'CELL_SCALAR_TYPES')


def cell_label(cell):
    validate_cell(cell)
    if cell['kind'] == 'OFF':
        label = 'off'
    else:
        alpha = cell['alpha_literal'].replace('+', 'p').replace('-', 'm')
        label = f"{cell['kind'].lower()}_r{cell['role_index']:02d}_{cell['tier']}_a{alpha}"
    return f"cell_{cell['cell_index']:03d}_{label}"


def cell_folder(root, cell, attempt):
    need(type(attempt) is int and 1 <= attempt <= 100, 'CELL_ATTEMPT_RANGE')
    return output_path(root, Path(root) / 'runs/train16' / f'{cell_label(cell)}_attempt{attempt}')


def parent_arguments():
    return SimpleNamespace(root=PARENT_ROOT, patch=TOOL_DIR / 'je01_native_role_jvp_patch.py',
                           expected_patch_sha=parent.PATCH_SHA, expected_adapter_sha=parent.ADAPTER_SHA,
                           expected_smoke_receipt=parent.SMOKE_SHA, smoke_receipt=SMOKE_RECEIPT,
                           expected_patched_cpp_sha=CPP_PIN, expected_patched_c_sha=C_PIN,
                           expected_build_receipt=BUILD_PIN, fixed_audit=FIXED_AUDIT)


def prepare_ledger(root):
    """The unmodified bank verifier resolves its immutable ledger beside tools."""
    source = PARENT_ROOT / 'docs/research' / LEDGER_NAME
    sealed(source, LEDGER_PIN)
    target = output_path(root, Path(root) / 'docs/research' / LEDGER_NAME)
    if target.exists():
        need(sha(target) == sha(source), 'LEDGER_BYTES_DRIFT')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(source.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())


def validate_native_output_root(native_c, root):
    """Check the pinned C_OPEN guard itself, not just Python directory policy."""
    anchor = 'const char *je01_jvp_mode=getenv("JE01_NATIVE_JVP_MODE");'
    need(type(native_c) is str and native_c.count(anchor) == 1, 'NATIVE_C_OPEN_ANCHOR')
    start = native_c.index(anchor)
    finish = native_c.find('int fd=open(je01_jvp_path,', start)
    need(finish > start, 'NATIVE_C_OPEN_END')
    block = native_c[start:finish]
    prefixes = re.findall(r'const char prefix\[\]="([^"\n]+)";', block)
    need(prefixes == [NATIVE_OUTPUT_PREFIX]
         and 'strncmp(je01_jvp_path,prefix,sizeof(prefix)-1)' in block
         and 'strcmp(je01_jvp_mode,"ROLE_DIRECTION_FIXED_Q4Q5")' in block,
         'NATIVE_C_OUTPUT_GUARD_CHANGED')
    root = Path(root)
    need(root.is_absolute() and str(root).startswith(prefixes[0]), 'NATIVE_C_OPEN_REJECTS_OUTPUT_ROOT')
    return {'source_c_sha256': C_PIN, 'required_absolute_prefix': prefixes[0],
            'output_root': str(root), 'validated_before_any_native_execution': True}


def recovery_contract():
    return {'reason': 'SOURCE_ENFORCED_JVP_OUTPUT_ROOT_MISMATCH_FOUND_BEFORE_FIRST_JVP',
            'change': 'FRESH_NATIVE_COMPATIBLE_ROOT_ONLY_NO_NATIVE_BINARY_OR_MATH_CHANGE',
            'v1_root': str(V1_ROOT), 'v1_plan_receipt_sha256': V1_PLAN_PIN,
            'v1_runner_source_sha256': V1_RUNNER_SHA,
            'v1_off_receipt_sha256': V1_OFF_PIN, 'v1_off_file_sha256': V1_OFF_FILE_PIN,
            'v1_interrupted_fd_receipt_sha256': V1_FAILED_FD_PIN,
            'v1_interrupted_fd_file_sha256': V1_FAILED_FD_FILE_PIN,
            'v1_workflow_receipt_sha256': V1_WORKFLOW_PIN,
            'v1_workflow_file_sha256': V1_WORKFLOW_FILE_PIN,
            'v1_independent_confirmation_receipt_sha256': V1_AUDIT_PIN,
            'v1_independent_confirmation_file_sha256': V1_AUDIT_FILE_PIN,
            'v1_independent_auditor_source_sha256': V1_AUDITOR_SHA,
            'inherited_successful_cell_indices': [0], 'v1_jvp_columns_measured': 0,
            'off_native_reexecution_forbidden': True,
            'interrupted_fd_was_not_a_completed_numerical_result': True,
            'v1_evidence_is_preserved': True}


def verify_v1_recovery(context):
    prior_plan = sealed(V1_ROOT / 'PLAN_RECEIPT.json', V1_PLAN_PIN)
    need(prior_plan.get('runner_source_sha256') == V1_RUNNER_SHA
         and prior_plan.get('root') == str(V1_ROOT), 'V1_ORIGINAL_PLAN_BINDING')
    old_context = {**context, 'plan': prior_plan,
                   'identity': {**context['identity'], 'plan_receipt_sha256': V1_PLAN_PIN}}
    off_cell, failed_cell = v1.execution_sequence()[:2]
    off_folder = v1.cell_folder(V1_ROOT, off_cell, 1)
    failed_folder = v1.cell_folder(V1_ROOT, failed_cell, 1)
    need({path.name for path in (V1_ROOT / 'runs/train16').iterdir()} ==
         {off_folder.name, failed_folder.name}, 'V1_MUST_HAVE_ONLY_ONE_OFF_AND_ONE_INTERRUPTED_FD')
    need(sha(off_folder / 'RUN_RECEIPT.json') == V1_OFF_FILE_PIN, 'V1_OFF_EXTERNAL_FILE_PIN')
    off, failures = v1.inspect_attempts(V1_ROOT, old_context, off_cell, None)
    need(off is not None and off['receipt_sha256'] == V1_OFF_PIN and failures == [], 'V1_OFF_RAW_REPLAY')
    failed_path = failed_folder / 'FAILED_RUN_RECEIPT.json'
    need(sha(failed_path) == V1_FAILED_FD_FILE_PIN, 'V1_INTERRUPTED_FD_EXTERNAL_FILE_PIN')
    failed = sealed(failed_path, V1_FAILED_FD_PIN)
    completed, failed_pins = v1.inspect_attempts(V1_ROOT, old_context, failed_cell, V1_OFF_PIN)
    need(completed is None and failed_pins == [V1_FAILED_FD_PIN]
         and failed.get('execution', {}).get('native_exit_code') == -15
         and failed.get('execution', {}).get('stop_reason') == 'MONITOR_OR_INTERRUPTION_ERROR',
         'V1_FD_INTERRUPTION_NOT_COMPLETED_NUMERICAL_RESULT')
    workflow_path = V1_ROOT / 'WORKFLOW_RECEIPT_0001.json'
    need(sha(workflow_path) == V1_WORKFLOW_FILE_PIN, 'V1_WORKFLOW_EXTERNAL_FILE_PIN')
    workflow = sealed(workflow_path, V1_WORKFLOW_PIN)
    need(workflow.get('status') == 'FAILED_OR_INTERRUPTED_NO_COMPLETION_CLAIM'
         and workflow.get('root') == str(V1_ROOT)
         and workflow.get('plan_receipt_sha256') == V1_PLAN_PIN
         and workflow.get('runner_source_sha256') == V1_RUNNER_SHA
         and workflow.get('analysis_receipt_sha256') is None
         and workflow.get('independent_audit_receipt_sha256') is None,
         'V1_TERMINAL_WORKFLOW_BINDING')
    audit_path = V1_ROOT / 'JE01_ALL48_TRAIN16_INDEPENDENT_AUDIT_V1_INTERRUPTED.json'
    need(sha(audit_path) == V1_AUDIT_FILE_PIN, 'V1_INDEPENDENT_CONFIRMATION_FILE_PIN')
    audit = sealed(audit_path, V1_AUDIT_PIN)
    need(audit.get('status') == 'INDEPENDENT_AUDIT_CONFIRMED_PARTIAL_NO_PASS'
         and audit.get('evidence_integrity') == 'PASS' and audit.get('experiment_numerical_verdict') == 'NOT_COMPLETE'
         and audit.get('plan_receipt_sha256') == V1_PLAN_PIN and audit.get('runner_source_sha256') == V1_RUNNER_SHA
         and audit.get('auditor_source_sha256') == V1_AUDITOR_SHA
         and audit.get('completed_native_cell_count') == 1 and audit.get('native_jvp_columns') == 0,
         'V1_INDEPENDENT_OFF_CONFIRMATION')
    for name, pin in [('gold_native.bin', off['output_sha256']), ('routes.bin', off['route_output_sha256']),
                      ('stdout.txt', off['stdout_sha256']), ('stderr.txt', off['stderr_sha256'])]:
        claim = audit.get('verified_files', {}).get(str(off_folder / name))
        need(isinstance(claim, dict) and claim.get('sha256') == pin
             and claim.get('size_bytes') == (off_folder / name).stat().st_size, 'V1_AUDITED_OFF_RAW_FILE:' + name)
    return off


def execution_provenance(cell):
    inherited = cell['kind'] == 'OFF'
    return {'native_executed_in_this_plan': not inherited,
            'executed_runner_source_sha256': V1_RUNNER_SHA if inherited else sha(Path(__file__)),
            'reused_from_v1_plan_receipt_sha256': V1_PLAN_PIN if inherited else None,
            'original_run_receipt_sha256': V1_OFF_PIN if inherited else None,
            'original_run_file_sha256': V1_OFF_FILE_PIN if inherited else None,
            'original_run_folder': str(v1.cell_folder(V1_ROOT, v1.execution_sequence()[0], 1)) if inherited else None,
            'v1_independent_confirmation_receipt_sha256': V1_AUDIT_PIN if inherited else None,
            'recovered_v1_interrupted_cell_receipt_sha256': V1_FAILED_FD_PIN if cell['cell_index'] == 1 else None}


def inherit_off(args, context):
    """Bridge the original OFF proof into this plan without running its binary."""
    cell = execution_sequence()[0]
    folder = cell_folder(args.root, cell, 1)
    if (folder / 'RUN_RECEIPT.json').exists():
        saved, failures = inspect_attempts(args.root, context, cell, None)
        need(saved is not None and not failures, 'INHERITED_OFF_RESUME')
        return saved
    need(not (folder / 'FAILED_RUN_RECEIPT.json').exists(), 'INHERITED_OFF_CANNOT_HAVE_NATIVE_FAILURE_ATTEMPT')
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    original = context['v1_off']
    source = original['_folder']
    copies = {name: source / name for name in ('gold_native.bin', 'routes.bin', 'stdout.txt', 'stderr.txt')}
    copies['INHERITED_ORIGINAL_RUN_RECEIPT.json'] = source / 'RUN_RECEIPT.json'
    for name, origin in copies.items():
        target = output_path(args.root, folder / name)
        if target.exists():
            need(sha(target) == sha(origin), 'PARTIAL_INHERITED_COPY_DRIFT:' + name)
        else:
            with target.open('xb') as stream:
                stream.write(origin.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
    raw = parse_raw_cell(folder, context, cell)
    receipt = seal_receipt(folder / 'RUN_RECEIPT.json', {
        **cell_contract(context, cell, 1, None, []), 'execution': original['execution'], **raw})
    print(json.dumps({'event': 'ALL48_V2_OFF_INHERITED_NO_NATIVE_EXECUTION',
                      'receipt_sha256': receipt['receipt_sha256'], 'original_run_receipt_sha256': V1_OFF_PIN,
                      'native_executed_in_this_plan': False}, sort_keys=True), flush=True)
    return {**receipt, '_folder': folder}


def preflight(args):
    root = require_root(args.root)
    need(root.is_dir() and not root.is_symlink(), 'PLAN_ROOT_MISSING')
    verify_import_sources()
    native_c = PARENT_ROOT / 'source/qwen_infer.c'
    need(sha(native_c) == C_PIN, 'PINNED_NATIVE_C_REQUIRED_BEFORE_PATH_CHECK')
    output_contract = validate_native_output_root(native_c.read_text(), root)
    context = parent.preflight(parent_arguments())
    need(context['build']['receipt_sha256'] == BUILD_PIN
         and context['build']['binary_sha256'] == BINARY_PIN
         and sha(context['library'] / 'libmlx.dylib') == LIBMLX_PIN
         and context['build']['mlx_library_sha256'] == LIBMLX_PIN, 'PROVEN_BINARY_OR_MLX_PIN')
    patch = sealed(PARENT_ROOT / 'SOURCE_PATCH_RECEIPT.json',
                   context['build']['source_patch_receipt_sha256'])
    need(patch.get('supported_role_indices') == list(range(24))
         and patch.get('supported_tiers') == list(TIERS) and patch.get('alpha') == 0
         and patch.get('one_direction_per_run') is True and patch.get('runtime_mode') == parent.MODE,
         'NATIVE_ALL48_SELECTOR_CONTRACT')
    parent_plan = sealed(PARENT_ROOT / 'PLAN_RECEIPT.json', PARENT_PLAN_PIN)
    report_path = PARENT_ROOT / 'JE01_THREE_ROLE_JVP_DIAGNOSTIC_attempt1.json'
    audit_path = PARENT_ROOT / 'JE01_THREE_ROLE_INDEPENDENT_AUDIT.json'
    need(sha(report_path) == PARENT_REPORT_FILE_PIN and sha(audit_path) == PARENT_AUDIT_FILE_PIN,
         'PARENT_EVIDENCE_EXTERNAL_FILE_PINS')
    report, audit = sealed(report_path, PARENT_REPORT_PIN), sealed(audit_path, PARENT_AUDIT_PIN)
    need(report.get('status') == 'THREE_ROLE_NATIVE_JVP_FIXED_FD_PASS'
         and audit.get('status') == 'INDEPENDENT_AUDIT_CONFIRMED_THREE_ROLE_PASS'
         and audit.get('evidence_integrity') == audit.get('experiment_numerical_verdict') == 'PASS'
         and audit.get('source_report_receipt_sha256') == PARENT_REPORT_PIN
         and report.get('plan_receipt_sha256') == audit.get('plan_receipt_sha256') == PARENT_PLAN_PIN
         and audit.get('external_plan_pin') == PARENT_PLAN_PIN
         and report.get('roles') == audit.get('roles') and audit.get('native_cells_raw_replayed') == 14
         and audit.get('native_inference_rerun') is False, 'PARENT_INDEPENDENT_PASS_CONTRACT')
    need(len(audit.get('verified_files', {})) == 132, 'PARENT_RAW_FILE_COVERAGE')
    for path, pin in audit['verified_files'].items():
        path = Path(path)
        need(path.is_absolute() and path.is_relative_to(WORK) and path.resolve() == path
             and sha(path) == pin['sha256'] and path.stat().st_size == pin['size_bytes'],
             'PARENT_RAW_FILE_DRIFT:' + str(path))
    for evidence in (report, audit):
        need(evidence.get('scope') == parent.SCOPE and evidence.get('case_ids') == list(range(16))
             and evidence.get('gold_tokens') == 656
             and all(evidence.get(k) is False for k in ('allocation_selected', 'production_modified',
                                                       'independent_validation_ood_accessed')),
             'PARENT_SCOPE_OR_HOLDOUT_GUARD')
    need(context['dataset']['case_ids'] == list(range(16))
         and sum(map(len, context['dataset']['target_token_ids'])) == 656, 'TRAIN16_ONLY')
    identity = {**context['identity'],
                'derivative_definition': 'ALL48_NATIVE_ALPHA_ZERO_FIXED_HARD_ROUTE_FULL_PREFIX_DIRECTIONS',
                'parent_three_role_report_sha256': PARENT_REPORT_PIN,
                'parent_three_role_independent_audit_sha256': PARENT_AUDIT_PIN}
    result = {**context, 'identity': identity, 'parent_plan': parent_plan,
              'parent_report': report, 'parent_audit': audit, 'native_output_path_contract': output_contract}
    result['v1_off'] = verify_v1_recovery(result)
    return result


def plan_payload(args, context):
    estimates = {'OFF': 73.8836, 'JVP': 154.9563, 'FD': 71.4456}
    sequence = execution_sequence()
    return {'schema': PLAN_SCHEMA, 'scope': SCOPE, 'root': str(require_root(args.root)),
            'source_identity': context['identity'], 'runner_source_sha256': sha(Path(__file__)),
            'helper_source_sha256': IMPORT_PINS, 'patch_source_sha256': parent.PATCH_SHA,
            'proven_native_build_receipt_sha256': BUILD_PIN, 'native_binary_sha256': BINARY_PIN,
            'parent_root': str(PARENT_ROOT), 'parent_report_receipt_sha256': PARENT_REPORT_PIN,
            'parent_report_file_sha256': PARENT_REPORT_FILE_PIN,
            'parent_audit_receipt_sha256': PARENT_AUDIT_PIN, 'parent_audit_file_sha256': PARENT_AUDIT_FILE_PIN,
            'parent_plan_receipt_sha256': PARENT_PLAN_PIN,
            'runtime': {'library_path': str(context['library']),
                        'libmlx_path': str(context['library'] / 'libmlx.dylib'), 'libmlx_sha256': LIBMLX_PIN,
                        'manifest_path': str(context['manifest']), 'manifest_sha256': sha(context['manifest']),
                        'metadata_path': str(context['metadata']), 'metadata_sha256': sha(context['metadata']),
                        'native_executable_path': str(context['executable']),
                        'fixed_fd_executable_path': str(context['fd_executable']),
                        'fixed_fd_executable_sha256': legacy.FIXED_NATIVE_SHA,
                        'baseline_routes_path': str(fixed.OFF_FOLDER / 'routes.bin'),
                        'baseline_routes_sha256': fixed.ROUTE_SHA, 'baseline_gold_sha256': common.BASE_SHA,
                        'bank_path': str(common.BANK / 'JE01_ROLE_DIRECTION_BANK.bin'),
                        'bank_receipt_sha256': fixed.BANK_RECEIPT, 'bank_sha256': fixed.BANK_SHA},
            'coordinate_order': coordinate_order(), 'column_count': 48,
            'coordinate_definition': 'ALTERNATIVE_MINUS_ORIGINAL_NATIVE_WEIGHT_ACTION_AT_ALPHA_ZERO',
            'evidence_execution_sequence': sequence, 'evidence_cell_count': 79,
            'native_execution_sequence': sequence[1:], 'native_cell_count': 78, 'inherited_native_cell_count': 1,
            'native_output_path_contract': context['native_output_path_contract'],
            'recovery': recovery_contract(),
            'new_fd_checkpoints': [{'role_index': r, 'tier': t, 'column_index': column_index(r, t)}
                                   for r, t in NEW_CHECKPOINTS],
            'reused_fd_checkpoints': [{'role_index': r, 'tier': t, 'column_index': column_index(r, t)}
                                      for r, t in PRIOR_CHECKPOINTS],
            'new_fd_cell_count': 30, 'fd_verified_coordinate_count': 9,
            'fd_coverage_interpretation': 'NINE_PREDECLARED_CHECKPOINTS_NOT_ALL_48_INDIVIDUAL_FD_VALIDATIONS',
            'alpha_grid': list(ALPHAS), 'jvp_alpha': 0,
            'case_ids': list(range(16)), 'gold_tokens': 656,
            'target_token_ids': context['dataset']['target_token_ids'],
            'full_prefix_completion_counters': dict(legacy.COMPLETION),
            'fd_convergence_rule': {'coarse_h': .05, 'fine_h': .025, 'rms_absolute_tolerance': 1e-4,
                                    'rms_relative_tolerance': .05, 'reference': 'ALL_656_FINE_FD_TOKEN_DERIVATIVES'},
            'jvp_agreement_rule': {'rms_absolute_tolerance': 1e-4, 'rms_relative_tolerance': .05,
                                   'reference': 'ALL_656_FINE_FD_TOKEN_DERIVATIVES'},
            'prior_q4_jvp_requires_exact_gradient_output_sha': True,
            'every_zero_and_jvp_primal_requires_original_gold_sha': common.BASE_SHA,
            'every_fixed_route_output_requires_baseline_sha': fixed.ROUTE_SHA,
            'natural_routing_stability': 'NOT_ESTABLISHED_BY_FORCED_ROUTE_REPLAY',
            'matrix_rules': {'raw_shape': [656, 48], 'g_definition': 'MEAN_TOKEN_DNLL',
                             'score_definition': 'NEGATIVE_DNLL',
                             'gram_definition': 'MEAN_TOKEN_DNLL_OUTER_PRODUCT',
                             'centering': 'NONE', 'sentence_partition': '16_FROZEN_CASE_IDS_IN_METADATA_ORDER',
                             'psd_min_eigenvalue_lower_bound': '-1e-8 * max(1, trace(Gram))',
                             'eigen_residual_tolerance_times_max_1_abs_trace': 1e-12,
                             'publish_only_if_all_48_raw_and_9_checkpoints_and_psd_pass': True},
            'execution_policy': {'sequential': True, 'default_max_new_cells': 12,
                                 'maximum_max_new_cells': 79, 'native_cell_timeout_seconds_max': 900,
                                 'initial_memory_free_percent_min': 55, 'memory_free_percent_floor': 35,
                                 'rss_limit_gib': 18, 'retry': 'SEALED_EXECUTION_OR_INTEGRITY_FAILURE_ONLY',
                                 'successful_cells_never_reexecuted': True,
                                 'numerical_failure': 'SEAL_HOLD_AND_STOP_FURTHER_NATIVE_CELLS',
                                 'plan_or_threshold_changes_after_first_cell': False},
            'timing_estimate': {'source': 'MEASURED_PRIOR_THREE_ROLE_NATIVE_CELL_RECEIPTS',
                                'seconds_per_cell': estimates,
                                'observed_up_jvp_seconds': 151.2950,
                                'estimated_native_seconds': math.fsum(estimates[c['kind']] for c in sequence if c['kind'] != 'OFF'),
                                'inherited_off_elapsed_seconds': context['v1_off']['execution']['elapsed_seconds'],
                                'excludes_preflight_hashing_and_audit': True,
                                'estimate_is_not_a_runtime_guarantee': True},
            'result_adaptive_selection': False, 'independent_audit_required_before_train240': True,
            **NO_ACTION}


def establish_plan(args, context, *, creating=False):
    root = require_root(args.root)
    path = output_path(root, root / 'PLAN_RECEIPT.json')
    payload = plan_payload(args, context)
    if creating:
        need(path.exists() or not (root / 'runs').exists(), 'PLAN_MUST_PRECEDE_NATIVE_CELLS')
        result = seal_receipt(path, payload, resume=True)
    else:
        need(legacy.is_sha(args.expected_plan_receipt), 'EXTERNAL_PLAN_PIN_REQUIRED')
        result = sealed(path, args.expected_plan_receipt)
        need({k: v for k, v in result.items() if k != 'receipt_sha256'} == payload,
             'PLAN_SOURCE_SCOPE_OR_RUNTIME_DRIFT')
    return {**context, 'plan': result,
            'identity': {**context['identity'], 'plan_receipt_sha256': result['receipt_sha256']}}


def plan(args):
    root = require_root(args.root)
    root.mkdir(mode=0o700, exist_ok=True)
    with exclusive_root(root):
        prepare_ledger(root)
        context = establish_plan(args, preflight(args), creating=True)
    result = context['plan']
    print(json.dumps({'event': 'ALL48_PLAN_SEALED_NO_INFERENCE', 'receipt_sha256': result['receipt_sha256'],
                      'native_cell_count': 78, 'evidence_cell_count': 79, 'inherited_native_cell_count': 1,
                      'columns': 48, 'fd_checkpoints': 9,
                      'estimated_native_seconds': result['timing_estimate']['estimated_native_seconds']},
                     sort_keys=True), flush=True)
    return result


def cell_environment(folder, context, cell):
    validate_cell(cell)
    env = legacy.environment(folder, context['manifest'], context['metadata'], context['library'], jvp=False)
    if cell['kind'] != 'OFF':
        env.update({'JE01_REAL_Q35_BUNDLE': str(common.BANK / 'JE01_ROLE_DIRECTION_BANK.bin'),
                    'JE01_REAL_Q35_TSV': str(common.BANK / 'JE01_ROLE_DIRECTION_BANK.tsv'),
                    'JE01_DIRECTION_ROLE': str(cell['role_index']), 'JE01_DIRECTION_TIER': cell['tier'],
                    'JE01_DIRECTION_ALPHA': cell['alpha_literal'],
                    'JE01_DIRECTION_BASELINE_ROUTES': str(fixed.OFF_FOLDER / 'routes.bin')})
    if cell['kind'] == 'JVP':
        env.update({'JE01_NATIVE_JVP_MODE': parent.MODE,
                    'JE01_NATIVE_JVP_GOLD_OUTPUT': str(folder / 'jvp_gold.bin')})
    return env


def validate_native_configuration(stderr, cell):
    matches = re.findall(r'^JE01_ROLE_DIRECTION_CONFIG role=(\d+) layer=(\d+) expert=(\d+) tier=(Q4|Q5) alpha=(\S+)\s*$',
                         stderr, flags=re.MULTILINE)
    if cell['kind'] == 'OFF':
        need(not matches and 'JE01_ROLE_DIRECTION_CONFIG' not in stderr, 'OFF_ROLE_CONFIG_FORBIDDEN')
    else:
        layer, expert, _ = common.ROLE_ORDER[cell['role_index']]
        expected = (str(cell['role_index']), str(layer), str(expert), cell['tier'], cell['alpha_literal'])
        need(matches and set(matches) == {expected}
             and stderr.count('JE01_ROLE_DIRECTION_CONFIG') == len(matches),
             'NATIVE_EXECUTED_ROLE_TIER_ALPHA_MISMATCH')


def parse_raw_cell(folder, context, cell):
    validate_cell(cell)
    raw = parent.parse_raw_cell(folder, context, cell['kind'], cell['role_index'], cell['alpha_literal'])
    validate_native_configuration((folder / 'stderr.txt').read_text(errors='strict'), cell)
    if cell['kind'] == 'OFF':
        original_path = folder / 'INHERITED_ORIGINAL_RUN_RECEIPT.json'
        need(sha(original_path) == V1_OFF_FILE_PIN, 'INHERITED_ORIGINAL_RECEIPT_BYTES')
        original = sealed(original_path, V1_OFF_PIN)
        need(all(original.get(key) == value for key, value in raw.items()), 'INHERITED_OFF_RAW_PARITY')
    pair = (cell['role_index'], cell['tier'])
    if cell['kind'] == 'JVP' and pair in PRIOR_CHECKPOINTS:
        prior = context['parent_audit']['roles'][cell['role_index']]
        need(raw['gradient_output_sha256'] == prior['jvp_output_sha256'], 'PRIOR_Q4_JVP_EXACT_BYTES_CHANGED')
        expected = context['parent_audit']['raw_directional_derivatives'][str(cell['role_index'])]['native_jvp']
        need(legacy.flatten(raw['gradients']['per_token_dnll']) == expected, 'PRIOR_Q4_RAW_DERIVATIVE_CHANGED')
    return raw


def cell_contract(context, cell, attempt, control_pin, prior_failure_pins):
    role = cell['role_index']
    return {'schema': CELL_SCHEMA, 'status': 'SUCCEEDED_EXIT0', 'scope': SCOPE,
            **cell, 'role_key': common.ROLE_ORDER[role] if role is not None else None,
            'alpha': common.ALPHAS[cell['alpha_literal']], 'attempt': attempt,
            'source_identity': context['identity'], 'plan_receipt_sha256': context['plan']['receipt_sha256'],
            'runner_source_sha256': sha(Path(__file__)), 'case_ids': list(range(16)),
            **execution_provenance(cell),
            'executed_binary_sha256': legacy.FIXED_NATIVE_SHA if cell['kind'] == 'FD' else BINARY_PIN,
            'new_off_receipt_sha256': control_pin, 'prior_failure_receipt_sha256': prior_failure_pins,
            'native_jvp_enabled': cell['kind'] == 'JVP', 'native_overlay_enabled': cell['kind'] != 'OFF',
            'fixed_route_replay_enabled': cell['kind'] != 'OFF',
            'full_prompt_and_KV_recomputed': True, 'router_scores_and_selected_weights_dynamic': True,
            'full_prefix_tangent_carried': cell['kind'] == 'JVP',
            'route_equality_interpretation': 'REPLAY_CONFORMANCE_NOT_NATURAL_ROUTING_STABILITY', **NO_ACTION}


def validate_saved_contract(receipt, context, cell, attempt, control_pin, failures):
    expected = cell_contract(context, cell, attempt, control_pin, failures)
    need(all(receipt.get(key) == value for key, value in expected.items()), 'RESUME_CELL_CONTRACT')
    need(type(receipt.get('cell_index')) is int and type(receipt.get('attempt')) is int
         and (receipt.get('role_index') is None if cell['kind'] == 'OFF' else
              type(receipt.get('role_index')) is int and type(receipt.get('column_index')) is int)
         and type(receipt.get('alpha')) in (int, float)
         and type(receipt.get('native_executed_in_this_plan')) is bool
         and all(receipt.get(k) is False for k in ('allocation_selected', 'production_modified',
                                                  'independent_validation_ood_accessed')),
         'RESUME_CELL_SCALAR_TYPES_OR_SCOPE_FLAGS')
    execution = receipt.get('execution', {})
    need(type(execution.get('native_exit_code')) is int and execution.get('native_exit_code') == 0
         and execution.get('stop_reason') is None
         and execution.get('native_started') is True
         and type(execution.get('elapsed_seconds')) in (int, float)
         and math.isfinite(execution['elapsed_seconds']) and execution['elapsed_seconds'] >= 0
         and type(execution.get('timeout_seconds')) is int and 1 <= execution['timeout_seconds'] <= 900
         and execution.get('rss_limit_gib') == 18 and execution.get('memory_free_floor_pct') == 35,
         'RESUME_EXECUTION_TELEMETRY')


def inspect_attempts(root, context, cell, control_pin):
    """A successful cell is immutable; retries require a preceding sealed failure."""
    base = Path(root) / 'runs/train16'
    paths = sorted(base.glob(cell_label(cell) + '_attempt*')) if base.exists() else []
    attempts = {}
    for folder in paths:
        match = re.fullmatch(re.escape(cell_label(cell)) + r'_attempt(\d+)', folder.name)
        need(match is not None, 'UNRECOGNIZED_ATTEMPT_PATH')
        number = int(match.group(1))
        need(folder == cell_folder(root, cell, number) and folder.is_dir() and not folder.is_symlink(),
             'CELL_ATTEMPT_PATH')
        attempts[number] = folder
    need(sorted(attempts) == list(range(1, len(attempts) + 1)), 'CELL_ATTEMPT_GAP')
    failures, successful = [], None
    for attempt, folder in sorted(attempts.items()):
        good, bad = folder / 'RUN_RECEIPT.json', folder / 'FAILED_RUN_RECEIPT.json'
        need(good.exists() != bad.exists(), 'INCOMPLETE_OR_AMBIGUOUS_CELL_REQUIRES_REVIEW')
        need(cell['kind'] != 'OFF' or (attempt == 1 and good.exists()),
             'OFF_IS_ONE_INHERITED_RECORD_NEVER_A_NATIVE_ATTEMPT')
        need(successful is None, 'SUCCESSFUL_CELL_WAS_REEXECUTED')
        if good.exists():
            receipt = sealed(good)
            validate_saved_contract(receipt, context, cell, attempt, control_pin, failures)
            actual = parse_raw_cell(folder, context, cell)
            need(all(receipt.get(key) == value for key, value in actual.items()), 'RESUME_RAW_OUTPUT_DRIFT')
            successful = {**receipt, '_folder': folder}
        else:
            failed = sealed(bad)
            need(failed.get('schema') == CELL_SCHEMA and failed.get('status') == 'EXECUTION_OR_INTEGRITY_FAILED'
                 and failed.get('scope') == SCOPE and failed.get('cell') == cell
                 and failed.get('attempt') == attempt and failed.get('source_identity') == context['identity']
                 and failed.get('plan_receipt_sha256') == context['plan']['receipt_sha256']
                 and failed.get('runner_source_sha256') == sha(Path(__file__))
                 and failed.get('prior_failure_receipt_sha256') == failures, 'FAILED_RETRY_EVIDENCE_CONTRACT')
            failures.append(failed['receipt_sha256'])
    return successful, failures


def verify_inventory(root):
    directory = Path(root) / 'runs/train16'
    if not directory.exists():
        return
    need(directory.is_dir() and directory.resolve() == directory and not directory.is_symlink(), 'RUNS_DIRECTORY')
    labels = {cell_label(cell) for cell in execution_sequence()}
    for path in directory.iterdir():
        match = re.fullmatch(r'(.+)_attempt([1-9]\d*)', path.name)
        need(match is not None and match.group(1) in labels and 1 <= int(match.group(2)) <= 100,
             'UNPLANNED_NATIVE_CELL_OR_ATTEMPT')


def run_cell(args, context, cell, control_pin, failures):
    need(cell['kind'] != 'OFF', 'V2_OFF_MUST_BE_INHERITED_NEVER_NATIVE_REEXECUTED')
    attempt = len(failures) + 1
    need(attempt <= args.max_cell_attempt, 'EXPLICIT_HIGHER_MAX_CELL_ATTEMPT_REQUIRED_AFTER_SEALED_FAILURE')
    folder = cell_folder(args.root, cell, attempt)
    need(not folder.exists() and not folder.is_symlink(), 'NEW_CELL_FOLDER_REQUIRED')
    folder.mkdir(mode=0o700, parents=True)
    (folder / 'tmp').mkdir(mode=0o700)
    executable = context['fd_executable'] if cell['kind'] == 'FD' else context['executable']
    expected = legacy.FIXED_NATIVE_SHA if cell['kind'] == 'FD' else BINARY_PIN
    execution = None
    print(json.dumps({'event': 'ALL48_CELL_START', **cell, 'attempt': attempt}, sort_keys=True), flush=True)
    try:
        need(sha(executable) == expected and sha(context['library'] / 'libmlx.dylib') == LIBMLX_PIN,
             'PRE_CELL_NATIVE_BINARY_OR_MLX_DRIFT')
        execution = legacy.bounded_native(executable, folder, cell_environment(folder, context, cell),
                                          args.timeout_seconds)
        need(execution['native_exit_code'] == 0 and execution['stop_reason'] is None, 'NATIVE_EXECUTION_FAILED')
        raw = parse_raw_cell(folder, context, cell)
        receipt = seal_receipt(folder / 'RUN_RECEIPT.json', {
            **cell_contract(context, cell, attempt, control_pin, failures), 'execution': execution, **raw})
    except BaseException as error:
        if isinstance(error, legacy.NativeExecutionError):
            execution = error.execution
        seal_receipt(folder / 'FAILED_RUN_RECEIPT.json', {
            'schema': CELL_SCHEMA, 'status': 'EXECUTION_OR_INTEGRITY_FAILED', 'scope': SCOPE,
            'cell': cell, 'attempt': attempt, 'source_identity': context['identity'],
            'plan_receipt_sha256': context['plan']['receipt_sha256'], 'runner_source_sha256': sha(Path(__file__)),
            'prior_failure_receipt_sha256': failures, 'execution': execution,
            'error': f'{type(error).__name__}: {error}', **NO_ACTION})
        raise
    print(json.dumps({'event': 'ALL48_CELL_DONE', **cell, 'attempt': attempt,
                      'receipt_sha256': receipt['receipt_sha256'],
                      'elapsed_seconds': execution['elapsed_seconds']}, sort_keys=True), flush=True)
    return {**receipt, '_folder': folder}


def reshape_by_targets(values, targets):
    need(isinstance(values, list) and len(values) == sum(map(len, targets)) == 656
         and all(type(v) in (int, float) and math.isfinite(v) for v in values), 'DERIVATIVE_COLUMN_SHAPE')
    rows, offset = [], 0
    for target in targets:
        rows.append(values[offset:offset + len(target)])
        offset += len(target)
    return rows


def checkpoint_payload(context, measured, records):
    role, tier = measured['role_index'], measured['tier']
    need(measured['kind'] == 'JVP' and (role, tier) in ALL_CHECKPOINTS, 'CHECKPOINT_COORDINATE')
    targets = context['dataset']['target_token_ids']
    if (role, tier) in PRIOR_CHECKPOINTS:
        previous = context['parent_audit']['raw_directional_derivatives'][str(role)]
        coarse = reshape_by_targets(previous['fixed_coarse_fd'], targets)
        fine = reshape_by_targets(previous['fixed_fine_fd'], targets)
        references = {'parent_audit_receipt_sha256': PARENT_AUDIT_PIN,
                      'parent_fd_reference_receipt_sha256': context['parent_audit']['roles'][role]['fd_reference_receipt_sha256']}
        kind = 'REUSED_INDEPENDENTLY_AUDITED_Q4_FD'
    else:
        fd = {r['alpha_literal']: r for r in records.values()
              if r['kind'] == 'FD' and (r['role_index'], r['tier']) == (role, tier)}
        need(set(fd) == set(ALPHAS), 'ALL_FIVE_CHECKPOINT_FD_CELLS_REQUIRED')
        coarse = parent.derivative_rows(fd['+0.05']['per_token_nll'], fd['-0.05']['per_token_nll'], .1)
        fine = parent.derivative_rows(fd['+0.025']['per_token_nll'], fd['-0.025']['per_token_nll'], .05)
        references = {'fd_run_receipt_sha256': {alpha: fd[alpha]['receipt_sha256'] for alpha in ALPHAS}}
        kind = 'NEW_FIVE_CELL_FIXED_FD_CHECKPOINT'
    convergence = legacy.compare_derivatives(coarse, fine)
    comparison = legacy.compare_derivatives(measured['gradients']['per_token_dnll'], fine)
    passed = convergence['passed'] is True and comparison['passed'] is True
    return {'schema': CHECK_SCHEMA, 'scope': SCOPE,
            'status': 'FD_CONVERGENCE_AND_JVP_AGREEMENT_PASS' if passed else 'NUMERICAL_REJECTED',
            'passed': passed, 'kind': kind, 'role_index': role, 'role_key': common.ROLE_ORDER[role],
            'tier': tier, 'column_index': column_index(role, tier),
            'plan_receipt_sha256': context['plan']['receipt_sha256'], 'source_identity': context['identity'],
            'jvp_run_receipt_sha256': measured['receipt_sha256'], 'jvp_output_sha256': measured['gradient_output_sha256'],
            'fd_convergence': convergence, 'jvp_vs_fine_fd': comparison,
            'per_token_coarse_fd': coarse, 'per_token_fine_fd': fine, **references, **NO_ACTION}


def save_checkpoint(args, context, measured, records):
    payload = checkpoint_payload(context, measured, records)
    path = output_path(args.root, Path(args.root) / f"CHECKPOINT_r{measured['role_index']:02d}_{measured['tier']}.json")
    result = seal_receipt(path, payload, resume=True)
    if not result['passed']:
        seal_receipt(output_path(args.root, Path(args.root) / 'NUMERICAL_HOLD_RECEIPT.json'), {
            'schema': HOLD_SCHEMA, 'scope': SCOPE, 'status': 'NUMERICAL_HOLD_NO_FURTHER_NATIVE_CELLS',
            'plan_receipt_sha256': context['plan']['receipt_sha256'],
            'checkpoint_receipt_sha256': result['receipt_sha256'],
            'checkpoint_path': str(path), 'completed_native_cell_count': len(records),
            'completed_evidence_cell_count': len(records), 'inherited_native_cell_count': 1,
            'native_cells_executed_in_this_plan': len(records) - 1,
            'completed_cell_indices': sorted(records), 'g': None, 'observed_label_directional_gram': None,
            **NO_ACTION}, resume=True)
    print(json.dumps({'event': 'ALL48_CHECKPOINT', 'role_index': measured['role_index'],
                      'tier': measured['tier'], 'passed': result['passed'],
                      'receipt_sha256': result['receipt_sha256']}, sort_keys=True), flush=True)
    return result


def load_prefix(args, context, *, create_checkpoints):
    verify_inventory(args.root)
    records, checkpoints, gap = {}, {}, False
    for cell in execution_sequence():
        control_pin = records[0]['receipt_sha256'] if records and cell['kind'] != 'OFF' else None
        saved, failures = inspect_attempts(args.root, context, cell, control_pin)
        need(not gap or (saved is None and not failures), 'NATIVE_ATTEMPT_AFTER_SEQUENCE_GAP_OR_HOLD')
        if saved is None:
            gap = True
            continue
        need(not gap, 'SUCCESSFUL_CELL_AFTER_INCOMPLETE_SEQUENCE_GAP')
        records[cell['cell_index']] = saved
        pair = (cell['role_index'], cell['tier'])
        if cell['kind'] == 'JVP' and pair in ALL_CHECKPOINTS:
            if create_checkpoints:
                checkpoint = save_checkpoint(args, context, saved, records)
            else:
                path = Path(args.root) / f'CHECKPOINT_r{pair[0]:02d}_{pair[1]}.json'
                checkpoint = sealed(path)
                need({k: v for k, v in checkpoint.items() if k != 'receipt_sha256'} ==
                     checkpoint_payload(context, saved, records), 'CHECKPOINT_RAW_REPLAY_DRIFT')
            checkpoints[pair] = checkpoint
            if not checkpoint['passed']:
                # No successful output may exist beyond a completed numerical rejection.
                gap = True
    return records, checkpoints


def token_partition(metadata, targets):
    starts, decoded = common.read_meta(metadata)
    need(decoded == targets and len(targets) == 16 and sum(map(len, targets)) == 656, 'TOKEN_PARTITION_ALIGNMENT')
    tokens, groups = [], []
    for case, row in enumerate(targets):
        offset = len(tokens)
        tokens.extend({'case_id': case, 'token_index': index, 'target_token_id': target,
                       'position': starts[case] + index - 1} for index, target in enumerate(row))
        groups.append({'case_id': case, 'token_offset_start': offset, 'token_offset_end': len(tokens),
                       'gold_tokens': len(row)})
    return tokens, groups


def jacobi_eigenvalues(matrix):
    """Small real-symmetric eigensolver; return a conservative residual bound."""
    n = len(matrix)
    need(n > 0 and all(len(row) == n for row in matrix), 'EIGEN_MATRIX_SHAPE')
    need(all(type(v) in (int, float) and math.isfinite(v) for row in matrix for v in row), 'EIGEN_NONFINITE')
    need(all(matrix[i][j] == matrix[j][i] for i in range(n) for j in range(n)), 'EIGEN_ASYMMETRY')
    a = [list(row) for row in matrix]
    scale = max(1.0, math.fsum(abs(a[i][i]) for i in range(n)))
    tolerance = 1e-15 * scale / max(1, n)
    for sweep in range(100):
        largest = max((abs(a[p][q]) for p in range(n) for q in range(p + 1, n)), default=0.0)
        if largest <= tolerance:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) <= tolerance:
                    continue
                tau = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = math.copysign(1.0 / (abs(tau) + math.hypot(1.0, tau)), tau)
                cosine = 1.0 / math.sqrt(1.0 + t * t)
                sine = t * cosine
                a[p][p] -= t * apq
                a[q][q] += t * apq
                a[p][q] = a[q][p] = 0.0
                for k in range(n):
                    if k not in (p, q):
                        old_p, old_q = a[k][p], a[k][q]
                        a[k][p] = a[p][k] = cosine * old_p - sine * old_q
                        a[k][q] = a[q][k] = sine * old_p + cosine * old_q
    residual = max((math.fsum(abs(a[i][j]) for j in range(n) if i != j) for i in range(n)), default=0.0)
    need(math.isfinite(residual) and residual <= 1e-12 * scale, 'EIGEN_RESIDUAL_TOO_LARGE')
    return sorted(a[i][i] for i in range(n)), residual


def matrix_statistics(matrix, groups):
    need(isinstance(matrix, list) and len(matrix) == 656
         and all(isinstance(row, list) and len(row) == 48 for row in matrix), 'ALL48_MATRIX_EXACT_SHAPE')
    need(all(type(v) in (int, float) and math.isfinite(v) for row in matrix for v in row), 'MATRIX_NONFINITE')
    need(len(groups) == 16, 'SIXTEEN_SENTENCE_GROUPS_REQUIRED')
    cursor = 0
    for case, group in enumerate(groups):
        need(group.get('case_id') == case and group.get('token_offset_start') == cursor
             and type(group.get('gold_tokens')) is int and group['gold_tokens'] > 0
             and group.get('token_offset_end') == cursor + group['gold_tokens'], 'SENTENCE_PARTITION_GAP_OR_OVERLAP')
        cursor = group['token_offset_end']
    need(cursor == 656, 'SENTENCE_PARTITION_TOTAL')
    columns = [list(column) for column in zip(*matrix)]
    g = [math.fsum(column) / 656 for column in columns]
    gram = [[math.fsum(a * b for a, b in zip(left, right)) / 656 for right in columns] for left in columns]
    need(all(math.isfinite(v) for v in g + [v for row in gram for v in row]), 'MOMENT_OVERFLOW')
    eigenvalues, residual = jacobi_eigenvalues(gram)
    trace = math.fsum(gram[i][i] for i in range(48))
    lower_bound = -1e-8 * max(1.0, trace)
    psd = {'minimum_eigenvalue': eigenvalues[0], 'maximum_eigenvalue': eigenvalues[-1],
           'trace': trace, 'eigenvalues_ascending': eigenvalues, 'offdiagonal_residual_bound': residual,
           'acceptance_lower_bound': lower_bound,
           'passed': eigenvalues[0] - residual >= lower_bound,
           'method': 'REAL_SYMMETRIC_CYCLIC_JACOBI_WITH_RESIDUAL_BOUND'}
    sentences = []
    for group in groups:
        part = matrix[group['token_offset_start']:group['token_offset_end']]
        sums = [math.fsum(row[index] for row in part) for index in range(48)]
        sentences.append({**group, 'sum_dnll': sums,
                          'mean_dnll': [value / group['gold_tokens'] for value in sums]})
    return {'g': g, 'gram': gram, 'psd': psd, 'sentence_statistics': sentences}


def analyze_complete(args, context, records, checkpoints):
    need(set(records) == set(range(79)) and set(checkpoints) == set(ALL_CHECKPOINTS)
         and all(c['passed'] is True for c in checkpoints.values()), 'ALL79_AND_NINE_CHECKPOINT_PASSES_REQUIRED')
    hold_path = output_path(args.root, Path(args.root) / 'NUMERICAL_HOLD_RECEIPT.json')
    previous_hold = sealed(hold_path) if hold_path.exists() else None
    need(previous_hold is None or (previous_hold.get('reason') == 'GRAM_PSD_REJECTED'
         and previous_hold.get('plan_receipt_sha256') == context['plan']['receipt_sha256']),
         'CHECKPOINT_NUMERICAL_HOLD_BLOCKS_MATRIX')
    measured = {r['column_index']: r for r in records.values() if r['kind'] == 'JVP'}
    need(set(measured) == set(range(48)), 'ALL48_COLUMNS_REQUIRED')
    columns = [legacy.flatten(measured[index]['gradients']['per_token_dnll']) for index in range(48)]
    matrix = [list(row) for row in zip(*columns)]
    tokens, groups = token_partition(context['metadata'].read_bytes(), context['dataset']['target_token_ids'])
    statistics = matrix_statistics(matrix, groups)
    original = common.MODEL / 'weights_moe/deepseek_moe_af.bin'
    need(sha(original) == common.AF_SHA, 'POST_RUN_ORIGINAL_AF4_DRIFT')
    if not statistics['psd']['passed']:
        hold = seal_receipt(hold_path, {
            'schema': HOLD_SCHEMA, 'scope': SCOPE, 'status': 'NUMERICAL_HOLD_NO_FURTHER_NATIVE_CELLS',
            'reason': 'GRAM_PSD_REJECTED', 'plan_receipt_sha256': context['plan']['receipt_sha256'],
            'completed_native_cell_count': 79, 'completed_cell_indices': list(range(79)),
            'completed_evidence_cell_count': 79, 'inherited_native_cell_count': 1,
            'native_cells_executed_in_this_plan': 78,
            'psd': statistics['psd'], 'g': None, 'observed_label_directional_gram': None,
            'post_run_original_af4_sha256': common.AF_SHA, **NO_ACTION}, resume=True)
        return seal_receipt(output_path(args.root, Path(args.root) / 'JE01_ALL48_TRAIN16_ANALYSIS.json'), {
            'schema': ANALYSIS_SCHEMA, 'scope': SCOPE, 'status': 'NUMERICAL_HOLD_NO_FURTHER_NATIVE_CELLS',
            'plan_receipt_sha256': context['plan']['receipt_sha256'], 'reason': 'GRAM_PSD_REJECTED',
            'evidence_integrity': 'PASS', 'checkpoint_numerical_verdict': 'PASS', 'psd_verdict': 'FAIL',
            'completed_native_cell_count': 79, 'completed_evidence_cell_count': 79,
               'native_cells_executed_in_this_plan': 78, 'inherited_native_cell_count': 1,
               'native_jvp_columns': 48, 'fd_verified_coordinates': 9,
            'hold_receipt_sha256': hold['receipt_sha256'], 'psd': statistics['psd'],
            'g': None, 'observed_label_directional_gram': None, **NO_ACTION}, resume=True)
    need(previous_hold is None, 'PREVIOUS_NUMERICAL_HOLD_CANNOT_BE_OVERRIDDEN')
    matrix_payload = {'schema': MATRIX_SCHEMA, 'scope': SCOPE,
                      'plan_receipt_sha256': context['plan']['receipt_sha256'],
                      'runner_source_sha256': sha(Path(__file__)), 'shape': [656, 48],
                      'coordinate_order': coordinate_order(), 'token_rows': tokens,
                      'per_token_dnll': matrix, 'sentence_groups': groups,
                      'jvp_receipt_sha256': [measured[i]['receipt_sha256'] for i in range(48)],
                      'jvp_output_sha256': [measured[i]['gradient_output_sha256'] for i in range(48)], **NO_ACTION}
    matrix_path = output_path(args.root, Path(args.root) / 'RAW_DERIVATIVE_MATRIX.json')
    matrix_receipt = seal_receipt(matrix_path, matrix_payload, resume=True)
    directions = []
    for coordinate in coordinate_order():
        pair = (coordinate['role_index'], coordinate['tier'])
        directions.append({**coordinate, 'raw_native_validation': 'PRIMAL_ROUTE_TARGET_FINITE_FULL_PREFIX_PASS',
                           'fd_validation': 'CHECKPOINT_PASS' if pair in checkpoints else 'NOT_INDIVIDUALLY_FD_CHECKED',
                           'checkpoint_receipt_sha256': checkpoints[pair]['receipt_sha256'] if pair in checkpoints else None,
                           'jvp_receipt_sha256': measured[coordinate['column_index']]['receipt_sha256']})
    elapsed = {kind: [r['execution']['elapsed_seconds'] for r in records.values() if r['kind'] == kind]
               for kind in ('OFF', 'FD', 'JVP')}
    payload = {'schema': ANALYSIS_SCHEMA, 'scope': SCOPE, 'status': COMPLETE_STATUS,
               'evidence_integrity': 'PASS', 'checkpoint_numerical_verdict': 'PASS',
               'source_identity': context['identity'], 'plan_receipt_sha256': context['plan']['receipt_sha256'],
               'runner_source_sha256': sha(Path(__file__)), 'case_ids': list(range(16)), 'gold_tokens': 656,
               'completed_native_cell_count': 79, 'completed_evidence_cell_count': 79,
               'native_cells_executed_in_this_plan': 78, 'inherited_native_cell_count': 1,
               'native_jvp_columns': 48, 'fd_verified_coordinates': 9,
               'fd_coverage_interpretation': 'NINE_PREDECLARED_CHECKPOINTS_NOT_ALL_48_INDIVIDUAL_FD_VALIDATIONS',
               'coordinate_order': coordinate_order(), 'directions': directions,
               'cell_receipts': [{'cell_index': i, 'attempt': records[i]['attempt'],
                                  'path': str(records[i]['_folder'] / 'RUN_RECEIPT.json'),
                                  'receipt_sha256': records[i]['receipt_sha256']} for i in range(79)],
               'checkpoint_receipts': [{'role_index': r, 'tier': t,
                                        'receipt_sha256': checkpoints[(r, t)]['receipt_sha256']}
                                       for r, t in ALL_CHECKPOINTS],
               'raw_derivative_matrix': {'path': str(matrix_path), 'shape': [656, 48],
                                         'receipt_sha256': matrix_receipt['receipt_sha256'],
                                         'file_sha256': sha(matrix_path)},
               'g': statistics['g'], 'g_definition': 'MEAN_TOKEN_DNLL', 'score_definition': 'NEGATIVE_DNLL',
               'observed_label_directional_gram': {
                   'definition': 'MEAN_TOKEN_DNLL_OUTER_PRODUCT', 'matrix': statistics['gram'],
                   'centering': 'NONE', 'tokens': 656, 'coordinate_order': coordinate_order(),
                   'interpretation': 'EXPLORATORY_48_DIRECTION_FIXED_BRANCH_GRAM_ON_SEEN_TRAIN16',
                   'used_for_policy_selection': False},
               'psd': statistics['psd'], 'sentence_statistics': statistics['sentence_statistics'],
               'measured_timing': {'seconds_by_kind': elapsed,
                                   'total_native_elapsed_seconds': math.fsum(v for values in elapsed.values() for v in values),
                                   'native_elapsed_seconds_this_plan': math.fsum(r['execution']['elapsed_seconds'] for r in records.values() if r['native_executed_in_this_plan']),
                                   'inherited_native_elapsed_seconds': records[0]['execution']['elapsed_seconds']},
               'post_run_original_af4_sha256': common.AF_SHA,
               'natural_routing_stability': 'NOT_ESTABLISHED_BY_FORCED_ROUTE_REPLAY',
               'independent_audit_required_before_train240': True, **NO_ACTION}
    return seal_receipt(output_path(args.root, Path(args.root) / 'JE01_ALL48_TRAIN16_ANALYSIS.json'),
                       payload, resume=True)


def next_receipt_path(root, prefix):
    index = 1
    while (Path(root) / f'{prefix}_{index:04d}.json').exists():
        index += 1
    return output_path(root, Path(root) / f'{prefix}_{index:04d}.json')


def run(args):
    root = require_root(args.root)
    need(type(args.max_new_cells) is int and 1 <= args.max_new_cells <= 79, 'MAX_NEW_CELLS_RANGE_1_TO79')
    need(type(args.max_cell_attempt) is int and 1 <= args.max_cell_attempt <= 100, 'MAX_CELL_ATTEMPT_RANGE')
    with exclusive_root(root):
        context = establish_plan(args, preflight(args))
        if (root / 'runs').exists():
            need(args.resume, 'EXPLICIT_RESUME_REQUIRED')
        inherit_off(args, context)
        records, checkpoints = load_prefix(args, context, create_checkpoints=True)
        before, fresh, started = len(records), 0, time.monotonic()
        hold = any(not c['passed'] for c in checkpoints.values())
        if not hold and (root / 'NUMERICAL_HOLD_RECEIPT.json').exists():
            previous_hold = sealed(root / 'NUMERICAL_HOLD_RECEIPT.json')
            need(len(records) == 79 and previous_hold.get('reason') == 'GRAM_PSD_REJECTED'
                 and previous_hold.get('plan_receipt_sha256') == context['plan']['receipt_sha256'],
                 'UNEXPLAINED_NUMERICAL_HOLD')
        if not hold:
            for cell in execution_sequence()[len(records):]:
                if fresh == args.max_new_cells:
                    break
                control_pin = records[0]['receipt_sha256'] if records and cell['kind'] != 'OFF' else None
                saved, failures = inspect_attempts(root, context, cell, control_pin)
                need(saved is None, 'NEXT_CELL_ALREADY_COMPLETE')
                measured = run_cell(args, context, cell, control_pin, failures)
                records[cell['cell_index']] = measured
                fresh += 1
                pair = (cell['role_index'], cell['tier'])
                if cell['kind'] == 'JVP' and pair in ALL_CHECKPOINTS:
                    checkpoint = save_checkpoint(args, context, measured, records)
                    checkpoints[pair] = checkpoint
                    if not checkpoint['passed']:
                        hold = True
                        break
        status = 'NUMERICAL_HOLD_NO_FURTHER_NATIVE_CELLS' if hold else 'PARTIAL_NO_MATRIX_NO_PASS'
        analysis = None
        if len(records) == 79 and not hold:
            # Reopen all raw bytes after execution, before any aggregate is published.
            records, checkpoints = load_prefix(args, context, create_checkpoints=False)
            analysis = analyze_complete(args, context, records, checkpoints)
            status = analysis['status']
        chunk = seal_receipt(next_receipt_path(root, 'CHUNK_RECEIPT'), {
            'schema': CHUNK_SCHEMA, 'scope': SCOPE, 'status': status,
            'plan_receipt_sha256': context['plan']['receipt_sha256'], 'runner_source_sha256': sha(Path(__file__)),
            'completed_cells_before': before, 'completed_cells_after': len(records), 'new_native_cells': fresh,
            'inherited_native_cell_count': 1,
            'native_cells_executed_in_this_plan': len(records) - 1,
            'requested_max_new_cells': args.max_new_cells, 'requested_max_cell_attempt': args.max_cell_attempt,
            'chunk_elapsed_seconds': time.monotonic() - started,
            'completed_cell_receipt_sha256': [records[i]['receipt_sha256'] for i in range(len(records))],
            'next_cell_index': len(records) if len(records) < 79 and not hold else None,
            'analysis_receipt_sha256': analysis['receipt_sha256'] if analysis else None,
            'more_native_cells_required': len(records) < 79 and not hold, **NO_ACTION})
    print(json.dumps({'event': 'ALL48_RUN_DONE', 'status': status, 'completed_cells': len(records),
                      'new_native_cells': fresh, 'inherited_native_cell_count': 1,
                      'native_cells_executed_in_this_plan': len(records) - 1,
                      'receipt_sha256': chunk['receipt_sha256'],
                      'analysis_receipt_sha256': analysis['receipt_sha256'] if analysis else None},
                     sort_keys=True), flush=True)
    return analysis or chunk


def analyze(args):
    root = require_root(args.root)
    with exclusive_root(root):
        context = establish_plan(args, preflight(args))
        records, checkpoints = load_prefix(args, context, create_checkpoints=True)
        hold = any(not c['passed'] for c in checkpoints.values())
        if len(records) == 79 and not hold:
            result = analyze_complete(args, context, records, checkpoints)
        else:
            result = seal_receipt(next_receipt_path(root, 'PARTIAL_ANALYSIS'), {
                'schema': ANALYSIS_SCHEMA, 'scope': SCOPE,
                'status': 'NUMERICAL_HOLD_NO_FURTHER_NATIVE_CELLS' if hold else 'PARTIAL_NO_MATRIX_NO_PASS',
                'plan_receipt_sha256': context['plan']['receipt_sha256'],
                'completed_native_cell_count': len(records), 'completed_evidence_cell_count': len(records),
                'inherited_native_cell_count': int(bool(records)),
                'native_cells_executed_in_this_plan': max(0, len(records) - 1),
                'native_jvp_columns': sum(r['kind'] == 'JVP' for r in records.values()),
                'checkpoint_receipt_sha256': [c['receipt_sha256'] for c in checkpoints.values()],
                'g': None, 'observed_label_directional_gram': None, **NO_ACTION})
    print(json.dumps({'event': 'ALL48_ANALYSIS_DONE', 'status': result['status'],
                      'receipt_sha256': result['receipt_sha256']}, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run', 'analyze'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-plan-receipt')
    parser.add_argument('--plan-only', action='store_true', help='Seal the full plan without native inference.')
    parser.add_argument('--max-new-cells', type=int, default=12)
    parser.add_argument('--max-cell-attempt', type=int, default=1,
                        help='Raise explicitly only to retry a previously sealed execution/integrity failure.')
    parser.add_argument('--timeout-seconds', type=int, default=900)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    legacy.memory_stop_reason(0, 100, 0, args.timeout_seconds)
    if args.action == 'plan' or args.plan_only:
        plan(args)
    elif args.action == 'run':
        result = run(args)
        if result['status'] == 'NUMERICAL_HOLD_NO_FURTHER_NATIVE_CELLS':
            raise SystemExit(2)
    else:
        result = analyze(args)
        if result['status'] == 'NUMERICAL_HOLD_NO_FURTHER_NATIVE_CELLS':
            raise SystemExit(2)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError(f'JE01_ALL48_SIGNAL:{signum}')
    signal.signal(signal.SIGTERM, interrupted)
    main()
