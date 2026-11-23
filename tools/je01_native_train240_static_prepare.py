#!/usr/bin/env python3
"""Prepare fifteen exact-counter native source variants, after the all48 gate.

This program does not compile or run a model. `spec` prints an unsealed design.
`manifest` first requires the actual independent all48 TRAIN16 PASS, then
reopens TRAIN240 source/input evidence and seals all expected source hashes.
`stage` requires that external manifest pin and produces fifteen source copies.
Only the completion token, layer-step, and gold counts change. The case counter
stays exactly 32. Restoring the two replaced lines must restore both base SHAs.
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
import shutil
import sys

TOOL_DIR = Path(__file__).resolve().parent
WORK = Path('/Volumes/D50/TailnetCommander/work')
ROOT_PREFIX = 'je01_native_role_jvp_train240_static_'
PARENT = WORK / 'je01_native_role_jvp_followup_20261123_v1'
UPSTREAM = WORK / 'je01_native_role_jvp_all48_train16_20261123_v2'
ORDINARY = WORK / 'je01_role_directional_20261123_v1'
PREP_ROOT = ORDINARY / 'train240_inputs'
BANK = WORK / 'je01_role_direction_bank_20261123_v1'
MODEL = WORK / 'hcp-p18-external-multimetric-20261015/native_moe_base_source_copy/weights_moe/deepseek_moe_af.bin'
SMOKE = WORK / 'je01_native_jvp_smoke_20261123_v4/JE01_NATIVE_JVP_SMOKE_RECEIPT.json'
CMAKE = WORK / 'je01_postnll_bc_native_compile_20261123_v1/venv/bin/cmake'
UPSTREAM_PLAN = '260c199a18f1b60bf94a6388fd9df306803a950383a199bd2665b88a6d553aba'
UPSTREAM_RUNNER = 'ad07f04937a79868a67f8eacedee22b6fe091399cf0557e5d2575234ebc50df6'
UPSTREAM_AUDITOR = 'a3cf5a2dc8048f51208f8a44ce3038eb51b9f44147ba4f62bb8a65aac0524d23'
PARENT_BUILD = '2cbeb4dc1a7e1ab5642a4abb22395498c10dde43bee051eec3c303c7e53fa008'
BASE_CPP = '6055ca3b261075fe5fcf76e5829f0fb20b8ecdd52599248f4b9b1032cfbeb49f'
BASE_C = '5e2a6ea76c22c1ec0ff9f2fb6d093b7d3a471679adbe439ed1c4b898a56a52a6'
BASE_CMAKE = '1479cf49e18da616e6fcfdde2ed9433b468701a3d5a4482fa7e9ee398683da19'
ADAPTER = '9ae9d98b921334bae321e583908e71bdc9c08178cbf3ca5d7a1d4d4e552a09ec'
BINARY = '110b1978d97bcd7c6bdfd6d42142569e56e9dacffa4ff462dc0786b3128e49f4'
AF4 = '9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4'
MLX = 'afb1939edc96dc10e452132e4e665276fe0c48bf8cdc1a69286feb8a14de1da0'
SMOKE_PIN = 'b0825ab7a048f7d95b155592760e10d73e3df2b112230adbf7b348f388ca1595'
BANK_PIN = '3b18257d22caddd0916292bef692110f9eeaf0edf83ad3fd4d5417fdf3443bb8'
BANK_TSV = 'fdc801f59f9c6438dd60bb33fab9a1332cb19c6e15eb15bef7425b7fbd522e4f'
BANK_RECEIPT = '4994ec762b3a9b8a44015bad18f4b5bc7ea26505208384146238e7def28464d4'
PREPARATION = 'f0a95e7ee7fbb1ce001c419c8626744e3730e85b235b9a726897ae38f946e477'
PREPARATION_FILE = '4dfb79c79bc564eaf1f7372c33b80fcaa3aba966ff280d0ac66ecb9ace565b50'
HELPERS = {
    'je01_role_directional_train240_inputs.py': '99f4d6da38880e3eaadcd8a32e9f288a75ae87b7f13c9721fe00109616b7175b',
    'je01_role_directional_native_runner.py': '5607c0afef3bd4b1a61bcab6622267acc6b6137fc65ba0e79873e92e31d7e9e6',
}
GOLD_COUNTS = (634, 702, 664, 784, 791, 772, 589, 646, 742, 698, 660, 700, 776, 617, 593)
INPUT_COUNTS = (2226, 2193, 2252, 2432, 2489, 2403, 1822, 2142, 2558, 2184, 2154, 2247, 2553, 1998, 1907)
DATASET_PINS = (
    'a571d5f863287d8b274cae19075e608ff60f5415df8badcd153bc1584372d99d',
    'e4b2dfd72f327a537d1af12cb8c6d5a66899a7d73af73759351cd770f666372d',
    '8a8b773b0f1719daf7a61bc5a56427b07e29f6f52e328eea64b5c990afc642b7',
    '613c0a270e2c9facffc9b0447de992d2d4bcb0faa8228ce310c9132363734501',
    'dfe2ca29b0c96a8c98469ca0a6270a4a9110b11518f79dc6d865b459a102dd7c',
    '676c2c33ecf9a718725e2ec50a4ee3f9914f7051300377e52eaf7dc92db0b0f6',
    '8cc1c9571fa57881f1d0e0b6d97f0052100bc06de8e7b5c767e38ca8e01cbd8a',
    'ff1b28a556b8b6b05a0dae9aeed18b2e52ed50739a52dc85ef0d7f7b043794af',
    'bd666828cd2e761b0bc1f63931c8bdcb9907cdb05290ea9063d62ee91f8bec21',
    '50414c9448c8da5b7bbf5fb7b62c45e6f0fffb9e9e259974ebfe4c0186379ad5',
    'f85377f0559c796ccd65138e84084af4bef5135eb7c083030a2a116980fb1f65',
    '5815116722db534ed08e5b48f3fc8e4cf12bd0fe696d7a046a4f9bea7399bc21',
    '80d2b8f19be1e866d5a96da9f4fbf698badc3aeeb11e26d21903c217d39a05ee',
    '8a07c5262bc5735dca1049702afd719992d04cb6d974233a7d5d960ecc197649',
    'c4a469243dd7827972154a955c9b4ebb1f4a2dfa00e827cb47499cb1a6fbc916',
)
CPP_ANCHOR = 'g_je01_one_tokens!=4334 || g_je01_one_cases!=32 || g_je01_one_layers!=117018)return 0;'
C_ANCHOR = 'if(je01_jvp_gold_count!=656 || !mlx_gpu_je01_one_column_jvp_finish(&tokens,&cases,&layers) ||'
SCOPE = 'TRAIN240_ROWS16_THROUGH255_STATIC_COMPLETION_SOURCE_PREPARATION_ONLY'
MANIFEST_SCHEMA = 'beglin-je01-train240-static-counter-build-manifest/1'
STAGE_SCHEMA = 'beglin-je01-train240-static-counter-source-stage/1'
SOURCE_STAGE_HEADROOM_BYTES = 1 << 30
NO_RUN = {'native_compilation': 'NOT_RUN', 'native_model_forward': 'NOT_RUN', 'derivative_collection': 'NOT_RUN',
          'allocation_selected': False, 'independent_validation_ood_accessed': False, 'production_modified': False}


def need(condition, message):
    if not condition:
        raise ValueError('JE01_TRAIN240_STATIC_' + message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def is_sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def sha(path):
    path = Path(path)
    need(path.is_file() and path.resolve() == path and not path.is_symlink(), 'FILE_MISSING_OR_LINK:' + str(path))
    state = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2 << 20), b''):
            state.update(block)
    return state.hexdigest()


def sealed(path, expected, *, field='receipt_sha256'):
    need(is_sha(expected), 'EXTERNAL_CANONICAL_PIN_REQUIRED')
    sha(path)
    value = json.loads(Path(path).read_bytes())
    need(type(value) is dict and value.get(field) == expected
         and canonical({k: v for k, v in value.items() if k != field}) == expected,
         'RECEIPT_CANONICAL_DRIFT:' + str(path))
    return value


def seal_once(path, payload, *, resume=False):
    path = Path(path)
    need('receipt_sha256' not in payload and path.resolve() == path and not path.is_symlink(), 'SEAL_PATH')
    expected = canonical(payload)
    if path.exists():
        need(resume, 'APPEND_ONLY_RECEIPT_EXISTS')
        result = sealed(path, expected)
        need({k: v for k, v in result.items() if k != 'receipt_sha256'} == payload, 'RESUME_PAYLOAD_DRIFT')
        return result
    result = {**payload, 'receipt_sha256': expected}
    with path.open('x') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return result


def require_root(value):
    path = Path(value)
    need(path.is_absolute() and path.parent == WORK and path.name.startswith(ROOT_PREFIX)
         and path.name != ROOT_PREFIX and path.resolve() == path and not path.is_symlink(), 'FRESH_NATIVE_COMPATIBLE_ROOT')
    return path


@contextmanager
def exclusive_root(root):
    lock_path = root / 'STATIC_PREPARATION_LOCK'
    need(lock_path.resolve() == lock_path and not lock_path.is_symlink(), 'LOCK_PATH')
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def validate_full48_audit(audit):
    """A three-role PASS, partial result, or failed checkpoint cannot open this gate."""
    expected = {'schema': 'beglin-je01-native-all48-train16-independent-raw-audit/2',
                'status': 'INDEPENDENT_AUDIT_CONFIRMED_ALL48_TRAIN16_RAW_AND_NINE_CHECKPOINTS_PASS',
                'scope': 'PREVIOUSLY_SEEN_TRAIN16_FIXED_HARD_ROUTE_48_DIRECTION_DIAGNOSTIC_ONLY',
                'root': str(UPSTREAM), 'evidence_integrity': 'PASS', 'experiment_numerical_verdict': 'PASS',
                'completed_evidence_cell_count': 79, 'completed_native_cell_count': 79,
                'native_cells_executed_in_this_plan': 78, 'inherited_native_cell_count': 1,
                'native_jvp_columns': 48, 'fd_verified_coordinates': 9,
                'gold_tokens': 656, 'case_ids': list(range(16)), 'sentence_groups_count': 16,
                'plan_receipt_sha256': UPSTREAM_PLAN, 'external_plan_pin': UPSTREAM_PLAN,
                'runner_source_sha256': UPSTREAM_RUNNER, 'auditor_source_sha256': UPSTREAM_AUDITOR,
                'all_48_raw_directions_verified': True, 'all_nine_fd_checkpoints_pass': True,
                'original_af4_full_file_sha_verified_after_run': AF4,
                'native_inference_rerun': False, 'experiment_runner_or_helper_imported': False,
                'allocation_selected': False, 'independent_validation_ood_accessed': False, 'production_modified': False}
    need(type(audit) is dict and all(audit.get(k) == v and type(audit.get(k)) is type(v)
                                    for k, v in expected.items()), 'ACTUAL_INDEPENDENT_ALL48_PASS_REQUIRED')
    need(audit.get('independent_psd_check', {}).get('passed') is True
         and audit.get('numerical_hold_receipt_sha256') is None
         and is_sha(audit.get('source_report_receipt_sha256')) and is_sha(audit.get('raw_matrix_receipt_sha256')),
         'UPSTREAM_MATRIX_OR_PSD_GATE')
    need(type(audit.get('g')) is list and len(audit['g']) == 48
         and all(type(v) in (int, float) and math.isfinite(v) for v in audit['g'])
         and type(audit.get('observed_label_directional_gram')) is list
         and len(audit['observed_label_directional_gram']) == 48
         and all(type(row) is list and len(row) == 48 and
                 all(type(v) in (int, float) and math.isfinite(v) for v in row)
                 for row in audit['observed_label_directional_gram']), 'UPSTREAM_NUMERICAL_SHAPE')


def require_full48_gate(args):
    # Deliberately first: no TRAIN240/runtime helper or source file is opened
    # until the externally pinned independent numerical gate has passed.
    need(is_sha(args.expected_all48_audit_receipt) and is_sha(args.expected_all48_audit_file_sha),
         'ACTUAL_ALL48_EXTERNAL_CANONICAL_AND_FILE_PINS_REQUIRED')
    path = UPSTREAM / 'JE01_ALL48_TRAIN16_INDEPENDENT_AUDIT.json'
    need(sha(path) == args.expected_all48_audit_file_sha, 'UPSTREAM_AUDIT_FILE_PIN')
    audit = sealed(path, args.expected_all48_audit_receipt)
    validate_full48_audit(audit)
    need(sha(UPSTREAM / 'tools/je01_native_all48_train16_runner_v2.py') == UPSTREAM_RUNNER
         and sha(UPSTREAM / 'tools/je01_native_all48_train16_independent_audit_v2.py') == UPSTREAM_AUDITOR,
         'UPSTREAM_EXECUTED_SOURCE_PINS')
    plan = sealed(UPSTREAM / 'PLAN_RECEIPT.json', UPSTREAM_PLAN)
    analysis = sealed(UPSTREAM / 'JE01_ALL48_TRAIN16_ANALYSIS.json', audit['source_report_receipt_sha256'])
    matrix = sealed(UPSTREAM / 'RAW_DERIVATIVE_MATRIX.json', audit['raw_matrix_receipt_sha256'])
    need(analysis.get('status') == 'ALL48_NATIVE_COMPLETE_CHECKPOINTS_AND_PSD_PASS'
         and analysis.get('plan_receipt_sha256') == matrix.get('plan_receipt_sha256') == UPSTREAM_PLAN
         and matrix.get('shape') == [656, 48]
         and canonical(matrix.get('per_token_dnll')) == audit.get('reconstructed_matrix_values_sha256'),
         'UPSTREAM_RAW_MATRIX_REPORT_BINDING')
    return {'audit_receipt_sha256': audit['receipt_sha256'], 'audit_file_sha256': sha(path),
            'audit_path': str(path), 'plan_receipt_sha256': UPSTREAM_PLAN,
            'runner_source_sha256': UPSTREAM_RUNNER, 'auditor_source_sha256': UPSTREAM_AUDITOR,
            'analysis_receipt_sha256': analysis['receipt_sha256'], 'raw_matrix_receipt_sha256': matrix['receipt_sha256'],
            'status': 'INDEPENDENT_ALL48_TRAIN16_GATE_PASS'}, plan


def checked_helpers():
    modules = {}
    for filename, pin in HELPERS.items():
        path = TOOL_DIR / filename
        need(sha(path) == pin, 'IMMUTABLE_HELPER_PIN:' + filename)
        name = path.stem
        if name in sys.modules:
            module = sys.modules[name]
            need(Path(module.__file__).resolve() == path, 'IMPORTED_HELPER_PATH')
        else:
            spec = importlib.util.spec_from_file_location(name, path)
            need(spec is not None and spec.loader is not None, 'HELPER_IMPORT_SPEC')
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        modules[name] = module
    return modules['je01_role_directional_train240_inputs'], modules['je01_role_directional_native_runner']


def counter_profile(chunk, native_input_tokens, gold_tokens):
    need(type(chunk) is int and 1 <= chunk <= 15
         and type(native_input_tokens) is int and native_input_tokens == INPUT_COUNTS[chunk - 1]
         and type(gold_tokens) is int and gold_tokens == GOLD_COUNTS[chunk - 1], 'FROZEN_CHUNK_COUNTS')
    tokens = 2 * native_input_tokens
    return {'chunk': chunk, 'case_ids': list(range(16 * chunk, 16 * (chunk + 1))),
            'native_local_case_ids': list(range(16)), 'native_input_tokens': native_input_tokens,
            'completion': {'tokens': tokens, 'cases': 32, 'layers': 27 * tokens, 'gold': gold_tokens},
            'expected_route_records': tokens * 26, 'expected_route_bytes': 30 + tokens * 26 * 40,
            'counter_definition': 'TWO_NATIVE_PASSES_OF_FULL_TEACHER_FORCED_INPUT;27_LAYERS;16_CASES_PER_PASS'}


def derive_profile(data, rows):
    chunk = data.get('chunk')
    need(type(chunk) is int and 1 <= chunk <= 15 and data.get('scope') == 'TRAIN240'
         and data.get('receipt_sha256') == DATASET_PINS[chunk - 1]
         and data.get('case_ids') == list(range(16 * chunk, 16 * (chunk + 1)))
         and type(rows) is list and len(rows) == 16
         and [row.get('row') for row in rows] == data['case_ids'], 'SOURCE_ROWS_AND_DATASET_BINDING')
    lengths = [len(row['prompt_token_ids']) + len(row['target_token_ids']) - 1 for row in rows]
    need(all(2 <= length < 384 for length in lengths), 'NATIVE_CONTEXT_BOUNDS')
    gold = sum(len(row['target_token_ids']) for row in rows)
    need(data.get('gold_tokens') == gold and data.get('target_token_ids') == [row['target_token_ids'] for row in rows],
         'GOLD_SOURCE_COUNT_ALIGNMENT')
    profile = counter_profile(chunk, sum(lengths), gold)
    return {**profile, 'dataset_path': str(PREP_ROOT / f'chunk{chunk:02d}/dataset.json'),
            'dataset_receipt_sha256': data['receipt_sha256'], 'manifest_path': data['manifest'],
            'manifest_sha256': data['manifest_sha256'], 'metadata_path': data['gold_metadata'],
            'metadata_sha256': data['gold_metadata_sha256'], 'input_sha256': data['input_sha256'],
            'teacher_forced_lengths_in_local_order': lengths, 'target_token_ids': data['target_token_ids'],
            'historical_gold_output_sha256': data['baseline_output_sha256'],
            'baseline_routes': 'MUST_BE_FRESHLY_CAPTURED_BY_CHUNK_OFF;NOT_TRAIN16_ROUTE_HASH'}


def rewrite_completion_literals(cpp, c, profile):
    expected = counter_profile(profile['chunk'], profile['native_input_tokens'], profile['completion']['gold'])
    need(profile['completion'] == expected['completion'], 'EXACT_COMPLETION_PROFILE')
    need(cpp.count(CPP_ANCHOR) == c.count(C_ANCHOR) == 1, 'UNIQUE_EXACT_COMPLETION_ANCHORS')
    counts = profile['completion']
    new_cpp = (f"g_je01_one_tokens!={counts['tokens']} || g_je01_one_cases!=32 || "
               f"g_je01_one_layers!={counts['layers']})return 0;")
    new_c = (f"if(je01_jvp_gold_count!={counts['gold']} || "
             '!mlx_gpu_je01_one_column_jvp_finish(&tokens,&cases,&layers) ||')
    patched_cpp, patched_c = cpp.replace(CPP_ANCHOR, new_cpp, 1), c.replace(C_ANCHOR, new_c, 1)
    need(patched_cpp.count(new_cpp) == patched_c.count(new_c) == 1
         and patched_cpp.replace(new_cpp, CPP_ANCHOR, 1) == cpp
         and patched_c.replace(new_c, C_ANCHOR, 1) == c, 'ONLY_TWO_COMPLETION_LINES_MAY_CHANGE')
    changes = {'cpp': {'old': CPP_ANCHOR, 'new': new_cpp}, 'c': {'old': C_ANCHOR, 'new': new_c}}
    return patched_cpp, patched_c, changes


def apply_to_pinned_sources(cpp_bytes, c_bytes, profile):
    need(digest(cpp_bytes) == BASE_CPP and digest(c_bytes) == BASE_C, 'EXACT_PROVEN_PARENT_NATIVE_SOURCES_REQUIRED')
    cpp, c, changes = rewrite_completion_literals(cpp_bytes.decode(), c_bytes.decode(), profile)
    need(digest(cpp.replace(changes['cpp']['new'], CPP_ANCHOR, 1).encode()) == BASE_CPP
         and digest(c.replace(changes['c']['new'], C_ANCHOR, 1).encode()) == BASE_C,
         'SOURCE_MATH_AND_ALL_OTHER_GUARDS_MUST_RESTORE_BYTE_EXACTLY')
    return cpp.encode(), c.encode(), changes


def native_output_contract(c_bytes, root):
    text = c_bytes.decode()
    anchor = 'const char *je01_jvp_mode=getenv("JE01_NATIVE_JVP_MODE");'
    need(text.count(anchor) == 1, 'NATIVE_C_OPEN_ANCHOR')
    start = text.index(anchor)
    finish = text.find('int fd=open(je01_jvp_path,', start)
    need(finish > start, 'NATIVE_C_OPEN_END')
    block = text[start:finish]
    prefix = re.findall(r'const char prefix\[\]="([^"\n]+)";', block)
    expected = str(WORK) + '/je01_native_role_jvp_'
    need(prefix == [expected] and 'strncmp(je01_jvp_path,prefix,sizeof(prefix)-1)' in block
         and str(root).startswith(expected), 'NATIVE_C_OPEN_REJECTS_STATIC_ROOT')
    return {'base_c_sha256': BASE_C, 'required_absolute_prefix': expected, 'output_root': str(root),
            'native_guard_read_from_pinned_source': True}


def source_tree(root):
    need(root.is_dir() and root.resolve() == root and not root.is_symlink(), 'SOURCE_ROOT')
    files = {}
    for path in sorted(root.rglob('*')):
        need(not path.is_symlink(), 'SOURCE_TREE_SYMLINK_FORBIDDEN')
        if path.is_file():
            files[str(path.relative_to(root))] = {'sha256': sha(path), 'size_bytes': path.stat().st_size}
        else:
            need(path.is_dir(), 'SOURCE_TREE_SPECIAL_FILE_FORBIDDEN')
    need(files, 'EMPTY_SOURCE_TREE')
    return files


def require_source_only_phase(root, build_paths):
    """These receipts cannot describe a root that has already compiled or run."""
    paths = [Path(root) / 'runs'] + [Path(path) for path in build_paths]
    need(not any(path.exists() or path.is_symlink() for path in paths), 'BUILD_OR_NATIVE_RUN_ALREADY_EXISTS')


def verified_context(args):
    gate, upstream_plan = require_full48_gate(args)
    training, common = checked_helpers()
    preparation_path = PREP_ROOT / 'JE01_TRAIN240_INPUT_PREPARATION_RECEIPT.json'
    need(sha(preparation_path) == PREPARATION_FILE, 'PREPARATION_FILE_PIN')
    preparation = sealed(preparation_path, PREPARATION)
    need(preparation.get('prepared_case_ids') == list(range(16, 256))
         and preparation.get('prepared_training_cases') == 240 and preparation.get('prepared_gold_tokens') == 10368
         and preparation.get('native_model_forward') == 'NOT_RUN'
         and preparation.get('independent_validation_ood_accessed') is False, 'PREPARATION_SCOPE')
    profiles = []
    for index, claim in enumerate(preparation['chunks'], 1):
        need(index <= 15 and claim['chunk'] == index and claim['dataset_receipt_sha256'] == DATASET_PINS[index - 1]
             and claim['dataset_path'] == str(PREP_ROOT / f'chunk{index:02d}/dataset.json'), 'PREPARATION_CHUNK_PIN')
        data = sealed(Path(claim['dataset_path']), DATASET_PINS[index - 1])
        rows = training.validate_source_linkage(data)
        common.validate_dataset(data)  # Rehash the actual prepared input and metadata bytes too.
        profiles.append(derive_profile(data, rows))
    need(len(profiles) == 15 and sum(p['completion']['gold'] for p in profiles) == 10368
         and sum(p['native_input_tokens'] for p in profiles) == 33560, 'ALL_FIFTEEN_CHUNK_TOTALS')
    build = sealed(PARENT / 'NATIVE_BUILD_RECEIPT.json', PARENT_BUILD)
    need(build.get('binary_sha256') == BINARY and build.get('configuration', {}).get('patched_cpp_sha256') == BASE_CPP
         and build.get('configuration', {}).get('patched_c_sha256') == BASE_C
         and build.get('configuration', {}).get('adapter_sha256') == ADAPTER
         and build.get('cmake_sha256') == BASE_CMAKE and build.get('mlx_library_sha256') == MLX,
         'PARENT_BUILD_CONTRACT')
    library = Path(upstream_plan['runtime']['library_path'])
    pins = {PARENT / 'source/mlx_moe.cpp': BASE_CPP, PARENT / 'source/qwen_infer.c': BASE_C,
            PARENT / 'source/CMakeLists.txt': BASE_CMAKE,
            PARENT / 'source/je01_mlx_native_jvp_adapters.h': ADAPTER,
            PARENT / 'bin/je01_native_role_jvp': BINARY, library / 'libmlx.dylib': MLX,
            MODEL: AF4, BANK / 'JE01_ROLE_DIRECTION_BANK.bin': BANK_PIN,
            BANK / 'JE01_ROLE_DIRECTION_BANK.tsv': BANK_TSV}
    for path, pin in pins.items():
        need(sha(path) == pin, 'CURRENT_NATIVE_MODEL_BANK_OR_RUNTIME_PIN:' + str(path))
    need(MODEL.stat().st_size == 9814343680, 'FULL_ORIGINAL_MODEL_BYTES')
    sealed(BANK / 'JE01_ROLE_DIRECTION_BANK_RECEIPT.json', BANK_RECEIPT)
    smoke = sealed(SMOKE, SMOKE_PIN, field='canonical_sha256')
    need(smoke.get('status') == 'PASS' and smoke['runtime']['dylib_sha256'] == MLX, 'GUARDED_SMOKE_PIN')
    header_pins = {}
    need(set(smoke['runtime']['header_sha256']) == {'array.h', 'ops.h', 'transforms.h'}, 'MLX_HEADER_COVERAGE')
    for name, pin in smoke['runtime']['header_sha256'].items():
        path = library.parent / 'include/mlx' / name
        need(sha(path) == pin, 'INSTALLED_MLX_HEADER_DRIFT')
        header_pins[str(path)] = pin
    return {'gate': gate, 'profiles': profiles, 'library': library,
            'parent_source_tree': source_tree(PARENT / 'source'),
            'runtime_file_sha256': {str(p): pin for p, pin in pins.items()},
            'mlx_header_sha256': header_pins, 'cmake_sha256': sha(CMAKE)}


def manifest_payload(args, context):
    root = require_root(args.root)
    cpp_base, c_base = (PARENT / 'source/mlx_moe.cpp').read_bytes(), (PARENT / 'source/qwen_infer.c').read_bytes()
    output_contract = native_output_contract(c_base, root)
    chunks = []
    for profile in context['profiles']:
        cpp, c, changes = apply_to_pinned_sources(cpp_base, c_base, profile)
        source = root / f"static_builds/chunk{profile['chunk']:02d}/source"
        build = source.parent / 'build'
        tree = {**context['parent_source_tree'], 'mlx_moe.cpp': {'sha256': digest(cpp), 'size_bytes': len(cpp)},
                'qwen_infer.c': {'sha256': digest(c), 'size_bytes': len(c)}}
        chunks.append({**profile, 'source_path': str(source), 'build_path': str(build),
                       'expected_binary_path': str(build / 'je01_native_role_jvp'),
                       'expected_cpp_sha256': digest(cpp), 'expected_c_sha256': digest(c),
                       'expected_source_tree_sha256': canonical(tree), 'completion_replacements': changes,
                       'expected_source_total_bytes': sum(item['size_bytes'] for item in tree.values()),
                       'compile_commands': [[str(CMAKE), '-S', str(source), '-B', str(build), '-G', 'Ninja',
                                             '-DCMAKE_BUILD_TYPE=Release',
                                             '-DMLX_CMAKE_DIR=' + str(context['library'].parent / 'share/cmake')],
                                            [str(CMAKE), '--build', str(build), '--target', 'qwen_infer_gpu', '--parallel', '2']]})
    require_source_only_phase(root, [chunk['build_path'] for chunk in chunks])
    return {'schema': MANIFEST_SCHEMA, 'scope': SCOPE, 'status': 'STATIC_SOURCE_BUILD_MANIFEST_SEALED_NO_BUILD_NO_INFERENCE',
            'root': str(root), 'prepare_source_sha256': sha(Path(__file__)), 'upstream_gate': context['gate'],
            'helper_source_sha256': HELPERS, 'preparation_receipt_sha256': PREPARATION,
            'preparation_file_sha256': PREPARATION_FILE, 'parent_source_root': str(PARENT / 'source'),
            'parent_build_receipt_sha256': PARENT_BUILD, 'base_cpp_sha256': BASE_CPP, 'base_c_sha256': BASE_C,
            'parent_source_files': context['parent_source_tree'],
            'parent_source_tree_sha256': canonical(context['parent_source_tree']),
            'parent_source_total_bytes': sum(item['size_bytes'] for item in context['parent_source_tree'].values()),
            'planned_source_copy_bytes': sum(chunk['expected_source_total_bytes'] for chunk in chunks),
            'source_stage_capacity_rule': {'required_free_bytes': 'REMAINING_UNSTAGED_SOURCE_BYTES_PLUS_HEADROOM',
                                            'headroom_bytes': SOURCE_STAGE_HEADROOM_BYTES,
                                            'compilation_and_native_outputs_not_covered': True},
            'runtime_file_sha256': context['runtime_file_sha256'], 'mlx_header_sha256': context['mlx_header_sha256'],
            'cmake_path': str(CMAKE), 'cmake_sha256': context['cmake_sha256'],
            'native_output_required_prefix': str(WORK) + '/je01_native_role_jvp_',
            'native_output_path_contract': output_contract,
            'allowed_source_change': 'TWO_EXACT_COMPLETION_LINES_ONLY;TOKENS_LAYERS_GOLD_CHANGE;CASES32_UNCHANGED',
            'jvp_math_and_other_guards_byte_preserved': True, 'installed_mlx_modified': False,
            'total_training_sentences': 240, 'global_case_ids': list(range(16, 256)),
            'native_local_case_ids_per_chunk': list(range(16)), 'total_gold_tokens': 10368,
            'total_native_input_tokens': 33560, 'per_direction_total_completion': {
                'tokens': 67120, 'cases': 480, 'layers': 1812240, 'gold': 10368},
            'future_collection_requires': ['FIFTEEN_SUCCESSFUL_EXACT_SOURCE_BUILD_RECEIPTS',
                                            'ONE_FRESH_OFF_PER_CHUNK_MATCHING_HISTORICAL_GOLD_SHA',
                                            'PER_CHUNK_NEW_FULL_PREFIX_ROUTES;NEVER_TRAIN16_ROUTE_HASH',
                                            '48_ALPHA_ZERO_FULL_PREFIX_JVPS_PER_CHUNK',
                                            'RAW_10368_BY48_MATRIX_AND_240_SENTENCE_CLUSTER_AUDIT'],
            'chunks': chunks, **NO_RUN}


def manifest(args):
    context = verified_context(args)  # Full independent PASS is the first executable gate.
    root = require_root(args.root)
    root.mkdir(mode=0o700, exist_ok=True)
    with exclusive_root(root):
        payload = manifest_payload(args, context)
        result = seal_once(root / 'TRAIN240_STATIC_BUILD_MANIFEST.json', payload, resume=args.resume)
    return result


def stage(args):
    context = verified_context(args)  # Recheck the gate even when a manifest already exists.
    root = require_root(args.root)
    need(root.is_dir() and is_sha(args.expected_manifest_receipt), 'EXTERNAL_BUILD_MANIFEST_PIN_REQUIRED')
    with exclusive_root(root):
        saved = sealed(root / 'TRAIN240_STATIC_BUILD_MANIFEST.json', args.expected_manifest_receipt)
        expected = manifest_payload(args, context)
        need({k: v for k, v in saved.items() if k != 'receipt_sha256'} == expected, 'MANIFEST_RECOMPUTATION_DRIFT')
        cpp_base, c_base = (PARENT / 'source/mlx_moe.cpp').read_bytes(), (PARENT / 'source/qwen_infer.c').read_bytes()
        remaining = 0
        for chunk in saved['chunks']:
            path = Path(chunk['source_path'])
            if path.exists() or path.is_symlink():
                need(args.resume and canonical(source_tree(path)) == chunk['expected_source_tree_sha256'],
                     'EXISTING_SOURCE_REQUIRES_EXACT_RESUME')
            else:
                remaining += chunk['expected_source_total_bytes']
        need(shutil.disk_usage(root).free >= remaining + SOURCE_STAGE_HEADROOM_BYTES,
             'INSUFFICIENT_DISK_FOR_EXACT_SOURCE_COPIES_AND_HEADROOM')
        staged = []
        for chunk in saved['chunks']:
            path = Path(chunk['source_path'])
            need(path.is_relative_to(root) and path.resolve() == path and not path.is_symlink(), 'SOURCE_OUTPUT_PATH')
            cpp, c, _ = apply_to_pinned_sources(cpp_base, c_base, chunk)
            if path.exists():
                need(args.resume, 'SOURCE_ALREADY_EXISTS_REQUIRE_RESUME')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(PARENT / 'source', path, symlinks=False)
                (path / 'mlx_moe.cpp').write_bytes(cpp)
                (path / 'qwen_infer.c').write_bytes(c)
            files = source_tree(path)
            need(canonical(files) == chunk['expected_source_tree_sha256']
                 and files['mlx_moe.cpp']['sha256'] == chunk['expected_cpp_sha256']
                 and files['qwen_infer.c']['sha256'] == chunk['expected_c_sha256'], 'EXACT_STAGED_SOURCE_TREE')
            staged.append({'chunk': chunk['chunk'], 'source_path': str(path),
                           'source_tree_sha256': canonical(files), 'cpp_sha256': files['mlx_moe.cpp']['sha256'],
                           'c_sha256': files['qwen_infer.c']['sha256'], 'completion': chunk['completion']})
        # Reopen the original parent after all copies: no parent mutation is allowed.
        need(source_tree(PARENT / 'source') == context['parent_source_tree'], 'PARENT_SOURCE_CHANGED_DURING_STAGE')
        result = seal_once(root / 'TRAIN240_STATIC_SOURCE_STAGE_RECEIPT.json', {
            'schema': STAGE_SCHEMA, 'scope': SCOPE, 'status': 'FIFTEEN_STATIC_COUNTER_SOURCES_STAGED_NOT_COMPILED',
            'root': str(root), 'manifest_receipt_sha256': saved['receipt_sha256'],
            'prepare_source_sha256': sha(Path(__file__)), 'upstream_gate': context['gate'],
            'parent_source_tree_sha256': saved['parent_source_tree_sha256'],
            'exact_source_only_change_verified': True, 'chunks': staged, **NO_RUN}, resume=args.resume)
    return result


def specification():
    return {'schema': 'beglin-je01-train240-static-counter-preparation-spec/1',
            'status': 'DRAFT_SPEC_NOT_A_SEALED_EXPERIMENT_PLAN', 'scope': SCOPE,
            'required_upstream': 'ACTUAL_INDEPENDENT_ALL48_TRAIN16_PASS_WITH_EXTERNAL_CANONICAL_AND_FILE_PINS',
            'base_cpp_sha256': BASE_CPP, 'base_c_sha256': BASE_C,
            'exact_cpp_anchor': CPP_ANCHOR, 'exact_c_anchor': C_ANCHOR,
            'chunks': [counter_profile(i, INPUT_COUNTS[i - 1], GOLD_COUNTS[i - 1]) for i in range(1, 16)],
            **NO_RUN}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('spec', 'manifest', 'stage'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--expected-all48-audit-receipt')
    parser.add_argument('--expected-all48-audit-file-sha')
    parser.add_argument('--expected-manifest-receipt')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = specification() if args.action == 'spec' else manifest(args) if args.action == 'manifest' else stage(args)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
