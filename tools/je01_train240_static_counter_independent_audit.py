#!/usr/bin/env python3
"""Independent TRAIN240 static-counter source audit; never build or infer.

The read-only evidence path is gated by an externally pinned complete all48
TRAIN16 independent audit.  The local metadata review and synthetic self-test
do not open TRAIN240 input binaries and cannot return a native-result PASS.
Only the three numeric completion constants (tokens, layers, gold) may differ
from the proven source; the case/reset counter remains exactly 32.
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
ROOT = WORK / 'je01_native_role_jvp_train240_static_20261123_v1'
PARENT = WORK / 'je01_native_role_jvp_followup_20261123_v1'
ALL48 = WORK / 'je01_native_role_jvp_all48_train16_20261123_v2'
INPUT_ROOT = WORK / 'je01_role_directional_20261123_v1'
PREPARED = INPUT_ROOT / 'train240_inputs'
ORIGINAL_INPUTS = WORK / 'je01_real_original256_activation_20261122_v1/portable512/je01_inputs'
GOLD_PATH = INPUT_ROOT / 'train_gold_source/JE01_ORIGINAL_REAL_TRAIN256_FULL_GOLD_IDS.json'
BASELINE_PATH = INPUT_ROOT / 'source/docs/research/JE01_ORIGINAL_AF_REAL256_FULL_TARGET_GOLD_NLL_20261118.json'
META_MAGIC = b'JE01_ORIGINAL_FULL_GOLD_TARGETS_META_V1\n'
NATIVE_PREFIX = '/Volumes/D50/TailnetCommander/work/je01_native_role_jvp_'
MODEL = WORK / 'hcp-p18-external-multimetric-20261015/native_moe_base_source_copy/weights_moe/deepseek_moe_af.bin'
BANK = WORK / 'je01_role_direction_bank_20261123_v1'
CMAKE = WORK / 'je01_postnll_bc_native_compile_20261123_v1/venv/bin/cmake'
SMOKE_PATH = WORK / 'je01_native_jvp_smoke_20261123_v4/JE01_NATIVE_JVP_SMOKE_RECEIPT.json'
SCOPE = 'TRAIN240_ROWS16_THROUGH255_STATIC_COMPLETION_SOURCE_PREPARATION_ONLY'
NO_RUN = {'native_compilation': 'NOT_RUN', 'native_model_forward': 'NOT_RUN', 'derivative_collection': 'NOT_RUN',
          'allocation_selected': False, 'independent_validation_ood_accessed': False, 'production_modified': False}
PIN = {
    'all48_plan': '260c199a18f1b60bf94a6388fd9df306803a950383a199bd2665b88a6d553aba',
    'all48_runner': 'ad07f04937a79868a67f8eacedee22b6fe091399cf0557e5d2575234ebc50df6',
    'all48_auditor': 'a3cf5a2dc8048f51208f8a44ce3038eb51b9f44147ba4f62bb8a65aac0524d23',
    'parent_build': '2cbeb4dc1a7e1ab5642a4abb22395498c10dde43bee051eec3c303c7e53fa008',
    'cpp': '6055ca3b261075fe5fcf76e5829f0fb20b8ecdd52599248f4b9b1032cfbeb49f',
    'c': '5e2a6ea76c22c1ec0ff9f2fb6d093b7d3a471679adbe439ed1c4b898a56a52a6',
    'cmake': '1479cf49e18da616e6fcfdde2ed9433b468701a3d5a4482fa7e9ee398683da19',
    'adapter': '9ae9d98b921334bae321e583908e71bdc9c08178cbf3ca5d7a1d4d4e552a09ec',
    'binary': '110b1978d97bcd7c6bdfd6d42142569e56e9dacffa4ff462dc0786b3128e49f4',
    'model': '9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4',
    'mlx_library': 'afb1939edc96dc10e452132e4e665276fe0c48bf8cdc1a69286feb8a14de1da0',
    'gold': '355a169cfb7c9ab79901ae7848afa21a2f5a576d048a5476f527bca699136fb7',
    'gold_file': 'b86e08bb54f65d0fa29dd545801d2fb3ef0bfb45bc1993743f21cd30cde7d7fd',
    'preparation': 'f0a95e7ee7fbb1ce001c419c8626744e3730e85b235b9a726897ae38f946e477',
    'preparation_file': '4dfb79c79bc564eaf1f7372c33b80fcaa3aba966ff280d0ac66ecb9ace565b50',
    'native_inputs': '2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34',
    'baseline': 'f71a4c4397cbadf938c7a9631ba8306547b31e33028dd9a54b556a1a758e3f41',
    'frozen_manifest': 'a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147',
    'metadata_review_file': '3f18b5fe9232133f260dcbddd8e35144c6994f80857f3c6c5e4d948429d098fa',
    'restored_helper': '99f4d6da38880e3eaadcd8a32e9f288a75ae87b7f13c9721fe00109616b7175b',
    'common_helper': '5607c0afef3bd4b1a61bcab6622267acc6b6137fc65ba0e79873e92e31d7e9e6',
    'bank': '3b18257d22caddd0916292bef692110f9eeaf0edf83ad3fd4d5417fdf3443bb8',
    'bank_tsv': 'fdc801f59f9c6438dd60bb33fab9a1332cb19c6e15eb15bef7425b7fbd522e4f',
    'bank_receipt': '4994ec762b3a9b8a44015bad18f4b5bc7ea26505208384146238e7def28464d4',
    'smoke': 'b0825ab7a048f7d95b155592760e10d73e3df2b112230adbf7b348f388ca1595',
}
CHUNK_DATASET_PINS = (
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
COUNTS = ((634, 2226), (702, 2193), (664, 2252), (784, 2432), (791, 2489),
          (772, 2403), (589, 1822), (646, 2142), (742, 2558), (698, 2184),
          (660, 2154), (700, 2247), (776, 2553), (617, 1998), (593, 1907))
CPP_ANCHOR = b'g_je01_one_tokens!=4334 || g_je01_one_cases!=32 || g_je01_one_layers!=117018)return 0;'
C_ANCHOR = b'if(je01_jvp_gold_count!=656 || !mlx_gpu_je01_one_column_jvp_finish(&tokens,&cases,&layers) ||'


def need(condition, label):
    if not condition:
        raise ValueError('JE01_TRAIN240_STATIC_INDEPENDENT_AUDIT:' + label)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def sha_string(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def exact(actual, expected, label):
    need(type(actual) is type(expected), label + ':type')
    if isinstance(expected, dict):
        need(actual.keys() == expected.keys(), label + ':keys')
        for key in expected:
            exact(actual[key], expected[key], label + '.' + key)
    elif isinstance(expected, list):
        need(len(actual) == len(expected), label + ':length')
        for index, value in enumerate(expected):
            exact(actual[index], value, label + '[' + str(index) + ']')
    else:
        need(actual == expected, label + ':value')


def strict_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            need(key not in result, 'duplicate_json_key:' + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('nonfinite_json_constant:' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def sealed(value, pin=None, field='receipt_sha256'):
    need(isinstance(value, dict) and sha_string(value.get(field)), 'receipt_type')
    body = {key: item for key, item in value.items() if key != field}
    need(value[field] == canonical(body), 'receipt_integrity')
    need(pin is None or value[field] == pin, 'external_receipt_pin')
    return value


def safe_path(path, directory=False):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink(), 'unsafe_path:' + str(path))
    need(path.is_dir() if directory else path.is_file(), 'missing_path:' + str(path))
    return path


class Evidence:
    def __init__(self):
        self.files = {}

    def read(self, path, expected=None, collect=True):
        path = safe_path(path)
        before = path.stat()
        digest, pieces = hashlib.sha256(), []
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
                if collect:
                    pieces.append(block)
        after = path.stat()
        need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
             'file_changed_while_reading:' + str(path))
        pin = digest.hexdigest()
        need(expected is None or pin == expected, 'file_sha256:' + str(path))
        entry = {'sha256': pin, 'size_bytes': after.st_size}
        need(str(path) not in self.files or self.files[str(path)] == entry, 'file_changed_between_reads')
        self.files[str(path)] = entry
        return b''.join(pieces) if collect else None

    def receipt(self, path, pin=None, file_pin=None, *, field='receipt_sha256'):
        return sealed(strict_json(self.read(path, file_pin)), pin, field)


def chunk_counts(chunk, gold_tokens=None, native_input_tokens=None):
    need(type(chunk) is int and 1 <= chunk <= 15, 'chunk_range')
    expected_gold, expected_inputs = COUNTS[chunk - 1]
    gold_tokens = expected_gold if gold_tokens is None else gold_tokens
    native_input_tokens = expected_inputs if native_input_tokens is None else native_input_tokens
    need(type(gold_tokens) is int and type(native_input_tokens) is int and
         (gold_tokens, native_input_tokens) == (expected_gold, expected_inputs), 'chunk_source_counts')
    return {'tokens': 2 * native_input_tokens, 'cases': 32,
            'layers': 54 * native_input_tokens, 'gold': gold_tokens,
            'route_events': 52 * native_input_tokens}


def verify_native_output_guard(source, root):
    root = Path(root)
    need(root.is_absolute() and root.parent == WORK and str(root).startswith(NATIVE_PREFIX), 'native_output_root')
    anchor = b'const char *je01_jvp_mode=getenv("JE01_NATIVE_JVP_MODE");'
    opener = b'int fd=open(je01_jvp_path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);'
    need(source.count(anchor) == source.count(opener) == 1, 'native_open_anchors')
    start, end = source.index(anchor), source.index(opener)
    need(start < end, 'native_open_order')
    block = source[start:end]
    need(block.count(('const char prefix[]="' + NATIVE_PREFIX + '";').encode()) == 1 and
         b'strncmp(je01_jvp_path,prefix,sizeof(prefix)-1)' in block and
         b'strcmp(je01_jvp_mode,"ROLE_DIRECTION_FIXED_Q4Q5")' in block and
         b'!je01_jvp_path' in block and b'!je01_gold_file' in block and b'exit(1)' in block,
         'native_open_guard_weakened')


def expected_counter_sources(base_cpp, base_c, counts):
    """Independent exact bytes construction; never accept a broad replacement."""
    need(set(counts) == {'tokens', 'cases', 'layers', 'gold', 'route_events'} and
         all(type(value) is int and value > 0 for value in counts.values()), 'counter_types')
    need(counts['cases'] == 32 and counts['tokens'] % 2 == 0 and
         counts['layers'] == 27 * counts['tokens'] and
         counts['route_events'] == 26 * counts['tokens'], 'counter_two_pass_invariants')
    need(base_cpp.count(CPP_ANCHOR) == 1 and base_c.count(C_ANCHOR) == 1, 'unique_completion_anchors')
    cpp_new = ('g_je01_one_tokens!=' + str(counts['tokens']) +
               ' || g_je01_one_cases!=32 || g_je01_one_layers!=' + str(counts['layers']) + ')return 0;').encode()
    c_new = ('if(je01_jvp_gold_count!=' + str(counts['gold']) +
             ' || !mlx_gpu_je01_one_column_jvp_finish(&tokens,&cases,&layers) ||').encode()
    patched_cpp, patched_c = base_cpp.replace(CPP_ANCHOR, cpp_new, 1), base_c.replace(C_ANCHOR, c_new, 1)
    need(patched_cpp.count(cpp_new) == 1 and patched_c.count(c_new) == 1 and
         patched_cpp.replace(cpp_new, CPP_ANCHOR, 1) == base_cpp and
         patched_c.replace(c_new, C_ANCHOR, 1) == base_c, 'completion_patch_reverse_exact')
    return patched_cpp, patched_c


def verify_counter_sources(base_cpp, base_c, staged_cpp, staged_c, counts):
    expected_cpp, expected_c = expected_counter_sources(base_cpp, base_c, counts)
    need(staged_cpp == expected_cpp and staged_c == expected_c, 'only_three_completion_constants_may_change')
    return {'cpp_sha256': sha_bytes(staged_cpp), 'c_sha256': sha_bytes(staged_c),
            'exact_reverse_base_cpp_sha256': sha_bytes(base_cpp), 'exact_reverse_base_c_sha256': sha_bytes(base_c),
            'changed_numeric_constants': ['tokens', 'layers', 'gold'], 'case_counter_unchanged': 32,
            'all_other_source_bytes_identical': True}


def review_metadata(path):
    evidence = Evidence()
    data = strict_json(evidence.read(path, PIN['metadata_review_file']))
    need(data['schema'] == 'beglin-je01-train240-source-metadata-review/1' and
         data['status'] == 'SOURCE_METADATA_REVIEWED_NO_TRAIN240_NATIVE_RUN', 'review_scope')
    exact(data['totals'], {'sentences': 240, 'gold_tokens': 10368, 'prompt_bos_tokens': 23432,
                          'native_input_tokens': 33560}, 'review_totals')
    need(len(data['chunks']) == 15, 'review_fifteen_chunks')
    result = []
    for chunk, row in enumerate(data['chunks'], 1):
        gold, inputs = COUNTS[chunk - 1]
        expected = {'chunk': chunk, 'case_first': 16 * chunk, 'case_last': 16 * chunk + 15,
                    'gold_tokens': gold, 'prompt_bos_tokens': inputs - gold + 16,
                    'native_input_tokens': inputs, 'dataset_receipt_sha256': CHUNK_DATASET_PINS[chunk - 1]}
        exact(row, expected, 'metadata_chunk')
        result.append({**row, 'expected_counters': chunk_counts(chunk)})
    need(sum(row['gold_tokens'] for row in result) == 10368 and
         sum(row['native_input_tokens'] for row in result) == 33560, 'review_sum_counts')
    return {'schema': 'beglin-je01-train240-static-counter-metadata-review/1',
            'status': 'PREPARATION_ONLY_NO_NATIVE_RESULT', 'chunks': result,
            'metadata_file_sha256': PIN['metadata_review_file'], 'source_raw_binary_reverified': False,
            'source_gold_and_preparation_reopened': False, 'all48_gate_claimed': False,
            'source_staged': False, 'compiled': False, 'native_model_forward': 'NOT_RUN',
            'independent_validation_ood_accessed': False, 'production_modified': False}


def verify_all48_gate(evidence, expected_canonical, expected_file):
    """Must be called before opening any original/prepared TRAIN240 binary."""
    need(sha_string(expected_canonical) and sha_string(expected_file), 'external_all48_pins_required')
    path = ALL48 / 'JE01_ALL48_TRAIN16_INDEPENDENT_AUDIT.json'
    gate = evidence.receipt(path, expected_canonical, expected_file)
    expected = {
        'schema': 'beglin-je01-native-all48-train16-independent-raw-audit/2',
        'status': 'INDEPENDENT_AUDIT_CONFIRMED_ALL48_TRAIN16_RAW_AND_NINE_CHECKPOINTS_PASS',
        'scope': 'PREVIOUSLY_SEEN_TRAIN16_FIXED_HARD_ROUTE_48_DIRECTION_DIAGNOSTIC_ONLY',
        'root': str(ALL48),
        'evidence_integrity': 'PASS', 'experiment_numerical_verdict': 'PASS',
        'completed_native_cell_count': 79, 'completed_evidence_cell_count': 79,
        'native_cells_executed_in_this_plan': 78, 'inherited_native_cell_count': 1,
        'native_jvp_columns': 48, 'fd_verified_coordinates': 9, 'all_48_raw_directions_verified': True,
        'all_nine_fd_checkpoints_pass': True, 'plan_receipt_sha256': PIN['all48_plan'],
        'runner_source_sha256': PIN['all48_runner'], 'auditor_source_sha256': PIN['all48_auditor'],
        'proven_native_build_receipt_sha256': PIN['parent_build'],
        'original_af4_full_file_sha_verified_after_run': PIN['model'],
        'numerical_hold_receipt_sha256': None, 'allocation_selected': False,
        'independent_validation_ood_accessed': False, 'production_modified': False,
    }
    for key, value in expected.items():
        exact(gate.get(key), value, 'all48_gate.' + key)
    need(sha_string(gate.get('source_report_receipt_sha256')) and sha_string(gate.get('raw_matrix_receipt_sha256')),
         'all48_complete_artifact_bindings')
    need(gate.get('independent_psd_check', {}).get('passed') is True, 'all48_psd_pass')
    evidence.read(ALL48 / 'tools/je01_native_all48_train16_independent_audit_v2.py', PIN['all48_auditor'], collect=False)
    evidence.read(ALL48 / 'tools/je01_native_all48_train16_runner_v2.py', PIN['all48_runner'], collect=False)
    plan = evidence.receipt(ALL48 / 'PLAN_RECEIPT.json', PIN['all48_plan'])
    analysis_path = ALL48 / 'JE01_ALL48_TRAIN16_ANALYSIS.json'
    matrix_path = ALL48 / 'RAW_DERIVATIVE_MATRIX.json'
    analysis = evidence.receipt(analysis_path, gate['source_report_receipt_sha256'])
    matrix = evidence.receipt(matrix_path, gate['raw_matrix_receipt_sha256'])
    need(analysis['status'] == 'ALL48_NATIVE_COMPLETE_CHECKPOINTS_AND_PSD_PASS' and
         analysis['plan_receipt_sha256'] == matrix['plan_receipt_sha256'] == PIN['all48_plan'] and
         matrix['shape'] == [656, 48] and canonical(matrix['per_token_dnll']) == gate['reconstructed_matrix_values_sha256'],
         'upstream_analysis_raw_matrix_binding')
    for path in (analysis_path, matrix_path):
        exact(evidence.files[str(path)], gate['verified_files'][str(path)], 'all48_audited_artifact_bytes')
    return gate, plan, {
        'audit_receipt_sha256': expected_canonical, 'audit_file_sha256': expected_file,
        'audit_path': str(ALL48 / 'JE01_ALL48_TRAIN16_INDEPENDENT_AUDIT.json'),
        'plan_receipt_sha256': PIN['all48_plan'], 'runner_source_sha256': PIN['all48_runner'],
        'auditor_source_sha256': PIN['all48_auditor'], 'analysis_receipt_sha256': analysis['receipt_sha256'],
        'raw_matrix_receipt_sha256': matrix['receipt_sha256'], 'status': 'INDEPENDENT_ALL48_TRAIN16_GATE_PASS'}


def source_documents(evidence):
    preparation = evidence.receipt(PREPARED / 'JE01_TRAIN240_INPUT_PREPARATION_RECEIPT.json',
                                   PIN['preparation'], PIN['preparation_file'])
    gold = evidence.receipt(GOLD_PATH, PIN['gold'], PIN['gold_file'])
    baseline = evidence.receipt(BASELINE_PATH, PIN['baseline'])
    native = evidence.receipt(ORIGINAL_INPUTS / 'JE01_NATIVE_384_INPUTS_RECEIPT.json', PIN['native_inputs'])
    need(preparation['schema'] == 'beglin-je01-native-role-directional-train240-source-preparation/1' and
         preparation['status'] == 'TRAIN256_ALL_SOURCE_VERIFIED_TRAIN240_FIFTEEN_CHUNKS_PREPARED', 'preparation_status')
    for key, value in {'validated_original_training_cases': 256, 'prepared_training_cases': 240,
                       'prepared_gold_tokens': 10368, 'prepared_case_ids': list(range(16, 256)),
                       'native_model_forward': 'NOT_RUN', 'independent_validation_ood_accessed': False,
                       'production_modified': False}.items():
        exact(preparation.get(key), value, 'preparation.' + key)
    need(gold['status'] == 'TRAIN256_FULL_GOLD_SOURCE_READY' and gold['original_256_source_rows'] == 256 and
         gold['aligned_to_original_prompt_prefix_count'] == 256 and gold['same_source_validation_and_ood_labels_used'] is False,
         'gold_train_only_source')
    need(native['total_input_records'] == 512 and native['bos_id'] == 100000 and native['native_context_positions'] == 384 and
         native['source_manifest_sha256'] == PIN['frozen_manifest'], 'native_input_provenance')
    need(baseline['status'] == 'REAL_ORIGINAL_AF_TRAIN_FULL_TARGET_NLL_CAL256_PASS' and
         baseline['source_gold_sha256'] == PIN['gold'] and baseline['original_512_input_sha256'] == PIN['native_inputs'] and
         baseline['n_calibration_train_cases'] == 256 and baseline['gold_full_target_tokens_scored'] == 11024 and
         baseline['training_only'] is True and baseline['heldout_gold_labels_used'] is False, 'historical_full256_baseline')
    need(len(native['cohorts']['calibration']['file_receipts']) == len(gold['cases']) ==
         len(baseline['per_case_real_NLL']) == 256, 'source_256_rows')
    verify_source_row_identity(native['cohorts']['calibration']['file_receipts'],
                               gold['cases'], baseline['per_case_real_NLL'])
    need(len(preparation['chunks']) == 15 and len(baseline['per_chunk_native_output_sha256']) == 16,
         'source_chunk_coverage')
    for chunk, row in enumerate(preparation['chunks'], 1):
        expected = {'chunk': chunk, 'dataset_id': f'train240_chunk{chunk:02d}',
                    'dataset_path': str(PREPARED / f'chunk{chunk:02d}/dataset.json'),
                    'dataset_receipt_sha256': CHUNK_DATASET_PINS[chunk - 1],
                    'case_ids': list(range(chunk * 16, chunk * 16 + 16)), 'gold_tokens': COUNTS[chunk - 1][0]}
        exact(row, expected, 'source_preparation_chunk')
    need(sum(row['gold_token_count'] for row in gold['cases']) == 11024 and
         sum(row['gold_token_count'] for row in gold['cases'][:16]) == 656 and
         sum(row['gold_token_count'] for row in gold['cases'][16:]) == 10368, 'original_gold_partition')
    for chunk, row in enumerate(baseline['per_chunk_native_output_sha256']):
        need(row['chunk'] == chunk and sha_string(row['result_sha256']), 'historical_chunk_pin')
    nll_sum = math.fsum(row['nll_sum'] for row in baseline['per_case_real_NLL'])
    need(abs(nll_sum / 11024 - baseline['mean_gold_token_NLL']) < 1e-12 and
         abs(baseline['mean_gold_token_NLL'] - 2.46035653719724) < 1e-13, 'historical_baseline_token_weighting')
    return preparation, gold, baseline, native


def verify_source_row_identity(prompts, gold_cases, baseline_cases):
    """Metadata check across TRAIN256; opens no excluded prompt or heldout file."""
    need(len(prompts) == len(gold_cases) == len(baseline_cases) == 256, 'complete_original_source_rows')
    source_ids = [row['input_id'] for row in prompts]
    need(all(sha_string(value) for value in source_ids) and len(set(source_ids)) == 256,
         'training_source_ids_must_be_unique_including_excluded_train16')
    for index, (source_id, gold, baseline) in enumerate(zip(source_ids, gold_cases, baseline_cases)):
        exact(gold['row'], index, 'original_gold_row_index')
        exact(baseline['row'], index, 'original_baseline_row_index')
        need(source_id == gold['input_id'] == baseline['source_id'], 'full256_prompt_gold_baseline_source_link')
    need(set(source_ids[:16]).isdisjoint(source_ids[16:]), 'train16_source_id_leak')


def source_linkage_contract():
    return {'native_input_root': str(ORIGINAL_INPUTS),
            'native_input_receipt_path': str(ORIGINAL_INPUTS / 'JE01_NATIVE_384_INPUTS_RECEIPT.json'),
            'native_input_receipt_sha256': PIN['native_inputs'], 'gold_source_path': str(GOLD_PATH),
            'gold_source_receipt_sha256': PIN['gold'], 'baseline_source_path': str(BASELINE_PATH),
            'baseline_source_receipt_sha256': PIN['baseline'], 'frozen_manifest_sha256': PIN['frozen_manifest'],
            'all_256_training_source_rows_validated': True, 'excluded_diagnostic_case_ids': list(range(16))}


def decode_training_row(evidence, prompt, gold, baseline, global_row):
    need(type(global_row) is int and 16 <= global_row < 256, 'train16_or_heldout_row_forbidden')
    source_id = prompt['input_id']
    need(sha_string(source_id), 'input_id')
    exact(gold['row'], global_row, 'gold_global_row')
    exact(baseline['row'], global_row, 'baseline_global_row')
    need(source_id == gold['input_id'] == baseline['source_id'], 'original_row_identity')
    raw = evidence.read(ORIGINAL_INPUTS / 'calibration' / (source_id + '.bin'), prompt['native_int32_sha256'])
    need(sha_bytes(raw) == gold['native_BOS_prompt_sha256'] and len(raw) == prompt['native_int32_bytes'] and
         len(raw) >= 8 and len(raw) % 4 == 0, 'original_prompt_byte_identity')
    tokens = list(struct.unpack('<' + 'i' * (len(raw) // 4), raw))
    targets = gold['gold_continuation_token_ids']
    need(tokens[0] == 100000 and all(0 <= token < 102400 for token in tokens) and
         isinstance(targets, list) and targets and all(type(token) is int and 0 <= token < 102400 for token in targets),
         'source_token_domain')
    need(len(tokens) == gold['original_BOS_prompt_positions'] == prompt['token_count_with_bos'] and
         len(targets) == gold['gold_token_count'] == baseline['tokens'] and
         len(tokens) + len(targets) == gold['model_full_BOS_sequence_positions'] <= 384, 'source_prefix_target_lengths')
    need(all(type(baseline[key]) in (int, float) and math.isfinite(baseline[key]) and baseline[key] >= 0
             for key in ('first_gold_nll', 'nll_sum')), 'historical_nll_domain')
    prepared_tokens = tokens + targets[:-1]
    prepared = struct.pack('<' + 'i' * len(prepared_tokens), *prepared_tokens)
    expected_metadata = struct.pack('<II384i', len(tokens), len(targets), *(targets + [-1] * (384 - len(targets))))
    source_record = {'row': global_row, 'input_id': source_id,
                     'original_prompt_path': str(ORIGINAL_INPUTS / 'calibration' / (source_id + '.bin')),
                     'original_prompt_sha256': sha_bytes(raw), 'original_prompt_positions': len(tokens),
                     'gold_tokens': len(targets), 'original_baseline_first_gold_nll': baseline['first_gold_nll'],
                     'original_baseline_nll_sum': baseline['nll_sum']}
    return {'global_case_id': global_row, 'source_id': source_id, 'prompt_tokens': len(tokens),
            'gold_tokens': len(targets), 'input_tokens': len(prepared_tokens), 'targets': targets,
            'prepared_bytes': prepared, 'metadata_record': expected_metadata, 'case_source_record': source_record}


def verify_datasets(evidence, documents):
    preparation, source_gold, baseline, native = documents
    exact(preparation['source_linkage'], source_linkage_contract(), 'preparation_source_linkage')
    chunks, source_ids, global_rows = [], set(), []
    for chunk in range(1, 16):
        folder = PREPARED / f'chunk{chunk:02d}'
        dataset = evidence.receipt(folder / 'dataset.json', CHUNK_DATASET_PINS[chunk - 1])
        need(dataset['schema'] == 'beglin-je01-role-directional-train-input/1' and
             dataset['scope'] == 'TRAIN240' and dataset['dataset_id'] == f'train240_chunk{chunk:02d}' and
             dataset['chunk'] == chunk, 'dataset_scope_id')
        expected_rows = list(range(chunk * 16, chunk * 16 + 16))
        exact(dataset['case_ids'], expected_rows, 'dataset_global_rows')
        exact(dataset['source_linkage'], source_linkage_contract(), 'dataset_source_linkage')
        for key, value in {'baseline_output_is_historical_reference_not_new_execution': True,
                           'independent_validation_ood_accessed': False, 'native_model_forward': 'NOT_RUN',
                           'production_modified': False}.items():
            exact(dataset.get(key), value, 'dataset.' + key)
        rows = [decode_training_row(evidence, native['cohorts']['calibration']['file_receipts'][index],
                                   source_gold['cases'][index], baseline['per_case_real_NLL'][index], index)
                for index in expected_rows]
        need(all(row['source_id'] not in source_ids for row in rows) and len({row['source_id'] for row in rows}) == 16,
             'duplicate_source_training_row')
        source_ids.update(row['source_id'] for row in rows)
        global_rows.extend(row['global_case_id'] for row in rows)
        exact(dataset['target_token_ids'], [row['targets'] for row in rows], 'dataset_targets')
        exact(dataset['case_sources'], [row['case_source_record'] for row in rows], 'dataset_case_source_record')
        exact(dataset['input_sha256'], [sha_bytes(row['prepared_bytes']) for row in rows], 'derived_input_hashes')
        exact(dataset['manifest'], str(folder / 'manifest.txt'), 'dataset_manifest_path')
        exact(dataset['gold_metadata'], str(folder / 'gold_meta.bin'), 'dataset_metadata_path')
        expected_manifest = ''.join(str(PREPARED / 'inputs' / (row['source_id'] + '.bin')) + ' 1\n' for row in rows).encode()
        need(evidence.read(folder / 'manifest.txt', dataset['manifest_sha256']) == expected_manifest,
             'manifest_exact_original_order')
        expected_meta = META_MAGIC + b''.join(row['metadata_record'] for row in rows)
        need(evidence.read(folder / 'gold_meta.bin', dataset['gold_metadata_sha256']) == expected_meta,
             'metadata_exact_source_positions_targets_padding')
        for row in rows:
            actual = evidence.read(PREPARED / 'inputs' / (row['source_id'] + '.bin'), sha_bytes(row['prepared_bytes']))
            need(actual == row['prepared_bytes'], 'full_teacher_forced_input_bytes')
        total_gold = sum(row['gold_tokens'] for row in rows)
        total_inputs = sum(row['input_tokens'] for row in rows)
        counters = chunk_counts(chunk, total_gold, total_inputs)
        historical = baseline['per_chunk_native_output_sha256'][chunk]
        need(dataset['gold_tokens'] == total_gold == historical['gold_targets_scored'] and
             dataset['baseline_output_sha256'] == historical['result_sha256'], 'historical_chunk_gold_binding')
        chunks.append({'chunk': chunk, 'dataset_path': str(folder / 'dataset.json'),
                       'dataset_receipt_sha256': dataset['receipt_sha256'], 'case_ids': expected_rows,
                       'local_case_ids': list(range(16)), 'gold_tokens': total_gold,
                       'prompt_bos_tokens': sum(row['prompt_tokens'] for row in rows),
                       'native_input_tokens': total_inputs, 'expected_counters': counters,
                       'manifest_path': dataset['manifest'], 'manifest_sha256': dataset['manifest_sha256'],
                       'metadata_path': dataset['gold_metadata'], 'metadata_sha256': dataset['gold_metadata_sha256'],
                       'input_sha256': dataset['input_sha256'], 'target_token_ids': dataset['target_token_ids'],
                       'teacher_forced_lengths_in_local_order': [row['input_tokens'] for row in rows],
                       'historical_gold_output_sha256': historical['result_sha256'],
                       'local_to_global_rows': [{'local_case_id': local, 'global_case_id': row['global_case_id'],
                                                'source_id': row['source_id'], 'prompt_tokens': row['prompt_tokens'],
                                                'gold_tokens': row['gold_tokens'], 'input_tokens': row['input_tokens']}
                                               for local, row in enumerate(rows)]})
    exact(global_rows, list(range(16, 256)), 'complete_disjoint_train240_global_rows')
    need(len(source_ids) == 240 and sum(row['gold_tokens'] for row in chunks) == 10368 and
         sum(row['native_input_tokens'] for row in chunks) == 33560 and
         sum(row['prompt_bos_tokens'] for row in chunks) == 23432, 'full_train240_totals')
    return chunks


def source_tree(evidence, root):
    root = safe_path(root, directory=True)
    result = {}
    for path in sorted(root.rglob('*')):
        need(not path.is_symlink() and (path.is_file() or path.is_dir()), 'source_tree_link_or_special_file')
        if path.is_file():
            evidence.read(path, collect=False)
            result[str(path.relative_to(root))] = dict(evidence.files[str(path)])
    need(result, 'empty_source_tree')
    return result


def runtime_context(evidence, root, upstream_plan):
    build = evidence.receipt(PARENT / 'NATIVE_BUILD_RECEIPT.json', PIN['parent_build'])
    required_build = {'binary_sha256': PIN['binary'], 'cmake_sha256': PIN['cmake'],
                      'mlx_library_sha256': PIN['mlx_library'], 'model_original_af4_sha256': PIN['model']}
    for key, value in required_build.items():
        exact(build[key], value, 'parent_build.' + key)
    for key, value in {'patched_cpp_sha256': PIN['cpp'], 'patched_c_sha256': PIN['c'],
                       'adapter_sha256': PIN['adapter']}.items():
        exact(build['configuration'][key], value, 'parent_configuration.' + key)
    library = safe_path(upstream_plan['runtime']['library_path'], directory=True)
    runtime_pins = {PARENT / 'source/mlx_moe.cpp': PIN['cpp'], PARENT / 'source/qwen_infer.c': PIN['c'],
                    PARENT / 'source/CMakeLists.txt': PIN['cmake'],
                    PARENT / 'source/je01_mlx_native_jvp_adapters.h': PIN['adapter'],
                    PARENT / 'bin/je01_native_role_jvp': PIN['binary'], library / 'libmlx.dylib': PIN['mlx_library'],
                    MODEL: PIN['model'], BANK / 'JE01_ROLE_DIRECTION_BANK.bin': PIN['bank'],
                    BANK / 'JE01_ROLE_DIRECTION_BANK.tsv': PIN['bank_tsv']}
    for path, pin in runtime_pins.items():
        evidence.read(path, pin, collect=False)
    need(evidence.files[str(MODEL)]['size_bytes'] == 9814343680, 'entire_original_model_length')
    evidence.receipt(BANK / 'JE01_ROLE_DIRECTION_BANK_RECEIPT.json', PIN['bank_receipt'])
    smoke = evidence.receipt(SMOKE_PATH, PIN['smoke'], field='canonical_sha256')
    need(smoke['status'] == 'PASS' and smoke['runtime']['dylib_sha256'] == PIN['mlx_library'] and
         set(smoke['runtime']['header_sha256']) == {'array.h', 'ops.h', 'transforms.h'}, 'pinned_smoke_runtime')
    headers = {}
    for name, pin in smoke['runtime']['header_sha256'].items():
        path = library.parent / 'include/mlx' / name
        evidence.read(path, pin, collect=False)
        headers[str(path)] = pin
    evidence.read(CMAKE, collect=False)
    helpers = {'je01_role_directional_train240_inputs.py': PIN['restored_helper'],
               'je01_role_directional_native_runner.py': PIN['common_helper']}
    for name, pin in helpers.items():
        evidence.read(root / 'tools' / name, pin, collect=False)
    cpp = evidence.read(PARENT / 'source/mlx_moe.cpp', PIN['cpp'])
    c = evidence.read(PARENT / 'source/qwen_infer.c', PIN['c'])
    verify_native_output_guard(c, root)
    return {'library': library, 'runtime_file_sha256': {str(path): pin for path, pin in runtime_pins.items()},
            'mlx_header_sha256': headers, 'cmake_sha256': evidence.files[str(CMAKE)]['sha256'],
            'helper_source_sha256': helpers, 'base_cpp': cpp, 'base_c': c,
            'parent_source_files': source_tree(evidence, PARENT / 'source')}


def manifest_profile(chunk):
    counts = chunk['expected_counters']
    return {'chunk': chunk['chunk'], 'case_ids': chunk['case_ids'], 'native_local_case_ids': list(range(16)),
            'native_input_tokens': chunk['native_input_tokens'],
            'completion': {key: counts[key] for key in ('tokens', 'cases', 'layers', 'gold')},
            'expected_route_records': counts['route_events'], 'expected_route_bytes': 30 + 40 * counts['route_events'],
            'counter_definition': 'TWO_NATIVE_PASSES_OF_FULL_TEACHER_FORCED_INPUT;27_LAYERS;16_CASES_PER_PASS',
            **{key: chunk[key] for key in ('dataset_path', 'dataset_receipt_sha256', 'manifest_path', 'manifest_sha256',
                                          'metadata_path', 'metadata_sha256', 'input_sha256',
                                          'teacher_forced_lengths_in_local_order', 'target_token_ids',
                                          'historical_gold_output_sha256')},
            'baseline_routes': 'MUST_BE_FRESHLY_CAPTURED_BY_CHUNK_OFF;NOT_TRAIN16_ROUTE_HASH'}


def reconstruct_manifest(root, prepare_sha, upstream_gate, chunks, context):
    variants = []
    for chunk in chunks:
        counts = chunk['expected_counters']
        cpp, c = expected_counter_sources(context['base_cpp'], context['base_c'], counts)
        cpp_new = ('g_je01_one_tokens!=' + str(counts['tokens']) +
                   ' || g_je01_one_cases!=32 || g_je01_one_layers!=' + str(counts['layers']) + ')return 0;')
        c_new = ('if(je01_jvp_gold_count!=' + str(counts['gold']) +
                 ' || !mlx_gpu_je01_one_column_jvp_finish(&tokens,&cases,&layers) ||')
        source = root / f"static_builds/chunk{chunk['chunk']:02d}/source"
        build = source.parent / 'build'
        tree = {**context['parent_source_files'], 'mlx_moe.cpp': {'sha256': sha_bytes(cpp), 'size_bytes': len(cpp)},
                'qwen_infer.c': {'sha256': sha_bytes(c), 'size_bytes': len(c)}}
        variants.append({**manifest_profile(chunk), 'source_path': str(source), 'build_path': str(build),
                         'expected_binary_path': str(build / 'je01_native_role_jvp'),
                         'expected_cpp_sha256': sha_bytes(cpp), 'expected_c_sha256': sha_bytes(c),
                         'expected_source_tree_sha256': canonical(tree),
                         'expected_source_total_bytes': sum(item['size_bytes'] for item in tree.values()),
                         'completion_replacements': {'cpp': {'old': CPP_ANCHOR.decode(), 'new': cpp_new},
                                                     'c': {'old': C_ANCHOR.decode(), 'new': c_new}},
                         'compile_commands': [[str(CMAKE), '-S', str(source), '-B', str(build), '-G', 'Ninja',
                                               '-DCMAKE_BUILD_TYPE=Release',
                                               '-DMLX_CMAKE_DIR=' + str(context['library'].parent / 'share/cmake')],
                                              [str(CMAKE), '--build', str(build), '--target', 'qwen_infer_gpu', '--parallel', '2']]})
    return {'schema': 'beglin-je01-train240-static-counter-build-manifest/1', 'scope': SCOPE,
            'status': 'STATIC_SOURCE_BUILD_MANIFEST_SEALED_NO_BUILD_NO_INFERENCE', 'root': str(root),
            'prepare_source_sha256': prepare_sha, 'upstream_gate': upstream_gate,
            'helper_source_sha256': context['helper_source_sha256'], 'preparation_receipt_sha256': PIN['preparation'],
            'preparation_file_sha256': PIN['preparation_file'], 'parent_source_root': str(PARENT / 'source'),
            'parent_build_receipt_sha256': PIN['parent_build'], 'base_cpp_sha256': PIN['cpp'], 'base_c_sha256': PIN['c'],
            'parent_source_files': context['parent_source_files'],
            'parent_source_tree_sha256': canonical(context['parent_source_files']),
            'parent_source_total_bytes': sum(item['size_bytes'] for item in context['parent_source_files'].values()),
            'planned_source_copy_bytes': sum(variant['expected_source_total_bytes'] for variant in variants),
            'source_stage_capacity_rule': {'required_free_bytes': 'REMAINING_UNSTAGED_SOURCE_BYTES_PLUS_HEADROOM',
                                            'headroom_bytes': 1073741824,
                                            'compilation_and_native_outputs_not_covered': True},
            'runtime_file_sha256': context['runtime_file_sha256'], 'mlx_header_sha256': context['mlx_header_sha256'],
            'cmake_path': str(CMAKE), 'cmake_sha256': context['cmake_sha256'],
            'native_output_required_prefix': NATIVE_PREFIX,
            'native_output_path_contract': {'base_c_sha256': PIN['c'], 'required_absolute_prefix': NATIVE_PREFIX,
                                            'output_root': str(root), 'native_guard_read_from_pinned_source': True},
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
            'chunks': variants, **NO_RUN}


def verify_staged_sources(evidence, root, manifest, expected_stage_pin, chunks, context):
    stage = evidence.receipt(root / 'TRAIN240_STATIC_SOURCE_STAGE_RECEIPT.json', expected_stage_pin)
    staged = []
    for chunk, variant in zip(chunks, manifest['chunks']):
        source = Path(variant['source_path'])
        need(source == root / f"static_builds/chunk{chunk['chunk']:02d}/source", 'staged_chunk_path')
        build_path = Path(variant['build_path'])
        need(not build_path.exists() and not build_path.is_symlink(), 'source_audit_must_precede_native_compilation')
        cpp = evidence.read(source / 'mlx_moe.cpp', variant['expected_cpp_sha256'])
        c = evidence.read(source / 'qwen_infer.c', variant['expected_c_sha256'])
        verify_counter_sources(context['base_cpp'], context['base_c'], cpp, c, chunk['expected_counters'])
        tree = source_tree(evidence, source)
        expected_cpp, expected_c = expected_counter_sources(context['base_cpp'], context['base_c'], chunk['expected_counters'])
        expected_tree = {**context['parent_source_files'],
                         'mlx_moe.cpp': {'sha256': sha_bytes(expected_cpp), 'size_bytes': len(expected_cpp)},
                         'qwen_infer.c': {'sha256': sha_bytes(expected_c), 'size_bytes': len(expected_c)}}
        exact(tree, expected_tree, 'entire_staged_source_tree')
        need(canonical(tree) == variant['expected_source_tree_sha256'], 'staged_tree_canonical')
        staged.append({'chunk': chunk['chunk'], 'source_path': str(source), 'source_tree_sha256': canonical(tree),
                       'cpp_sha256': sha_bytes(cpp), 'c_sha256': sha_bytes(c), 'completion': variant['completion']})
    expected = {'schema': 'beglin-je01-train240-static-counter-source-stage/1', 'scope': SCOPE,
                'status': 'FIFTEEN_STATIC_COUNTER_SOURCES_STAGED_NOT_COMPILED', 'root': str(root),
                'manifest_receipt_sha256': manifest['receipt_sha256'],
                'prepare_source_sha256': manifest['prepare_source_sha256'], 'upstream_gate': manifest['upstream_gate'],
                'parent_source_tree_sha256': manifest['parent_source_tree_sha256'],
                'exact_source_only_change_verified': True, 'chunks': staged, **NO_RUN}
    exact({key: value for key, value in stage.items() if key != 'receipt_sha256'}, expected, 'entire_stage_receipt')
    exact(source_tree(evidence, PARENT / 'source'), context['parent_source_files'], 'parent_source_still_unmodified')
    return stage


def audit(args):
    evidence = Evidence()
    gate, upstream_plan, upstream_gate = verify_all48_gate(evidence, args.expected_all48_audit_receipt,
                                                         args.expected_all48_audit_file_sha)
    # All binary/source reads below are unreachable when the upstream result is partial or HOLD.
    root = safe_path(args.root, directory=True)
    need(root == ROOT, 'exact_static_source_root')
    need(not (root / 'runs').exists() and not (root / 'runs').is_symlink(), 'static_source_root_has_native_run_evidence')
    need(all(sha_string(getattr(args, name)) for name in ('expected_manifest_receipt', 'expected_stage_receipt',
                                                         'expected_prepare_sha')), 'external_static_source_pins')
    evidence.read(root / 'tools/je01_native_train240_static_prepare.py', args.expected_prepare_sha, collect=False)
    manifest = evidence.receipt(root / 'TRAIN240_STATIC_BUILD_MANIFEST.json', args.expected_manifest_receipt)
    chunks = verify_datasets(evidence, source_documents(evidence))
    context = runtime_context(evidence, root, upstream_plan)
    reconstructed = reconstruct_manifest(root, args.expected_prepare_sha, upstream_gate, chunks, context)
    exact({key: value for key, value in manifest.items() if key != 'receipt_sha256'}, reconstructed,
          'entire_static_build_manifest')
    stage = verify_staged_sources(evidence, root, manifest, args.expected_stage_receipt, chunks, context)
    result = {'schema': 'beglin-je01-train240-static-counter-independent-source-audit/1',
              'status': 'INDEPENDENT_STATIC_SOURCE_PREPARATION_PASS_NOT_COMPILED',
              'observed_at': datetime.now(timezone.utc).isoformat(), 'root': str(root), 'scope': SCOPE,
              'evidence_integrity': 'PASS', 'upstream_gate': upstream_gate,
              'manifest_receipt_sha256': manifest['receipt_sha256'], 'stage_receipt_sha256': stage['receipt_sha256'],
              'prepare_source_sha256': args.expected_prepare_sha, 'auditor_source_sha256': sha_bytes(Path(__file__).read_bytes()),
              'chunk_count': 15, 'training_sentences': 240, 'global_case_ids': list(range(16, 256)),
              'native_local_case_ids_per_chunk': list(range(16)), 'total_gold_tokens': 10368,
              'total_prompt_bos_tokens': 23432, 'total_native_input_tokens': 33560,
              'per_direction_total_completion': {'tokens': 67120, 'cases': 480, 'layers': 1812240, 'gold': 10368},
              'per_direction_total_route_records': 1745120,
              'original_and_prepared_training_input_bytes_reverified': True,
              'all_fifteen_source_trees_independently_reconstructed': True,
              'all_planned_build_directories_absent': True,
              'allowed_changed_numeric_constants': ['tokens', 'layers', 'gold'], 'case_counter_per_chunk': 32,
              'case_counter_is_two_pass_telemetry_not_independent_observation_count': True,
              'jvp_math_and_other_source_bytes_preserved': True,
              'base_cpp_sha256': PIN['cpp'], 'base_c_sha256': PIN['c'],
              'parent_source_tree_sha256': canonical(context['parent_source_files']),
              'chunks': chunks, 'future_collection_requires': reconstructed['future_collection_requires'],
              'scientific_limits': [
                  'Source preparation only; no new compiled binary or TRAIN240 model result is verified.',
                  'Fifteen future binaries require externally pinned build receipts bound to the respective dataset and source tree.',
                  'Fresh chunk-specific OFF gold parity and full-prefix routes remain mandatory before each chunk JVP collection.',
                  'The 240 source rows exclude TRAIN16; document-level independence is not established by these source IDs.',
                  'Native two-pass counters do not multiply the number of independent sentences.',
                  'Fixed-hard-route local derivatives do not establish finite natural-route policy loss or heldout performance.',
              ],
              'experiment_runner_or_helper_imported': False, 'verified_files': evidence.files, **NO_RUN}
    result['receipt_sha256'] = canonical(result)
    return result


def write_output(path, root, result):
    path, root = Path(path), Path(root)
    need(path.is_absolute() and path.parent == root and path.parent.resolve() == path.parent and
         re.fullmatch(r'TRAIN240_STATIC_INDEPENDENT_AUDIT(?:_[A-Za-z0-9_-]+)?\.json', path.name) is not None and
         not path.exists() and not path.is_symlink(), 'fresh_audit_output_required')
    with path.open('x') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def self_test():
    need(sum(chunk_counts(i)['tokens'] for i in range(1, 16)) == 67120 and
         sum(chunk_counts(i)['layers'] for i in range(1, 16)) == 1812240 and
         sum(chunk_counts(i)['gold'] for i in range(1, 16)) == 10368, 'selftest_counter_totals')
    cpp = b'UNMODIFIED_PREFIX\n' + CPP_ANCHOR + b'\nUNMODIFIED_SUFFIX'
    c = b'UNMODIFIED_PREFIX\n' + C_ANCHOR + b'\nUNMODIFIED_SUFFIX'
    for chunk in range(1, 16):
        patched_cpp, patched_c = expected_counter_sources(cpp, c, chunk_counts(chunk))
        verify_counter_sources(cpp, c, patched_cpp, patched_c, chunk_counts(chunk))
        try:
            verify_counter_sources(cpp, c, patched_cpp + b'\nMATH_DRIFT', patched_c, chunk_counts(chunk))
        except ValueError:
            pass
        else:
            raise AssertionError('source mutation was not rejected')
    return {'status': 'PASS_STATIC_COUNTER_SYNTHETIC_SELF_TEST_ONLY', 'actual_source_or_input_verified': False,
            'chunks': 15, 'native_model_forward': 'NOT_RUN', 'compiled': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-test', action='store_true')
    group.add_argument('--review-metadata', type=Path)
    group.add_argument('--root', type=Path)
    parser.add_argument('--expected-all48-audit-receipt')
    parser.add_argument('--expected-all48-audit-file-sha')
    parser.add_argument('--expected-manifest-receipt')
    parser.add_argument('--expected-stage-receipt')
    parser.add_argument('--expected-prepare-sha')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.review_metadata:
        result = review_metadata(args.review_metadata)
    else:
        need(args.output is not None, 'fresh_output_path_required')
        result = audit(args)
        write_output(args.output, args.root, result)
        result = {key: result[key] for key in ('status', 'receipt_sha256', 'chunk_count', 'total_gold_tokens',
                                              'native_compilation', 'native_model_forward')}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({'status': 'FAIL_STATIC_COUNTER_EVIDENCE_INTEGRITY',
                          'error': f'{type(error).__name__}: {error}', 'native_model_forward': 'NOT_RUN'}), file=sys.stderr)
        sys.exit(1)
