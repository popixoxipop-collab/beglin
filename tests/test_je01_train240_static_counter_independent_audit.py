"""Synthetic source and input-corruption tests; no model or remote filesystem."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'tools/je01_train240_static_counter_independent_audit.py'
spec = importlib.util.spec_from_file_location('independent_static_audit_tested', SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

GUARD = b'''const char *je01_jvp_mode=getenv("JE01_NATIVE_JVP_MODE");
const char prefix[]="/Volumes/D50/TailnetCommander/work/je01_native_role_jvp_";
if(strcmp(je01_jvp_mode,"ROLE_DIRECTION_FIXED_Q4Q5") || !je01_gold_file || !je01_jvp_path ||
strncmp(je01_jvp_path,prefix,sizeof(prefix)-1)) { exit(1); }
int fd=open(je01_jvp_path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
'''


class MemoryEvidence:
    def __init__(self):
        self.raw, self.files = {}, {}

    def read(self, path, expected=None, collect=True):
        raw = self.raw[str(path)]
        digest = audit.sha_bytes(raw)
        audit.need(expected is None or digest == expected, 'synthetic_file_sha')
        self.files[str(path)] = {'sha256': digest, 'size_bytes': len(raw)}
        return raw if collect else None

    def receipt(self, path, pin=None, file_pin=None, *, field='receipt_sha256'):
        return audit.sealed(audit.strict_json(self.read(path, file_pin)), pin, field)

    def put_receipt(self, path, payload):
        result = {**payload, 'receipt_sha256': audit.canonical(payload)}
        self.raw[str(path)] = json.dumps(result, sort_keys=True, allow_nan=False).encode()
        return result['receipt_sha256']


def train_fixture():
    """Build a complete synthetic 240-row corpus with the real aggregate lengths."""
    evidence, dataset_pins = MemoryEvidence(), []
    gold = {'cases': [{} for _ in range(256)]}
    baseline = {'per_case_real_NLL': [{} for _ in range(256)],
                'per_chunk_native_output_sha256': [{} for _ in range(16)]}
    native = {'cohorts': {'calibration': {'file_receipts': [{} for _ in range(256)]}}}
    for chunk, (total_gold, total_inputs) in enumerate(audit.COUNTS, 1):
        folder = audit.PREPARED / f'chunk{chunk:02d}'
        targets, input_pins, case_sources, lines, metadata = [], [], [], [], bytearray(audit.META_MAGIC)
        for local in range(16):
            row = 16 * chunk + local
            n = total_gold // 16 + int(local < total_gold % 16)
            s = total_inputs // 16 + int(local < total_inputs % 16)
            p = s - n + 1
            source_id = audit.sha_bytes(f'SYNTHETIC_TRAIN_ONLY_ROW_{row}'.encode())
            prompt_tokens = [100000] + [row + 1] * (p - 1)
            gold_tokens = [1000 + row * 4 + index for index in range(n)]
            prompt = struct.pack('<' + 'i' * p, *prompt_tokens)
            prompt_path = audit.ORIGINAL_INPUTS / 'calibration' / (source_id + '.bin')
            evidence.raw[str(prompt_path)] = prompt
            native['cohorts']['calibration']['file_receipts'][row] = {
                'input_id': source_id, 'native_int32_sha256': audit.sha_bytes(prompt),
                'native_int32_bytes': len(prompt), 'token_count_with_bos': p}
            gold['cases'][row] = {'row': row, 'input_id': source_id, 'native_BOS_prompt_sha256': audit.sha_bytes(prompt),
                                  'gold_continuation_token_ids': gold_tokens, 'gold_token_count': n,
                                  'original_BOS_prompt_positions': p, 'model_full_BOS_sequence_positions': p + n}
            baseline['per_case_real_NLL'][row] = {'row': row, 'source_id': source_id, 'tokens': n,
                                                 'first_gold_nll': 1., 'nll_sum': float(n)}
            tokens = prompt_tokens + gold_tokens[:-1]
            prepared = struct.pack('<' + 'i' * s, *tokens)
            prepared_path = audit.PREPARED / 'inputs' / (source_id + '.bin')
            evidence.raw[str(prepared_path)] = prepared
            input_pins.append(audit.sha_bytes(prepared))
            targets.append(gold_tokens)
            metadata.extend(struct.pack('<II384i', p, n, *(gold_tokens + [-1] * (384 - n))))
            lines.append(str(prepared_path) + ' 1\n')
            case_sources.append({'row': row, 'input_id': source_id, 'original_prompt_path': str(prompt_path),
                                 'original_prompt_sha256': audit.sha_bytes(prompt), 'original_prompt_positions': p,
                                 'gold_tokens': n, 'original_baseline_first_gold_nll': 1.,
                                 'original_baseline_nll_sum': float(n)})
        manifest = ''.join(lines).encode()
        evidence.raw[str(folder / 'manifest.txt')] = manifest
        evidence.raw[str(folder / 'gold_meta.bin')] = bytes(metadata)
        historical = audit.sha_bytes(f'SYNTHETIC_BASELINE_CHUNK_{chunk}'.encode())
        baseline['per_chunk_native_output_sha256'][chunk] = {'chunk': chunk, 'result_sha256': historical,
                                                            'gold_targets_scored': total_gold}
        dataset = {'schema': 'beglin-je01-role-directional-train-input/1', 'scope': 'TRAIN240', 'chunk': chunk,
                   'dataset_id': f'train240_chunk{chunk:02d}', 'case_ids': list(range(16 * chunk, 16 * chunk + 16)),
                   'manifest': str(folder / 'manifest.txt'), 'gold_metadata': str(folder / 'gold_meta.bin'),
                   'manifest_sha256': audit.sha_bytes(manifest), 'gold_metadata_sha256': audit.sha_bytes(metadata),
                   'input_sha256': input_pins, 'target_token_ids': targets, 'baseline_output_sha256': historical,
                   'gold_tokens': total_gold, 'case_sources': case_sources, 'source_linkage': audit.source_linkage_contract(),
                   'baseline_output_is_historical_reference_not_new_execution': True,
                   'independent_validation_ood_accessed': False, 'native_model_forward': 'NOT_RUN',
                   'production_modified': False}
        dataset_pins.append(evidence.put_receipt(folder / 'dataset.json', dataset))
    preparation = {'source_linkage': audit.source_linkage_contract()}
    return evidence, (preparation, gold, baseline, native), tuple(dataset_pins)


class StaticCounterAuditTests(unittest.TestCase):
    def test_counter_profiles_distinguish_layers_routes_and_independent_cases(self):
        first = audit.chunk_counts(1)
        self.assertEqual(first, {'tokens': 4452, 'cases': 32, 'layers': 120204, 'gold': 634, 'route_events': 115752})
        self.assertEqual(audit.chunk_counts(15),
                         {'tokens': 3814, 'cases': 32, 'layers': 102978, 'gold': 593, 'route_events': 99164})
        self.assertEqual(sum(audit.chunk_counts(i)['tokens'] for i in range(1, 16)), 67120)
        self.assertEqual(sum(audit.chunk_counts(i)['layers'] for i in range(1, 16)), 1812240)
        self.assertEqual(sum(audit.chunk_counts(i)['route_events'] for i in range(1, 16)), 1745120)
        for args in ((0,), (16,), (True,), (1, 656, 2167), (1, 634., 2226)):
            with self.assertRaises(ValueError):
                audit.chunk_counts(*args)

    def test_exact_source_patch_rejects_math_guard_and_wrong_chunk_mutations(self):
        cpp = b'auto derivative = unchanged_math;\n' + audit.CPP_ANCHOR + b'\n'
        c = GUARD + audit.C_ANCHOR + b'\nfsync(fd);\n'
        for chunk in range(1, 16):
            counts = audit.chunk_counts(chunk)
            new_cpp, new_c = audit.expected_counter_sources(cpp, c, counts)
            proof = audit.verify_counter_sources(cpp, c, new_cpp, new_c, counts)
            self.assertTrue(proof['all_other_source_bytes_identical'])
            mutations = [(new_cpp.replace(b'unchanged_math', b'new_math'), new_c),
                         (new_cpp.replace(b'g_je01_one_cases!=32', b'g_je01_one_cases!=16'), new_c),
                         (new_cpp, new_c.replace(b'O_EXCL|O_NOFOLLOW', b'0')),
                         (new_cpp, new_c.replace(b'fsync(fd);', b''))]
            for altered_cpp, altered_c in mutations:
                with self.assertRaisesRegex(ValueError, 'only_three_completion_constants'):
                    audit.verify_counter_sources(cpp, c, altered_cpp, altered_c, counts)
        chunk1 = audit.expected_counter_sources(cpp, c, audit.chunk_counts(1))
        with self.assertRaises(ValueError):
            audit.verify_counter_sources(cpp, c, *chunk1, audit.chunk_counts(2))
        for altered_cpp, altered_c in ((cpp + audit.CPP_ANCHOR, c), (cpp, c + audit.C_ANCHOR)):
            with self.assertRaises(ValueError):
                audit.expected_counter_sources(altered_cpp, altered_c, audit.chunk_counts(1))

    def test_c_open_prefix_and_full_guard_are_preserved(self):
        audit.verify_native_output_guard(GUARD, audit.ROOT)
        for root in (audit.WORK / 'je01_native_all48_train16_old', Path('/tmp/je01_native_role_jvp_x')):
            with self.assertRaises(ValueError):
                audit.verify_native_output_guard(GUARD, root)
        for text in (GUARD.replace(b'!je01_gold_file || ', b''),
                     GUARD.replace(b'strncmp(je01_jvp_path,prefix,sizeof(prefix)-1)', b'0'),
                     GUARD.replace(b'O_NOFOLLOW', b'0')):
            with self.assertRaises(ValueError):
                audit.verify_native_output_guard(text, audit.ROOT)

    def test_all240_raw_dataset_bytes_and_mapping_are_independently_checked(self):
        evidence, documents, pins = train_fixture()
        with patch.object(audit, 'CHUNK_DATASET_PINS', pins):
            result = audit.verify_datasets(evidence, documents)
        self.assertEqual(len(result), 15)
        self.assertEqual(sum(row['gold_tokens'] for row in result), 10368)
        self.assertEqual(sum(row['native_input_tokens'] for row in result), 33560)
        self.assertEqual(result[0]['local_to_global_rows'][0]['global_case_id'], 16)
        self.assertEqual(result[-1]['local_to_global_rows'][-1]['global_case_id'], 255)
        self.assertEqual(result[-1]['local_to_global_rows'][-1]['local_case_id'], 15)
        self.assertFalse(any('/validation/' in path or '/ood/' in path for path in evidence.files))
        prepared = [path for path in evidence.files if '/train240_inputs/inputs/' in path]
        originals = [path for path in evidence.files if '/je01_inputs/calibration/' in path]
        self.assertEqual((len(prepared), len(originals)), (240, 240))

    def test_manifest_source_copy_budget_uses_actual_variant_byte_lengths(self):
        evidence, documents, pins = train_fixture()
        with patch.object(audit, 'CHUNK_DATASET_PINS', pins):
            chunks = audit.verify_datasets(evidence, documents)
        cpp, c = b'prefix\n' + audit.CPP_ANCHOR + b'\n', GUARD + audit.C_ANCHOR + b'\n'
        tree = {'mlx_moe.cpp': {'sha256': audit.sha_bytes(cpp), 'size_bytes': len(cpp)},
                'qwen_infer.c': {'sha256': audit.sha_bytes(c), 'size_bytes': len(c)},
                'unchanged.h': {'sha256': audit.sha_bytes(b'header'), 'size_bytes': 6}}
        context = {'base_cpp': cpp, 'base_c': c, 'parent_source_files': tree,
                   'library': Path('/synthetic/runtime/lib'), 'helper_source_sha256': {},
                   'runtime_file_sha256': {}, 'mlx_header_sha256': {}, 'cmake_sha256': 'c' * 64}
        manifest = audit.reconstruct_manifest(audit.ROOT, 'a' * 64, {}, chunks, context)
        parent_bytes = len(cpp) + len(c) + 6
        self.assertEqual(manifest['parent_source_total_bytes'], parent_bytes)
        # Only chunk07's layer count has five digits; the original has six.
        self.assertEqual(manifest['planned_source_copy_bytes'], 15 * parent_bytes - 1)
        self.assertEqual(manifest['chunks'][6]['expected_source_total_bytes'], parent_bytes - 1)
        self.assertEqual(manifest['source_stage_capacity_rule']['headroom_bytes'], 1 << 30)
        self.assertEqual(manifest['chunks'][0]['expected_route_bytes'], 30 + 115752 * 40)
        self.assertEqual(manifest['chunks'][0]['native_local_case_ids'], list(range(16)))

    def test_prepared_payload_drift_rejected_even_after_dataset_resealing(self):
        evidence, documents, pins = train_fixture()
        path = audit.PREPARED / 'chunk01/dataset.json'
        dataset = json.loads(evidence.raw[str(path)])
        input_path = dataset['manifest']
        first_input = evidence.raw[input_path].decode().splitlines()[0].rsplit(' ', 1)[0]
        changed = bytearray(evidence.raw[first_input])
        struct.pack_into('<i', changed, 4, 8888)
        evidence.raw[first_input] = bytes(changed)
        dataset['input_sha256'][0] = audit.sha_bytes(changed)
        del dataset['receipt_sha256']
        replaced = evidence.put_receipt(path, dataset)
        with patch.object(audit, 'CHUNK_DATASET_PINS', (replaced,) + pins[1:]), self.assertRaisesRegex(ValueError, 'derived_input_hashes'):
            audit.verify_datasets(evidence, documents)

    def test_train16_global_mapping_and_metadata_padding_drift_are_rejected(self):
        for corruption in ('global_case', 'metadata_pad'):
            evidence, documents, pins = train_fixture()
            path = audit.PREPARED / 'chunk01/dataset.json'
            data = json.loads(evidence.raw[str(path)])
            if corruption == 'global_case':
                data['case_ids'][0] = 0
            else:
                meta_path = data['gold_metadata']
                raw = bytearray(evidence.raw[meta_path])
                raw[-4:] = struct.pack('<i', 0)
                evidence.raw[meta_path] = bytes(raw)
                data['gold_metadata_sha256'] = audit.sha_bytes(raw)
            del data['receipt_sha256']
            changed_pin = evidence.put_receipt(path, data)
            with patch.object(audit, 'CHUNK_DATASET_PINS', (changed_pin,) + pins[1:]), self.assertRaises(ValueError):
                audit.verify_datasets(evidence, documents)

    def test_train16_source_identity_cannot_reappear_under_a_new_global_row(self):
        identifiers = [audit.sha_bytes(f'original_source_{index}'.encode()) for index in range(256)]
        prompts = [{'input_id': source_id} for source_id in identifiers]
        gold = [{'row': index, 'input_id': source_id} for index, source_id in enumerate(identifiers)]
        baseline = [{'row': index, 'source_id': source_id} for index, source_id in enumerate(identifiers)]
        audit.verify_source_row_identity(prompts, gold, baseline)
        prompts[16]['input_id'] = gold[16]['input_id'] = baseline[16]['source_id'] = identifiers[0]
        with self.assertRaisesRegex(ValueError, 'source_ids_must_be_unique'):
            audit.verify_source_row_identity(prompts, gold, baseline)

    def test_partial_or_hold_blocks_before_any_train240_read(self):
        evidence = MemoryEvidence()
        path = audit.ALL48 / 'JE01_ALL48_TRAIN16_INDEPENDENT_AUDIT.json'
        for status in ('INDEPENDENT_AUDIT_CONFIRMED_PARTIAL_NO_PASS', 'INDEPENDENT_AUDIT_CONFIRMED_NUMERICAL_HOLD'):
            payload = {'schema': 'beglin-je01-native-all48-train16-independent-raw-audit/2', 'status': status}
            pin = evidence.put_receipt(path, payload)
            file_pin = audit.sha_bytes(evidence.raw[str(path)])
            with patch.object(audit, 'Evidence', return_value=evidence), self.assertRaises(ValueError):
                audit.audit(SimpleNamespace(expected_all48_audit_receipt=pin,
                                            expected_all48_audit_file_sha=file_pin))
            self.assertEqual(set(evidence.files), {str(path)})

    def test_entire_source_tree_and_output_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            source = root / 'source'
            source.mkdir()
            (source / 'header.h').write_bytes(b'original header')
            first = audit.source_tree(audit.Evidence(), source)
            (source / 'header.h').write_bytes(b'mutated header')
            second = audit.source_tree(audit.Evidence(), source)
            self.assertNotEqual(audit.canonical(first), audit.canonical(second))
            (source / 'linked.h').symlink_to(source / 'header.h')
            with self.assertRaises(ValueError):
                audit.source_tree(audit.Evidence(), source)
            output = root / 'TRAIN240_STATIC_INDEPENDENT_AUDIT.json'
            audit.write_output(output, root, {'synthetic_test_only': True})
            before = output.read_bytes()
            with self.assertRaises(ValueError):
                audit.write_output(output, root, {'overwrite': True})
            self.assertEqual(output.read_bytes(), before)
            with self.assertRaises(ValueError):
                audit.write_output(root / 'TRAIN240_STATIC_BUILD_MANIFEST.json', root, {})

    def test_reviewed_metadata_is_not_raw_or_gate_pass(self):
        root = SOURCE.parents[1]
        path = root / 'docs/JE01_TRAIN240_INPUT_METADATA_20261123.json'
        if not path.exists():
            path = root / 'docs/research/JE01_TRAIN240_INPUT_METADATA_20261123.json'
        result = audit.review_metadata(path)
        self.assertEqual(result['status'], 'PREPARATION_ONLY_NO_NATIVE_RESULT')
        self.assertFalse(result['source_raw_binary_reverified'])
        self.assertFalse(result['all48_gate_claimed'])
        self.assertFalse(result['compiled'])

    def test_duplicate_json_and_external_pin_rejection(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError):
                audit.strict_json(raw)
        value = {'source': 'synthetic'}
        original = audit.canonical(value)
        changed = {'source': 'different'}
        changed['receipt_sha256'] = audit.canonical(changed)
        with self.assertRaises(ValueError):
            audit.sealed(changed, original)
        with self.assertRaises(ValueError):
            audit.exact(True, 1, 'bool_is_not_case_count')


if __name__ == '__main__':
    unittest.main()
