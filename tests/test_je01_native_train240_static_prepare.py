"""Synthetic source-contract tests; no TRAIN240 source staging or model run."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
spec = importlib.util.spec_from_file_location('train240_static_tested', TOOLS / 'je01_native_train240_static_prepare.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
patch_spec = importlib.util.spec_from_file_location('train240_original_patch_fixture', TOOLS / 'je01_native_role_jvp_patch.py')
original = importlib.util.module_from_spec(patch_spec)
patch_spec.loader.exec_module(original)


def synthetic_rows(chunk):
    gold, inputs = p.GOLD_COUNTS[chunk - 1], p.INPUT_COUNTS[chunk - 1]
    rows = []
    for local in range(16):
        count = gold // 16 + int(local < gold % 16)
        length = inputs // 16 + int(local < inputs % 16)
        prompt = length - count + 1
        rows.append({'row': chunk * 16 + local, 'prompt_token_ids': [100000] + [2] * (prompt - 1),
                     'target_token_ids': [3] * count})
    data = {'scope': 'TRAIN240', 'chunk': chunk, 'receipt_sha256': p.DATASET_PINS[chunk - 1],
            'case_ids': list(range(chunk * 16, chunk * 16 + 16)), 'gold_tokens': gold,
            'target_token_ids': [r['target_token_ids'] for r in rows],
            'manifest': str(p.PREP_ROOT / f'chunk{chunk:02d}/manifest.txt'), 'manifest_sha256': 'a' * 64,
            'gold_metadata': str(p.PREP_ROOT / f'chunk{chunk:02d}/gold_meta.bin'), 'gold_metadata_sha256': 'b' * 64,
            'input_sha256': ['c' * 64] * 16, 'baseline_output_sha256': 'd' * 64}
    return data, rows


def synthetic_pass_audit():
    return {'schema': 'beglin-je01-native-all48-train16-independent-raw-audit/2',
            'status': 'INDEPENDENT_AUDIT_CONFIRMED_ALL48_TRAIN16_RAW_AND_NINE_CHECKPOINTS_PASS',
            'scope': 'PREVIOUSLY_SEEN_TRAIN16_FIXED_HARD_ROUTE_48_DIRECTION_DIAGNOSTIC_ONLY',
            'root': str(p.UPSTREAM), 'evidence_integrity': 'PASS', 'experiment_numerical_verdict': 'PASS',
            'completed_evidence_cell_count': 79, 'completed_native_cell_count': 79,
            'native_cells_executed_in_this_plan': 78, 'inherited_native_cell_count': 1,
            'native_jvp_columns': 48, 'fd_verified_coordinates': 9,
            'gold_tokens': 656, 'case_ids': list(range(16)), 'sentence_groups_count': 16,
            'plan_receipt_sha256': p.UPSTREAM_PLAN, 'external_plan_pin': p.UPSTREAM_PLAN,
            'runner_source_sha256': p.UPSTREAM_RUNNER, 'auditor_source_sha256': p.UPSTREAM_AUDITOR,
            'all_48_raw_directions_verified': True, 'all_nine_fd_checkpoints_pass': True,
            'original_af4_full_file_sha_verified_after_run': p.AF4,
            'native_inference_rerun': False, 'experiment_runner_or_helper_imported': False,
            'allocation_selected': False, 'independent_validation_ood_accessed': False, 'production_modified': False,
            'independent_psd_check': {'passed': True}, 'numerical_hold_receipt_sha256': None,
            'source_report_receipt_sha256': 'e' * 64, 'raw_matrix_receipt_sha256': 'f' * 64,
            'g': [0.0] * 48, 'observed_label_directional_gram': [[0.0] * 48 for _ in range(48)]}


class StaticTrain240SourceContract(unittest.TestCase):
    def test_source_derived_all_fifteen_profiles_and_global_local_mapping(self):
        profiles = [p.derive_profile(*synthetic_rows(chunk)) for chunk in range(1, 16)]
        self.assertEqual([x for profile in profiles for x in profile['case_ids']], list(range(16, 256)))
        self.assertTrue(all(profile['native_local_case_ids'] == list(range(16)) for profile in profiles))
        self.assertEqual(sum(v['native_input_tokens'] for v in profiles), 33560)
        self.assertEqual(sum(v['completion']['gold'] for v in profiles), 10368)
        self.assertEqual(sum(v['completion']['tokens'] for v in profiles), 67120)
        self.assertEqual(sum(v['completion']['cases'] for v in profiles), 480)
        self.assertEqual(sum(v['completion']['layers'] for v in profiles), 1812240)
        self.assertEqual(profiles[0]['completion'], {'tokens': 4452, 'cases': 32, 'layers': 120204, 'gold': 634})
        data, rows = synthetic_rows(1)
        wrong = copy.deepcopy(data); wrong['case_ids'][0] = 0
        with self.assertRaises(ValueError): p.derive_profile(wrong, rows)
        bad_rows = copy.deepcopy(rows); bad_rows[0]['prompt_token_ids'].append(4)
        with self.assertRaises(ValueError): p.derive_profile(data, bad_rows)
        for args in [(True, 2226, 634), (16, 2226, 634), (1, 2225, 634), (1, 2226, 635)]:
            with self.assertRaises(ValueError): p.counter_profile(*args)

    def test_only_completion_lines_change_and_restore_original_fragments(self):
        for chunk in range(1, 16):
            profile = p.counter_profile(chunk, p.INPUT_COUNTS[chunk - 1], p.GOLD_COUNTS[chunk - 1])
            old_cpp, old_c = original.FINALIZE_WRAPPER, original.C_CLOSE
            cpp, c, changes = p.rewrite_completion_literals(old_cpp, old_c, profile)
            self.assertEqual(cpp.replace(changes['cpp']['new'], p.CPP_ANCHOR, 1), old_cpp)
            self.assertEqual(c.replace(changes['c']['new'], p.C_ANCHOR, 1), old_c)
            self.assertIn('g_je01_one_cases!=32', cpp)
            self.assertIn('!g_je01_direction_pending_routes.empty()', cpp)
            self.assertIn('fflush(je01_jvp_file)!=0', c)
            self.assertIn('fsync(fileno(je01_jvp_file))!=0', c)
        duplicate = old_cpp + '\n' + p.CPP_ANCHOR
        with self.assertRaises(ValueError): p.rewrite_completion_literals(duplicate, old_c, profile)
        changed = copy.deepcopy(profile); changed['completion']['cases'] = 16
        with self.assertRaises(ValueError): p.rewrite_completion_literals(old_cpp, old_c, changed)
        with self.assertRaises(ValueError):
            p.apply_to_pinned_sources(old_cpp.encode(), old_c.encode(), profile)

    def test_exact_compiled_finish_guards_for_each_static_profile(self):
        compiler = shutil.which('c++')
        if not compiler: self.skipTest('A C++ compiler is required for this local source-contract check.')
        prefix = '''#include <cstdint>
#include <vector>
#include <cassert>
void *g_cbatch_x=nullptr;
int g_cbatch_layers_done=0;
std::vector<int> g_je01_direction_pending_routes;
uint64_t g_je01_one_tokens=0,g_je01_one_cases=0,g_je01_one_layers=0;
'''
        finish = original.FINALIZE_WRAPPER[original.FINALIZE_WRAPPER.index('extern "C" int mlx_gpu_je01_one_column_jvp_finish'):]
        with tempfile.TemporaryDirectory(prefix='je01_train240_static_contract_') as directory:
            source, binary = Path(directory) / 'guards.cpp', Path(directory) / 'guards'
            bodies, checks = [], []
            for chunk in range(1, 16):
                profile = p.counter_profile(chunk, p.INPUT_COUNTS[chunk - 1], p.GOLD_COUNTS[chunk - 1])
                cpp, c, _ = p.rewrite_completion_literals(finish, original.C_CLOSE, profile)
                name = f'finish_{chunk}'
                bodies.append(cpp.replace('mlx_gpu_je01_one_column_jvp_finish', name))
                tokens, layers = profile['completion']['tokens'], profile['completion']['layers']
                self.assertEqual(int(re.search(r'je01_jvp_gold_count!=(\d+)', c).group(1)), profile['completion']['gold'])
                checks.append(f'''g_je01_one_tokens={tokens};g_je01_one_cases=32;g_je01_one_layers={layers};
assert({name}(&t,&n,&l)==1);assert(t=={tokens}&&n==32&&l=={layers});
++g_je01_one_tokens;assert({name}(&t,&n,&l)==0);--g_je01_one_tokens;
--g_je01_one_cases;assert({name}(&t,&n,&l)==0);++g_je01_one_cases;
++g_je01_one_layers;assert({name}(&t,&n,&l)==0);--g_je01_one_layers;
g_je01_direction_pending_routes.push_back(1);assert({name}(&t,&n,&l)==0);g_je01_direction_pending_routes.clear();
g_cbatch_layers_done=1;assert({name}(&t,&n,&l)==0);g_cbatch_layers_done=0;
assert({name}(nullptr,&n,&l)==0);''')
            source.write_text(prefix + '\n'.join(bodies) + '\nint main(){uint64_t t=0,n=0,l=0;\n' + '\n'.join(checks) + '\n}')
            subprocess.run([compiler, '-std=c++17', '-O0', str(source), '-o', str(binary)], check=True,
                           capture_output=True, text=True, timeout=30)
            subprocess.run([str(binary)], check=True, capture_output=True, timeout=5)

    def test_full48_gate_refuses_partial_three_role_and_false_completion(self):
        good = synthetic_pass_audit()
        p.validate_full48_audit(good)
        mutations = [('status', 'INDEPENDENT_AUDIT_CONFIRMED_PARTIAL_NO_PASS'),
                     ('status', 'INDEPENDENT_AUDIT_CONFIRMED_THREE_ROLE_PASS'),
                     ('native_jvp_columns', 3), ('fd_verified_coordinates', 8),
                     ('native_cells_executed_in_this_plan', 79), ('all_nine_fd_checkpoints_pass', False),
                     ('runner_source_sha256', '0' * 64), ('plan_receipt_sha256', '0' * 64),
                     ('independent_validation_ood_accessed', True), ('allocation_selected', 0)]
        for key, value in mutations:
            wrong = copy.deepcopy(good); wrong[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): p.validate_full48_audit(wrong)
        wrong = copy.deepcopy(good); wrong['g'][3] = math.nan
        with self.assertRaises(ValueError): p.validate_full48_audit(wrong)
        wrong = copy.deepcopy(good); wrong['independent_psd_check']['passed'] = False
        with self.assertRaises(ValueError): p.validate_full48_audit(wrong)

    def test_missing_gate_stops_before_any_source_or_output_access(self):
        with tempfile.TemporaryDirectory(prefix='je01_train240_gate_test_') as directory:
            destination = Path(directory) / 'must_not_be_created'
            args = SimpleNamespace(root=destination, expected_all48_audit_receipt=None,
                                   expected_all48_audit_file_sha=None, resume=False)
            with self.assertRaisesRegex(ValueError, 'ACTUAL_ALL48_EXTERNAL_CANONICAL_AND_FILE_PINS_REQUIRED'):
                p.manifest(args)
            self.assertFalse(destination.exists())

    def test_native_root_guard_and_unsealed_spec_are_explicit(self):
        root = p.WORK / 'je01_native_role_jvp_train240_static_20261123_v1'
        p.require_root(root)
        result = p.native_output_contract(original.C_OPEN.encode(), root)
        self.assertEqual(result['required_absolute_prefix'], str(p.WORK) + '/je01_native_role_jvp_')
        with self.assertRaises(ValueError):
            p.native_output_contract(original.C_OPEN.encode(), p.WORK / 'je01_native_all48_train16_20261123_v1')
        spec = p.specification()
        self.assertEqual(spec['status'], 'DRAFT_SPEC_NOT_A_SEALED_EXPERIMENT_PLAN')
        self.assertNotIn('receipt_sha256', spec)
        self.assertEqual(spec['native_compilation'], 'NOT_RUN')
        self.assertEqual(spec['native_model_forward'], 'NOT_RUN')

    def test_source_tree_rejects_links_and_receipt_changes(self):
        with tempfile.TemporaryDirectory(prefix='je01_train240_source_test_') as directory:
            root = Path(directory)
            (root / 'source.cpp').write_text('unchanged math\n')
            tree = p.source_tree(root)
            (root / 'alias.cpp').symlink_to(root / 'source.cpp')
            with self.assertRaises(ValueError): p.source_tree(root)
            (root / 'alias.cpp').unlink()
            self.assertEqual(p.source_tree(root), tree)
            path = root / 'synthetic_receipt.json'
            before = p.seal_once(path, {'synthetic_only': True})
            self.assertEqual(p.seal_once(path, {'synthetic_only': True}, resume=True), before)
            with self.assertRaises(ValueError): p.seal_once(path, {'synthetic_only': False}, resume=True)

    def test_source_only_receipts_refuse_existing_build_or_run(self):
        with tempfile.TemporaryDirectory(prefix='je01_train240_phase_test_') as directory:
            root = Path(directory)
            build = root / 'static_builds/chunk01/build'
            p.require_source_only_phase(root, [build])
            build.mkdir(parents=True)
            with self.assertRaises(ValueError): p.require_source_only_phase(root, [build])
            build.rmdir()
            (root / 'runs').mkdir()
            with self.assertRaises(ValueError): p.require_source_only_phase(root, [build])


if __name__ == '__main__':
    unittest.main()
