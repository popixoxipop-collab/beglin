"""Recovery-v2 synthetic negative controls; no native model execution."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('all48_v2_tested', TOOLS / 'je01_native_all48_train16_runner_v2.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def synthetic_context():
    return {'identity': {'synthetic_test_only': True}, 'plan': {'receipt_sha256': 'a' * 64},
            'dataset': {'target_token_ids': [list(range(41)) for _ in range(16)]}}


def new_checkpoint_fixture(coarse_slope=1.0, fine_slope=1.0, jvp=1.0):
    context = synthetic_context()
    records = {}
    for index, alpha in enumerate(r.ALPHAS):
        numeric = r.common.ALPHAS[alpha]
        slope = coarse_slope if abs(numeric) == .05 else fine_slope
        records[index] = {'kind': 'FD', 'role_index': 0, 'tier': 'Q5', 'alpha_literal': alpha,
                          'receipt_sha256': str(index) * 64,
                          'per_token_nll': [[2.0 + numeric * slope] * 41 for _ in range(16)]}
    measured = {'kind': 'JVP', 'role_index': 0, 'tier': 'Q5', 'receipt_sha256': 'b' * 64,
                'gradient_output_sha256': 'c' * 64,
                'gradients': {'per_token_dnll': [[jvp] * 41 for _ in range(16)]}}
    return context, measured, records


class All48V2Contracts(unittest.TestCase):
    def test_exact_all48_order_and_early_q5_checkpoints(self):
        coordinates, sequence = r.coordinate_order(), r.execution_sequence()
        self.assertEqual(len(coordinates), 48)
        self.assertEqual(coordinates[0], {'column_index': 0, 'role_index': 0,
                                         'role_key': [12, 13, 'gate_proj'], 'tier': 'Q4'})
        self.assertEqual(coordinates[-1], {'column_index': 47, 'role_index': 23,
                                          'role_key': [24, 60, 'down_proj'], 'tier': 'Q5'})
        self.assertEqual([c['column_index'] for c in coordinates], list(range(48)))
        self.assertEqual(len(sequence), 79)
        self.assertEqual(sum(c['kind'] == 'OFF' for c in sequence), 1)
        self.assertEqual(sum(c['kind'] == 'JVP' for c in sequence), 48)
        self.assertEqual(sum(c['kind'] == 'FD' for c in sequence), 30)
        self.assertEqual({(c['role_index'], c['tier']) for c in sequence if c['kind'] == 'FD'},
                         {(0, 'Q5'), (1, 'Q5'), (2, 'Q5'), (3, 'Q4'), (13, 'Q5'), (23, 'Q4')})
        first_jvp = [c for c in sequence if c['kind'] == 'JVP'][:3]
        self.assertEqual([(c['role_index'], c['tier'], c['cell_index']) for c in first_jvp],
                         [(0, 'Q5', 6), (1, 'Q5', 12), (2, 'Q5', 18)])
        self.assertEqual(sorted(c['column_index'] for c in sequence if c['kind'] == 'JVP'), list(range(48)))
        self.assertEqual(len({r.cell_label(c) for c in sequence}), 79)

    def test_cell_contract_rejects_scope_aliases_and_wrong_coordinates(self):
        good = r.execution_sequence()[6]
        r.validate_cell(good)
        for key, value in [('role_index', 24), ('role_index', False), ('tier', 'Q3'),
                           ('alpha_literal', '+0.05'), ('column_index', 0), ('cell_index', True)]:
            bad = copy.deepcopy(good); bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                r.validate_cell(bad)
        for role, tier in [(True, 'Q4'), (-1, 'Q4'), (24, 'Q4'), (0, 'Q3')]:
            with self.assertRaises(ValueError):
                r.column_index(role, tier)

    def test_native_log_checks_actual_late_layer_role_tier_and_alpha(self):
        cell = next(c for c in r.execution_sequence()
                    if c['kind'] == 'FD' and c['role_index'] == 23 and c['alpha_literal'] == '+0.025')
        correct = 'JE01_ROLE_DIRECTION_CONFIG role=23 layer=24 expert=60 tier=Q4 alpha=+0.025\n'
        r.validate_native_configuration(correct + correct, cell)
        for bad in [correct.replace('role=23', 'role=2'), correct.replace('layer=24', 'layer=12'),
                    correct.replace('expert=60', 'expert=13'), correct.replace('tier=Q4', 'tier=Q5'),
                    correct.replace('+0.025', '0.025'), correct + 'JE01_ROLE_DIRECTION_CONFIG malformed\n']:
            with self.assertRaises(ValueError):
                r.validate_native_configuration(bad, cell)
        with self.assertRaises(ValueError):
            r.validate_native_configuration(correct, r.execution_sequence()[0])

    def test_receipts_are_no_clobber_and_partial_writes_block_resume(self):
        with tempfile.TemporaryDirectory(prefix='je01_all48_synthetic_') as directory:
            path = Path(directory) / 'receipt.json'
            payload = {'synthetic_only': True, 'value': 4}
            first = r.seal_receipt(path, payload)
            original = path.read_bytes()
            self.assertEqual(r.seal_receipt(path, payload, resume=True), first)
            for changed, resume in [(payload, False), ({'synthetic_only': True, 'value': 5}, True)]:
                with self.assertRaises(ValueError):
                    r.seal_receipt(path, changed, resume=resume)
            self.assertEqual(path.read_bytes(), original)
            partial = Path(directory) / 'partial.json'
            partial.write_bytes(b'{"receipt_sha256":')
            with self.assertRaises(ValueError):
                r.seal_receipt(partial, payload, resume=True)
            self.assertEqual(partial.read_bytes(), b'{"receipt_sha256":')

    def test_saved_cell_rejects_altered_plan_binary_tier_and_scope(self):
        context = synthetic_context()
        cell = r.execution_sequence()[6]
        receipt = r.cell_contract(context, cell, 1, 'd' * 64, [])
        receipt['execution'] = {'native_exit_code': 0, 'stop_reason': None, 'native_started': True,
                                'elapsed_seconds': 1.0, 'timeout_seconds': 900,
                                'rss_limit_gib': 18, 'memory_free_floor_pct': 35}
        r.validate_saved_contract(receipt, context, cell, 1, 'd' * 64, [])
        for key, value in [('role_index', False), ('tier', 'Q4'), ('column_index', 0),
                           ('plan_receipt_sha256', 'f' * 64), ('executed_binary_sha256', 'e' * 64),
                           ('new_off_receipt_sha256', None), ('full_prefix_tangent_carried', False),
                           ('alpha', False), ('allocation_selected', 0),
                           ('independent_validation_ood_accessed', True), ('prior_failure_receipt_sha256', ['a' * 64])]:
            bad = copy.deepcopy(receipt); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.validate_saved_contract(bad, context, cell, 1, 'd' * 64, [])

    def test_checkpoint_requires_both_fd_convergence_and_jvp_agreement(self):
        good = r.checkpoint_payload(*new_checkpoint_fixture())
        self.assertTrue(good['passed'])
        fd_bad = r.checkpoint_payload(*new_checkpoint_fixture(coarse_slope=1, fine_slope=2, jvp=2))
        self.assertFalse(fd_bad['fd_convergence']['passed'])
        self.assertTrue(fd_bad['jvp_vs_fine_fd']['passed'])
        self.assertFalse(fd_bad['passed'])
        jvp_bad = r.checkpoint_payload(*new_checkpoint_fixture(coarse_slope=1, fine_slope=1, jvp=2))
        self.assertTrue(jvp_bad['fd_convergence']['passed'])
        self.assertFalse(jvp_bad['jvp_vs_fine_fd']['passed'])
        self.assertFalse(jvp_bad['passed'])
        context, measured, records = new_checkpoint_fixture()
        del records[0]
        with self.assertRaises(ValueError):
            r.checkpoint_payload(context, measured, records)

    def test_gram_g_and_unequal_sentence_partition_use_token_weighting(self):
        matrix = [[(i - 327.5) / 656, 2 * (i - 327.5) / 656 + .25] + [0.0] * 46 for i in range(656)]
        groups, cursor = [], 0
        for case in range(16):
            count = 26 + 2 * case
            groups.append({'case_id': case, 'token_offset_start': cursor,
                           'token_offset_end': cursor + count, 'gold_tokens': count})
            cursor += count
        result = r.matrix_statistics(matrix, groups)
        self.assertAlmostEqual(result['g'][0], 0, places=14)
        self.assertAlmostEqual(result['g'][1], .25, places=14)
        second_moment = math.fsum(row[0] ** 2 for row in matrix) / 656
        self.assertAlmostEqual(result['gram'][0][0], second_moment, places=14)
        self.assertAlmostEqual(result['gram'][0][1], 2 * second_moment, places=14)
        self.assertAlmostEqual(result['gram'][1][1], 4 * second_moment + .25 ** 2, places=14)
        self.assertTrue(result['psd']['passed'])
        self.assertEqual(len(result['psd']['eigenvalues_ascending']), 48)
        weighted = math.fsum(s['sum_dnll'][0] for s in result['sentence_statistics']) / 656
        unweighted = math.fsum(s['mean_dnll'][0] for s in result['sentence_statistics']) / 16
        self.assertAlmostEqual(weighted, result['g'][0], places=14)
        self.assertGreater(abs(unweighted - weighted), .01)
        wrong_groups = copy.deepcopy(groups); wrong_groups[1]['token_offset_start'] += 1
        for bad_matrix, bad_groups in [(matrix[:-1], groups), ([row[:-1] for row in matrix], groups),
                                        (matrix, wrong_groups)]:
            with self.assertRaises(ValueError):
                r.matrix_statistics(bad_matrix, bad_groups)
        matrix[0][0] = math.nan
        with self.assertRaises(ValueError):
            r.matrix_statistics(matrix, groups)

    def test_eigenvalues_and_residual_bound_on_known_indefinite_matrix(self):
        values, residual = r.jacobi_eigenvalues([[1.0, 2.0], [2.0, 1.0]])
        self.assertAlmostEqual(values[0], -1, places=14)
        self.assertAlmostEqual(values[1], 3, places=14)
        self.assertLessEqual(residual, 1e-12)
        for bad in [[[1.0, 2.0], [3.0, 1.0]], [[math.inf]], [[True]]]:
            with self.assertRaises(ValueError):
                r.jacobi_eigenvalues(bad)

    def test_token_keys_and_sentence_offsets_come_from_binary_metadata(self):
        raw = bytearray(r.common.META_MAGIC)
        targets = []
        for case in range(16):
            ids = list(range(case * 41 + 100, (case + 1) * 41 + 100))
            targets.append(ids)
            raw.extend(struct.pack('<II384i', 2, 41, *(ids + [-1] * (384 - 41))))
        tokens, groups = r.token_partition(bytes(raw), targets)
        self.assertEqual(len(tokens), 656)
        self.assertEqual(tokens[41], {'case_id': 1, 'token_index': 0, 'target_token_id': 141, 'position': 1})
        self.assertEqual(groups[-1]['token_offset_end'], 656)
        wrong = copy.deepcopy(targets); wrong[3][2] += 1
        with self.assertRaises(ValueError):
            r.token_partition(bytes(raw), wrong)


    def test_compiled_c_open_contract_rejects_v1_root_and_accepts_v2(self):
        patch = r.legacy.pinned_module(TOOLS / 'je01_native_role_jvp_patch.py', r.parent.PATCH_SHA,
                                       'all48_v2_path_guard_patch_fixture')
        new_root = r.WORK / 'je01_native_role_jvp_all48_train16_20261123_v2'
        r.require_root(new_root)
        contract = r.validate_native_output_root(patch.C_OPEN, new_root)
        self.assertEqual(contract['required_absolute_prefix'], r.NATIVE_OUTPUT_PREFIX)
        self.assertTrue(contract['validated_before_any_native_execution'])
        with self.assertRaisesRegex(ValueError, 'NATIVE_C_OPEN_REJECTS_OUTPUT_ROOT'):
            r.validate_native_output_root(patch.C_OPEN, r.V1_ROOT)
        with self.assertRaises(ValueError):
            r.require_root(r.V1_ROOT)
        for source in [patch.C_OPEN.replace('je01_native_role_jvp_', 'je01_native_all48_train16_'),
                       patch.C_OPEN.replace('strncmp(je01_jvp_path,prefix,sizeof(prefix)-1)', '0'),
                       patch.C_OPEN + patch.C_OPEN]:
            with self.assertRaises(ValueError):
                r.validate_native_output_root(source, new_root)

    def test_inherited_off_provenance_cannot_claim_a_new_native_execution(self):
        context = synthetic_context()
        off = r.execution_sequence()[0]
        contract = r.cell_contract(context, off, 1, None, [])
        self.assertFalse(contract['native_executed_in_this_plan'])
        self.assertEqual(contract['executed_runner_source_sha256'], r.V1_RUNNER_SHA)
        self.assertEqual(contract['reused_from_v1_plan_receipt_sha256'], r.V1_PLAN_PIN)
        self.assertEqual(contract['original_run_receipt_sha256'], r.V1_OFF_PIN)
        self.assertEqual(contract['original_run_file_sha256'], r.V1_OFF_FILE_PIN)
        self.assertNotEqual(contract['runner_source_sha256'], contract['executed_runner_source_sha256'])
        contract['execution'] = {'native_exit_code': 0, 'stop_reason': None, 'native_started': True,
                                 'elapsed_seconds': 72.38797466736287, 'timeout_seconds': 900,
                                 'rss_limit_gib': 18, 'memory_free_floor_pct': 35}
        r.validate_saved_contract(contract, context, off, 1, None, [])
        for key, value in [('native_executed_in_this_plan', True), ('native_executed_in_this_plan', 0),
                           ('executed_runner_source_sha256', contract['runner_source_sha256']),
                           ('original_run_receipt_sha256', 'd' * 64),
                           ('reused_from_v1_plan_receipt_sha256', context['plan']['receipt_sha256'])]:
            wrong = copy.deepcopy(contract); wrong[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.validate_saved_contract(wrong, context, off, 1, None, [])
        with self.assertRaisesRegex(ValueError, 'OFF_MUST_BE_INHERITED_NEVER_NATIVE_REEXECUTED'):
            r.run_cell(None, context, off, None, [])
        new_fd = r.execution_provenance(r.execution_sequence()[1])
        self.assertTrue(new_fd['native_executed_in_this_plan'])
        self.assertIsNone(new_fd['original_run_receipt_sha256'])
        self.assertEqual(new_fd['recovered_v1_interrupted_cell_receipt_sha256'], r.V1_FAILED_FD_PIN)

    def test_v1_source_is_preserved_and_recovery_requires_independent_confirmation(self):
        self.assertEqual(r.sha(TOOLS / 'je01_native_all48_train16_runner.py'),
                         '7451c867ebb810cff9fe854322e00937bf8b1807c2d0475006d6703be32dd4de')
        contract = r.recovery_contract()
        self.assertEqual(contract['inherited_successful_cell_indices'], [0])
        self.assertEqual(contract['v1_jvp_columns_measured'], 0)
        self.assertTrue(contract['off_native_reexecution_forbidden'])
        self.assertEqual(contract['v1_independent_confirmation_receipt_sha256'],
                         'b21f0e037e27ce30c47bb5ec4ef8b17a44d09c3b38f18389e33ae2fb62df6d3f')


if __name__ == '__main__':
    unittest.main()
