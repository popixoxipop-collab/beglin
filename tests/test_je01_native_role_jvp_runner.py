"""Negative contract tests for the new runner; no real-model claim."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('role_runner_tested', TOOLS / 'je01_native_role_jvp_runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def synthetic_meta_and_jvp():
    """Explicit synthetic 16x41 shape for binary parser corruption tests."""
    meta = bytearray(runner.common.META_MAGIC)
    raw = bytearray(runner.legacy.JVP_MAGIC)
    expected = []
    for row in range(16):
        targets = list(range(100 + row * 41, 100 + (row + 1) * 41))
        meta.extend(struct.pack('<II384i', 2, 41, *(targets + [-1] * (384 - 41))))
        nlls = []
        for index, target in enumerate(targets):
            nll = 2.0 + row * .001 + index * .00001
            nlls.append(nll)
            raw.extend(struct.pack('<4I4d', row, index, target, index + 1, nll, .001, .002, .003))
        expected.append(nlls)
    return bytes(meta), bytes(raw), expected


class NativeRoleRunnerContract(unittest.TestCase):
    def test_predeclared_cell_grid_rejects_scope_expansion(self):
        for role in (0, 1, 2):
            runner.validate_cell('JVP', role, '0', 1)
            for alpha in ('0',) + runner.FD_ALPHAS:
                runner.validate_cell('FD', role, alpha, 1)
        invalid = [('JVP', 3, '0', 1), ('JVP', True, '0', 1), ('JVP', 1, '+0.05', 1),
                   ('FD', 0, '1', 1), ('FD', 0, '+0.01', 1), ('OFF', 0, '0', 1),
                   ('OFF', None, '+0.05', 1), ('JVP', 0, '0', 0), ('JVP', 0, '0', 101)]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                runner.validate_cell(*values)

    def test_fd_parser_rejects_shape_nonfinite_and_wrong_denominator(self):
        good = [[2.0] * 41 for _ in range(16)]
        self.assertEqual(runner.derivative_rows(good, good, .05), [[0.0] * 41 for _ in range(16)])
        short = copy.deepcopy(good); short[2].pop()
        nonfinite = copy.deepcopy(good); nonfinite[3][4] = math.nan
        for left, right, denominator in [(good, short, .05), (nonfinite, good, .05),
                                         (good, good, 0), (good, good, True)]:
            with self.assertRaises(ValueError):
                runner.derivative_rows(left, right, denominator)

    def test_binary_jvp_parser_exact_cardinality_and_primal_alignment(self):
        meta, raw, baseline = synthetic_meta_and_jvp()
        parsed = runner.legacy.read_jvp(raw, meta, baseline)
        self.assertEqual(sum(map(len, parsed['per_token_dnll'])), 656)
        offset = len(runner.legacy.JVP_MAGIC)
        corrupt_target = bytearray(raw); struct.pack_into('<I', corrupt_target, offset + 8, 0)
        corrupt_nll = bytearray(raw); struct.pack_into('<d', corrupt_nll, offset + 16, 2.000000000000001)
        corrupt_tangent = bytearray(raw); struct.pack_into('<d', corrupt_tangent, offset + 24, math.inf)
        for broken in [b'bad' + raw, raw[:-1], raw + b'\0', bytes(corrupt_target),
                       bytes(corrupt_nll), bytes(corrupt_tangent)]:
            with self.subTest(sha=hashlib.sha256(broken).hexdigest()), self.assertRaises(ValueError):
                runner.legacy.read_jvp(broken, meta, baseline)

    def test_resume_seal_is_append_only_and_recomputed(self):
        with tempfile.TemporaryDirectory(prefix='je01_runner_test_') as temporary:
            path = Path(temporary) / 'receipt.json'
            first = runner.write_or_compare(path, {'synthetic': True, 'value': 3}, resume=False)
            before = path.read_bytes()
            same = runner.write_or_compare(path, {'synthetic': True, 'value': 3}, resume=True)
            self.assertEqual(first, same)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                runner.write_or_compare(path, {'synthetic': True, 'value': 4}, resume=True)
            with self.assertRaises(ValueError):
                runner.write_or_compare(path, {'synthetic': True, 'value': 3}, resume=False)
            self.assertEqual(path.read_bytes(), before)

    def test_resume_contract_rejects_other_role_plan_or_runtime(self):
        targets = [list(range(41)) for _ in range(16)]
        context = {'identity': {'synthetic': True}, 'dataset': {'target_token_ids': targets},
                   'build': {'binary_sha256': 'a' * 64}, 'plan': {'receipt_sha256': 'b' * 64}}
        receipt = {'schema': runner.RUN_SCHEMA, 'status': 'SUCCEEDED_EXIT0', 'scope': runner.SCOPE,
                   'kind': 'JVP', 'source_identity': context['identity'], 'role_index': 1,
                   'role_key': runner.common.ROLE_ORDER[1], 'tier': 'Q4', 'alpha': 0,
                   'alpha_literal': '0', 'attempt': 1, 'case_ids': list(range(16)),
                   'target_token_ids': targets, 'plan_receipt_sha256': 'b' * 64,
                   'runner_source_sha256': runner.sha(TOOLS / 'je01_native_role_jvp_runner.py'),
                   'execution': {'native_exit_code': 0, 'stop_reason': None},
                   'native_jvp_enabled': True, 'native_overlay_enabled': True,
                   'fixed_route_replay_enabled': True, 'full_prompt_and_KV_recomputed': True,
                   'router_scores_and_selected_weights_dynamic': True, 'full_prefix_tangent_carried': True,
                   'full48_empirical_fisher': None, 'allocation_selected': False,
                   'independent_validation_ood_accessed': False, 'production_modified': False,
                   'executed_binary_sha256': 'a' * 64}
        runner.validate_saved_contract(receipt, context, 'JVP', 1, '0', 1)
        for key, bad in [('role_index', 2), ('role_key', runner.common.ROLE_ORDER[0]),
                         ('tier', 'Q5'), ('plan_receipt_sha256', 'c' * 64),
                         ('executed_binary_sha256', 'd' * 64), ('fixed_route_replay_enabled', False),
                         ('full_prefix_tangent_carried', False), ('allocation_selected', True),
                         ('independent_validation_ood_accessed', True), ('alpha', .05)]:
            changed = copy.deepcopy(receipt); changed[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_saved_contract(changed, context, 'JVP', 1, '0', 1)


if __name__ == '__main__':
    unittest.main()
