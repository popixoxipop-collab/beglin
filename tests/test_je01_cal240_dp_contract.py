#!/usr/bin/env python3
"""Synthetic-only tests for JE01 CAL240 real-activation fixed-byte DP.

No checkpoint, original hidden activation, holdout gold, GPU, or production runner
is read by this test. The function under test is extracted from the actual
source by Python AST so that its mathematical logic is executed verbatim.
"""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools' / 'je01_eoe_actual_cal240_h2_sw_native_activation_dp.py'
ROLES = ('gate_proj', 'up_proj', 'down_proj')
FROZEN = ((12, 13), (13, 22), (13, 28), (14, 63),
          (20, 17), (21, 61), (24, 26), (24, 60))
FIELDS = ('route_weighted_output_SSE', 'route_weighted_output_relative_MSE')


def read_contract():
    source = SOURCE.read_text(encoding='utf-8')
    tree = ast.parse(source, str(SOURCE))
    names = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('BUDGET', 'COSTS'):
                    names[target.id] = ast.literal_eval(node.value)
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'solve']
    if len(funcs) != 1:
        raise AssertionError('one JE01 solve function required')
    sub = ast.Module(body=funcs, type_ignores=[])
    ast.fix_missing_locations(sub)
    env = {}
    exec(compile(sub, str(SOURCE), 'exec'), env, env)
    return source, tree, names, env['solve']


def fake_roles():
    rows = []
    for layer, expert in FROZEN:
        for role in ROLES:
            record = {'layer': layer, 'expert': expert, 'role': role}
            for f in FIELDS:
                record[f + '_AF4'] = 1.0
                record[f + '_Q4'] = 1.40
                record[f + '_Q5'] = 0.90
            rows.append(record)
    return rows


class TestJE01Cal240DP(unittest.TestCase):
    def test_exact_original_bytes_and_tier_balancing(self):
        _, _, meta, _ = read_contract()
        self.assertEqual(meta['BUDGET'], 43_253_760)
        self.assertEqual(meta['COSTS'], {'AF4': 1_802_240, 'Q4': 1_622_016, 'Q5': 1_982_464})
        self.assertEqual(24 * meta['COSTS']['AF4'], meta['BUDGET'])
        self.assertEqual(meta['COSTS']['AF4'] - meta['COSTS']['Q4'],
                         meta['COSTS']['Q5'] - meta['COSTS']['AF4'])

    def test_unprofitable_equal_byte_trade_keeps_all_original_af4(self):
        _, _, _, solve = read_contract()
        for f in FIELDS:
            out = solve(fake_roles(), f)
            self.assertEqual(out['best_overall']['risk_difference'], 0.0)
            self.assertEqual(out['best_overall']['unchanged_AF4_roles'], 24)
            self.assertEqual(out['best_nonbaseline_exactbyte']['q4_roles'], 1)
            self.assertEqual(out['best_nonbaseline_exactbyte']['q5_roles'], 1)
            self.assertAlmostEqual(out['best_nonbaseline_exactbyte']['risk_difference'], 0.30, places=8)

    def test_profitable_pair_is_selected_without_exceeding_budget(self):
        _, _, meta, solve = read_contract()
        rows = fake_roles()
        for f in FIELDS:
            rows[0][f + '_Q4'] = 1.01  # cheapest sacrificed Q4 role
            rows[1][f + '_Q5'] = 0.80  # highest recovered Q5 role
            out = solve(rows, f)
            best = out['best_overall']
            self.assertEqual(best['q4_roles'], 1)
            self.assertEqual(best['q5_roles'], 1)
            self.assertEqual(best['unchanged_AF4_roles'], 22)
            self.assertAlmostEqual(best['risk_difference'], -0.19, places=8)
            self.assertEqual(sum(meta['COSTS'][choice[-1]] for choice in best['source_assignment']),
                             meta['BUDGET'])

    def test_equal_byte_optimization_is_deterministic(self):
        _, _, _, solve = read_contract()
        rows = fake_roles()
        self.assertEqual(solve(rows, FIELDS[0]), solve(rows, FIELDS[0]))

    def test_source_excludes_pilot16_and_does_not_claim_fisher_or_nll(self):
        source, tree, _, _ = read_contract()
        comparisons = [ast.unparse(node.test) for node in ast.walk(tree)
                       if isinstance(node, ast.If)]
        self.assertIn('global_case >= 16', comparisons)
        self.assertIn('training_unseen240_prompt_count', source)
        self.assertIn('source_heldout_OOD_labels_used', source)
        self.assertIn('true_logprob_gradients', source)
        self.assertIn('NOT_MEASURED', source)
        self.assertIn('NOT_TRUE_LOSS_FISHER_OR_HESSIAN', source)
        self.assertIn('down_proj', source)


if __name__ == '__main__':
    unittest.main()
