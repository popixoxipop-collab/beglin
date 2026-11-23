"""Independent-auditor corruption and mathematical controls; no model claims."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'tools/je01_native_all48_train16_independent_audit.py'
SPEC = importlib.util.spec_from_file_location('independent_je01_all48_auditor_under_test', SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture():
    counts = [40, 42] + [41] * 14
    starts = [3 + case % 2 for case in range(16)]
    targets = [[100 + case * 50 + token for token in range(count)] for case, count in enumerate(counts)]
    meta = audit.META_MAGIC + b''.join(struct.pack('<II384i', starts[case], count,
                                        *(targets[case] + [-1] * (384 - count)))
                                     for case, count in enumerate(counts))
    gold, jvp, baseline = bytearray(audit.GOLD_MAGIC), bytearray(audit.JVP_MAGIC), []
    for case, row in enumerate(targets):
        values = []
        for index, token in enumerate(row):
            key = case, index, token, starts[case] + index - 1
            nll = 1. + case * .01 + index * .001
            gold.extend(struct.pack('<4IdIf', *key, nll, token, 2.))
            jvp.extend(struct.pack('<4I4d', *key, nll, .001 * (case + index), -.02, .04))
            values.append(nll)
        baseline.append(values)
    return starts, targets, meta, bytes(gold), bytes(jvp), baseline


class MemoryEvidence:
    def __init__(self, value):
        self.value = value
    def receipt(self, path, *args):
        return self.value


class IndependentAuditTests(unittest.TestCase):
    def setUp(self):
        self.starts, self.targets, self.meta, self.gold, self.jvp, self.baseline = fixture()

    def test_raw_records_and_unequal_sentence_partition(self):
        self.assertEqual(audit.metadata(self.meta), (self.starts, self.targets))
        self.assertEqual(audit.gold(self.gold, self.starts, self.targets), self.baseline)
        decoded = audit.jvp(self.jvp, self.starts, self.targets, self.baseline)
        self.assertEqual(sum(map(len, decoded['per_token_dnll'])), 656)
        self.assertEqual(audit.token_partition(self.targets)[:2], [
            {'case_id': 0, 'token_offset_start': 0, 'token_offset_end': 40, 'gold_tokens': 40},
            {'case_id': 1, 'token_offset_start': 40, 'token_offset_end': 82, 'gold_tokens': 42}])

    def test_raw_token_target_position_and_primal_corruption(self):
        base = len(audit.JVP_MAGIC)
        corruptions = []
        for offset, value in [(0, 1), (4, 1), (8, 999), (12, 55)]:
            altered = bytearray(self.jvp)
            struct.pack_into('<I', altered, base + offset, value)
            corruptions.append(bytes(altered))
        altered = bytearray(self.jvp)
        struct.pack_into('<d', altered, base + 16, math.nextafter(self.baseline[0][0], math.inf))
        corruptions.append(bytes(altered))
        for raw in corruptions:
            with self.assertRaises(ValueError):
                audit.jvp(raw, self.starts, self.targets, self.baseline)

    def test_nonfinite_values_truncation_and_trailing_bytes(self):
        for offset, value in [(24, math.nan), (32, math.inf), (40, -.1)]:
            altered = bytearray(self.jvp)
            struct.pack_into('<d', altered, len(audit.JVP_MAGIC) + offset, value)
            with self.assertRaises(ValueError):
                audit.jvp(bytes(altered), self.starts, self.targets, self.baseline)
        for raw in (self.jvp[:-1], self.jvp + b'\0', b'X' + self.jvp[1:]):
            with self.assertRaises(ValueError):
                audit.jvp(raw, self.starts, self.targets, self.baseline)
        with self.assertRaises(ValueError):
            audit.gold(self.gold + b'\0', self.starts, self.targets)

    def test_json_resealing_does_not_defeat_external_pin(self):
        original = {'value': 1}
        pin = audit.canonical(original)
        sealed = {**original, 'receipt_sha256': pin}
        audit.sealed_object(sealed, pin)
        altered = {'value': 2}
        altered['receipt_sha256'] = audit.canonical(altered)
        with self.assertRaises(ValueError):
            audit.sealed_object(altered, pin)
        for raw in (b'{"value":1,"value":2}', b'{"value":Infinity}', b'{"value":NaN}'):
            with self.assertRaises(ValueError):
                audit.strict_json(raw)
        for actual, expected in ((True, 1), (False, 0.), (1., 1)):
            with self.assertRaises(ValueError):
                audit.exact(actual, expected, 'numeric_boolean_alias')

    def test_fd_numerical_failure_and_zero_direction_remain_explicit(self):
        self.assertTrue(audit.comparison([.1] * 656, [.1] * 656)['passed'])
        self.assertFalse(audit.comparison([.2] * 656, [.1] * 656)['passed'])
        result = audit.comparison([0.] * 656, [0.] * 656)
        self.assertTrue(result['passed'])
        self.assertIsNone(result['cosine_similarity'])
        self.assertEqual(audit.finite_difference([2.] * 656, [1.] * 656, .05), [10.] * 656)

    def test_known_full48_spectrum_and_indefinite_control(self):
        n = 48
        signs = [(-1.) ** index for index in range(n)]
        matrix = [[.03 / n + .12 * signs[i] * signs[j] / n for j in range(n)] for i in range(n)]
        result = audit.spectral_check(matrix)
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['eigenvalues_ascending'][-1], .12, places=13)
        self.assertAlmostEqual(result['eigenvalues_ascending'][-2], .03, places=13)
        self.assertLess(max(map(abs, result['eigenvalues_ascending'][:-2])), 1e-12)
        self.assertFalse(audit.spectral_check([[1., 2.], [2., 1.]])['passed'])
        self.assertTrue(audit.spectral_check([[0.] * n for _ in range(n)])['passed'])
        with self.assertRaises(ValueError):
            audit.spectral_check([[1., 0.], [.1, 1.]])

    def test_g_gram_and_sentence_weighting_are_token_correct(self):
        # One nonzero direction is a ramp with a closed-form first/second moment.
        matrix = [[float(index + 1)] + [0.] * 47 for index in range(656)]
        gradient, gram, groups = audit.gram_statistics(matrix, self.targets)
        self.assertEqual(gradient[0], 657. / 2)
        self.assertEqual(gram[0][0], 657. * 1313. / 6)
        self.assertEqual(gradient[1:], [0.] * 47)
        self.assertEqual(sum(g['sum_dnll'][0] for g in groups), 656. * 657. / 2)
        self.assertNotEqual(sum(g['mean_dnll'][0] for g in groups) / 16, gradient[0])
        self.assertNotEqual(gram[0][0], gradient[0] ** 2)
        self.assertEqual(groups[1]['sum_dnll'][0], sum(range(41, 83)))
        changed = copy.deepcopy(matrix)
        changed[15][3] = math.nan
        with self.assertRaises(ValueError):
            audit.gram_statistics(changed, self.targets)

    def test_native_configuration_counter_and_down_mapping(self):
        prefix = ('JE01_ROLE_DIRECTION_REGISTERED\nJE01_FIXED_BASELINE_ROUTES_ENABLED records=112684\n'
                  'JE01_ROLE_DIRECTION_CONFIG role=23 layer=24 expert=60 tier=Q4 alpha=0\n'
                  'JE01_ONE_COLUMN_JVP_PREFIX_COMPLETE tokens=4334 cases=32 layers=117018 gold=656\n'
                  'RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16\n')
        audit.native_log(prefix.encode(), 'JVP', 23, 'Q4', '0')
        corruptions = [prefix.replace('expert=60', 'expert=59'), prefix.replace('tier=Q4', 'tier=Q5'),
                       prefix.replace('gold=656', 'gold=655'), prefix + 'JE01_ROLE_DIRECTION_CONFIG malformed\n',
                       prefix + 'JE01_ONE_COLUMN_JVP_PREFIX_COMPLETE malformed\n', prefix + 'FATAL: bad\n']
        for text in corruptions:
            with self.assertRaises(ValueError):
                audit.native_log(text.encode(), 'JVP', 23, 'Q4', '0')

    def test_exact_sequence_and_checkpoint_coverage(self):
        cells = audit.sequence()
        self.assertEqual(len(cells), 79)
        self.assertEqual([(r['role_index'], r['tier']) for r in cells if r['kind'] == 'JVP'][:3],
                         [(0, 'Q5'), (1, 'Q5'), (2, 'Q5')])
        self.assertEqual({r['column_index'] for r in cells if r['kind'] == 'JVP'}, set(range(48)))
        self.assertEqual(sum(r['kind'] == 'FD' for r in cells), 30)
        self.assertEqual(len(audit.NEW_FD + audit.PRIOR_FD), 9)

    def test_resealed_matrix_and_partition_corruptions_rejected(self):
        matrix = [[float(i * 48 + j) for j in range(48)] for i in range(656)]
        measured = {i: {'receipt_sha256': str(i), 'gradient_output_sha256': str(i + 48)} for i in range(48)}
        context = {'root': Path('/tmp/audit'), 'plan': {'receipt_sha256': 'plan'}, 'runner_sha': 'runner',
                   'token_rows': audit.token_keys(self.starts, self.targets),
                   'sentence_groups': audit.token_partition(self.targets)}
        expected = {'schema': 'beglin-je01-native-all48-train16-raw-matrix/1', 'scope': audit.SCOPE,
                    'plan_receipt_sha256': 'plan', 'runner_source_sha256': 'runner', 'shape': [656, 48],
                    'coordinate_order': audit.coordinate_order(), 'token_rows': context['token_rows'],
                    'per_token_dnll': matrix, 'sentence_groups': context['sentence_groups'],
                    'jvp_receipt_sha256': [str(i) for i in range(48)],
                    'jvp_output_sha256': [str(i + 48) for i in range(48)], **audit.NO_ACTION}
        expected['receipt_sha256'] = audit.canonical(expected)
        audit.verify_matrix_artifact(MemoryEvidence(expected), context, measured, matrix)
        for corruption in ('swap_columns', 'group_overlap', 'target_id', 'hidden_extra_claim'):
            altered = copy.deepcopy(expected)
            if corruption == 'swap_columns':
                for row in altered['per_token_dnll']:
                    row[1], row[2] = row[2], row[1]
            elif corruption == 'group_overlap':
                altered['sentence_groups'][1]['token_offset_start'] -= 1
            elif corruption == 'target_id':
                altered['token_rows'][7]['target_token_id'] += 1
            else:
                altered['heldout_validated'] = True
            altered['receipt_sha256'] = audit.canonical({k: v for k, v in altered.items() if k != 'receipt_sha256'})
            with self.assertRaises(ValueError):
                audit.verify_matrix_artifact(MemoryEvidence(altered), context, measured, matrix)

    def test_no_overwrite_or_escape_and_symlink_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            path = root / 'JE01_ALL48_TRAIN16_INDEPENDENT_AUDIT.json'
            audit.write_output(path, root, {'status': 'SYNTHETIC_TEST_ONLY'})
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                audit.write_output(path, root, {'status': 'OVERWRITE'})
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                audit.write_output(root / 'PLAN_RECEIPT.json', root, {})
            preflight = root / 'ALL48_PREFLIGHT_INDEPENDENT_AUDIT.json'
            audit.write_output(preflight, root, {'status': 'PARTIAL_NO_PASS'})
            self.assertEqual(json.loads(preflight.read_text())['status'], 'PARTIAL_NO_PASS')
            link = root / 'linked.json'
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                audit.Evidence().read(link)

    def test_failed_attempt_after_prefix_gap_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            next_cell = audit.sequence()[1]
            path = root / 'runs/train16' / (audit.cell_label(next_cell) + '_attempt1')
            path.mkdir(parents=True)
            (path / 'FAILED_RUN_RECEIPT.json').write_text('{}')
            context = {'root': root}
            with self.assertRaisesRegex(ValueError, 'native_attempt_after_gap'):
                audit.verify_prefix(MemoryEvidence({}), context, {}, {})

    def test_actual_sealed_plan_schema_matches_independent_contract(self):
        """Schema comparison only: deliberately does not claim remote file checks."""
        folder = SOURCE.parents[1]
        path = folder / 'docs/research/JE01_ALL48_TRAIN16_PRERUN_PLAN_20261123.json'
        if not path.exists():
            path = folder / 'ALL48_PLAN_RECEIPT.json'
        if not path.exists():
            self.skipTest('Pinned actual plan is an external optional schema fixture')
        raw = path.read_bytes()
        self.assertEqual(audit.sha_bytes(raw), 'dc4cf87f557ef806d44c02cd9bf66a60dbec893303d74aa89cbf9ef0c5a3454f')
        plan = audit.sealed_object(audit.strict_json(raw),
                                  '60fdced19fad43cbf2b73801dd68effbbd75c186f8a5f39de9274cad39453419')
        parent_path = folder / 'docs/research/JE01_THREE_ROLE_JVP_PRERUN_PLAN_20261123.json'
        if not parent_path.exists():
            parent_path = folder / 'THREE_ROLE_PLAN_RECEIPT.json'
        parent_plan = audit.strict_json(parent_path.read_bytes())
        patch_contract = {'supported_role_indices': list(range(24)), 'supported_tiers': ['Q4', 'Q5'],
                          'alpha': 0, 'one_direction_per_run': True, 'runtime_mode': 'ROLE_DIRECTION_FIXED_Q4Q5'}
        class SchemaEvidence:
            def read(self, *args, **kwargs):
                return b''
            def receipt(self, source, *args):
                if source == audit.ROOT / 'PLAN_RECEIPT.json':
                    return plan
                if source == audit.PARENT / 'PLAN_RECEIPT.json':
                    return parent_plan
                if source == audit.PARENT / 'SOURCE_PATCH_RECEIPT.json':
                    return patch_contract
                raise AssertionError(source)
        context = {'dataset': {'manifest': plan['runtime']['manifest_path'],
                               'gold_metadata': plan['runtime']['metadata_path']},
                   'targets': plan['target_token_ids']}
        args = SimpleNamespace(root=audit.ROOT, expected_plan_receipt=plan['receipt_sha256'],
                               expected_runner_sha=plan['runner_source_sha256'])
        with patch.object(audit, 'safe_path', side_effect=lambda p, **kwargs: Path(p)):
            checked, identity = audit.verify_plan(args, SchemaEvidence(), context)
        self.assertEqual(checked['receipt_sha256'], plan['receipt_sha256'])
        self.assertEqual(identity['plan_receipt_sha256'], plan['receipt_sha256'])


if __name__ == '__main__':
    unittest.main()
