#!/usr/bin/env python3
"""Synthetic-only offline contract tests; zero original Qwen or ARC items used."""
import hashlib
import json
import math
import pathlib
import tempfile
import unittest

import qv20_native_calibration_receipt_gate as gate


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())


def save_json(p, obj):
    write(p, json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


class GateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qv20_receipt_synthetic_')
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name) / 'synthetic_not_actual_EOE'
        self.exe = self.root / 'dummy_C_binary'
        write(self.exe, b'test-only-cannot-run')
        self.pins = {'binary': digest(self.exe)}
        public = self.root / 'Qv20_PREEXEC_PUBLIC.json'
        write(public, '{"synthetic":true}\n')
        self.pins['public'] = digest(public)
        self.rows = [
            {'correct': 'ABCD'[i % 4], 'prefix_len': 20 if i < 16 else 200}
            for i in range(32)
        ]
        cal = self.root / 'cal_arc_train' / 'frozen_inputs_private.json'
        save_json(cal, {'selected_rows': self.rows, 'synthetic_only': True})
        self.pins['cal'] = digest(cal)
        for arm in ('static_source', 'static_random'):
            mask = self.root / 'masks' / (arm + '.txt')
            write(mask, ('synthetic-' + arm).encode())
            self.pins[arm] = digest(mask)
            self._make_arm(arm)

    def _folder(self, arm):
        return self.root / 'native_original_C_static' / 'cal_arc_train' / arm

    def _make_arm(self, arm):
        short_wins = arm == 'static_source'
        scores = [0.25 if (i < 16) == short_wins else 2.0 for i in range(32)]
        lines = ['projection_role_tensors: f32=0 n8=28 q4=168']
        for i, (score, row) in enumerate(zip(scores, self.rows)):
            answer = 'ABCD'.index(row['correct'])
            lines.append(f'[qv5-choice] window {i} NLL {score:.8f} expected {answer} predicted {answer} prefix {row["prefix_len"]}')
        mean = sum(scores) / 32
        lines.append(f'qv5_choice NLL {sum(scores):.8f} items 32 correct 32 mean {mean:.8f} ppl {math.exp(mean):.8f}')
        folder = self._folder(arm)
        log = folder / 'native_original_C_Qwen28.log'
        write(log, '\n'.join(lines) + '\n')
        receipt = {
            'schema': gate.SCHEMA,
            'state': 'ACTUAL_ORIGINAL_QWEN28_C_STATIC_' + arm.upper() + '_PASS',
            'cohort': 'cal_arc_train', 'arm': arm, 'n': 32,
            'original_Qwen28_C_executable_sha256': self.pins['binary'],
            'source_frozen_private_SHA256': self.pins['cal'],
            'source_public_preexec_SHA256': self.pins['public'],
            'real_native_original_C_log_SHA256': digest(log),
            'original_full_vocab_gold_nexttoken_NLL': scores,
            'original_choice_correct_each': [1] * 32,
            'correct': 32, 'mean_NLL': mean,
            'real_native_Q4_N8_weights_active': True,
            'no_model_kernel_or_operator_release_changes': True,
        }
        private = folder / 'PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json'
        save_json(private, receipt)
        self._refresh_public(arm)

    def _refresh_public(self, arm):
        folder = self._folder(arm)
        private = folder / 'PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json'
        doc = json.loads(private.read_text())
        public = {k: v for k, v in doc.items()
                  if k not in ('original_full_vocab_gold_nexttoken_NLL', 'original_choice_correct_each')}
        public['private_scores_receipt_SHA256'] = digest(private)
        save_json(folder / 'PUBLIC_AGGREGATE.json', public)

    def audit(self):
        return gate.audit(root=self.root, original_exe=self.exe, pins=self.pins)

    def test_valid_paired_receipts_without_any_heldout_data(self):
        assert not (self.root / 'test_arc_validation').exists()
        assert not (self.root / 'test_ob_validation').exists()
        result = self.audit()
        self.assertEqual(result['status'], 'BOTH_ORIGINAL_NATIVE_TRAIN_ARMS_VERIFIED')
        self.assertFalse(result['heldout_data_opened'])
        self.assertEqual(set(result['aggregate']), {'static_source', 'static_random'})
        self.assertEqual(result['aggregate']['static_source']['n'], 32)

    def test_missing_random_arm_fails_closed(self):
        (self._folder('static_random') / 'PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json').unlink()
        with self.assertRaises(OSError):
            self.audit()

    def test_wrong_mask_hash_fails(self):
        with (self.root / 'masks' / 'static_random.txt').open('ab') as f:
            f.write(b'not-frozen')
        with self.assertRaises(gate.AuditFailure):
            self.audit()

    def test_corrupted_native_log_fails(self):
        with (self._folder('static_random') / 'native_original_C_Qwen28.log').open('ab') as f:
            f.write(b'altered')
        with self.assertRaises(gate.AuditFailure):
            self.audit()

    def test_wrong_receipt_arm_even_with_matching_public_digest_fails(self):
        arm = 'static_random'
        private = self._folder(arm) / 'PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json'
        doc = json.loads(private.read_text())
        doc['arm'] = 'static_source'
        save_json(private, doc)
        self._refresh_public(arm)
        with self.assertRaises(gate.AuditFailure):
            self.audit()

    def test_individual_score_tamper_even_with_rehashed_public_fails(self):
        arm = 'static_random'
        private = self._folder(arm) / 'PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json'
        doc = json.loads(private.read_text())
        doc['original_full_vocab_gold_nexttoken_NLL'][0] += 0.25
        doc['mean_NLL'] += 0.25 / 32
        save_json(private, doc)
        self._refresh_public(arm)
        with self.assertRaises(gate.AuditFailure):
            self.audit()

    def test_corrupt_public_digest_fails(self):
        arm = 'static_random'
        pub = self._folder(arm) / 'PUBLIC_AGGREGATE.json'
        doc = json.loads(pub.read_text())
        doc['private_scores_receipt_SHA256'] = '0' * 64
        save_json(pub, doc)
        with self.assertRaises(gate.AuditFailure):
            self.audit()

    def test_corrupt_native_case_mapping_with_matching_log_digest_fails(self):
        arm = 'static_random'
        folder = self._folder(arm)
        log = folder / 'native_original_C_Qwen28.log'
        raw = log.read_text().replace('window 0 NLL ', 'window 30 NLL ', 1)
        write(log, raw)
        private = folder / 'PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json'
        doc = json.loads(private.read_text())
        doc['real_native_original_C_log_SHA256'] = digest(log)
        save_json(private, doc)
        self._refresh_public(arm)
        with self.assertRaises(gate.AuditFailure):
            self.audit()


if __name__ == '__main__':
    unittest.main(verbosity=2)
