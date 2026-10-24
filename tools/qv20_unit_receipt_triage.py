#!/usr/bin/env python3
"""Read-only Qv20 / Tailnet development unit receipt triage.

Separates failures observed before test-run interruption from cascading
cancellations, and enforces the sealed test receipt's stdout checksum.
Never edits logs, dev sessions, model data, release state, or permissions.
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys
import unittest

LOG_LIMIT = 4_000_000
HEADER = re.compile(r"^✖ (.+?)(?:\s+\((\d+(?:\.\d+)?)ms\))?$", re.M)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def basename(s: str) -> str:
    return pathlib.PurePosixPath(s.rsplit(' (', 1)[0]).name[:128]


def analyze(log: bytes, receipt: dict):
    if len(log) > LOG_LIMIT:
        raise ValueError('log_size_limit')
    if receipt.get('schema') != 'tc-development-test-receipt/1':
        raise ValueError('receipt_schema_invalid')
    if receipt.get('profile') != 'unit':
        raise ValueError('wrong_test_profile')
    if receipt.get('stdout_sha256') != digest(log):
        raise ValueError('stdout_receipt_sha256_mismatch')
    txt = ANSI.sub('', log.decode('utf-8', 'replace'))
    cut = txt.find('Interrupted while running:')
    before, after = (txt[:cut], txt[cut:]) if cut >= 0 else (txt, '')
    direct = [(basename(m.group(1)), float(m.group(2)) if m.group(2) else None)
              for m in HEADER.finditer(before)]
    cancelled = [basename(m.group(1)) for m in HEADER.finditer(after)]
    return {
        'schema': 'BEGLIN-Qv20-isolated-unit-receipt-triage/1',
        'read_only': True,
        'receipt_sha256_valid': True,
        'timed_out': receipt.get('timed_out') is True,
        'exit_code': receipt.get('exit_code'),
        'interruption_marker_present': cut >= 0,
        'direct_failed_headers_before_interruption': len(direct),
        'post_interruption_failed_headers': len(cancelled),
        'direct_failure_samples': [{'test': n, 'reported_ms': ms}
                                   for n, ms in direct[:25]],
        'post_interruption_sample_names': cancelled[:12],
        'inference_limit': 'not_all_post_interruption_tests_can_be_called_direct_test_failures',
        'full_unit_gate_pass': receipt.get('state') == 'PASS' and
                               receipt.get('exit_code') == 0 and
                               receipt.get('timed_out') is False,
    }


def run_selftest():
    class Test(unittest.TestCase):
        def record(self, text, **kwargs):
            b = text.encode('utf-8')
            rec = dict(schema='tc-development-test-receipt/1', profile='unit',
                       state='FAIL', exit_code=1, timed_out=True,
                       stdout_sha256=digest(b))
            rec.update(kwargs)
            return b, rec

        def test_separates_direct_and_interrupted(self):
            b, r = self.record('✖ alpha.test.mjs (21.3ms)\nInterrupted while running:\n✖ /x/beta.test.mjs\n')
            o = analyze(b, r)
            self.assertEqual(o['direct_failed_headers_before_interruption'], 1)
            self.assertEqual(o['post_interruption_failed_headers'], 1)
            self.assertEqual(o['direct_failure_samples'][0]['test'], 'alpha.test.mjs')
            self.assertFalse(o['full_unit_gate_pass'])

        def test_digest_fail_closed(self):
            b, r = self.record('✔ ok\n')
            with self.assertRaisesRegex(ValueError, 'sha256'):
                analyze(b + b'corrupt', r)

        def test_schema_fail_closed(self):
            b, r = self.record('✔ ok\n', profile='policy')
            with self.assertRaisesRegex(ValueError, 'profile'):
                analyze(b, r)

        def test_clean_run(self):
            b, r = self.record('✔ ok\n', state='PASS', exit_code=0, timed_out=False)
            self.assertTrue(analyze(b, r)['full_unit_gate_pass'])

        def test_limit(self):
            b, r = self.record('x')
            with self.assertRaisesRegex(ValueError, 'size'):
                analyze(b'x' * (LOG_LIMIT + 1), r)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if res.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=pathlib.Path)
    parser.add_argument('--stdout-log', type=pathlib.Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return run_selftest()
    if not args.receipt or not args.stdout_log:
        parser.error('--receipt and --stdout-log must both be supplied')
    try:
        receipt = json.loads(args.receipt.read_text('utf-8'))
        with args.stdout_log.open('rb') as f:
            log = f.read(LOG_LIMIT + 1)
        output = analyze(log, receipt)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({'status': 'TRIAGE_NOT_READY', 'error_type': type(exc).__name__,
                          'no_gate_override': True}, sort_keys=True))
        return 4
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if output['full_unit_gate_pass'] else 2


if __name__ == '__main__':
    sys.exit(main())
