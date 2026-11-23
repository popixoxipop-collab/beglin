"""Execution-controller checks; no model, host connection, or dataset reads."""
from __future__ import annotations

import importlib.util
import json
import fcntl
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'tools/je01_all48_train16_workflow_v2.py'
spec = importlib.util.spec_from_file_location('je01_workflow_tested', SOURCE)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class WorkflowBoundaries(unittest.TestCase):
    def test_native_allowed_root_and_whole_workflow_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            old = work / 'je01_native_all48_train16_rejected_v1'
            good = work / 'je01_native_role_jvp_all48_train16_allowed_v2'
            old.mkdir(); good.mkdir()
            with patch.object(workflow, 'WORK', work):
                with self.assertRaises(ValueError):
                    workflow.checked_root(old)
                self.assertEqual(workflow.checked_root(good), good)
                argv = ['controller', '--root', str(good)]
                for name in ('plan-receipt', 'runner-sha', 'auditor-sha', 'controller-sha'):
                    argv.extend(['--expected-' + name, 'a' * 64])
                with (good / 'WORKFLOW_LOCK').open('a') as held:
                    fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'concurrent_workflow'):
                        workflow.main()

    def test_receipt_pin_and_append_only_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            record = workflow.write_receipt(path, {'test_only': True, 'count': 79})
            raw = path.read_bytes()
            self.assertEqual(workflow.receipt(path, record['receipt_sha256']), record)
            with self.assertRaises(ValueError):
                workflow.receipt(path, '0' * 64)
            with self.assertRaises(ValueError):
                workflow.write_receipt(path, {'test_only': True, 'count': 48})
            self.assertEqual(path.read_bytes(), raw)
            record['count'] = 78
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                workflow.receipt(path)

    def test_actual_child_exit_code_is_preserved(self):
        self.assertEqual(workflow.run_child([sys.executable, '-c', 'raise SystemExit(7)']), 7)

    @unittest.skipUnless(hasattr(signal, 'SIGALRM'), 'POSIX interruption check')
    def test_interruption_terminates_and_reaps_owned_child(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'child.pid'
            command = [sys.executable, '-c',
                       'import os,signal,sys;from pathlib import Path;'
                       'Path(sys.argv[1]).write_text(str(os.getpid()));signal.pause()', str(path)]
            previous = signal.getsignal(signal.SIGALRM)
            def interrupt(signum, frame):
                raise InterruptedError('synthetic controller interruption')
            signal.signal(signal.SIGALRM, interrupt)
            try:
                signal.alarm(1)
                with self.assertRaises(InterruptedError):
                    workflow.run_child(command)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
            self.assertTrue(path.is_file())
            with self.assertRaises(ProcessLookupError):
                os.kill(int(path.read_text()), 0)


if __name__ == '__main__':
    unittest.main()
