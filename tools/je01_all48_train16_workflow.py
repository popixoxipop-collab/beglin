#!/usr/bin/env python3
"""Run the sealed TRAIN16 study and its separate independent audit once.

This controller does not implement model mathematics. It binds the exact plan,
runner, and independent auditor before launching either subprocess. It never
submits additional jobs, selects an allocation, or opens TRAIN240/holdouts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

WORK = Path('/Volumes/D50/TailnetCommander/work')
BUILD = '2cbeb4dc1a7e1ab5642a4abb22395498c10dde43bee051eec3c303c7e53fa008'
RUNNER_NAME = 'je01_native_all48_train16_runner.py'
AUDITOR_NAME = 'je01_native_all48_train16_independent_audit.py'
COMPLETE = 'ALL48_NATIVE_COMPLETE_CHECKPOINTS_AND_PSD_PASS'


def require(condition, message):
    if not condition:
        raise ValueError('JE01_ALL48_WORKFLOW:' + message)


def file_sha(path):
    path = Path(path)
    require(path.is_file() and path.resolve() == path and not path.is_symlink(), 'file_path:' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def receipt(path, expected=None):
    file_sha(path)
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), 'receipt_type')
    body = {key: item for key, item in value.items() if key != 'receipt_sha256'}
    require(value.get('receipt_sha256') == canonical(body), 'receipt_content')
    require(expected is None or value['receipt_sha256'] == expected, 'receipt_external_pin')
    return value


def write_receipt(path, value):
    require(not Path(path).exists() and not Path(path).is_symlink(), 'append_only_workflow_receipt')
    result = {**value, 'receipt_sha256': canonical(value)}
    with Path(path).open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return result


def checked_root(value):
    path = Path(value)
    require(path.is_absolute() and path.parent == WORK and path.resolve() == path
            and path.name.startswith('je01_native_all48_train16_')
            and path.is_dir() and not path.is_symlink(), 'isolated_research_root')
    return path


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}, sort_keys=True), flush=True)


def run_child(command):
    """Interrupt only this controller's exact child; its native guard cleans up."""
    process = subprocess.Popen(command)
    try:
        return process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=12)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-plan-receipt', required=True)
    parser.add_argument('--expected-runner-sha', required=True)
    parser.add_argument('--expected-auditor-sha', required=True)
    parser.add_argument('--expected-controller-sha', required=True)
    parser.add_argument('--workflow-attempt', type=int, default=1)
    args = parser.parse_args()
    require(type(args.workflow_attempt) is int and 1 <= args.workflow_attempt <= 100, 'attempt_range')
    for name in ('expected_plan_receipt', 'expected_runner_sha', 'expected_auditor_sha',
                 'expected_controller_sha'):
        require(re.fullmatch('[0-9a-f]{64}', getattr(args, name)) is not None, 'sha_argument:' + name)
    root = checked_root(args.root)
    lock_path = root / 'WORKFLOW_LOCK'
    require(lock_path.resolve() == lock_path and not lock_path.is_symlink(), 'workflow_lock_path')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('JE01_ALL48_WORKFLOW:concurrent_workflow_forbidden') from error
        return run_workflow(args, root)


def run_workflow(args, root):
    output = root / f'WORKFLOW_RECEIPT_{args.workflow_attempt:04d}.json'
    require(not output.exists() and not output.is_symlink(), 'workflow_attempt_already_recorded')
    runner, auditor = root / 'tools' / RUNNER_NAME, root / 'tools' / AUDITOR_NAME
    require(file_sha(Path(__file__).absolute()) == args.expected_controller_sha, 'controller_source_pin')
    require(file_sha(runner) == args.expected_runner_sha, 'runner_source_pin')
    require(file_sha(auditor) == args.expected_auditor_sha, 'auditor_source_pin')
    plan = receipt(root / 'PLAN_RECEIPT.json', args.expected_plan_receipt)
    require(plan.get('runner_source_sha256') == args.expected_runner_sha, 'plan_runner_binding')
    started = time.monotonic()
    data = {'schema': 'beglin-je01-all48-train16-workflow/1',
            'started_at': datetime.now(timezone.utc).isoformat(),
            'root': str(root), 'workflow_attempt': args.workflow_attempt,
            'plan_receipt_sha256': args.expected_plan_receipt,
            'runner_source_sha256': args.expected_runner_sha,
            'auditor_source_sha256': args.expected_auditor_sha,
            'controller_source_sha256': args.expected_controller_sha,
            'runner_exit_code': None, 'auditor_exit_code': None,
            'analysis_receipt_sha256': None, 'independent_audit_receipt_sha256': None,
            'status': 'STARTED_NOT_COMPLETE', 'train240_execution': 'NOT_RUN',
            'allocation_selected': False, 'independent_validation_ood_accessed': False,
            'production_modified': False}
    code = 1
    emit('ALL48_WORKFLOW_START', plan=args.expected_plan_receipt, root=str(root))
    try:
        runner_command = [sys.executable, str(runner), 'run', '--root', str(root),
                          '--expected-plan-receipt', args.expected_plan_receipt,
                          '--max-new-cells', '79', '--resume', '--timeout-seconds', '900']
        data['runner_exit_code'] = run_child(runner_command)
        require(file_sha(runner) == args.expected_runner_sha and
                file_sha(auditor) == args.expected_auditor_sha, 'source_drift_after_native')
        if data['runner_exit_code'] == 0:
            analysis = receipt(root / 'JE01_ALL48_TRAIN16_ANALYSIS.json')
            require(analysis.get('status') == COMPLETE and
                    analysis.get('plan_receipt_sha256') == args.expected_plan_receipt and
                    analysis.get('completed_native_cell_count') == 79 and
                    analysis.get('native_jvp_columns') == 48 and
                    analysis.get('fd_verified_coordinates') == 9, 'runner_full_completion')
            data['analysis_receipt_sha256'] = analysis['receipt_sha256']
        elif data['runner_exit_code'] == 2:
            hold = receipt(root / 'NUMERICAL_HOLD_RECEIPT.json')
            require(hold.get('plan_receipt_sha256') == args.expected_plan_receipt, 'hold_plan_binding')
            data['numerical_hold_receipt_sha256'] = hold['receipt_sha256']
        else:
            raise RuntimeError('native_runner_failed:' + str(data['runner_exit_code']))
        audit_output = root / 'JE01_ALL48_TRAIN16_INDEPENDENT_AUDIT.json'
        require(not audit_output.exists(), 'independent_audit_already_exists_reconcile_first')
        audit_command = [sys.executable, str(auditor), '--root', str(root),
                         '--expected-build-receipt', BUILD,
                         '--expected-plan-receipt', args.expected_plan_receipt,
                         '--expected-runner-sha', args.expected_runner_sha,
                         '--output', str(audit_output)]
        emit('ALL48_WORKFLOW_INDEPENDENT_AUDIT_START', native_exit_code=data['runner_exit_code'])
        data['auditor_exit_code'] = run_child(audit_command)
        require(file_sha(runner) == args.expected_runner_sha and
                file_sha(auditor) == args.expected_auditor_sha and
                file_sha(Path(__file__).absolute()) == args.expected_controller_sha,
                'source_drift_after_independent_audit')
        audit = receipt(audit_output)
        require(audit.get('plan_receipt_sha256') == args.expected_plan_receipt,
                'independent_audit_plan_binding')
        require(audit.get('auditor_source_sha256') == args.expected_auditor_sha,
                'independent_audit_source_binding')
        data['independent_audit_receipt_sha256'] = audit['receipt_sha256']
        data['independent_audit_status'] = audit.get('status')
        require(audit.get('evidence_integrity') == 'PASS', 'independent_integrity_gate')
        if data['runner_exit_code'] == 0 and data['auditor_exit_code'] == 0:
            require(audit.get('status') ==
                    'INDEPENDENT_AUDIT_CONFIRMED_ALL48_TRAIN16_RAW_AND_NINE_CHECKPOINTS_PASS'
                    and audit.get('experiment_numerical_verdict') == 'PASS'
                    and audit.get('completed_native_cell_count') == 79
                    and audit.get('source_report_receipt_sha256') == data['analysis_receipt_sha256'],
                    'independent_full_completion_gate')
            data['status'], code = 'NATIVE_ALL48_AND_INDEPENDENT_AUDIT_COMPLETED', 0
        elif data['runner_exit_code'] == 2 and data['auditor_exit_code'] == 2:
            require(audit.get('status') == 'INDEPENDENT_AUDIT_CONFIRMED_NUMERICAL_HOLD'
                    and audit.get('experiment_numerical_verdict') == 'FAIL',
                    'independent_numerical_hold_gate')
            data['status'], code = 'NUMERICAL_HOLD_INDEPENDENTLY_RECORDED', 2
        else:
            raise RuntimeError('independent_audit_failed_or_disagreed')
    except BaseException as error:
        data['status'] = 'FAILED_OR_INTERRUPTED_NO_COMPLETION_CLAIM'
        data['error'] = f'{type(error).__name__}: {error}'
        code = 1
    finally:
        data['ended_at'] = datetime.now(timezone.utc).isoformat()
        data['elapsed_seconds'] = time.monotonic() - started
        result = write_receipt(output, data)
        emit('ALL48_WORKFLOW_DONE', status=data['status'], receipt_sha256=result['receipt_sha256'],
             runner_exit_code=data['runner_exit_code'], auditor_exit_code=data['auditor_exit_code'])
    return code


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError(f'JE01_ALL48_WORKFLOW_SIGNAL:{signum}')
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
