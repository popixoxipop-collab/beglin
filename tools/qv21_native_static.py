#!/usr/bin/env python3
"""Original pinned Apple Qwen28 C scorer for ONE Qv21 cohort and static arm.

Must be launched as an admitted managed EOE job. Frozen inputs required; never
reopens the Qv20 heldout. Writes owner-only, create-only logs and receipts.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import struct
import subprocess
import time
import psutil

ROOT = pathlib.Path('/Volumes/D50/TailnetCommander/work/beglin-qv21-tailrisk-20261025')
SPEC_SHA256 = '408c661e66259784480b18eefc15f75d3bc7d034124194baaf3a218637e2f2d0'


def read_spec():
    raw = (ROOT / 'qv21_spec.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPEC_SHA256:
        raise RuntimeError('QV21_STUDY_SPEC_DRIFT')
    return json.loads(raw)

EXE = pathlib.Path('/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_mixed')
MODEL = pathlib.Path('/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89')
QWEN = pathlib.Path('/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021')
R_ROW = re.compile(r'^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$', re.M)
R_SUM = re.compile(r'^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$', re.M)


def filehash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(262144), b''):
            h.update(chunk)
    return h.hexdigest()


def require(ok, reason):
    if not ok:
        raise RuntimeError('QV21_ORIGINAL_C_STATIC_FAIL_' + reason)


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode()


def save_new(p, raw):
    with p.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    p.chmod(0o600)


def run(split, arm, preflight_only=False):
    spec = read_spec()
    require(split in spec['splits'], 'SPLIT_NAME')
    require(arm in spec['static_masks'], 'MASK_NAME')
    assert split in ('cal_arc_train96', 'eval_arc_train48', 'eval_ob_train48')
    require(filehash(EXE) == spec['original_native_c_sha256'], 'EXACT_ORIGINAL_C_SHA')
    public_file = ROOT / 'Qv21_PREEXEC_PUBLIC.json'
    public = json.loads(public_file.read_text('utf8'))
    require(public.get('state') == 'FROZEN_BEFORE_ANY_QV21_NATIVE_SCORING' and
            public.get('study_spec_SHA256') == hashlib.sha256(
                (pathlib.Path(__file__).resolve().parent.parent / 'qv21_spec.json').read_bytes()).hexdigest(),
            'PREEXEC_STATUS_OR_SPEC_SHA')
    manifest = ROOT / split / 'frozen_inputs_private.json'
    require(filehash(manifest) == public['frozen_private_input_SHA256'][split], 'PRIVATE_FROZEN_SHA')
    require(filehash(ROOT / 'masks' / (arm + '.txt')) == spec['static_masks'][arm], 'MASK_PIN')
    rows = json.loads(manifest.read_text('utf8'))['selected_rows']
    n = 96 if split == 'cal_arc_train96' else 48
    require(len(rows) == n, 'EXACT_CASE_COUNT')
    work = ROOT / split / arm / 'eval/ppl_work'
    for i, row in enumerate(rows):
        ids = struct.pack('<' + 'i' * len(row['prefix_ids']), *row['prefix_ids'])
        choices = (' '.join(str(row['suffix_ids'][k]) for k in 'ABCD') + ' ' +
                   str('ABCD'.index(row['correct'])) + '\n').encode()
        require((work / f'w{i}.i32').read_bytes() == ids, 'TOKEN_BINARY_' + str(i))
        require((work / f'w{i}.choices').read_bytes() == choices, 'CHOICE_BINARY_' + str(i))
    if preflight_only:
        print('QV21_ORIGINAL_C_STATIC_PREFLIGHT_PASS ' + json.dumps({'split': split, 'arm': arm, 'n': n,
               'original_C_sha256': spec['original_native_c_sha256'],
               'public_freeze_sha256': filehash(public_file), 'no_model_scoring': True}, sort_keys=True), flush=True)
        return
    output_dir = ROOT / 'native_original_C_static' / split / arm
    require(not output_dir.exists() and not output_dir.is_symlink(), 'OUTPUT_ALREADY_EXISTS')
    os.umask(0o077)
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('QWEN_', 'BEGLIN_Q_', 'BEGLIN_Q_LAB_', 'BEGLIN_QV'))}
    env.update({
        'VECLIB_MAXIMUM_THREADS': '1', 'Q4_THREADS': '1', 'Q4_POOL_SPIN': '0',
        'QWEN_SAFETENSORS': str(MODEL / 'model.safetensors'),
        'QWEN_HF_CONFIG': str(MODEL / 'config.json'),
        'QWEN_SAFETENSORS_MAXSEQ': '256',
        'QWEN_BASE': str(ROOT / split / arm),
        'QWEN_PROMPT': str(ROOT / split / arm / 'prompt_ids.i32'),
        'BEGLIN_Q_LAB_PRECISION': 'mixed', 'BEGLIN_QV5_FROZEN_ARC_CHALLENGE': '1',
        'BEGLIN_QV2_NATIVE_ASYNC': '0',
        'BEGLIN_Q_LAB_N8_ROLES_FILE': str(ROOT / 'masks' / (arm + '.txt')),
    })
    started = time.monotonic()
    proc = subprocess.Popen([str(EXE), 'masked_choice_qv5', str(n)], cwd=str(QWEN),
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    peak = 0
    while proc.poll() is None:
        try:
            peak = max(peak, psutil.Process(proc.pid).memory_info().rss)
        except psutil.NoSuchProcess:
            pass
        time.sleep(.08)
    stdout, stderr = proc.communicate()
    wall = time.monotonic() - started
    logfile = output_dir / 'native_original_C_Qwen28.log'
    save_new(logfile, stdout + b'\n' + stderr)
    require(proc.returncode == 0, 'NATIVE_EXEC_EXIT_' + str(proc.returncode))
    text = logfile.read_text('utf8')
    cases = R_ROW.findall(text); sums = R_SUM.findall(text)
    require(len(cases) == n and len(sums) == 1, 'ORIGINAL_C_CASE_COUNT')
    require(text.count('projection_role_tensors: f32=0 n8=28 q4=168') == 1, 'ORIGINAL_C_PROJECTION_ROLES')
    losses = []; correct = []
    for i, (idx, gold_nll, answer, pred, plen) in enumerate(cases):
        row = rows[i]
        value = float(gold_nll)
        require(int(idx) == i and int(plen) == row['prefix_len'] and
                int(answer) == 'ABCD'.index(row['correct']) and
                math.isfinite(value) and value >= 0, 'ORIGINAL_C_ROW_' + str(i))
        losses.append(value); correct.append(int(answer == pred))
    total, count, wins, mean, ppl = map(float, sums[0])
    require(int(count) == n and int(wins) == sum(correct) and
            abs(total - sum(losses)) <= 1e-7 and abs(mean - total / n) < 1e-8 and
            math.isfinite(ppl) and ppl > 0, 'ORIGINAL_C_AGGREGATE')
    receipt = {
        'schema': 'BEGLIN-Qv21-actual-original-Qwen28-C-single-static-mask-case-scores/1',
        'state': 'ACTUAL_ORIGINAL_QWEN28_C_STATIC_' + arm.upper() + '_PASS',
        'split': split, 'arm': arm, 'n': n,
        'original_Qwen28_C_executable_sha256': spec['original_native_c_sha256'],
        'source_frozen_private_SHA256': filehash(manifest),
        'source_public_preexec_SHA256': filehash(public_file),
        'mask_sha256': spec['static_masks'][arm],
        'real_native_original_C_log_SHA256': filehash(logfile),
        'original_full_vocab_gold_nexttoken_NLL': losses,
        'original_choice_correct_each': correct,
        'correct': sum(correct), 'mean_NLL': mean,
        'actual_native_process_wall_s': wall, 'actual_native_peak_RSS_MiB': peak / 1048576,
        'Q4_THREADS': 1, 'real_native_Q4_N8_weights_active': True,
        'no_original_C_or_operator_release_changes': True,
    }
    priv = output_dir / 'PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json'
    save_new(priv, canonical(receipt))
    public_receipt = {k: v for k, v in receipt.items()
                      if k not in ('original_full_vocab_gold_nexttoken_NLL', 'original_choice_correct_each')}
    public_receipt['private_scores_receipt_SHA256'] = filehash(priv)
    pub = output_dir / 'PUBLIC_AGGREGATE.json'
    save_new(pub, canonical(public_receipt))
    print('QV21_ACTUAL_STATIC_NATIVE_PASS ' + json.dumps({
        'split': split, 'arm': arm, 'n': n, 'correct': sum(correct), 'mean_NLL': mean,
        'real_native_wall_s': round(wall, 3), 'public_receipt_sha256': filehash(pub)}, sort_keys=True), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--split', required=True, choices=('cal_arc_train96', 'eval_arc_train48', 'eval_ob_train48'))
    cli.add_argument('--arm', required=True, choices=('static_source', 'static_random'))
    cli.add_argument('--preflight-only', action='store_true')
    args = cli.parse_args()
    run(args.split, args.arm, preflight_only=args.preflight_only)
