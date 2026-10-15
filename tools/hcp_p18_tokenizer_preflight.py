#!/usr/bin/env python3
"""No-inference P18 native tokenizer compatibility preflight.

Inputs are the hash-pinned XOX tokenizer copy on EOE D50 and P17/P18 frozen
sources. This never changes P17, serves a model, or certifies P18 quality.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
from collections import Counter, defaultdict

XOX_TOKENIZER_SHA = '41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669'
XOX_TOKENCFG_SHA = '31181eaf79394ea26728d95ecb54fe7c8413e6f56085dbabc8b0818134380ec8'
XOX_CONFIG_SHA = 'f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d'
P17_SHA = 'a8e6e42b11a0f881317ab8afddebf7b266a965f4100bfddd7342183e73728521'
P18_CANDIDATES_SHA = 'ea7cb77649fcdc285393234e019d56db0d7178fb7787cf23268a7c52c072bb5a'
MAX_VOCAB = 102400


def digest(path: pathlib.Path) -> str:
    path = pathlib.Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError('Source must be a regular file: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_sha(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError('Pinned source identity mismatch: ' + str(path))
    return actual


def token_digest(ids):
    return hashlib.sha256(','.join(map(str, ids)).encode('ascii')).hexdigest()


def encode(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=True).ids
    if not ids or any(type(x) != int or not 0 <= x < MAX_VOCAB for x in ids):
        raise ValueError('Unexpected empty or non-DeepSeek token sequence')
    return ids


def audit_p17(tokenizer, frozen):
    items = frozen.get('items')
    if frozen.get('question_count') != 12 or not isinstance(items, list) or len(items) != 12:
        raise ValueError('P17 reference source is not 12 original questions')
    passed = []
    for item in items:
        prefix = encode(tokenizer, item['prompt'])
        full = encode(tokenizer, item['prompt'] + item['reference_answer'])
        ok = (prefix == item['prompt_ids']
              and full == item['teacher_full_context_ids']
              and full[:len(prefix)] == prefix
              and full[len(prefix):] == item['gold_answer_ids']
              and token_digest(prefix) == item['prompt_token_sha256']
              and token_digest(full) == item['teacher_full_context_token_sha256'])
        passed.append(ok)
    return {'total': len(items), 'exact_matches': sum(passed), 'all_exact': all(passed)}


def preview_p18(tokenizer, bank):
    items = bank.get('items')
    if bank.get('schema') != 'beglin-hcp-p18-public-candidate-question-bank-v1':
        raise ValueError('P18 input schema mismatch')
    if not isinstance(items, list) or len(items) != 480:
        raise ValueError('P18 requires 480 exactly source-selected questions')
    counts = Counter()
    records = []
    seen = set()
    for item in items:
        item_id = item['item_id']
        dataset = item['dataset']
        if item_id in seen or dataset not in ('mmlu', 'arc_challenge', 'gsm8k'):
            raise ValueError('P18 wrong ID/dataset or duplicated source')
        seen.add(item_id)
        counts[dataset] += 1
        pre = encode(tokenizer, item['prompt'])
        full = encode(tokenizer, item['prompt'] + item['reference_answer'])
        stable = full[:len(pre)] == pre
        gold = full[len(pre):] if stable else []
        records.append({
            'item_id': item_id,
            'dataset': dataset,
            'record_sha256': item['record_sha256'],
            'prompt_token_count': len(pre),
            'gold_token_count': len(gold),
            'prompt_token_sha256': token_digest(pre),
            'gold_token_sha256': token_digest(gold),
            'prefix_stable': stable,
            'prompt_plus_8_fits_native_128': len(pre) + 8 <= 128,
            'prompt_plus_32_fits_native_128': len(pre) + 32 <= 128,
            'prompt_plus_128_fits_model_4096': len(pre) + 128 <= 4096,
        })
    if counts != Counter({'mmlu': 160, 'arc_challenge': 160, 'gsm8k': 160}):
        raise ValueError('P18 required balanced source strata')
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--d50-root', required=True)
    p.add_argument('--p17-frozen', required=True)
    p.add_argument('--report', required=True)
    a = p.parse_args()
    root = pathlib.Path(a.d50_root)
    target = pathlib.Path(a.report)
    if not root.is_dir() or root.is_symlink() or target.is_symlink() or target.exists():
        raise ValueError('Unsafe D50 root, or output already exists; no overwrite allowed')
    pinned = root / 'tokenizer_exact_XOX'
    for fn, sha in (('tokenizer.json', XOX_TOKENIZER_SHA),
                    ('tokenizer_config.json', XOX_TOKENCFG_SHA),
                    ('config.json', XOX_CONFIG_SHA)):
        require_sha(pinned / fn, sha)
    require_sha(pathlib.Path(a.p17_frozen), P17_SHA)
    require_sha(root / 'candidate_bank_PRIVATE.json', P18_CANDIDATES_SHA)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(pinned / 'tokenizer.json'))
    p17 = audit_p17(tokenizer, json.loads(pathlib.Path(a.p17_frozen).read_text()))
    if not p17['all_exact']:
        raise ValueError('P17 tokenizer exact token-ID parity failed; refuse P18 inputs')
    records = preview_p18(tokenizer, json.loads((root / 'candidate_bank_PRIVATE.json').read_text()))
    agg = defaultdict(dict)
    for name in ('mmlu', 'arc_challenge', 'gsm8k'):
        sample = [r for r in records if r['dataset'] == name]
        agg[name] = {'count': len(sample),
                     'boundary_stable_count': sum(r['prefix_stable'] for r in sample),
                     'prompt_min': min(r['prompt_token_count'] for r in sample),
                     'prompt_max': max(r['prompt_token_count'] for r in sample),
                     'gold_max': max(r['gold_token_count'] for r in sample),
                     'fits_128_after_8': sum(r['prompt_plus_8_fits_native_128'] for r in sample),
                     'fits_128_after_32': sum(r['prompt_plus_32_fits_native_128'] for r in sample),
                     'fits_4096_after_128': sum(r['prompt_plus_128_fits_model_4096'] for r in sample)}
    report = {'schema': 'beglin-hcp-p18-tokenizer-preview-only-v1',
              'state': 'P18_ZERO_SHOT_TOKENIZER_PREFLIGHT_NOT_FROZEN_NOT_NATIVE',
              'p17_token_id_parity': p17,
              'xox_original_tokenizer_sha256': XOX_TOKENIZER_SHA,
              'model_index_sha256': 'd2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf',
              'p18_selected_480_sha256': P18_CANDIDATES_SHA,
              'per_dataset': agg,
              'prompt_answer_boundary_stable': sum(r['prefix_stable'] for r in records),
              'private_prompt_hash_records': records,
              'real_model_forward_count': 0,
              'native_A_B_evaluations': 0,
              'actual_4096_native_context_verified': False,
              'automatic_live_promotion': False,
              'production_touched': False}
    if target.parent.resolve() != root.resolve():
        raise ValueError('Output must be directly under the designated D50 root')
    with target.open('x', encoding='utf-8') as f:
        json.dump(report, f, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps({'state': report['state'],
                      'p17_exact': p17['exact_matches'],
                      'source_counts': {k: v['count'] for k, v in agg.items()},
                      'boundary_stable': report['prompt_answer_boundary_stable'],
                      'report_sha256': digest(target)}, sort_keys=True))


if __name__ == '__main__':
    main()
