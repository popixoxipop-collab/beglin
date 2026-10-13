#!/usr/bin/env python3
"""Capture actual DeepSeek prompt activation/routing traces via fresh scratch processes.

Uses original Beglin QWEN_MOE_PROMPT_IDS override and a source-verified scratch
research-only h2 hook. All raw token IDs, routes and hidden vectors stay in the
isolated runner scratch directory; only hashed manifest and metric aggregates
are eligible for upload. No network, no service changes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from transformers import AutoTokenizer

SCHEMA="beglin-prompt-suite-v1"
PROMPTS=[
  # Training group: conceptual diversity, separated by fresh process to reset KV.
  ('train', 'Explain why the seasons change on Earth and why summer differs by hemisphere.'),
  ('train', 'Describe how photosynthesis converts sunlight into chemical energy.'),
  ('train', 'Write a short explanation of Newton second law in simple language.'),
  ('train', 'If a train travels forty kilometers in thirty minutes, estimate its average speed.'),
  ('train', 'What is the difference between precision and recall in a classification model?'),
  ('train', 'Explain how a hash table differs from a sorted array for searching.'),
  ('train', 'Summarize the causes of the French Revolution using one paragraph.'),
  ('train', 'How do clouds form and why can different clouds produce rain?'),
  ('train', 'Describe two ways to reduce energy consumption in a modern building.'),
  ('train', 'Explain the role of mitochondria in a living animal cell.'),
  ('train', 'How does a neural network use gradient descent to update parameters?'),
  ('train', 'Compare a renewable solar panel with a conventional gas power station.'),
  # Validation group is reserved for selecting a candidate method, not fitting.
  ('validation', 'Describe what a black hole event horizon means for light and time.'),
  ('validation', 'Explain how vaccines train the immune system to recognize a pathogen.'),
  ('validation', 'What problems arise when training data contain duplicated examples?'),
  ('validation', 'How would you test a small robot navigation algorithm safely?'),
  ('validation', 'Discuss why ocean temperature influences the weather along a coast.'),
  ('validation', 'Describe how binary search locates a value in an ordered collection.'),
  # Final holdout: never used to train or choose method/hyperparameters.
  ('holdout', 'Explain what makes prime numbers different from composite numbers.'),
  ('holdout', 'How does a heat pump move thermal energy into a room during winter?'),
  ('holdout', 'Describe a safe way to check a machine learning system for data leakage.'),
  ('holdout', 'How can astronomers estimate distances to stars using parallax?'),
  ('holdout', 'What is the purpose of a checksum in reliable file transfer?'),
  ('holdout', 'Describe how friction changes the motion of a rolling wheel.'),
]

def sha256_bytes(buf:bytes)->str:return hashlib.sha256(buf).hexdigest()
def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def run_suite(binary:Path,index:Path,tokenizer_dir:Path,model_root:Path,out:Path,
              *,layer:int=1, hidden:int=2048, max_positions:int=24, limit_prompts:int|None=None,
              capture_env:str='BEGLIN_CAPTURE_PRE_FFN_INPUT'):
    assert binary.is_file() and index.is_file() and model_root.is_dir()
    if capture_env!='BEGLIN_CAPTURE_PRE_FFN_INPUT':raise ValueError('unsafe capture variable')
    tok=AutoTokenizer.from_pretrained(str(tokenizer_dir),local_files_only=True,
                                      trust_remote_code=False,use_fast=True)
    assert 1<=max_positions<=40
    items=PROMPTS[:limit_prompts] if limit_prompts is not None else PROMPTS
    assert len(items)>=1
    text_shas=set();id_shas=set()
    out.mkdir(parents=True,exist_ok=True)
    index_sha=sha256_file(index)
    rows=[]
    t0=time.monotonic()
    for i,(split,prompt) in enumerate(items):
        prompt_id=f'p{i:02d}'
        rawids=tok.encode(prompt,add_special_tokens=True)
        ids=[int(z) for z in rawids[:max_positions]]
        if len(ids)<8 or any(not 0<=z<102400 for z in ids):
            raise RuntimeError(f'prompt not eligible: {prompt_id}')
        strids=','.join(map(str,ids))
        if len(strids)>=900:raise ValueError('input environment length exceeds safe bound')
        ih=sha256_bytes(strids.encode())
        th=sha256_bytes(prompt.encode())
        if ih in id_shas or th in text_shas:raise ValueError('duplicate independent prompt content')
        id_shas.add(ih);text_shas.add(th)
        p=out/prompt_id;p.mkdir(exist_ok=False)
        env={k:os.environ[k] for k in ('PATH','HOME','TMPDIR','LANG') if k in os.environ}
        env.update({'QWEN_MOE_BASE':str(model_root),
                    'QWEN_MOE_PROMPT_IDS':strids,
                    'QWEN_MOE_NEARTIE_CORRECT':'0',
                    capture_env:str(p/'true_h2.bin')})
        r=subprocess.run([str(binary)],cwd=str(p),env=env,text=True,
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=100)
        if r.returncode!=0:
            raise RuntimeError(f'scratch forward failed for {prompt_id}: rc={r.returncode}, last={r.stderr[-280:]}')
        cap=p/'true_h2.bin';route=p/'moe3a_c_routing.txt'
        if not cap.is_file() or not route.is_file():
            raise RuntimeError(f'missing real trace for {prompt_id}')
        expected_bytes=len(ids)*(12+hidden*4)
        if cap.stat().st_size!=expected_bytes:
            raise RuntimeError(f'capture count mismatch for {prompt_id}: {cap.stat().st_size}!={expected_bytes}')
        rows.append({'prompt_id':prompt_id,'split':split,'position_count':len(ids),
                     'input_ids_sha256':ih,'prompt_text_sha256':th,
                     'capture_sha256':sha256_file(cap),'routing_sha256':sha256_file(route)})
        print('PROMPT_CAPTURE_PASS',prompt_id,split,'positions',len(ids),
              'elapsed_s',round(time.monotonic()-t0,1),flush=True)
    if len(items)==len(PROMPTS):
        assert [sum(v['split']==s for v in rows) for s in ('train','validation','holdout')]==[12,6,6]
    payload={'schema':SCHEMA,'model_id':'DeepSeek-V2-Lite','layer':layer,'hidden':hidden,
             'tokenizer_root_basename':tokenizer_dir.name,
             'checkpoint_index_sha256':index_sha,
             'macro_top_k':6,'prompt_count':len(rows),
             'source_model':'quantized AF DeepSeek-V2-Lite scratch forward',
             'fresh_process_per_prompt':True,'raw_activation_uploaded':False,
             'raw_token_ids_uploaded':False,'production_touched':False,
             'runs':rows}
    (out/'runs.json').write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n')
    print('PROMPT_SUITE_DONE',json.dumps({'prompts':len(rows),
      'positions':sum(x['position_count'] for x in rows),
      'split_counts':{s:sum(x['split']==s for x in rows) for s in ('train','validation','holdout')},
      'total_seconds':round(time.monotonic()-t0,2)},sort_keys=True),flush=True)
    return payload

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',required=True,type=Path)
    ap.add_argument('--checkpoint-index',required=True,type=Path)
    ap.add_argument('--tokenizer-dir',required=True,type=Path)
    ap.add_argument('--model-root',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--max-positions',type=int,default=24)
    a=ap.parse_args()
    run_suite(a.binary,a.checkpoint_index,a.tokenizer_dir,a.model_root,a.out,max_positions=a.max_positions)