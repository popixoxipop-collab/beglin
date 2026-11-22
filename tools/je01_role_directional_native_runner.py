#!/usr/bin/env python3
"""Receipt-bound native role directions, with complete teacher-forced context.

Each finite-difference cell reruns the unchanged 27-layer AF4 model from each
prompt's first token. Thus upstream changes to later KV state and routing are
included. A cell is a numerical computational-direction observation, not an
autodiff result or a Hessian. TRAIN16 is diagnostic only; fitting requires the
separately sealed TRAIN240 input set. No validation/OOD data are opened here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import time

WORK = Path('/Volumes/D50/TailnetCommander/work')
ROOT = WORK / 'je01_role_directional_20261123_v1'
BANK = WORK / 'je01_role_direction_bank_20261123_v1'
OLD = WORK / 'je01_eoe_realaf_nll_pilot16_20261122_v1'
MODEL = WORK / 'hcp-p18-external-multimetric-20261015/native_moe_base_source_copy'
AF_SHA = '9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4'
BASE_SHA = 'b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c'
BASE_RECEIPT = '7dd5da25e500007a2a35fd724d82a7c8c6966727458763160f6d4ee7b3ac54ec'
OLD_STAGE_RECEIPT = 'a814cdfd84f544132c4d9d8aa9e6e3c4845252e49a467d61c623e93931d3106e'
PAIRS = ((12,13),(13,22),(13,28),(14,63),(20,17),(21,61),(24,26),(24,60))
ROLES = ('gate_proj','up_proj','down_proj')
ROLE_ORDER = [[l,e,r] for l,e in PAIRS for r in ROLES]
META_MAGIC = b'JE01_ORIGINAL_FULL_GOLD_TARGETS_META_V1\n'
GOLD_MAGIC = b'JE01_ORIGINAL_AF_FULL_GOLD_NLL_V1\n'
ROUTE_MAGIC = b'JE01_ROLE_DIRECTION_ROUTES_V1\n'
ALPHAS = {'0':0., '1':1., '+0.05':.05, '-0.05':-.05,
          '+0.025':.025, '-0.025':-.025}
SCHEMA = 'beglin-je01-native-role-directional-run/1'
MARKER = 'JE01_ROLE_DIRECTION_REGISTERED'


def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sealed(path, expected=None):
    need(path.is_file() and not path.is_symlink(), 'JE01_DIRECTION_RECEIPT_MISSING:'+str(path))
    d = json.loads(path.read_text())
    claim = d.get('receipt_sha256')
    need(claim == canonical({k:v for k,v in d.items() if k != 'receipt_sha256'}),
         'JE01_DIRECTION_RECEIPT_CONTENT_DRIFT:'+path.name)
    need(expected is None or claim == expected, 'JE01_DIRECTION_RECEIPT_PIN_DRIFT:'+path.name)
    return d


def write_seal(path, data):
    data['receipt_sha256'] = canonical(data)
    with path.open('x') as f:
        f.write(json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())
    return data


def read_meta(raw):
    need(raw.startswith(META_MAGIC), 'JE01_DIRECTION_GOLD_META_MAGIC')
    offset = len(META_MAGIC)
    starts, targets = [], []
    for _ in range(16):
        need(offset+1544 <= len(raw), 'JE01_DIRECTION_GOLD_META_TRUNCATED')
        start,count = struct.unpack_from('<II', raw, offset)
        gold = list(struct.unpack_from('<384i', raw, offset+8))
        offset += 1544
        need(start >= 2 and 1 <= count and start+count <= 384,
             'JE01_DIRECTION_GOLD_META_BOUNDS')
        need(all(0 <= x < 102400 for x in gold[:count]) and
             all(x == -1 for x in gold[count:]), 'JE01_DIRECTION_GOLD_META_TARGETS')
        starts.append(start)
        targets.append(gold[:count])
    need(offset == len(raw), 'JE01_DIRECTION_GOLD_META_TRAILING')
    return starts,targets


def read_gold(raw, meta):
    starts,targets = read_meta(meta)
    need(raw.startswith(GOLD_MAGIC), 'JE01_DIRECTION_NATIVE_GOLD_MAGIC')
    offset = len(GOLD_MAGIC)
    nlls = []
    for row,ids in enumerate(targets):
        values = []
        for index,target in enumerate(ids):
            need(offset+32 <= len(raw), 'JE01_DIRECTION_NATIVE_GOLD_TRUNCATED')
            rr,ii,tt,pos,nll,top1,logit = struct.unpack_from('<4IdIf', raw, offset)
            offset += 32
            need((rr,ii,tt,pos) == (row,index,target,starts[row]+index-1),
                 'JE01_DIRECTION_NATIVE_GOLD_ALIGNMENT')
            need(math.isfinite(nll) and nll >= 0 and math.isfinite(logit) and
                 0 <= top1 < 102400, 'JE01_DIRECTION_NATIVE_GOLD_VALUES')
            values.append(nll)
        nlls.append(values)
    need(offset == len(raw), 'JE01_DIRECTION_NATIVE_GOLD_TRAILING')
    return nlls,targets


def read_routes(raw):
    need(raw.startswith(ROUTE_MAGIC), 'JE01_DIRECTION_ROUTE_MAGIC')
    body = raw[len(ROUTE_MAGIC):]
    need(body and len(body) % 40 == 0, 'JE01_DIRECTION_ROUTE_RECORD_SIZE')
    rows = list(struct.iter_unpack('<10i',body))
    need(len(rows) % 26 == 0,'JE01_DIRECTION_ROUTE_LAYER_COVERAGE')
    for i,row in enumerate(rows):
        event,layer,slot,pos,*ids = row
        need(event == i and layer == 1+i%26 and slot == 0 and 0 <= pos < 384 and
             len(set(ids)) == 6 and all(0 <= expert < 64 for expert in ids),
             'JE01_DIRECTION_ROUTE_COORDINATE_DRIFT')
        if i%26:
            need(pos == rows[i-1][3],'JE01_DIRECTION_ROUTE_TOKEN_POSITION_DRIFT')
    return rows


def validate_dataset(data):
    need(data['scope'] in ('TRAIN16_DIAGNOSTIC','TRAIN240'), 'JE01_DIRECTION_TRAIN_ONLY')
    ids = data['case_ids']
    need(len(ids) == 16 and len(set(ids)) == 16 and all(type(i) is int for i in ids),
         'JE01_DIRECTION_SIXTEEN_UNIQUE_CASES')
    if data['scope'] == 'TRAIN16_DIAGNOSTIC':
        need(ids == list(range(16)), 'JE01_DIRECTION_PILOT_CASES_CHANGED')
    else:
        need(all(16 <= i < 256 for i in ids) and ids == sorted(ids),
             'JE01_DIRECTION_TRAIN240_SPLIT_LEAK')
        need(isinstance(data.get('source_linkage'),dict),
             'JE01_DIRECTION_TRAIN240_SOURCE_LINKAGE_REQUIRED')
        import je01_role_directional_train240_inputs as training
        training.validate_source_linkage(data)
    manifest,meta = Path(data['manifest']),Path(data['gold_metadata'])
    need(all(p.is_file() and not p.is_symlink() for p in (manifest,meta)),
         'JE01_DIRECTION_DATA_FILES_MISSING')
    need(sha(manifest) == data['manifest_sha256'] and sha(meta) == data['gold_metadata_sha256'],
         'JE01_DIRECTION_DATA_FILES_HASH_DRIFT')
    starts,targets = read_meta(meta.read_bytes())
    need(targets == data['target_token_ids'], 'JE01_DIRECTION_DATA_TARGET_DRIFT')
    lines = manifest.read_text().splitlines()
    need(len(lines) == 16 and len(data['input_sha256']) == 16,
         'JE01_DIRECTION_MANIFEST_COUNT')
    for row,line in enumerate(lines):
        file,one = line.rsplit(' ',1)
        p = Path(file)
        need(one == '1' and p.is_file() and not p.is_symlink() and
             sha(p) == data['input_sha256'][row], 'JE01_DIRECTION_INPUT_HASH_DRIFT')
        raw = p.read_bytes()
        need(len(raw) == 4*(starts[row]+len(targets[row])-1),
             'JE01_DIRECTION_INPUT_FULL_PREFIX_LENGTH')
        tokens = struct.unpack('<'+'i'*(len(raw)//4),raw)
        need(tokens[0] == 100000 and all(0 <= t < 102400 for t in tokens) and
             list(tokens[starts[row]:]) == targets[row][:-1],
             'JE01_DIRECTION_TEACHER_FORCED_SEQUENCE_DRIFT')
    return manifest,meta


def prepare_pilot():
    baseline = sealed(OLD/'JE01_EOE_PILOT16_REAL_NLL_RESULT.json',BASE_RECEIPT)
    stage = sealed(OLD/'JE01_EOE_STAGED_PILOT16_PROVENANCE.json',OLD_STAGE_RECEIPT)
    need(baseline['native_output_sha256'] == BASE_SHA and baseline['full_gold_tokens'] == 656 and
         baseline['original_AF_source_sha256'] == AF_SHA, 'JE01_DIRECTION_PRIOR_NATIVE_BASELINE_DRIFT')
    candidates = [p for p in (OLD/'inputs',OLD/'inputs/je01_real_pilot16')
                  if (p/'JE01_EOE_ORIGINAL_PILOT16_INPUT_RECEIPT.json').is_file()]
    need(len(candidates) == 1, 'JE01_DIRECTION_PILOT_ROOT_AMBIGUOUS')
    meta = candidates[0]/'chunk00/gold_meta.bin'
    manifest = OLD/'manifest.eoe.txt'
    need(sha(meta) == stage['gold_meta_sha256'] and
         sha(manifest) == stage['manifest_current_host_sha256'] and
         sha(OLD/'gold_original_nll.bin') == BASE_SHA,
         'JE01_DIRECTION_PRIOR_INPUT_OR_OUTPUT_SHA_DRIFT')
    _,targets = read_gold((OLD/'gold_original_nll.bin').read_bytes(),meta.read_bytes())
    data = {'schema':'beglin-je01-role-directional-train-input/1',
            'scope':'TRAIN16_DIAGNOSTIC','case_ids':list(range(16)),
            'manifest':str(manifest),'gold_metadata':str(meta),
            'manifest_sha256':sha(manifest),'gold_metadata_sha256':sha(meta),
            'input_sha256':[sha(Path(x.rsplit(' ',1)[0])) for x in manifest.read_text().splitlines()],
            'target_token_ids':targets,'baseline_output_sha256':BASE_SHA,
            'independent_validation_ood_accessed':False}
    validate_dataset(data)
    directory = ROOT/'datasets'
    directory.mkdir(exist_ok=True,mode=0o700)
    return write_seal(directory/'pilot16.json',data)


def memory_free():
    text = subprocess.check_output(['/usr/bin/memory_pressure','-Q'],text=True,timeout=6)
    found = re.search(r'System-wide memory free percentage:\s*(\d+)%',text)
    need(found is not None, 'JE01_DIRECTION_MEMORY_SENSOR_MISSING')
    return int(found.group(1))


def preflight(dataset):
    import je01_role_directional_bank as bank_module
    need(ROOT.is_dir() and not ROOT.is_symlink(), 'JE01_DIRECTION_ISOLATED_ROOT_MISSING')
    data = sealed(dataset)
    manifest,meta = validate_dataset(data)
    build = sealed(ROOT/'NATIVE_BUILD_RECEIPT.json')
    need(build['status'] == 'JE01_ROLE_DIRECTION_NATIVE_COMPILE_PASS' and
         build['original_cpp_sha256'] == '640da6e8afcace5e5440177ef9953732bee1cf83927680992f809a7af3f6950a' and
         build['production_modified'] is False, 'JE01_DIRECTION_NATIVE_BUILD_CONTRACT')
    exe = ROOT/'bin/je01_role_directional_native'
    need(exe.is_file() and not exe.is_symlink() and sha(exe) == build['binary_sha256'],
         'JE01_DIRECTION_NATIVE_BINARY_HASH')
    bank = bank_module.verify_bank(BANK)
    original = MODEL/'weights_moe/deepseek_moe_af.bin'
    need(original.is_file() and not original.is_symlink() and
         original.stat().st_size == 9814343680 and sha(original) == AF_SHA,
         'JE01_DIRECTION_ORIGINAL_AF4_FULL_SHA')
    import mlx,mlx.core as mx
    need(mx.__version__ == '0.32.1', 'JE01_DIRECTION_MLX_ABI_DRIFT')
    lib = Path(list(mlx.__path__)[0])/'lib'
    need((lib/'libmlx.dylib').is_file(), 'JE01_DIRECTION_MLX_LIBRARY_MISSING')
    identity = {'original_af4_sha256':AF_SHA,'native_executable_sha256':build['binary_sha256'],
                'bank_sha256':bank['physical_file_sha256'],
                'input_manifest_sha256':sha(manifest),'gold_metadata_sha256':sha(meta),
                'role_order_sha256':canonical(ROLE_ORDER)}
    return data,manifest,meta,exe,lib,identity


def run_cell(dataset,role,tier,alpha,off=False,attempt=1):
    need(type(role) is int and role in range(24) and tier in ('Q4','Q5') and
         alpha in ALPHAS and type(attempt) is int and attempt >= 1,
         'JE01_DIRECTION_CELL_OUTSIDE_REGISTERED_GRID')
    need(not off or alpha == '0', 'JE01_DIRECTION_OFF_ALPHA_NONZERO')
    data,manifest,meta,exe,lib,identity = preflight(dataset)
    label = 'off' if off else 'r%02d_%s_a%s'%(role,tier,alpha.replace('+','p').replace('-','m'))
    dataset_id = data.get('dataset_id',dataset.stem)
    need(re.fullmatch(r'(pilot16|train240_chunk(?:0[1-9]|1[0-5]))',dataset_id),
         'JE01_DIRECTION_DATASET_ID_INVALID')
    if alpha != '0':
        controls = [ROOT/'runs'/dataset_id/name/'RUN_RECEIPT.json' for name in
                    ('off_attempt1','r00_Q4_a0_attempt1')]
        for path in controls:
            control = sealed(path)
            need(control['status'] == 'SUCCEEDED_EXIT0' and control['alpha'] == 0 and
                 control['source_identity'] == identity and control['case_ids'] == data['case_ids'] and
                 control['output_sha256'] == data['baseline_output_sha256'] and
                 control['control_reference_output_sha256'] == control['output_sha256'],
                 'JE01_DIRECTION_ORIGINAL_PARITY_CONTROLS_REQUIRED')
    folder = ROOT/'runs'/dataset_id/(label+'_attempt'+str(attempt))
    need(not folder.exists() and not folder.is_symlink(), 'JE01_DIRECTION_CELL_ALREADY_ATTEMPTED')
    folder.mkdir(parents=True,mode=0o700)
    (folder/'tmp').mkdir(mode=0o700)
    target,route = folder/'gold_native.bin',folder/'routes.bin'
    env = {'PATH':'/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin','HOME':'/Users/eoe',
           'TMPDIR':str(folder/'tmp'),'DYLD_LIBRARY_PATH':str(lib),
           'QWEN_MOE_BASE':str(MODEL),'QWEN_MOE_GPU_CBATCH_ONLINE':'1',
           'QWEN_MOE_CB_SLOTS':'1','QWEN_MOE_CB_REQS':'16','QWEN_MOE_CB_PREFILL_BUDGET':'1',
           'QWEN_MOE_CB_PROMPT_MANIFEST':str(manifest),'QWEN_MOE_GPU_SORT_THRESHOLD':'64',
           'QWEN_MOE_GPU_CBATCH_BUDGET':'0','JE01_ORIGINAL_GOLD_META':str(meta),
           'JE01_ORIGINAL_GOLD_OUTPUT':str(target),'JE01_DIRECTION_ROUTE_OUTPUT':str(route)}
    if not off:
        env.update({'JE01_REAL_Q35_BUNDLE':str(BANK/'JE01_ROLE_DIRECTION_BANK.bin'),
                    'JE01_REAL_Q35_TSV':str(BANK/'JE01_ROLE_DIRECTION_BANK.tsv'),
                    'JE01_DIRECTION_ROLE':str(role),'JE01_DIRECTION_TIER':tier,
                    'JE01_DIRECTION_ALPHA':alpha})
    started = time.monotonic()
    lowest = memory_free()
    need(lowest >= 55,'JE01_DIRECTION_INITIAL_MEMORY_BELOW55')
    print(json.dumps({'event':'NATIVE_DIRECTION_START','cell':label,'scope':data['scope'],
                      'cases':data['case_ids'],'memory_free_pct':lowest}),flush=True)
    stop = None
    with (folder/'stdout.txt').open('xb') as out,(folder/'stderr.txt').open('xb') as err:
        proc = subprocess.Popen([str(exe)],cwd=str(folder),env=env,stdout=out,stderr=err)
        try:
            while proc.poll() is None:
                free = memory_free()
                lowest = min(lowest,free)
                if free < 35: stop = 'MEMORY_BELOW35'
                if time.monotonic()-started > 420: stop = 'CELL_TIMEOUT420'
                rss = subprocess.run(['/bin/ps','-o','rss=','-p',str(proc.pid)],
                                     capture_output=True,text=True,timeout=5).stdout.strip()
                if rss.isdigit() and int(rss) > 16*1024*1024: stop = 'RSS_OVER16GB'
                if stop:
                    proc.terminate()
                    break
                time.sleep(.8)
        except BaseException:
            proc.terminate()
            try: proc.wait(timeout=6)
            except subprocess.TimeoutExpired: proc.kill();proc.wait()
            raise
        if stop and proc.poll() is None:
            try: proc.wait(timeout=6)
            except subprocess.TimeoutExpired: proc.kill()
        code = proc.wait()
    errors = (folder/'stderr.txt').read_text(errors='replace')
    need(code == 0 and stop is None and 'FATAL:' not in errors and
         'RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16' in errors and
         ((MARKER in errors) == (not off)),
         'JE01_DIRECTION_NATIVE_CELL_FAILED:'+str(code)+':'+str(stop)+':'+errors[-1800:])
    values,targets = read_gold(target.read_bytes(),meta.read_bytes())
    need(route.is_file(),'JE01_DIRECTION_ROUTE_TRACE_MISSING')
    route_rows = read_routes(route.read_bytes())
    output_sha = sha(target)
    if alpha == '0' and data.get('baseline_output_sha256'):
        need(output_sha == data['baseline_output_sha256'], 'JE01_DIRECTION_ALPHA_ZERO_NOT_EXACT_ORIGINAL')
    record = {'schema':SCHEMA,'status':'SUCCEEDED_EXIT0','scope':data['scope'],
              'source_identity':identity,'direction_index':None if alpha == '0' else role*2+(tier=='Q5'),
              'alpha':ALPHAS[alpha],'case_ids':data['case_ids'],'target_token_ids':targets,
              'per_token_nll':values,'output_sha256':output_sha,
              'control_reference_output_sha256':data.get('baseline_output_sha256') if alpha == '0' else None,
              'native_overlay_enabled':not off,'role_index':role if not off else None,
              'role_key':ROLE_ORDER[role] if not off else None,'tier':tier if not off else None,
              'route_output_sha256':sha(route),'route_output_bytes':route.stat().st_size,
              'routed_token_layer_events':len(route_rows),
              'route_output_path':str(route),'mean_gold_nll':math.fsum(map(math.fsum,values))/sum(map(len,values)),
              'derivative_kind':'FINITE_DIFFERENCE_NATIVE_LINEAR_ROLE_OUTPUT_COMPUTATIONAL_DIRECTION',
              'full_prompt_and_KV_recomputed':True,'min_memory_free_pct':lowest,
              'elapsed_seconds':time.monotonic()-started,'native_exit_code':code,
              'runner_source_sha256':sha(Path(__file__)),
              'independent_validation_ood_accessed':False,'production_modified':False}
    result = write_seal(folder/'RUN_RECEIPT.json',record)
    print(json.dumps({'event':'NATIVE_DIRECTION_DONE','cell':label,'receipt_sha256':result['receipt_sha256'],
                      'mean_gold_nll':result['mean_gold_nll'],'output_sha256':output_sha,
                      'elapsed_seconds':result['elapsed_seconds']},sort_keys=True),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-pilot',action='store_true')
    parser.add_argument('--dataset',type=Path,default=ROOT/'datasets/pilot16.json')
    parser.add_argument('--role',type=int,default=0)
    parser.add_argument('--tier',choices=('Q4','Q5'),default='Q4')
    parser.add_argument('--alpha',choices=tuple(ALPHAS),default='0')
    parser.add_argument('--off',action='store_true')
    parser.add_argument('--attempt',type=int,default=1)
    args = parser.parse_args()
    if args.prepare_pilot:
        print(json.dumps(prepare_pilot(),sort_keys=True))
    else:
        run_cell(args.dataset,args.role,args.tier,args.alpha,args.off,args.attempt)
