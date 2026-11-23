#!/usr/bin/env python3
"""Separate six-cell fixed-hard-route diagnostic; original runs stay immutable.

The only default experiment is original-routing OFF with margin capture,
replayed-route alpha zero, then role0/Q4 at four fixed finite-difference steps.
The two controls must reproduce the original raw gold and route SHA before any
nonzero cell starts. Scores, selected weights and all prompt/KV states are
recomputed. This module neither selects a policy nor reads validation/OOD data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

import je01_role_directional_native_runner as common
import je01_role_directional_fixed_routes_patch as patch

WORK = Path('/Volumes/D50/TailnetCommander/work')
ROOT = WORK/'je01_role_directional_fixed_20261123_v1'
ORDINARY = WORK/'je01_role_directional_20261123_v1'
BC_BUILD = WORK/'je01_postnll_bc_native_compile_20261123_v1'
DATASET = ORDINARY/'datasets/pilot16.json'
OFF_FOLDER = ORDINARY/'runs/pilot16/off_attempt1'
OFF_RECEIPT = '85d8e28d51fbc0e2d319f0e60e5f9e6ac3d2f9050b2e386d81a969f6ad8f672a'
DATASET_RECEIPT = 'a0091a208adaa559788a942149fe28f12f1de4e03dfbbafe840a5acce50f3e2a'
ORDINARY_BUILD_RECEIPT = '3e7a2fd489b5cac1b00946fdf2369062bf0758e687e25f379cc3d4a2c60a07fe'
ORDINARY_EXE_SHA = 'cccd8b3389d62f76ec6392122eace86be62be1e12d782829b9cb0e6792a5a9a9'
COMMON_RUNNER_SHA = '5607c0afef3bd4b1a61bcab6622267acc6b6137fc65ba0e79873e92e31d7e9e6'
BANK_RECEIPT = '4994ec762b3a9b8a44015bad18f4b5bc7ea26505208384146238e7def28464d4'
BANK_SHA = '3b18257d22caddd0916292bef692110f9eeaf0edf83ad3fd4d5417fdf3443bb8'
BANK_TSV_SHA = 'fdc801f59f9c6438dd60bb33fab9a1332cb19c6e15eb15bef7425b7fbd522e4f'
ROUTE_SHA = '7e5811d13953f99db5df55169bf465f039acb9224f626336d247168e0f570711'
ROUTE_BYTES = 4_507_390
ROUTE_RECORDS = 112_684
MANIFEST_SHA = 'fab455198ef37d443b20af8e3029df3ddf4f4ba0f6d584eceeb262d26d6a146d'
META_SHA = 'be0dab12624efb4fae17538b4c5e6ef76bd09aebf4c09b1b428fc1d8f0fff406'
ROLE_ORDER_SHA = 'd232ca637eb17a1b40bf29f75e0590d8f63ecbc9aa950fb838a34b6fa90259a1'
FIXED_CPP_SHA = '390fb6b136bdd9f4978e6482b7a18b11951333e453752a15c3fd094a7d3b07f3'
SCHEMA = 'beglin-je01-native-fixed-hard-route-directional-run/1'
FD_ALPHAS = ('+0.05','-0.05','+0.025','-0.025')
EXPECTED_ORDINARY_IDENTITY = {
    'original_af4_sha256': common.AF_SHA, 'native_executable_sha256': ORDINARY_EXE_SHA,
    'bank_sha256': BANK_SHA, 'input_manifest_sha256': MANIFEST_SHA,
    'gold_metadata_sha256': META_SHA, 'role_order_sha256': ROLE_ORDER_SHA,
}


def require_fixed_root(path):
    path = Path(path)
    common.need(path.is_absolute() and path.name.startswith('je01_role_directional_fixed_') and
                path.parent == WORK and path.resolve() == path and not path.is_symlink() and
                path != ORDINARY, 'JE01_FIXED_SEPARATE_RESEARCH_ROOT_REQUIRED')
    return path


def require_fixed_output(path):
    root = require_fixed_root(ROOT)
    path = Path(path)
    common.need(path.is_absolute() and path.is_relative_to(root) and path != root and
                path.resolve() == path and not path.is_symlink(),
                'JE01_FIXED_OUTPUT_PARENT_LINK_OR_ESCAPE')
    return path


def validate_cell_request(role, tier, alpha, ordinary=False, margins=False, attempt=1):
    common.need(type(role) is int and role in range(24) and tier in ('Q4','Q5') and
                alpha in ('0',)+FD_ALPHAS and type(attempt) is int and attempt >= 1,
                'JE01_FIXED_CELL_GRID_OR_ALPHA1_FORBIDDEN')
    common.need(not ordinary or alpha == '0', 'JE01_FIXED_ORDINARY_CONTROL_ONLY')
    common.need(not margins or alpha == '0', 'JE01_FIXED_MARGINS_REQUIRE_ZERO')


def validate_baseline_identity(off, dataset):
    common.need(off.get('receipt_sha256') == OFF_RECEIPT and
                dataset.get('receipt_sha256') == DATASET_RECEIPT,
                'JE01_FIXED_BASELINE_RECEIPT_PIN')
    common.need(off.get('schema') == common.SCHEMA and off.get('status') == 'SUCCEEDED_EXIT0' and
                off.get('native_overlay_enabled') is False and off.get('alpha') == 0 and
                off.get('output_sha256') == common.BASE_SHA and
                off.get('source_identity') == EXPECTED_ORDINARY_IDENTITY,
                'JE01_FIXED_ORIGINAL_OFF_IDENTITY')
    common.need(off.get('route_output_sha256') == ROUTE_SHA and
                off.get('route_output_bytes') == ROUTE_BYTES and
                off.get('routed_token_layer_events') == ROUTE_RECORDS and
                off.get('route_output_path') == str(OFF_FOLDER/'routes.bin'),
                'JE01_FIXED_BASELINE_ROUTE_PIN')
    common.need(dataset.get('scope') == off.get('scope') == 'TRAIN16_DIAGNOSTIC' and
                dataset.get('case_ids') == off.get('case_ids') == list(range(16)) and
                dataset.get('manifest_sha256') == MANIFEST_SHA and
                dataset.get('gold_metadata_sha256') == META_SHA and
                dataset.get('baseline_output_sha256') == common.BASE_SHA and
                dataset.get('target_token_ids') == off.get('target_token_ids') and
                sum(map(len, off['target_token_ids'])) == 656,
                'JE01_FIXED_BASELINE_INPUT_ALIGNMENT')


def verified_baseline():
    common.need(common.sha(Path(common.__file__)) == COMMON_RUNNER_SHA,
                'JE01_FIXED_COMMON_READER_SOURCE_DRIFT')
    off = common.sealed(OFF_FOLDER/'RUN_RECEIPT.json', OFF_RECEIPT)
    dataset = common.sealed(DATASET, DATASET_RECEIPT)
    validate_baseline_identity(off, dataset)
    common.validate_dataset(dataset)
    route = OFF_FOLDER/'routes.bin'
    common.need(route.is_file() and not route.is_symlink() and
                route.stat().st_size == ROUTE_BYTES and common.sha(route) == ROUTE_SHA,
                'JE01_FIXED_ORIGINAL_ROUTE_FILE_HASH')
    rows = patch.parse_baseline_routes(route.read_bytes())
    common.need(len(rows) == ROUTE_RECORDS, 'JE01_FIXED_ORIGINAL_ROUTE_COUNT')
    gold = OFF_FOLDER/'gold_native.bin'
    common.need(gold.is_file() and not gold.is_symlink() and common.sha(gold) == common.BASE_SHA,
                'JE01_FIXED_ORIGINAL_GOLD_HASH')
    values, targets = common.read_gold(gold.read_bytes(), Path(dataset['gold_metadata']).read_bytes())
    common.need(values == off['per_token_nll'] and targets == off['target_token_ids'],
                'JE01_FIXED_ORIGINAL_GOLD_REPLAY')
    return off, dataset, rows


def build():
    root = require_fixed_root(ROOT)
    common.need(not any((root/name).exists() or (root/name).is_symlink()
                        for name in ('source','build','bin','NATIVE_BUILD_RECEIPT.json')),
                'JE01_FIXED_NEW_BUILD_ROOT_REQUIRED')
    old = common.sealed(ORDINARY/'NATIVE_BUILD_RECEIPT.json', ORDINARY_BUILD_RECEIPT)
    common.need(old['binary_sha256'] == ORDINARY_EXE_SHA and
                common.sha(ORDINARY/'bin/je01_role_directional_native') == ORDINARY_EXE_SHA,
                'JE01_FIXED_ORDINARY_NATIVE_BUILD_PIN')
    source = ORDINARY/'source'
    common.need(source.is_dir() and not source.is_symlink() and
                common.sha(source/'mlx_moe.cpp') == patch.CPP_SHA256 and
                common.sha(source/'qwen_infer.c') == patch.C_SHA256,
                'JE01_FIXED_ORDINARY_SOURCE_PIN')
    common.need(not any(p.is_symlink() for p in source.rglob('*')),
                'JE01_FIXED_SOURCE_COPY_MUST_NOT_FOLLOW_LINKS')
    root.mkdir(exist_ok=True, mode=0o700)
    shutil.copytree(source, root/'source', symlinks=True)
    # The shared bank verifier resolves its immutable calibration ledger beside
    # the staged tools. Copy that signed JSON without rewriting its contents.
    ledger = 'JE01_ACTUAL_CAL240_FFN_H2_SWIGLU_NATIVE_OUTPUT_DP_20261122.json'
    common.sealed(source/'docs/research'/ledger,
                  '35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb')
    require_fixed_output(root/'docs/research').mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/'docs/research'/ledger,require_fixed_output(root/'docs/research'/ledger))
    applied = patch.stage(root/'source')
    common.need(applied['patched_cpp_sha256'] == FIXED_CPP_SHA,
                'JE01_FIXED_EXPECTED_SOURCE_PATCH_MISMATCH')
    common.write_seal(root/'SOURCE_PATCH_RECEIPT.json', applied)
    cm = root/'source/CMakeLists.txt'
    original = cm.read_text()
    anchor = 'OUTPUT_NAME je01_role_directional_native'
    common.need(original.count(anchor) == 1, 'JE01_FIXED_CMAKE_OUTPUT_ANCHOR')
    cm.write_text(original.replace(anchor,'OUTPUT_NAME je01_role_directional_fixed_native',1))
    import mlx
    cmake = BC_BUILD/'venv/bin/cmake'
    mlx_cmake = Path(list(mlx.__path__)[0])/'share/cmake'
    common.need(cmake.is_file() and mlx_cmake.is_dir(), 'JE01_FIXED_TOOLCHAIN_MISSING')
    subprocess.run([str(cmake),'-S',str(root/'source'),'-B',str(root/'build'),'-G','Ninja',
                    '-DCMAKE_BUILD_TYPE=Release','-DMLX_CMAKE_DIR='+str(mlx_cmake)],check=True)
    subprocess.run([str(cmake),'--build',str(root/'build'),'--target','qwen_infer_gpu',
                    '--parallel','2'],check=True)
    (root/'bin').mkdir(mode=0o700)
    exe = root/'bin/je01_role_directional_fixed_native'
    shutil.copy2(root/'build/je01_role_directional_fixed_native',exe)
    exe.chmod(0o700)
    common.need(exe.stat().st_size > 100000, 'JE01_FIXED_NATIVE_BINARY_TOO_SMALL')
    result = {'schema':'beglin-je01-native-fixed-route-build/1',
              'status':'JE01_FIXED_ROUTE_NATIVE_COMPILE_PASS',
              'ordinary_build_receipt_sha256':ORDINARY_BUILD_RECEIPT,
              'ordinary_native_sha256':ORDINARY_EXE_SHA,
              'base_cpp_sha256':patch.CPP_SHA256,'unchanged_c_sha256':patch.C_SHA256,
              'patched_cpp_sha256':common.sha(root/'source/mlx_moe.cpp'),
              'source_patch_receipt_sha256':applied['receipt_sha256'],
              'binary_sha256':common.sha(exe),'binary_bytes':exe.stat().st_size,
              'build_and_runner_source_sha256':common.sha(Path(__file__)),
              'patch_source_sha256':common.sha(Path(patch.__file__)),
              'ordinary_off_parity':'NOT_RUN','replay_zero_parity':'NOT_RUN',
              'native_directional_derivative':'NOT_MEASURED','production_modified':False}
    common.write_seal(root/'NATIVE_BUILD_RECEIPT.json',result)
    print(json.dumps({'event':'FIXED_NATIVE_BUILD_DONE','binary_sha256':result['binary_sha256'],
                      'receipt_sha256':result['receipt_sha256']},sort_keys=True),flush=True)
    return result


def preflight():
    require_fixed_root(ROOT)
    off, dataset, routes = verified_baseline()
    data, manifest, meta, ordinary_exe, lib, identity = common.preflight(DATASET)
    common.need(identity == EXPECTED_ORDINARY_IDENTITY and data == dataset and
                common.sha(ordinary_exe) == ORDINARY_EXE_SHA,
                'JE01_FIXED_ORDINARY_PREFLIGHT_DRIFT')
    bank = common.sealed(common.BANK/'JE01_ROLE_DIRECTION_BANK_RECEIPT.json', BANK_RECEIPT)
    common.need(bank['physical_file_sha256'] == BANK_SHA and
                bank['role_manifest_sha256'] == BANK_TSV_SHA,
                'JE01_FIXED_BANK_RECEIPT_PIN')
    receipt = common.sealed(ROOT/'NATIVE_BUILD_RECEIPT.json')
    common.need(receipt.get('status') == 'JE01_FIXED_ROUTE_NATIVE_COMPILE_PASS' and
                receipt.get('ordinary_build_receipt_sha256') == ORDINARY_BUILD_RECEIPT and
                receipt.get('base_cpp_sha256') == patch.CPP_SHA256 and
                receipt.get('patched_cpp_sha256') == FIXED_CPP_SHA and
                receipt.get('unchanged_c_sha256') == patch.C_SHA256 and
                receipt.get('production_modified') is False,
                'JE01_FIXED_BUILD_RECEIPT_CONTRACT')
    exe = ROOT/'bin/je01_role_directional_fixed_native'
    common.need(exe.is_file() and not exe.is_symlink() and
                common.sha(exe) == receipt['binary_sha256'] and
                common.sha(ROOT/'source/mlx_moe.cpp') == FIXED_CPP_SHA and
                common.sha(ROOT/'source/qwen_infer.c') == patch.C_SHA256,
                'JE01_FIXED_NATIVE_OR_SOURCE_SHA_DRIFT')
    identity = {**identity,'native_executable_sha256':receipt['binary_sha256'],
                'ordinary_off_receipt_sha256':OFF_RECEIPT,
                'ordinary_native_sha256':ORDINARY_EXE_SHA,
                'baseline_routes_sha256':ROUTE_SHA,
                'diagnostic_definition_sha256':common.canonical({
                    'mode':'FIXED_BASELINE_HARD_TOPK_DYNAMIC_SCORES_AND_WEIGHTS',
                    'route_sha256':ROUTE_SHA,'records':ROUTE_RECORDS,
                    'full_prompt_KV_recomputed':True,'alpha1_forbidden':True})}
    return off, dataset, routes, manifest, meta, exe, lib, identity


def read_margins(raw, routes):
    common.need(raw.startswith(patch.MARGIN_MAGIC), 'JE01_FIXED_MARGIN_MAGIC')
    body = raw[len(patch.MARGIN_MAGIC):]
    common.need(len(body) == 28*len(routes), 'JE01_FIXED_MARGIN_RECORD_COUNT')
    minimum = math.inf
    ties = 0
    by_layer = {}
    for record, route in zip(struct.iter_unpack('<4i3f',body),routes):
        common.need(record[:4] == route[:4], 'JE01_FIXED_MARGIN_ALIGNMENT')
        selected,unselected,gap = record[4:]
        common.need(all(math.isfinite(x) for x in (selected,unselected,gap)) and
                    0 <= unselected <= selected <= 1 and gap >= 0 and
                    math.isclose(gap,selected-unselected,rel_tol=2e-7,abs_tol=1e-12),
                    'JE01_FIXED_MARGIN_VALUE_INVALID')
        minimum = min(minimum,gap)
        ties += gap == 0
        key = str(route[1])
        item = by_layer.setdefault(key,{'events':0,'minimum_gap':math.inf,'exact_ties':0})
        item['events'] += 1
        item['minimum_gap'] = min(item['minimum_gap'],gap)
        item['exact_ties'] += gap == 0
    return {'events':len(routes),'minimum_selected_minus_unselected':minimum,
            'exact_boundary_ties':ties,'all_membership_gaps_positive':ties == 0,
            'by_layer':by_layer,'within_selected_order_ties':'NOT_MEASURED',
            'finite_h_ordinary_routing_stability':'NOT_ESTABLISHED_BY_BASELINE_MARGINS'}


def folder_for(role,tier,alpha,ordinary=False,attempt=1):
    label = 'ordinary_off' if ordinary else 'r%02d_%s_a%s'%(role,tier,alpha.replace('+','p').replace('-','m'))
    return require_fixed_output(ROOT/'runs/pilot16'/(label+'_attempt'+str(attempt)))


def verify_record(folder, identity, *, role, tier, alpha, ordinary, margins):
    record = common.sealed(folder/'RUN_RECEIPT.json')
    common.need(record.get('schema') == SCHEMA and record.get('status') == 'SUCCEEDED_EXIT0' and
                record.get('source_identity') == identity and record.get('alpha') == common.ALPHAS[alpha] and
                record.get('fixed_route_replay_enabled') == (not ordinary) and
                record.get('native_overlay_enabled') == (not ordinary) and
                record.get('role_index') == (None if ordinary else role) and
                record.get('tier') == (None if ordinary else tier) and
                record.get('margin_capture_enabled') == margins,
                'JE01_FIXED_RESUME_CELL_IDENTITY')
    common.need(common.sha(folder/'gold_native.bin') == record['output_sha256'] and
                common.sha(folder/'routes.bin') == record['route_output_sha256'] == ROUTE_SHA,
                'JE01_FIXED_RESUME_OUTPUT_HASH')
    dataset = common.sealed(DATASET, DATASET_RECEIPT)
    values,targets = common.read_gold((folder/'gold_native.bin').read_bytes(),
                                      Path(dataset['gold_metadata']).read_bytes())
    common.need(values == record['per_token_nll'] and targets == record['target_token_ids'] and
                record['case_ids'] == list(range(16)), 'JE01_FIXED_RESUME_GOLD_REPLAY')
    rows = patch.parse_baseline_routes((folder/'routes.bin').read_bytes())
    common.need(len(rows) == ROUTE_RECORDS, 'JE01_FIXED_RESUME_ROUTE_COUNT')
    if alpha == '0':
        common.need(record['output_sha256'] == common.BASE_SHA, 'JE01_FIXED_ZERO_CONTROL_HASH_MISMATCH')
    if margins:
        common.need(common.sha(folder/'baseline_margins.bin') == record['margin_output_sha256'] and
                    read_margins((folder/'baseline_margins.bin').read_bytes(),rows) == record['baseline_margins'],
                    'JE01_FIXED_RESUME_MARGIN_HASH_OR_REPLAY')
    return record


def run_cell(role=0,tier='Q4',alpha='0',*,ordinary=False,margins=False,attempt=1,resume=False):
    validate_cell_request(role,tier,alpha,ordinary,margins,attempt)
    off,data,base_routes,manifest,meta,exe,lib,identity = preflight()
    # A repeated run is only accepted when its exact result is verified.
    folder = folder_for(role,tier,alpha,ordinary,attempt)
    if folder.exists():
        common.need(resume,'JE01_FIXED_CELL_ALREADY_ATTEMPTED')
        return verify_record(folder,identity,role=role,tier=tier,alpha=alpha,
                             ordinary=ordinary,margins=margins)
    if not ordinary:
        verify_record(folder_for(0,'Q4','0',True,attempt),identity,
                      role=0,tier='Q4',alpha='0',ordinary=True,margins=True)
    if alpha != '0':
        verify_record(folder_for(0,'Q4','0',False,attempt),identity,
                      role=0,tier='Q4',alpha='0',ordinary=False,margins=False)
    common.need(not folder.is_symlink(),'JE01_FIXED_OUTPUT_SYMLINK')
    folder.mkdir(parents=True,mode=0o700)
    (folder/'tmp').mkdir(mode=0o700)
    target,route = folder/'gold_native.bin',folder/'routes.bin'
    env = {'PATH':'/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin','HOME':'/Users/eoe',
           'TMPDIR':str(folder/'tmp'),'DYLD_LIBRARY_PATH':str(lib),'QWEN_MOE_BASE':str(common.MODEL),
           'QWEN_MOE_GPU_CBATCH_ONLINE':'1','QWEN_MOE_CB_SLOTS':'1','QWEN_MOE_CB_REQS':'16',
           'QWEN_MOE_CB_PREFILL_BUDGET':'1','QWEN_MOE_CB_PROMPT_MANIFEST':str(manifest),
           'QWEN_MOE_GPU_SORT_THRESHOLD':'64','QWEN_MOE_GPU_CBATCH_BUDGET':'0',
           'JE01_ORIGINAL_GOLD_META':str(meta),'JE01_ORIGINAL_GOLD_OUTPUT':str(target),
           'JE01_DIRECTION_ROUTE_OUTPUT':str(route)}
    if not ordinary:
        env.update({'JE01_REAL_Q35_BUNDLE':str(common.BANK/'JE01_ROLE_DIRECTION_BANK.bin'),
                    'JE01_REAL_Q35_TSV':str(common.BANK/'JE01_ROLE_DIRECTION_BANK.tsv'),
                    'JE01_DIRECTION_ROLE':str(role),'JE01_DIRECTION_TIER':tier,
                    'JE01_DIRECTION_ALPHA':alpha,
                    'JE01_DIRECTION_BASELINE_ROUTES':str(OFF_FOLDER/'routes.bin')})
    if margins:
        env['JE01_DIRECTION_MARGIN_OUTPUT'] = str(folder/'baseline_margins.bin')
    started = time.monotonic()
    lowest = common.memory_free()
    common.need(lowest >= 55,'JE01_FIXED_INITIAL_MEMORY_BELOW55')
    print(json.dumps({'event':'FIXED_NATIVE_CELL_START','cell':folder.name,
                      'ordinary_routing':ordinary,'memory_free_pct':lowest}),flush=True)
    stop = None
    with (folder/'stdout.txt').open('xb') as out,(folder/'stderr.txt').open('xb') as err:
        proc = subprocess.Popen([str(exe)],cwd=str(folder),env=env,stdout=out,stderr=err)
        try:
            while proc.poll() is None:
                free = common.memory_free(); lowest = min(lowest,free)
                if free < 35: stop = 'MEMORY_BELOW35'
                if time.monotonic()-started > 420: stop = 'TIMEOUT420'
                rss = subprocess.run(['/bin/ps','-o','rss=','-p',str(proc.pid)],
                                     capture_output=True,text=True,timeout=5).stdout.strip()
                if rss.isdigit() and int(rss) > 16*1024*1024: stop = 'RSS_OVER16GB'
                if stop:
                    proc.terminate(); break
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
    common.need(code == 0 and stop is None and 'FATAL:' not in errors and
                'RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16' in errors and
                ((common.MARKER in errors) == (not ordinary)) and
                (('JE01_FIXED_BASELINE_ROUTES_ENABLED records='+str(ROUTE_RECORDS)) in errors) == (not ordinary),
                'JE01_FIXED_NATIVE_CELL_FAILED:'+str(code)+':'+str(stop)+':'+errors[-1800:])
    values,targets = common.read_gold(target.read_bytes(),meta.read_bytes())
    rows = patch.parse_baseline_routes(route.read_bytes())
    common.need(rows == base_routes and common.sha(route) == ROUTE_SHA,
                'JE01_FIXED_REPLAY_ROUTE_OUTPUT_MISMATCH')
    output_sha = common.sha(target)
    if alpha == '0':
        common.need(output_sha == common.BASE_SHA and values == off['per_token_nll'],
                    'JE01_FIXED_ALPHA_ZERO_NOT_EXACT_ORIGINAL')
    margin_summary = read_margins((folder/'baseline_margins.bin').read_bytes(),base_routes) if margins else None
    record = {'schema':SCHEMA,'status':'SUCCEEDED_EXIT0','scope':'TRAIN16_DIAGNOSTIC',
              'source_identity':identity,'alpha':common.ALPHAS[alpha],
              'direction_index':None if alpha == '0' else role*2+(tier=='Q5'),
              'case_ids':data['case_ids'],'target_token_ids':targets,'per_token_nll':values,
              'output_sha256':output_sha,'baseline_original_gold_sha256':common.BASE_SHA,
              'native_overlay_enabled':not ordinary,'fixed_route_replay_enabled':not ordinary,
              'role_index':None if ordinary else role,'role_key':None if ordinary else common.ROLE_ORDER[role],
              'tier':None if ordinary else tier,'routing_mode':'ORDINARY_CONTROL' if ordinary else 'FIXED_BASELINE_HARD_INDICES_DYNAMIC_WEIGHTS',
              'baseline_route_sha256':ROUTE_SHA,'baseline_route_bytes':ROUTE_BYTES,
              'baseline_route_records':ROUTE_RECORDS,'original_off_receipt_sha256':OFF_RECEIPT,
              'route_output_sha256':common.sha(route),'route_output_path':str(route),
              'route_output_bytes':route.stat().st_size,'routed_token_layer_events':len(rows),
              'route_equality_interpretation':'REPLAY_CONFORMANCE_NOT_NATURAL_ROUTING_STABILITY',
              'margin_capture_enabled':margins,'baseline_margins':margin_summary,
              'margin_output_sha256':common.sha(folder/'baseline_margins.bin') if margins else None,
              'mean_gold_nll':math.fsum(map(math.fsum,values))/656,
              'derivative_kind':'FINITE_DIFFERENCE_FIXED_HARD_BRANCH_NATIVE_ROLE_DIRECTION',
              'full_prompt_and_KV_recomputed':True,'router_scores_and_selected_weights_dynamic':True,
              'min_memory_free_pct':lowest,'elapsed_seconds':time.monotonic()-started,
              'native_exit_code':code,'runner_source_sha256':common.sha(Path(__file__)),
              'allocation_selected':False,'independent_validation_ood_accessed':False,'production_modified':False}
    common.write_seal(folder/'RUN_RECEIPT.json',record)
    print(json.dumps({'event':'FIXED_NATIVE_CELL_DONE','cell':folder.name,'output_sha256':output_sha,
                      'mean_gold_nll':record['mean_gold_nll'],'receipt_sha256':record['receipt_sha256']},sort_keys=True),flush=True)
    return record


def summarize_gate(runs):
    common.need(len(runs) == 6,'JE01_FIXED_SIX_CELL_DIAGNOSTIC_REQUIRED')
    control,zero,*perturbed = runs
    common.need(control['output_sha256'] == zero['output_sha256'] == common.BASE_SHA and
                control['route_output_sha256'] == zero['route_output_sha256'] == ROUTE_SHA and
                control['per_token_nll'] == zero['per_token_nll'], 'JE01_FIXED_CONTROL_PARITY_REQUIRED')
    for r in runs:
        common.need(r['schema'] == SCHEMA and r['source_identity'] == control['source_identity'] and
                    r['case_ids'] == control['case_ids'] and r['target_token_ids'] == control['target_token_ids'] and
                    r['route_output_sha256'] == ROUTE_SHA,'JE01_FIXED_DIAGNOSTIC_RUN_ALIGNMENT')
    columns = {r['alpha']:[x for row in r['per_token_nll'] for x in row] for r in perturbed}
    common.need(set(columns) == {-.05,.05,-.025,.025} and
                all(r['role_index'] == 0 and r['tier'] == 'Q4' and r['fixed_route_replay_enabled']
                    for r in perturbed),'JE01_FIXED_GATE_Q4_ONLY')
    coarse = [(a-b)/.1 for a,b in zip(columns[.05],columns[-.05])]
    fine = [(a-b)/.05 for a,b in zip(columns[.025],columns[-.025])]
    common.need(len(fine) == 656 and all(math.isfinite(x) for x in coarse+fine),
                'JE01_FIXED_DERIVATIVE_CARDINALITY_OR_FINITE')
    rms = math.sqrt(math.fsum(x*x for x in fine)/656)
    disagreement = math.sqrt(math.fsum((a-b)**2 for a,b in zip(coarse,fine))/656)
    threshold = 1e-4+.05*rms
    denominator = math.sqrt(math.fsum(x*x for x in coarse)*math.fsum(x*x for x in fine))
    convergence = {'coarse_h':.05,'fine_h':.025,'absolute_tolerance':1e-4,'relative_tolerance':.05,
                   'rms_fine_derivative':rms,'rms_disagreement':disagreement,
                   'rms_acceptance_threshold':threshold,'passed':disagreement <= threshold,
                   'cosine_similarity':math.fsum(a*b for a,b in zip(coarse,fine))/denominator if denominator else None,
                   'max_absolute_disagreement':max(abs(a-b) for a,b in zip(coarse,fine))}
    return {'schema':'beglin-je01-native-fixed-hard-route-six-cell-gate-diagnostic/1',
            'status':'FIXED_BRANCH_CONVERGED' if convergence['passed'] else 'FIXED_BRANCH_NUMERICAL_REJECTED',
            'source_identity':control['source_identity'],'scope':'TRAIN16_DIAGNOSTIC',
            'cases':16,'gold_tokens':656,'role_key':common.ROLE_ORDER[0],'tier':'Q4',
            'ordinary_off_raw_gold_and_routes_identical':True,'replay_zero_raw_gold_and_routes_identical':True,
            'baseline_margins':control['baseline_margins'],'convergence':convergence,
            'mean_coarse_directional_nll_derivative':math.fsum(coarse)/656,
            'mean_fine_directional_nll_derivative':math.fsum(fine)/656,
            'per_token_fine_derivative':fine,
            'observed_label_directional_gradient_square_mean':math.fsum(x*x for x in fine)/656,
            'run_receipt_sha256':[r['receipt_sha256'] for r in runs],
            'full_48_direction_fisher':'NOT_MEASURED','allocation_selected':False,
            'ordinary_finite_h_routing_stability':'NOT_INFERRED_FROM_FORCED_ROUTE_EQUALITY',
            'baseline_positive_gap_scope':'LOCAL_MEMBERSHIP_ONLY_NOT_WITHIN_SELECTED_ORDER_OR_FINITE_STEP_STABILITY',
            'independent_validation_ood_accessed':False,'production_modified':False}


def diagnostic(resume=False):
    runs = [run_cell(ordinary=True,margins=True,resume=resume),
            run_cell(resume=resume)]
    for alpha in FD_ALPHAS:
        runs.append(run_cell(alpha=alpha,resume=resume))
    report = summarize_gate(runs)
    common.write_seal(ROOT/'JE01_FIXED_ROUTE_GATE_SIX_CELL_DIAGNOSTIC.json',report)
    print(json.dumps({'event':'FIXED_GATE_DIAGNOSTIC_COMPLETE','status':report['status'],
                      'convergence':report['convergence'],'receipt_sha256':report['receipt_sha256']},sort_keys=True),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('build','diagnostic','cell'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--alpha',choices=('0',)+FD_ALPHAS,default='0')
    parser.add_argument('--ordinary',action='store_true')
    parser.add_argument('--margins',action='store_true')
    args = parser.parse_args()
    if args.action == 'build': build()
    elif args.action == 'diagnostic': diagnostic(args.resume)
    else: run_cell(alpha=args.alpha,ordinary=args.ordinary,margins=args.margins,resume=args.resume)
