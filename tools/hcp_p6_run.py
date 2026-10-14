#!/usr/bin/env python3
"""Bounded sequential XOX P6 research, real native ACKs, no serving writes."""
from __future__ import annotations
import argparse,contextlib,hashlib,json,math,os,pathlib,random,shutil,subprocess,time
import numpy as np
import precision_context as pc
import gpu_runtime_control as grc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
from hcp_p4_long_prompt_registry import prepare
from hcp_p4_native_policies_ab import A,B,sha
from hcp_p6_metrics import read_dump,logit_metrics,teacher_forced,summarize_speed
ROOT=pathlib.Path('/Users/xox/mcp-sandbox/tailnet-commander/hcp-p6-af-logits-20261014-1104')
REPO=ROOT/'src'
TOKENIZER=pathlib.Path('/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors')
FEATURE_SHA='c3b54b1ebe41d9a85ed94ddd0da87cde0de86ecf0af56d2e621733e439944f53'
INDEX_SHA='d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf'
BINARY_SHA='45bb3b04c621a25af27b312797e2054847764c061302cc6a3876f772e079ab25'
STAGE=pathlib.Path(ps.PERSISTENT_BINARY)
STAGE_SHA=ps.EXPECTED_PERSISTENT_BINARY_SHA
LIVE=pathlib.Path('/Users/xox/vdsp_serving/active-route.json')
POLICIES={'AF':[],'A':A,'B':B}
LIMIT_RSS=14*1024**3

def dump_json(path,obj):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def sequence_sha(seq):
    return hashlib.sha256(','.join(map(str,seq)).encode()).hexdigest()

def stage_workers():
    text=subprocess.check_output(['/bin/ps','-axo','pid=,comm='],text=True)
    return sorted(int(line.split()[0]) for line in text.splitlines() if str(STAGE) in line)

def preserved():
    return {'stage_sha256':sha(STAGE),'active_route_sha256':sha(LIVE) if LIVE.exists() else None,
            'live_stage_worker_pids':stage_workers()}

def preflight(mode):
    if ROOT.resolve()!=ROOT or REPO.resolve()!=REPO:raise ValueError('research root is symlinked')
    if sha(STAGE)!=STAGE_SHA:raise ValueError('original stage hash changed')
    if sha(ROOT/'build/qwen_infer_gpu')!=BINARY_SHA:raise ValueError('unrecognized research binary')
    if sha(TOKENIZER/'model.safetensors.index.json')!=INDEX_SHA:raise ValueError('checkpoint index changed')
    if sha(ROOT/'inputs/p4_derived_long_features.json')!=FEATURE_SHA:raise ValueError('risk source changed')
    if shutil.disk_usage(ROOT).free<3*1024**3:raise ValueError('research free disk below 3 GiB')
    if int(subprocess.check_output(['/usr/sbin/sysctl','-n','hw.memsize']))<40*1024**3:raise ValueError('host below memory bound')
    for name in ['qwen_infer.c','mlx_moe.cpp','CMakeLists.txt']:
        expected=json.loads((ROOT/'build-source.json').read_text())['instrumented_source_sha256'][name]
        if sha(ROOT/'probe-src'/name)!=expected:raise ValueError('instrumented source changed')
    registry=prepare(ROOT/'inputs/bank',TOKENIZER)
    selected=[registry['prompts'][3*i+i%3] for i in range(12)]
    frozen={r['prompt_id']:r for r in json.loads((ROOT/'inputs/p4_derived_long_features.json').read_text())['runs']}
    for row in selected:
        ref=frozen[row['id']]
        if ref['token_sha256']!=row['token_sha256'] or ref['prompt_length']!=row['tokens']:
            raise ValueError('not exact original P5 input')
        if not 104<=row['tokens']<=112 or row['tokens']+4>=128:raise ValueError('input bounds')
    if len({r['domain'] for r in selected})!=12:raise ValueError('missing domains')
    state=preserved()
    provenance={'schema':'beglin-hcp-p6-preflight-v1','mode':mode,'before':state,
        'instrumented_binary_sha256':BINARY_SHA,'index_sha256':INDEX_SHA,
        'af_binding_source':'original AF loaded by the identical GPU online path; empty promotion map',
        'protocol_sha256':sha(REPO/'experiments/hcp_p6/protocol.json'),
        'runner_source_sha256':sha(REPO/'tools/hcp_p6_run.py'),
        'metrics_source_sha256':sha(REPO/'tools/hcp_p6_metrics.py'),
        'research_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'prompts':[{k:v for k,v in r.items() if k!='token_ids'} for r in selected],
        'checkpoint_shards_rehashed_this_run':False,
        'production_touched':False,'automatic_live_promotion':False}
    dump_json(ROOT/f'{mode}-preflight.json',provenance)
    ps.PERSISTENT_BINARY=ROOT/'build/qwen_infer_gpu';ps.EXPECTED_PERSISTENT_BINARY_SHA=BINARY_SHA
    ps.REQUEST_TIMEOUT_SECONDS=max(ps.REQUEST_TIMEOUT_SECONDS,180)
    base.REPO=REPO # only cwd in this research process; never write original checkout.
    return selected,state

@contextlib.contextmanager
def worker(name, *, af=False, capture=False):
    path=ROOT/'workers'/name
    if path.exists():raise ValueError('worker directory must be fresh; do not overwrite evidence')
    cls=ps.PersistentRouteWorker if af else ps.AdaptivePersistentRouteWorker
    w=cls(route=base.baseline_route() if af else base.candidate_route(),root=path)
    native_env=w._env
    def env():
        e=native_env();e.update({'QWEN_MOE_CB_SLOTS':'1','QWEN_MOE_CB_REQS':'1',
                                'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        if capture:e['BEGLIN_HCP_P6_LOGITS_PATH']=str(path/'logits.bin')
        return e
    w._env=env
    try:
        health=w.start();ack=grc.read_runtime_ack(w.ack_path)
        expected=POLICIES['AF' if af else 'A']
        if not health['alive'] or ack['active_policy']!=expected or ack['active_policy_hash']!=pc.policy_hash(expected):
            raise ValueError('real native startup policy mismatch')
        if af and ack['status']!='STARTUP_STATE':raise ValueError('original AF startup ACK mismatch')
        if int(health['rss_bytes'])>LIMIT_RSS:raise ValueError('startup RSS over bound')
        print('P6_WORKER_READY',json.dumps({'name':name,'pid':health['pid'],'policy':'AF' if af else 'A','capture':capture}),flush=True)
        yield w
    finally:
        w.stop(force=True)

def infer(w,row,policy,admission_id):
    target=POLICIES[policy]
    start=time.perf_counter_ns()
    if policy=='AF':
        result=w.submit([(row['token_ids'],4)])
    else:
        result=w.submit_with_precision_policy([(row['token_ids'],4)],target_policy=target,admission_id=admission_id)
    wall=(time.perf_counter_ns()-start)/1e6
    if result.get('finite_logits') is not True or result.get('inference_passes',1)!=1:
        raise ValueError('nonfinite logits or unapproved correction pass')
    answers=result.get('responses')
    if len(answers)!=1 or not 1<=len(answers[0])<=4:raise ValueError('invalid native token response')
    seq=list(map(int,answers[0]));ack=grc.read_runtime_ack(w.ack_path)
    if ack['active_policy']!=target or ack['active_policy_hash']!=pc.policy_hash(target):
        raise ValueError('real native post-admission policy mismatch')
    health=w.health()
    if not health['alive'] or int(health['rss_bytes'])>LIMIT_RSS:raise ValueError('worker death/RSS bound')
    epoch=result.get('precision_epoch') or {};switched=bool(epoch.get('transitioned'))
    if switched:
        if ack['status']!='REBIND_SET_APPLIED' or ack['txn_id']!=epoch['txn_id'] or len(epoch['changed_targets'])!=2 or epoch['after_epoch']!=epoch['before_epoch']+1:
            raise ValueError('missing exact two-target transaction ACK')
    timing={'engine_ms':float(result['engine_wall_ms']),'roundtrip_ms':float(result['roundtrip_ms']),
            'admission_wall_ms':wall,'transition_ms':float((epoch.get('transition_cost') or {}).get('transition_wall_ms',0)),
            'rss_bytes':int(health['rss_bytes']),'phase':'switch' if switched else 'steady',
            'epoch':int(ack['weight_epoch']),'native_ack_status':ack['status'],
            'sequence_sha256':sequence_sha(seq),'generated_tokens':len(seq)}
    if not all(math.isfinite(timing[k]) and timing[k]>0 for k in ['engine_ms','roundtrip_ms','admission_wall_ms']):
        raise ValueError('invalid native timing')
    return seq,timing

def verify_preserved(before):
    after=preserved()
    if before['stage_sha256']!=after['stage_sha256'] or before['active_route_sha256']!=after['active_route_sha256']:
        raise ValueError('production artifact changed during study')
    return {'before':before,'after':after,'stage_and_route_hashes_equal':True,
            'same_live_stage_worker_pids':before['live_stage_worker_pids']==after['live_stage_worker_pids']}

def quality():
    selected,before=preflight('quality');refs={};records=[];raw_hashes={}
    refdir=ROOT/'private_af_reference';refdir.mkdir(exist_ok=False)
    with worker('quality_af',af=True,capture=True) as w:
        for i,row in enumerate(selected):
            seq,timing=infer(w,row,'AF',f'af-{i}')
            path=w.root/'logits.bin';arr=read_dump(path,positions=row['tokens']+len(seq)-1)
            dest=refdir/(row['id']+'.bin');path.rename(dest)
            refs[row['id']]={'seq':seq,'dump_sha256':sha(dest),'teacher_forced':teacher_forced(arr[:row['tokens']],row['token_ids']),
                'positions':len(arr),'generated_tokens':len(seq)}
            raw_hashes['AF/'+row['id']]=sha(dest)
            print('P6_AF_CAPTURE',json.dumps({'completed':i+1,'prompt_tokens':row['tokens'],'full_vocab_positions':len(arr)}),flush=True)
    with worker('quality_ab',capture=True) as w:
        for i,row in enumerate(selected):
            ref=refs[row['id']];reference=read_dump(refdir/(row['id']+'.bin'),positions=ref['positions'])
            out={'prompt_id':row['id'],'prompt_index':i,'domain':row['domain'],'original_split':row['split'],
                 'prompt_tokens':row['tokens'],'token_sha256':row['token_sha256'],
                 'AF':{k:v for k,v in ref.items() if k!='seq'}}
            generated={}
            for policy in (('A','B') if i%2==0 else ('B','A')):
                seq,timing=infer(w,row,policy,f'quality-{i}-{policy}')
                path=w.root/'logits.bin';arr=read_dump(path,positions=row['tokens']+len(seq)-1)
                raw_hashes[policy+'/'+row['id']]=sha(path)
                out[policy]={'matched_prefill':logit_metrics(reference[:row['tokens']],arr[:row['tokens']]),
                    'last_prompt':logit_metrics(reference[row['tokens']-1:row['tokens']],arr[row['tokens']-1:row['tokens']]),
                    'teacher_forced':teacher_forced(arr[:row['tokens']],row['token_ids']),
                    'af_first_generated_token_agrees':seq[0]==ref['seq'][0],
                    'af_full_generated_sequence_agrees':seq==ref['seq'],
                    'sequence_sha256':sequence_sha(seq),'generated_tokens':len(seq),
                    'dump_sha256':raw_hashes[policy+'/'+row['id']],
                    'runtime_policy_hash':pc.policy_hash(POLICIES[policy])}
                generated[policy]=seq;path.unlink()
            out['a_b_first_generated_token_agrees']=generated['A'][0]==generated['B'][0]
            out['a_b_full_generated_sequence_agrees']=generated['A']==generated['B']
            records.append(out);dump_json(ROOT/'quality-progress.json',records)
            print('P6_QUALITY_PAIR',json.dumps({'completed':i+1,'domain':row['domain'],
                'A_AF_first':out['A']['af_first_generated_token_agrees'],'B_AF_first':out['B']['af_first_generated_token_agrees'],
                'A_last_rel_l2':out['A']['last_prompt']['relative_l2'],'B_last_rel_l2':out['B']['last_prompt']['relative_l2']}),flush=True)
    integrity=verify_preserved(before)
    summary={}
    total=sum(r['prompt_tokens'] for r in records)
    for name in ('AF','A','B'):
        n=sum(r[name]['teacher_forced']['predicted_input_tokens'] for r in records)
        loss=sum(r[name]['teacher_forced']['nll_sum'] for r in records)
        correct=sum(r[name]['teacher_forced']['correct_next_input_tokens'] for r in records)
        summary[name]={'next_input_tokens':n,'mean_prompt_nll':loss/n,'prompt_perplexity':math.exp(loss/n),
                       'next_input_token_accuracy':correct/n,'next_input_token_correct':correct,
                       'is_task_answer_accuracy':False}
        if name!='AF':
            summary[name].update({'first_token_AF_agreement_count':sum(r[name]['af_first_generated_token_agrees'] for r in records),
             'full_sequence_AF_agreement_count':sum(r[name]['af_full_generated_sequence_agrees'] for r in records),
             'prefill_top1_agreement':sum(r[name]['matched_prefill']['top1_agree_count'] for r in records)/total,
             'prefill_kl_mean':sum(r[name]['matched_prefill']['kl_af_to_candidate_mean']*r['prompt_tokens'] for r in records)/total,
             'last_prompt_rel_l2_mean':float(np.mean([r[name]['last_prompt']['relative_l2'] for r in records])),
             'last_prompt_rmse_mean':float(np.mean([r[name]['last_prompt']['rmse'] for r in records]))})
    # Keep sealed aggregates, not token values or huge model vectors.
    for path in refdir.glob('*.bin'):path.unlink()
    for path in (ROOT/'workers').glob('quality_*/worker.log'):path.unlink()
    for path in (ROOT/'workers').glob('quality_*/neartie.jsonl'):path.unlink()
    result={'schema':'beglin-hcp-p6-full-logit-quality-v1','status':'COMPLETE',
        'sample_prompts':12,'matched_prefill_positions':total,'vocabulary':102400,
        'same_gpu_binary_all_policies':BINARY_SHA,'summary':summary,'records':records,
        'a_b_first_agreement_count':sum(r['a_b_first_generated_token_agrees'] for r in records),
        'a_b_full_sequence_agreement_count':sum(r['a_b_full_generated_sequence_agrees'] for r in records),
        'reference_is_bf16_or_human_ground_truth':False,'holdout_is_new_untouched':False,
        'timing_from_logit_capture_excluded_from_speed_claims':True,
        'raw_dump_sha256_before_deletion':raw_hashes,'raw_logit_files_removed':True,
        'integrity':integrity,'production_touched':False,'automatic_live_promotion':False}
    dump_json(ROOT/'quality.json',result)
    print('P6_QUALITY_COMPLETE',json.dumps({'summary':summary,'positions':total,'integrity':integrity},sort_keys=True),flush=True)

def speed():
    selected,before=preflight('speed')
    quality_result=json.loads((ROOT/'quality.json').read_text())
    if quality_result['status']!='COMPLETE':raise ValueError('complete quality evidence required')
    expected={r['prompt_id']:{p:r[p]['sequence_sha256'] for p in ('A','B')} for r in quality_result['records']}
    protocol=json.loads((REPO/'experiments/hcp_p6/protocol.json').read_text())['speed']
    if protocol['blocks']!=6 or protocol['seed']!=20261014:raise ValueError('protocol changed')
    rng=random.Random(protocol['seed']);plans=[]
    for block in range(6):
        indices=list(range(12));rng.shuffle(indices)
        first=['A']*6+['B']*6;rng.shuffle(first)
        plans.extend({'block':block,'prompt_index':i,'first':f} for i,f in zip(indices,first))
    dump_json(ROOT/'speed-order-preregistered.json',plans)
    records=[];warmups=0;preconditions=0
    with worker('speed_ab',capture=False) as w:
        # Warm every input shape and both policies; retain counts, exclude from timing.
        for i,row in enumerate(selected):
            for policy in ('A','B'):
                seq,_=infer(w,row,policy,f'warm-{i}-{policy}');warmups+=1
                if sequence_sha(seq)!=expected[row['id']][policy]:raise ValueError('capture-on/off token parity failed')
        print('P6_SPEED_WARMUP_PASS',json.dumps({'requests':warmups,'capture_on_off_sequence_parity':True}),flush=True)
        for plan in plans:
            block=plan['block'];i=plan['prompt_index'];row=selected[i]
            first=plan['first'];second='B' if first=='A' else 'A'
            ack=grc.read_runtime_ack(w.ack_path)
            if ack['active_policy_hash']!=pc.policy_hash(POLICIES[second]):
                infer(w,row,second,f'prepare-{block}-{i}');preconditions+=1
            for occurrence,policy in enumerate((first,first,second,second)):
                seq,timing=infer(w,row,policy,f'speed-{block}-{i}-{occurrence}-{policy}')
                phase='switch' if occurrence in (0,2) else 'steady'
                if timing['phase']!=phase:raise ValueError('not exact balanced switch/steady crossover')
                timing.update({'block':block,'prompt_index':i,'prompt_id':row['id'],'domain':row['domain'],
                               'policy':policy,'order':first+first+second+second,
                               'sequence_matches_quality':sequence_sha(seq)==expected[row['id']][policy]})
                if not timing['sequence_matches_quality']:raise ValueError('benchmark output changed against quality run')
                records.append(timing)
            dump_json(ROOT/'speed-progress.json',records)
            print('P6_SPEED_PAIR',json.dumps({'block':block+1,'prompt_index':i,'measured_requests':len(records)}),flush=True)
    if (ROOT/'workers/speed_ab/logits.bin').exists():raise ValueError('logit dump unexpectedly enabled in speed phase')
    integrity=verify_preserved(before)
    result={'schema':'beglin-hcp-p6-repeated-speed-v1','status':'COMPLETE',
        'blocks':6,'prompts':12,'measured_admissions':len(records),'warmup_admissions_excluded':warmups,
        'preconditioning_admissions_excluded':preconditions,
        'speed_order_sha256':sha(ROOT/'speed-order-preregistered.json'),
        'binary_sha256':BINARY_SHA,'summary':summarize_speed(records),
        'engine_time_already_includes_native_transition':True,'transition_added_twice':False,
        'primary_timing_is_full_python_admission':True,'capture_on_off_sequence_parity':True,
        'workers_sequential_not_concurrent':True,
        'integrity':integrity,'records':records,'production_touched':False,
        'automatic_live_promotion':False}
    dump_json(ROOT/'speed.json',result)
    for name in ('worker.log','neartie.jsonl'):
        path=ROOT/'workers/speed_ab'/name
        if path.exists():path.unlink()
    print('P6_SPEED_COMPLETE',json.dumps({'summary':result['summary'],'integrity':integrity},sort_keys=True),flush=True)

if __name__=='__main__':
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['quality','speed'],required=True);a=p.parse_args()
    try:(quality if a.mode=='quality' else speed)()
    except Exception as error:
        dump_json(ROOT/(a.mode+'-failure.json'),{'status':'FAILED','type':type(error).__name__,'message':str(error)[:3000],
                                             'production_touched':False})
        raise
