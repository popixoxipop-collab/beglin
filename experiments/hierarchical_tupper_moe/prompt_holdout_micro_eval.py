#!/usr/bin/env python3
"""Read-only prompt-heldout micro-expert routing evaluation on real DeepSeek inputs.

Consumes independently captured real h2 vectors and macro top-K routes produced by
an isolated Beglin scratch binary. Trains per expert only on actual routed
training prompts, selects a method using validation prompts, and reports final
holdout prompt errors with prompt-bootstrap intervals. No production mutation,
no native qNg64 parity, and no claim that an offline compensated FFN runs faster.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from real_activation_micro_router import read_capture, read_route, channels
from permutation_alignment_probe import read_rows
from real_checkpoint_reader import RealExpertTileReader
from trainable_micro_router import fit_one_expert
from hierarchical_moe_probe import tupper_encode, tupper_decode

SCHEMA="beglin-prompt-holdout-micro-router-v1"
METHODS=("static","static_mean_residual","dynamic","dynamic_mean_residual","dynamic_ridge_residual")
SPLITS=("train","validation","holdout")


def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def load_runs(root: Path, *, layer: int, hidden: int, expected_index_hash: str,
              minimum_prompts: int=12):
    meta=json.loads((root/'runs.json').read_text())
    if meta.get('schema')!='beglin-prompt-suite-v1' or meta.get('layer')!=layer:
        raise ValueError('invalid capture manifest contract')
    if meta.get('checkpoint_index_sha256')!=expected_index_hash:
        raise ValueError('checkpoint identity does not match local index')
    recs=meta.get('runs')
    if not isinstance(recs,list) or len(recs)<minimum_prompts:
        raise ValueError('insufficient independent prompts')
    seen_ids=set(); seen_token_hash=set(); counts=defaultdict(int)
    captures={}; routes={}; sources=[]
    for record in recs:
        name=record.get('prompt_id'); split=record.get('split')
        if not isinstance(name,str) or not name.startswith('p') or not name[1:].isdigit() or name in seen_ids:
            raise ValueError('duplicate or invalid prompt identity')
        if split not in SPLITS:
            raise ValueError('invalid group split')
        tokenhash=record.get('input_ids_sha256')
        if not isinstance(tokenhash,str) or len(tokenhash)!=64 or tokenhash in seen_token_hash:
            raise ValueError('missing or duplicate tokenized prompt identity')
        seen_ids.add(name);seen_token_hash.add(tokenhash)
        p=root/name
        hpath=p/'true_h2.bin'; rpath=p/'moe3a_c_routing.txt'
        if not hpath.is_file() or not rpath.is_file():
            raise ValueError('missing real hidden/routing evidence')
        if _sha256(hpath)!=record.get('capture_sha256') or _sha256(rpath)!=record.get('routing_sha256'):
            raise ValueError('prompt evidence hash mismatch')
        acts=read_capture(hpath,layer,hidden)
        if len(acts)!=record.get('position_count') or sorted(acts)!=list(range(len(acts))):
            raise ValueError('missing or incorrect real hidden positions')
        route=read_route(rpath,layer,set(acts))
        for pos,choices in route.items():
            if len(choices)!=meta.get('macro_top_k') or any(not(0<=eid<64) for eid,_ in choices):
                raise ValueError('incorrect actual macro-router topology')
            if any(not np.isfinite(weight) or weight<0 for _,weight in choices):
                raise ValueError('invalid macro-router probability')
        for pos,h in acts.items():
            key=(name,pos)
            captures[key]=h
            routes[key]=route[pos]
        counts[split]+=1
        sources.append({'prompt_id':name,'split':split,'positions':len(acts),'input_ids_sha256':tokenhash,
                        'capture_sha256':record['capture_sha256'],'routing_sha256':record['routing_sha256']})
    if any(counts[k]<3 for k in SPLITS):
        raise ValueError('need at least three independent prompts per split')
    return captures,routes,sources,dict(counts),meta


def _routed_indices(routes:dict, group_keys:list[tuple[str,int]])->dict[int,list[tuple[str,int]]]:
    d=defaultdict(list)
    for key in group_keys:
        for eid,weight in routes[key]:
            if weight>0:d[eid].append(key)
    return dict(d)


def _sums_by_prompt(keys:list[tuple[str,int]], targets:np.ndarray, predictions:np.ndarray):
    by_prompt=defaultdict(lambda:[0.,0.,0.])
    for i,(p,_) in enumerate(keys):
        r=targets[i]
        z=predictions[i]-r
        stat=by_prompt[p]
        stat[0]+=float(np.dot(r,r))
        stat[1]+=float(np.dot(z,z))
        stat[2]=max(stat[2],float(np.linalg.norm(z)/max(np.linalg.norm(r),1e-12)))
    return dict(by_prompt)


def _metric(keys, target, guess, *,bootstrap_seed=101, bootstrap_draws=300):
    sums=_sums_by_prompt(keys,target,guess)
    if not sums:raise ValueError('evaluation set is empty')
    numer=sum(v[1] for v in sums.values());denom=sum(v[0] for v in sums.values())
    rel=float(math.sqrt(numer/max(denom,1e-25)))
    groups=list(sorted(sums))
    rng=np.random.default_rng(bootstrap_seed)
    boots=[]
    if len(groups)>=3:
        for _ in range(bootstrap_draws):
            chosen=rng.choice(groups,size=len(groups),replace=True)
            boots.append(math.sqrt(sum(sums[p][1] for p in chosen)/max(sum(sums[p][0] for p in chosen),1e-25)))
    return {'relative_l2':rel,'worst_position_relative_l2':max(v[2] for v in sums.values()),
            'prompt_count':len(groups),'position_count':len(keys),
            'prompt_bootstrap_95ci':list(map(float,np.quantile(boots,[.025,.975]))) if boots else None}


def evaluate(index:Path, suite:Path, *,layer:int=1,hidden:int=2048, intermediate:int=1408,
             widths=(16,64),fractions=(.25,.5,.75), min_train_examples=4, bootstrap_draws=300):
    index=Path(index)
    index_sha=_sha256(index)
    hidden_state,routes,sources,prompt_counts,meta=load_runs(Path(suite),layer=layer,hidden=hidden,
                                                         expected_index_hash=index_sha)
    prompt_split={s['prompt_id']:s['split'] for s in sources}
    splits={k:[key for key in sorted(hidden_state) if prompt_split[key[0]]==k] for k in SPLITS}
    train,val,test=splits['train'],splits['validation'],splits['holdout']
    train_routes=_routed_indices(routes,train)
    eval_keys=val+test
    nval=len(val)
    lookup={key:i for i,key in enumerate(eval_keys)}
    eval_routes=_routed_indices(routes,eval_keys)
    seen_experts=sorted(set(train_routes)|set(eval_routes))
    if not seen_experts or any(e>=64 for e in seen_experts):
        raise ValueError('unexpected expert coverage')
    reader=RealExpertTileReader(index,layer=layer,expert_ids=seen_experts,
                                hidden=hidden,intermediate=intermediate)
    results={}
    for width in widths:
        if width<=0 or intermediate%width:raise ValueError('invalid micro width')
        ng=intermediate//width
        full=np.zeros((len(eval_keys),hidden),np.float64)
        guesses={f:{m:np.zeros_like(full) for m in METHODS} for f in fractions}
        activation_executions={f:0 for f in fractions}
        fallback_counts={f:0 for f in fractions}
        trained_pairs={f:0 for f in fractions}
        parameter_bytes={f:0 for f in fractions}
        train_sample_sizes={}
        for eid in seen_experts:
            train_k=train_routes.get(eid,[])
            eval_k=eval_routes.get(eid,[])
            if not eval_k:continue
            indices=[lookup[k] for k in eval_k]
            weights=np.array([dict(routes[k])[eid] for k in eval_k],np.float64)
            xq=np.stack([hidden_state[k] for k in eval_k])
            matrix=read_rows(reader,eid,np.arange(intermediate))
            cq=np.stack([channels(matrix,x,width) for x in xq])
            yt=np.sum(cq,axis=1,dtype=np.float64)
            full[indices]+=weights[:,None]*yt
            train_sample_sizes[eid]=len(train_k)
            if len(train_k)<min_train_examples:
                del matrix
                for f in fractions:
                    for m in METHODS:
                        guesses[f][m][indices]+=weights[:,None]*yt
                    fallback_counts[f]+=len(eval_k)
                    activation_executions[f]+=len(eval_k)*ng
                continue
            xc=np.stack([hidden_state[k] for k in train_k])
            cc=np.stack([channels(matrix,x,width) for x in xc])
            del matrix
            for f in fractions:
                if not 0<f<=1:raise ValueError('invalid sparsity fraction')
                fit=fit_one_expert(xc,cc,xq,cq,f)
                for m in METHODS:
                    guesses[f][m][indices]+=weights[:,None]*fit[m]
                parameter_bytes[f]+=4*fit.get('stored_router_floats_per_expert',fit.get('trained_coefficients',0)+fit.get('trained_feature_floats',0))
                chosen_count=max(1,math.ceil(ng*f))
                activation_executions[f]+=len(eval_k)*chosen_count
                trained_pairs[f]+=len(eval_k)
                if not np.all(fit['dynamic_mask'].sum(axis=1)==chosen_count):
                    raise AssertionError('wrong number of routed channels')
                # Spot-check the generalized Tupper layout; this is an encoding check,
                # not a test of compression or quality.
                bitmap=np.zeros((ng,64),dtype=np.uint8)
                bitmap[:,eid]=fit['dynamic_mask'][0].astype(np.uint8)
                packed=tupper_encode(bitmap)
                if not np.array_equal(bitmap,tupper_decode(packed['k'],ng,64)):
                    raise AssertionError('Tupper mask did not roundtrip')
                del fit
            del cq,cc
        for f in fractions:
            denval=max(float(np.sum(full[:nval]**2)),1e-25)
            validations={m:float(math.sqrt(np.sum((guesses[f][m][:nval]-full[:nval])**2)/denval)) for m in METHODS}
            locked=min(METHODS,key=lambda m:(validations[m],METHODS.index(m)))
            report={'validation_errors':validations,'method_selected_by_validation':locked,
                    'holdout_methods':{m:_metric(test,full[nval:],guesses[f][m][nval:],bootstrap_draws=bootstrap_draws) for m in METHODS},
                    'train_samples_per_expert_minmax':([min(train_sample_sizes.values()),max(train_sample_sizes.values())] if train_sample_sizes else None),
                    'distinct_routed_experts':len(seen_experts),
                    'validation_routed_pair_count':sum(len(routes[k]) for k in val),
                    'holdout_routed_pair_count':sum(len(routes[k]) for k in test),
                    'full_channel_fallback_pair_count':fallback_counts[f],
                    'trained_pair_count_val_plus_test':trained_pairs[f],
                    'approx_router_resident_fp32_bytes':parameter_bytes[f],
                    'idealized_channel_compute_fraction_excluding_gate_and_corrector':activation_executions[f]/max(ng*sum(len(routes[k]) for k in eval_keys),1),
                    'tupper_grid':[ng,64], 'tupper_roundtrip':True}
            report['locked_holdout']=report['holdout_methods'][locked]
            results.setdefault(str(width),{})[str(f)]=report
    return {'schema':SCHEMA,'status':'EVALUATED_NOT_PROMOTABLE',
            'source':'actual independent textual prompts, actual quantized AF pre-FFN h2 and macro TopK, BF16 expert checkpoint',
            'model_id':'DeepSeek-V2-Lite','layer':layer,'hidden':hidden,'intermediate':intermediate,
            'checkpoint_index_sha256':index_sha,'dataset_manifest_sha256':_sha256(Path(suite)/'runs.json'),
            'prompt_split_counts':prompt_counts,'position_split_counts':{s:len(v) for s,v in splits.items()},
            'prompt_count':len(sources), 'seen_routed_expert_count':len(seen_experts),
            'capture_evidence_hashes':{s['prompt_id']:s['capture_sha256'] for s in sources},
            'route_evidence_hashes':{s['prompt_id']:s['routing_sha256'] for s in sources},
            'heldout_method_selection':'validation_only','prompt_bootstrap_draws':bootstrap_draws,
            'real_checkpoint_full_hash_verified':False,
            'input_source_mismatch':'quantized AF h2 with BF16 expert weights',
            'native_qng64_parity_tested':False,'actual_task_accuracy_tested':False,
            'runtime_speed_measured':False,'raw_hidden_uploaded':False,'production_touched':False,
            'automatic_live_promotion':False,'results':results}

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--checkpoint-index',required=True,type=Path)
    a.add_argument('--prompt-suite',required=True,type=Path)
    a.add_argument('--layer',type=int,default=1)
    a.add_argument('--hidden',type=int,default=2048)
    a.add_argument('--intermediate',type=int,default=1408)
    a.add_argument('--widths',type=int,nargs='+',default=[16,64])
    a.add_argument('--fractions',type=float,nargs='+',default=[.25,.5,.75])
    a.add_argument('--output',required=True,type=Path)
    ns=a.parse_args()
    o=evaluate(ns.checkpoint_index,ns.prompt_suite,layer=ns.layer,hidden=ns.hidden,intermediate=ns.intermediate,widths=ns.widths,fractions=ns.fractions)
    ns.output.parent.mkdir(parents=True,exist_ok=True)
    ns.output.write_text(json.dumps(o,sort_keys=True,indent=2)+'\n')
    print('PROMPT_HOLDOUT_RESULT',json.dumps({
        'status':o['status'],'prompt_split_counts':o['prompt_split_counts'],
        'position_split_counts':o['position_split_counts'],
        'results':{w:{f:{'locked':row['method_selected_by_validation'],
               'locked_test_rel_l2':round(row['locked_holdout']['relative_l2'],6),
               'static_test_rel_l2':round(row['holdout_methods']['static']['relative_l2'],6),
               'dynamic_test_rel_l2':round(row['holdout_methods']['dynamic']['relative_l2'],6),
               'dynamic_ridge_test_rel_l2':round(row['holdout_methods']['dynamic_ridge_residual']['relative_l2'],6),
               'fallback_pairs':row['full_channel_fallback_pair_count']} for f,row in fr.items()}
               for w,fr in o['results'].items()}},sort_keys=True))

if __name__=='__main__':main()