#!/usr/bin/env python3
"""P4 causal true Top6/7 gap forecasting over next 16 token positions.

Four-way H-JEPA-inspired ablation: static, flat, shared 2-level, separate
2-level; additional current-risk persistence comparator. Non-JEPA PCA/Ridge
surrogate ONLY. All transforms and threshold learned from 12 training prompts,
lambda/mixture from 12 validation prompts, heldout 12 prompts evaluated once.

No qNg64 policy permissions or native speedup statements can be inferred.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,time
import numpy as np
import hcp_p2_multiscale as p2

SCHEMA="beglin-hcp-p4-112token-future16-hierarchy-v1"
ARMS=("static","persistence","flat","shared_twolevel","separate_twolevel")
LAM=(.01,.1,1.,10.,100.)
ALPHA=(0.,.25,.5,.75,1.)
STRIDE=16
RANK=3

def load(path):
    raw=pathlib.Path(path).read_bytes()
    src=json.loads(raw)
    if src.get("schema")!="beglin-hcp-p4-long-actual-af-risk-dataset-v1" or src.get("status")!="ACTUAL_DEEPSEEK_LONGISH_CONTEXT_RISK_TRACES_ONLY":
        raise ValueError("invalid actual P4 long-input training lineage")
    if src.get("train_val_holdout")!={"train":12,"validation":12,"holdout":12} or src.get("record_count")!=36:
        raise ValueError("P4 prompt-level group split invalid")
    if src.get("production_touched") is not False or src.get("native_precision_changed") is not False or src.get("full_logits_exported") is not False:
        raise ValueError("forbidden native authority or raw logits upload")
    seen=set();data={k:[] for k in ("train","validation","holdout")}
    for case in src["sequences"]:
        pid=case["prompt_id"];split=case["split"]
        if pid in seen or split not in data:raise ValueError("unknown or duplicate independent prompt")
        seen.add(pid)
        rows=case["rows"]
        if not isinstance(rows,list) or len(rows)<96:
            raise ValueError("not a genuine 104-112-token sequence")
        for i in range(len(rows)):
            if i+STRIDE>=len(rows):break
            cur=rows[i]
            if int(cur.get("position",-1))!=i:raise ValueError("noncausal temporal alignment")
            fast=np.asarray(cur["fast"],dtype=float);slow=np.asarray(cur["slow16"],dtype=float)
            y=float(cur["true_future_gap_min"][str(STRIDE)])
            next1=float(rows[i+1]["observed_min_kth_gap"])
            current=float(cur["observed_min_kth_gap"])
            if fast.shape!=(8,) or slow.shape!=(8,) or not np.isfinite(fast).all() or not np.isfinite(slow).all():
                raise ValueError("nonfinite or wrong slow/fast latent source")
            if not all(math.isfinite(v) and 0<=v<=1 for v in (y,next1,current)):
                raise ValueError("not an actual bounded rank-6/7 gap")
            data[split].append({"pid":pid,"pos":i,
                    "fast":fast,"slow":slow,"next1":next1,"next4":y,
                    "current_gap":current})
    if len(seen)!=36 or [len({r["pid"] for r in data[k]}) for k in data]!=[12,12,12]:
        raise ValueError("improper isolated prompt holdout")
    return data,hashlib.sha256(raw).hexdigest()

def pred(train,rows,arm,lam,alpha):
    if arm=="static":
        y=np.array([r["next4"] for r in train])
        return np.full(len(rows),float(y.mean())),8
    if arm=="persistence":
        return np.array([r["current_gap"] for r in rows]),0
    ft,st=p2.data_matrix(train,"fast"),p2.data_matrix(train,"slow")
    fv,sv=p2.data_matrix(rows,"fast"),p2.data_matrix(rows,"slow")
    next1=p2.data_matrix(train,"next1")
    future=p2.data_matrix(train,"next4")
    if arm=="flat":
        features=np.concatenate((ft,st),axis=1)
        normal=p2.Center(features)
        weights=p2.fit_ridge(normal(features),future,lam)
        target=np.concatenate((fv,sv),axis=1)
        return p2.infer_ridge(weights,normal(target)),weights.nbytes+normal.storage()
    # Identical causal feature budget for both. The 2-level upper predictor
    # uses only slow16 at block start and holds its value for 16 token positions.
    standard=p2.Center(np.concatenate((ft,st),axis=0))
    xf=standard(ft);xs=standard(st);vf=standard(fv);vs=standard(sv)
    if arm=="shared_twolevel":
        w=p2.pca(np.concatenate((xf,xs),axis=0),rank=RANK)
        wf=ws=w
    elif arm=="separate_twolevel":
        wf=p2.pca(xf,rank=RANK)
        ws=p2.pca(xs,rank=RANK)
    else:raise ValueError("bad predictive baseline")
    first=p2.fit_ridge(xf@wf,next1,lam)
    upper=p2.fit_ridge(xs@ws,future,lam)
    hold={} ; upper_predictions=[]
    for i,row in enumerate(rows):
        bucket=(row["pid"],row["pos"]//STRIDE)
        if bucket not in hold:
            hold[bucket]=float(p2.infer_ridge(upper,vs[i:i+1]@ws)[0])
        upper_predictions.append(hold[bucket])
    lower=p2.infer_ridge(first,vf@wf)
    output=np.clip(alpha*np.asarray(upper_predictions)+(1.-alpha)*lower,0,1)
    storage=first.nbytes+upper.nbytes+wf.nbytes+ws.nbytes+standard.storage()
    if arm=="shared_twolevel":storage-=ws.nbytes
    return output,int(storage)

def metric(rows,target,estimate):
    x=p2.metric(rows,target,estimate)
    return {k:float(v) if isinstance(v,(float,np.floating)) else v for k,v in x.items()}

def evaluate(source,output):
    data,source_sha=load(source)
    tr,va,ho=[data[k] for k in ("train","validation","holdout")]
    vy=np.array([row["next4"] for row in va])
    hy=np.array([row["next4"] for row in ho])
    fitted={};choices={}
    for arm in ARMS:
        cases=[(1.,1.)] if arm in ("static","persistence") else (
            [(lam,1.) for lam in LAM] if arm=="flat" else
            [(lam,alpha) for lam in LAM for alpha in ALPHA])
        grid=[]
        for lam,alpha in cases:
            values,_=pred(tr,va,arm,lam,alpha)
            loss=float(np.mean((values-vy)**2))
            if not math.isfinite(loss):raise ValueError("validation error not finite")
            grid.append((loss,lam,alpha))
        loss,lam,alpha=min(grid)
        choices[arm]={"lambda":lam,"upper_alpha":alpha,
                      "validation_rmse":math.sqrt(loss)}
    elapsed={};bytes_used={};final={};preds={}
    for arm in ARMS:
        c=choices[arm]
        start=time.perf_counter_ns()
        estimate,nbytes=pred(tr,ho,arm,c["lambda"],c["upper_alpha"])
        elapsed[arm]=(time.perf_counter_ns()-start)/1e6
        preds[arm]=estimate;bytes_used[arm]=int(nbytes)
        final[arm]=metric(ho,hy,estimate)
    paired={arm:p2.paired_bootstrap(ho,hy,preds["flat"],preds[arm],draws=1000)
            for arm in ("shared_twolevel","separate_twolevel")}
    deeper=[]
    for arm in paired:
        gain=paired[arm]["improvement_vs_flat_prompt_rmse"]
        ci=paired[arm]["paired_prompt_bootstrap_95ci"]
        passed=(gain>=.01 and ci[0]>0)
        deeper.append({"arm":arm,"paired_improvement":gain,"prompt_bootstrap_95ci":ci,
                       ">=1percent_and_CI_positive":passed})
    result={
       "schema":SCHEMA,"status":"REAL_MEDIUM_CONTEXT_RISK_PREDICTION_ONLY",
       "source_sha256":source_sha,
       "source":"REAL_DEEPSEEK_112TOKEN_AF_26LAYER_TOP6_TOP7",
       "target":"min future 16-token true Kth/(K+1) relative gap",
       "train_prompts":12,"validation_prompts":12,"holdout_prompts":12,
       "supervised_positions":{k:len(data[k]) for k in data},
       "training_learns_no_holdout_targets":True,
       "validation_hyperparameters":choices,
       "heldout":final,"paired_against_flat":paired,
       "predictive_depth_gate":deeper,
       "meaningful_hierarchy_gain_supported":any(row[">=1percent_and_CI_positive"] for row in deeper),
       "estimated_parameter_bytes":bytes_used,
       "batch_training_and_prediction_cpu_ms":elapsed,
       "genuine_JEPA_training_performed":False,
       "native_qng64_quality_labels_available":False,
       "full_model_speedup_measured":False,
       "production_touched":False,"native_precision_policy_change_allowed":False,
       "automatic_live_promotion":False
    }
    output=pathlib.Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("P4_TRUE_MEDIUM_CONTEXT_MULTIHORIZON_FORECAST",json.dumps({
        "source_sha256":source_sha,
        "positions":result["supervised_positions"],
        "prompt_macro_rmse":{k:round(v["prompt_macro_rmse"],8) for k,v in final.items()},
        "paired_against_flat":paired,
        "depth_gate":deeper,
        "native_policy_change_allowed":False},sort_keys=True))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--features",type=pathlib.Path,required=True)
    p.add_argument("--output",type=pathlib.Path,required=True)
    a=p.parse_args()
    evaluate(a.features,a.output)
