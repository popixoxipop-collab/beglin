#!/usr/bin/env python3
"""Prompt-heldout true rank6/rank7 next-horizon risk forecasting ablation.

5 fixed arms: static prior, observed-margin persistence, flat Ridge, two-level
shared-PCA, and two-level separate-PCA. Risk threshold computed using TRAIN
only; validation chooses lambda/slow mixing; holdout never used for training.
No precision policy change, no native task/throughput claims.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib
import numpy as np
import hcp_p2_multiscale as p2

SCHEMA="beglin-hcp-p3-real-topk-boundary-forecast-v1"
ARMS=("static","persistence","flat","shared_twolevel","separate_twolevel")
SPLITS=("train","validation","holdout")
LAMS=(.01,.1,1.,10.)
ALPHAS=(0.,.25,.5,.75,1.)

def load(source):
    raw=pathlib.Path(source).read_bytes();d=json.loads(raw)
    if d.get("schema")!="beglin-hcp-p3-true-router-boundary-features-v1" or d.get("status")!="ACTUAL_TOP6_TOP7_RISK_PROXY_ONLY":
        raise ValueError("not genuine full-softmax boundary dataset")
    if d.get("true_rank7_measured") is not True or d.get("target_is_certified_precision_risk") is not False or d.get("precision_rebinding_allowed") is not False or d.get("production_touched") is not False:
        raise ValueError("invalid router source/precision safety contract")
    if d.get("prompt_counts")!={"train":72,"validation":24,"holdout":24}:
        raise ValueError("bad prompt-heldout cardinality")
    seen=set();rows={k:[] for k in SPLITS}
    for doc in d["prompts"]:
        pid=doc.get("id");split=doc.get("split")
        if pid in seen or not isinstance(pid,str) or split not in SPLITS:
            raise ValueError("bad independent prompt identity")
        seen.add(pid)
        seq=doc.get("rows")
        if not isinstance(seq,list) or len(seq)<3:raise ValueError("too few sequence positions")
        for i,s in enumerate(seq):
            f=np.asarray(s.get("fast"),dtype=np.float64)
            z=np.asarray(s.get("slow"),dtype=np.float64)
            n1=float(s.get("target_next1"))
            n4=float(s.get("target_next4"))
            ob=float(s.get("observed_margin"))
            if (int(s.get("position",-1))!=i or f.shape!=(8,) or z.shape!=(8,) or
                not np.isfinite(f).all() or not np.isfinite(z).all() or
                any(not 0<=x<=1 for x in (n1,n4,ob)) or
                abs(f[6]-ob)>1e-7 or n4>n1+1e-9):
                raise ValueError("invalid causality, target, or rank gap feature")
            rows[split].append({"pid":pid,"pos":i,"fast":f,"slow":z,
                "next1":n1,"next4":n4,"observed_margin":ob})
    if len(seen)!=120 or [len(set(r["pid"] for r in rows[k])) for k in SPLITS]!=[72,24,24]:
        raise ValueError("missing group identities")
    return rows,hashlib.sha256(raw).hexdigest()

def auc(truth,score):
    labels=np.asarray(truth,dtype=bool);s=np.asarray(score,dtype=float)
    p=int(labels.sum());n=len(labels)-p
    if not p or not n:return None
    order=np.argsort(s,kind="stable")
    ranks=np.zeros(len(s),dtype=float)
    i=0
    while i<len(s):
        j=i+1
        while j<len(s) and s[order[j]]==s[order[i]]:
            j+=1
        ranks[order[i:j]]=(i+1+j)/2
        i=j
    return float((ranks[labels].sum()-p*(p+1)/2)/(p*n))

def average_precision(truth,score):
    labels=np.asarray(truth,dtype=bool)
    if not labels.any():return None
    order=np.argsort(-np.asarray(score),kind="stable")
    t=labels[order].astype(int)
    p=np.cumsum(t);r=np.arange(1,len(t)+1)
    return float(np.dot((p/r),t)/max(1,t.sum()))

def metric(rows,truth,pred,threshold):
    truth=np.asarray(truth);pred=np.asarray(pred)
    if len(rows)!=len(truth) or not len(rows) or not np.isfinite(pred).all():
        raise ValueError("invalid test metric")
    residual=pred-truth;by={}
    for i,r in enumerate(rows):
        a=by.setdefault(r["pid"],[0.,0])
        a[0]+=float(residual[i]**2);a[1]+=1
    labels=truth<=threshold
    return {"token_rmse":float(np.sqrt(np.mean(residual**2))),
        "mae":float(np.mean(np.abs(residual))),
        "prompt_macro_rmse":float(np.sqrt(np.mean([u[0]/u[1] for u in by.values()]))),
        "near_tie_auroc":auc(labels,-pred),
        "near_tie_average_precision":average_precision(labels,-pred),
        "near_tie_event_rate":float(np.mean(labels)),
        "prompt_count":len(by),"token_position_count":len(rows)}

def train_predict(train,val,arm,lam,alpha):
    if arm=="persistence":
        x=np.array([r["observed_margin"] for r in val],dtype=np.float64)
        return x,0
    return p2.timed_predictor(train,val,arm,lam,alpha,rank=2)

def evaluate(source,output):
    data,sha=load(source);train,val,hold=[data[k] for k in SPLITS]
    ty=np.array([r["next4"] for r in train])
    vy=np.array([r["next4"] for r in val])
    hy=np.array([r["next4"] for r in hold])
    threshold=float(np.quantile(ty,.20))
    chosen={};estim={}
    for arm in ARMS:
        if arm in ("static","persistence"):
            combos=[(1.,1.)]
        elif arm=="flat":combos=[(lam,1.) for lam in LAMS]
        else:combos=[(lam,a) for lam in LAMS for a in ALPHAS]
        candidates=[]
        for lam,a in combos:
            pred,_=train_predict(train,val,arm,lam,a)
            mse=float(np.mean((pred-vy)**2))
            if not math.isfinite(mse):raise ValueError("bad validation risk metric")
            candidates.append((mse,lam,a))
        loss,lam,a=min(candidates)
        chosen[arm]={"lam":lam,"slow_alpha":a,"validation_rmse":math.sqrt(loss)}
    predictions={};quality={};parameter_bytes={}
    for arm in ARMS:
        cfg=chosen[arm]
        pred,b=train_predict(train,hold,arm,cfg["lam"],cfg["slow_alpha"])
        predictions[arm]=pred
        quality[arm]=metric(hold,hy,pred,threshold)
        parameter_bytes[arm]=int(b)
    improvements={
      key:p2.paired_bootstrap(hold,hy,predictions["flat"],predictions[key])
      for key in ("persistence","shared_twolevel","separate_twolevel")}
    choices=[]
    for arm in ("shared_twolevel","separate_twolevel"):
        improvement=improvements[arm]
        ci=improvement["paired_prompt_bootstrap_95ci"]
        qualifies=(ci[0]>0 and improvement["improvement_vs_flat_prompt_rmse"]>=.01
                   and quality[arm]["near_tie_auroc"] is not None
                   and quality["flat"]["near_tie_auroc"] is not None
                   and quality[arm]["near_tie_auroc"]>=quality["flat"]["near_tie_auroc"])
        choices.append({"arm":arm,"depth_screen_pass":bool(qualifies),
                        "paired_prompt_95ci":ci,
                        "improvement":improvement["improvement_vs_flat_prompt_rmse"]})
    report={"schema":SCHEMA,"status":"TRUE_ROUTER_RISK_FORECAST_EVALUATED_RESEARCH_ONLY",
        "source_feature_sha256":sha,
        "true_K_vs_Kplus1_measured":True,
        "full_64_expert_probabilities_before_renormalization":True,
        "boundary_label":"min across 26 layers (p6-p7)/p6 next 1..4 tokens",
        "train_only_near_tie_quantile":.20,
        "near_tie_relative_gap_threshold":threshold,
        "train_prompts":72,"validation_prompts":24,"holdout_prompts":24,
        "supervised_token_positions":{"train":len(train),"validation":len(val),"holdout":len(hold)},
        "per_arm_validation_choice":chosen,"heldout":quality,
        "paired_vs_flat":improvements,"model_parameter_bytes_approx":parameter_bytes,
        "hierarchy_qualified_for_more_research":any(z["depth_screen_pass"] for z in choices),
        "depth_screen":choices,
        "certified_precision_trigger_training_labels":False,
        "context_specific_qng64_acceptance_verified":False,
        "same_precision_qng64_token_accuracy_tested":False,
        "native_policy_rebind_allowed":False,
        "production_touched":False,"auto_live_promotion":False}
    output=pathlib.Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("P3_TRUE_TOPK_BOUNDARY_ABLATION",json.dumps({"status":report["status"],
       "sample_counts":report["supervised_token_positions"],
       "threshold":threshold,
       "near_tie_event_rate":quality["flat"]["near_tie_event_rate"],
       "heldout_macro_rmse":{k:round(v["prompt_macro_rmse"],8) for k,v in quality.items()},
       "heldout_auroc":{k:v["near_tie_auroc"] for k,v in quality.items()},
       "paired_vs_flat":improvements,
       "depth_screen":choices,
       "native_policy_rebind_allowed":False},sort_keys=True))
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--features",type=pathlib.Path,required=True)
    p.add_argument("--output",type=pathlib.Path,required=True)
    a=p.parse_args();evaluate(a.features,a.output)
