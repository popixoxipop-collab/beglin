#!/usr/bin/env python3
"""H-JEPA-inspired separate slow/fast predictive representations, OFFLINE ONLY.

Not JEPA training: linear PCA encoders and ridge prediction of actual DeepSeek
future *top-six selected-score entropy*. This surrogate is NOT a certified
near-tie margin, runtime qNg64 accuracy, or authority to change any policy.
Compared arms: static mean, flat ridge, 2-level shared PCA and independent PCA.
All learned transforms fit TRAIN prompts only. Validation tunes hyperparameters;
the 24 HOLDOUT prompts are touched once for final reporting.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,time
import numpy as np

SCHEMA="beglin-hcp-p2-multiscale-predictor-v1"
SPLITS=("train","validation","holdout")
ARMS=("static","flat","shared_twolevel","separate_twolevel")
LAMS=(.01,.1,1.,10.)
ALPHAS=(0.,.25,.5,.75,1.)
RANK=2
STRIDE=4

class P2Error(ValueError):pass

def load(path):
    raw=pathlib.Path(path).read_bytes()
    d=json.loads(raw)
    if d.get("schema")!="beglin-hcp-p2-causal-real-trace-v1" or d.get("source")!="REAL_DEEPSEEK_QUANTIZED_AF_H2_AND_TOP6_ROUTING":
        raise P2Error("not real DeepSeek causal trace")
    if d.get("status")!="PROXY_ONLY_NO_CERTIFIED_RUNTIME_PERMISSION" or d.get("policy_change_allowed") is not False or d.get("production_touched") is not False:
        raise P2Error("invalid trust boundary")
    if d.get("target_is_genuine_topK_boundary_margin") is not False or d.get("input_uses_same_token_post_ffn_logits") is not False:
        raise P2Error("future information leakage or false certification")
    docs=d.get("prompts",[])
    if len(docs)!=120 or d.get("prompt_counts")!={"train":72,"validation":24,"holdout":24}:
        raise P2Error("wrong prompt-level split")
    seen=set();dataset={s:[] for s in SPLITS}
    for doc in docs:
        pid=doc.get("id");split=doc.get("split")
        if not isinstance(pid,str) or pid in seen or split not in SPLITS:raise P2Error("duplicate or invalid prompt")
        seen.add(pid)
        rows=doc.get("rows")
        if not isinstance(rows,list) or len(rows)<3:raise P2Error("too few causal samples")
        for j,r in enumerate(rows):
            f=np.asarray(r.get("fast"),dtype=np.float64)
            sl=np.asarray(r.get("slow"),dtype=np.float64)
            yn=float(r.get("target_next1"))
            ys=float(r.get("target_next4"))
            if int(r.get("position",-1))!=j or f.shape!=(6,) or sl.shape!=(6,) or not np.isfinite(f).all() or not np.isfinite(sl).all() or not(0<=yn<=1 and 0<=ys<=1):
                raise P2Error("invalid timestamp, feature, or target")
            dataset[split].append({"pid":pid,"pos":j,"fast":f,"slow":sl,"next1":yn,"next4":ys})
    if [len(set(r["pid"] for r in dataset[s])) for s in SPLITS]!=[72,24,24]:
        raise P2Error("split group cardinality changed")
    return dataset,hashlib.sha256(raw).hexdigest()

def data_matrix(rows,field):
    return np.stack([r[field] for r in rows]) if field in ("fast","slow") else np.array([r[field] for r in rows])

class Center:
    def __init__(self,x):
        self.mu=x.mean(axis=0)
        self.scale=np.maximum(x.std(axis=0),1e-7)
    def __call__(self,x):return (x-self.mu)/self.scale
    def storage(self):return self.mu.nbytes+self.scale.nbytes

def pca(x,rank=RANK):
    centered=x-x.mean(axis=0)
    _,_,vt=np.linalg.svd(centered,full_matrices=False)
    return vt[:min(rank,vt.shape[0])].T

def fit_ridge(x,y,lam):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if x.ndim!=2 or y.ndim!=1 or len(x)!=len(y) or lam<=0:raise P2Error("invalid ridge")
    a=np.c_[np.ones(len(x)),x]
    penalty=np.eye(a.shape[1])*lam;penalty[0,0]=0
    coeff=np.linalg.solve(a.T@a+penalty,a.T@y)
    return coeff

def infer_ridge(beta,x):
    return np.clip(np.c_[np.ones(len(x)),x]@beta,0.,1.)

def timed_predictor(train,val,method,lam,alpha,rank=RANK):
    ft,st=data_matrix(train,"fast"),data_matrix(train,"slow")
    fv,sv=data_matrix(val,"fast"),data_matrix(val,"slow")
    yn=data_matrix(train,"next1");ys=data_matrix(train,"next4")
    if method=="static":
        return np.full(len(val),float(np.mean(ys))),1
    if method=="flat":
        x=np.concatenate([ft,st],axis=1);v=np.concatenate([fv,sv],axis=1)
        norm=Center(x)
        b=fit_ridge(norm(x),ys,lam)
        return infer_ridge(b,norm(v)),b.nbytes+norm.storage()
    normal=Center(np.concatenate([ft,st]))
    f=normal(ft);s=normal(st);vf=normal(fv);vs=normal(sv)
    if method=="shared_twolevel":
        w=pca(np.concatenate([f,s]),rank)
        wfast=w;wslow=w
    elif method=="separate_twolevel":
        wfast=pca(f,rank);wslow=pca(s,rank)
    else:raise P2Error("unknown arm")
    fast=fit_ridge(f@wfast,yn,lam)
    slow=fit_ridge(s@wslow,ys,lam)
    # Each prompt gets a separate, causal four-step slow waypoint.
    seen={}
    low=[]
    for i,row in enumerate(val):
        pid=row["pid"];block=row["pos"]//STRIDE
        key=(pid,block)
        if key not in seen:
            seen[key]=infer_ridge(slow,(vs[i:i+1]@wslow))[0]
        low.append(seen[key])
    upper=np.asarray(low)
    lower=infer_ridge(fast,vf@wfast)
    pred=np.clip(alpha*upper+(1-alpha)*lower,0.,1.)
    size=int(fast.nbytes+slow.nbytes+wfast.nbytes+wslow.nbytes+normal.storage())
    if method=="shared_twolevel":size-=int(wslow.nbytes)
    return pred,size

def metric(rows,truth,pred):
    e=pred-truth
    by={}
    for idx,row in enumerate(rows):
        key=row["pid"]
        sums=by.setdefault(key,[0.,0])
        sums[0]+=float(e[idx]*e[idx]);sums[1]+=1
    if len(by)<3:raise P2Error("need at least 3 heldout prompts")
    prompt_losses=np.array([v[0]/v[1] for k,v in sorted(by.items())])
    return {"rmse":float(np.sqrt(np.mean(e*e))),
            "mae":float(np.mean(np.abs(e))),
            "prompt_macro_rmse":float(np.sqrt(np.mean(prompt_losses))),
            "prompt_count":len(by),
            "sample_count":len(rows)}

def paired_bootstrap(rows,y,flat,other,*,draws=500):
    per={}
    for i,r in enumerate(rows):
        a,b=per.setdefault(r["pid"],[[],[]])
        a.append(float((flat[i]-y[i])**2))
        b.append(float((other[i]-y[i])**2))
    groups=sorted(per);aa=np.array([np.mean(per[g][0]) for g in groups]);bb=np.array([np.mean(per[g][1]) for g in groups])
    if len(groups)<3:raise P2Error("insufficient independent holdout prompts")
    rng=np.random.default_rng(20261014)
    improvements=[]
    for _ in range(draws):
        idx=rng.integers(0,len(groups),len(groups))
        a=float(np.mean(aa[idx]));b=float(np.mean(bb[idx]))
        if a<=1e-18:raise P2Error("zero flat error invalid paired ratio")
        improvements.append(1.-math.sqrt(b/a))
    return {"improvement_vs_flat_prompt_rmse":float(1.-math.sqrt(np.mean(bb)/np.mean(aa))),
            "paired_prompt_bootstrap_95ci":list(map(float,np.quantile(improvements,[.025,.975]))),
            "bootstrap_draws":draws}

def evaluate(path,out,*,rank=RANK):
    data,sha=load(path)
    tr,va,ho=[data[s] for s in SPLITS]
    yv=data_matrix(va,"next4");yh=data_matrix(ho,"next4")
    chosen={};validation={}
    for arm in ARMS:
        experiments=[]
        candidates=[(1.,1.)] if arm=="static" else [(lam,a) for lam in LAMS for a in ((1.,) if arm=="flat" else ALPHAS)]
        for lam,alpha in candidates:
            v,_=timed_predictor(tr,va,arm,lam,alpha,rank=rank)
            err=float(np.mean((v-yv)**2))
            experiments.append((err,lam,alpha))
        score,lam,alpha=min(experiments,key=lambda t:(t[0],t[1],t[2]))
        chosen[arm]={"ridge_lambda":float(lam),"slow_alpha":float(alpha),
                     "validation_proxy_rmse":float(math.sqrt(score))}
        validation[arm]=float(math.sqrt(score))
    predictions={};quality={};sizes={};timings={}
    for arm in ARMS:
        hp=chosen[arm]
        t=time.perf_counter_ns()
        values,size=timed_predictor(tr,ho,arm,hp["ridge_lambda"],hp["slow_alpha"],rank=rank)
        elapsed=(time.perf_counter_ns()-t)/1e6
        predictions[arm]=values
        quality[arm]=metric(ho,yh,values)
        timings[arm]={"whole_holdout_batch_python_ms":float(elapsed),
                      "not_native_kernel_latency":True}
        sizes[arm]=int(size)
    comparisons={arm:paired_bootstrap(ho,yh,predictions["flat"],predictions[arm])
                 for arm in ("shared_twolevel","separate_twolevel")}
    static=quality["static"]["prompt_macro_rmse"];flat=quality["flat"]["prompt_macro_rmse"]
    result={"schema":SCHEMA,"status":"REAL_SURROGATE_EVALUATED_NATIVE_POLICY_BLOCKED",
            "source_feature_sha256":sha,"source":"REAL_QUANTIZED_AF_H2_TOP6_ROUTING",
            "train_prompt_count":72,"validation_prompt_count":24,"holdout_prompt_count":24,
            "train_token_positions":len(tr),"validation_token_positions":len(va),
            "holdout_token_positions":len(ho),
            "source_proxy_is_certified_precision_risk":False,
            "source_target":"next-4-selected-top6-score-entropy",
            "temporal_prediction":True,"slow_stride":STRIDE,
            "slow_latent_rank":rank,"fast_latent_rank":rank,
            "trained_method":"PCA and supervised Ridge; NOT H-JEPA joint training",
            "validation_hyperparameter_selection":chosen,
            "final_heldout":quality,"paired_improvement_vs_flat":comparisons,
            "estimated_predictor_parameter_bytes":sizes,
            "offline_python_training_plus_prediction_ms":timings,
            "full_model_latency_measured":False,
            "exact_context_certificates_verified":False,
            "native_policy_rebinding_allowed":False,
            "production_touched":False,"automatic_live_promotion":False}
    out=pathlib.Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("P2_REAL_SURROGATE_ABLATION",json.dumps({
        "status":result["status"],"prompts":[72,24,24],
        "heldout_proxy_rmse":{k:round(v["prompt_macro_rmse"],7) for k,v in quality.items()},
        "paired_vs_flat":comparisons,
        "native_policy_rebinding_allowed":False},sort_keys=True))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--features",type=pathlib.Path,required=True)
    p.add_argument("--output",type=pathlib.Path,required=True)
    a=p.parse_args()
    evaluate(a.features,a.output)
