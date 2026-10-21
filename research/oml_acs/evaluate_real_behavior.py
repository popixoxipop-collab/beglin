#!/usr/bin/env python3
"""Real SST-2/CARER train->dev cross-model ACS: never a final/holdout verdict."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from acs_core import unit

ROOT=Path.home()/".local/share/beglin/oml_acs/20261021"
DATA_SHA="b99fb5d7e8c909e6821f555ac590b36ff52393b9d4e7a01828ee936c615bd274"
ANCHOR_SHA="47d5ddcb37b81212e3f43a34441662062b42a038d4cc606d9d7081c31131f83b"
MODELS={
"qwen":(1536,"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",ROOT/"qwen25_300_anchor_projector_v1.npz"),
"mistral":(4096,"c170c708c41dac9275d15a8fff4eca08d52bab71",ROOT/"mistral7b_layer16_300_projector_bf16.npz"),
"phi":(3072,"cfbefacb99257ffa30c83adab238a50856ac3083",ROOT/"phi4mini_layer16_300_projector_bf16.npz"),
}
AXES=("sst2_positive_vs_negative","carer_joy_vs_anger")

def digest(b):return hashlib.sha256(b).hexdigest()

def auc(pos,neg):
    p,n=np.asarray(pos,dtype=float),np.asarray(neg,dtype=float)
    if p.ndim!=1 or n.ndim!=1 or not len(p) or not len(n) or not np.isfinite(p).all() or not np.isfinite(n).all():
        raise ValueError("nonempty finite 1D class score vectors required")
    delta=p[:,None]-n[None,:]
    return float((np.count_nonzero(delta>0)+.5*np.count_nonzero(delta==0))/delta.size)

def bootstrap(pos,neg,seed=42,n=999):
    rng=np.random.default_rng(seed);p=np.asarray(pos);q=np.asarray(neg)
    return [float(x) for x in np.quantile([auc(p[rng.integers(len(p),size=len(p))],
                                               q[rng.integers(len(q),size=len(q))]) for _ in range(n)],[.025,.975])]

def load(which):
    if which not in MODELS:raise ValueError("unknown model")
    hidden,rev,projfile=MODELS[which]
    arrfile=ROOT/f"behavior_{which}_600_bf16.npz"
    for path in (arrfile,projfile):
        if not path.is_file() or (path.stat().st_mode&0o777)!=0o600:
            raise RuntimeError("private file missing/unsecured")
    with np.load(arrfile,allow_pickle=False) as a:
        if a["behavior_source_sha256"].item()!=DATA_SHA or a["anchor_source_sha256"].item()!=ANCHOR_SHA or a["model_revision"].item()!=rev:
            raise RuntimeError("model/source revision mismatched")
        X=a["activations"].astype(np.float64)
        ids=a["text_sha256"].tolist();axis=a["axis"].tolist()
        split=a["split"].tolist();labels=a["labels"].astype(int)
    with np.load(projfile,allow_pickle=False) as p:
        if p["source_raw_sha256"].item()!=ANCHOR_SHA or p["ahat"].shape!=(300,hidden):
            raise RuntimeError("projector revision or dimensions mismatched")
        mu=p["mu"].copy();A=p["ahat"].copy()
    if X.shape!=(600,hidden) or not np.isfinite(X).all() or not np.isfinite(A).all():
        raise RuntimeError("invalid model activation matrix")
    return dict(X=X,ids=ids,axis=np.asarray(axis),split=np.asarray(split),y=labels,mu=mu,A=A,rev=rev)

def select(x,axis,split,label):
    return (x["axis"]==axis)&(x["split"]==split)&(x["y"]==label)

def source_axis(x,axis):
    pos=x["X"][select(x,axis,"behavior_train",1)]
    neg=x["X"][select(x,axis,"behavior_train",0)]
    if len(pos)!=100 or len(neg)!=100:raise RuntimeError("not 100 training samples per class")
    return unit(pos.mean(axis=0)-neg.mean(axis=0))

def evaluate(source,target,axis):
    if source["ids"]!=target["ids"] or not np.array_equal(source["y"],target["y"]) or not np.array_equal(source["axis"],target["axis"]) or not np.array_equal(source["split"],target["split"]):
        raise RuntimeError("source/target row identity, split or label mismatch")
    c=unit(source["A"]@source_axis(source,axis))
    where=np.where((target["axis"]==axis)&(target["split"]=="behavior_dev"))[0]
    if len(where)!=100:raise RuntimeError("missing target development rows")
    h=target["X"][where]-target["mu"]
    norms=np.linalg.norm(h,axis=1)
    if np.any(norms<=1e-10):raise RuntimeError("degenerate target activations")
    projected=(h/norms[:,None])@target["A"].T
    y=target["y"][where];is_pos=y==1;is_neg=y==0
    if is_pos.sum()!=50 or is_neg.sum()!=50:raise RuntimeError("imbalanced dev labels")
    scores=projected@c
    observed=auc(scores[is_pos],scores[is_neg])
    rng=np.random.default_rng(42)
    shuffled=[];randoms=[]
    for _ in range(199):
        s=projected[:,rng.permutation(300)]@c
        shuffled.append(auc(s[is_pos],s[is_neg]))
        s=projected@unit(rng.normal(size=300))
        randoms.append(auc(s[is_pos],s[is_neg]))
    def ctrl(values):
        return dict(mean=float(np.mean(values)),p95=float(np.quantile(values,.95)),
                    empirical_p_ge_observed=float((1+sum(v>=observed for v in values))/(len(values)+1)),n=len(values))
    native_target=source_axis(target,axis)  # target TRAIN-only supervised comparison
    native_scores=target["X"][where]@native_target
    return dict(axis=axis,source_train_per_class=100,target_dev_per_class=50,
                target_dev_label_used_to_fit=False,source_polarity_fixed=True,
                transferred_auroc=observed,
                bootstrap_95ci=bootstrap(scores[is_pos],scores[is_neg]),
                shuffle_anchor=ctrl(shuffled),random_axis=ctrl(randoms),
                target_native_trained_baseline_auroc=auc(native_scores[is_pos],native_scores[is_neg]))

def run(source_name,target_name):
    if source_name==target_name:raise ValueError("different models required")
    source,target=load(source_name),load(target_name)
    out=dict(schema="oml-acs-crossmodel-real-behavior-dev/1",status="DEV_ONLY_NO_FINAL_VERDICT",
             source=source_name,target=target_name,source_revision=source["rev"],target_revision=target["rev"],
             axes=[evaluate(source,target,axis) for axis in AXES],
             rows_same_order=source["ids"]==target["ids"],behavior_sha256=DATA_SHA,anchor_sha256=ANCHOR_SHA,
             dev_labels_not_used_for_axis_construction=True,final_holdout_accessed=False,
             V6_24_attested_excluded=False,paper_exact_anchor_ids=False,production_touched=False)
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",choices=sorted(MODELS),required=True)
    p.add_argument("--target",choices=sorted(MODELS),required=True)
    args=p.parse_args()
    answer=run(args.source,args.target)
    os.umask(0o077)
    path=ROOT/f"realbehavior_300anchors_{args.source}_to_{args.target}_dev_v1.json"
    with path.open("x",encoding="utf-8") as f:
        json.dump(answer,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
    print("REAL_ACS_DEV_EVALUATION",json.dumps(answer,sort_keys=True),flush=True)
