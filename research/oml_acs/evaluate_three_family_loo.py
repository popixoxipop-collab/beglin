#!/usr/bin/env python3
"""Three-family leave-one-out ACS development pilot (not a final study).

Uniform raw average of two SOURCE canonical behavioral axes (paper Eq. 4),
then unit-normalized ONLY for direction scoring. No target-dev label fitting.
The third family may not be substituted silently: the actual participants
are Qwen2.5-1.5B, Mistral-7B, Phi-4-mini (Mistral is a prereg deviation).
"""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np
from acs_core import unit
from evaluate_real_behavior import ROOT,AXES,MODELS,load,source_axis,auc,bootstrap

PARTICIPANTS=("qwen","mistral","phi")

def raw_mean(a,b):
    x=np.asarray(a,dtype=float)
    y=np.asarray(b,dtype=float)
    if x.ndim!=1 or x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("equal finite canonical axis dimensions required")
    return 0.5*(x+y)

def dev_project(target,axis):
    idx=np.where((target["axis"]==axis)&(target["split"]=="behavior_dev"))[0]
    if len(idx)!=100:raise RuntimeError("100 development items required per axis")
    h=target["X"][idx]-target["mu"][None,:]
    norms=np.linalg.norm(h,axis=1)
    if not np.isfinite(h).all() or (norms<=1e-10).any():
        raise RuntimeError("nonfinite/degenerate target dev features")
    projected=(h/norms[:,None])@target["A"].T
    y=target["y"][idx]
    if (y==1).sum()!=50 or (y==0).sum()!=50:raise RuntimeError("50+50 dev class balance required")
    return projected,y,idx

def evaluate_loo(source_a,source_b,target,axis):
    ids=target["ids"]
    for src in (source_a,source_b):
        if src["ids"]!=ids or not np.array_equal(src["y"],target["y"]) or not np.array_equal(src["axis"],target["axis"]) or not np.array_equal(src["split"],target["split"]):
            raise RuntimeError("three-family row order/label contract mismatch")
    p1=source_a["A"]@source_axis(source_a,axis)
    p2=source_b["A"]@source_axis(source_b,axis)
    c_raw=raw_mean(p1,p2)
    c=unit(c_raw)
    pt,y,idx=dev_project(target,axis)
    scores=pt@c
    pos=y==1;neg=y==0
    val=auc(scores[pos],scores[neg])
    individual=[auc((pt@unit(u))[pos],(pt@unit(u))[neg]) for u in (p1,p2)]
    rng=np.random.default_rng(42)
    shuffled=[]
    randoms=[]
    for _ in range(199):
        s=pt[:,rng.permutation(300)]@c
        shuffled.append(auc(s[pos],s[neg]))
        s=pt@unit(rng.normal(size=300))
        randoms.append(auc(s[pos],s[neg]))
    def baseline(q):
        return dict(mean=float(np.mean(q)),p95=float(np.quantile(q,.95)),
                    empirical_p_ge_observed=float((sum(x>=val for x in q)+1)/(len(q)+1)),n=len(q))
    native=source_axis(target,axis)
    ns=target["X"][idx]@native
    return dict(axis=axis,target_dev_pos=50,target_dev_neg=50,
                source_a_only_AUROC=individual[0],source_b_only_AUROC=individual[1],
                uniform_two_source_AUROC=val,bootstrap_95ci=bootstrap(scores[pos],scores[neg]),
                shuffled_anchor_null=baseline(shuffled),random_canonical_null=baseline(randoms),
                target_native_train_only_baseline_AUROC=auc(ns[pos],ns[neg]),
                literal_eq4_raw_axis_norm=float(np.linalg.norm(c_raw)),
                normalized_axis_used_for_detection=True,labels_not_used_to_fit_target_direction=True)

def run(target_name):
    if target_name not in PARTICIPANTS:raise ValueError("unknown target")
    other=[a for a in PARTICIPANTS if a!=target_name]
    a,b,t=load(other[0]),load(other[1]),load(target_name)
    return dict(schema="oml-acs-three-family-dev-loo/1",
        status="DEV_ONLY_NO_PREREG_FINAL_VERDICT",
        model_families=list(PARTICIPANTS),
        source_families=other,target_family=target_name,
        axes=[evaluate_loo(a,b,t,axis) for axis in AXES],
        source_average="literal_0.5*(projected_source_axis_A+projected_source_axis_B)",
        anchors="same 300 revision-matched development prompts",
        anchor_source_identical=True,
        independent_final_holdout_used=False,V6_consumed_24_not_attested_excluded=True,
        prereg_model_deviation="Mistral7B substitutes for inaccessible gated Llama3.2-3B",
        randomization_null_count=199,
        original_HELM_row_exact_match=False,
        production_touched=False)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--target",required=True,choices=PARTICIPANTS)
    args=ap.parse_args()
    result=run(args.target)
    os.umask(0o077)
    path=ROOT/f"realbehavior_300anchors_threefamily_target_{args.target}_dev_v1.json"
    with path.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True)
        f.write("\n");f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
    print("THREE_FAMILY_LOO_DEV_RECEIPT",json.dumps(result,sort_keys=True),flush=True)
