#!/usr/bin/env python3
"""C1: 5-fold source-only OOF reliability vs paired co-error weighted ACS axes.

Only SST-2 and CARER real training/validation rows. Target-dev labels are
read exclusively by evaluator, never weight selection or source-axis fitting.
This is already-seen development data, NOT blind/final hypothesis validation.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
import hashlib
import json
from pathlib import Path
import argparse
import numpy as np
from acs_core import unit,canonical_mean_raw,weights_from_source_errors
from evaluate_real_behavior import ROOT,AXES,MODELS,load,source_axis,auc

PARTICIPANTS=("qwen","mistral","phi")
SOURCE_SHA="b99fb5d7e8c909e6821f555ac590b36ff52393b9d4e7a01828ee936c615bd274"
ANCHOR_SHA="47d5ddcb37b81212e3f43a34441662062b42a038d4cc606d9d7081c31131f83b"
EXPECTED_ARCHIVES={
 "qwen":("169ce4c3846ca64ac4f05c2d86535f9377c5e2545e5fd2eeeb0fb42813f727cd",
         "664f24c3aed82e9dcc15e895d695fcdfc90215dd6ba17e2d708421bb4f0c9722"),
 "mistral":("14b9303bae9be5e721561b8223eba04185fbc89ae648d7b0134483e787fdaa90",
            "ec38c4068528b73dcd3d32407e6a57a6183cdebc8ead8a758210f3edc820ee24"),
 "phi":("47c4a7446de59633a8136133be4479b1af915d4de958239e0c496c80ea7849ad",
        "2a2b645c7dcae69a2203c6b3554f7f17bbc70fb7eaee54d1c0207d031d3c7fa1"),
}
FOLDS=5
SEED=42
RIDGE=1e-6
BOOTSTRAPS=999
GAIN_THRESHOLD=0.02

def sha(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()

def verify_artifacts():
    f=ROOT/"behavior_sst2_carer_train_dev_v1.jsonl"
    if not f.is_file() or (f.stat().st_mode&0o777)!=0o600 or sha(f.read_bytes())!=SOURCE_SHA:
        raise RuntimeError("source behavior training/development text integrity failed")
    anchors=ROOT/"anchors_300_local.jsonl"
    if not anchors.is_file() or (anchors.stat().st_mode&0o777)!=0o600 or sha(anchors.read_bytes())!=ANCHOR_SHA:
        raise RuntimeError("300 development anchors integrity failed")
    for name,(arr_sha,proj_sha) in EXPECTED_ARCHIVES.items():
        proj=MODELS[name][2]
        arr=ROOT/f"behavior_{name}_600_bf16.npz"
        for path,expected in ((arr,arr_sha),(proj,proj_sha)):
            if not path.is_file() or (path.stat().st_mode&0o777)!=0o600:
                raise RuntimeError("required private artifact missing/insecure: "+str(path))
            if sha(path.read_bytes())!=expected:
                raise RuntimeError("artifact SHA changed: "+str(path))

def shared_fold_plan(reference:dict,axis:str)->dict:
    """Deterministic 5-fold stratification on TRAIN-ONLY row identities."""
    train_idx=np.flatnonzero((reference["axis"]==axis)&(reference["split"]=="behavior_train"))
    if len(train_idx)!=200:raise RuntimeError("need 200 train samples (100 per class)")
    labels=reference["y"][train_idx]
    if (labels==1).sum()!=100 or (labels==0).sum()!=100:
        raise RuntimeError("source train must be balanced 100/100")
    seed=SEED+AXES.index(axis)
    rng=np.random.default_rng(seed)
    folds=np.full(200,-1,dtype=np.int8)
    for y in (0,1):
        members=np.flatnonzero(labels==y)
        parts=np.array_split(rng.permutation(members),FOLDS)
        for fold,part in enumerate(parts):
            if len(part)!=20:raise RuntimeError("unbalanced fold")
            folds[part]=fold
    if (folds<0).any():raise RuntimeError("unassigned fold")
    digest=sha(json.dumps([{"row_text_sha256":reference["ids"][int(i)],"fold":int(folds[j])}
           for j,i in enumerate(train_idx)],sort_keys=True,separators=(",",":")).encode())
    return dict(indices=train_idx,fold_ids=folds,labels=labels,manifest_sha256=digest,seed=seed)

def source_oof(model:dict,axis:str,fold_plan:dict)->dict:
    """Own-model OOF classification error on same 200 training questions.

    Held-out fold excluded from native direction and fold classification
    cutoff, computed as midpoint of two train class mean scores.
    """
    indices=fold_plan["indices"]
    folds=fold_plan["fold_ids"]
    labels=fold_plan["labels"]
    if not np.array_equal(model["y"][indices],labels):
        raise RuntimeError("source training label mismatch")
    centered=model["X"][indices]-model["mu"][None,:]
    n=np.linalg.norm(centered,axis=1)
    if (n<=1e-10).any() or not np.isfinite(n).all():
        raise RuntimeError("degenerate/nonfinite source point")
    projected=(centered/n[:,None])@model["A"].T # 200 x 300
    errors=np.full(len(indices),-1,dtype=np.int8)
    signed_margin=np.zeros(len(indices),dtype=np.float64)
    thresholds=[]
    for f in range(FOLDS):
        fit=folds!=f; hold=folds==f
        if fit.sum()!=160 or hold.sum()!=40:
            raise RuntimeError("fold size drift")
        positive=indices[fit&(labels==1)]
        negative=indices[fit&(labels==0)]
        if len(positive)!=80 or len(negative)!=80:
            raise RuntimeError("source fold class balance drift")
        native=unit(model["X"][positive].mean(0)-model["X"][negative].mean(0))
        canonical=unit(model["A"]@native)
        scores=projected@canonical
        cutoff=0.5*(scores[fit&(labels==1)].mean()+scores[fit&(labels==0)].mean())
        thresholds.append(float(cutoff))
        estimate=(scores[hold]>=cutoff).astype(np.int8)
        errors[hold]=(estimate!=labels[hold]).astype(np.int8)
        signed_margin[hold]=(2*labels[hold]-1)*(scores[hold]-cutoff)
    if np.any(errors<0) or not set(errors.tolist()) <= {0,1}:
        raise RuntimeError("OOF coverage incomplete")
    return dict(error=errors,accuracy=float(1-errors.mean()),
                positive_oof_accuracy=float(1-errors[labels==1].mean()),
                negative_oof_accuracy=float(1-errors[labels==0].mean()),
                signed_margin=signed_margin,thresholds=thresholds)

def target_points(target:dict,axis:str)->tuple[np.ndarray,np.ndarray]:
    idx=np.flatnonzero((target["axis"]==axis)&(target["split"]=="behavior_dev"))
    if len(idx)!=100:raise RuntimeError("target development subset wrong size")
    labels=target["y"][idx]
    if (labels==1).sum()!=50 or (labels==0).sum()!=50:
        raise RuntimeError("target dev label balance drift")
    centered=target["X"][idx]-target["mu"][None,:]
    norms=np.linalg.norm(centered,axis=1)
    if not np.isfinite(norms).all() or (norms<=1e-10).any():
        raise RuntimeError("target hidden states degenerate")
    return (centered/norms[:,None])@target["A"].T,labels

def paired_bootstrap(scores:dict[str,np.ndarray],labels:np.ndarray,seed:int=42,
                    draws:int=BOOTSTRAPS)->dict:
    """Resample identical target-dev rows across conditions for paired CI."""
    pos=np.flatnonzero(labels==1);neg=np.flatnonzero(labels==0)
    if len(pos)!=50 or len(neg)!=50:
        raise RuntimeError("bootstrap must operate on 50+50 development rows")
    expected={"uniform","reliability_only","correlated_error"}
    if not expected <= set(scores):
        raise RuntimeError("missing comparison")
    rng=np.random.default_rng(seed)
    deltas={"correlated_minus_uniform":[],"correlated_minus_reliability_only":[]}
    for _ in range(draws):
        p=pos[rng.integers(0,len(pos),len(pos))]
        n=neg[rng.integers(0,len(neg),len(neg))]
        val={kind:auc(x[p],x[n]) for kind,x in scores.items() if kind in expected}
        deltas["correlated_minus_uniform"].append(val["correlated_error"]-val["uniform"])
        deltas["correlated_minus_reliability_only"].append(val["correlated_error"]-val["reliability_only"])
    return {name:{"lower":float(np.quantile(value,.025)),
                  "upper":float(np.quantile(value,.975)),
                  "resamples":draws}
            for name,value in deltas.items()}

def compute_case(target_name:str,axis:str,models:dict)->dict:
    if target_name not in PARTICIPANTS or axis not in AXES:
        raise ValueError("unrecognized preregistered target or axis")
    sources=[k for k in PARTICIPANTS if k!=target_name]
    target=models[target_name]
    for m in sources:
        obj=models[m]
        if obj["ids"]!=target["ids"] or not np.array_equal(obj["axis"],target["axis"]) or not np.array_equal(obj["split"],target["split"]) or not np.array_equal(obj["y"],target["y"]):
            raise RuntimeError("model train/dev row order or label mismatch")
    plan=shared_fold_plan(models[sources[0]],axis)
    stats=[source_oof(models[m],axis,plan) for m in sources]
    errors=np.column_stack([r["error"] for r in stats])
    if errors.shape!=(200,2):
        raise RuntimeError("source-only OOF error input wrong shape")
    weights={
        "uniform":np.array([0.5,0.5]),
        "reliability_only":weights_from_source_errors(errors,dependence=0,ridge=RIDGE,iterations=2000),
        "correlated_error":weights_from_source_errors(errors,dependence=1,ridge=RIDGE,iterations=2000),
    }
    # independent source-only choice: never target-dev best-model selection
    idx=int(np.argmax([x["accuracy"] for x in stats]))
    weights["source_oof_best_single"]=np.eye(2,dtype=float)[idx]
    weights["source_a_only"]=np.array([1.,0.])
    weights["source_b_only"]=np.array([0.,1.])
    projected_sources=np.stack([models[m]["A"]@source_axis(models[m],axis) for m in sources])
    projected_target,labels=target_points(target,axis)
    conditions={}
    for name,w in weights.items():
        if not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1.):
            raise RuntimeError("invalid source weights")
        direction=unit(canonical_mean_raw(projected_sources,weights=w))
        conditions[name]=projected_target@direction
    estimates={name:auc(v[labels==1],v[labels==0]) for name,v in conditions.items()}
    baseline=estimates["uniform"]
    corr=estimates["correlated_error"]
    diag=estimates["reliability_only"]
    ci=paired_bootstrap(conditions,labels,seed=SEED+AXES.index(axis),draws=BOOTSTRAPS)
    return dict(
        target=target_name,axis=axis,sources=sources,
        fold_count=FOLDS,source_oof_folds_sha256=plan["manifest_sha256"],
        source_oof_fold_seed=plan["seed"],source_oof_error_count=200,
        source_oof_error_rate={model:float(stats[i]["error"].mean()) for i,model in enumerate(sources)},
        source_oof_balanced_accuracy={model:float(stats[i]["accuracy"]) for i,model in enumerate(sources)},
        source_oof_class_accuracy={model:{"positive":stats[i]["positive_oof_accuracy"],
            "negative":stats[i]["negative_oof_accuracy"]} for i,model in enumerate(sources)},
        source_error_coprobability_matrix=(errors.astype(np.float64).T@errors.astype(np.float64)/len(errors)).tolist(),
        source_error_centered_covariance=np.cov(errors.astype(np.float64),rowvar=False,ddof=0).tolist(),
        source_error_cross_correlation=float(np.corrcoef(errors.T)[0,1])
            if all(float(np.std(errors[:,i]))>1e-10 for i in range(2)) else None,
        source_weights={name:[float(x) for x in w] for name,w in weights.items()},
        AUROC={name:float(val) for name,val in estimates.items()},
        source_only_best_choice_model=sources[idx],
        target_dev_oracle_best_single_AUROC=max(estimates["source_a_only"],estimates["source_b_only"]),
        target_dev_oracle_non_deployable=True,
        C1_correlated_minus_uniform=float(corr-baseline),
        C1_correlated_minus_reliability_only=float(corr-diag),
        C1_dev_gain_at_least_point_02=bool(corr-baseline>=GAIN_THRESHOLD),
        paired_999_bootstrap_deltas=ci,
        source_only_weights=True,target_dev_labels_used_only_as_metrics=True,
        no_test_or_final_evaluation=True)

def run()->dict:
    verify_artifacts()
    models={name:load(name) for name in PARTICIPANTS}
    for name in PARTICIPANTS[1:]:
        for k in ("ids","axis","split","y"):
            if k=="ids":
                if models[name][k]!=models["qwen"][k]:raise RuntimeError("source row hashes unequal")
            elif not np.array_equal(models[name][k],models["qwen"][k]):
                raise RuntimeError("source labeled data mismatch")
    cases=[compute_case(target,axis,models) for target in PARTICIPANTS for axis in AXES]
    wins=sum(x["C1_dev_gain_at_least_point_02"] for x in cases)
    mean_delta=float(np.mean([x["C1_correlated_minus_uniform"] for x in cases]))
    return dict(
        schema="oml-acs-c1-source-only-weighting-dev/1",
        status="DEVELOPMENT_EXPLORATORY_COMPLETE_NO_FINAL_HYPOTHESIS_VERDICT",
        scientific_hypothesis="C1",participants=list(PARTICIPANTS),
        model_revisions={k:MODELS[k][1] for k in PARTICIPANTS},
        behavior_train_dev_source_sha256=SOURCE_SHA,anchor_source_sha256=ANCHOR_SHA,
        fold_count=FOLDS,seed=SEED,
        fold_assignment_shared_across_models=True,
        weights_trained_from_source_train_OOF_only=True,
        target_dev_used_to_fit_weights=False,
        normalization="literal weighted Eq4 raw source projection then unit-normalize once for detection",
        conditions=["uniform","source_a_only","source_b_only","source_oof_best_single","reliability_only","correlated_error"],
        cases=cases,case_count=len(cases),
        C1_dev_threshold_absolute_AUROC=GAIN_THRESHOLD,
        C1_dev_cases_at_least_point_02=int(wins),
        C1_dev_mean_AUROC_delta_vs_uniform=mean_delta,
        oracle_not_a_deployable_baseline=True,
        confidence_intervals_development_only=True,
        prior_target_dev_already_seen_by_B=True,
        independent_blind_final_validation_run=False,
        old_V6_24_attested_excluded=False,license_and_semantic_overlap_audit_complete=False,
        objective_changed_after_dev_observation=False,
        A_B_C2_Q_verdicts_not_modified=True,production_touched=False,
        conclusion="Source-only OOF candidate experiment only; report all successes/failures, no final C1 acceptance.")

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--output",default=str(ROOT/"OML_ACS_C1_SOURCE_OOF_DEVELOPMENT_20261021.json"))
    args=cli.parse_args()
    os.umask(0o077)
    result=run()
    path=Path(args.output)
    if path.exists():raise FileExistsError("not overwriting prior C1 experiment")
    with path.open("x",encoding="utf-8") as fh:
        json.dump(result,fh,indent=2,sort_keys=True,ensure_ascii=True)
        fh.write("\n");fh.flush();os.fsync(fh.fileno())
    path.chmod(0o600)
    print("C1_SOURCE_ONLY_OOF_RESULT "+json.dumps(result,sort_keys=True),flush=True)
