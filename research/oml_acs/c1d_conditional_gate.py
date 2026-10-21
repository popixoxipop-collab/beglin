#!/usr/bin/env python3
"""C1-D input-conditional 2-source reliability/correlation gate.

Fixed pre-evaluation protocol docs: C1D_PROSPECTIVE_DEV_FREEZE_20261021.json.
Gate receives only source model canonical input points and source-TRAIN OOF
binary errors. Target labels are passed separately to scoring, not gating.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
import hashlib,json
from pathlib import Path
import numpy as np
from acs_core import unit
from evaluate_c1_source_oof import shared_fold_plan,source_oof,PARTICIPANTS,AXES,SEED
from evaluate_real_behavior import ROOT,load,source_axis,auc

NEW=ROOT/"C1D_NEW_PROSPECTIVE_DEV_200_20261021.jsonl"
SELECTION=ROOT/"C1D_NEW_PROSPECTIVE_DEV_200_20261021_receipt.json"
K=32
SHRINK=0.5
RIDGE=1e-6
BOOT=999

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()

def unit_rows(x):
    z=np.asarray(x,dtype=np.float64)
    if z.ndim!=2 or not np.isfinite(z).all():
        raise ValueError("2D finite point matrix required")
    norms=np.linalg.norm(z,axis=1)
    if (norms<=1e-10).any():raise ValueError("degenerate canonical input")
    return z/norms[:,None]

def closed_form_weight(R):
    """Exact unconstrained minimum of two-source quadratic projected to simplex."""
    mat=np.asarray(R,dtype=np.float64)
    if mat.shape!=(2,2) or not np.isfinite(mat).all() or not np.allclose(mat,mat.T,atol=1e-12):
        raise ValueError("2x2 finite symmetric error co-moment needed")
    a,b,c=float(mat[0,0]+RIDGE),float(mat[0,1]),float(mat[1,1]+RIDGE)
    den=a+c-2*b
    if den<=1e-12:
        return np.array([.5,.5])
    w0=float(np.clip((c-b)/den,0,1))
    return np.array([w0,1-w0])

def compute_gate_weights(old_train_a,old_train_b,query_a,query_b,paired_source_errors):
    """No target labels, target embeddings, or target performance in signature."""
    A,B,Q,R=[unit_rows(x) for x in (old_train_a,old_train_b,query_a,query_b)]
    e=np.asarray(paired_source_errors,dtype=np.float64)
    if len(A)!=len(B) or A.shape[0]!=e.shape[0] or e.ndim!=2 or e.shape[1]!=2:
        raise ValueError("source OOF rows not paired")
    if Q.shape[0]!=R.shape[0] or Q.shape[1]!=A.shape[1] or R.shape[1]!=B.shape[1]:
        raise ValueError("source query point/order mismatch")
    if len(A)<K or (e<0).any() or (e>1).any():
        raise ValueError("need >=32 binary source OOF training errors")
    if not np.isin(e,[0,1]).all():
        raise ValueError("source OOF must use 0/1 errors")
    S=.5*(Q@A.T + R@B.T)
    # Stable tie resolution by original, SHA-bound source row order.
    nearest=np.argsort(-S,axis=1,kind="stable")[:,:K]
    global_R=(e.T@e)/len(e)
    static=closed_form_weight(global_R)
    dynamic=[]
    for chosen in nearest:
        local=e[chosen]
        local_R=(local.T@local)/K
        fused=SHRINK*global_R+(1-SHRINK)*local_R
        dynamic.append(closed_form_weight(fused))
    weights=np.asarray(dynamic)
    if weights.shape!=(len(Q),2) or not np.isfinite(weights).all() or (weights<0).any() or not np.allclose(weights.sum(axis=1),1):
        raise RuntimeError("invalid conditional simplex weights")
    return weights,static,global_R,nearest

def load_new_rows():
    if not NEW.is_file() or not SELECTION.is_file() or NEW.stat().st_mode&0o777!=0o600 or SELECTION.stat().st_mode&0o777!=0o600:
        raise RuntimeError("new prospective data absent or insecure")
    content=NEW.read_bytes()
    source=json.loads(SELECTION.read_text())
    if sha(content)!=source["source_file_sha256"] or source["status"]!="SELECTED_NOT_EVALUATED":
        raise RuntimeError("new evaluation corpus identity is not pinned")
    rows=[json.loads(x) for x in content.decode().splitlines() if x.strip()]
    if len(rows)!=200 or len({r["canonical_sha256"] for r in rows})!=200:
        raise RuntimeError("new eval row count/uniqueness failed")
    if {r["axis"] for r in rows}!=set(AXES):
        raise RuntimeError("behavior axes changed")
    for axis in AXES:
        for y in (0,1):
            if sum(r["axis"]==axis and r["class_label"]==y for r in rows)!=50:
                raise RuntimeError("new prospective class balance changed")
    return rows,source

def load_new_model(which,rows,source_sha):
    from evaluate_real_behavior import MODELS
    if which not in PARTICIPANTS:raise ValueError("model not in prereg")
    file=ROOT/f"C1D_newdev_{which}_200_bf16.npz"
    if not file.is_file() or file.stat().st_mode&0o777!=0o600:
        raise RuntimeError("model new data artifact missing/insecure")
    with np.load(file,allow_pickle=False) as z:
        if z["data_source_sha256"].item()!=source_sha or z["model_revision"].item()!=MODELS[which][1]:
            raise RuntimeError("new activation source/model revision drift")
        if z["text_sha256"].tolist()!=[r["text_sha256"] for r in rows]:
            raise RuntimeError("new model row identity/order drift")
        if z["axis"].tolist()!=[r["axis"] for r in rows]:
            raise RuntimeError("new model axis order drift")
        X=z["activations"].astype(np.float64)
    if X.shape!=(200,MODELS[which][0]) or not np.isfinite(X).all():
        raise RuntimeError("new model activation shape invalid")
    return X

def canonical_point(model,values):
    centered=np.asarray(values,dtype=np.float64)-model["mu"][None,:]
    norms=np.linalg.norm(centered,axis=1)
    if not np.isfinite(centered).all() or (norms<=1e-10).any():
        raise ValueError("zero/nonfinite anchored source point")
    return (centered/norms[:,None])@model["A"].T

def score_condition(target_points,axes,weights):
    q=np.asarray(target_points,dtype=np.float64)
    a=np.stack([np.asarray(v,dtype=np.float64) for v in axes])
    w=np.asarray(weights,dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=a.shape[1] or a.shape[0]!=2 or w.shape!=(len(q),2):
        raise ValueError("per-input gate dimensions mismatch")
    if np.any(w<0) or not np.allclose(w.sum(axis=1),1):
        raise ValueError("invalid per-input simplex")
    mixed=w@a
    mixed=unit_rows(mixed)
    return np.sum(q*mixed,axis=1)

def paired_ci(scores,labels,n=BOOT,seed=73):
    y=np.asarray(labels)
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    if len(pos)!=50 or len(neg)!=50:
        raise ValueError("class balance must be 50/50")
    rng=np.random.default_rng(seed)
    a,b=np.asarray(scores["dynamic"]),np.asarray(scores["static"])
    differences=[]
    for _ in range(n):
        i=pos[rng.integers(len(pos),size=len(pos))]
        j=neg[rng.integers(len(neg),size=len(neg))]
        differences.append(auc(a[i],a[j])-auc(b[i],b[j]))
    return [float(v) for v in np.quantile(differences,[.025,.975])]

def evaluate(target_name,axis,old,new,rows):
    if target_name not in PARTICIPANTS or axis not in AXES:
        raise ValueError("invalid target or behavior axis")
    sources=[v for v in PARTICIPANTS if v!=target_name]
    base=old[sources[0]]
    fold=shared_fold_plan(base,axis)
    errors=[]
    for name in sources:
        m=old[name]
        if m["ids"]!=base["ids"] or not np.array_equal(m["y"],base["y"]):
            raise RuntimeError("OOF source row identity mismatch")
        errors.append(source_oof(m,axis,fold)["error"])
    error=np.column_stack(errors)
    # Cache 200 TRAIN-only source input canonical points for nearest neighbor.
    history=[canonical_point(old[s],old[s]["X"][fold["indices"]]) for s in sources]
    ids=np.asarray([r["axis"]==axis for r in rows])
    if ids.sum()!=100:raise RuntimeError("new dev axis count invalid")
    input_points=[canonical_point(old[s],new[s][ids]) for s in sources]
    w_dynamic,w_static,global_R,neighbors=compute_gate_weights(history[0],history[1],
                                             input_points[0],input_points[1],error)
    target_points=canonical_point(old[target_name],new[target_name][ids])
    axes=[old[s]["A"]@source_axis(old[s],axis) for s in sources]
    scores={
      "uniform":score_condition(target_points,axes,np.tile([.5,.5],(100,1))),
      "static":score_condition(target_points,axes,np.tile(w_static,(100,1))),
      "dynamic":score_condition(target_points,axes,w_dynamic),
    }
    labels=np.asarray([rows[i]["class_label"] for i in range(len(rows)) if ids[i]],dtype=np.int8)
    if (labels==1).sum()!=50 or (labels==0).sum()!=50:
        raise RuntimeError("new class counts invalid")
    aucs={k:auc(v[labels==1],v[labels==0]) for k,v in scores.items()}
    ci=paired_ci(scores,labels)
    variation=np.std(w_dynamic[:,0])
    return dict(target=target_name,source_models=sources,axis=axis,dev_count=100,
      OOF_memory_rows=len(error),knn_neighbors=K,global_fraction=SHRINK,
      fold_manifest_sha256=fold["manifest_sha256"],
      source_global_OOF_error_R=global_R.tolist(),
      static_weights=[float(x) for x in w_static],
      auc=aucs,delta_dynamic_minus_static=float(aucs["dynamic"]-aucs["static"]),
      delta_dynamic_minus_uniform=float(aucs["dynamic"]-aucs["uniform"]),
      bootstrap_999_dynamic_minus_static_ci95=ci,
      threshold_met=bool(aucs["dynamic"]-aucs["static"]>=.02),
      dynamic_weight_w0_summary=dict(min=float(w_dynamic[:,0].min()),
        p25=float(np.quantile(w_dynamic[:,0],.25)),median=float(np.median(w_dynamic[:,0])),
        p75=float(np.quantile(w_dynamic[:,0],.75)),max=float(w_dynamic[:,0].max()),
        standard_deviation=float(variation),
        fraction_at_boundary=float(np.mean((w_dynamic[:,0]<=1e-12)|(w_dynamic[:,0]>=1-1e-12))),
        fraction_near_static_1e6=float(np.mean(np.abs(w_dynamic[:,0]-w_static[0])<=1e-6))),
      no_target_dev_labels_passed_to_gate=True)

def run():
    rows,src=load_new_rows()
    old={name:load(name) for name in PARTICIPANTS}
    new={name:load_new_model(name,rows,src["source_file_sha256"]) for name in PARTICIPANTS}
    out=[evaluate(t,ax,old,new,rows) for t in PARTICIPANTS for ax in AXES]
    return dict(schema="oml-acs-C1D-prospective-dev-local-gate/1",
      status="NEW_SOURCE_DEVELOPMENT_ONLY_NOT_FINAL",
      protocol_file="configs/research/OML_ACS_C1D_PROSPECTIVE_DEV_FREEZE_20261021.json",
      new_dataset_sha256=src["source_file_sha256"],new_dataset_count=200,
      old_behavior_sha256="b99fb5d7e8c909e6821f555ac590b36ff52393b9d4e7a01828ee936c615bd274",
      original_300_anchor_sha256="47d5ddcb37b81212e3f43a34441662062b42a038d4cc606d9d7081c31131f83b",
      rows=out,case_count=len(out),mean_delta_dynamic_vs_static=float(np.mean([x["delta_dynamic_minus_static"] for x in out])),
      wins_gt_point02=sum(x["threshold_met"] for x in out),
      source_features_only_gate=True,target_labels_used_to_fit_gate=False,
      new_data_labels_only_for_AUROC=True,
      datasets_were_source_train_only=True,
      independent_holdout_not_opened=True,
      old_V6_24_identifiers_not_attested=True,
      model_mixture_latency_not_measured=True,
      BEGLIN_tensor_bit_allocation_not_tested=True,
      main_or_production_modified=False)

if __name__=="__main__":
    result=run()
    os.umask(0o077)
    path=ROOT/"C1D_NEW_DEV_DYNAMIC_GATE_RESULTS_20261021.json"
    if path.exists():raise FileExistsError("refuse overwrite previous C1D result")
    with path.open("x") as f:
        json.dump(result,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
    print("C1D_NEW_DEV_EVIDENCE",json.dumps(result,sort_keys=True),flush=True)
