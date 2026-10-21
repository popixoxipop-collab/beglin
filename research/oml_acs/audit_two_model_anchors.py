#!/usr/bin/env python3
"""Development-only paired ACS anchor geometry audit: Qwen1.5B vs Mistral7B.

Requires exact text-hash/row-order equivalence of both 300-anchor matrices.
Computes similarity and fixed-seed anchor-order permutation nulls.
Cannot establish behavioral transfer, generation steering or final AUROC.
Private input arrays are never published; output is aggregate JSON only.
"""
from __future__ import annotations
import collections
import hashlib
import json
import pathlib
import numpy as np

BASE = pathlib.Path.home() / ".local/share/beglin/oml_acs/20261021"
INPUT_SHA = "47d5ddcb37b81212e3f43a34441662062b42a038d4cc606d9d7081c31131f83b"
TARGET_Q_SHA = "cbd5535b5247ea7b153a7835e5c27d9b8cc680cc6d031108b87db1a49136f692"
Q = BASE / "qwen25_300_anchor_projector_v1.npz"
M = BASE / "mistral7b_layer16_300_projector_bf16.npz"
QA = BASE / "qwen25_15layer_300_anchors_bf16.npz"
MA = BASE / "mistral7b_layer16_300_anchors_bf16.npz"
MREC = BASE / "mistral7b_layer16_300_anchors_bf16_receipt.json"
OUT = BASE / "qwen_mistral_300_anchor_geometry_audit_20261021.json"

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def center_kernel(k: np.ndarray) -> np.ndarray:
    """Double center Gram kernel over prompt examples, not feature dimensions."""
    if k.ndim!=2 or k.shape[0]!=k.shape[1]:
        raise ValueError("a square Gram matrix is required")
    return k - k.mean(axis=0)[None,:] - k.mean(axis=1)[:,None] + k.mean()

def linear_cka_gram(k: np.ndarray, l: np.ndarray) -> float:
    """Biased CKA; report against row-shuffle null due finite-n effects."""
    a,b=center_kernel(k),center_kernel(l)
    denom=float(np.linalg.norm(a)*np.linalg.norm(b))
    if denom<=1e-12:raise ValueError("degenerate CKA")
    return float(np.sum(a*b)/denom)

def upper_pearson(k: np.ndarray, l: np.ndarray) -> float:
    """Exclude equal-row diagonal (always 1.0), use all unordered pair cells."""
    if k.shape!=l.shape or k.ndim!=2 or k.shape[0]!=k.shape[1]:
        raise ValueError("incompatible Gram shapes")
    i=np.triu_indices(k.shape[0],k=1)
    x,y=k[i],l[i]
    if x.std()<1e-12 or y.std()<1e-12:
        raise ValueError("zero-variance Gram upper triangle")
    return float(np.corrcoef(x,y)[0,1])

def metrics(q: np.ndarray, m: np.ndarray, repeats: int=199, seed: int=42)->dict:
    if q.ndim!=2 or m.ndim!=2 or q.shape[0]!=m.shape[0] or q.shape[0]<3:
        raise ValueError("paired row counts required")
    if not np.isfinite(q).all() or not np.isfinite(m).all():
        raise ValueError("nonfinite inputs")
    if not np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-8) or not np.allclose(np.linalg.norm(m,axis=1),1,atol=1e-8):
        raise ValueError("each anchor row must be L2 normalized")
    k=q@q.T
    l=m@m.T
    observed_c=linear_cka_gram(k,l)
    observed_p=upper_pearson(k,l)
    rng=np.random.default_rng(seed)
    null_c,null_p=[],[]
    for _ in range(repeats):
        order=rng.permutation(len(q))
        z=l[np.ix_(order,order)]
        null_c.append(linear_cka_gram(k,z))
        null_p.append(upper_pearson(k,z))
    return dict(
        row_count=len(q), source_feature_dim=q.shape[1], target_feature_dim=m.shape[1],
        alignment_CKA_biased=observed_c,
        CKA_row_shuffle_mean=float(np.mean(null_c)),
        CKA_row_shuffle_95th=float(np.quantile(null_c,.95)),
        CKA_row_shuffle_empirical_p=float((1+sum(t>=observed_c for t in null_c))/(1+repeats)),
        offdiag_pairwise_cosine_pearson=observed_p,
        offdiag_pearson_shuffle_mean=float(np.mean(null_p)),
        offdiag_pearson_shuffle_95th=float(np.quantile(null_p,.95)),
        offdiag_pearson_shuffle_empirical_p=float((1+sum(t>=observed_p for t in null_p))/(1+repeats)),
        shuffle_count=repeats,shuffle_seed=seed,
    )

def main():
    assert not OUT.exists(), "refuse overwrite"
    for p in (Q,M,QA,MA,MREC):
        if not p.is_file() or p.stat().st_mode&0o777!=0o600:
            raise RuntimeError("private input missing or wrong mode: "+str(p))
    receipt=json.loads(MREC.read_text())
    assert receipt["status"]=="PASS_TECHNICAL_EXTRACTION"
    assert receipt["rows_same_order_as_qwen"] is True and receipt["source_anchors_sha256"]==INPUT_SHA
    assert receipt["source_qwen_array_sha256"]==TARGET_Q_SHA
    with np.load(Q,allow_pickle=False) as srcp, np.load(M,allow_pickle=False) as targetp:
        q=srcp["ahat"]
        m=targetp["ahat"]
        assert q.shape==(300,1536) and m.shape==(300,4096)
        assert srcp["source_raw_sha256"].item()==INPUT_SHA
        assert targetp["source_raw_sha256"].item()==INPUT_SHA
        assert targetp["activation_array_sha256"].item()==receipt["target_activation_array_sha256"]
    with np.load(QA,allow_pickle=False) as qa, np.load(MA,allow_pickle=False) as ma:
        assert qa["text_sha256"].tolist()==ma["text_sha256"].tolist()
        assert qa["scenario"].tolist()==ma["scenario"].tolist()
        assert qa["source_root_sha256"].item()==ma["source_raw_sha256"].item()==INPUT_SHA
        assert sha(ma["activations"].tobytes())==receipt["target_activation_array_sha256"]
    report={
        "schema":"oml-acs-qwen-mistral-300-geometry-audit/1",
        "status":"PASS_DEV_GEOMETRY_DIAGNOSTIC_ONLY",
        "source_anchors_sha256":INPUT_SHA,
        "source_activation_sha256":TARGET_Q_SHA,
        "target_activation_sha256":receipt["target_activation_array_sha256"],
        "source_projector_file_sha256":sha(Q.read_bytes()),
        "target_projector_file_sha256":sha(M.read_bytes()),
        "source_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "target_revision":"c170c708c41dac9275d15a8fff4eca08d52bab71",
        "unit_normalized_rows":True,
        "matched_anchor_row_identifiers_and_order":True,
        "diagnostic_metrics":metrics(q,m),
        "benchmark_label_accessed":False,
        "behavior_steering_proven":False,"heldout_AUROC_proven":False,
        "replicated_author_300_ids":False,
        "final_eval_opened":False,"v6_used_24_exclusion_verified":False,
        "raw_text_or_activations_published":False,
        "scientific_hypothesis_verdict":"NO_VERDICT",
        "limitations":[
            "CKA on the construction anchors is not cross-model behavior transfer accuracy",
            "CKA estimator is biased and shuffle null only diagnoses row correspondence",
            "No steering, behavioral labels, final holdout or prospective samples were used",
            "Reimplementation anchors are not author-row-identical HELM anchors",
        ],
    }
    import os
    os.umask(0o077)
    with OUT.open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2,sort_keys=True)
        f.write("\n");f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    print("GEOMETRY_AUDIT_RECEIPT",json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
