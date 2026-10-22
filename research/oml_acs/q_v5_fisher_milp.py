#!/usr/bin/env python3
"""Exact 28-of-196 mixed precision HiGHS MILP, Qv5 Fisher gains/source-only.

The Qv2 measured Apple NEON kernel bytes, serial latency and QKV/FFN stage
timings are unchanged. Training Fisher is original BF16 only and frozen BEFORE
the locked ARC validation split is ever scored. No live runtime write.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v2_async_activation_milp as v2

PROTO=ROOT/"configs/research/BEGLIN_QV5_FISHER_GRAD_PRELOCK_20261022.json"
PUB=ROOT/"docs/research/BEGLIN_QV5_FISHER_SOURCE_AGGREGATE_20261022.json"
INPUT_LOCK=ROOT/"docs/research/BEGLIN_QV5_ARC_FREEZE_PUBLIC_AGGREGATE_20261022.json"
MASK=ROOT/"configs/research/BEGLIN_QV5_FISHER_28N8_SOURCE_POLICY_20261022.txt"
REPORT=ROOT/"docs/research/BEGLIN_QV5_FISHER_HIGHS_SOURCE_MILP_20261022.json"
ORIGINAL=ROOT/"lab_local/qv5_fisher_20261022/fisher_196_train_private_receipt.json"
PRE_SHA="a886418affc9f665d64ce4ca9eb0b5ea2aef4ae97bc59bf58ea1affb53bf19ef"
LOCK_SHA="43f693c8697b4e8889f807780034093c62c5813a3ad31e7a3dea7428083a3f84"
ROLES=v2.ROLES
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def verified():
    if REPORT.exists() or MASK.exists():
        raise FileExistsError("policy was previously frozen; do not refit after evaluation")
    if sha(PROTO.read_bytes())!=PRE_SHA or sha(INPUT_LOCK.read_bytes())!=LOCK_SHA:
        raise RuntimeError("source/data prelock not original")
    rec=json.loads(PUB.read_text())
    if rec["status"]!="SOURCE_TRAIN_ONLY_ALL_196_NATIVE_GRADIENTS_AND_QUANT_ERRORS" or \
       not rec["no_dev_labels_accessed"] or rec["source_train_SHA256"]!="631bb02d22b696ddcabfd3d7ab759218fd75e1e8f3d8384cbf4a5df4de3f6994":
        raise RuntimeError("Fisher source not independently graduated / validation leakage")
    if (ORIGINAL.stat().st_mode&0o777)!=0o600 or sha(ORIGINAL.read_bytes())!=rec["private_full_source_receipt_SHA256"]:
        raise RuntimeError("missing or tampered raw original-BF16 directional Fisher receipt")
    gains=rec["quality_upgrade_by_role_layer"]
    if set(gains)!=set(ROLES) or any(len(gains[r])!=28 for r in ROLES):
        raise RuntimeError("not 196 source tensors")
    vec=np.asarray([gains[r][i] for r in ROLES for i in range(28)],dtype=np.float64)
    if not np.isfinite(vec).all() or not np.max(np.abs(vec))>0:
        raise RuntimeError("source directional Fisher nonfinite/zero")
    return rec,vec

def run():
    rec,vec=verified()
    t0=time.monotonic()
    _,_,stages,serial,memory=v2.verified_sources()
    scale=float(np.max(np.abs(vec)))
    scaled={role:list((np.array(rec["quality_upgrade_by_role_layer"][role],dtype=np.float64)/scale).astype(float)) for role in ROLES}
    candidate=v2.solve({"quality_upgrade_by_role_layer":scaled},stages,serial,memory)
    mask="".join(f"{role} {i}\n" for role,i in candidate["selected_ids"]).encode()
    if len(candidate["selected_ids"])!=28 or candidate["mip_gap"]>1e-7 or candidate["n4_tensors"]!=168:
        raise RuntimeError("exact mixed precision MILP proof absent")
    os.umask(0o077)
    with MASK.open("xb") as f:
        f.write(mask);f.flush();os.fsync(f.fileno())
    # Policy choice stays research-only even when the source split-half stability is good.
    output={
      "schema":"BEGLIN-Qv5-original-BF16-directional-Fisher-source-HiGHS-MILP/1",
      "status":"FROZEN_SOURCE_ONLY_EXACT_MILP_NOT_YET_ARC_VALIDATION_SCORED",
      "preregistration_sha256":PRE_SHA,"original_input_lock_sha256":LOCK_SHA,
      "original_fisher_source_pub_sha256":sha(PUB.read_bytes()),
      "original_fisher_source_private_sha256":rec["private_full_source_receipt_SHA256"],
      "prior_native_kernel_stage_cost_sha256":v2.SHA_STAGE,
      "prior_original_diagonal_activation_score_SHA256":v2.SHA_SCORES,
      "source_only_objective":"mean original BF16 source prompt of [(grad·(Q4(W)-W))^2-(grad·(Q8(W)-W))^2]",
      "score_scale_to_max_abs":scale,
      "negative_gain_tensors":rec["negative_gain_tensors"],
      "odd_even_source_spearman":rec["split_half_odd_even_spearman"],
      "source_split_half_stability_advisory_met":rec["split_half_odd_even_spearman"]>=0.3,
      "mask_sha256":sha(mask),"optimizer":candidate,
      "source_original_BF16":rec["original_HF_rev"],
      "development_ARC_40_labels_unopened":True,
      "no_downstream_one-tensor_full_model_interventions":True,
      "production_write_allowed":False,"native_fullmodel_async_default_enabled":False,
      "model_accuracy_improvement_proved":False,
      "scientific_Q_verdict":"NO_VERDICT","elapsed_s":round(time.monotonic()-t0,2)
    }
    text=(json.dumps(output,indent=2,sort_keys=True)+"\n").encode()
    with REPORT.open("xb") as f:
        f.write(text);f.flush();os.fsync(f.fileno())
    print("QV5_FISHER_REAL_SOURCE_ONLY_MILP_FROZEN",json.dumps({
        "source_fisher_gain_scale":scale,"policy_SHA":sha(mask),"receipt_SHA":sha(text),
        "n8_counts":candidate["selected_role_counts"],
        "memory_bytes":candidate["estimated_projection_weight_bytes"],
        "predicted_serial_ms":candidate["original_fullmodel_serial_estimated_ms"],
        "stage_max_ms":candidate["isolated_async_CPU_stage_total_ms"],
        "source_split_half_spearman":rec["split_half_odd_even_spearman"],
        "MILP_gap":candidate["mip_gap"]},sort_keys=True),flush=True)
if __name__=="__main__":run()
