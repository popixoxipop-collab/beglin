#!/usr/bin/env python3
"""Independent Qv8 source-only signed BF16 196x96 audit, no estimator imports.

Recomputes the global first-order signed Q4/n8 joint-loss objective from
individual sealed original BF16 source gradients. Checks local-search
reported score against original held dataset/admission boundaries. Does not
regenerate full-model logits or claim exact Hessian/causality.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,struct
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv8_joint_signed_20261022"
PRE=ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_SOURCE_PRELOCK_20261022.json"
ARC=LAB/"pinned_original_ARC_train96.json"
HELD=LAB/"pinned_HellaSwag_val48.json"
RAW=LAB/"original_BF16_signed_196x96_source_private.json"
PUBLIC_SRC=ROOT/"docs/research/BEGLIN_QV8_BF16_SIGNED_CROSS_TENSOR_SOURCE_PUBLIC_20261022.json"
POL=ROOT/"docs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_OPTIMIZER_20261022.json"
MASK=ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt"
OUT=LAB/"independent_Qv8_source_joint_signed_audit.json"
PUB=ROOT/"docs/research/BEGLIN_QV8_SOURCE_JOINT_SIGNED_INDEPENDENT_PUBLIC_20261022.json"
PRE_SHA="9b833f5fc2729d13320c87e368baaff952c890ed068586622badfc7da63b4db1"
ARC_SHA="8da2f96b6ec02a5a45a2308e84074a1de14e7a93dc30e9517960596e3b2cee0c"
HELD_SHA="45c95170a2eb01b4c79ad44719ab9a1877e568f2583a90d84340d7448e0d1a33"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
def sha(b:bytes):return hashlib.sha256(b).hexdigest()
def secure(p):
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("original Qv8 BF16 source private receipt unsealed "+str(p))
    return p.read_bytes()
def measure(mask,baseline:np.ndarray,delta:np.ndarray):
    x=np.zeros(196,dtype=np.float64)
    for role,n in mask:
        j=ROLES.index(role)*28+n;x[j]=1.
    return float(np.mean((baseline+delta@x)**2))
def source():
    if sha(PRE.read_bytes())!=PRE_SHA or sha(secure(ARC))!=ARC_SHA or sha(secure(HELD))!=HELD_SHA:
        raise RuntimeError("Qv8 source/eval or source-only prelock changed")
    orig=json.loads(secure(ARC))["source_train96"]
    if len(orig)!=96 or len({q["id"] for q in orig})!=96:
        raise RuntimeError("original source train96 invalid")
    data=secure(RAW);values=json.loads(data)
    agg=json.loads(PUBLIC_SRC.read_text())
    if agg["full_source_private_SHA256"]!=sha(data) or values["status"]!="SOURCE_ONLY_196x96_SIGNED_NATIVE_Q4N8_DELTA_VERIFIED" or \
       not values["eval_logits_unopened"] or values["unopened_HellaSwag48_SHA256"]!=HELD_SHA:
        raise RuntimeError("original BF16 signed source SHA not verified")
    rows=[];checkpoint=[]
    for i,q in enumerate(orig):
        p=LAB/"signed_source96_case_receipts"/f"signed_{i:03d}.json"
        b=secure(p)
        if sha(b)!=values["private_per_case_SHA256"][i]:
            raise RuntimeError("Qv8 signed original per-source gradient receipt changed")
        x=json.loads(b)
        if x["index"]!=i or x["id_sha256"]!=sha(q["id"].encode()) or \
           len(x["signed_q4_per_tensor"])!=196 or len(x["signed_n8_per_tensor"])!=196:
            raise RuntimeError("original signed scalar role / source item mismatch")
        rows.append(x)
    q4=np.asarray([x["signed_q4_per_tensor"] for x in rows],dtype=np.float64)
    n8=np.asarray([x["signed_n8_per_tensor"] for x in rows],dtype=np.float64)
    if q4.shape!=(96,196) or n8.shape!=(96,196) or \
       not (np.isfinite(q4).all() and np.isfinite(n8).all()):
        raise RuntimeError("source 96x196 signed task gradients nonfinite")
    if not np.array_equal(q4,np.asarray(values["original_tensor_grad_dot_q4"])) or \
       not np.array_equal(n8,np.asarray(values["original_tensor_grad_dot_n8"])):
        raise RuntimeError("original BF16 96 per-tensor signs not reproducible from private checkpoints")
    if abs(values["first_order_joint_Q4_source_rms"]**2-float(np.mean((q4.sum(axis=1))**2)))>1e-7:
        raise RuntimeError("first-order original Q4 task loss prediction mis-scored")
    policy=json.loads(POL.read_text())
    if policy["status"]!="SOURCE_ONLY_28N8_STRICT_FEASIBLE_LOCAL_SWAP_DONE_NO_EVAL_OPENED" or \
       policy["source_96_signed_private_sha256"]!=sha(data) or \
       not policy["no_held_data_labels_or_logits_loaded"] or policy["source_optimization_exact_global"]:
        raise RuntimeError("joint signed source-only bounded optimizer evidence untrustworthy")
    entries=[line.split() for line in MASK.read_text().splitlines()]
    if len(entries)!=28 or len(set(tuple(x) for x in entries))!=28 or \
       any(len(x)!=2 or x[0] not in ROLES or not x[1].isdigit() or int(x[1])>=28 for x in entries):
        raise RuntimeError("source selected exactly 28 original Qwen tensors required")
    if policy["mask_sha256"]!=sha(MASK.read_bytes()):
        raise RuntimeError("real source Qv8 28n8 policy SHA drift")
    selected={(a,int(b)) for a,b in entries}
    old=q4.sum(axis=1);d=n8-q4
    final=measure(selected,old,d)
    if abs(final-policy["selected_source_signed_loss_squared"])>max(1e-4,1e-5*abs(final)):
        raise RuntimeError("cross tensor signed source objective not reproducible")
    if not policy["strict_28N8_and_168Q4"] or \
       policy["final_original_Apple_kernel_costs"]["memory_bytes"]>830570496 or \
       policy["final_original_Apple_kernel_costs"]["serial_estimate_ms"]>100.0001 or \
       policy["final_original_Apple_kernel_costs"]["isolated_async_stage_ms"]>50.0001:
        raise RuntimeError("native C actual budget source admission invalid")
    basis={}
    checks=[
      ("qv6_64","BEGLIN_QV6_FISHER_64N8_SOURCE_POLICY_20261022.txt"),
      ("qv6_128","BEGLIN_QV6_FISHER_128N8_SOURCE_POLICY_20261022.txt"),
      ("qv5_24","BEGLIN_QV5_FISHER_28N8_SOURCE_POLICY_20261022.txt")]
    for key,name in checks:
        mask2={(a,int(b)) for a,b in (line.split() for line in (ROOT/"configs/research"/name).read_text().splitlines())}
        value=measure(mask2,old,d)
        expect=policy["source_old_MILP_objective_competitors"][key]["signed_joint_source_objective"]
        if abs(value-expect)>max(1e-4,abs(value)*1e-5):
            raise RuntimeError("baseline signed-loss score cannot be reproduced "+key)
        basis[key]=value
    if abs(policy["objective_global_loss_change_squared_empirical_mean_Q4"]-float(np.mean(old**2)))>max(1e-4,abs(float(np.mean(old**2)))*1e-5):
        raise RuntimeError("all-Q4 source objective inconsistent")
    # Recompute pairwise Gram; confirms cross terms really present.
    G=(d.T@d)/96
    offdiag=float(np.linalg.norm(G-np.diag(np.diag(G))))
    if not math.isfinite(offdiag) or offdiag<=0 or abs(offdiag-policy["pairwise_cross_term_frobenius_norm"])>max(1e-3,offdiag*1e-6):
        raise RuntimeError("reported signed cross-tensor interactions not genuine")
    return {"source_private_SHA":sha(data),"selected_policy_SHA":sha(MASK.read_bytes()),
       "original_source_gradient_cases":96,"gradients_per_case":196,
       "source_Q4_full_squared_loss":float(np.mean(old**2)),
       "source_new_joint_squared_loss":final,
       "old_mask_objectives":basis,
       "cross_tensor_Gram_offdiag_norm":offdiag,
       "no_global_optimality_claim":True,
       "no_held_data_or_labels_used":True,
       "no_operational_changes":True}
def main():
    if OUT.exists() or PUB.exists():raise FileExistsError("cannot overwrite original independent BF16 signed source audit")
    record={"schema":"BEGLIN-Qv8-separate-original-BF16-signed-source-96x196-gradient-audit/1",
        "status":"PASS_RAW_96x196_SIGNED_CROSS_TENSOR_SOURCE_AND_MASK_COST_CONTRACT",
        **source()}
    os.umask(0o077)
    b=(json.dumps(record,indent=2,sort_keys=True)+"\n").encode()
    with OUT.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    pub={**record,"private_source_audit_SHA256":sha(b)}
    data=(json.dumps(pub,indent=2,sort_keys=True)+"\n").encode()
    with PUB.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    print("QV8_INDEPENDENT_ORIGINAL_BF16_SIGNED_CROSS_TENSOR_AUDIT_PASS",
        json.dumps({"source_Q4":record["source_Q4_full_squared_loss"],
         "new":record["source_new_joint_squared_loss"],
         "basis":record["old_mask_objectives"],"public_SHA":sha(data),"private_SHA":sha(b)},sort_keys=True),flush=True)
if __name__=="__main__":main()
