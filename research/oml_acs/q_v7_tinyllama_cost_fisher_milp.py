#!/usr/bin/env python3
"""Original BF16 TinyLlama Fisher x real Apple q4/n8 costs, exact 22/154 MILP."""
from pathlib import Path
import json,hashlib,math,os,numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[2]
SCORE=ROOT/"docs/research/BEGLIN_QV7_TINYLLAMA_SOURCE_154_FISHER_PUBLIC_20261022.json"
COST=ROOT/"docs/research/BEGLIN_QV7_LLAMA_NATIVE_SEVEN_ROLE_COSTS_20261022.json"
MASK=ROOT/"configs/research/BEGLIN_QV7_TINYLLAMA_22N8_FISHER_FROZEN_POLICY_20261022.txt"
PUB=ROOT/"docs/research/BEGLIN_QV7_TINYLLAMA_154_COST_FISHER_POLICY_20261022.json"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
def sha(b):return hashlib.sha256(b).hexdigest()
def run():
    if MASK.exists() or PUB.exists():raise FileExistsError("frozen policy cannot be refit")
    src=json.loads(SCORE.read_text());kernel=json.loads(COST.read_text())
    if src["status"]!="SOURCE_ONLY_REAL_LLAMA_ORIGINAL_BF16_64_GRAD_AND_NATIVE_C_QUANT" or \
        src["source_cases"]!=64 or not src["crosscorpus_labels_or_logits_unopened"]:
        raise RuntimeError("not original BF16 source 64 disjoint held")
    if kernel["status"]!="SEVEN_REAL_SOURCE_BF16_LLAMA_ROLE_NATIVE_KERNELS_PROFILLED" or \
       src["official_rev"]!=kernel["source_model_revision"]:
        raise RuntimeError("not same pinned original TinyLlama native kernel costs")
    rows={v["role"]:v for v in kernel["role_kernel_cost_measured"]}
    if set(rows)!=set(ROLES):raise RuntimeError("must have seven original Llama roles")
    mem4={};delta_mem={};time4={};delta_time={}
    for role in ROLES:
        x=rows[role];times=x["real_original_native_kernel_costs"]
        m4=x["q4_with_f32_scales_bytes"];m8=x["n8_with_f32_scales_bytes"]
        t4=min(times["q4_1t"]["median_ms"],times["q4_6t"]["median_ms"])
        t8=times["n8_repacked"]["median_ms"]
        if not 0<m4<m8 or not 0<t4<t8:raise RuntimeError("C kernel cost cannot be inferred "+role)
        mem4[role]=m4;delta_mem[role]=m8-m4;time4[role]=t4;delta_time[role]=t8-t4
    base_m=22*sum(mem4.values())
    base_t=22*sum(time4.values())
    budget=base_m+3*sum(delta_mem.values())+delta_mem["k_proj"]
    cap=base_t*1.25
    scores=src["quality_upgrade_by_role_layer"]
    if set(scores)!=set(ROLES) or any(len(scores[r])!=22 for r in ROLES):
        raise RuntimeError("not 154 original BF16 source projection gains")
    g=np.array([scores[r][i] for r in ROLES for i in range(22)],dtype=np.float64)
    if not np.isfinite(g).all() or np.max(np.abs(g))<=0:
        raise RuntimeError("source Fisher values invalid")
    dmem=np.array([delta_mem[r] for r in ROLES for i in range(22)],dtype=np.float64)
    dtime=np.array([delta_time[r] for r in ROLES for i in range(22)],dtype=np.float64)
    A=np.stack((np.ones(154),dmem,dtime))
    rhs=[22,budget-base_m,cap-base_t]
    normalized=g/max(abs(g))
    eps=1e-12*(154-np.arange(154))/154
    result=milp(c=-(normalized+eps),integrality=np.ones(154,dtype=np.int8),
       bounds=Bounds(np.zeros(154),np.ones(154)),
       constraints=LinearConstraint(A,[22,0,0],rhs),
       options={"time_limit":120.0,"mip_rel_gap":1e-9})
    if result.status!=0 or result.x is None or result.mip_gap>1e-7:
        raise RuntimeError("exact TinyLlama source-only MILP failed "+result.message)
    bits=result.x>.5
    if bits.sum()!=22 or not np.all(np.isclose(result.x,np.round(result.x),atol=1e-6)):
        raise RuntimeError("source-only 22 mixed tier invariant violated")
    ids=[(r,i) for j,r in enumerate(ROLES) for i in range(22) if bits[22*j+i]]
    packed=int(base_m+np.dot(bits.astype(int),dmem))
    latency=float(base_t+np.dot(bits.astype(int),dtime))
    if packed>budget or latency>cap+1e-7:raise RuntimeError("actual native bytes or time proxy exceeds cap")
    mask="".join(f"{r} {i}\n" for r,i in ids).encode()
    with MASK.open("xb") as f:f.write(mask);f.flush();os.fsync(f.fileno())
    doc={"schema":"BEGLIN-Qv7-original-BF16-Llama-154-Fisher-native-NEON-MILP/1",
        "status":"SOURCE_ONLY_EXACT_LLAMA_22N8_MASK_FROZEN_BEFORE_CROSSCORPUS_NATIVE_QUALITY",
        "model_revision":src["official_rev"],
        "source_fisher_154_private_sha256":src["private_original_source_SHA256"],
        "source_154_public_sha256":sha(SCORE.read_bytes()),
        "source_original_Llama_seven_native_costs_sha256":sha(COST.read_bytes()),
        "n8_count":22,"q4_count":132,"baseline_q4_projection_bytes":base_m,
        "selected_projection_bytes":packed,"rolebalanced_22_upgrade_memory_cap_bytes":budget,
        "source_q4_projection_cost_ms":base_t,"selected_projection_cost_ms":latency,
        "source_measured_projection_cost_budget_ms":cap,
        "selected_role_counts":{r:sum(1 for x,j in ids if x==r) for r in ROLES},
        "selected_ids":ids,"mask_sha256":sha(mask),
        "solver_status":int(result.status),"mip_gap":float(result.mip_gap),
        "source_split_half_rank_spearman":src["source_split_half_odd_even_spearman"],
        "source_normalization_max_absolute_gain":float(np.max(np.abs(g))),
        "trained_only_original_BF16_no_quantized_forward":True,
        "model_native_latency_not_measured_by_this_source_model":True,
        "unseen_48_test_accuracy_not_opened_by_optimizer":True,
        "production_promotion":False}
    blob=(json.dumps(doc,indent=2,sort_keys=True)+"\n").encode()
    with PUB.open("xb") as f:f.write(blob);f.flush();os.fsync(f.fileno())
    print("QV7_TINYLLAMA_22N8_SOURCE_ONLY_HIGHS_MILP_FROZEN",
      json.dumps({"n8_role_counts":doc["selected_role_counts"],"budget_bytes":budget,
       "memory_bytes":packed,"predicted_projection_ms":latency,
       "source_rank_rho":doc["source_split_half_rank_spearman"],
       "mask_sha256":doc["mask_sha256"],"solver_gap":doc["mip_gap"],
       "public_sha256":sha(blob)},sort_keys=True),flush=True)
if __name__=="__main__":run()
