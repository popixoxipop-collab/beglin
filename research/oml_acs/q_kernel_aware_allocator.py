#!/usr/bin/env python3
"""Exact BEGLIN CPU kernel-aware Q4G64/n8 tensor bit+kernel allocator.

Uses only previously recorded 196 original-BF16 tensor weight distortion,
measured Q4 1t/6t and repacked n8 native kernel costs, and old known
full-model *latency calibration*, never target logits or PPL for selection.

Seven groups share per-role shape, memory increment, kernel cost. Within a
role the mathematically best k weight-proxy upgrades are top k deltas.
Enumerate 7-role integer counts summing to 28; exact under the surrogate.
"""
from __future__ import annotations
import hashlib,json,math,os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
QSCAN=ROOT/"lab_local/q_tensor_proxy_20261021/q_196_proxy_results_private.json"
PROFILE=ROOT/"lab_local/q_kernel_aware_role_costs_20261022_v2/seven_role_native_cost_private.json"
TARGET=ROOT/"lab_local/q_kernel_aware_allocator_20261022"
EXPECTED_SCAN="720de694838d3a85156afae3732974f3867db1c8ce4957e55778da77724a0d8a"
EXPECTED_PROFILE="280ca338745d0ebb2c44986d18e8b240135fe7fa4b1077f49f65f5d8fb3eff93"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
NUM_UPGRADES=28
MEMORY_MAX=830570496
NATIVE_BASE_Q4_MS=75.26
REFERENCE_ROLE4_MS=116.18
KERNEL_AWARE_MAX_MS=100.0

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def read_secure(path:Path,checksum:str)->dict:
    if not path.is_file() or path.is_symlink() or path.stat().st_mode&0o777!=0o600:
        raise RuntimeError("source evidence missing/insecure")
    buf=path.read_bytes()
    if sha(buf)!=checksum:raise RuntimeError("pinned source-only evidence SHA mismatch "+str(path))
    return json.loads(buf)
def mk_roles(scan,profile):
    source=scan["rows"]
    if len(source)!=196 or len({(x["role"],x["layer"]) for x in source})!=196:
        raise RuntimeError("not 196 real source projections")
    roledata={r["role"]:r for r in profile["role_samples"]}
    if set(roledata)!=set(ROLES):raise RuntimeError("7 kernel roles not matched")
    result={}
    for role in ROLES:
        subset=[x for x in source if x["role"]==role]
        if len(subset)!=28 or {r["layer"] for r in subset}!=set(range(28)):
            raise RuntimeError("not 28 layers per role")
        kern=roledata[role]
        if kern["shape"]!=subset[0]["weight_shape"]:
            raise RuntimeError("actual native kernel and original tensor geometry drift")
        for row in subset:
            if kern["q4_with_scales_bytes"]!=row["q4_with_scales_bytes"] or kern["n8_with_scales_bytes"]!=row["qng64_8_with_scales_bytes"]:
                raise RuntimeError("actual tensor memory cost differs from original source encoder")
        # Native Q4 can choose ONE of its real kernel variants, per actual role.
        options=("q4_1t","q4_6t")
        q4_kernel=min(options,key=lambda k:(kern["times"][k]["median_ms"],k))
        q4ms=kern["times"][q4_kernel]["median_ms"]
        n8ms=kern["times"]["n8_repacked"]["median_ms"]
        if not (math.isfinite(q4ms) and math.isfinite(n8ms) and n8ms>0 and q4ms>0):
            raise RuntimeError("invalid actual kernel latency")
        ordered=sorted(subset,key=lambda x:(-x["squared_normalized_error_reduction_q4_minus_n8"],x["layer"]))
        gains=[float(x["squared_normalized_error_reduction_q4_minus_n8"]) for x in ordered]
        if any(not math.isfinite(x) or x<=0 for x in gains):
            raise RuntimeError("native source-only 8bit benefit nonpositive/drift")
        prefix=[0.]
        for x in gains:prefix.append(prefix[-1]+x)
        result[role]=dict(
            source_layer_ids=[int(x["layer"]) for x in ordered],
            prefix_benefit=prefix,
            Q4_kernel=q4_kernel,
            Q4_median_ms=q4ms,
            N8_kernel="n8_repacked_neon",
            N8_median_ms=n8ms,
            repack_at_load_ms=kern["n8_one_time_repack_ms"],
            extra_byte_cost=int(kern["n8_with_scales_bytes"]-kern["q4_with_scales_bytes"]),
            extra_median_ms=float(n8ms-q4ms),
        )
    return result

def build_optimizer(roledata):
    # Per-bit memory costs: source-verified group64 Q4 scale + nibble arrays.
    q4_memory=0
    scan=read_secure(QSCAN,EXPECTED_SCAN)
    for row in scan["rows"]:q4_memory+=int(row["q4_with_scales_bytes"])
    if q4_memory!=736985088:raise RuntimeError("incorrect known Qwen Q4 projection byte budget")
    deltas=[roledata[r]["extra_median_ms"] for r in ROLES]
    if any(x<=0 for x in deltas):
        raise RuntimeError("reported n8 kernel unexpectedly faster; recalibrate model before selection")
    gamma=(REFERENCE_ROLE4_MS-NATIVE_BASE_Q4_MS)/(4*sum(deltas))
    if not 0.5<=gamma<=2.0:
        raise RuntimeError("old full-model latency not consistent with role microbench")
    base_proxy=sum(float(row["q4_relative_l2"])**2 for row in scan["rows"])
    return q4_memory,gamma,base_proxy

def exhaustive_groups(roledata,q4_memory,gamma):
    """Provably exhaustive over all 7-tuple integer counts summing to 28.

    For a role with k upgraded tensors, choosing the top-k source errors
    maximizes the chosen objective because role shape/kernel costs are equal.
    No gradient/heuristic decisions are made inside this limited objective.
    """
    prefix=[roledata[r]["prefix_benefit"] for r in ROLES]
    bytes_delta=[roledata[r]["extra_byte_cost"] for r in ROLES]
    time_delta=[roledata[r]["extra_median_ms"] for r in ROLES]
    budget=MEMORY_MAX-q4_memory
    best={"kernel_aware":None,"memory_only":None}
    stats=dict(all_count_compositions=0,memory_feasible=0,kernel_time_feasible=0)
    def choose(name,score,counts,deltaB,deltaT):
        existing=best[name]
        # deterministic tie-break: lexicographically smaller role counts
        if existing is None or score>existing["source_proxy_gain"]+1e-12 or (
            abs(score-existing["source_proxy_gain"])<=1e-12 and counts<tuple(existing["counts"])
        ):
            best[name]=dict(counts=list(counts),source_proxy_gain=float(score),
              native_approx_total_decode_ms=float(NATIVE_BASE_Q4_MS+gamma*deltaT),
              packed_projection_memory_bytes=q4_memory+deltaB,
              extra_kernel_cost_ms=float(deltaT),
              source_proxy_tie_break="lexicographic role-count ascending")
    def traverse(index,remaining,counts,score,deltaB,deltaT):
        if deltaB>budget:
            return
        if index==len(ROLES)-1:
            count=remaining
            cb=deltaB+count*bytes_delta[index]
            if cb>budget:return
            ct=deltaT+count*time_delta[index]
            gain=score+prefix[index][count]
            chosen=tuple(counts)+(count,)
            stats["all_count_compositions"]+=1
            stats["memory_feasible"]+=1
            choose("memory_only",gain,chosen,cb,ct)
            if NATIVE_BASE_Q4_MS+gamma*ct<=KERNEL_AWARE_MAX_MS+1e-9:
                stats["kernel_time_feasible"]+=1
                choose("kernel_aware",gain,chosen,cb,ct)
            return
        for count in range(remaining+1):
            cb=deltaB+count*bytes_delta[index]
            if cb>budget:break
            traverse(index+1,remaining-count,counts+[count],
                score+prefix[index][count],cb,deltaT+count*time_delta[index])
    traverse(0,NUM_UPGRADES,[],0.,0,0.)
    if best["memory_only"] is None or best["kernel_aware"] is None:
        raise RuntimeError("policy infeasible; do not silently relax prereg constraints")
    return best,stats,gamma

def policy(roledata,desc):
    out=[]
    for role,count in zip(ROLES,desc["counts"]):
        out.extend((role,idx) for idx in roledata[role]["source_layer_ids"][:count])
    if len(out)!=NUM_UPGRADES or len(set(out))!=NUM_UPGRADES:
        raise RuntimeError("not exactly 28 unique role-layer tensor upgrades")
    return sorted(out,key=lambda x:(ROLES.index(x[0]),x[1]))

def put(path:Path,blob:bytes):
    with path.open("xb") as f:
        f.write(blob);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)

def run():
    if TARGET.exists():raise FileExistsError("allocator output already exists; do not fit after seeing model metrics")
    scan=read_secure(QSCAN,EXPECTED_SCAN)
    profile=read_secure(PROFILE,EXPECTED_PROFILE)
    if profile["nroles"]!=7 or profile["source_BF16_snapshot"]!="4ae77cb209f06199b8df1c94e21ff341332a3a89":
        raise RuntimeError("not pinned BF16 original source")
    roledata=mk_roles(scan,profile)
    q4bytes,gamma,base_proxy=build_optimizer(roledata)
    policies,stats,_=exhaustive_groups(roledata,q4bytes,gamma)
    os.umask(0o077)
    TARGET.mkdir(parents=True,mode=0o700)
    policy_results={}
    for name,p in policies.items():
        selected=policy(roledata,p)
        data="".join(f"{role} {layer}\n" for role,layer in selected).encode()
        file=TARGET/f"{name}_n8_roles.txt"
        put(file,data)
        p["policy_sha256"]=sha(data)
        p["n8_upgraded_role_layers"]={r:sorted(l for x,l in selected if x==r) for r in ROLES}
        p["decoder_and_precision_by_role"]={r:dict(bits4_kernel=roledata[r]["Q4_kernel"],
             bits8_kernel=roledata[r]["N8_kernel"],
             projected_decode_4_ms=roledata[r]["Q4_median_ms"],
             projected_decode_8_ms=roledata[r]["N8_median_ms"]) for r in ROLES}
        p["source_proxy_expected_residual_error"]=base_proxy-p["source_proxy_gain"]
        policy_results[name]=p
    if not policies["kernel_aware"]["native_approx_total_decode_ms"]<=KERNEL_AWARE_MAX_MS:
        raise RuntimeError("latency ceiling violated")
    result=dict(schema="BEGLIN-Q-kernel-aware-196-exact-source-only-allocator/1",
        status="SELECTED_AND_FROZEN_BEFORE_NATIVE_MODEL_EVALUATION",
        fixed_n8_promotions=NUM_UPGRADES,
        fixed_n4_count=196-NUM_UPGRADES,
        memory_ceiling_bytes=MEMORY_MAX,
        kernel_aware_predicted_latency_ceiling_ms=KERNEL_AWARE_MAX_MS,
        old_Q4_fullmodel_single_token_ms=NATIVE_BASE_Q4_MS,
        old_role_balanced_mixed_native_single_token_ms=REFERENCE_ROLE4_MS,
        role_kernel_calibration_gamma=gamma,
        original_196_source_weight_scan_sha256=EXPECTED_SCAN,
        native_role_cost_profile_sha256=EXPECTED_PROFILE,
        source_original_BF16_not_modified=True,
        exact_solver_enumeration_stats=stats,
        public_test_inputs_opened=False,
        final_V6_24_used_rows_still_unknown=True,
        source_only_proxy_not_actual_behavior_sensitivity=True,
        full_model_latency_prediction_not_guaranteed=True,
        policies=policy_results,
        production_main_untouched=True)
    output=TARGET/"kernel_aware_source_only_selection.json"
    put(output,(json.dumps(result,indent=2,sort_keys=True)+"\n").encode())
    print("BEGLIN_KERNEL_AWARE_POLICY_FROZEN",json.dumps({
        "selection_sha256":sha(output.read_bytes()),
        "gamma":gamma,"policies":{
          name:dict(counts=policies[name]["counts"],
                bytes=policies[name]["packed_projection_memory_bytes"],
                predicted_ms=policies[name]["native_approx_total_decode_ms"],
                gain=policies[name]["source_proxy_gain"],sha=policies[name]["policy_sha256"])
          for name in policies},"stats":stats},sort_keys=True),flush=True)
if __name__=="__main__":run()
