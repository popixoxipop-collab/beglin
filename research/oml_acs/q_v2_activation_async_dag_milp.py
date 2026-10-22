#!/usr/bin/env python3
"""BEGLIN Q v2: source-only activation-aware, async-DAG mixed-precision MILP.

196 original Qwen BF16 tensors, each 4bit Q4G64 or 8bit qNg64 repacked.
Measured Q/K/V and gate/up joint stage wall timings enter each of 128
per-layer mixed-bit masks; o/down remain serial. The source-BF16 input
activation diagonal score is an offline LOCAL proxy, not true model behavior.
Only prerecorded 12 calibration public prompts were used in the scores;
DO NOT read the 8 evaluation prompts or any actual new model outputs here.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,time
from collections import Counter
from itertools import product
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import coo_matrix

ROOT=pathlib.Path(__file__).resolve().parents[2]
V2=ROOT/"configs/research/BEGLIN_QV2_ASYNC_DAG_MILP_PRELOCK_20261022.json"
QUALITY=ROOT/"docs/research/BEGLIN_Q_V2_BF16_ACTIVATION_LOCAL_AGGREGATE_20261022.json"
STAGES=ROOT/"docs/research/BEGLIN_Q_V2_ASYNC_NATIVE_STAGE_COSTS_20261022.json"
ROLES_COST=ROOT/"docs/research/BEGLIN_Q_KERNEL_AWARE_ALLOCATOR_NATIVE_EVIDENCE_20261022.json"
LAB=pathlib.Path.home()/".local/share/beglin/oml_acs/20261022_qv2_dag_joint_milp_prelocked"
ROLE_ORDER=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
OLD_ROLE_RANDOM_SHA="72065ac7a4ae5d68b1c7944dc4a67e40ac9c3c825db54a1d441ca3002ec015c5"
ACTIVATION_SHA="ee7b177b4dde984a5d83e0cdb5983d2ae00c34ee752fef9113a79568a1d12663"
ASYNC_SHA="380670d42ccd2f12c4a015600be3d921f672c570a3d3a4a1652c61e217791808"
ROLE_SHA="280ca338745d0ebb2c44986d18e8b240135fe7fa4b1077f49f65f5d8fb3eff93"
N_LAYERS=28
N_MASKS=128

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def load(path:pathlib.Path)->tuple[dict,str]:
    if not path.is_file():raise RuntimeError("missing pinned source "+str(path))
    blob=path.read_bytes()
    return json.loads(blob),sha(blob)
def put(path:pathlib.Path,data:bytes):
    if path.exists():raise FileExistsError("overwrite forbidden "+str(path))
    with path.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)

def mask_bits(mask:int)->tuple[int,...]:
    return tuple(8 if (mask>>(6-i))&1 else 4 for i in range(7))
def collect_profiles():
    cfg,cfgsha=load(V2)
    q,qsha=load(QUALITY)
    stages,stsha=load(STAGES)
    cost,costsha=load(ROLES_COST)
    assert cfg["status"]=="FROZEN_BEFORE_POLICY_AND_HELD_EVAL"
    assert cfg["role_order"]==list(ROLE_ORDER) and cfg["choices_per_layer"]==128
    assert q["private_activation_moment_sha256"]==ACTIVATION_SHA
    assert q["tensor_count"]==196 and q["source_BF16_revision"]==cfg["revision"]
    assert stages["private_sha256"]==ASYNC_SHA and len(stages["stages"]["qkv"])==8
    assert len(stages["stages"]["ffn_gu"])==4
    assert cost["source_7role_kernel_profile_sha256"]==ROLE_SHA
    role_cost={row["role"]:row for row in cost["role_kernel_cost_measured"]}
    assert set(role_cost)==set(ROLE_ORDER)
    gains=q["quality_upgrade_by_role_layer"]
    assert set(gains)==set(ROLE_ORDER)
    assert all(len(gains[r])==28 and min(gains[r])>0 for r in ROLE_ORDER)
    return cfg,q,stages,role_cost,(cfgsha,qsha,stsha,costsha)

def native_stage(bits:tuple[int,...],stage:dict,rolecost:dict):
    q="".join(str(x) for x in bits[:3])
    gu="".join(str(x) for x in bits[4:6])
    qstage=stage["qkv"][q];gustage=stage["ffn_gu"][gu]
    def cost(index):
        r=rolecost[ROLE_ORDER[index]]
        return float(r["q4_neon_6thread_median_ms"] if bits[index]==4 else r["n8_repacked_neon_median_ms"])
    serial=qstage["serial_q4_6t_ms"]+gustage["serial_q4_6t_ms"]+cost(3)+cost(6)
    parallel=qstage["best_ms"]+gustage["best_ms"]+cost(3)+cost(6)
    assert parallel<=serial+1e-9
    return float(serial),float(parallel)

def calibrate(stage:dict,rolecost:dict):
    rng=np.random.default_rng(73)
    old_random={}
    for role in ROLE_ORDER:
        old_random[role]=tuple(sorted(int(x) for x in rng.choice(N_LAYERS,4,replace=False)))
    old_txt="".join(f"{role} {idx}\n" for role in ROLE_ORDER for idx in old_random[role])
    if sha(old_txt.encode())!=OLD_ROLE_RANDOM_SHA:
        raise RuntimeError("frozen seed73 random control changed")
    all4=(4,)*7
    base_serial,_=native_stage(all4,stage,rolecost)
    base_total=base_serial*N_LAYERS
    random_total=0.
    for layer in range(N_LAYERS):
        row=tuple(8 if layer in old_random[r] else 4 for r in ROLE_ORDER)
        serial,_=native_stage(row,stage,rolecost)
        random_total+=serial
    diff=random_total-base_total
    if diff<=0 or not np.isfinite(diff):
        raise RuntimeError("invalid true native stage calibration")
    gamma=(116.18-75.26)/diff
    if not 0.05<gamma<10:
        raise RuntimeError("implausible cost-to-whole-model calibration")
    return gamma,base_total,old_random

def option_bank(scores,stage,roles):
    basebytes=sum(int(roles[r]["q4_with_scales_bytes"])*28 for r in ROLE_ORDER)
    assert basebytes==736985088
    choices=[]
    for mask in range(N_MASKS):
        bits=mask_bits(mask)
        n8=sum(b==8 for b in bits)
        byte_delta=sum(int(roles[r]["n8_with_scales_bytes"]-
                roles[r]["q4_with_scales_bytes"]) for r,b in zip(ROLE_ORDER,bits) if b==8)
        ser,async_ms=native_stage(bits,stage["stages"],roles)
        choices.append((bits,n8,byte_delta,ser,async_ms))
    quality=np.empty((28,N_MASKS),dtype=np.float64)
    for layer in range(28):
        for mask,item in enumerate(choices):
            bits=item[0]
            quality[layer,mask]=sum(scores[r][layer] for r,b in zip(ROLE_ORDER,bits) if b==8)
    if not np.isfinite(quality).all():
        raise RuntimeError("nonfinite source-only activation quality proxy")
    return basebytes,choices,quality

def solve_one(name:str,cfg:dict,basebytes:int,choices:list,quality:np.ndarray,
              gamma:float,base_serial_total:float)->dict:
    if name=="async_DAG":
        serial_ceiling=float(cfg["primary_limits"]["native_serial_predicted_ms_max"])
        async_ceiling=float(cfg["primary_limits"]["asynchronous_DAG_predicted_ms_max"])
    elif name=="serial_only":
        serial_ceiling=float(cfg["control_limits"]["native_serial_predicted_ms_max"])
        async_ceiling=None
    else:raise ValueError(name)
    nvar=N_LAYERS*N_MASKS
    rix=[];cix=[];vals=[]
    for layer in range(N_LAYERS):
        for mask in range(N_MASKS):
            idx=layer*N_MASKS+mask
            bits,n8,delta,serial,asyncms=choices[mask]
            rix += [layer,N_LAYERS,N_LAYERS+1,N_LAYERS+2,N_LAYERS+3]
            cix += [idx]*5
            vals += [1.,float(n8),float(delta)/1e6,
                     gamma*(serial-choices[0][3]),gamma*(asyncms-choices[0][3])]
    A=coo_matrix((np.asarray(vals,dtype=np.float64),(rix,cix)),
                 shape=(N_LAYERS+4,nvar)).tocsr()
    lb=np.concatenate([np.ones(N_LAYERS),[28,-np.inf,-np.inf,-np.inf]])
    upper_async=np.inf if async_ceiling is None else async_ceiling-75.26
    ub=np.concatenate([np.ones(N_LAYERS),[28,(cfg["projection_memory_max_bytes"]-basebytes)/1e6,
                       serial_ceiling-75.26,upper_async]])
    options=dict(time_limit=150.0,mip_rel_gap=1e-6,presolve=True)
    t=time.monotonic()
    sol=milp(c=-quality.reshape(-1),integrality=np.ones(nvar,dtype=int),
             bounds=Bounds(np.zeros(nvar),np.ones(nvar)),
             constraints=LinearConstraint(A,lb,ub),options=options)
    elapsed=time.monotonic()-t
    if sol.x is None or sol.status!=0 or not bool(sol.success):
        raise RuntimeError(f"MILP {name} not certified optimal: status={sol.status}; message={sol.message}; mip_gap={getattr(sol,'mip_gap',None)}")
    grid=sol.x.reshape((N_LAYERS,N_MASKS))
    if np.max(np.abs(grid-np.round(grid)))>1e-5:
        raise RuntimeError("MILP not integral")
    masked=[int(np.argmax(row)) for row in grid]
    if any(np.sum(np.round(row))!=1 for row in grid):
        raise RuntimeError("not exactly one native kernel/bitmask per layer")
    selected=[(r,layer) for layer,mask in enumerate(masked)
              for r,b in zip(ROLE_ORDER,choices[mask][0]) if b==8]
    assert len(selected)==28 and len(set(selected))==28
    extra_bytes=sum(choices[k][2] for k in masked)
    measured_mem=basebytes+extra_bytes
    serial=75.26+gamma*sum(choices[k][3]-choices[0][3] for k in masked)
    parallel=75.26+gamma*sum(choices[k][4]-choices[0][3] for k in masked)
    if measured_mem>cfg["projection_memory_max_bytes"] or serial>serial_ceiling+1e-5 or (
        async_ceiling is not None and parallel>async_ceiling+1e-5):
        raise RuntimeError("MILP native constraints violated")
    gain=float(sum(quality[layer,mask] for layer,mask in enumerate(masked)))
    if abs(gain-(-sol.fun))>1e-6:raise RuntimeError("objective inconsistent")
    policy_txt="".join(f"{r} {layer}\n" for r in ROLE_ORDER
                  for layer in sorted(l for role,l in selected if role==r))
    return dict(name=name,status="MILP_OPTIMAL_CERTIFIED",native_mask_for_28_layers=masked,
      n8_role_layer_pairs=selected,
      n8_role_counts={r:sum(x==r for x,l in selected) for r in ROLE_ORDER},
      source_activation_local_quality_gain=gain,
      projection_bytes=measured_mem,
      predicted_native_serial_ms=float(serial),
      predicted_hypothetical_parallel_graph_ms=float(parallel),
      native_QKV_FFN_stage_dispatch_policy="each layer selects its measured best native serial or independent Q4 threadpool parallel mode",
      frozen_policy_sha256=sha(policy_txt.encode()),
      frozen_policy_text=policy_txt,
      solver=dict(status=int(sol.status),message=str(sol.message),
                  mip_gap=float(getattr(sol,"mip_gap",0) or 0),
                  mip_dual_bound=float(getattr(sol,"mip_dual_bound",0) or 0),
                  node_count=int(getattr(sol,"mip_node_count",0) or 0),
                  elapsed_seconds=round(elapsed,3)))

def main():
    if LAB.exists():raise FileExistsError("cannot overwrite source-only MILP policies")
    cfg,quality,stages,roles,shas=collect_profiles()
    gamma,base_total,old_random=calibrate(stages["stages"],roles)
    basebytes,choices,score=option_bank(quality["quality_upgrade_by_role_layer"],stages,roles)
    results=[]
    for name in ("async_DAG","serial_only"):
        a=solve_one(name,cfg,basebytes,choices,score,gamma,base_total)
        results.append(a)
        print("BEGLIN_QV2_MILP_SOURCE_FROZEN",name,json.dumps({
          "counts":a["n8_role_counts"],"predicted_serial_ms":a["predicted_native_serial_ms"],
          "predicted_async_ms":a["predicted_hypothetical_parallel_graph_ms"],
          "proxy_gain":a["source_activation_local_quality_gain"],"bytes":a["projection_bytes"],
          "gap":a["solver"]["mip_gap"],"solver_elapsed":a["solver"]["elapsed_seconds"],
          "policy_sha256":a["frozen_policy_sha256"]},sort_keys=True),flush=True)
    os.umask(0o077);LAB.mkdir(parents=True,mode=0o700)
    for result in results:
        blob=result.pop("frozen_policy_text").encode()
        put(LAB/(result["name"]+"_policy.txt"),blob)
        assert sha(blob)==result["frozen_policy_sha256"]
    manifest=dict(schema="BEGLIN-QV2-activation-aware-async-DAG-MILP/1",
      status="OPTIMAL_SOURCE_ONLY_POLICIES_FROZEN_BEFORE_EVAL",
      config_schema=cfg["schema"],
      original_source_sha256=dict(zip(("prelock","quality_scores","stage_profile","role_profile"),shas)),
      original_activation_calibration_sha256=ACTIVATION_SHA,
      original_async_native_stage_sha256=ASYNC_SHA,
      actual_original_BF16_source_revision=cfg["revision"],
      original_q4_baseline_native_ms=75.26,
      original_frozen_role_balanced_random_ms=116.18,
      native_cost_scale_gamma=float(gamma),
      layer_choices_per_decoder=128,
      total_binary_decision_variables=28*128,
      original_q4_projection_bytes=basebytes,
      total_upgraded_n8=28,
      role_order=ROLE_ORDER,
      policies=results,
      eval_prompts_not_loaded_for_selection=True,
      blocked_fullmodel_per_tensor_interventions_not_retried=True,
      production_model_unchanged=True,
      scientific_quality_verdict="NO_VERDICT")
    receipt=LAB/"source_only_DAG_MILP_receipt.json"
    put(receipt,(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode())
    public=dict(manifest)
    pub=LAB/"source_only_DAG_MILP_public_manifest.json"
    put(pub,(json.dumps(public,indent=2,sort_keys=True)+"\n").encode())
    print("BEGLIN_QV2_MILP_POLICIES_SHA_BOUND",json.dumps({
      "private_sha256":sha(receipt.read_bytes()),
      "public_aggregate_sha256":sha(pub.read_bytes()),
      "policies":{x["name"]:x["frozen_policy_sha256"] for x in results},
      "changed_role_assignment":results[0]["n8_role_counts"]!=results[1]["n8_role_counts"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
