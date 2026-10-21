#!/usr/bin/env python3
"""BEGLIN Qv2 source-only activation-aware and asynchronous-stage MILP.

448 binary choices (28 layers * [8 QKV + 4 gate/up + 2 O + 2 down]).
Each layer chooses one precision tuple per stage. Quality objective uses
original BF16 pre-calibration local *diagonal* activation-weighted errors.
Kernel cost uses measured Apple native concurrent QKV/gate-up wall time,
including actual independent 1/6t q4 pools and contention. Both a serial
upper-budget and a *hypothetical* async-DAG budget are enforced.
Current full BEGLIN model is still sequential! No claims of realized DAG TPS.

No model prompts, checkpoints, labels, evaluation logits or hidden activation
arrays are accessed. Only preregistered 196 scalar gains + 7 kernel role costs
+ 12 measured stage-pattern costs. Strict HiGHS optimality required.
"""
from __future__ import annotations
import hashlib,itertools,json,math,os,pathlib,time
from collections import Counter
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_array

ROOT=pathlib.Path(__file__).resolve().parents[2]
QUALITY=ROOT/"docs/research/BEGLIN_QV2_ACTIVATION_LOCAL_196_SOURCE_AGGREGATE_20261022.json"
ASYNC=ROOT/"docs/research/BEGLIN_QV2_ASYNC_STAGE_REAL_COST_AGGREGATE_20261022.json"
OLD=ROOT/"docs/research/BEGLIN_Q_KERNEL_AWARE_ALLOCATOR_NATIVE_EVIDENCE_20261022.json"
PREREG=ROOT/"configs/research/BEGLIN_QV2_ACTIVATION_ASYNC_MILP_PRELOCK_20261022.json"
PRIVATE=pathlib.Path.home()/".local/share/beglin/oml_acs/20261022_qv2_joint_policy"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
STAGES=(("qkv",("q_proj","k_proj","v_proj")),
        ("ffn_gu",("gate_proj","up_proj")),
        ("o",("o_proj",)),
        ("down",("down_proj",)))
EXPECTED_BLOBS={
 "quality":"f58672e001a447cd82b15faa0a06cfdc25e60452",
 "async":"a029f55f175610ebba294353663f02e4e7219948",
 "old":"6474d8d1c974935f87f5c46ec49b8f9597360d6a"
}
MODELREV="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SOURCE_MOMENTS_SHA="ee7b177b4dde984a5d83e0cdb5983d2ae00c34ee752fef9113a79568a1d12663"
SOURCE_ASYNC_SHA="380670d42ccd2f12c4a015600be3d921f672c570a3d3a4a1652c61e217791808"
Q4_BASELINE_MS=75.26
GAMMA=1.039406406482483
MAX_MEMORY=830570496
UPGRADES=28
MAX_SERIAL_MS=100.
MAX_ASYNC_CANDIDATE_MS=80.

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def source_json(path,blob):
    if not path.is_file():raise RuntimeError("missing preregistered aggregate: "+str(path))
    import subprocess
    actual=subprocess.run(["git","hash-object",str(path)],capture_output=True,text=True,check=True).stdout.strip()
    if actual!=blob:raise RuntimeError("original GitHub aggregate blob changed "+str(path))
    return json.loads(path.read_text())
def secure_write(path,content):
    with path.open("xb") as f:
        f.write(content);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)

def prepare_sources():
    pre=json.loads(PREREG.read_text())
    c=pre["optimization"]["constraints"]
    if (c["num_n8_exactly"],c["max_projection_weight_bytes"],c["max_serial_predicted_decode_ms"],
        c["max_parallel_candidate_DAG_predicted_decode_ms"])!=(UPGRADES,MAX_MEMORY,MAX_SERIAL_MS,MAX_ASYNC_CANDIDATE_MS):
        raise RuntimeError("original locked MILP constraints unexpectedly changed")
    quality=source_json(QUALITY,EXPECTED_BLOBS["quality"])
    stages=source_json(ASYNC,EXPECTED_BLOBS["async"])
    prev=source_json(OLD,EXPECTED_BLOBS["old"])
    if quality["source_BF16_revision"]!=MODELREV or quality["private_activation_moment_sha256"]!=SOURCE_MOMENTS_SHA:
        raise RuntimeError("original BF16 calibration source drift")
    if quality["tensor_count"]!=196 or quality["test_prompts_not_used"] is not True:
        raise RuntimeError("not original 196 source-only quality gains")
    if stages["private_receipt_SHA256"]!=SOURCE_ASYNC_SHA:
        raise RuntimeError("not independently measured real native stage costs")
    if prev["source_196_weights_sha256"]!="720de694838d3a85156afae3732974f3867db1c8ce4957e55778da77724a0d8a":
        raise RuntimeError("BEGLIN native original quant format contract changed")
    arr=quality["quality_upgrade_by_role_layer"]
    if set(arr)!=set(ROLES) or any(len(arr[r])!=28 for r in ROLES):
        raise RuntimeError("196 original role-layer activation benefits incomplete")
    for role in ROLES:
        if not all(np.isfinite(v) and v>0 for v in arr[role]):
            raise RuntimeError("bad source BF16 activation local output benefit")
    kc={r["role"]:r for r in prev["role_kernel_cost_measured"]}
    if set(kc)!=set(ROLES):raise RuntimeError("native kernel costs missing")
    return pre,quality,stages,kc

def build_options(quality,stages,kernel):
    benefits=quality["quality_upgrade_by_role_layer"]
    async_stages=stages["stages"]
    qkv=async_stages["qkv"]
    gu=async_stages["ffn_gu"]
    if set(qkv)!={"".join(map(str,x)) for x in itertools.product((4,8),repeat=3)}:
        raise RuntimeError("not eight measured native QKV stage combinations")
    if set(gu)!={"".join(map(str,x)) for x in itertools.product((4,8),repeat=2)}:
        raise RuntimeError("not four measured native FFN gate/up combos")
    roleinfo={}
    for role,k in kernel.items():
        q4options=[("q4_1t",k["q4_neon_1thread_median_ms"]),
                   ("q4_6t",k["q4_neon_6thread_median_ms"])]
        chosen=min(q4options,key=lambda x:(x[1],x[0]))
        x=tuple(int(k[n]) for n in ("q4_with_scales_bytes","n8_with_scales_bytes"))
        roleinfo[role]=dict(q4_cost=float(chosen[1]),q4_kernel=chosen[0],
                            n8_cost=float(k["n8_repacked_neon_median_ms"]),
                            n8_kernel="n8_repacked_NEON",
                            additional_n8_bytes=x[1]-x[0],q4_weight_bytes=x[0])
        if x[1]<=x[0]:raise RuntimeError("invalid Q4/N8 model memory cost")
    baseline_q4_bytes=28*sum(roleinfo[r]["q4_weight_bytes"] for r in ROLES)
    if baseline_q4_bytes!=736985088:raise RuntimeError("q4 native projection costs don't agree")
    baseline_serial_stage=(
      qkv["444"]["serial6_ms"]+gu["44"]["serial6_ms"]+
      roleinfo["o_proj"]["q4_cost"]+roleinfo["down_proj"]["q4_cost"])
    residual_cost=Q4_BASELINE_MS-28*baseline_serial_stage
    if not (10<residual_cost<50):
        raise RuntimeError("full model unmeasured nonprojection baseline is invalid")
    stages_opts={}
    for layer in range(28):
        for stage,roles in STAGES:
            opts=[]
            for bits in itertools.product((4,8),repeat=len(roles)):
                bitstring="".join(str(x) for x in bits)
                nup=sum(b==8 for b in bits)
                if stage=="qkv" or stage=="ffn_gu":
                    data=async_stages[stage][bitstring]
                    measured_cost=float(data["best_ms"])
                    mode=data["best_mode"]
                    if mode not in ("serial_q4_6t","parallel_q4_1t","parallel_q4_6t"):
                        raise RuntimeError("unknown native async stage mode")
                    if not (0<measured_cost<=float(data["serial6_ms"])+1e-6):
                        raise RuntimeError("asynchronous stage cost not measured or inconsistent")
                else:
                    role=roles[0]
                    k=roleinfo[role]
                    measured_cost=k["q4_cost"] if bits[0]==4 else k["n8_cost"]
                    mode=k["q4_kernel"] if bits[0]==4 else k["n8_kernel"]
                benefit=sum(float(benefits[r][layer]) for r,b in zip(roles,bits) if b==8)
                extra_bytes=sum(roleinfo[r]["additional_n8_bytes"] for r,b in zip(roles,bits) if b==8)
                serial_delta=sum(roleinfo[r]["n8_cost"]-roleinfo[r]["q4_cost"] for r,b in zip(roles,bits) if b==8)
                opts.append(dict(layer=layer,stage=stage,roles=roles,bits=bits,bitstring=bitstring,
                    source_quality_gain=benefit,n8_count=nup,memory_increase_bytes=extra_bytes,
                    predicted_serial_delta_ms=serial_delta,
                    async_stage_measured_ms=measured_cost,async_mode=mode))
            stages_opts[(layer,stage)]=opts
    assert len(stages_opts)==112 and sum(map(len,stages_opts.values()))==448
    return stages_opts,roleinfo,baseline_q4_bytes,residual_cost

def solve_milp(stages_opts,baseline_q4_bytes,residual_cost,async_limit=True):
    candidates=[o for key in stages_opts for o in stages_opts[key]]
    n=len(candidates)
    group_indices=[]
    offset=0
    for key,opts in stages_opts.items():
        group_indices.append(list(range(offset,offset+len(opts))))
        offset+=len(opts)
    assert offset==448
    # 112 one-choice equalities, exact upgrades, memory cap, serial prediction
    # cap, and optionally true measured-stage asynchronous-candidate cap.
    rows=[];cols=[];data=[]
    m=112+4
    low=np.full(m,-np.inf,dtype=np.float64)
    high=np.full(m,np.inf,dtype=np.float64)
    for i,indices in enumerate(group_indices):
        low[i]=1.;high[i]=1.
        for j in indices:rows.append(i);cols.append(j);data.append(1.)
    low[112]=UPGRADES;high[112]=UPGRADES
    high[113]=(MAX_MEMORY-baseline_q4_bytes)/1e6
    high[114]=(MAX_SERIAL_MS-Q4_BASELINE_MS)/GAMMA
    high[115]=MAX_ASYNC_CANDIDATE_MS-residual_cost if async_limit else np.inf
    for j,o in enumerate(candidates):
        for idx,value in ((112,o["n8_count"]),(113,o["memory_increase_bytes"]/1e6),
                          (114,o["predicted_serial_delta_ms"]),(115,o["async_stage_measured_ms"])):
            if value:
                rows.append(idx);cols.append(j);data.append(float(value))
    A=coo_array((np.array(data,dtype=np.float64),
        (np.array(rows,dtype=np.int32),np.array(cols,dtype=np.int32))),shape=(m,n)).tocsr()
    c=-np.asarray([x["source_quality_gain"] for x in candidates],dtype=np.float64)
    started=time.monotonic()
    result=milp(c,
        integrality=np.ones(n,dtype=np.int32),
        bounds=Bounds(np.zeros(n),np.ones(n)),
        constraints=LinearConstraint(A,low,high),
        options={"disp":False,"time_limit":120.,"mip_rel_gap":1e-8})
    elapsed=time.monotonic()-started
    if result.status!=0 or result.x is None or result.mip_gap is None or result.mip_gap>1e-6:
        raise RuntimeError(f"MILP did not certify optimal solution: status={result.status}, gap={result.mip_gap}, message={result.message}")
    discrete=np.round(result.x).astype(np.int8)
    if not np.allclose(discrete,result.x,atol=1e-7):
        raise RuntimeError("solver returned fractional binary choice")
    chosen=[candidates[i] for i,x in enumerate(discrete) if x]
    if len(chosen)!=112:
        raise RuntimeError("not one stage assignment per decoder layer")
    upgrade=sum(o["n8_count"] for o in chosen)
    bytes_used=baseline_q4_bytes+sum(o["memory_increase_bytes"] for o in chosen)
    serial_ms=Q4_BASELINE_MS+GAMMA*sum(o["predicted_serial_delta_ms"] for o in chosen)
    async_ms=residual_cost+sum(o["async_stage_measured_ms"] for o in chosen)
    benefit=sum(o["source_quality_gain"] for o in chosen)
    if upgrade!=28 or bytes_used>MAX_MEMORY+1 or serial_ms>MAX_SERIAL_MS+1e-5:
        raise RuntimeError("program result violates hard source-only constraints")
    if async_limit and async_ms>MAX_ASYNC_CANDIDATE_MS+1e-5:
        raise RuntimeError("program violated hypothetical async-DAG constraint")
    tensor_selected=set()
    schedule=[]
    for o in chosen:
        for r,b in zip(o["roles"],o["bits"]):
            if b==8:tensor_selected.add((r,o["layer"]))
        schedule.append(dict(layer=o["layer"],stage=o["stage"],
            bits=o["bitstring"],native_stage_schedule=o["async_mode"],
            observed_stage_cost_ms=o["async_stage_measured_ms"]))
    if len(tensor_selected)!=28:
        raise RuntimeError("duplicate layer-role upgrade")
    sorted_policy=sorted(tensor_selected,key=lambda x:(ROLES.index(x[0]),x[1]))
    return dict(
        count28=upgrade,bytes=bytes_used,serial_prediction_ms=serial_ms,
        async_candidate_DAG_prediction_ms=async_ms,
        activation_local_proxy_gain=benefit,
        role_counts={r:sum(a==r for a,b in sorted_policy) for r in ROLES},
        selected_roles=sorted_policy,
        per_layer_stage_schedule=schedule,
        solver_status=result.status,solver_mip_gap=float(result.mip_gap),
        solver_dual_bound=float(result.mip_dual_bound),
        solver_rel_gap=float(result.mip_gap),
        solver_seconds=elapsed,
        objective_proxy=-float(result.fun)
    )

def run():
    if PRIVATE.exists():raise FileExistsError("policy already frozen; will not tune against evaluation")
    pre,quality,stages,kernel=prepare_sources()
    options,roleinfo,base_bytes,overhead=build_options(quality,stages,kernel)
    actual=solve_milp(options,base_bytes,overhead,async_limit=True)
    ablate=solve_milp(options,base_bytes,overhead,async_limit=False)
    if abs(actual["activation_local_proxy_gain"]-actual["objective_proxy"])>1e-7:
        raise RuntimeError("MILP objective cannot be independently recalculated")
    if ablate["activation_local_proxy_gain"]+1e-9<actual["activation_local_proxy_gain"]:
        raise RuntimeError("removing constraint should not reduce optimum")
    os.umask(0o077);PRIVATE.mkdir(parents=True,mode=0o700)
    for name,r in (("joint_async",actual),("no_async_constraint",ablate)):
        b="".join(f"{role} {layer}\n" for role,layer in r["selected_roles"]).encode()
        path=PRIVATE/f"{name}_n8_policy.txt"
        secure_write(path,b)
        r["policy_SHA256"]=sha(b)
    receipt=dict(
        schema="BEGLIN-Q-v2-activation-plus-async-MILP/1",
        status="FROZEN_OPTIMAL_SOURCE_ONLY_BEFORE_NEW_EVAL",
        model_revision=MODELREV,
        source_quality_activation_sha=SOURCE_MOMENTS_SHA,
        source_async_private_receipt_sha=SOURCE_ASYNC_SHA,
        original_component_git_blobs=EXPECTED_BLOBS,
        exact_28_upgrades=True,
        original_projection_Q4_bytes=base_bytes,
        native_Q4_full_model_serial_reference_ms=Q4_BASELINE_MS,
        stage_base_residual_ms=overhead,
        constraints={"num_8bit":28,"memory_bytes":MAX_MEMORY,
            "serial_model_ms":MAX_SERIAL_MS,"hypothetical_parallel_DAG_ms":MAX_ASYNC_CANDIDATE_MS},
        policies={"joint_async":actual,"no_async_constraint":ablate},
        selection_uses_prelocked_source_only_prompts=True,
        native_full_Qwen_still_executes_sequentially=True,
        no_prior_blocked_output_interventions=True,
        no_old_V6_final_data_used=True,
        native_model_not_run_on_evaluation_prompts=True,
        scientific_quality_claim="NO_VERDICT"
    )
    path=PRIVATE/"qv2_source_only_policy_selection_private.json"
    secure_write(path,(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_QV2_JOINT_MILP_FROZEN",json.dumps({
        "receipt_SHA256":sha(path.read_bytes()),
        "counterfactual_async_constraint_changes":actual["policy_SHA256"]!=ablate["policy_SHA256"],
        "joint":{k:actual[k] for k in ("policy_SHA256","role_counts","bytes","serial_prediction_ms",
           "async_candidate_DAG_prediction_ms","activation_local_proxy_gain","solver_mip_gap")},
        "no_async_constraint":{k:ablate[k] for k in ("policy_SHA256","role_counts","bytes",
           "serial_prediction_ms","async_candidate_DAG_prediction_ms","activation_local_proxy_gain","solver_mip_gap")}}
        ,sort_keys=True),flush=True)
if __name__=="__main__":run()
