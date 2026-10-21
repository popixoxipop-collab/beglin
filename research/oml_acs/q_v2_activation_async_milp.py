#!/usr/bin/env python3
"""BEGLIN Qv2 196-tensor activation/real-async-DAG MILP bit+kernel selector.

SciPy HiGHS exact binary MILP: one of 128 mixed q4/n8 role masks per
Qwen decoder layer, 28 layers, 3584 choices. Optimize only ORIGINAL BF16
source calibration activation-diagonal local error reduction under:
28 total n8 upgrades, physical projection bytes, previous native serial
latency ceiling, and empirical Apple QKV+gate/up async-stage DAG ceiling.
Async DAG applies ONLY to a future stage executor, not current full model.

This script never reads heldout evaluation prompts or outputs.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,time,math
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
ROOT=pathlib.Path(__file__).resolve().parents[2]
P=ROOT/"configs/research/BEGLIN_QV2_196_DAG_QUALITY_MILP_PRELOCK_20261022.json"
G=ROOT/"docs/research/BEGLIN_Q_V2_ACTIVATION_AWARE_LOCAL_SCORES_20261022.json"
A=ROOT/"docs/research/BEGLIN_Q_V2_REAL_ASYNC_STAGE_COSTS_20261022.json"
R=ROOT/"docs/research/BEGLIN_Q_KERNEL_AWARE_ALLOCATOR_NATIVE_EVIDENCE_20261022.json"
OUT=pathlib.Path.home()/".local/share/beglin/oml_acs/20261022_qv2_milp"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
L=28
M=128
EXPECTED={
 "quality_private":"7b3ad80294275807cb3775601915b18ec3caeeef430849d8cea48389b3623616",
 "stage_private":"380670d42ccd2f12c4a015600be3d921f672c570a3d3a4a1652c61e217791808",
 "role_native_private":"e634f73fd78bf35181d5cfd8aafc54675b8aa810182a1163483423f0c9da6769"
}
BASE_MS=75.26
MAX_SERIAL_MS=100.0
MAX_ASYNC_MS=90.0
MAX_BYTES=830570496
PROMOTIONS=28
ROLE_SCALING_GAMMA=1.039406406482483
EXPECTED_SOURCE_SNAPSHOT="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def load():
    for x in (P,G,A,R):
        if not x.is_file():raise RuntimeError("preregistered source aggregate missing "+str(x))
    conf=json.loads(P.read_text())
    if conf["status"]!="FROZEN_BEFORE_NEW_POLICY_AND_NATIVE_EVALUATION":
        raise RuntimeError("source policy freeze absent")
    if conf["capacity"]!={"n8_exact":28,"n4_exact":168,
       "projection_bytes_max":MAX_BYTES,
       "serial_native_model_predicted_ms_max":MAX_SERIAL_MS,
       "async_DAG_hypothetical_predicted_ms_max":MAX_ASYNC_MS}:
        raise RuntimeError("MILP constraint prereg changed")
    q=json.loads(G.read_text());a=json.loads(A.read_text());r=json.loads(R.read_text())
    if q["private_196_source_score_receipt_sha256"]!=EXPECTED["quality_private"]:
        raise RuntimeError("source BF16 calibration/quality score provenance mismatch")
    if a["source_private_receipt_sha256"]!=EXPECTED["stage_private"]:
        raise RuntimeError("actual native Apple CPU async-stage evidence changed")
    if r["source_private_audit_sha256"]!=EXPECTED["role_native_private"]:
        raise RuntimeError("actual native 7-role Apple kernel profile changed")
    if q["source_BF16_revision"]!=EXPECTED_SOURCE_SNAPSHOT:
        raise RuntimeError("original source model Qwen revision changed")
    scores=q["quality_upgrade_by_role_layer"]
    if set(scores)!=set(ROLES) or any(len(scores[x])!=L for x in ROLES):
        raise RuntimeError("incomplete 196 BF16-source tensor score table")
    if any(not math.isfinite(float(v)) or v<=0 for row in scores.values() for v in row):
        raise RuntimeError("invalid local activation sensitivity")
    if set(a["results"]["qkv"])!={"".join(map(str,x)) for x in
         [(4+4*int(i>=4),4+4*int(i%4>=2),4+4*int(i%2)) for i in range(8)]}:
        raise RuntimeError("not eight observed QKV masks")
    if set(a["results"]["ffn_gu"])!={"44","48","84","88"}:
        raise RuntimeError("not four observed gate/up masks")
    roles={row["role"]:row for row in r["role_kernel_cost_measured"]}
    if set(roles)!=set(ROLES):raise RuntimeError("7 kernel roles missing")
    return conf,q,a,r,scores,roles

def masks():
    return [tuple(8 if mask&(1<<j) else 4 for j in range(7)) for mask in range(M)]

def evaluate_mask_costs(roles,async_data):
    native=async_data["results"]
    baseqkv=native["qkv"]["444"]["best_verified_stage_ms"]
    basegu=native["ffn_gu"]["44"]["best_verified_stage_ms"]
    sourceQ4=lambda name:roles[name]["q4_neon_6thread_median_ms"]
    source8=lambda name:roles[name]["n8_repacked_neon_median_ms"]
    extra_memory={r:int(roles[r]["n8_with_scales_bytes"]-roles[r]["q4_with_scales_bytes"]) for r in ROLES}
    memory_base=28*sum(int(roles[r]["q4_with_scales_bytes"]) for r in ROLES)
    if memory_base!=736985088:raise RuntimeError("source Q4 196 projection byte identity changed")
    tuples=masks()
    options=[]
    for bits in tuples:
        qkv="".join(map(str,bits[:3]));gu=f"{bits[4]}{bits[5]}"
        if qkv not in native["qkv"] or gu not in native["ffn_gu"]:
            raise RuntimeError("missing native concurrent stage observation")
        serial=sum((source8(r)-sourceQ4(r)) for r,b in zip(ROLES,bits) if b==8)
        concurrent=native["qkv"][qkv]["best_verified_stage_ms"]-baseqkv
        concurrent+=native["ffn_gu"][gu]["best_verified_stage_ms"]-basegu
        concurrent+= (source8("o_proj")-sourceQ4("o_proj")) if bits[3]==8 else 0.
        concurrent+= (source8("down_proj")-sourceQ4("down_proj")) if bits[6]==8 else 0.
        serial_calibrated=serial*ROLE_SCALING_GAMMA
        if not all(math.isfinite(x) for x in (serial,concurrent,serial_calibrated)):
            raise RuntimeError("invalid calibrated native serial/async cost")
        options.append(dict(bits=bits,
            n8_count=sum(b==8 for b in bits),
            extra_bytes=sum(extra_memory[r] for r,b in zip(ROLES,bits) if b==8),
            serial_increment_ms=serial_calibrated,
            async_increment_ms=concurrent,
            preferred_qkv_executor_mode=native["qkv"][qkv]["best_verified_mode"],
            preferred_gateup_executor_mode=native["ffn_gu"][gu]["best_verified_mode"],
            observed_native_qkv_stage_ms=native["qkv"][qkv]["best_verified_stage_ms"],
            observed_native_gateup_stage_ms=native["ffn_gu"][gu]["best_verified_stage_ms"]))
    return options,memory_base

def solve():
    config,src,async_profile,native,scores,roles=load()
    options,basebytes=evaluate_mask_costs(roles,async_profile)
    n=L*M
    utility=np.empty(n,dtype=np.float64)
    rows=[];cols=[];values=[]
    for layer in range(L):
        for mask,opt in enumerate(options):
            col=layer*M+mask
            utility[col]=sum(scores[role][layer] for role,b in zip(ROLES,opt["bits"]) if b==8)
            for row,val in (
                (layer,1.0),
                (L,float(opt["n8_count"])),
                (L+1,float(opt["extra_bytes"])/196608.),
                (L+2,opt["serial_increment_ms"]),
                (L+3,opt["async_increment_ms"])):
                rows.append(row);cols.append(col);values.append(val)
    A=coo_matrix((np.asarray(values),(np.asarray(rows),np.asarray(cols))),
                 shape=(L+4,n)).tocsc()
    lb=np.full(L+4,-np.inf,dtype=np.float64)
    ub=np.full(L+4,np.inf,dtype=np.float64)
    lb[:L]=ub[:L]=1
    lb[L]=ub[L]=PROMOTIONS
    if (MAX_BYTES-basebytes)%196608:raise RuntimeError("mixed encoder memory not exact 192KiB units")
    ub[L+1]=(MAX_BYTES-basebytes)//196608
    ub[L+2]=MAX_SERIAL_MS-BASE_MS
    ub[L+3]=MAX_ASYNC_MS-BASE_MS
    # Zero MILP integrality gap requested. Suboptimal timeout is FAILURE.
    start=time.perf_counter()
    result=milp(c=-utility,integrality=np.ones(n,dtype=np.int8),
        bounds=Bounds(np.zeros(n),np.ones(n)),
        constraints=LinearConstraint(A,lb,ub),
        options=dict(time_limit=180.0,mip_rel_gap=0.0,disp=False))
    elapsed=time.perf_counter()-start
    if result.status!=0 or result.x is None or not result.success:
        raise RuntimeError(f"Qv2 MILP not proven optimal: {result.status} {result.message}; cannot open test")
    x=np.asarray(result.x)
    if np.any(np.abs(x-np.round(x))>1e-6):
        raise RuntimeError("fractional mixed quantization returned")
    chosen=[]
    policies=[]
    for layer in range(L):
        segment=x[layer*M:(layer+1)*M]
        hits=np.where(segment>.5)[0]
        if len(hits)!=1:raise RuntimeError("one mask per source Qwen layer violated")
        opt=options[int(hits[0])]
        row=dict(layer=layer,mask=int(hits[0]),bits=dict(zip(ROLES,opt["bits"])),
                 executor_QKV=opt["preferred_qkv_executor_mode"],
                 executor_gate_up=opt["preferred_gateup_executor_mode"],
                 model_serial_extra_ms=opt["serial_increment_ms"],
                 model_async_extra_ms=opt["async_increment_ms"],
                 extra_weight_bytes=opt["extra_bytes"],
                 source_calibration_quality_gain=float(utility[layer*M+int(hits[0])]))
        policies.append(row)
        for role,bit in row["bits"].items():
            if bit==8:chosen.append((role,layer))
    if len(chosen)!=PROMOTIONS or len(set(chosen))!=PROMOTIONS:
        raise RuntimeError("selected q8 tensor count wrong")
    selected_extra=sum(row["extra_weight_bytes"] for row in policies)
    t_serial=BASE_MS+sum(row["model_serial_extra_ms"] for row in policies)
    t_async=BASE_MS+sum(row["model_async_extra_ms"] for row in policies)
    if basebytes+selected_extra>MAX_BYTES or t_serial>MAX_SERIAL_MS+1e-7 or t_async>MAX_ASYNC_MS+1e-7:
        raise RuntimeError("native decoder or async DAG cost constraints violated")
    selected_gain=sum(row["source_calibration_quality_gain"] for row in policies)
    if abs(selected_gain+result.fun)>1e-7:
        raise RuntimeError("MILP score objective mismatch")
    labels=sorted(chosen,key=lambda z:(ROLES.index(z[0]),z[1]))
    text="".join(f"{role} {layer}\n" for role,layer in labels).encode()
    os.umask(0o077)
    if OUT.exists():raise FileExistsError("source-only Qv2 MILP policy already exists; cannot overwrite")
    OUT.mkdir(parents=True,mode=0o700)
    with (OUT/"selected_n8_roles.txt").open("xb") as f:f.write(text)
    (OUT/"selected_n8_roles.txt").chmod(0o600)
    diag=dict(schema="BEGLIN-Qv2-196-full-MILP-native-kernel-quality-async/1",
        status="OPTIMAL_HIGHS_SOURCE_ONLY_BEFORE_TEST",
        original_model_revision=EXPECTED_SOURCE_SNAPSHOT,
        input_preregistered_SHA256=sha(P.read_bytes()),
        input_source_activation_quality_SHA256=sha(G.read_bytes()),
        input_native_concurrency_measured_SHA256=sha(A.read_bytes()),
        input_native_role_profile_SHA256=sha(R.read_bytes()),
        quality_private_receipt_SHA256=EXPECTED["quality_private"],
        stage_private_receipt_SHA256=EXPECTED["stage_private"],
        source_scores_196_only_no_hidden_raw_activation_export=True,
        all_roles=ROLES,n_layers=L,n_mask_options_per_layer=M,binary_option_count=n,
        actual_worker_scheduling_scope="Apple native CPU sibling projections only, measured Python threadpool with independent Q4 native pools",
        full_model_runtime_currently_sequential=True,
        full_model_parallel_runtime_claim=False,
        selected_n8=28,selected_q4=168,
        native_model_storage_bytes=basebytes+selected_extra,
        native_model_storage_limit_bytes=MAX_BYTES,
        native_serial_predicted_ms=t_serial,serial_ms_bound=MAX_SERIAL_MS,
        native_hypothetical_async_DAG_predicted_ms=t_async,async_ms_bound=MAX_ASYNC_MS,
        source_calibration_gain_sum=selected_gain,
        MILP_status=int(result.status),MILP_message=str(result.message),
        MILP_mip_gap=float(result.mip_gap),
        MILP_mip_node_count=int(result.mip_node_count),
        optimizer_wall_seconds=elapsed,
        source_quality_objective="diagonal input-second-moment local projection reconstruction error benefit, no NLL",
        source_policy_tensor_mask_sha256=sha(text),
        per_layer=policies,
        selected_upgraded_roles={role:sorted(layer for role2,layer in chosen if role2==role) for role in ROLES},
        tested_IID_OOD_or_prelocked_eval_prompts_used=False,
        modified_main_or_release=False,
        calibration_intervention_banned_full_model_logit_tests_unmodified=True,
        limits_of_model=["Do not call input-moment score full-model quality sensitivity", "Async DAG observed stage costs not yet wired to whole Qwen runtime", "Result is MILP optimum for the specified discrete-source surrogate and measured stage table, not globally optimal LLM quality", "GPU Metal/other bitwidths not profiled"]
    )
    receipt=OUT/"policy_private_receipt.json"
    with receipt.open("xb") as f:
        f.write((json.dumps(diag,sort_keys=True,indent=2)+"\n").encode())
    receipt.chmod(0o600)
    print("BEGLIN_QV2_ACTIVATION_ASYNC_MILP_OPTIMAL",json.dumps({
      "policy_sha256":sha(text),"receipt_sha256":sha(receipt.read_bytes()),
      "upgrades_by_role":{r:len(diag["selected_upgraded_roles"][r]) for r in ROLES},
      "storage_bytes":diag["native_model_storage_bytes"],
      "predicted_serial_ms":t_serial,"predicted_async_ms":t_async,
      "source_proxy_gain":selected_gain,
      "mip_nodes":diag["MILP_mip_node_count"],"mip_gap":diag["MILP_mip_gap"],"seconds":round(elapsed,2)
    },sort_keys=True),flush=True)
if __name__=="__main__":solve()
