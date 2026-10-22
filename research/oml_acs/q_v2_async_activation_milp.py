#!/usr/bin/env python3
"""BEGLIN Qv2 source-only activation-aware, measured-concurrent-DAG MILP.

448 binary options = (28 decoder layers)*(8 qkv + 4 gate/up + 2 o + 2 down).
Exactly one option per stage/layer. MILP constraints:
  - exactly 28 real n8 tensor upgrades, 168 packed Q4G64
  - original measured packed memory <= 830570496 bytes
  - original calibrated sequential one-token latency proxy <= 100 ms
  - original measured best-concurrent independent native CPU stage DAG
    sum over 28 layers <= 50ms (NOT full-model runtime latency)
Maximize BF16 source-calibrated local *diagonal activation-weighted* quality.

HiGHS integrality must prove optimum; no evaluation prompts are loaded.
"""
from __future__ import annotations
import hashlib,itertools,json,math,os,pathlib,time
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import coo_matrix

ROOT=pathlib.Path(__file__).resolve().parents[2]
CONFIG=ROOT/"configs/research/BEGLIN_Q_V2_ASYNC_DAG_MILP_PRELOCK_20261022.json"
QUAL=ROOT/"configs/research/BEGLIN_Q_V2_ACTIVATION_DIAGONAL_GAINS_20261022.json"
STAGE=ROOT/"configs/research/BEGLIN_Q_V2_NATIVE_ASYNC_STAGE_COSTS_20261022.json"
ROLE=ROOT/"docs/research/BEGLIN_Q_KERNEL_AWARE_ALLOCATOR_NATIVE_EVIDENCE_20261022.json"
OUT=pathlib.Path.home()/".local/share/beglin/oml_acs/20261022_qv2_async_allocator"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
GROUPS=(("qkv",ROLES[:3]),("o",(ROLES[3],)),("ffn_gu",ROLES[4:6]),("down",(ROLES[6],)))
SHA_SCORES="94ffe5c18ee4f92fadee912d1046d2ed104f3934e968081e87dfd1ac07ed1bf0"
SHA_STAGE="380670d42ccd2f12c4a015600be3d921f672c570a3d3a4a1652c61e217791808"
MEMORY_LIMIT=830570496
N8_TARGET=28
SERIAL_CEILING_MS=100.
ASYNC_DAG_STAGE_CEILING_MS=50.
FULL_Q4_MS=75.26
GAMMA_SERIAL=1.039406406482483

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def verified_sources():
    pre=json.loads(CONFIG.read_text())
    if pre.get("status")!="SOURCE_METRICS_AVAILABLE__NEW_POLICY_AND_EVALUATION_NOT_YET_RUN":
        raise RuntimeError("wrong prelock status")
    gains=json.loads(QUAL.read_text())
    async_cost=json.loads(STAGE.read_text())
    native=json.loads(ROLE.read_text())
    if gains["source_SHA"]!=SHA_SCORES or async_cost["source_SHA"]!=SHA_STAGE:
        raise RuntimeError("source calibration or measured async stage provenance changed")
    if sha(QUAL.read_bytes())==sha(STAGE.read_bytes()):
        raise RuntimeError("unexpected provenance collision")
    for key in ROLES:
        arr=gains["quality_upgrade_by_role_layer"].get(key)
        if not isinstance(arr,list) or len(arr)!=28 or any(not math.isfinite(x) or x<=0 for x in arr):
            raise RuntimeError("missing/nonpositive actual BF16 source-gain for "+key)
    costs={row["role"]:row for row in native["role_kernel_cost_measured"]}
    if set(costs)!=set(ROLES):
        raise RuntimeError("missing seven measured native kernels")
    serial={}
    memory={}
    for role in ROLES:
        x=costs[role]
        q4=min(float(x["q4_neon_1thread_median_ms"]),float(x["q4_neon_6thread_median_ms"]))
        n8=float(x["n8_repacked_neon_median_ms"])
        if not 0<q4<n8:raise RuntimeError("kernel cost sign mismatch")
        serial[role]=dict(q4=q4,n8=n8,delta=n8-q4)
        memory[role]=dict(q4=int(x["q4_with_scales_bytes"]),n8=int(x["n8_with_scales_bytes"]))
    for stage,roles in GROUPS:
        if len(roles)>1:
            table=async_cost["data"][stage]
            expected={"".join(str(v) for v in mask) for mask in itertools.product((4,8),repeat=len(roles))}
            if set(table)!=expected:raise RuntimeError("missing real concurrency scenario for "+stage)
            for name,obs in table.items():
                if obs["best_mode"] not in ("serial_q4_6t","parallel_q4_1t","parallel_q4_6t"):
                    raise RuntimeError("unsupported async mode")
                if obs["best_ms"]<=0 or obs["best_ms"]>obs["serial_ms"]+1e-8:
                    raise RuntimeError("async stage cost must be measured and no worse than serial")
    return pre,gains,async_cost,serial,memory

def solve(quality,stages,serial,memory):
    q4_total=sum(28*memory[role]["q4"] for role in ROLES)
    if q4_total!=736985088:raise RuntimeError("original packed Q4 baseline bytes drift")
    budget=MEMORY_LIMIT-q4_total
    combos=[]
    constraint_rows=[]; constraint_cols=[];constraint_vals=[]
    groups_equality=[]
    for layer in range(28):
        for stage,roles in GROUPS:
            stage_indices=[]
            for mask in itertools.product((4,8),repeat=len(roles)):
                details=dict(zip(roles,mask))
                bits="".join(str(x) for x in mask)
                if len(roles)>1:
                    obs=stages["data"][stage][bits]
                    asynctime=float(obs["best_ms"])
                    concurrency=obs["best_mode"]
                else:
                    role=roles[0]
                    asynctime=serial[role]["q4" if mask[0]==4 else "n8"]
                    concurrency="serial_one_role"
                benefit=sum(quality["quality_upgrade_by_role_layer"][role][layer]
                            for role,bits in details.items() if bits==8)
                count=sum(b==8 for b in mask)
                bdelta=sum(memory[role]["n8"]-memory[role]["q4"]
                           for role,bits in details.items() if bits==8)
                serialdelta=sum(serial[role]["delta"] for role,bits in details.items() if bits==8)
                idx=len(combos)
                combos.append(dict(layer=layer,stage=stage,roles=roles,mask=mask,
                  N8_count=count,extra_memory_bytes=bdelta,
                  serial_kernel_extra_ms=serialdelta,
                  async_stage_ms=asynctime,
                  source_activation_local_quality_gain=benefit,
                  validated_async_dispatch=concurrency))
                stage_indices.append(idx)
            groups_equality.append(stage_indices)
    n=len(combos)
    if n!=448 or len(groups_equality)!=112:raise RuntimeError("not 448 measured joint stage choices")
    # Build 115 row MILP: 112 stage equality + exactly 28 bit choices,
    # capped memory and calibrated serial time and measured async DAG stage.
    constraint_count=len(groups_equality)+4
    for row,group in enumerate(groups_equality):
        for col in group:
            constraint_rows.append(row);constraint_cols.append(col);constraint_vals.append(1.)
    extra=np.array([x["N8_count"] for x in combos],dtype=np.float64)
    memb=np.array([x["extra_memory_bytes"] for x in combos],dtype=np.float64)
    seriald=np.array([x["serial_kernel_extra_ms"] for x in combos],dtype=np.float64)
    stagecost=np.array([x["async_stage_ms"] for x in combos],dtype=np.float64)
    for offset,weights in enumerate((extra,memb,seriald,stagecost)):
        row=len(groups_equality)+offset
        for col,value in enumerate(weights):
            if value:
                constraint_rows.append(row);constraint_cols.append(col);constraint_vals.append(float(value))
    mat=coo_matrix((constraint_vals,(constraint_rows,constraint_cols)),shape=(constraint_count,n)).tocsr()
    lower=np.concatenate([np.ones(len(groups_equality)),[N8_TARGET,-np.inf,-np.inf,-np.inf]])
    upper=np.concatenate([np.ones(len(groups_equality)),[
         N8_TARGET,budget,(SERIAL_CEILING_MS-FULL_Q4_MS)/GAMMA_SERIAL,ASYNC_DAG_STAGE_CEILING_MS]])
    objective=np.array([x["source_activation_local_quality_gain"] for x in combos])
    # tiny deterministic tie-breaker cannot change any meaningful quality score
    perturb=1.e-12*(n-np.arange(n,dtype=np.float64))/n
    result=milp(c=-(objective+perturb),integrality=np.ones(n,dtype=np.int8),
        bounds=Bounds(np.zeros(n),np.ones(n)),
        constraints=LinearConstraint(mat,lower,upper),
        options={"time_limit":180.0,"mip_rel_gap":1.e-9,"presolve":True})
    if result.status!=0 or result.x is None or (getattr(result,"mip_gap",1)>1.e-7):
        raise RuntimeError("HiGHS did not prove exact optimum "+str(result.message))
    x=result.x
    if not np.all((np.abs(x)<1e-5)|(np.abs(x-1)<1e-5)):
        raise RuntimeError("fractional MILP decisions")
    selected=[c for c,v in zip(combos,x) if v>0.5]
    assert len(selected)==112
    ids=[]
    stageplan=[]
    for c in selected:
        for role,bits in zip(c["roles"],c["mask"]):
            if bits==8:ids.append((role,c["layer"]))
        stageplan.append(dict(layer=c["layer"],stage=c["stage"],bits="".join(str(n) for n in c["mask"]),
            dispatcher=c["validated_async_dispatch"],observed_stage_median_ms=c["async_stage_ms"]))
    ids.sort(key=lambda k:(ROLES.index(k[0]),k[1]))
    if len(ids)!=28 or len(set(ids))!=28:
        raise RuntimeError("selected 28 unique n8 tensor IDs contract failed")
    totalmem=q4_total+sum(c["extra_memory_bytes"] for c in selected)
    predictedserial=FULL_Q4_MS+GAMMA_SERIAL*sum(c["serial_kernel_extra_ms"] for c in selected)
    predictedasyncstage=sum(c["async_stage_ms"] for c in selected)
    if totalmem>MEMORY_LIMIT or predictedserial>SERIAL_CEILING_MS+1.e-7 or predictedasyncstage>ASYNC_DAG_STAGE_CEILING_MS+1.e-7:
        raise RuntimeError("chosen policy violates actual calculated constraints")
    return dict(layer_stage_options=stageplan,selected_ids=ids,estimated_projection_weight_bytes=totalmem,
       original_fullmodel_serial_estimated_ms=predictedserial,
       isolated_async_CPU_stage_total_ms=predictedasyncstage,
       source_diag_activation_quality_gain=sum(c["source_activation_local_quality_gain"] for c in selected),
       source_only_solver_status=result.status,mip_gap=float(result.mip_gap),
       explored_mip_node_count=int(getattr(result,"mip_node_count",-1)),
       milp_stage_variable_count=n,
       n8_tensors=28,n4_tensors=168,
       selected_role_counts={r:sum(role==r for role,l in ids) for r in ROLES},
       estimated_unquantized_base_full_q4_ms=FULL_Q4_MS,
       async_stage_makespan_is_not_full_model_walltime=True)

def main():
    if OUT.exists():raise FileExistsError("source-only policy already frozen; do not refit on evaluation")
    os.umask(0o077)
    start=time.monotonic()
    protocol,quality,stages,serial,memory=verified_sources()
    model=solve(quality,stages,serial,memory)
    OUT.mkdir(parents=True,mode=0o700)
    policy=OUT/"qv2_async_activation_28n8_frozen.txt"
    content="".join(f"{role} {layer}\n" for role,layer in model["selected_ids"]).encode()
    with policy.open("xb") as f:
        f.write(content);f.flush();os.fsync(f.fileno())
    policy.chmod(0o600)
    inputs={
        "source_activation_scores_sha256":sha(QUAL.read_bytes()),
        "source_parallel_native_stage_sha256":sha(STAGE.read_bytes()),
        "native_seven_role_costs_sha256":sha(ROLE.read_bytes()),
        "source_protocol_sha256":sha(CONFIG.read_bytes()),
        "source_activation_private_receipt_sha256":quality["source_SHA"],
        "source_native_parallel_private_receipt_sha256":stages["source_SHA"],
        "selected_policy_sha256":sha(content),
    }
    result={
      "schema":"BEGLIN-Q-v2-activation-and-async-DAG-native-ILP/1",
      "status":"SOURCE_ONLY_PROVED_MILP_OPTIMUM__NOT_YET_MODEL_EVALUATED",
      "model":"Qwen/Qwen2.5-1.5B-Instruct",
      "source_bf16_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
      "native_hardware_cost":"EOE Apple Mac13,1, quantized BEGLIN C native Q4 6t/n8-repacked",
      "activation_calibration":"original BF16 on Alienware CUDA, 12 public prompts/461 token positions",
      "optimizer":model,
      "input_sha":inputs,
      "output_test_window_not_used":True,
      "parallel_execution_full_model_unverified":True,
      "core_decoder_sequential_until_explicit_async_integration":True,
      "model_quality_score_is_diagonal_local_error_proxy":True,
      "scientific_Q_verdict":"NO_VERDICT",
      "prior_blocked_full_model_tensor_intervention_not_retried":True,
      "production_main_unchanged":True,
      "elapsed_seconds":round(time.monotonic()-start,3)
    }
    manifest=OUT/"qv2_async_activation_allocator_receipt.json"
    with manifest.open("xb") as f:
        f.write((json.dumps(result,indent=2,sort_keys=True)+"\n").encode())
        f.flush();os.fsync(f.fileno())
    manifest.chmod(0o600)
    print("BEGLIN_Q_V2_ASYNC_ACTIVATION_POLICY_FROZEN",
          json.dumps({"receipt_sha256":sha(manifest.read_bytes()),
          "policy_sha256":sha(content),"role_counts":model["selected_role_counts"],
          "source_quality_gain":model["source_diag_activation_quality_gain"],
          "projected_bytes":model["estimated_projection_weight_bytes"],
          "predicted_serial_ms":model["original_fullmodel_serial_estimated_ms"],
          "measured_stage_DAG_ms":model["isolated_async_CPU_stage_total_ms"],
          "solver_gap":model["mip_gap"],"node_count":model["explored_mip_node_count"]},
          sort_keys=True),flush=True)
if __name__=="__main__":main()
