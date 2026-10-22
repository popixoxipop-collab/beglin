#!/usr/bin/env python3
"""Qv8 quadratic cross-tensor source task-loss allocator, source-only.

Unlike per-tensor E[(g·delta_i)^2], this evaluates the *joint*
original-BF16 first-order loss perturbation for all 196 tensors simultaneously:
  r_s(x) = sum_i [a_si + x_i * (b_si-a_si)]
  J(x)   = mean_s r_s(x)^2
subject to 28 of 196 n8 and the same original measured BEGLIN native Apple
Q4/n8 byte, serial kernel proxy and measured QKV/FFN stage costs.

Exactly constrained deterministic multi-start feasible 1-exchange local
search. NOT a globally optimal quadratic binary solver / exact Hessian.
No held HellaSwag validation results or per-tensor fullmodel interventions.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,sys,time
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v2_async_activation_milp as v2

PRE=ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_SOURCE_PRELOCK_20261022.json"
SOURCE=ROOT/"docs/research/BEGLIN_QV8_BF16_SIGNED_CROSS_TENSOR_SOURCE_PUBLIC_20261022.json"
FULL=ROOT/"lab_local/qv8_joint_signed_20261022/original_BF16_signed_196x96_source_private.json"
HELD=ROOT/"lab_local/qv8_joint_signed_20261022/pinned_HellaSwag_val48.json"
Q6_64=ROOT/"configs/research/BEGLIN_QV6_FISHER_64N8_SOURCE_POLICY_20261022.txt"
Q6_128=ROOT/"configs/research/BEGLIN_QV6_FISHER_128N8_SOURCE_POLICY_20261022.txt"
Q5_24=ROOT/"configs/research/BEGLIN_QV5_FISHER_28N8_SOURCE_POLICY_20261022.txt"
MASK=ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt"
OUT=ROOT/"docs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_OPTIMIZER_20261022.json"
PRE_SHA="9b833f5fc2729d13320c87e368baaff952c890ed068586622badfc7da63b4db1"
HELD_SHA="45c95170a2eb01b4c79ad44719ab9a1877e568f2583a90d84340d7448e0d1a33"
BASELINE_MASK_SHA={
 "qv6_64":"74fb949e45d13f63a26d412effc39377492ea71bda20ae3690f56fddf6303e08",
 "qv6_128":"8b9e2404f9a38d7b57784b7d663efa19661553ccaa7dc79bfcbf6e6180fc7997",
 "qv5_24":"88fdc38e6f55a16b0bdd4f7385e70f6de053db577a26d334a3ef3cb831113533"}
N=196;K=28
ROLES=v2.ROLES
def sha(b:bytes):return hashlib.sha256(b).hexdigest()

def input_score():
    if OUT.exists() or MASK.exists():raise FileExistsError("Qv8 policy already finalized, no refitting")
    if sha(PRE.read_bytes())!=PRE_SHA or sha(HELD.read_bytes())!=HELD_SHA:
        raise RuntimeError("Qv8 old source/dataset prereg violated")
    scores=json.loads(SOURCE.read_text())
    data=FULL.read_bytes()
    if scores["status"]!="SOURCE_ONLY_196x96_SIGNED_NATIVE_Q4N8_DELTA_VERIFIED" or \
       scores["full_source_private_SHA256"]!=sha(data) or \
       scores["eval_logits_unopened"] is not True:
        raise RuntimeError("source BF16 signed score not validated")
    obj=json.loads(data)
    q4=np.asarray(obj["original_tensor_grad_dot_q4"],dtype=np.float64)
    n8=np.asarray(obj["original_tensor_grad_dot_n8"],dtype=np.float64)
    if q4.shape!=(96,196) or n8.shape!=(96,196) or not np.isfinite(q4).all() or not np.isfinite(n8).all():
        raise RuntimeError("missing 96x196 signed scalar projections")
    return scores,q4,n8

def make_engine():
    _,_,stages,serial,mem=v2.verified_sources()
    base_memory=sum(28*mem[r]["q4"] for r in ROLES)
    if base_memory!=736985088:
        raise RuntimeError("actual original Q4G64 weight/scale memory changed")
    base_serial=v2.FULL_Q4_MS
    mapping={}
    for layer in range(28):
        for stage,roles in v2.GROUPS:
            for role in roles:
                mapping[(role,layer)]=(layer,stage,roles)
    extras={}
    for stage,roles in v2.GROUPS:
        if len(roles)==1:
            extras[stage]=roles
    def stage_one(layer,stage,roles,selected):
        pattern="".join("8" if (role,layer) in selected else "4" for role in roles)
        if len(roles)==1:
            val=serial[roles[0]]["n8" if pattern=="8" else "q4"]
        else:
            val=float(stages["data"][stage][pattern]["best_ms"])
        if not math.isfinite(val) or val<=0:raise RuntimeError("native measured async stage cost invalid")
        return val
    def all_stage(selected):
        return sum(stage_one(layer,stage,roles,selected)
            for layer in range(28) for stage,roles in v2.GROUPS)
    dmem={(r,l):int(mem[r]["n8"]-mem[r]["q4"]) for r in ROLES for l in range(28)}
    dtime={(r,l):float(serial[r]["delta"]) for r in ROLES for l in range(28)}
    def cost(selected):
        return {"memory_bytes":int(base_memory+sum(dmem[r] for r in selected)),
                "serial_estimate_ms":float(base_serial+v2.GAMMA_SERIAL*sum(dtime[r] for r in selected)),
                "isolated_async_stage_ms":float(all_stage(selected))}
    def okay(k):
        return k["memory_bytes"]<=v2.MEMORY_LIMIT and k["serial_estimate_ms"]<=v2.SERIAL_CEILING_MS+1e-6 and \
            k["isolated_async_stage_ms"]<=v2.ASYNC_DAG_STAGE_CEILING_MS+1e-6
    def swap_cost(selected,cur,old_id,new_id):
        memnew=cur["memory_bytes"]-dmem[old_id]+dmem[new_id]
        serialnew=cur["serial_estimate_ms"]+v2.GAMMA_SERIAL*(-dtime[old_id]+dtime[new_id])
        if memnew>v2.MEMORY_LIMIT or serialnew>v2.SERIAL_CEILING_MS+1e-6:
            return None
        affected={mapping[old_id],mapping[new_id]}
        proposed=(selected-{old_id})|{new_id}
        change=0.
        for layer,stage,roles in affected:
            change+=stage_one(layer,stage,roles,proposed)-stage_one(layer,stage,roles,selected)
        stage_new=cur["isolated_async_stage_ms"]+change
        if stage_new>v2.ASYNC_DAG_STAGE_CEILING_MS+1e-6:
            return None
        return {"memory_bytes":int(memnew),"serial_estimate_ms":float(serialnew),
                "isolated_async_stage_ms":float(stage_new)}
    return cost,swap_cost,okay

def ids_from_mask(path,dig):
    if not path.is_file() or sha(path.read_bytes())!=dig:
        raise RuntimeError("source-only starting bitmask changed "+str(path))
    rows=set()
    for line in path.read_text().splitlines():
        a,b=line.split()
        if a not in ROLES or not b.isdigit() or int(b)>=28:
            raise RuntimeError("invalid original 28layer projection "+line)
        rows.add((a,int(b)))
    if len(rows)!=28:raise RuntimeError("requires exact 28 original native bitmask")
    return rows

def allocate():
    scores,q4,n8=input_score()
    t=time.monotonic()
    a=q4.sum(axis=1)
    D=n8-q4
    Gram=(D.T@D)/96
    linear=(D.T@a)/96
    j0=float(np.mean(a*a))
    if not np.isfinite(Gram).all() or not np.isfinite(linear).all():
        raise RuntimeError("signed BF16 source first-order cross-term invalid")
    matrix=Gram+Gram.T
    IDs=[(r,idx) for r in ROLES for idx in range(28)]
    ix_of={key:i for i,key in enumerate(IDs)}
    real_cost,swap_cost,okay=make_engine()
    starts=[
       ("qv6_64",ids_from_mask(Q6_64,BASELINE_MASK_SHA["qv6_64"])),
       ("qv6_128",ids_from_mask(Q6_128,BASELINE_MASK_SHA["qv6_128"])),
       ("qv5_24",ids_from_mask(Q5_24,BASELINE_MASK_SHA["qv5_24"])) ]
    for name,x in starts:
        if not okay(real_cost(x)):raise RuntimeError("original source MILP mask does not satisfy native costs "+name)
    rng=np.random.default_rng(20261022)
    for seed in (17,41,73):
        basis=starts[seed%len(starts)][1].copy()
        for j in range(10):
            for tries in range(800):
                i=IDs[int(rng.choice([ix_of[key] for key in basis]))]
                remain=[z for z in IDs if z not in basis]
                o=remain[int(rng.integers(len(remain)))]
                trial=(basis-{i})|{o}
                if okay(real_cost(trial)):
                    basis=trial;break
        starts.append((f"source_shuffle_{seed}",basis))
    def obj(x):
        v=np.zeros(N,dtype=np.float64)
        for item in x:v[ix_of[item]]=1.
        return j0+2*linear@v+v@Gram@v
    details=[];candidates=[]
    for name,begin in starts:
        current=begin.copy();c=real_cost(current);before=float(obj(current))
        swaps=[];local_eval=0
        for iteration in range(140):
            x=np.zeros(N,dtype=np.float64)
            for z in current:x[ix_of[z]]=1.
            gradient=2*(linear+Gram@x)
            left=sorted(current,key=lambda z:ix_of[z])
            right=[z for z in IDs if z not in current]
            best=None
            for out_id in left:
                i=ix_of[out_id]
                for in_id in right:
                    j=ix_of[in_id]
                    delta=(-gradient[i]+gradient[j]
                      +Gram[i,i]+Gram[j,j]-2*Gram[i,j])
                    if delta>=-1e-10:continue
                    nextcost=swap_cost(current,c,out_id,in_id)
                    if nextcost is None:continue
                    local_eval+=1
                    if best is None or delta<best[0]-1e-12 or (abs(delta-best[0])<=1e-12 and (i,j)<best[3]):
                        best=(delta,out_id,in_id,(i,j),nextcost)
            if best is None:break
            diff,old,new,_,costnew=best
            previous_score=obj(current)
            proposal=(current-{old})|{new}
            new_score=obj(proposal)
            if abs(new_score-previous_score-diff)>max(1e-4,abs(diff)*1e-5):
                raise RuntimeError("quadratic analytical pairwise Δ differs from exact signed gradient Gram")
            swaps.append({"drop":list(old),"add":list(new),
                          "signed_joint_loss_change":float(diff)})
            current=proposal;c=costnew
        if len(current)!=28 or not okay(real_cost(current)):
            raise RuntimeError("local search produced infeasible actual Apple kernel policy")
        final=float(obj(current))
        details.append({"start_name":name,"start_score":before,
           "final_score":final,"feasible_swaps":len(swaps),
           "best_improving_exchanges_accepted":swaps,
           "last_cost":real_cost(current),"proven_global_optimum":False})
        candidates.append((final,name,current))
        print("QV8_SOURCE_GLOBAL_SIGNED_MODEL_LOCAL_EXCHANGE",name,
             "start",round(before,7),"final",round(final,7),
             "swaps",len(swaps),"elapsed",round(time.monotonic()-t,1),flush=True)
    final,name,selected=min(candidates,key=lambda t:(t[0],t[1]))
    asbytes="".join(f"{key[0]} {key[1]}\n" for key in sorted(selected,key=lambda v:ix_of[v])).encode()
    os.umask(0o077)
    with MASK.open("xb") as f:f.write(asbytes);f.flush();os.fsync(f.fileno())
    baselines={}
    for baseline_name,group in starts[:3]:
        baselines[baseline_name]={"signed_joint_source_objective":float(obj(group)),
           "costs":real_cost(group),"mask_SHA256":sha("".join(f"{k[0]} {k[1]}\n" for k in sorted(group,key=lambda v:ix_of[v])).encode())}
    out={
       "schema":"BEGLIN-Qv8-first-order-cross-tensor-signed-task-loss-source-local-search/1",
       "status":"SOURCE_ONLY_28N8_STRICT_FEASIBLE_LOCAL_SWAP_DONE_NO_EVAL_OPENED",
       "prelock_sha256":PRE_SHA,
       "source_96_signed_private_sha256":scores["full_source_private_SHA256"],
       "source_96_signed_public_sha256":sha(SOURCE.read_bytes()),
       "held48_HellaSwag_SHA256":HELD_SHA,
       "original_model_rev":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
       "full_original_model_layer_count":28,
       "source_signed_task_gradient_cross_tensor_pairwise_covariance":True,
       "source_diagonal_no_cross_objective_comparison_only":False,
       "objective_global_loss_change_squared_empirical_mean_Q4":j0,
       "source_old_MILP_objective_competitors":baselines,
       "strict_28N8_and_168Q4":True,
       "source_optimization_exact_global":False,
       "source_optimization_feasible_pairwise_1swap_local_only":True,
       "source_starts":details,
       "selected_best_start":name,
       "selected_source_signed_loss_squared":float(final),
       "source_quadratic_Gram_full":False,
       "selected_ids":[list(k) for k in sorted(selected,key=lambda v:ix_of[v])],
       "mask_sha256":sha(asbytes),
       "final_original_Apple_kernel_costs":real_cost(selected),
       "pairwise_cross_term_frobenius_norm":float(np.linalg.norm(Gram-np.diag(np.diag(Gram)))),
       "diag_term_frobenius_norm":float(np.linalg.norm(np.diag(Gram))),
       "no_held_data_labels_or_logits_loaded":True,
       "no_main_or_immutable_release_changes":True,
       "auto_serving_promotion_forbidden":True,
       "elapsed_s":round(time.monotonic()-t,2)}
    blob=(json.dumps(out,sort_keys=True,indent=2)+"\n").encode()
    with OUT.open("xb") as f:f.write(blob);f.flush();os.fsync(f.fileno())
    print("QV8_28N8_REAL_BEGLIN_SOURCE_SIGNED_JOINT_MASK_FROZEN",json.dumps({
      "best":name,"baseline_q4":j0,"new_score":float(final),
      "baselines":{k:round(v["signed_joint_source_objective"],6) for k,v in baselines.items()},
      "mask_sha":sha(asbytes),"public_sha":sha(blob),
      "cost":real_cost(selected),"seconds":out["elapsed_s"]},sort_keys=True),flush=True)

if __name__=="__main__":allocate()
