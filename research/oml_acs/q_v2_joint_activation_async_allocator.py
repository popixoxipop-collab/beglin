#!/usr/bin/env python3
"""BEGLIN Qv2 joint source activation + measured kernel + async stage allocator.

Goal: choose exactly 28 n8 upgrades among 196 true Qwen role/layer matrices.
Quality: SOURCE BF16 activation input diagonal local reconstruction benefit.
Cost: actual Apple Q4(1/6 worker) and n8 predecoded NEON stage measurements,
plus real QKV and gate/up parallel stage observed cost (not claimed runtime).

Exact within deterministic *top-k activation-score per role* policy family,
not globally exact across all 196 arbitrary bitmask combinations because
asynchronous stage costs couple the roles within a layer.
No model output, eval prompts, previous blocked intervention accessed.
"""
from __future__ import annotations
import hashlib,json,math,os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
QUALITY=ROOT/"docs/research/BEGLIN_Q_V2_ACTIVATION_DIAG_196_PUBLIC_AGGREGATE_20261022.json"
KERNEL=ROOT/"lab_local/q_kernel_aware_role_costs_20261022_v2/seven_role_native_cost_private.json"
ASYNC=ROOT/"lab_local/q_v2_async_stage_20261022/async_sibling_stage_costs_private.json"
LAB=ROOT/"lab_local/q_v2_joint_activation_async_20261022"
SRC_PROTO=ROOT/"configs/research/BEGLIN_Q_ACTIVATION_ASYNC_PARALLEL_V2_PRELOCK_20261022.json"
ROLE_ORDER=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
ROLE_N=28
UPGRADES=28
BASE_Q4_BYTES=736985088
MEMORY_LIMIT=830570496
SINGLE_TOKEN_BASE_Q4_MS=75.26
ROLE_BALANCED_RANDOM_MS=116.18
SOURCE_ROLE4_POLICY=ROOT/"lab_local/q_full_model_mixed_20261021/policy_random.txt"
QUALITY_ACTIV_SHA="ee7b177b4dde984a5d83e0cdb5983d2ae00c34ee752fef9113a79568a1d12663"
KERNEL_SHA="280ca338745d0ebb2c44986d18e8b240135fe7fa4b1077f49f65f5d8fb3eff93"
ASYNC_SHA="380670d42ccd2f12c4a015600be3d921f672c570a3d3a4a1652c61e217791808"
PROTO_SHA="0d9177b152286ccd413028e74838529e2a5aad57b42ec793df3095f99208d4af"
BOUND_SERIAL_MS=100.
BOUND_PARALLEL_MS=80.
MASK28=(1<<ROLE_N)-1

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()

def get_data():
    pub=json.loads(QUALITY.read_text())
    kernel_raw=KERNEL.read_bytes();async_raw=ASYNC.read_bytes();proto=SRC_PROTO.read_bytes()
    if sha(kernel_raw)!=KERNEL_SHA or sha(async_raw)!=ASYNC_SHA or sha(proto)!=PROTO_SHA:
        raise RuntimeError("source native CPU kernel costs or registered protocol changed")
    if pub.get("private_activation_moment_sha256")!=QUALITY_ACTIV_SHA or pub.get("tensor_count")!=196:
        raise RuntimeError("BF16 source activation scores not trusted")
    quality=pub["quality_upgrade_by_role_layer"]
    if set(quality)!=set(ROLE_ORDER) or any(len(vals)!=28 for vals in quality.values()):
        raise RuntimeError("source-only input incomplete")
    if any(any(not math.isfinite(float(x)) or float(x)<=0 for x in vals) for vals in quality.values()):
        raise RuntimeError("bad source activation-weighted local benefit")
    kernel=json.loads(kernel_raw);asyncstage=json.loads(async_raw)
    if kernel["nroles"]!=7 or set(asyncstage["stages"])!={"qkv","ffn_gu"}:
        raise RuntimeError("missing actual kernel stage")
    return pub,kernel,asyncstage

def family(quality,kernel):
    profile={x["role"]:x for x in kernel["role_samples"]}
    indexed={};memory_deltas={}
    for role in ROLE_ORDER:
        vals=quality[role]
        ranking=sorted(range(28),key=lambda layer:(-vals[layer],layer))
        masks=[0];prefix=[0.]
        for layer in ranking:
            masks.append(masks[-1]|(1<<layer))
            prefix.append(prefix[-1]+float(vals[layer]))
        row=profile[role]
        memory_deltas[role]=int(row["n8_with_scales_bytes"]-row["q4_with_scales_bytes"])
        if memory_deltas[role]<=0 or not math.isfinite(prefix[-1]):
            raise RuntimeError("bad model memory/quality role")
        indexed[role]=dict(rank=ranking,masks=masks,prefix=prefix,extra_bytes=memory_deltas[role],
                           q4_median_ms=float(row["times"]["q4_6t"]["median_ms"]),
                           n8_median_ms=float(row["times"]["n8_repacked"]["median_ms"]))
    return indexed

def old_control_masks():
    if sha(SOURCE_ROLE4_POLICY.read_bytes())!="72065ac7a4ae5d68b1c7944dc4a67e40ac9c3c825db54a1d441ca3002ec015c5":
        raise RuntimeError("previous successful native role-matched control mask SHA drift")
    mask={role:0 for role in ROLE_ORDER}
    for line in SOURCE_ROLE4_POLICY.read_text().splitlines():
        role,v=line.split();idx=int(v)
        if role not in mask or not 0<=idx<28 or (mask[role]>>idx)&1:raise RuntimeError("control source invalid")
        mask[role]|=1<<idx
    if any(bin(x).count('1')!=4 for x in mask.values()):raise RuntimeError("baseline not 4 per role")
    return mask

def stage_total(msks,profile,parallel:bool):
    s=profile["stages"]
    key="best_verified_stage_ms" if parallel else "serial_q4_6t"
    total=0.
    qkv=msks["q_proj"],msks["k_proj"],msks["v_proj"]
    gu=msks["gate_proj"],msks["up_proj"]
    for i in range(28):
        bits="".join(str(8 if (m>>i)&1 else 4) for m in qkv)
        c=s["qkv"][bits]
        total+=float(c["best_verified_stage_ms"] if parallel else c["serial_q4_6t"]["median_stage_ms"])
        bb="".join(str(8 if (m>>i)&1 else 4) for m in gu)
        c=s["ffn_gu"][bb]
        total+=float(c["best_verified_stage_ms"] if parallel else c["serial_q4_6t"]["median_stage_ms"])
        for role in ("o_proj","down_proj"):
            row=_COSTS[role]
            total+=row["n8_median_ms"] if (msks[role]>>i)&1 else row["q4_median_ms"]
    return total

_COSTS={}

def precompute(indexed,asyncstage):
    global _COSTS
    _COSTS=indexed
    base=stage_total({role:0 for role in ROLE_ORDER},asyncstage,False)
    control=stage_total(old_control_masks(),asyncstage,False)
    if control<=base:raise RuntimeError("role-balanced native staging unexpectedly cheaper than all Q4")
    gamma=(ROLE_BALANCED_RANDOM_MS-SINGLE_TOKEN_BASE_Q4_MS)/(control-base)
    if not 0.1<=gamma<=4.0:raise RuntimeError("real native calibration outside sane range")
    def combo(stage,roles):
        options={role:indexed[role]["masks"] for role in roles}
        if len(roles)==3:
            out={}
            for a in range(29):
                for b in range(29):
                    for c in range(29):
                        masks={role:options[role][count] for role,count in zip(roles,(a,b,c))}
                        serial=0.;parallel=0.
                        for layer in range(28):
                            bits="".join(str(8 if ((masks[role]>>layer)&1) else 4) for role in roles)
                            row=asyncstage["stages"][stage][bits]
                            serial+=row["serial_q4_6t"]["median_stage_ms"]
                            parallel+=row["best_verified_stage_ms"]
                        out[(a,b,c)]=(serial,parallel)
            return out
        out={}
        for a in range(29):
            for b in range(29):
                masks={role:options[role][count] for role,count in zip(roles,(a,b))}
                serial=0.;parallel=0.
                for layer in range(28):
                    bits="".join(str(8 if ((masks[role]>>layer)&1) else 4) for role in roles)
                    row=asyncstage["stages"][stage][bits]
                    serial+=row["serial_q4_6t"]["median_stage_ms"]
                    parallel+=row["best_verified_stage_ms"]
                out[(a,b)]=(serial,parallel)
        return out
    qkv=combo("qkv",ROLE_ORDER[:3])
    gu=combo("ffn_gu",("gate_proj","up_proj"))
    return dict(base_serial_stage_ms=base,calibration_control_serial_stage_ms=control,
        original_BEG_native_gamma=gamma,qkv=qkv,gu=gu)

def enumerate_policies(indexed,cost,asyncstage):
    assert len(indexed)==7
    roles=ROLE_ORDER
    md=[indexed[x]["extra_bytes"] for x in roles]
    prefs=[indexed[x]["prefix"] for x in roles]
    base=cost["base_serial_stage_ms"]
    gamma=cost["original_BEG_native_gamma"]
    qkv=cost["qkv"];gu=cost["gu"]
    q4o=indexed["o_proj"]["q4_median_ms"];n8o=indexed["o_proj"]["n8_median_ms"]
    q4d=indexed["down_proj"]["q4_median_ms"];n8d=indexed["down_proj"]["n8_median_ms"]
    # Single true memory upper bound and fixed bit count for BOTH policies.
    budget=MEMORY_LIMIT-BASE_Q4_BYTES
    best={"serial100":None,"parallel80":None}
    stats=dict(total_role_compositions=0,valid_memory=0,valid_serial=0,valid_parallel=0)
    def update(name,counts,gain,total_bytes,serial_ms,parallel_ms):
        old=best[name]
        item=dict(counts=list(counts),score=float(gain),projection_bytes=total_bytes,
            predicted_serial_ms=float(serial_ms),hypothetical_parallel_ms=float(parallel_ms))
        if old is None or gain>old["score"]+1e-12 or (abs(gain-old["score"])<=1e-12 and tuple(counts)<tuple(old["counts"])):
            best[name]=item
    # Iterate 7-category compositions, but use fast 3+2 stage block cost lookup.
    def dfs(i,remain,counts,accB,accG):
        if accB>budget:return
        if i==6:
            c=remain;totB=accB+c*md[6]
            if totB>budget:return
            vals=counts+[c]
            stats["total_role_compositions"]+=1
            stats["valid_memory"]+=1
            q=cost["qkv"][tuple(vals[:3])]
            ff=cost["gu"][(vals[4],vals[5])]
            serial_st=q[0]+ff[0] + 28*(q4o+q4d)+vals[3]*(n8o-q4o)+vals[6]*(n8d-q4d)
            parallel_st=q[1]+ff[1]+28*(q4o+q4d)+vals[3]*(n8o-q4o)+vals[6]*(n8d-q4d)
            serial_ms=SINGLE_TOKEN_BASE_Q4_MS+gamma*(serial_st-base)
            parallel_ms=SINGLE_TOKEN_BASE_Q4_MS+gamma*(parallel_st-base)
            gain=accG+prefs[6][c]
            if serial_ms<=BOUND_SERIAL_MS+1e-8:
                stats["valid_serial"]+=1
                update("serial100",vals,gain,BASE_Q4_BYTES+totB,serial_ms,parallel_ms)
                if parallel_ms<=BOUND_PARALLEL_MS+1e-8:
                    stats["valid_parallel"]+=1
                    update("parallel80",vals,gain,BASE_Q4_BYTES+totB,serial_ms,parallel_ms)
            return
        for c in range(remain+1):
            b=accB+c*md[i]
            if b>budget:break
            dfs(i+1,remain-c,counts+[c],b,accG+prefs[i][c])
    dfs(0,UPGRADES,[],0,0.)
    if any(v is None for v in best.values()):
        raise RuntimeError("no source-only candidate meets serial / hypothetical parallel constraints")
    for name,info in best.items():
        counts=info["counts"]
        groups={}
        selected=[]
        for role,count in zip(roles,counts):
            chosen=sorted(indexed[role]["rank"][:count])
            groups[role]=chosen
            selected.extend((role,l) for l in chosen)
        assert len(set(selected))==28
        info["layers_by_role"]=groups
        info["source_only_quality_measure"]="diagonal activation-scaled local projection error reduction; NOT true LLM NLL sensitivity"
        info["n8_tensors"]=28
        info["n4_tensors"]=168
        info["source_policy_family"]="top-k per each role; exact over role-count vectors, not all 2^196 assignments"
    return best,stats

def write_private(p:Path,bytestr:bytes):
    with p.open("xb") as f:
        f.write(bytestr);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)

def main():
    if LAB.exists():raise FileExistsError("source-only v2 policy already committed; no posthoc update")
    raw,kernel,asyncstage=get_data()
    indexed=family(raw["quality_upgrade_by_role_layer"],kernel)
    c=precompute(indexed,asyncstage)
    selected,stats=enumerate_policies(indexed,c,asyncstage)
    os.umask(0o077)
    LAB.mkdir(parents=True,mode=0o700)
    for name,policy in selected.items():
        rows=[(role,l) for role in ROLE_ORDER for l in policy["layers_by_role"][role]]
        p=LAB/f"{name}_source_only_roles.txt"
        data="".join(f"{r} {i}\n" for r,i in rows).encode()
        write_private(p,data)
        policy["policy_sha256"]=sha(data)
        policy["role_kernel_choices"]={role:dict(bit4=indexed[role]["q4_median_ms"],
             bit8=indexed[role]["n8_median_ms"]) for role in ROLE_ORDER}
    results=dict(schema="BEGLIN-Qv2-source-activation-async-196-allocator/1",
        status="FROZEN_BEFORE_NATIVE_CANDIDATE_EVALUATION",
        calibration_type="BF16 original projections source input diagonal 461 token positions",
        source_original_HF_rev="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        source_activation_aggregate_sha256=sha(QUALITY.read_bytes()),
        private_source_activation_moments_sha256=QUALITY_ACTIV_SHA,
        original_cpu_7_role_kernel_profile_sha256=KERNEL_SHA,
        real_async_QKV_FFN_stage_profile_sha256=ASYNC_SHA,
        preregistered_cal_eval_split_sha256=PROTO_SHA,
        actual_NATIVE_CPU_Qwen_body_is_sequential=True,
        hypothetical_parallel_stage_model_only=True,
        frozen_fixed_n8_promotions=UPGRADES,
        frozen_memory_budget_bytes=MEMORY_LIMIT,
        serial_predicted_time_max_ms=BOUND_SERIAL_MS,
        hypothetical_async_predicted_max_ms=BOUND_PARALLEL_MS,
        serial_reference_original_Q4_ms=SINGLE_TOKEN_BASE_Q4_MS,
        original_4_each_control_ms=ROLE_BALANCED_RANDOM_MS,
        kernel_stage_cost_normalization_gamma=c["original_BEG_native_gamma"],
        search_stats=stats,selected_policies=selected,
        independent_new_IID_OOD_not_opened=True,
        prior_per_tensor_logit_sensitivity_blocked_not_retried=True,
        main_or_production_modified=False,
        scientific_Q_verdict="NO_VERDICT",
        limitations=["Activation diagonal moments omit correlated covariance, full Hessian, end-to-end causal NLL and head Jacobians",
          "Native asynchronous stage timing only measured on isolated original layer0 Qwen projections",
          "No whole-model actual asynchronous execution; hypothetical parallel predicted time must not be called actual speed",
          "Top-k-per-role exact over 7 counts only, not all possible 196 individual assignments",
          "Calibration native speed gamma anchors previous all4 and frozen random 28n8 kernel measurements",
          "Activation scores from 12 public prompts/461 token positions; new evaluation 8 different prelisted public prompts remain sealed",
          "Prior V6 24 used identities unknown, no final task-AUROC or real BF16 native arithmetic"]
    )
    path=LAB/"frozen_v2_source_only_policies.json"
    write_private(path,(json.dumps(results,indent=2,sort_keys=True)+"\n").encode())
    print("BEGLIN_QV2_ACTIVATION_ASYNC_POLICIES_FROZEN",json.dumps(dict(
       source_receipt_sha256=sha(path.read_bytes()),
       serial_calibration_gamma=c["original_BEG_native_gamma"],
       stats=stats,policies={name:{k:r[k] for k in ("counts","score","projection_bytes","predicted_serial_ms","hypothetical_parallel_ms","policy_sha256")}
            for name,r in selected.items()}),sort_keys=True),flush=True)
if __name__=="__main__":main()
