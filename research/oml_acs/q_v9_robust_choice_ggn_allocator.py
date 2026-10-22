#!/usr/bin/env python3
"""Qv9 conditional-four-choice Gauss–Newton plus ridge/minimax bit allocation.

  d_s,k(x) = sum_{i=196} [J_{s,k,i} δQ4_i +
                      x_i J_{s,k,i}(δN8_i−δQ4_i)], k in {A,B,C}, D=0.
  GGN_s(x) = .5 * Var_{p_s(A,B,C,D)}[d_s(x)].
  Ridge_s(x) = sum_i .5 * Var_{p_s}[J_i δ_i(x)].
  Train-only J(x) = .65 max(GGN_even12,GGN_odd12)+.35 mean(Ridge_all24).

Strict source-only, fixed coefficient, deterministic multi-start one-swap
local search, measured Apple native kernels and exact 28 n8/168 q4.
Not global quadratic optimum; not full-vocab NLL Hessian.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v8_signed_joint_optimizer as q8

LAB=ROOT/"lab_local/qv9_ggn_20261023"
INPUT=LAB/"original_BF16_choice_GGN_train24_private.json"
SOURCE_PUBLIC=ROOT/"docs/research/BEGLIN_QV9_TRAIN24_CONDITIONAL_GGN_PUBLIC_20261023.json"
PROTO=ROOT/"configs/research/BEGLIN_QV9_GGN_SOURCE_PRELOCK_20261023.json"
POLICY=ROOT/"docs/research/BEGLIN_QV9_ROBUST_CONDITIONAL_GGN_SOURCE_POLICY_20261023.json"
MASK=ROOT/"configs/research/BEGLIN_QV9_GGN_28N8_FROZEN_MASK_20261023.txt"
GGN_SHA_PRE="73dc2829da79637a97d68f903f72fed753eace2b154caec111c1fe60a7f4f983"
PRIOR={
"qv6_64":(ROOT/"configs/research/BEGLIN_QV6_FISHER_64N8_SOURCE_POLICY_20261022.txt","74fb949e45d13f63a26d412effc39377492ea71bda20ae3690f56fddf6303e08"),
"qv6_128":(ROOT/"configs/research/BEGLIN_QV6_FISHER_128N8_SOURCE_POLICY_20261022.txt","8b9e2404f9a38d7b57784b7d663efa19661553ccaa7dc79bfcbf6e6180fc7997"),
"qv8_signed96":(ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt","9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922"),
"qv5_24":(ROOT/"configs/research/BEGLIN_QV5_FISHER_28N8_SOURCE_POLICY_20261022.txt","88fdc38e6f55a16b0bdd4f7385e70f6de053db577a26d334a3ef3cb831113533")
}
K=28;N=196;ALPHA=.35
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
KEYS=[(r,i) for r in ROLES for i in range(28)]
def sha(x):return hashlib.sha256(x).hexdigest()
def read_sealed(p):
    if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:
        raise RuntimeError("private original BF16 GGN source no longer sealed")
    return p.read_bytes()
def conditional_risk(z3,p):
    # Batch n x 3 relative logit shifts anchored at D=0.
    z=np.concatenate((z3,np.zeros((z3.shape[0],1))),axis=1)
    mean=np.sum(z*p,axis=1)
    return .5*(np.sum(z*z*p,axis=1)-mean*mean)
def score_train_inputs():
    if POLICY.exists() or MASK.exists():raise FileExistsError("source policy previously selected; cannot refit")
    if sha(PROTO.read_bytes())!=GGN_SHA_PRE or not SOURCE_PUBLIC.is_file():
        raise RuntimeError("precommitted Qv9 source GGN proof missing")
    pub=json.loads(SOURCE_PUBLIC.read_text())
    raw=read_sealed(INPUT)
    if pub["full_private_receipt_sha256"]!=sha(raw) or \
       pub["source_group"]!="train" or pub["n_contrastive_logits_backprop"]!=3 or \
       pub["n_source_examples"]!=24:
        raise RuntimeError("original BF16 source 24×196×3 authentic triplets unbound")
    x=json.loads(raw)
    q4=np.asarray(x["q4_signed_relative_logit_delta"],dtype=np.float64)
    n8=np.asarray(x["n8_signed_relative_logit_delta"],dtype=np.float64)
    prob=np.asarray(x["original_conditional_four_letter_probabilities"],dtype=np.float64)
    if q4.shape!=(24,N,3) or n8.shape!=(24,N,3) or prob.shape!=(24,4) or \
       not (np.isfinite(q4).all() and np.isfinite(n8).all() and np.isfinite(prob).all()):
        raise RuntimeError("source BF16 conditional GGN original arrays invalid")
    if np.max(abs(prob.sum(axis=1)-1))>1e-5 or np.min(prob)<0:
        raise RuntimeError("BF16 next-letter original probability invalid")
    return pub,q4,n8,prob

def run():
    pub,q4,n8,p=score_train_inputs()
    started=time.monotonic()
    d=n8-q4
    base=q4.sum(axis=1)
    indep4=np.zeros((24,N)); indep8=np.zeros((24,N))
    for i in range(N):
        indep4[:,i]=conditional_risk(q4[:,i,:],p)
        indep8[:,i]=conditional_risk(n8[:,i,:],p)
    diag4=indep4.mean(axis=0)
    diag_delta=(indep8-indep4).mean(axis=0)
    if not np.isfinite([indep4.sum(),indep8.sum()]).all() or np.min(indep4)<-1e-8:
        raise RuntimeError("source conditional GGN PSD violation")
    physical_cost,swap_cost,feasible=q8.make_engine()
    ix={v:i for i,v in enumerate(KEYS)}
    def set_bits(mask):
        v=np.zeros(N,dtype=np.float64)
        for k in mask:v[ix[k]]=1.
        return v
    def value(mask):
        x=set_bits(mask)
        z=base+np.einsum("sik,i->sk",d,x,optimize=True)
        r=conditional_risk(z,p)
        individual=float(diag4.sum()+diag_delta@x)
        result=(1-ALPHA)*max(float(r[::2].mean()),float(r[1::2].mean()))+ALPHA*individual
        return float(result),{"GGN_even12":float(r[::2].mean()),"GGN_odd12":float(r[1::2].mean()),
            "GGN_mean":float(r.mean()),"ridge_diagonal_GGN":individual}
    seeds=[(k,q8.ids_from_mask(file,digest)) for k,(file,digest) in PRIOR.items()]
    for name,start in seeds:
        if len(start)!=K or not feasible(physical_cost(start)):
            raise RuntimeError("original fixed n8 source policy violates same measured Apple constraints")
    # Three deterministically generated feasible exploratory starts.
    rng=np.random.default_rng(20261023)
    for seed in (17,41,73):
        start=seeds[seed%len(seeds)][1].copy()
        for t in range(10):
            candidates=list(start)
            for _ in range(1000):
                old=candidates[int(rng.integers(len(candidates)))]
                other=KEYS[int(rng.integers(N))]
                if other in start:continue
                new=(start-{old})|{other}
                if feasible(physical_cost(new)):
                    start=new;break
        seeds.append((f"source_randomstart_{seed}",start))
    results=[]
    for name,init in seeds:
        chosen=init.copy(); c=physical_cost(chosen); start_score,_=value(chosen)
        accepted=[]
        for iteration in range(100):
            current,diag=value(chosen)
            pairs=None
            for old in sorted(chosen,key=lambda k:ix[k]):
                for new in KEYS:
                    if new in chosen:continue
                    costnew=swap_cost(chosen,c,old,new)
                    if costnew is None:continue
                    val,_=value((chosen-{old})|{new})
                    gap=val-current
                    if gap>=-1e-10:continue
                    tie=(ix[old],ix[new])
                    if pairs is None or gap<pairs[0]-1e-10 or (abs(gap-pairs[0])<1e-10 and tie<pairs[3]):
                        pairs=(gap,old,new,tie,costnew)
            if pairs is None:break
            gap,old,new,_,cnew=pairs
            newset=(chosen-{old})|{new}
            assert feasible(physical_cost(newset))
            accepted.append({"removed":list(old),"added":list(new),
                "source_regularized_choice_GGN_decrease":float(-gap)})
            chosen=newset;c=cnew
        final,bitscore=value(chosen)
        results.append({"start":name,"start_objective":start_score,"final_objective":final,
          "accepted_improving_1swaps":accepted,"selected_ids":[list(x) for x in sorted(chosen,key=lambda x:ix[x])],
          "source_risk_components":bitscore,"costs":physical_cost(chosen),
          "global_optimum_proven":False})
        print("QV9_SOURCE_ROBUST_CONDITIONAL_GGN_LOCAL_SEARCH",name,
          "initial",round(start_score,5),"final",round(final,5),
          "swaps",len(accepted),"seconds",round(time.monotonic()-started,1),flush=True)
    best=min(results,key=lambda v:(v["final_objective"],v["start"]))
    ids=[tuple(x) for x in best["selected_ids"]]
    assert len(ids)==K and len(set(ids))==K
    text="".join(f"{r} {l}\n" for r,l in ids).encode()
    os.umask(0o077)
    with MASK.open("xb") as f:f.write(text);f.flush();os.fsync(f.fileno())
    comparisons={"allQ4":None}
    baseobj,basecomp=value(set())
    comparisons["allQ4"]={"objective":baseobj,"components":basecomp}
    for name,mask in seeds[:4]:
        orig,comps=value(mask)
        comparisons[name]={"objective":orig,"components":comps}
    answer={
       "schema":"BEGLIN-Qv9-conditioned-choice-output-GGN-minimax-ridge-local-28n8/1",
       "status":"GGN_SOURCE_ONLY_FROZEN_BEFORE_DIAG_OR_MMLU",
       "prelock_sha256":GGN_SHA_PRE,
       "input_source_24_private_sha256":pub["full_private_receipt_sha256"],
       "input_source_24_public_sha256":sha(SOURCE_PUBLIC.read_bytes()),
       "original_checkpoint_rev":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
       "source_only_choice_GGN_not_full_vocab":True,
       "source_training_cases":24,"source_folds":[12,12],
       "fixed_ridge_alpha":ALPHA,
       "objective_formula":"0.65*max(GGN_odd12_mean,GGN_even12_mean)+0.35*sum_i(original_conditional_output_GGN_for_individual_tensor)",
       "Gauss_Newton_positive_semidefinite_probability_curvature":True,
       "prior_mask_source_objectives":comparisons,
       "source_all_possible_starts":results,
       "selected_best_start":best["start"],"selected_regularized_GGN_risk":best["final_objective"],
       "selected_components":best["source_risk_components"],
       "n8_projection_count":28,"q4_projection_count":168,
       "selected_original_model_roles":best["selected_ids"],
       "mask_sha256":sha(text),
       "final_Apple_native_source_costs":best["costs"],
       "strict_same_resource_constraints_as_Qv8":True,
       "global_optimality_proved":False,
       "source_1swap_local_optimality_only":True,
       "conditional_GGN_does_not_include_unselected_other_vocab_logits":True,
       "source_diagnostic_24_unopened":True,
       "MMLU_eval_48_logits_unopened":True,
       "no_fullmodel_single_tensor_interventions":True,
       "production_promotion":False,
       "elapsed_source_optimizer_s":round(time.monotonic()-started,2)
    }
    blob=(json.dumps(answer,sort_keys=True,indent=2)+"\n").encode()
    with POLICY.open("xb") as f:f.write(blob);f.flush();os.fsync(f.fileno())
    print("QV9_FIXED_REGULARIZED_GGN_28N8_SOURCE_MASK_FROZEN",json.dumps({
        "best_start":best["start"],"joint_risk":best["final_objective"],
        "q4_risk":baseobj,"best_components":best["source_risk_components"],
        "mask_sha256":sha(text),"public_sha256":sha(blob),
        "budget":best["costs"]},sort_keys=True),flush=True)
if __name__=="__main__":run()
