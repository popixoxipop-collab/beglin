#!/usr/bin/env python3
"""Source-only grouped fullvocab-mass top2+OTHER trust-regularized bit allocation.

For original BF16 Qwen fullvocab next-token group masses:
group A,B,C,top1 nonchoice,top2 nonchoice,OTHER (all others),D anchor.
Signed actual C Q4/n8 errors Δu in 6 relative logit channels:
  grouped_exact_KL_linearized = logsumexp(log p+u)-p·u.
  trust_penalty = .05 * sum p max(0,|u_Drelative|-2)^2.
  isolated_curvature_ridge = .10/196 sum_i .5 Var_p(u_i).
  J = max(even12 mean(KL + trust), odd12 mean(...)) + ridge.
This is exact grouped-KL for *first-order linearized grouped logits* only.
It is NOT exact full-model quantized output KL, nor does it capture within-OTHER
logit Fisher curvature. Original native Apple CPU kernel/memory/stage constraints
from prior Qv8 hold, fixed 28n8/168q4. Strict feasible 1-swap local search.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,sys,time
import numpy as np
from scipy.special import logsumexp
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v8_signed_joint_optimizer as prior

PRE=ROOT/"configs/research/BEGLIN_QV10_TOPK_OTHER_TRUST_PRELOCK_20261023.json"
SOURCE=ROOT/"lab_local/qv10_topk_trust_20261023/original_BF16_top2other_train24_private.json"
SOURCEPUB=ROOT/"docs/research/BEGLIN_QV10_TOP2OTHER_TRAIN24_PUBLIC_20261023.json"
HELD=ROOT/"lab_local/qv10_topk_trust_20261023/RACE_middle_validation_48_locked.json"
MASK=ROOT/"configs/research/BEGLIN_QV10_TOP2OTHER_TRUST_28N8_SOURCE_POLICY_20261023.txt"
POL=ROOT/"docs/research/BEGLIN_QV10_TOPK_OTHER_TRUST_SOURCE_POLICY_20261023.json"
PRE_SHA="4ef4d31a4b9d0be2e015ffaac84a8ee86feb86047a8a8aae4a50318144d5d207"
RACE_SHA="acdfd36477df4f440eeb9d03ad68342a34ce2435028eff1b8cef0de4c9f859de"
BASELINES={
 "Qv6_Fisher64":(ROOT/"configs/research/BEGLIN_QV6_FISHER_64N8_SOURCE_POLICY_20261022.txt",
      "74fb949e45d13f63a26d412effc39377492ea71bda20ae3690f56fddf6303e08"),
 "Qv6_Fisher128":(ROOT/"configs/research/BEGLIN_QV6_FISHER_128N8_SOURCE_POLICY_20261022.txt",
      "8b9e2404f9a38d7b57784b7d663efa19661553ccaa7dc79bfcbf6e6180fc7997"),
 "Qv9_ChoiceGGN24":(ROOT/"configs/research/BEGLIN_QV9_GGN_28N8_FROZEN_MASK_20261023.txt",
      "0553e76e9a9d225e6c6e16a5b5a2ccb75e1794d16ee5c5608bc637a56504c4b2"),
 "Qv8_Signed96":(ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt",
      "9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922")}
N=196;K=28
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
IDS=[(r,i) for r in ROLES for i in range(28)]
def sha(b):return hashlib.sha256(b).hexdigest()
def sealed(p):
 if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:
  raise RuntimeError("original BF16 Qwen signed grouped source source 0600 missing")
 return p.read_bytes()
def inputs():
 if POL.exists() or MASK.exists():raise FileExistsError("source mask previously frozen; no post-evaluation refitting")
 if sha(PRE.read_bytes())!=PRE_SHA or sha(sealed(HELD))!=RACE_SHA:
  raise RuntimeError("new source prior prelock or held RACE article hashes drift")
 raw=sealed(SOURCE);meta=json.loads(SOURCEPUB.read_text())
 if meta["private_full_grouped_source_SHA256"]!=sha(raw) or \
    not meta["numeric_group_probability_mass_original_fullvocab_exact"] or \
    meta["stage"]!="train" or meta["num_cases"]!=24:
  raise RuntimeError("24 source original BF16 topK+OTHER not authentic")
 x=json.loads(raw)
 a=np.asarray(x["signed_Q4_relative_logit_groups"],dtype=np.float64)
 b=np.asarray(x["signed_n8_relative_logit_groups"],dtype=np.float64)
 p=np.asarray(x["source_original_BF16_output_group_probabilities"],dtype=np.float64)
 if a.shape!=(24,N,6) or b.shape!=(24,N,6) or p.shape!=(24,7) or \
    not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(p).all() or np.min(p)<0 or \
    np.max(abs(p.sum(axis=1)-1))>5e-5:
  raise RuntimeError("original grouped signed Q4/n8 source invalid")
 return meta,a,b,p

def grouped_KL(u,p):
 # relative-to-D logit shifts have six channels, D=0; grouped logsumexp
 # is exact for first-order u, NOT for the true nonlinear quantized model.
 z=np.concatenate((u,np.zeros((len(u),1),dtype=np.float64)),axis=1)
 logp=np.log(np.clip(p,1.e-300,1.))
 kl=logsumexp(logp+z,axis=1)-(p*z).sum(axis=1)
 if np.min(kl)<-1e-7 or not np.isfinite(kl).all():
  raise RuntimeError("negative or nonfinite first-order grouped KL")
 return np.maximum(kl,0)
def GGN_per_tensor(original,p):
 # Original: 24×196×6. Returns 24×196 diagonal grouped GGN.
 z=np.concatenate((original,np.zeros((24,196,1),dtype=np.float64)),axis=-1)
 mu=np.einsum("sg,sig->si",p,z)
 return np.maximum(0,.5*(np.einsum("sg,sig->si",p,z*z)-mu*mu))

def optimize():
 meta,q4,n8,p=inputs()
 now=time.monotonic()
 diff=n8-q4
 index={key:i for i,key in enumerate(IDS)}
 costs,swapcost,valid=prior.make_engine()
 q4_d=GGN_per_tensor(q4,p)
 q8_d=GGN_per_tensor(n8,p)
 ridge0=float(q4_d.mean(axis=0).sum()/N)
 rd=(q8_d-q4_d).mean(axis=0)/N
 baseline=q4.sum(axis=1)
 def eval_summary(u,bit):
  kl=grouped_KL(u,p)
  group=np.concatenate((u,np.zeros((len(u),1))),axis=1)
  penalty=.05*np.sum(p*np.maximum(0,abs(group)-2.)**2,axis=1)
  fold=(kl+penalty)
  ridge=float(ridge0+np.dot(bit,rd))
  obj=float(max(fold[::2].mean(),fold[1::2].mean())+.1*ridge)
  return obj,{"grouped_KL_mean":float(kl.mean()),
     "trust_penalty_mean":float(penalty.mean()),
     "even12_mean_KL_plus_trust":float(fold[::2].mean()),
     "odd12_mean_KL_plus_trust":float(fold[1::2].mean()),
     "mean_isolated_grouped_output_GGN_div_196":ridge,
     "objective":obj}
 def embed(chosen):
  b=np.zeros(N,dtype=np.float64)
  for role,layer in chosen:b[index[(role,layer)]]=1
  return b
 def quality(chosen):
  bits=embed(chosen)
  z=baseline+np.einsum("sik,i->sk",diff,bits,optimize=True)
  return eval_summary(z,bits)
 starts=[]
 for name,(path,dig) in BASELINES.items():
  selected=prior.ids_from_mask(path,dig)
  if len(selected)!=K or not valid(costs(selected)):
   raise RuntimeError("prior frozen native NEON cost-feasible allocation no longer valid "+name)
  starts.append((name,selected))
 rng=np.random.default_rng(20261023)
 for seed in (17,41):
  current=starts[seed%len(starts)][1].copy()
  for _ in range(10):
   for tries in range(700):
    old=list(current)[int(rng.integers(len(current)))]
    other=IDS[int(rng.integers(N))]
    if other in current:continue
    candidate=(current-{old})|{other}
    if valid(costs(candidate)):
     current=candidate;break
  starts.append((f"prelock_source_shuffle_{seed}",current))
 reports=[];finished=[]
 for name,seed in starts:
  selected=seed.copy();c=costs(selected)
  bit=embed(selected);u=baseline+np.einsum("sik,i->sk",diff,bit,optimize=True)
  first,components=eval_summary(u,bit)
  swaps=[]
  for it in range(90):
   score,comps=eval_summary(u,bit)
   best=None
   # Evaluate every feasible one-swap change, preserving the same physical stage constraints.
   for old in sorted(selected,key=lambda v:index[v]):
    i=index[old]
    for candidate in IDS:
     if candidate in selected:continue
     j=index[candidate]
     kc=swapcost(selected,c,old,candidate)
     if kc is None:continue
     new_u=u+diff[:,j,:]-diff[:,i,:]
     newbit=bit.copy();newbit[i]=0;newbit[j]=1
     value,_=eval_summary(new_u,newbit)
     step=value-score
     pair=(i,j)
     if step>=-1e-9:continue
     if best is None or step<best[0]-1e-10 or (abs(step-best[0])<=1e-10 and pair<best[4]):
      best=(step,old,candidate,new_u,pair,kc)
   if best is None:break
   amount,old,new,new_u,_,new_cost=best
   newselected=(selected-{old})|{new}
   if not valid(costs(newselected)):raise RuntimeError("new source bit policy violates actual CPU resource constraints")
   newscore,_=quality(newselected)
   if abs(newscore-score-amount)>max(1.e-4,abs(amount)*1.e-5):
    raise RuntimeError("incremental topK+other source objective differs from exact recomputation")
   bit[index[old]]=0;bit[index[new]]=1
   selected=newselected;u=new_u;c=new_cost
   swaps.append([list(old),list(new),float(amount)])
  val,comps=eval_summary(u,bit)
  print("QV10_REAL_ORIGINAL_BF16_TOPK_OTHER_TRUST_SOURCE_LOCAL_SEARCH",
      name,"start",round(first,5),"final",round(val,5),"swaps",len(swaps),
      "seconds",round(time.monotonic()-now,1),flush=True)
  reports.append({"start":name,"source_before":first,"source_after":val,
      "accepted_feasible_swaps":swaps,"source_components":comps,
      "selected_ids":[list(x) for x in sorted(selected,key=lambda k:index[k])],
      "selected_native_kernel_resources":costs(selected),
      "not_global_optimality":True})
  finished.append((val,name,selected))
 best=min(finished,key=lambda v:(v[0],v[1]))
 selected=best[2];ordered=sorted(selected,key=lambda k:index[k])
 mask="".join(f"{role} {layer}\n" for role,layer in ordered).encode()
 if len(selected)!=K or len(set(selected))!=K or not valid(costs(selected)):
  raise RuntimeError("bad 28 tensor grouped full-vocab source allocation")
 os.umask(0o077)
 with MASK.open("xb") as f:f.write(mask);f.flush();os.fsync(f.fileno())
 baseline_stats={"all_Q4":quality(set())[1]}
 for name,case in starts[:4]:
  baseline_stats[name]=quality(case)[1]
 final_stats=quality(selected)[1]
 receipt={"schema":"BEGLIN-Qv10-real-original-BF16-top2-other-trust-grouped-fullvocab-mass-28n8/1",
  "status":"SOURCE_ONLY_REAL_TOPK_OTHER_GROUPED_KL_28N8_MASK_FROZEN_BEFORE_DIAG_OR_RACE",
  "prelock_SHA256":PRE_SHA,"original_RACE48_input_SHA256":RACE_SHA,
  "original_BF16_source24_private_SHA256":meta["private_full_grouped_source_SHA256"],
  "original_BF16_source24_public_SHA256":sha(SOURCEPUB.read_bytes()),
  "full_vocabulary_mass_exactly_conserved_by_7group_partition":True,
  "full_vocabulary_Hessian_not_exact_due_to_OTHER_collapsing":True,
  "grouped_exact_KL_is_only_of_linearly_predicted_logits":True,
  "top_nonchoice_tokens_per_source":2,
  "fixed_trust_relative_logit_threshold":2.0,"fixed_trust_regularization_weight":.05,
  "fixed_isolated_per_tensor_grouped_GGN_weight":.10,
  "source_fold_worst_even_odd_12_optimization":True,
  "prior_mask_source_only_objectives":baseline_stats,
  "start_modes_explored":reports,
  "selected_start":best[1],
  "selected_source_train_objective":final_stats,
  "selected_ids":[list(x) for x in ordered],
  "n8_count":28,"q4_count":168,
  "mask_sha256":sha(mask),
  "original_Apple_NEON_kernel_resources":costs(selected),
  "source_only_single_swap_local_no_global_optimum_proof":True,
  "source24_used_for_selection":True,
  "source_diagnostic24_not_opened":True,
  "new_RACE48_logits_and_labels_not_used_to_tune":True,
  "normal_source_not_independent_across_Qv9_prior_work":True,
  "no_automatic_precision_promotion":True,"no_runtime_main_operator_changes":True,
  "elapsed_s":round(time.monotonic()-now,2)}
 data=(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode()
 with POL.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 print("QV10_28N8_TOPK_OTHER_FULLVOCAB_MASS_TRUST_POLICY_FROZEN",json.dumps({
    "chosen":best[1],"training_obj":final_stats["objective"],
    "q4_obj":baseline_stats["all_Q4"]["objective"],
    "mask_sha":sha(mask),"policy_sha":sha(data),
    "source24_mean_source_other_mass":meta["mean_mass_OTHER"],
    "seconds":receipt["elapsed_s"]},sort_keys=True),flush=True)
if __name__=="__main__":optimize()
