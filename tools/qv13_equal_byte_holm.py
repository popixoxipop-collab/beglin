#!/usr/bin/env python3
"""Qv13 fixed-mask comparison: paired item bootstrap + exploratory Holm sign-flip audit.
Never adapts policies or exports raw questions/case predictions.
"""
import argparse, hashlib, json, pathlib, random
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv13-20261023")
SOURCE="source_qv8"
RANDOMS=("random73","random137","random211","random389","random577")
PUBLISHED_PRE_SHA="5cb944375fa271c4528b4846f4b3c22cdf8179d4a38a2f5c17babbc03c4df92a"
LOCK_SHA="79826eb944ac2a4f87b637a430471f069a3789da37cdb12a9b7d1060d49db9b8"
BATCH_GROUPS={"pilot16":("pilot16",),"arc128":("arc128",),"ob128":("ob128",),"full256":("arc128","ob128")}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def boot_ci(d,draws=10000,seed=20261023):
 rng=random.Random(seed);n=len(d)
 samples=sorted(sum(d[rng.randrange(n)] for _ in range(n))/n for _ in range(draws))
 return [samples[int(.025*draws)],samples[int(.975*draws)-1]]
def signed_p(d,draws=10000,seed=20261023):
 # One-sided random-sign test of mean difference. Exploratory only:
 # relies on sign exchangeability (NOT implied by a matched observational sample).
 r=random.Random(seed);n=len(d);observed=sum(d)/n
 t=0
 for j in range(draws):
  x=sum((1 if r.getrandbits(1) else -1)*v for v in d)/n
  if x>=observed:t+=1
 return (t+1)/(draws+1)
def analyze(batch):
 if batch not in BATCH_GROUPS:raise ValueError(batch)
 b=BATCH_GROUPS[batch]
 lockp=LAB/"qv13_lock_private.json"
 if sha(lockp)!=LOCK_SHA:raise RuntimeError("Qv13 frozen original lock drift")
 lock=read(lockp)
 if lock["prelock_sha256"]!=PUBLISHED_PRE_SHA:raise RuntimeError("prelock drift")
 names=(SOURCE,)+RANDOMS
 data={x:[] for x in names};correct={x:[] for x in names};file_shas={}
 for segment in b:
  p=LAB/"native_runs"/segment/"input_manifest.json"
  if sha(p)!=lock["batches"][segment]["manifest_sha256"]:raise RuntimeError("input manifest drift")
  for name in names:
   receipt=LAB/"native_runs"/segment/name/"native_qv13_original_C_receipt.json"
   if not receipt.is_file():
    return dict(schema="BEGLIN-Qv13-five-random-policy-exploratory/1",
      status="NOT_ENOUGH_ACTUAL_NATIVE_POLICY_RESULTS",batch=batch,
      missing=[f"{x}:{z}" for x in names for z in b if not (LAB/"native_runs"/z/x/"native_qv13_original_C_receipt.json").is_file()])
   r=read(receipt)
   if r["status"]!="ORIGINAL_NATIVE_28L_REAL_Q4_OR_28N8_SCORING_PASS" or r["preexecution_lock_sha256"]!=LOCK_SHA or r["input_manifest_sha256"]!=sha(p):
    raise RuntimeError("receipt independent authority mismatch")
   if r["mask_sha256"]!=lock["masks"][name]["mask_sha256"]:raise RuntimeError("mask changed")
   if r["n"]!=lock["batches"][segment]["count"]:raise RuntimeError("count altered")
   if sha(LAB/"native_runs"/segment/name/"native_qv13_original_C_run.log")!=r["native_log_sha256"]:
    raise RuntimeError("original C log sha drift")
   data[name].extend(r["article_fullvocab_gold_NLL"])
   correct[name].extend(r["article_correct"])
   file_shas[segment+":"+name]=sha(receipt)
 n=len(data[SOURCE])
 if not n or any(len(data[x])!=n for x in names):raise RuntimeError("paired sample missing")
 costs=[lock["masks"][x]["projection_packed_bytes"] for x in names]
 if len(set(costs))!=1 or costs[0]!=765493248:raise RuntimeError("five control costs not matched")
 counts=[lock["masks"][x]["role_counts"] for x in names]
 if not all(x==counts[0] for x in counts):raise RuntimeError("five control role counts not matched")
 rawp=[]
 for index,other in enumerate(RANDOMS):
  d=[x-y for x,y in zip(data[other],data[SOURCE])]
  rawp.append(signed_p(d,seed=20261023+index))
 order=sorted(range(len(RANDOMS)),key=lambda i:rawp[i])
 holm=[None]*len(RANDOMS);running=0.
 for rank,index in enumerate(order):
  running=max(running,min(1.,(len(RANDOMS)-rank)*rawp[index]))
  holm[index]=running
 comps={}
 for i,name in enumerate(RANDOMS):
  d=[x-y for x,y in zip(data[name],data[SOURCE])]
  comps[name]={"random_correct":sum(correct[name]),
   "random_mean_NLL":sum(data[name])/n,
   "source_minus_random_correct":sum(correct[SOURCE])-sum(correct[name]),
   "paired_mean_random_minus_source_NLL":sum(d)/n,
   "paired_bootstrap_95_unadjusted":boot_ci(d,seed=20261023+i),
   "source_better_question_count":sum(v>0 for v in d),
   "source_worse_question_count":sum(v<0 for v in d),
   "exploratory_sign_flip_one_sided_p":rawp[i],
   "exploratory_sign_flip_Holm_p":holm[i]}
 mean_random=[sum(data[name][j] for name in RANDOMS)/len(RANDOMS) for j in range(n)]
 combined=[x-y for x,y in zip(mean_random,data[SOURCE])]
 return dict(schema="BEGLIN-Qv13-five-frozen-equal-byte-random-controls-exploratory/1",
    status="PILOT_EXPLORATORY_ALL_SIX_REAL_NATIVE" if batch=="pilot16" else "NEW_DATA_RESULT_DESCRIPTIVE_NOT_CONFIRMATORY",
    batch=batch,articles=n,source_correct=sum(correct[SOURCE]),
    source_mean_gold_fullvocab_NLL=sum(data[SOURCE])/n,
    matched_packed_projection_bytes=costs[0], matched_role_counts=counts[0],
    mean_random_gold_fullvocab_NLL=sum(mean_random)/n,
    source_improvement_vs_mean_random_NLL=sum(combined)/n,
    source_improvement_vs_mean_random_bootstrap95=boot_ci(combined,seed=20261023),
    five_random_controls=comps,
    bootstrap_draws=10000,sign_flip_draws=10000,
    multiplicity="Holm over five one-sided paired sign-flip comparisons; approximate Monte Carlo",
    sign_flip_assumption="requires null sign exchangeability, not guaranteed; exploratory p-values only",
    context="Partial pilot results inspected before full 256; preregistration wording deviates from code; this is not independent confirmation",
    preexecution_private_lock_sha256=LOCK_SHA,original_receipt_sha256=file_shas,
    no_per_item_losses_exported=True,no_dynamic_precision_switch_proven=True,no_live_promotion=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--batch",choices=tuple(BATCH_GROUPS),default="pilot16")
 args=p.parse_args()
 print(json.dumps(analyze(args.batch),sort_keys=True,indent=2))
