#!/usr/bin/env python3
"""BEGLIN Qv13: disjoint-from-pilot 240-item aggregate analysis, original C.
No fitting, mask adjustment, new native forward, production change, or per-item export.
The 240-only analysis is POST-PILOT specified; it is not pristine preregistration.
"""
import argparse, collections, hashlib, json, math, pathlib, random
from typing import Mapping

ROOT=pathlib.Path(__file__).resolve().parents[1]
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv13-20261023")
PRE=ROOT/"configs/research/BEGLIN_QV13_ACTUAL_NATIVE_EVAL_PRELOCK_20261023.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV13_ORIGINAL_NATIVE_256_PREEXEC_LOCK_PUBLIC_20261023.json"
PRIVATE=LAB/"qv13_lock_private.json"
SHA={
 "pre":"5cb944375fa271c4528b4846f4b3c22cdf8179d4a38a2f5c17babbc03c4df92a",
 "public":"9d045e1056e5ec8bc3a093a8a3d198277cbbc4083fb054a929b566cb2bb03c8c",
 "private":"79826eb944ac2a4f87b637a430471f069a3789da37cdb12a9b7d1060d49db9b8"}
BATCHES=("arc128","ob128")
ARMS=("q4","source_qv8","random73","random137","random211","random389","random577","original_v2")
RANDOM=("random73","random137","random211","random389","random577")
RNGSEED=20261023

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for chunk in iter(lambda:f.read(1<<18),b""):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def must(cond,why):
 if not cond:raise RuntimeError("QV13_FAIL_CLOSED_"+why)
def verify_freeze():
 for name,p in (("pre",PRE),("public",PUBLIC),("private",PRIVATE)):
  must(p.is_file() and sha(p)==SHA[name],name+"_identity")
 lock=read(PRIVATE)
 must(lock["prelock_sha256"]==SHA["pre"],"prelock")
 must(lock["status"]=="FROZEN_BEFORE_QV13_NATIVE_C_FORWARD","status")
 manifests={}
 for batch in ("pilot16",)+BATCHES:
  p=LAB/"native_runs"/batch/"input_manifest.json"
  must(sha(p)==lock["batches"][batch]["manifest_sha256"],batch+"_manifest")
  data=read(p)
  must(data["count"]==(16 if batch=="pilot16" else 128),"count_"+batch)
  ids=[x["item_id_sha256"] for x in data["original_items"]]
  must(len(ids)==len(set(ids)),"unique_"+batch)
  manifests[batch]=data
 pilot=manifests["pilot16"]["original_items"]
 arc=manifests["arc128"]["original_items"];ob=manifests["ob128"]["original_items"]
 must([v["item_id_sha256"] for v in pilot]==[v["item_id_sha256"] for v in arc[:8]+ob[:8]],"pilot16_not_exact_first8_each")
 must(set(x["item_id_sha256"] for x in arc).isdisjoint(set(x["item_id_sha256"] for x in ob)),"cross_domain_overlap")
 for arm in ARMS[1:]:
  info=lock["masks"][arm]
  path=LAB/"masks"/f"{arm}.txt"
  must(sha(path)==info["mask_sha256"],"mask_"+arm)
 must(all(lock["masks"][r]["projection_packed_bytes"]==lock["masks"]["source_qv8"]["projection_packed_bytes"] for r in RANDOM),"random_byte_mismatch")
 must(all(lock["masks"][r]["role_counts"]==lock["masks"]["source_qv8"]["role_counts"] for r in RANDOM),"random_role_mismatch")
 return lock,manifests

def load_original_receipts(lock,manifests):
 # Independently validate every native raw C log before accepting its receipt.
 # Imported here so --check does not require any full-model results.
 import importlib.util
 spec=importlib.util.spec_from_file_location("qv13_native_log_crosscheck",ROOT/"tools/qv13_native_log_crosscheck.py")
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 for batch in BATCHES:
  summary=mod.scan(batch)
  must(summary["status"]=="ALL_EIGHT_NATIVE_C_LOG_REPARSE_PASS","incomplete_independent_raw_C_"+batch)
 groups={k:{} for k in ARMS}
 receipts={}
 for batch in BATCHES:
  for arm in ARMS:
   loc=LAB/"native_runs"/batch/arm/"native_qv13_original_C_receipt.json"
   r=read(loc)
   must(r["n"]==128 and len(r["article_fullvocab_gold_NLL"])==128,"native_sample_"+arm+batch)
   must(r["input_manifest_sha256"]==sha(LAB/"native_runs"/batch/"input_manifest.json"),"input_sha_"+arm+batch)
   vals=r["article_fullvocab_gold_NLL"][8:]
   acc=r["article_correct"][8:]
   must(len(vals)==120 and len(acc)==120,"held120_partition_"+arm+batch)
   must(all(math.isfinite(v) and v>=0 for v in vals),"finite_NLL_"+arm+batch)
   must(all(v in (0,1) for v in acc),"accuracy_binary_"+arm+batch)
   groups[arm][batch]={"nll":vals,"correct":acc}
   receipts[batch+":"+arm]=sha(loc)
 return groups,receipts

def strat_ci(samples_by_domain,seed,nboot=10000):
 # Pre-specified equal two-domain mixture: resample within each domain.
 rng=random.Random(seed);reps=[]
 for k in range(nboot):
  means=[]
  for batch in BATCHES:
   d=samples_by_domain[batch];n=len(d)
   means.append(sum(d[rng.randrange(n)] for _ in range(n))/n)
  reps.append(sum(means)/2)
 reps.sort()
 return [reps[int(.025*nboot)],reps[int(.975*nboot)-1]]

def sign_flip_p(samples_by_domain,seed,draws=10000):
 # Diagnostic random-sign test, one-sided; requires exchangeability under null.
 observations=[v for batch in BATCHES for v in samples_by_domain[batch]]
 n=len(observations);actual=sum(observations)/n;rng=random.Random(seed);large=0
 for k in range(draws):
  trial=sum(v*(1 if rng.getrandbits(1) else -1) for v in observations)/n
  large+=trial>=actual
 return (large+1)/(draws+1)

def mcnemar_discordant(src,comp):
 # Two-sided exact McNemar binomial tail on discordant pairs.
 lose=win=0
 for batch in BATCHES:
  for a,b in zip(src[batch]["correct"],comp[batch]["correct"]):
   win+=int(a==1 and b==0)
   lose+=int(a==0 and b==1)
 discordant=win+lose
 if discordant==0:p=1.
 else:p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(win,lose)+1))/2**discordant)
 return {"source_only_correct":win,"comparator_only_correct":lose,"mcnemar_exact_two_sided_exploratory_p":p}

def analyze():
 lock,manifest=verify_freeze()
 groups,receipts=load_original_receipts(lock,manifest)
 per_arm={}
 for arm in ARMS:
  d=groups[arm]
  pooled=[v for b in BATCHES for v in d[b]["nll"]]
  acc=[v for b in BATCHES for v in d[b]["correct"]]
  per_arm[arm]={
   "n_postpilot":len(pooled),"correct":sum(acc),"mean_correct_fullvocab_NLL":sum(pooled)/len(pooled),
   "projection_packed_bytes":lock["original_q4_packed_bytes"] if arm=="q4" else lock["masks"][arm]["projection_packed_bytes"],
   "per_dataset":{b:{"n":120,"correct":sum(d[b]["correct"]),"mean_NLL":sum(d[b]["nll"])/120} for b in BATCHES}}
 src=groups["source_qv8"]
 comparisons={}
 pvals=[]
 for i,arm in enumerate(RANDOM):
  alt=groups[arm]
  diff={b:[x-y for x,y in zip(alt[b]["nll"],src[b]["nll"])] for b in BATCHES}
  mu=sum(v for b in BATCHES for v in diff[b])/240
  p=sign_flip_p(diff,RNGSEED+i)
  pvals.append(p)
  comparisons[arm]={"paired_NLL_gain_source_over_random":mu,"stratified_bootstrap95":strat_ci(diff,RNGSEED+i),
   "source_better_items":sum(v>0 for b in BATCHES for v in diff[b]),
   "source_worse_items":sum(v<0 for b in BATCHES for v in diff[b]),
   "source_correct_minus_random":per_arm["source_qv8"]["correct"]-per_arm[arm]["correct"],
   "paired_accuracy_discordance":mcnemar_discordant(src,alt),
   "sign_flip_one_sided_p_exploratory":p}
 indexed=sorted(range(len(RANDOM)),key=lambda i:pvals[i])
 monotonic=0
 for rank,i in enumerate(indexed):
  monotonic=max(monotonic,min(1.,(len(RANDOM)-rank)*pvals[i]))
  comparisons[RANDOM[i]]["sign_flip_Holm_adjusted_p_exploratory"]=monotonic
 diffmean={b: [
   sum(groups[arm][b]["nll"][i] for arm in RANDOM)/5-src[b]["nll"][i]
   for i in range(120)] for b in BATCHES}
 return {
 "schema":"BEGLIN-Qv13-postpilot-240-independent-original-C-aggregate/1",
 "status":"POSTPILOT_240_REAL_28L_C_EVALUATION_REPARSED_EXPLORATORY_NOT_PRISTINE_PREREG",
 "sample":"ARC-Challenge test 120 + OpenBookQA test 120, excluding first eight pilot items of each domain",
 "n_postpilot":240,"n_excluded_pilot":16,
 "primary_descriptive_estimand":"Qv8 source minus mean of five matched random models, paired fullvocab answer-token NLL; positive random-minus-source favors Qv8",
 "source_gain_vs_5random_mean_NLL":sum(v for b in BATCHES for v in diffmean[b])/240,
 "source_gain_vs_5random_stratified95":strat_ci(diffmean,RNGSEED),
 "per_arm":per_arm,"five_matched_random":comparisons,
 "original_C_receipt_SHA256":receipts,
 "source_prelock_SHA256":SHA["pre"],"source_private_lock_SHA256":SHA["private"],
 "original_input_SHA256":{b:lock["batches"][b]["manifest_sha256"] for b in BATCHES},
 "limits":["This analysis was devised after 16 pilot results were inspected; not a pristine pre-registered final claim",
 "Qv13 source prelock has documented implementation-text errata",
 "Public test data may have appeared in base-model pretraining",
 "Results are for static tensor masks, not per-request dynamic changes",
 "Projection packed bytes are not total peak RSS or guaranteed speedups",
 "Source-vs-random sign-flip p-values require conditional sign exchangeability; exploratory only",
 "No per-item question, token sequence or predictions exported"],
 "no_live_precision_promotion":True}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--check",action="store_true")
 p.add_argument("--output",type=pathlib.Path)
 args=p.parse_args()
 if args.check:
  lock,manifest=verify_freeze()
  print(json.dumps({"status":"POSTPILOT_PARTITION_240_FROZEN_INPUTS_VERIFIED","pilot_overlap":16,
     "heldout_arc":120,"heldout_ob":120,"qv13_private_SHA256":SHA["private"],
     "batch_manifest_SHA256":{b:lock["batches"][b]["manifest_sha256"] for b in BATCHES},
     "same_bytes_role_random_controls":True},sort_keys=True))
 else:
  result=analyze()
  s=json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n"
  if args.output:
   args.output.parent.mkdir(parents=True,exist_ok=True)
   with args.output.open("x",encoding="utf-8") as f:f.write(s)
   print("SAVED_QV13_AGGREGATE_ONLY",args.output)
  else:print(s)
if __name__=="__main__":main()
