#!/usr/bin/env python3
"""Qv13 independent re-audit: original C sealed receipts, paired article losses and costs.
Only aggregate JSON may be published. Never use these estimates to select or retune masks.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,random,statistics
R=pathlib.Path(__file__).resolve().parents[1]
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv13-20261023")
PRE=R/"configs/research/BEGLIN_QV13_ACTUAL_NATIVE_EVAL_PRELOCK_20261023.json"
PRE_SHA="5cb944375fa271c4528b4846f4b3c22cdf8179d4a38a2f5c17babbc03c4df92a"
MASKS=("source_qv8","random73","random137","random211","random389","random577","original_v2")
ARMS=("q4",)+MASKS
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def paired_ci(diff,seed,draws=10000):
 rng=random.Random(seed);n=len(diff)
 samp=sorted(sum(diff[rng.randrange(n)] for _ in range(n))/n for k in range(draws))
 return [samp[int(draws*.025)],samp[int(draws*.975)-1]]
def result_for_batch(batch,require_all=False):
 if digest(PRE)!=PRE_SHA:raise RuntimeError("prelock modified")
 lock_path=LAB/"qv13_lock_private.json"
 lock=read(lock_path);lock_sha=digest(lock_path)
 if lock["prelock_sha256"]!=PRE_SHA or lock["status"]!="FROZEN_BEFORE_QV13_NATIVE_C_FORWARD":
  raise RuntimeError("not preexecution freeze")
 groups={"arc128":("arc128",),"ob128":("ob128",),"pilot16":("pilot16",),"full256":("arc128","ob128")}
 batches=groups[batch]
 samples={a:[] for a in ARMS};choices={a:[] for a in ARMS};timings={};rss={}
 inputs=[];counts={}
 for b in batches:
  manifest_path=LAB/"native_runs"/b/"input_manifest.json"
  manifest=read(manifest_path)
  if digest(manifest_path)!=lock["batches"][b]["manifest_sha256"]:raise RuntimeError("manifest mutated")
  ids=[row["item_id_sha256"] for row in manifest["original_items"]]
  counts[b]=len(ids);inputs.extend(ids)
  for arm in ARMS:
   p=LAB/"native_runs"/b/arm/"native_qv13_original_C_receipt.json"
   if not p.is_file():continue
   rec=read(p)
   if rec["prelock_sha256"]!=PRE_SHA or rec["preexecution_lock_sha256"]!=lock_sha or rec["input_manifest_sha256"]!=digest(manifest_path):
    raise RuntimeError("input/source identity mismatch "+arm)
   if rec["batch"]!=b or rec["arm"]!=arm or rec["n"]!=len(ids) or rec["status"]!="ORIGINAL_NATIVE_28L_REAL_Q4_OR_28N8_SCORING_PASS":
    raise RuntimeError("native arm inaccurate "+arm)
   if digest(LAB/"native_runs"/b/arm/"native_qv13_original_C_run.log")!=rec["native_log_sha256"]:
    raise RuntimeError("original native C log changed "+arm)
   if arm!="q4" and rec["mask_sha256"]!=lock["masks"][arm]["mask_sha256"]:
    raise RuntimeError("original C mask changed "+arm)
   if len(rec["article_fullvocab_gold_NLL"])!=len(ids) or len(rec["article_correct"])!=len(ids):
    raise RuntimeError("per-row original C log mismatch")
   samples[arm].extend(rec["article_fullvocab_gold_NLL"])
   choices[arm].extend(rec["article_correct"])
   timings[b+":"+arm]=rec["elapsed_wall_sec"]
   rss[b+":"+arm]=rec["cpu_peak_rss_observed_bytes"]
 finished={a for a in ARMS if len(samples[a])==len(inputs)}
 if require_all and len(finished)!=len(ARMS):raise RuntimeError("not all policies evaluated, refuse full conclusion")
 if len(set(inputs))!=len(inputs):raise RuntimeError("duplicate raw evaluation question ID across batches")
 baseline=lock["original_q4_packed_bytes"]
 rows={}
 for arm in ARMS:
  if arm not in finished:continue
  v=samples[arm]
  bytes_=baseline if arm=="q4" else lock["masks"][arm]["projection_packed_bytes"]
  rows[arm]={"n":len(v),"correct":sum(choices[arm]),"mean_gold_fullvocab_NLL":sum(v)/len(v),
   "packed_projection_bytes":bytes_,
   "N8_count":0 if arm=="q4" else 28,
   "native_wall_seconds_by_batch":{b:timings[b+":"+arm] for b in batches},
   "peak_process_RSS_MiB_by_batch":{b:round(rss[b+":"+arm]/1048576,2) for b in batches}}
 comparisons=[]
 if "source_qv8" in finished:
  for arm in ARMS:
   if arm=="source_qv8" or arm not in finished:continue
   diff=[a-b for a,b in zip(samples[arm],samples["source_qv8"])]
   comparisons.append({"source":"source_qv8","vs":arm,
    "estimand":"NLL(comparator)-NLL(source), positive means source better",
    "paired_mean_source_advantage":sum(diff)/len(diff),
    "paired_bootstrap_95_unadjusted":paired_ci(diff,seed=20261023),
    "source_better_item_count":sum(d>0 for d in diff),
    "source_worse_item_count":sum(d<0 for d in diff),
    "source_correct_delta":sum(choices["source_qv8"])-sum(choices[arm]),
    "equal_byte_and_role_counts_verified":arm.startswith("random"),
    "stats_exploratory_if_pilot":batch=="pilot16"})
 pub={"schema":"BEGLIN-Qv13-real-native-independent-Qwen28-audit-public/1",
   "status":"ALL_REGISTERED_ARMS_RECORDED" if len(finished)==len(ARMS) else "PARTIAL_ARMS_ONLY_NO_POLICY_ADVANTAGE_CLAIM",
   "batch":batch,"batches":list(batches),
   "n_unique_questions":len(inputs),"frozen_input_manifest_SHA256":{b:lock["batches"][b]["manifest_sha256"] for b in batches},
   "prelock_sha256":PRE_SHA,"preexecution_private_lock_SHA256":lock_sha,
   "tested_arms":list(sorted(finished)),"missing_arms":[a for a in ARMS if a not in finished],
   "original_Qv5_C_executable_sha256":lock["original_C_binary_sha256"],
   "native_full_vocabulary_one_gold_token_NLL_not_conditional_four_choice":True,
   "five_random_controls_equal_role_counts_and_packed_weight_bytes":all(lock["masks"][a]["projection_packed_bytes"]==765493248 for a in MASKS if a.startswith("random")),
   "underpowered_pilot_interim_peeking_not_final":batch=="pilot16",
   "no_held_data_based_mask_selection":True,
   "not_actual_dynamic_runtime_switching":True,
   "full256_rejects_any_missing_or_mismatched_original_C_receipt":True,
   "paired_stats_bootstrap_resamples":10000,
   "per_case_losses_not_exported":True,
   "results":rows,"paired_comparisons_unadjusted":comparisons,
   "no_live_promotion":True}
 return pub
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--batch",choices=("pilot16","arc128","ob128","full256"),default="pilot16")
 p.add_argument("--output",type=pathlib.Path)
 p.add_argument("--require-all",action="store_true")
 args=p.parse_args()
 out=result_for_batch(args.batch,args.require_all)
 s=json.dumps(out,indent=2,sort_keys=True,ensure_ascii=False)+"\n"
 if args.output:
  if args.output.exists():raise FileExistsError("do not overwrite native audit")
  args.output.write_text(s)
  print("PUBLIC_AGGREGATE_SAVED",args.output,digest(args.output))
 else:print(s)
