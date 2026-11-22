#!/usr/bin/env python3
"""JE01 full27 native independently paired TRAIN16 original/A/B/C NLL audit.

This script verifies real sealed original OFF/B/C 656-token native output
receipts and pre-existing 27-layer A source, then recomputes paired 16-cluster
uncertainty. It never loads heldout data, never promotes policy, and does not
claim empirical Fisher without weight/mixing gradients.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

WORK=Path("/Volumes/D50/TailnetCommander/work")
RUN=WORK/"je01_postnll_bc_original27_native_gold16_20261123_v1"
BINARY_SHA="d53db80fd7326d5c247fc8d9544b70b60b4601abd1fbe84715dd180b297737ac"
ORIGINAL_SHA="b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c"
A_ON_SHA="9b0f4644ab8cfbd8c4119bb59d3aaf5e34205d4338208fd1f95b21f828242f8d"
BASELINE_NLL=2.291799702354307
BUNDLE_SHA={
 "B":"fcf33feb58b87724f8be673c8457cf2d0b3e2ab0cf947ed6c3b0cb6e91f67fcf",
 "C":"9a61a792d7e529868109ee3e5c240fea545479ca89d681cbf0b0ea6543b17997"}
SRC=Path(__file__).resolve().parents[1]
A_REC=SRC/"docs/research/JE01_REAL27_CAL240_ROLEWISE_ON_656GOLD_NLL_20261122.json"
A_SRC_SHA="54afc4f7e7faa86b6c4a9c21286a95cae02856de2d6e3c8dfa309bb1011c0510"


def fsha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()


def canonical(data):
 return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def need(cond,msg):
 if not cond:raise ValueError(msg)


def signed(path):
 need(path.is_file() and not path.is_symlink(),"JE01_SIGNED_DATA_MISSING")
 d=json.loads(path.read_text())
 seal=d.get("receipt_sha256")
 need(isinstance(seal,str) and canonical({k:v for k,v in d.items() if k!="receipt_sha256"})==seal,
      "JE01_SIGNED_DATA_NOT_CANONICAL")
 return d


def validate(rec,mode):
 need(rec["gold_tokens"]==656 and rec["gold_train_cases"]==16 and
      rec["native_full27_model"] is True and
      rec["native_executable_sha256"]==BINARY_SHA and
      rec["original_AF4_output_sha256"]==ORIGINAL_SHA and
      rec["same_source_original_NLL"]==BASELINE_NLL and
      rec["heldout_validation128"]=="NOT_ACCESSED" and
      rec["heldout_OOD128"]=="NOT_ACCESSED" and
      rec["empirical_fisher_weights"]=="NOT_MEASURED" and
      rec["production_modified"] is False and
      rec["original_model_weights_modified"] is False and
      rec["mode"]==mode and len(rec["per_case"])==16 and
      len(rec["per_gold_target_actual_NLL"])==16,
      "JE01_REAL27_NATIVE_ONOFF_SCOPE_DRIFT")
 for i,(c,vals) in enumerate(zip(rec["per_case"],rec["per_gold_target_actual_NLL"])):
  need(c["row"]==i and len(vals)==c["tokens"] and
       all(math.isfinite(x) and x>=0 for x in vals) and
       abs(math.fsum(vals)-c["measured_mode_NLL_sum"])<1e-8,
       "JE01_REAL27_INDEPENDENT_NATIVE_TOKENS_NOT_MATCHED")
 need(sum(c["tokens"] for c in rec["per_case"])==656 and
      abs(math.fsum(c["measured_mode_NLL_sum"] for c in rec["per_case"])/656 -
          rec["mean_gold_NLL"])<1e-10,
      "JE01_REAL27_NATIVE_MEAN_NOT_RECOMPUTABLE")
 if mode in BUNDLE_SHA:
  need(rec["status"]=="JE01_BC_REAL_ORIGINAL27_ON_NLL_MEASURED" and
       rec["policy_arm"]==("B_PROTECT_EARLY_GATE" if mode=="B" else "C_PROTECT_EARLY_UP") and
       rec["real_physical_bundle_sha256"]==BUNDLE_SHA[mode] and
       rec["real_physical_budget_bytes"]==43253760 and
       rec["real_tier_counts"]=={"AF4":12,"Q4":6,"Q5":6},
       "JE01_REAL27_NEW_POLICY_BYTE_DRIFT")
 else:
  need(mode=="off" and rec["status"]=="JE01_BC_TRUE_ORIGINAL27_OFF_EXACT_SHA_PASS" and
       rec["native_output_sha256"]==ORIGINAL_SHA and
       rec["NLL_candidate_minus_original"]==0 and
       rec["policy_arm"]=="ORIGINAL_AF4",
       "JE01_REAL27_COMPILED_BINARY_OFF_NOT_EXACT_ORIGINAL")
 return rec["per_gold_target_actual_NLL"]


def compare(ref,candidate):
 need(len(ref)==len(candidate)==16,"JE01_PAIRED_16_MISMATCH")
 per_token=[]
 for x,y in zip(ref,candidate):
  need(len(x)==len(y)>0,"JE01_PAIRED_TOKEN_COUNT_MISMATCH")
  row=[b-a for a,b in zip(x,y)]
  need(all(math.isfinite(v) for v in row),"JE01_PAIR_NLL_NONFINITE")
  per_token.append(row)
 token_count=sum(map(len,per_token))
 need(token_count==656,"JE01_PAIR_NOT_EXACT_656")
 delta=math.fsum(math.fsum(x) for x in per_token)/token_count
 return {"delta_NLL":delta,
         "sentence_delta_sums":[math.fsum(x) for x in per_token],
         "sentence_lengths":list(map(len,per_token)),
         "worsened_sentences":sum(math.fsum(x)>0 for x in per_token),
         "improved_sentences":sum(math.fsum(x)<0 for x in per_token),
         "worsened_tokens":sum(v>0 for x in per_token for v in x),
         "improved_tokens":sum(v<0 for x in per_token for v in x)}


def shared_cluster_bootstrap(contrasts,draws=20000,seed=20261123):
 names=list(contrasts)
 for item in contrasts.values():
  need(len(item["sentence_delta_sums"])==len(item["sentence_lengths"])==16,"JE01_PAIRED_BOOTSTRAP_INVALID")
 rng=random.Random(seed)
 samples={k:[] for k in names}
 for _ in range(draws):
  idx=[rng.randrange(16) for _ in range(16)]
  for k in names:
   d=contrasts[k]
   total=math.fsum(d["sentence_delta_sums"][j] for j in idx)
   n=sum(d["sentence_lengths"][j] for j in idx)
   samples[k].append(total/n)
 result={}
 for k in names:
  a=sorted(samples[k])
  result[k]={
   "percentile95_NLL_delta":[a[int(.025*draws)],a[int(.975*draws)]],
   "fraction_bootstrap_delta_lower":sum(x<0 for x in a)/draws}
 return {"seed":seed,"draws":draws,"method":"SHARED_SENTENCE_CLUSTER_PERCENTILE_BOOTSTRAP_EXPLORATORY",
         "contrasts":result}


def audit():
 off=signed(RUN/"off/JE01_BC_REAL27_GOLD16_NATIVE_NLL.json")
 orig=validate(off,"off")
 recs={"OFF":off}
 for mode in ("B","C"):
  d=signed(RUN/mode/"JE01_BC_REAL27_GOLD16_NATIVE_NLL.json")
  validate(d,mode)
  for i,(o,c) in enumerate(zip(orig,d["per_gold_target_actual_NLL"])):
   need(len(o)==len(c),"JE01_SOURCE_GOLD_TOKEN_ALIGNMENT_CHANGED")
  recs[mode]=d
 need(fsha(A_REC)==A_SRC_SHA,"JE01_FROZEN_A_ON_RECEIPT_SHA_CHANGED")
 d=signed(A_REC)
 need(d["status"]=="ACTUAL_REAL_ORIGINAL27_CAL240_ROLEWISE_PILOT16_MEASURED" and
      d["gold_tokens"]==656 and d["actual_model_complete_27layers"] is True and
      d["real_native_output_sha256"]==A_ON_SHA and
      d["production_checkpoint_modified"] is False and
      d["true_source_heldout_ood_labels_accessed"] is False,
      "JE01_OLD_A_NATIVE_ON_SOURCE_CHANGED")
 values={"OFF":orig,"A":d["per_gold_target_actual_NLL"],
         "B":recs["B"]["per_gold_target_actual_NLL"],
         "C":recs["C"]["per_gold_target_actual_NLL"]}
 pairs=[("OFF","A"),("OFF","B"),("OFF","C"),("A","B"),("A","C"),("B","C")]
 comp={a+"_vs_"+b:compare(values[a],values[b]) for a,b in pairs}
 bs=shared_cluster_bootstrap(comp)
 out={
  "schema":"beglin-je01-original27-post-nll-bc-actual-fullgold-independent-audit/1",
  "status":"JE01_REAL27_POSTNLL_BC_PAIRED_NATIVE_NLL_INDEPENDENT_AUDIT_PASS",
  "actual_NLL_means":{"OFF":off["mean_gold_NLL"],"A":d["actual_native_mean_gold_NLL"],
                      "B":recs["B"]["mean_gold_NLL"],"C":recs["C"]["mean_gold_NLL"]},
  "source_receipt_sha256":{"OFF":off["receipt_sha256"],"A":d["receipt_sha256"],
                           "B":recs["B"]["receipt_sha256"],"C":recs["C"]["receipt_sha256"]},
  "pre_registered_binary_sha":BINARY_SHA,
  "physical_B_C_sha256":BUNDLE_SHA,
  "source_original_AF4_output_sha256":ORIGINAL_SHA,
  "source_A_on_output_sha256":A_ON_SHA,
  "cases":16,"gold_tokens":656,
  "comparisons":comp,
  "shared_cluster_bootstrap":bs,
  "design_scope":"B_C_EQUAL43MB_PREREG_TWO_EARLY_GATE_UP_POLICIES_ALREADY_SEEN_TRAIN16_ONLY",
  "true_empirical_Fisher_parameter_matrix":"NOT_MEASURED",
  "independent_validation128":"NOT_ACCESSED",
  "independent_OOD128":"NOT_ACCESSED",
  "policy_adoption":"HOLD",
  "production_modified":False}
 out["receipt_sha256"]=canonical(out)
 return out


if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--output",type=Path,required=True)
 args=p.parse_args()
 r=audit()
 if args.output.exists() or args.output.is_symlink():
  raise FileExistsError("JE01_AUDIT_APPEND_ONLY_SOURCE_MUST_NOT_OVERWRITE")
 args.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
 print(json.dumps({k:v for k,v in r.items() if k not in ("comparisons",)},sort_keys=True),flush=True)
 for k,v in r["comparisons"].items():
  print("JE01_REAL27_PAIR",k,v["delta_NLL"],r["shared_cluster_bootstrap"]["contrasts"][k],flush=True)
