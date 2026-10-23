#!/usr/bin/env python3
"""Second independent Qv13 research audit. READ-ONLY original evidence, aggregate only.
Reconstructs source test data, n8 masks and individual original C scores independently
of qv13_native_256, qv13_native_independent_audit, and qv13_equal_byte_holm.
Does not invoke inference, edit original receipts or publish raw items.
"""
from pathlib import Path
from collections import Counter
import hashlib, difflib, json, random, re, struct, math
import numpy as np
from scipy.stats import binomtest
from datasets import Dataset
from tokenizers import Tokenizer

R=Path(__file__).resolve().parents[1]
L=Path("/Volumes/D50/TailnetCommander/work/beglin-qv13-20261023")
PYQ5=Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin")
SOURCE=R/"configs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt"
V2=R/"configs/research/BEGLIN_Q_V2_SELECTED_28_N8_ROLE_LAYER_POLICY_20261022.txt"
TOKEN=Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89/tokenizer.json")
ARROW={
 "arc128":Path("/Users/eoe/.cache/huggingface/datasets/allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-test.arrow"),
 "ob128":Path("/Users/eoe/.cache/huggingface/datasets/allenai___openbookqa/main/0.0.0/388097ea7776314e93a529163e0fea805b8a6454/openbookqa-test.arrow")}
LOCK_SHA="79826eb944ac2a4f87b637a430471f069a3789da37cdb12a9b7d1060d49db9b8"
PRE_SHA="5cb944375fa271c4528b4846f4b3c22cdf8179d4a38a2f5c17babbc03c4df92a"
PUB_SHA="9d045e1056e5ec8bc3a093a8a3d198277cbbc4083fb054a929b566cb2bb03c8c"
ARROW_SHA={"arc128":"e56914301ff03032cabc30b284e0e8d612cc56bac7b6a5ab4e7f35b31d6c3e32","ob128":"586293629974e2ed0fb06f33ef5336a737097cac11e063faf554799caacf791e"}
MASK_SHA={"source_qv8":"9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922","original_v2":"9324c5ec891d638f2713d3bbd4187399e1aae8a9a74df129dfabc4e8ea870212"}
BIN_SHA={"q4":"408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c","mixed":"f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221"}
ARMS=("q4","source_qv8","random73","random137","random211","random389","random577","original_v2")
RANDOMS=("random73","random137","random211","random389","random577")
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
DELTA={"q_proj":1179648,"k_proj":196608,"v_proj":196608,"o_proj":1179648,"gate_proj":6881280,"up_proj":6881280,"down_proj":6881280}
ROW=re.compile(rb"^\[qv5-choice\] window (\d+) NLL ([^ ]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
TOTAL=re.compile(rb"^qv5_choice NLL ([^ ]+) items (\d+) correct (\d+) mean ([^ ]+) ppl ([^ ]+)$",re.M)

def hash_bytes(b):return hashlib.sha256(b).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for c in iter(lambda:f.read(262144),b""):h.update(c)
 return h.hexdigest()
def check(ok,what):
 if not ok:raise RuntimeError("INDEPENDENT_AUDIT_FAIL:"+what)
def assert_file(p,h,name):check(p.is_file() and file_sha(p)==h,name)
def load_mask(path):
 rows=[]
 for line in path.read_text().splitlines():
  a,b=line.split();rows.append((a,int(b)))
 check(len(rows)==28 and len(set(rows))==28 and all(r in ROLES and 0<=n<28 for r,n in rows),"28_distinct_role_layers")
 return rows
def tokenized(dataset,r,tok):
 c=r["choices"]; labels=c["label"];texts=c["text"];stem=r["question"] if dataset=="arc128" else r["question_stem"]
 if len(labels)!=4 or len(texts)!=4 or len(set(labels))!=4 or r["answerKey"] not in labels or not stem.strip() or any(not x.strip() for x in texts):return None
 text="Choose the correct answer.\nQuestion: "+stem.strip()+"\n"+"\n".join(f"{k}. {v.strip()}" for k,v in zip("ABCD",texts))+"\nAnswer using one letter A, B, C, or D."
 prompt="<|im_start|>user\n"+text+"<|im_end|>\n<|im_start|>assistant\n"
 ids=tok.encode(prompt,add_special_tokens=False).ids
 if not 18<=len(ids)<=208:return None
 suffix={}
 for x in "ABCD":
  t=tok.encode(prompt+x,add_special_tokens=False).ids
  if len(t)!=len(ids)+1 or t[:-1]!=ids:return None
  suffix[x]=t[-1]
 if len(set(suffix.values()))!=4:return None
 correct="ABCD"[labels.index(r["answerKey"])]
 key=hash_bytes((dataset+"/"+r["id"]+"/"+hash_bytes(stem.encode())).encode())
 return {"key":key,"id":r["id"],"correct":correct,"ids":ids,"suffix":suffix,
         "qhash":hash_bytes(stem.encode()),"phash":hash_bytes(text.encode())}
def verify_source_data(lock):
 tok=Tokenizer.from_file(str(TOKEN))
 inputs={};question_hashes={}
 for ds,arrow in ARROW.items():
  assert_file(arrow,ARROW_SHA[ds],"original_arrow_"+ds)
  original=Dataset.from_file(str(arrow))
  candidates=[x for row in original if (x:=tokenized(ds,row,tok)) is not None]
  candidates.sort(key=lambda x:(x["key"],x["id"]))
  selected=[];seen=set()
  for x in candidates:
   if x["id"] in seen:continue
   seen.add(x["id"]);selected.append(x)
   if len(selected)==128:break
  private=L/"data"/(ds+".json")
  assert_file(private,lock["datasets"][ds]["sha256"],"frozen_original_Qwen_tokens_"+ds)
  frozen=json.loads(private.read_text())["items"]
  check(len(selected)==len(frozen)==128,"selected128_"+ds)
  for i,(raw,f) in enumerate(zip(selected,frozen)):
   check(raw["id"]==f["source_id"] and raw["correct"]==f["correct"],"source_id_gold_label_"+ds)
   check(raw["qhash"]==f["question_sha256"] and raw["phash"]==f["full_prompt_sha256"],"exact_prompt_"+ds)
   check(raw["ids"]==f["prefix_ids"] and raw["suffix"]==f["suffix_ids"],"token_boundary_"+ds)
  question_hashes[ds]={r["qhash"] for r in selected}
  inputs[ds]={"total_original":len(original),"eligible":len(candidates),"selected_exact_sha":128,"token_and_gold_label_match":128,"no_truncation":True}
 check(len(question_hashes["arc128"]&question_hashes["ob128"])==0,"cross_domain_question_collision")
 return inputs
def verify_masks(lock):
 assert_file(SOURCE,MASK_SHA["source_qv8"],"original_qv8_mask")
 assert_file(V2,MASK_SHA["original_v2"],"original_v2_mask")
 masks={}
 for arm in ARMS[1:]:
  p=L/"masks"/(arm+".txt")
  assert_file(p,lock["masks"][arm]["mask_sha256"],"sealed_mask_"+arm)
  masks[arm]=load_mask(p)
  delta=sum(DELTA[r] for r,n in masks[arm])
  check(delta==lock["masks"][arm]["incremental_packed_bytes"],"per_role_packed_cost_"+arm)
  check(736985088+delta==lock["masks"][arm]["projection_packed_bytes"],"total_packed_cost_"+arm)
 check(set(masks["source_qv8"])==set(load_mask(SOURCE)),"source_mask")
 check(set(masks["original_v2"])==set(load_mask(V2)),"original_v2")
 n8perrole=Counter(r for r,n in masks["source_qv8"])
 for seed in (73,137,211,389,577):
  rng=random.Random(seed)
  restored=[(r,i) for r in ROLES for i in sorted(rng.sample(range(28),n8perrole[r]))]
  check(restored==masks["random"+str(seed)],"random_seed_reproduction_"+str(seed))
 check(all(Counter(r for r,n in masks[x])==n8perrole and
  lock["masks"][x]["projection_packed_bytes"]==765493248 for x in RANDOMS),"same_budget_five_controls")
 return {"qv8_vs_random_same_role_and_packed_bytes":True,"matched_random_seeds_reproduced":5,
 "source_role_counts":dict(n8perrole),"source_bytes":765493248,
 "q4_bytes":736985088,"v2_bytes":776896512}
def native_scores(lock):
 data={}
 checked=0;means={};accs={}
 for arm in ARMS:
  data[arm]={}
  for ds in ("arc128","ob128"):
   base=L/"native_runs"/ds/arm
   receipt=base/"native_qv13_original_C_receipt.json"
   log=base/"native_qv13_original_C_run.log"
   a=json.loads(receipt.read_text())
   check(a["status"]=="ORIGINAL_NATIVE_28L_REAL_Q4_OR_28N8_SCORING_PASS","receipt_"+arm+ds)
   check(a["preexecution_lock_sha256"]==LOCK_SHA and
         a["prelock_sha256"]==PRE_SHA,"receipts_pinned_"+arm+ds)
   check(a["native_log_sha256"]==file_sha(log),"native_log_sha_"+arm+ds)
   kind="q4" if arm=="q4" else "mixed"
   check(a["native_original_binary_sha256"]==BIN_SHA[kind],"binary_"+arm+ds)
   check(a["input_manifest_sha256"]==lock["batches"][ds]["manifest_sha256"],"input_manifest_"+ds)
   check(a["n"]==128,"number_items_"+arm+ds)
   b=log.read_bytes()
   count=(b"projection_role_tensors: f32=0 n8=0 q4=196" if arm=="q4" else b"projection_role_tensors: f32=0 n8=28 q4=168")
   check(b.count(count)==1,"executed_native_quant_roles_"+arm+ds)
   logs=ROW.findall(b);summ=TOTAL.findall(b)
   check(len(logs)==128 and len(summ)==1,"per_question_raw_C_log_count_"+arm+ds)
   loss=np.zeros(128,dtype=float);right=np.zeros(128,dtype=int)
   for i,(window,nll,y,pred,n) in enumerate(logs):
    check(int(window)==i and 18<=int(n)<=208,"native_window_and_prefix_"+arm+ds)
    nll=float(nll);pred=int(pred);y=int(y)
    check(math.isfinite(nll) and nll>=0 and 0<=pred<4 and 0<=y<4,"native_gold_logit_"+arm+ds)
    loss[i]=nll;right[i]=int(y==pred)
    c=base/"eval/ppl_work"/("w"+str(i)+".choices")
    ids=base/"eval/ppl_work"/("w"+str(i)+".i32")
    meta=json.loads((L/"native_runs"/ds/"input_manifest.json").read_text())["original_items"][i]
    check(file_sha(ids)==meta["ids_sha256"] and file_sha(c)==meta["label_sha256"],"input_bytes_"+arm+ds)
    check(int(n)==meta["prefix_len"] and ids.stat().st_size==4*int(n),"i32_bytes_"+arm+ds)
    check(int(c.read_text().split()[-1])==y,"native_answer_idx_"+arm+ds)
   total=float(summ[0][0]);n=int(summ[0][1]);correct=int(summ[0][2]);mean=float(summ[0][3])
   check(n==128 and abs(total-np.sum(loss))<1e-7 and correct==int(right.sum()) and abs(mean-float(loss.mean()))<1e-8,"native_total_"+arm+ds)
   check(np.allclose(loss,a["article_fullvocab_gold_NLL"],rtol=0,atol=1e-12),"receipt_losses_"+arm+ds)
   check(np.array_equal(right,a["article_correct"]),"receipt_correct_"+arm+ds)
   checked+=128
   data[arm][ds]={"loss":loss,"correct":right}
  means[arm]=float(np.mean(np.concatenate([data[arm][ds]["loss"] for ds in ("arc128","ob128")])))
  accs[arm]=int(sum(np.sum(data[arm][ds]["correct"]) for ds in ("arc128","ob128")))
 return data,means,accs,checked
def bootstrap_gap(vals,source,off,seed):
 rng=np.random.default_rng(seed);bs=[];parts=[];B=40000
 for ds in ("arc128","ob128"):
  x=np.mean(np.stack([vals[r][ds]["loss"][off:] for r in RANDOMS]),axis=0)-source[ds]["loss"][off:]
  parts.append(x)
  bs.append(np.mean(x[rng.integers(0,len(x),size=(B,len(x)))],axis=1))
 combined=(bs[0]+bs[1])/2
 return {"n":len(parts[0])+len(parts[1]),"mean_random_minus_qv8_NLL":float(np.mean(np.concatenate(parts))),
 "CI_40k_stratified_question_bootstrap_95":np.quantile(combined,[.025,.975]).tolist(),
 "ARC_mean":float(parts[0].mean()),"OpenBookQA_mean":float(parts[1].mean())}
def check_stats(data,means,accs):
 result={}
 for label,offset in (("all256",0),("postpilot240",8)):
  summary=bootstrap_gap(data,data["source_qv8"],offset,20261023)
  diff=np.concatenate([data["random73"][d]["loss"][offset:]-data["source_qv8"][d]["loss"][offset:] for d in ("arc128","ob128")])
  summary["QV8_NLL_lower_than_random73_items"]=int((diff>0).sum())
  summary["random73_NLL_lower_items"]=int((diff<0).sum())
  summary["random73_vs_Qv8_exact_NLL_sign_test_two_sided_p"]=binomtest(int((diff>0).sum()),len(diff),.5).pvalue
  summary["source_vs_Q4_mean_NLL"]=float(np.mean(np.concatenate([data["q4"][d]["loss"][offset:]-data["source_qv8"][d]["loss"][offset:] for d in ("arc128","ob128")])))
  summary["source_correct"]=sum(int(np.sum(data["source_qv8"][d]["correct"][offset:])) for d in ("arc128","ob128"))
  summary["Q4_correct"]=sum(int(np.sum(data["q4"][d]["correct"][offset:])) for d in ("arc128","ob128"))
  summary["V2_correct"]=sum(int(np.sum(data["original_v2"][d]["correct"][offset:])) for d in ("arc128","ob128"))
  # Only 5 random control masks: rank of fixed source in sample of six policies
  summary["finite_mask_sample_source_rank"]=1+sum(1 for r in RANDOMS if
   sum(np.mean(data[r][d]["loss"][offset:]) for d in ("arc128","ob128")) <
   sum(np.mean(data["source_qv8"][d]["loss"][offset:]) for d in ("arc128","ob128")))
  summary["mask_level_rank_p_floor_if_label_exchangeable"]=1/6
  result[label]=summary
 return result

def main():
 assert_file(R/"configs/research/BEGLIN_QV13_ACTUAL_NATIVE_EVAL_PRELOCK_20261023.json",PRE_SHA,"original_prelock")
 assert_file(R/"docs/research/BEGLIN_QV13_ORIGINAL_NATIVE_256_PREEXEC_LOCK_PUBLIC_20261023.json",PUB_SHA,"public_freeze")
 assert_file(L/"qv13_lock_private.json",LOCK_SHA,"frozen_private_256")
 assert_file(PYQ5/"beglin_qv5_base",BIN_SHA["q4"],"actual_original_q4_C_executable")
 assert_file(PYQ5/"beglin_qv5_mixed",BIN_SHA["mixed"],"actual_original_n8_C_executable")
 assert_file(TOKEN,"a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf","original_tokenizer")
 lock=json.loads((L/"qv13_lock_private.json").read_text())
 for ds in ("arc128","ob128"):
  m=L/"native_runs"/ds/"input_manifest.json"
  assert_file(m,lock["batches"][ds]["manifest_sha256"],"frozen_input_"+ds)
 input_report=verify_source_data(lock)
 mask_report=verify_masks(lock)
 scores,means,acc,checked=native_scores(lock)
 summary=check_stats(scores,means,acc)
 diff=list(difflib.unified_diff((PYQ5/"qv5_base.c").read_text().splitlines(),(PYQ5/"qv5_mixed.c").read_text().splitlines()))
 hunk_count=sum(x.startswith("@@") for x in diff)
 check(hunk_count==5,"localized_quant_source_diffs")
 doc={
 "schema":"BEGLIN-Qv13-new-independent-full-native-Qwen28-C-recheck/1",
 "verdict":"ENGINE_EVIDENCE_PASS__AVERAGE_NLL_MATCHED_RANDOM_SAMPLE_ADVANTAGE_EXPLORATORY__GENERAL_OPTIMALITY_NOT_DEMONSTRATED__ACCURACY_GAIN_NOT_DEMONSTRATED__DYNAMIC_PRECISION_NOT_DEMONSTRATED",
 "actual_original_28layer_native_16_C_runs":16,
 "fullvocab_correct_single_token_NLL_rows_recomputed_from_original_C_log":checked,
 "source_dataset_and_tokenization_independently_reconstructed":input_report,
 "matched_role_layer_random_masks_independently_reconstructed":mask_report,
 "source_original_C_base_vs_mixed_diff_hunks":hunk_count,
 "independent_mean_fullvocab_correct_token_NLL_all256":means,
 "independent_4_choice_accuracy_correct_all256":acc,
 "stratified_bootstrap_and_finite_mask_rank_analysis":summary,
 "prelock_digest":PRE_SHA,"sealed_private_input_and_mask_digest":LOCK_SHA,"public_freeze_digest":PUB_SHA,
 "statistical_caveat":"Question-pair bootstrap estimates mean over prompts CONDITIONAL on these five fixed random masks, NOT an inference over all random masks. Finite mask-level rank among six comparisons cannot yield a one-sided p below 1/6.",
 "protocol_caveat":"Prereg SHA field for original Qv8 is incorrect and seed role order + sorted dataset hash text differ from executable freeze. Separate immutable provenance errata exists.",
 "other_caveats":["Static masks not per-request dynamic precision","Current accuracy not better than Q4 or original V2","Comparison to Q4 uses separate original Q4/mixed compiled C binaries, with localized five-hunk quantization-loader differences","NLL is full-vocabulary gold A/B/C/D token rather than conditional multiple choice NLL","Exact external BF16 weight-file source revision across ALL tensors not newly certified here","Public test may have appeared in model pretraining","No per-request switching cost, total system memory budget, high-confidence throughput or Metal GPU evaluation"],
 "no_raw_tokens_answers_or_per_case_model_scores_exported":True,
 "source_model_and_operator_runtime_not_modified":True,
 "research_only_no_main_merge":True}
 print(json.dumps(doc,sort_keys=True,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
