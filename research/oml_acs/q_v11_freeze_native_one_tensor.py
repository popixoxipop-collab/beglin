#!/usr/bin/env python3
"""Freeze original RACE middle TEST 16 distinct articles for native 1-N8 ablation.

No model forward/answer labels used for selection, no prompt truncation.
Source BF16 Fisher128 high/low tensor identities independently recomputed.
"""
from __future__ import annotations
import hashlib,json,os,pathlib
from datasets import load_dataset
from tokenizers import Tokenizer
ROOT=pathlib.Path(__file__).resolve().parents[2]
PRE=ROOT/"configs/research/BEGLIN_QV11_REAL_ONE_N8_INTERVENTION_PRELOCK_20261023.json"
TRAIN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv6-fisher-20261022/docs/research/BEGLIN_QV6_FISHER_SCALING_196_SOURCE_AGGREGATE_20261022.json")
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
LAB=ROOT/"lab_local/qv11_single_original_C_20261023"
OUT=LAB/"RACE_middle_test16_distinct_article_locked.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV11_RACE_TEST16_PREMODEL_INPUTS_20261023.json"
PRE_SHA="2cd480dc4c72b5383014a1325d350e51a0b47b1335ec0013a6d069773b546a39"
REV="2fec9fd81f1dc971569a9b729c43f2f0e6436637"
def sha(b):return hashlib.sha256(b).hexdigest()
def put(p,obj):
 raw=(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
 with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
 return sha(raw)
def encode(row,tok):
 if row["answer"] not in "ABCD" or len(row["options"])!=4 or not all(isinstance(x,str) and x.strip() for x in row["options"]):return None
 article=row["article"].strip();q=row["question"].strip()
 if not article or not q:return None
 s=("Choose the correct answer.\nPassage: "+article+"\nQuestion: "+q+"\n"+
   "\n".join(f"{a}. {v.strip()}" for a,v in zip("ABCD",row["options"]))+
   "\nAnswer using one letter A, B, C, or D.")
 prompt="<|im_start|>user\n"+s+"<|im_end|>\n<|im_start|>assistant\n"
 prefix=tok.encode(prompt,add_special_tokens=False).ids
 if not 18<=len(prefix)<=208:return None
 answer={}
 for k in "ABCD":
  total=tok.encode(prompt+k,add_special_tokens=False).ids
  if total[:len(prefix)]!=prefix or len(total)!=len(prefix)+1:return None
  answer[k]=total[-1]
 if len(set(answer.values()))!=4:return None
 return {"id":row["example_id"]+"/"+sha(q.encode())[:16],
    "article_original_id":row["example_id"],"id_sha256":sha(row["example_id"].encode()),
    "original_passage_sha256":sha(article.encode()),
    "question_sha256":sha(q.encode()),"full_prompt_sha256":sha(s.encode()),
    "full_untruncated_passage":True,"prompt_len":len(prefix),
    "prefix_ids":prefix,"suffix_ids":answer,"correct":row["answer"]}
def main():
 if OUT.exists() or PUBLIC.exists():raise FileExistsError("original 16 RACE test cohorts already fixed")
 if sha(PRE.read_bytes())!=PRE_SHA:raise RuntimeError("Qv11 preregistration changed after launch")
 cfg=json.loads(PRE.read_text())
 src=json.loads(TRAIN.read_text())["quality_upgrade_by_role_layer_per_size"]["128"]
 checked=[]
 for role,s in cfg["role_extremes"].items():
  vals=src[role];hi=max(range(28),key=lambda i:vals[i]);lo=min(range(28),key=lambda i:vals[i])
  if hi!=s["high_layer"] or lo!=s["low_layer"] or abs(vals[hi]-s["high_gain"])>1e-10 or abs(vals[lo]-s["low_gain"])>1e-10:
   raise RuntimeError("Qv11 top/bottom Fisher source tensor identities changed")
  checked+= [f"{role}:{hi}",f"{role}:{lo}"]
 if len(set(checked))!=6:raise RuntimeError("not six distinct role high/low scalar experiments")
 tok=Tokenizer.from_file(str(SNAP/"tokenizer.json"))
 ds=load_dataset("ehovy/race","middle",split="test",revision=REV)
 sorted_ds=sorted(ds,key=lambda d:(sha((REV+"/test/"+d["example_id"]+"/"+d["question"]).encode()),d["example_id"],d["question"]))
 selected=[];seen=set();eligible=0
 for row in sorted_ds:
  x=encode(row,tok)
  if x is None:continue
  eligible+=1
  if x["article_original_id"] in seen:continue
  seen.add(x["article_original_id"])
  selected.append(x)
  if len(selected)==16:break
 if len(selected)!=16 or len({x["article_original_id"] for x in selected})!=16:
  raise RuntimeError("not 16 distinct untruncated RACE test articles")
 os.umask(0o077);LAB.mkdir(parents=True,mode=0o700,exist_ok=True)
 original={"schema":"BEGLIN-Qv11-actual-1N8-vs-195Q4-original-RACEtest16/1",
  "status":"SEALED_RACE_TEST16_BEFORE_NATIVE_SINGLE_TENSOR_INTERVENTIONS",
  "prelock_sha256":PRE_SHA,"original_Qwen_HF_revision":cfg["model_HF_revision"],
  "RACE_middle_revision":REV,"original_data_split":"test",
  "one_original_question_per_article":True,"no_truncation":True,"sample_count":16,
  "real_question_source":selected,"model_logits_not_accessed":True}
 source_digest=put(OUT,original)
 summary={"schema":"BEGLIN-Qv11-original-16-new-RACE-articles-and-6-source-1N8-interventions/1",
  "status":"FROZEN_SOURCE_SIGNED_SCORE_6_SINGLETON_C_PROBES_AND_RACE_TEST16",
  "prelock_SHA256":PRE_SHA,"RACE_test16_private_SHA256":source_digest,
  "original_test_data_revision":REV,"RACE_middletest_question_count":16,"num_eligible_questions_before_unique_article":eligible,
  "RACE_article_ID_order_sha256":sha(json.dumps([x["article_original_id"] for x in selected]).encode()),
  "prompt_min_max_token_length":[min(x["prompt_len"] for x in selected),max(x["prompt_len"] for x in selected)],
  "original_Fisher128_source_evidence_public_SHA256":sha(TRAIN.read_bytes()),
  "selected_source_extremes":cfg["role_extremes"],"no_native_output_seen":True,
  "operator_main_model_unchanged":True}
 raw=(json.dumps(summary,sort_keys=True,indent=2)+"\n").encode()
 with PUBLIC.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 print("QV11_RACE_MIDDLE_TEST16_AND_SOURCE6_EXTREMES_FROZEN",json.dumps({
  "n":16,"source_extremes":checked,"token_range":summary["prompt_min_max_token_length"],
  "private_sha":source_digest,"public_sha":sha(raw)},sort_keys=True),flush=True)
if __name__=="__main__":main()
