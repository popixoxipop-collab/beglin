#!/usr/bin/env python3
"""Freeze 48 unseen RACE-middle articles / answers before Qv10 source upgrade.

Complete article, question and all four original options are preserved, no
truncation. One question per article avoids article-level pseudoreplication.
Only SHA256(dataset commit + article ID + question) determines selection.
Never reads model logits, native outcomes or Qv9 previous evaluation labels.
"""
from __future__ import annotations
import hashlib,json,os,pathlib
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT=pathlib.Path(__file__).resolve().parents[2]
OLD=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv9-ggn-20261023")
QTRAIN=OLD/"lab_local/qv9_ggn_20261023/ARC_train24_GGN_source_frozen.json"
QDIAG=OLD/"lab_local/qv9_ggn_20261023/ARC_diag24_independent_frozen.json"
QJAC_TRAIN=OLD/"lab_local/qv9_ggn_20261023/original_BF16_choice_GGN_train24_private.json"
QJAC_DIAG=OLD/"lab_local/qv9_ggn_20261023/original_BF16_choice_GGN_diagnostic24_private.json"
PRE=ROOT/"configs/research/BEGLIN_QV10_TOPK_OTHER_TRUST_PRELOCK_20261023.json"
LAB=ROOT/"lab_local/qv10_topk_trust_20261023"
LOCK=LAB/"RACE_middle_validation_48_locked.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV10_RACE_INPUT_PREMODEL_FREEZE_20261023.json"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
PRE_SHA="4ef4d31a4b9d0be2e015ffaac84a8ee86feb86047a8a8aae4a50318144d5d207"
REV="2fec9fd81f1dc971569a9b729c43f2f0e6436637"
def sha(b):return hashlib.sha256(b).hexdigest()
def secure(p):
 if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
  raise RuntimeError("prior original BF16 source scalar/inputs were not 0600")
 return p.read_bytes()
def save(p,obj):
 data=(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
 with p.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
 return sha(data)
def encode(row,tok):
 article=row["article"].strip();question=row["question"].strip()
 opts=row["options"];key=row["answer"]
 if not article or not question or len(opts)!=4 or key not in "ABCD" or any(not isinstance(t,str) or not t.strip() for t in opts):
  return None
 content=("Choose the correct answer.\nPassage: "+article+
   "\nQuestion: "+question+"\n"+
   "\n".join(f"{l}. {text.strip()}" for l,text in zip("ABCD",opts))+
   "\nAnswer using one letter A, B, C, or D.")
 user_chat="<|im_start|>user\n"+content+"<|im_end|>\n<|im_start|>assistant\n"
 ids=tok.encode(user_chat,add_special_tokens=False).ids
 if not 18<=len(ids)<=208:return None
 cands={}
 for letter in "ABCD":
  new=tok.encode(user_chat+letter,add_special_tokens=False).ids
  if new[:len(ids)]!=ids or len(new)!=len(ids)+1:return None
  cands[letter]=new[-1]
 if len(set(cands.values()))!=4:return None
 return {"id":row["example_id"]+"/"+sha(question.encode())[:16],
   "passage_original_id":row["example_id"],"answer":key,
   "prompt_token_ids":ids,"correct":key,"option_ids":cands,
   "prompt_token_count":len(ids),"article_sha":sha(article.encode()),
   "question_sha":sha(question.encode()),"content_sha":sha(content.encode())}
def main():
 if LOCK.exists() or PUBLIC.exists():raise FileExistsError("RACE study already frozen")
 cfg=json.loads(PRE.read_text())
 if sha(PRE.read_bytes())!=PRE_SHA:raise RuntimeError("Qv10 prereg altered")
 required={
  QTRAIN:cfg["source_train24_original_Qv9_SHA256"],
  QDIAG:cfg["source_diagnostic24_original_Qv9_SHA256"],
  QJAC_TRAIN:cfg["train_original_4choice_relative_logits_Jacobians_SHA256"],
  QJAC_DIAG:cfg["diagnostic_original_4choice_relative_logits_Jacobians_SHA256"]}
 for p,dig in required.items():
  if sha(secure(p))!=dig:raise RuntimeError("source BF16 signed original Qv9 data changed "+str(p))
 q_train=json.loads(secure(QTRAIN))["questions"];q_diag=json.loads(secure(QDIAG))["questions"]
 if len(q_train)!=24 or len(q_diag)!=24 or {v["id"] for v in q_train}&{v["id"] for v in q_diag}:
  raise RuntimeError("source and diagnostic original ARC labels not disjoint")
 tok=Tokenizer.from_file(str(SNAP/"tokenizer.json"))
 raw=load_dataset("ehovy/race","middle",split="validation",revision=REV)
 ordered=sorted(raw,key=lambda row:(sha((REV+"/"+row["example_id"]+"/"+row["question"]).encode()),
     row["example_id"],row["question"]))
 selected=[];seen_articles=set();candidate=0
 for r in ordered:
  v=encode(r,tok)
  if v is None:continue
  candidate+=1
  if v["passage_original_id"] in seen_articles:continue
  seen_articles.add(v["passage_original_id"])
  selected.append(v)
  if len(selected)>=48:break
 if len(selected)!=48 or len(seen_articles)!=48:
  raise RuntimeError("not 48 distinct original untruncated RACE articles with one-token answers")
 os.umask(0o077);LAB.mkdir(parents=True,exist_ok=True,mode=0o700)
 payload={"schema":"BEGLIN-Qv10-actual-RACE-middle-validation-48-lock/1",
   "status":"48_DISTINCT_FULL_PASSAGES_SHA_FROZEN_BEFORE_TOPK_BF16_JACOBIAN",
   "preregistration_SHA256":PRE_SHA,"original_RACE_commit":REV,
   "Qv9_old24train_SHA256":cfg["source_train24_original_Qv9_SHA256"],
   "Qv9_old24diag_SHA256":cfg["source_diagnostic24_original_Qv9_SHA256"],
   "number_questions":48,"questions":selected,
   "one_per_article":True,"no_truncation":True,
   "no_model_scores_loaded":True}
 lockedsha=save(LOCK,payload)
 pub={"schema":"BEGLIN-Qv10-original-source-and-RACE48-input-freeze/1",
   "status":"ALL_3_COHORTS_SHA_LOCKED_BEFORE_TOPK_PLUS_OTHER_MODEL_EXECUTION",
   "prereg_sha":PRE_SHA,"original_48_RACE_revision":REV,
   "RACE48_private_SHA256":lockedsha,
   "train24_private_SHA256":cfg["source_train24_original_Qv9_SHA256"],
   "source_diag24_private_SHA256":cfg["source_diagnostic24_original_Qv9_SHA256"],
   "source_train_24_and_diag_24_have_prior_Qv9_analyses":True,
   "eval_48_distinct_article_count":48,"eligible_prompt_candidates":candidate,
   "prompt_min_max":[min(v["prompt_token_count"] for v in selected),
                     max(v["prompt_token_count"] for v in selected)],
   "source_ids_SHA256":sha(json.dumps([x["id"] for x in q_train+q_diag]).encode()),
   "held_48_articleIDs_SHA256":sha(json.dumps([x["passage_original_id"] for x in selected]).encode()),
   "test_model_performance_unseen":True,"main_and_live_unchanged":True}
 data=(json.dumps(pub,sort_keys=True,indent=2)+"\n").encode()
 with PUBLIC.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 print("QV10_RACE48_DISTINCT_ORIGINAL_ARTICLE_INPUTS_FROZEN",json.dumps({
   "source_train":24,"diag":24,"RACE":48,"token_lengths":pub["prompt_min_max"],
   "RACE_private_sha":lockedsha,"public_sha":sha(data)},sort_keys=True),flush=True)
if __name__=="__main__":main()
