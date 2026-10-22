#!/usr/bin/env python3
"""One-time source-train / development-val ARC-Challenge freeze, before Fisher fitting.

Selections depend only on a pinned public dataset revision, exact Qwen token
identity and SHA-sorted IDs, never logits, NLL or policy outcomes.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,sys
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT=pathlib.Path(__file__).resolve().parents[2]
PROTO=ROOT/"configs/research/BEGLIN_QV5_FISHER_GRAD_PRELOCK_20261022.json"
EXPECTED_PROTO="a886418affc9f665d64ce4ca9eb0b5ea2aef4ae97bc59bf58ea1affb53bf19ef"
PRIVATE=ROOT/"lab_local/qv5_fisher_20261022"
PUB=ROOT/"docs/research/BEGLIN_QV5_ARC_FREEZE_PUBLIC_AGGREGATE_20261022.json"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
MAX_PROMPT_TOKENS=208

def sha(b:bytes):return hashlib.sha256(b).hexdigest()
def put(path:pathlib.Path,obj:dict):
    b=(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
    with path.open("xb") as f:
        f.write(b);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
    return sha(b)

def parse(item,tokenizer):
    labels=item["choices"]["label"]
    options=item["choices"]["text"]
    if labels!=["A","B","C","D"] or item["answerKey"] not in labels or len(options)!=4:
        return None
    if not item["id"] or not item["question"] or any(not str(t).strip() for t in options):
        return None
    question="Choose the correct answer from these four options.\nQuestion: "+item["question"].strip()
    question+="\n"+"\n".join(f"{letter}. {text.strip()}" for letter,text in zip(labels,options))
    question+="\nAnswer using exactly one letter: A, B, C, or D."
    chat="<|im_start|>user\n"+question+"<|im_end|>\n<|im_start|>assistant\n"
    prefix=tokenizer.encode(chat,add_special_tokens=False).ids
    if len(prefix)<18 or len(prefix)>MAX_PROMPT_TOKENS: return None
    suffix={}
    for letter in labels:
        arr=tokenizer.encode(chat+letter,add_special_tokens=False).ids
        if arr[:len(prefix)]!=prefix or len(arr)!=len(prefix)+1:
            return None
        suffix[letter]=arr[-1]
    if len(set(suffix.values()))!=4:return None
    return {"id":item["id"],"prefix_ids":prefix,"correct":item["answerKey"],
       "suffix_ids":suffix,"prompt_token_count":len(prefix),
       "input_sha256":sha(bytes(str(prefix),"utf-8")),
       "question_sha256":sha(item["question"].strip().encode())}

def choose(split,count,tokenizer,model_hash):
    ds=load_dataset("allenai/ai2_arc","ARC-Challenge",split=split,revision=model_hash)
    candidates=[]
    for x in ds:
        row=parse(x,tokenizer)
        if row is not None:
            row["sortkey"]=sha((model_hash+"/"+split+"/"+row["id"]).encode())
            candidates.append(row)
    candidates.sort(key=lambda x:(x["sortkey"],x["id"]))
    seen=set();rows=[]
    for row in candidates:
        if row["question_sha256"] in seen:continue
        seen.add(row["question_sha256"])
        if len(rows)==count: break
        rows.append({k:v for k,v in row.items() if k!="sortkey"})
    if len(rows)!=count:
        raise RuntimeError(f"Insufficient pinned dataset single-letter cases: {split} {len(rows)}/{count} after {len(candidates)} candidates")
    return rows,len(candidates)

def main():
    if PUB.exists() or (PRIVATE/"source_train_frozen.json").exists() or (PRIVATE/"development_val_frozen.json").exists():
        raise FileExistsError("never refreeze selected ARC questions after native outputs")
    if sha(PROTO.read_bytes())!=EXPECTED_PROTO:raise RuntimeError("preregistration changed")
    c=json.loads(PROTO.read_text())
    tokfile=SNAP/"tokenizer.json"
    if not tokfile.is_file() or c["dataset_sha"]!="210d026faf9955653af8916fad021475a3f00453":
        raise RuntimeError("pinned original tokenizer/dataset absent")
    tokenizer=Tokenizer.from_file(str(tokfile))
    train,availtrain=choose("train",c["source_questions"],tokenizer,c["dataset_sha"])
    val,availval=choose("validation",c["development_questions"],tokenizer,c["dataset_sha"])
    if {x["question_sha256"] for x in train}&{x["question_sha256"] for x in val}:
        raise RuntimeError("source and development ARC texts overlap")
    os.umask(0o077)
    PRIVATE.mkdir(parents=True,exist_ok=True,mode=0o700)
    src={"schema":"BEGLIN-Qv5-ARC-source-train-24/1",
         "prelock_sha256":EXPECTED_PROTO,"dataset_sha":c["dataset_sha"],
         "question_count":len(train),"questions":train,
         "only_source_train_labels":True,"dev_logits_unopened":True}
    held={"schema":"BEGLIN-Qv5-ARC-development-validation-40/1",
         "prelock_sha256":EXPECTED_PROTO,"dataset_sha":c["dataset_sha"],
         "question_count":len(val),"questions":val,
         "locked_before_source_Fisher_policy":True,"never_use_for_policy_selection":True}
    srch=put(PRIVATE/"source_train_frozen.json",src)
    valh=put(PRIVATE/"development_val_frozen.json",held)
    pub={"schema":"BEGLIN-Qv5-public-input-freeze/1",
      "status":"BOTH_SPLITS_FROZEN_BEFORE_ORIGINAL_BF16_FISHER",
      "preregistration_sha256":EXPECTED_PROTO,"dataset_repo":c["dataset_repo"],
      "dataset_commit_sha256":c["dataset_sha"],"BF16_original_HF_revision":c["bf16_official_rev"],
      "tokenizer_sha256":sha(tokfile.read_bytes()),
      "source_items":len(train),"source_candidate_eligible":availtrain,
      "development_items":len(val),"development_candidate_eligible":availval,
      "source_receipt_sha256":srch,"development_receipt_sha256":valh,
      "selected_source_ids_sha256":sha(json.dumps([x["id"] for x in train]).encode()),
      "selected_dev_ids_sha256":sha(json.dumps([x["id"] for x in val]).encode()),
      "selected_validation_answers_not_exposed_in_public_source_policy":True,
      "no_V6_existing_unknown_24_explicitly_opened":True,
      "source_and_development_distinct":True,"production_touched":False}
    pub_bytes=(json.dumps(pub,sort_keys=True,indent=2)+"\n").encode()
    with PUB.open("xb") as f:
        f.write(pub_bytes);f.flush();os.fsync(f.fileno())
    print("QV5_ARC_INPUTS_FROZEN",json.dumps({
       "source":len(train),"development":len(val),
       "source_sha256":srch,"dev_sha256":valh,"pub_sha256":sha(pub_bytes),
       "input_range":[min(q["prompt_token_count"] for q in val),max(q["prompt_token_count"] for q in val)]
    },sort_keys=True),flush=True)
if __name__=="__main__": main()
