#!/usr/bin/env python3
"""Lock 64 TinyLlama source ARC questions and unseen OpenBookQA TEST 48.

All task decisions derive from pinned original public dataset revisions, not
model logits or observed performance. Original Qv6 Qwen train first64 IDs
are reused semantically but separately tokenized for official Llama model.
"""
from __future__ import annotations
import hashlib,json,os,pathlib
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT=pathlib.Path(__file__).resolve().parents[2]
PRE=ROOT/"configs/research/BEGLIN_QV7_CROSSFAMILY_CALIBRATION_PRELOCK_20261022.json"
PRE_SHA="1b5fa904cf11f25292720efc75f658fc10cedc8923236b4d5b0ae7e70d3e88c6"
QV6=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv6-fisher-20261022")
QV6SRC=QV6/"lab_local/qv6_fisher_scaling_20261022/source_arc_128_frozen.json"
QV6HELD=QV6/"lab_local/qv6_fisher_scaling_20261022/openbookqa_val48_frozen.json"
SRC_SHA="78d267bef54240cfb32b292c5d1885b6716fadd9bbd78ec7b3b1bf97822a1ab2"
EVAL_SHA="c09831bf86c6eedebed433dc2a0398a0992afb19fddc9f86750c2d137d71d1b0"
LAB=ROOT/"lab_local/qv7_tinyllama_20261022"
MODEL=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv7-tinyllama-model-20261022")
TRAIN=LAB/"source_arc64_tinyllama_frozen.json"
TEST=LAB/"eval_openbookqa_test48_tinyllama_frozen.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV7_CROSSFAMILY_TINYLLAMA_INPUT_FREEZE_20261022.json"

def sha(b:bytes):return hashlib.sha256(b).hexdigest()
def secure(p:pathlib.Path):
    if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:
        raise RuntimeError("old Qv6 sealed source not permitted "+str(p))
    return p.read_bytes()
def persist(p,obj):
    b=(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
    return sha(b)

def parse(row,tok):
    a=row["choices"]
    options=a["text"];labels=a["label"];correct=row["answerKey"]
    text=row["question"] if "question" in row else row["question_stem"]
    if labels!=["A","B","C","D"] or correct not in labels or not text.strip() or len(options)!=4:
        return None
    content="Choose the correct answer from the following four options.\nQuestion: "+text.strip()+"\n"
    content+="\n".join(f"{k}. {v.strip()}" for k,v in zip(labels,options))
    content+="\nAnswer using exactly one letter: A, B, C, or D."
    prefix=tok.apply_chat_template([{"role":"user","content":content}],tokenize=False,add_generation_prompt=True)
    ids=tok.encode(prefix,add_special_tokens=False)
    if not 18<=len(ids)<=208:return None
    answers={}
    for letter in labels:
        candidate=tok.encode(prefix+letter,add_special_tokens=False)
        if candidate[:len(ids)]!=ids or len(candidate)!=len(ids)+1:return None
        answers[letter]=candidate[-1]
    if len(set(answers.values()))!=4:return None
    return {"id":row["id"],"correct":correct,
       "prefix_ids":ids,"suffix_ids":answers,
       "prompt_token_count":len(ids),
       "input_sha256":sha(json.dumps(ids,separators=(',',':')).encode()),
       "question_sha256":sha(text.strip().encode())}

def run():
    if TRAIN.exists() or TEST.exists() or PUBLIC.exists():
        raise FileExistsError("cannot refreeze before/after Llama source calibration")
    if sha(PRE.read_bytes())!=PRE_SHA or sha(secure(QV6SRC))!=SRC_SHA or sha(secure(QV6HELD))!=EVAL_SHA:
        raise RuntimeError("original prereg, Qwen ARC source, prior OpenBookQA val provenance drift")
    cfg=json.loads(PRE.read_text())
    if cfg["replication_revision"]!="fe8a4ea1ffedaf415f4da2f062534de366a451e6" or \
       cfg["source_count"]!=64 or cfg["eval_count"]!=48 or cfg["no_main_changes"] is not True:
        raise RuntimeError("wrong model family/protocol")
    if not (MODEL/"config.json").is_file() or not (MODEL/"tokenizer.json").is_file():
        raise RuntimeError("pinned TinyLlama source tokenizer unavailable")
    model_cfg=json.loads((MODEL/"config.json").read_text())
    if model_cfg["model_type"]!="llama" or model_cfg["num_hidden_layers"]!=22 or \
       model_cfg["hidden_size"]!=2048 or model_cfg["vocab_size"]!=32000:
        raise RuntimeError("Llama model architecture unexpected")
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
    old=json.loads(secure(QV6SRC))["all_source_questions"][:64]
    train_rows=load_dataset("allenai/ai2_arc","ARC-Challenge",split="train",revision=cfg["replication_source_data"].split("@")[1])
    lookup={x["id"]:x for x in train_rows}
    source=[]
    for x in old:
        if x["id"] not in lookup:raise RuntimeError("original frozen ARC ID missing")
        val=parse(lookup[x["id"]],tok)
        if val is None or val["question_sha256"]!=x["question_sha256"]:
            raise RuntimeError("cross-family original source question semantics drift")
        source.append(val)
    original48=json.loads(secure(QV6HELD))["unseen_questions"]
    used={x["question_sha256"] for x in original48}|{x["question_sha256"] for x in source}
    dataset=load_dataset("allenai/openbookqa","main",split="test",revision=cfg["new_cross_corpus"].split("@")[1])
    chosen=[];elig=0
    ordered=sorted(dataset,key=lambda x:(sha((cfg["new_cross_corpus"].split("@")[1]+"/test/"+x["id"]).encode()),x["id"]))
    for row in ordered:
        x=parse(row,tok)
        if x is None:continue
        elig+=1
        if x["question_sha256"] in used:continue
        used.add(x["question_sha256"])
        chosen.append(x)
        if len(chosen)==48:break
    if len(source)!=64 or len(chosen)!=48 or {x["id"] for x in source}&{x["id"] for x in chosen}:
        raise RuntimeError("source/test 64+48 disjoint frozen cohort missing")
    os.umask(0o077)
    LAB.mkdir(parents=True,exist_ok=True,mode=0o700)
    source_receipt={"schema":"BEGLIN-Qv7-TinyLlama-independent-ARC-train64/1",
      "prelock_sha256":PRE_SHA,"source_dataset_sha":cfg["replication_source_data"].split("@")[1],
      "model_revision":cfg["replication_revision"],"source64":source,
      "source_Qv6_64_ids_exact":True,"raw_text_not_published":True}
    test_receipt={"schema":"BEGLIN-Qv7-TinyLlama-new-OpenBookQA-test48/1",
      "prelock_sha256":PRE_SHA,"test_dataset_sha":cfg["new_cross_corpus"].split("@")[1],
      "model_revision":cfg["replication_revision"],"test48":chosen,
      "not_previous_OpenBookQA_val48":True,"held_test_scores_unopened":True}
    s= persist(TRAIN,source_receipt)
    t= persist(TEST,test_receipt)
    agg={"schema":"BEGLIN-Qv7-cross-family-ARC64-test48-source-freeze/1",
      "status":"FROZEN_LLAMA_MODEL_SOURCE64_AND_UNSEEN_TEST48_BEFORE_CALIBRATION",
      "prelock_sha256":PRE_SHA,"model_revision":cfg["replication_revision"],
      "model_type":"llama","decoder_layers":22,"original_model_metadata_config_sha256":sha((MODEL/"config.json").read_bytes()),
      "source_count":64,"held_count":48,
      "source_private_sha256":s,"held_private_sha256":t,
      "source_ids_sha256":sha(json.dumps([x["id"] for x in source]).encode()),
      "held_ids_sha256":sha(json.dumps([x["id"] for x in chosen]).encode()),
      "previous_OpenBookQA_val_ids_not_reused":True,
      "held_eligible_four_option_candidates":elig,
      "held_prompt_token_range":[min(x["prompt_token_count"] for x in chosen),max(x["prompt_token_count"] for x in chosen)],
      "no_source_or_model_logits_accessed":True,
      "live_release_unchanged":True,"automatic_policy_promotion_off":True}
    b=(json.dumps(agg,sort_keys=True,indent=2)+"\n").encode()
    with PUBLIC.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    print("QV7_PINNED_LLAMA_FAMILY_64_48_INPUT_FREEZE_PASS",
         json.dumps({"src_sha":s,"held_sha":t,"public_sha":sha(b),
            "held_range":agg["held_prompt_token_range"]},sort_keys=True),flush=True)
if __name__=="__main__":run()
