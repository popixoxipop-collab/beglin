#!/usr/bin/env python3
"""Pre-freeze nested ARC train (24,64,128) and 48 unseen OpenBookQA questions.

Never accesses original BF16 weights/quantized logits, the Qv5 development
labels as optimizer data, or the legacy V6 holdout. All data fixed BEFORE Qv6
Fisher gradients. Original Qv5 24 source IDs and token arrays must match.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,sys
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT=pathlib.Path(__file__).resolve().parents[2]
BASE5=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
sys.path.insert(0,str(ROOT/"research/oml_acs"))
from q_v5_freeze_arc_inputs import choose,parse,sha

PRE=ROOT/"configs/research/BEGLIN_QV6_SAMPLE_SCALING_PRELOCK_20261022.json"
PRE_SHA="2e35764878191717ef7b84ed546e72e2a1c513523ecb17d24b54489bd29f319b"
LAB=ROOT/"lab_local/qv6_fisher_scaling_20261022"
TRAIN=LAB/"source_arc_128_frozen.json"
CROSS=LAB/"openbookqa_val48_frozen.json"
PUB=ROOT/"docs/research/BEGLIN_QV6_SPLITS_PREMODEL_FREEZE_20261022.json"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")

def private(p,obj):
    b=(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
    return sha(b)

def run():
    if TRAIN.exists() or CROSS.exists() or PUB.exists():
        raise FileExistsError("Qv6 dataset already frozen; no post-observation data reselection")
    if sha(PRE.read_bytes())!=PRE_SHA:raise RuntimeError("Qv6 prelock changed")
    c=json.loads(PRE.read_text())
    oldsrc=BASE5/"lab_local/qv5_fisher_20261022/source_train_frozen.json"
    olddev=BASE5/"lab_local/qv5_fisher_20261022/development_val_frozen.json"
    if (sha(oldsrc.read_bytes())!=c["source_24_private_receipt_sha256"] or
        sha(olddev.read_bytes())!=c["prior_arc_development_split_SHA256"]):
        raise RuntimeError("Qv5 old train/dev source identity unknown")
    orig=json.loads(oldsrc.read_text())["questions"]
    dev=json.loads(olddev.read_text())["questions"]
    tok=Tokenizer.from_file(str(SNAP/"tokenizer.json"))
    train,eligible=choose("train",128,tok,c["source_dataset_revision"])
    if len(train)!=128 or len(orig)!=24:
        raise RuntimeError("source nested train not complete")
    for i in range(24):
        if orig[i]!=train[i]:
            raise RuntimeError("first 24 were NOT original Fisher source tokens; refuse gradient reuse at "+str(i))
    raw=load_dataset(c["cross_dataset"],c["cross_dataset_subset"],
          split=c["cross_dataset_split"],revision=c["cross_dataset_revision"])
    seen={x["question_sha256"] for x in train}|{x["question_sha256"] for x in dev}
    ood=[]
    eligiblecross=0
    ordered=sorted(raw,key=lambda row:(sha((c["cross_dataset_revision"]+"/validation/"+row["id"]).encode()),row["id"]))
    for d in ordered:
        item={"id":d["id"],"question":d["question_stem"],"choices":d["choices"],"answerKey":d["answerKey"]}
        row=parse(item,tok)
        if not row:continue
        eligiblecross+=1
        if row["question_sha256"] in seen:continue
        seen.add(row["question_sha256"])
        if len(ood)<48:ood.append(row)
    if len(ood)!=48:raise RuntimeError("OpenBookQA has insufficient frozen 4-choice validations")
    if set(x["question_sha256"] for x in train)&set(x["question_sha256"] for x in ood):
        raise RuntimeError("ARC training and OpenBookQA validation texts collide")
    os.umask(0o077);LAB.mkdir(parents=True,mode=0o700)
    trainrecord={"schema":"BEGLIN-Qv6-pinned-ARC-Challenge-train-128-nested/1",
      "prelock_sha256":PRE_SHA,"dataset_sha256":c["source_dataset_revision"],
      "source_case_count":len(train),"original_Qv5_first_24_exactly_reproduced":True,
      "source_original_Qv5_24_SHA256":c["source_24_private_receipt_sha256"],
      "index_24_64_128":[24,64,128],"all_source_questions":train,
      "not_from_val_test_or_Qv5_dev":True}
    heldrecord={"schema":"BEGLIN-Qv6-pinned-OpenBookQA-validation-48/1",
      "prelock_sha256":PRE_SHA,"dataset_sha256":c["cross_dataset_revision"],
      "cross_corpus_data_locks_before_gradients":True,
      "sample_count":len(ood),"unseen_questions":ood,
      "no_ARC_source_or_prior_dev_exact_question_overlap":True,
      "not_an_official_OOD_generalization_estimate":True}
    a=private(TRAIN,trainrecord);b=private(CROSS,heldrecord)
    report={"schema":"BEGLIN-Qv6-original-nested-128-and-cross-corpus-48-pre-model-freeze/1",
      "status":"FROZEN_SOURCE_NESTED_AND_CROSSCORPUS_BEFORE_ANY_NEW_FISHER_OR_EVAL",
      "protocol_SHA256":PRE_SHA,"ARC_original_rev":c["source_dataset_revision"],
      "OpenBookQA_rev":c["cross_dataset_revision"],"original_model_rev":c["official_BF16_revision"],
      "source128_hash":a,"cross48_hash":b,"ARC_train_eligible":eligible,
      "OpenBookQA_eligible":eligiblecross,
      "source24_prefix_exact":True,
      "source64_prefix_sha256":sha(json.dumps([x["id"] for x in train[:64]]).encode()),
      "source128_idsha":sha(json.dumps([x["id"] for x in train]).encode()),
      "cross48_idsha":sha(json.dumps([x["id"] for x in ood]).encode()),
      "cross_prompt_length_range":[min(x["prompt_token_count"] for x in ood),max(x["prompt_token_count"] for x in ood)],
      "development_labels_not_used_to_select_models":True,
      "research_only":True,"production_changed":False}
    p=(json.dumps(report,sort_keys=True,indent=2)+"\n").encode()
    with PUB.open("xb") as f:f.write(p);f.flush();os.fsync(f.fileno())
    print("QV6_128_SOURCE_AND_48_CROSSCORPUS_FROZEN",
        json.dumps({"source24_prefix_match":True,"src_sha":a,"held_sha":b,
         "public_sha":sha(p),"held_prompt_range":report["cross_prompt_length_range"]},sort_keys=True),flush=True)
if __name__=="__main__":run()
