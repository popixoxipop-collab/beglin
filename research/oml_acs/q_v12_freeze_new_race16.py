#!/usr/bin/env python3
"""Freeze new RACE-middle test16, article-disjoint from Qv11's prior 16.

Three singleton masks and all three pairs are preregistered from previous
Qv11 development native outcomes; no new RACE output or held label is used
to select bit positions. Full original article, question, options preserved.
"""
from __future__ import annotations
import hashlib,json,os,pathlib
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT=pathlib.Path(__file__).resolve().parents[2]
PRE=ROOT/"configs/research/BEGLIN_QV12_TWO_N8_INTERACTION_PRELOCK_20261023.json"
OLD=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv11-trueintervention-20261023")
OLD_DATA=OLD/"lab_local/qv11_single_original_C_20261023/RACE_middle_test16_distinct_article_locked.json"
OLD_AUDIT=OLD/"docs/research/BEGLIN_QV11_REAL_ONE_TENSOR_NATIVE_CAUSAL_AUDIT_PUBLIC_20261023.json"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
LAB=ROOT/"lab_local/qv12_twotensor_20261023"
OUT=LAB/"new_RACE_middle_test16_disjoint_Qv11_frozen.json"
PUB=ROOT/"docs/research/BEGLIN_QV12_RACE16_NEW_INPUT_FREEZE_PUBLIC_20261023.json"
PRE_SHA="a565bdec341d675ce52a37132b06355760c2fef96b8e4329c17508a237e4374f"
REV="2fec9fd81f1dc971569a9b729c43f2f0e6436637"
def sha(b):return hashlib.sha256(b).hexdigest()
def sealed(p):
    if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:
        raise RuntimeError("original Qv11 private inputs not sealed")
    return p.read_bytes()
def write_private(p,obj):
    b=(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
    return sha(b)
def tokenize(r,tok):
    if not isinstance(r["article"],str) or not r["article"].strip() or \
       not isinstance(r["question"],str) or not r["question"].strip() or \
       r["answer"] not in "ABCD" or len(r["options"])!=4 or \
       any(not isinstance(x,str) or not x.strip() for x in r["options"]):
        return None
    article=r["article"].strip();q=r["question"].strip()
    text=("Choose the correct answer.\nPassage: "+article+
          "\nQuestion: "+q+"\n"+
          "\n".join(f"{k}. {v.strip()}" for k,v in zip("ABCD",r["options"]))+
          "\nAnswer using one letter A, B, C, or D.")
    prefix="<|im_start|>user\n"+text+"<|im_end|>\n<|im_start|>assistant\n"
    ids=tok.encode(prefix,add_special_tokens=False).ids
    if not 18<=len(ids)<=208:return None
    choices={}
    for a in "ABCD":
        new=tok.encode(prefix+a,add_special_tokens=False).ids
        if new[:len(ids)]!=ids or len(new)!=len(ids)+1:return None
        choices[a]=new[-1]
    if len(set(choices.values()))!=4:return None
    articleid=r["example_id"]
    return {"id":articleid+"/"+sha(q.encode())[:16],
      "article_original_id":articleid,
      "article_sha256":sha(article.encode()),"question_sha256":sha(q.encode()),
      "original_full_prompt_sha256":sha(text.encode()),
      "prefix_ids":ids,"suffix_ids":choices,
      "correct":r["answer"],"prompt_len":len(ids),
      "source_passage_question_and_options_untruncated":True}
def main():
    if OUT.exists() or PUB.exists():raise FileExistsError("Qv12 held input already SHA locked")
    if sha(PRE.read_bytes())!=PRE_SHA:raise RuntimeError("preregistered pairs have drifted")
    cfg=json.loads(PRE.read_text())
    oldraw=sealed(OLD_DATA)
    if sha(oldraw)!=cfg["prior_Qv11_original_test16_private_SHA256"]:
        raise RuntimeError("original prior 16 RACE articles changed")
    prev=json.loads(oldraw)["real_question_source"]
    priorids={r["article_original_id"] for r in prev}
    if len(priorids)!=16:raise RuntimeError("Qv11 test16 previous unique article data missing")
    parent=OLD_AUDIT.read_bytes()
    if sha(parent)!=cfg["prior_Qv11_7arm_true_N8_public_audit_SHA256"]:
        raise RuntimeError("previous observed singleton gains authority drift")
    rank=json.loads(parent)["measured_per_tensor_fullmodel_actual_NLL_effect"]
    selected=cfg["three_preselected_intervention_tensors"]
    expected=set()
    for row in selected:
        key=f"{row['role']}_{row['layer']}"
        expected.add(key)
        if abs(rank[key]["actual_N8_improvement_fullQ4_minus_N8"]-row["observed_Qv11_prior_source16_improvement_NLL"])>1e-12:
            raise RuntimeError("Qv12 prelocked singleton source previous outcome changed")
    if expected!={"down_proj_10","v_proj_20","k_proj_0"}:
        raise RuntimeError("prelock selected tensor set changed")
    for role in ("down_proj","v_proj","k_proj"):
        eligible=[(r["actual_N8_improvement_fullQ4_minus_N8"],name) for name,r in rank.items() if r["role"]==role]
        best=max(eligible)
        if best[1] not in expected:
            raise RuntimeError("Qv12 prereg was not previous best within selected role")
    tok=Tokenizer.from_file(str(SNAP/"tokenizer.json"))
    rows=load_dataset("ehovy/race","middle",split="test",revision=REV)
    ordered=sorted(rows,key=lambda r:(sha((REV+"/test/"+r["example_id"]+"/"+r["question"]).encode()),r["example_id"],r["question"]))
    selectedq=[];articleids=set();eligible_count=0;excluded_prior=0
    for r in ordered:
        v=tokenize(r,tok)
        if v is None:continue
        eligible_count+=1
        if v["article_original_id"] in priorids:
            excluded_prior+=1;continue
        if v["article_original_id"] in articleids:continue
        articleids.add(v["article_original_id"])
        selectedq.append(v)
        if len(selectedq)==16:break
    if len(selectedq)!=16 or len(articleids)!=16 or articleids & priorids:
        raise RuntimeError("new RACE test16 not independent of Qv11 prior article IDs")
    os.umask(0o077)
    LAB.mkdir(parents=True,exist_ok=True,mode=0o700)
    priv={"schema":"BEGLIN-Qv12-frozen-new-disjoint-RACE-middle-test16/1",
      "status":"NEW_RACE_TEST16_SHA_LOCKED_BEFORE_NATIVE_Q4_SINGLE_AND_PAIR_FORWARD",
      "preregistration_SHA256":PRE_SHA,"original_HF_rev":cfg["model_HF_rev"],
      "dataset_revision":REV,"dataset_split":"test","dataset_config":"middle",
      "previous_Qv11_private_SHA256":cfg["prior_Qv11_original_test16_private_SHA256"],
      "article_not_in_Qv11_previous16":True,"one_question_per_article":True,
      "all_original_full_passages_untruncated":True,
      "question_count":16,"full_frozen_questions":selectedq,
      "new_model_inference_or_answer_accuracy_unused_to_select":True}
    fullsha=write_private(OUT,priv)
    pub={"schema":"BEGLIN-Qv12-prelocked-three-native-singleton-and-three-pair-original-RACEtest16/1",
      "status":"FROZEN_THREE_TENSORS_SIX_NATIVE_UPGRADES_AND_16_NEW_RACE_ARTICLES_BEFORE_EXECUTION",
      "prelock_SHA256":PRE_SHA,"RACE_new_test16_private_SHA256":fullsha,
      "old_Qv11_RACE_test16_SHA256":cfg["prior_Qv11_original_test16_private_SHA256"],
      "old_Qv11_source_outcome_SHA256":cfg["prior_Qv11_7arm_true_N8_public_audit_SHA256"],
      "preselected_tensors":selected,"preselected_source_observed_singleton_outcome_reused":True,
      "new_test16_disjoint_original_article_ids":True,"new_test16_distinct_article_count":16,
      "new_test16_article_id_order_sha256":sha(json.dumps([x["article_original_id"] for x in selectedq]).encode()),
      "new_test16_prompt_len_range":[min(q["prompt_len"] for q in selectedq),max(q["prompt_len"] for q in selectedq)],
      "number_valid_eligible_before_16_article_selection":eligible_count,
      "number_prior_articles_excluded_during_selection":excluded_prior,
      "native_endtask_NLL_not_observed_at_freeze":True,
      "no_operational_promotion":True}
    data=(json.dumps(pub,sort_keys=True,indent=2)+"\n").encode()
    with PUB.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    print("QV12_PRE_FROZEN_NEW_DISJOINT_RACE_TEST16_FOR_7_PAIR_ARMS",json.dumps({
      "prev_excluded":excluded_prior,"prompt_range":pub["new_test16_prompt_len_range"],
      "source_pairs":list(expected),"private_sha":fullsha,"public_sha":sha(data)},sort_keys=True),flush=True)
if __name__=="__main__":main()
