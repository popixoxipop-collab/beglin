#!/usr/bin/env python3
"""Freeze disjoint source 24+24 ARC questions and MMLU psychology/CS 24+24.

The source 48 are positions 128:176 AFTER original Qv6 SHA-sorted 128 source
questions and never chosen by any model output. MMLU sampled by ID/content
only. No model forward or per-policy scores used at this step.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,sys
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT=pathlib.Path(__file__).resolve().parents[2]
Q6=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv6-fisher-20261022")
Q6SOURCE=Q6/"lab_local/qv6_fisher_scaling_20261022/source_arc_128_frozen.json"
Q5=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
sys.path.insert(0,str(Q5/"research/oml_acs"))
from q_v5_freeze_arc_inputs import choose
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
CONFIG=ROOT/"configs/research/BEGLIN_QV9_GGN_SOURCE_PRELOCK_20261023.json"
LAB=ROOT/"lab_local/qv9_ggn_20261023"
CAL=LAB/"ARC_train24_GGN_source_frozen.json"
DIAG=LAB/"ARC_diag24_independent_frozen.json"
MMLU=LAB/"MMLU_psych_CS_48_eval_frozen.json"
PUB=ROOT/"docs/research/BEGLIN_QV9_INPUT_PREMODEL_FREEZE_20261023.json"
PRE_SHA="73dc2829da79637a97d68f903f72fed753eace2b154caec111c1fe60a7f4f983"
Q6_SHA="78d267bef54240cfb32b292c5d1885b6716fadd9bbd78ec7b3b1bf97822a1ab2"
LETTERS="ABCD"
def sha(x:bytes):return hashlib.sha256(x).hexdigest()
def verify_private(p):
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:raise RuntimeError("Qv6 original private source unsealed")
    return p.read_bytes()
def write(p,o):
    b=(json.dumps(o,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
    return sha(b)
def parse_mmlu(row,subject,idx,tok):
    if not isinstance(row["question"],str) or not row["question"].strip() or len(row["choices"])!=4: return None
    if not isinstance(row["answer"],int) or not 0<=row["answer"]<4: return None
    q=row["question"].strip()
    if any(not isinstance(x,str) or not x.strip() for x in row["choices"]):return None
    body=("Choose the single best answer.\nQuestion: "+q+"\n"+
          "\n".join(f"{letter}. {t.strip()}" for letter,t in zip(LETTERS,row["choices"]))+
          "\nAnswer using exactly one letter: A, B, C, or D.")
    prompt="<|im_start|>user\n"+body+"<|im_end|>\n<|im_start|>assistant\n"
    prefix=tok.encode(prompt,add_special_tokens=False).ids
    if not 18<=len(prefix)<=208:return None
    options={}
    for letter in LETTERS:
        full=tok.encode(prompt+letter,add_special_tokens=False).ids
        if full[:len(prefix)]!=prefix or len(full)!=len(prefix)+1:return None
        options[letter]=full[-1]
    if len(set(options.values()))!=4:return None
    key=f"{subject}/{idx}"
    return {
      "id":key,"id_sha256":sha(key.encode()),"subject":subject,
      "prompt_token_count":len(prefix),"prefix_ids":prefix,
      "suffix_ids":options,"correct":LETTERS[row["answer"]],
      "question_sha256":sha(q.encode()),"content_SHA256":sha((q+"\n"+"\n".join(row["choices"])).encode())
    }
def main():
    if CAL.exists() or DIAG.exists() or MMLU.exists() or PUB.exists():
        raise FileExistsError("Qv9 input freeze exists, no late reselection")
    if sha(CONFIG.read_bytes())!=PRE_SHA or sha(verify_private(Q6SOURCE))!=Q6_SHA:
        raise RuntimeError("Qv9 original prelock/Qv6 source data provenance changed")
    protocol=json.loads(CONFIG.read_text())
    tok=Tokenizer.from_file(str(SNAP/"tokenizer.json"))
    train,eligible=choose("train",176,tok,protocol["source_revision"])
    q6=json.loads(verify_private(Q6SOURCE))["all_source_questions"]
    if len(q6)!=128 or train[:128]!=q6:raise RuntimeError("Qv9 new source list not 100% after original 128")
    cal=train[128:152]
    diag=train[152:176]
    if len(cal)!=24 or len(diag)!=24 or {x["question_sha256"] for x in cal}&{x["question_sha256"] for x in diag}:
        raise RuntimeError("24+24 source calibration and diagnostic split overlap")
    chosen=[];candidate={}
    seen={x["question_sha256"] for x in train}
    for subject in protocol["subjects"]:
        ds=load_dataset(protocol["new_validation_dataset"],subject,split="test",revision=protocol["validation_rev"])
        sorted_rows=sorted(enumerate(ds),key=lambda pair:(
            sha((protocol["validation_rev"]+"/test/"+subject+"/"+str(pair[0])).encode()),pair[0]))
        per=[];ok=0
        for index,row in sorted_rows:
            item=parse_mmlu(row,subject,index,tok)
            if item is None:continue
            ok+=1
            if item["question_sha256"] in seen:continue
            seen.add(item["question_sha256"])
            per.append(item)
            if len(per)==24:break
        if len(per)!=24:raise RuntimeError("not enough frozen MMLU single-token cases "+subject)
        candidate[subject]={"total":len(ds),"eligible_examined":ok}
        chosen+=per
    if len(chosen)!=48 or len({x["id"] for x in chosen})!=48:
        raise RuntimeError("MMLU new subject-balanced cross-domain validation cohort not 48")
    os.umask(0o077);LAB.mkdir(parents=True,mode=0o700)
    cal_hash=write(CAL,{"schema":"BEGLIN-Qv9-24-original-BF16-GGN-training-ARC/1",
       "protocol_SHA256":PRE_SHA,"prior_Qv6_128_input_SHA256":Q6_SHA,
       "train_items":24,"dataset_rev":protocol["source_revision"],"questions":cal,
       "MMLU_validation_output_unopened":True})
    diag_hash=write(DIAG,{"schema":"BEGLIN-Qv9-original-independent-source-ARC24-diagnostic-only/1",
       "protocol_SHA256":PRE_SHA,"source_train24_disjoint":True,
       "dataset_rev":protocol["source_revision"],"questions":diag,
       "no_Qv9_policy_tuning_allowed":True})
    val_hash=write(MMLU,{"schema":"BEGLIN-Qv9-original-MMLU-psychology-CS-48-crossdomain-input-lock/1",
       "protocol_SHA256":PRE_SHA,"dataset_rev":protocol["validation_rev"],
       "source_dataset_never_used_for_eval":True,
       "subjects":protocol["subjects"],"items":chosen,
       "new_eval_quality_labels_not_used_by_optimizer":True})
    pub={"schema":"BEGLIN-Qv9-ARC24-train24-diagnostic-MMLU48-pre-model-freeze/1",
      "status":"ALL_THREE_COHORTS_FROZEN_BEFORE_BF16_GGN_BACKWARD_AND_NATIVE",
      "prelock_sha256":PRE_SHA,
      "Qwen_original_rev":protocol["original_model_rev"],
      "source128_sha256":Q6_SHA,
      "original_source64_128_176_sequence_proven":True,
      "source_train24_private_SHA256":cal_hash,
      "source_diagnostic24_private_SHA256":diag_hash,
      "MMLU_crossdomain48_private_SHA256":val_hash,
      "source_subjects":protocol["subjects"],"source_dataset_rev":protocol["source_revision"],
      "MMLU_rev":protocol["validation_rev"],"eval_count":len(chosen),
      "subject_eligible_counts":candidate,
      "test_prompt_token_range":[min(x["prompt_token_count"] for x in chosen),max(x["prompt_token_count"] for x in chosen)],
      "private_input_token_arrays_not_published":True,
      "model_logit_or_accuracy_accessed":False,
      "production_runtime_unchanged":True}
    b=(json.dumps(pub,indent=2,sort_keys=True)+"\n").encode()
    with PUB.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    print("QV9_REAL_NEW_BF16_GGN_MMLU_INPUTS_FROZEN",json.dumps({
        "prelock_sha":PRE_SHA,"source_sha":cal_hash,"diagnostic_sha":diag_hash,
        "MMLU_sha":val_hash,"public_sha":sha(b),
        "eval_token_range":pub["test_prompt_token_range"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
