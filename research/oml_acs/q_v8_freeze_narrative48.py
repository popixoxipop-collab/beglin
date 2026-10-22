#!/usr/bin/env python3
"""Freeze 96 original Qv6 ARC BF16 source questions and new HellaSwag val48.

All inputs selected independently from native model logits / answer loss. No
prior ARC-dev/OpenBookQA-dev validation questions used to optimize the mask.
"""
import hashlib,json,os,pathlib
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT=pathlib.Path(__file__).resolve().parents[2]
Q6=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv6-fisher-20261022")
SRC_Q6=Q6/"lab_local/qv6_fisher_scaling_20261022/source_arc_128_frozen.json"
PROTO=ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_SOURCE_PRELOCK_20261022.json"
LAB=ROOT/"lab_local/qv8_joint_signed_20261022"
SOURCE=LAB/"pinned_original_ARC_train96.json"
HELD=LAB/"pinned_HellaSwag_val48.json"
PUB=ROOT/"docs/research/BEGLIN_QV8_SOURCE_AND_HELLASWAG_PREMODEL_FREEZE_20261022.json"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
PRE_SHA="9b833f5fc2729d13320c87e368baaff952c890ed068586622badfc7da63b4db1"
SRC_Q6_SHA="78d267bef54240cfb32b292c5d1885b6716fadd9bbd78ec7b3b1bf97822a1ab2"
def sha(b):return hashlib.sha256(b).hexdigest()
def sealed(p):
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("Qv6 model source train128 not privately sealed")
    return p.read_bytes()
def save(p,obj):
    b=(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
    return sha(b)
def parse(row,tokenizer):
    endings=row["endings"]
    if len(endings)!=4 or any(not isinstance(t,str) or not t.strip() for t in endings):
        return None
    try:label=int(row["label"])
    except (ValueError,TypeError):return None
    if not 0<=label<=3:return None
    context=row["ctx"].strip()
    if not context:return None
    # Unmodified natural-language context/candidate content, no truncation.
    prompt="Choose the most plausible continuation of the situation.\nContext: "+context
    prompt+="\n"+"\n".join(f"{letter}. {text.strip()}" for letter,text in zip("ABCD",endings))
    prompt+="\nAnswer using exactly one letter: A, B, C, or D."
    chat="<|im_start|>user\n"+prompt+"<|im_end|>\n<|im_start|>assistant\n"
    ids=tokenizer.encode(chat,add_special_tokens=False).ids
    if len(ids)<18 or len(ids)>208:return None
    options={}
    for letter in "ABCD":
        all_tokens=tokenizer.encode(chat+letter,add_special_tokens=False).ids
        if all_tokens[:len(ids)]!=ids or len(all_tokens)!=len(ids)+1:
            return None
        options[letter]=all_tokens[-1]
    if len(set(options.values()))!=4:return None
    key=f"{row['ind']}:{row['source_id']}"
    text=" ".join([context]+endings)
    return {"id":key,"source_ind":row["ind"],"prefix_ids":ids,
        "correct":"ABCD"[label],"suffix_ids":options,
        "prompt_token_count":len(ids),"context_sha256":sha(context.encode()),
        "content_sha256":sha(text.encode()),
        "id_sha256":sha(key.encode())}
def main():
    if SOURCE.exists() or HELD.exists() or PUB.exists():
        raise FileExistsError("cannot refreeze evaluation questions after study")
    if sha(PROTO.read_bytes())!=PRE_SHA or sha(sealed(SRC_Q6))!=SRC_Q6_SHA:
        raise RuntimeError("previous source or preregistration identity mismatch")
    proto=json.loads(PROTO.read_text())
    q6=json.loads(sealed(SRC_Q6))
    rows=q6["all_source_questions"][:96]
    if len(rows)!=96 or len({q["id"] for q in rows})!=96:
        raise RuntimeError("original pinned ARC source 96 incomplete")
    tok=Tokenizer.from_file(str(SNAP/"tokenizer.json"))
    data=load_dataset(proto["evaluation_repo"],split="validation",revision=proto["evaluation_revision"])
    ranked=sorted(data,key=lambda q:(sha((proto["evaluation_revision"]+"/validation/"+str(q["ind"])).encode()),int(q["ind"])))
    held=[];eligible=0;seen=set()
    for item in ranked:
        row=parse(item,tok)
        if row is None:continue
        eligible+=1
        if row["content_sha256"] in seen:continue
        seen.add(row["content_sha256"])
        held.append(row)
        if len(held)==48:break
    if len(held)!=48:raise RuntimeError("48 HellaSwag single-token-label questions unavailable")
    os.umask(0o077);LAB.mkdir(mode=0o700,parents=True,exist_ok=True)
    s=save(SOURCE,{"schema":"BEGLIN-Qv8-source-original-Qv6-ARC96/1",
       "preregistration_SHA256":PRE_SHA,"old_Qv6_ARC128_SHA256":SRC_Q6_SHA,
       "source_train_count":96,"source_train96":rows,
       "same_exact_source_Qwen_chat_tokens_as_Qv6":True})
    v=save(HELD,{"schema":"BEGLIN-Qv8-HellaSwag-val48-locked/1",
       "preregistration_SHA256":PRE_SHA,
       "dataset_revision":proto["evaluation_revision"],"held_count":48,
       "held_validation48":held,"source_train_optimization_has_no_access_to_held_logits":True})
    pub={"schema":"BEGLIN-Qv8-original-ARC96-and-new-HellaSwag48-dataset-lock/1",
       "status":"SOURCE_AND_OUTOFDOMAIN_EVALUATION_FROZEN_BEFORE_SIGNED_GRADIENT",
       "preregistration_sha256":PRE_SHA,"frozen_ARC96_SHA256":s,
       "frozen_HellaSwag48_SHA256":v,
       "HF_dataset_revision":proto["evaluation_revision"],
       "source_count":96,"eval_count":48,"candidate_full_prompt_count":eligible,
       "source96_id_digest":sha(json.dumps([r["id"] for r in rows]).encode()),
       "held48_id_digest":sha(json.dumps([r["id"] for r in held]).encode()),
       "held_min_max_prompt_tokens":[min(q["prompt_token_count"] for q in held),max(q["prompt_token_count"] for q in held)],
       "no_prior_ARC_or_OpenBookQA_eval_prompts_used":True,
       "source_and_eval_dataset_different":True,"production_untouched":True}
    b=(json.dumps(pub,indent=2,sort_keys=True)+"\n").encode()
    with PUB.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    print("BEGLIN_QV8_ARC96_HELLASWAG48_FROZEN",json.dumps({
       "src_sha":s,"held_sha":v,"pub_sha":sha(b),
       "min_max":pub["held_min_max_prompt_tokens"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
