#!/usr/bin/env python3
"""Independent official Qwen BF16 Transformers reference on frozen ARC 40.

Reads the validation inputs only after the Qv5 source-only Fisher mask was
sealed. Executes the original model without ANY quantization/weight mutation.
Provides an external arithmetic oracle rather than trusting native C logs.
Not a native BF16-vs-F32 exact arithmetic equivalence proof.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv5_fisher_20261022"
DEV=LAB/"development_val_frozen.json"
PROTO=ROOT/"configs/research/BEGLIN_QV5_FISHER_GRAD_PRELOCK_20261022.json"
POLICY=ROOT/"docs/research/BEGLIN_QV5_FISHER_HIGHS_SOURCE_MILP_20261022.json"
MASK=ROOT/"configs/research/BEGLIN_QV5_FISHER_28N8_SOURCE_POLICY_20261022.txt"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
OUT=LAB/"native_40val_6mode/independent_original_BF16_HF_mps_reference.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV5_ORIGINAL_BF16_HF_40_ARC_AGGREGATE_20261022.json"
DEV_SHA="a00985cc73f80518f448c2b8a1295cf894283723f87b5af53321d486ce0187ae"
PROTO_SHA="a886418affc9f665d64ce4ca9eb0b5ea2aef4ae97bc59bf58ea1affb53bf19ef"
def sha(data):return hashlib.sha256(data).hexdigest()
def private(p,obj):
    b=(json.dumps(obj,sort_keys=True,indent=2)+"\n").encode()
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
    return sha(b)
def run():
    if OUT.exists() or PUBLIC.exists():raise FileExistsError("BF16 oracle already sealed")
    if sha(PROTO.read_bytes())!=PROTO_SHA or sha(DEV.read_bytes())!=DEV_SHA:
        raise RuntimeError("not frozen ARC original validation")
    if not POLICY.exists():raise RuntimeError("never evaluate BF16 validation before source Fisher policy freeze")
    pol=json.loads(POLICY.read_text())
    if pol["status"]!="FROZEN_SOURCE_ONLY_EXACT_MILP_NOT_YET_ARC_VALIDATION_SCORED" or sha(MASK.read_bytes())!=pol["mask_sha256"]:
        raise RuntimeError("source Fisher policy not frozen")
    val=json.loads(DEV.read_text())
    if len(val["questions"])!=40:raise RuntimeError("validation cohort not 40")
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():raise RuntimeError("Apple MPS BF16 reference unavailable")
    t=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(str(SNAP),dtype=torch.bfloat16,local_files_only=True,
        trust_remote_code=False,attn_implementation="eager").eval().to("mps")
    results=[];total=0.;passed=0
    with torch.inference_mode():
        for i,row in enumerate(val["questions"]):
            x=torch.tensor([row["prefix_ids"]],dtype=torch.long,device="mps")
            logits=model(input_ids=x,use_cache=False).logits[0,-1,:].float()
            target=torch.tensor([row["suffix_ids"][row["correct"]]],dtype=torch.long,device="mps")
            nll=float(F.cross_entropy(logits.unsqueeze(0),target).cpu())
            ids=[row["suffix_ids"][ch] for ch in "ABCD"]
            best="ABCD"[int(logits[ids].argmax().cpu())]
            if not math.isfinite(nll) or nll<0:raise RuntimeError("HF BF16 oracle nonfinite")
            passed+=int(best==row["correct"]);total+=nll
            results.append({"question_id_sha":sha(row["id"].encode()),
                "source_prompt_SHA256":row["input_sha256"],"mean_answer_nll":nll,
                "predicted":best,"correct":row["correct"],"correct_prediction":best==row["correct"]})
            if (i+1)%10==0:
                print("QV5_BF16_MPS_ORACLE_PROGRESS",i+1,"/",len(val),flush=True)
    dest={"schema":"BEGLIN-Qv5-original-BF16-HF-40-ARC-developer-oracle/1",
      "status":"INDEPENDENT_OFFICIAL_BF16_MPS_FORWARD_REFERENCE_ONLY",
      "original_BF16_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
      "local_source_original_BF16_cache":SNAP.name,"original_dataset_sha":json.loads(PROTO.read_text())["dataset_sha"],
      "dev_frozen_sha256":DEV_SHA,"source_only_fisher_mask_sha256":pol["mask_sha256"],
      "device":"Apple MPS BF16 Torch 2.13; no native BEGLIN quantization",
      "correct":passed,"total":len(results),
      "accuracy":passed/len(results),"correct_label_NLL":total/len(results),
      "individual_examples":results,"source_gradients_not_recomputed":True,
      "no_any_model_weight_changes":True,"no_production_promotion":True,
      "elapsed_wall_s":round(time.monotonic()-t,2)}
    os.umask(0o077)
    h=private(OUT,dest)
    pub={k:v for k,v in dest.items() if k!="individual_examples"}
    pub["original_private_oracle_sha256"]=h
    pubh=(json.dumps(pub,sort_keys=True,indent=2)+"\n").encode()
    with PUBLIC.open("xb") as f:f.write(pubh);f.flush();os.fsync(f.fileno())
    print("QV5_ORIGINAL_BF16_HF_ARC40_ORACLE_PASS",json.dumps({"correct":passed,
        "nll":total/len(results),"private_SHA256":h,"public_SHA256":sha(pubh),
        "elapsed_s":round(time.monotonic()-t,2)},sort_keys=True),flush=True)
if __name__=="__main__":run()
