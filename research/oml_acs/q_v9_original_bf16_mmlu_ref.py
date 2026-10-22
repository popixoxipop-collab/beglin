#!/usr/bin/env python3
"""Unmodified Qwen original BF16 MPS reference on exactly frozen new MMLU 48.

Read validation only after source-only Qv9 GGN 28n8 mask is frozen.
Does not modify original checkpoint; is not byte-identical native C widened F32.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT=pathlib.Path(__file__).resolve().parents[2]
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
LAB=ROOT/"lab_local/qv9_ggn_20261023"
DATA=LAB/"MMLU_psych_CS_48_eval_frozen.json"
POL=ROOT/"docs/research/BEGLIN_QV9_ROBUST_CONDITIONAL_GGN_SOURCE_POLICY_20261023.json"
OUT=LAB/"original_BF16_Qwen_MMLU48_reference_private.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV9_ORIGINAL_BF16_MMLU48_PUBLIC_20261023.json"
DATA_SHA="2203bf54dfbc20e07488a234a012011efd78b098876e981de8ebed12cf8045c9"
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    if OUT.exists() or PUBLIC.exists():raise FileExistsError("reference already sealed")
    if sha(DATA.read_bytes())!=DATA_SHA:raise RuntimeError("Qv9 exact MMLU frozen tokens drift")
    if not POL.exists() or json.loads(POL.read_text())["status"]!="GGN_SOURCE_ONLY_FROZEN_BEFORE_DIAG_OR_MMLU":
        raise RuntimeError("cannot evaluate MMLU before source-only GGN fixed")
    samples=json.loads(DATA.read_text())["items"]
    if len(samples)!=48:raise RuntimeError("not 48 frozen MMLU")
    if not torch.backends.mps.is_available():raise RuntimeError("requires Apple MPS original BF16")
    torch.set_num_threads(2)
    t=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),dtype=torch.bfloat16,
       local_files_only=True,trust_remote_code=False,attn_implementation="eager").eval().to("mps")
    outputs=[]
    with torch.inference_mode():
        for i,ex in enumerate(samples):
            x=torch.tensor([ex["prefix_ids"]],dtype=torch.long,device="mps")
            logits=model(input_ids=x,use_cache=False).logits[0,-1,:].float()
            answer=torch.tensor([ex["suffix_ids"][ex["correct"]]],dtype=torch.long,device="mps")
            nll=float(F.cross_entropy(logits.unsqueeze(0),answer).cpu())
            candidate_ids=[ex["suffix_ids"][c] for c in "ABCD"]
            selected="ABCD"[int(logits[candidate_ids].argmax().cpu())]
            if not math.isfinite(nll):raise RuntimeError("original BF16 MMLU NLL nonfinite")
            outputs.append({"original_question_id_sha256":sha(ex["id"].encode()),"subject":ex["subject"],
              "correct":ex["correct"],"predicted":selected,
              "correct_label_full_vocab_NLL":nll,"right":selected==ex["correct"]})
            if (i+1)%12==0:print("QV9_TRUE_ORIGINAL_BF16_MMLU_PROGRESS",i+1,"/48",flush=True)
    count=sum(x["right"] for x in outputs)
    nll=sum(x["correct_label_full_vocab_NLL"] for x in outputs)/48
    receipt={
      "schema":"BEGLIN-Qv9-original-BF16-Qwen-MMLU48-reference/1",
      "status":"ACTUAL_ORIGINAL_UNMODIFIED_HF_BF16_MMLU48_REFERENCE",
      "official_model_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
      "pinned_MMLU_frozen_inputs_SHA256":DATA_SHA,
      "GGN_source_only_mask_sha256":json.loads(POL.read_text())["mask_sha256"],
      "source_dataset":"MMLU test high_school_psychology and high_school_computer_science",
      "subject_count_each":24,"correct":count,"n":48,
      "mean_correct_answer_full_vocab_NLL":nll,
      "per_case_original_BF16":outputs,
      "device":"PyTorch original BF16 Apple MPS",
      "original_model_weights_mutated":False,
      "numerical_parity_with_native_quantized_C_not_claimed":True,
      "elapsed_s":round(time.monotonic()-t,2)}
    os.umask(0o077)
    data=(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode()
    with OUT.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    doc={k:v for k,v in receipt.items() if k!="per_case_original_BF16"}
    doc["private_original_reference_receipt_sha256"]=sha(data)
    body=(json.dumps(doc,sort_keys=True,indent=2)+"\n").encode()
    with PUBLIC.open("xb") as f:f.write(body);f.flush();os.fsync(f.fileno())
    print("QV9_ORIGINAL_BF16_MMLU48_REFERENCE_PASS",json.dumps({
      "correct":count,"NLL":nll,"private_SHA":sha(data),"public_SHA":sha(body)},sort_keys=True),flush=True)
if __name__=="__main__":main()
