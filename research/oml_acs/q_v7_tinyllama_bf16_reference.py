#!/usr/bin/env python3
"""Actual original TinyLlama BF16 unmodified MPS model on frozen 48 test IDs.

Runs independent HuggingFace original model architecture (Llama), no bit mask
or quantized model changes. Scores correct token under full vocab and four-
option argmax. This is reference, not proof of native BEGLIN BF16 parity.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT=pathlib.Path(__file__).resolve().parents[2]
MODEL=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv7-tinyllama-model-20261022")
LAB=ROOT/"lab_local/qv7_tinyllama_20261022"
DATA=LAB/"eval_openbookqa_test48_tinyllama_frozen.json"
OUT=LAB/"TinyLlama_original_BF16_test48_private.json"
PUB=ROOT/"docs/research/BEGLIN_QV7_TINYLLAMA_ORIGINAL_BF16_TEST48_PUBLIC.json"
TEST_SHA="8807002e79afadcb2d150b32dffdefa204afb0d70678375a608b7900f18d0ab0"
def sha(b):return hashlib.sha256(b).hexdigest()
def run():
    if OUT.exists() or PUB.exists():raise FileExistsError("cannot overwrite BF16 original TinyLlama reference")
    if sha(DATA.read_bytes())!=TEST_SHA:raise RuntimeError("TinyLlama frozen test 48 changed")
    d=json.loads(DATA.read_text())["test48"]
    if not (MODEL/"model.safetensors").is_file() or len(d)!=48:
        raise RuntimeError("pinned original llama source not fully downloaded")
    if not torch.backends.mps.is_available():raise RuntimeError("Apple BF16 MPS required")
    torch.set_num_threads(2)
    t=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),dtype=torch.bfloat16,local_files_only=True,
        trust_remote_code=False,attn_implementation="eager").eval().to("mps")
    if model.config.model_type!="llama" or model.config.num_hidden_layers!=22:raise RuntimeError("wrong Llama HF model")
    outs=[]
    with torch.inference_mode():
        for ix,q in enumerate(d):
            input_ids=torch.tensor([q["prefix_ids"]],dtype=torch.long,device="mps")
            logits=model(input_ids=input_ids,use_cache=False).logits[0,-1,:].float()
            target=torch.tensor([q["suffix_ids"][q["correct"]]],dtype=torch.long,device="mps")
            nll=float(F.cross_entropy(logits.unsqueeze(0),target).cpu())
            guess="ABCD"[int(logits[[q["suffix_ids"][letter] for letter in "ABCD"]].argmax().cpu())]
            if not math.isfinite(nll):raise RuntimeError("nonfinite BF16 TinyLlama output")
            outs.append({"id_sha256":sha(q["id"].encode()),
                 "answer":q["correct"],"prediction":guess,
                 "correct":guess==q["correct"],"nll":nll})
            if (ix+1)%12==0:print("QV7_ORIGINAL_LLAMA_BF16_48_PROGRESS",ix+1,flush=True)
    right=sum(z["correct"] for z in outs);nll=sum(z["nll"] for z in outs)/48
    receipt={"schema":"BEGLIN-Qv7-original-HF-TinyLlama-BF16-MPS-48/1",
     "status":"ACTUAL_ORIGINAL_LLAMA_FAMILY_BF16_UNMODIFIED_MODEL_COMPLETE",
     "official_revision":"fe8a4ea1ffedaf415f4da2f062534de366a451e6",
     "source_model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
     "dataset":"OpenBookQA test", "frozen_test48_sha256":TEST_SHA,
     "source_arch":"llama","decoder_layers":22,
     "correct_count":right,"case_count":48,"correct_label_NLL":nll,
     "per_case_results":outs,"device":"Apple MPS BF16 PyTorch original",
     "elapsed_seconds":round(time.monotonic()-t,2),
     "no_original_weight_changes":True,"no_live_promotion":True}
    os.umask(0o077)
    raw=(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode()
    with OUT.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    public={k:v for k,v in receipt.items() if k!="per_case_results"}
    public["private_receipt_sha256"]=sha(raw)
    out=(json.dumps(public,sort_keys=True,indent=2)+"\n").encode()
    with PUB.open("xb") as f:f.write(out);f.flush();os.fsync(f.fileno())
    print("QV7_ORIGINAL_TINYLLAMA_BF16_TEST48_DONE",json.dumps({
      "correct":right,"NLL":nll,"audit_sha":sha(raw),"public_sha":sha(out)},sort_keys=True),flush=True)
if __name__=="__main__":run()
