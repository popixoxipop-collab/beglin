#!/usr/bin/env python3
"""C1-D pinned model BF16 activation extraction on NEW source-separated dev.

Extract only final-token decoder block activations. Text and labels are private.
This is not the final IID/OOD holdout. No model fine-tuning or target labels for
gating; task labels are attached after extraction solely to evaluate outcomes.
"""
from __future__ import annotations
import os
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"

import argparse,hashlib,json,time,collections
from pathlib import Path
import numpy as np

ROOT=Path.home()/".local/share/beglin/oml_acs/20261021"
DATA=ROOT/"C1D_NEW_PROSPECTIVE_DEV_200_20261021.jsonl"
DATA_SHA="6ee84d7b25a1025d0ae41dda32fbd1098a82499e8192cc9cde71808c52f3b4bb"
ANCHOR_SHA="47d5ddcb37b81212e3f43a34441662062b42a038d4cc606d9d7081c31131f83b"
MODELS={
 "qwen":dict(name="Qwen/Qwen2.5-1.5B-Instruct",rev="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
   snapshot=Path.home()/".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
   hidden=1536,num_layers=28,layer=14,shards=1,kind="qwen2",offload=False),
 "mistral":dict(name="mistralai/Mistral-7B-Instruct-v0.3",rev="c170c708c41dac9275d15a8fff4eca08d52bab71",
   snapshot=Path.home()/".cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71",
   hidden=4096,num_layers=32,layer=16,shards=3,kind="mistral",offload=True),
 "phi":dict(name="microsoft/Phi-4-mini-instruct",rev="cfbefacb99257ffa30c83adab238a50856ac3083",
   snapshot=Path("/mnt/g/TailnetCommander/work/oml_acs_hf/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083"),
   hidden=3072,num_layers=32,layer=16,shards=2,kind="phi3",offload=False)
}
def sha(b):return hashlib.sha256(b).hexdigest()
def load_rows():
    if not DATA.is_file() or DATA.stat().st_mode&0o777!=0o600:
        raise RuntimeError("new private data missing/unsecured")
    content=DATA.read_bytes()
    if sha(content)!=DATA_SHA:raise RuntimeError("new data source digest drift")
    rows=[json.loads(x) for x in content.decode().splitlines() if x.strip()]
    if len(rows)!=200 or len({r["canonical_sha256"] for r in rows})!=200:
        raise RuntimeError("new source has wrong number of rows/duplicates")
    if len(set(r["axis"] for r in rows))!=2:
        raise RuntimeError("new source axes count changed")
    for r in rows:
        if r["purpose"]!="prospective_development_only" or r["source_split"]!="train":
            raise RuntimeError("unapproved final/test samples")
        if sha(r["text"].encode())!=r["text_sha256"]:raise RuntimeError("row SHA mismatch")
    stats=collections.Counter((r["axis"],r["class_label"]) for r in rows)
    if sorted(stats.values())!=[50,50,50,50]:
        raise RuntimeError("class balance changed")
    return rows

def run(which):
    if which not in MODELS:raise ValueError("unapproved target/model")
    config=MODELS[which]
    out=ROOT/f"C1D_newdev_{which}_200_bf16.npz"
    receipt=ROOT/f"C1D_newdev_{which}_200_bf16_receipt.json"
    if out.exists() or receipt.exists():raise FileExistsError("will not overwrite prior measurements")
    rows=load_rows()
    snap=config["snapshot"]
    if snap.name!=config["rev"] or not (snap/"config.json").is_file() or not (snap/"tokenizer.json").is_file():
        raise RuntimeError("pinned model snapshot incomplete")
    if len(list(snap.glob("*.safetensors")))!=config["shards"]:
        raise RuntimeError("incomplete pinned model shard set")
    from transformers import AutoTokenizer,AutoModelForCausalLM
    import torch
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    os.umask(0o077)
    start=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(str(snap),local_files_only=True,trust_remote_code=False)
    kwargs=dict(dtype=torch.bfloat16,local_files_only=True,trust_remote_code=False,use_safetensors=True)
    if config["offload"]:
        kwargs.update(device_map="auto",max_memory={0:"12GiB","cpu":"20GiB"},low_cpu_mem_usage=True)
    model=AutoModelForCausalLM.from_pretrained(str(snap),**kwargs)
    if not config["offload"]:model=model.to("cuda")
    model.eval()
    if model.config.hidden_size!=config["hidden"] or model.config.model_type!=config["kind"] or len(model.model.layers)!=config["num_layers"]:
        raise RuntimeError("pinned architecture contract mismatch")
    print("NEW_DEV_MODEL_READY",which,round(time.perf_counter()-start,2),flush=True)
    vals=[];hooks=[]
    def capture(_mod,_args,output):
        h=output[0] if isinstance(output,tuple) else output
        if h.ndim!=3 or h.shape[-1]!=config["hidden"]:raise RuntimeError("activation shape mismatch")
        hooks.append(h[0,-1,:].detach().float().cpu().numpy().copy())
    handle=model.model.layers[config["layer"]].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for i,row in enumerate(rows):
                e=tok.apply_chat_template([{"role":"user","content":row["text"]}],
                    tokenize=True,add_generation_prompt=True,return_tensors="pt",return_dict=True)
                inputs={k:v.to(model.device) for k,v in e.items() if k in ("input_ids","attention_mask")}
                if "attention_mask" not in inputs:inputs["attention_mask"]=torch.ones_like(inputs["input_ids"])
                hooks.clear()
                model(**inputs,use_cache=False)
                if len(hooks)!=1:raise RuntimeError("expected exactly one post-block capture")
                vals.append(hooks[0])
                if (i+1)%50==0:print("NEW_DEV_PROGRESS",which,i+1,"of 200",round(time.perf_counter()-start,2),flush=True)
    finally:
        handle.remove()
    activations=np.asarray(vals,dtype=np.float32)
    if activations.shape!=(200,config["hidden"]) or not np.isfinite(activations).all():
        raise RuntimeError("extraction failed/incomplete")
    digest=sha(activations.tobytes(order="C"))
    temp=out.with_name(out.name+".tmp")
    with temp.open("xb") as f:
        np.savez_compressed(f,activations=activations,
            text_sha256=np.asarray([r["text_sha256"] for r in rows]),
            axis=np.asarray([r["axis"] for r in rows]),
            data_source_sha256=np.asarray(DATA_SHA),
            anchor_source_sha256=np.asarray(ANCHOR_SHA),
            model_revision=np.asarray(config["rev"]),
            layer_index_zero_based=np.asarray(config["layer"]))
        f.flush();os.fsync(f.fileno())
    temp.replace(out);out.chmod(0o600)
    torch.cuda.synchronize()
    rec=dict(schema="oml-acs-c1d-new-200-bf16-activations/1",
       status="PASS_TECHNICAL_EXTRACTION_NO_FINAL",
       model=which,model_id=config["name"],revision=config["rev"],
       layer_zero_based=config["layer"],data_source_sha256=DATA_SHA,
       anchor_source_sha256=ANCHOR_SHA,
       shape=[200,config["hidden"]],dtype="float32",bf16_model=True,
       array_sha256=digest,archive_sha256=sha(out.read_bytes()),archive_size=out.stat().st_size,
       private_mode="0600",all_finite=True,
       old_behavior_model_not_tuned=True,original_test_split_read=False,
       target_labels_not_used_for_inference=True,
       elapsed_seconds=round(time.perf_counter()-start,2),
       cuda_peak_allocated_mib=round(torch.cuda.max_memory_allocated()/1048576,2),
       production_touched=False)
    with receipt.open("x",encoding="utf-8") as f:
        json.dump(rec,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
    receipt.chmod(0o600)
    print("NEW_DEV_ACTIVATION_RECEIPT",json.dumps(rec,sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=sorted(MODELS),required=True)
    run(p.parse_args().model)
