#!/usr/bin/env python3
"""Extract BF16 final-token block activations for pinned real behavior train/dev rows.

This is technical development-only data processing. The 600 prompts/labels
come exclusively from pinned SST-2 and CARER train+validation; no test/holdout.
All raw text and activation arrays stay private (0600), no published text.
"""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import collections
import hashlib
import json
from pathlib import Path
import time
import numpy as np

BEHAVIOR_ROOT_SHA = "b99fb5d7e8c909e6821f555ac590b36ff52393b9d4e7a01828ee936c615bd274"
ANCHORS_SHA = "47d5ddcb37b81212e3f43a34441662062b42a038d4cc606d9d7081c31131f83b"
BASE = Path.home() / ".local/share/beglin/oml_acs/20261021"
SOURCE = BASE / "behavior_sst2_carer_train_dev_v1.jsonl"
SOURCES = {
    "qwen": {
        "id":"Qwen/Qwen2.5-1.5B-Instruct",
        "revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "snapshot":Path.home()/".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "layer":14, "layers":28, "hidden":1536, "shards":1,
        "anchor_projector":BASE/"qwen25_300_anchor_projector_v1.npz",
        "device_map":False,
    },
    "mistral": {
        "id":"mistralai/Mistral-7B-Instruct-v0.3",
        "revision":"c170c708c41dac9275d15a8fff4eca08d52bab71",
        "snapshot":Path.home()/".cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71",
        "layer":16, "layers":32, "hidden":4096, "shards":3,
        "anchor_projector":BASE/"mistral7b_layer16_300_projector_bf16.npz",
        "device_map":True,
    },
    "phi": {
        "id":"microsoft/Phi-4-mini-instruct",
        "revision":"cfbefacb99257ffa30c83adab238a50856ac3083",
        "snapshot":Path("/mnt/g/TailnetCommander/work/oml_acs_hf/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083"),
        "layer":16, "layers":32, "hidden":3072, "shards":2,
        "anchor_projector":BASE/"phi4mini_layer16_300_projector_bf16.npz",
        "device_map":False,
    },
}

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def load_source() -> list[dict]:
    if not SOURCE.is_file() or (SOURCE.stat().st_mode & 0o777) != 0o600:
        raise RuntimeError("behavior data missing or not private")
    content=SOURCE.read_bytes()
    if sha(content)!=BEHAVIOR_ROOT_SHA:
        raise RuntimeError("behavior corpus SHA mismatch")
    rows=[json.loads(line) for line in content.decode().splitlines() if line]
    if len(rows)!=600:
        raise RuntimeError("wrong behavior row count")
    allowed={"behavior_train","behavior_dev"}
    if set(r["split"] for r in rows)!=allowed:
        raise RuntimeError("final/test split must remain unopened")
    if len(set(r["canonical_text_sha256"] for r in rows))!=600:
        raise RuntimeError("behavior duplicates")
    for r in rows:
        if sha(r["text"].encode("utf-8"))!=r["text_sha256"]:
            raise RuntimeError("behavior text source drift")
        if r["split"] not in allowed or r["class_label"] not in (0,1):
            raise RuntimeError("split/label mismatch")
    counts=collections.Counter((r["axis"],r["split"],int(r["class_label"])) for r in rows)
    axes={"sst2_positive_vs_negative","carer_joy_vs_anger"}
    for axis in axes:
        for split,n in (("behavior_train",100),("behavior_dev",50)):
            for cls in (0,1):
                if counts[(axis,split,cls)]!=n:
                    raise RuntimeError("unbalanced behavior strata")
    if len(axes)!=len({r["axis"] for r in rows}):
        raise RuntimeError("unexpected behavior axis")
    return rows

def ensure_checkpoint(spec: dict)->None:
    snap=spec["snapshot"]
    if snap.name!=spec["revision"] or not (snap/"config.json").is_file():
        raise RuntimeError("model revision or config mismatch")
    if len(list(snap.glob("*.safetensors"))) != spec["shards"]:
        raise RuntimeError("incomplete pinned BF16 safetensors")
    if not spec["anchor_projector"].is_file():
        raise RuntimeError("corresponding 300-anchor projector not yet verified")
    with np.load(spec["anchor_projector"],allow_pickle=False) as p:
        if p["source_raw_sha256"].item()!=ANCHORS_SHA:
            raise RuntimeError("anchor-root mismatch")
        if p["ahat"].shape!=(300,spec["hidden"]):
            raise RuntimeError("expected full 300-anchor projector")

def extract(which: str)->None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    start=time.perf_counter()
    if which not in SOURCES:
        raise ValueError("unknown model alias")
    spec=SOURCES[which]
    out=BASE/f"behavior_{which}_600_bf16.npz"
    receipt=BASE/f"behavior_{which}_600_bf16_receipt.json"
    for path in (out,receipt):
        if path.exists():
            raise FileExistsError("refuse overwrite "+str(path))
    os.umask(0o077)
    rows=load_source()
    ensure_checkpoint(spec)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for isolated source comparison")
    tokenizer=AutoTokenizer.from_pretrained(str(spec["snapshot"]),local_files_only=True,
                                            trust_remote_code=False)
    opts=dict(local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16)
    if spec["device_map"]:
        opts.update(device_map="auto",max_memory={0:"12GiB","cpu":"20GiB"},low_cpu_mem_usage=True)
    model=AutoModelForCausalLM.from_pretrained(str(spec["snapshot"]),**opts)
    if not spec["device_map"]:
        model=model.to("cuda")
    model.eval()
    if model.config.hidden_size!=spec["hidden"] or len(model.model.layers)!=spec["layers"]:
        raise RuntimeError("model architecture changed")
    if model.config.model_type not in {"phi3","qwen2","mistral"}:
        raise RuntimeError("unexpected architecture/type")
    block=model.model.layers[spec["layer"]]
    print("BEHAVIOR_MODEL_LOADED",which,round(time.perf_counter()-start,2),flush=True)
    activations=[]
    observed=[]
    def hook(_module,_args,output):
        h=output[0] if isinstance(output,tuple) else output
        if h.ndim!=3 or h.shape[-1]!=spec["hidden"]:
            raise RuntimeError("unexpected post-block activation")
        observed.append(h[0,-1,:].detach().float().cpu().numpy().copy())
    handle=block.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for idx,row in enumerate(rows):
                enc=tokenizer.apply_chat_template([{"role":"user","content":row["text"]}],
                     tokenize=True,add_generation_prompt=True,return_tensors="pt",return_dict=True)
                inp={k:v.to(model.device) for k,v in enc.items() if k in ("input_ids","attention_mask")}
                if "attention_mask" not in inp:
                    inp["attention_mask"]=torch.ones_like(inp["input_ids"])
                observed.clear()
                model(**inp,use_cache=False)
                if len(observed)!=1:
                    raise RuntimeError("layer hook called zero or multiple times")
                activations.append(observed[0])
                if (idx+1)%100==0:
                    print("BEHAVIOR_ACTIVATION_PROGRESS",which,idx+1,"of 600",
                          "elapsed",round(time.perf_counter()-start,2),flush=True)
    finally:
        handle.remove()
    arr=np.asarray(activations,dtype=np.float32)
    if arr.shape!=(600,spec["hidden"]) or not np.isfinite(arr).all():
        raise RuntimeError("incomplete/non-finite behavior embeddings")
    raw_array_sha=sha(arr.tobytes(order="C"))
    tmp=out.with_name(out.name+".tmp")
    with tmp.open("xb") as f:
        np.savez_compressed(f,activations=arr,
          text_sha256=np.asarray([r["text_sha256"] for r in rows]),
          axis=np.asarray([r["axis"] for r in rows]),
          split=np.asarray([r["split"] for r in rows]),
          labels=np.asarray([r["class_label"] for r in rows],dtype=np.int8),
          behavior_source_sha256=np.asarray(BEHAVIOR_ROOT_SHA),
          anchor_source_sha256=np.asarray(ANCHORS_SHA),
          model_revision=np.asarray(spec["revision"]),
          layer_index_zero_based=np.asarray(spec["layer"]))
        f.flush();os.fsync(f.fileno())
    tmp.replace(out);out.chmod(0o600)
    torch.cuda.synchronize()
    rec={
      "schema":"oml-acs-behavior-600-bf16-activations/1",
      "status":"PASS_TECHNICAL_EXTRACTION",
      "not_final_eval":True,"scientific_verdict":"NONE",
      "model":spec["id"],"model_revision":spec["revision"],
      "layer_zero_based":spec["layer"],"precision":"BF16",
      "behavior_source_sha256":BEHAVIOR_ROOT_SHA,
      "anchor_source_sha256":ANCHORS_SHA,
      "shape":[600,spec["hidden"]],"array_sha256":raw_array_sha,
      "npz_sha256":sha(out.read_bytes()),"npz_bytes":out.stat().st_size,
      "private_mode":"0600","split_names":["behavior_train","behavior_dev"],
      "gpu":torch.cuda.get_device_name(0),
      "peak_cuda_allocated_mb":round(torch.cuda.max_memory_allocated()/1048576,2),
      "elapsed_seconds":round(time.perf_counter()-start,2),
      "target_eval_labels_used_for_axis_fit":False,
      "V6_consumed_24_verified_excluded":False,
      "production_touched":False,
    }
    with receipt.open("x",encoding="utf-8") as f:
        json.dump(rec,f,indent=2,sort_keys=True);f.write("\n")
        f.flush();os.fsync(f.fileno())
    receipt.chmod(0o600)
    print("BEHAVIOR_ACTIVATION_RECEIPT",json.dumps(rec,sort_keys=True),flush=True)

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--model",required=True,choices=sorted(SOURCES))
    args=cli.parse_args()
    extract(args.model)
