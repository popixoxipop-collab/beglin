#!/usr/bin/env python3
"""G0 development-anchor activation capture (not evaluation).
Read local SHA-locked 300-row private dataset; write *only* private NumPy
activation matrix and non-content aggregate receipt. BF16 Qwen, no downloads,
no labels, no model generation and no operational mutations.
"""
from __future__ import annotations
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json, hashlib, time, pathlib, collections, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
EXPECTED_RAW = "47d5ddcb37b81212e3f43a34441662062b42a038d4cc606d9d7081c31131f83b"
BASE = pathlib.Path.home()/".local/share/beglin/oml_acs/20261021"
SOURCE = BASE/"anchors_300_local.jsonl"
SNAP = pathlib.Path.home()/".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"/REV
OUT = BASE/"qwen25_15layer_300_anchors_bf16.npz"
REPORT = BASE/"qwen25_15layer_300_anchors_bf16_receipt.json"
def sha(value:bytes)->str:
    return hashlib.sha256(value).hexdigest()
def main():
    started=time.perf_counter()
    assert SOURCE.is_file() and (SOURCE.stat().st_mode&0o777)==0o600, "private dataset not secured"
    blob=SOURCE.read_bytes()
    assert sha(blob)==EXPECTED_RAW, "data identity drift"
    rows=[json.loads(line) for line in blob.decode("utf-8").splitlines() if line]
    assert len(rows)==300 and len(set(r["text_sha256"] for r in rows))==300
    counts=collections.Counter(r["scenario"] for r in rows)
    assert len(counts)==15 and all(n==20 for n in counts.values())
    for r in rows:
        assert sha(r["text"].encode())==r["text_sha256"], "sample SHA mismatch"
        assert 10<=len(r["text"])<=300
    assert not OUT.exists() and not REPORT.exists(), "refuse to overwrite existing measurement"
    assert (SNAP/"model.safetensors").is_file() and (SNAP/"config.json").is_file(), "pinned checkpoint unavailable"
    os.umask(0o077)
    tok=AutoTokenizer.from_pretrained(str(SNAP),local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(str(SNAP),local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16).to("cuda").eval()
    assert len(model.model.layers)==28 and model.config.hidden_size==1536
    torch.cuda.reset_peak_memory_stats()
    captures=[]
    def capture(_layer,_args,out):
        h=out[0] if isinstance(out,tuple) else out
        assert h.ndim==3 and h.shape[-1]==1536
        captures.append(h[0,-1,:].detach().float().cpu().numpy().copy())
    handle=model.model.layers[14].register_forward_hook(capture)
    activations=[]
    try:
        with torch.inference_mode():
            for i,row in enumerate(rows):
                inputs=tok.apply_chat_template([{"role":"user","content":row["text"]}],tokenize=True,add_generation_prompt=True,return_tensors="pt",return_dict=True)
                inp={k:v.to("cuda") for k,v in inputs.items() if k in {"input_ids","attention_mask"}}
                if "attention_mask" not in inp: inp["attention_mask"]=torch.ones_like(inp["input_ids"])
                captures.clear()
                model(**inp,use_cache=False)
                if len(captures)!=1:raise RuntimeError("unexpected model hook count")
                activations.append(captures[0])
                if (i+1)%50==0:
                    print("ACTIVATION_PROGRESS",i+1,"of",len(rows),"elapsed",round(time.perf_counter()-started,2),flush=True)
    finally:
        handle.remove()
    M=np.asarray(activations,dtype=np.float32)
    assert M.shape==(300,1536) and np.isfinite(M).all()
    centered=M.astype(np.float64)-M.astype(np.float64).mean(axis=0)
    norms=np.linalg.norm(centered,axis=1)
    assert np.isfinite(norms).all() and (norms>1e-10).all()
    activation_sha=sha(M.tobytes(order="C"))
    temp=OUT.with_name(OUT.name+".tmp")
    with temp.open("xb") as f:
        np.savez_compressed(f,activations=M,text_sha256=np.asarray([r["text_sha256"] for r in rows]),scenario=np.asarray([r["scenario"] for r in rows]),model_revision=np.asarray(REV),source_root_sha256=np.asarray(EXPECTED_RAW))
        f.flush();os.fsync(f.fileno())
    temp.replace(OUT);OUT.chmod(0o600)
    report={
        "schema":"oml-acs-300-anchor-activation-receipt/1","status":"PASS_TECHNICAL_EXTRACTION",
        "scientific_hypothesis_verdict":"NONE","source_raw_sha256":EXPECTED_RAW,
        "model":MODEL,"model_revision":REV,"precision":"BF16","layer_index_zero_based":14,
        "shape":[300,1536],"scenarios":dict(sorted(counts.items())),
        "activation_array_sha256":activation_sha,
        "stored_npz_sha256":sha(OUT.read_bytes()),"stored_npz_bytes":OUT.stat().st_size,
        "mean_centered_row_norm_min":float(norms.min()),"mean_centered_row_norm_max":float(norms.max()),
        "all_finite":True,"gpu":torch.cuda.get_device_name(0),
        "peak_cuda_allocated_mb":round(torch.cuda.max_memory_allocated()/1048576,2),
        "elapsed_seconds":round(time.perf_counter()-started,2),
        "final_dataset_frozen":False,"v6_consumed_24_excluded":False,
        "raw_text_in_report":False,"labels_used":False,"target_model_evaluation":False,
        "production_touched":False
    }
    with REPORT.open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2,ensure_ascii=True);f.write("\n");f.flush();os.fsync(f.fileno())
    REPORT.chmod(0o600)
    print("G0_300_CAPTURE_RECEIPT",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    main()
