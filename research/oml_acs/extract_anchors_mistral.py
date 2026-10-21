#!/usr/bin/env python3
"""G0 cross-family development-anchor capture, pinned 300 rows: Mistral 7B BF16.

No generation/steering, no final labels or final split. Raw texts and both
activation matrices remain private on WSL ext4; GitHub receives receipts only.
Unlike source code hashes, this script's product SHA is recorded at runtime.
"""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import collections
import hashlib
import json
import pathlib
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from acs_core import AnchorProjector

SOURCE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SOURCE_REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TARGET_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TARGET_REV = "c170c708c41dac9275d15a8fff4eca08d52bab71"
DATA_SHA = "47d5ddcb37b81212e3f43a34441662062b42a038d4cc606d9d7081c31131f83b"
LAYER = 16
BASE = pathlib.Path.home() / ".local/share/beglin/oml_acs/20261021"
DATA = BASE / "anchors_300_local.jsonl"
QWEN = BASE / "qwen25_15layer_300_anchors_bf16.npz"
QWEN_RECEIPT = BASE / "qwen25_15layer_300_anchors_bf16_receipt.json"
MODEL = pathlib.Path.home() / ".cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots" / TARGET_REV
CAPTURE = BASE / "mistral7b_layer16_300_anchors_bf16.npz"
PROJECTOR = BASE / "mistral7b_layer16_300_projector_bf16.npz"
RECEIPT = BASE / "mistral7b_layer16_300_anchors_bf16_receipt.json"

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def write_npz_atomic(path: pathlib.Path, **kwargs) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("xb") as stream:
        np.savez_compressed(stream, **kwargs)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    path.chmod(0o600)

def main():
    began = time.perf_counter()
    assert DATA.is_file() and (DATA.stat().st_mode & 0o777) == 0o600, "unsecured private source"
    raw = DATA.read_bytes()
    assert sha(raw) == DATA_SHA, "source SHA mismatch"
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    assert len(rows) == 300, "not exactly 300 source rows"
    hashes = [r["text_sha256"] for r in rows]
    scenarios = [r["scenario"] for r in rows]
    assert len(set(hashes)) == 300
    assert len(set(scenarios)) == 15
    assert all(v == 20 for v in collections.Counter(scenarios).values())
    assert all(sha(r["text"].encode("utf-8")) == r["text_sha256"] for r in rows), "per-row hash mismatch"
    assert all(10 <= len(r["text"]) <= 300 for r in rows)

    assert QWEN.is_file() and QWEN_RECEIPT.is_file(), "source-model matrix missing"
    qrec = json.loads(QWEN_RECEIPT.read_text())
    assert qrec["model_revision"] == SOURCE_REV and qrec["source_raw_sha256"] == DATA_SHA
    assert sha(QWEN.read_bytes()) == qrec["stored_npz_sha256"]
    with np.load(QWEN, allow_pickle=False) as q:
        assert q["activations"].shape == (300, 1536)
        assert q["text_sha256"].tolist() == hashes, "source Qwen row identity/order mismatch"
        assert q["scenario"].tolist() == scenarios, "source Qwen scenario/order mismatch"
        assert sha(q["activations"].tobytes()) == qrec["activation_array_sha256"]
    for file in (CAPTURE, PROJECTOR, RECEIPT):
        assert not file.exists(), "would overwrite earlier result: " + str(file)
    assert MODEL.name == TARGET_REV and (MODEL / "config.json").is_file()
    assert len(list(MODEL.glob("*.safetensors"))) == 3, "incomplete cached Mistral safetensors"

    os.umask(0o077)
    assert torch.cuda.is_available(), "CUDA unavailable"
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.bfloat16, device_map="auto", max_memory={0:"12GiB", "cpu":"20GiB"},
        low_cpu_mem_usage=True
    ).eval()
    if model.config.hidden_size != 4096 or len(model.model.layers) != 32:
        raise RuntimeError("unexpected target architecture or layer count")
    print("MODEL_READY", round(time.perf_counter() - began, 2), "seconds",
          "GPU",torch.cuda.get_device_name(0),flush=True)

    activations=[]
    observed=[]
    def capture(_module, _inp, output):
        h=output[0] if isinstance(output,tuple) else output
        if h.ndim != 3 or h.shape[-1] != 4096:
            raise RuntimeError("unexpected target activation shape")
        observed.append(h[0,-1,:].detach().float().cpu().numpy().copy())

    hook=model.model.layers[LAYER].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for i,row in enumerate(rows):
                tokenized=tokenizer.apply_chat_template(
                    [{"role":"user","content":row["text"]}],tokenize=True,
                    add_generation_prompt=True,return_tensors="pt",return_dict=True)
                inputs={k:v.to(model.device) for k,v in tokenized.items()
                        if k in {"input_ids","attention_mask"}}
                if "attention_mask" not in inputs:
                    inputs["attention_mask"]=torch.ones_like(inputs["input_ids"])
                observed.clear()
                model(**inputs, use_cache=False)
                if len(observed)!=1:
                    raise RuntimeError("expected exactly one selected-layer activation hook")
                activations.append(observed[0])
                if (i+1)%50==0:
                    print("ACTIVATION_PROGRESS",i+1,"of 300",
                          "elapsed",round(time.perf_counter()-began,2),flush=True)
    finally:
        hook.remove()

    array = np.asarray(activations, dtype=np.float32)
    assert array.shape==(300,4096) and np.isfinite(array).all()
    projector=AnchorProjector.fit(array)
    assert projector.ahat.shape==(300,4096)
    norms=np.linalg.norm(projector.ahat,axis=1)
    assert np.isfinite(norms).all() and np.allclose(norms,1.0,atol=1e-12)
    gram=projector.ahat@projector.ahat.T
    eig=np.maximum(np.linalg.eigvalsh(gram),0)
    positive=eig[eig>1e-12]
    p=positive/positive.sum()
    entropy_rank=float(np.exp(-np.sum(p*np.log(p))))
    rank=int(np.sum(eig>1e-6))

    arr_sha=sha(array.tobytes(order="C"))
    write_npz_atomic(
        CAPTURE, activations=array, text_sha256=np.asarray(hashes),
        scenario=np.asarray(scenarios), source_raw_sha256=np.asarray(DATA_SHA),
        model_revision=np.asarray(TARGET_REV), layer_index_zero_based=np.asarray(LAYER))
    write_npz_atomic(
        PROJECTOR, mu=projector.mu, ahat=projector.ahat, model_revision=np.asarray(TARGET_REV),
        source_raw_sha256=np.asarray(DATA_SHA), activation_array_sha256=np.asarray(arr_sha),
        layer_index_zero_based=np.asarray(LAYER))
    assert (CAPTURE.stat().st_mode & 0o777)==0o600
    assert (PROJECTOR.stat().st_mode & 0o777)==0o600

    torch.cuda.synchronize()
    receipt = {
        "schema":"oml-acs-target-300-activation/1",
        "status":"PASS_TECHNICAL_EXTRACTION", "scientific_verdict":"NO_VERDICT",
        "source_anchors_sha256":DATA_SHA,
        "source_model":SOURCE_MODEL,"source_revision":SOURCE_REV,
        "source_qwen_array_sha256":qrec["activation_array_sha256"],
        "target_model":TARGET_MODEL,"target_revision":TARGET_REV,
        "precision":"BF16","layer_index_zero_based":LAYER,
        "target_activation_shape":[300,4096], "rows_same_order_as_qwen":True,
        "target_activation_array_sha256":arr_sha,
        "capture_npz_sha256":sha(CAPTURE.read_bytes()),
        "capture_npz_size_bytes":CAPTURE.stat().st_size,
        "target_projector_shape":[300,4096],
        "target_projector_npz_sha256":sha(PROJECTOR.read_bytes()),
        "target_projector_npz_size_bytes":PROJECTOR.stat().st_size,
        "all_finite":True,"projector_row_norm_min":float(norms.min()),
        "projector_row_norm_max":float(norms.max()),
        "projector_numeric_rank_tolerance_1e6":rank,
        "projector_effective_rank_entropy":round(entropy_rank,4),
        "peak_cuda_allocated_mb":round(torch.cuda.max_memory_allocated()/1048576,2),
        "run_elapsed_seconds":round(time.perf_counter()-began,2),
        "device":torch.cuda.get_device_name(0),
        "labels_used":False,"original_author_300_same_ids":False,
        "final_eval_opened":False,"old_V6_24_attested_excluded":False,
        "source_raw_text_published":False,"production_touched":False,
    }
    with RECEIPT.open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    RECEIPT.chmod(0o600)
    print("MISTRAL_300_RECEIPT",json.dumps(receipt,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
