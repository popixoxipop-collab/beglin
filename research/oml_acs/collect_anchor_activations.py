#!/usr/bin/env python3
"""Collect real BF16 residual-stream anchors from pinned cached checkpoints.

Input: private G: 300-anchor JSONL, no labels, no generation.
Output: local .npy float32 activations + .json receipt; no prompt text in Git.
Fails closed on source/model mismatch and on any pre-existing different output.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, time
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS={
"qwen":{"id":"Qwen/Qwen2.5-1.5B-Instruct","cache":"models--Qwen--Qwen2.5-1.5B-Instruct",
"rev":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306","layer":14,"dim":1536,"layers":28},
"mistral":{"id":"mistralai/Mistral-7B-Instruct-v0.3","cache":"models--mistralai--Mistral-7B-Instruct-v0.3",
"rev":"c170c708c41dac9275d15a8fff4eca08d52bab71","layer":16,"dim":4096,"layers":32}
}
BASE=pathlib.Path("/mnt/g/TailnetCommander/work/oml_acs_20261021_anchors")
INPUT_SHA="47d5ddcb37b81212e3f43a34441662062b42a038d4cc606d9d7081c31131f83b"
def hash_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for buf in iter(lambda:f.read(1024*1024),b""):
            h.update(buf)
    return h.hexdigest()

def load_inputs():
    path=BASE/"anchors_300_local.jsonl"
    if not path.is_file() or hash_file(path)!=INPUT_SHA:
        raise RuntimeError("raw 300-anchor identity changed; stop")
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    meta=json.loads((BASE/"anchor_manifest_public_hashes_only.json").read_text())
    if len(rows)!=300 or meta.get("total")!=300 or not meta.get("quota_ok") or not meta.get("unique_text"):
        raise RuntimeError("incomplete original anchor manifest")
    for idx,(row,public_row) in enumerate(zip(rows,meta["ids"])):
        if hashlib.sha256(row["text"].encode()).hexdigest()!=row["text_sha256"]:
            raise RuntimeError(f"anchor text identity invalid at {idx}")
        if row["text_sha256"]!=public_row["text_sha256"] or not 10<=len(row["text"])<=300:
            raise RuntimeError(f"anchor order mismatch or invalid prompt {idx}")
    return rows

def load_model(name):
    cfg=MODELS[name]
    p=pathlib.Path.home()/".cache/huggingface/hub"/cfg["cache"]/"snapshots"/cfg["rev"]
    if not p.is_dir() or not (p/"config.json").is_file() or not list(p.glob("*.safetensors")):
        raise RuntimeError("required fully cached pinned checkpoint absent: "+str(p))
    tk=AutoTokenizer.from_pretrained(str(p),local_files_only=True,trust_remote_code=False)
    kw=dict(local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16)
    if name=="mistral":
        kw.update(device_map="auto",max_memory={0:"12GiB","cpu":"20GiB"},low_cpu_mem_usage=True)
        m=AutoModelForCausalLM.from_pretrained(str(p),**kw).eval()
    else:
        m=AutoModelForCausalLM.from_pretrained(str(p),**kw).to("cuda").eval()
    if len(m.model.layers)!=cfg["layers"] or m.config.hidden_size!=cfg["dim"]:
        raise RuntimeError("model topology differs from pinned research configuration")
    return cfg,tk,m,m.model.layers[cfg["layer"]]

def capture(tk,m,block,text,expected_dim):
    got=[]
    def read(_module,_args,out):
        hidden=out[0] if isinstance(out,tuple) else out
        got.append(hidden[0,-1,:].detach().float().cpu().numpy().copy())
    hook=block.register_forward_hook(read)
    try:
        batch=tk.apply_chat_template(
            [{"role":"user","content":text}],add_generation_prompt=True,
            return_tensors="pt",return_dict=True)
        ids=batch["input_ids"].to(m.device)
        mask=batch.get("attention_mask",torch.ones_like(ids)).to(m.device)
        with torch.inference_mode():
            m(input_ids=ids,attention_mask=mask,use_cache=False)
    finally:
        hook.remove()
    if len(got)!=1 or got[0].shape!=(expected_dim,) or not np.isfinite(got[0]).all():
        raise RuntimeError("invalid hidden state/hook count")
    return got[0]

def save_unique(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if path.exists():
        if hash_file(path)==hashlib.sha256(payload).hexdigest():
            print("UNCHANGED_EXISTING",str(path),flush=True)
            return
        raise RuntimeError("existing artifact SHA differs; refuse rewrite: "+str(path))
    tmp=path.with_name(path.name+".tmp")
    fd=os.open(str(tmp),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,"wb") as f:
        f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(str(tmp),str(path));os.chmod(path,0o600)

def run(name):
    start=time.perf_counter()
    rows=load_inputs()
    cfg,tk,model,block=load_model(name)
    print("MODEL_LOADED",name,cfg["rev"],"at",round(time.perf_counter()-start,1),flush=True)
    features=[]
    for i,row in enumerate(rows):
        features.append(capture(tk,model,block,row["text"],cfg["dim"]))
        if (i+1)%25==0:
            print("PROGRESS",name,i+1,round(time.perf_counter()-start,1),flush=True)
    a=np.stack(features).astype(np.float32)
    mu=a.mean(axis=0,dtype=np.float64)
    c=a.astype(np.float64)-mu
    norms=np.linalg.norm(c,axis=1)
    if a.shape!=(300,cfg["dim"]) or np.min(norms)<=1e-10:
        raise RuntimeError("bad matrix shape/degenerate anchor")
    normalized=(c/norms[:,None]).astype(np.float32)
    # Verify first example readout is deterministic within the same loaded model.
    replay=capture(tk,model,block,rows[0]["text"],cfg["dim"])
    replay_max_abs=float(np.max(np.abs(replay-a[0])))
    if replay_max_abs>1e-3: raise RuntimeError("replay deviates from first capture")
    import io
    def arr_bytes(obj):
        z=io.BytesIO();np.save(z,obj,allow_pickle=False);return z.getvalue()
    root=BASE/"features"
    stem=name+"_"+cfg["rev"][:12]+"_layer"+str(cfg["layer"])
    raw=arr_bytes(a);unit=arr_bytes(normalized)
    save_unique(root/(stem+"_native.npy"),raw)
    save_unique(root/(stem+"_normalized_acs_rows.npy"),unit)
    rec={"schema":"oml-acs-anchor-activation-capture/1",
         "status":"ACTIVATIONS_ONLY_PASS","model":cfg["id"],"revision":cfg["rev"],
         "precision":"BF16","zero_based_layer":cfg["layer"],"dtype_saved":"float32",
         "input_sha256":INPUT_SHA,"anchor_count":300,
         "shape":list(a.shape),"normalized_shape":list(normalized.shape),
         "max_replay_abs_diff":replay_max_abs,"native_npy_sha256":hashlib.sha256(raw).hexdigest(),
         "normalized_npy_sha256":hashlib.sha256(unit).hexdigest(),
         "gpu":torch.cuda.get_device_name(0),"peak_cuda_mb":round(torch.cuda.max_memory_allocated()/2**20,2),
         "elapsed_sec":round(time.perf_counter()-start,2),
         "held_out_behavior_labels_used":False,"model_intervention_done":False,
         "scientific_verdict":"NONE","original_author_anchor_ids_identical":"UNKNOWN",
         "V6_holdout_exclusion_certified":False}
    data=(json.dumps(rec,sort_keys=True,indent=2)+"\n").encode()
    save_unique(root/(stem+"_receipt.json"),data)
    print("ANCHOR_CAPTURE_RECEIPT",json.dumps(rec,sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=list(MODELS),required=True)
    run(p.parse_args().model)
