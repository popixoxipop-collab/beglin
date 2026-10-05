#!/usr/bin/env python3
"""Emit deterministic group64 Q observations and a Q-Heatmap from real Qwen weights."""
from __future__ import annotations
import argparse,array,hashlib,json,math,struct,urllib.request
from collections import Counter
from pathlib import Path
from p12_qwen25_cross_backend_fixture import read_tensor,quant_planes,GROUP,bf16_to_f32
import qt_heatmap as qth

DEFAULT_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors"
DEFAULT_TENSOR="model.layers.0.self_attn.q_proj.weight"

def canonical_bytes(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def fetch_range(url,start,end):
    req=urllib.request.Request(url,headers={"Range":f"bytes={start}-{end}","User-Agent":"beglin-qt-qheatmap/1"})
    with urllib.request.urlopen(req,timeout=60) as r:
        data=r.read()
    expected=end-start+1
    if len(data)!=expected:
        raise RuntimeError(f"range length mismatch expected={expected} actual={len(data)}")
    return data

def read_remote_tensor(url,name):
    head=fetch_range(url,0,2*1024*1024-1)
    hlen=struct.unpack("<Q",head[:8])[0]
    if 8+hlen>len(head):
        head=fetch_range(url,0,8+hlen-1)
    meta=json.loads(head[8:8+hlen])
    ent=meta[name]; start,end=ent["data_offsets"]; base=8+hlen
    raw=fetch_range(url,base+start,base+end-1)
    shape=ent["shape"]; dtype=ent["dtype"]
    vals=bf16_to_f32(raw) if dtype=="BF16" else array.array("f",raw)
    return vals,int(shape[0]),int(shape[1]),dtype,hashlib.sha256(raw).hexdigest()

def observe(vals,out_dim,in_dim,bits,current_n=5):
    x=[math.sin((i+1)*0.173)+0.25*math.cos((i+1)*0.071) for i in range(in_dim)]
    events=[]; ng=in_dim//GROUP
    for n in bits:
        _,_,deq=quant_planes(vals,out_dim,in_dim,n)
        for r in range(out_dim):
            for g in range(ng):
                off=r*in_dim+g*GROUP
                xoff=g*GROUP
                err=abs(sum((deq[off+i]-vals[off+i])*x[xoff+i] for i in range(GROUP)))
                events.append({
                    "kind":"quant",
                    "target_key":f"qwen2.5/L0/q_proj/row={r}/group64={g}",
                    "candidate_n":n,
                    "current_n":current_n,
                    "output_max_abs_error":err,
                })
    return sorted(events,key=lambda e:(e["target_key"],e["candidate_n"]))

def build_outputs(vals,out_dim,in_dim,dtype,bits,current_n=5,error_budget=5e-4,min_samples=3,tensor_sha256=None,tensor_name=DEFAULT_TENSOR):
    bits=sorted(set(bits))
    events=observe(vals,out_dim,in_dim,bits,current_n=current_n)
    observations={
        "schema":"beglin-qt-qwen25-group64-observations-v1",
        "tensor_name":tensor_name,
        "shape":[out_dim,in_dim],
        "dtype":dtype,
        "tensor_sha256":tensor_sha256,
        "group_size":GROUP,
        "bits":bits,
        "current_n":current_n,
        "events":events,
    }
    observations["observations_sha256"]=hashlib.sha256(canonical_bytes(observations)).hexdigest()
    qmap,_=qth.project(events,generation=1,error_budget=error_budget,min_samples=min_samples)
    qmap["source_observations_sha256"]=observations["observations_sha256"]
    qmap["error_budget"]=error_budget
    qmap["minimum_samples"]=min_samples
    qmap["recommended_n_distribution"]=dict(sorted(Counter(str(c["recommended_n"]) for c in qmap["cells"]).items()))
    qmap["heatmap_sha256"]=qth.digest(qmap)
    summary={
        "schema":"beglin-qt-qwen25-qheatmap-summary-v1",
        "status":"PASS",
        "tensor_name":tensor_name,
        "shape":[out_dim,in_dim],
        "group_size":GROUP,
        "group_cells":len(qmap["cells"]),
        "event_count":len(events),
        "bits":bits,
        "current_n":current_n,
        "error_budget":error_budget,
        "tensor_sha256":tensor_sha256,
        "observations_sha256":observations["observations_sha256"],
        "heatmap_sha256":qmap["heatmap_sha256"],
        "recommended_n_distribution":qmap["recommended_n_distribution"],
        "production_touched":False,
        "automatic_live_promotion":False,
    }
    return observations,qmap,summary

def main():
    ap=argparse.ArgumentParser()
    src=ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint")
    src.add_argument("--url")
    ap.add_argument("--output",required=True)
    ap.add_argument("--qheatmap-output",required=True)
    ap.add_argument("--summary-output",required=True)
    ap.add_argument("--tensor",default=DEFAULT_TENSOR)
    ap.add_argument("--bits",nargs="+",type=int,default=[4,5,6])
    ap.add_argument("--current-n",type=int,default=5)
    ap.add_argument("--error-budget",type=float,default=5e-4)
    ap.add_argument("--min-samples",type=int,default=3)
    a=ap.parse_args()
    if a.url:
        vals,out_dim,in_dim,dtype,tensor_sha=read_remote_tensor(a.url,a.tensor)
    else:
        vals,out_dim,in_dim,dtype=read_tensor(Path(a.checkpoint),a.tensor)
        tensor_sha=None
    if in_dim%GROUP:
        raise SystemExit("input dimension must be divisible by group64")
    observations,qmap,summary=build_outputs(
        vals,out_dim,in_dim,dtype,a.bits,current_n=a.current_n,error_budget=a.error_budget,
        min_samples=a.min_samples,tensor_sha256=tensor_sha,tensor_name=a.tensor
    )
    for path,obj in [(a.output,observations),(a.qheatmap_output,qmap),(a.summary_output,summary)]:
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
    print(json.dumps(summary,sort_keys=True))

if __name__=="__main__":
    main()
