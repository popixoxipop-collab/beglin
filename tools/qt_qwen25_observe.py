#!/usr/bin/env python3
"""Emit deterministic group64 Q observations from a real safetensors tensor."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from p12_qwen25_cross_backend_fixture import read_tensor,quant_planes,GROUP

def canonical_bytes(obj): return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def observe(vals,out_dim,in_dim,bits):
    x=[math.sin((i+1)*0.173)+0.25*math.cos((i+1)*0.071) for i in range(in_dim)]
    events=[]; ng=in_dim//GROUP
    for n in bits:
        _,_,deq=quant_planes(vals,out_dim,in_dim,n)
        for r in range(out_dim):
            for g in range(ng):
                off=r*in_dim+g*GROUP
                err=abs(sum((deq[off+i]-vals[off+i])*x[g*GROUP+i] for i in range(GROUP)))
                events.append({"kind":"quant","target_key":f"qwen2.5/L0/q_proj/row={r}/group64={g}","candidate_n":n,"current_n":max(bits),"output_max_abs_error":err})
    return sorted(events,key=lambda e:(e["target_key"],e["candidate_n"]))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--tensor",default="model.layers.0.self_attn.q_proj.weight"); ap.add_argument("--bits",nargs="+",type=int,default=[4,5,6])
    a=ap.parse_args(); vals,out_dim,in_dim,dtype=read_tensor(Path(a.checkpoint),a.tensor)
    if in_dim%GROUP: raise SystemExit("input dimension must be divisible by group64")
    events=observe(vals,out_dim,in_dim,sorted(set(a.bits)))
    payload={"schema":"beglin-qt-qwen25-group64-observations-v1","tensor_name":a.tensor,"shape":[out_dim,in_dim],"dtype":dtype,"group_size":GROUP,"bits":sorted(set(a.bits)),"events":events}
    payload["observations_sha256"]=hashlib.sha256(canonical_bytes(payload)).hexdigest()
    Path(a.output).write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"PASS","event_count":len(events),"observations_sha256":payload["observations_sha256"]},sort_keys=True))
if __name__=="__main__": main()
