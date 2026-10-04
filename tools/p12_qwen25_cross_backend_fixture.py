#!/usr/bin/env python3
"""Build real Qwen dense qNg64 n5 cross-backend fixture from safetensors."""
from __future__ import annotations
import argparse, array, hashlib, json, math, struct
from pathlib import Path

GROUP=64

def bf16_to_f32(raw: bytes):
    out=array.array("f")
    for (v,) in struct.iter_unpack("<H", raw):
        out.append(struct.unpack("<f", struct.pack("<I", v<<16))[0])
    return out

def read_tensor(path: Path, name: str):
    with path.open("rb") as f:
        hlen=struct.unpack("<Q",f.read(8))[0]
        h=json.loads(f.read(hlen))
        ent=h[name]; start,end=ent["data_offsets"]
        base=8+hlen; f.seek(base+start); raw=f.read(end-start)
    shape=ent["shape"]
    vals=bf16_to_f32(raw) if ent["dtype"]=="BF16" else array.array("f",raw)
    return vals,int(shape[0]),int(shape[1]),ent["dtype"]

def quant_planes(vals,out_dim,in_dim,n):
    ng=in_dim//GROUP; qmax=(1<<(n-1))-1; qmin=-(1<<(n-1))
    planes=bytearray(out_dim*ng*n*8); scales=array.array("f")
    deq=array.array("f",[0.0])*(out_dim*in_dim)
    for r in range(out_dim):
      for g in range(ng):
        off=r*in_dim+g*GROUP
        mx=max(abs(vals[off+p]) for p in range(GROUP))
        scale=mx/qmax if mx>1e-12 else 1.0
        scales.append(scale); inv=1.0/scale; err=0.0; codes=[]
        for p in range(GROUP):
            x=vals[off+p]+err
            q=max(qmin,min(qmax,int(round(x*inv))))
            dq=q*scale; err=x-dq; codes.append(q); deq[off+p]=dq
        po=((r*ng+g)*n)*8
        bias=1<<(n-1)
        for bit in range(n):
            for byte in range(8):
                v=0
                for k in range(8):
                    u=codes[byte*8+k]+bias
                    v |= ((u>>bit)&1)<<k
                planes[po+bit*8+byte]=v
    return planes,scales,deq

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--output-dir",required=True)
    ap.add_argument("--tensor",default="model.layers.0.self_attn.q_proj.weight"); ap.add_argument("--n",type=int,default=5)
    a=ap.parse_args(); root=Path(a.output_dir); root.mkdir(parents=True,exist_ok=True)
    vals,out_dim,in_dim,dtype=read_tensor(Path(a.checkpoint),a.tensor)
    assert in_dim%GROUP==0
    planes,scales,deq=quant_planes(vals,out_dim,in_dim,a.n)
    x=array.array("f",[math.sin((i+1)*0.173)+0.25*math.cos((i+1)*0.071) for i in range(in_dim)])
    y=array.array("f")
    for r in range(out_dim):
        off=r*in_dim
        y.append(sum(deq[off+i]*x[i] for i in range(in_dim)))
    files={"planes.bin":bytes(planes),"scales.bin":scales.tobytes(),"input.bin":x.tobytes(),"cpu_expected.bin":y.tobytes()}
    shas={}
    for name,data in files.items():
        (root/name).write_bytes(data); shas[name]=hashlib.sha256(data).hexdigest()
    manifest={"schema":"beglin-p12-qwen25-dense-cross-backend-fixture-v1","tensor_name":a.tensor,"shape":[out_dim,in_dim],
      "dtype":dtype,"n":a.n,"group_size":GROUP,"planes_bytes":len(planes),"scale_count":len(scales),
      "input_count":len(x),"output_count":len(y),"files_sha256":shas,
      "production_touched":False,"production_write_allowed":False,"automatic_live_promotion":False}
    raw=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode(); manifest["manifest_sha256"]=hashlib.sha256(raw).hexdigest()
    (root/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    print(json.dumps(manifest,sort_keys=True))
if __name__=="__main__": main()
