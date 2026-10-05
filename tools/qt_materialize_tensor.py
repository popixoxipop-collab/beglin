#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,struct,urllib.request,hashlib
from pathlib import Path
URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors"
TENSOR="model.layers.0.self_attn.q_proj.weight"
EXPECTED="7af566022f07072d911a22135d8835121b1fe44e05e2d9a009c1263cba98fdef"
def get(a,b):
 r=urllib.request.Request(URL,headers={"Range":f"bytes={a}-{b}","User-Agent":"beglin-qt-fixture/2"});d=urllib.request.urlopen(r,timeout=60).read()
 if len(d)!=b-a+1:raise SystemExit("RANGE_LENGTH_MISMATCH")
 return d
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args()
 h=get(0,2*1024*1024-1);n=struct.unpack("<Q",h[:8])[0]
 if 8+n>len(h):h=get(0,8+n-1)
 m=json.loads(h[8:8+n]);e=m[TENSOR];s,z=e["data_offsets"];raw=get(8+n+s,8+n+z-1)
 if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise SystemExit("TENSOR_SHA_MISMATCH")
 hdr=json.dumps({TENSOR:{"dtype":e["dtype"],"shape":e["shape"],"data_offsets":[0,len(raw)]}},separators=(",",":")).encode();hdr+=b" "*((-len(hdr)-8)%8)
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+".tmp");tmp.write_bytes(struct.pack("<Q",len(hdr))+hdr+raw);tmp.replace(out)
 print(json.dumps({"status":"PASS","tensor_sha256":EXPECTED,"output":str(out),"bytes":len(raw)},sort_keys=True))
if __name__=="__main__":main()
