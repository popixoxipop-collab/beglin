#!/usr/bin/env python3
from __future__ import annotations
import argparse,array,hashlib,json,math
from pathlib import Path
from qt_qwen25_observe import read_remote_tensor,DEFAULT_URL,DEFAULT_TENSOR
from p12_qwen25_cross_backend_fixture import quant_planes,GROUP

def build(vals,out_dim,in_dim,n):
    planes,scales,deq=quant_planes(vals,out_dim,in_dim,n)
    x=array.array("f",[math.sin((i+1)*0.173)+0.25*math.cos((i+1)*0.071) for i in range(in_dim)])
    y=array.array("f")
    for r in range(out_dim):
        off=r*in_dim
        y.append(sum(deq[off+i]*x[i] for i in range(in_dim)))
    return {"planes.bin":bytes(planes),"scales.bin":scales.tobytes(),"input.bin":x.tobytes(),"cpu_expected.bin":y.tobytes()}

def write_fixture(root,files,meta):
    root=Path(root); root.mkdir(parents=True,exist_ok=True); shas={}
    for name,data in files.items():
        (root/name).write_bytes(data); shas[name]=hashlib.sha256(data).hexdigest()
    m={**meta,"files_sha256":shas}
    raw=json.dumps(m,sort_keys=True,separators=(",",":")).encode()
    m["manifest_sha256"]=hashlib.sha256(raw).hexdigest()
    (root/"manifest.json").write_text(json.dumps(m,sort_keys=True,indent=2)+"\n")
    return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--url",default=DEFAULT_URL); ap.add_argument("--tensor",default=DEFAULT_TENSOR)
    ap.add_argument("--n",type=int,required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--expected-tensor-sha256")
    a=ap.parse_args()
    if a.n<3 or a.n>8: raise SystemExit("n must be in [3,8]")
    vals,out_dim,in_dim,dtype,tsha=read_remote_tensor(a.url,a.tensor)
    if a.expected_tensor_sha256 and tsha!=a.expected_tensor_sha256: raise SystemExit("tensor SHA mismatch")
    files=build(vals,out_dim,in_dim,a.n)
    m=write_fixture(a.output_dir,files,{"schema":"beglin-qt-qwen25-uniform-fixture-v1","tensor_name":a.tensor,"tensor_sha256":tsha,"shape":[out_dim,in_dim],"dtype":dtype,"n":a.n,"group_size":GROUP,"production_touched":False,"automatic_live_promotion":False})
    print(json.dumps({"status":"PASS","n":a.n,"tensor_sha256":tsha,"manifest_sha256":m["manifest_sha256"],"files_sha256":m["files_sha256"]},sort_keys=True))
if __name__=="__main__": main()
