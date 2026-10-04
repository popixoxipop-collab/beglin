#!/usr/bin/env python3
"""P12 real-Qwen dense qNg64 MLX probe manifest builder.

Produces the exact isolated probe contract consumed by the XOX MLX runner.
It intentionally contains no production paths and binds one Qwen2.5 tensor.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

SCHEMA="beglin-p12-qwen25-dense-mlx-probe-v1"

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def build(*, checkpoint: str, tensor: str, n: int, output: str) -> dict:
    p=Path(checkpoint).expanduser().resolve()
    if "/vdsp_serving/" in str(p):
        raise SystemExit("production path refused")
    if n not in {2,3,5,6,7,9,10,11,12,13,14,15}:
        raise SystemExit("unsupported qNg64 precision")
    out={
      "schema":SCHEMA,
      "model":"Qwen/Qwen2.5-0.5B-Instruct",
      "checkpoint":str(p),
      "checkpoint_file_sha256":sha256(p),
      "tensor_name":tensor,
      "target_key":"checkpoint-538528b2cf263385/L0/q_proj",
      "backend":"mlx_metal",
      "candidate_n":n,
      "group_size":64,
      "probe_steps":[
        "register_real_qng64_from_safetensors",
        "bind_mlx_gpu_af",
        "verify_binding_identity",
        "metal_matvec",
        "cpu_oracle_compare",
        "snapshot_restore",
        "repeat_metal_matvec"
      ],
      "production_touched":False,
      "production_write_allowed":False,
      "automatic_live_promotion":False
    }
    raw=json.dumps(out,sort_keys=True,separators=(",",":")).encode()
    out["manifest_sha256"]=hashlib.sha256(raw).hexdigest()
    Path(output).write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--tensor",default="model.layers.0.self_attn.q_proj.weight")
    ap.add_argument("--n",type=int,default=5)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    print(json.dumps(build(checkpoint=a.checkpoint,tensor=a.tensor,n=a.n,output=a.output),sort_keys=True))
if __name__=="__main__": main()
