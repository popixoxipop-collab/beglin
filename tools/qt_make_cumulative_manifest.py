#!/usr/bin/env python3
import argparse,re,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--end-layer",type=int,required=True);p.add_argument("--output",required=True);a=p.parse_args()
if not 0<=a.end_layer<24: raise SystemExit("end-layer must be 0..23")
src=Path(a.source);dst=Path(a.output);shutil.rmtree(dst,ignore_errors=True);dst.mkdir(parents=True)
n=0
for f in src.glob("*.bits"):
 m=re.search(r"model_layers_(\d+)_self_attn_",f.name)
 if m and int(m.group(1))<=a.end_layer: shutil.copy2(f,dst/f.name);n+=1
exp=(a.end_layer+1)*4
if n!=exp: raise SystemExit(f"expected {exp} maps, got {n}")
print(f"QT_MANIFEST_PASS end_layer={a.end_layer} tensors={n}")
