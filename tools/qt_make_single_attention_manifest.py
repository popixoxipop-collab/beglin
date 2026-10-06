#!/usr/bin/env python3
import argparse,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--layer",type=int,required=True);p.add_argument("--role",choices=["q_proj","k_proj","v_proj","o_proj"],required=True);p.add_argument("--output",required=True);a=p.parse_args()
src=Path(a.source);dst=Path(a.output);shutil.rmtree(dst,ignore_errors=True);dst.mkdir(parents=True)
name=f"model_layers_{a.layer}_self_attn_{a.role}_weight.bits";f=src/name
if not f.is_file(): raise SystemExit(f"missing map {f}")
shutil.copy2(f,dst/name)
print(f"QT_SINGLE_MANIFEST_PASS layer={a.layer} role={a.role}")
