#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
EXPECTED_SCHEMA="beglin-q-incumbent-map-v1"
EXPECTED_SHA256="a87ed7d350b1152545c46ad25c662e7560008727928624eab4b3241e17c951ba"
EXPECTED_TENSOR="model.layers.0.self_attn.q_proj.weight"
EXPECTED_OUT=896;EXPECTED_IN=896;GROUP=64;EXPECTED_CELLS=12544
class IncumbentMapError(RuntimeError): pass
def load_incumbent(path:str|Path,*,tensor_name:str,out_dim:int,in_dim:int)->dict:
 p=Path(path).expanduser().resolve(strict=True);raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
 if sha!=EXPECTED_SHA256: raise IncumbentMapError(f"incumbent SHA mismatch: {sha}")
 x=json.loads(raw)
 if x.get("schema")!=EXPECTED_SCHEMA: raise IncumbentMapError("schema mismatch")
 if tensor_name!=EXPECTED_TENSOR: raise IncumbentMapError("tensor identity mismatch")
 if (out_dim,in_dim)!=(EXPECTED_OUT,EXPECTED_IN): raise IncumbentMapError("shape mismatch")
 bits=x.get("bits")
 if not isinstance(bits,dict) or len(bits)!=EXPECTED_CELLS: raise IncumbentMapError("cell count mismatch")
 ng=in_dim//GROUP;ordered=[]
 for r in range(out_dim):
  for g in range(ng):
   k=f"{r}:{g}"
   if k not in bits: raise IncumbentMapError(f"missing cell {k}")
   n=int(bits[k])
   if n not in (4,5,6): raise IncumbentMapError(f"unsupported n={n} at {k}")
   ordered.append(n)
 if x.get("automatic_live_promotion") is not False: raise IncumbentMapError("automatic promotion must be false")
 return {"schema":"beglin-q-incumbent-runtime-binding-v1","incumbent_sha256":sha,"tensor_name":tensor_name,"shape":[out_dim,in_dim],"group_size":GROUP,"cell_count":len(ordered),"bits":ordered,"average_bits":sum(ordered)/len(ordered),"production_write_allowed":False,"automatic_live_promotion":False}
