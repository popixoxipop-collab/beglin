#!/usr/bin/env python3
import argparse,json,struct,hashlib
from pathlib import Path
ROLES=("q_proj","k_proj","v_proj","o_proj")
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--safetensors",required=True);ap.add_argument("--output",required=True);a=ap.parse_args()
 p=Path(a.safetensors);f=p.open("rb");n=struct.unpack("<Q",f.read(8))[0];h=json.loads(f.read(n));f.close()
 rows=[]
 for name,e in h.items():
  if name=="__metadata__":continue
  for role in ROLES:
   suffix=f".self_attn.{role}.weight"
   if name.startswith("model.layers.") and name.endswith(suffix):
    layer=int(name.split(".")[2]);shape=list(map(int,e["shape"]))
    if len(shape)!=2 or shape[1]%64: raise SystemExit(f"bad shape {name}: {shape}")
    rows.append({"layer":layer,"role":role,"tensor_name":name,"shape":shape,"group_size":64,
      "groups_per_row":shape[1]//64,"cell_count":shape[0]*(shape[1]//64),"dtype":e["dtype"]})
 rows.sort(key=lambda x:(x["layer"],ROLES.index(x["role"])))
 layers=sorted({x["layer"] for x in rows});expected={(l,r) for l in layers for r in ROLES};got={(x["layer"],x["role"]) for x in rows}
 if got!=expected: raise SystemExit(f"incomplete attention inventory missing={sorted(expected-got)}")
 out={"schema":"beglin-qt-attention-inventory-v1","layer_count":len(layers),"tensor_count":len(rows),
      "total_cells":sum(x["cell_count"] for x in rows),"tensors":rows}
 raw=(json.dumps(out,sort_keys=True,indent=2)+"\n").encode();Path(a.output).write_bytes(raw)
 print(json.dumps({"status":"PASS","layer_count":out["layer_count"],"tensor_count":out["tensor_count"],
 "total_cells":out["total_cells"],"inventory_sha256":hashlib.sha256(raw).hexdigest()},sort_keys=True))
if __name__=="__main__":main()
