#!/usr/bin/env python3
import argparse,hashlib,json,struct
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--shard",required=True);ap.add_argument("--output-dir",required=True);a=ap.parse_args()
 sh=json.load(open(a.shard));p=Path(a.checkpoint);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 with p.open("rb") as f:n=struct.unpack("<Q",f.read(8))[0];h=json.loads(f.read(n));base=8+n
 rec=[]
 with p.open("rb") as f:
  for t in sh["tensors"]:
   e=h[t["tensor_name"]]
   if list(map(int,e["shape"]))!=t["shape"]:raise SystemExit("shape mismatch "+t["tensor_name"])
   s,z=e["data_offsets"];f.seek(base+s);raw=f.read(z-s)
   q=hashlib.sha256(raw).hexdigest();rec.append({"tensor_name":t["tensor_name"],"layer":t["layer"],"role":t["role"],"shape":t["shape"],"cell_count":t["cell_count"],"bytes":len(raw),"sha256":q})
 total=sum(x["cell_count"] for x in rec)
 if total!=sh["total_cells"]:raise SystemExit("coverage mismatch")
 ev={"schema":"beglin-qt-attention-materialization-v1","shard":sh["shard"],"tensor_count":len(rec),"total_cells":total,"tensors":rec}
 raw=(json.dumps(ev,sort_keys=True,indent=2)+"\n").encode();(out/"MATERIALIZATION.json").write_bytes(raw)
 print(json.dumps({"status":"PASS","shard":sh["shard"],"tensor_count":len(rec),"total_cells":total,"materialization_sha256":hashlib.sha256(raw).hexdigest()},sort_keys=True))
if __name__=="__main__":main()
