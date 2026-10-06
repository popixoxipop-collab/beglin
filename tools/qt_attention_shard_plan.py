#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--inventory",required=True);ap.add_argument("--shards",type=int,default=8);ap.add_argument("--output-dir",required=True);a=ap.parse_args()
 inv=json.load(open(a.inventory)); n=a.shards
 if n<1 or n>32: raise SystemExit("shards must be 1..32")
 bins=[{"shard":i,"total_cells":0,"tensors":[]} for i in range(n)]
 # deterministic LPT: large tensors first, stable layer/role tie-break
 ts=sorted(inv["tensors"],key=lambda x:(-x["cell_count"],x["layer"],x["role"]))
 for t in ts:
  b=min(bins,key=lambda x:(x["total_cells"],x["shard"]));b["tensors"].append(t);b["total_cells"]+=t["cell_count"]
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 manifest={"schema":"beglin-qt-attention-shard-plan-v1","inventory_sha256":hashlib.sha256(Path(a.inventory).read_bytes()).hexdigest(),"shard_count":n,"total_cells":sum(b["total_cells"] for b in bins),"shards":[]}
 for b in bins:
  raw=(json.dumps(b,sort_keys=True,indent=2)+"\n").encode();p=out/f"shard-{b['shard']:02d}.json";p.write_bytes(raw)
  manifest["shards"].append({"shard":b["shard"],"tensor_count":len(b["tensors"]),"total_cells":b["total_cells"],"sha256":hashlib.sha256(raw).hexdigest(),"path":p.name})
 raw=(json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode();(out/"SHARD_PLAN.json").write_bytes(raw)
 print(json.dumps({"status":"PASS","shards":n,"total_cells":manifest["total_cells"],"min_cells":min(b["total_cells"] for b in bins),"max_cells":max(b["total_cells"] for b in bins),"plan_sha256":hashlib.sha256(raw).hexdigest()},sort_keys=True))
if __name__=="__main__":main()
