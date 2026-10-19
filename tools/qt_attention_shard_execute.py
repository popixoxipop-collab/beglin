#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--shard",required=True);ap.add_argument("--checkpoint",required=True);ap.add_argument("--output",required=True);ap.add_argument("--dry-run",action="store_true");a=ap.parse_args()
 sh=json.load(open(a.shard)); out=Path(a.output);out.mkdir(parents=True,exist_ok=True); results=[]
 for t in sh["tensors"]:
  rec={k:t[k] for k in ("layer","role","tensor_name","shape","cell_count","groups_per_row")}
  rec["status"]="PLANNED" if a.dry_run else "READY"
  results.append(rec)
 total=sum(x["cell_count"] for x in results)
 if total!=sh["total_cells"] or len(results)!=len(sh["tensors"]):raise SystemExit("shard coverage mismatch")
 ev={"schema":"beglin-qt-attention-shard-execution-v1","shard":sh["shard"],"tensor_count":len(results),"total_cells":total,"checkpoint_sha256":sha(a.checkpoint),"dry_run":a.dry_run,"tensors":results}
 raw=(json.dumps(ev,sort_keys=True,indent=2)+"\n").encode();p=out/"EXECUTION.json";p.write_bytes(raw)
 print(json.dumps({"status":"PASS","shard":sh["shard"],"tensor_count":len(results),"total_cells":total,"execution_sha256":hashlib.sha256(raw).hexdigest(),"dry_run":a.dry_run},sort_keys=True))
if __name__=="__main__":main()
