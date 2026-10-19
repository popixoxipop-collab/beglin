#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--output",required=True);a=p.parse_args()
rows=[]
for end in (0,3,7,11,15,19,23):
 f=Path(a.root)/f"beval-L{end}.json"
 x=json.load(open(f))
 rows.append({"end_layer":end,"tensor_count":(end+1)*4,"status":x["status"],"max_relative_l2":x["max_relative_l2"],"margin":x["margin"],"candidate_sha256":x["candidate_sha256"]})
Path(a.output).write_text(json.dumps({"schema":"beglin-qt-full-attention-sweep-v1","rows":rows},sort_keys=True,indent=2)+"\n")
print(json.dumps(rows,sort_keys=True))
