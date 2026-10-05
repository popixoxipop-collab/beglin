#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,re
from collections import Counter
from pathlib import Path

CELL_RE=re.compile(r"/group64=(\d+)$")

def canon(o):
    return json.dumps(o,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def build(qmap,summary):
    if qmap.get("schema")!="beglin-qheatmap-v2": raise ValueError("bad qheatmap schema")
    if summary.get("schema")!="beglin-qt-qwen25-qheatmap-summary-v1": raise ValueError("bad summary schema")
    if qmap.get("heatmap_sha256")!=summary.get("heatmap_sha256"): raise ValueError("heatmap identity mismatch")
    current=int(summary["current_n"]); cells=[]
    for c in sorted(qmap["cells"],key=lambda x:x["target_key"]):
        m=CELL_RE.search(c["target_key"])
        if not m: raise ValueError("non-group64 target")
        n=int(c["recommended_n"]); err=float(c["error_by_n"][str(n)])
        cells.append({"target_key":c["target_key"],"qgroup_index":int(m.group(1)),"before_n":current,"proposed_n":n,"expected_error":err,"confidence":float(c["confidence"])})
    dist=dict(sorted(Counter(str(c["proposed_n"]) for c in cells).items()))
    base={"schema":"beglin-qpolicy-candidate-v1","source_heatmap_sha256":summary["heatmap_sha256"],"tensor_name":summary["tensor_name"],"tensor_sha256":summary["tensor_sha256"],"generation":int(qmap["generation"]),"current_n":current,"cells":cells,"distribution":dist,"approved_by_beval":False,"production_write_allowed":False,"automatic_live_promotion":False}
    sha=hashlib.sha256(canon(base)).hexdigest()
    base["policy_id"]="qpc_"+sha[:16]; base["candidate_sha256"]=sha
    return base

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--qheatmap",required=True); ap.add_argument("--summary",required=True); ap.add_argument("--output",required=True)
    a=ap.parse_args(); q=json.load(open(a.qheatmap)); s=json.load(open(a.summary)); out=build(q,s)
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
    transitions=Counter(f'{c["before_n"]}->{c["proposed_n"]}' for c in out["cells"])
    print(json.dumps({"status":"PASS","policy_id":out["policy_id"],"candidate_sha256":out["candidate_sha256"],"cells":len(out["cells"]),"distribution":out["distribution"],"transitions":dict(sorted(transitions.items())),"approved_by_beval":False},sort_keys=True))
if __name__=="__main__": main()
