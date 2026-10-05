#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
def wilson(k,n,z=1.96):
 p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return max(0,c-h),min(1,c+h)
def main():
 p=argparse.ArgumentParser();p.add_argument("--observations",required=True);p.add_argument("--output",required=True);p.add_argument("--approve-lcb",type=float,default=.95);a=p.parse_args();o=json.load(open(a.observations));rows=[]
 for x in o["rows"]:
  lo,hi=wilson(x["safe_count"],x["contexts"])
  decision="APPROVE" if lo>=a.approve_lcb else "FALLBACK" if hi<.5 else "MORE_EVIDENCE"
  rows.append({**x,"safe_ci95":[lo,hi],"decision":decision})
 dist={k:sum(x["decision"]==k for x in rows) for k in ["APPROVE","MORE_EVIDENCE","FALLBACK"]}
 out={"schema":"beglin-qt-calibrated-beval-v1","approve_lcb":a.approve_lcb,"decision_distribution":dist,"rows":rows,"automatic_live_promotion":False};Path(a.output).write_text(json.dumps(out,sort_keys=True,indent=2)+"\n");print(json.dumps({"status":"PASS","decision_distribution":dist},sort_keys=True))
if __name__=="__main__":main()
