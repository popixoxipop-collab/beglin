#!/usr/bin/env python3
import argparse,array,json,math
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--baseline",required=True);ap.add_argument("--candidate",required=True);ap.add_argument("--layers",type=int,required=True);ap.add_argument("--hidden",type=int,required=True);ap.add_argument("--start-layer",type=int,default=0);ap.add_argument("--budget",type=float,default=.0025);a=ap.parse_args()
 def rd(p):
  x=array.array("f");x.frombytes(Path(p).read_bytes());return x
 b,c=rd(a.baseline),rd(a.candidate);need=a.layers*a.hidden
 if len(b)!=need or len(c)!=need:raise SystemExit(f"dump size mismatch {len(b)} {len(c)} expected {need}")
 rows=[];mx=0.
 for l in range(a.layers):
  off=l*a.hidden;num=den=0.
  for i in range(a.hidden):
   d=c[off+i]-b[off+i];num+=d*d;den+=b[off+i]*b[off+i]
  rel=math.sqrt(num)/(math.sqrt(den)+1e-12);rows.append({"layer":l,"relative_l2":rel})
  if l>=a.start_layer:mx=max(mx,rel)
 ok=mx<=a.budget
 print(json.dumps({"schema":"beglin-qt-transformer-hidden-beval-v1","status":"PASS" if ok else "FAIL","layers":a.layers,"hidden":a.hidden,"start_layer":a.start_layer,"budget":a.budget,"max_relative_l2":mx,"margin":a.budget-mx,"per_layer":rows,"production_write_allowed":False,"automatic_live_promotion":False},sort_keys=True))
 raise SystemExit(0 if ok else 1)
if __name__=="__main__":main()
