#!/usr/bin/env python3
import argparse,array,json,math,struct
from pathlib import Path
G=64
def bf16(raw):
 a=array.array("f")
 for (v,) in struct.iter_unpack("<H",raw):a.append(struct.unpack("<f",struct.pack("<I",v<<16))[0])
 return a
def q(xs,n):
 qm=(1<<(n-1))-1;qn=-(1<<(n-1));mx=max(abs(x) for x in xs);s=mx/qm if mx>1e-12 else 1.;iv=1/s;res=0.;o=[]
 for x in xs:
  z=x+res;c=max(qn,min(qm,int(round(z*iv))));y=c*s;res=z-y;o.append(y)
 return o
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--tensor",required=True);ap.add_argument("--bits",required=True);ap.add_argument("--windows",type=int,default=8);ap.add_argument("--budget",type=float,default=.0025);a=ap.parse_args()
 p=Path(a.checkpoint)
 with p.open("rb") as f:n=struct.unpack("<Q",f.read(8))[0];h=json.loads(f.read(n));e=h[a.tensor];s,z=e["data_offsets"];f.seek(8+n+s);v=bf16(f.read(z-s))
 O,I=map(int,e["shape"]);ng=I//G;bits=Path(a.bits).read_bytes()
 if len(bits)!=O*ng or any(x not in (4,5,6,7,8) for x in bits):raise SystemExit("invalid bits map")
 qv=array.array("f",[0.0])*(O*I);k=0
 for r in range(O):
  for g in range(ng):
   d=q([float(v[r*I+g*G+i]) for i in range(G)],bits[k]);k+=1
   for i,x in enumerate(d):qv[r*I+g*G+i]=x
 vals=[]
 for w in range(a.windows):
  x=[math.sin((i+1)*(.173+.013*w))+.25*math.cos((i+1)*(.071+.007*w)) for i in range(I)]
  num=den=0.
  for r in range(O):
   b=m=0.;off=r*I
   for i in range(I):b+=v[off+i]*x[i];m+=qv[off+i]*x[i]
   d=m-b;num+=d*d;den+=b*b
  vals.append(math.sqrt(num)/(math.sqrt(den)+1e-12))
 mx=max(vals);mean=sum(vals)/len(vals);ok=mx<=a.budget
 print(json.dumps({"schema":"beglin-qt-downstream-beval-v1","status":"PASS" if ok else "FAIL","tensor_name":a.tensor,"shape":[O,I],"cell_count":len(bits),"windows":a.windows,"relative_l2_mean":mean,"relative_l2_max":mx,"budget":a.budget,"margin":a.budget-mx,"production_write_allowed":False,"automatic_live_promotion":False},sort_keys=True))
 raise SystemExit(0 if ok else 1)
if __name__=="__main__":main()
