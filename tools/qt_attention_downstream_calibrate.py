#!/usr/bin/env python3
import argparse,array,json,math,struct,hashlib
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
 ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--tensor",required=True);ap.add_argument("--bits",required=True);ap.add_argument("--output",required=True);ap.add_argument("--windows",type=int,default=8);ap.add_argument("--budget",type=float,default=.0025);a=ap.parse_args()
 p=Path(a.checkpoint)
 with p.open("rb") as f:n=struct.unpack("<Q",f.read(8))[0];h=json.loads(f.read(n));e=h[a.tensor];s,z=e["data_offsets"];f.seek(8+n+s);v=bf16(f.read(z-s))
 O,I=map(int,e["shape"]);ng=I//G;bits=list(Path(a.bits).read_bytes())
 if len(bits)!=O*ng:raise SystemExit("bits size")
 xs=[[math.sin((i+1)*(.173+.013*w))+.25*math.cos((i+1)*(.071+.007*w)) for i in range(I)] for w in range(a.windows)]
 # Precompute baseline and each cell's output delta at n4/n5/n6 for every window.
 base=[[0.0]*O for _ in xs]; delta={}
 for w,x in enumerate(xs):
  for r in range(O):
   off=r*I;base[w][r]=sum(v[off+i]*x[i] for i in range(I))
 for r in range(O):
  for g in range(ng):
   k=r*ng+g;orig=[float(v[r*I+g*G+i]) for i in range(G)]
   for nb in (4,5,6,7,8):
    dq=q(orig,nb);delta[(k,nb)]=[sum((dq[i]-orig[i])*xs[w][g*G+i] for i in range(G)) for w in range(a.windows)]
 def metric(bs):
  vals=[]
  for w in range(a.windows):
   num=den=0.
   for r in range(O):
    d=sum(delta[(r*ng+g,bs[r*ng+g])][w] for g in range(ng));num+=d*d;den+=base[w][r]*base[w][r]
   vals.append(math.sqrt(num)/(math.sqrt(den)+1e-12))
  return max(vals),sum(vals)/len(vals)
 # Rank promotion benefit by reduction in squared window deltas, deterministic tie by cell.
 def score(k):
  n=bits[k]
  if n>=8:return -1.
  a0=delta[(k,n)];a1=delta[(k,n+1)]
  return sum(x*x-y*y for x,y in zip(a0,a1))
 history=[];mx,mean=metric(bits);history.append({"step":0,"max":mx,"mean":mean,"avg_bits":sum(bits)/len(bits)})
 batch=max(1,len(bits)//32);step=0
 while mx>a.budget and any(n<8 for n in bits):
  cand=[k for k,n in enumerate(bits) if n<8];cand.sort(key=lambda k:(-score(k),k))
  for k in cand[:batch]:bits[k]+=1
  step+=1;mx,mean=metric(bits);history.append({"step":step,"max":mx,"mean":mean,"avg_bits":sum(bits)/len(bits)})
 out=bytes(bits);Path(a.output).write_bytes(out);ok=mx<=a.budget
 print(json.dumps({"schema":"beglin-qt-downstream-calibration-v1","status":"PASS" if ok else "FAIL","tensor_name":a.tensor,"cell_count":len(bits),"budget":a.budget,"relative_l2_max":mx,"relative_l2_mean":mean,"average_bits":sum(bits)/len(bits),"histogram":{str(n):bits.count(n) for n in (4,5,6,7,8)},"steps":step,"history":history,"bits_sha256":hashlib.sha256(out).hexdigest(),"production_write_allowed":False,"automatic_live_promotion":False},sort_keys=True))
 raise SystemExit(0 if ok else 1)
if __name__=="__main__":main()
