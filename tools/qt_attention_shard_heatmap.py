#!/usr/bin/env python3
import argparse,array,hashlib,json,math,struct
from pathlib import Path
G=64
def bf16(raw):
 a=array.array("f")
 for (v,) in struct.iter_unpack("<H",raw):a.append(struct.unpack("<f",struct.pack("<I",v<<16))[0])
 return a
def qerr(xs,n):
 qmax=(1<<(n-1))-1;qmin=-(1<<(n-1));mx=max(abs(x) for x in xs);sc=mx/qmax if mx>1e-12 else 1.;inv=1/sc;res=0.;dq=[]
 for x in xs:
  z=x+res;q=max(qmin,min(qmax,int(round(z*inv))));y=q*sc;res=z-y;dq.append(y)
 return dq
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--shard",required=True);ap.add_argument("--output-dir",required=True);ap.add_argument("--windows",type=int,default=4);a=ap.parse_args()
 sh=json.load(open(a.shard));p=Path(a.checkpoint);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 with p.open("rb") as f:n=struct.unpack("<Q",f.read(8))[0];h=json.loads(f.read(n));base=8+n
 summaries=[];total=0
 with p.open("rb") as f:
  for t in sh["tensors"]:
   e=h[t["tensor_name"]];s,z=e["data_offsets"];f.seek(base+s);raw=f.read(z-s)
   vals=bf16(raw);O,I=t["shape"];ng=I//G;hist={4:0,5:0,6:0};unc=[];cells=[]
   for r in range(O):
    ro=r*I
    for g in range(ng):
     xs=[float(vals[ro+g*G+i]) for i in range(G)]; per={}
     for nbit in (4,5,6):
      dq=qerr(xs,nbit); errs=[]
      for w in range(a.windows):
       p1=.173+.013*w;p2=.071+.007*w
       basev=qv=0.
       for i in range(G):
        col=g*G+i;x=math.sin((col+1)*p1)+.25*math.cos((col+1)*p2);basev+=xs[i]*x;qv+=dq[i]*x
       errs.append(abs(qv-basev)/(abs(basev)+1e-6))
      per[nbit]={"mean":sum(errs)/len(errs),"max":max(errs),"spread":max(errs)-min(errs)}
     # pilot selection: smallest n whose max relative local error <= n6 max + 0.0025.
     ref=per[6]["max"]; chosen=next((n for n in (4,5,6) if per[n]["max"]<=ref+.0025),6)
     hist[chosen]+=1;unc.append(per[chosen]["spread"]);cells.append(chosen)
   total+=len(cells);rawmap=bytes(cells);mp=out/(t["tensor_name"].replace(".","_")+".bits")
   mp.write_bytes(rawmap);summaries.append({"tensor_name":t["tensor_name"],"layer":t["layer"],"role":t["role"],"cell_count":len(cells),"bit_histogram":{str(k):v for k,v in hist.items()},"average_bits":sum(cells)/len(cells),"uncertainty_mean_spread":sum(unc)/len(unc),"bits_sha256":hashlib.sha256(rawmap).hexdigest()})
 ev={"schema":"beglin-qt-attention-shard-heatmap-pilot-v1","shard":sh["shard"],"windows":a.windows,"tensor_count":len(summaries),"total_cells":total,"selection_status":"PILOT_NOT_DOWNSTREAM_CERTIFIED","tensors":summaries}
 raw=(json.dumps(ev,sort_keys=True,indent=2)+"\n").encode();(out/"HEATMAP_PILOT.json").write_bytes(raw)
 if total!=sh["total_cells"]:raise SystemExit("coverage mismatch")
 print(json.dumps({"status":"PASS","shard":sh["shard"],"tensor_count":len(summaries),"total_cells":total,"evidence_sha256":hashlib.sha256(raw).hexdigest(),"downstream_certified":False},sort_keys=True))
if __name__=="__main__":main()
