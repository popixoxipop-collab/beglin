#!/usr/bin/env python3
from __future__ import annotations
import argparse,array,json,math,re
from pathlib import Path
from p12_qwen25_cross_backend_fixture import read_tensor,quant_planes,GROUP
KEY=re.compile(r"row=(\d+)/group64=(\d+)$")
def contexts(n):
 return [[math.sin((i+1)*(0.071+0.013*k))+0.25*math.cos((i+1)*(0.173+0.009*k)) for i in range(896)] for k in range(n)]
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--uncertainty",required=True);p.add_argument("--output",required=True);p.add_argument("--contexts",type=int,default=16);p.add_argument("--top-k",type=int,default=256);p.add_argument("--budget",type=float,default=5e-4);a=p.parse_args()
 vals,od,idim,dtype=read_tensor(Path(a.checkpoint),"model.layers.0.self_attn.q_proj.weight");u=json.load(open(a.uncertainty))
 targets=sorted(u["cells"],key=lambda x:(-x["active_observation_priority"],x["target_key"]))[:a.top_k];ns=sorted({int(x["recommended_n"]) for x in targets})
 deq={n:quant_planes(vals,od,idim,n)[2] for n in ns};xs=contexts(a.contexts);rows=[]
 for t in targets:
  m=KEY.search(t["target_key"]);r,g=map(int,m.groups());n=int(t["recommended_n"]);off=r*idim+g*GROUP;xoff=g*GROUP
  es=[abs(sum((deq[n][off+i]-vals[off+i])*x[xoff+i] for i in range(GROUP))) for x in xs];safe=sum(e<=a.budget for e in es)
  rows.append({"target_key":t["target_key"],"recommended_n":n,"contexts":a.contexts,"safe_count":safe,"safe_rate":safe/a.contexts,"mean_error":sum(es)/len(es),"max_error":max(es),"errors":es})
 out={"schema":"beglin-qt-active-observation-v1","budget":a.budget,"context_count":a.contexts,"target_count":len(rows),"rows":rows};Path(a.output).write_text(json.dumps(out,sort_keys=True,indent=2)+"\n");print(json.dumps({"status":"PASS","targets":len(rows),"contexts":a.contexts},sort_keys=True))
if __name__=="__main__":main()
