#!/usr/bin/env python3
import argparse,json,subprocess,shutil,hashlib
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--checkpoint",required=True);p.add_argument("--config",required=True);p.add_argument("--prompt",required=True)
p.add_argument("--engine",required=True);p.add_argument("--maps",required=True);p.add_argument("--tensor",required=True)
p.add_argument("--baseline",required=True);p.add_argument("--work",required=True);p.add_argument("--budget",type=float,default=.0025)
p.add_argument("--max-rounds",type=int,default=256);a=p.parse_args()
work=Path(a.work);work.mkdir(parents=True,exist_ok=True)
safe=a.tensor.replace(".","_");src=Path(a.maps)/(safe+".bits")
bits=bytearray(src.read_bytes())
# Deterministic promotion priority: lowest current bit first, then stable cell index.
# This outer search uses REAL Transformer BEVAL; projection calibrator remains only a prior.
hist=[]
for rd in range(a.max_rounds+1):
 d=work/f"r{rd:02d}";d.mkdir(exist_ok=True)
 cur=d/(safe+".bits");cur.write_bytes(bits)
 dump=d/"hidden.bin"
 env=f"QWEN_SAFETENSORS={a.checkpoint} QWEN_HF_CONFIG={a.config} QWEN_PROMPT={a.prompt} QWEN_QT_MIXED_MANIFEST={d} QWEN_DEBUG_LAYERDUMP={dump} QWEN_DEBUG_LAYERDUMP_POS=15"
 subprocess.run(["/bin/sh","-lc",env+" "+a.engine+" greedy 1"],check=True)
 out=d/"beval.json"
 r=subprocess.run(["python3","tools/qt_hidden_dump_beval.py","--baseline",a.baseline,"--candidate",str(dump),"--layers","24","--hidden","896","--start-layer","0","--budget",str(a.budget)],stdout=out.open("w"))
 x=json.load(open(out));avg=sum(bits)/len(bits)
 hist.append({"round":rd,"status":x["status"],"max_relative_l2":x["max_relative_l2"],"margin":x["margin"],"average_bits":avg,"bits_sha256":hashlib.sha256(bits).hexdigest()})
 if r.returncode==0: break
 cand=[i for i,n in enumerate(bits) if n<8]
 if not cand: break
 cand.sort(key=lambda i:(bits[i],i))
 batch=max(1,len(bits)//32)
 for i in cand[:batch]: bits[i]+=1
(work/"FINAL.bits").write_bytes(bits)
(work/"SEARCH.json").write_text(json.dumps({"schema":"beglin-qt-transformer-promotion-search-v1","requires_fp32_control":True,"tensor":a.tensor,"budget":a.budget,"history":hist,"final_status":hist[-1]["status"],"final_average_bits":sum(bits)/len(bits)},sort_keys=True,indent=2)+"\n")
print(json.dumps(hist[-1],sort_keys=True))
