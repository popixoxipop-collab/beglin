#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math
from collections import defaultdict

def _canon(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)

def project(events,generation=0):
    q=defaultdict(lambda:{"count":0,"errors":defaultdict(list),"current_n":None,"supported":set()})
    t=defaultdict(lambda:{"count":0,"scores":[]})
    for e in events:
        k=e["target_key"]
        if e["kind"]=="quant":
            r=q[k]; r["count"]+=1; n=int(e["candidate_n"]); r["supported"].add(n)
            r["errors"][n].append(float(e["output_max_abs_error"])); r["current_n"]=int(e["current_n"])
        elif e["kind"]=="train":
            r=t[k]; r["count"]+=1
            r["scores"].append(abs(float(e["delta_w_quant"]))*abs(float(e["grad"])))
        else: raise ValueError("unknown event kind")
    qcells=[]
    for k,r in sorted(q.items()):
        err={str(n):sum(v)/len(v) for n,v in sorted(r["errors"].items())}
        safe=[n for n in sorted(r["supported"]) if err[str(n)] <= 5e-4]
        rec=min(safe) if safe else max(r["supported"])
        conf=min(1.0,r["count"]/3.0)
        qcells.append({"target_key":k,"current_n":r["current_n"],"supported_n":sorted(r["supported"]),"recommended_n":rec,"minimum_safe_n":rec if safe else None,"confidence":conf,"sample_count":r["count"],"state":"CANDIDATE" if r["count"]>=3 else "OBSERVED","error_by_n":err})
    tcells=[]
    allscores={k:(sum(r["scores"])/len(r["scores"])) for k,r in t.items()}
    ordered=sorted(allscores.values())
    for k,r in sorted(t.items()):
        s=allscores[k]; rank=(sum(v<=s for v in ordered)/len(ordered)) if ordered else 0.0
        lr=1.0 if rank>.99 else .7 if rank>.95 else .3 if rank>.8 else .1 if rank>.5 else 0.0
        conf=min(1.0,r["count"]/3.0)
        tcells.append({"target_key":k,"sensitivity":s,"recommended_trainable":lr>0,"lr_scale":lr,"confidence":conf,"sample_count":r["count"],"state":"CANDIDATE" if r["count"]>=3 else "OBSERVED"})
    qmap={"schema":"beglin-qheatmap-v2","generation":generation,"cells":qcells}
    tmap={"schema":"beglin-theatmap-v2","generation":generation,"cells":tcells}
    return qmap,tmap

def digest(obj):
    return hashlib.sha256(_canon(obj).encode()).hexdigest()
