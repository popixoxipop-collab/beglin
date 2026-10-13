#!/usr/bin/env python3
"""Verified next-token and four-step TOP6/TOP7 risk labels from live model router.

Every observation is a full-softmax rank6/rank7 margin measured BEFORE TopK
renormalization in a scratch C forward, not selected-top6 entropy proxy.
The label is never an authorized precision trigger by itself.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,re
import numpy as np
from hcp_p2_real_feature_builder import read_hidden,read_route,logits_entropy_file,sha as sha_file

SCHEMA="beglin-hcp-p3-true-router-boundary-features-v1"
P3_RE=re.compile(r"^pos (\d+) layer (\d+) kth (\d+) runnerup (\d+) p6 ([\d.eE+-]+) p7 ([\d.eE+-]+) relgap ([\d.eE+-]+) logratio ([\d.eE+-]+) entropy ([\d.eE+-]+)$")
ROUTED_LAYERS=tuple(range(1,27))

def read_boundaries(path,positions):
    seen={}
    raw=path.read_text()
    lines=raw.splitlines()
    if len(lines)!=len(ROUTED_LAYERS)*positions:
        raise ValueError("incomplete rank6/7 observations")
    for line in lines:
        m=P3_RE.match(line)
        if not m:raise ValueError("malformed full-softmax boundary trace")
        p,l,kth,nxt=map(int,m.group(1,2,3,4))
        score6,score7,ratio,logratio,entropy=map(float,m.group(5,6,7,8,9))
        if p<0 or p>=positions or l not in ROUTED_LAYERS or (p,l) in seen or kth==nxt:
            raise ValueError("incorrect rank source, layer or duplicate")
        if not (0<=kth<64 and 0<=nxt<64 and
                0<=score7<=score6<=1.0+1e-5 and
                0<=ratio<=1+1e-4 and 0<=entropy<=1+1e-4 and
                np.isfinite(logratio) and logratio>=-1e-5):
            raise ValueError("invalid actual rank6/rank7 score")
        expected=(score6-score7)/max(score6,1e-30)
        if abs(expected-ratio)>3e-5:
            raise ValueError("wrong rank6/7 gap formula")
        seen[p,l]={"relative_gap":ratio,"logratio":logratio,"entropy":entropy,
                  "kth":kth,"runnerup":nxt,"p6":score6,"p7":score7}
    if len(seen)!=len(lines):raise ValueError("wrong number of unique boundary observations")
    return seen

def actual_layer1_selected_ids(path,positions):
    out={}
    for raw in path.read_text().splitlines():
        if not raw.startswith("pos "):continue
        m=re.match(r"^pos (\d+) layer (\d+) experts (.+)$",raw)
        if not m or int(m.group(2))!=1:continue
        pos=int(m.group(1))
        if pos in out:raise ValueError("duplicate actual layer1 selected route")
        ids={int(word.split(":",1)[0]) for word in m.group(3).split()}
        if len(ids)!=6:raise ValueError("Top-6 identity not preserved")
        out[pos]=ids
    if set(out)!=set(range(positions)):raise ValueError("actual top6 route incomplete")
    return out

def from_one_prompt(base,positions):
    h=read_hidden(base/"true_h2.bin",positions)
    routed=read_route(base/"moe3a_c_routing.txt",positions)
    layer1=actual_layer1_selected_ids(base/"moe3a_c_routing.txt",positions)
    margins=read_boundaries(base/"true_boundary.txt",positions)
    prior_logit_entropy,vocab=logits_entropy_file(base/"moe3a_c_logits.bin",positions)
    raw=[];ema=None
    for t in range(positions):
        scores=routed[t]
        # Original selected Top-6 post-renorm scores are a separate
        # comparison feature; true risk is measured with all 64 scores.
        border=margins[t,1]
        if border["kth"] not in layer1[t] or border["runnerup"] in layer1[t]:
            raise ValueError("TopK identities disagree between pre- and post-renorm route")
        min_gap=min(margins[t,l]["relative_gap"] for l in ROUTED_LAYERS)
        l1_gap=border["relative_gap"]
        worst_layer=min(ROUTED_LAYERS,key=lambda l:margins[t,l]["relative_gap"])
        feat=np.array([
            math.log1p(float(np.linalg.norm(h[t])))/5,
            math.log1p(float(np.std(h[t])))/2,
            scores["entropy"],scores["gap"],scores["concentration"],
            l1_gap,min_gap,
            (0. if t==0 else prior_logit_entropy[t-1]),
        ],dtype=np.float64)
        if not np.isfinite(feat).all() or not(0<=min_gap<=1):
            raise ValueError("untrusted state feature")
        ema=feat if ema is None else .75*ema+.25*feat
        raw.append({"pos":t,"fast":feat.tolist(),"slow":ema.tolist(),
                    "observed_min_ratio":min_gap,
                    "observed_layer1_ratio":l1_gap,
                    "worst_layer":worst_layer})
    rows=[]
    for t in range(positions-1):
        next4=[raw[k]["observed_min_ratio"] for k in range(t+1,min(t+5,positions))]
        rows.append({"position":t,
            "fast":raw[t]["fast"],"slow":raw[t]["slow"],
            "observed_margin":raw[t]["observed_min_ratio"],
            "target_next1":float(next4[0]),
            "target_next4":float(min(next4)),
            "next4_window_length":len(next4)})
    return rows,vocab

def build(source,output):
    src=pathlib.Path(source);doc=json.loads((src/"runs_p3.json").read_text())
    if doc.get("schema")!="beglin-p3-120prompt-full-softmax-boundary-v1" or doc.get("prompt_count")!=120 or doc.get("prompt_split_counts")!={"train":72,"validation":24,"holdout":24}:
        raise ValueError("not the exact frozen 120 prompt/cross-layer corpus")
    if doc.get("production_touched") is not False or doc.get("raw_logits_uploaded") is not False:
        raise ValueError("untrusted production/privacy manifest")
    seen=set();out=[];counter={"train":0,"validation":0,"holdout":0};vocab=None
    for row in doc["records"]:
        name=row["id"];split=row["split"];path=src/name
        if name in seen or split not in counter or path.is_symlink() or not path.is_dir():
            raise ValueError("duplicate source identity or missing prompt")
        seen.add(name)
        for fname,col in (("true_h2.bin","h2_sha256"),("moe3a_c_routing.txt","routing_sha256"),
                          ("true_boundary.txt","boundary_sha256"),("moe3a_c_logits.bin","logits_sha256")):
            p=path/fname
            if not p.is_file() or p.is_symlink() or sha_file(p)!=row[col]:
                raise ValueError("raw data content identity mismatch")
        records,v=from_one_prompt(path,int(row["positions"]))
        if v!=vocab and vocab is not None:raise ValueError("vocab changed across prompts")
        vocab=v
        if len(records)<3:raise ValueError("not enough positions for future risk")
        counter[split]+=1
        out.append({"id":name,"split":split,"domain":row.get("domain"),"rows":records})
    if counter!={"train":72,"validation":24,"holdout":24}:
        raise ValueError("wrong split")
    result={"schema":SCHEMA,"status":"ACTUAL_TOP6_TOP7_RISK_PROXY_ONLY",
            "source":"REAL_DEEPSEEK_QUANTIZED_AF_COMPLETE_TOP6_TOP7",
            "source_manifest_sha256":sha_file(src/"runs_p3.json"),
            "checkpoint_index_sha256":doc["source_index_sha256"],
            "prompts":out,"prompt_counts":counter,
            "margin_source":"real full 64-expert softmax before topK rerank/renorm",
            "route_label":"min across 26 routed layers of (p6-p7)/p6",
            "future_horizon":"min of future 1..4 tokens",
            "lookahead_in_features":False,
            "target_is_certified_precision_risk":False,
            "true_rank7_measured":True,
            "raw_hidden_uploaded":False,"raw_logits_uploaded":False,
            "precision_rebinding_allowed":False,"production_touched":False}
    output=pathlib.Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n")
    print("P3_VERIFIED_FUTURE_KTH_BOUNDARY",json.dumps({
        "prompts":sum(counter.values()),"positions_with_future":sum(len(x["rows"]) for x in out),
        "routed_layers":len(ROUTED_LAYERS),"train_val_holdout":counter,
        "true_rank7":True,"precision_rebinding_allowed":False},sort_keys=True))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--suite",type=pathlib.Path,required=True)
    p.add_argument("--output",type=pathlib.Path,required=True)
    a=p.parse_args();build(a.suite,a.output)
