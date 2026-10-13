#!/usr/bin/env python3
"""Causal slow/fast observations from offline, prompt-independent real DeepSeek runs.

This is an exploration of predictive *surrogates*, not native safety triggers.
All actual h2 vectors, token IDs, original logits and routing text stay on XOX.
Only aggregate model metrics are eligible for uploading after evaluation.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,re,struct
import numpy as np

SCHEMA="beglin-hcp-p2-causal-real-trace-v1"
ROUTE=re.compile(r"^pos\s+(\d+)\s+layer\s+(\d+)\s+experts\s+(.+)$")

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def read_hidden(p,expected):
    raw=p.read_bytes(); size=12+4*2048
    if len(raw)!=expected*size:raise ValueError("h2 record length mismatch")
    out={}
    for off in range(0,len(raw),size):
        pos,layer,hidden=struct.unpack_from("<iii",raw,off)
        if layer!=1 or hidden!=2048 or pos in out or pos<0:
            raise ValueError("wrong/duplicate h2 site")
        a=np.frombuffer(raw,dtype="<f4",count=2048,offset=off+12).astype(np.float64)
        if not np.isfinite(a).all() or np.linalg.norm(a)<=0:raise ValueError("invalid hidden activation")
        out[pos]=a
    if sorted(out)!=list(range(expected)):raise ValueError("missing h2 positions")
    return out

def read_route(p,expected):
    out={}
    for line in p.read_text(errors="replace").splitlines():
        m=ROUTE.match(line.strip())
        if not m or int(m.group(2))!=1:continue
        pos=int(m.group(1))
        if pos in out or not(0<=pos<expected):raise ValueError("duplicate/wrong route position")
        pairs=[]
        for part in m.group(3).split():
            e,w=part.split(":",1)
            eid=int(e); weight=float(w)
            if not(0<=eid<64 and math.isfinite(weight) and weight>=0):
                raise ValueError("invalid routed expert score")
            pairs.append((eid,weight))
        if len(pairs)!=6 or len(set(k for k,_ in pairs))!=6:raise ValueError("Top-K route cardinality mismatch")
        scores=np.array([w for _,w in pairs],dtype=float)
        scores/=max(float(scores.sum()),1e-25)
        entropy=float(-np.dot(scores,np.log(np.maximum(scores,1e-30)))/math.log(6))
        gap=float(np.sort(scores)[-1]-np.sort(scores)[-2])
        out[pos]={"entropy":entropy,"gap":gap,"concentration":float(max(scores))}
    if sorted(out)!=list(range(expected)):raise ValueError("missing actual layer-1 routes")
    return out

def logits_entropy_file(p,positions):
    if not p.is_file():raise ValueError("missing full-forward logits")
    length=p.stat().st_size
    if length % (4*positions):raise ValueError("logit length mismatch")
    vocab=length//(4*positions)
    if not 100000<=vocab<=110000:raise ValueError("unexpected DeepSeek vocabulary/logit size")
    out=[]
    # Iterate rows via read-only mmap; never export logits.
    buf=np.memmap(p,dtype="<f4",mode="r",shape=(positions,vocab))
    for row in buf:
        x=np.asarray(row,dtype=np.float64)
        if not np.isfinite(x).all():raise ValueError("nonfinite true logits")
        weights=np.exp(x-np.max(x))
        norm=float(np.sum(weights))
        probs=weights/norm
        out.append(float(-np.dot(probs,np.log(np.maximum(probs,1e-300)))/math.log(vocab)))
    del buf
    return out,vocab

def extract_prompt(prompt_dir,expected):
    h=read_hidden(prompt_dir/"true_h2.bin",expected)
    route=read_route(prompt_dir/"moe3a_c_routing.txt",expected)
    logits,vocab=logits_entropy_file(prompt_dir/"moe3a_c_logits.bin",expected)
    raw=[];ema=np.zeros(6,np.float64)
    for pos in range(expected):
        a=h[pos];r=route[pos]
        instant=np.array([
            math.log1p(float(np.linalg.norm(a)))/5,
            math.log1p(float(np.std(a)))/2,
            r["entropy"],r["gap"],r["concentration"],
            0.0 if pos==0 else logits[pos-1]
        ],dtype=np.float64)
        ema=instant if pos==0 else .75*ema+.25*instant
        raw.append({"position":pos,"fast":instant.tolist(),"slow":ema.tolist(),
                    "proxy":r["entropy"]})
    rows=[]
    for t in range(expected-1):
        # Predict next-token/next-four-token top-six distribution entropy. This is
        # *not* true topK/topK+1 routing ambiguity or certified precision margin.
        future=[x["proxy"] for x in raw[t+1:min(expected,t+5)]]
        z=raw[t].copy();z["target_next4"]=float(np.mean(future))
        z["target_next1"]=float(raw[t+1]["proxy"])
        rows.append(z)
    return rows,vocab

def build(root,output):
    root=pathlib.Path(root)
    meta=json.loads((root/"runs.json").read_text())
    if meta.get("schema")!="beglin-prompt-suite-v1" or meta.get("prompt_count")!=120:
        raise ValueError("not the 120-prompt independently captured corpus")
    records=meta.get("runs",[])
    if len(records)!=120:raise ValueError("incomplete prompt manifest")
    counts={"train":0,"validation":0,"holdout":0};docs=[];ids=set();tokenhash=set();vocab=None
    for rec in records:
        pid=rec["prompt_id"];split=rec["split"]
        if pid in ids or split not in counts or rec["input_ids_sha256"] in tokenhash:
            raise ValueError("prompt identity or split duplicate")
        ids.add(pid);tokenhash.add(rec["input_ids_sha256"])
        p=root/pid
        if not p.is_dir() or p.is_symlink():raise ValueError("missing prompt")
        for name,field in (("true_h2.bin","capture_sha256"),("moe3a_c_routing.txt","routing_sha256")):
            if digest(p/name)!=rec[field]:raise ValueError("raw trace identity mismatch")
        rows,v=extract_prompt(p,int(rec["position_count"]))
        if vocab is None:vocab=v
        if v!=vocab:raise ValueError("vocabulary changed between independent prompts")
        if len(rows)<3:raise ValueError("too few tokens for slow-fast target")
        counts[split]+=1
        docs.append({"id":pid,"split":split,"domain":rec.get("domain"),"rows":rows})
    if counts!={"train":72,"validation":24,"holdout":24}:
        raise ValueError("wrong prompt independent split")
    out={"schema":SCHEMA,"source":"REAL_DEEPSEEK_QUANTIZED_AF_H2_AND_TOP6_ROUTING",
         "status":"PROXY_ONLY_NO_CERTIFIED_RUNTIME_PERMISSION",
         "prompts":docs,"prompt_counts":counts,"target":"next 1 or 4 token top-6 selected-score entropy",
         "target_is_genuine_topK_boundary_margin":False,
         "input_uses_same_token_post_ffn_logits":False,
         "features_use_only_current_pre_ffn_h2_current_top6_route_and_prior_logits":True,
         "causal_time_split":True,"vocabulary":vocab,
         "source_index_sha256":meta["checkpoint_index_sha256"],
         "raw_hidden_uploaded":False,"raw_logits_uploaded":False,
         "policy_change_allowed":False,"production_touched":False}
    output=pathlib.Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,sort_keys=True,separators=(",",":"))+"\n")
    print("P2_REAL_CAUSAL_FEATURES",json.dumps({"prompts":sum(counts.values()),
       "tokens_with_future_target":sum(len(p["rows"]) for p in docs),
       "split":counts,"source":out["source"],"proxy_only":True},sort_keys=True))
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--prompt-suite",type=pathlib.Path,required=True)
    p.add_argument("--output",type=pathlib.Path,required=True)
    a=p.parse_args()
    build(a.prompt_suite,a.output)
