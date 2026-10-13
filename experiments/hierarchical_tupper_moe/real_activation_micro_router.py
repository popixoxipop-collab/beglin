#!/usr/bin/env python3
"""Offline true-activation Micro-Expert evaluation; does not mutate inference.

Requires real Beglin pre-FFN h2 capture and actual layer-1 top-K trace.
The trace comes from quantized AF inference; expert BF16 weights may therefore
disagree with the quantized forward. Uses calibration/holdout POSITIONS, not
random Gaussian hidden inputs. Static salience sees only calibration positions.
"""
import argparse
import hashlib
import json
import math
import pathlib
import re
import struct
import numpy as np
from real_checkpoint_reader import RealExpertTileReader
from permutation_alignment_probe import read_rows
from hierarchical_moe_probe import tupper_encode,tupper_decode

def read_capture(file,layer,hidden):
    data=file.read_bytes()
    n=12+4*hidden
    if not data or len(data)%n:
        raise ValueError("empty or invalid real pre-FFN hidden trace")
    out={}
    for i in range(0,len(data),n):
        pos,l,d=struct.unpack_from("<iii",data,i)
        if l!=layer or d!=hidden or pos<0 or pos in out:
            raise ValueError("duplicate positions or incorrect actual layer")
        h=np.frombuffer(data,dtype="<f4",offset=i+12,count=d).astype(np.float64)
        if not np.isfinite(h).all() or np.linalg.norm(h)==0:
            raise ValueError("nonfinite or zero true hidden vector")
        out[pos]=h
    return dict(sorted(out.items()))

def read_route(file,layer,positions):
    seen={}
    regex=re.compile(r"^pos\s+(\d+)\s+layer\s+(\d+)\s+experts\s+(.+)$")
    for raw in file.read_text(errors="replace").splitlines():
        m=regex.match(raw.strip())
        if m is None:continue
        pos,l=int(m[1]),int(m[2])
        if l!=layer or pos not in positions:continue
        if pos in seen:raise ValueError("duplicate actual router position")
        vals=[]
        for token in m[3].split():
            if ":" not in token:raise ValueError("malformed router score")
            a,b=token.split(":",1);e=int(a);p=float(b)
            if e<0 or not np.isfinite(p) or p<0:raise ValueError("bad actual router score")
            vals.append((e,p))
        if not vals or len(set(e for e,_ in vals))!=len(vals):
            raise ValueError("empty or duplicated actual top-K")
        seen[pos]=vals
    if set(seen)!=set(positions):
        raise ValueError("BLOCKED: incomplete real macro-router trace")
    return seen

def channels(rows,h,width):
    d=rows.shape[1]//3;mid=rows.shape[0]
    if d!=h.size or mid%width:raise ValueError("wrong expert/capture dimensions")
    g=rows[:,:d].astype(np.float64)@h
    u=rows[:,d:2*d].astype(np.float64)@h
    z=(g/(1+np.exp(-np.clip(g,-40,40))))*u
    D=rows[:,2*d:].astype(np.float64)
    return np.stack([z[i:i+width]@D[i:i+width,:] for i in range(0,mid,width)])

def evaluate(index,capture,router,layer,hidden,intermediate,widths=(16,64),fractions=(.25,.5,.75,1.)):
    acts=read_capture(capture,layer,hidden)
    if len(acts)<4:raise ValueError("need >=4 real activation positions")
    route=read_route(router,layer,set(acts))
    positions=sorted(acts)
    cut=len(positions)//2;cal=positions[:cut];held=positions[cut:]
    eids=sorted({e for pair in route.values() for e,_ in pair})
    if not eids or max(eids)>=64:raise ValueError("model is not configured for 64 experts")
    reader=RealExpertTileReader(index,layer=layer,expert_ids=eids,hidden=hidden,intermediate=intermediate)
    results={}
    for width in widths:
        n=intermediate//width
        if intermediate%width:raise ValueError("width must divide expert hidden neurons")
        fixed={f:{} for f in fractions};out={}
        for eid in eids:
            row=read_rows(reader,eid,np.arange(intermediate))
            train=np.stack([channels(row,acts[pos],width) for pos in cal])
            importance=np.mean(np.linalg.norm(train,axis=2),axis=0)
            for f in fractions:
                count=max(1,math.ceil(n*f))
                fixed[f][eid]=np.sort(np.argsort(importance,kind="stable")[-count:])
            out[eid]={pos:channels(row,acts[pos],width) for pos in held}
            del row,train
        metrics={}
        for f in fractions:
            total=0.;error=0.;worst=0.
            mask=np.zeros((n,64),np.uint8)
            for e,selected in fixed[f].items():mask[selected,e]=1
            packed=tupper_encode(mask)
            if not np.array_equal(tupper_decode(packed["k"],n,64),mask):
                raise AssertionError("generalized Tupper N x M mask not lossless")
            for pos in held:
                ref=np.zeros(hidden,np.float64);candidate=np.zeros(hidden,np.float64)
                for eid,wt in route[pos]:
                    vals=out[eid][pos]
                    ref+=wt*vals.sum(axis=0)
                    candidate+=wt*vals[fixed[f][eid]].sum(axis=0)
                err=np.linalg.norm(ref-candidate)/(np.linalg.norm(ref)+1e-30)
                worst=max(worst,float(err))
                total+=float(np.sum(ref**2));error+=float(np.sum((ref-candidate)**2))
            metrics[str(f)]={
                "heldout_macro_output_relative_l2":float(np.sqrt(error/max(total,1e-30))),
                "heldout_macro_output_worst_relative_l2":worst,
                "real_heldout_positions":len(held),
                "retained_per_routed_expert":len(next(iter(fixed[f].values()))),
                "full_micro_channels_per_expert":n,
                "theoretical_expert_ffn_compute_fraction":len(next(iter(fixed[f].values())))/n,
                "tupper_grid":[n,64],"tupper_mask_roundtrip":True}
        results[str(width)]=metrics
    return {"schema":"beglin-actual-micro-router-v1","status":"PASS",
        "real_pre_ffn_capture_sha256":hashlib.sha256(capture.read_bytes()).hexdigest(),
        "real_router_sha256":hashlib.sha256(router.read_bytes()).hexdigest(),
        "expert_checkpoint_index_sha256":reader.provenance()["index_sha256"],
        "source_mismatch":"actual hidden from quantized AF forward, expert matrices from BF16 safetensors",
        "calibration_positions":len(cal),"holdout_positions":len(held),
        "actual_router_top_k":len(next(iter(route.values()))),
        "observed_macro_expert_union_count":len(eids),"true_activation":True,
        "true_macro_route":True,"counterfactual_calibration_for_unselected_experts":True,
        "sparse_policy_trained":False,"task_metrics":False,"runtime_speed_measured":False,
        "production_touched":False,"automatic_promotion":False,"results":results}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-index",required=True,type=pathlib.Path)
    p.add_argument("--capture",required=True,type=pathlib.Path)
    p.add_argument("--routing",required=True,type=pathlib.Path)
    p.add_argument("--layer",type=int,default=1)
    p.add_argument("--hidden",type=int,default=2048)
    p.add_argument("--intermediate",type=int,default=1408)
    p.add_argument("--output",required=True,type=pathlib.Path)
    a=p.parse_args()
    result=evaluate(a.checkpoint_index,a.capture,a.routing,a.layer,a.hidden,a.intermediate)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("ACTUAL_MICRO_ROUTER_RESULT",json.dumps({
        "calibration_tokens":result["calibration_positions"],"heldout_tokens":result["holdout_positions"],
        "real_macro_router":result["true_macro_route"],"errors":{
          w:{f:round(v["heldout_macro_output_relative_l2"],6) for f,v in ff.items()}
          for w,ff in result["results"].items()}},sort_keys=True))
if __name__=="__main__":
    main()
