#!/usr/bin/env python3
"""Read-only, function-preserving permutation/sign alignment of real MoE experts.

Joint permutation of gate/up rows and down columns, plus simultaneous sign
flips of up rows and down columns, leaves SwiGLU unchanged. Alignment reads
only designated signature coordinates. The heldout low-rank comparison excludes
all signature and coefficient-fit coordinates. Never writes model weights.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from real_checkpoint_reader import RealExpertTileReader
from hierarchical_moe_probe import assemble_swiglu, fit_and_predict

SCHEMA="beglin-expert-permutation-alignment-probe-v1"

def _cosine_rows(a):
    a=np.asarray(a,dtype=np.float32)
    return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-10)

def triplet_signature(reader,e,coords):
    z=[]
    for role in ("gate_proj","up_proj","down_proj"):
        raw=reader._array(e,role)
        t=raw[coords,:].T if role=="down_proj" else raw[:,coords]
        z.append(_cosine_rows(reader._float(t,reader.tensors[(e,role)]["dtype"])))
    return tuple(z)

def sign_invariant_similarity(ref,cand):
    g=ref[0]@cand[0].T
    pair=(ref[1]@cand[1].T+ref[2]@cand[2].T)*.5
    return (.45*g+.55*np.abs(pair)).astype(np.float32)

def align_descriptors(anchor,candidate):
    sim=sign_invariant_similarity(anchor,candidate)
    row,col=linear_sum_assignment(-sim)
    if not np.array_equal(row,np.arange(len(row))):
        raise RuntimeError("Hungarian assignment changed row ordering")
    pair=np.sum(anchor[1]*candidate[1][col],axis=1)+np.sum(anchor[2]*candidate[2][col],axis=1)
    signs=np.where(pair<0,-1.,1.).astype(np.float32)
    return col.astype(np.int32),signs,float(np.mean(sim[row,col]))

def similarity_holdout(reference,candidate,permutation,sign):
    g=np.sum(reference[0]*candidate[0][permutation],axis=1)
    ud=(np.sum(reference[1]*candidate[1][permutation],axis=1)
       +np.sum(reference[2]*candidate[2][permutation],axis=1))*.5*sign
    return float(np.mean(.45*g+.55*ud))

def permute_sign_rows(rows,permutation,signs):
    n,t=rows.shape
    if t%3 or len(permutation)!=n or len(signs)!=n:
        raise ValueError("dimension mismatch")
    if np.unique(permutation).size!=n or np.min(permutation)!=0 or np.max(permutation)!=n-1:
        raise ValueError("invalid joint permutation")
    d=t//3
    out=np.asarray(rows,dtype=np.float32)[permutation].copy()
    out[:,d:2*d]*=signs[:,None]
    out[:,2*d:]*=signs[:,None]
    return out

def read_rows(reader,e,original_rows):
    d=reader.hidden
    out=np.empty((len(original_rows),3*d),dtype=np.float32)
    for role,loc in (("gate_proj",slice(0,d)),("up_proj",slice(d,2*d)),("down_proj",slice(2*d,3*d))):
        raw=reader._array(e,role)
        chosen=raw[:,original_rows].T if role=="down_proj" else raw[original_rows,:]
        out[:,loc]=reader._float(chosen,reader.tensors[(e,role)]["dtype"])
    return out

def block_rows(reader,e,target_rows,perm,sign):
    out=read_rows(reader,e,perm[target_rows])
    d=reader.hidden
    out[:,d:]*=sign[target_rows,None]
    return out

def source_tile(reader,perm,signs,tile_idx):
    idx=np.arange(tile_idx*128,(tile_idx+1)*128)
    if np.max(idx)>=reader.intermediate:
        raise ValueError("out-of-range tile")
    out=np.empty((len(reader.ids),128,3*reader.hidden),dtype=np.float32)
    for j,e in enumerate(reader.ids):
        out[j]=block_rows(reader,e,idx,perm[j],signs[j])
    return out

def strict_factorization_evaluation(reader,perms,signs,signature_coords,*,widths,rank,train,seed):
    """Same heldout weight coordinates for aligned, native, random controls.

    Matching uses only signature coords; coefficient fitting uses separate
    calibration coords; neither appears in evaluation coords. Training expert
    weights can be fully observed, but no heldout expert evaluation coord is
    exposed to the alignment or coefficient-fit oracle.
    """
    E=len(reader.ids);d=reader.hidden
    tiles=reader.intermediate//128
    if reader.intermediate%128 or train<=rank+1 or train>=E-1:
        raise ValueError("invalid shape/expert count")
    forbidden=np.concatenate([signature_coords,signature_coords+d,signature_coords+2*d])
    allowed=np.ones(3*d,dtype=bool);allowed[forbidden]=False
    stats={mode:{str(w):{"err_sq":0.,"mean_sq":0.,"numel":0} for w in widths} for mode in perms}
    for ti in range(tiles):
        for mode in perms:
            x=source_tile(reader,perms[mode],signs[mode],ti)
            for width in widths:
                for sl in range(0,128,width):
                    v=x[:,sl:sl+width,:].reshape(E,-1)
                    possible=np.flatnonzero(np.tile(allowed,width))
                    rnd=np.random.default_rng(seed+32452843*ti+127*width+sl)
                    idx=rnd.permutation(possible)
                    ncal=max(rank+8,int(.2*len(idx)))
                    calib=idx[:ncal]
                    eval_idx=idx[ncal:]
                    basis_idx=idx[ncal:ncal+min(4096,len(eval_idx))]
                    pred,mu,_=fit_and_predict(v,train,rank,calib,basis_idx)
                    delta=(v[train:,eval_idx]-pred[train:,eval_idx]).astype(np.float64)
                    base=(v[train:,eval_idx]-mu[eval_idx]).astype(np.float64)
                    cell=stats[mode][str(width)]
                    cell["err_sq"]+=float(np.sum(delta*delta))
                    cell["mean_sq"]+=float(np.sum(base*base))
                    cell["numel"]+=int(delta.size)
    return {mode:{w:{
        "heldout_mse_vs_shared_mean_explained":1-cell["err_sq"]/max(cell["mean_sq"],1e-30),
        "heldout_error_over_mean_error":cell["err_sq"]/max(cell["mean_sq"],1e-30),
        "independent_evaluation_parameter_count":cell["numel"]
    } for w,cell in values.items()} for mode,values in stats.items()}

def make_permutations(reader,seed,signature_width,validation_width):
    d=reader.hidden;n=reader.intermediate;E=len(reader.ids)
    rng=np.random.default_rng(seed)
    coords=rng.permutation(d)
    sig=np.sort(coords[:signature_width])
    val=np.sort(coords[signature_width:signature_width+validation_width])
    if signature_width+validation_width>d:
        raise ValueError("signature + validation exceeds hidden width")
    ref=triplet_signature(reader,reader.ids[0],sig)
    test_ref=triplet_signature(reader,reader.ids[0],val)
    native=np.tile(np.arange(n,dtype=np.int32),(E,1))
    native_sign=np.ones((E,n),np.float32)
    aligned=native.copy();signs=native_sign.copy()
    checks=[]
    for j,e in enumerate(reader.ids):
        cand=triplet_signature(reader,e,sig)
        test=triplet_signature(reader,e,val)
        if j:
            aligned[j],signs[j],train_sim=align_descriptors(ref,cand)
        else:
            train_sim=1.
        checks.append({
            "expert_id":int(e),"signature_matching_score":train_sim,
            "natural_heldout_signature_cos":similarity_holdout(test_ref,test,native[j],native_sign[j]),
            "aligned_heldout_signature_cos":similarity_holdout(test_ref,test,aligned[j],signs[j]),
            "joint_up_down_sign_flips":int(np.sum(signs[j]<0)),
            "changed_row_fraction":float(np.mean(aligned[j]!=np.arange(n)))})
    sham=native.copy()
    for j in range(1,E):
        sham[j]=rng.permutation(n)
    perms={"unaligned":native,"aligned":aligned,"sham_random":sham}
    all_signs={"unaligned":native_sign,"aligned":signs,"sham_random":np.ones_like(signs)}
    diag={
        "permutation_sign_sha256":hashlib.sha256(aligned.tobytes()+signs.tobytes()).hexdigest(),
        "signature_coordinate_hash":hashlib.sha256(sig.tobytes()).hexdigest(),
        "heldout_signature_hash":hashlib.sha256(val.tobytes()).hexdigest(),
        "reference_expert":reader.ids[0],
        "per_expert":checks,
        "aligned_heldout_cos":float(np.mean([x["aligned_heldout_signature_cos"] for x in checks[1:]])),
        "unaligned_heldout_cos":float(np.mean([x["natural_heldout_signature_cos"] for x in checks[1:]]))}
    return perms,all_signs,diag,sig

def real_function_check(reader,perms,signs,seed):
    j=1 if len(reader.ids)>1 else 0
    e=reader.ids[j]
    native=read_rows(reader,e,np.arange(reader.intermediate))
    aligned=block_rows(reader,e,np.arange(reader.intermediate),perms["aligned"][j],signs["aligned"][j])
    rng=np.random.default_rng(seed+1943)
    errors=[]
    for _ in range(2):
        h=rng.normal(0,.5,reader.hidden).astype(np.float64)
        y=assemble_swiglu(native,h,16)
        y2=assemble_swiglu(aligned,h,16)
        errors.append(float(np.linalg.norm(y-y2)/(np.linalg.norm(y)+1e-30)))
    if not np.all(np.isfinite(errors)) or max(errors)>1e-8:
        raise AssertionError(f"Function-preserving permutation failed: {errors}")
    return {"expert_id":int(e),"input_count":2,"max_relative_l2":max(errors),"pass":True}

def execute(args):
    start=time.time()
    reader=RealExpertTileReader(args.checkpoint_index,layer=args.layer,
        expert_ids=list(range(args.experts)),hidden=args.hidden,intermediate=args.intermediate)
    perms,signs,diag,sig=make_permutations(reader,args.seed,args.signature_width,args.validation_width)
    exact=real_function_check(reader,perms,signs,args.seed)
    metrics=strict_factorization_evaluation(reader,perms,signs,sig,
        widths=args.widths,rank=args.rank,train=args.train_experts,seed=args.seed)
    result={
        "schema":SCHEMA,"status":"PASS","identity":reader.provenance(),
        "source_sha256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "shape":{"layer":args.layer,"hidden":args.hidden,"intermediate":args.intermediate,
                 "experts":args.experts,"train_experts":args.train_experts,
                 "heldout_experts":args.experts-args.train_experts},
        "alignment_signature_width":args.signature_width,
        "heldout_signature_width":args.validation_width,
        "rank":args.rank,"alignment":diag,"function_equivalence":exact,
        "heldout_coordinate_factorization":metrics,
        "cpu_wall_seconds":round(time.time()-start,3),
        "production_touched":False,"automatic_live_promotion":False,
        "limitations":[
            "Real checkpoint weights only, no real transformer hidden activations",
            "Does not implement native Beglin qNg64",
            "Heldout experts use separately observed calibration coordinates, not zero-shot prediction",
            "Heldout coordinate scores exclude matching and coefficient-fit coordinates",
            "No runtime/router change; one layer and one anchor expert"] }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("ALIGNMENT_RESULT",json.dumps({
        "status":result["status"],"cpu_wall_s":result["cpu_wall_seconds"],
        "function_relative_l2":exact["max_relative_l2"],
        "natural_heldout_cos":diag["unaligned_heldout_cos"],
        "aligned_heldout_cos":diag["aligned_heldout_cos"],
        "heldout":{name:{w:round(t["heldout_mse_vs_shared_mean_explained"],9) for w,t in records.items()}
                   for name,records in metrics.items()}},sort_keys=True))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-index",required=True,type=pathlib.Path)
    p.add_argument("--layer",type=int,default=1)
    p.add_argument("--experts",type=int,default=8)
    p.add_argument("--train-experts",type=int,default=4)
    p.add_argument("--rank",type=int,default=2)
    p.add_argument("--hidden",type=int,default=2048)
    p.add_argument("--intermediate",type=int,default=1408)
    p.add_argument("--widths",type=int,nargs="+",default=[16,64])
    p.add_argument("--signature-width",type=int,default=64)
    p.add_argument("--validation-width",type=int,default=64)
    p.add_argument("--seed",type=int,default=20261013)
    p.add_argument("--output",required=True,type=pathlib.Path)
    a=p.parse_args()
    if a.intermediate%128 or any(128%w for w in a.widths):
        p.error("Widths must divide 128 and intermediate must be a multiple of 128")
    if a.signature_width+a.validation_width>a.hidden:
        p.error("Signature and validation widths exceed hidden dimensions")
    execute(a)

if __name__=="__main__":
    main()
