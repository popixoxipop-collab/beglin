#!/usr/bin/env python3
"""JE01 frozen BF16 8 Experts -> native qNg64 n=3/5 real packed tier artifacts.

Strict exact original Chat full-shard SHA and HTTP206, no heldout labels, no
production checkpoint, no original AF q4 identity or model-level quality claim.
"""
from __future__ import annotations
import argparse,ctypes,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
import requests
sys.path.insert(0,str(Path(__file__).resolve().parent))
from je01_hf_exact_bf16_range_tiers import obtain_pinned_source
from je01_selected_bf16_native_artifacts import (
    SHAPES, SOURCE_MODEL_ID, compile_original_quantizer, bf16_to_float, save_unique
)
from je01_native_exact_byte_presolve import (
    read_verified_ledger, validate_ledger, canonical, EXPECTED_BASELINE_BYTES
)
SCHEMA="beglin-je01-original-BF16-expert8-real-qNg64-n3-n5-physical-artifacts/1"
LEDGER=Path(__file__).resolve().parents[1]/"docs/research/JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json"
TIERS=(3,5)
ROLES=("gate_proj","up_proj","down_proj")
EXPECTED_REAL_ALL8={"3":30277632,"5":47579136}
SOURCE_REV="85864749cd611b4353ce1decdb286193298f64c7"

def source_weight(wmap,shards,layer,expert,role):
    key=f"model.layers.{layer}.mlp.experts.{expert}.{role}.weight"
    sn=wmap.get(key)
    if sn not in shards:raise ValueError("JE01_NATIVE_SOURCE_SELECTED_ROLE_MISSING")
    sh=shards[sn]
    if sh.header is None:sh.header_only()
    meta=sh.header.get(key)
    shape=SHAPES[role]
    if not isinstance(meta,dict) or meta.get("shape")!=list(shape) or meta.get("dtype")!="BF16":
        raise ValueError("JE01_SOURCE_SELECTED_TENSOR_METADATA_DRIFT")
    offs=meta.get("data_offsets")
    if (not isinstance(offs,list) or len(offs)!=2 or any(type(o)!=int for o in offs)
            or offs[0]<0 or offs[1]-offs[0]!=int(np.prod(shape))*2
            or sh.base+offs[1]>sh.size):
        raise ValueError("JE01_SOURCE_SELECTED_TENSOR_OFFSETS_INVALID")
    raw=sh.range(sh.base+offs[0],offs[1]-offs[0])
    return bf16_to_float(raw,shape),hashlib.sha256(raw).hexdigest(),sn

def quant_matrix(lib,w,bits):
    if bits not in TIERS or w.dtype!=np.dtype("float32") or not w.flags.c_contiguous:
        raise ValueError("JE01_QN_NATIVE_MATRIX_INVALID")
    out,inp=w.shape
    if inp%64!=0 or out<1:raise ValueError("JE01_QN_GROUP_MISMATCH")
    ngrp=inp//64
    p=np.zeros(w.size*bits//8,dtype=np.uint8)
    s=np.zeros(out*ngrp,dtype=np.float32)
    lib.gguf_quantize_qNg64(
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out,inp,bits,ctypes.c_void_p(p.ctypes.data),
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    if not np.isfinite(s).all() or not np.all(s>0):
        raise ValueError("JE01_QN_SCALES_NONFINITE")
    # Independent unpacker: native grouped bitplanes, little-endian each plane.
    planes=np.unpackbits(p.reshape(out,ngrp,bits,8),axis=-1,bitorder="little")
    code=np.zeros((out,ngrp,64),dtype=np.int16)
    for b in range(bits):code|=planes[:,:,b,:].astype(np.int16)<<b
    y=((code-(1<<(bits-1)))*s.reshape(out,ngrp,1)).reshape(w.shape)
    err=y.astype(np.float64)-w.astype(np.float64)
    relative=float(np.mean(err**2)/max(np.mean(w.astype(np.float64)**2),1e-30))
    if not np.isfinite(relative):raise ValueError("JE01_QN_REAL_BF16_MSE_NONFINITE")
    return p,s,relative

def generate(dest):
    if dest.exists() or dest.is_symlink() or not dest.parent.is_dir():
        raise ValueError("JE01_NEW_OUTPUT_DIR_REQUIRED")
    slots=validate_ledger(read_verified_ledger(LEDGER))
    if len(slots)!=8:raise ValueError("JE01_ORIGINAL_EIGHT_MISSING")
    sess=requests.Session()
    sess.headers.update({"User-Agent":"BEGLIN-JE01-actual-original-BF16-n3n5/1"})
    model,wmap,shards=obtain_pinned_source(sess)
    if model["index_sha256"]!="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf":
        raise ValueError("JE01_BF16_ORIGINAL_INDEX_SHA_CHANGED")
    dest.mkdir(mode=0o700)
    lib=compile_original_quantizer(dest/"je01_native_qng64.dylib")
    records=[];totals={"3":0,"5":0}
    for slot in slots:
        layer,expert=slot["layer"],slot["expert"]
        for role in ROLES:
            w,source_sha,shard=source_weight(wmap,shards,layer,expert,role)
            for bit in TIERS:
                p,s,rmse=quant_matrix(lib,w,bit)
                prefix=f"L{layer:02d}_E{expert:02d}_{role}_n{bit}"
                parts={"packed":save_unique(dest/(prefix+".packed"),p),
                       "scale":save_unique(dest/(prefix+".scales.f32"),s)}
                physical=sum(x["actual_bytes"] for x in parts.values())
                expected=w.size*bit//8+(w.size//64)*4
                if physical!=expected:raise ValueError("JE01_NATIVE_Q3Q5_ROLE_SIZE_DRIFT")
                totals[str(bit)]+=physical
                records.append({"layer":layer,"expert":expert,"role":role,"bits":bit,
                                "BF16_source_shard":shard,"BF16_tensor_sha256":source_sha,
                                "relative_weight_MSE_to_original_BF16":rmse,
                                "physical_bytes":physical,"files":parts})
        print(json.dumps({"progress_experts":len(records)//6,"total_experts":8}),flush=True)
    if totals!=EXPECTED_REAL_ALL8 or len(records)!=48:
        raise ValueError("JE01_Q3Q5_SAVED_PHYSICAL_BYTES_NOT_AS_PROJECTED")
    cost3=totals["3"]//8;cost5=totals["5"]//8
    assert 2*cost3+6*cost5==EXPECTED_BASELINE_BYTES
    weights={}
    for slot in slots:
        k=(slot["layer"],slot["expert"])
        losses={}
        for bit in TIERS:
            relevant=[x for x in records if (x["layer"],x["expert"],x["bits"])==(*k,bit)]
            if len(relevant)!=3:raise ValueError("JE01_Q3Q5_ROLE_INCOMPLETE")
            losses[str(bit)]=sum(x["relative_weight_MSE_to_original_BF16"] for x in relevant)/3
        weights[k]=losses
    # Pre-NLL pilot candidates; never select on heldout data.
    frequency=tuple((s["layer"],s["expert"]) for s in sorted(
        slots,key=lambda z:(z["calibration_last_token_routed_calls"],z["layer"],z["expert"]))[:2])
    weight=tuple(sorted(weights,key=lambda k:(weights[k]["3"]-weights[k]["5"],k))[:2])
    keys=sorted(weights)
    hashed=sorted(keys,key=lambda k:hashlib.sha256(("JE01_Q3Q5_PRE_LABEL_20261118|"+str(k)).encode()).digest())
    placebo=tuple(hashed[:2])
    if set(placebo) in (set(frequency),set(weight)):
        for index in range(1,7):
            v=tuple(hashed[index:index+2])
            if len(v)==2 and set(v) not in (set(frequency),set(weight)):
                placebo=v;break
    fixed={"route_frequency":sorted(frequency),
           "BF16_weight_MSE_proxy":sorted(weight),
           "fixed_permutation":sorted(placebo)}
    report={"schema":SCHEMA,"status":"REAL_SOURCE_BF16_8EXPERT_48_NATIVE_Q3Q5_ROLE_TIERS_GENERATED",
            "source_original_BF16_identity_SHA256":SOURCE_MODEL_ID,
            "source_verified_immutable_Chat_revision":SOURCE_REV,
            "actual_native_codec":"BEGLIN_gguf_quantize_qNg64_3_5",
            "actual_full8_physical_tier_bytes":totals,
            "equal_original_AF_budget":EXPECTED_BASELINE_BYTES,
            "physical_cost3_per_expert":cost3,"physical_cost5_per_expert":cost5,
            "equal_byte_2x3_6x5_combinations":len(list(itertools.combinations(keys,2))),
            "frozen_before_gold_three_candidate_assignments":fixed,
            "records":records,
            "full_model_NLL":"NOT_MEASURED",
            "original_AF_q4_equivalence":"NOT_VERIFIED",
            "real_27layer_model_forward":"NOT_RUN",
            "heldout_OOD_labels_accessed":False,
            "production_modified":False}
    report["receipt_sha256"]=canonical(report)
    (dest/"JE01_ORIGINAL_BF16_E8_Q3Q5_PHYSICAL_RECEIPT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    try:
        doc=generate(a.output)
        print(json.dumps({"status":doc["status"],"receipt_sha256":doc["receipt_sha256"],
                          "actual_bytes":doc["actual_full8_physical_tier_bytes"],
                          "pre_NLL_candidate_maps":doc["frozen_before_gold_three_candidate_assignments"]},
                         sort_keys=True),flush=True)
    except Exception as ex:
        print(json.dumps({"status":"BLOCKED","reason":str(ex)[:280]}),flush=True)
        raise SystemExit(2)
