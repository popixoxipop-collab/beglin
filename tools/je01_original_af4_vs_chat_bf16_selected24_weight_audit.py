#!/usr/bin/env python3
"""JE01 original AF4 vs original DeepSeek-V2-Lite-Chat BF16 actual weight audit.

Compare all frozen 8 Expert × 3 role matrices. Read source-only pinned full
safetensors and original 9.8GB compressed AF4; independently unpack true
affine nibble, scale AND bias, match original tensor orientation and SHA.
Evidence is tensor agreement, not original BF16 file identity nor target NLL.
No model runs, no labels and no production writes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path

import numpy as np

AF_ROOT = Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015/native_moe_base_source_copy/weights_moe")
BF_ROOT = Path("/Volumes/D50/deepseek_v2lite_bf16_safetensors")
OUTPUT_PARENT = Path("/Volumes/D50/TailnetCommander/work")
SCHEMA = "beglin-je01-original-AF4-vs-Chat-BF16-selected8-24role-actual-weight-dequant/1"
AF_FILE_SHA = "9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
AF_LAYOUT_SHA = "ba3898f3bf366b1aec8471d9896885f101bf64b78c816828ea82bd7d58851c3e"
INDEX_SHA = "d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
CONFIG_SHA = "f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d"
SHARD_SHA = {
    "model-00001-of-000004.safetensors": ("a314b79a30810b1038f9db08345178e16fb5742aa266c4a3dc05054c59800082",8594887408),
    "model-00002-of-000004.safetensors": ("b0bd088feada0af954614da3a5b179c4d4d5d8bb2482ec83194d101f3fe91796",8591757448),
    "model-00003-of-000004.safetensors": ("d31b192e488f0b3a04e007d842c3c9ba5783d944cb1cc27402fe8b8bb837c490",8590718520),
    "model-00004-of-000004.safetensors": ("304499be0ea06bf8a3937d7a3cc515bc3738dbd288bdd0bbc1b2d3560fce4930",5636263200),
}
FROZEN_SLOTS=((21,61),(24,26),(13,28),(20,17),(12,13),(24,60),(14,63),(13,22))
ROLES=("gate_proj","up_proj","down_proj")
SHAPES={"gate_proj":(1408,2048),"up_proj":(1408,2048),"down_proj":(2048,1408)}

def sha_file(path):
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda:f.read(2<<20),b""):
            digest.update(buf)
    return digest.hexdigest()

def canonical(doc):
    return hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def safe_file(root,name,expected_sha=None,expected_size=None,rehash=True):
    path=root/name
    if (root.is_symlink() or path.is_symlink() or
        not path.is_file() or path.resolve().parent != root.resolve()):
        raise ValueError("JE01_ORIGINAL_SOURCE_PATH_MISSING_OR_SYMLINK:"+name)
    if expected_size is not None and path.stat().st_size!=expected_size:
        raise ValueError("JE01_ORIGINAL_SOURCE_SIZE_DRIFT:"+name)
    if expected_sha and rehash and sha_file(path)!=expected_sha:
        raise ValueError("JE01_ORIGINAL_SOURCE_COMPLETE_SHA_CHANGED:"+name)
    return path

def read_af_layout():
    file=safe_file(AF_ROOT,"layout_af.txt",AF_LAYOUT_SHA)
    layout={}
    for line in file.read_text().splitlines():
        bits=line.split()
        if len(bits)!=9:raise ValueError("JE01_ORIGINAL_AF_LAYOUT_FIELDS")
        name=bits[0]
        if name in layout:raise ValueError("JE01_ORIGINAL_AF_DUPLICATE_TENSOR")
        vals=tuple(map(int,bits[1:]))
        layout[name]=vals
    if len(layout)<80:raise ValueError("JE01_ORIGINAL_AF_LAYOUT_UNEXPECTED")
    return layout

def read_bf16_headers():
    ipath=safe_file(BF_ROOT,"model.safetensors.index.json",INDEX_SHA,rehash=True)
    conf=safe_file(BF_ROOT,"config.json",CONFIG_SHA,rehash=True)
    d=json.loads(ipath.read_text())
    arch=json.loads(conf.read_text())
    if d["metadata"]["total_size"]!=31412968448 or arch.get("model_type")!="deepseek_v2" or \
       arch.get("num_hidden_layers")!=27 or arch.get("n_routed_experts")!=64:
        raise ValueError("JE01_BF16_ORIGINAL_MODEL_ARCH_DRIFT")
    paths={}
    headers={}
    for name,(sha,size) in SHARD_SHA.items():
        file=safe_file(BF_ROOT,name,sha,size,rehash=True)
        paths[name]=file
        with file.open("rb") as f:
            sizeheader=struct.unpack("<Q",f.read(8))[0]
            if not 64<=sizeheader<=32000000:
                raise ValueError("JE01_ORIGINAL_BF16_HEADER_SIZE_OUT_OF_RANGE")
            raw=f.read(sizeheader)
            if len(raw)!=sizeheader:raise ValueError("JE01_ORIGINAL_BF16_HEADER_TRUNCATED")
            headers[name]=(json.loads(raw),8+sizeheader)
    return d["weight_map"],paths,headers

def affine4_original(af,layout,layer,expert,role):
    key=f"model.layers.{layer}.mlp.switch_mlp.{role}"
    if key not in layout:raise ValueError("JE01_ORIGINAL_AF4_ROLE_NOT_IN_LAYOUT:"+key)
    E,out,inp,ng,po,pbytes,so,bo=layout[key]
    if E!=64 or (out,inp)!=SHAPES[role] or ng!=inp//64 or \
       pbytes!=E*out*inp//2:
        raise ValueError("JE01_ORIGINAL_AF4_LAYOUT_ROLE_SHAPE_CHANGED:"+key)
    span=out*inp//2
    nscale=out*ng
    with af.open("rb") as f:
        f.seek(po+expert*span)
        pb=f.read(span)
        f.seek(so+expert*nscale*4)
        scale=f.read(nscale*4)
        f.seek(bo+expert*nscale*4)
        bias=f.read(nscale*4)
    if len(pb)!=span or len(scale)!=nscale*4 or len(bias)!=nscale*4:
        raise ValueError("JE01_ORIGINAL_AF4_EXPERT_BYTES_INCOMPLETE")
    codes=np.frombuffer(pb,dtype=np.uint8)
    nib=np.empty(codes.size*2,dtype=np.uint8)
    nib[0::2]=codes&15
    nib[1::2]=codes>>4
    scale=np.frombuffer(scale,dtype="<f4")
    bias=np.frombuffer(bias,dtype="<f4")
    if not np.isfinite(scale).all() or not np.isfinite(bias).all():
        raise ValueError("JE01_ORIGINAL_AF4_GROUP_SCALE_OR_BIAS_NONFINITE")
    # The ORIGINAL affine AF4 stores SIGNED per-group scales. On the
    # real L21/E61 gate, 22,465/45,056 group scales are negative. This
    # is valid: native code computes (nibble*scale + bias) verbatim.
    # C original BEGLIN moe_decode_af: affine (nibble * scale + bias),
    # no LUT enabled in original online gate; group index is col/64.
    dequant=(nib.reshape(out*ng,64).astype(np.float32)
             *scale[:,None]+bias[:,None]).reshape(out,inp)
    return dequant,nib.reshape(out*ng,64),scale,bias,{
       "af_packed_sha256":hashlib.sha256(pb).hexdigest(),
       "af_scales_sha256":hashlib.sha256(scale.tobytes()).hexdigest(),
       "af_biases_sha256":hashlib.sha256(bias.tobytes()).hexdigest(),
       "original_af_offsets":{"packed":[po+expert*span,po+(expert+1)*span],
          "scale":[so+expert*nscale*4,so+(expert+1)*nscale*4],
          "bias":[bo+expert*nscale*4,bo+(expert+1)*nscale*4]}
    }

def bf16_selected(weightmap,paths,headers,layer,expert,role):
    key=f"model.layers.{layer}.mlp.experts.{expert}.{role}.weight"
    name=weightmap.get(key)
    if name not in paths:
        raise ValueError("JE01_CHATSOURCE_SELECTED_TENSOR_NOT_FOUND:"+key)
    header,base=headers[name]
    meta=header.get(key)
    shape=SHAPES[role]
    if not isinstance(meta,dict) or meta.get("dtype")!="BF16" or meta.get("shape")!=list(shape):
        raise ValueError("JE01_BF16_SOURCE_TENSOR_META_CHANGED:"+key)
    off=meta.get("data_offsets")
    if not isinstance(off,list) or len(off)!=2 or any(type(x)!=int for x in off) or \
       off[0]<0 or off[1]-off[0]!=np.prod(shape)*2 or \
       base+off[1]>paths[name].stat().st_size:
        raise ValueError("JE01_BF16_TENSOR_RANGE_CHANGED:"+key)
    with paths[name].open("rb") as f:
        f.seek(base+off[0])
        raw=f.read(off[1]-off[0])
    if len(raw)!=np.prod(shape)*2:raise ValueError("JE01_BF16_TENSOR_BYTES_SHORT")
    float32=(np.frombuffer(raw,dtype="<u2").astype(np.uint32)<<16).view("<f4").reshape(shape)
    if not np.isfinite(float32).all():
        raise ValueError("JE01_BF16_WEIGHT_HAS_NONFINITE")
    return float32,{
      "tensor":key,"source_shard":name,
      "original_bf16_tensor_sha256":hashlib.sha256(raw).hexdigest()
    }

def stats(a,b,nibbles,scale,bias):
    if a.shape!=b.shape or a.shape[1]%64:
        raise ValueError("JE01_COMPARE_MATRIX_SHAPE_WRONG")
    x=a.astype(np.float64,copy=False)
    y=b.astype(np.float64,copy=False)
    delta=x-y
    mse=float(np.mean(delta*delta))
    ref=float(np.mean(y*y))
    if ref<1e-24:raise ValueError("JE01_SOURCE_WEIGHT_REF_ZERO")
    # If the AF4 scale+bias groups were generated from this very BF16
    # tensor, quantizing this BF16 with the EXISTING ORIGINAL affine
    # scale+bias should yield the stored nibble codes for most elements.
    groups=y.reshape(-1,64)
    valid_scale=scale!=0
    if not valid_scale.any():
        raise ValueError("JE01_ORIGINAL_AF4_ALL_GROUP_SCALES_ZERO")
    predicted=np.rint(
        (groups[valid_scale]-bias[valid_scale].astype(np.float64)[:,None])/
         scale[valid_scale].astype(np.float64)[:,None])
    np.clip(predicted,0,15,out=predicted)
    agreement=float(np.mean(predicted.astype(np.uint8)==nibbles[valid_scale]))
    xm=float(np.mean(x));ym=float(np.mean(y))
    cov=float(np.mean((x-xm)*(y-ym)))
    vx=float(np.mean((x-xm)**2));vy=float(np.mean((y-ym)**2))
    corr=cov/math.sqrt(vx*vy) if vx>1e-24 and vy>1e-24 else 0
    source_abs=np.abs(y)
    per_group_delta=np.abs(x.reshape(-1,64)-groups)
    per_group_rmse=np.sqrt(np.mean(per_group_delta*per_group_delta,axis=1))
    return {
     "relative_weight_RMSE_AF4_vs_trueBF16":math.sqrt(mse/ref),
     "relative_weight_MSE_AF4_vs_trueBF16":mse/ref,
     "pearson_weight_correlation":corr,
     "max_absolute_weight_diff":float(np.max(np.abs(delta))),
     "mean_absolute_weight_diff":float(np.mean(np.abs(delta))),
     "original_AF4_nibble_code_agreement_using_BF16_at_AF4_scale_bias":agreement,
     "original_AF4_nibble_code_agreement_tested_fraction":float(np.mean(valid_scale)),
     "original_AF4_group_scale_negative_count":int(np.sum(scale<0)),
     "original_AF4_group_scale_zero_count":int(np.sum(scale==0)),
     "fraction_original_AF4_exact_element_value_equal_BF16":float(np.mean(a==b)),
     "mean_BF16_abs_weight":float(np.mean(source_abs)),
     "original_affine_scale_min":float(np.min(scale)),
     "original_affine_scale_max":float(np.max(scale)),
     "original_affine_bias_min":float(np.min(bias)),
     "original_affine_bias_max":float(np.max(bias)),
     "groupwise_RMSE_p50":float(np.median(per_group_rmse)),
     "groupwise_RMSE_p95":float(np.quantile(per_group_rmse,.95))
    }

def run(out,max_roles):
    if max_roles not in (1,24):raise ValueError("JE01_EXPLICIT_1_OR_24_ROLES_ONLY")
    if out.exists() or out.is_symlink() or not out.parent.is_dir() or out.parent.is_symlink():
        raise ValueError("JE01_NEW_RESEARCH_OUTPUT_REQUIRED")
    layout=read_af_layout()
    af=safe_file(AF_ROOT,"deepseek_moe_af.bin",AF_FILE_SHA,9814343680,rehash=True)
    weightmap,paths,headers=read_bf16_headers()
    roles=[]
    for layer,expert in FROZEN_SLOTS:
      for role in ROLES:
        roles.append((layer,expert,role))
    results=[]
    for count,(layer,expert,role) in enumerate(roles[:max_roles],1):
        a,nibs,scale,bias,original=affine4_original(af,layout,layer,expert,role)
        b,reference=bf16_selected(weightmap,paths,headers,layer,expert,role)
        score=stats(a,b,nibs,scale,bias)
        result={"layer":layer,"expert":expert,"role":role,
                "coefficients":int(b.size),
                **original,**reference,**score}
        results.append(result)
        print(json.dumps({"progress":count,"total":max_roles,"layer":layer,
             "expert":expert,"role":role,"correlation":score["pearson_weight_correlation"],
             "relative_RMSE":score["relative_weight_RMSE_AF4_vs_trueBF16"],
             "same_original_affine_code_agreement":
                score["original_AF4_nibble_code_agreement_using_BF16_at_AF4_scale_bias"]},
             sort_keys=True),flush=True)
    complete=max_roles==24
    doc={
      "schema":SCHEMA,
      "status":"REAL_SOURCE_AF4_VS_CHAT_BF16_ALL_24_ROLE_DEQUANT_COMPARE_COMPLETE"
        if complete else "REAL_SOURCE_AF4_VS_CHAT_BF16_ONE_ROLE_PREFLIGHT_ONLY",
      "original_AF4_file_sha256":AF_FILE_SHA,
      "original_AF4_layout_sha256":AF_LAYOUT_SHA,
      "true_BF16_Chat_index_sha256":INDEX_SHA,
      "true_BF16_Chat_shard_sha256":{name:val[0] for name,val in SHARD_SHA.items()},
      "true_source_weight_family":"DeepSeek-V2-Lite-Chat",
      "source_complete_four_safetensors_sha_check":True,
      "original_AF4_9point8GB_full_file_sha_check":True,
      "original_AF4_decoder":"affine_nibble_times_group_scale_plus_group_bias",
      "true_BF16_decoder":"bf16_payload_u16_leftshift16_to_float32",
      "original_8_experts":[list(pair) for pair in FROZEN_SLOTS],
      "role_comparison_count":len(results),
      "compare_coefficients":sum(v["coefficients"] for v in results),
      "per_role":results,
      "model_full_forward":"NOT_RUN",
      "original_256_real_paired_NLL":"NOT_REPEATED",
      "NIG_vs_original16_gold_NLL_from_prior":"0.014955175691470757_WORSE",
      "source_checkpoint_bitwise_identical":"NOT_APPLICABLE_AF4_IS_LOSSY",
      "proof_exact_quantizer_exporter":"NOT_TESTED",
      "independent_128_128_OOD_labels_used":False,
      "operating_model_modified":False
    }
    if results:
        weights=np.array([v["coefficients"] for v in results],dtype=np.float64)
        totals=weights.sum()
        doc["aggregate_summary"]={
          "weighted_24role_relative_RMSE_mean":float(np.sum(
            weights*np.array([r["relative_weight_RMSE_AF4_vs_trueBF16"] for r in results]))/totals),
          "weighted_24role_nibble_agreement":float(np.sum(
            weights*np.array([r["original_AF4_nibble_code_agreement_using_BF16_at_AF4_scale_bias"] for r in results]))/totals),
          "min_weight_correlation":min(r["pearson_weight_correlation"] for r in results),
          "max_relative_RMSE":max(r["relative_weight_RMSE_AF4_vs_trueBF16"] for r in results)
        }
    doc["receipt_sha256"]=canonical(doc)
    out.write_text(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return doc

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--max-roles",type=int,choices=[1,24],default=1)
    opt=ap.parse_args()
    try:
        d=run(opt.output,opt.max_roles)
        print(json.dumps({"status":d["status"],"receipt_sha256":d["receipt_sha256"],
                          "aggregation":d.get("aggregate_summary")},sort_keys=True),flush=True)
    except Exception as exc:
        print(json.dumps({"status":"BLOCKED","type":type(exc).__name__,
                          "reason":str(exc)[:320]},sort_keys=True),flush=True)
        raise SystemExit(2)
