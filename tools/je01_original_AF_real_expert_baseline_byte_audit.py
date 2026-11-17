#!/usr/bin/env python3
"""JE01 exact ACTUAL original native AF compressed baseline Expert byte ledger.

These are the real byte offsets/sizes of the existing XOX five-file BEGLIN
DeepSeek-V2-Lite compressed AF model, NOT a hypothetical 6-bit codec cost and
not a claim that alternate Expert precision packing already exists. Verify
full 25KiB layout file SHA and 256-real-router receipt before computing any
eight selected (layer,expert) role payload costs.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib

LAYOUT_SHA="ba3898f3bf366b1aec8471d9896885f101bf64b78c816828ea82bd7d58851c3e"
ROUTER256_SHA="62d0a86956c1a07fa8143efcb52b0b5dea6808767716cd160e05ceb1c0c8a8eb"
AF_BYTES=9814343680
SCHEMA="beglin-je01-actual-original-AF-eight-expert-exact-baseline-bytes/1"
ROLES=("gate_proj","up_proj","down_proj")

def canonical(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def records(layout_path):
    raw=pathlib.Path(layout_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=LAYOUT_SHA:
        raise ValueError("JE01_ORIGINAL_AF_LAYOUT_SHA_CHANGED")
    lines=raw.decode().strip().splitlines()
    if len(lines)!=269:raise ValueError("JE01_ORIGINAL_AF_LAYOUT_TENSOR_COUNT_CHANGED")
    parsed=[]
    for line in lines:
        c=line.split()
        if len(c)!=9 or not c[0] or any(not v.isdigit() for v in c[1:]):
            raise ValueError("JE01_ORIGINAL_AF_TENSOR_LAYOUT_FORMAT_INVALID")
        name=c[0];e,out,inp,ng,off,bsize,scale,bias=map(int,c[1:])
        if e<1 or out<1 or inp<1 or off<0 or bsize<1 or scale<0 or bias<0:
            raise ValueError("JE01_ORIGINAL_AF_TENSOR_LAYOUT_VALUES_INVALID")
        parsed.append({"tensor":name,"e":e,"out":out,"in":inp,"ng":ng,
                       "packed_off":off,"packed_bytes":bsize,"scale_off":scale,"bias_off":bias})
    for i,x in enumerate(parsed):
        nxt=parsed[i+1]["packed_off"] if i+1<len(parsed) else AF_BYTES
        if not x["packed_off"]<x["scale_off"]<x["bias_off"]<nxt:
            raise ValueError("JE01_ORIGINAL_AF_TENSOR_OFFSETS_NOT_MONOTONIC")
        if x["scale_off"]-x["packed_off"]!=x["packed_bytes"]:
            raise ValueError("JE01_ORIGINAL_AF_REAL_PACKED_LENGTH_DRIFT")
        x["scale_bytes"]=x["bias_off"]-x["scale_off"]
        x["bias_bytes"]=nxt-x["bias_off"]
        if x["scale_bytes"]!=x["bias_bytes"] or x["scale_bytes"]!=x["e"]*x["out"]*x["ng"]*4:
            raise ValueError("JE01_ORIGINAL_AF_SCALE_BIAS_F32_BYTES_DRIFT")
    return parsed

def audit(layout_path,router_json_path):
    source=json.loads(pathlib.Path(router_json_path).read_text())
    proof=source.pop("receipt_sha256",None)
    if proof!=ROUTER256_SHA or canonical(source)!=proof:
        raise ValueError("JE01_ORIGINAL_256_ROUTE_SHA_CHANGED")
    if source["calibration_prompt_count"]!=256 or \
       source["status"]!="PASS_256_REAL_ORIGINAL_AF_ROUTER_LAST_PROMPT_ONLY":
        raise ValueError("JE01_ORIGINAL_256_SELECTED_SLOT_SCOPE_CHANGED")
    layout=records(layout_path);byname={x["tensor"]:x for x in layout}
    slots=source["exactly_eight_pilot_slots_frozen_by_unlabeled_calibration_last_token_routing"]
    if len(slots)!=8:raise ValueError("JE01_ORIGINAL_REAL_SELECTED_EIGHT_SLOTS_MISSING")
    output=[];seen=set()
    for item in slots:
        layer,e=item["layer"],item["expert"]
        if not 1<=layer<=26 or not 0<=e<64 or (layer,e) in seen:
            raise ValueError("JE01_ORIGINAL_EXPERT_INDEX_OUT_OF_RANGE")
        seen.add((layer,e))
        roles=[]
        for role in ROLES:
            tensor="model.layers.%d.mlp.switch_mlp.%s"%(layer,role)
            m=byname[tensor]
            if m["e"]!=64 or m["ng"] not in (22,32) or m["out"]*m["in"]!=1408*2048:
                raise ValueError("JE01_ORIGINAL_EXPERT_SHAPE_DRIFT:"+tensor)
            for key in ("packed_bytes","scale_bytes","bias_bytes"):
                if m[key]%64:raise ValueError("JE01_ORIGINAL_EXPERT_BYTE_BOUNDARY_DRIFT:"+tensor)
            packed=m["packed_bytes"]//64
            scale=m["scale_bytes"]//64
            bias=m["bias_bytes"]//64
            start_p=m["packed_off"]+e*packed
            start_s=m["scale_off"]+e*scale
            start_b=m["bias_off"]+e*bias
            roles.append({"role":role,"tensor":tensor,"individual_expert_bits":4,
                "packed_weight_bytes":packed,
                "individual_expert_scale_f32_bytes":scale,
                "individual_expert_bias_f32_bytes":bias,
                "total_expert_role_bytes":packed+scale+bias,
                "source_Af_global_byte_regions":{
                    "packed":[start_p,start_p+packed],
                    "scale":[start_s,start_s+scale],
                    "bias":[start_b,start_b+bias]}})
        output.append({"layer":layer,"expert":e,
                       "calibration_last_token_routed_calls":item["real_original_calibration_routed_calls"],
                       "baseline_assumed_native_AF_bits":4,
                       "roles":roles,
                       "actual_packed_baseline_bytes":sum(v["total_expert_role_bytes"] for v in roles)})
    total=sum(x["actual_packed_baseline_bytes"] for x in output)
    return {"schema":SCHEMA,"status":"EIGHT_SELECTED_ORIGINAL_AF_BASELINE_BYTES_AUDITED",
      "source_real_original_256_route_receipt_sha256":ROUTER256_SHA,
      "source_original_AF_layout_SHA256":LAYOUT_SHA,
      "source_original_AF_total_bytes":AF_BYTES,
      "actual_selected_expert_slots":8,
      "real_256_selected_expert_baseline_eight_bytes":total,
      "real_original_baseline_bit_width_per_selected_expert":4,
      "selected_slots":output,
      "real_alternate_2_6_8_precision_byte_artifacts":"NOT_CREATED",
      "real_mixed_expert_2_6_8_bits_end_to_end_NLL":"NOT_MEASURED",
      "cannot_use_formula_to_certify_equal_real_bytes":True,
      "native_reference_model_checkpoint_unchanged":True,
      "production_promotion_allowed":False}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original-layout-af",required=True)
    p.add_argument("--original-frozen256-router-json",required=True)
    a=p.parse_args()
    try:
        report=audit(a.original_layout_af,a.original_frozen256_router_json)
        report["receipt_sha256"]=canonical(report)
        print(json.dumps(report,sort_keys=True,separators=(",",":"),allow_nan=False))
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:270]}))
        raise SystemExit(2)
