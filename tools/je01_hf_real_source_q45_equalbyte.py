#!/usr/bin/env python3
"""JE01 actual pinned original DeepSeek Chat BF16 Q4/Q5 real physical challenger.

Generate 12 native q4g64 symmetric NIBBLE-packed FFN roles using the existing
C codec and original Chat BF16 HTTP206 SHA-pinned tensor ranges; copy 12 true
qNg64 n5 bitplane roles from immutable physical 48-tier artifact. Exactly
43,253,760 bytes. No data labels or model forward, nonconfirmatory postpilot.
"""
from __future__ import annotations
import argparse,ctypes,hashlib,json,os,sys
from pathlib import Path
import requests
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from je01_native_exact_byte_presolve import canonical,read_verified_ledger,validate_ledger,EXPECTED_BASELINE_BYTES
from je01_hf_exact_bf16_range_tiers import obtain_pinned_source
from je01_real_native_q3q5_from_bf16 import source_weight
from je01_selected_bf16_native_artifacts import SHAPES,SOURCE_MODEL_ID,compile_original_quantizer
TIERS_RECEIPT="04ddca2f84367f41098357b5fae905e76c1207ef023d648a9f388aaddb3bb0ca"
MODEL_INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
BF16_REV="85864749cd611b4353ce1decdb286193298f64c7"
EXPECTED_SLOTS=((21,61),(24,26),(13,28),(20,17),(12,13),(24,60),(14,63),(13,22))
Q4_SLOTS={(13,28),(24,60),(13,22),(14,63)}
ROLES=("gate_proj","up_proj","down_proj")
LEDGER=Path(__file__).resolve().parents[1]/"docs/research/JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json"
SCHEMA="beglin-je01-source-pinned-real-BF16-Q4nibble-Q5planes-12-12-exactbudget/1"
Q4_COST=1622016
Q5_COST=1982464

def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def choose_root(base):
    valid=[p for p in (base,base/"je01_real_q3q5") if p.is_dir() and
           not p.is_symlink() and (p/"JE01_ORIGINAL_BF16_E8_Q3Q5_PHYSICAL_RECEIPT.json").is_file()]
    if len(valid)!=1:raise ValueError("JE01_SEALED_ORIGINAL_REAL_Q5_ARTIFACT_ROOT_AMBIGUOUS")
    return valid[0]
def read_original_q5(root,slots):
    root=choose_root(root)
    file=root/"JE01_ORIGINAL_BF16_E8_Q3Q5_PHYSICAL_RECEIPT.json"
    d=json.loads(file.read_text())
    sig=d.pop("receipt_sha256",None)
    if sig!=TIERS_RECEIPT or canonical(d)!=TIERS_RECEIPT or \
       d["status"]!="REAL_SOURCE_BF16_8EXPERT_48_NATIVE_Q3Q5_ROLE_TIERS_GENERATED" or \
       d["source_original_BF16_identity_SHA256"]!=SOURCE_MODEL_ID or \
       d["actual_full8_physical_tier_bytes"]!={"3":30277632,"5":47579136}:
        raise ValueError("JE01_ORIGINAL_Q5_REAL_TIER_SOURCE_SHA_DRIFT")
    expected={(layer,expert,role) for layer,expert in slots for role in ROLES}
    q5={}
    for row in d["records"]:
        if row["bits"]!=5:continue
        key=row["layer"],row["expert"],row["role"]
        if key not in expected or key in q5:raise ValueError("JE01_ORIGINAL_Q5_ROLE_UNKNOWN_OR_DUPLICATE")
        blobs={}
        for name in ("packed","scale"):
            source=row["files"][name]
            p=root/source["filename"]
            if p.parent!=root or p.is_symlink() or not p.is_file() or \
               p.stat().st_size!=source["actual_bytes"]:
                raise ValueError("JE01_Q5_SAVED_FILE_PATH_OR_SIZE_DRIFT")
            data=p.read_bytes()
            if sha(data)!=source["sha256"]:
                raise ValueError("JE01_Q5_SAVED_FILE_SHA_DRIFT")
            blobs[name]=data
        if len(blobs["packed"])+len(blobs["scale"])!=Q5_COST:
            raise ValueError("JE01_ORIGINAL_Q5_PHYSICAL_GROUP64_SIZE_DRIFT")
        q5[key]={"file":blobs,"tensor_sha":row["BF16_tensor_sha256"],
                 "relative_MSE":row["relative_weight_MSE_to_original_BF16"]}
    if set(q5)!=expected:raise ValueError("JE01_Q5_ORIGINAL_24_ROLE_COVERAGE_MISSING")
    return q5

def native_q4(lib,w):
    out,inn=w.shape
    if w.dtype!=np.float32 or not w.flags.c_contiguous or inn%64:
        raise ValueError("JE01_NATIVE_Q4_BF16_INPUT_F32_SHAPE")
    packed=np.zeros(w.size//2,dtype=np.uint8)
    scales=np.zeros((out,inn//64),dtype=np.float32)
    ptr=ctypes.POINTER(ctypes.c_float)
    lib.gguf_quantize_q4g64_error_feedback(w.ctypes.data_as(ptr),out,inn,
         ctypes.c_void_p(packed.ctypes.data),scales.ctypes.data_as(ptr))
    if not np.isfinite(scales).all() or not np.all(scales>0):
        raise ValueError("JE01_REAL_Q4_NATIVE_SCALE_NONFINITE")
    code=np.empty(packed.size*2,dtype=np.uint8)
    code[0::2]=packed&15
    code[1::2]=packed>>4
    y=((code.astype(np.int16).reshape(out,inn//64,64)-8)
       *scales.reshape(out,inn//64,1)).reshape(w.shape)
    delta=y.astype(np.float64)-w.astype(np.float64)
    mse=float(np.mean(delta*delta)/np.mean(w.astype(np.float64)**2))
    if not np.isfinite(mse):raise ValueError("JE01_Q4_REAL_WEIGHT_RECONSTRUCTION_NONFINITE")
    return {"packed":packed.tobytes(),"scale":scales.tobytes()},mse

def build(q5root,out):
    if out.exists() or out.is_symlink() or not out.parent.is_dir():
        raise ValueError("JE01_ONLY_NEW_RESEARCH_OUTPUT")
    slots=validate_ledger(read_verified_ledger(LEDGER))
    fixed=tuple((v["layer"],v["expert"]) for v in slots)
    if fixed!=EXPECTED_SLOTS or len(Q4_SLOTS)!=4 or \
       12*Q4_COST+12*Q5_COST!=EXPECTED_BASELINE_BYTES:
        raise ValueError("JE01_FROZEN_EIGHT_EXPERTS_OR_EQUAL_BYTE_BUDGET_DRIFT")
    old_q5=read_original_q5(q5root,fixed)
    session=requests.Session()
    session.headers["User-Agent"]="BEGLIN-JE01-original-Chat-BF16-q4q5-exactbyte-pilot/1"
    model,wmap,shards=obtain_pinned_source(session)
    if model["index_sha256"]!=MODEL_INDEX_SHA or SOURCE_MODEL_ID!=\
       "36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc":
        raise ValueError("JE01_REAL_SOURCE_BF16_CHAT_INDEX_SHA_CHANGED")
    out.mkdir(mode=0o700)
    lib=compile_original_quantizer(out/"je01_source_q4g64.so")
    lib.gguf_quantize_q4g64_error_feedback.argtypes=[
       ctypes.POINTER(ctypes.c_float),ctypes.c_int,ctypes.c_int,ctypes.c_void_p,
       ctypes.POINTER(ctypes.c_float)]
    lib.gguf_quantize_q4g64_error_feedback.restype=None
    target=out/"JE01_SOURCE_REAL_Q4Q5_12_12_EXACT_BYTE.bin"
    records=[];regions=[];h=hashlib.sha256();q4=q5=0
    with target.open("xb") as fd:
        for layer,expert in fixed:
            for role in ROLES:
                key=layer,expert,role
                bit=4 if (layer,expert) in Q4_SLOTS else 5
                if bit==4:
                    w,source_sha,shard=source_weight(wmap,shards,layer,expert,role)
                    streams,mse=native_q4(lib,w)
                    q4+=1
                else:
                    q=old_q5[key]
                    streams=q["file"]
                    mse=q["relative_MSE"]
                    source_sha=q["tensor_sha"]
                    shard="previous-original-Chat-verified-q5"
                    q5+=1
                length=sum(len(v)for v in streams.values())
                if length!=(Q4_COST if bit==4 else Q5_COST):
                    raise ValueError("JE01_Q45_ACTUAL_FILE_REGION_BYTE_SIZE_DRIFT")
                shapes=SHAPES[role]
                for kind in ("packed","scale"):
                    data=streams[kind]
                    start=fd.tell();fd.write(data);h.update(data);end=fd.tell()
                    regions.append({"layer":layer,"expert":expert,"role":role,
                                    "bits":bit,"kind":kind,"offsets":[start,end],
                                    "sha256":sha(data)})
                records.append({"layer":layer,"expert":expert,"role":role,"bits":bit,
                       "coefficients":int(np.prod(shapes)),
                       "actual_bytes":length,
                       "BF16_original_tensor_SHA256":source_sha,
                       "BF16_shard_reference":shard,
                       "reconstruction_relative_weight_MSE_to_BF16":mse})
            print(json.dumps({"finished_layer":layer,"expert":expert,
                               "roles_completed":len(records)}),flush=True)
        fd.flush();os.fsync(fd.fileno())
    if q4!=12 or q5!=12 or len(regions)!=48 or \
       target.stat().st_size!=EXPECTED_BASELINE_BYTES or \
       sha(target.read_bytes())!=h.hexdigest():
        raise ValueError("JE01_REAL_Q45_PHYSICAL_FILE_BYTES_OR_FULL_SHA_FAIL")
    tuples=sorted(tuple(r["offsets"])for r in regions)
    if tuples[0][0]!=0 or tuples[-1][1]!=EXPECTED_BASELINE_BYTES or \
       any(tuples[i][1]!=tuples[i+1][0] for i in range(47)):
        raise ValueError("JE01_Q45_BYTE_REGION_GAP_OR_OVERLAP")
    tsv=out/"JE01_SOURCE_REAL_Q4Q5_24_ROLES.tsv"
    lines=[]
    for layer,expert in fixed:
        for role in ROLES:
            bit=4 if (layer,expert) in Q4_SLOTS else 5
            outsize,insize=SHAPES[role]
            rr=[r for r in regions if (r["layer"],r["expert"],r["role"])==(layer,expert,role)]
            if len(rr)!=2:raise ValueError("JE01_Q45_OUTPUT_INCOMPLETE_ROLE")
            p=next(r["offsets"] for r in rr if r["kind"]=="packed")
            s=next(r["offsets"] for r in rr if r["kind"]=="scale")
            lines.append(f"{layer}\t{expert}\t{role}\t{bit}\t{outsize}\t{insize}\t{p[0]}\t{p[1]}\t{s[0]}\t{s[1]}")
    tsv.write_text("\n".join(lines)+"\n")
    report={"schema":SCHEMA,"status":"ORIGINAL_REAL_BF16_Q4G64_Q5G64_12EACH_43MB_SHA_PASS",
        "scientific_scope":"POST_NIG_16TRAIN_PILOT_EXPLORATORY_CHALLENGER_NOT_CONFIRMATORY",
        "original_BF16_Chat_source_revision":BF16_REV,
        "original_BF16_fullSHA_provenance":"4-of-4 signed pinned shard X-Linked-Etag+size and exact HTTP206 ranges for selected Q4 roles; Q5 prior signed full-shard proof",
        "original_BF16_Chat_index_sha256":MODEL_INDEX_SHA,
        "source_true_original_pretrained_BF16_id_sha256":SOURCE_MODEL_ID,
        "old_real_Q5_physical_source_receipt":TIERS_RECEIPT,
        "real_selected_Q4_experts":[list(x)for x in sorted(Q4_SLOTS)],
        "q4_role_count":q4,"q5_role_count":q5,
        "exact_original_AF_eight_expert_budget_bytes":EXPECTED_BASELINE_BYTES,
        "actual_full_binary_bytes":target.stat().st_size,
        "actual_full_binary_sha256":h.hexdigest(),
        "real_native_tsv_sha256":sha(tsv.read_bytes()),
        "Q4_native_codec":"BEGLIN_gguf_quantize_q4g64_error_feedback_NIBBLE_group64_symmetric",
        "Q5_native_codec":"BEGLIN_gguf_quantize_qNg64_n5_BITPLANES_group64_symmetric",
        "Q4_MLX_requires_synthetic_bias_minus_eight_scale":True,
        "q4_group_bias_persisted_in_blob":False,
        "tensor_records":records,"physical_regions":regions,
        "real_27layer_gold_NLL":"NOT_MEASURED",
        "heldout_OOD_labels_used":False,
        "production_modified":False}
    report["receipt_sha256"]=canonical(report)
    (out/"JE01_REAL_Q45_EQUALBYTE_RECEIPT.json").write_text(
      json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return report

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--prior-q5-source",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    args=ap.parse_args()
    try:
        res=build(args.prior_q5_source,args.output)
        print(json.dumps({"status":res["status"],"sha256":res["actual_full_binary_sha256"],
           "bytes":res["actual_full_binary_bytes"],"q4":res["q4_role_count"],
           "q5":res["q5_role_count"],"receipt_sha256":res["receipt_sha256"],
           "real_Q4_MSEs":[r["reconstruction_relative_weight_MSE_to_BF16"]
             for r in res["tensor_records"] if r["bits"]==4]},sort_keys=True),flush=True)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:320],
           "exception":type(e).__name__},sort_keys=True),flush=True)
        raise SystemExit(2)
