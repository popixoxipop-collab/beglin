#!/usr/bin/env python3
"""JE01 true original27 routed 2048-d h2-weighted AF4 vs Q4sym/Q5sym FFN
GATE/UP expert GEMV perturbation. 45 actual sample/router-weighted h2
records from 16 training prompts; original 656-gold output is bitwise baseline.
Output quadratic is E[p||DeltaW h||^2] with identity expert-output metric.
This is NOT the actual loss Hessian, Fisher, gradient, role-down or 256-case
activation estimate. No evaluation labels, heldout, promotion, or source edits.
"""
from __future__ import annotations
import argparse,ctypes,hashlib,importlib.util,json,math,struct
from pathlib import Path
import numpy as np
WORK=Path("/Volumes/D50/TailnetCommander/work")
ACT=WORK/"je01_eoe_original27_h2_trace16_20261122_v2"
NATIVE=WORK/"je01_real_router256_native_weight_dp_20261122_v1"
MAGIC=b"JE01_ACTUAL_REAL_AF4_TRAIN16_ROUTED_H2_V1\n"
ACT_SHA="238642649f68d111df3b60f60083a47d3a6074ada881ccb36f813dd606f9a5b4"
ACT_RECEIPT_SHA="42213c246c488dcef8176c468811fe516d542dd367532ea191c21cfd1d17d6ca"
SOURCE_AUDIT_GIT="a16828e9b17db0d16d75946a54fdad943938398c"
NATIVE_DRIVER_GIT="848f0cde9b2ef915cbeb2c6c67fcc50933f43fbc"
ORIGINAL_GOLD_SHA="b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c"
FROZEN={(12,13),(13,22),(13,28),(14,63),(20,17),(21,61),(24,26),(24,60)}
REC_BYTES=8212
def sha(p):
    d=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(2<<20),b""):d.update(b)
    return d.hexdigest()
def canon(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def gitblob(p):
    b=p.read_bytes()
    return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def signed(p,digest):
    if p.is_symlink() or not p.is_file():raise ValueError("JE01_ACTUAL_H2_EVIDENCE_MISSING")
    d=json.loads(p.read_text())
    sig=d.pop("receipt_sha256",None)
    if sig!=digest or canon(d)!=digest:raise ValueError("JE01_ACTUAL_REAL_H2_SIGNED_RECEIPT_SHA_DRIFT")
    return d
def trace_records():
    out=ACT/"run"
    doc=signed(out/"JE01_TRUE_ORIGINAL_ROUTED_H2_TRAIN16_RECEIPT.json",ACT_RECEIPT_SHA)
    if doc["status"]!="JE01_ACTUAL_ORIGINAL27_B1_LASTPROMPT_ROUTED_H2_TRAIN16_TRACE_AND_GOLD_PARITY_PASS" or \
       doc["actual_routed_hidden_records"]!=45 or \
       doc["actual_binary_trace_SHA256"]!=ACT_SHA or \
       doc["actual_gold16_output_sha256"]!=ORIGINAL_GOLD_SHA or \
       doc["source_true_256_training_activation_vectors"]!="NOT_MEASURED" or \
       doc["actual_weight_Fisher_Hessian"]!="NOT_MEASURED" or \
       doc["source_heldout_or_ood_labels_used"] is not False or \
       doc["production_weights_modified"] is not False:
        raise ValueError("JE01_ACTUAL_H2_TRUE_MODEL_RECEIPT_SCHEMA_CHANGED")
    p=out/"native_original_h2.bin"
    if p.is_symlink() or not p.is_file() or sha(p)!=ACT_SHA or p.stat().st_size!=doc["actual_binary_trace_size_bytes"]:
        raise ValueError("JE01_ACTUAL_45_H2_REAL_BINARY_SHA_CHANGED")
    raw=p.read_bytes()
    if not raw.startswith(MAGIC) or (len(raw)-len(MAGIC))//REC_BYTES!=45 or \
       (len(raw)-len(MAGIC))%REC_BYTES:
        raise ValueError("JE01_TRUE_ORIGINAL_H2_RECORD_LAYOUT_CORRUPT")
    records=[];seen=set()
    for i in range(45):
        off=len(MAGIC)+i*REC_BYTES
        case,l,e,pos=struct.unpack_from("<4I",raw,off)
        prob=struct.unpack_from("<f",raw,off+16)[0]
        x=np.frombuffer(raw,dtype="<f4",count=2048,offset=off+20).copy()
        if case>=16 or (l,e) not in FROZEN or (case,l,e) in seen or \
           not 0<prob<=1 or not np.isfinite(x).all():
            raise ValueError("JE01_ACTUAL_ROUTED_H2_DATA_INVALID")
        seen.add((case,l,e))
        records.append({"case":case,"layer":l,"expert":e,"last_prompt_position":pos,
                        "router_probability":float(prob),"actual_h2":x})
    if len(records)!=doc["actual_routed_hidden_records"]:raise ValueError("JE01_H2_COUNT_WRONG")
    return records,doc
def load_actual_native():
    source=NATIVE/"sources"
    af=source/"original_audit.py";native=source/"driver_v2.py"
    for p,git in ((af,SOURCE_AUDIT_GIT),(native,NATIVE_DRIVER_GIT)):
        if p.is_symlink() or not p.is_file() or gitblob(p)!=git:
            raise ValueError("JE01_ACTUAL_ORIGINAL_SOURCE_GIT_DRIFT")
    def import_py(name,p):
        spec=importlib.util.spec_from_file_location(name,p)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod
    return import_py("je01_true_original_affine",af),import_py("je01_true_native_q45",native)
def run(out):
    if out.exists() or out.is_symlink() or out.parent!=WORK:
        raise ValueError("JE01_ACTUAL_H2_ANALYSIS_FRESH_D50_OUTPUT_REQUIRED")
    records,trace=trace_records()
    original,native=load_actual_native()
    af=original.safe_file(original.AF_ROOT,"deepseek_moe_af.bin",
        original.AF_FILE_SHA,9814343680,rehash=True)
    layout=original.read_af_layout()
    weightmap,shards,headers=original.read_bf16_headers()
    libfile=NATIVE/"result"/"je01_native_q4q5.dylib"
    if not libfile.is_file() or libfile.is_symlink():raise ValueError("JE01_TRUE_NATIVE_Q4Q5_C_COMPILER_OUTPUT_MISSING")
    lib=ctypes.CDLL(str(libfile))
    f32=ctypes.POINTER(ctypes.c_float)
    lib.gguf_quantize_q4g64_error_feedback.argtypes=[f32,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    lib.gguf_quantize_qNg64.argtypes=[f32,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    out.mkdir(mode=0o700)
    entries=[]
    for l,e in sorted(FROZEN):
        rs=[r for r in records if (r["layer"],r["expert"])==(l,e)]
        if not rs:continue
        weights=np.array([r["router_probability"] for r in rs],dtype=np.float64)
        x=np.stack([r["actual_h2"] for r in rs]).T.astype(np.float32)
        h_energy=float(np.sum(weights*np.sum(x.astype(np.float64)**2,axis=0)))
        for role in ("gate_proj","up_proj"):
            w_af,_,_,_,src=original.affine4_original(af,layout,l,e,role)
            bf,meta=original.bf16_selected(weightmap,shards,headers,l,e,role)
            w_q4,shaq4,scale4=native.quant(lib,bf,4)
            w_q5,shaq5,scale5=native.quant(lib,bf,5)
            # All are true pretrained FFN weights; x is true post-attention RMSNorm h2.
            af_y=w_af@x
            bf_y=bf@x
            q4_y=w_q4@x
            q5_y=w_q5@x
            def sse(a,b):
                diff=(a.astype(np.float64)-b.astype(np.float64))
                return float(np.sum(weights*np.sum(diff**2,axis=0)))
            orig_energy=float(np.sum(weights*np.sum(af_y.astype(np.float64)**2,axis=0)))
            bf_energy=float(np.sum(weights*np.sum(bf_y.astype(np.float64)**2,axis=0)))
            if min(orig_energy,bf_energy,h_energy)<=0 or not np.isfinite(orig_energy):
                raise ValueError("JE01_TRUE_ACTUAL_INPUT_OR_OUTPUT_ENERGY_INVALID")
            rec={"layer":l,"expert":e,"role":role,
              "actual_routed_prompt_observations":len(rs),
              "source_actual_h2_trace_sha256":ACT_SHA,
              "router_prob_weight_sum":float(weights.sum()),
              "actual_h2_weighted_energy":h_energy,
              "source_original_BF16_role_sha256":meta["original_bf16_tensor_sha256"],
              "original_AF4_packed_sha256":src["af_packed_sha256"],
              "true_BF16_AF4_symmetric_output_energy":bf_energy,
              "original_AF4_output_energy":orig_energy,
              "actual_activation_output_relative_MSE_AF4_vs_BF16":sse(af_y,bf_y)/bf_energy,
              "actual_activation_output_relative_MSE_Q4_vs_BF16":sse(q4_y,bf_y)/bf_energy,
              "actual_activation_output_relative_MSE_Q5_vs_BF16":sse(q5_y,bf_y)/bf_energy,
              "actual_activation_output_relative_MSE_Q4_vs_AF4":sse(q4_y,af_y)/orig_energy,
              "actual_activation_output_relative_MSE_Q5_vs_AF4":sse(q5_y,af_y)/orig_energy,
              "source_q4_native_packed_sha256":shaq4,"source_q5_native_packed_sha256":shaq5,
              "source_q4_native_scale_sha256":scale4,"source_q5_native_scale_sha256":scale5}
            if any(not math.isfinite(v) or v<0 for k,v in rec.items() if k.startswith("actual_activation_output_relative_MSE_")):
                raise ValueError("JE01_REAL_ACTIVATION_WEIGHTED_GEMV_REL_MSE_NONFINITE")
            entries.append(rec)
        print(json.dumps({"layer":l,"expert":e,"actual_train16_routed_h2":len(rs),"gate_up_roles_complete":len(entries)}),flush=True)
    if len(entries)!=16 or len({(v["layer"],v["expert"],v["role"]) for v in entries})!=16:
        raise ValueError("JE01_ORIGINAL_8EXPERT_GATE_UP_ACTUAL_H2_NOT_COMPLETE")
    def mean(k):
        return float(np.mean([v[k] for v in entries]))
    summary={"actual_original27_hidden_records":45,
       "actual_real_FFN_gate_up_role_sensitivity_count":len(entries),
       "per_role_normalized_output_MSE_sqrt_mean_AF4":math.sqrt(mean("actual_activation_output_relative_MSE_AF4_vs_BF16")),
       "per_role_normalized_output_MSE_sqrt_mean_Q4":math.sqrt(mean("actual_activation_output_relative_MSE_Q4_vs_BF16")),
       "per_role_normalized_output_MSE_sqrt_mean_Q5":math.sqrt(mean("actual_activation_output_relative_MSE_Q5_vs_BF16")),
       "source_router_prob_weighted_h2_proxy_available":True,
       "not_measured":["real_down_proj_swiglu_activation","real_256_train_h2_records",
                       "true_model_loss_Fisher","true_model_loss_Hessian","loss_gradient",
                       "heldout_and_OOD_quality","new_27layer_NLL"]}
    doc={"schema":"beglin-je01-true-original27-train16-routed-H2-activation-weighted-FFN-gateup-output-MSE/1",
       "status":"ACTUAL_ORIGINAL27_ROUTED_H2_45_OBSERVATIONS_GATEUP_16ROLE_GEMV_PROXY_MEASURED",
       "source_actual_original_h2_receipt_sha256":ACT_RECEIPT_SHA,
       "source_actual_h2_trace_sha256":ACT_SHA,
       "source_training_prompts":16,"source_real_routed_h2_observations":45,
       "source_actual_original_BF16_4shard_full_SHA_reverified":True,
       "actual_original_AF4_checkpoint_sha256":original.AF_FILE_SHA,
       "actual_original_symmetric_Q4Q5_native_codec":"BEGLIN_gguf_transcode_verified",
       "analysis_scope":"ROUTER_PROB_WEIGHTED_TRUE_PRETRAINED_GATE_UP_INPUT_H2_X_WEIGHT_GEMV_IDENTITY_OUTPUT_METRIC_NOT_TRUE_FISHER",
       "per_role":entries,"summary":summary,
       "source_training_gold_NLL_original_bitwise_same":True,
       "independent_heldout_OOD_accessed":False,"production_modified":False}
    doc["receipt_sha256"]=original.canonical(doc)
    (out/"JE01_TRUE_ORIGINAL_ROUTED_H2_NATIVE_GATEUP_ACTUAL_GEMV_PROXY.json").write_text(
        json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":doc["status"],"receipt_sha256":doc["receipt_sha256"],
       "summary":summary},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    try:run(a.output)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","type":type(e).__name__,
             "reason":str(e)[:390]},sort_keys=True),flush=True)
        raise SystemExit(2)
