#!/usr/bin/env python3
"""JE01 real 256 TRAIN prompt activation-conditioned FFN precision DP.

This is an independent source-aware sensitivity audit using actual original27
Metal h2 2048-d vectors for native gate/up, actual original SwiGLU 1408-d
vectors for down, 8 selected Experts and source BF16/AF4/native Q4/Q5.
First 16 previously seen pilot prompts are parity controls only; choose policy
using CALIBRATION PROMPTS 16..255, not repeated gold16. No labels, no
Fisher/Hessian/gradient or model NLL, no production changes.
"""
from __future__ import annotations
import argparse,ctypes,hashlib,importlib.util,json,math,struct
from pathlib import Path
import numpy as np

WORK=Path("/Volumes/D50/TailnetCommander/work")
ACT=WORK/"je01_real_original256_activation_20261122_v1"
NATIVE=WORK/"je01_real_router256_native_weight_dp_20261122_v1"
MAGIC=b"JE01_REAL_ORIGINAL_AF4_CAL256_ACTUAL_H2_SWIGLU_V1\n"
BYTES_PER_RECORD=16+4+(2048+1408)*4
AF_GIT="a16828e9b17db0d16d75946a54fdad943938398c"
NATIVE_GIT="848f0cde9b2ef915cbeb2c6c67fcc50933f43fbc"
NATIVE_BIN_SHA="77fd53111cf8316c9d9186f21ace65bccaf3c6fb1aba54c43c698076d5dc914d"
AF_SHA="9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
ROUTER_SHA="62d0a86956c1a07fa8143efcb52b0b5dea6808767716cd160e05ceb1c0c8a8eb"
FROZEN={(12,13),(13,22),(13,28),(14,63),(20,17),(21,61),(24,26),(24,60)}
ROLES=("gate_proj","up_proj","down_proj")
COSTS={"AF4":1802240,"Q4":1622016,"Q5":1982464}
BUDGET=43253760
SCHEMA="beglin-je01-actual-original27-cal240-clean-H2-SwiGLU-native-24-role-activation-relative-error-exactbyte-DP/1"
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for buf in iter(lambda:f.read(1<<20),b""):h.update(buf)
    return h.hexdigest()
def canonical(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def gitblob(p):
    raw=p.read_bytes();return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
def load():
    p=ACT/"run256"/"JE01_ORIGINAL27_CAL256_TRUE_H2_SWIGLU_RECEIPT.json"
    if not p.is_file() or p.is_symlink():raise ValueError("JE01_TRUE_CAL256_ACTIVATION_RECEIPT_NOT_FINISHED")
    seal=json.loads(p.read_text());claimed=seal.pop("receipt_sha256",None)
    if claimed!=canonical(seal) or \
       seal["status"]!="JE01_TRUE_ORIGINAL27_256TRAIN_682ROUTED_H2_SWIGLU_MODEL_FORWARD_PASS" or \
       seal["actual_original_routed_hidden_and_swiglu_records"]!=682 or \
       seal["total_unique_original_training_prompts"]!=256 or \
       seal["native_original_FFN_input_H2_dimension"]!=2048 or \
       seal["native_original_FFN_down_SwiGLU_dimension"]!=1408 or \
       seal["original_27layer_MLX_native_binary_sha256"]!=NATIVE_BIN_SHA or \
       seal["source_original_router256_receipt_sha256"]!=ROUTER_SHA or \
       seal["original_AF4_full_checkpoint_sha256"]!=AF_SHA or \
       seal["independent_holdout_OOD_gold_labels_accessed"] or seal["production_modified"]:
        raise ValueError("JE01_NATIVE_REAL256_ACTIVATION_EVIDENCE_SHA_DRIFT")
    if len(seal["per_chunk"])!=16:raise ValueError("JE01_CAL256_TRUE16_SOURCE_CHUNKS_NOT_COMPLETE")
    routed={pair:[] for pair in FROZEN};seen=set();fullcount=0
    for blk,chunk in enumerate(seal["per_chunk"]):
        path=ACT/"run256"/("chunk"+str(blk).zfill(2))/"actual_h2_swiglu.bin"
        if path.is_symlink() or not path.is_file() or sha(path)!=chunk["real_binary_sha256"] or \
           path.stat().st_size!=chunk["binary_file_bytes"] or chunk["chunk"]!=blk:
            raise ValueError("JE01_REAL256_ORIGINAL_H2_BINARY_SOURCE_SHA_CHANGED")
        raw=path.read_bytes()
        if not raw.startswith(MAGIC) or (len(raw)-len(MAGIC))%BYTES_PER_RECORD:
            raise ValueError("JE01_REAL256_ACTIVATION_TRACE_NATIVE_RECORD_CONTRACT")
        for pos in range(len(MAGIC),len(raw),BYTES_PER_RECORD):
            case,l,e,last=struct.unpack_from("<4I",raw,pos)
            prob=struct.unpack_from("<f",raw,pos+16)[0]
            h=np.frombuffer(raw,dtype="<f4",count=2048,offset=pos+20).copy()
            sw=np.frombuffer(raw,dtype="<f4",count=1408,offset=pos+20+2048*4).copy()
            global_case=blk*16+case
            if case>=16 or (l,e) not in FROZEN or not 0<prob<=1 or not np.isfinite(h).all() or \
               not np.isfinite(sw).all() or (global_case,l,e) in seen:
                raise ValueError("JE01_REAL256_ROUTED_ACTUAL_SAMPLES_INVALID")
            seen.add((global_case,l,e));fullcount+=1
            # First 16 are SOURCE/PARITY controls used in previous pilot.
            # Do not use them to choose or tune new precision allocation.
            if global_case>=16:routed[(l,e)].append((global_case,float(prob),h,sw))
    if fullcount!=682 or len(seen)!=682:
        raise ValueError("JE01_ACTUAL_ORIGINAL256_LASTTOKEN_SELECTED_H2_682_NOT_COMPLETE")
    return routed,claimed,seal
def original():
    p=NATIVE/"sources"/"original_audit.py"
    n=NATIVE/"sources"/"driver_v2.py"
    if any(x.is_symlink() or not x.is_file() for x in (p,n)) or \
       gitblob(p)!=AF_GIT or gitblob(n)!=NATIVE_GIT:
        raise ValueError("JE01_REAL_ORIGINAL_NATIVE_CODEC_SOURCE_SHA_CHANGED")
    def imp(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
    return imp("je01_source_original_BF16",p),imp("je01_actual_q45",n)
def solve(rows,field):
    # Dynamic programming EXACT 43,253,760B: same 24 role total as AF4.
    # One Q4 gives -180224B, one Q5 gives +180224B.
    states={(0,False):(0.0,())}
    for r in rows:
        states_next={}
        for (delta,changed),(risk,plan) in states.items():
            for tier,m in (("AF4",0),("Q4",-1),("Q5",1)):
                key=(delta+m,changed or m!=0)
                score=risk+r[field+"_"+tier]-r[field+"_AF4"]
                candidate=(score,plan+((r["layer"],r["expert"],r["role"],tier),))
                old=states_next.get(key)
                if old is None or candidate<old:states_next[key]=candidate
        states=states_next
    if abs(states[(0,False)][0])>1e-10:
        raise ValueError("JE01_REAL_ACTUAL_INPUT_DP_BASELINE_ZERO_NOT_PRESERVED")
    def detail(v):
        return {"risk_difference":v[0],"source_assignment":[list(x) for x in v[1]],
                "q4_roles":sum(x[3]=="Q4" for x in v[1]),
                "q5_roles":sum(x[3]=="Q5" for x in v[1]),
                "unchanged_AF4_roles":sum(x[3]=="AF4" for x in v[1])}
    return {"best_overall":detail(min(states[(0,False)],states[(0,True)])),
            "best_nonbaseline_exactbyte":detail(states[(0,True)]),
            "number_final_dynamic_states":len(states)}
def run(output):
    if output.exists() or output.is_symlink() or output.parent!=WORK:
        raise ValueError("JE01_RESEARCH_NEW_D50_OUTPUT_ONLY")
    data,trace_sha,trace=load()
    native,helper=original()
    af=native.safe_file(native.AF_ROOT,"deepseek_moe_af.bin",native.AF_FILE_SHA,9814343680,rehash=True)
    layout=native.read_af_layout()
    weightmap,shards,headers=native.read_bf16_headers()
    libfile=NATIVE/"result"/"je01_native_q4q5.dylib"
    if libfile.is_symlink() or not libfile.is_file():
        raise ValueError("JE01_TRUE_SOURCE_NATIVE_C_CODEC_LIBRARY_MISSING")
    lib=ctypes.CDLL(str(libfile));f32=ctypes.POINTER(ctypes.c_float)
    lib.gguf_quantize_q4g64_error_feedback.argtypes=[f32,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    lib.gguf_quantize_qNg64.argtypes=[f32,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    output.mkdir(mode=0o700)
    entries=[];diagonal={}
    for l,e in sorted(FROZEN):
        selected=data[(l,e)]
        if len(selected)<32:
            raise ValueError("JE01_REAL256_EXPERT_LESS_THAN32_CLEAN_UNSEEN_PROMPTS")
        probability=np.array([r[1] for r in selected],dtype=np.float64)
        X=np.stack([r[2] for r in selected],axis=1).astype(np.float32)
        SW=np.stack([r[3] for r in selected],axis=1).astype(np.float32)
        if X.shape!=(2048,len(selected)) or SW.shape!=(1408,len(selected)):
            raise ValueError("JE01_TRUE_ORIGINAL_FFN_ACTIVATION_DIMENSIONS_CHANGED")
        diagonal[f"L{l}E{e}_gateup_h2_second_moment"]=np.average(X.astype(np.float64)**2,axis=1,weights=probability).astype(np.float32)
        diagonal[f"L{l}E{e}_down_swiglu_second_moment"]=np.average(SW.astype(np.float64)**2,axis=1,weights=probability).astype(np.float32)
        for role in ROLES:
            source,_,_,_,afsrc=native.affine4_original(af,layout,l,e,role)
            bf,ref=native.bf16_selected(weightmap,shards,headers,l,e,role)
            q4,_,_=helper.quant(lib,bf,4)
            q5,_,_=helper.quant(lib,bf,5)
            x=SW if role=="down_proj" else X
            # Float64 reductions, native true float32 matvec. This is an
            # output-space diagonal Gauss-Newton (identity output metric)
            # conditional loss proxy: NOT true next-token Hessian/Fisher.
            target=(bf@x).astype(np.float64)
            energy=float(np.sum(probability*np.sum(target*target,axis=0)))
            if not energy>0 or not math.isfinite(energy):
                raise ValueError("JE01_TRUE_BF16_GEMV_OUTPUT_ZERO_OR_INVALID")
            base={"AF4":source,"Q4":q4,"Q5":q5}
            r={"layer":l,"expert":e,"role":role,
               "actual_clean_unseen_training_router_observations":len(selected),
               "routed_probability_mass":float(probability.sum()),
               "actual_input_dimension":x.shape[0],
               "actual_output_dimension":bf.shape[0],
               "actual_original_BF16_tensor_sha256":ref["original_bf16_tensor_sha256"],
               "actual_original_AF4_packed_sha256":afsrc["af_packed_sha256"],
               "original_BF16_activation_output_energy":energy}
            for tier,weights in base.items():
                diff=(weights@x).astype(np.float64)-target
                squared=float(np.sum(probability*np.sum(diff*diff,axis=0)))
                r["route_weighted_output_SSE_"+tier]=squared
                r["route_weighted_output_relative_MSE_"+tier]=squared/energy
            if any(not math.isfinite(r["route_weighted_output_relative_MSE_"+x]) or
                  r["route_weighted_output_relative_MSE_"+x]<0 for x in base):
                raise ValueError("JE01_REAL_ACTUAL_WEIGHTED_FFN_OUTPUT_MSE_NONFINITE")
            entries.append(r)
        print(json.dumps({"layer":l,"expert":e,"actual_clean_routed_H2_SW":len(selected),
                          "roles_done":len(entries)},sort_keys=True),flush=True)
    if len(entries)!=24 or len(diagonal)!=16:
        raise ValueError("JE01_REAL256_GEMV_ALL24_ROLE_MOMENTS_INCOMPLETE")
    npz=output/"JE01_CAL240_ACTUAL_ORIGINAL_FFN_INPUT_DIAGONAL_MOMENTS.npz"
    np.savez_compressed(npz,**diagonal)
    results={key:solve(entries,key) for key in ("route_weighted_output_SSE","route_weighted_output_relative_MSE")}
    d={"schema":SCHEMA,
       "status":"JE01_ACTUAL_ORIGINAL27_CAL240_UNSEEN_FFN_H2_SWIGLU_24ROLE_ACTIVATION_DP_COMPLETE",
       "source_actual_training_original27_256_h2_sw_trace_receipt_sha256":trace_sha,
       "source_original_router256_receipt_sha256":ROUTER_SHA,
       "source_full_original_AF4_sha256":native.AF_FILE_SHA,
       "true_original_Chat_BF16_four_shard_SHA_reverified":True,
       "actual_original_256_training_prompts_all_capture":True,
       "training_first16_prior_pilot_excluded_from_precision_selection":True,
       "training_unseen240_prompt_count":240,
       "observed_total_original256_routed_activation_records":682,
       "role_count":24,"selected_frozen_expert_count":8,
       "observed_FFN_input_dimensions":{"gate_up":2048,"down_SwiGLU":1408},
       "observed_source_input_moment_npz_sha256":sha(npz),
       "per_role":entries,"separate_exactbyte_dp_objective_results":results,
       "scientific_scope":"ACTUAL_TRAINED_ORIGINAL_ROUTE_WEIGHTED_GEMV_DIAGONAL_OUTPUT_METRIC_NOT_TRUE_LOSS_FISHER_OR_HESSIAN",
       "true_logprob_gradients":"NOT_MEASURED","full_native_27layer_new_NLL":"NOT_RUN",
       "source_heldout_OOD_labels_used":False,"production_modified":False}
    d["receipt_sha256"]=canonical(d)
    (output/"JE01_ACTUAL_CAL240_FFN_H2_SWIGLU_NATIVE_OUTPUT_DP.json").write_text(
        json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":d["status"],"receipt_sha256":d["receipt_sha256"],
           "objective_summaries":{k:{"best":v["best_overall"]["risk_difference"],
                 "changed_Q4":v["best_overall"]["q4_roles"],
                 "changed_Q5":v["best_overall"]["q5_roles"],
                 "best_nonbaseline":v["best_nonbaseline_exactbyte"]["risk_difference"]}
                    for k,v in results.items()}},sort_keys=True),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    try:run(a.output)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","error_type":type(e).__name__,
           "reason":str(e)[:440]},sort_keys=True),flush=True)
        raise SystemExit(2)
