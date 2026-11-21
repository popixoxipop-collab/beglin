#!/usr/bin/env python3
"""JE01: REAL 256 original last-token Top6 frequency weighted native AF4/Q4/Q5
24-role exact-byte DP. NOT real activation vectors, Fisher, Hessian or NLL.
Original 9.8GB AF4 & 31GB true Chat BF16 SHA checked; 256 unlabeled native
router counts; exact BEGLIN q4g64/qNg64 C codec, per-role source errors.
"""
import argparse,ctypes,hashlib,importlib.util,json,math,subprocess
from pathlib import Path
import numpy as np
WORK=Path("/Volumes/D50/TailnetCommander/work")
ROUTER_DIGEST="62d0a86956c1a07fa8143efcb52b0b5dea6808767716cd160e05ceb1c0c8a8eb"
BUDGET=43253760
SLOTS={(21,61),(24,26),(13,28),(20,17),(12,13),(24,60),(14,63),(13,22)}
ROLES=("gate_proj","up_proj","down_proj")
PRICES={"AF4":1802240,"Q4":1622016,"Q5":1982464}
GIT_BLOBS={"router":"913765968deb284ec9c96a11f86441a8173d90d9",
           "af":"a16828e9b17db0d16d75946a54fdad943938398c",
           "c":"ff739d886f55b5a35843cf82fb55c9cd8f76104f",
           "h":"2f8767868cb5576044c1b20e174855ade13ecf33"}
def canon(o):
    return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def gitblob(p):
    s=p.read_bytes();return hashlib.sha1(b"blob "+str(len(s)).encode()+b"\0"+s).hexdigest()
def load_routes(p):
    d=json.loads(p.read_text());signature=d.pop("receipt_sha256",None)
    if signature!=ROUTER_DIGEST or canon(d)!=signature or d["calibration_prompt_count"]!=256 or \
       d["observed_route_events"]!=6656 or d["observed_routed_expert_ids"]!=39936 or \
       d["unlabeled_calibration_only"] is not True:
        raise ValueError("JE01_REAL_CAL256_FULL_ROUTER_OR_RECEIPT_DRIFT")
    m=d["real256_frequency_matrix"]
    slot=d["exactly_eight_pilot_slots_frozen_by_unlabeled_calibration_last_token_routing"]
    if len(m)!=26 or any(len(x)!=64 for x in m) or len(slot)!=8 or \
       {(x["layer"],x["expert"]) for x in slot}!=SLOTS:
        raise ValueError("JE01_CAL256_SELECTED_SLOT_DRIFT")
    counts={(v["layer"],v["expert"]):v["real_original_calibration_routed_calls"] for v in slot}
    if any(m[l-1][e]!=cnt or cnt<32 for (l,e),cnt in counts.items()):
        raise ValueError("JE01_REAL_CAL256_ROUTER_COUNTER_DRIFT")
    return counts,d
def quant(lib,w,b):
    rows,cols=w.shape;g=cols//64
    packed=np.zeros(w.size*b//8,dtype=np.uint8)
    scales=np.zeros((rows,g),dtype=np.float32)
    args=(w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),rows,cols)
    if b==4:
        lib.gguf_quantize_q4g64_error_feedback(*args,packed.ctypes.data,scales.ctypes.data)
        nib=np.empty(packed.size*2,dtype=np.int16)
        nib[0::2]=packed&15;nib[1::2]=packed>>4
        y=(nib.reshape(rows,g,64)-8)*scales[:,:,None]
    else:
        lib.gguf_quantize_qNg64(*args,b,packed.ctypes.data,scales.ctypes.data)
        bitplanes=np.unpackbits(packed.reshape(rows,g,b,8),axis=-1,bitorder="little")
        codes=np.zeros((rows,g,64),dtype=np.int16)
        for i in range(b):codes|=(bitplanes[:,:,i,:].astype(np.int16)<<i)
        y=(codes-16)*scales[:,:,None]
    if not np.isfinite(y).all() or not np.isfinite(scales).all() or not (scales>0).all():
        raise ValueError("JE01_REAL_NATIVE_QUANTIZATION_NONFINITE")
    return y.reshape(w.shape),hashlib.sha256(packed.tobytes()).hexdigest(),hashlib.sha256(scales.tobytes()).hexdigest()
def mseratio(a,b):
    d=a.astype(np.float64)-b.astype(np.float64);y=b.astype(np.float64)
    return float(np.mean(d*d)/np.mean(y*y))
def dp(rows):
    # One AF4 baseline (1802240 B), Q4 -180224 B, Q5 +180224 B.
    # Exact byte budget iff #Q4 = #Q5. This solves all 3^24 assignments.
    states={(0,False):(0.0,())}
    for row in rows:
        cost=row["routed_cal256_last_token_count"]
        base=row["relative_MSE_AF4"]
        nxt={}
        for (diff,changed),(risk,plan) in states.items():
            for tier,change in (("AF4",0),("Q4",-1),("Q5",1)):
                key=(diff+change,changed or change!=0)
                cand=(risk+cost*(row["relative_MSE_"+tier]-base),
                      plan+((row["layer"],row["expert"],row["role"],tier),))
                old=nxt.get(key)
                if old is None or cand<old:nxt[key]=cand
        states=nxt
    if (0,False) not in states or abs(states[(0,False)][0])>1e-12 or (0,True) not in states:
        raise ValueError("JE01_DP_EXACT_BASELINE_NONBASELINE_NOT_VALID")
    def record(v):
        risk,plan=v
        return {"source_router_weighted_relative_MSE_delta":risk,
                "roles_changed":sum(x[3]!="AF4" for x in plan),
                "q4_roles":sum(x[3]=="Q4" for x in plan),
                "q5_roles":sum(x[3]=="Q5" for x in plan),
                "assignment":[list(x) for x in plan]}
    return {"best_exactbudget_overall":record(min((states[(0,False)],states[(0,True)]))),
            "best_nonbaseline_exactbudget":record(states[(0,True)]),
            "tested_final_states":len(states)}
def run(src,out):
    if out.exists() or out.is_symlink() or out.parent!=WORK or not src.is_dir():
        raise ValueError("JE01_NEW_D50_RESEARCH_PATH_ONLY")
    paths={k:src/name for k,name in {"router":"JE01_ROUTER256.json","af":"original_audit.py",
                                     "c":"gguf_transcode.c","h":"gguf_transcode.h"}.items()}
    for key,p in paths.items():
        if p.is_symlink() or not p.is_file() or gitblob(p)!=GIT_BLOBS[key]:
            raise ValueError("JE01_VERIFIED_RESEARCH_INPUT_GIT_BLOB_CHANGED:"+key)
    counts,router=load_routes(paths["router"])
    spec=importlib.util.spec_from_file_location("je01_actual_bf16",paths["af"])
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    af=mod.safe_file(mod.AF_ROOT,"deepseek_moe_af.bin",mod.AF_FILE_SHA,9814343680,rehash=True)
    layout=mod.read_af_layout()
    m,files,headers=mod.read_bf16_headers() # all full 31GB original shard SHA
    out.mkdir(mode=0o700)
    dylib=out/"je01_native_q4q5.dylib"
    subprocess.run(["/usr/bin/clang","-O2","-std=c11","-fPIC","-dynamiclib",
                    str(paths["c"]),"-o",str(dylib),"-lm"],check=True,timeout=120)
    lib=ctypes.CDLL(str(dylib));f32=ctypes.POINTER(ctypes.c_float)
    lib.gguf_quantize_q4g64_error_feedback.argtypes=[f32,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    lib.gguf_quantize_qNg64.argtypes=[f32,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    records=[]
    for layer,expert in sorted(SLOTS):
        for role in ROLES:
            original,_,_,_,source=mod.affine4_original(af,layout,layer,expert,role)
            bf,reference=mod.bf16_selected(m,files,headers,layer,expert,role)
            q4,hash4,scale4=quant(lib,bf,4)
            q5,hash5,scale5=quant(lib,bf,5)
            v={"layer":layer,"expert":expert,"role":role,
               "routed_cal256_last_token_count":counts[(layer,expert)],
               "true_original_BF16_sha256":reference["original_bf16_tensor_sha256"],
               "original_AF4_packed_sha256":source["af_packed_sha256"],
               "relative_MSE_AF4":mseratio(original,bf),
               "relative_MSE_Q4":mseratio(q4,bf),
               "relative_MSE_Q5":mseratio(q5,bf),
               "Q4_packed_sha256":hash4,"Q4_group_scale_sha256":scale4,
               "Q5_packed_sha256":hash5,"Q5_group_scale_sha256":scale5,
               "coefficient_count":int(bf.size),"physical_bytes":PRICES}
            if any(not math.isfinite(v["relative_MSE_"+b]) for b in PRICES):
                raise ValueError("JE01_REAL_BF16_NONFINITE_MSE")
            records.append(v)
        print(json.dumps({"expert":expert,"layer":layer,"completed_true_source_roles":len(records)}),flush=True)
    if len(records)!=24 or sum(v["coefficient_count"] for v in records)!=69206016 or \
       12*PRICES["Q4"]+12*PRICES["Q5"]!=BUDGET:
        raise ValueError("JE01_REAL_24ROLE_OR_NATIVE_BYTE_COST_DRIFT")
    result=dp(records)
    deltas=[r["routed_cal256_last_token_count"]*(r["relative_MSE_Q4"]-r["relative_MSE_AF4"]) for r in records]
    benefits=[r["routed_cal256_last_token_count"]*(r["relative_MSE_AF4"]-r["relative_MSE_Q5"]) for r in records]
    summary={"all24_Q4_have_more_weight_MSE_than_AF4":all(v>0 for v in deltas),
             "all24_Q5_have_less_weight_MSE_than_AF4":all(v>0 for v in benefits),
             "min_weighted_Q4_excess_MSE":min(deltas),
             "max_weighted_Q5_recovery_MSE":max(benefits),
             "min_one_to_one_transfer_lower_bound":min(deltas)-max(benefits),
             "source_router_counts_min_max":[min(counts.values()),max(counts.values())]}
    doc={"schema":"beglin-je01-true-original-cal256-router-AF4-Q4-Q5-24role-exactbytes-weight-proxy/1",
         "status":"REAL_UNLABELLED_ROUTER256_BF16_ORIGINAL_NATIVE_Q4Q5_EXACTBYTE_DP_COMPLETE",
         "router256_source_receipt_sha256":ROUTER_DIGEST,
         "router256_original_prompt_manifest_sha256":router["source_calibration256_manifest_sha256"],
         "original_AF4_full_SHA256":mod.AF_FILE_SHA,
         "actual_true_source_original_Chat_BF16_full4shard_SHA_verified":True,
         "actual_256_last_token_router_count_observations":256,
         "actual_real_model_selected_expert_roles":24,
         "actual_real_BF16_coefficients":69206016,
         "physical_eight_expert_original_AF_budget":BUDGET,
         "all_original_24role_native_source_measurements":records,
         "exactbyte_dp":result,"summary":summary,
         "not_measured":["hidden_states","FFN_activations","loss_gradients","Fisher","Hessian",
                          "true_all-token_router_sensitivity","new_27layer_NLL"],
         "risk_measure":"last_token_router_count * BF16_normalized_true_weight_MSE_difference",
         "source_calibration_only_labels_used":False,"independent_holdout_OOD_labels_used":False,
         "production_modified":False}
    doc["receipt_sha256"]=canon(doc)
    (out/"JE01_ROUTER256_REAL_NATIVE_Q45_EXACTBYTE_DP.json").write_text(
        json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":doc["status"],"receipt_sha256":doc["receipt_sha256"],
          "dp":result,"summary":summary},sort_keys=True)[:7000],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--sources",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    try:run(a.sources,a.output)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","type":type(e).__name__,
             "reason":str(e)[:340]},sort_keys=True),flush=True)
        raise SystemExit(2)
