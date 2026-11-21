#!/usr/bin/env python3
"""JE01 true original DeepSeek 27layer routed FFN-input covariance diagnostic.

True 45 EOE source SHA-pinned original h2 2048-d routing events and native
AF4/BF16 Q4/Q5 role matrices. Compare exact weighted output perturbation
tr(DeltaW C DeltaW^T) with diagonal approximation
tr(DeltaW diag(C) DeltaW^T) where C=sum p_t h_t h_t^T, p_t=actual router.
No gradients, true Fisher, Hessian, true NLL gain, down_proj h2 substitute,
heldout, runtime model edit or production promotion.
"""
from __future__ import annotations
import argparse,ctypes,hashlib,importlib.util,json,math
from pathlib import Path
import numpy as np

WORK=Path("/Volumes/D50/TailnetCommander/work")
RAW_SOURCE=WORK/"je01_real_h2_actweighted_gateup_20261122_v1.py"
ACT_SRC_BLOB="3176273ecfee804a667c9d6238e4c1e50bbd73bc"
PRIOR=WORK/"je01_real_h2_actweighted_gateup_20261122_v1"/"JE01_TRUE_ORIGINAL_ROUTED_H2_NATIVE_GATEUP_ACTUAL_GEMV_PROXY.json"
PRIOR_SHA="d55f4b506076a3a30f73c582ede270c38d41ac12b06190e959ac5a7383157a89"
SCHEMA="beglin-je01-real27-train16-h2-raw-covariance-diagonal-approx-vs-exact-gateup/1"
def gitblob(f):
    s=f.read_bytes()
    return hashlib.sha1(b"blob "+str(len(s)).encode()+b"\0"+s).hexdigest()
def canonical(j):
    return hashlib.sha256(json.dumps(j,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def read_source():
    if RAW_SOURCE.is_symlink() or not RAW_SOURCE.is_file() or gitblob(RAW_SOURCE)!=ACT_SRC_BLOB:
        raise ValueError("JE01_PINNED_TRUE_ORIGINAL_ACTIVATION_GATEUP_SOURCE_SHA_CHANGED")
    spec=importlib.util.spec_from_file_location("je01_physical_routed_h2",RAW_SOURCE)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
def source_prior():
    if PRIOR.is_symlink() or not PRIOR.is_file():
        raise ValueError("JE01_ORIGINAL_REAL_H2_GATEUP_SIGNED_SOURCE_MISSING")
    j=json.loads(PRIOR.read_text())
    claim=j.pop("receipt_sha256",None)
    if claim!=PRIOR_SHA or canonical(j)!=PRIOR_SHA or \
       j["status"]!="ACTUAL_ORIGINAL27_ROUTED_H2_45_OBSERVATIONS_GATEUP_16ROLE_GEMV_PROXY_MEASURED":
        raise ValueError("JE01_SOURCE_16_REAL_GEMV_ROLE_RECEIPT_CHANGED")
    return {(x["layer"],x["expert"],x["role"]):x for x in j["per_role"]}
def output_delta(W,ref,X,weights):
    diff=(W.astype(np.float32,copy=False)-ref.astype(np.float32,copy=False))
    Y=(diff@X).astype(np.float64)
    exact=float(np.sum(weights*np.sum(Y*Y,axis=0)))
    # The uncentered *actual routed input second moment* C=sum_t p h h^T
    # yields exact trace(DeltaW C DeltaW^T). Diagonal approximation discards
    # all off-diagonal correlated input directions. No loss Jacobian here.
    second=np.sum(weights[np.newaxis,:]*(X.astype(np.float64)**2),axis=1)
    diag=float(np.sum(np.sum(diff.astype(np.float64)**2,axis=0)*second))
    return exact,diag
def run(out):
    if out.exists() or out.is_symlink() or out.parent!=WORK:
        raise ValueError("JE01_ONLY_FRESH_D50_RESEARCH_COVARIANCE_OUTPUT")
    h2=read_source()
    observations,receipt=h2.trace_records()
    gold=source_prior()
    if len(observations)!=45 or len(gold)!=16 or len(set((x["case"],x["layer"],x["expert"]) for x in observations))!=45:
        raise ValueError("JE01_ACTUAL_ORIGINAL_ROUTED_H2_45_GROUP_ID_INCONSISTENCY")
    org,nat=h2.load_actual_native()
    af=org.safe_file(org.AF_ROOT,"deepseek_moe_af.bin",org.AF_FILE_SHA,9814343680,rehash=True)
    layout=org.read_af_layout()
    weightmap,shards,heads=org.read_bf16_headers()
    lib=ctypes.CDLL(str(h2.NATIVE/"result"/"je01_native_q4q5.dylib"))
    f32=ctypes.POINTER(ctypes.c_float)
    lib.gguf_quantize_q4g64_error_feedback.argtypes=[f32,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    lib.gguf_quantize_qNg64.argtypes=[f32,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    out.mkdir(mode=0o700)
    result=[]
    for l,e in sorted(h2.FROZEN):
        obs=[x for x in observations if (x["layer"],x["expert"])==(l,e)]
        if not obs:raise ValueError("JE01_ACTUAL_H2_SELECTED_ROUTE_NEVER_HIT")
        X=np.ascontiguousarray(np.stack([x["actual_h2"] for x in obs],axis=1),dtype=np.float32)
        p=np.array([x["router_probability"] for x in obs],dtype=np.float64)
        cov=np.float64(X) if False else None
        for role in ("gate_proj","up_proj"):
            original,_,_,_,_=org.affine4_original(af,layout,l,e,role)
            bf,meta=org.bf16_selected(weightmap,shards,heads,l,e,role)
            q4,_,_=nat.quant(lib,bf,4)
            q5,_,_=nat.quant(lib,bf,5)
            ref=float(np.sum(p*np.sum((bf@X).astype(np.float64)**2,axis=0)))
            if ref<=0 or not math.isfinite(ref):raise ValueError("JE01_ACTUAL_BF16_REAL_OUTPUT_ENERGY_MISSING")
            v={"layer":l,"expert":e,"role":role,
               "actual_routed_last_prompt_count":len(obs),
               "actual_router_weight_sum":float(p.sum()),
               "original_2048d_ffn_h2_vector_sha256":hashlib.sha256(X.tobytes()).hexdigest(),
               "effective_uncentered_input_covariance_rank_max":len(obs),
               "actual_output_bf16_energy":ref,
               "source_original_bf16_role_tensor_sha256":meta["original_bf16_tensor_sha256"],
               "tier_diagnostics":{}}
            for name,W in (("AF4",original),("Q4",q4),("Q5",q5)):
                exact,diag=output_delta(W,bf,X,p)
                known=gold[(l,e,role)]["actual_activation_output_relative_MSE_"+name+"_vs_BF16"]
                if abs(exact/ref-known)>2e-6:
                    raise ValueError("JE01_ORIGINAL_REAL_GEMV_45H2_BIT_SOURCE_MSE_MISMATCH:"+str((l,e,role,name,exact/ref,known)))
                if diag<=0 or exact<=0:
                    raise ValueError("JE01_ORIGINAL_ACTUAL_H2_DIAG_COVARIANCE_NONPOSITIVE")
                v["tier_diagnostics"][name]={
                    "actual_exact_weighted_output_sse":exact,
                    "actual_diagonal_approx_output_sse":diag,
                    "actual_cov_offdiagonal_output_interaction_sse":exact-diag,
                    "actual_exact_relative_output_MSE_to_BF16":exact/ref,
                    "diagonal_approx_to_exact_ratio":diag/exact,
                    "diagonal_approx_relative_error":(diag-exact)/exact}
            result.append(v)
        print(json.dumps({"layer":l,"expert":e,"completed_true_source_roles":len(result)}),flush=True)
    if len(result)!=16 or len({(x["layer"],x["expert"],x["role"]) for x in result})!=16:
        raise ValueError("JE01_REAL_H2_GATEUP_COVARIANCE_16_ROLE_COVERAGE_NOT_COMPLETE")
    summary={}
    for tier in ("AF4","Q4","Q5"):
        errors=[r["tier_diagnostics"][tier]["diagonal_approx_relative_error"] for r in result]
        ratios=[r["tier_diagnostics"][tier]["diagonal_approx_to_exact_ratio"] for r in result]
        summary[tier]={"diagonal_approx_ratio_p50":float(np.median(ratios)),
          "diagonal_approx_ratio_min":min(ratios),
          "diagonal_approx_ratio_max":max(ratios),
          "mean_absolute_relative_diagonal_approx_error":float(np.mean(np.abs(errors))),
          "count_ratio_outside_0p8_to_1p2":sum(v<.8 or v>1.2 for v in ratios),
          "count_diagonal_exceeds_exact":sum(v>1 for v in ratios)}
    doc={"schema":SCHEMA,"status":"ACTUAL_ORIGINAL27_H2_ROUTED_COVARIANCE_DIAG_VS_EXACT_16_GATEUP_REAL_NORM_PASS",
       "source_45_actual_routed_original_h2_SHA256":h2.ACT_SHA,
       "source_original_gold_output_SHA256":h2.ORIGINAL_GOLD_SHA,
       "source_16_role_actual_gemv_receipt_SHA256":PRIOR_SHA,
       "actual_source_original_AF4_SHA256":org.AF_FILE_SHA,
       "source_all_four_BF16_shards_fullsha_rechecked":True,
       "actual_original_routed_h2_events":45,
       "source_training_prompt_cases":16,"actual_2048d_hidden_vectors":45,
       "actual_16_gate_up_role_measurements":16,
       "actual_uncentered_covariance_not_empirical_Fisher":True,
       "calibration_scope":"TRAIN16_ROUTED_EVENT_LAST_PROMPT_ONLY_NOT_256",
       "per_role":result,"summary":summary,
       "not_measured":["true_loss_Fisher","true_loss_Hessian","true_loss_gradients",
          "down_proj_swiglu_activation","256_prompt_activations",
          "heldout128_ood128_or_prod_metrics","new_27_layer_NLL"],
       "independent_heldout_labels_used":False,"production_modified":False}
    doc["receipt_sha256"]=canonical(doc)
    (out/"JE01_REAL_H2_DIAGONAL_VS_FULL_COVARIANCE_GATEUP_20261122.json").write_text(
        json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":doc["status"],"receipt_sha256":doc["receipt_sha256"],
          "summary":summary},sort_keys=True),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    o=ap.parse_args()
    try:run(o.output)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","type":type(e).__name__,
           "reason":str(e)[:420]},sort_keys=True),flush=True)
        raise SystemExit(2)
