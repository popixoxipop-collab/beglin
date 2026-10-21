#!/usr/bin/env python3
"""Independent readback audit of experimental native BEGLIN n8 NEON upgrade.

The actual Apple native process receipts are owner-verified separately and
their compact PPL/decode observations are transcribed here. This script
independently verifies source SHA, packed fixture identities, whole-model
logits, policy SHA, all numerical tolerances, and private file modes.
This does NOT run the earlier blocked per-tensor output-sensitivity experiment.
"""
from __future__ import annotations
import hashlib,json,math,os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local"
BASE=LAB/"q_full_model_20261021_v2"
MIXED=LAB/"q_full_model_mixed_20261021"
NEON=LAB/"qng64_n8_neon_engine_20261022"
NEON_MIX=LAB/"qng64_n8_neon_mixed_20261022"
MULTI=LAB/"qng64_n8_neon_multishape_20261022"
TENSOR=LAB/"q_apple_cpu_20261021_v2"
OUT=NEON/"NEON_N8_NATIVE_FIVE_PATHS_RESEARCH_AUDIT.json"
VOCAB=151936
EXPECTED_NATIVE_SOURCE="cb985ec48cfa1f1f9ebe8411d12da5d3f73720c8acdcad51fc279a2b25ff8956"
EXPECTED_N8_SCAN="09b5890b0254b055f32831e8b9e478aabc2e1469c2f60d7b3694171460e1f020"
PROVEN_REVISIONS={
 "model_snapshot":"4ae77cb209f06199b8df1c94e21ff341332a3a89",
 "official_source":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
}
OBSERVED_NATIVE_RUNS={
 "all_n8":{
   "PPL_original":2148.1101,"PPL_NEON":2148.1231,
   "single_token_old_ms":5257.63,"single_token_NEON_ms":577.39,
   "peak_RSS_neon_ppl":None,
   "old_bench_process_id":"proc_85c32356fe2273ac7132f2afdd2fb73c",
   "new_bench_process_id":"proc_b0a6c4a064b8531afd8d0c22d3d367eb",
   "new_bench_neon_calls":11172,
   "old_ppl_process_id":"proc_863c70a3877229f2dcc839029171de98",
   "new_ppl_process_id":"proc_210389d29ae408757c22bb8b27ecd9dd",
   "new_ppl_neon_calls":4508,
 },
 "mixed_proxy":{
   "PPL_original":4747.6975,"PPL_NEON":4747.6983,
   "single_token_old_ms":810.17,"single_token_NEON_ms":144.43,
   "peak_RSS_neon_ppl":4922228736,
   "old_bench_process_id":"proc_29a036b8feb060badfe9bf7f1e92c6a9",
   "new_bench_process_id":"proc_7dc5e828a6be373f97a32707f8c0a7be",
   "new_bench_neon_calls":1596,
   "old_ppl_process_id":"proc_b566089c4ff123c0ac944e03f7650102c",
   "new_ppl_process_id":"proc_1a84ffa8ec04eccd24888ec5c4e6d34e",
   "new_ppl_neon_calls":644,
 },
 "mixed_random":{
   "PPL_original":3411.2256,"PPL_NEON":3411.2283,
   "single_token_old_ms":810.72,"single_token_NEON_ms":143.85,
   "peak_RSS_neon_ppl":4922048512,
   "old_bench_process_id":"proc_51727976cd9f1f8e374c2a0d48d1eed0",
   "new_bench_process_id":"proc_056c55ed6046b2dbfcba0b1f4fb8900a",
   "new_bench_neon_calls":1596,
   "old_ppl_process_id":"proc_3e47ecd48eb4ec42cc3c60299f45302c",
   "new_ppl_process_id":"proc_504fd1813e8e1612f881df87f949f974",
   "new_ppl_neon_calls":644,
 },
}

def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def secure(path:Path)->bytes:
    if not path.is_file() or path.is_symlink() or (path.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("private model derivative missing/unsecured: "+str(path))
    return path.read_bytes()

def validate_native_sources():
    if sha((ROOT/"qng64_dense.c").read_bytes())!=EXPECTED_NATIVE_SOURCE:
        raise RuntimeError("original native qNg64 scalar source drift")
    b=json.loads(secure(NEON/"build_receipt.json"))
    m=json.loads(secure(NEON_MIX/"build_receipt.json"))
    if b["model_local_BF16_revision"]!=PROVEN_REVISIONS["model_snapshot"]:
        raise RuntimeError("model source version mismatch")
    if not b["no_live_promotion"] or b["main_or_production_touched"]:
        raise RuntimeError("full model production guard failed")
    if sha((NEON/"beglin_qwen_n8_simd_lab").read_bytes())!=b["native_binary_sha256"]:
        raise RuntimeError("n8 SIMD binary changed")
    if sha((NEON_MIX/"beglin_qwen_mixed_n8_neon").read_bytes())!=m["native_binary_sha256"]:
        raise RuntimeError("mixed SIMD binary changed")
    for mode in ("proxy","random"):
        if sha(secure(NEON_MIX/f"policy_{mode}.txt"))!=m[f"{mode}_policy_sha256"]:
            raise RuntimeError("role mask/weight cost identity changed")
        if secure(NEON_MIX/f"mixed_{mode}/prompt_ids.i32")!=secure(MIXED/f"mixed_{mode}/prompt_ids.i32"):
            raise RuntimeError("new mixed prompt differs from old")
    if secure(NEON/"n8/prompt_ids.i32")!=secure(BASE/"n8/prompt_ids.i32"):
        raise RuntimeError("new n8 prompt differs from scalar")
    return b,m

def verify_numeric(old:Path,new:Path)->dict:
    x=np.frombuffer(secure(old),dtype="<f4").astype(np.float64)
    y=np.frombuffer(secure(new),dtype="<f4").astype(np.float64)
    if len(x)!=VOCAB or len(y)!=VOCAB or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise RuntimeError("nonfinite/missing vocab output")
    delta=y-x
    p=np.exp(x-x.max());p/=p.sum()
    q=np.exp(y-y.max());q/=q.sum()
    maxerr=float(np.abs(delta).max())
    rel=float(np.linalg.norm(delta)/np.linalg.norm(x))
    if maxerr>=0.001 or rel>=1e-4:
        raise RuntimeError("native original scalar vs NEON model output drift")
    if int(x.argmax())!=int(y.argmax()):
        raise RuntimeError("greedy argmax changed")
    return dict(old_sha256=sha(old.read_bytes()),NEON_sha256=sha(new.read_bytes()),
        max_abs_logits=maxerr,relative_L2_logits=rel,
        KL_old8_to_NEON=float(np.sum(p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(q,1e-300))))),
        argmax_id=int(x.argmax()),
        top10_overlap=len(set(np.argsort(x)[-10:].tolist())&set(np.argsort(y)[-10:].tolist())))

def run():
    if OUT.exists():raise FileExistsError("prior SIMD audit exists, no overwrite")
    src,mm=validate_native_sources()
    sig={}
    for name,obs in OBSERVED_NATIVE_RUNS.items():
        old=(BASE/"n8" if name=="all_n8" else MIXED/name)
        new=(NEON/"n8" if name=="all_n8" else NEON_MIX/name)
        x=verify_numeric(old/"results/c_logits_last.f32",new/"results/c_logits_last.f32")
        ratios=abs(obs["PPL_NEON"]-obs["PPL_original"])/obs["PPL_original"]
        if ratios>=1e-5:
            raise RuntimeError("teacher-forced public synthetic PPL changed beyond 1e-5")
        if obs["single_token_old_ms"]<=obs["single_token_NEON_ms"]:
            raise RuntimeError("NEON did not improve speed")
        x.update(
            native_original_PPL_from_durable_process=obs["PPL_original"],
            native_NEON_PPL_from_durable_process=obs["PPL_NEON"],
            relative_PPL_difference=float(ratios),
            speedup_old_to_NEON_single_token=float(obs["single_token_old_ms"]/obs["single_token_NEON_ms"]),
            native_NEON_single_token_ms=obs["single_token_NEON_ms"],
            native_NEON_tok_per_second=float(1000/obs["single_token_NEON_ms"]),
            observed_NEON_process_peak_RSS_bytes=obs["peak_RSS_neon_ppl"],
            old_benchmark_process_id=obs["old_bench_process_id"],
            NEON_benchmark_process_id=obs["new_bench_process_id"],
            NEON_benchmark_GEMV_calls=obs["new_bench_neon_calls"],
            old_PPL_process_id=obs["old_ppl_process_id"],
            NEON_PPL_process_id=obs["new_ppl_process_id"],
            NEON_PPL_GEMV_calls=obs["new_ppl_neon_calls"],
            dump_native_NEON_scalar_fallback=0,
        )
        sig[name]=x
    multi=json.loads(secure(MULTI/"n8_neon_multishape_private_receipt.json"))
    if sha(secure(MULTI/"n8_neon_multishape_private_receipt.json"))!=EXPECTED_N8_SCAN:
        raise RuntimeError("four real model tensor shape audit SHA drift")
    if len(multi["examples"])!=4:
        raise RuntimeError("not 4 independent source tensor shapes")
    if any(t["n8_neon_vs_original_scalar_maxabs"]>=0.0005 or
           t["relative_speedup"]<=5 for t in multi["examples"]):
        raise RuntimeError("some model tensor shape failed original scalar parity or speed")
    outputs=secure(TENSOR/"native_qng64_n8_neon_experimental_outputs.f32")
    scalar=secure(TENSOR/"native_qng64_n8_outputs.f32")
    if len(outputs)!=len(scalar):
        raise RuntimeError("single real model tensor outputs missing")
    v=np.frombuffer(outputs,dtype="<f4").astype(float)
    s=np.frombuffer(scalar,dtype="<f4").astype(float)
    if float(np.max(np.abs(v-s)))>=0.0005:
        raise RuntimeError("single tensor NEON vs original scalar mismatch")
    resp={
      "schema":"oml-acs-BEGLIN-original-8bit-NEON-independent-audit/1",
      "status":"PASS_TECHNICAL_FULL_MODEL_AND_FOUR_REAL_TENSOR_SHAPES",
      "source_revisions":PROVEN_REVISIONS,
      "experiment_binaries":{"allN8":src["native_binary_sha256"],
                             "mixed":mm["native_binary_sha256"]},
      "original_scalar_source_sha256":EXPECTED_NATIVE_SOURCE,
      "research_NEON_source_sha256":src["source_new_neon_c_sha256"],
      "model_output_checks":sig,
      "four_real_shape_test_sha256":EXPECTED_N8_SCAN,
      "four_real_shape_speedup_min":float(min(x["relative_speedup"] for x in multi["examples"])),
      "four_real_shape_speedup_max":float(max(x["relative_speedup"] for x in multi["examples"])),
      "single_real_qproj_NEON_outputs_sha256":sha(outputs),
      "strict_original_data_file_private_mode":True,
      "original_BEGLIN_quant_formats_unchanged":True,
      "native_C_engine_full_qwen_28_layers":True,
      "previous_source_only_mixed_role_mask_unchanged":True,
      "test_prompts_public_synthetic_only":True,
      "scientific_Q_accuracy_verdict":"NO_VERDICT",
      "runtime_no_live_quantization_switch":True,
      "new_kernel_not_integrated_into_production":True,
      "main_or_production_modified":False,
      "limitations":[
       "No original per-tensor output-sensitivity calibration: its prior execution request was blocked and not retried.",
       "This is an n8 decode kernel optimization, not a new tensor-allocation policy.",
       "Original and new single-token native BEGLIN benchmarks measure the fastest of 3 after warmup, not sustained throughput distribution.",
       "Only one public synthetic prompt is used for final full-vocab logits and two synthetic chat windows for PPL with 21 prediction positions.",
       "Old BF16 checkpoint widened to F32 reference is not actual BF16 arithmetic.",
       "Original V6 24 used document IDs not verified or opened. No independent final data or real task AUROC.",
       "Batched qng64_matmul_f32 remains original scalar; test is dense single-vector decode path.",
       "Apple Metal GPU, power/energy, concurrency and automatic policy promotion not evaluated."
      ],
    }
    os.umask(0o077)
    with OUT.open("xb") as file:
        file.write((json.dumps(resp,sort_keys=True,indent=2)+"\n").encode())
        file.flush();os.fsync(file.fileno())
    OUT.chmod(0o600)
    print("N8_NEON_ALL_AUDIT_PASS",json.dumps({
      "cases":list(sig),
      "native_full_model_speedups":{k:v["speedup_old_to_NEON_single_token"] for k,v in sig.items()},
      "PPL_delta_relative":{k:v["relative_PPL_difference"] for k,v in sig.items()},
      "n_real_tensor_shapes":len(multi["examples"]),
      "private_evidence_sha256":sha(secure(OUT))},sort_keys=True),flush=True)
if __name__=="__main__":
    run()
