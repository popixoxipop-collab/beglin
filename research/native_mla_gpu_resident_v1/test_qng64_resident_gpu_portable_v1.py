#!/usr/bin/env python3
"""EOE-only actual Metal GPU *device-owned* packed qNg64 weight residency QA.

Register compressed 4096x512 group64 n7 planes/F32 scales once as evaluated
MLX GPU arrays. Compare every output with the previous independently tested
single fused Metal attention. Overwrite caller host buffers AFTER registration
to prove resident GPU weights were deep copied and stay unchanged. Then verify
handle invalidation, malformed input rejection, active allocator memory, and
repeated-call timings. Synthetic input only; not a serving TPOT certificate.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["VECLIB_MAXIMUM_THREADS"]="1"

import numpy as np

ROOT=Path(__file__).resolve().parent
NEW=ROOT/"build"/"libbeglin_mla_qng64_resident_weight_v1.dylib"
SRC=ROOT/"mla_qng64_resident_weight_handle_v1.cpp"
ORIG=ROOT/"build"/"libbeglin_mla_qng64_single_fused_v1.dylib"
BASE_C=ROOT/"build"/"libbeglin_qng64_reference.dylib"
RECEIPT=ROOT/"build"/"gpu_device_resident_qng64_synthetic_receipt_v1.json"
SHA={
 SRC:"4128d4d469c0312d0927d190f8971da157780bee3ffc966882e72e479b1891ac",
 ROOT/"mla_qng64_metal_absorbed_staged_v1.cpp":"43ae243c819f125e3f97c768cf0046d1fd6311536e948e1667d079190cbdfbbd",
 ROOT/"mla_qng64_metal_single_fused_v1.cpp":"46292b78e8da1c467e52e6ac24e703e5d9c56cdc81fc6ae892976bb4de87ff33"
}
H=16
K=128
LAT=512
R=64
ROWS=4096
N=7
NG=8
SCALE=np.float32(0.073)
U=ctypes.POINTER(ctypes.c_uint8)
F=ctypes.POINTER(ctypes.c_float)
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def need(ok,why):
    if not ok:raise RuntimeError("BEGLIN_RESIDENT_QNG64_DEVICE_FAIL_CLOSED: "+why)
def ptr(a,t):
    return a.ctypes.data_as(t)
def metric(a,b):
    diff=a.astype(np.float64)-b.astype(np.float64)
    denom=np.linalg.norm(b.astype(np.float64))
    need(np.isfinite(a).all() and np.isfinite(b).all() and denom>1e-10,"source GPU results finite")
    return {"relative_L2":float(np.linalg.norm(diff)/denom),
            "max_abs":float(np.max(np.abs(diff)))}
def main():
    need(not RECEIPT.exists(),"immutable GPU device weight receipt already exists")
    for file,pin in SHA.items():
        need(file.is_file() and sha(file)==pin,"SHA drift: "+file.name)
    need(BASE_C.exists() and NEW.exists() and ORIG.exists(),
         "portable original C quantizer + MLX real binaries missing")
    rng=np.random.default_rng(20261019)
    original_weights=np.ascontiguousarray(rng.normal(0,.013,(ROWS,LAT)),dtype=np.float32)
    num_planes=ROWS*NG*7*8
    planes=np.empty(num_planes,dtype=np.uint8)
    scales=np.empty(ROWS*NG,dtype=np.float32)
    origin=ctypes.CDLL(str(BASE_C))
    quant=origin.qng64_quantize_f32
    quant.restype=ctypes.c_int
    quant.argtypes=(F,ctypes.c_int,ctypes.c_int,ctypes.c_int,U,F)
    need(quant(ptr(original_weights,F),ROWS,LAT,N,ptr(planes,U),ptr(scales,F))==0,
         "source physical qNg64 n7 format contract")
    need(planes.nbytes==1835008 and scales.nbytes==131072,
         "exact source qNg64 group64 bytes")
    query=np.ascontiguousarray(rng.normal(0,.2,(H,K)),dtype=np.float32)
    q_rope=np.ascontiguousarray(rng.normal(0,.3,(H,R)),dtype=np.float32)
    latent=np.ascontiguousarray(rng.normal(0,.25,(64,LAT)),dtype=np.float32)
    rope=np.ascontiguousarray(rng.normal(0,.3,(64,R)),dtype=np.float32)
    native=ctypes.CDLL(str(NEW))
    baseline=ctypes.CDLL(str(ORIG))
    native.beglin_mla_qng64_resident_gpu_available_v1.restype=ctypes.c_int
    need(native.beglin_mla_qng64_resident_gpu_available_v1()==1,"actual Apple GPU unavailable")
    register=native.beglin_mla_qng64_resident_register_v1
    register.restype=ctypes.c_uint64
    register.argtypes=(U,F,ctypes.c_int)
    forward=native.beglin_mla_qng64_resident_forward_v1
    forward.restype=ctypes.c_int
    forward.argtypes=(ctypes.c_uint64,F,F,F,F,ctypes.c_int,ctypes.c_float,F)
    unregister=native.beglin_mla_qng64_resident_unregister_v1
    unregister.restype=ctypes.c_int
    unregister.argtypes=(ctypes.c_uint64,)
    active_count=native.beglin_mla_qng64_resident_active_weights_v1
    active_count.restype=ctypes.c_size_t
    bytes_payload=native.beglin_mla_qng64_resident_payload_bytes_v1
    bytes_payload.restype=ctypes.c_size_t
    bytes_payload.argtypes=(ctypes.c_uint64,)
    active_mem=native.beglin_mla_qng64_resident_mlx_active_memory_v1
    active_mem.restype=ctypes.c_size_t
    malloc_cache=native.beglin_mla_qng64_resident_mlx_allocator_cache_v1
    malloc_cache.restype=ctypes.c_size_t
    baseline_fn=baseline.beglin_mla_qng64_fused_metal_v1
    baseline_fn.argtypes=(U,F,ctypes.c_int,F,F,F,F,ctypes.c_int,ctypes.c_float,F)
    baseline_fn.restype=ctypes.c_int
    before_gpu=int(active_mem())
    before_cached=int(malloc_cache())
    started=time.perf_counter()
    token_id=register(ptr(planes,U),ptr(scales,F),7)
    resident_bind_seconds=time.perf_counter()-started
    need(token_id>0 and active_count()==1 and
         bytes_payload(token_id)==planes.nbytes+scales.nbytes,
         "native GPU weight handle registration failed")
    after_gpu=int(active_mem())
    print(json.dumps({"stage":"GPU_WEIGHT_ALLOCATED",
        "mlX_active_before":before_gpu,"mlx_active_after":after_gpu,
        "expected_packed_scale_bytes":planes.nbytes+scales.nbytes,
        "registry_active":int(active_count())}),flush=True)
    cases=[]
    original_output={}
    for T in (1,2,5,7,16,32,64):
        selected_lat=np.ascontiguousarray(latent[:T],dtype=np.float32)
        selected_rope=np.ascontiguousarray(rope[:T],dtype=np.float32)
        a=np.full((H,K),np.nan,dtype=np.float32)
        b=np.full((H,K),np.nan,dtype=np.float32)
        tr=time.perf_counter()
        rc=forward(token_id,ptr(query,F),ptr(q_rope,F),ptr(selected_lat,F),
                   ptr(selected_rope,F),T,float(SCALE),ptr(a,F))
        resident_seconds=time.perf_counter()-tr
        tb=time.perf_counter()
        baseline_rc=baseline_fn(ptr(planes,U),ptr(scales,F),7,
                   ptr(query,F),ptr(q_rope,F),ptr(selected_lat,F),
                   ptr(selected_rope,F),T,float(SCALE),ptr(b,F))
        baseline_seconds=time.perf_counter()-tb
        if rc!=1 or baseline_rc!=1:
            print(json.dumps({"resident_code":rc,"baseline_code":baseline_rc,
                              "tokens":T,"handle":bool(token_id)},sort_keys=True),flush=True)
        need(rc==1 and baseline_rc==1,"real qNg64 physical Metal output failed")
        errors=metric(a,b)
        need(errors["relative_L2"]<1e-6 and errors["max_abs"]<1e-5,
             "GPU resident packed qNg64 vs original physical nonresident Metal divergence")
        original_output[T]=a.copy()
        cases.append({"T":T,"relative_L2_vs_same_Metal_kernel":errors["relative_L2"],
              "max_abs":errors["max_abs"],
              "resident_weight_handle_once_GPU_wall_s":resident_seconds,
              "one_shot_rebinding_GPU_wall_s":baseline_seconds})
        print(json.dumps({"phase":"RESIDENT_METAL_PERSISTENT_MATRIX_PARITY",
                         "tokens":T,"L2":errors["relative_L2"]}),flush=True)
    # Source host input memory MUST NOT remain authoritative after registration.
    host_copy_planes=planes.copy()
    host_copy_scales=scales.copy()
    planes[:]=np.uint8(255)
    scales[:]=np.float32(0.125)
    post=np.empty((H,K),dtype=np.float32)
    rc=forward(token_id,ptr(query,F),ptr(q_rope,F),ptr(latent[:7].copy(),F),
               ptr(rope[:7].copy(),F),7,float(SCALE),ptr(post,F))
    need(rc==1 and np.array_equal(post,original_output[7]),
         "changing caller-host source incorrectly mutated GPU-resident weights")
    # Negative rejection never attempts a kernel with unauthorized pointers.
    need(register(ptr(host_copy_planes,U),ptr(host_copy_scales,F),6)==0,
         "unsupported bit width registered")
    wrong=host_copy_scales.copy()
    wrong[0]=0.0
    need(register(ptr(host_copy_planes,U),ptr(wrong,F),7)==0,
         "invalid source qNg64 scale registered")
    need(active_count()==1,"ineligible registrations mutated valid GPU ownership")
    rejects=0
    for invalid in (0,token_id+992):
        before=post.copy()
        rc=forward(invalid,ptr(query,F),ptr(q_rope,F),
              ptr(latent[:7].copy(),F),ptr(rope[:7].copy(),F),
              7,float(SCALE),ptr(post,F))
        need(rc==0 and np.array_equal(post,before),
             "unregistered handle accessed device or mutated caller output")
        rejects+=1
    repeated=[]
    for i in range(8):
        zz=np.empty((H,K),dtype=np.float32)
        t=time.perf_counter()
        rc=forward(token_id,ptr(query,F),ptr(q_rope,F),
              ptr(latent[:7].copy(),F),ptr(rope[:7].copy(),F),
              7,float(SCALE),ptr(zz,F))
        elapsed=time.perf_counter()-t
        need(rc==1 and np.array_equal(zz,original_output[7]),
             "same source GPU-resident packed weights not deterministic")
        repeated.append(elapsed)
    need(unregister(token_id)==1 and active_count()==0 and
         bytes_payload(token_id)==0,
         "GPU owned packed weight handle not released")
    post_delete=np.empty((H,K),dtype=np.float32)
    need(forward(token_id,ptr(query,F),ptr(q_rope,F),
         ptr(latent[:7].copy(),F),ptr(rope[:7].copy(),F),7,
         float(SCALE),ptr(post_delete,F))==0,
         "stale GPU packed handle not invalidated")
    need(unregister(token_id)==0,"second deletion accepted")
    after_delete_gpu=int(active_mem())
    receipt={
      "schema":"beglin.qNg64-n7-real-Metal-GPU-device-packed-MLA-opaque-handle/1",
      "status":"PASS_GPU_DEVICE_OWNED_N7_BITPLANES_REUSED_ACROSS_REAL_METAL_MLA_QUERIES",
      "actual_physical_Metal_GPU":True,
      "source_CPP_SHA256":SHA[SRC],
      "compiled_research_GPU_dylib_sha256":sha(NEW),
      "prior_original_fused_CPP_dylib_sha256":sha(ORIG),
      "real_source_checkpoint_used":False,
      "synthetic_qNg64_weights_encoded_with_native_C_quantizer":True,
      "packed_planes_and_FP32_group_scales_GPU_owned_bytes":1835008+131072,
      "resident_backend":"MLX explicit GPU copy and eval held by C++ process-local registry",
      "GPU_MLX_allocator_active_before_register":before_gpu,
      "GPU_MLX_allocator_active_after_register":after_gpu,
      "GPU_MLX_allocator_active_after_unregister":after_delete_gpu,
      "GPU_MLX_allocator_cache_before_register":before_cached,
      "GPU_MLX_allocator_cache_after_unregister":int(malloc_cache()),
      "resident_registration_wall_seconds":resident_bind_seconds,
      "regression_cases":cases,
      "source_host_mutation_after_registration_did_not_change_GPU_results":True,
      "invalid_ID_or_scale_or_bit_rejected_count":rejects+2,
      "GPU_weights_handle_after_cleanup_count":int(active_count()),
      "real_device_resident_repeated_same_7token_GPU_queries_wall_times":repeated,
      "real_device_resident_repeated_same_7token_GPU_queries_median_seconds":float(statistics.median(repeated)),
      "same_Metal_fused_attention_output_matched_original_max_relative_L2":
          max(t["relative_L2_vs_same_Metal_kernel"] for t in cases),
      "source_checkpoint_payload_uploaded_or_modified":False,
      "full_original_model_resident":False,
      "original_DeepSeek_source_quality_validated":False,
      "gpu_residency_benchmark_scope":"one synthetic 4096x512 qNg64 kv_b projection only",
      "full_engine_serving_TPOT_certified":False,
      "production_modified":False
    }
    with RECEIPT.open("x",encoding="utf-8") as f:
        json.dump(receipt,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps({"status":receipt["status"],
        "GPU_active_before_B":before_gpu,"GPU_active_after_register_B":after_gpu,
        "GPU_active_after_unregister_B":after_delete_gpu,
        "host_mutation_survived":True,
        "highest_GPURelL2":receipt["same_Metal_fused_attention_output_matched_original_max_relative_L2"],
        "median_reused_GPU_query_seconds":float(statistics.median(repeated)),
        "receipt_sha256":sha(RECEIPT)},sort_keys=True))

if __name__=="__main__":
    main()
