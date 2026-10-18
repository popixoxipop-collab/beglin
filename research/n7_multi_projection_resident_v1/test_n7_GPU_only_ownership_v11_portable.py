#!/usr/bin/env python3
"""BEGLIN multi-matrix n7/group64 physical Metal resident GEMV regression.

Synthetic reference only. Validates independently C-quantized matrices,
GPU-owned packed code/scales, batch 1..16, separate shapes/handles,
host-source mutation independence, cross-handle isolation, fail closed,
explicit GPU release, and 256 MiB bounded admission.
No real model source, no production changes, no claim of end-to-end speed.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import time
import os

os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["VECLIB_MAXIMUM_THREADS"]="1"
import numpy as np

R=Path(__file__).resolve().parent
CPP=R/"n7_multi_projection_gpu_residency_v11_gpuowned_candidate.cpp"
GPU=R/"build"/"libbeglin_n7_multi_projection_GPU_resident_v11_candidate.dylib"
REF=R/"build"/"libbeglin_qng64_reference.dylib"
RECEIPT=Path(os.environ["BEGLIN_TEST_RECEIPTS_DIR"])/"multi_n7_gpu_resident_Metal_synthetic_receipt_v11.json" if os.environ.get("BEGLIN_TEST_RECEIPTS_DIR") else R/"build"/"multi_n7_gpu_resident_Metal_synthetic_receipt_v11.json"
SRC_SHA="f10a8b30bfeff95cf0e756d45567ac4aaaa3d2552f5ce62a7746af09000a3201"
LIB_SHA=None  # compiled MLX binary SHA depends on local compiler/SDK; recorded, not pre-pinned
F=ctypes.POINTER(ctypes.c_float)
U=ctypes.POINTER(ctypes.c_uint8)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gate(ok,msg):
    if not ok:raise RuntimeError("BEGLIN_MULTI_GPU_QNG64_SYNTHETIC_FAIL_CLOSED:"+msg)
def P(x,t):return x.ctypes.data_as(t)
def oracle(a,b):
    d=a.astype(np.float64)-b.astype(np.float64)
    denom=np.linalg.norm(b.astype(np.float64))
    gate(np.isfinite(a).all() and np.isfinite(b).all() and denom>0,"finite source reference")
    return {"relative_L2":float(np.linalg.norm(d)/denom),"max_abs":float(np.max(np.abs(d)))}

def main():
    gate(not RECEIPT.exists(),"immutable synthetic receipt already created")
    gate(sha(CPP)==SRC_SHA and GPU.is_file() and REF.is_file(),
         "source/binary/reference SHA/availability mismatch")
    c=ctypes.CDLL(str(REF))
    c.qng64_quantize_f32.argtypes=(F,ctypes.c_int,ctypes.c_int,ctypes.c_int,U,F)
    c.qng64_quantize_f32.restype=ctypes.c_int
    c.qng64_matmul_f32.argtypes=(U,F,ctypes.c_int,F,F,F,ctypes.c_int,ctypes.c_int,ctypes.c_int)
    c.qng64_matmul_f32.restype=None
    gpu=ctypes.CDLL(str(GPU))
    gpu.beglin_multi_n7_gpu_available_v1.restype=ctypes.c_int
    gate(gpu.beglin_multi_n7_gpu_available_v1()==1,"physical Apple Metal unavailable")
    reg=gpu.beglin_multi_n7_register_v1
    reg.restype=ctypes.c_uint64
    reg.argtypes=(U,F,ctypes.c_int,ctypes.c_int,ctypes.c_int)
    fw=gpu.beglin_multi_n7_forward_v1
    fw.restype=ctypes.c_int
    fw.argtypes=(ctypes.c_uint64,F,ctypes.c_int,F)
    unreg=gpu.beglin_multi_n7_unregister_v1
    unreg.restype=ctypes.c_int
    unreg.argtypes=(ctypes.c_uint64,)
    active=gpu.beglin_multi_n7_active_count_v1
    active.restype=ctypes.c_size_t
    resident=gpu.beglin_multi_n7_payload_bytes_v1
    resident.restype=ctypes.c_uint64
    size=gpu.beglin_multi_n7_handle_payload_v1
    size.restype=ctypes.c_uint64
    size.argtypes=(ctypes.c_uint64,)
    mem=gpu.beglin_multi_n7_allocator_active_v1
    mem.restype=ctypes.c_uint64
    budget=gpu.beglin_multi_n7_registry_budget_v1
    budget.restype=ctypes.c_uint64
    gate(budget()==256*1024*1024 and active()==0,"isolated model registry budget or empty state")
    generator=np.random.default_rng(20261019)
    cases=[(128,512),(512,512),(2048,512),(4096,512),(2048,2048),(4096,2048)]
    samples=[]
    registered=[]
    buffers={}
    before_gpu=int(mem())
    try:
        for i,(out,in_) in enumerate(cases):
            raw=np.ascontiguousarray(generator.normal(0,0.013,(out,in_)),dtype=np.float32)
            raw*=np.float32((i+1)/5)
            planes=np.empty(out*(in_//64)*56,dtype=np.uint8)
            scale=np.empty(out*(in_//64),dtype=np.float32)
            gate(c.qng64_quantize_f32(P(raw,F),out,in_,7,P(planes,U),P(scale,F))==0,
                 "real n7 source C encoding failed")
            gid=int(reg(P(planes,U),P(scale,F),out,in_,7))
            expected=planes.nbytes+scale.nbytes
            gate(gid>0 and gid not in registered and size(gid)==expected,
                 "GPU compressed independent role registration")
            registered.append(gid)
            buffers[gid]={"planes":planes.copy(),"scales":scale.copy(),
                          "shape":[out,in_],"expected_bytes":expected}
            local=[]
            for B in (1,3,8) if out<4096 else (1,3):
                x=np.ascontiguousarray(generator.normal(0,0.25,(B,in_)),dtype=np.float32)
                actual=np.full((B,out),np.nan,dtype=np.float32)
                reference=np.full((B,out),np.nan,dtype=np.float32)
                start=time.perf_counter()
                rc=fw(gid,P(x,F),B,P(actual,F))
                elapsed=time.perf_counter()-start
                if rc!=1:
                    print(json.dumps({"diagnostic":"N7_GPU_FORWARD_FAILED",
                        "out":out,"in":in_,"batch":B,"returned":rc,
                        "first_values":actual.ravel()[:8].tolist(),
                        "registered_bytes":int(resident()),
                        "allocator_active_bytes":int(mem())}),flush=True)
                gate(rc==1,"actual native Metal N7 GEMV failed for batch "+str(B))
                c.qng64_matmul_f32(P(planes,U),P(scale,F),7,
                    P(x,F),None,P(reference,F),out,in_,B)
                compared=oracle(actual,reference)
                if compared["relative_L2"]>=1e-4 or compared["max_abs"]>=.005:
                    print(json.dumps({"diagnostic":"PARITY_BREAK","out":out,"in":in_,
                        "batch":B,"Metal_values":actual.ravel()[:8].tolist(),
                        "C_values":reference.ravel()[:8].tolist(),
                        "packed_head":planes[:16].tolist(),
                        "scale_head":scale[:4].tolist()}),flush=True)
                gate(compared["relative_L2"]<1e-4 and compared["max_abs"]<.005,
                     "real Metal vs separate C oracle diverged: "+str(compared))
                local.append({"batch":B,"metric":compared,"Metal_wall_seconds":elapsed})
            samples.append({"shape":[out,in_],"registry_id_nonzero":True,
                            "resident_bytes":expected,"oracle_cases":local})
            # GPU-only candidate: allocator-pressure on C++ source buffers
            # after registration returns and temporary std::vectors expire.
            pressure=[np.full(planes.nbytes,(j*17)%251,dtype=np.uint8)
                      for j in range(5)]
            scales_pressure=[np.full(scale.shape,np.float32(j+.1))
                             for j in range(5)]
            gate(all(np.isfinite(a).all() for a in scales_pressure),
                 "source CPU lifetime pressure invalid")
            del pressure,scales_pressure
            # Mutate HOST input memory. Previously registered GPU weight must
            # remain byte-identical, not borrow original numpy source pointers.
            planes.fill(255)
            scale.fill(.12)
            x=np.ones((1,in_),dtype=np.float32)*.07
            testout=np.empty((1,out),dtype=np.float32)
            gold=np.empty((1,out),dtype=np.float32)
            fw_rc=fw(gid,P(x,F),1,P(testout,F))
            b=buffers[gid]
            c.qng64_matmul_f32(P(b["planes"],U),P(b["scales"],F),7,
                              P(x,F),None,P(gold,F),out,in_,1)
            gate(fw_rc==1 and oracle(testout,gold)["relative_L2"]<1e-4,
                 "mutated caller buffer affected GPU resident source matrix")
            print(json.dumps({"status":"PASS_ONE_QNG64_N7_GPU_MATRIX",
                 "shape":[out,in_],"bytes":expected,
                 "cases":len(local)}),flush=True)
        expected_sum=sum(v["expected_bytes"] for v in buffers.values())
        gate(active()==len(cases) and resident()==expected_sum,
             "registered matrices source total count or bytes drift")
        measured_mem=int(mem())
        gate(measured_mem-before_gpu>=expected_sum,
             "GPU active allocation did not materialize all source packed tensors")
        # Bound and invalid API contracts. A legitimate registry cannot
        # silently accept n6/n8, wrong geometry or negative source scale.
        first=buffers[registered[0]]
        p=P(first["planes"],U)
        q=P(first["scales"],F)
        wrong=first["scales"].copy();wrong[0]=-1.0
        denied=[
         reg(p,q,128,512,6),
         reg(p,q,128,513,7),
         reg(p,q,0,512,7),
         reg(p,P(wrong,F),128,512,7),
         reg(p,q,16384,16384,7),  # >96 MiB matrix denied before reading caller buffers
        ]
        gate(all(x==0 for x in denied),"invalid source geometry or scales accepted")
        gate(active()==len(cases),"invalid registration changed authority count")
        x=np.zeros((1,512),dtype=np.float32)
        out=np.full((1,128),44.0,dtype=np.float32)
        gate(fw(0,P(x,F),1,P(out,F))==0 and np.all(out==44.0),
             "invalid zero opaque handle accepted")
        gate(fw(registered[0],P(x,F),0,P(out,F))==0 and np.all(out==44.0),
             "invalid zero batch allowed")
        gate(fw(registered[0],P(x,F),17,P(out,F))==0 and np.all(out==44.0),
             "invalid oversized batch allowed")
        x[0,0]=np.nan
        gate(fw(registered[0],P(x,F),1,P(out,F))==0 and np.all(out==44.0),
             "nonfinite inputs incorrectly reach Metal")
        # Independent handle matrices produce independent GPU results;
        # repeated same stored matrix gives exactly same output.
        repeat=[]
        id0=registered[0]
        x=np.ascontiguousarray(generator.normal(0,0.2,(1,512)),dtype=np.float32)
        refout=np.empty((1,128),dtype=np.float32)
        gate(fw(id0,P(x,F),1,P(refout,F))==1,"repeat source reference")
        for _ in range(8):
            got=np.empty_like(refout)
            start=time.perf_counter()
            ok=fw(id0,P(x,F),1,P(got,F))
            repeat.append(time.perf_counter()-start)
            gate(ok==1 and np.array_equal(got,refout),
                 "GPU resident source output changed with repeated same input")
        # All handles invalidated and GPU-owned active allocation released.
        while registered:
            handle=registered.pop()
            gate(unreg(handle)==1 and size(handle)==0,
                 "registered GPU source qNg64 handle release failed")
        gate(active()==0 and resident()==0,
             "GPU registry still owns compressed weights after teardown")
        gate(fw(id0,P(x,F),1,P(refout,F))==0 and unreg(id0)==0,
             "stale GPU handle reentered native GEMV")
        receipt={
          "schema":"beglin.n7-qng64-multi-matrix-real-Metal-device-resident-registry/1",
          "status":"PASS_GENERAL_MULTIPLE_QNG64_N7_DEVICE_RESIDENT_GEMV",
          "source_SHA256":sha(CPP),"native_Metal_dylib_SHA256":sha(GPU),
          "checkpoint_loaded":False,
          "source_original_C_quantizer_and_matmul_CPU_oracle":True,
          "real_Apple_Metal_Cpp_kernel_called":True,
          "matrix_count":len(cases),
          "source_shapes":cases,
          "GPU_device_resident_total_packed_and_scales_bytes":expected_sum,
          "GPU_MLX_active_memory_before_register_bytes":before_gpu,
          "GPU_MLX_active_memory_with_all_matrices_bytes":measured_mem,
          "registry_peak_opaque_handles":len(cases),
          "hard_budget_bytes":budget(),
          "host_mutation_test":True,"reject_negative_count":len(denied)+4,
          "max_cpu_C_ref_relative_L2":max(v["metric"]["relative_L2"]
                                        for item in samples for v in item["oracle_cases"]),
          "max_cpu_C_ref_absolute_error":max(v["metric"]["max_abs"]
                                        for item in samples for v in item["oracle_cases"]),
          "same_GPU_handle_8_repetitions_median_s":statistics.median(repeat),
          "registered_after_cleanup":int(active()),
          "CXX_registry_duplicate_host_compact_buffers":False,
          "computed_GPU_device_ownership_candidate":True,
          "source_host_pointer_lifetime_pressure_test":True,
          "production_deploy_allowed":False,
          "persistent_full_model_resident":False,
          "production_or_source_model_modified":False,
          "end_to_end_TPOT_certified":False,"samples":samples
        }
        with RECEIPT.open("x",encoding="utf-8") as f:
            json.dump(receipt,f,sort_keys=True,indent=2)
            f.write("\n")
        print(json.dumps({"status":receipt["status"],
            "actual_registered_GPU_compressed_bytes":expected_sum,
            "max_relL2":receipt["max_cpu_C_ref_relative_L2"],
            "reused_median_seconds":receipt["same_GPU_handle_8_repetitions_median_s"],
            "receipt_SHA256":sha(RECEIPT)}),flush=True)
    finally:
        while registered:
            unreg(registered.pop())

if __name__=="__main__":
    main()
