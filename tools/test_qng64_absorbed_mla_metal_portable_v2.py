#!/usr/bin/env python3
"""Portable BEGLIN native C++/Metal qNg64 absorbed-MLA synthetic stress regression.

Requires macOS/Apple Silicon, Python+numpy+mlx installed, Xcode compiler.
Compiles source files into a temporary directory; never writes model data,
serving configurations or research credentials. No checkpoint dependency.

Only B=1 single-query MLA attention H=16, latent=512, RoPE=64, K/V=128,
context length 1..64; output is original per-head V context, not projected
LLM logits or an end-to-end decoder speed measurement.

Native C packs and expands the **CPU-only** independent oracle;
the native Metal candidate consumes the original packed n7 uint8 bitplanes
and FP32 group scales, never allocating full expanded historical K/V on GPU.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE=Path(__file__).resolve().parent
SRC=HERE/"qng64_absorbed_mla_cpp_metal_v1.cpp"
CPP_PIN="74c705b7cec2a5b8394d9d17226763eaff3fbf44e28ba15a9b5e664d2465b1ff"
H,K,V,R,D=16,128,128,64,512
OUT=H*(K+V)
PLANE_BYTES=OUT*(D//64)*7*8
SCALE_COUNT=OUT*(D//64)

def require(x,why):
    if not x:raise RuntimeError("BEGLIN_MLA_PORTABLE_GPU_STRESS_FAIL_CLOSED: "+why)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ptr(a,typ):
    require(isinstance(a,np.ndarray) and a.flags["C_CONTIGUOUS"],
            "noncontiguous native array")
    return a.ctypes.data_as(ctypes.POINTER(typ))

def f32(a):return np.ascontiguousarray(a,dtype=np.float32)

def build(root,tmp):
    require(platform.system()=="Darwin" and platform.machine()=="arm64",
            "only real Apple ARM64 allowed")
    mlx=importlib.util.find_spec("mlx.core")
    require(mlx is not None and mlx.origin,"installed MLX package missing")
    base=Path(mlx.origin).parent
    header=base/"include"/"mlx"/"mlx.h"
    lib=base/"lib"/"libmlx.dylib"
    require(header.is_file() and lib.is_file(),"matching MLX C++ SDK missing")
    c=root/"qng64_dense.c"
    h=root/"qng64_dense.h"
    require(c.is_file() and h.is_file(),"original BEGLIN C n7 source absent")
    require(sha(SRC)==CPP_PIN,"native Metal source pin drift")
    obj=tmp/"qng64_dense.o"
    cc=tmp/"libbeglin_mla_cpu_oracle.dylib"
    metal=tmp/"libbeglin_qng64_mla_metal.dylib"
    for cmd in (
      ["/usr/bin/clang","-std=c11","-O2","-fPIC","-I",str(root),
       "-c",str(c),"-o",str(obj)],
      ["/usr/bin/clang","-dynamiclib",str(obj),"-o",str(cc)],
      ["/usr/bin/clang++","-std=c++20","-O2","-fPIC","-dynamiclib",
       "-Wno-unused-parameter","-Wno-deprecated-copy",
       "-I",str(base/"include"),str(SRC),"-L",str(base/"lib"),"-lmlx",
       "-Wl,-rpath,"+str(base/"lib"),"-o",str(metal)],
    ):
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
        require(p.returncode==0,"compile failed "+(p.stderr or "")[-1400:])
    return cc,metal

def configure(cc,metal):
    c=ctypes.CDLL(str(cc))
    m=ctypes.CDLL(str(metal))
    u8=ctypes.POINTER(ctypes.c_uint8)
    fp=ctypes.POINTER(ctypes.c_float)
    z=ctypes.c_size_t
    d=ctypes.POINTER(ctypes.c_double)
    c.qng64_packed_bytes.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int]
    c.qng64_packed_bytes.restype=z
    c.qng64_quantize_f32.argtypes=[fp,ctypes.c_int,ctypes.c_int,ctypes.c_int,u8,fp]
    c.qng64_quantize_f32.restype=ctypes.c_int
    c.qng64_matmul_f32.argtypes=[u8,fp,ctypes.c_int,fp,fp,fp,
                                ctypes.c_int,ctypes.c_int,ctypes.c_int]
    c.qng64_matmul_f32.restype=None
    m.beglin_qng64_mla_absorbed_available_v1.restype=ctypes.c_int
    require(m.beglin_qng64_mla_absorbed_available_v1()==1,
            "actual Apple Metal unavailable")
    f=m.beglin_qng64_mla_absorbed_probe_v1
    f.argtypes=[u8,z,fp,z,fp,z,fp,z,fp,z,fp,z,
                ctypes.c_int,ctypes.c_float,fp,z,fp,z,d]
    f.restype=ctypes.c_int
    return c,f,metal

def pack(c,w):
    require(tuple(w.shape)==(OUT,D) and np.isfinite(w).all(),"synthetic source W")
    nbytes=c.qng64_packed_bytes(OUT,D,7)
    require(nbytes==PLANE_BYTES,"exact n7 group64 bitplane size")
    planes=np.zeros(nbytes,dtype=np.uint8)
    scales=np.zeros(SCALE_COUNT,dtype=np.float32)
    rc=c.qng64_quantize_f32(ptr(w,ctypes.c_float),OUT,D,7,
                             ptr(planes,ctypes.c_uint8),ptr(scales,ctypes.c_float))
    require(rc==0 and np.isfinite(scales).all() and np.all(scales>0),
            "original C compressed n7 quantizer fail")
    return planes,scales

def oracle(c,planes,scales,qn,qr,latent,kpe,scale):
    T=int(latent.shape[0])
    expanded=np.empty((T,OUT),dtype=np.float32)
    c.qng64_matmul_f32(ptr(planes,ctypes.c_uint8),ptr(scales,ctypes.c_float),7,
       ptr(latent,ctypes.c_float),None,ptr(expanded,ctypes.c_float),OUT,D,T)
    expanded=expanded.reshape(T,H,256).transpose(1,0,2).astype(np.float64)
    k0=expanded[:,:,:K]
    value=expanded[:,:,K:]
    scores=(np.einsum("hr,htr->ht",qn.astype(np.float64),k0)
            +np.einsum("hr,tr->ht",qr.astype(np.float64),kpe.astype(np.float64)))
    scores*=float(scale)
    e=np.exp(scores-np.max(scores,axis=1,keepdims=True))
    prob=e/np.sum(e,axis=1,keepdims=True)
    result=np.einsum("ht,htv->hv",prob,value)
    return f32(scores),f32(result)

def run(f,planes,scales,qn,qr,latent,kpe,scale,cap=None):
    T=int(latent.shape[0])
    out=np.full((H,V),-404.0,dtype=np.float32)
    scores=np.full((H,T),-405.0,dtype=np.float32)
    measured=ctypes.c_double(-1)
    args=[ptr(planes,ctypes.c_uint8),planes.nbytes,
          ptr(scales,ctypes.c_float),scales.size,
          ptr(qn,ctypes.c_float),qn.size,
          ptr(qr,ctypes.c_float),qr.size,
          ptr(latent,ctypes.c_float),latent.size,
          ptr(kpe,ctypes.c_float),kpe.size,
          T,scale,ptr(out,ctypes.c_float),out.size,
          ptr(scores,ctypes.c_float),scores.size,ctypes.byref(measured)]
    if cap is not None:
        for i,v in cap.items():args[i]=v
    ok=f(*args)
    return ok,out,scores,measured.value

def metric(a,b):
    a=a.astype(np.float64).ravel()
    b=b.astype(np.float64).ravel()
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
            "malformed values in GPU math")
    diff=a-b
    return float(np.linalg.norm(diff)/max(np.linalg.norm(b),1e-30))

def invalid(f,planes,scales,qn,qr,latent,kpe,scale):
    u8=ctypes.POINTER(ctypes.c_uint8)
    fp=ctypes.POINTER(ctypes.c_float)
    bad=[
      ("null-planes",{0:u8()}),
      ("wrong-packed-size",{1:planes.nbytes-1}),
      ("wrong-scale-count",{3:scales.size+1}),
      ("wrong-qnope-count",{5:qn.size-1}),
      ("wrong-qrope-count",{7:qr.size-1}),
      ("wrong-latent-count",{9:latent.size-1}),
      ("wrong-kpe-count",{11:kpe.size-1}),
      ("T-zero",{12:0}),
      ("T-over-cap",{12:65}),
      ("T-inconsistent",{12:latent.shape[0]+1}),
      ("zero-scale-parameter",{13:0.0}),
      ("nan-scale-parameter",{13:float("nan")}),
      ("small-output",{15:H*V-1}),
      ("small-scores",{17:H*latent.shape[0]-1}),
      ("null-result",{14:fp()}),
    ]
    # Note all GPU buffers should remain unchanged following an invalid call.
    for name,overrides in bad:
        ok,out,scores,_=run(f,planes,scales,qn,qr,latent,kpe,scale,overrides)
        require(ok==0 and np.all(out==-404) and np.all(scores==-405),
                "malformed input accepted "+name)
    qbad=qn.copy()
    qbad[0,0]=np.nan
    ok,out,scores,_=run(f,planes,scales,qn,qr,latent,kpe,scale,
                         {4:ptr(qbad,ctypes.c_float)})
    require(ok==0 and np.all(out==-404) and np.all(scores==-405),
            "nonfinite q accepted")
    sbad=scales.copy()
    sbad[0]=0
    ok,out,scores,_=run(f,planes,scales,qn,qr,latent,kpe,scale,
                         {2:ptr(sbad,ctypes.c_float)})
    require(ok==0 and np.all(out==-404) and np.all(scores==-405),
            "invalid source scale accepted")
    return len(bad)+2

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo-root",type=Path,default=HERE.parent,
                        help="repo path containing original qng64_dense.c/.h")
    parser.add_argument("--emit",type=Path,help="optional immutable local receipt")
    args=parser.parse_args()
    gate= require
    gate(not args.emit or not args.emit.exists(),"immutable local receipt exists")
    tic=time.perf_counter()
    cases=[]
    with tempfile.TemporaryDirectory(prefix="beglin-mla-cpp-metal-") as t:
      cpath,mpath=build(args.repo_root.resolve(),Path(t))
      c,f,dylib=configure(cpath,mpath)
      for seed in (23,71,173):
        rng=np.random.default_rng(seed)
        # Three independent source-like input regimes: Gaussian, outlier-rich,
        # and distribution-skewed. All are synthetic, never checkpoint weights.
        w=rng.normal(0,0.02,size=(OUT,D)).astype(np.float32)
        if seed==71:
            w[::13,:]*=4
        if seed==173:
            w[::7,:]+=0.025
        W=f32(w)
        planes,scales=pack(c,W)
        qn=f32(rng.normal(0,0.35,size=(H,K)))
        qr=f32(rng.normal(0,0.30,size=(H,R)))
        for T in (1,5,7,16,64):
            latent=f32(rng.normal(0,0.24,size=(T,D)))
            kpe=f32(rng.normal(0,0.20,size=(T,R)))
            scale=np.float32(0.073)
            sc_ref,out_ref=oracle(c,planes,scales,qn,qr,latent,kpe,scale)
            rc,out,sc,first_ms=run(f,planes,scales,qn,qr,latent,kpe,scale)
            gate(rc==1,"real compressed Metal pipeline did not run")
            gap_score=metric(sc,sc_ref)
            gap_out=metric(out,out_ref)
            gate(gap_score<0.003 and gap_out<0.003,
                 f"native CPP/MLX GPU vs independent C numeric discrepancy seed={seed},T={T},gap={gap_out}")
            warm=[]
            for _ in range(2):
                rr,o,s,ms=run(f,planes,scales,qn,qr,latent,kpe,scale)
                gate(rr==1 and metric(o,out)<2e-5 and metric(s,sc)<2e-5,
                     "compiled Metal nondeterminism in isolated test")
                warm.append(ms)
            cases.append({"seed":seed,"tokens":T,"context_output_relative_L2":gap_out,
                          "score_relative_L2":gap_score,
                          "prototype_warm_probe_median_ms":statistics.median(warm)})
        if seed==23:
            guards=invalid(f,planes,scales,qn,qr,latent,kpe,scale)
      gate(len(cases)==15 and guards==17,"adversarial / randomized-case coverage")
      report={
        "schema":"beglin.cpp-qng64-n7-absorbed-MLA-Metal-portable-regression/1",
        "status":"PASS_REAL_METAL_QNG64_COMPRESSED_MLA_15_RANDOMIZED_CASES",
        "model_checkpoint_access":False,
        "native_BEGLIN_C_bitplanes":True,
        "native_CPP_MLX_Metal_executed":True,
        "expanded_per_head_source_KV_materialized_on_Metal":False,
        "CPU_oracle_expands_source_KV_for_independent_comparison":True,
        "source_file_sha256":sha(SRC),"compiled_native_Metal_binary_sha256":sha(dylib),
        "independent_seeds":3,"context_lengths":[1,5,7,16,64],
        "cases":cases,"rejected_malformed_inputs":guards,
        "max_output_relative_L2":max(x["context_output_relative_L2"] for x in cases),
        "max_attention_score_relative_L2":max(x["score_relative_L2"] for x in cases),
        "full_DeepSeek_model_certified":False,
        "production_throughput_claim":False,
        "production_modified":False,
        "wall_seconds":time.perf_counter()-tic}
      if args.emit:
          with args.emit.open("x",encoding="utf-8") as fd:
              json.dump(report,fd,indent=2,sort_keys=True);fd.write("\n")
      print(json.dumps({"status":report["status"],
            "cases":len(cases),"max_output_relL2":report["max_output_relative_L2"],
            "max_score_relL2":report["max_attention_score_relative_L2"],
            "rejected_malformed":guards,"program_seconds":report["wall_seconds"],
            "receipt_sha256":sha(args.emit) if args.emit else None},sort_keys=True))

if __name__=="__main__":
    main()
