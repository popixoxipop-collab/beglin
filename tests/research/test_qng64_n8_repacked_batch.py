#!/usr/bin/env python3
"""Proof: repacked native BEGLIN n8 M>1 GEMM does not call original bitplanes.

Builds isolated native shared library with original qng64 matvec/matmul
symbols aliased and research-only dispatch wrapper. Verifies bit-plane
original n8 M=3 against in-place decoded n8 M=3, and other valid n fallback.
No source model or production modifications.
"""
from __future__ import annotations
import ctypes,hashlib,json,os
from pathlib import Path
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
FIX=ROOT/"lab_local/q_apple_cpu_20261021_v2"
LAB=ROOT/"lab_local/qng64_n8_repacked_batch_20261022"
def sha(b):return hashlib.sha256(b).hexdigest()
def run(args):
    subprocess.run(args,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def bind(lib,name,args,restype=None):
    f=getattr(lib,name);f.argtypes=args;f.restype=restype
    return f
def main():
    if LAB.exists():raise FileExistsError("batch parity fixture already exists")
    os.umask(0o077);LAB.mkdir(mode=0o700,parents=True)
    original=LAB/"scalar_alias.o"
    run(["clang","-O3","-fPIC","-c",str(ROOT/"qng64_dense.c"),
         "-Dqng64_matvec_f32=beglin_original_scalar_matvec_f32",
         "-Dqng64_matmul_f32=beglin_original_scalar_matmul_f32",
         "-o",str(original)])
    out=LAB/"librepacked_batch_parity.dylib"
    run(["clang","-dynamiclib","-O3","-fPIC","-I",str(ROOT),
         str(ROOT/"research/oml_acs/qng64_n8_repacked_neon.c"),
         str(ROOT/"research/oml_acs/qng64_n8_repacked_lab_dispatch.c"),
         str(original),"-lm","-o",str(out)])
    l=ctypes.CDLL(str(out))
    ptr=ctypes.c_void_p
    orig=bind(l,"beglin_original_scalar_matmul_f32",[ptr,ptr,ctypes.c_int,ptr,ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int])
    tested=bind(l,"qng64_matmul_f32",[ptr,ptr,ctypes.c_int,ptr,ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int])
    repack=bind(l,"beglin_qng64_n8_repack_inplace",[ptr,ctypes.c_int,ctypes.c_int],ctypes.c_int)
    group_bytes=bind(l,"qng64_group_bytes",[ctypes.c_int],ctypes.c_size_t)
    encode=bind(l,"qng64_quantize_f32",[ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr,ptr],ctypes.c_int)
    W=np.fromfile(FIX/"native_qng64_n8.planes",dtype=np.uint8)
    scales=np.fromfile(FIX/"native_qng64_n8.scales.f32",dtype="<f4")
    X=np.fromfile(FIX/"layer0_token_norm_inputs.f32",dtype="<f4").reshape(96,1536)[:3].copy()
    assert len(W)==1536**2 and len(scales)==1536*1536//64
    decoded=W.copy()
    assert repack(decoded.ctypes.data,1536,1536)==0
    original_out=np.empty((3,1536),dtype=np.float32)
    repacked_out=np.empty_like(original_out)
    orig(W.ctypes.data,scales.ctypes.data,8,X.ctypes.data,None,original_out.ctypes.data,1536,1536,3)
    tested(decoded.ctypes.data,scales.ctypes.data,8,X.ctypes.data,None,repacked_out.ctypes.data,1536,1536,3)
    delta=repacked_out.astype(float)-original_out.astype(float)
    maxerr=float(np.abs(delta).max())
    rel=float(np.linalg.norm(delta)/np.linalg.norm(original_out.astype(float)))
    assert maxerr<.0005 and rel<1e-5,(maxerr,rel)
    fallback={}
    for n in (3,5,6,7):
        gb=group_bytes(n)
        if not gb:continue
        rng=np.random.default_rng(42+n)
        original_weights=rng.normal(size=(64,64)).astype(np.float32)
        packed=np.empty((64*gb,),dtype=np.uint8)
        gs=np.empty(64,dtype=np.float32)
        if encode(original_weights.ctypes.data,64,64,n,packed.ctypes.data,gs.ctypes.data)!=0:
            continue
        xb=rng.normal(size=(3,64)).astype(np.float32)
        a=np.empty((3,64),dtype=np.float32)
        b=np.empty_like(a)
        orig(packed.ctypes.data,gs.ctypes.data,n,xb.ctypes.data,None,a.ctypes.data,64,64,3)
        tested(packed.ctypes.data,gs.ctypes.data,n,xb.ctypes.data,None,b.ctypes.data,64,64,3)
        assert np.array_equal(a,b),"other-bit batched fallback not identical"
        fallback[str(n)]=dict(batch_rows=3,bitplane_group_bytes=int(gb),bitwise_output_identical=True)
    assert fallback,"no other bitwidth recognized for fallback testing"
    result=dict(schema="BEGLIN-native-n8-repacked-batch-parity/1",
      status="PASS_NATIVE_M3_SAME_CODE_BYTES",
      shape=[1536,1536],batch_rows=3,
      n8_original_scalar_vs_repacked_maxabs=maxerr,
      n8_original_scalar_vs_repacked_rel_l2=rel,
      other_bitwidth_original_fallback=fallback,
      native_library_sha256=sha(out.read_bytes()),
      original_BEG_LIN_source_SHA256=sha((ROOT/"qng64_dense.c").read_bytes()),
      original_n8_bitplanes_SHA256=sha(W.tobytes()),
      original_model_data_unchanged=True,
      production_modified=False)
    rec=LAB/"receipt.json"
    with rec.open("xb") as f:
        f.write((json.dumps(result,sort_keys=True,indent=2)+"\n").encode())
        f.flush();os.fsync(f.fileno())
    rec.chmod(0o600)
    print("REPACKED_BEG_LIN_BATCH_PARITY_PASS",json.dumps(result,sort_keys=True),flush=True)
if __name__=="__main__":main()
