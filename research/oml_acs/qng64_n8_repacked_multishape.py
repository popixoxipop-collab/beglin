#!/usr/bin/env python3
"""Original Qwen BF16 real 4-shape n8 bitplanes -> in-place codes parity.

Original native BEGLIN encoder and scalar bitplane GEMV serve as oracle.
Research-only repacked code storage equals the original n8 packed byte count.
No model sensitivity, task labels or production changes.
"""
from __future__ import annotations
import ctypes,hashlib,json,mmap,os
from pathlib import Path
import struct,subprocess,time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PREVIOUS=ROOT/"research/oml_acs/qng64_n8_neon_multishape_test.py"
LAB=ROOT/"lab_local/qng64_n8_repacked_multishape_20261022"
MODEL=Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89/model.safetensors")
NAMES=(
"model.layers.0.self_attn.k_proj.weight",
"model.layers.14.self_attn.o_proj.weight",
"model.layers.0.mlp.gate_proj.weight",
"model.layers.0.mlp.down_proj.weight")
SEED=20261022
def sha(blob:bytes)->str:return hashlib.sha256(blob).hexdigest()
def convert(bits:bytes,shape):
    raw=np.frombuffer(bits,dtype="<u2")
    return (raw.astype("<u4")<<16).view("<f4").reshape(shape).copy()
def build():
    so=LAB/"liborig_and_repacked.dylib"
    cmd=["clang","-dynamiclib","-O3","-fPIC","-I",str(ROOT),
         str(ROOT/"qng64_dense.c"),
         str(ROOT/"research/oml_acs/qng64_n8_repacked_neon.c"),
         "-lm","-o",str(so)]
    subprocess.run(cmd,check=True)
    l=ctypes.CDLL(str(so))
    p=ctypes.c_void_p
    l.qng64_quantize_f32.argtypes=[p,ctypes.c_int,ctypes.c_int,ctypes.c_int,p,p]
    l.qng64_quantize_f32.restype=ctypes.c_int
    l.qng64_matvec_f32.argtypes=[p,p,ctypes.c_int,p,p,p,ctypes.c_int,ctypes.c_int]
    l.beglin_qng64_n8_repack_inplace.argtypes=[p,ctypes.c_int,ctypes.c_int]
    l.beglin_qng64_n8_repack_inplace.restype=ctypes.c_int
    l.beglin_qng64_n8_repacked_matvec_neon.argtypes=[p,p,p,p,p,ctypes.c_int,ctypes.c_int]
    l.beglin_qng64_n8_repacked_matvec_neon.restype=ctypes.c_int
    return l,sha(so.read_bytes())
def compare(model,name,node,mm,offset,seed):
    out,dim=node["shape"]
    a,b=node["data_offsets"]
    w=convert(bytes(mm[offset+a:offset+b]),(out,dim))
    n=out*dim
    orig=np.empty((n,),dtype=np.uint8)
    scales=np.empty((n//64,),dtype=np.float32)
    if model.qng64_quantize_f32(w.ctypes.data,out,dim,8,orig.ctypes.data,scales.ctypes.data):
        raise RuntimeError("native model encoder failed")
    packed=orig.copy()
    if model.beglin_qng64_n8_repack_inplace(packed.ctypes.data,out,dim):
        raise RuntimeError("research prepack failed")
    # Verify all 64 codes in first/last/middle group, including sign bit.
    ng=n//64
    for g in sorted({0,1,ng//2,ng-1}):
        u=orig[g*64:(g+1)*64].reshape(8,8)
        q=packed[g*64:(g+1)*64].view(np.int8)
        for c in range(64):
            v=sum(((int(u[b,c//8])>>(c%8))&1)<<b for b in range(8))
            if q[c]!=v-128:raise RuntimeError("data buffer decode mismatch")
    rng=np.random.default_rng(seed)
    x=rng.standard_normal((8,dim)).astype(np.float32)*.05
    oracle=np.zeros((out,),dtype=np.float32)
    computed=np.zeros_like(oracle)
    maxerr=0.;err2=0.;ref2=0.
    for row in x:
        model.qng64_matvec_f32(orig.ctypes.data,scales.ctypes.data,8,
                 row.ctypes.data,None,oracle.ctypes.data,out,dim)
        rc=model.beglin_qng64_n8_repacked_matvec_neon(packed.ctypes.data,scales.ctypes.data,
                 row.ctypes.data,None,computed.ctypes.data,out,dim)
        if rc:raise RuntimeError("wrong shape SIMD")
        diff=computed.astype(np.float64)-oracle.astype(np.float64)
        maxerr=max(maxerr,float(np.max(np.abs(diff))))
        err2+=float(np.square(diff).sum())
        ref2+=float(np.square(oracle.astype(np.float64)).sum())
    rel=(err2/ref2)**.5
    if maxerr>=.0005 or rel>=1e-5:
        raise RuntimeError(f"real model {name} fails numerical parity {maxerr} {rel}")
    timings={}
    for method in ("scalar","repacked"):
        rounds=[]
        for rep in range(5):
            t=time.perf_counter()
            for row in x:
                if method=="scalar":
                    model.qng64_matvec_f32(orig.ctypes.data,scales.ctypes.data,8,
                        row.ctypes.data,None,oracle.ctypes.data,out,dim)
                else:
                    model.beglin_qng64_n8_repacked_matvec_neon(packed.ctypes.data,scales.ctypes.data,
                        row.ctypes.data,None,computed.ctypes.data,out,dim)
            rounds.append((time.perf_counter()-t)*1000/len(x))
        timings[method]=float(np.median(rounds))
    return dict(name=name,shape=[out,dim],original_BF16_tensor_sha256=sha(bytes(mm[offset+a:offset+b])),
        native_n8_packed_sha256=sha(orig.tobytes()),repacked_in_memory_sha256=sha(packed.tobytes()),
        native_bitplane_weight_bytes=n,repacked_code_weight_bytes=n,
        max_abs_error=maxerr,relative_L2=rel,scalar_ms=timings["scalar"],
        repacked_ms=timings["repacked"],speedup=timings["scalar"]/timings["repacked"],private=True)
def main():
    if LAB.exists():raise FileExistsError("multishape experiment exists")
    os.umask(0o077);LAB.mkdir(parents=True,mode=0o700)
    model,binary_sha=build()
    with MODEL.open("rb") as f:
        header_size=struct.unpack("<Q",f.read(8))[0]
        data=json.loads(f.read(header_size))
        offset=8+header_size
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
            rows=[]
            for idx,name in enumerate(NAMES):
                out=compare(model,name,data[name],mm,offset,SEED+idx)
                rows.append(out)
                print("REPACKED_REAL_TENSOR",json.dumps(out,sort_keys=True),flush=True)
    j=dict(schema="BEGLIN-Q-repacked-n8-multishape-real/1",
        status="PASS_4_REAL_SHAPES",binary_sha256=binary_sha,rows=rows,
        original_model="Qwen2.5-1.5B-Instruct BF16 cached revision 4ae77cb209f06199b8df1c94e21ff341332a3a89",
        full_model_tested=False,production_unchanged=True)
    output=LAB/"multishape_private_receipt.json"
    with output.open("xb") as f:
        f.write((json.dumps(j,indent=2,sort_keys=True)+"\n").encode());f.flush();os.fsync(f.fileno())
    output.chmod(0o600)
    print("REPACKED_4_SHAPES_VERIFIED",json.dumps({"sha256":sha(output.read_bytes()),
        "min_speedup":min(q["speedup"] for q in rows),"maximum_error":max(q["max_abs_error"] for q in rows)}),flush=True)
if __name__=="__main__":main()
