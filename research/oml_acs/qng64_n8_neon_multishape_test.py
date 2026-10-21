#!/usr/bin/env python3
"""Isolated multi-shape regression of BEGLIN native qNg64 n8 NEON decoder.

Read-only ORIGINAL public Qwen BF16 safetensors weights. Both encoder and
scalar oracle are compiled from unmodified BEGLIN qng64_dense.c. The candidate
is separate research qng64_n8_neon_experimental.c, not linked into serving.
No prompts, secrets, behavioral labels, production, or old holdout accessed.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
MODEL=Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89/model.safetensors")
FIX=ROOT/"lab_local/q_apple_cpu_20261021_v2"
LAB=ROOT/"lab_local/qng64_n8_neon_multishape_20261022"
TENSORS=(
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.14.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
)
PASS_REL_L2=1e-5
PASS_MAX_ABS=5e-4
SEED=20261022

def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def build_c():
    so=LAB/"liboriginal_vs_neon.dylib"
    cmd=["clang","-dynamiclib","-O3","-fPIC","-I",str(ROOT),
         str(ROOT/"qng64_dense.c"),
         str(ROOT/"research/oml_acs/qng64_n8_neon_experimental.c"),
         "-lm","-o",str(so)]
    subprocess.run(cmd,check=True)
    a=ctypes.CDLL(str(so))
    ptr=ctypes.c_void_p
    a.qng64_quantize_f32.argtypes=[ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr,ptr]
    a.qng64_quantize_f32.restype=ctypes.c_int
    a.qng64_matvec_f32.argtypes=[ptr,ptr,ctypes.c_int,ptr,ptr,ptr,ctypes.c_int,ctypes.c_int]
    a.qng64_matvec_f32.restype=None
    a.beglin_qng64_n8_matvec_neon.argtypes=[ptr,ptr,ptr,ptr,ptr,ctypes.c_int,ctypes.c_int]
    a.beglin_qng64_n8_matvec_neon.restype=ctypes.c_int
    return a,sha(so.read_bytes())

def bf16_f32(raw:bytes,shape):
    if len(raw)!=int(np.prod(shape))*2:
        raise RuntimeError("wrong raw bf16 shape")
    bits=np.frombuffer(raw,dtype="<u2")
    return (bits.astype("<u4")<<16).view("<f4").reshape(shape).copy()

def unpack_reference_oracle(planes:np.ndarray,scale:np.ndarray,
                            out:int,inp:int,chosen_rows:np.ndarray)->dict:
    ng=inp//64
    codes=planes.reshape(out,ng,8,8)
    inp_cols=(0,1,7,8,15,31,32,55,63)
    checked=0
    for row in chosen_rows:
        for group in (0,ng//2,ng-1):
            for col in inp_cols:
                value=0
                for bit in range(8):
                    value|=int((int(codes[row,group,bit,col//8])>>(col%8))&1)<<bit
                q=value-128
                if not -128<=q<=127:
                    raise AssertionError("manual encoded code outside full n8 range")
                checked+=1
    return dict(bitplane_positions_independently_checked=checked,bitplanes=8,range="[-128,127]")

def run_tensor(lib,name,node,mm,byte_offset,base_inputs):
    begin,end=node["data_offsets"]
    shape=node["shape"]
    out,inp=(int(shape[0]),int(shape[1]))
    if node["dtype"]!="BF16" or inp%64:
        raise RuntimeError("tensor not supported")
    raw=bytes(mm[byte_offset+begin:byte_offset+end])
    weight=bf16_f32(raw,shape)
    ng=out*(inp//64)
    planes=np.empty((out*inp,),dtype=np.uint8)
    scales=np.empty((ng,),dtype=np.float32)
    if lib.qng64_quantize_f32(weight.ctypes.data,out,inp,8,
                            planes.ctypes.data,scales.ctypes.data)!=0:
        raise RuntimeError("native quantizer refused real shape")
    rng=np.random.default_rng(SEED+out*13+inp)
    if inp==1536:
        sample=base_inputs[[0,1,2,3,10,30,60,95]].astype(np.float32,copy=True)
        interpretation="real Qwen embedding RMSNorm input (1536)"
    else:
        sample=rng.normal(size=(8,inp)).astype(np.float32)*0.06
        interpretation="seeded synthetic activation for high-dim 8960"
    scalar=np.empty((out,),dtype=np.float32)
    simd=np.empty((out,),dtype=np.float32)
    diffs=[];ref_energy=0.;err_energy=0.
    for x in sample:
        lib.qng64_matvec_f32(planes.ctypes.data,scales.ctypes.data,8,
                            x.ctypes.data,None,scalar.ctypes.data,out,inp)
        if lib.beglin_qng64_n8_matvec_neon(planes.ctypes.data,scales.ctypes.data,
                          x.ctypes.data,None,simd.ctypes.data,out,inp)!=0:
            raise RuntimeError("NEON failed real shape")
        d=simd.astype(np.float64)-scalar.astype(np.float64)
        if not np.isfinite(d).all():
            raise RuntimeError("nonfinite candidates")
        diffs.append(float(np.abs(d).max()))
        ref_energy+=float(np.square(scalar.astype(np.float64)).sum())
        err_energy+=float(np.square(d).sum())
    maximum=max(diffs)
    rel=float(np.sqrt(err_energy/ref_energy))
    if maximum>PASS_MAX_ABS or rel>PASS_REL_L2:
        raise RuntimeError(f"parity failed for {name}: max={maximum}, rel={rel}")
    manually=unpack_reference_oracle(planes,scales,out,inp,
                                    np.asarray([0,min(out-1,7),out-1]))
    timing={}
    for kind in ("scalar","neon"):
        values=[]
        for rep in range(5):
            start=time.perf_counter()
            for j in range(8):
                x=sample[j]
                if kind=="scalar":
                    lib.qng64_matvec_f32(planes.ctypes.data,scales.ctypes.data,8,
                          x.ctypes.data,None,scalar.ctypes.data,out,inp)
                elif lib.beglin_qng64_n8_matvec_neon(planes.ctypes.data,scales.ctypes.data,
                          x.ctypes.data,None,simd.ctypes.data,out,inp)!=0:
                    raise RuntimeError("NEON timing failed")
            values.append((time.perf_counter()-start)/8*1000)
        timing[kind]=[float(x) for x in values]
    return {
      "tensor":name,"weight_shape":[out,inp],
      "raw_BF16_weight_sha256":sha(raw),
      "packed_native_BEG_N8_planes_sha256":sha(planes.tobytes()),
      "source_bits_BF16":16,"stored_n8_bytes_including_f32_scales":len(planes)+len(scales.tobytes()),
      "inputs_per_shape":len(sample),"input_origin":interpretation,
      "n8_neon_vs_original_scalar_maxabs":maximum,
      "n8_neon_vs_original_scalar_relative_L2":rel,
      "parity_gate_max_abs":PASS_MAX_ABS,"parity_gate_rel_L2":PASS_REL_L2,
      "manual_bitplane_spot_oracle":manually,
      "scalar_median_ms":float(np.median(timing["scalar"])),
      "neon_median_ms":float(np.median(timing["neon"])),
      "relative_speedup":float(np.median(timing["scalar"])/np.median(timing["neon"])),
      "timed_rounds_each":5,
    }

def main():
    if LAB.exists():raise FileExistsError("refuse overwrite multi-shape evidence")
    if not MODEL.is_file():raise RuntimeError("pinned real BF16 checkpoint absent")
    os.umask(0o077)
    LAB.mkdir(parents=True,mode=0o700)
    lib,binsha=build_c()
    base=np.fromfile(FIX/"layer0_token_norm_inputs.f32",dtype="<f4").reshape(96,1536)
    with MODEL.open("rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]
        hdr=json.loads(f.read(n))
        off=8+n
        rows=[]
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
            for name in TENSORS:
                row=run_tensor(lib,name,hdr[name],mm,off,base)
                rows.append(row)
                print("N8_MULTISHAPE_REAL",json.dumps(row,sort_keys=True),flush=True)
    r={
      "schema":"beglin-qng64-n8-neon-multishape/1",
      "status":"PASS_ORIGINAL_SOURCE_REAL_TENSORS",
      "model":"Qwen/Qwen2.5-1.5B-Instruct",
      "source_rev":"4ae77cb209f06199b8df1c94e21ff341332a3a89",
      "research_binary_sha256":binsha,
      "native_scalar_source_sha256":sha((ROOT/"qng64_dense.c").read_bytes()),
      "research_neon_source_sha256":sha((ROOT/"research/oml_acs/qng64_n8_neon_experimental.c").read_bytes()),
      "parity_gates":{"max_abs":PASS_MAX_ABS,"relative_L2":PASS_REL_L2},
      "examples":rows,
      "full_model_forward_tested":False,
      "production_touched":False,
      "main_touched":False,
      "note":"Real weight matrices and native packed qNg64, but 8960 input vector synthetic; do not overstate as full model end-to-end speedup",
    }
    path=LAB/"n8_neon_multishape_private_receipt.json"
    with path.open("xb") as file:
        file.write((json.dumps(r,indent=2,sort_keys=True)+"\n").encode())
        file.flush();os.fsync(file.fileno())
    path.chmod(0o600)
    print("N8_MULTISHAPE_COMPLETE",json.dumps({
      "count":len(rows),"sha256":sha(path.read_bytes()),
      "min_speedup":min(x["relative_speedup"] for x in rows),
      "max_parity_error":max(x["n8_neon_vs_original_scalar_maxabs"] for x in rows)
    },sort_keys=True),flush=True)
if __name__=="__main__":main()
