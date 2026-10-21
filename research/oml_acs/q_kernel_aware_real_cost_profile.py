#!/usr/bin/env python3
"""Measure native BEGLIN q4 1/6-thread and n8 repacked NEON by source Qwen role."""
import ctypes, hashlib, json, mmap, os, struct, subprocess, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SNAP=Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
LAB=ROOT/"lab_local/q_kernel_aware_role_costs_20261022"
SRC=SNAP/"model.safetensors"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
EXPECTED={
 "gguf_transcode.c":"7371cc352226e912dbf42762e5e63a70e91acb5c90691d90e32401e18af7ec5f",
 "qng64_dense.c":"cb985ec48cfa1f1f9ebe8411d12da5d3f73720c8acdcad51fc279a2b25ff8956",
 "research/oml_acs/qng64_n8_repacked_neon.c":"19a3cdf92af1764233d0eac3e61339387c9de52568022508bea3538470e85c76",
}
def sha(b):return hashlib.sha256(b).hexdigest()
def library():
    if LAB.exists():raise FileExistsError("role costs already generated")
    for path,dig in EXPECTED.items():
        if sha((ROOT/path).read_bytes())!=dig:raise RuntimeError("BEGLIN native encoder SHA drift "+path)
    os.umask(0o077);LAB.mkdir(mode=0o700,parents=True)
    out=LAB/"libkernels.dylib"
    subprocess.run(["clang","-dynamiclib","-O3","-fPIC","-I",str(ROOT),
       str(ROOT/"research/oml_acs/q_kernel_aware_profile_native.c"),
       str(ROOT/"research/oml_acs/qng64_n8_repacked_neon.c"),
       str(ROOT/"gguf_transcode.c"),str(ROOT/"qng64_dense.c"),
       "-framework","Accelerate","-lpthread","-lm","-o",str(out)],check=True)
    lib=ctypes.CDLL(str(out));p=ctypes.c_void_p
    def api(name,args,result=None):
        f=getattr(lib,name);f.argtypes=args;f.restype=result;return f
    return {
     "start":api("beglin_kernel_profile_init",[ctypes.c_int],ctypes.c_int),
     "stop":api("beglin_kernel_profile_release",[],ctypes.c_int),
     "q4":api("beglin_kernel_profile_q4",[ctypes.c_int,p,p,p,p,ctypes.c_int,ctypes.c_int],ctypes.c_int),
     "enc4":api("gguf_quantize_q4g64_error_feedback",[p,ctypes.c_int,ctypes.c_int,p,p]),
     "enc8":api("qng64_quantize_f32",[p,ctypes.c_int,ctypes.c_int,ctypes.c_int,p,p],ctypes.c_int),
     "repack":api("beglin_qng64_n8_repack_inplace",[p,ctypes.c_int,ctypes.c_int],ctypes.c_int),
     "n8":api("beglin_qng64_n8_repacked_matvec_neon",[p,p,p,p,p,ctypes.c_int,ctypes.c_int],ctypes.c_int),
     "sha":sha(out.read_bytes())
    }
def time_role(lib,role,weight,base):
    out,inp=weight.shape;n=out*inp;ng=n//64
    q4=np.empty(n//2,dtype=np.uint8);s4=np.empty(ng,dtype=np.float32)
    q8=np.empty(n,dtype=np.uint8);s8=np.empty(ng,dtype=np.float32)
    lib["enc4"](weight.ctypes.data,out,inp,q4.ctypes.data,s4.ctypes.data)
    if lib["enc8"](weight.ctypes.data,out,inp,8,q8.ctypes.data,s8.ctypes.data):raise RuntimeError("n8")
    if role=="q_proj":
        prior=ROOT/"lab_local/q_apple_cpu_20261021_v2"
        if sha(q4.tobytes())!=sha((prior/"native_packed4.codes").read_bytes()) or sha(q8.tobytes())!=sha((prior/"native_qng64_n8.planes").read_bytes()):
            raise RuntimeError("independent original BEGLIN source quantizer differs")
    t=time.perf_counter()
    if lib["repack"](q8.ctypes.data,out,inp):raise RuntimeError("n8 repack")
    repack=(time.perf_counter()-t)*1000
    inputs=base[[0,2,9,15,22,45,64,95]].copy() if inp==1536 else np.random.default_rng(73).normal(size=(8,inp)).astype(np.float32)*.05
    y=np.empty(out,dtype=np.float32)
    timings={}
    for name in ("q4_1t","q4_6t","n8_repacked"):
        def call(x):
            if name=="q4_1t":return lib["q4"](1,q4.ctypes.data,s4.ctypes.data,x.ctypes.data,y.ctypes.data,out,inp)
            if name=="q4_6t":return lib["q4"](6,q4.ctypes.data,s4.ctypes.data,x.ctypes.data,y.ctypes.data,out,inp)
            return lib["n8"](q8.ctypes.data,s8.ctypes.data,x.ctypes.data,None,y.ctypes.data,out,inp)
        for x in inputs[:2]:
            if call(x):raise RuntimeError("kernel warmup "+name)
        durations=[]
        for rep in range(7):
            t=time.perf_counter()
            for x in inputs:
                if call(x) or not np.isfinite(y).all():raise RuntimeError("kernel "+name)
            durations.append((time.perf_counter()-t)*1000/len(inputs))
        timings[name]=dict(median_ms=float(np.median(durations)),samples_ms=[round(x,7) for x in durations])
    return dict(role=role,shape=[out,inp],q4_with_scales_bytes=q4.nbytes+s4.nbytes,
        n8_with_scales_bytes=q8.nbytes+s8.nbytes,n8_one_time_repack_ms=repack,
        times=timings,source_layer=0,source_weight_f32_sha256=sha(weight.tobytes()),
        source_activations="BF16 model token embedding RMSNorm for input=1536; synthetic for input=8960",
        native_kernel_cost_only_not_full_model=True)
def main():
    lib=library()
    if lib["start"](8960):raise RuntimeError("q4 thread pool")
    rows=[]
    try:
        f32input=np.fromfile(ROOT/"lab_local/q_apple_cpu_20261021_v2/layer0_token_norm_inputs.f32",dtype="<f4").reshape(96,1536)
        with SRC.open("rb") as file:
            hlen=struct.unpack("<Q",file.read(8))[0];hdr=json.loads(file.read(hlen))
            with mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as mm:
                for role in ROLES:
                    category="self_attn" if role in ROLES[:4] else "mlp"
                    node=hdr[f"model.layers.0.{category}.{role}.weight"]
                    start,end=node["data_offsets"]
                    if node["dtype"]!="BF16":raise RuntimeError("checkpoint no longer BF16")
                    bits=np.frombuffer(mm,dtype="<u2",count=(end-start)//2,offset=8+hlen+start)
                    weight=(bits.astype("<u4")<<16).view("<f4").reshape(node["shape"]).copy()
                    item=time_role(lib,role,weight,f32input)
                    rows.append(item)
                    print("BEGLIN_KERNEL_ROLE_COST",json.dumps(item,sort_keys=True),flush=True)
    finally:
        if lib["stop"]():raise RuntimeError("q4 worker pool did not shutdown")
    result=dict(schema="BEGLIN-kernel-aware-role-cost-profile/1",source_BF16_snapshot=SNAP.name,
       nroles=len(rows),role_samples=rows,source_encoder_hashes=EXPECTED,
       native_shared_library_sha256=lib["sha"],model_unchanged=True,
       prompt_labels_used=False,full_model_tps_not_predicted_as_fact=True,
       actual_CPU_q4_6thread_pool=True,production_unchanged=True)
    p=LAB/"seven_role_native_cost_private.json"
    with p.open("xb") as f:
        f.write((json.dumps(result,indent=2,sort_keys=True)+"\n").encode())
        f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
    print("BEGLIN_KERNEL_PROFILE_DONE",json.dumps({"SHA256":sha(p.read_bytes()),"roles":len(rows),
        "medians_ms":{r["role"]:{k:round(v["median_ms"],6) for k,v in r["times"].items()} for r in rows}},sort_keys=True))
if __name__=="__main__":main()
