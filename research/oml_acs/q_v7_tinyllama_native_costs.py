#!/usr/bin/env python3
"""Seven-role original TinyLlama model/native BEGLIN Q4/n8 NEON kernel profiler.

Profiles each original BF16 projection role-0 on Apple CPU. Native Q4G64
1/6 threads and true in-place-repacked n8 are executed, not synthesized
from Qwen shapes. Includes one-time n8 repack time (not decode latency).
Strictly research-only and no full-model speed claims.
"""
from __future__ import annotations
import ctypes,hashlib,json,math,os,pathlib,subprocess,time
import numpy as np
from safetensors import safe_open
ROOT=pathlib.Path(__file__).resolve().parents[2]
MODEL=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv7-tinyllama-model-20261022/model.safetensors")
OUT=ROOT/"lab_local/qv7_tinyllama_20261022/native_kernel_costs"
PUBLIC=ROOT/"docs/research/BEGLIN_QV7_LLAMA_NATIVE_SEVEN_ROLE_COSTS_20261022.json"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
SHAS={
"gguf_transcode.c":"7371cc352226e912dbf42762e5e63a70e91acb5c90691d90e32401e18af7ec5f",
"qng64_dense.c":"cb985ec48cfa1f1f9ebe8411d12da5d3f73720c8acdcad51fc279a2b25ff8956",
"research/oml_acs/qng64_n8_repacked_neon.c":"19a3cdf92af1764233d0eac3e61339387c9de52568022508bea3538470e85c76",
"research/oml_acs/q_kernel_aware_profile_native.c":"7e4231bd5b6b447ec3163def72a5a57b7a996406641751bf2c0783c488b1df22"}
def sha(x:bytes)->str:return hashlib.sha256(x).hexdigest()
def put(p,obj):
    b=(json.dumps(obj,sort_keys=True,indent=2)+"\n").encode()
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
    return sha(b)
def build():
    if OUT.exists() or PUBLIC.exists():raise FileExistsError("Qv7 Llama kernel cost profile already captured")
    for p,h in SHAS.items():
        if sha((ROOT/p).read_bytes())!=h:raise RuntimeError("original BEGLIN C quantizer kernel source changed "+p)
    os.umask(0o077);OUT.mkdir(mode=0o700,parents=True)
    so=OUT/"libtinyllama_7roles_native_kernel.dylib"
    subprocess.run(["clang","-dynamiclib","-O3","-fPIC","-I",str(ROOT),
      str(ROOT/"research/oml_acs/q_kernel_aware_profile_native.c"),
      str(ROOT/"research/oml_acs/qng64_n8_repacked_neon.c"),
      str(ROOT/"gguf_transcode.c"),str(ROOT/"qng64_dense.c"),
      "-framework","Accelerate","-lpthread","-lm","-o",str(so)],check=True,timeout=100)
    lib=ctypes.CDLL(str(so));p=ctypes.c_void_p;ci=ctypes.c_int
    def api(name,args,rest=None):
        f=getattr(lib,name);f.argtypes=args;f.restype=rest;return f
    functions=dict(
      init=api("beglin_kernel_profile_init",[ci],ci),
      release=api("beglin_kernel_profile_release",[],ci),
      q4=api("beglin_kernel_profile_q4",[ci,p,p,p,p,ci,ci],ci),
      enc4=api("gguf_quantize_q4g64_error_feedback",[p,ci,ci,p,p]),
      enc8=api("qng64_quantize_f32",[p,ci,ci,ci,p,p],ci),
      repack=api("beglin_qng64_n8_repack_inplace",[p,ci,ci],ci),
      n8=api("beglin_qng64_n8_repacked_matvec_neon",[p,p,p,p,p,ci,ci],ci))
    return functions,sha(so.read_bytes())
def kernel_costs():
    native,libsha=build()
    # TinyLlama out dimensions q=2048, k/v=256, ffn=5632.
    if native["init"](5632):raise RuntimeError("native Apple six-thread Q4 worker did not initialize")
    rows=[]
    try:
        with safe_open(str(MODEL),framework="pt",device="cpu") as sf:
            for role in ROLES:
                category="self_attn" if role in ROLES[:4] else "mlp"
                key=f"model.layers.0.{category}.{role}.weight"
                if key not in sf.keys():raise RuntimeError("actual BF16 TinyLlama role absent "+key)
                w=sf.get_tensor(key).float().contiguous().numpy()
                o,inp=w.shape
                if inp%64 or o<1:raise RuntimeError("role group64 native encoding shape incorrect")
                cells=o*inp;group=cells//64
                packed4=np.empty(cells//2,np.uint8);scale4=np.empty(group,np.float32)
                packed8=np.empty(cells,np.uint8);scale8=np.empty(group,np.float32)
                native["enc4"](w.ctypes.data,o,inp,packed4.ctypes.data,scale4.ctypes.data)
                if native["enc8"](w.ctypes.data,o,inp,8,packed8.ctypes.data,scale8.ctypes.data):
                    raise RuntimeError("new Llama n8 encoded source failed")
                tic=time.perf_counter()
                if native["repack"](packed8.ctypes.data,o,inp):
                    raise RuntimeError("new Llama shape n8 repack failed")
                repacked_ms=(time.perf_counter()-tic)*1e3
                rng=np.random.default_rng(307+len(rows))
                inputs=rng.normal(loc=0.,scale=.05,size=(12,inp)).astype(np.float32)
                output=np.empty(o,dtype=np.float32)
                kernels={}
                def execute(name,x):
                    if name=="q4_1t":
                        return native["q4"](1,packed4.ctypes.data,scale4.ctypes.data,x.ctypes.data,output.ctypes.data,o,inp)
                    if name=="q4_6t":
                        return native["q4"](6,packed4.ctypes.data,scale4.ctypes.data,x.ctypes.data,output.ctypes.data,o,inp)
                    return native["n8"](packed8.ctypes.data,scale8.ctypes.data,x.ctypes.data,None,output.ctypes.data,o,inp)
                for kernel in ("q4_1t","q4_6t","n8_repacked"):
                    for x in inputs[:3]:
                        if execute(kernel,x)!=0:raise RuntimeError("native kernel warmup failed")
                    values=[]
                    for i in range(9):
                        ts=time.perf_counter()
                        for x in inputs:
                            if execute(kernel,x)!=0 or not np.isfinite(output).all():
                                raise RuntimeError("real kernel nan/fail "+kernel)
                        values.append((time.perf_counter()-ts)*1000/len(inputs))
                    kernels[kernel]=dict(median_ms=float(np.median(values)),
                        min_ms=float(min(values)),max_ms=float(max(values)),
                        repeated_kernel_medians_ms=values)
                row={"role":role,"shape":[int(o),int(inp)],"layer":0,
                     "source_original_BF16_weight_sha256":sha(w.tobytes()),
                     "q4_with_f32_scales_bytes":int(packed4.nbytes+scale4.nbytes),
                     "n8_with_f32_scales_bytes":int(packed8.nbytes+scale8.nbytes),
                     "native_n8_repack_once_ms":repacked_ms,
                     "real_original_native_kernel_costs":kernels,
                     "example_input_type":"12 independent fixed-seed synthetic activation vectors; source BF16 weights authentic; no downstream labels",
                     "not_full_model_serving_timing":True}
                rows.append(row)
                print("QV7_TINY_LLAMA_TRUE_NATIVE_SEVEN_ROLE",role,o,inp,
                      {k:round(v["median_ms"],6) for k,v in kernels.items()},flush=True)
    finally:
        if native["release"]()!=0:raise RuntimeError("original Q4 shared pool cleanup failed")
    full={"schema":"BEGLIN-Qv7-TinyLlama-real-Apple-q4n8-role-median-kernel-costs/1",
       "status":"SEVEN_REAL_SOURCE_BF16_LLAMA_ROLE_NATIVE_KERNELS_PROFILLED",
       "source_model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
       "source_model_revision":"fe8a4ea1ffedaf415f4da2f062534de366a451e6",
       "original_native_C_source_sha256":SHAS,"native_kernel_library_sha256":libsha,
       "seven_roles":rows,"layers":22,"roles":7,
       "unquantized_source_model_weight_unchanged":True,
       "actual_full_model_runtime_not_measured_here":True,
       "no_autopromotion":True}
    receipt=OUT/"source_original_tinyllama_role_profiles_private.json"
    rawsha=put(receipt,full)
    aggregate={k:v for k,v in full.items() if k!="seven_roles"}
    aggregate["role_kernel_cost_measured"]=rows
    aggregate["source_private_full_receipt_sha256"]=rawsha
    data=(json.dumps(aggregate,sort_keys=True,indent=2)+"\n").encode()
    with PUBLIC.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    print("QV7_REAL_LLAMA_CPU_KERNEL_COST_PROFILE_PASS",json.dumps({
      "private_sha":rawsha,"public_sha":sha(data),"nroles":len(rows)},sort_keys=True),flush=True)
if __name__=="__main__":kernel_costs()
