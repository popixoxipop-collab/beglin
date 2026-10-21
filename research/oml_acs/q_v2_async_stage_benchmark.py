#!/usr/bin/env python3
"""Real CPU QKV and gate/up BEGLIN async sibling-node scheduling benchmark.

Runs SAME q4/8 original source BF16 Qwen weights through independent native
kernels using Python ThreadPoolExecutor (ctypes C calls release GIL). Each
sibling q4 call owns an independently initialized native 1t/6t q4 pool;
never concurrent-mutates the same q4pool. Benchmarks all QKV 8 and FFN 4
precision combinations, serial 6t versus parallel 1t and parallel 6t.

Stage input x is a real pinned L0 embedding-RMSNorm vector. Timing may differ
from actual later-layer activation, caches or full model. Concurrency helps
only if this direct observed full-stage wall-time is less than serial.
"""
from __future__ import annotations
import ctypes,hashlib,json,mmap,os,pathlib,struct,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
SOURCE=SNAP/"model.safetensors"
LAB=ROOT/"lab_local/q_v2_async_stage_20261022"
FIX=ROOT/"lab_local/q_apple_cpu_20261021_v2/layer0_token_norm_inputs.f32"
STAGES={"qkv":("q_proj","k_proj","v_proj"),"ffn_gu":("gate_proj","up_proj")}
ROLLS=7
SAMPLES=8
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def pvt(path:pathlib.Path,data:bytes):
    with path.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def build():
    if LAB.exists():raise FileExistsError("prior async cost profile not overwritable")
    os.umask(0o077);LAB.mkdir(parents=True,mode=0o700)
    so=LAB/"libqv2async.dylib"
    subprocess.run(["clang","-O3","-dynamiclib","-fPIC","-I",str(ROOT),
       str(ROOT/"research/oml_acs/q_v2_async_stage_native.c"),
       str(ROOT/"research/oml_acs/qng64_n8_repacked_neon.c"),
       str(ROOT/"gguf_transcode.c"),str(ROOT/"qng64_dense.c"),
       "-o",str(so),"-framework","Accelerate","-lpthread","-lm"],check=True)
    lib=ctypes.CDLL(str(so));ptr=ctypes.c_void_p
    def fn(name,args,out=None):
        f=getattr(lib,name);f.argtypes=args;f.restype=out;return f
    api={
      "pool_start":fn("beglin_async_stage_profile_init",[ctypes.c_int],ctypes.c_int),
      "pool_stop":fn("beglin_async_stage_profile_shutdown",[],ctypes.c_int),
      "q4":fn("beglin_async_stage_profile_q4",[ctypes.c_int,ctypes.c_int,ptr,ptr,ptr,ptr,ctypes.c_int,ctypes.c_int],ctypes.c_int),
      "pack4":fn("gguf_quantize_q4g64_error_feedback",[ptr,ctypes.c_int,ctypes.c_int,ptr,ptr]),
      "pack8":fn("qng64_quantize_f32",[ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr,ptr],ctypes.c_int),
      "repack":fn("beglin_qng64_n8_repack_inplace",[ptr,ctypes.c_int,ctypes.c_int],ctypes.c_int),
      "n8":fn("beglin_qng64_n8_repacked_matvec_neon",[ptr,ptr,ptr,ptr,ptr,ctypes.c_int,ctypes.c_int],ctypes.c_int)
    }
    return api,sha(so.read_bytes())
def source_weights():
    roles=set(k for values in STAGES.values() for k in values)
    result={}
    with SOURCE.open("rb") as f:
        n=struct.unpack("<Q",f.read(8))[0];hdr=json.loads(f.read(n))
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
            for role in sorted(roles):
                group="self_attn" if role in STAGES["qkv"] else "mlp"
                key=f"model.layers.0.{group}.{role}.weight"
                item=hdr[key];a,b=item["data_offsets"]
                bits=np.frombuffer(mm,dtype="<u2",count=(b-a)//2,offset=8+n+a)
                array=(bits.astype("<u4")<<16).view("<f4").reshape(item["shape"]).copy()
                del bits
                if not np.isfinite(array).all():raise RuntimeError("nonfinite BF16 source")
                result[role]=array
    return result
def pack_all(api,wts):
    packed={}
    for role,w in wts.items():
        out,inp=w.shape;n=w.size;g=n//64
        p4=np.empty(n//2,dtype=np.uint8);s4=np.empty(g,dtype=np.float32)
        p8=np.empty(n,dtype=np.uint8);s8=np.empty(g,dtype=np.float32)
        api["pack4"](w.ctypes.data,out,inp,p4.ctypes.data,s4.ctypes.data)
        if api["pack8"](w.ctypes.data,out,inp,8,p8.ctypes.data,s8.ctypes.data):
            raise RuntimeError("native 8bit encoder refused")
        if api["repack"](p8.ctypes.data,out,inp):raise RuntimeError("native n8 layout conversion failed")
        packed[role]={"q4":(p4,s4),"n8":(p8,s8),"dims":(out,inp)}
    return packed

def bench_stage(api,packed,stage,roles,samples):
    results={}
    for mask in product((4,8),repeat=len(roles)):
        bits="".join(str(k) for k in mask)
        outputs={role:np.empty(packed[role]["dims"][0],dtype=np.float32) for role in roles}
        def call(pos,role,precision,threadmode,x):
            w=packed[role];out,inp=w["dims"]
            codes,scales=w["q4" if precision==4 else "n8"]
            y=outputs[role]
            if precision==4:
                rc=api["q4"](pos,threadmode,codes.ctypes.data,scales.ctypes.data,
                        x.ctypes.data,y.ctypes.data,out,inp)
            else:
                rc=api["n8"](codes.ctypes.data,scales.ctypes.data,
                        x.ctypes.data,None,y.ctypes.data,out,inp)
            if rc!=0:raise RuntimeError("real native stage dispatch failed")
            return y
        # Serial Q4 uses 6t; parallel has independent role-indexed q4 pools.
        serial_ref={}
        for i,role in enumerate(roles):
            serial_ref[role]=call(i,role,mask[i],6,samples[0]).copy()
        for mode in ("serial_q4_6t","parallel_q4_1t","parallel_q4_6t"):
            rounds=[]
            with ThreadPoolExecutor(max_workers=len(roles)) as pool:
                for rep in range(ROLLS):
                    t=time.perf_counter()
                    for sample in samples:
                        if mode=="serial_q4_6t":
                            for idx,role in enumerate(roles):
                                call(idx,role,mask[idx],6,sample)
                        else:
                            nthreads=1 if mode=="parallel_q4_1t" else 6
                            tasks=[pool.submit(call,idx,role,mask[idx],nthreads,sample)
                                   for idx,role in enumerate(roles)]
                            for task in tasks:task.result()
                        for role in roles:
                            if not np.isfinite(outputs[role]).all():
                                raise RuntimeError("nonfinite result of parallel stage")
                    rounds.append((time.perf_counter()-t)*1000/len(samples))
            for idx,role in enumerate(roles):
                if not np.allclose(serial_ref[role],
                     call(idx,role,mask[idx],6,samples[0]),rtol=1e-6,atol=1e-5):
                    raise RuntimeError("per-role output differs across dispatch")
            results.setdefault(bits,{})[mode]={
              "median_stage_ms":float(np.median(rounds)),
              "samples_ms":[round(float(x),7) for x in rounds]}
        fastest=min(results[bits],key=lambda name:results[bits][name]["median_stage_ms"])
        results[bits]["best_verified_mode"]=fastest
        results[bits]["best_verified_stage_ms"]=results[bits][fastest]["median_stage_ms"]
        results[bits]["actual_parallel_speedup_vs_serial"]=results[bits]["serial_q4_6t"]["median_stage_ms"]/results[bits]["best_verified_stage_ms"]
        print("ASYNC_CPU_STAGE",stage,bits,fastest,round(results[bits]["best_verified_stage_ms"],5),
              "serial",round(results[bits]["serial_q4_6t"]["median_stage_ms"],5),flush=True)
    return results
def main():
    api,dig=build()
    wts=source_weights()
    packed=pack_all(api,wts)
    if api["pool_start"](8960)!=0:raise RuntimeError("native separated q4 pools not ready")
    report={}
    inputs=np.fromfile(FIX,dtype="<f4").reshape(96,1536)[[0,2,9,15,22,45,64,95]].copy()
    try:
        for stage,roles in STAGES.items():
            report[stage]=bench_stage(api,packed,stage,roles,inputs)
    finally:
        if api["pool_stop"]()!=0:raise RuntimeError("native q4 pool cleanup failed")
    result={
       "schema":"beglin-Q-v2-actual-native-async-stage-dispatch-cost/1",
       "status":"PASS_ACTUAL_CPU_ASYNC_STAGE_BENCHMARK",
       "source_model":"Qwen2.5-1.5B-Instruct",
       "source_rev":SNAP.name,
       "native_lib_sha256":dig,
       "native_profile_wrapper_source_sha256":sha((ROOT/"research/oml_acs/q_v2_async_stage_native.c").read_bytes()),
       "stages":report,
       "source_weight_sha256":{name:sha(w.tobytes()) for name,w in wts.items()},
       "input_vectors_sha256":sha(inputs.tobytes()),
       "Q4_pool_not_shared_between_concurrent_calls":True,
       "concurrency_measured_not_implied":True,
       "stage_GPU_Metal_unmeasured":True,
       "native_full_model_parallel_execution_not_implemented":True,
       "original_operational_engine_unchanged":True,
       "protocol":"only q/k/v and gate/up independent same-x projections",
       "caveats":["Python ThreadPoolExecutor ctypes startup and future coordination are included in observed stage wall time",
          "Full BEGLIN decoder is currently sequential: these schedules are candidate latency models only",
          "Q4 6-thread pools are independent per role; parallel calls may oversubscribe the physical CPU",
          "One role0 layer tensor and public input; other layer hidden activations could behave differently"]
    }
    path=LAB/"async_sibling_stage_costs_private.json"
    pvt(path,(json.dumps(result,indent=2,sort_keys=True)+"\n").encode())
    print("BEGLIN_ASYNC_STAGE_REAL_RECEIPT",json.dumps({
        "sha256":sha(path.read_bytes()),"stages":{name:{
            k:{"best":v["best_verified_stage_ms"],
               "mode":v["best_verified_mode"],
               "benefit_ratio":v["actual_parallel_speedup_vs_serial"]} for k,v in data.items()}
            for name,data in report.items()}},sort_keys=True),flush=True)
if __name__=="__main__":
    main()
