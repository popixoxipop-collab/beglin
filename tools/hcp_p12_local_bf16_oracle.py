#!/usr/bin/env python3
"""P12 local, original BF16 DeepSeek weight projection error against native qNg64.

Not full model teacher. 2 verified matrices only, 5/6-bit group64 error-feedback
native C quantization and 8 deterministic source-independent probe vectors.
Read-only safetensors, no MPS/CUDA, no checkpoint bytes exported, <1.5GiB RSS.
"""
from __future__ import annotations
import argparse,ctypes,hashlib,json,math,os,pathlib,resource,subprocess,time
import numpy as np
import torch
from safetensors import safe_open

MODEL_INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
STAGE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p12-bf16-oracle-20261014")
MODEL=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
SOURCE_SHA="e2271d6fe758c6d09877130637dccb193427ea15"
SEED=20261014
MAX_RSS=1536*(1024**2)
KEYS={
 "shared_up_L3":"model.layers.3.mlp.shared_experts.up_proj.weight",
 "shared_down_L26":"model.layers.26.mlp.shared_experts.down_proj.weight",
}
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def memory_guard():
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 if rss >=MAX_RSS:raise MemoryError("P12 BF16 micro-oracle exceeded 1.5GiB process RSS")
 return int(rss)
def weighted_projection_checks(w,codes,scale,n,vectors,lib):
 nrows,ncols=w.shape
 assert ncols%64==0
 gb=n*8
 q=ctypes.c_int(n)
 packed_ptr=codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
 scale_ptr=scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
 inp=np.ascontiguousarray(vectors[0],dtype=np.float32)
 native=np.empty(nrows,dtype=np.float32)
 lib.qng64_matvec_f32(packed_ptr,scale_ptr,q,
   inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
   None,native.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),nrows,ncols)
 # Independently decode native bit-planes in bounded row blocks,
 # accumulate numeric weight errors and projections for all vectors.
 approx=np.empty((nrows,vectors.shape[0]),dtype=np.float32)
 se=0.0; maxdiff=0.0; sumw=0.0; suma=0.0
 for start in range(0,nrows,96):
  end=min(nrows,start+96)
  width=end-start
  sections=codes[start:end].reshape(width,ncols//64,n,8)
  vals=np.zeros((width,ncols//64,64),dtype=np.int16)
  for j in range(n):
   bits=np.unpackbits(sections[:,:,j,:],axis=-1,bitorder="little")
   vals+=(bits.astype(np.int16)<<j)
  vals-=(1<<(n-1))
  deq=(vals.astype(np.float32)*scale[start:end,:,None]).reshape(width,ncols)
  diff=deq-w[start:end]
  se+=float(np.sum(diff.astype(np.float64)**2))
  sumw+=float(np.sum(w[start:end].astype(np.float64)**2))
  suma+=float(np.sum(deq.astype(np.float64)**2))
  maxdiff=max(maxdiff,float(np.abs(diff).max()))
  approx[start:end]=np.matmul(deq,vectors.T).astype(np.float32)
 # The C dense matrix-vector kernel operates on actual packed bytes.
 native_delta=np.max(np.abs(native-approx[:,0]))
 if native_delta>0.0035+0.00001*np.max(np.abs(native)):
  raise ValueError(f"native qNg64 projection mismatch {native_delta}")
 if np.any(~np.isfinite(approx)):raise ValueError("quantized projected features non-finite")
 return {"approx":approx,"weight_rmse":math.sqrt(se/(nrows*ncols)),
  "weight_relative_l2":math.sqrt(se/sumw),
  "weight_max_abs_error":maxdiff,
  "native_C_vs_independent_unpack_projection_max_abs":float(native_delta),
  "packed_bytes":int(codes.nbytes),"scales_bytes":int(scale.nbytes)}

def run():
 if (not ROOT.is_dir() or ROOT.parent.name!=".tailnet-tmp" or
   ROOT.name!="hcp-p12-bf16-oracle-20261014" or ROOT.is_symlink()):
  raise RuntimeError("P12 run root not the exact owned scratch")
 torch.set_num_threads(1)
 if sha(MODEL/"model.safetensors.index.json")!=MODEL_INDEX_SHA:
  raise RuntimeError("BF16 model SHA index changed")
 if sha("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")!=STAGE_SHA or sha("/Users/xox/vdsp_serving/active-route.json")!=ROUTE_SHA:
  raise RuntimeError("original immutable stage identity mismatch")
 source=ROOT/"src"
 expected=subprocess.check_output(["/usr/bin/git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
 if expected!=SOURCE_SHA:raise RuntimeError("original pinned P11 Git source mismatch")
 for name in ("gguf_transcode.c","gguf_transcode.h","qng64_dense.c","qng64_dense.h"):
  if not (source/name).is_file() or (source/name).is_symlink():
   raise RuntimeError(f"missing source {name}")
 dylib=ROOT/"libp12_exact_quant.dylib"
 if not dylib.is_file():raise RuntimeError("missing prebuilt pure native C quantizer")
 lib=ctypes.CDLL(str(dylib))
 lib.gguf_quantize_qNg64.argtypes=[
  ctypes.POINTER(ctypes.c_float),ctypes.c_int,ctypes.c_int,ctypes.c_int,
  ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_float)]
 lib.gguf_quantize_qNg64.restype=None
 lib.qng64_matvec_f32.argtypes=[
  ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_float),ctypes.c_int,
  ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),
  ctypes.POINTER(ctypes.c_float),ctypes.c_int,ctypes.c_int]
 lib.qng64_matvec_f32.restype=None
 mapping=json.loads((MODEL/"model.safetensors.index.json").read_text())["weight_map"]
 records=[]
 for key,name in KEYS.items():
  shard=MODEL/mapping[name]
  if not shard.is_file() or shard.is_symlink() or shard.parent!=MODEL:raise RuntimeError("untrusted original BF16 shard")
  with safe_open(str(shard),framework="pt",device="cpu") as sf:
   tensor=sf.get_slice(name)
   shape=tensor.get_shape()
   if tensor.get_dtype()!="BF16" or len(shape)!=2 or (int(shape[1])%64):
    raise RuntimeError("original matrix not BF16 group64")
   nrow,ncol=map(int,shape)
   if nrow*ncol>6_000_000:raise RuntimeError("BF16 allowed tensor limit")
   bf=tensor[0:nrow,0:ncol]
   if bf.dtype!=torch.bfloat16 or bf.shape!=(nrow,ncol):raise RuntimeError("malformed BF16 safe-slice")
   fp32=np.ascontiguousarray(bf.to(torch.float32).numpy(),dtype=np.float32)
  del bf
  if not np.all(np.isfinite(fp32)):raise RuntimeError("non-finite original BF16")
  v=np.random.default_rng(SEED+sum(name.encode())).standard_normal((8,ncol)).astype(np.float32)
  # This BF16 CPU reference includes the exact same input and source weights
  # as native bit-pack, but is only an isolated matrix projection.
  ref=np.ascontiguousarray(np.matmul(fp32,v.T),dtype=np.float32)
  if not np.all(np.isfinite(ref)):raise RuntimeError("BF16 reference projection nonfinite")
  one={"target":key,"source_tensor":name,"source_shard":shard.name,
       "original_dtype":"BF16","weight_shape":[nrow,ncol],
       "fp32_weight_bytes":int(fp32.nbytes),
       "vector_count":8,"input_rng_seed":SEED+sum(name.encode()),
       "reference_project_sha256":hashlib.sha256(ref.tobytes()).hexdigest(),
       "source_full_weight_tile_sha256":hashlib.sha256(fp32.tobytes()).hexdigest(),
       "quantization_bits":{}}
  for n in (5,6):
   t0=time.perf_counter()
   packed=np.zeros((nrow,ncol//64,n*8),dtype=np.uint8)
   scale=np.empty((nrow,ncol//64),dtype=np.float32)
   lib.gguf_quantize_qNg64(
     fp32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),nrow,ncol,n,
     packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
     scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
   if not np.all(np.isfinite(scale)) or float(np.min(scale))<=0:
    raise RuntimeError("P12 qNg64 source quantization had invalid scales")
   metrics=weighted_projection_checks(fp32,packed,scale,n,v,lib)
   approx=metrics.pop("approx")
   err=(approx-ref).astype(np.float64)
   rel=math.sqrt(float(np.sum(err*err))/max(1.e-12,float(np.sum(ref.astype(np.float64)**2))))
   rmse=math.sqrt(float(np.mean(err*err)))
   mean_abs=float(np.mean(np.abs(err)))
   record={"bits":n,
    **metrics,
    "native_quantizer":"gguf_quantize_qNg64 original C, error-feedback 64",
    "C_transcode_layout":"group64, biased-bit-plane, little-endian",
    "packed_sha256":hashlib.sha256(packed.tobytes()).hexdigest(),
    "scale_sha256":hashlib.sha256(scale.tobytes()).hexdigest(),
    "projection_sha256":hashlib.sha256(approx.tobytes()).hexdigest(),
    "projection_rmse":rmse,"projection_relative_l2":rel,
    "projection_mean_absolute_error":mean_abs,
    "cpu_quantize_and_decode_wall_s":round(time.perf_counter()-t0,4),
    "RSS_after_bytes":memory_guard()}
   one["quantization_bits"][str(n)]=record
   print("P12_BF16_LOCAL_MATRIX",json.dumps({
      "target":key,"n":n,"weight_relative_l2":record["weight_relative_l2"],
      "projection_relative_l2":rel,"rss_MB":round(record["RSS_after_bytes"]/1048576,2)},sort_keys=True),flush=True)
   del approx,packed,scale
  # Important: separate target's BF16 matrix memory from next one.
  del fp32,ref,v
  records.append(one)
  memory_guard()
 final={"schema":"beglin-hcp-p12-real-native-BF16-local-two-tensor-projections-v1",
 "status":"TWO_ORIGINAL_BF16_MATRICES_EXACT_C_QNG64_LOCAL_ORACLE_PASS",
 "model_index_sha256":MODEL_INDEX_SHA,
 "source_commit":SOURCE_SHA,
 "original_BF16_shards_opened_readonly":True,"source_local_tensors":len(KEYS),
 "bits_tested":[5,6],"independent_projection_vectors":8,
 "compiled_source_sha256":{x:sha(source/x) for x in ("gguf_transcode.c","qng64_dense.c")},
 "C_compiled_library_sha256":sha(dylib),
 "records":records,
 "native_C_vs_independent_bitunpack_matvec_equivalence":True,
 "full_BF16_teacher_forward_executed":False,
 "model_BF16_logits_reference_measured":False,
 "tested_reference_is_full_model":False,
 "production_P9_weights_bit_identical_not_proven":True,
 "task_accuracy_noninferiority_proven":False,
 "whole_model_quality_certified":False,
 "max_RSS_bytes":memory_guard(),
 "same_quantization_algorithm_as_native":True,
 "production_touched":False,"automatic_promotion":False}
 result=ROOT/"evidence/p12_local_BF16_oracle.json"
 with result.open("x") as f:json.dump(final,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
 print("P12_BF16_LOCAL_ORACLE_COMPLETE",json.dumps({"sha256":sha(result),
   "n_matrices":len(records),"bits":[5,6],"reference_is_full_model":False,
   "peak_rss_MB":round(final["max_RSS_bytes"]/1048576,2),
   "promotion":False},sort_keys=True),flush=True)

if __name__=="__main__":run()
