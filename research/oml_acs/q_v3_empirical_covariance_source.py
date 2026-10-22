#!/usr/bin/env python3
"""Qv3 original BF16 Qwen empirical cross-channel local projection error.

Unlike v2 diagonal E[x_j**2]*E_weight[j], evaluates the exact empirical
quadratic form on ORIGINAL projection input activations from 12 PUBLIC
calibration prompts:
  D_i(b) = sum_inputs || x @ (W_b - W)^T ||^2 /
                       sum_inputs || source_BF16_projection(x) - bias ||^2.
No downstream model weight is replaced, no final-logit interventions,
no eval prompts, task labels or old V6 holdout accessed.

True original BEGLIN native C Q4G64 and qNg64-n8 encoders are used.
Quantized residual matrices cached as fp16 GPU buffers, with FP32 GEMV
to prevent exponential PCIe transfers. FP16 cache precision makes this a
high-precision approximation of empirical full covariance, NOT an exact
FP64 oracle; quantify that limitation and do not promote automatically.
"""
from __future__ import annotations
import ctypes,hashlib,json,os,pathlib,subprocess,time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT=pathlib.Path(__file__).resolve().parents[2]
OLD=ROOT/"configs/research/BEGLIN_Q_ACTIVATION_ASYNC_PARALLEL_V2_PRELOCK_20261022.json"
PRELOCK=ROOT/"configs/research/BEGLIN_Q_V3_EMPIRICAL_COVARIANCE_PRELOCK_20261022.json"
MODEL=pathlib.Path("/home/alienware-r13/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
OUT=pathlib.Path.home()/".local/share/beglin/oml_acs/20261022_qv3_empirical_covariance"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
SOURCE_OLD_SHA="0d9177b152286ccd413028e74838529e2a5aad57b42ec793df3095f99208d4af"
SOURCE_ENCODER_SHA={
 "gguf_transcode.c":"7371cc352226e912dbf42762e5e63a70e91acb5c90691d90e32401e18af7ec5f",
 "qng64_dense.c":"cb985ec48cfa1f1f9ebe8411d12da5d3f73720c8acdcad51fc279a2b25ff8956",
}

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def private(path:pathlib.Path,data:bytes):
    with path.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def quant_library():
    so=OUT/"libsource_BEG_nativeq4n8.so"
    subprocess.run(["gcc","-shared","-O3","-fPIC","-I",str(ROOT),
          str(ROOT/"gguf_transcode.c"),str(ROOT/"qng64_dense.c"),
          "-o",str(so),"-lm"],check=True)
    lib=ctypes.CDLL(str(so))
    ptr=ctypes.c_void_p
    lib.gguf_quantize_q4g64_error_feedback.argtypes=[ptr,ctypes.c_int,ctypes.c_int,ptr,ptr]
    lib.gguf_quantize_q4g64_error_feedback.restype=None
    lib.qng64_quantize_f32.argtypes=[ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr,ptr]
    lib.qng64_quantize_f32.restype=ctypes.c_int
    return lib,sha(so.read_bytes())
def source_errors(lib,module):
    w=module.weight.detach().float().cpu().numpy().copy()
    if not np.isfinite(w).all():raise RuntimeError("source FP32 weight nonfinite")
    out,inp=w.shape
    if inp%64:raise RuntimeError("invalid group64 original weight")
    nc=out*inp;g=nc//64
    p4=np.empty((nc//2,),dtype=np.uint8);s4=np.empty(g,dtype=np.float32)
    p8=np.empty((nc,),dtype=np.uint8);s8=np.empty(g,dtype=np.float32)
    lib.gguf_quantize_q4g64_error_feedback(w.ctypes.data,out,inp,p4.ctypes.data,s4.ctypes.data)
    if lib.qng64_quantize_f32(w.ctypes.data,out,inp,8,p8.ctypes.data,s8.ctypes.data):
        raise RuntimeError("invalid native qNg64 8bit quantized weight")
    ng=inp//64
    codes=p4.reshape(out,ng,32)
    unpack4=np.empty((out,ng,64),dtype=np.int8)
    unpack4[:,:,::2]=(codes&15).astype(np.int8)-8
    unpack4[:,:,1::2]=(codes>>4).astype(np.int8)-8
    w4=(unpack4.astype(np.float32)*s4.reshape(out,ng,1)).reshape(out,inp)
    planes=p8.reshape(out,ng,8,8)
    u=np.zeros((out,ng,64),dtype=np.uint16)
    for bit in range(8):
        u|=(np.unpackbits(planes[:,:,bit,:],bitorder="little",axis=-1).astype(np.uint16)<<bit)
    w8=((u.astype(np.int16)-128).astype(np.float32)*s8.reshape(out,ng,1)).reshape(out,inp)
    if not np.isfinite(w4).all() or not np.isfinite(w8).all():
        raise RuntimeError("nonfinite real BEGLIN code decode")
    d4=(w4-w).astype(np.float16)
    d8=(w8-w).astype(np.float16)
    check={"BF16_f32_weight_sha":sha(w.tobytes()),
      "packed_Q4G64_sha256":sha(p4.tobytes()),
      "native_qNg64_n8_bitplanes_sha256":sha(p8.tobytes()),
      "fp16_error_quantization":True}
    return torch.from_numpy(d4).to("cuda:0"),torch.from_numpy(d8).to("cuda:0"),check

def main():
    if OUT.exists():raise FileExistsError("previous v3 source calibration results exist")
    if not torch.cuda.is_available():raise RuntimeError("CUDA original BF16 source required")
    if not MODEL.is_dir() or not PRELOCK.is_file():raise RuntimeError("pinned source or v3 prereg absent")
    if sha(OLD.read_bytes())!=SOURCE_OLD_SHA:raise RuntimeError("previous frozen source prompt contract changed")
    for name,h in SOURCE_ENCODER_SHA.items():
        if sha((ROOT/name).read_bytes())!=h:raise RuntimeError("original BEGLIN C encoder changed")
    cfgraw=PRELOCK.read_bytes();cfg=json.loads(cfgraw)
    if cfg["status"]!="LOCKED_BEFORE_COVARIANCE_CALIBRATION_AND_NEW_PUBLIC_DEV_EVALUATION":
        raise RuntimeError("v3 protocol not pre-frozen")
    if cfg["model"]["pinned_official_bf16_rev"]!=MODEL.name:
        raise RuntimeError("source BF16 checkpoint revision mismatch")
    prev=json.loads(OLD.read_text());texts=prev["calibration_public_only"]
    if len(texts)!=12 or len(set(texts))!=12:raise RuntimeError("unexpected source calibration prompts")
    os.umask(0o077);OUT.mkdir(parents=True,mode=0o700)
    lib,so_hash=quant_library()
    torch.backends.cuda.matmul.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
    mdl=AutoModelForCausalLM.from_pretrained(str(MODEL),dtype=torch.bfloat16,
                    local_files_only=True,trust_remote_code=False,attn_implementation="sdpa")
    mdl=mdl.eval().to("cuda:0")
    modules=dict(mdl.named_modules())
    keys=[]
    for layer in range(28):
        for role in ROLES:
            part="self_attn" if role in ROLES[:4] else "mlp"
            name=f"model.layers.{layer}.{part}.{role}"
            if name not in modules:raise RuntimeError("original source model module disappeared")
            keys.append(name)
    if len(keys)!=196:raise RuntimeError("original model must have 196 projections")
    cache={}
    numerator={name:[0.,0.,0.] for name in keys}
    counts=defaultdict(int);extract_checks={}
    hooks=[]
    def make(name):
        def observer(module,inputs,output):
            if name not in cache:
                # Original checkpoint BF16 weights are read-only.
                delta4,delta8,check=source_errors(lib,module)
                cache[name]=(delta4,delta8)
                extract_checks[name]=check
            x=inputs[0].detach().reshape(-1,module.weight.shape[1]).float()
            y=output.detach().reshape(-1,module.weight.shape[0]).float()
            if module.bias is not None:
                y=y-module.bias.detach().float()
            d4,d8=cache[name]
            with torch.no_grad():
                # Source activations include cross-channel correlations exactly
                # in x@delta^T; FP16 residual cache + FP32 matmul approximation.
                p4=F.linear(x,d4.float())
                p8=F.linear(x,d8.float())
                numerator[name][0]+=float(torch.sum(p4.float()**2).item())
                numerator[name][1]+=float(torch.sum(p8.float()**2).item())
                numerator[name][2]+=float(torch.sum(y.float()**2).item())
            counts[name]+=int(x.shape[0])
        return observer
    for name in keys:hooks.append(modules[name].register_forward_hook(make(name)))
    total_tokens=0;lengths=[];t0=time.monotonic()
    try:
        with torch.inference_mode():
            for ix,text in enumerate(texts):
                encoded=tokenizer.apply_chat_template([{"role":"user","content":text}],
                        add_generation_prompt=True,tokenize=True,return_tensors="pt")
                ids=encoded["input_ids"] if not torch.is_tensor(encoded) else encoded
                if ids.ndim!=2 or not 8<=ids.shape[1]<=90:
                    raise RuntimeError("noncanonical Qwen source calibration token shape")
                total_tokens+=int(ids.shape[1]);lengths.append(int(ids.shape[1]))
                mdl(input_ids=ids.to("cuda:0"),use_cache=False,return_dict=True)
                print("Q3_FULL_COV_CAL_PROGRESS",ix+1,len(texts),
                        "CUDA_MEM_GB",round(torch.cuda.memory_allocated()/1e9,3),flush=True)
    finally:
        for h in hooks:h.remove()
    torch.cuda.synchronize()
    if len(cache)!=196 or len(counts)!=196 or set(counts.values())!={total_tokens}:
        raise RuntimeError("missing original BF16 source activation sample input counts")
    rows=[]
    for name in keys:
        u4,u8,d=numerator[name]
        if d<=0 or not np.isfinite([u4,u8,d]).all():
            raise RuntimeError("invalid original Qwen local projection energy")
        q4=float(u4/d);q8=float(u8/d)
        if not q4>q8>=0:
            # Reject unmeasured/local numerical reversals instead of hiding.
            raise RuntimeError("source local projection n8 not better than Q4 for "+name)
        parts=name.split(".")
        layer=int(parts[2]);role=parts[-1]
        rows.append(dict(layer=layer,role=role,
          local_full_input_covariance_Q4_relative_error=q4,
          local_full_input_covariance_n8_relative_error=q8,
          source_only_n8_upgrade_gain=q4-q8,
          tokens=counts[name],
          bf16_original_weight_sha256=extract_checks[name]["BF16_f32_weight_sha"],
          encoded_q4_sha256=extract_checks[name]["packed_Q4G64_sha256"],
          encoded_n8_sha256=extract_checks[name]["native_qNg64_n8_bitplanes_sha256"]))
    rows.sort(key=lambda v:(ROLES.index(v["role"]),v["layer"]))
    gains={role:[r["source_only_n8_upgrade_gain"] for r in rows if r["role"]==role] for role in ROLES}
    record={
     "schema":"BEGLIN-Q-v3-empirical-offdiagonal-source-projection-error/1",
     "status":"PASS_ORIGINAL_BF16_196_LOCAL_ERROR_NO_DOWNSTREAM_INTERVENTION",
     "model_source_rev":MODEL.name,
     "source_v3_protocol_sha256":sha(cfgraw),
     "source_v2_calibration_protocol_sha256":SOURCE_OLD_SHA,
     "native_C_quant_encoder_sha256":SOURCE_ENCODER_SHA,
     "native_quant_library_sha256":so_hash,
     "calibration_prompts":12,"original_source_tokens":total_tokens,
     "actual_module_count":196,
     "module_rows":rows,
     "quality_upgrade_by_role_layer":gains,
     "FP16_residual_cache_FP32_activation_matmul":True,
     "off_diagonal_covariance_retained_in_x_delta_projection":True,
     "exact_F64_covariance_computation":False,
     "no_downstream_final_output_interventions":True,
     "no_old_V6_or_private_task_labels":True,
     "no_eval_prompts_loaded":True,
     "production_main_unchanged":True,
     "elapsed_seconds":round(time.monotonic()-t0,2)
    }
    path=OUT/"v3_196_full_input_covariance_local_error_private.json"
    private(path,(json.dumps(record,sort_keys=True,indent=2)+"\n").encode())
    aggregate={k:v for k,v in record.items() if k!="module_rows"}
    aggregate["private_full_source_SHA256"]=sha(path.read_bytes())
    output=OUT/"v3_196_covariance_quality_public_aggregate.json"
    private(output,(json.dumps(aggregate,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_Q3_EMPIRICAL_LOCAL_COVARIANCE_SOURCE_PASS",
          json.dumps({"private_SHA256":sha(path.read_bytes()),
            "public_SHA256":sha(output.read_bytes()),
            "token_positions":total_tokens,"roles":196,
            "elapsed_s":record["elapsed_seconds"],
            "gains_range":[min(x["source_only_n8_upgrade_gain"] for x in rows),
                           max(x["source_only_n8_upgrade_gain"] for x in rows)]},sort_keys=True),flush=True)
if __name__=="__main__":main()
