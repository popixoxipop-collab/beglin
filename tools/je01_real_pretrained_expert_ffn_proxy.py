#!/usr/bin/env python3
"""JE01 actual pretrained BF16 Expert FFN quantization sensitivity, exploratory.

Read ONLY 8 predeclared Layer-11 DeepSeek-V2-Lite BF16 routed Experts' gate,
up, down from a full-SHA-pinned safetensors shard. Score deterministic SYNTHETIC
RMS-normalized hidden vectors and independent per-group affine 4/6/8-bit
reconstructions. No real router observations, gold labels, model-wide logits,
NLL, exact native qNg64 byte equivalence, or production promotion.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np

SCHEMA="beglin-je01-real-pretrained-expert-ffn-proxy/1"
ROOT=Path("/Volumes/D50/deepseek_v2lite_bf16_safetensors")
INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
BF16_ID="36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc"
BF16_SHARD2_SHA="b0bd088feada0af954614da3a5b179c4d4d5d8bb2482ec83194d101f3fe91796"
LAYER=11
EXPERTS=(0,7,15,23,31,39,47,55) # predeclared, NOT router-frequency selected
BITS=(4,6,8)
GROUP=64
N_INPUTS=16
SEED=20261117


def canonical(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),
                             allow_nan=False).encode()).hexdigest()


def affine_group_roundtrip(W,bits):
    """Exploratory signed affine min/max by contiguous groups; not BEGLIN codec."""
    if bits not in BITS or W.ndim!=2 or W.shape[1]%GROUP:
        raise ValueError("JE01_PROXY_SHAPE_OR_BIT_INVALID")
    arr=np.asarray(W,dtype=np.float32)
    o,i=arr.shape
    block=arr.reshape(o,i//GROUP,GROUP)
    lo=block.min(-1,keepdims=True)
    hi=block.max(-1,keepdims=True)
    scale=(hi-lo)/np.float32((1<<bits)-1)
    scale=np.where(scale>0,scale,1.0)
    q=np.rint((block-lo)/scale)
    out=(lo+np.clip(q,0,(1<<bits)-1)*scale).reshape(o,i)
    return out.astype(np.float32,copy=False)


def sigmoid_silu(x):
    x=np.asarray(x,dtype=np.float32)
    clipped=np.maximum(np.minimum(x,np.float32(40)),np.float32(-40))
    return x/(np.float32(1)+np.exp(-clipped))


def ffn(inp,parts):
    gate=inp@parts["gate"].T
    up=inp@parts["up"].T
    out=sigmoid_silu(gate)*up
    return out@parts["down"].T


def bf16_to_f32(raw,shape):
    cnt=math.prod(shape)
    if len(raw)!=2*cnt:
        raise ValueError("JE01_WEIGHT_BUFFER_SIZE")
    words=np.frombuffer(raw,dtype="<u2")
    return (words.astype(np.uint32)<<16).view(np.float32).reshape(shape).copy()


def load_header(f):
    data=f.read(8)
    if len(data)!=8:raise ValueError("JE01_SHARD_HEADER_MISSING")
    n=struct.unpack("<Q",data)[0]
    if n<64 or n>32_000_000:
        raise ValueError("JE01_SHARD_HEADER_BOUNDS")
    raw=f.read(n)
    if len(raw)!=n:raise ValueError("JE01_TRUNCATED_HEADER")
    return json.loads(raw),8+n


def native_weight(f,headers,origin,key,shape):
    meta=headers.get(key)
    if not isinstance(meta,dict) or meta.get("dtype")!="BF16" or meta.get("shape")!=list(shape):
        raise ValueError("JE01_SOURCE_TENSOR_META_DRIFT:"+key)
    start,end=meta["data_offsets"]
    if type(start)!=int or type(end)!=int or end-start!=2*math.prod(shape):
        raise ValueError("JE01_SOURCE_DATA_OFFSET_DRIFT:"+key)
    f.seek(origin+start)
    raw=f.read(end-start)
    if len(raw)!=end-start:
        raise ValueError("JE01_TRUNCATED_WEIGHT:"+key)
    return bf16_to_f32(raw,shape),hashlib.sha256(raw).hexdigest()


def audit():
    index_raw=(ROOT/"model.safetensors.index.json").read_bytes()
    if hashlib.sha256(index_raw).hexdigest()!=INDEX_SHA:
        raise ValueError("JE01_SOURCE_INDEX_SHA_DRIFT")
    names=json.loads(index_raw)["weight_map"]
    shard="model-00002-of-000004.safetensors"
    shard_path=ROOT/shard
    if not shard_path.is_file() or shard_path.is_symlink() or shard_path.stat().st_size!=8591757448:
        raise ValueError("JE01_SHARD2_SOURCE_MISSING")
    # BF16 full shard SHA was already measured and sealed separately; this
    # run separately hashes each consumed tensor's original BF16 bytes.
    rng=np.random.default_rng(SEED)
    X=rng.normal(loc=0,scale=1,size=(N_INPUTS,2048)).astype(np.float32)
    X/=np.sqrt(np.mean(X*X,axis=1,keepdims=True)+1e-12)
    input_sha=hashlib.sha256(X.tobytes()).hexdigest()
    result=[]
    with shard_path.open("rb") as fh:
        headers,origin=load_header(fh)
        for expert in EXPERTS:
            parts={}; tensor_sha={}
            for role,shape in (("gate",(1408,2048)),("up",(1408,2048)),
                               ("down",(2048,1408))):
                key=f"model.layers.{LAYER}.mlp.experts.{expert}.{role}_proj.weight"
                if names.get(key)!=shard:raise ValueError("JE01_ROUTED_WEIGHT_MAP_DRIFT")
                parts[role],tensor_sha[role]=native_weight(fh,headers,origin,key,shape)
                if not np.isfinite(parts[role]).all():raise ValueError("JE01_NAN_WEIGHT")
            base=ffn(X,parts)
            if not np.isfinite(base).all():raise ValueError("JE01_NAN_BF16_FFN")
            base_rms=float(np.sqrt(np.mean(base**2,dtype=np.float64)))
            if base_rms<=0:raise ValueError("JE01_ZERO_BASELINE_OUTPUT")
            losses={}
            for bits in BITS:
                qparts={role:affine_group_roundtrip(W,bits)
                        for role,W in parts.items()}
                qout=ffn(X,qparts)
                if not np.isfinite(qout).all():raise ValueError("JE01_NAN_QUANT_OUTPUT")
                err=np.asarray(qout,dtype=np.float64)-np.asarray(base,dtype=np.float64)
                relative_rms=float(np.sqrt(np.mean(err*err))/base_rms)
                losses[str(bits)]={"relative_output_rmse":relative_rms,
                                   "relative_output_mse":relative_rms**2}
            result.append({"layer":LAYER,"expert_id":expert,
                           "consumed_bf16_tensor_sha256":tensor_sha,
                           "baseline_output_rms":base_rms,
                           "bit_proxy_loss":losses})
    doc={
        "schema":SCHEMA,"status":"REAL_PRETRAINED_WEIGHTS_SYNTHETIC_ACTIVATIONS_ONLY",
        "original_bf16_checkpoint_identity_sha256":BF16_ID,
        "source_shard_index_sha256":INDEX_SHA,
        "previously_attested_full_shard_sha256":BF16_SHARD2_SHA,
        "synthetic_inputs_sha256":input_sha,"synthetic_seed":SEED,
        "num_synthetic_inputs":N_INPUTS,"layer":LAYER,
        "experts":list(EXPERTS),"quantization_bits":list(BITS),
        "codec":"EXPLORATORY_ROW_GROUP_AFFINE_MINMAX_NOT_NATIVE_BEGLIN",
        "groupsize":GROUP,"results":result,
        "actual_pretrained_weights_used":True,
        "actual_real_router_hidden_states":False,
        "full_pretrained_model_forward":"NOT_RUN",
        "per_prompt_gold_token_nll":"NOT_MEASURED",
        "same_byte_budget_certification":"NOT_MEASURED",
        "production_promotion_allowed":False
    }
    doc["receipt_sha256"]=canonical(doc)
    return doc


if __name__=="__main__":
    try:print(json.dumps(audit(),sort_keys=True,separators=(",",":"),allow_nan=False))
    except Exception as e:
        print(json.dumps({"schema":SCHEMA,"status":"BLOCKED","error":str(e)[:200]}))
        raise SystemExit(2)
