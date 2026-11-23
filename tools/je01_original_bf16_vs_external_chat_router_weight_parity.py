#!/usr/bin/env python3
"""JE01 BF16 original vs separately SHA-pinned MLX Chat F16 ROUTER weights.

These sources have different tokenizers and quantized non-router matrices. This
compares only the 26 pretrained MoE router gate matrices, in ORIGINAL BF16 vs
reference F16 representation, not their activations, routed behavior, logits
or quality. It fails closed on any unverified shard, shape, or numeric drift.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

SCHEMA="beglin-je01-original-vs-reference-26-router-weight-numeric-parity/1"
A=Path("/Volumes/D50/deepseek_v2lite_bf16_safetensors")
B=Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--DeepSeek-V2-Lite-Chat-4bit-mlx/snapshots/a33a8ad64477b9ca7a65b6dc1d8c06c997f87216")
A_INDEX="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
B_INDEX="1f840914b851127054b81b60ee1ad1a7412a218c170c7567770426289f7fd29c"
A_SOURCE_ID="36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc"
B_REFERENCE_ID="33730a3d5b3bec0ba72cea02f896a78ca367275d7fbf63e0e880e0e0c99b4d4c"
MAX_ALLOWED_ABS=1e-6
KEYS=[f"model.layers.{layer}.mlp.gate.weight" for layer in range(1,27)]


def canonical(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),
                             allow_nan=False).encode()).hexdigest()


def load_meta(root,expected_sha):
    raw=(root/"model.safetensors.index.json").read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError("JE01_MODEL_INDEX_SOURCE_DRIFT")
    return json.loads(raw)["weight_map"]


def take_weight(root,idx,key,dtype):
    name=idx.get(key)
    if not name or not name.startswith("model-") or not name.endswith(".safetensors"):
        raise ValueError("JE01_ROUTER_WEIGHT_NOT_INDEXED:"+key)
    file=root/name
    if not file.is_file() or file.stat().st_size<5000000:
        raise ValueError("JE01_ROUTER_SHARD_MISSING:"+name)
    with file.open("rb") as f:
        meta_length=struct.unpack("<Q",f.read(8))[0]
        if meta_length>32_000_000:raise ValueError("JE01_SHARD_HEADER_OVERFLOW")
        meta=json.loads(f.read(meta_length)).get(key)
        if not meta or meta["dtype"]!=dtype or meta["shape"]!=[64,2048]:
            raise ValueError("JE01_ROUTER_SHAPE_OR_DTYPE_DRIFT:"+key)
        start,end=meta["data_offsets"]
        if end-start!=64*2048*2:
            raise ValueError("JE01_ROUTER_BYTES_DRIFT:"+key)
        f.seek(8+meta_length+start)
        raw=f.read(end-start)
    if len(raw)!=64*2048*2:raise ValueError("JE01_TRUNCATED_ROUTER")
    data=((np.frombuffer(raw,dtype="<u2").astype(np.uint32)<<16).view(np.float32)
          if dtype=="BF16" else np.frombuffer(raw,dtype="<f2").astype(np.float32))
    if not np.isfinite(data).all():raise ValueError("JE01_ROUTER_NAN_OR_INF")
    return data,hashlib.sha256(raw).hexdigest()


def audit():
    aidx=load_meta(A,A_INDEX)
    bidx=load_meta(B,B_INDEX)
    recs=[]
    for layer,key in enumerate(KEYS,start=1):
        a,sha_a=take_weight(A,aidx,key,"BF16")
        b,sha_b=take_weight(B,bidx,key,"F16")
        diff=a.astype(np.float64)-b.astype(np.float64)
        rel=math.sqrt(float(np.mean(diff*diff))/float(np.mean(a.astype(np.float64)**2)))
        maxabs=float(np.max(np.abs(diff)))
        identical=int(np.count_nonzero(diff==0))
        recs.append({"layer":layer,"router_tensor":key,
                     "shape":[64,2048],
                     "original_BF16_data_sha256":sha_a,
                     "MLX_Chat_F16_data_sha256":sha_b,
                     "numerically_equal_elements":identical,
                     "total_elements":a.size,
                     "max_absolute_difference":maxabs,
                     "relative_rms_difference":rel})
        if maxabs>MAX_ALLOWED_ABS:raise ValueError("JE01_LARGE_ROUTER_WEIGHT_DRIFT_LAYER"+str(layer))
    report={
        "schema":SCHEMA,"status":"ALL_26_ROUTER_WEIGHT_NUMERIC_PARITY_PASS",
        "original_BF16_full_checkpoint_identity_sha256":A_SOURCE_ID,
        "reference_MLX_Chat4bit_checkpoint_identity_sha256":B_REFERENCE_ID,
        "compared_layers":list(range(1,27)),
        "n_routed_expert_gate_matrices":len(recs),
        "n_matrix_scalar_coefficients":sum(z["total_elements"] for z in recs),
        "max_absolute_coefficient_difference":max(z["max_absolute_difference"] for z in recs),
        "max_relative_rms_difference":max(z["relative_rms_difference"] for z in recs),
        "threshold_max_abs":MAX_ALLOWED_ABS,
        "per_layer_router_weights":recs,
        "same_model_tokenizer":False,
        "same_non_router_quantized_weights":"NOT_ESTABLISHED",
        "same_router_hidden_states":"NOT_MEASURED",
        "full_pretrained_BF16_model_forward":"NOT_RUN",
        "native_BEG_LIN_logit_parity":"NOT_MEASURED",
        "JE01_gold_token_NLL":"NOT_MEASURED",
        "production_promotion_allowed":False}
    report["receipt_sha256"]=canonical(report)
    return report


if __name__=="__main__":
    try:print(json.dumps(audit(),sort_keys=True,separators=(",",":"),allow_nan=False))
    except Exception as error:
        print(json.dumps({"schema":SCHEMA,"status":"BLOCKED","reason":str(error)[:240]}))
        raise SystemExit(2)
