#!/usr/bin/env python3
"""JE01 real pretrained external Chat4bit 26-layer 64way route/source-gate audit.

Actual reference full 27-layer forward on one fixed NON-JE01-study public prompt.
For each routed layer 1..26, independently compute two full64 softmax sets on
THE SAME real hidden activation: reference F16 gate and original BF16 source
gate rounded to F16. Need not and does NOT claim original BF16 hidden states,
end-to-end logits or matched numerical precision of all other weights.
Saves all 156 real token-layer full64 probabilities in compact IEEE FP16
base64, plus corresponding full source-gate FP16 outputs and exact selections.
NO original user data; NO official JE01 512 corpus; NO production writes.
"""
import base64
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import types
import traceback
import numpy as np

SCHEMA="beglin-je01-external-chat4bit-full26-real-source-gate-router/1"
ROOT=Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--DeepSeek-V2-Lite-Chat-4bit-mlx")
REV="a33a8ad64477b9ca7a65b6dc1d8c06c997f87216"
SOURCE_ROOT=Path("/Volumes/D50/deepseek_v2lite_bf16_safetensors")
SOURCE_INDEX="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
SOURCE_BF16_ID="36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc"
REF_MODEL_ID="33730a3d5b3bec0ba72cea02f896a78ca367275d7fbf63e0e880e0e0c99b4d4c"
MEMORY_CAP=10_200_000_000
PROMPT="The capital of France is"
REFERENCE_LOADER_GITBLOB="52630b35071d4e52f7afb6a4c0f2714a6c917326"


def canonical(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),
                  allow_nan=False).encode()).hexdigest()


def original_gates():
    index_bytes=(SOURCE_ROOT/"model.safetensors.index.json").read_bytes()
    if hashlib.sha256(index_bytes).hexdigest()!=SOURCE_INDEX:
        raise ValueError("JE01_SOURCE_INDEX_SHA_DRIFT")
    mapping=json.loads(index_bytes)["weight_map"]
    matrices={}
    provenance={}
    for layer in range(1,27):
        key=f"model.layers.{layer}.mlp.gate.weight"
        shard_name=mapping.get(key)
        if not shard_name or "/" in shard_name or not shard_name.endswith(".safetensors"):
            raise ValueError("JE01_SOURCE_ROUTER_KEY_MISSING")
        p=SOURCE_ROOT/shard_name
        with p.open("rb") as f:
            n=struct.unpack("<Q",f.read(8))[0]
            if n>32_000_000:raise ValueError("JE01_SOURCE_HEADER_SIZE")
            header=json.loads(f.read(n))[key]
            if header["dtype"]!="BF16" or header["shape"]!=[64,2048]:
                raise ValueError("JE01_SOURCE_ROUTER_DTYPE_SHAPE_DRIFT")
            lo,hi=header["data_offsets"]
            if hi-lo!=262144:raise ValueError("JE01_SOURCE_ROUTER_BYTE_SIZE_DRIFT")
            f.seek(8+n+lo)
            raw=f.read(hi-lo)
        if len(raw)!=262144:raise ValueError("JE01_SOURCE_ROUTER_TRUNCATED")
        np_f32=(np.frombuffer(raw,dtype="<u2").astype(np.uint32)<<16).view(np.float32)
        matrices[layer]=np_f32.reshape(64,2048).copy()
        provenance[str(layer)]=hashlib.sha256(raw).hexdigest()
    return matrices,provenance


def main():
    os.environ.update({"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
                       "TOKENIZERS_PARALLELISM":"false"})
    import mlx.core as mx
    import mlx.nn as nn
    from tokenizers import Tokenizer
    mx.set_memory_limit(MEMORY_CAP)
    mx.set_cache_limit(128*1024*1024)
    source_file=Path("/Users/eoe/mcp-sandbox/tailnet-commander/JE01_EXTERNAL_CHAT4BIT_FIRST_REAL_FORWARD_20261117.py")
    b=source_file.read_bytes()
    actual=hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
    if actual!=REFERENCE_LOADER_GITBLOB:
        raise ValueError("JE01_SOURCE_PINNED_REFERENCE_LOADER_DRIFT")
    spec=importlib.util.spec_from_file_location("je01_ref_forward_verified",source_file)
    ref=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    pkg=types.ModuleType("mlx_lm")
    pkg.__path__=["/opt/homebrew/lib/python3.11/site-packages/mlx_lm"]
    pkg.__package__="mlx_lm"
    pkg.__spec__=importlib.machinery.ModuleSpec("mlx_lm",loader=None,is_package=True)
    sys.modules["mlx_lm"]=pkg
    from mlx_lm.models.deepseek_v2 import Model,ModelArgs,MoEGate
    if "filelock" in sys.modules:raise RuntimeError("JE01_FILELOCK_SIDE_EFFECT")
    snap,cfg=ref.validate()
    if cfg.get("quantization")!={"bits":4,"group_size":64}:
        raise ValueError("JE01_EXTERNAL_MODEL_QUANTIZATION_DRIFT")
    weights={}
    for name in ref.SHARDS:
        weights.update(mx.load(str(snap/name)))
    model=Model(ModelArgs.from_dict(cfg))
    weights=model.sanitize(weights)
    def predicate(p,m):
        return (cfg["quantization"][p] if p in cfg["quantization"]
                else hasattr(m,"to_quantized") and (p+".scales") in weights)
    nn.quantize(model,group_size=64,bits=4,mode="affine",class_predicate=predicate)
    model.eval()
    model.load_weights(list(weights.items()),strict=True)
    del weights
    mx.eval(model.parameters())
    print(json.dumps({"step":"REAL_REFERENCE_MODEL_LOADED",
                      "gpu_active_bytes":mx.get_active_memory()}),flush=True)

    orig_matrices,orig_bf16_shas=original_gates()
    # The original BF16 gates are represented in this reference's exact F16
    # arithmetic for same-input router parity; no original BF16 main model.
    gate_arrays={l:mx.array(orig_matrices[l],dtype=mx.float16)
                 for l in range(1,27)}
    mx.eval(list(gate_arrays.values()))
    del orig_matrices
    tok=Tokenizer.from_file(str(snap/"tokenizer.json"))
    ids=[int(cfg["bos_token_id"]),*tok.encode(PROMPT,add_special_tokens=False).ids]
    if len(ids)!=6:raise ValueError("JE01_PROMPT_TOKENS_DRIFT")
    inp=mx.array([ids],dtype=mx.int32)
    baseline=model(inp)
    mx.eval(baseline)
    lastarg=int(mx.argmax(baseline[0,-1,:]).item())
    if lastarg!=8913:raise ValueError("JE01_REFERENCE_BASELINE_ARGMAX_DRIFT")
    print(json.dumps({"step":"BASELINE_REFERENCE_FORWARD_PASS","argmax":lastarg}),flush=True)

    gates={id(model.model.layers[l].mlp.gate):l for l in range(1,27)}
    if len(gates)!=26:raise ValueError("JE01_REAL_GATE_COLLECTION_DRIFT")
    orig=MoEGate.__call__
    capture={}
    def audited(self,x):
        layer=gates.get(id(self))
        if layer is None:return orig(self,x)
        ref_all=mx.softmax(x@self.weight.T,axis=-1,precise=True)
        source_all=mx.softmax(x@gate_arrays[layer].T,axis=-1,precise=True)
        chosen,chosen_p=orig(self,x)
        if layer in capture:raise ValueError("JE01_DUPLICATE_ROUTER_LAYER")
        capture[layer]=(ref_all,source_all,chosen,chosen_p)
        return chosen,chosen_p
    MoEGate.__call__=audited
    try:
        observed=model(inp)
        mx.eval(observed)
    finally:
        MoEGate.__call__=orig
    if len(capture)!=26:raise ValueError("JE01_ROUTER_LAYER_NOT_ALL_CAPTURED")
    parity=float(mx.max(mx.abs(observed-baseline)).item())
    if parity>1e-4:raise ValueError("JE01_INSTRUMENTATION_CHANGED_PRETRAINED_LOGITS")
    allarrays=[value for row in capture.values() for value in row]
    mx.eval(allarrays)
    print(json.dumps({"step":"INSTRUMENTED_26_LAYER_FORWARD_PASS",
                      "max_abs_logits_change":parity,"gpu_active_bytes":mx.get_active_memory()}),flush=True)
    events=[];hard_flips=0;soft_ties=0;max_source_ref_diff=0.0;max_sum_err=0.0
    for layer in range(1,27):
        r_all,s_all,inds,wgts=capture[layer]
        r_mat=r_all.tolist()[0]
        s_mat=s_all.tolist()[0]
        idx_mat=inds.tolist()[0]
        w_mat=wgts.tolist()[0]
        for pos,(rr,ss,chosen,weights) in enumerate(zip(r_mat,s_mat,idx_mat,w_mat)):
            if len(rr)!=64 or len(ss)!=64 or len(chosen)!=6 or len(set(chosen))!=6:
                raise ValueError("JE01_INVALID_ROUTER_SHAPES")
            sm=sum(rr)
            smerr=abs(sm-1)
            if smerr>.001:raise ValueError("JE01_FP16_ROUTER_NORMALIZER_DRIFT")
            max_sum_err=max(max_sum_err,smerr)
            max_ref_source=max(abs(x-y) for x,y in zip(rr,ss))
            max_source_ref_diff=max(max_source_ref_diff,max_ref_source)
            if max_ref_source>0.0005:raise ValueError("JE01_SOURCE_GATE_SAME_HIDDEN_MISMATCH")
            others=[rr[e] for e in range(64) if e not in chosen]
            margin=min(rr[e] for e in chosen)-max(others)
            if margin<0:hard_flips+=1
            elif margin==0:soft_ties+=1
            if any(abs(rr[e]-w)>.0005 for e,w in zip(chosen,weights)):
                raise ValueError("JE01_REAL_REFERENCE_SELECTED_SCORE_MISMATCH")
            # Full source and reference 64way probabilities are half-precision
            # bit-exact encoded, no rounding through JSON floats.
            events.append({
                "layer":layer,"token_position":pos,
                "reference_fp16_softmax64_b64":base64.b64encode(struct.pack("<64e",*rr)).decode(),
                "source_BF16_gate_on_reference_hidden_fp16_softmax64_b64":
                    base64.b64encode(struct.pack("<64e",*ss)).decode(),
                "chosen6_expert_ids":chosen,"selected_probs":weights,
                "boundary_margin":margin,"fp16_sum":sm,
                "source_reference_max_abs_prob_diff":max_ref_source})
    if len(events)!=26*6 or hard_flips:
        raise ValueError("JE01_REAL_ROUTER_EVENTS_OR_HARD_FLIPS")
    doc={
        "schema":SCHEMA,
        "status":"REAL_26_LAYER_SOURCE_GATE_SAME_REFERENCE_HIDDEN_ROUTER_PASS",
        "original_BF16_full_checkpoint_identity_sha256":SOURCE_BF16_ID,
        "external_MLX_Chat4bit_checkpoint_identity_sha256":REF_MODEL_ID,
        "reference_tokenizer_sha256":ref.TOK_SHA,
        "source_and_reference_tensors_not_identical_training_inference":True,
        "source_BF16_router_gates_converted_to_reference_fp16":True,
        "same_reference_model_hidden_states_for_both_router_gates":True,
        "original_BF16_model_hidden_states":"NOT_USED_OR_MEASURED",
        "layers_in_actual_reference_forward":27,
        "routed_layers_monitored":26,
        "total_real_reference_token_layer_events":len(events),
        "prompt_sha256":hashlib.sha256(PROMPT.encode()).hexdigest(),
        "sample_token_positions_per_layer":len(ids),
        "full_model_reference_logits_change_max_abs":parity,
        "original_BF16_router_matrix_per_layer_sha256":orig_bf16_shas,
        "max_BF16_gate_vs_ref_gate_same_hidden_probability_difference":max_source_ref_diff,
        "max_observed_reference_fp16_softmax_sum_error":max_sum_err,
        "hard_topk_selection_mismatches":hard_flips,
        "ambiguous_boundary_tie_events":soft_ties,
        "all64_score_codec":"base64_little_endian_ieee754_fp16_64_values",
        "actual_reference_router_events":events,
        "mlx_gpu_active_bytes":mx.get_active_memory(),
        "mlx_gpu_peak_bytes":mx.get_peak_memory(),
        "mlx_memory_cap_bytes":MEMORY_CAP,
        "JE01_512_official_training_and_holdout_used":False,
        "original_BF16_full_model_logits_parity":"NOT_TESTED",
        "original_BEG_LIN_full_model_gold_NLL":"NOT_MEASURED",
        "production_write_allowed":False}
    doc["receipt_sha256"]=canonical(doc)
    print(json.dumps(doc,sort_keys=True,separators=(",",":"),allow_nan=False),flush=True)


if __name__=="__main__":
    try:main()
    except BaseException as error:
        print(json.dumps({"schema":SCHEMA,"status":"BLOCKED",
                          "error":str(error)[:300],"kind":type(error).__name__}),flush=True)
        traceback.print_exc()
        raise SystemExit(2)
