#!/usr/bin/env python3
"""External real DeepSeek Chat4bit routed 64way softmax + same-model logits parity.

This observes actual pretrained model routing for a fixed PUBLIC short prompt,
not the original SHA-attested BF16 JE01 model, not its routed calibration
inputs and not BEGLIN native inference. Only a 1-layer (L11) trace, all token
positions. Instrumentation is performed in a separate Python process and
restored immediately, never modifying packages or production files. Full
reference model logits before and after instrumentation must numerically agree.
"""
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import types
import traceback

SCHEMA="beglin-je01-external-chat4bit-real-router-64way-l11/1"
ROOT=Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--DeepSeek-V2-Lite-Chat-4bit-mlx")
REV="a33a8ad64477b9ca7a65b6dc1d8c06c997f87216"
CHECKPOINT_SHA="33730a3d5b3bec0ba72cea02f896a78ca367275d7fbf63e0e880e0e0c99b4d4c"
SOURCE_FORWARD_SCRIPT_BLOB="52630b35071d4e52f7afb6a4c0f2714a6c917326"
LAYER=11
SAMPLE_PROMPT="The capital of France is"
MEM_CAP=10_200_000_000
start=time.monotonic()


def signal(step,**v):
    print(json.dumps({"event":step,"elapsed_s":round(time.monotonic()-start,3),**v},
                     sort_keys=True),flush=True)


def setup_package():
    pkg=types.ModuleType("mlx_lm")
    pkg.__path__=["/opt/homebrew/lib/python3.11/site-packages/mlx_lm"]
    pkg.__package__="mlx_lm"
    pkg.__spec__=importlib.machinery.ModuleSpec("mlx_lm",loader=None,is_package=True)
    sys.modules["mlx_lm"]=pkg
    from mlx_lm.models.deepseek_v2 import Model,ModelArgs,MoEGate
    if "filelock" in sys.modules:raise RuntimeError("JE01_UNEXPECTED_FILELOCK_INIT")
    return Model,ModelArgs,MoEGate


def load_ref(mx,nn,Model,ModelArgs,ref):
    snap=ROOT/"snapshots"/REV
    _,cfg=ref.validate()
    weights={}
    for name in ref.SHARDS:
        weights.update(mx.load(str(snap/name)))
    model=Model(ModelArgs.from_dict(cfg))
    weights=model.sanitize(weights)
    quant=cfg["quantization"]
    def predicate(p,m):
        return (quant[p] if p in quant else
                hasattr(m,"to_quantized") and (p+".scales") in weights)
    nn.quantize(model,group_size=quant["group_size"],bits=quant["bits"],
                mode="affine",class_predicate=predicate)
    model.eval()
    model.load_weights(list(weights.items()),strict=True)
    del weights
    mx.eval(model.parameters())
    return model,cfg


def run():
    os.environ.update({"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
                       "TOKENIZERS_PARALLELISM":"false"})
    from tokenizers import Tokenizer
    import mlx.core as mx
    import mlx.nn as nn
    mx.set_memory_limit(MEM_CAP)
    mx.set_cache_limit(128*1024*1024)
    model_file=Path("/Users/eoe/mcp-sandbox/tailnet-commander/JE01_EXTERNAL_CHAT4BIT_FIRST_REAL_FORWARD_20261117.py")
    raw=model_file.read_bytes()
    git_sha=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    if git_sha!=SOURCE_FORWARD_SCRIPT_BLOB:
        raise ValueError("JE01_PINNED_RESEARCH_FORWARD_LOADER_DRIFT")
    spec=importlib.util.spec_from_file_location("je01_external_sha_pinned",model_file)
    ref=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    Model,ModelArgs,MoEGate=setup_package()
    model,cfg=load_ref(mx,nn,Model,ModelArgs,ref)
    signal("REAL_REFERENCE_PRETRAINED_LOADED",
           memory_active_bytes=mx.get_active_memory(),
           reference_identity=CHECKPOINT_SHA)
    tok=Tokenizer.from_file(str(ROOT/"snapshots"/REV/"tokenizer.json"))
    input_ids=[int(cfg["bos_token_id"]),*tok.encode(SAMPLE_PROMPT,add_special_tokens=False).ids]
    if not 3<=len(input_ids)<=16:raise ValueError("JE01_INVALID_SHORT_PROMPT")
    mxinput=mx.array([input_ids],dtype=mx.int32)
    original_logits=model(mxinput)
    mx.eval(original_logits)
    signal("UNINSTRUMENTED_REAL_MODEL_FORWARD_PASS",
           original_last_argmax=int(mx.argmax(original_logits[0,-1,:]).item()))

    selected_gate=model.model.layers[LAYER].mlp.gate
    original_call=MoEGate.__call__
    captured=[]
    def audit(self,x):
        if self is selected_gate:
            softmax_all=mx.softmax(x @ self.weight.T, axis=-1, precise=True)
            indices,weights=original_call(self,x)
            captured.append((softmax_all,indices,weights))
            return indices,weights
        return original_call(self,x)
    MoEGate.__call__=audit
    try:
        observed_logits=model(mxinput)
        mx.eval(observed_logits)
    finally:
        MoEGate.__call__=original_call
    if len(captured)!=1:
        raise ValueError("JE01_REAL_ROUTER_NOT_CALLED_EXACTLY_ONCE")
    distribution, selected, selected_weights=captured[0]
    mx.eval(distribution,selected,selected_weights)
    parity=float(mx.max(mx.abs(original_logits-observed_logits)).item())
    argmax_a=int(mx.argmax(original_logits[0,-1,:]).item())
    argmax_b=int(mx.argmax(observed_logits[0,-1,:]).item())
    if parity>1e-4 or argmax_a!=argmax_b:
        raise ValueError("JE01_INSTRUMENTATION_CHANGED_LOGITS")
    full_probs=distribution.tolist()
    ids=selected.tolist()
    weights=selected_weights.tolist()
    if len(full_probs)!=1 or len(full_probs[0])!=len(input_ids):
        raise ValueError("JE01_REAL_TOKEN_EVENT_COUNT_DRIFT")
    events=[]
    for t,(row,chosen,chosen_probs) in enumerate(zip(full_probs[0],ids[0],weights[0])):
        if len(row)!=64 or len(chosen)!=6 or len(set(chosen))!=6:
            raise ValueError("JE01_INVALID_ROUTER_SHAPE")
        if any(not math.isfinite(float(p)) or p<0 for p in row):
            raise ValueError("JE01_ROUTER_NONFINITE")
        prob_sum=sum(row)
        # Original native reference MoEGate returns a float16 softmax; FP16
        # element-wise rounding can shift sum(64 entries) by >1e-4 even
        # with mathematically normalized true probabilities. The recorded
        # 1.0001554489135742 at position 0 grounds an explicit FP16-only
        # 0.001 tolerance, while float32 retains 1e-4; no renormalization.
        dtype_name=str(distribution.dtype)
        sum_tolerance=0.001 if dtype_name=="mlx.core.float16" else 1e-4
        if abs(prob_sum-1.0)>sum_tolerance:
            signal("ROUTER_SOFTMAX_ROUNDING_DIAGNOSTIC",
                   position=t,probability_sum=prob_sum,
                   absolute_sum_error=abs(prob_sum-1.0),
                   softmax_dtype=str(distribution.dtype),
                   min_probability=min(row),max_probability=max(row),
                   reference_selected=chosen,
                   reference_selected_weights=chosen_probs)
            raise ValueError("JE01_ROUTER_PROB_NOT_NORMALIZED")
        top6=sorted(range(64),key=lambda i:(-row[i],i))[:6]
        if set(top6)!=set(chosen):
            raise ValueError("JE01_REAL_ROUTER_TOP6_MISMATCH")
        if any(abs(row[k]-float(w))>5e-4 for k,w in zip(chosen,chosen_probs)):
            raise ValueError("JE01_REAL_ROUTER_WEIGHT_MISMATCH")
        min_selected=min(row[k] for k in chosen)
        max_unselected=max(row[k] for k in range(64) if k not in chosen)
        events.append({
            "token_position":t,
            "full_64way_softmax":row,
            "top6_chosen_ids_from_real_reference":chosen,
            "top6_chosen_weights":chosen_probs,
            "selected_unselected_boundary_probability_gap":min_selected-max_unselected,
            "probability_sum":prob_sum
        })
    result={
        "schema":SCHEMA,"status":"REAL_PRETRAINED_REFERENCE_ROUTER_AND_LOGIT_PARITY_PASS",
        "reference_model":"DeepSeek-V2-Lite-Chat-4bit-mlx",
        "reference_revision":REV,
        "reference_weights_sha256":CHECKPOINT_SHA,
        "reference_tokenizer_sha256":ref.TOK_SHA,
        "original_je01_model_sha":"DIFFERENT_NOT_EQUIVALENT",
        "model_is_original_beglin":False,
        "layers_actually_forwarded":27,
        "router_trace_layer":LAYER,
        "routed_experts_count":64,"per_token_topk":6,
        "prompt_sha256":hashlib.sha256(SAMPLE_PROMPT.encode()).hexdigest(),
        "token_ids_sha256":hashlib.sha256(json.dumps(input_ids,separators=(",",":")).encode()).hexdigest(),
        "n_real_token_events":len(events),
        "baseline_reference_last_argmax":argmax_a,
        "instrumented_reference_last_argmax":argmax_b,
        "full_logits_max_abs_diff":parity,
        "reference_mlx_active_bytes":mx.get_active_memory(),
        "reference_mlx_peak_bytes":mx.get_peak_memory(),
        "reference_mlx_memory_limit_bytes":MEM_CAP,
        "router_softmax_dtype":str(distribution.dtype),
        "router_probability_sum_tolerance":0.001 if str(distribution.dtype)=="mlx.core.float16" else 1e-4,
        "max_observed_probability_sum_error":max(abs(x["probability_sum"]-1) for x in events),
        "router_events":events,
        "JE01_training_or_holdout_inputs_used":False,
        "original_beglin_native_logits_parity":"NOT_MEASURED",
        "true_beglin_EXPERT_NLL":"NOT_MEASURED",
        "production_write_allowed":False
    }
    print(json.dumps(result,sort_keys=True,separators=(",",":")),flush=True)


if __name__=="__main__":
    try:run()
    except BaseException as error:
        signal("BLOCKED",error_type=type(error).__name__,
               explanation=str(error)[:400])
        traceback.print_exc()
        raise SystemExit(2)
