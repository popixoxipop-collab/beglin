#!/usr/bin/env python3
"""JE01 external DeepSeek-V2-Lite-Chat 4bit real single-prompt MLX forward.

Strictly reference-only: NOT pretrained BF16 JE01 model, NOT native BEGLIN
runtime and NOT original-tokenizer parity, Expert NLL, or promotion evidence.
Uses only an already SHA-attested local HF cache; no network, production edits,
quantization file writes, or holdout prompts. Bypasses the installed MLX-LM
package __init__ importer that blocks on filelock's hard-link probe ON EOE,
without patching or disabling the system's filelock/os.link implementation.
"""
from __future__ import annotations
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time
import types
import traceback

SCHEMA="beglin-je01-external-mlx-chat4bit-first-pretrained-forward/1"
ROOT=Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--DeepSeek-V2-Lite-Chat-4bit-mlx")
REV="a33a8ad64477b9ca7a65b6dc1d8c06c997f87216"
TOK_SHA="c6276ee9e3cfa46033f54dc46db3f6dba0c6a1f6a2e547543124f33b44873db4"
CFG_SHA="67134636123453e474a5b72d611be6130f59f8fc7478aac7927cbe6f9cd44d71"
INDEX_SHA="1f840914b851127054b81b60ee1ad1a7412a218c170c7567770426289f7fd29c"
MODEL_SHA="33730a3d5b3bec0ba72cea02f896a78ca367275d7fbf63e0e880e0e0c99b4d4c"
SHARDS={"model-00001-of-00002.safetensors":5316545126,
        "model-00002-of-00002.safetensors":3523543576}
MLX_LIMIT_BYTES=10_200_000_000
PROMPT="The capital of France is"
started=time.monotonic()


def report(phase,**fields):
    print(json.dumps({"event":phase,"elapsed_s":round(time.monotonic()-started,3),**fields},
                     sort_keys=True),flush=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    snap=ROOT/"snapshots"/REV
    if not snap.is_dir():raise ValueError("JE01_REF_REV_MISSING")
    meta={}
    for name,ref in [("config.json",CFG_SHA),("tokenizer.json",TOK_SHA),
                     ("model.safetensors.index.json",INDEX_SHA)]:
        f=snap/name
        if sha(f)!=ref:raise ValueError("JE01_REF_SMALL_FILE_SHA_DRIFT:"+name)
        meta[name]=ref
    config=json.loads((snap/"config.json").read_text(encoding="utf-8"))
    index=json.loads((snap/"model.safetensors.index.json").read_text(encoding="utf-8"))
    if (config.get("model_type")!="deepseek_v2" or
        config.get("n_routed_experts")!=64 or
        config.get("num_experts_per_tok")!=6 or
        config.get("num_hidden_layers")!=27 or
        config.get("quantization")!={"bits":4,"group_size":64} or
        index.get("metadata",{}).get("total_size")!=8839977984 or
        set(index["weight_map"].values())!=set(SHARDS)):
        raise ValueError("JE01_REF_MODEL_CONFIG_DRIFT")
    for fn,sz in SHARDS.items():
        f=snap/fn
        if not f.is_file() or f.stat().st_size!=sz:
            raise ValueError("JE01_REF_SHARD_LENGTH_DRIFT")
        if not f.resolve(strict=True).is_relative_to(ROOT.resolve(strict=True)):
            raise ValueError("JE01_REF_SYMLINK_ESCAPE")
    return snap,config


def main():
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    import mlx.core as mx
    import mlx.nn as nn
    from tokenizers import Tokenizer
    snap,cfg=validate()
    report("REF_SHA_METADATA_VALIDATED",quantization=cfg["quantization"],
           model_loaded=False)
    if not hasattr(mx,"metal") or not hasattr(mx.metal,"set_memory_limit"):
        raise RuntimeError("JE01_MLX_MEMORY_CAP_UNSUPPORTED")
    mx.metal.set_memory_limit(MLX_LIMIT_BYTES)
    mx.metal.set_cache_limit(128*1024*1024)
    report("MLX_GPU_HARD_MEMORY_LIMIT_SET",memory_cap_bytes=MLX_LIMIT_BYTES)

    # Equivalent original installed DeepseekV2 model architecture source module,
    # but skip package root "__init__.py": importing the latter forces
    # huggingface_hub->filelock->_strict._probe_link_follow_symlinks and
    # hangs on this host's os.link; do not modify the third-party package.
    package=Path("/opt/homebrew/lib/python3.11/site-packages/mlx_lm")
    m=types.ModuleType("mlx_lm")
    m.__path__=[str(package)]
    m.__package__="mlx_lm"
    m.__spec__=importlib.machinery.ModuleSpec("mlx_lm",loader=None,is_package=True)
    sys.modules["mlx_lm"]=m
    from mlx_lm.models.deepseek_v2 import Model,ModelArgs
    if "filelock" in sys.modules:
        raise RuntimeError("JE01_UNEXPECTED_FILELOCK_SIDE_EFFECT")
    report("DEEPSEEKV2_ARCHITECTURE_IMPORTED",model_loaded=False)

    tkn=Tokenizer.from_file(str(snap/"tokenizer.json"))
    ids=tkn.encode(PROMPT,add_special_tokens=False).ids
    bos=int(cfg.get("bos_token_id",100000))
    if len(ids)<2 or len(ids)>16 or bos<0 or bos>=int(cfg["vocab_size"]):
        raise ValueError("JE01_SMALL_REFERENCE_PROMPT_TOKENS")
    token_ids=[bos,*ids]
    report("REFERENCE_PROMPT_TOKENIZED",
           prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
           length=len(token_ids),bos_id=bos,tokenizer_sha256=TOK_SHA)

    weights={}
    for fn in SHARDS:
        weights.update(mx.load(str(snap/fn)))
        report("PRETRAINED_REFERENCE_SHARD_MAPPED",filename=fn,
               n_tensor_keys=len(weights))
    model=Model(ModelArgs.from_dict(cfg))
    weights=model.sanitize(weights)

    quantization=cfg["quantization"]
    def predicate(p,module):
        if p in quantization:return quantization[p]
        return hasattr(module,"to_quantized") and (p+".scales") in weights
    nn.quantize(model,group_size=quantization["group_size"],bits=quantization["bits"],
                mode="affine",class_predicate=predicate)
    model.eval()
    model.load_weights(list(weights.items()),strict=True)
    report("PRETRAINED_REFERENCE_WEIGHTS_BOUND_STRICT",
           n_tensor_keys=len(weights),
           active_gpu_bytes=mx.metal.get_active_memory())
    del weights
    mx.eval(model.parameters())
    report("ALL_REFERENCE_WEIGHTS_MATERIALIZED",
           active_gpu_bytes=mx.metal.get_active_memory())

    x=mx.array([token_ids],dtype=mx.int32)
    logits=model(x)
    if logits.shape!=(1,len(token_ids),int(cfg["vocab_size"])):
        raise RuntimeError("JE01_INCORRECT_FULL_LOGIT_SHAPE")
    last=logits[0,-1,:]
    maxlogit=mx.max(last)
    minlogit=mx.min(last)
    next_id=mx.argmax(last)
    nll_like=mx.logsumexp(last)-maxlogit
    mx.eval(maxlogit,minlogit,next_id,nll_like)
    top_id=int(next_id.item())
    score=float(maxlogit.item())
    min_score=float(minlogit.item())
    logsum=float(nll_like.item())
    if not all(map(__import__("math").isfinite,(score,min_score,logsum))):
        raise RuntimeError("JE01_NONFINITE_REAL_LOGITS")
    result={
        "schema":SCHEMA,"status":"EXTERNAL_CHAT4BIT_REAL_PRETRAINED_FORWARD_PASS",
        "reference_checkpoint_identity_sha256":MODEL_SHA,
        "reference_tokenizer_sha256":TOK_SHA,
        "source_model_original_bf16_identity":"DIFFERENT_NOT_PARITY",
        "reference_model_type":"deepseek_v2",
        "reference_revision":REV,"prompt_sha256":hashlib.sha256(PROMPT.encode()).hexdigest(),
        "input_token_count":len(token_ids),"actual_bos_id":bos,
        "real_pretrained_full_model_layers":int(cfg["num_hidden_layers"]),
        "logit_vocabulary_size":int(cfg["vocab_size"]),
        "next_token_argmax":top_id,
        "max_logit":score,"min_logit":min_score,
        "max_token_negative_log_probability":logsum,
        "reference_mlx_gpu_active_bytes":mx.metal.get_active_memory(),
        "reference_mlx_gpu_peak_bytes":mx.metal.get_peak_memory(),
        "reference_mlx_hard_memory_limit":MLX_LIMIT_BYTES,
        "filelock_imported":False,
        "JE01_frozen_training_or_holdout_used":False,
        "BEGLIN_original_full_logits_parity":"NOT_TESTED",
        "JE01_real_routed_Expert_NLL":"NOT_MEASURED",
        "production_write_allowed":False,
    }
    print(json.dumps(result,sort_keys=True,separators=(",",":")),flush=True)


if __name__=="__main__":
    try:
        main()
    except BaseException as exc:
        report("BLOCKED",error_type=type(exc).__name__,
               error=str(exc)[:500],model_quality_claim=False)
        traceback.print_exc()
        raise SystemExit(2)
