#!/usr/bin/env python3
"""Read-only complete DeepSeek-V2-Lite BF16 source identity on EOE D50.

No runtime, model load, file writes, token evaluation or GPU allocation.
The source model SHA is not the quantized running-worker checkpoint SHA.
"""
import hashlib
import json
from pathlib import Path

SCHEMA="beglin-je01-native-bf16-checkpoint-sha-v1"
MODEL=Path("/Volumes/D50/deepseek_v2lite_bf16_safetensors")
EXPECTED_CONFIG="f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d"
EXPECTED_INDEX="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
EXPECTED_SHARDS=[8594887408,8591757448,8590718520,5636263200]

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError("P0_MISSING_OR_SYMLINK:"+path.name)
    if MODEL.resolve() not in path.resolve().parents:
        raise ValueError("P0_PATH_ESCAPE")
    return path

def collect():
    cfgp,idxp,tokp=[read(MODEL/x) for x in ("config.json","model.safetensors.index.json","tokenizer.json")]
    cfg=json.loads(cfgp.read_text())
    idx=json.loads(idxp.read_text())
    if not all((cfg.get("model_type")=="deepseek_v2",cfg.get("n_routed_experts")==64,
                cfg.get("num_experts_per_tok")==6,cfg.get("norm_topk_prob") is False,
                cfg.get("first_k_dense_replace")==1,idx.get("metadata",{}).get("total_size")==31412968448)):
        raise ValueError("P0_MODEL_CONFIG_OR_WEIGHT_TOTAL_DRIFT")
    cfg_hash,idx_hash,tok_hash=sha(cfgp),sha(idxp),sha(tokp)
    if (cfg_hash,idx_hash)!=(EXPECTED_CONFIG,EXPECTED_INDEX):
        raise ValueError("P0_CONFIG_INDEX_SHA_DRIFT")
    shard_names=sorted(set(idx.get("weight_map",{}).values()))
    if shard_names!=[f"model-{i:05d}-of-000004.safetensors" for i in range(1,5)]:
        raise ValueError("P0_UNEXPECTED_MODEL_SHARDS")
    shards=[]
    for name,size in zip(shard_names,EXPECTED_SHARDS):
        p=read(MODEL/name)
        actual=p.stat().st_size
        if actual!=size:
            raise ValueError("P0_SHARD_BYTES_DRIFT:"+name)
        shards.append({"name":name,"size_bytes":actual,"sha256":sha(p)})
    model={"model_type":"deepseek_v2","n_routed_experts":64,
           "num_experts_per_tok":6,"norm_topk_prob":False,
           "first_k_dense_replace":1,
           "config_sha256":cfg_hash,"tokenizer_sha256":tok_hash,
           "index_sha256":idx_hash,"shards":shards}
    receipt={"schema":SCHEMA,"checkpoint_identity_sha256":digest(model),
             "model":model,"raw_model_checkpoint_byte_total":sum(EXPECTED_SHARDS),
             "actual_model_forward":"NOT_RUN","native_logit_parity":"NOT_RUN",
             "quantized_weight_binary_identity":"NOT_VERIFIED",
             "promotion_authorized":False,"status":"FULL_BF16_SHA_ONLY"}
    receipt["receipt_sha256"]=digest(receipt)
    return receipt

if __name__=="__main__":
    try:
        print(json.dumps(collect(),sort_keys=True,separators=(",",":")))
    except Exception as exc:
        print(json.dumps({"schema":SCHEMA,"status":"BLOCKED","error":str(exc)[:250],
                          "native_inference_occurred":False}))
        raise SystemExit(2)
