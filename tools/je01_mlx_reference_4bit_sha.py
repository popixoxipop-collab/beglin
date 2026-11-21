#!/usr/bin/env python3
"""JE01 READ-ONLY pretrained MLX-Chat reference SHA; NOT BEGLIN source parity.

Uses the existing HF cached 4bit-mlx reference, distinct from the frozen
BF16 DeepSeek source and XOX BEGLIN quantized runtime. No downloads, writes,
model load, or use of JE01 quality holdouts. Snapshot symlinks are permitted
only when resolving WITHIN the same models--... cache root.
"""
from pathlib import Path
import hashlib
import json
import os

SCHEMA="beglin-je01-external-mlx-4bit-reference-source-sha/1"
ROOT=Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--DeepSeek-V2-Lite-Chat-4bit-mlx")
REV="a33a8ad64477b9ca7a65b6dc1d8c06c997f87216"
EXPECT={
 "config.json":"67134636123453e474a5b72d611be6130f59f8fc7478aac7927cbe6f9cd44d71",
 "model.safetensors.index.json":"1f840914b851127054b81b60ee1ad1a7412a218c170c7567770426289f7fd29c",
 "tokenizer.json":"c6276ee9e3cfa46033f54dc46db3f6dba0c6a1f6a2e547543124f33b44873db4",
}
SHARD_LENGTHS={"model-00001-of-00002.safetensors":5316545126,
               "model-00002-of-00002.safetensors":3523543576}

def stable(obj):
 return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def verified(path, root=ROOT):
 real=path.resolve(strict=True)
 if not real.is_file() or not real.is_relative_to(root.resolve(strict=True)):
  raise ValueError("JE01_CACHE_ESCAPE_OR_NON_FILE")
 return real

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for chunk in iter(lambda:stream.read(1<<20),b""):
   h.update(chunk)
 return h.hexdigest()

def audit():
 snapshot=ROOT/"snapshots"/REV
 meta={}
 for name,expect in EXPECT.items():
  p=verified(snapshot/name)
  measured=sha(p)
  if measured!=expect:raise ValueError("JE01_REF_METADATA_DRIFT:"+name)
  meta[name]={"sha256":measured,"bytes":p.stat().st_size}
 cfg=json.loads((snapshot/"config.json").read_text())
 index=json.loads((snapshot/"model.safetensors.index.json").read_text())
 if (cfg.get("model_type")!="deepseek_v2" or cfg.get("n_routed_experts")!=64 or
     cfg.get("num_experts_per_tok")!=6 or cfg.get("num_hidden_layers")!=27 or
     cfg.get("quantization")!={"bits":4,"group_size":64} or
     index.get("metadata",{}).get("total_size")!=8839977984 or
     set(index.get("weight_map",{}).values())!=set(SHARD_LENGTHS)):
  raise ValueError("JE01_REF_CONFIG_OR_INDEX_DRIFT")
 shards=[]
 for name,bytes_expected in SHARD_LENGTHS.items():
  file=verified(snapshot/name)
  if file.stat().st_size!=bytes_expected:
   raise ValueError("JE01_REF_SHARD_LENGTH_DRIFT")
  shards.append({"name":name,"bytes":bytes_expected,"sha256":sha(file)})
 record={
  "schema":SCHEMA,"status":"REFERENCE_SOURCE_SHA_ONLY",
  "reference_family":"DeepSeek-V2-Lite-Chat-4bit-mlx",
  "source_revision":REV,"model_type":"deepseek_v2",
  "quantization":{"bits":4,"group_size":64},
  "metadata":meta,"shards":shards,
  "model_ref_identity_sha256":stable({"metadata":meta,"shards":shards,"revision":REV}),
  "source_bf16_checkpoint_sha256":"DIFFERENT_MODEL_NOT_EQUIVALENT",
  "same_as_beglin_quantized_runtime":False,
  "original_je01_tokenizer_sha256":"41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669",
  "reference_tokenizer_sha256":EXPECT["tokenizer.json"],
  "pretrained_forward":"NOT_RUN",
  "full_model_parity":"NOT_MEASURED",
  "production_promotion_allowed":False,
  "weights_modified":False
 }
 record["receipt_sha256"]=stable(record)
 return record

if __name__=="__main__":
 try:print(json.dumps(audit(),sort_keys=True,separators=(",",":")))
 except Exception as exc:
  print(json.dumps({"schema":SCHEMA,"status":"BLOCKED","error":str(exc)[:250]}))
  raise SystemExit(2)
