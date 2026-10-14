#!/usr/bin/env python3
"""XOX exact original BF16 safetensors *slice* oracle, never a full LLM teacher.

Read-only model shard access, 2 tensors x 16 rows x 256 features and float32
CPU tile-matvec. Explicit output says NOT full-model BF16 logits/accuracy.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,resource,sys
import torch
from safetensors import safe_open

ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p11-inflight-20261014-1816")
MODEL=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
STAGE=pathlib.Path("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")
STAGE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
KEYS=("model.layers.3.mlp.shared_experts.up_proj.weight",
      "model.layers.26.mlp.shared_experts.down_proj.weight")
ROWS=16;COLS=256
MAX_BF16_TILE_BYTES=ROWS*COLS*2
MAX_RSS=2*1024**3
def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def sanity():
    if not ROOT.is_dir() or ROOT.parent.name!=".tailnet-tmp" or not ROOT.name.startswith("hcp-p11-inflight-"):
        raise ValueError("outside exact P11 scratch")
    if sha(STAGE)!=STAGE_SHA or sha(ROUTE)!=ROUTE_SHA or sha(MODEL/"model.safetensors.index.json")!=INDEX_SHA:
        raise ValueError("model or stage SHA differs from originally certified source")
    index=json.loads((MODEL/"model.safetensors.index.json").read_text())
    for k in KEYS:
        if k not in index["weight_map"]:raise ValueError("target BF16 weight missing")
    return index
def run():
    index=sanity()
    torch.set_num_threads(1)
    lines=[]
    for key in KEYS:
        name=index["weight_map"][key]
        p=MODEL/name
        if not p.is_file() or p.is_symlink() or p.parent!=MODEL:
            raise ValueError("bad precise BF16 read-only shard")
        with safe_open(str(p),framework="pt",device="cpu") as handle:
            if key not in handle.keys():raise ValueError("model index/shard key mismatch")
            slice_=handle.get_slice(key)
            shape=slice_.get_shape()
            dtype=slice_.get_dtype()
            if len(shape)!=2 or min(shape)<COLS or dtype!="BF16":
                raise ValueError("unexpected source matrix shape or BF16 dtype")
            # Access only a small tensor region; DO NOT get_tensor(key) or .to(cuda/mps).
            tile=slice_[0:ROWS,0:COLS]
            if tile.dtype!=torch.bfloat16 or tile.shape!=torch.Size((ROWS,COLS)):
                raise ValueError("BF16 slice not precisely bounded")
            if tile.numel()*tile.element_size()!=MAX_BF16_TILE_BYTES:
                raise ValueError("BF16 slice memory budget exceeded")
            raw=tile.contiguous().view(torch.int16).numpy().tobytes()
            if len(raw)!=MAX_BF16_TILE_BYTES:
                raise ValueError("raw tile byte budget exceeded")
            wf=tile.to(dtype=torch.float32)
            x=torch.linspace(-1.0,1.0,COLS,dtype=torch.float32)
            z=torch.mv(wf,x)
            if not bool(torch.isfinite(wf).all() and torch.isfinite(z).all()):
                raise ValueError("nonfinite original BF16 slice")
            record={
             "tensor_key":key,"shard":name,"original_shard_bytes":p.stat().st_size,
             "source_weight_shape":shape,"source_dtype":dtype,
             "tile_shape":[ROWS,COLS],"tile_dtype":"BF16",
             "slice_bytes_read_for_values":MAX_BF16_TILE_BYTES,
             "tile_weight_sha256":hashlib.sha256(raw).hexdigest(),
             "test_vector":"deterministic linspace(-1,1,256) in FP32",
             "tile_matvec_result_sha256":hashlib.sha256(z.numpy().tobytes()).hexdigest(),
             "tile_matvec_mean":float(z.mean().item()),
             "tile_matvec_max_abs":float(z.abs().max().item()),
             "full_weight_tensor_loaded":False,
             "full_model_loaded":False}
            lines.append(record)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss>MAX_RSS:raise RuntimeError("P11 BF16 tile probe exceeded 2GiB process RSS gate")
    if sanity()["weight_map"]!=index["weight_map"]:
        raise ValueError("BF16 model index changed during read")
    report={
       "schema":"beglin-hcp-p11-readonly-bf16-weight-slice-oracle-v1",
       "status":"ORIGINAL_BF16_TILE_PROJECTION_ONLY",
       "source_index_sha256":INDEX_SHA,
       "samples":lines,
       "total_BF16_tile_value_bytes":len(lines)*MAX_BF16_TILE_BYTES,
       "sample_count":len(lines),"RSS_peak_bytes":rss,
       "device":"cpu",
       "whole_BF16_model_forward_executed":False,
       "BF16_reference_logits_measured":False,
       "quantized_A_or_B_noninferiority_proven":False,
       "quality_certified":False,
       "production_modified":False,"automatic_live_promotion":False}
    target=ROOT/"evidence/bf16_source_tiles.json"
    with target.open("x") as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    print("P11_BF16_SLICE_PROBE",json.dumps({"sha256":sha(target),
     "tile_count":len(lines),"total_source_value_bytes":len(lines)*MAX_BF16_TILE_BYTES,
     "peak_rss_bytes":rss,"full_teacher_executed":False},sort_keys=True),flush=True)
if __name__=="__main__":run()
