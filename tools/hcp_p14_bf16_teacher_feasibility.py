#!/usr/bin/env python3
"""P14 original BF16 full-model streaming feasibility, header-only, fail closed.

Inspect safetensors headers and configuration WITHOUT mapping/loading tensor
values. This is an engineering gate and a potential per-layer streaming plan,
not whole model BF16 logits or task-quality evidence.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,resource,struct,importlib.util
from collections import defaultdict

ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p11-inflight-p14-multiworker-20261014")
MODEL=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
ORIGINAL_BYTES=31413626576
MAX_RSS=512*(1024**2)
DTYPE_WIDTH={"BF16":2,"F16":2,"F32":4,"I64":8,"I32":4,"U8":1,"I8":1}
REGEX=re.compile(r"^model\.layers\.([0-9]+)\.")

def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()

def main():
    if ROOT.name!="hcp-p11-inflight-p14-multiworker-20261014" or not ROOT.is_dir():
        raise RuntimeError("P14 run-owned root missing")
    index=MODEL/"model.safetensors.index.json"
    if sha(index)!=INDEX_SHA:
        raise RuntimeError("P14 original source BF16 index differs")
    obj=json.loads(index.read_text())
    mapping=obj["weight_map"]
    shards=sorted(set(mapping.values()))
    if len(shards)!=4:
        raise RuntimeError("P14 original BF16 shard count differs from certified four")
    config=json.loads((MODEL/"config.json").read_text())
    count=int(config["num_hidden_layers"])
    assert count==27 and config["model_type"]=="deepseek_v2"
    weight_bytes=0
    discovered=set()
    layer_bytes=defaultdict(int)
    layer_nparams=defaultdict(int)
    tensor_bytes=defaultdict(int)
    shard_info={}
    all_metadata={}
    for name in shards:
        p=MODEL/name
        if p.is_symlink() or not p.is_file() or p.parent!=MODEL:
            raise ValueError("untrusted or symlinked original safetensors shard")
        with p.open("rb") as f:
            header_length=struct.unpack("<Q",f.read(8))[0]
            if not 0<header_length<=8*1024*1024:
                raise ValueError("original safetensors header size invalid")
            header=json.loads(f.read(header_length))
        data={k:v for k,v in header.items() if k!="__metadata__"}
        start_end=[]
        shard_total=0
        for key,meta in data.items():
            if key in discovered or mapping.get(key)!=name:
                raise ValueError("safetensors duplicated or index mismatched tensor")
            discovered.add(key)
            shape=meta["shape"];dtype=meta["dtype"]
            if not shape or dtype not in DTYPE_WIDTH or any(type(x)!=int or x<=0 for x in shape):
                raise ValueError("unrecognized BF16 tensor metadata")
            n=math.prod(shape)
            data_range=meta["data_offsets"]
            if not isinstance(data_range,list) or len(data_range)!=2:
                raise ValueError("original tensor offset not present")
            a,b=data_range
            bytes_expected=n*DTYPE_WIDTH[dtype]
            if (a<0 or b<a or b-a!=bytes_expected or
                b+8+header_length>p.stat().st_size):
                raise ValueError("model tensor byte offsets inconsistent with original BF16 file")
            start_end.append((a,b))
            shard_total+=bytes_expected
            weight_bytes+=bytes_expected
            match=REGEX.match(key)
            if match:
                layer=int(match.group(1))
                if layer>=count:raise ValueError("unexpected original model layer")
                layer_bytes[layer]+=bytes_expected
                layer_nparams[layer]+=n
            else:
                tensor_bytes[key]+=bytes_expected
            if bytes_expected>0:
                all_metadata[key]={"shape":shape,"dtype":dtype,"source":name,"bytes":bytes_expected}
        intervals=sorted(start_end)
        if any(x[0]<last[1] for last,x in zip(intervals,intervals[1:])):
            raise ValueError("overlapping actual shard data offsets")
        shard_info[name]={"model_file_bytes":p.stat().st_size,
             "tensor_count":len(data),"tensor_data_bytes":shard_total,
             "header_bytes":8+header_length}
    if discovered!=set(mapping):
        raise ValueError("original BF16 model index/shards incomplete")
    if sorted(layer_bytes)!=list(range(count)):
        raise ValueError("model missing expected contiguous 27 transformer layers")
    if sum(x["model_file_bytes"] for x in shard_info.values())!=ORIGINAL_BYTES:
        raise ValueError("original DeepSeek shards changed raw total file bytes")
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak>=MAX_RSS:raise MemoryError("P14 BF16 metadata-only teacher feasibility consumed too much RSS")
    try:available_transformers=bool(importlib.util.find_spec("transformers.models.deepseek_v2"))
    except Exception:available_transformers=False
    try:available_accelerate=bool(importlib.util.find_spec("accelerate"))
    except Exception:available_accelerate=False
    report={"schema":"beglin-hcp-p14-fully-original-BF16-header-only-resource-admission-v1",
        "status":"FULL_BF16_TEACHER_ENGINE_NOT_YET_STREAMING_CERTIFIED",
        "model_index_sha256":INDEX_SHA,"model_type":config["model_type"],
        "model_decoder_layer_count":count,
        "original_shard_total_file_bytes":ORIGINAL_BYTES,
        "original_weight_data_total_bytes":weight_bytes,
        "original_per_shard_headers_checked":shard_info,
        "original_all_tensors_indexed":len(discovered),
        "largest_layer_weight_bytes":max(layer_bytes.values()),
        "largest_layer_weight_gib":round(max(layer_bytes.values())/1024**3,4),
        "layer_weight_bytes_by_index":{str(k):layer_bytes[k] for k in sorted(layer_bytes)},
        "largest_single_tensor_bytes":max(v["bytes"] for v in all_metadata.values()),
        "non_layer_major_weight_count":len(tensor_bytes),
        "non_layer_top_weights_bytes":dict(sorted(tensor_bytes.items(),key=lambda x:-x[1])[:8]),
        "any_layer_subcomponent_reads_performed":False,
        "whole_BF16_model_loaded":False,"full_BF16_teacher_forward_executed":False,
        "exact_whole_BF16_teacher_logits_produced":False,
        "transformers_has_native_deepseek_v2_module":available_transformers,
        "accelerate_installed":available_accelerate,
        "a_streaming_teacher_might_fit_but_not_proven":True,
        "no_unverified_teacher_results_promoted":True,
        "RSS_peak_bytes":peak,"RSS_limit_bytes":MAX_RSS,
        "original_operational_checkpoint_not_modified":True,
        "production_touched":False,"automatic_live_promotion":False}
    p=ROOT/"evidence/BF16_TEACHER_FEASIBILITY.json"
    with p.open("x") as f:json.dump(report,f,indent=2,sort_keys=True);f.write("\n")
    print("P14_ORIGINAL_BF16_HEADER_ONLY_FEASIBILITY_PASS",json.dumps({
        "tensor_metadata_count":len(discovered),"layers":count,
        "max_layer_weight_gib":report["largest_layer_weight_gib"],
        "source_bytes":ORIGINAL_BYTES,"RSS_MB":round(peak/1024**2,2),
        "full_model_loaded":False,
        "streaming_full_logits_not_yet_implemented":True,
        "receipt_sha256":sha(p)},sort_keys=True),flush=True)
if __name__=="__main__":main()
