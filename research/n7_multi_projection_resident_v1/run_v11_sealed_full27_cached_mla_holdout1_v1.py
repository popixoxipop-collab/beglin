#!/usr/bin/env python3
"""Research-only original DeepSeek-V2-Lite HOLDOUT-1 / BEGLIN v11 prequant + fused MLA.

Use complete, source-pinned 5,181 n7/group64 packed-weights sealed on D50.
All selected source-defined attention, FFN and MoE expert qNg64 model linears
execute the actual C++ Metal device-resident GEMV. The original independently
source-pinned fused MLA decoder owns *one original qNg64 kv_b weight handle per
source decoder layer*, and preserves ONLY BF16 512 latent + 64 RoPE-k per token
for the autoregressive KV cache. No expanded persistent K/V is admitted.

Research only, not throughput, mixed precision, or production serving.
BF16 source layers still load their structure/source weights per step for
original-source comparability; eligible model linears load packed artifacts
without *any* C quantization in the runtime. Model/dylibs not modified.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["VECLIB_MAXIMUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="2"

import numpy as np
import torch
from tokenizers import Tokenizer

from run_v11_sealed_full27_recompute_v1 import (
    SealedModel, SealedMetalLinear, model_source, INDEX_SHA, gate, sha_path,
)
from research_n7_device_lru_cache_v3_GPUonly_candidate import ResidentN7Cache
from full27_sealed_materializer_v1 import SHARD_SHAS

LOCAL=Path(__file__).resolve().parent
PREVIOUS_SOURCE_SHA="3340c6b86f7085f90b6276e784fa791d188227aeb3d8d301fa3f97f779809cae"
P18=Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-real-activations-20261017")
NATIVE=Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-native-full27-20261018")

PRIVATE_SOURCE_SHAS={
    "native_mla_gpu_resident_decode_adapter_v1.py":
        "b1576b445b401adfe23a5deb63465cbe69aae447346cec8949067c5a9760e714",
    "native_mla_fused_decode_adapter_v1.py":
        "b33d0c9146cba9f322f24c613adebe02b2af956e31a48f4878fc8a810135e1bc",
    "latent_mla_kv_source_cache_v3.py":
        "2d96a2d6b2bbac169a76c91bbea7fed9e36e44e56429e73103adbcf8e2c35650",
    "mla_qng64_resident_weight_handle_v1.cpp":
        "4128d4d469c0312d0927d190f8971da157780bee3ffc966882e72e479b1891ac",
    "mla_qng64_metal_single_fused_v1.cpp":
        "46292b78e8da1c467e52e6ac24e703e5d9c56cdc81fc6ae892976bb4de87ff33",
    "libbeglin_mla_qng64_single_fused_v1.dylib":
        "03e1f21e2d1a840a89206ba21ec8128205e9c88852fdb41251756519c8a4074f",
    "libbeglin_mla_qng64_resident_weight_v1.dylib":
        "bd5f940227482d39d21db1dd14c501c17daa62e3f73a8446903e11b8f640bd3e",
}
C_SOURCE_SHA="6d76b36ef23341e7ac90ecda79541e93133e701e5538ab215e757c334526d786"
CPP_SOURCE_SHA="f10a8b30bfeff95cf0e756d45567ac4aaaa3d2552f5ce62a7746af09000a3201"

U=ctypes.POINTER(ctypes.c_uint8)
F=ctypes.POINTER(ctypes.c_float)


def import_private_fused():
    gate(sha_path(LOCAL/"run_v11_sealed_full27_recompute_v1.py")==PREVIOUS_SOURCE_SHA,
         "original independent sealed prequant model source changed")
    for name,pin in PRIVATE_SOURCE_SHAS.items():
        gate(sha_path(NATIVE/name)==pin,
             "original private BEGLIN fused MLA/latent weight authority drift: "+name)
    sys.path.insert(0,str(NATIVE))
    import native_mla_gpu_resident_decode_adapter_v1 as resident
    return resident


def metric(a,b):
    if isinstance(a,torch.Tensor):
        a=a.detach().float().numpy()
    if isinstance(b,torch.Tensor):
        b=b.detach().float().numpy()
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    gate(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
         "paired finite source tensor geometry")
    delta=a-b
    return {
        "relative_L2":float(np.linalg.norm(delta)/max(np.linalg.norm(b),1e-30)),
        "max_abs":float(np.max(np.abs(delta))),
    }


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir",required=True)
    ap.add_argument("--sealed-root",required=True)
    ap.add_argument("--manifest-sha",required=True)
    ap.add_argument("--native-c-dylib",required=True)
    ap.add_argument("--native-gpu-dylib",required=True)
    ap.add_argument("--receipt-dir",required=True)
    ap.add_argument("--max-layers",type=int,choices=(1,2,27),required=True)
    ap.add_argument("--greedy-tokens",type=int,choices=(0,2),required=True)
    args=ap.parse_args()
    started=time.monotonic()
    torch.set_num_threads(2)
    gate(args.max_layers==27 or args.greedy_tokens==0,
         "prefix-only MLA cache smoke cannot produce real 27L selected tokens")
    checkpoint=Path(args.checkpoint_dir)
    receipt_dir=Path(args.receipt_dir)
    gate(checkpoint.is_dir() and not checkpoint.is_symlink() and
         receipt_dir.is_dir() and not receipt_dir.is_symlink(),
         "original BF16 checkpoint or immutable research receipt root missing")
    for name,pin in [
        ("model.safetensors.index.json",INDEX_SHA),
        ("config.json","f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d"),
        ("modeling_deepseek.py","7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c")]:
        gate(sha_path(checkpoint/name)==pin,
             "original DeepSeek BF16 source identity drift: "+name)
    gate(sha_path(checkpoint/"tokenizer.json")==
         "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669",
         "original tokenizer SHA-256 source authority changed")
    resident=import_private_fused()
    packed,bf16,numeric=model_source()
    for obj in (packed,packed.exp,packed.exp.base,packed.source,
                packed.bf_reference,bf16,bf16.exp,bf16.exp.base,
                bf16.src,bf16.first_ref):
        obj.DATA=checkpoint
    gate(args.manifest_sha=="c82361f7fbd73884b633bb595334805c0b7c0cf2f82ab8de26311ca78a0142b8",
         "externally attested full 5181 n7 manifest digest changed")
    gpu_cpp=Path(args.native_gpu_dylib).parent.parent/"n7_multi_projection_gpu_residency_v11_gpuowned_candidate.cpp"
    gate(sha_path(gpu_cpp)==CPP_SOURCE_SHA,
         "computed GPU-owned n7 Metal operator source identity drift")
    cfile=Path(args.native_c_dylib)
    gate(sha_path(cfile)==C_SOURCE_SHA,
         "original native C group64 quantization/matmul source identity drift")
    c=ctypes.CDLL(str(cfile))
    ref=c.qng64_matmul_f32
    ref.argtypes=(U,F,ctypes.c_int,F,F,F,ctypes.c_int,ctypes.c_int,ctypes.c_int)
    ref.restype=None
    gpu_lib=ctypes.CDLL(args.native_gpu_dylib)
    gpu_lib.beglin_multi_n7_gpu_available_v1.restype=ctypes.c_int
    gate(gpu_lib.beglin_multi_n7_gpu_available_v1()==1,"genuine physical Metal GPU required")

    cfg=packed.exp.SimpleNamespace(**json.loads((checkpoint/"config.json").read_text()))
    cfg._attn_implementation="eager"
    gate(cfg.num_hidden_layers==27 and cfg.hidden_size==2048 and
         cfg.n_routed_experts==64 and cfg.num_experts_per_tok==6,
         "original model architecture changed")
    ns=packed.source.load_sources()
    index=json.loads((checkpoint/"model.safetensors.index.json").read_text())["weight_map"]
    known={r["name"]:r for r in json.loads(
        (P18/"full27_qng64_equal_budget_policy_manifest_v1.json").read_text())["rows"]}
    gate(len(known)==5181,"original frozen uniform7 full model policy disappeared")
    tokenizer=Tokenizer.from_file(str(checkpoint/"tokenizer.json"))
    prompt=bf16.CHOICES[1]
    ids=[cfg.bos_token_id]+tokenizer.encode(prompt,add_special_tokens=False).ids
    gate(len(ids)==7,"frozen holdout-1 source prompt must tokenize to seven positions")
    header={}
    emb,_,_=packed.exp.base.open_sample(
        "model.embed_tokens.weight",index["model.embed_tokens.weight"],
        np.array(ids,dtype=np.int64),header)
    prefix=torch.from_numpy(emb).reshape(1,len(ids),cfg.hidden_size).to(torch.bfloat16)
    gate(prefix.shape==(1,7,2048) and torch.isfinite(prefix).all(),
         "source BF16 original five tokens")

    general=ResidentN7Cache(gpu_lib,bytes_budget=256*1024*1024,max_handles=96)
    fused=None
    output=None
    source=None
    try:
        source=SealedModel(args.sealed_root,args.manifest_sha,known,general,ref)
        fused=resident.ResidentFusedMetalSourceAttention(
            ctypes.CDLL(str(NATIVE/"libbeglin_mla_qng64_single_fused_v1.dylib")),
            SealedMetalLinear,ns)
        caches=[]
        phase_recs=[]
        prediction=[]
        source_embedded_tokens=[]
        evidence_run_id=str(time.time_ns())
        prefix_len=len(ids)
        h=prefix

        def selected_ids(layer,l,hidd,mask,pos,cache):
            if l==0:return []
            norm=layer.input_layernorm(hidd)
            fork=cache.fork()
            with fork.capture():
                attn=layer.self_attn(
                    norm,attention_mask=mask,position_ids=pos,
                    past_key_value=fork,use_cache=True)[0]
            pre=layer.post_attention_layernorm(hidd+attn)
            chosen,weight,aux=layer.mlp.gate(pre)
            gate(aux is None and chosen.shape[-1]==cfg.num_experts_per_tok
                 and torch.isfinite(weight).all(),
                 "source MoEGate selected/top6 weight invalid under fused cache")
            return sorted(set(map(int,chosen.flatten().tolist())))

        def load_experts(layer,l,chosen):
            if l==0:return []
            gate(len(chosen)>0 and len(chosen)<=cfg.n_routed_experts,
                 "actual source router expert count invalid")
            out=[]
            for eid in chosen:
                prefix_=f"model.layers.{l}.mlp.experts.{eid}"
                expert=ns["DeepseekV2MLP"](
                    cfg,intermediate_size=cfg.moe_intermediate_size
                ).to(torch.bfloat16).eval()
                actual,_=packed.source.load_named_parameters(
                    expert,prefix_,index,header)
                gate(actual==3,"original MoE selected expert tensor count")
                for projection in packed.FFN:
                    role=f"{prefix_}.{projection}.weight"
                    mod=source.module(role,expert.get_submodule(projection))
                    setattr(expert,projection,mod)
                    out.append(mod)
                layer.mlp.experts[eid]=expert
            return out

        def do_phase(name,states,count,new_token_position):
            next_h=states
            mask=(numeric.mask_for(prefix_len) if count==0 else
                  torch.zeros((1,1,1,prefix_len+count),dtype=torch.bfloat16))
            pos=(torch.arange(prefix_len,dtype=torch.long).reshape(1,prefix_len)
                 if count==0 else
                 torch.tensor([[prefix_len+count-1]],dtype=torch.long))
            stage_start=time.monotonic()
            used_before=source.loaded
            n_routed=0
            for lid in range(args.max_layers):
                layer_start=time.monotonic()
                if lid==0:
                    layer=packed.first.read_original_decoder(ns,cfg,index,header)
                else:
                    layer,_=bf16.source_layer(ns,cfg,index,header,lid)
                fixed=source.replace_fixed(layer,lid,packed)
                kv_b=layer.self_attn.kv_b_proj
                gate(isinstance(kv_b,SealedMetalLinear) and
                     kv_b.n==7 and kv_b.out==4096 and kv_b.in_features==512,
                     "original source kv_b frozen prequant geometry or type")
                fused.patch_source(layer.self_attn)
                if count==0:
                    cached=resident.FusedSourceMLACache(layer.self_attn,lid,max_tokens=16)
                else:
                    cached=caches[lid]
                    gate(cached.attention is None and cached.size_tokens()==prefix_len+count-1,
                         "previous latent KV cache token count/rebind identity drift")
                    cached.bind(layer.self_attn)
                chosen=selected_ids(layer,lid,next_h,mask,pos,cached)
                experts=load_experts(layer,lid,chosen)
                n_routed+=len(chosen)
                with cached.capture():
                    y=layer(next_h,attention_mask=mask,position_ids=pos,
                            past_key_value=cached,use_cache=True)[0]
                gate(y.dtype==torch.bfloat16 and y.device.type=="cpu" and
                     y.shape==next_h.shape and torch.isfinite(y).all(),
                     "real original source MLA/MoE decoder hidden nonfinite")
                gate(cached.size_tokens()==prefix_len+count and cached.updates==count+1
                     and cached.rehydrate_calls==0 and cached.fused_calls==prefix_len+count
                     and cached.size_bytes()==(prefix_len+count)*(512+64)*2,
                     "actual source fused MLA latent KV append or source return inconsistent")
                mods=[m for m in fixed if m is not kv_b] + experts
                source.record(mods)
                gate(all(m.calls>0 for m in mods),
                     "packed original model n7 Metal projections unexecuted")
                cached.unbind()
                if count==0:caches.append(cached)
                next_h=y.clone()
                print(json.dumps({
                    "phase":"REAL_SOURCE_SEALED_FUSED_MLA",
                    "decode_phase":name,"layer":lid,
                    "selected_MoE_experts":len(chosen),
                    "executed_packed_model_roles":len(mods)+1,
                    "latent_KV_tokens":cached.size_tokens(),
                    "wall_seconds":round(time.monotonic()-layer_start,3)
                }),flush=True)
                del layer,fixed,experts,mods,kv_b,y
                gc.collect()
            stage={
                "source_phase":name,
                "source_layer_count":args.max_layers,
                "new_context_position":new_token_position,
                "latent_KV_size_bytes":sum(c.size_bytes() for c in caches),
                "source_full_model_C_gpu_oracle_bound_module_instances":
                    source.cpu_oracle_checked,
                "source_gpu_packed_module_instances_this_stage":
                    source.loaded-used_before,
                "MoE_experts_used_sum":n_routed,
                "runtime_seconds_including_BF16_source_loading":
                    round(time.monotonic()-stage_start,3),
            }
            return next_h,stage

        # Pre-fill (7 original holdout tokens), then two model-selected cached decodes.
        h,rec=do_phase("PREFILL",prefix,0,0)
        if args.max_layers==27:
            logits,_=packed.finish_logits(index,header,h[:,-1:,:],cfg)
            gate(logits.shape==(1,102400) and np.isfinite(logits).all(),
                 "full vocab source lm_head prefill failed")
            first=int(np.argmax(logits))
            baseline_path=P18/"qng64_uniform7_full27_prompt1_logits_v1.npy"
            gate(sha_path(baseline_path)==
                 "75e36edb9f696af89bf47c7720f2f124ff6a14c47eb3efaef5f2da2c95326f8c",
                 "frozen independent holdout1 prior qNg64 n7 logits SHA mismatch")
            old=np.load(str(baseline_path),allow_pickle=False)
            gate(old.shape==(7,102400),"historical independent uniform7 holdout1 logits missing")
            delta=metric(logits[0],old[-1])
            gate(delta["relative_L2"]<0.025 and first==int(np.argmax(old[-1])),
                 "original full27 fused MLA prefill vs uniform7 source logits mismatch")
            rec["logits_compare_original_uniform7"]=delta
            rec["prediction"]=first
            rec["logits_SHA256"]=hashlib.sha256(logits.tobytes()).hexdigest()
            logfile=receipt_dir/(
                "BEGLIN_v11_holdout1_cached_logits_"+evidence_run_id+"_step0.npy")
            with logfile.open("xb") as fd:
                np.save(fd,logits,allow_pickle=False)
                fd.flush()
                os.fsync(fd.fileno())
            rec["private_logits_basename"]=logfile.name
            rec["private_logits_npy_sha256"]=sha_path(logfile)
            prediction.append(first)
        phase_recs.append(rec)
        print(json.dumps({"phase":"PREFILL_RESULT","max_layers":args.max_layers,
                          "selected_next":prediction[-1] if prediction else None}),flush=True)
        if args.greedy_tokens==2:
            gate(args.max_layers==27 and len(prediction)==1 and 0<=prediction[0]<102400,
                 "only full model's first argmax may be embedded as real token")
            for index_dec in (1,2):
                token=prediction[-1]
                source_embedded_tokens.append(token)
                emb_dec,_,_=packed.exp.base.open_sample(
                    "model.embed_tokens.weight",index["model.embed_tokens.weight"],
                    np.array([token],dtype=np.int64),header)
                x=torch.from_numpy(emb_dec).reshape(1,1,cfg.hidden_size).to(torch.bfloat16)
                h,record=do_phase("DECODE_"+str(index_dec),x,index_dec,
                                  new_token_position=prefix_len+index_dec-1)
                logits,_=packed.finish_logits(index,header,h,cfg)
                gate(logits.shape==(1,102400) and np.isfinite(logits).all(),
                     "source full27 cached greedy vocabulary logits unavailable")
                predicted=int(np.argmax(logits))
                record["prediction"]=predicted
                record["logits_SHA256"]=hashlib.sha256(logits.tobytes()).hexdigest()
                logfile=receipt_dir/(
                    "BEGLIN_v11_holdout1_cached_logits_"+evidence_run_id+
                    "_step"+str(index_dec)+".npy")
                with logfile.open("xb") as fd:
                    np.save(fd,logits,allow_pickle=False)
                    fd.flush()
                    os.fsync(fd.fileno())
                record["private_logits_basename"]=logfile.name
                record["private_logits_npy_sha256"]=sha_path(logfile)
                phase_recs.append(record)
                prediction.append(predicted)
                print(json.dumps({"phase":"ACTUAL_SOURCE_GREEDY_CACHED_MODEL",
                                  "embedded_input":token,"predicted_next":predicted,
                                  "decode":index_dec}),flush=True)
            gate(len(source_embedded_tokens)==2 and len(prediction)==3
                 and source_embedded_tokens==prediction[:2]
                 and all(type(v) is int and 0<=v<102400 for v in prediction),
                 "true source model 27-layer cached greedy continuation changed")

        # Record GPU source/weight and latent-KV statuses BEFORE closing.
        general_stats=general.get_state()
        resident_stats=fused.describe()
        gate(resident_stats["actual_device_Metal_layer_count"]==args.max_layers
             and resident_stats["one_time_weight_register_calls"]==args.max_layers
             and resident_stats["resident_GPU_forward_calls"] ==
                 resident_stats["physical_fused_GPU_source_query_calls"],
             "source n7 resident kv_b registrations or real MLA kernel dispatch drift")
        gate(all(c.attention is None and c.rehydrate_calls==0 and
                 c.size_tokens()==prefix_len+args.greedy_tokens
                 for c in caches),
             "resident model layer reference leaked or K/V expanded")
        total_BF16_MLA_KV=sum(c.size_bytes() for c in caches)
        gate(total_BF16_MLA_KV==(prefix_len+args.greedy_tokens)*args.max_layers*(512+64)*2,
             "true GPU-fused source MLA latent-KV byte accounting mismatch")
        output={
            "schema":"beglin.v11.original-deepseek-full27-sealed-n7-fused-MLA-cached/1",
            "status":"PASS_REAL_FULL27_HOLDOUT1_TWO_GREEDY_SEALED_N7_FUSED_MLA_KV" if
                 args.max_layers==27 and args.greedy_tokens==2 else
                 "PASS_REAL_ORIGINAL_SEALED_N7_FUSED_MLA_PREFIX",
            "checkpoint_index_sha256":INDEX_SHA,
            "full_sealed_prequant_manifest_sha256":args.manifest_sha,
            "original_source_policy_sha256":
                 "39b98921c30a992f93ea896283336a6b5fe1810c5737443132f2410b474e9612",
            "source_prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
            "model_layers":args.max_layers,
            "source_prompt_id":1,
            "source_prompt_token_count":prefix_len,
            "greedy_input_tokens":source_embedded_tokens,
            "predictions":prediction,
            "source_model_true_MoE_gate_experts":True,
            "device_resident_fused_kvb_weight_layers":
                resident_stats["actual_device_Metal_layer_count"],
            "fused_mla_GPU_attention_query_calls":resident_stats["physical_fused_GPU_source_query_calls"],
            "fused_mla_kvb_resident_GPU_packed_bytes":
                resident_stats["resident_device_bitplanes_F32scales_total_bytes"],
            "per_layer_fused_kvb_weight_registered_once":True,
            "GPU_normal_qng64_model_forward_calls":general_stats["actual_GPU_forward_calls"],
            "GPU_normal_qng64_reuse_hits":general_stats["GPU_reuse_hits"],
            "GPU_normal_qng64_evictions":general_stats["GPU_LRU_evictions"],
            "CPU_original_qNg64_requant_during_model_generation":0,
            "source_new_model_role_compressed_file_loads":source.loaded,
            "source_gpu_C_oracle_checked_role_modules":source.cpu_oracle_checked,
            "FP32_C_vs_Metal_max_relative_L2_on_BF16_ambiguity":
                source.max_F32_vs_C_relative_L2,
            "BF16_neighbor_ULP_transition_count":source.BF16_rounding_one_ULP_flips,
            "persistent_KV_source_storage":"BF16 512D normalized latent + 64D rotated RoPE key",
            "persistent_MLA_KV_total_bytes":total_BF16_MLA_KV,
            "expanded_MLA_KV_persistent":False,
            "full_model_GPU_exclusively_resident":False,
            "BF16_original_source_layer_and_reference_weight_reads_still_present":True,
            "genuine_serving_TTFT_TPOT_certified":False,
            "original_source_modified":False,
            "production_promotion_authorized":False,
            "peak_process_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "wall_seconds":round(time.monotonic()-started,3),
            "stage_receipts":phase_recs,
        }
    finally:
        general.close()
        if fused is not None:fused.close()
    gate(gpu_lib.beglin_multi_n7_active_count_v1()==0 and
         gpu_lib.beglin_multi_n7_payload_bytes_v1()==0 and
         fused.closed and len(fused.handles)==0,
         "original GPU-computed n7 / fused MLA native weight cleanup failed")
    output["all_GPU_weight_handles_fully_released"]=True
    output["all_MLA_KV_layers_detached_from_original_attention"]=True
    output["completed_without_original_checkpoint_changes"]=True
    receipt=(json.dumps(output,sort_keys=True,indent=2)+"\n").encode()
    receipt_path=receipt_dir/(
        "BEGLIN_v11_holdout1_real_27L_sealed_fused_mla_cached_"+
        str(args.max_layers)+"L_"+str(args.greedy_tokens)+"tokens_"+
        str(time.time_ns())+".json")
    with receipt_path.open("xb") as f:
        f.write(receipt)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({
        "status":output["status"],"greedy_input_tokens":source_embedded_tokens,
        "predictions":prediction,"layers":args.max_layers,
        "model_source_native_Metal_GEMVs":general_stats["actual_GPU_forward_calls"],
        "fused_MLA_query_calls":resident_stats["physical_fused_GPU_source_query_calls"],
        "BF16_MLA_KV_bytes":total_BF16_MLA_KV,
        "source_qNg64_C_requantizations":0,
        "all_GPU_handles_released":True,
        "receipt_basename":receipt_path.name,
        "receipt_SHA256":hashlib.sha256(receipt).hexdigest(),
        "wall_seconds":output["wall_seconds"],
        "production_promotion_allowed":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
