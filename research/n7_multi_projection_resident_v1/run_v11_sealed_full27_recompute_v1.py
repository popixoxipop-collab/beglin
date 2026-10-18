#!/usr/bin/env python3
"""Research-only real DeepSeek-V2-Lite 27-layer sealed n7 Metal autoregression.

Full model is evaluated through the ORIGINAL source-defined DeepSeek-V2
attention, MoE gate, routed experts, residuals, RMSNorm and lm_head.
All 5,181 eligible weights are admitted from the hash-sealed, previously
C-quantized qNg64 n7 source roles; only actually routed experts are opened.
Native C++/MLX physical Metal computes *every packed role* on demand.

Unlike the earlier fused MLA cache path, this test RECOMPUTES the full context
for each predicted token. It is a numerical integration gate, NOT a KV cache
or TTFT/TPOT benchmark. Original BF16 source layers and experts are still
loaded for source-reference structures, so elimination of BF16 file reads
is not claimed. No original source is edited or executed via dynamic patches.
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
from torch import nn
from tokenizers import Tokenizer

from prequant_full27_execution_adapter_v1 import SealedPrequantFull27
from full27_sealed_materializer_v1 import (
    INDEX_SHA, SHARD_SHAS, FULL_ROLES, FULL_Q_BYTES, gate, sha_path,
)
from research_n7_device_lru_cache_v3_GPUonly_candidate import ResidentN7Cache

BASE=Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-real-activations-20261017")
# SHA-pinned research modules, not production serving files.
SOURCE_SHA={
 "stream_deepseek_full27_qng64_packed_metal_v1.py":"1aaa1f556596c2c24d74bf54cf7d452adaa7e8e75c26c025dc76cbbb5dc247c9",
 "stream_deepseek_27layer_bf16_reference_v1.py":"414f549727393c7ec4b9eaf6c2c5746dd28c1816a61e254a113e8c068995d550",
 "test_full_first_decoder_quantized_propagation.py":"8c07f14c4d67c59b5016fe83fb828409617ff0fa3e471f9d241fddc0c0ec9335",
 "research_real_deepseek_two_layer_moe_bf16_v1.py":"3b6cbd9f0c8c99fe16a44c09f111cc1b0e4482994e629a043697bfce4f3cbef6",
 "full27_qng64_equal_budget_policy_manifest_v1.json":"39b98921c30a992f93ea896283336a6b5fe1810c5737443132f2410b474e9612",
}
GPU_CPP_SHA="f10a8b30bfeff95cf0e756d45567ac4aaaa3d2552f5ce62a7746af09000a3201"
GPU_LIB_SHA=None  # Real compiled dylib hashes are recorded in receipts.

U=ctypes.POINTER(ctypes.c_uint8)
F=ctypes.POINTER(ctypes.c_float)


def model_source():
    for name,sha in SOURCE_SHA.items():
        gate(sha_path(BASE/name)==sha,"original numerical research source digest drift: "+name)
    sys.path.insert(0,str(BASE))
    import stream_deepseek_full27_qng64_packed_metal_v1 as packed
    import stream_deepseek_27layer_bf16_reference_v1 as bf16
    import test_full_first_decoder_quantized_propagation as numeric
    return packed,bf16,numeric


class SealedMetalLinear(nn.Module):
    def __init__(self,role,source,cache,reference):
        super().__init__()
        self.role=role
        self.source=source
        self.cache=cache
        self.reference=reference
        self.n=7
        self.out=source.out
        self.in_features=source.in_features
        self.serialized_bytes=source.serialized_bytes
        self.calls=0
        self.native_C_first_call_parity=False
        self.max_C_relative_L2=0.0
        self.extra_F32_oracle_GPU_dispatches=0
        self.bf16_single_ULP_flips=0
        self.max_F32_relative_L2=0.0

    def forward(self,x):
        y=self.cache.call(self.role,self.source,x)
        if not self.native_C_first_call_parity:
            X=np.ascontiguousarray(
                x.detach().float().numpy().reshape(-1,self.in_features),
                dtype=np.float32)
            ref=np.empty((X.shape[0],self.out),np.float32)
            self.reference(
                self.source.packed.ctypes.data_as(U),
                self.source.scales.ctypes.data_as(F),
                7,X.ctypes.data_as(F),None,ref.ctypes.data_as(F),
                self.out,self.in_features,int(X.shape[0]))
            answer=torch.from_numpy(ref.reshape(*x.shape[:-1],self.out)).to(torch.bfloat16)
            delta=torch.linalg.vector_norm((y.float()-answer.float()).reshape(-1))
            norm=torch.linalg.vector_norm(answer.float().reshape(-1)).clamp_min(1e-30)
            rel=float(delta/norm)
            if not(np.isfinite(rel) and rel<3e-4):
                # Strict two-stage numerical gate, NOT a free BF16 tolerance:
                # a BF16 nearest-rounding boundary can flip by one ULP even
                # when the native C and GPU F32 outputs agree to 1e-7.
                # Independently re-read the *same* pinned opaque GPU handle
                # and compare its F32 outputs with the original native C.
                entry=self.cache.lru.get(self.role)
                gate(entry is not None,"GPU source role missing before FP32 diagnosis")
                raw_gpu=np.empty_like(ref)
                ok=self.cache.forward_fn(
                    int(entry["handle"]),X.ctypes.data_as(F),
                    int(X.shape[0]),raw_gpu.ctypes.data_as(F))
                self.extra_F32_oracle_GPU_dispatches+=1
                gate(ok==1 and np.isfinite(raw_gpu).all(),
                     "native Metal raw F32 diagnostic dispatch failed")
                c64=ref.astype(np.float64)
                gpu64=raw_gpu.astype(np.float64)
                raw_rel=float(np.linalg.norm(c64-gpu64)/
                    max(np.linalg.norm(c64),1e-30))
                max_abs=float(np.max(np.abs(c64-gpu64)))
                self.max_F32_relative_L2=raw_rel
                gpu_bf16=torch.from_numpy(
                    raw_gpu.reshape(*x.shape[:-1],self.out)).to(torch.bfloat16)
                next_plus=torch.nextafter(
                    answer,torch.full_like(answer,float("inf")))
                next_minus=torch.nextafter(
                    answer,torch.full_like(answer,float("-inf")))
                within_one_ulp=(y==answer)|(y==next_plus)|(y==next_minus)
                mismatch=int(torch.count_nonzero(y!=answer))
                gate(raw_rel<1e-5 and max_abs<1e-4 and
                     rel<0.005 and bool(within_one_ulp.all()) and
                     torch.equal(y,gpu_bf16),
                     "GPU vs C source arithmetic/ULP parity drift role="+
                     self.role+" BF16_relL2="+str(rel)+
                     " FP32_relL2="+str(raw_rel)+
                     " FP32_max_abs="+str(max_abs)+
                     " BF16_flips="+str(mismatch))
                self.bf16_single_ULP_flips=mismatch

            self.native_C_first_call_parity=True
            self.max_C_relative_L2=rel
        self.calls+=1
        return y


class SealedModel:
    def __init__(self,artifact,manifest_sha,known,gpu,original_c):
        # Manifest/source/exporter authority all checked by shared reader.
        from prequant_n7_sealed_v1 import PackReader
        from prequant_full27_execution_adapter_v1 import (
            SOURCE_RUN_AUTHORITY_SHA256,SOURCE_EXPORTER_SHA256,SOURCE_NATIVE_C_SHA256,
        )
        root=Path(artifact)
        authority=root/"RUN_AUTHORITY.json"
        gate(authority.is_file() and not authority.is_symlink() and
             hashlib.sha256(authority.read_bytes()).hexdigest() ==
                 SOURCE_RUN_AUTHORITY_SHA256,
             "source and exporter run authority mismatch")
        a=json.loads(authority.read_bytes())
        gate(a.get("exporter_source_sha256")==SOURCE_EXPORTER_SHA256 and
             a.get("native_quantizer_sha256")==SOURCE_NATIVE_C_SHA256 and
             a.get("source_shard_sha256")==SHARD_SHAS,
             "original packed materializer/source model not pinned")
        self.reader=PackReader(root,manifest_sha,INDEX_SHA)
        gate(len(self.reader.roles)==FULL_ROLES and
             self.reader.total_packed_bytes==FULL_Q_BYTES and
             set(self.reader.roles)==set(known),
             "source/model policy not complete or output manifest swapped")
        self.gpu=gpu
        self.original_c=original_c
        self.loaded=0
        self.packed_bytes=0
        self.roles=set()
        self.cpu_oracle_checked=0
        self.max_cpu_gpu_relative_L2=0.0
        self.raw_F32_GPU_rechecks=0
        self.max_F32_vs_C_relative_L2=0.0
        self.BF16_rounding_one_ULP_flips=0

    def module(self,role,original):
        gate(role in self.reader.roles and
             tuple(original.weight.shape)==(
                 self.reader.roles[role]["out"],
                 self.reader.roles[role]["in_features"]),
             "real BF16 model source role identity/dims mismatch")
        src=self.reader.load(role)
        m=SealedMetalLinear(role,src,self.gpu,self.original_c)
        self.loaded+=1
        self.roles.add(role)
        self.packed_bytes+=src.serialized_bytes
        return m

    def replace_fixed(self,layer,idx,packed):
        roles=packed.ATTENTION+(
            tuple("mlp."+p for p in packed.FFN) if idx==0
            else tuple("mlp.shared_experts."+p for p in packed.FFN))
        gate(len(roles)==7,"frozen decoder layer fixed roles")
        mods=[]
        for path in roles:
            parent,field=path.rsplit(".",1)
            orig=layer.get_submodule(path)
            role=f"model.layers.{idx}.{path}.weight"
            m=self.module(role,orig)
            setattr(layer.get_submodule(parent),field,m)
            mods.append(m)
        return mods

    def load_routed(self,layer,idx,hidden,mask,pos,packed,ns,cfg,index,headers):
        if idx==0:return [],[]
        norm=layer.input_layernorm(hidden)
        with torch.no_grad():
            a=layer.self_attn(norm,attention_mask=mask,
                              position_ids=pos,use_cache=False)[0]
            pre=layer.post_attention_layernorm(hidden+a)
            selected,weights,aux=layer.mlp.gate(pre)
        gate(aux is None and selected.shape==(hidden.shape[1],6),
             "actual source MoE top6 router contract")
        expert_ids=sorted(set(int(z) for z in selected.reshape(-1).tolist()))
        gate(0<len(expert_ids)<=64,"source-defined router selected expert bound")
        used=[]
        for eid in expert_ids:
            base=f"model.layers.{idx}.mlp.experts.{eid}"
            expert=ns["DeepseekV2MLP"](
                cfg,intermediate_size=cfg.moe_intermediate_size).to(torch.bfloat16).eval()
            original_n,_=packed.source.load_named_parameters(
                expert,base,index,headers)
            gate(original_n==3,"source expert has 3 eligible FFN projections")
            for role in packed.FFN:
                key=f"{base}.{role}.weight"
                orig=expert.get_submodule(role)
                mod=self.module(key,orig)
                setattr(expert,role,mod)
                used.append(mod)
            layer.mlp.experts[eid]=expert
        return expert_ids,used

    def record(self,mods):
        for m in mods:
            gate(m.calls>0 and m.native_C_first_call_parity,
                 "real model packed projection not physically executed")
            self.cpu_oracle_checked+=1
            self.max_cpu_gpu_relative_L2=max(
                self.max_cpu_gpu_relative_L2,m.max_C_relative_L2)
            self.raw_F32_GPU_rechecks+=m.extra_F32_oracle_GPU_dispatches
            self.max_F32_vs_C_relative_L2=max(
                self.max_F32_vs_C_relative_L2,m.max_F32_relative_L2)
            self.BF16_rounding_one_ULP_flips+=m.bf16_single_ULP_flips


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir",required=True)
    parser.add_argument("--sealed-root",required=True)
    parser.add_argument("--manifest-sha",required=True)
    parser.add_argument("--native-c-dylib",required=True)
    parser.add_argument("--native-gpu-dylib",required=True)
    parser.add_argument("--max-layers",type=int,choices=(1,2,5,27),required=True)
    parser.add_argument("--greedy-tokens",type=int,choices=(0,2),required=True)
    parser.add_argument("--receipt-dir",required=True)
    args=parser.parse_args()

    started=time.monotonic()
    torch.set_num_threads(2)
    gate(args.max_layers==27 or args.greedy_tokens==0,
         "partial layer smoke cannot claim real model-selected tokens")
    d=Path(args.checkpoint_dir)
    receipt_dir=Path(args.receipt_dir)
    gate(d.is_dir() and not d.is_symlink() and
         receipt_dir.is_dir() and not receipt_dir.is_symlink(),
         "invalid original checkpoint/receipt location")
    for name,sha in (
        ("model.safetensors.index.json",INDEX_SHA),
        ("config.json","f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d"),
        ("modeling_deepseek.py","7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c")):
        gate(sha_path(d/name)==sha,"source checkpoint pinned metadata drift")
    packed,bf16,numeric=model_source()
    # The original model research modules have a default D50 copy root.
    # This explicit per-process source root is independently SHA-pinned.
    packed.DATA=d
    packed.exp.DATA=d
    packed.exp.base.DATA=d
    packed.source.DATA=d
    packed.bf_reference.DATA=d
    bf16.DATA=d
    bf16.exp.DATA=d
    bf16.exp.base.DATA=d
    bf16.src.DATA=d
    bf16.first_ref.DATA=d
    gpu_source=Path(args.native_gpu_dylib)
    gate(sha_path(gpu_source.parent.parent/
        "n7_multi_projection_gpu_residency_v11_gpuowned_candidate.cpp") ==
        GPU_CPP_SHA,"MLX C++ GPU ownership candidate source SHA drift")
    lib=ctypes.CDLL(str(args.native_gpu_dylib))
    lib.beglin_multi_n7_gpu_available_v1.restype=ctypes.c_int
    gate(lib.beglin_multi_n7_gpu_available_v1()==1,
         "physical Apple Silicon Metal required")
    c=ctypes.CDLL(str(args.native_c_dylib))
    ref=c.qng64_matmul_f32
    ref.argtypes=(U,F,ctypes.c_int,F,F,F,ctypes.c_int,ctypes.c_int,ctypes.c_int)
    ref.restype=None
    from prequant_full27_execution_adapter_v1 import SOURCE_NATIVE_C_SHA256
    gate(sha_path(Path(args.native_c_dylib))==SOURCE_NATIVE_C_SHA256,
         "original native C ABI identity mismatch")
    cache=ResidentN7Cache(lib,bytes_budget=256*1024*1024,max_handles=96)
    source=None
    result=None
    try:
        policy=json.loads((BASE/"full27_qng64_equal_budget_policy_manifest_v1.json").read_text())
        known={row["name"]:row for row in policy["rows"]}
        gate(len(known)==5181 and policy["qng64_matrix_count"]==5181,
             "frozen uniform7 complete qNg64 model manifest")
        source=SealedModel(args.sealed_root,args.manifest_sha,known,cache,ref)
        config=json.loads((d/"config.json").read_text())
        cfg=packed.exp.SimpleNamespace(**config)
        cfg._attn_implementation="eager"
        gate(cfg.num_hidden_layers==27 and cfg.n_routed_experts==64 and
             cfg.num_experts_per_tok==6 and cfg.hidden_size==2048,
             "DeepSeek source model architecture changed")
        ns=packed.source.load_sources()
        index=json.loads((d/"model.safetensors.index.json").read_text())["weight_map"]
        tokenizer=Tokenizer.from_file(str(d/"tokenizer.json"))
        prompt=bf16.CHOICES[0]
        prefix=[cfg.bos_token_id]+tokenizer.encode(prompt,add_special_tokens=False).ids
        gate(len(prefix)==5,"authored original 5-token input changed")
        tokens=[]
        stages=[]
        # 0: five-token prefill; 1: selected token as sixth input;
        # 2: selected second token as seventh input. No cached KV.
        for step in range(1+args.greedy_tokens):
            ids=prefix+tokens
            emb,_,_=packed.exp.base.open_sample(
                "model.embed_tokens.weight",index["model.embed_tokens.weight"],
                np.array(ids,dtype=np.int64),{})
            hidden=torch.from_numpy(emb).unsqueeze(0).to(torch.bfloat16)
            length=hidden.shape[1]
            mask=numeric.mask_for(length)
            pos=torch.arange(length,dtype=torch.long).unsqueeze(0)
            headers={}
            stage_start=time.monotonic()
            loaded_before=source.loaded
            count_experts=0
            for i in range(args.max_layers):
                layer_start=time.monotonic()
                if i==0:
                    layer=packed.first.read_original_decoder(ns,cfg,index,headers)
                else:
                    layer,_=bf16.source_layer(ns,cfg,index,headers,i)
                mods=source.replace_fixed(layer,i,packed)
                chosen,experts=source.load_routed(
                    layer,i,hidden,mask,pos,packed,ns,cfg,index,headers)
                mods.extend(experts)
                count_experts+=len(chosen)
                with torch.no_grad():
                    next_hidden=layer(hidden,attention_mask=mask,
                                      position_ids=pos,use_cache=False)[0]
                gate(next_hidden.shape==hidden.shape and
                     bool(torch.isfinite(next_hidden).all()),
                     "nonfinite or wrong DeepSeek source decoder output")
                source.record(mods)
                hidden=next_hidden.clone()
                print(json.dumps({
                    "phase":"ORIGINAL_MODEL_SEALED_N7_REAL_METAL_LAYER",
                    "context_length":length,"layer":i,
                    "qng64_model_roles_executed":len(mods),
                    "router_expert_count":len(chosen),
                    "wall_seconds":round(time.monotonic()-layer_start,3)
                }),flush=True)
                del layer,mods,chosen,experts,next_hidden
                gc.collect()
            stage=dict(context_length=length,
                       model_layers=args.max_layers,
                       actually_routed_experts_across_layers=count_experts,
                       packed_source_module_instances=source.loaded-loaded_before,
                       completed_seconds=round(time.monotonic()-stage_start,3),
                       full_model_logits_computed=False)
            if args.max_layers==27:
                with torch.no_grad():
                    logits,_=packed.finish_logits(
                        index,headers,hidden[0,-1:,:],cfg)
                gate(logits.shape==(1,102400) and np.isfinite(logits).all(),
                     "full-vocabulary original BF16 source lm_head invalid")
                token=int(np.argmax(logits[0]))
                stage["full_model_logits_computed"]=True
                stage["vocab_logits_sha256"]=hashlib.sha256(
                    logits.tobytes()).hexdigest()
                stage["next_prediction"]=token
                # Existing separate original-model 27L frozen qNg64 baseline.
                if step==0:
                    old=np.load(
                        "/Users/eoe/mcp-sandbox/tailnet-commander/"
                        "beglin-real-activations-20261017/"
                        "qng64_uniform7_full27_prompt0_logits_v1.npy",
                        allow_pickle=False)
                    gate(old.shape==(5,102400),"independent old source q7 logits unavailable")
                    base=old[-1].astype(np.float64)
                    got=logits[0].astype(np.float64)
                    rel=float(np.linalg.norm(got-base)/max(np.linalg.norm(base),1e-30))
                    stage["prefill_logits_relative_L2_vs_prior_uniform7"]=rel
                    gate(rel<0.025 and token==int(np.argmax(base)),
                         "original source q7 prefill logits parity declined")
                    del old,base,got
                if step<args.greedy_tokens:
                    tokens.append(token)
            stages.append(stage)
            print(json.dumps({
                "phase":"FULL27_SEALED_STEP_COMPLETE",
                "step":step,"context_length":length,
                "next_prediction":stage.get("next_prediction"),
                "wall_seconds":stage["completed_seconds"]
            }),flush=True)
            del hidden,mask,pos,headers,emb
        stats=cache.get_state()
        if args.max_layers==27 and args.greedy_tokens==2:
            gate(tokens==[429,4281] and stages[-1]["next_prediction"]==6230,
                 "original v11 greedy 429->4281->6230 contract changed")
        result={
            "schema":"beglin.v11.real-deepseek-full27-sealed-n7-metal-uncached-recompute/1",
            "status":"PASS_REAL_FULL27_PREQUANT_TWO_GREEDY_UNCACHED" if
                     args.max_layers==27 and args.greedy_tokens==2 else
                     "PASS_ORIGINAL_SEALED_N7_PREFIX_HARDWARE_SMOKE",
            "checkpoint_index_sha256":INDEX_SHA,
            "packed_source_manifest_sha256":args.manifest_sha,
            "model_layers":args.max_layers,
            "source_prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
            "generated_token_inputs":tokens,
            "predicted_token_after_last_input":stages[-1].get("next_prediction"),
            "actual_model_source_expert_routing":True,
            "native_C_cpp_Metal_v11_GPU_compressed_all_matmuls":True,
            "GPU_original_C_oracle_checked_instances":source.cpu_oracle_checked,
            "GPU_vs_C_max_BF16_relative_L2":source.max_cpu_gpu_relative_L2,
            "extra_native_Metal_F32_oracle_validation_calls":source.raw_F32_GPU_rechecks,
            "GPU_vs_native_C_max_F32_relL2_on_BF16_ambiguity":source.max_F32_vs_C_relative_L2,
            "BF16_rounding_one_ULP_flips":source.BF16_rounding_one_ULP_flips,
            "dynamic_C_quantizations_during_inference":0,
            "compressed_disk_role_load_instances":source.loaded,
            "compressed_disk_logical_bytes":source.packed_bytes,
            "physical_Metal_GPU_forward_calls":stats["actual_GPU_forward_calls"] + source.raw_F32_GPU_rechecks,
            "normal_Model_GPU_forward_calls":stats["actual_GPU_forward_calls"],
            "Metal_GPU_registration_count":stats["GPU_registration_count"],
            "Metal_GPU_reuse_hits":stats["GPU_reuse_hits"],
            "Metal_GPU_evictions":stats["GPU_LRU_evictions"],
            "MLA_cached_KV_tested":False,
            "recomputed_full_context_each_step":True,
            "BF16_original_source_layer_and_expert_loads_still_present":True,
            "real_serving_TTFT_TPOT_validated":False,
            "source_model_weights_gpu_full_resident":False,
            "original_checkpoint_modified":False,
            "production_promotion_allowed":False,
            "stage_receipts":stages,
            "peak_process_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "wall_seconds":round(time.monotonic()-started,3),
        }
    finally:
        cache.close()
    gate(lib.beglin_multi_n7_active_count_v1()==0 and
         lib.beglin_multi_n7_payload_bytes_v1()==0,
         "v11 full27 actual Metal packed weight handles not all released")
    result["all_GPU_handles_released"]=True
    output=(json.dumps(result,indent=2,sort_keys=True)+"\n").encode()
    name=("BEGLIN_v11_sealed_full27_uncached_"+str(args.max_layers)+"L_"+
          str(args.greedy_tokens)+"tokens_"+str(time.time_ns())+".json")
    receipt=receipt_dir/name
    with receipt.open("xb") as fd:
        fd.write(output)
        fd.flush()
        os.fsync(fd.fileno())
    print(json.dumps({
        "status":result["status"],"layers":args.max_layers,
        "token_inputs":tokens,
        "prediction":result["predicted_token_after_last_input"],
        "GPU_role_module_instances":source.loaded,
        "GPU_forward_calls":stats["actual_GPU_forward_calls"],
        "GPU_cache_hits":stats["GPU_reuse_hits"],
        "all_GPU_handles_released":True,
        "receipt_sha256":hashlib.sha256(output).hexdigest(),
        "wall_seconds":result["wall_seconds"],
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
