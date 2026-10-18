#!/usr/bin/env python3
"""Original DeepSeek-V2-Lite layer0: frozen 7 role v11 prequant adapter/Metal smoke.

Uses a research-only TEMP preview manifest of seven ALREADY sealed source roles,
not a full27 manifest. The production-bound adapter's full27 policy gates are
overridden only inside this isolated test process to seven roles; the real
model-wide adapter still requires 5,181 roles. Reads original packed weights,
NOT original BF16 tensor payloads, and feeds synthetic BF16 activation vectors.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

import prequant_full27_execution_adapter_v1 as adapter
from prequant_n7_sealed_v1 import role_files, SCHEMA
from full27_sealed_materializer_v1 import (
    INDEX_SHA, load_boundary, c_quantizer, existing_receipt, json_ascii, gate,
)
from research_n7_device_lru_cache_v3_GPUonly_candidate import ResidentN7Cache

F=ctypes.POINTER(ctypes.c_float)
U=ctypes.POINTER(ctypes.c_uint8)
ATTENTION=("self_attn.q_proj","self_attn.kv_a_proj_with_mqa",
           "self_attn.kv_b_proj","self_attn.o_proj")
FFN=("gate_proj","up_proj","down_proj")


class EmptyPackedNative(nn.Module):
    n_constructed=0
    def forward(self,x):
        raise RuntimeError("CPU reference is tested separately via original C; this is metadata-only constructor")


class ShapeModel:
    def __init__(self, seven):
        self.self_attn=SimpleNamespace()
        self.mlp=SimpleNamespace()
        for role,dims in seven.items():
            key=role.split(".")
            if key[3]=="self_attn":
                setattr(self.self_attn,key[4],SimpleNamespace(weight=SimpleNamespace(shape=dims)))
            else:
                setattr(self.mlp,key[4],SimpleNamespace(weight=SimpleNamespace(shape=dims)))
    def get_submodule(self,path):
        obj=self
        for seg in path.split("."):
            obj=getattr(obj,seg)
        return obj


def ptr(a,t):
    return a.ctypes.data_as(t)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",required=True)
    parser.add_argument("--boundary-tsv",required=True)
    parser.add_argument("--native-c-dylib",required=True)
    parser.add_argument("--native-gpu-dylib",required=True)
    parser.add_argument("--policy-json",required=True)
    args=parser.parse_args()
    stage=Path(args.stage)
    rows,excluded,meta=load_boundary(Path(args.boundary_tsv))
    seven={k:v for k,v in rows.items() if k.startswith("model.layers.0.")}
    gate(len(seven)==7,"original first decoder exact 7 prequant roles")
    library,lib_sha=c_quantizer(Path(args.native_c_dylib))
    policy=json.loads(Path(args.policy_json).read_text())
    known={v["name"]:v for v in policy["rows"] if v["name"] in seven}
    gate(len(known)==7,"real seven policies absent")
    native=ctypes.CDLL(str(args.native_c_dylib))
    fn=native.qng64_matmul_f32
    fn.argtypes=(U,F,ctypes.c_int,F,F,F,ctypes.c_int,ctypes.c_int,ctypes.c_int)
    fn.restype=None
    gpu=ctypes.CDLL(str(args.native_gpu_dylib))
    gpu.beglin_multi_n7_gpu_available_v1.restype=ctypes.c_int
    gate(gpu.beglin_multi_n7_gpu_available_v1()==1,"physical Metal required")

    # Hardlinks refer to the original read-only packed bytes; a temporary
    # seven-role manifest exists only during this opt-in EOE test.
    with tempfile.TemporaryDirectory(prefix="beglin_l0_sealed_preview_") as td:
        preview=Path(td)
        manifest={"schema":SCHEMA,"source_checkpoint_index_sha256":INDEX_SHA,"roles":[]}
        for role in sorted(seven):
            r=existing_receipt(stage,seven[role],lib_sha)
            gate(r is not None,"first layer has no completed native packed role")
            pn,sn=role_files(role)
            os.link(stage/pn,preview/pn)
            os.link(stage/sn,preview/sn)
            manifest["roles"].append(dict(
                role=role, out=r["out"], in_features=r["in_features"],
                planes_sha256=r["planes_sha256"],scales_sha256=r["scales_sha256"]))
        raw=json_ascii(manifest)
        (preview/"manifest.json").write_bytes(raw)
        digest=hashlib.sha256(raw).hexdigest()
        packed_bytes=sum(seven[k]["serialized_bytes"] for k in seven)
        gate(packed_bytes==75939840,"first decoder original exact packed bytes")
        adapter.FULL_ROLES=7
        adapter.FULL_Q_BYTES=packed_bytes
        class Meta:
            ATTENTION=ATTENTION
            FFN=FFN
        hooks=adapter.SealedPrequantFull27(preview,digest,known,EmptyPackedNative,Meta)
        model=ShapeModel({k:(v["out"],v["in_features"]) for k,v in seven.items()})
        count,total,bf16,mods=hooks.prepare_weights(model,0,"uniform7",known,native,None)
        gate(count==7 and total==packed_bytes and hooks.loads==7 and
             EmptyPackedNative.n_constructed==7,
             "decoder native class fast constructor/gate failure")
        registry=ResidentN7Cache(gpu,bytes_budget=128*1024*1024,max_handles=12)
        observed=0.0
        try:
            for path in ATTENTION+tuple("mlp."+p for p in FFN):
                role="model.layers.0."+path+".weight"
                m=model.get_submodule(path)
                x=torch.linspace(-0.6,0.6,m.in_features,dtype=torch.float32).to(torch.bfloat16)[None,:]
                gpu_y=registry.call(role,m,x).float().numpy()
                xin=np.ascontiguousarray(x.float().numpy())
                cpu_y=np.empty((1,m.out),np.float32)
                fn(ptr(m.packed,U),ptr(m.scales,F),7,ptr(xin,F),None,
                   ptr(cpu_y,F),m.out,m.in_features,1)
                cpu_bf16=torch.from_numpy(cpu_y).to(torch.bfloat16).float().numpy()
                delta=np.linalg.norm(gpu_y.astype(np.float64)-cpu_bf16.astype(np.float64))
                baseline=np.linalg.norm(cpu_bf16.astype(np.float64))
                rel=float(delta/max(baseline,1e-30))
                gate(rel<3e-4,"real packed weight physical Metal/C output disagreement")
                observed=max(observed,rel)
            state=registry.get_state()
            gate(state["actual_GPU_forward_calls"]==7 and
                 state["GPU_registration_count"]==7,
                 "physical Metal 7 original packed model role count mismatch")
        finally:
            registry.close()
        gate(gpu.beglin_multi_n7_active_count_v1()==0 and
             gpu.beglin_multi_n7_payload_bytes_v1()==0,
             "first decoder model role GPU handle leak")
        print(json.dumps({
            "status":"PASS_REAL_DEEPSEEK_LAYER0_SEVEN_PREQUANT_ADAPTER_GPU",
            "original_model_packed_roles":7,
            "original_BF16_source_payload_read_during_GPU_test":False,
            "prequant_uniform_n7_bytes":packed_bytes,
            "physical_gpu_native_forward_calls":7,
            "max_BF16_Output_relL2_vs_original_native_C":observed,
            "prequant_sha_verified":True,
            "gpu_handles_after_teardown":0,
            "full27_inference_verified":False,
            "production_promotion_allowed":False,
        },sort_keys=True))


if __name__=="__main__":
    main()
