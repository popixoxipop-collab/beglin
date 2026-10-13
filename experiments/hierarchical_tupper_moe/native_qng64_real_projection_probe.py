#!/usr/bin/env python3
"""Read-only real DeepSeek BF16 gate projection through Beglin qNg64 C codec.

Offline local projection/precision/CPU timing only. Does NOT evaluate a trained
micro-router, GPU parity, native serving latency, or whole-model token quality.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
import pathlib
import statistics
import time
import numpy as np
from real_checkpoint_reader import RealExpertTileReader

F32P=ct.POINTER(ct.c_float)
U8P=ct.POINTER(ct.c_uint8)

def verify_real_projection(index, library, output, *, layer=1, expert=0, hidden=2048, intermediate=1408):
    idx=pathlib.Path(index).resolve(strict=True)
    dll=ct.CDLL(str(pathlib.Path(library).resolve(strict=True)))
    dll.qng64_packed_bytes.argtypes=[ct.c_int,ct.c_int,ct.c_int]
    dll.qng64_packed_bytes.restype=ct.c_size_t
    dll.qng64_quantize_f32.argtypes=[F32P,ct.c_int,ct.c_int,ct.c_int,U8P,F32P]
    dll.qng64_quantize_f32.restype=ct.c_int
    dll.qng64_matvec_f32.argtypes=[U8P,F32P,ct.c_int,F32P,F32P,F32P,ct.c_int,ct.c_int]
    dll.qng64_matvec_f32.restype=None
    rdr=RealExpertTileReader(idx,layer=layer,expert_ids=[expert],hidden=hidden,intermediate=intermediate)
    ref=rdr._array(expert,"gate_proj")
    w=np.ascontiguousarray(rdr._float(ref,rdr.tensors[(expert,"gate_proj")]["dtype"]),dtype=np.float32)
    if w.shape!=(intermediate,hidden):raise ValueError("unexpected true model projection")
    rng=np.random.default_rng(20261014)
    x=np.ascontiguousarray(rng.normal(size=hidden).astype(np.float32)*(.40))
    reference=w.astype(np.float64)@x.astype(np.float64)
    records=[]
    for n in (3,5,7,9):
        nb=int(dll.qng64_packed_bytes(intermediate,hidden,n))
        if nb<=0:raise ValueError("native qNg64 size invalid")
        packed=np.empty(nb,dtype=np.uint8)
        scale=np.empty(intermediate*(hidden//64),dtype=np.float32)
        rc=dll.qng64_quantize_f32(w.ctypes.data_as(F32P),intermediate,hidden,n,
                                 packed.ctypes.data_as(U8P),scale.ctypes.data_as(F32P))
        if rc:raise RuntimeError("native qNg64 quantizer returned error")
        y=np.empty(intermediate,dtype=np.float32)
        samples=[]
        for repeat in range(12):
            t=time.perf_counter_ns()
            dll.qng64_matvec_f32(packed.ctypes.data_as(U8P),scale.ctypes.data_as(F32P),n,
                                  x.ctypes.data_as(F32P),None,y.ctypes.data_as(F32P),intermediate,hidden)
            dt=(time.perf_counter_ns()-t)/1e6
            if repeat>=2:samples.append(dt)
        error=y.astype(np.float64)-reference
        rel=float(np.linalg.norm(error)/max(np.linalg.norm(reference),1e-25))
        if not np.isfinite(rel):raise ValueError("nonfinite native qNg64 projection output")
        records.append({"bits":n,"packed_bytes":nb,"scale_bytes":int(scale.nbytes),
            "total_bytes":int(nb+scale.nbytes),
            "relative_l2_vs_bf16_projection":rel,
            "max_abs_error_vs_bf16_projection":float(np.max(np.abs(error))),
            "cpu_median_matvec_ms":float(statistics.median(samples)),
            "cpu_sample_count":len(samples)})
    body={"schema":"beglin-qng64-real-weight-local-cpu-v1","status":"LOCAL_PROJECTION_ONLY",
        "checkpoint_index_sha256":hashlib.sha256(idx.read_bytes()).hexdigest(),
        "model":"DeepSeek-V2-Lite","layer":layer,"expert":expert,"role":"gate_proj",
        "shape":[intermediate,hidden],"bf16_weight_bytes":2*intermediate*hidden,
        "real_pretrained_weights":True,"actual_hidden_inputs":False,"input_kind":"seeded synthetic norm matched by amplitude, not real h2",
        "native_beglin_cpu_qng64_quantize_and_matvec":True,"whole_model_token_accuracy":False,
        "cpu_expert_micro_router_used":False,"gpu_parity":False,"production_touched":False,
        "source_shard_content_sha_not_verified_this_run":True,"records":records}
    output=pathlib.Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(body,sort_keys=True,indent=2)+"\n")
    print("NATIVE_QNG64_REAL_PROJECTION",json.dumps({"n":len(records),"relative_l2":{r["bits"]:round(r["relative_l2_vs_bf16_projection"],6) for r in records},"median_cpu_ms":{r["bits"]:round(r["cpu_median_matvec_ms"],4) for r in records}},sort_keys=True))
    return body

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint-index",required=True,type=pathlib.Path)
    p.add_argument("--library",required=True,type=pathlib.Path)
    p.add_argument("--output",required=True,type=pathlib.Path)
    a=p.parse_args()
    verify_real_projection(a.checkpoint_index,a.library,a.output)
