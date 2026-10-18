#!/usr/bin/env python3
"""Independent read-only whole-artifact audit for BEGLIN v11 packed qNg64 n7.

Verifies ALL 5,181 stored planes+scales SHA/shape/bitwidth contracts,
source checkpoint shards, and 27 independently C-requantized real q_proj
role bytes against frozen output. Does not execute full-model logits itself.
All raw weights remain on local researcher host; receipt hashes only.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
from safetensors import safe_open

from prequant_n7_sealed_v1 import PackReader
from full27_sealed_materializer_v1 import (
    INDEX_SHA, SHARD_SHAS, FULL_Q_BYTES, FULL_ROLES, FULL_TENSORS,
    BOUNDARY_SHA, load_boundary, index_authority,
    validate_full_shard_shas, c_quantizer, existing_receipt, gate, json_ascii,
    sha_path,
)

U=ctypes.POINTER(ctypes.c_uint8)
F=ctypes.POINTER(ctypes.c_float)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir",required=True)
    ap.add_argument("--boundary-tsv",required=True)
    ap.add_argument("--sealed-root",required=True)
    ap.add_argument("--manifest-sha",required=True)
    ap.add_argument("--native-c-dylib",required=True)
    ap.add_argument("--receipt-dir",required=True)
    args=ap.parse_args()
    started=time.monotonic()
    checkpoint=Path(args.checkpoint_dir)
    root=Path(args.sealed_root)
    boundary=Path(args.boundary_tsv)
    receipt_dir=Path(args.receipt_dir)
    gate(root.is_dir() and not root.is_symlink() and
         receipt_dir.is_dir() and not receipt_dir.is_symlink(),
         "sealed source or audit receipt root not valid")
    rows,excluded,_=load_boundary(boundary)
    index_authority(checkpoint,rows,excluded)
    validate_full_shard_shas(checkpoint)
    quant,quant_sha=c_quantizer(Path(args.native_c_dylib))
    reader=PackReader(root,args.manifest_sha,INDEX_SHA)
    gate(set(reader.roles)==set(rows) and len(reader.roles)==FULL_ROLES
         and reader.total_packed_bytes==FULL_Q_BYTES and
         len(excluded)==FULL_TENSORS-FULL_ROLES,
         "whole-model sealed packed manifest not complete")
    authority=json.loads((root/"RUN_AUTHORITY.json").read_text())
    gate(authority["source_shard_sha256"]==SHARD_SHAS and
         authority["index_sha256"]==INDEX_SHA and
         authority["boundary_tsv_sha256"]==BOUNDARY_SHA and
         authority["native_quantizer_sha256"]==quant_sha,
         "materializer run did not match original frozen policy/source/C")
    all_role_bytes=0
    for i,role in enumerate(sorted(rows)):
        metadata=existing_receipt(root,rows[role],quant_sha)
        gate(metadata is not None,"one of the 5,181 roles lacks committed native output")
        data=reader.load(role)
        gate(data.serialized_bytes==rows[role]["serialized_bytes"] and
             data.out==rows[role]["out"] and
             data.in_features==rows[role]["in_features"],"original frozen role geometry")
        all_role_bytes+=data.serialized_bytes
        del data
        if (i+1)%1000==0:
            print(json.dumps({"audit_role_count":i+1,
                              "elapsed_seconds":round(time.monotonic()-started,2)}),
                  flush=True)
    gate(all_role_bytes==FULL_Q_BYTES, "full model actual serialized SHA verified bytes")

    # Independent full-original-BF16 -> native C qNg64 byte parity on one
    # predetermined Q projection from EACH of the 27 decoder layers.
    qproj_sha={}
    for layer in range(27):
        role=f"model.layers.{layer}.self_attn.q_proj.weight"
        entry=rows[role]
        with safe_open(str(checkpoint/entry["shard"]),framework="pt",device="cpu") as file:
            bf16=file.get_tensor(role)
            gate(str(bf16.dtype)=="torch.bfloat16" and
                 tuple(bf16.shape)==(entry["out"],entry["in_features"]),
                 "independent original source Q tensor type/shape mismatch")
            weights=np.ascontiguousarray(bf16.float().numpy(),dtype=np.float32)
            del bf16
        groups=entry["out"]*entry["in_features"]//64
        plane=np.empty(groups*56,dtype=np.uint8)
        scale=np.empty(groups,dtype=np.float32)
        gate(quant(weights.ctypes.data_as(F),entry["out"],entry["in_features"],7,
                   plane.ctypes.data_as(U),scale.ctypes.data_as(F))==0,
             "independent native C quantization failed")
        original=reader.load(role)
        gate(np.array_equal(plane,original.packed) and
             np.array_equal(scale,original.scales),
             "stored n7 role drifted from actual BF16 source")
        qproj_sha[str(layer)]=hashlib.sha256(
            plane.tobytes()+scale.tobytes()).hexdigest()
        del original, weights, plane, scale
    result={
        "schema":"beglin.v11.full27.prequant-independent-artifact-audit/1",
        "status":"PASS_REAL_FULL27_5181_SEALED_ALL_HASHES_27_INDEPENDENT_C_REQUANT",
        "source_index_sha256":INDEX_SHA,
        "source_shard_sha256":SHARD_SHAS,
        "source_boundary_sha256":BOUNDARY_SHA,
        "native_quantizer_sha256":quant_sha,
        "sealed_manifest_sha256":args.manifest_sha,
        "all_5291_original_model_tensors_classified":True,
        "all_qng64_group64_bitwidth7_roles":FULL_ROLES,
        "BF16_excluded_tensor_count":len(excluded),
        "serialized_qng64_verified_bytes":all_role_bytes,
        "independently_source_requantized_qproj_layers":27,
        "independent_C_quantized_Qrole_SHA256_per_layer":qproj_sha,
        "all_materialized_planes_and_scales_SHA_verified":True,
        "separate_full_model_greedy_tokens_evaluated":False,
        "original_model_modified":False,
        "production_promotion_allowed":False,
        "max_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds":round(time.monotonic()-started,3),
    }
    data=json_ascii(result)
    path=receipt_dir/"BEGLIN_V11_FULL27_5181_SEALED_AUDIT_v1.json"
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({
        "status":result["status"],
        "verified_qng64_bytes":all_role_bytes,
        "original_C_requant_parity_qproj_layers":27,
        "receipt_sha256":hashlib.sha256(data).hexdigest(),
        "elapsed_seconds":result["elapsed_seconds"],
        "production_promotion_allowed":False},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
