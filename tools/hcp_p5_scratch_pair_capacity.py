#!/usr/bin/env python3
"""Increase DeepSeek scratch C and MLX GPU cache lengths together, never live."""
from __future__ import annotations
import argparse,hashlib,json,pathlib
from hcp_p5_scratch_capacity128 import apply as patch_scratch_c

MLX_GIT_BLOB_SHA = "b6cbc1c98cfedcd0f3c99f437d1f6c5978f2ce5a"

def patch_both(root):
    root=pathlib.Path(root).resolve(strict=True)
    mlx=root/"mlx_moe.cpp"
    if not mlx.is_file() or mlx.is_symlink():
        raise ValueError("MLX scratch source not regular file")
    original=mlx.read_bytes()
    blob=hashlib.sha1(b"blob "+str(len(original)).encode()+bytes([0])+original).hexdigest()
    if blob!=MLX_GIT_BLOB_SHA:
        raise ValueError("MLX backend source commit mismatch")
    text=original.decode("utf-8")
    old="#define MLA_L0_MAXPOS 32"
    new="#define MLA_L0_MAXPOS 128"
    if text.count(old)!=1 or text.count("#define GQA_L0_MAXPOS 32")!=1:
        raise ValueError("unexpected MLX MLA/GQA positional limit definitions")
    for marker in ("mlx_gpu_cbatch_layer_step_lazy","spos[i] >= MLA_L0_MAXPOS",
                   "g_fused_K.push_back","g_fused_V.push_back"):
        if marker not in text:
            raise ValueError("missing MLX fused KV bounds check")
    result=patch_scratch_c(root)
    mlx.write_text(text.replace(old,new))
    receipt={"schema":"beglin-p5-dual-c-mlx-128-scratch-v1",
        "original_mlx_blob_sha":blob,
        "patched_mlx_sha256":hashlib.sha256(mlx.read_bytes()).hexdigest(),
        "native_c_capacity":128,"gpu_mla_KV_capacity":128,
        "separate_gqa_capacity_left_unchanged":32,
        "source_c_receipt":result,
        "production_touched":False,"live_promotion":False}
    print("P5_C_MLX_SCRATCH_BOUNDS_MATCH",json.dumps(receipt,sort_keys=True))
    return receipt

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source-root",type=pathlib.Path,required=True)
    a=p.parse_args()
    patch_both(a.source_root)
