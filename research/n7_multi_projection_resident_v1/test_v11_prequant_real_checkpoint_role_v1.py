#!/usr/bin/env python3
"""Opt-in physical Metal EOE research: one ORIGINAL BF16 checkpoint role.

Attests BOTH DeepSeek-V2-Lite index and the source 8-GiB shard SHA before
reading one BF16 projection. It quantizes ONCE using original BEGLIN native C,
seals the packed source, then loads through v11 read-only prequant mmap and
dispatches repeated C++ Metal GPU-resident GEMVs without re-quantization.
Only receipt statistics and hashes are printed; weight payloads / prompts
are never uploaded. A single role is NOT full-model prequant loader proof.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from safetensors import safe_open

from prequant_n7_sealed_v1 import PackWriter, PackReader
from research_n7_device_lru_cache_v3_GPUonly_candidate import ResidentN7Cache

SOURCE_INDEX_SHA = "d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
SOURCE_SHARD_SHA = "a314b79a30810b1038f9db08345178e16fb5742aa266c4a3dc05054c59800082"
SOURCE_SHARD = "model-00001-of-000004.safetensors"
ROLE = "model.layers.0.self_attn.q_proj.weight"
CPP_SHA = "f10a8b30bfeff95cf0e756d45567ac4aaaa3d2552f5ce62a7746af09000a3201"
HERE = Path(__file__).resolve().parent
GPU = HERE / "build" / "libbeglin_n7_multi_projection_GPU_resident_v11_candidate.dylib"
C = HERE / "build" / "libbeglin_qng64_reference.dylib"
U = ctypes.POINTER(ctypes.c_uint8)
F = ctypes.POINTER(ctypes.c_float)


def gate(ok, why):
    if not ok:
        raise RuntimeError("BEGLIN_V11_REAL_ROLE_PREQUANT_FAIL_CLOSED: " + why)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ptr(a, datatype):
    return a.ctypes.data_as(datatype)


def run(checkpoint, scratch):
    checkpoint = Path(checkpoint)
    scratch = Path(scratch)
    gate(checkpoint.is_dir() and scratch.is_dir() and not checkpoint.is_symlink(),
         "source checkpoint or staging root not accessible")
    index = checkpoint / "model.safetensors.index.json"
    shard = checkpoint / SOURCE_SHARD
    gate(index.is_file() and not index.is_symlink() and digest(index) == SOURCE_INDEX_SHA,
         "original BF16 checkpoint index SHA mismatch")
    idx = json.loads(index.read_text())
    gate(idx.get("weight_map", {}).get(ROLE) == SOURCE_SHARD,
         "real checkpoint mapping does not match frozen role/shard")
    sh = shard.stat()
    gate(sh.st_size >= 8 * 1024**3 and digest(shard) == SOURCE_SHARD_SHA,
         "original immutable source BF16 shard SHA mismatch")
    gate((sh.st_ino, sh.st_size, sh.st_mtime_ns) ==
         (shard.stat().st_ino, shard.stat().st_size, shard.stat().st_mtime_ns),
         "source shard changed during SHA attestation")
    gate(GPU.is_file() and C.is_file() and
         digest(HERE / "n7_multi_projection_gpu_residency_v11_gpuowned_candidate.cpp") == CPP_SHA,
         "original C/v11 candidate source and native build missing/mismatched")
    ref = ctypes.CDLL(str(C))
    quantize = ref.qng64_quantize_f32
    quantize.argtypes = (F, ctypes.c_int, ctypes.c_int, ctypes.c_int, U, F)
    quantize.restype = ctypes.c_int
    reference = ref.qng64_matmul_f32
    reference.argtypes = (U, F, ctypes.c_int, F, F, F,
                          ctypes.c_int, ctypes.c_int, ctypes.c_int)
    reference.restype = None
    with safe_open(str(shard), framework="pt", device="cpu") as reader:
        bf16 = reader.get_tensor(ROLE)
        gate(bf16.dtype == torch.bfloat16 and bf16.ndim == 2 and
             tuple(bf16.shape) == (3072, 2048),
             "original source BF16 q_proj shape/dtype mismatch")
        w = np.ascontiguousarray(bf16.float().numpy(), dtype=np.float32)
    out, in_features = w.shape
    groups = out * in_features // 64
    packed = np.empty((groups * 56,), dtype=np.uint8)
    scales = np.empty((groups,), dtype=np.float32)
    gate(quantize(ptr(w, F), out, in_features, 7,
                  ptr(packed, U), ptr(scales, F)) == 0,
         "real role original C qNg64 n7 quantization failed")
    source_packed_sha = hashlib.sha256(packed.tobytes()+scales.tobytes()).hexdigest()
    gpu = ctypes.CDLL(str(GPU))
    gpu.beglin_multi_n7_gpu_available_v1.restype = ctypes.c_int
    gate(gpu.beglin_multi_n7_gpu_available_v1() == 1, "real physical Apple Metal required")
    with tempfile.TemporaryDirectory(prefix="beglin_real_prequant_v11_", dir=scratch) as temp:
        root = Path(temp) / "sealed"
        writer = PackWriter(root, SOURCE_INDEX_SHA)
        writer.add(ROLE, out, in_features, packed, scales)
        manifest_sha = writer.seal()
        prequant = PackReader(root, manifest_sha, SOURCE_INDEX_SHA).load(ROLE)
        gate(np.array_equal(prequant.packed, packed) and
             np.array_equal(prequant.scales, scales) and
             not prequant.packed.flags.writeable,
             "original source quantized planes changed in disk/reload")
        del w, bf16, packed, scales
        # This synthetic hidden-state vector is intentionally NOT a task-quality
        # or original prompt evaluation. It only checks genuine original WEIGHTS.
        inputs = torch.linspace(-0.7, 0.7, in_features, dtype=torch.float32).to(torch.bfloat16)[None, :]
        x = np.ascontiguousarray(inputs.float().numpy())
        y_ref = np.empty((1, out), dtype=np.float32)
        reference(ptr(prequant.packed, U), ptr(prequant.scales, F),
                  7, ptr(x, F), None, ptr(y_ref, F), out, in_features, 1)
        bf_ref = torch.from_numpy(y_ref).to(torch.bfloat16)
        cache = ResidentN7Cache(gpu, bytes_budget=16*1024*1024, max_handles=2)
        start = time.monotonic()
        try:
            checks = []
            for _ in range(20):
                y = cache.call(ROLE, prequant, inputs)
                gate(y.shape == bf_ref.shape and torch.isfinite(y).all(),
                     "GPU native BF16 output invalid")
                d = torch.linalg.vector_norm((y.float()-bf_ref.float())).item()
                denom = torch.linalg.vector_norm(bf_ref.float()).item()
                gate(denom > 0 and d/denom < 3e-4,
                     "loaded real original q_proj Metal / C reference diverged")
                checks.append(d/denom)
            metrics = cache.get_state()
        finally:
            cache.close()
        gate(gpu.beglin_multi_n7_active_count_v1() == 0 and
             gpu.beglin_multi_n7_payload_bytes_v1() == 0,
             "original real role GPU handle was not freed")
        gate(metrics["actual_GPU_forward_calls"] == 20 and
             metrics["GPU_registration_count"] == 1 and
             metrics["GPU_reuse_hits"] == 19,
             "real source GPU reuse/quant-free model role contract failed")
        result = {
            "schema": "beglin.v11.real-source-prequant-one-role-Metal/1",
            "status": "PASS_REAL_QPROJ_SEALED_PREQUANT_DEVICE_GPU_REUSE",
            "role": ROLE, "shape": [out, in_features],
            "checkpoint_index_sha256": SOURCE_INDEX_SHA,
            "original_shard_sha256": SOURCE_SHARD_SHA,
            "C_quantized_payload_sha256": source_packed_sha,
            "sealed_manifest_sha256": manifest_sha,
            "original_C_quantize_calls_during_export": 1,
            "source_BF16_read_calls_during_inference": 0,
            "CPU_packed_requantization_during_inference": 0,
            "actual_physical_GPU_Metal_calls": metrics["actual_GPU_forward_calls"],
            "GPU_cache_reuse_hits": metrics["GPU_reuse_hits"],
            "max_gpu_vs_native_C_BF16_relative_L2": max(checks),
            "actual_GPU_handle_count_after_close": 0,
            "packed_bytes": prequant.serialized_bytes,
            "test_seconds_excluding_shard_sha": round(time.monotonic()-start, 4),
            "full_model_prequant_loader_verified": False,
            "model_prompt_quality_verified": False,
            "production_promotion_allowed": False,
        }
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--scratch-dir", required=True)
    args = parser.parse_args()
    run(args.checkpoint_dir, args.scratch_dir)
