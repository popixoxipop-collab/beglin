#!/usr/bin/env python3
"""JE01 source-attested BF16 -> actual qNg64(2,6)/q8g64(8) research artifacts.

Reads 24 *selected* original BF16 Expert roles, invokes the PINNED existing
BEGLIN native C quantizers, independently decodes the exact saved packed bits
and writes SHA-attested scratch-only artifacts. Not a mixed routed GPU model,
not a matched-byte complete assignment, and not NLL performance evidence.
Only runs on EOE D50 scratch; never overwrites an existing output directory.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import re

import numpy as np

from je01_native_exact_byte_presolve import (
    canonical, projected_slot_bytes, read_verified_ledger, validate_ledger,
)

SCHEMA = "beglin-je01-bf16-selected8-native-packed-tiers-research-only/1"
SOURCE_RECEIPT = "689c78abe0280c1cc961222f8c8601e54ad89c97641aeee62ef3c4e05951fe7b"
SOURCE_MODEL_ID = "36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc"
C_BLOB_SHA1 = "ff739d886f55b5a35843cf82fb55c9cd8f76104f"
SOURCE_ROOT = Path("/Volumes/D50/deepseek_v2lite_bf16_safetensors")
OUTPUT_PARENT = Path("/Volumes/D50/TailnetCommander/work")
SRC_RECORD = Path(__file__).resolve().parents[1] / "docs/research/JE01_REAL_BF16_CHECKPOINT_SHA_20261117.json"
LEDGER = Path(__file__).resolve().parents[1] / "docs/research/JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json"
C_SOURCE = Path(__file__).resolve().parents[1] / "gguf_transcode.c"
GROUP = 64
TIERS = (2, 6, 8)
SHAPES = {"gate_proj": (1408, 2048), "up_proj": (1408, 2048), "down_proj": (2048, 1408)}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2 << 20), b""):
            h.update(block)
    return h.hexdigest()


def secure_regular(root: Path, path: Path) -> Path:
    if not path.is_file() or path.is_symlink() or root.resolve() not in path.resolve().parents:
        raise ValueError("JE01_SOURCE_MISSING_SYMLINK_OR_PATH_ESCAPE:" + path.name)
    return path


def attestation():
    d = json.loads(SRC_RECORD.read_text())
    h = d.pop("receipt_sha256", None)
    if h != SOURCE_RECEIPT or canonical(d) != SOURCE_RECEIPT:
        raise ValueError("JE01_BF16_SOURCE_ATTESTATION_CHANGED")
    if d.get("checkpoint_identity_sha256") != SOURCE_MODEL_ID or d.get("status") != "FULL_BF16_SHA_ONLY":
        raise ValueError("JE01_BF16_MODEL_IDENTITY_CHANGED")
    return d["model"]


@contextmanager
def verify_source(model):
    root = SOURCE_ROOT
    if not root.is_dir() or root.is_symlink() or root.resolve() != root:
        raise ValueError("JE01_BF16_SOURCE_ROOT_UNAVAILABLE")
    idxpath = secure_regular(root, root / "model.safetensors.index.json")
    cfgpath = secure_regular(root, root / "config.json")
    idxraw = idxpath.read_bytes()
    cfgraw = cfgpath.read_bytes()
    if (hashlib.sha256(idxraw).hexdigest() != model["index_sha256"] or
            hashlib.sha256(cfgraw).hexdigest() != model["config_sha256"]):
        raise ValueError("JE01_BF16_INDEX_CONFIG_DRIFT")
    index = json.loads(idxraw)
    if index.get("metadata", {}).get("total_size") != 31_412_968_448:
        raise ValueError("JE01_BF16_INDEX_TOTAL_DRIFT")
    headers = {}
    with ExitStack() as stack:
        for shard in model["shards"]:
            path = secure_regular(root, root / shard["name"])
            f = stack.enter_context(path.open("rb"))
            if os.fstat(f.fileno()).st_size != shard["size_bytes"]:
                raise ValueError("JE01_BF16_SHARD_SIZE_DRIFT:" + shard["name"])
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(2 << 20), b""):
                digest.update(block)
            if digest.hexdigest() != shard["sha256"]:
                raise ValueError("JE01_BF16_SHARD_SHA_DRIFT:" + shard["name"])
            f.seek(0)
            rawlen = f.read(8)
            if len(rawlen) != 8:
                raise ValueError("JE01_BF16_SHARD_HEADER_SHORT")
            n = struct.unpack("<Q", rawlen)[0]
            if n < 64 or n > 32_000_000:
                raise ValueError("JE01_BF16_SHARD_HEADER_BOUNDS")
            header = f.read(n)
            if len(header) != n:
                raise ValueError("JE01_BF16_SHARD_HEADER_TRUNCATED")
            headers[shard["name"]] = (f, json.loads(header), 8 + n)
        yield index["weight_map"], headers


def tensor_raw(weightmap, opened, layer, expert, role):
    key = f"model.layers.{layer}.mlp.experts.{expert}.{role}.weight"
    shard = weightmap.get(key)
    if shard not in opened:
        raise ValueError("JE01_BF16_SELECTED_EXPERT_NOT_IN_INDEX:" + key)
    fh, header, base = opened[shard]
    metadata = header.get(key)
    shape = SHAPES[role]
    if not isinstance(metadata, dict) or metadata.get("dtype") != "BF16" or metadata.get("shape") != list(shape):
        raise ValueError("JE01_BF16_SELECTED_TENSOR_META_DRIFT:" + key)
    offsets = metadata.get("data_offsets")
    if (not isinstance(offsets, list) or len(offsets) != 2 or
            type(offsets[0]) is not int or type(offsets[1]) is not int or
            offsets[0] < 0 or offsets[1] - offsets[0] != math.prod(shape) * 2):
        raise ValueError("JE01_BF16_SELECTED_TENSOR_OFFSETS_DRIFT:" + key)
    fh.seek(base + offsets[0])
    raw = fh.read(offsets[1] - offsets[0])
    if len(raw) != math.prod(shape) * 2:
        raise ValueError("JE01_BF16_SELECTED_TENSOR_TRUNCATED:" + key)
    return raw


def bf16_to_float(raw, shape):
    if len(raw) != math.prod(shape) * 2:
        raise ValueError("JE01_BF16_DECODE_SIZE")
    u = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    w = (u << np.uint32(16)).view("<f4").reshape(shape)
    if not np.isfinite(w).all():
        raise ValueError("JE01_BF16_NONFINITE_TENSOR")
    return np.ascontiguousarray(w)


def git_blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\x00" + raw).hexdigest()


def compile_original_quantizer(lib_path: Path):
    raw = C_SOURCE.read_bytes()
    if git_blob_sha(raw) != C_BLOB_SHA1 or C_SOURCE.is_symlink():
        raise ValueError("JE01_ORIGINAL_QUANTIZER_GIT_BLOB_DRIFT")
    cc = os.environ.get("CC", "cc")
    mode = "-dynamiclib" if platform.system() == "Darwin" else "-shared"
    subprocess.run([cc, "-O2", "-std=c11", "-fPIC", mode, str(C_SOURCE),
                    "-o", str(lib_path), "-lm"], check=True, timeout=120)
    lib = ctypes.CDLL(str(lib_path))
    f32 = ctypes.POINTER(ctypes.c_float)
    ptr = ctypes.c_void_p
    lib.gguf_quantize_qNg64.argtypes = [f32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr, f32]
    lib.gguf_quantize_qNg64.restype = None
    lib.gguf_quantize_q8g64.argtypes = [f32, ctypes.c_int, ctypes.c_int, ptr, f32]
    lib.gguf_quantize_q8g64.restype = None
    return lib


def decode_native(packed, scales, shape, bits):
    rows, cols = shape
    if cols % GROUP:
        raise ValueError("JE01_DECODE_INVALID_GROUP")
    n_groups = cols // GROUP
    if scales.size != rows * n_groups:
        raise ValueError("JE01_DECODE_SCALE_SIZE")
    sc = scales.reshape(rows, n_groups, 1)
    if bits == 8:
        if packed.dtype != np.int8 or packed.size != rows * cols:
            raise ValueError("JE01_DECODE_Q8_LAYOUT")
        signed = packed.reshape(rows, n_groups, GROUP).astype(np.float32)
    elif bits in (2, 6):
        if packed.dtype != np.uint8 or packed.size != rows * cols * bits // 8:
            raise ValueError("JE01_DECODE_QNG_LAYOUT")
        planes = np.unpackbits(
            packed.reshape(rows, n_groups, bits, 8), axis=-1, bitorder="little"
        )
        u = np.zeros((rows, n_groups, GROUP), dtype=np.int16)
        for j in range(bits):
            u |= planes[:, :, j, :].astype(np.int16) << j
        signed = (u - (1 << (bits - 1))).astype(np.float32)
    else:
        raise ValueError("JE01_DECODE_TIER_UNSUPPORTED")
    return (signed * sc).reshape(shape)


def quantize_matrix(lib, weight, bits):
    if weight.ndim != 2 or weight.dtype != np.float32 or not weight.flags.c_contiguous:
        raise ValueError("JE01_QUANT_INPUT_NOT_C_F32")
    rows, cols = weight.shape
    if rows < 1 or cols < 1 or cols % GROUP or bits not in TIERS:
        raise ValueError("JE01_QUANT_SHAPE_OR_TIER")
    scales = np.empty((rows, cols // GROUP), dtype=np.float32)
    fp = weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sp = scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if bits == 8:
        packed = np.empty((rows * cols,), dtype=np.int8)
        lib.gguf_quantize_q8g64(fp, rows, cols, packed.ctypes.data, sp)
    else:
        packed = np.empty((rows * cols * bits // 8,), dtype=np.uint8)
        lib.gguf_quantize_qNg64(fp, rows, cols, bits, packed.ctypes.data, sp)
    decoded = decode_native(packed, scales, weight.shape, bits)
    if not np.isfinite(scales).all() or not (scales > 0).all() or not np.isfinite(decoded).all():
        raise ValueError("JE01_QUANT_NONFINITE")
    diff = decoded.astype(np.float64) - weight.astype(np.float64)
    rmse = float(np.sqrt(np.mean(diff * diff, dtype=np.float64)))
    scale = float(np.sqrt(np.mean(weight.astype(np.float64) ** 2)))
    return packed, scales, rmse, rmse / max(scale, 1e-20)


def save_unique(path, arr):
    with path.open("xb") as f:
        f.write(arr.tobytes(order="C"))
        f.flush()
        os.fsync(f.fileno())
    if path.stat().st_size != arr.nbytes:
        raise ValueError("JE01_REAL_ARTIFACT_BYTE_COUNT_DRIFT")
    return {"filename": path.name, "actual_bytes": path.stat().st_size,
            "sha256": file_sha(path)}


def enforce_safe_memory():
    """Fail closed on real macOS pressure before each BF16 expert extraction."""
    if platform.system() != "Darwin":
        raise ValueError("JE01_ACTUAL_BF16_HOST_MUST_BE_MACOS")
    try:
        output = subprocess.check_output(
            ["/usr/bin/memory_pressure", "-Q"], text=True, timeout=8
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("JE01_MEMORY_PRESSURE_SENSOR_UNAVAILABLE") from exc
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", output)
    if not match:
        raise ValueError("JE01_MEMORY_PRESSURE_SENSOR_UNPARSEABLE")
    free = int(match.group(1))
    if free < 35:
        raise ValueError("JE01_MEMORY_PRESSURE_FREE_BELOW_35PCT")
    return free


def make_artifacts(out_dir: Path):
    enforce_safe_memory()
    ledger = read_verified_ledger(LEDGER)
    slots = validate_ledger(ledger)
    source = attestation()
    # Never let the user-supplied path alias the original BF16 model, worker
    # directories, checkout, or pre-existing data. Only D50 scratch is valid.
    dest = out_dir.absolute()
    if (out_dir.is_symlink() or OUTPUT_PARENT not in dest.parents or
            not OUTPUT_PARENT.is_dir() or dest.exists()):
        raise ValueError("JE01_OUTPUT_MUST_BE_NEW_D50_SCRATCH_DIRECTORY")
    # Finish source digest attestation BEFORE creating any scratch files.
    with verify_source(source) as (weightmap, handles):
        # Read tensors from the SAME open descriptors whose full SHA was verified.
        dest.mkdir(mode=0o700)
        lib_path = dest / ("je01_native_quant.dylib" if platform.system() == "Darwin" else "je01_native_quant.so")
        lib = compile_original_quantizer(lib_path)
        results = []
        totals = {str(n): 0 for n in TIERS}
        for slot in slots:
            enforce_safe_memory()
            for role in ("gate_proj", "up_proj", "down_proj"):
                raw = tensor_raw(weightmap, handles, slot["layer"], slot["expert"], role)
                tensor_sha = hashlib.sha256(raw).hexdigest()
                w = bf16_to_float(raw, SHAPES[role])
                for bits in TIERS:
                    packed, scales, rmse, relrmse = quantize_matrix(lib, w, bits)
                    stem = f"L{slot['layer']:02d}_E{slot['expert']:02d}_{role}_n{bits}"
                    packed_rec = save_unique(dest / (stem + ".packed"), packed)
                    scales_rec = save_unique(dest / (stem + ".scales.f32"), scales)
                    actual = packed_rec["actual_bytes"] + scales_rec["actual_bytes"]
                    totals[str(bits)] += actual
                    results.append({"layer": slot["layer"], "expert": slot["expert"],
                                    "role": role, "bits": bits, "source_bf16_sha256": tensor_sha,
                                    "packed": packed_rec, "scales": scales_rec,
                                    "actual_total_bytes": actual,
                                    "roundtrip_RMSE_vs_BF16": rmse,
                                    "roundtrip_relative_RMSE_vs_BF16": relrmse})
        for bits in TIERS:
            exp = sum(projected_slot_bytes(slot, bits) for slot in slots)
            if totals[str(bits)] != exp:
                raise ValueError("JE01_TRUE_SAVED_BYTES_NOT_EQUAL_CODEC_PRESOLVE")
        if len(results) != 24 * 3:
            raise ValueError("JE01_BF16_ROLE_TIER_COUNT_DRIFT")
        manifest = {
            "schema": SCHEMA, "status": "NATIVE_C_QUANTIZED_SELECTED_BF16_ROLE_ARTIFACTS_ONLY",
            "source_original_bf16_identity": SOURCE_MODEL_ID,
            "source_attestation_receipt_sha256": SOURCE_RECEIPT,
            "AF_original_unchanged_8expert_baseline_bytes": 43_253_760,
            "quantizer_C_git_blob": C_BLOB_SHA1,
            "actual_physical_tier_bytes_for_all8": totals,
            "actual_role_tier_payload_count": len(results),
            "results": results,
            "model_wide_mixed_routed_expert_gpu_forward": "NOT_RUN",
            "native_original_AF_equivalence_to_BF16": "NOT_VERIFIED",
            "matched_byte_joint_candidate": "NOT_ASSEMBLED",
            "calibration_gold_token_NLL": "NOT_MEASURED",
            "heldout_or_ood_opened": False,
            "production_model_or_worker_changed": False,
            "promotion_allowed": False
        }
        manifest["receipt_sha256"] = canonical(manifest)
        with (dest / "JE01_BF16_SELECTED8_TIERS_ACTUAL_MANIFEST.json").open("x") as f:
            json.dump(manifest, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
        return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    try:
        doc = make_artifacts(args.out_dir)
        print(json.dumps({k: doc[k] for k in (
            "status", "receipt_sha256", "actual_role_tier_payload_count",
            "actual_physical_tier_bytes_for_all8",
            "model_wide_mixed_routed_expert_gpu_forward")}, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)[:240]}))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
