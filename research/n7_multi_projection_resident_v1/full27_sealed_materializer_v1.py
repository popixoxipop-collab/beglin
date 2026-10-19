#!/usr/bin/env python3
"""Research-only DeepSeek-V2-Lite 27-layer n7/group64 prequant materializer.

Reads the SHA-pinned *original* BF16 source in tensor-sized streaming slices.
Quantizes each one time using BEGLIN's ORIGINAL native qng64_dense.c, then
writes canonical qNg64 n7 planes/FP32-scales to an isolated append-only stage.
A complete 5,181-role manifest is published only when all original-source
roles have been processed and their payload SHA checks succeed.

Safe checkpoint resume verifies every previous role's authority/size/digest.
Incomplete outputs are never accepted as a model or made serving-visible.
Does not modify BF16 source, XOX, production, or the old v10 fallback.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

import numpy as np
from safetensors import safe_open
import torch

from prequant_n7_sealed_v1 import (
    PackReader, SCHEMA, file_sha, role_files, geometry, canonical_role, require,
)

INDEX_SHA = "d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
BOUNDARY_SHA = "274247e626f67579e56256e356f554fd1b1e5e5967df2603ec82bfb902af4963"
CONFIG_SHA = "f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d"
MODEL_CODE_SHA = "7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c"
SHARD_SHAS = {
    "model-00001-of-000004.safetensors": "a314b79a30810b1038f9db08345178e16fb5742aa266c4a3dc05054c59800082",
    "model-00002-of-000004.safetensors": "b0bd088feada0af954614da3a5b179c4d4d5d8bb2482ec83194d101f3fe91796",
    "model-00003-of-000004.safetensors": "d31b192e488f0b3a04e007d842c3c9ba5783d944cb1cc27402fe8b8bb837c490",
    "model-00004-of-000004.safetensors": "304499be0ea06bf8a3937d7a3cc515bc3738dbd288bdd0bbc1b2d3560fce4930",
}
FULL_ROLES = 5181
FULL_TENSORS = 5291
FULL_Q_BYTES = 14328299520
FULL_BF16_BYTES = 31412968448
MAX_RAM_SCRATCH_B = 128 * 1024 * 1024
ROLE_RECEIPT_SCHEMA = "beglin.qng64.n7.source-role-attested/1"
AUTHORITY_SCHEMA = "beglin.deepseek-v2-lite.full27-prequant-progress/1"
HEX64 = re.compile(r"[0-9a-f]{64}", re.ASCII)
U = ctypes.POINTER(ctypes.c_uint8)
F = ctypes.POINTER(ctypes.c_float)


def gate(ok, msg):
    require(bool(ok), "FULL27_MATERIALIZER: " + msg)


def sha_path(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regular(path):
    gate(not path.is_symlink() and path.is_file(), "unexpected regular source/payload file")
    info = path.stat()
    gate(stat.S_ISREG(info.st_mode) and info.st_size >= 0, "source not a regular file")
    return info


def atomic_xb(path, raw):
    gate(not path.exists() and not path.is_symlink(), "attempted overwrite")
    with path.open("xb") as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())
    path.chmod(0o400)


def json_ascii(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def load_boundary(tsv):
    regular(tsv)
    gate(tsv.stat().st_size < 3 * 1024 * 1024 and sha_path(tsv) == BOUNDARY_SHA,
         "canonical immutable source TSV hash")
    lines = tsv.read_text(encoding="ascii").splitlines()
    header = lines[0].split("\t")
    gate(header == [
        "BEGLIN-SOURCE-MANIFEST/1", INDEX_SHA,
        str(FULL_TENSORS), str(FULL_ROLES), "27", "4",
        str(FULL_Q_BYTES), str(FULL_BF16_BYTES)
    ], "canonical manifest count/size header")
    mapping, excluded = {}, set()
    source_shards = set()
    shard_meta = {}
    for line in lines[1:]:
        row = line.split("\t")
        if row[0] == "S":
            gate(len(row) == 6 and row[1] in SHARD_SHAS and
                 row[1] not in source_shards, "untrusted shard metadata")
            source_shards.add(row[1])
            shard_meta[row[1]] = {
                "bytes": int(row[2]), "header_length": int(row[3]),
                "header_sha256": row[4], "tensor_count": int(row[5]),
            }
        else:
            gate(row[0] == "T" and len(row) == 9, "bad tensor boundary row")
            name, shard, dtype, shape, begin, end, bits, size = row[1:]
            gate(dtype == "BF16" and shard in SHARD_SHAS, "source tensor identity")
            dims = tuple(int(x) for x in shape.split(","))
            if bits == "7":
                canonical_role(name)
                gate(len(dims) == 2, "packed tensor rank")
                expected, scale = geometry(dims[0], dims[1])
                gate(size == str(expected + scale), "packed n7 byte format mismatch")
                gate(name not in mapping, "repeated canonical source tensor")
                mapping[name] = dict(
                    role=name, shard=shard, out=dims[0], in_features=dims[1],
                    original_bf16_bytes=2 * math.prod(dims),
                    serialized_bytes=expected+scale)
            else:
                gate(bits == "0" and size == "0", "non-quant tensor policy changed")
                excluded.add(name)
    gate(source_shards == set(SHARD_SHAS) and len(mapping) == FULL_ROLES
         and len(excluded) == FULL_TENSORS-FULL_ROLES
         and sum(v["serialized_bytes"] for v in mapping.values()) == FULL_Q_BYTES,
         "all quant tensor policy population/bytes mismatch")
    for layer in range(27):
        gate(sum(1 for k in mapping if k.startswith("model.layers.%d." % layer)) ==
             (7 if layer == 0 else 199), "per-layer quantization coverage")
    return mapping, excluded, shard_meta


def index_authority(checkpoint, boundary_rows, boundary_excluded):
    gate(checkpoint.is_dir() and not checkpoint.is_symlink(), "source root symlink/bad")
    meta = (("model.safetensors.index.json", INDEX_SHA),
            ("config.json", CONFIG_SHA), ("modeling_deepseek.py", MODEL_CODE_SHA))
    for name, digest in meta:
        file = checkpoint / name
        regular(file)
        gate(file.stat().st_size < 2_000_000 and sha_path(file) == digest,
             "original source metadata sha mismatch: " + name)
    index = json.loads((checkpoint/"model.safetensors.index.json").read_text())["weight_map"]
    gate(type(index) is dict and len(index) == FULL_TENSORS and
         set(index) == set(boundary_rows).union(boundary_excluded) and
         set(index.values()) == set(SHARD_SHAS),
         "original safetensors index coverage drift")
    for name, row in boundary_rows.items():
        gate(index[name] == row["shard"], "source role shard mismatch")
    return index


def validate_full_shard_shas(checkpoint):
    for name, expected in SHARD_SHAS.items():
        file = checkpoint / name
        before = regular(file)
        digest = sha_path(file)
        after = regular(file)
        gate(digest == expected and
             (before.st_ino, before.st_dev, before.st_size,
              before.st_mtime_ns, before.st_ctime_ns) ==
             (after.st_ino, after.st_dev, after.st_size,
              after.st_mtime_ns, after.st_ctime_ns),
             "full original BF16 shard SHA/identity drift: " + name)
        print(json.dumps({"phase": "SOURCE_SHARD_VERIFIED",
                          "shard_number": int(name[6:11]),
                          "sha256": digest, "size": before.st_size}), flush=True)


def c_quantizer(dylib):
    regular(dylib)
    lib = ctypes.CDLL(str(dylib))
    group = lib.qng64_group_bytes
    group.argtypes = (ctypes.c_int,)
    group.restype = ctypes.c_size_t
    gate(group(7) == 56 and group(4) == 0, "native C qNg64 width contract")
    quant = lib.qng64_quantize_f32
    quant.argtypes = (F, ctypes.c_int, ctypes.c_int, ctypes.c_int, U, F)
    quant.restype = ctypes.c_int
    return quant, sha_path(dylib)


def stage_id(row):
    role = row["role"]
    role_files(role)
    return hashlib.sha256(role.encode("ascii")).hexdigest()


def check_digest_file(path, expected_len, digest):
    regular(path)
    gate(path.stat().st_size == expected_len, "previous payload size mismatch")
    gate(sha_path(path) == digest, "previous packed role tampered")
    return True


def existing_receipt(root, row, quant_sha):
    ident = stage_id(row)
    path = root / "progress" / (ident + ".json")
    plane_name, scale_name = role_files(row["role"])
    planes, scales = root/plane_name, root/scale_name
    if not path.exists():
        # Abandoned uncommitted files can appear after abrupt process death.
        # Only these two statically derived paths inside our exclusive stage
        # may be cleared, and only after checking they are regular.
        for candidate in (planes, scales):
            if candidate.exists() or candidate.is_symlink():
                regular(candidate)
                candidate.chmod(0o600)
                candidate.unlink()
        return None
    regular(path)
    obj = json.loads(path.read_text(encoding="ascii"))
    gate(type(obj) is dict and obj.get("schema") == ROLE_RECEIPT_SCHEMA and
         obj.get("role") == row["role"] and
         obj.get("out") == row["out"] and
         obj.get("in_features") == row["in_features"] and
         obj.get("serialized_bytes") == row["serialized_bytes"] and
         obj.get("native_quantizer_sha256") == quant_sha and
         isinstance(obj.get("original_bf16_role_sha256"), str) and
         HEX64.fullmatch(obj["original_bf16_role_sha256"]) and
         isinstance(obj.get("planes_sha256"), str) and
         HEX64.fullmatch(obj["planes_sha256"]) and
         isinstance(obj.get("scales_sha256"), str) and
         HEX64.fullmatch(obj["scales_sha256"]),
         "incomplete or modified role receipt")
    p_len, s_len = geometry(row["out"], row["in_features"])
    check_digest_file(planes, p_len, obj["planes_sha256"])
    check_digest_file(scales, s_len, obj["scales_sha256"])
    return obj


def export_role(root, row, ref_tensor, quant, quant_sha):
    role = row["role"]
    gate(ref_tensor.dtype == torch.bfloat16 and
         tuple(ref_tensor.shape) == (row["out"], row["in_features"]) and
         ref_tensor.is_contiguous(), "actual source BF16 tensor/type/shape mismatch")
    source_sha = hashlib.sha256(
        ref_tensor.view(torch.int16).numpy().tobytes(order="C")).hexdigest()
    w = np.ascontiguousarray(ref_tensor.float().numpy(), dtype=np.float32)
    gate(np.isfinite(w).all(), "nonfinite original BF16 weights")
    p_len, s_len = geometry(row["out"], row["in_features"])
    planes = np.empty(p_len, np.uint8)
    scales = np.empty(s_len//4, np.float32)
    gate(quant(w.ctypes.data_as(F), row["out"], row["in_features"], 7,
               planes.ctypes.data_as(U), scales.ctypes.data_as(F)) == 0,
         "native C original qNg64 quantization failed")
    gate(np.isfinite(scales).all() and np.all(scales > 0), "quantized scales invalid")
    plane_raw, scale_raw = planes.tobytes(), scales.tobytes()
    gate(len(plane_raw)+len(scale_raw) == row["serialized_bytes"],
         "native packed role byte bound drift")
    pn, sn = role_files(role)
    pb, sb = file_sha(plane_raw), file_sha(scale_raw)
    atomic_xb(root/pn, plane_raw)
    atomic_xb(root/sn, scale_raw)
    receipt = {
        "schema": ROLE_RECEIPT_SCHEMA, "role": role,
        "out": row["out"], "in_features": row["in_features"],
        "serialized_bytes": row["serialized_bytes"],
        "original_bf16_role_sha256": source_sha,
        "planes_sha256": pb, "scales_sha256": sb,
        "native_quantizer_sha256": quant_sha,
    }
    atomic_xb(root/"progress"/(stage_id(row)+".json"), json_ascii(receipt))
    return receipt


def make_manifest(root, rows, receipts):
    gate(set(rows) == set(receipts) and len(rows) == FULL_ROLES,
         "cannot seal an incomplete original 5,181 matrix model")
    meta = {
        "schema": SCHEMA, "source_checkpoint_index_sha256": INDEX_SHA,
        "roles": [dict(
            role=k, out=rows[k]["out"], in_features=rows[k]["in_features"],
            planes_sha256=receipts[k]["planes_sha256"],
            scales_sha256=receipts[k]["scales_sha256"],
        ) for k in sorted(rows)],
    }
    raw = json_ascii(meta)
    gate(len(raw) <= 4*1024*1024, "model-wide manifest grew beyond loader bound")
    path = root/"manifest.json"
    atomic_xb(path, raw)
    digest = file_sha(raw)
    reader = PackReader(root, digest, INDEX_SHA)
    gate(set(reader.roles) == set(rows) and reader.total_packed_bytes == FULL_Q_BYTES,
         "sealed n7 model-wide role contract drift")
    return digest


def run(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    checkpoint = Path(args.checkpoint_dir)
    boundary = Path(args.boundary_tsv)
    root = Path(args.output_root)
    quant_dylib = Path(args.native_c_dylib)
    gate(not root.is_symlink(), "output stage must not be symlink")
    mapping, excluded, _shard_info = load_boundary(boundary)
    index_authority(checkpoint, mapping, excluded)
    quant, quant_sha = c_quantizer(quant_dylib)

    if args.inspect_only:
        print(json.dumps({"status": "PASS_FULL27_PREQUANT_PLAN",
                          "qng64_roles":len(mapping), "BF16_exclusions":len(excluded),
                          "packed_eligible_bytes":FULL_Q_BYTES,
                          "total_model_serialized_BF16_exclusions_plus_n7_bytes":
                              FULL_Q_BYTES + FULL_BF16_BYTES -
                              sum(r["original_bf16_bytes"] for r in mapping.values()),
                          "checkpoint_index_sha256":INDEX_SHA,
                          "boundary_manifest_sha256":BOUNDARY_SHA},sort_keys=True))
        return
    gate(args.max_new_roles is None or 1 <= args.max_new_roles <= FULL_ROLES,
         "invalid partial execution bound")

    vfs = os.statvfs(root.parent)
    free = vfs.f_frsize * vfs.f_bavail
    gate(free >= FULL_Q_BYTES + 50*1024**3,
         "D50 research disk headroom insufficient; no serving changes")
    root.mkdir(mode=0o700, exist_ok=True)
    gate(not root.is_symlink() and root.is_dir(), "research stage path symlink")
    root = root.resolve(strict=True)
    lock = root/".single_writer.lock"
    with lock.open("a+b") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress = root/"progress"
        progress.mkdir(mode=0o700, exist_ok=True)
        auth = {
            "schema": AUTHORITY_SCHEMA, "index_sha256": INDEX_SHA,
            "boundary_tsv_sha256":BOUNDARY_SHA, "source_shard_sha256":SHARD_SHAS,
            "native_quantizer_sha256":quant_sha,
            "qng64_role_count":FULL_ROLES,
            "BF16_exclusion_count":FULL_TENSORS-FULL_ROLES,
            "quantized_expected_bytes":FULL_Q_BYTES,
            "exporter_source_sha256":sha_path(Path(__file__)),
        }
        authority_file = root/"RUN_AUTHORITY.json"
        if authority_file.exists():
            regular(authority_file)
            gate(authority_file.read_bytes() == json_ascii(auth),
                 "existing stage incompatible with current pinned code/native source")
        else:
            atomic_xb(authority_file, json_ascii(auth))
        # Even resumes perform full source SHA checks; no trust in timestamps alone.
        validate_full_shard_shas(checkpoint)
        sorted_roles = sorted(mapping.values(), key=lambda r:(r["shard"],r["role"]))
        found = {}
        for row in sorted_roles:
            prior = existing_receipt(root, row, quant_sha)
            if prior is not None:
                found[row["role"]] = prior
        processed = 0
        bytes_completed = sum(mapping[k]["serialized_bytes"] for k in found)
        print(json.dumps({"phase":"RESUME_AUDITED", "completed_roles":len(found),
                          "completed_bytes":bytes_completed,
                          "new_role_limit":args.max_new_roles}),flush=True)
        if len(found) < FULL_ROLES:
            for shard in sorted(SHARD_SHAS):
                todo=[r for r in sorted_roles if r["shard"]==shard and
                      r["role"] not in found]
                if not todo: continue
                with safe_open(str(checkpoint/shard),framework="pt",device="cpu") as loader:
                    for row in todo:
                        tensor=loader.get_tensor(row["role"])
                        found[row["role"]]=export_role(root,row,tensor,quant,quant_sha)
                        del tensor
                        processed+=1
                        bytes_completed+=row["serialized_bytes"]
                        if processed%32==0 or processed==1:
                            print(json.dumps({"phase":"QNG64_N7_PROGRESS",
                                              "new_roles":processed,
                                              "total_roles_completed":len(found),
                                              "total_bytes_completed":bytes_completed,
                                              "elapsed_seconds":round(time.monotonic()-started,2)}),
                                  flush=True)
                        if args.max_new_roles is not None and processed>=args.max_new_roles:
                            break
                if args.max_new_roles is not None and processed>=args.max_new_roles:
                    break
        sealed = len(found)==FULL_ROLES
        digest = make_manifest(root,mapping,found) if sealed else None
        print(json.dumps({
            "status":"PASS_SEALED_FULL27_5181_QNG64_N7" if sealed else
                     "PASS_PARTIAL_PREQUANT_NOT_SEALED",
            "modelwide_prequant_complete":sealed,
            "new_roles_processed":processed,
            "roles_total_completed":len(found),
            "roles_required":FULL_ROLES,
            "completed_qng64_bytes":bytes_completed,
            "expected_qng64_bytes":FULL_Q_BYTES,
            "BF16_exclusions":len(excluded),
            "sealed_manifest_sha256":digest,
            "native_quantizer_sha256":quant_sha,
            "wall_seconds":round(time.monotonic()-started,3),
            "production_promotion_allowed":False,
            "full27_model_decode_verified":False},sort_keys=True),flush=True)


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--boundary-tsv", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--native-c-dylib", required=True)
    ap.add_argument("--inspect-only",action="store_true")
    ap.add_argument("--max-new-roles",type=int,default=None)
    run(ap.parse_args())


if __name__=="__main__":
    cli()
