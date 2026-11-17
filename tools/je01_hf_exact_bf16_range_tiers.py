#!/usr/bin/env python3
"""JE01 immutable upstream BF16 range-attested 8-Expert native codec artifact trial.

Original BF16 HF revision and all four shard SHA256/lengths must match JE01's
already EOE-attested real model. Every remote HEAD must identify the exact
content-addressed full shard via X-Linked-Etag. Every GET must be HTTP 206
with strict Content-Range. Fail closed if CDN ignores Range. No gold labels,
generation, whole-model inference, production writes, or Bayesian claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
import urllib.parse

import requests
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from je01_selected_bf16_native_artifacts import (
    SOURCE_MODEL_ID, SHAPES, TIERS, C_BLOB_SHA1, attestation,
    bf16_to_float, compile_original_quantizer, quantize_matrix,
    save_unique
)
from je01_native_exact_byte_presolve import (
    canonical, read_verified_ledger, validate_ledger, projected_slot_bytes
)

REVISION = "419ef8b6b7a816343af14bc1a4436d4e33affb5d"
HF_REPO = "deepseek-ai/DeepSeek-V2-Lite"
SCHEMA = "beglin-je01-exact-attested-HF-range-actual-bf16-native-nbit/1"
ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/research/JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json"
ALLOWED_REDIRECT_HOSTS = (
    "cas-bridge.xethub.hf.co", "cdn-lfs.hf.co",
    "cas-server.xethub.hf.co", "huggingface.co"
)
MAX_TENSOR_BYTES = 5_767_168
MAX_HEADER_BYTES = 32_000_000


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def secure_redirect(uri: str):
    p = urllib.parse.urlsplit(uri)
    if p.scheme != "https" or not p.hostname:
        raise ValueError("JE01_REMOTE_REDIRECT_PROTOCOL_UNTRUSTED")
    if p.hostname not in ALLOWED_REDIRECT_HOSTS:
        raise ValueError("JE01_REMOTE_REDIRECT_HOST_UNTRUSTED:" + str(p.hostname))


class Shard:
    def __init__(self, session, spec):
        self.session = session
        self.spec = spec
        self.name = spec["name"]
        self.size = spec["size_bytes"]
        self.sha = spec["sha256"]
        self.redirect = ""
        self.checked_at = 0
        self.base = None
        self.header = None

    def handshake(self):
        url = (f"https://huggingface.co/{HF_REPO}/resolve/{REVISION}/"
               f"{self.name}?download=true")
        res = self.session.head(url, timeout=(20, 60), allow_redirects=False)
        try:
            if res.status_code not in (302, 307):
                raise ValueError("JE01_CDN_LFS_HEAD_NOT_REDIRECT")
            hdr = res.headers
            digest = (hdr.get("X-Linked-Etag") or "").strip('"').lower()
            linked_size = hdr.get("X-Linked-Size")
            if digest != self.sha or linked_size != str(self.size):
                # Public source metadata only; never log the signed Location URL.
                print(json.dumps({
                    "event": "JE01_REMOTE_BF16_FULL_SHARD_IDENTITY_FAILURE",
                    "shard": self.name,
                    "pinned_expected_sha256": self.sha,
                    "remote_x_linked_etag": digest,
                    "full_sha256_matches": digest == self.sha,
                    "pinned_expected_bytes": self.size,
                    "remote_x_linked_size": linked_size,
                    "full_size_matches": linked_size == str(self.size),
                    "remote_revision_commit": hdr.get("X-Repo-Commit"),
                    "response_status": res.status_code
                }, sort_keys=True), flush=True)
                raise ValueError("JE01_CDN_FULL_SHARD_SHA_SIZE_MISMATCH:" + self.name)
            self.redirect = hdr.get("Location") or ""
            secure_redirect(self.redirect)
            self.checked_at = time.monotonic()
        finally:
            res.close()

    def range(self, start: int, length: int) -> bytes:
        if not isinstance(start, int) or not isinstance(length, int):
            raise ValueError("JE01_INVALID_RANGE_TYPE")
        if start < 0 or length < 1 or start + length > self.size:
            raise ValueError("JE01_OUT_OF_BOUNDS_RANGE")
        if not self.redirect or time.monotonic() - self.checked_at > 75:
            self.handshake()
        end = start + length - 1
        response = self.session.get(
            self.redirect,
            headers={"Range": f"bytes={start}-{end}",
                     "Accept-Encoding": "identity"},
            stream=True, allow_redirects=False, timeout=(20, 120)
        )
        try:
            if response.status_code != 206:
                raise ValueError("JE01_RANGE_NOT_RESPECTED:" + str(response.status_code))
            expected_range = f"bytes {start}-{end}/{self.size}"
            if response.headers.get("Content-Range") != expected_range:
                raise ValueError("JE01_RANGE_RESPONSE_OFFSET_OR_SIZE_DRIFT")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("JE01_RANGE_COMPRESSED_RESPONSE_UNSUPPORTED")
            raw = bytearray()
            for chunk in response.iter_content(chunk_size=1<<20):
                if chunk:
                    raw.extend(chunk)
                    if len(raw) > length:
                        raise ValueError("JE01_RANGE_OVERREAD")
            if len(raw) != length:
                raise ValueError("JE01_RANGE_TRUNCATED")
            return bytes(raw)
        finally:
            response.close()

    def header_only(self):
        n = struct.unpack("<Q", self.range(0,8))[0]
        if not 64 <= n <= MAX_HEADER_BYTES:
            raise ValueError("JE01_SHARD_HEADER_LENGTH_BOUNDS")
        data = self.range(8,n)
        header = json.loads(data)
        if not isinstance(header,dict):
            raise ValueError("JE01_SHARD_HEADER_BAD_JSON")
        self.base = 8 + n
        self.header = header
        return header


def obtain_pinned_source(session):
    model = attestation()  # JE01 EOE physical original BF16 SHA receipt
    if model["index_sha256"] != "d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf":
        raise ValueError("JE01_EOE_BF16_INDEX_SHA_NOT_EXPECTED")
    index_url = (f"https://huggingface.co/{HF_REPO}/resolve/{REVISION}/"
                 "model.safetensors.index.json?download=true")
    index_resp = session.get(index_url,timeout=(20,90))
    index_resp.raise_for_status()
    raw = index_resp.content
    if len(raw) > 2_000_000 or hashlib.sha256(raw).hexdigest() != model["index_sha256"]:
        raise ValueError("JE01_HF_INDEX_FULL_SHA_DIFFERS_FROM_EOE")
    index = json.loads(raw)
    if index.get("metadata",{}).get("total_size") != 31_412_968_448:
        raise ValueError("JE01_HF_INDEX_WEIGHT_SIZE_DRIFT")
    shards = {s["name"]:Shard(session,s) for s in model["shards"]}
    for shard in shards.values():
        shard.handshake()
    return model,index["weight_map"],shards


def run(mode, output=None):
    assert mode in ("preflight","extract")
    ledger = read_verified_ledger(LEDGER)
    slots = validate_ledger(ledger)
    if mode == "extract":
        if output is None or output.exists() or output.is_symlink() or not output.parent.is_dir():
            raise ValueError("JE01_OUTPUT_NOT_FRESH")
    session = requests.Session()
    session.headers.update({"User-Agent":"BEGLIN-JE01-range-audit/1"})
    model, weightmap, shards = obtain_pinned_source(session)
    wanted = []
    for slot in slots:
        for role in ("gate_proj","up_proj","down_proj"):
            key = f"model.layers.{slot['layer']}.mlp.experts.{slot['expert']}.{role}.weight"
            shard_name = weightmap.get(key)
            if shard_name not in shards:
                raise ValueError("JE01_SELECTED_TENSOR_INDEX_MISSING:"+key)
            sh = shards[shard_name]
            if sh.header is None:
                sh.header_only()
            metadata = sh.header.get(key)
            size=2*np.prod(SHAPES[role],dtype=np.int64)
            if not isinstance(metadata,dict) or metadata.get("dtype")!="BF16" or metadata.get("shape")!=list(SHAPES[role]):
                raise ValueError("JE01_SELECTED_BF16_TENSOR_DTYPE_SHAPE_DRIFT:"+key)
            offset=metadata.get("data_offsets")
            if not isinstance(offset,list) or len(offset)!=2 or any(type(i)!=int for i in offset):
                raise ValueError("JE01_BF16_TENSOR_BAD_OFFSET")
            if not 0<=offset[0]<offset[1] or offset[1]-offset[0]!=size or size>MAX_TENSOR_BYTES or sh.base+offset[1]>sh.size:
                raise ValueError("JE01_BF16_TENSOR_OFFSET_DRIFT")
            wanted.append((slot,role,key,sh,sh.base+offset[0],int(size)))
    if len(wanted)!=24:
        raise ValueError("JE01_EXPECTED_24_SELECTED_EXPERT_ROLES")
    checked_shards=sorted({rec[3].name for rec in wanted})
    if mode == "preflight":
        # A real range request from one selected BF16 tensor; checks 206/offset
        # without opening full pretrained weights or any heldout labels.
        first=wanted[0]
        snippet=first[3].range(first[4],64)
        return {"schema":SCHEMA,"status":"REMOTE_RANGE_PREFLIGHT_ONLY",
                "source_model":SOURCE_MODEL_ID,"source_index_sha256":model["index_sha256"],
                "matched_shard_full_etags": checked_shards,
                "selected_role_count":24,"first_selected_tensor_64byte_sha256":hashlib.sha256(snippet).hexdigest(),
                "true_bf16_tensor_files_created":0,"original_model_forward":"NOT_RUN","production_modified":False}
    output.mkdir(mode=0o700)
    lib = compile_original_quantizer(output/"je01_native_quant.so")
    records=[]; totals={str(bit):0 for bit in TIERS}
    for index,(slot,role,key,sh,offset,nbytes) in enumerate(wanted):
        raw=sh.range(offset,nbytes)
        tensor_sha=hashlib.sha256(raw).hexdigest()
        w=bf16_to_float(raw,SHAPES[role])
        for bit in TIERS:
            packed,scales,rmse,relrmse=quantize_matrix(lib,w,bit)
            stem=f"L{slot['layer']:02d}_E{slot['expert']:02d}_{role}_n{bit}"
            p=save_unique(output/(stem+".packed"),packed)
            s=save_unique(output/(stem+".scales.f32"),scales)
            cost=p["actual_bytes"]+s["actual_bytes"]
            totals[str(bit)]+=cost
            records.append({"layer":slot["layer"],"expert":slot["expert"],"role":role,"tier":bit,
                            "source_exact_bf16_tensor_sha256":tensor_sha,
                            "source_full_shard_sha256":sh.sha,"source_shard":sh.name,
                            "packed":p,"scale":s,"saved_physical_bytes":cost,
                            "rmse_bf16":rmse,"relative_rmse_bf16":relrmse})
        print(json.dumps({"progress":index+1,"total":24,"layer":slot["layer"],
                          "expert":slot["expert"],"role":role,
                          "BF16_tensor_sha256":tensor_sha}),flush=True)
    for bit in TIERS:
        predicted=sum(projected_slot_bytes(s,bit) for s in slots)
        if totals[str(bit)]!=predicted:
            raise ValueError("JE01_ALL_REAL_BYTES_NOT_MATCHED_CODEC_BUDGET")
    receipt={"schema":SCHEMA,"status":"REAL_ORIGINAL_BF16_SHA_ATTESTED_24_TENSORS_72_NATIVE_TIERS_PACKED",
             "source_original_bf16_checkpoint_sha256":SOURCE_MODEL_ID,
             "source_index_sha256":model["index_sha256"],
             "source_shards_full_etag_sha256_verified":checked_shards,
             "original_baseline_actual_eight_expert_bytes":43_253_760,
             "codec_pinned_original_C_blob":C_BLOB_SHA1,
             "real_role_tier_count":len(records),"real_physical_byte_totals_all8":totals,
             "records":records,"original_native4bit_AF_source_parity":"NOT_VERIFIED",
             "matched_byte_joint_expert_candidate":"NOT_ASSEMBLED",
             "original_full_model_mixed_precision_forward":"NOT_RUN",
             "actual_gold_token_NLL_improvement":"NOT_MEASURED",
             "heldout_or_ood_read":False,"production_modified":False}
    receipt["receipt_sha256"]=canonical_sha(receipt)
    (output/"JE01_HF_RANGE_NATIVE_BF16_72_ARTIFACTS.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n"
    )
    return {key:receipt[key] for key in ("status","receipt_sha256","real_role_tier_count",
                                        "real_physical_byte_totals_all8",
                                        "actual_gold_token_NLL_improvement")}


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--mode",choices=["preflight","extract"],required=True)
    a.add_argument("--output",type=Path)
    opt=a.parse_args()
    try:
        print(json.dumps(run(opt.mode,opt.output),sort_keys=True,allow_nan=False),flush=True)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:350]}),flush=True)
        raise SystemExit(2)
