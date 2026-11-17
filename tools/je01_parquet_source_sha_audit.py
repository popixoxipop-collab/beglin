#!/usr/bin/env python3
"""JE01 native XOX read-only full public Parquet source SHA audit.

Fetches exactly 4 declared files via pinned Hugging Face repository revisions,
streams into SHA256, keeps no source bytes on disk, does not inspect labels or
execute any model. Complements (does not rewrite) the 512-row manifest.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import urllib.parse
import urllib.request

SCHEMA = "beglin-je01-full-source-parquet-sha256-v1"
ROW_MANIFEST_SHA256 = "a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147"
SOURCES = (
    ("wikitext_train_0", "Salesforce/wikitext",
     "b08601e04326c79dfdd32d625aee71d232d685c3",
     "wikitext-103-raw-v1/train-00000-of-00002.parquet", 156987808),
    ("wikitext_train_1", "Salesforce/wikitext",
     "b08601e04326c79dfdd32d625aee71d232d685c3",
     "wikitext-103-raw-v1/train-00001-of-00002.parquet", 157088770),
    ("wikitext_validation", "Salesforce/wikitext",
     "b08601e04326c79dfdd32d625aee71d232d685c3",
     "wikitext-103-raw-v1/validation-00000-of-00001.parquet", 657209),
    ("hellaswag_validation", "Rowan/hellaswag",
     "218ec52e09a7e7462a5400043bb9a69a41d06b76",
     "data/validation-00000-of-00001.parquet", 6315951),
)


def identity(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha_source(item, opener=urllib.request.urlopen):
    role, name, revision, relative_path, expected_length = item
    url = "https://huggingface.co/datasets/" + name + "/resolve/" + revision + "/" + relative_path
    req = urllib.request.Request(url, headers={
        "User-Agent": "BEGLIN-JE01-source-SHA-audit-v1",
        "Accept-Encoding": "identity",
    })
    with opener(req, timeout=90) as response:
        final_host = urllib.parse.urlsplit(response.geturl()).hostname or ""
        if (final_host != "huggingface.co" and
            final_host != "hf.co" and not final_host.endswith(".hf.co")):
            raise ValueError("JE01_UNTRUSTED_DATA_REDIRECT")
        if response.status != 200 or response.headers.get("content-encoding") not in (None, "identity"):
            raise ValueError("JE01_TRANSFER_NOT_RAW_200")
        hlen = response.headers.get("content-length")
        if hlen is not None and int(hlen) != expected_length:
            raise ValueError("JE01_SOURCE_LENGTH_HEADER_DRIFT:" + role)
        h = hashlib.sha256()
        first = b""
        last = b""
        size = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            if not first:
                first = chunk[:4]
            last = (last + chunk)[-4:]
            size += len(chunk)
            if size > expected_length:
                raise ValueError("JE01_PARQUET_OVERSIZE:" + role)
            h.update(chunk)
    if size != expected_length:
        raise ValueError("JE01_PARQUET_TOTAL_SIZE_DRIFT:" + role)
    if first != b"PAR1" or last != b"PAR1":
        raise ValueError("JE01_PARQUET_MAGIC_INVALID:" + role)
    return {"role": role, "dataset": name, "revision": revision,
            "relative_path": relative_path, "byte_size": size,
            "file_sha256": h.hexdigest(), "parquet_magic_verified": True}


def audit():
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(sha_source, SOURCES))
    if len(results) != 4 or len({r["file_sha256"] for r in results}) != 4:
        raise ValueError("JE01_SOURCE_SHARD_IDENTITY_INVALID")
    report = {
        "schema": SCHEMA,
        "input_row_manifest_sha256": ROW_MANIFEST_SHA256,
        "stage": "FOUR_COMPLETE_PUBLIC_PARQUET_SOURCES_SHA256_PINNED",
        "files": results,
        "total_source_bytes": sum(r["byte_size"] for r in results),
        "model_checkpoint_sha256": "NOT_MEASURED",
        "native_moe_inference": "NOT_RUN",
        "production_authorized": False,
        "weights_modified": False,
    }
    report["receipt_sha256"] = identity(report)
    return report


if __name__ == "__main__":
    try:
        print(json.dumps(audit(), sort_keys=True, separators=(",", ":")))
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "stage": "BLOCKED",
                          "error": type(exc).__name__, "reason": str(exc)[:160],
                          "native_inference": False}))
        raise SystemExit(2)
