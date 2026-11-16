#!/usr/bin/env python3
"""JE01 read-only full checkpoint + engine binary inventory.

Never loads weights into a model, executes native code, quantizes, or writes.
Computed identity is a candidate content-addressed reference, not a trusted
model-quality or runtime parity receipt. New checkpoint requires new study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

SCHEMA = "BEGLIN-JE01-full-checkpoint-readonly-attestation/1"


def sha256_file(path):
    hash_value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            hash_value.update(block)
    return hash_value.hexdigest()


def canonical_sha(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _safe(path, *, base, allow_symlinks=True):
    p = pathlib.Path(path)
    if not p.exists() or not p.is_file():
        raise ValueError("JE01_MISSING_OR_NONFILE_" + str(p.name))
    if not allow_symlinks and p.is_symlink():
        raise ValueError("JE01_FORBIDDEN_SYMLINK_" + p.name)
    resolved = p.resolve(strict=True)
    try:
        rel = resolved.relative_to(base.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("JE01_FILE_OUTSIDE_ALLOWED_ROOT_" + p.name) from exc
    if ".." in rel.parts:
        raise ValueError("JE01_TRAVERSAL")
    return resolved


def inspect(*, model_snapshot, allowed_model_root, runtime_binary,
            expected_runtime_sha256, engine_git_commit,
            expected_checkpoint_identity=None):
    snap = pathlib.Path(model_snapshot).resolve(strict=True)
    allowed = pathlib.Path(allowed_model_root).resolve(strict=True)
    if not snap.is_dir():
        raise ValueError("JE01_SNAPSHOT_NOT_DIRECTORY")
    if len(engine_git_commit) != 40 or any(c not in "0123456789abcdef" for c in engine_git_commit):
        raise ValueError("JE01_GIT_COMMIT_UNPINNED")
    if len(expected_runtime_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in expected_runtime_sha256
    ):
        raise ValueError("JE01_RUNTIME_SHA_UNPINNED")
    config_path = _safe(snap / "config.json", base=allowed)
    index_path = _safe(snap / "model.safetensors.index.json", base=allowed)
    tokenizer_path = _safe(snap / "tokenizer.json", base=allowed)
    with open(config_path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(cfg, dict) or cfg.get("model_type") != "deepseek_v2":
        raise ValueError("JE01_NOT_DEEPSEEK_V2")
    routed = cfg.get("n_routed_experts")
    top_k = cfg.get("num_experts_per_tok")
    if type(routed) is not int or routed != 64 or (
        type(top_k) is not int or not 1 <= top_k <= 64
    ):
        raise ValueError("JE01_UNEXPECTED_ROUTED_EXPERT_CONFIG")
    weights = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weights, dict) or not weights:
        raise ValueError("JE01_MODEL_WEIGHT_MAP_EMPTY")
    shard_names = sorted(set(weights.values()))
    if any(not isinstance(name, str) or not name.endswith(".safetensors") or
           name.startswith("/") or "/" in name or "\\" in name or name in (".", "..")
           for name in shard_names):
        raise ValueError("JE01_SHARD_PATH_INVALID")
    shards = []
    for name in shard_names:
        shard = _safe(snap / name, base=allowed)
        shards.append({"name": name, "sha256": sha256_file(shard),
                       "size_bytes": shard.stat().st_size})
    if not shards or any(x["size_bytes"] <= 0 for x in shards):
        raise ValueError("JE01_EMPTY_SHARD")
    runtime = pathlib.Path(runtime_binary).resolve(strict=True)
    if not runtime.is_file():
        raise ValueError("JE01_BINARY_NOT_FILE")
    runtime_sha = sha256_file(runtime)
    if runtime_sha != expected_runtime_sha256:
        raise ValueError("JE01_RUNTIME_BINARY_SHA_DRIFT")
    model_identity = {
        "model_type": cfg["model_type"], "n_routed_experts": routed,
        "num_experts_per_tok": top_k,
        "config_sha256": sha256_file(config_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "index_sha256": sha256_file(index_path),
        "shards": shards,
    }
    checkpoint_identity = canonical_sha(model_identity)
    if expected_checkpoint_identity is not None and (
        checkpoint_identity != expected_checkpoint_identity
    ):
        raise ValueError("JE01_CHECKPOINT_IDENTITY_DRIFT")
    record = {
        "schema": SCHEMA,
        "checkpoint_identity_sha256": checkpoint_identity,
        "model": model_identity,
        "engine_git_commit": engine_git_commit,
        "runtime_binary_sha256": runtime_sha,
        "runtime_parity": "NOT_MEASURED",
        "selected_expert_routing": "NOT_OBSERVED",
        "complete_shard_content_sha": True,
        "status": "HASH_PINNED_ONLY" if expected_checkpoint_identity else "DISCOVERY_ONLY",
        "model_quality_or_optimizer_gain": "NOT_MEASURED",
    }
    record["receipt_sha256"] = canonical_sha(record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-snapshot", required=True)
    p.add_argument("--allowed-model-root", required=True)
    p.add_argument("--runtime-binary", required=True)
    p.add_argument("--expected-runtime-sha256", required=True)
    p.add_argument("--engine-git-commit", required=True)
    p.add_argument("--expected-checkpoint-identity")
    a = p.parse_args()
    try:
        result = inspect(model_snapshot=a.model_snapshot,
                         allowed_model_root=a.allowed_model_root,
                         runtime_binary=a.runtime_binary,
                         expected_runtime_sha256=a.expected_runtime_sha256,
                         engine_git_commit=a.engine_git_commit,
                         expected_checkpoint_identity=a.expected_checkpoint_identity)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema": SCHEMA, "status": "BLOCKED",
                          "reason": str(exc), "no_model_execution": True}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
