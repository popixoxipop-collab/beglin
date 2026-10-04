#!/usr/bin/env python3
"""P12 loader runtime planning and source binding verification.

Turns a LoaderContract inside a ModelCapabilityBundle into an explicit,
fail-closed runtime plan. This module does not load weights or mutate production.

Safety properties:
- unsupported loader contracts fail closed
- unverified loader contracts compile only to VALIDATION_REQUIRED
- source files are bound by absolute path, size, and SHA-256
- symlinks are rejected at execution time
- silent dense fallback is forbidden
- compressed/source quantization semantics are preserved in the plan
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import model_capability as mc


class LoaderRuntimeError(RuntimeError):
    pass


_FORMAT_TO_MODE = {
    "GGUF": "GGUF_DIRECT",
    "SAFETENSORS_SINGLE": "SAFETENSORS_SINGLE",
    "SAFETENSORS_SHARDED": "SAFETENSORS_SHARDED",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _stable_hash(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return sha256_json(payload)


def _require_bundle(bundle: Mapping[str, Any]) -> dict:
    value = dict(bundle)
    if value.get("schema") != "beglin-model-capability-bundle-v1":
        raise LoaderRuntimeError("unsupported model capability bundle schema")
    got = str(value.get("bundle_sha256") or "")
    if got != mc.stable_identity_sha256(value):
        raise LoaderRuntimeError("model capability bundle hash mismatch")
    return value


def _require_loader_contract(contract: Mapping[str, Any]) -> dict:
    value = dict(contract)
    if value.get("schema") != "beglin-loader-contract-v1":
        raise LoaderRuntimeError("unsupported loader contract schema")
    got = str(value.get("loader_contract_sha256") or "")
    if got != mc.stable_identity_sha256(value):
        raise LoaderRuntimeError("loader contract self-hash mismatch")
    if value.get("silent_dense_fallback_allowed") is not False:
        raise LoaderRuntimeError("silent dense fallback must remain disabled")
    if value.get("status") == "UNSUPPORTED":
        raise LoaderRuntimeError("loader contract is unsupported")
    return value


def _source_bindings(source: Mapping[str, Any]) -> list[dict]:
    rows = []
    seen = set()
    for raw in source.get("file_hashes") or []:
        path = Path(str(raw.get("path") or ""))
        if not path.is_absolute():
            raise LoaderRuntimeError("loader source path must be absolute")
        key = str(path)
        if key in seen:
            raise LoaderRuntimeError(f"duplicate source file binding: {path}")
        seen.add(key)
        sha = str(raw.get("sha256") or "")
        if len(sha) != 64:
            raise LoaderRuntimeError(f"invalid source SHA for {path}")
        rows.append({
            "name": str(raw.get("name") or path.name),
            "path": key,
            "size_bytes": int(raw.get("size_bytes") or 0),
            "sha256": sha,
        })
    if not rows:
        raise LoaderRuntimeError("model source has no file bindings")
    return sorted(rows, key=lambda r: (r["name"], r["path"]))


def compile_runtime_plan(bundle: Mapping[str, Any]) -> dict:
    value = _require_bundle(bundle)
    source = dict(value.get("source_manifest") or {})
    loader = _require_loader_contract(value.get("loader_contract") or {})

    source_format = str(source.get("source_format") or "")
    if source_format not in _FORMAT_TO_MODE:
        raise LoaderRuntimeError(f"unsupported runtime source format: {source_format}")
    if loader.get("source_format") != source_format:
        raise LoaderRuntimeError("loader/source format mismatch")

    checkpoint = str(value.get("checkpoint_identity_sha256") or "")
    if checkpoint != str(source.get("checkpoint_identity_sha256") or ""):
        raise LoaderRuntimeError("bundle/source checkpoint identity mismatch")

    status = str(loader.get("status") or "")
    execution_allowed = status == "VERIFIED"
    plan_status = "READY" if execution_allowed else "VALIDATION_REQUIRED"
    memory = dict(loader.get("memory_preflight") or {})
    source_files = _source_bindings(source)

    primary = str(source.get("primary_path") or "")
    shard_paths = [str(p) for p in source.get("shard_paths") or []]
    if not primary:
        raise LoaderRuntimeError("model source primary_path is missing")
    if source_format == "SAFETENSORS_SHARDED" and not shard_paths:
        raise LoaderRuntimeError("sharded safetensors runtime requires shard paths")

    plan = {
        "schema": "beglin-loader-runtime-plan-v1",
        "status": plan_status,
        "execution_allowed": execution_allowed,
        "model_capability_bundle_sha256": value["bundle_sha256"],
        "loader_contract_sha256": loader["loader_contract_sha256"],
        "checkpoint_identity_sha256": checkpoint,
        "architecture_id": value["architecture_descriptor"]["architecture_id"],
        "source_format": source_format,
        "execution_mode": _FORMAT_TO_MODE[source_format],
        "primary_path": primary,
        "shard_paths": sorted(shard_paths),
        "source_files": source_files,
        "source_quantization": loader.get("source_quantization"),
        "encountered_formats": list(loader.get("encountered_formats") or []),
        "unsupported_formats": list(loader.get("unsupported_formats") or []),
        "mmap_strategy": loader.get("mmap_strategy"),
        "transcode_strategy": loader.get("transcode_strategy"),
        "cache_strategy": loader.get("cache_strategy"),
        "memory_preflight": memory,
        "runtime_probe_required": bool(memory.get("requires_runtime_probe", True)),
        "silent_dense_fallback_allowed": False,
        "production_write_allowed": False,
        "automatic_live_promotion": False,
    }
    plan["runtime_plan_sha256"] = _stable_hash(plan, "runtime_plan_sha256")
    return plan


def verify_runtime_plan(plan: Mapping[str, Any]) -> dict:
    value = dict(plan)
    if value.get("schema") != "beglin-loader-runtime-plan-v1":
        raise LoaderRuntimeError("unsupported loader runtime plan schema")
    if value.get("silent_dense_fallback_allowed") is not False:
        raise LoaderRuntimeError("silent dense fallback must remain disabled")
    got = str(value.get("runtime_plan_sha256") or "")
    actual = _stable_hash(value, "runtime_plan_sha256")
    if got != actual:
        raise LoaderRuntimeError(
            f"loader runtime plan hash mismatch: expected={got} actual={actual}"
        )
    return value


def verify_source_files(plan: Mapping[str, Any]) -> dict:
    value = verify_runtime_plan(plan)
    checked = []
    total = 0
    for row in value.get("source_files") or []:
        path = Path(str(row["path"]))
        if path.is_symlink():
            raise LoaderRuntimeError(f"loader source may not be a symlink: {path}")
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise LoaderRuntimeError(f"loader source disappeared: {path}") from exc
        if not resolved.is_file():
            raise LoaderRuntimeError(f"loader source is not a regular file: {path}")
        stat = resolved.stat()
        expected_size = int(row["size_bytes"])
        if stat.st_size != expected_size:
            raise LoaderRuntimeError(
                f"loader source size drift: {path} expected={expected_size} actual={stat.st_size}"
            )
        actual_sha = mc.sha256_file(resolved)
        if actual_sha != row["sha256"]:
            raise LoaderRuntimeError(
                f"loader source SHA drift: {path} expected={row['sha256']} actual={actual_sha}"
            )
        total += stat.st_size
        checked.append({
            "path": str(resolved),
            "size_bytes": stat.st_size,
            "sha256": actual_sha,
        })

    result = {
        "schema": "beglin-loader-runtime-source-verification-v1",
        "status": "SOURCE_FILES_VERIFIED",
        "runtime_plan_sha256": value["runtime_plan_sha256"],
        "model_capability_bundle_sha256": value["model_capability_bundle_sha256"],
        "loader_contract_sha256": value["loader_contract_sha256"],
        "checkpoint_identity_sha256": value["checkpoint_identity_sha256"],
        "file_count": len(checked),
        "total_bytes": total,
        "files": checked,
        "execution_allowed": bool(value["execution_allowed"]),
        "runtime_probe_required": bool(value["runtime_probe_required"]),
        "production_write_allowed": False,
    }
    result["result_sha256"] = sha256_json(result)
    return result


def require_executable_plan(plan: Mapping[str, Any]) -> dict:
    value = verify_runtime_plan(plan)
    if value.get("status") != "READY" or value.get("execution_allowed") is not True:
        raise LoaderRuntimeError("loader runtime plan requires validation before execution")
    if value.get("unsupported_formats"):
        raise LoaderRuntimeError("loader runtime plan contains unsupported formats")
    return value


def plan_from_bundle(bundle: Mapping[str, Any]) -> dict:
    return compile_runtime_plan(bundle)
