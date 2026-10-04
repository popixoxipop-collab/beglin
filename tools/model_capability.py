#!/usr/bin/env python3
"""P12 canonical model/capability contracts.

This module is deliberately pure and side-effect free.  It does not load model
weights or mutate a runtime.  It defines deterministic identities used by the
source inspector, architecture adapters, backend planners and P8-P11 gates.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping

HEX64 = re.compile(r"^[0-9a-f]{64}$")

KNOWN_ARCHITECTURES = {
    "qwen2": {"family": "qwen2", "dense_or_moe": "DENSE", "attention_kind": "GQA"},
    "llama": {"family": "llama", "dense_or_moe": "DENSE", "attention_kind": "GQA"},
    "deepseek_v2": {"family": "deepseek_v2", "dense_or_moe": "MOE", "attention_kind": "MLA"},
    "qwen3_moe": {"family": "qwen3", "dense_or_moe": "MOE", "attention_kind": "GQA"},
    "qwen3moe": {"family": "qwen3", "dense_or_moe": "MOE", "attention_kind": "GQA"},
    "olmoe": {"family": "olmoe", "dense_or_moe": "MOE", "attention_kind": "GQA"},
    "gpt-oss": {"family": "gpt-oss", "dense_or_moe": "MOE", "attention_kind": "HYBRID"},
}

TOKENIZER_STATUSES = {
    "IN_ENGINE_VERIFIED", "EXTERNAL_VERIFIED", "IMPLEMENTED_UNVERIFIED", "UNSUPPORTED"
}
LOADER_STATUSES = {"VERIFIED", "IMPLEMENTED_UNVERIFIED", "PARTIAL", "UNSUPPORTED"}
CAPABILITY_LEVELS = {"VERIFIED", "IMPLEMENTED_UNVERIFIED", "RESTART_REQUIRED", "UNSUPPORTED"}
MUTATION_MODES = {"HOT_REBIND_SINGLE", "HOT_REBIND_MULTI", "RESTART_REQUIRED", "IMMUTABLE"}
ELIGIBILITY = {"FULL", "PARTIAL", "DENIED"}
MAPPING_STATUSES = {"MAPPED", "IGNORE", "UNSUPPORTED"}


class ModelCapabilityError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def require_sha(name: str, value: Any) -> str:
    value = str(value or "").lower()
    if not HEX64.fullmatch(value):
        raise ModelCapabilityError(f"{name} must be 64 lowercase hex chars")
    return value


def _stable_without(value: Mapping[str, Any], *fields: str) -> dict:
    out = dict(value)
    for field in fields:
        out.pop(field, None)
    return out


def build_model_source_manifest(
    *,
    model_id: str,
    model_revision: str,
    source_format: str,
    files: Iterable[Mapping[str, Any]],
    root_path: str | None = None,
    discovered_at: str | None = None,
) -> dict:
    if source_format not in {"GGUF", "SAFETENSORS_SINGLE", "SAFETENSORS_SHARDED", "LEGACY_BEG_LIN"}:
        raise ModelCapabilityError(f"unsupported source_format={source_format!r}")
    normalized = []
    seen = set()
    for raw in files:
        logical = str(raw["logical_path"]).replace("\\", "/")
        if not logical or logical.startswith("/") or ".." in logical.split("/"):
            raise ModelCapabilityError(f"logical_path must be relative and normalized: {logical!r}")
        if logical in seen:
            raise ModelCapabilityError(f"duplicate source logical_path: {logical}")
        seen.add(logical)
        normalized.append({
            "logical_path": logical,
            "sha256": require_sha("source file sha256", raw["sha256"]),
            "size_bytes": int(raw["size_bytes"]),
            "kind": str(raw["kind"]),
        })
    if not normalized:
        raise ModelCapabilityError("source manifest requires at least one file")
    normalized.sort(key=lambda row: (row["logical_path"], row["sha256"]))
    identity = {
        "schema": "beglin-model-source-identity-v1",
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        "source_format": source_format,
        "files": normalized,
    }
    manifest = {
        "schema": "beglin-model-source-v1",
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        "source_format": source_format,
        "root_path": root_path,
        "files": normalized,
        "checkpoint_identity_sha256": sha256_json(identity),
        "discovered_at": discovered_at,
        "immutable": True,
    }
    return manifest


def identify_architecture(
    source_name: str | None,
    *,
    facts: Mapping[str, Any] | None = None,
    adapter_version: str = "p12-v1",
) -> dict:
    raw = str(source_name or "").strip().lower().replace("-", "_")
    aliases = {
        "deepseek_v2": "deepseek_v2",
        "deepseekv2": "deepseek_v2",
        "qwen3moe": "qwen3_moe",
        "qwen3_moe": "qwen3_moe",
        "qwen2": "qwen2",
        "llama": "llama",
        "olmoe": "olmoe",
        "gpt_oss": "gpt-oss",
        "gpt-oss": "gpt-oss",
    }
    architecture_id = aliases.get(raw, raw or "unknown")
    profile = KNOWN_ARCHITECTURES.get(architecture_id)
    payload = {
        "schema": "beglin-architecture-descriptor-v1",
        "status": "KNOWN" if profile else "UNKNOWN_ARCHITECTURE",
        "architecture_id": architecture_id,
        "architecture_family": profile["family"] if profile else "unknown",
        "dense_or_moe": profile["dense_or_moe"] if profile else "UNKNOWN",
        "attention_kind": profile["attention_kind"] if profile else "UNKNOWN",
        "architecture_adapter_id": f"beglin-{architecture_id}" if profile else "none",
        "architecture_adapter_version": str(adapter_version),
        "inference_allowed": bool(profile),
        "facts": dict(facts or {}),
    }
    payload["descriptor_sha256"] = sha256_json(payload)
    return payload


def canonical_target_key(
    model_id: str,
    *,
    role: str,
    layer: int | None = None,
    expert_id: int | None = None,
) -> str:
    parts = [str(model_id)]
    if layer is not None:
        if int(layer) < 0:
            raise ModelCapabilityError("layer must be non-negative")
        parts.append(f"L{int(layer)}")
    parts.append(str(role).lower())
    if expert_id is not None:
        if int(expert_id) < 0:
            raise ModelCapabilityError("expert_id must be non-negative")
        parts.append(f"E{int(expert_id)}")
    return "/".join(parts)


def build_tensor_role_graph(*, model_id: str, nodes: Iterable[Mapping[str, Any]]) -> dict:
    out = []
    targets = set()
    sources = set()
    unclaimed = 0
    for raw in nodes:
        status = str(raw.get("mapping_status", "MAPPED")).upper()
        if status not in MAPPING_STATUSES:
            raise ModelCapabilityError(f"unsupported mapping_status={status!r}")
        source = str(raw["source_tensor_name"])
        if source in sources:
            raise ModelCapabilityError(f"duplicate source tensor: {source}")
        sources.add(source)
        role = str(raw["role"]).upper()
        layer = None if raw.get("layer") is None else int(raw["layer"])
        expert_id = None if raw.get("expert_id") is None else int(raw["expert_id"])
        key = str(raw.get("canonical_target_key") or canonical_target_key(
            model_id, role=role, layer=layer, expert_id=expert_id
        ))
        if key in targets:
            raise ModelCapabilityError(f"duplicate canonical target: {key}")
        targets.add(key)
        row = {
            "canonical_target_key": key,
            "source_tensor_name": source,
            "layer": layer,
            "role": role,
            "expert_id": expert_id,
            "shape": [int(v) for v in raw.get("shape", [])],
            "dtype": raw.get("dtype"),
            "source_quant_format": raw.get("source_quant_format"),
            "mapping_status": status,
        }
        if status == "UNSUPPORTED":
            unclaimed += 1
        out.append(row)
    out.sort(key=lambda row: row["canonical_target_key"])
    graph = {
        "schema": "beglin-tensor-role-graph-v1",
        "model_id": str(model_id),
        "nodes": out,
        "unclaimed_tensor_count": unclaimed,
    }
    graph["graph_sha256"] = sha256_json(graph)
    return graph


def build_operator_graph(
    *,
    model_id: str,
    operators: Iterable[Mapping[str, Any]],
    unsupported_primitives: Iterable[str] = (),
) -> dict:
    rows = [dict(row) for row in operators]
    ids = [str(row.get("operator_id", "")) for row in rows]
    if any(not value for value in ids):
        raise ModelCapabilityError("every operator requires operator_id")
    if len(ids) != len(set(ids)):
        raise ModelCapabilityError("duplicate operator_id")
    rows.sort(key=lambda row: str(row["operator_id"]))
    graph = {
        "schema": "beglin-operator-graph-v1",
        "model_id": str(model_id),
        "operators": rows,
        "unsupported_primitives": sorted({str(v) for v in unsupported_primitives}),
    }
    graph["graph_sha256"] = sha256_json(graph)
    return graph


def build_model_skeleton(
    *,
    model_source_sha256: str,
    architecture_descriptor: Mapping[str, Any],
    layer_skeletons: Iterable[Mapping[str, Any]],
    global_tensors: Iterable[Mapping[str, Any]] = (),
    operator_graph_ref: str | None = None,
    tensor_role_graph_ref: str | None = None,
    tokenizer_contract_ref: str | None = None,
    loader_contract_ref: str | None = None,
    required_primitives: Iterable[str] = (),
    optional_primitives: Iterable[str] = (),
    unsupported_primitives: Iterable[str] = (),
) -> dict:
    desc_sha = require_sha("architecture descriptor", architecture_descriptor["descriptor_sha256"])
    require_sha("model source", model_source_sha256)
    layers = [dict(row) for row in layer_skeletons]
    layer_ids = [int(row["layer_index"]) for row in layers]
    if len(layer_ids) != len(set(layer_ids)):
        raise ModelCapabilityError("duplicate layer_index")
    layers.sort(key=lambda row: int(row["layer_index"]))
    skeleton = {
        "schema": "beglin-model-skeleton-v1",
        "model_source_sha256": model_source_sha256,
        "architecture_descriptor_sha256": desc_sha,
        "architecture": str(architecture_descriptor["architecture_id"]),
        "layer_count": len(layers),
        "layer_skeletons": layers,
        "global_tensors": [dict(row) for row in global_tensors],
        "operator_graph_ref": operator_graph_ref,
        "tensor_role_graph_ref": tensor_role_graph_ref,
        "tokenizer_contract_ref": tokenizer_contract_ref,
        "loader_contract_ref": loader_contract_ref,
        "required_primitives": sorted({str(v) for v in required_primitives}),
        "optional_primitives": sorted({str(v) for v in optional_primitives}),
        "unsupported_primitives": sorted({str(v) for v in unsupported_primitives}),
    }
    skeleton["skeleton_sha256"] = sha256_json(skeleton)
    return skeleton


def build_tokenizer_contract(
    *, tokenizer_id: str, tokenizer_family: str, status: str,
    vocab_size: int | None = None, evidence_refs: Iterable[Mapping[str, Any]] = (),
) -> dict:
    if status not in TOKENIZER_STATUSES:
        raise ModelCapabilityError(f"invalid tokenizer status={status}")
    payload = {
        "schema": "beglin-tokenizer-contract-v1",
        "tokenizer_id": str(tokenizer_id),
        "tokenizer_family": str(tokenizer_family),
        "vocab_size": None if vocab_size is None else int(vocab_size),
        "status": status,
        "evidence_refs": normalize_evidence_refs(evidence_refs),
    }
    payload["contract_sha256"] = sha256_json(payload)
    return payload


def build_loader_contract(
    *, source_format: str, status: str,
    supported_formats: Iterable[str] = (), unsupported_formats: Iterable[str] = (),
    evidence_refs: Iterable[Mapping[str, Any]] = (),
) -> dict:
    if status not in LOADER_STATUSES:
        raise ModelCapabilityError(f"invalid loader status={status}")
    payload = {
        "schema": "beglin-loader-contract-v1",
        "source_format": str(source_format),
        "status": status,
        "supported_formats": sorted({str(v) for v in supported_formats}),
        "unsupported_formats": sorted({str(v) for v in unsupported_formats}),
        "evidence_refs": normalize_evidence_refs(evidence_refs),
    }
    payload["contract_sha256"] = sha256_json(payload)
    return payload


def normalize_evidence_refs(rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    out = []
    for raw in rows:
        status = str(raw.get("status", "VERIFIED")).upper()
        if status not in {"VERIFIED", "UNVERIFIED"}:
            raise ModelCapabilityError("evidence status must be VERIFIED or UNVERIFIED")
        sha = require_sha("evidence sha256", raw["sha256"])
        out.append({
            "kind": str(raw["kind"]),
            "sha256": sha,
            "status": status,
            "ref": str(raw.get("ref", "")),
        })
    return sorted(out, key=lambda row: (row["kind"], row["sha256"], row["ref"]))


def validate_capability_cell(cell: Mapping[str, Any]) -> dict:
    level = str(cell["inference_status"])
    mutation = str(cell["mutation_mode"])
    if level not in CAPABILITY_LEVELS:
        raise ModelCapabilityError(f"invalid capability level={level}")
    if mutation not in MUTATION_MODES:
        raise ModelCapabilityError(f"invalid mutation_mode={mutation}")
    evidence = normalize_evidence_refs(cell.get("evidence_refs", []))
    if level == "VERIFIED" and not any(row["status"] == "VERIFIED" for row in evidence):
        raise ModelCapabilityError("VERIFIED capability requires VERIFIED evidence")
    return {
        "target_key": str(cell["target_key"]),
        "backend": str(cell["backend"]),
        "inference_status": level,
        "mutation_mode": mutation,
        "supported_n": sorted({int(v) for v in cell.get("supported_n", [])}),
        "quant_formats": sorted({str(v) for v in cell.get("quant_formats", [])}),
        "evidence_refs": evidence,
        "reason_code": cell.get("reason_code"),
    }


def build_backend_capability_matrix(*, model_id: str, cells: Iterable[Mapping[str, Any]]) -> dict:
    rows = [validate_capability_cell(row) for row in cells]
    keys = [(row["target_key"], row["backend"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ModelCapabilityError("duplicate backend capability cell")
    rows.sort(key=lambda row: (row["target_key"], row["backend"]))
    matrix = {"schema": "beglin-backend-capability-v1", "model_id": str(model_id), "cells": rows}
    matrix["matrix_sha256"] = sha256_json(matrix)
    return matrix


def derive_pipeline_eligibility(
    *,
    architecture_status: str,
    tokenizer_status: str,
    loader_status: str,
    backend_cells: Iterable[Mapping[str, Any]],
    unsupported_targets: Iterable[str] = (),
) -> tuple[str, list[str]]:
    reasons = []
    if architecture_status != "KNOWN":
        reasons.append("architecture_not_verified")
    if tokenizer_status == "UNSUPPORTED":
        reasons.append("tokenizer_unsupported")
    if loader_status == "UNSUPPORTED":
        reasons.append("loader_unsupported")
    cells = list(backend_cells)
    verified = [c for c in cells if c.get("inference_status") == "VERIFIED"]
    if not verified:
        reasons.append("no_verified_backend_target")
    if reasons:
        return "DENIED", sorted(reasons)
    partial = (
        tokenizer_status != "IN_ENGINE_VERIFIED"
        or loader_status != "VERIFIED"
        or bool(list(unsupported_targets))
        or any(c.get("inference_status") != "VERIFIED" for c in cells)
    )
    return ("PARTIAL" if partial else "FULL"), ([] if not partial else ["capability_subset_only"])


def build_model_capability_bundle(
    *,
    model_id: str,
    checkpoint_identity: str,
    skeleton_sha256: str,
    architecture_status: str,
    tokenizer_status: str,
    loader_status: str,
    backend_matrix: Iterable[Mapping[str, Any]],
    quant_matrix: Iterable[Mapping[str, Any]] = (),
    mutation_matrix: Iterable[Mapping[str, Any]] = (),
    unsupported_targets: Iterable[str] = (),
    evidence_refs: Iterable[Mapping[str, Any]] = (),
) -> dict:
    require_sha("checkpoint identity", checkpoint_identity)
    require_sha("skeleton", skeleton_sha256)
    backend_rows = [validate_capability_cell(row) for row in backend_matrix]
    unsupported = sorted({str(v) for v in unsupported_targets})
    eligibility, reasons = derive_pipeline_eligibility(
        architecture_status=architecture_status,
        tokenizer_status=tokenizer_status,
        loader_status=loader_status,
        backend_cells=backend_rows,
        unsupported_targets=unsupported,
    )
    hot = sorted({
        row["target_key"] for row in backend_rows
        if row["mutation_mode"] in {"HOT_REBIND_SINGLE", "HOT_REBIND_MULTI"}
        and row["inference_status"] == "VERIFIED"
    })
    hot_set = set(hot)
    restart_only = sorted({
        row["target_key"] for row in backend_rows
        if row["mutation_mode"] == "RESTART_REQUIRED"
        and row["target_key"] not in hot_set
    })
    search = sorted({
        row["target_key"] for row in backend_rows
        if row["inference_status"] == "VERIFIED" and row["supported_n"]
    })
    bundle = {
        "schema": "beglin-model-capability-bundle-v1",
        "model_id": str(model_id),
        "checkpoint_identity": checkpoint_identity,
        "skeleton_sha256": skeleton_sha256,
        "architecture_status": architecture_status,
        "tokenizer_status": tokenizer_status,
        "loader_status": loader_status,
        "backend_matrix": sorted(backend_rows, key=lambda r: (r["target_key"], r["backend"])),
        "quant_matrix": [dict(row) for row in quant_matrix],
        "mutation_matrix": [dict(row) for row in mutation_matrix],
        "unsupported_targets": unsupported,
        "restart_only_targets": restart_only,
        "hot_rebind_targets": hot,
        "precision_search_targets": search,
        "p8_p11_eligibility": eligibility,
        "eligibility_reasons": reasons,
        "evidence_refs": normalize_evidence_refs(evidence_refs),
    }
    bundle["bundle_sha256"] = sha256_json(bundle)
    return bundle


def verify_identity(value: Mapping[str, Any], hash_field: str, *, volatile_fields: Iterable[str] = ()) -> None:
    expected = require_sha(hash_field, value[hash_field])
    payload = _stable_without(value, hash_field, *volatile_fields)
    actual = sha256_json(payload)
    if actual != expected:
        raise ModelCapabilityError(f"{hash_field} mismatch: expected={expected} actual={actual}")
