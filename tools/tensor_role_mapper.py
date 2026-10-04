#!/usr/bin/env python3
"""P12 safetensors tensor inventory and semantic role mapper.

The inspector reads only safetensors headers. Weight payloads are never
materialized. Unknown tensor names remain UNSUPPORTED rather than being guessed.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import struct
from typing import Any

import model_capability as mc


class TensorRoleMappingError(RuntimeError):
    pass


_MAX_HEADER_BYTES = 64 * 1024 * 1024


def read_safetensors_header(path: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    with p.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise TensorRoleMappingError(f"invalid safetensors header length: {p}")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
            raise TensorRoleMappingError(
                f"safetensors header size out of range: {header_len}"
            )
        raw = handle.read(header_len)
        if len(raw) != header_len:
            raise TensorRoleMappingError(f"truncated safetensors header: {p}")
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise TensorRoleMappingError(f"invalid safetensors JSON header: {p}") from exc
    if not isinstance(obj, dict):
        raise TensorRoleMappingError("safetensors header root must be an object")

    out: dict[str, dict[str, Any]] = {}
    for name, meta in obj.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(meta, dict):
            raise TensorRoleMappingError("invalid safetensors tensor entry")
        shape = meta.get("shape")
        dtype = meta.get("dtype")
        offsets = meta.get("data_offsets")
        if not isinstance(shape, list) or not all(isinstance(v, int) and v >= 0 for v in shape):
            raise TensorRoleMappingError(f"invalid tensor shape for {name}")
        if not isinstance(dtype, str) or not dtype:
            raise TensorRoleMappingError(f"invalid tensor dtype for {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(v, int) and v >= 0 for v in offsets)
            or offsets[1] < offsets[0]
        ):
            raise TensorRoleMappingError(f"invalid data_offsets for {name}")
        out[name] = {
            "shape": [int(v) for v in shape],
            "dtype": dtype,
            "data_offsets": [int(offsets[0]), int(offsets[1])],
        }
    return out


def inventory_safetensors(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if source.is_file():
        if source.suffix != ".safetensors":
            raise TensorRoleMappingError("tensor inventory file must be .safetensors")
        return read_safetensors_header(source)

    if not source.is_dir():
        raise TensorRoleMappingError(f"model path does not exist: {source}")

    index = source / "model.safetensors.index.json"
    if index.is_file():
        try:
            idx = json.loads(index.read_text())
            weight_map = idx["weight_map"]
        except Exception as exc:
            raise TensorRoleMappingError("invalid safetensors index") from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise TensorRoleMappingError("safetensors weight_map must be non-empty")

        shard_headers: dict[str, dict[str, dict[str, Any]]] = {}
        for shard in sorted(set(str(v) for v in weight_map.values())):
            if "/" in shard or "\\" in shard or shard.startswith("."):
                raise TensorRoleMappingError(f"unsafe shard name: {shard!r}")
            shard_headers[shard] = read_safetensors_header(source / shard)

        out: dict[str, dict[str, Any]] = {}
        for tensor_name, shard_name_raw in weight_map.items():
            tensor_name = str(tensor_name)
            shard_name = str(shard_name_raw)
            header = shard_headers.get(shard_name)
            if header is None or tensor_name not in header:
                raise TensorRoleMappingError(
                    f"index/header mismatch for tensor {tensor_name!r}"
                )
            if tensor_name in out:
                raise TensorRoleMappingError(f"duplicate tensor in index: {tensor_name}")
            out[tensor_name] = dict(header[tensor_name])
        return out

    shards = sorted(source.glob("*.safetensors"))
    if len(shards) != 1:
        raise TensorRoleMappingError(
            "tensor inventory requires one safetensors file or an index"
        )
    return read_safetensors_header(shards[0])


_GLOBAL_PATTERNS = [
    (re.compile(r"^(?:model\.)?embed_tokens\.(weight|bias)$"), "EMBEDDING"),
    (re.compile(r"^(?:model\.)?norm\.(weight|bias)$"), "FINAL_NORM"),
    (re.compile(r"^lm_head\.(weight|bias)$"), "LM_HEAD"),
]

_LAYER_PATTERNS = [
    (re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.(weight|bias)$"), "ATTN_NORM"),
    (re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.(weight|bias)$"), "FFN_NORM"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$"), "Q_PROJ"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$"), "K_PROJ"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$"), "V_PROJ"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$"), "O_PROJ"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_norm\.(weight|bias)$"), "Q_NORM"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_norm\.(weight|bias)$"), "K_NORM"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_a_proj\.(weight|bias)$"), "Q_A_PROJ"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_b_proj\.(weight|bias)$"), "Q_B_PROJ"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.(weight|bias)$"), "KV_A_PROJ"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.kv_b_proj\.(weight|bias)$"), "KV_B_PROJ"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_a_layernorm\.(weight|bias)$"), "Q_A_NORM"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.kv_a_layernorm\.(weight|bias)$"), "KV_A_NORM"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$"), "DENSE_GATE"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$"), "DENSE_UP"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$"), "DENSE_DOWN"),
    (re.compile(r"^model\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.gate\.(weight|bias)$"), "ROUTER"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_expert_gate\.(weight|bias)$"), "SHARED_EXPERT_ROUTER"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_expert\.gate_proj\.(weight|bias)$"), "SHARED_GATE"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_expert\.up_proj\.(weight|bias)$"), "SHARED_UP"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_expert\.down_proj\.(weight|bias)$"), "SHARED_DOWN"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"), "SHARED_GATE"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.up_proj\.(weight|bias)$"), "SHARED_UP"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.(weight|bias)$"), "SHARED_DOWN"),
]

_EXPERT_PATTERNS = [
    (
        re.compile(
            r"^model\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts\.(\d+)\.gate_proj\.(weight|bias)$"
        ),
        "EXPERT_GATE",
    ),
    (
        re.compile(
            r"^model\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts\.(\d+)\.up_proj\.(weight|bias)$"
        ),
        "EXPERT_UP",
    ),
    (
        re.compile(
            r"^model\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts\.(\d+)\.down_proj\.(weight|bias)$"
        ),
        "EXPERT_DOWN",
    ),
]

_IGNORE_PATTERNS = [
    re.compile(r"^model\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq$"),
]


def _with_parameter_kind(role: str, parameter: str) -> str:
    return role if parameter == "weight" else f"{role}_BIAS"


def map_tensor_name(architecture_id: str, name: str) -> dict[str, Any]:
    """Map one tensor name. Unknown names are explicit UNSUPPORTED."""
    architecture_id = str(architecture_id)
    for pattern in _IGNORE_PATTERNS:
        if pattern.fullmatch(name):
            return {"role": "NON_PARAMETER_METADATA", "mapping_status": "IGNORE"}

    for pattern, role in _GLOBAL_PATTERNS:
        m = pattern.fullmatch(name)
        if m:
            return {
                "role": _with_parameter_kind(role, m.group(1)),
                "mapping_status": "MAPPED",
            }

    for pattern, role in _EXPERT_PATTERNS:
        m = pattern.fullmatch(name)
        if m:
            return {
                "layer": int(m.group(1)),
                "expert_id": int(m.group(2)),
                "role": _with_parameter_kind(role, m.group(3)),
                "mapping_status": "MAPPED",
            }

    for pattern, role in _LAYER_PATTERNS:
        m = pattern.fullmatch(name)
        if m:
            return {
                "layer": int(m.group(1)),
                "role": _with_parameter_kind(role, m.group(2)),
                "mapping_status": "MAPPED",
            }

    # Architecture-specific fail-closed marker is retained in the role so the
    # diagnostic is useful while still counting as unsupported.
    return {
        "role": f"UNMAPPED_{architecture_id.upper().replace('-', '_')}",
        "mapping_status": "UNSUPPORTED",
    }


def build_tensor_role_graph_from_safetensors(
    *,
    model_id: str,
    architecture_id: str,
    path: str | Path,
) -> dict:
    inventory = inventory_safetensors(path)
    nodes = []
    for tensor_name, meta in inventory.items():
        mapped = map_tensor_name(architecture_id, tensor_name)
        nodes.append({
            "source_tensor_name": tensor_name,
            "role": mapped["role"],
            "layer": mapped.get("layer"),
            "expert_id": mapped.get("expert_id"),
            "shape": meta["shape"],
            "dtype": meta["dtype"],
            "source_quant_format": meta["dtype"],
            "mapping_status": mapped["mapping_status"],
        })
    return mc.build_tensor_role_graph(model_id=model_id, nodes=nodes)
