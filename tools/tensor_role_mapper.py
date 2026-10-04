#!/usr/bin/env python3
"""P12 safetensors tensor inventory and semantic role mapper.

The inspector reads only safetensors headers. Weight payloads are never
materialized. Unknown architectures and unknown tensor names remain UNSUPPORTED
rather than being guessed from similar-looking names.
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
_KNOWN_SEMANTIC_ARCHITECTURES = {
    "qwen2",
    "llama",
    "deepseek_v2",
    "qwen3_moe",
    "olmoe",
    "gpt-oss",
}


def read_safetensors_header(path: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    size = p.stat().st_size
    with p.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise TensorRoleMappingError(f"invalid safetensors header length: {p}")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
            raise TensorRoleMappingError(
                f"safetensors header size out of range: {header_len}"
            )
        if 8 + header_len > size:
            raise TensorRoleMappingError(
                f"truncated safetensors header: {p}"
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

    data_bytes = size - 8 - header_len
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
        if offsets[1] > data_bytes:
            raise TensorRoleMappingError(
                f"data_offsets exceed file for {name}"
            )
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

_GGUF_GLOBAL_PATTERNS = [
    (re.compile(r"^token_embd\.(weight|bias)$"), "EMBEDDING"),
    (re.compile(r"^output_norm\.(weight|bias)$"), "FINAL_NORM"),
    (re.compile(r"^output\.(weight|bias)$"), "LM_HEAD"),
]

_GGUF_LAYER_PATTERNS = [
    (re.compile(r"^blk\.(\d+)\.attn_norm\.(weight|bias)$"), "ATTN_NORM"),
    (re.compile(r"^blk\.(\d+)\.ffn_norm\.(weight|bias)$"), "FFN_NORM"),
    (re.compile(r"^blk\.(\d+)\.attn_q\.(weight|bias)$"), "Q_PROJ"),
    (re.compile(r"^blk\.(\d+)\.attn_k\.(weight|bias)$"), "K_PROJ"),
    (re.compile(r"^blk\.(\d+)\.attn_v\.(weight|bias)$"), "V_PROJ"),
    (re.compile(r"^blk\.(\d+)\.attn_output\.(weight|bias)$"), "O_PROJ"),
    (re.compile(r"^blk\.(\d+)\.attn_qkv\.(weight|bias)$"), "QKV_PROJ"),
    (re.compile(r"^blk\.(\d+)\.attn_q_norm\.(weight|bias)$"), "Q_NORM"),
    (re.compile(r"^blk\.(\d+)\.attn_k_norm\.(weight|bias)$"), "K_NORM"),
    (re.compile(r"^blk\.(\d+)\.attn_q_a\.(weight|bias)$"), "Q_A_PROJ"),
    (re.compile(r"^blk\.(\d+)\.attn_q_b\.(weight|bias)$"), "Q_B_PROJ"),
    (re.compile(r"^blk\.(\d+)\.attn_kv_a_mqa\.(weight|bias)$"), "KV_A_PROJ"),
    (re.compile(r"^blk\.(\d+)\.attn_kv_b\.(weight|bias)$"), "KV_B_PROJ"),
    (re.compile(r"^blk\.(\d+)\.attn_q_a_norm\.(weight|bias)$"), "Q_A_NORM"),
    (re.compile(r"^blk\.(\d+)\.attn_kv_a_norm\.(weight|bias)$"), "KV_A_NORM"),
    (re.compile(r"^blk\.(\d+)\.attn_sinks\.(weight|bias)$"), "ATTN_SINKS"),
    (re.compile(r"^blk\.(\d+)\.ffn_gate\.(weight|bias)$"), "DENSE_GATE"),
    (re.compile(r"^blk\.(\d+)\.ffn_up\.(weight|bias)$"), "DENSE_UP"),
    (re.compile(r"^blk\.(\d+)\.ffn_down\.(weight|bias)$"), "DENSE_DOWN"),
    (re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.(weight|bias)$"), "ROUTER"),
    (re.compile(r"^blk\.(\d+)\.ffn_gate_exp[s]?\.(weight|bias)$"), "EXPERT_GATE_PACKED"),
    (re.compile(r"^blk\.(\d+)\.ffn_up_exp[s]?\.(weight|bias)$"), "EXPERT_UP_PACKED"),
    (re.compile(r"^blk\.(\d+)\.ffn_down_exp[s]?\.(weight|bias)$"), "EXPERT_DOWN_PACKED"),
    (re.compile(r"^blk\.(\d+)\.ffn_gate_shexp\.(weight|bias)$"), "SHARED_GATE"),
    (re.compile(r"^blk\.(\d+)\.ffn_up_shexp\.(weight|bias)$"), "SHARED_UP"),
    (re.compile(r"^blk\.(\d+)\.ffn_down_shexp\.(weight|bias)$"), "SHARED_DOWN"),
]

_GGUF_IGNORE_PATTERNS = [
    re.compile(r"^rope_freqs\.weight$"),
    re.compile(r"^blk\.\d+\.rope_freqs\.weight$"),
]


def _with_parameter_kind(role: str, parameter: str) -> str:
    return role if parameter == "weight" else f"{role}_BIAS"


def map_tensor_name(architecture_id: str, name: str) -> dict[str, Any]:
    """Map one tensor name. Unknown architecture/name is explicit UNSUPPORTED."""
    architecture_id = str(architecture_id)
    if architecture_id not in _KNOWN_SEMANTIC_ARCHITECTURES:
        return {
            "role": f"UNMAPPED_{architecture_id.upper().replace('-', '_')}",
            "mapping_status": "UNSUPPORTED",
        }

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


def map_gguf_tensor_name(architecture_id: str, name: str) -> dict[str, Any]:
    """Map standardized GGUF tensor names without inventing architecture roles."""
    architecture_id = str(architecture_id)
    if architecture_id not in _KNOWN_SEMANTIC_ARCHITECTURES:
        return {
            "role": f"UNMAPPED_{architecture_id.upper().replace('-', '_')}",
            "mapping_status": "UNSUPPORTED",
        }

    for pattern in _GGUF_IGNORE_PATTERNS:
        if pattern.fullmatch(name):
            return {"role": "NON_PARAMETER_METADATA", "mapping_status": "IGNORE"}

    for pattern, role in _GGUF_GLOBAL_PATTERNS:
        m = pattern.fullmatch(name)
        if m:
            return {
                "role": _with_parameter_kind(role, m.group(1)),
                "mapping_status": "MAPPED",
            }

    for pattern, role in _GGUF_LAYER_PATTERNS:
        m = pattern.fullmatch(name)
        if m:
            return {
                "layer": int(m.group(1)),
                "role": _with_parameter_kind(role, m.group(2)),
                "mapping_status": "MAPPED",
            }

    return {
        "role": f"UNMAPPED_{architecture_id.upper().replace('-', '_')}",
        "mapping_status": "UNSUPPORTED",
    }


def build_tensor_role_graph_from_gguf_inventory(
    *,
    model_id: str,
    architecture_id: str,
    inventory: Mapping[str, Any],
) -> dict:
    nodes = []
    for tensor in inventory.get("tensors", []):
        name = str(tensor["name"])
        mapped = map_gguf_tensor_name(architecture_id, name)
        nodes.append({
            "source_tensor_name": name,
            "role": mapped["role"],
            "layer": mapped.get("layer"),
            "expert_id": None,
            "shape": list(tensor.get("shape", [])),
            "dtype": tensor.get("ggml_type"),
            "source_quant_format": tensor.get("ggml_type"),
            "mapping_status": mapped["mapping_status"],
        })
    return mc.build_tensor_role_graph(model_id=model_id, nodes=nodes)
