#!/usr/bin/env python3
"""P12 model capability compiler.

Read-only front door for Beglin model generalization:
source inspection -> architecture descriptor -> ModelSkeleton ->
TensorRoleGraph/OperatorGraph -> tokenizer/loader/backend/quant/mutation
capabilities -> stable ModelCapabilityBundle.

This module deliberately performs no production mutation.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import struct
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "v1"
SUPPORTED_ARCHITECTURES = {
    "qwen2",
    "llama",
    "deepseek_v2",
    "qwen3_moe",
    "olmoe",
    "gpt-oss",
}
ARCH_ALIASES = {
    "qwen2": "qwen2",
    "qwen2_moe": "qwen3_moe",
    "qwen3_moe": "qwen3_moe",
    "qwen3moe": "qwen3_moe",
    "llama": "llama",
    "deepseek_v2": "deepseek_v2",
    "olmoe": "olmoe",
    "gpt_oss": "gpt-oss",
    "gpt-oss": "gpt-oss",
}
GGUF_SUPPORTED_ARCH = {"qwen2", "llama", "qwen3_moe", "gpt-oss"}
SAFETENSORS_SUPPORTED_ARCH = {"qwen2", "llama", "deepseek_v2", "qwen3_moe", "olmoe"}

GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    30: "BF16",
    39: "MXFP4",
}
LOADER_SUPPORTED_QUANTS = {
    "F32", "F16", "BF16", "Q4_0", "Q5_0", "Q8_0",
    "Q3_K", "Q4_K", "Q5_K", "Q6_K", "MXFP4",
}
LOADER_EXPLICIT_UNSUPPORTED = {"Q2_K", "IQ_SERIES"}

CPU_QNG64_WIDTHS = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
MLX_QNG64_WIDTHS = [2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]

PRECISION_ROLES = {
    "Q_PROJ", "K_PROJ", "V_PROJ", "O_PROJ",
    "Q_A_PROJ", "Q_B_PROJ", "KV_A_PROJ", "KV_B_PROJ",
    "DENSE_GATE", "DENSE_UP", "DENSE_DOWN",
    "SHARED_GATE", "SHARED_UP", "SHARED_DOWN",
    "EXPERT_GATE", "EXPERT_UP", "EXPERT_DOWN",
}
GLOBAL_ROLES = {"EMBEDDING", "LM_HEAD", "FINAL_NORM"}


class ModelCapabilityError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


_IDENTITY_EPHEMERAL_KEYS = {
    "discovered_at", "observed_at", "collected_at", "captured_at",
}
_IDENTITY_LOCATION_KEYS = {
    "root_path", "primary_path", "config_path", "tokenizer_paths",
    "shard_paths", "path",
}
_SELF_HASH_KEYS = {
    "source_manifest_sha256", "architecture_descriptor_sha256",
    "tensor_role_graph_sha256", "operator_graph_sha256",
    "tokenizer_contract_sha256", "loader_contract_sha256",
    "skeleton_sha256", "backend_capability_sha256",
    "quant_capability_sha256", "runtime_mutation_sha256",
    "bundle_sha256", "transition_plan_sha256", "validation_plan_sha256",
}


def stable_identity_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: stable_identity_payload(v)
            for k, v in value.items()
            if k not in _IDENTITY_EPHEMERAL_KEYS
            and k not in _IDENTITY_LOCATION_KEYS
            and k not in _SELF_HASH_KEYS
        }
    if isinstance(value, list):
        return [stable_identity_payload(v) for v in value]
    return value


def stable_identity_sha256(value: Any) -> str:
    return sha256_json(stable_identity_payload(value))


def normalize_architecture(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    return ARCH_ALIASES.get(raw, raw or "unknown")


def _require_regular(path: Path) -> None:
    if path.is_symlink():
        raise ModelCapabilityError(f"symlink source is not accepted: {path}")
    if not path.is_file():
        raise ModelCapabilityError(f"not a regular file: {path}")


class _FileCursor:
    """Bounds-checked streaming cursor over GGUF metadata/tensor headers.

    P12 inspection must work for multi-GB GGUF files without reading the model
    data section into memory. Only metadata and tensor-info records are read.
    """
    def __init__(self, handle, size: int):
        self.handle = handle
        self.size = int(size)
        self.pos = 0

    def need(self, n: int) -> None:
        if n < 0 or self.pos + n > self.size:
            raise ModelCapabilityError(
                f"truncated binary metadata at offset={self.pos} need={n}"
            )

    def take(self, n: int) -> bytes:
        self.need(n)
        out = self.handle.read(n)
        if len(out) != n:
            raise ModelCapabilityError(
                f"short read at offset={self.pos} expected={n} actual={len(out)}"
            )
        self.pos += n
        return out

    def skip(self, n: int) -> None:
        self.need(n)
        self.handle.seek(n, 1)
        self.pos += n

    def u8(self) -> int:
        return self.take(1)[0]

    def i8(self) -> int:
        return struct.unpack("<b", self.take(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.take(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.take(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.take(8))[0]

    def string(self) -> str:
        n = self.u64()
        if n > 64 * 1024 * 1024:
            raise ModelCapabilityError(f"GGUF string length is unreasonable: {n}")
        raw = self.take(n)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelCapabilityError("GGUF metadata string is not UTF-8") from exc


_GGUF_FIXED_WIDTH = {
    0: 1,  # UINT8
    1: 1,  # INT8
    2: 2,  # UINT16
    3: 2,  # INT16
    4: 4,  # UINT32
    5: 4,  # INT32
    6: 4,  # FLOAT32
    7: 1,  # BOOL
    10: 8, # UINT64
    11: 8, # INT64
    12: 8, # FLOAT64
}


def _gguf_scalar(cur: _FileCursor, type_id: int) -> Any:
    readers = {
        0: cur.u8, 1: cur.i8, 2: cur.u16, 3: cur.i16,
        4: cur.u32, 5: cur.i32, 6: cur.f32, 7: cur.u8,
        8: cur.string, 10: cur.u64, 11: cur.i64, 12: cur.f64,
    }
    if type_id not in readers:
        raise ModelCapabilityError(f"unsupported GGUF metadata value type={type_id}")
    return readers[type_id]()


def _gguf_skip_array(cur: _FileCursor, elem_type: int, count: int) -> None:
    if count < 0 or count > 100_000_000:
        raise ModelCapabilityError(f"GGUF metadata array count is unreasonable: {count}")
    if elem_type in _GGUF_FIXED_WIDTH:
        cur.skip(_GGUF_FIXED_WIDTH[elem_type] * count)
        return
    if elem_type == 8:  # STRING
        for _ in range(count):
            n = cur.u64()
            if n > 64 * 1024 * 1024:
                raise ModelCapabilityError(f"GGUF array string length is unreasonable: {n}")
            cur.skip(n)
        return
    if elem_type == 9:  # nested ARRAY; unusual but parse fail-closed
        for _ in range(count):
            nested_type = cur.u32()
            nested_count = cur.u64()
            _gguf_skip_array(cur, nested_type, nested_count)
        return
    raise ModelCapabilityError(f"unsupported GGUF array element type={elem_type}")


def _gguf_read_value(cur: _FileCursor, type_id: int) -> Any:
    if type_id != 9:
        return _gguf_scalar(cur, type_id)
    elem_type = cur.u32()
    count = cur.u64()
    _gguf_skip_array(cur, elem_type, count)
    return {"array_type": elem_type, "count": count}


def parse_gguf(path: str | Path) -> dict:
    p = Path(path)
    _require_regular(p)
    size = p.stat().st_size
    with p.open("rb") as handle:
        cur = _FileCursor(handle, size)
        if cur.take(4) != b"GGUF":
            raise ModelCapabilityError(f"bad GGUF magic: {p}")
        version = cur.u32()
        tensor_count = cur.u64()
        kv_count = cur.u64()
        if tensor_count > 10_000_000 or kv_count > 1_000_000:
            raise ModelCapabilityError(
                f"unreasonable GGUF counts tensors={tensor_count} kv={kv_count}"
            )
        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = cur.string()
            type_id = cur.u32()
            metadata[key] = _gguf_read_value(cur, type_id)
        tensors = []
        for _ in range(tensor_count):
            name = cur.string()
            n_dims = cur.u32()
            if n_dims > 8:
                raise ModelCapabilityError(f"unreasonable GGUF tensor rank={n_dims}: {name}")
            dims = [cur.u64() for _ in range(n_dims)]
            type_id = cur.u32()
            offset = cur.u64()
            tensors.append({
                "name": name,
                "shape": dims,
                "dtype": GGML_TYPE_NAMES.get(type_id, f"GGML_TYPE_{type_id}"),
                "ggml_type_id": type_id,
                "data_offset": offset,
            })
    return {
        "version": version,
        "metadata": metadata,
        "tensors": tensors,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
    }

def parse_safetensors_header(path: str | Path) -> dict:
    p = Path(path)
    _require_regular(p)
    with p.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ModelCapabilityError(f"truncated safetensors header length: {p}")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len <= 1 or header_len > 128 * 1024 * 1024:
            raise ModelCapabilityError(f"invalid safetensors header length={header_len}")
        raw = f.read(header_len)
        if len(raw) != header_len:
            raise ModelCapabilityError(f"truncated safetensors JSON header: {p}")
    try:
        header = json.loads(raw)
    except Exception as exc:
        raise ModelCapabilityError(f"invalid safetensors JSON header: {p}") from exc
    tensors = []
    for name, row in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(row, dict):
            raise ModelCapabilityError(f"bad safetensors tensor entry: {name}")
        tensors.append({
            "name": name,
            "shape": [int(x) for x in row.get("shape", [])],
            "dtype": str(row.get("dtype", "UNKNOWN")),
            "data_offsets": [int(x) for x in row.get("data_offsets", [])],
        })
    return {
        "metadata": header.get("__metadata__", {}),
        "tensors": tensors,
        "tensor_count": len(tensors),
    }


def _find_config(root: Path) -> Path | None:
    p = root / "config.json"
    return p if p.is_file() else None


def _load_json(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise ModelCapabilityError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ModelCapabilityError(f"expected JSON object: {path}")
    return value


def _tokenizer_files(root: Path) -> list[Path]:
    names = (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "tokenizer.model", "spiece.model", "merges.txt", "vocab.json",
    )
    return [root / name for name in names if (root / name).is_file()]


def _file_record(path: Path) -> dict:
    return {
        "name": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def inspect_model_source(path: str | Path) -> dict:
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise ModelCapabilityError(f"model source does not exist: {src}")

    root = src if src.is_dir() else src.parent
    config_path = _find_config(root)
    config = _load_json(config_path)
    tokenizer_paths = _tokenizer_files(root)

    source_format = None
    primary: Path | None = None
    shard_paths: list[Path] = []
    tensor_inventory: list[dict] = []
    metadata: dict[str, Any] = {}

    if src.is_file() and src.suffix.lower() == ".gguf":
        source_format = "GGUF"
        primary = src
    elif src.is_file() and src.suffix.lower() == ".safetensors":
        source_format = "SAFETENSORS_SINGLE"
        primary = src
    elif src.is_file() and src.name.endswith(".safetensors.index.json"):
        source_format = "SAFETENSORS_SHARDED"
        primary = src
    elif src.is_dir():
        ggufs = sorted(src.glob("*.gguf"))
        indices = sorted(src.glob("*.safetensors.index.json"))
        sts = sorted(src.glob("*.safetensors"))
        if len(ggufs) == 1:
            source_format, primary = "GGUF", ggufs[0]
        elif indices:
            source_format, primary = "SAFETENSORS_SHARDED", indices[0]
        elif len(sts) == 1:
            source_format, primary = "SAFETENSORS_SINGLE", sts[0]
        elif (src / "manifest.json").is_file():
            source_format, primary = "LEGACY_BEG_LIN", src / "manifest.json"
        else:
            raise ModelCapabilityError(
                f"could not select a unique model source in directory: {src}"
            )
    else:
        raise ModelCapabilityError(f"unsupported model source: {src}")

    identity_files: list[Path] = []
    if config_path:
        identity_files.append(config_path)
    identity_files.extend(tokenizer_paths)

    if source_format == "GGUF":
        assert primary is not None
        parsed = parse_gguf(primary)
        metadata = parsed["metadata"]
        tensor_inventory = parsed["tensors"]
        identity_files.append(primary)
    elif source_format == "SAFETENSORS_SINGLE":
        assert primary is not None
        parsed = parse_safetensors_header(primary)
        metadata = parsed["metadata"]
        tensor_inventory = parsed["tensors"]
        identity_files.append(primary)
    elif source_format == "SAFETENSORS_SHARDED":
        assert primary is not None
        index = _load_json(primary)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelCapabilityError("safetensors index missing non-empty weight_map")
        shard_names = sorted({str(v) for v in weight_map.values()})
        shard_paths = [root / name for name in shard_names]
        missing = [str(p) for p in shard_paths if not p.is_file()]
        if missing:
            raise ModelCapabilityError(f"missing safetensors shard(s): {missing}")
        identity_files.append(primary)
        identity_files.extend(shard_paths)
        seen = set()
        for shard in shard_paths:
            parsed = parse_safetensors_header(shard)
            for row in parsed["tensors"]:
                if row["name"] in seen:
                    raise ModelCapabilityError(
                        f"duplicate tensor across safetensors shards: {row['name']}"
                    )
                seen.add(row["name"])
                tensor_inventory.append(row)
        metadata = index.get("metadata", {}) if isinstance(index.get("metadata"), dict) else {}
    elif source_format == "LEGACY_BEG_LIN":
        assert primary is not None
        identity_files.append(primary)
        metadata = _load_json(primary)

    records = [_file_record(p) for p in sorted(set(identity_files), key=str)]
    checkpoint_identity_sha256 = sha256_json(
        [{"name": r["name"], "sha256": r["sha256"], "size_bytes": r["size_bytes"]}
         for r in records]
    )

    manifest = {
        "schema": "beglin-model-source-v1",
        "model_id": (
            Path(str(config.get("_name_or_path"))).name
            if config.get("_name_or_path")
            else str(metadata.get("general.name") or root.name)
        ),
        "model_revision": str(config.get("_commit_hash") or ""),
        "source_format": source_format,
        "root_path": str(root),
        "primary_path": str(primary) if primary else None,
        "config_path": str(config_path) if config_path else None,
        "tokenizer_paths": [str(p) for p in tokenizer_paths],
        "shard_paths": [str(p) for p in shard_paths],
        "file_hashes": records,
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
        "tensor_count": len(tensor_inventory),
        "metadata": metadata,
        "config": config,
        "tensor_inventory": tensor_inventory,
        "immutable": True,
    }
    manifest["source_manifest_sha256"] = stable_identity_sha256(manifest)
    return manifest


ARCH_REGISTRY = {
    "qwen2": {
        "family": "qwen",
        "dense_or_moe": "DENSE",
        "attention_kind": "GQA",
        "rope_kind": "ROPE",
        "norm_kind": "RMS_NORM",
        "qk_norm_kind": "NONE",
        "tokenizer_family": "QWEN2_BPE",
    },
    "llama": {
        "family": "llama",
        "dense_or_moe": "DENSE",
        "attention_kind": "GQA",
        "rope_kind": "LLAMA_ROPE",
        "norm_kind": "RMS_NORM",
        "qk_norm_kind": "NONE",
        "tokenizer_family": "LLAMA_BPE",
    },
    "deepseek_v2": {
        "family": "deepseek",
        "dense_or_moe": "MOE",
        "attention_kind": "MLA",
        "rope_kind": "YARN",
        "norm_kind": "RMS_NORM",
        "qk_norm_kind": "MLA_QK_A_NORM",
        "tokenizer_family": "DEEPSEEK_BPE",
    },
    "qwen3_moe": {
        "family": "qwen",
        "dense_or_moe": "MOE",
        "attention_kind": "GQA",
        "rope_kind": "ROPE",
        "norm_kind": "RMS_NORM",
        "qk_norm_kind": "PER_HEAD",
        "tokenizer_family": "QWEN2_BPE",
    },
    "olmoe": {
        "family": "olmo",
        "dense_or_moe": "MOE",
        "attention_kind": "GQA",
        "rope_kind": "ROPE",
        "norm_kind": "RMS_NORM",
        "qk_norm_kind": "WHOLE_VECTOR",
        "tokenizer_family": "OLMO_BPE",
    },
    "gpt-oss": {
        "family": "gpt-oss",
        "dense_or_moe": "MOE",
        "attention_kind": "HYBRID",
        "rope_kind": "YARN_NEOX",
        "norm_kind": "RMS_NORM",
        "qk_norm_kind": "NONE",
        "tokenizer_family": "O200K_HARMONY",
    },
}


def _config_architecture(source: Mapping[str, Any]) -> str:
    config = source.get("config") or {}
    metadata = source.get("metadata") or {}
    if config:
        model_type = config.get("model_type")
        if model_type:
            return normalize_architecture(str(model_type))
        archs = config.get("architectures")
        if isinstance(archs, list) and archs:
            token = str(archs[0]).lower()
            if "deepseek" in token:
                return "deepseek_v2"
            if "qwen3" in token and "moe" in token:
                return "qwen3_moe"
            if "qwen" in token:
                return "qwen2"
            if "llama" in token:
                return "llama"
            if "olmoe" in token:
                return "olmoe"
    raw = metadata.get("general.architecture")
    if isinstance(raw, str):
        return normalize_architecture(raw)
    return "unknown"


def _first_value(config: Mapping[str, Any], keys: Iterable[str], default=None):
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _gguf_arch_value(metadata: Mapping[str, Any], raw_arch: str, suffixes: Iterable[str], default=None):
    for suffix in suffixes:
        key = f"{raw_arch}.{suffix}"
        if key in metadata:
            return metadata[key]
    return default


def build_architecture_descriptor(source: Mapping[str, Any]) -> dict:
    arch = _config_architecture(source)
    config = source.get("config") or {}
    metadata = source.get("metadata") or {}
    reg = ARCH_REGISTRY.get(arch)
    raw_arch = str(metadata.get("general.architecture") or arch).replace("_", "-")
    if raw_arch == "qwen3-moe":
        raw_arch = "qwen3moe"
    if raw_arch == "gpt_oss":
        raw_arch = "gpt-oss"

    def fact(config_keys, gguf_suffixes, default=None):
        value = _first_value(config, config_keys, None)
        if value is not None:
            return value
        return _gguf_arch_value(metadata, raw_arch, gguf_suffixes, default)

    n_heads = fact(["num_attention_heads", "n_head"], ["attention.head_count"])
    n_kv = fact(["num_key_value_heads"], ["attention.head_count_kv"], n_heads)
    hidden = fact(["hidden_size", "n_embd"], ["embedding_length"])
    head_dim = fact(["head_dim"], ["attention.key_length"])
    if head_dim is None and hidden and n_heads:
        head_dim = int(hidden) // int(n_heads)

    expert_count = fact(
        ["n_routed_experts", "num_experts", "num_local_experts"],
        ["expert_count"],
    )
    experts_per_token = fact(
        ["num_experts_per_tok", "num_experts_per_token", "top_k"],
        ["expert_used_count"],
    )

    if reg is None:
        desc = {
            "schema": "beglin-architecture-descriptor-v1",
            "architecture_id": "unknown",
            "architecture_family": "unknown",
            "architecture_variant": str(config.get("model_type") or metadata.get("general.architecture") or ""),
            "source_architecture_name": str(config.get("model_type") or metadata.get("general.architecture") or ""),
            "status": "UNKNOWN_ARCHITECTURE",
            "dense_or_moe": "UNKNOWN",
            "attention_kind": "UNKNOWN",
            "rope_kind": "UNKNOWN",
            "norm_kind": "UNKNOWN",
            "qk_norm_kind": "UNKNOWN",
            "architecture_adapter_id": None,
            "architecture_adapter_version": None,
        }
    else:
        attention_kind = reg["attention_kind"]
        if arch in {"qwen2", "llama"} and n_heads and n_kv and int(n_heads) == int(n_kv):
            attention_kind = "MHA"
        desc = {
            "schema": "beglin-architecture-descriptor-v1",
            "architecture_id": arch,
            "architecture_family": reg["family"],
            "architecture_variant": str(config.get("model_type") or metadata.get("general.architecture") or arch),
            "source_architecture_name": str(config.get("model_type") or metadata.get("general.architecture") or arch),
            "status": "VERIFIED_ADAPTER",
            "dense_or_moe": reg["dense_or_moe"],
            "attention_kind": attention_kind,
            "rope_kind": reg["rope_kind"],
            "norm_kind": reg["norm_kind"],
            "qk_norm_kind": reg["qk_norm_kind"],
            "architecture_adapter_id": f"beglin-{arch}-v1",
            "architecture_adapter_version": 1,
        }

    desc.update({
        "hidden_size": int(hidden) if hidden is not None else None,
        "intermediate_size": (
            int(fact(["intermediate_size", "moe_intermediate_size"], ["feed_forward_length", "expert_feed_forward_length"]))
            if fact(["intermediate_size", "moe_intermediate_size"], ["feed_forward_length", "expert_feed_forward_length"]) is not None
            else None
        ),
        "num_layers": (
            int(fact(["num_hidden_layers", "n_layer"], ["block_count"]))
            if fact(["num_hidden_layers", "n_layer"], ["block_count"]) is not None
            else None
        ),
        "num_attention_heads": int(n_heads) if n_heads is not None else None,
        "num_kv_heads": int(n_kv) if n_kv is not None else None,
        "head_dim": int(head_dim) if head_dim is not None else None,
        "vocab_size": (
            int(fact(["vocab_size"], ["vocab_size"]))
            if fact(["vocab_size"], ["vocab_size"]) is not None
            else None
        ),
        "context_length": (
            int(fact(["max_position_embeddings", "max_sequence_length"], ["context_length"]))
            if fact(["max_position_embeddings", "max_sequence_length"], ["context_length"]) is not None
            else None
        ),
        "expert_count": int(expert_count) if expert_count is not None else None,
        "experts_per_token": int(experts_per_token) if experts_per_token is not None else None,
        "shared_expert_count": (
            int(_first_value(config, ["n_shared_experts", "num_shared_experts"], 0) or 0)
            if arch != "unknown" else None
        ),
        "dense_prefix_layers": (
            int(_first_value(config, ["first_k_dense_replace"], 0) or 0)
            if arch == "deepseek_v2" else 0
        ),
        "sliding_window": _first_value(config, ["sliding_window"], _gguf_arch_value(metadata, raw_arch, ["attention.sliding_window"], None)),
        "attention_sink": bool(arch == "gpt-oss"),
    })
    desc["architecture_descriptor_sha256"] = stable_identity_sha256(desc)
    return desc


_HF_LAYER = r"(?:model\.)?layers\.(?P<layer>\d+)\."
_HF_PATTERNS = [
    (re.compile(r"^(?:model\.)?embed_tokens\.weight$"), "EMBEDDING", None),
    (re.compile(r"^lm_head\.weight$"), "LM_HEAD", None),
    (re.compile(r"^(?:model\.)?norm\.weight$"), "FINAL_NORM", None),
    (re.compile("^" + _HF_LAYER + r"self_attn\.q_proj\.weight$"), "Q_PROJ", None),
    (re.compile("^" + _HF_LAYER + r"self_attn\.k_proj\.weight$"), "K_PROJ", None),
    (re.compile("^" + _HF_LAYER + r"self_attn\.v_proj\.weight$"), "V_PROJ", None),
    (re.compile("^" + _HF_LAYER + r"self_attn\.o_proj\.weight$"), "O_PROJ", None),
    (re.compile("^" + _HF_LAYER + r"self_attn\.q_a_proj\.weight$"), "Q_A_PROJ", None),
    (re.compile("^" + _HF_LAYER + r"self_attn\.q_b_proj\.weight$"), "Q_B_PROJ", None),
    (re.compile("^" + _HF_LAYER + r"self_attn\.kv_a_proj_with_mqa\.weight$"), "KV_A_PROJ", None),
    (re.compile("^" + _HF_LAYER + r"self_attn\.kv_b_proj\.weight$"), "KV_B_PROJ", None),
    (re.compile("^" + _HF_LAYER + r"self_attn\.q_norm\.weight$"), "Q_NORM", None),
    (re.compile("^" + _HF_LAYER + r"self_attn\.k_norm\.weight$"), "K_NORM", None),
    (re.compile("^" + _HF_LAYER + r"self_attn\.q_a_layernorm\.weight$"), "Q_NORM", None),
    (re.compile("^" + _HF_LAYER + r"self_attn\.kv_a_layernorm\.weight$"), "K_NORM", None),
    (re.compile("^" + _HF_LAYER + r"mlp\.gate_proj\.weight$"), "DENSE_GATE", None),
    (re.compile("^" + _HF_LAYER + r"mlp\.up_proj\.weight$"), "DENSE_UP", None),
    (re.compile("^" + _HF_LAYER + r"mlp\.down_proj\.weight$"), "DENSE_DOWN", None),
    (re.compile("^" + _HF_LAYER + r"mlp\.gate\.weight$"), "ROUTER", None),
    (re.compile("^" + _HF_LAYER + r"mlp\.shared_experts\.gate_proj\.weight$"), "SHARED_GATE", None),
    (re.compile("^" + _HF_LAYER + r"mlp\.shared_experts\.up_proj\.weight$"), "SHARED_UP", None),
    (re.compile("^" + _HF_LAYER + r"mlp\.shared_experts\.down_proj\.weight$"), "SHARED_DOWN", None),
    (re.compile("^" + _HF_LAYER + r"mlp\.experts\.(?P<expert>\d+)\.gate_proj\.weight$"), "EXPERT_GATE", "expert"),
    (re.compile("^" + _HF_LAYER + r"mlp\.experts\.(?P<expert>\d+)\.up_proj\.weight$"), "EXPERT_UP", "expert"),
    (re.compile("^" + _HF_LAYER + r"mlp\.experts\.(?P<expert>\d+)\.down_proj\.weight$"), "EXPERT_DOWN", "expert"),
]

_GGUF_PATTERNS = [
    (re.compile(r"^token_embd\.weight$"), "EMBEDDING"),
    (re.compile(r"^output\.weight$"), "LM_HEAD"),
    (re.compile(r"^output_norm\.weight$"), "FINAL_NORM"),
    (re.compile(r"^blk\.(?P<layer>\d+)\.attn_q\.weight$"), "Q_PROJ"),
    (re.compile(r"^blk\.(?P<layer>\d+)\.attn_k\.weight$"), "K_PROJ"),
    (re.compile(r"^blk\.(?P<layer>\d+)\.attn_v\.weight$"), "V_PROJ"),
    (re.compile(r"^blk\.(?P<layer>\d+)\.attn_output\.weight$"), "O_PROJ"),
    (re.compile(r"^blk\.(?P<layer>\d+)\.attn_q_norm\.weight$"), "Q_NORM"),
    (re.compile(r"^blk\.(?P<layer>\d+)\.attn_k_norm\.weight$"), "K_NORM"),
    (re.compile(r"^blk\.(?P<layer>\d+)\.ffn_gate\.weight$"), "DENSE_GATE"),
    (re.compile(r"^blk\.(?P<layer>\d+)\.ffn_up\.weight$"), "DENSE_UP"),
    (re.compile(r"^blk\.(?P<layer>\d+)\.ffn_down\.weight$"), "DENSE_DOWN"),
    (re.compile(r"^blk\.(?P<layer>\d+)\.ffn_gate_inp\.weight$"), "ROUTER"),
    (re.compile(r"^blk\.(?P<layer>\d+)\.ffn_gate_exps\.weight$"), "EXPERT_GATE"),
    (re.compile(r"^blk\.(?P<layer>\d+)\.ffn_up_exps\.weight$"), "EXPERT_UP"),
    (re.compile(r"^blk\.(?P<layer>\d+)\.ffn_down_exps\.weight$"), "EXPERT_DOWN"),
]


def _target_key(model_id: str, layer: int | None, role: str, expert: int | None = None) -> str:
    scope = "global" if layer is None else f"L{layer}"
    suffix = f"/E{expert}" if expert is not None else ""
    return f"{model_id}/{scope}/{role.lower()}{suffix}"


def classify_tensor(name: str, source_format: str, model_id: str) -> dict | None:
    if source_format == "GGUF":
        for pattern, role in _GGUF_PATTERNS:
            m = pattern.match(name)
            if m:
                layer = int(m.group("layer")) if "layer" in m.groupdict() and m.group("layer") else None
                return {
                    "role": role,
                    "layer": layer,
                    "expert_id": None,
                    "stacked_experts": role.startswith("EXPERT_"),
                    "canonical_target_key": _target_key(model_id, layer, role),
                }
    else:
        for pattern, role, expert_group in _HF_PATTERNS:
            m = pattern.match(name)
            if m:
                layer = int(m.group("layer")) if "layer" in m.groupdict() and m.group("layer") else None
                expert = (
                    int(m.group(expert_group))
                    if expert_group and m.groupdict().get(expert_group) is not None
                    else None
                )
                return {
                    "role": role,
                    "layer": layer,
                    "expert_id": expert,
                    "stacked_experts": False,
                    "canonical_target_key": _target_key(model_id, layer, role, expert),
                }
    # Biases and norms are valid model tensors but not precision targets in v1.
    if name.endswith(".bias") or name.endswith("_norm.weight") or ".input_layernorm.weight" in name or ".post_attention_layernorm.weight" in name:
        return {
            "role": "IGNORE_NON_PRECISION",
            "layer": None,
            "expert_id": None,
            "stacked_experts": False,
            "canonical_target_key": f"{model_id}/ignore/{name}",
        }
    return None


def build_tensor_role_graph(source: Mapping[str, Any], descriptor: Mapping[str, Any]) -> dict:
    model_id = str(source["model_id"])
    nodes = []
    unmapped = []
    seen_targets: dict[str, str] = {}
    for tensor in source.get("tensor_inventory", []):
        name = str(tensor["name"])
        mapped = classify_tensor(name, str(source["source_format"]), model_id)
        if mapped is None:
            unmapped.append(name)
            continue
        target = mapped["canonical_target_key"]
        # Multiple physical tensors may only share an IGNORE key; semantic targets must be unique.
        if mapped["role"] != "IGNORE_NON_PRECISION" and target in seen_targets:
            raise ModelCapabilityError(
                f"duplicate semantic target {target}: {seen_targets[target]} and {name}"
            )
        seen_targets[target] = name
        nodes.append({
            **mapped,
            "source_tensor_name": name,
            "shape": [int(x) for x in tensor.get("shape", [])],
            "dtype": str(tensor.get("dtype", "UNKNOWN")),
            "source_quant_format": str(tensor.get("dtype", "UNKNOWN")),
            "semantic_group": (
                "GLOBAL" if mapped["role"] in GLOBAL_ROLES else
                "ATTENTION" if mapped["role"] in {"Q_PROJ","K_PROJ","V_PROJ","O_PROJ","Q_A_PROJ","Q_B_PROJ","KV_A_PROJ","KV_B_PROJ","Q_NORM","K_NORM"} else
                "MOE" if mapped["role"] == "ROUTER" or mapped["role"].startswith(("EXPERT_","SHARED_")) else
                "DENSE_FFN"
            ),
        })
    graph = {
        "schema": "beglin-tensor-role-graph-v1",
        "model_id": model_id,
        "architecture_id": descriptor["architecture_id"],
        "tensor_count": int(source.get("tensor_count", 0)),
        "mapped_tensor_count": len(nodes),
        "unmapped_tensor_count": len(unmapped),
        "mapping_coverage": (
            float(len(nodes)) / float(source.get("tensor_count", 1))
            if int(source.get("tensor_count", 0)) > 0 else 0.0
        ),
        "nodes": sorted(nodes, key=lambda r: (str(r["canonical_target_key"]), str(r["source_tensor_name"]))),
        "unmapped_tensors": sorted(unmapped),
    }
    graph["tensor_role_graph_sha256"] = stable_identity_sha256(graph)
    return graph


def _attention_operator(descriptor: Mapping[str, Any]) -> str:
    kind = descriptor.get("attention_kind")
    return {
        "MHA": "ATTENTION_MHA",
        "GQA": "ATTENTION_GQA",
        "MLA": "ATTENTION_MLA",
        "HYBRID": "SLIDING_WINDOW_ATTENTION",
        "SLIDING_WINDOW": "SLIDING_WINDOW_ATTENTION",
    }.get(str(kind), "UNSUPPORTED_ATTENTION")


def build_operator_graph(descriptor: Mapping[str, Any]) -> dict:
    arch = str(descriptor["architecture_id"])
    n_layers = int(descriptor.get("num_layers") or 0)
    ops = []
    unsupported = []
    if arch == "unknown":
        unsupported.append("UNKNOWN_ARCHITECTURE")
    attn_op = _attention_operator(descriptor)
    if attn_op == "UNSUPPORTED_ATTENTION":
        unsupported.append(str(descriptor.get("attention_kind") or "UNKNOWN_ATTENTION"))

    for layer in range(n_layers):
        if attn_op != "UNSUPPORTED_ATTENTION":
            ops.append({
                "operator_id": f"L{layer}/attention",
                "operator_type": attn_op,
                "layer": layer,
                "numeric_semantics": {
                    "qk_norm_kind": descriptor.get("qk_norm_kind"),
                    "rope_kind": descriptor.get("rope_kind"),
                    "sliding_window": descriptor.get("sliding_window"),
                    "attention_sink": descriptor.get("attention_sink"),
                },
                "precision_sensitive": True,
                "mutation_scope": "WEIGHTS_ONLY",
            })
        dense_prefix = int(descriptor.get("dense_prefix_layers") or 0)
        if descriptor.get("dense_or_moe") == "MOE" and layer >= dense_prefix:
            ops.append({
                "operator_id": f"L{layer}/router",
                "operator_type": "ROUTER_TOPK",
                "layer": layer,
                "numeric_semantics": {
                    "expert_count": descriptor.get("expert_count"),
                    "experts_per_token": descriptor.get("experts_per_token"),
                },
                "precision_sensitive": True,
                "mutation_scope": "WEIGHTS_ONLY",
            })
            ops.append({
                "operator_id": f"L{layer}/moe",
                "operator_type": "MOE_EXPERT",
                "layer": layer,
                "numeric_semantics": {
                    "shared_expert_count": descriptor.get("shared_expert_count"),
                },
                "precision_sensitive": True,
                "mutation_scope": "WEIGHTS_ONLY",
            })
        else:
            ops.append({
                "operator_id": f"L{layer}/ffn",
                "operator_type": "DENSE_FFN",
                "layer": layer,
                "numeric_semantics": {},
                "precision_sensitive": True,
                "mutation_scope": "WEIGHTS_ONLY",
            })
    graph = {
        "schema": "beglin-operator-graph-v1",
        "architecture_id": arch,
        "operators": ops,
        "unsupported_primitives": sorted(set(unsupported)),
    }
    graph["operator_graph_sha256"] = stable_identity_sha256(graph)
    return graph


def build_tokenizer_contract(source: Mapping[str, Any], descriptor: Mapping[str, Any]) -> dict:
    arch = str(descriptor["architecture_id"])
    files = [Path(p).name for p in source.get("tokenizer_paths", [])]
    family = ARCH_REGISTRY.get(arch, {}).get("tokenizer_family", "UNKNOWN")
    if arch in {"qwen2", "qwen3_moe", "llama", "olmoe"}:
        status = "IN_ENGINE_VERIFIED" if files else "IMPLEMENTED_UNVERIFIED"
        encode_backend = "beglin_bpe"
    elif arch == "gpt-oss":
        status = "EXTERNAL_VERIFIED"
        encode_backend = "tiktoken_o200k_harmony"
    elif arch == "deepseek_v2":
        status = "UNSUPPORTED"
        encode_backend = None
    else:
        status = "UNSUPPORTED"
        encode_backend = None
    contract = {
        "schema": "beglin-tokenizer-contract-v1",
        "architecture_id": arch,
        "tokenizer_family": family,
        "source_files": sorted(files),
        "status": status,
        "encode_backend": encode_backend,
        "decode_backend": encode_backend,
        "text_io_supported": status in {"IN_ENGINE_VERIFIED", "EXTERNAL_VERIFIED"},
        "silent_fallback_allowed": False,
    }
    contract["tokenizer_contract_sha256"] = stable_identity_sha256(contract)
    return contract


def build_loader_contract(source: Mapping[str, Any], descriptor: Mapping[str, Any]) -> dict:
    arch = str(descriptor["architecture_id"])
    fmt = str(source["source_format"])
    encountered = sorted({
        str(row.get("dtype", "UNKNOWN"))
        for row in source.get("tensor_inventory", [])
    })
    if fmt == "GGUF":
        supported_arch = arch in GGUF_SUPPORTED_ARCH
        unsupported_formats = sorted(
            q for q in encountered
            if q not in LOADER_SUPPORTED_QUANTS
        )
    elif fmt in {"SAFETENSORS_SINGLE", "SAFETENSORS_SHARDED"}:
        supported_arch = arch in SAFETENSORS_SUPPORTED_ARCH
        # Safetensors dtype names (F32/F16/BF16) plus architecture-specific AF paths.
        unsupported_formats = sorted(
            q for q in encountered
            if q not in {"F32", "F16", "BF16", "float32", "float16", "bfloat16", "I8", "U8"}
        )
    else:
        supported_arch = False
        unsupported_formats = encountered
    status = (
        "IMPLEMENTED_UNVERIFIED"
        if supported_arch and not unsupported_formats
        else "UNSUPPORTED"
    )
    contract = {
        "schema": "beglin-loader-contract-v1",
        "architecture_id": arch,
        "source_format": fmt,
        "status": status,
        "mmap_strategy": "MMAP" if fmt == "GGUF" else "SHARD_AWARE",
        "transcode_strategy": "BEGLIN_QNG64_WHEN_ELIGIBLE",
        "cache_strategy": "CONTENT_IDENTITY",
        "encountered_formats": encountered,
        "unsupported_formats": unsupported_formats,
        "silent_dense_fallback_allowed": False,
    }
    contract["loader_contract_sha256"] = stable_identity_sha256(contract)
    return contract


def build_model_skeleton(
    source: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    tensor_graph: Mapping[str, Any],
    operator_graph: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    loader: Mapping[str, Any],
) -> dict:
    n_layers = int(descriptor.get("num_layers") or 0)
    layers = []
    by_layer: dict[int, list[str]] = {}
    for node in tensor_graph.get("nodes", []):
        layer = node.get("layer")
        if layer is None:
            continue
        by_layer.setdefault(int(layer), []).append(str(node["role"]))
    for layer in range(n_layers):
        layers.append({
            "layer_index": layer,
            "tensor_roles": sorted(set(by_layer.get(layer, []))),
            "attention_kind": descriptor.get("attention_kind"),
            "ffn_kind": (
                "MOE"
                if descriptor.get("dense_or_moe") == "MOE"
                and layer >= int(descriptor.get("dense_prefix_layers") or 0)
                else "DENSE"
            ),
        })
    globals_ = sorted(
        node["role"] for node in tensor_graph.get("nodes", [])
        if node.get("role") in GLOBAL_ROLES
    )
    skeleton = {
        "schema": "beglin-model-skeleton-v1",
        "model_id": source["model_id"],
        "model_source_sha256": source["source_manifest_sha256"],
        "architecture_descriptor_sha256": descriptor["architecture_descriptor_sha256"],
        "architecture": descriptor["architecture_id"],
        "layer_count": n_layers,
        "layer_skeletons": layers,
        "global_tensors": globals_,
        "operator_graph_ref": operator_graph["operator_graph_sha256"],
        "tensor_role_graph_ref": tensor_graph["tensor_role_graph_sha256"],
        "tokenizer_contract_ref": tokenizer["tokenizer_contract_sha256"],
        "loader_contract_ref": loader["loader_contract_sha256"],
        "required_primitives": sorted({
            row["operator_type"] for row in operator_graph.get("operators", [])
        }),
        "optional_primitives": [],
        "unsupported_primitives": list(operator_graph.get("unsupported_primitives", [])),
    }
    skeleton["skeleton_sha256"] = stable_identity_sha256(skeleton)
    return skeleton


def build_backend_capability_matrix(
    tensor_graph: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    *,
    requested_backend: str | None = None,
    cpu_runtime_verified: bool = False,
    mlx_runtime_verified: bool = False,
) -> dict:
    if requested_backend not in {None, "cpu", "mlx_metal"}:
        raise ModelCapabilityError(f"unsupported backend request: {requested_backend}")
    rows = []
    for node in tensor_graph.get("nodes", []):
        role = str(node["role"])
        if role == "IGNORE_NON_PRECISION":
            continue
        target = str(node["canonical_target_key"])
        precision_role = role in PRECISION_ROLES
        for backend in ("cpu", "mlx_metal"):
            if requested_backend and backend != requested_backend:
                continue
            if backend == "cpu":
                inference_status = "VERIFIED" if cpu_runtime_verified else "IMPLEMENTED_UNVERIFIED"
                qng64_status = (
                    "VERIFIED" if precision_role and cpu_runtime_verified
                    else ("IMPLEMENTED_UNVERIFIED" if precision_role else "UNSUPPORTED_ROLE")
                )
                mutation_mode = "RESTART_REQUIRED" if precision_role else "IMMUTABLE"
                supported_n = CPU_QNG64_WIDTHS if precision_role else []
            else:
                arch = str(descriptor.get("architecture_id") or "unknown")
                target_verified = bool(mlx_runtime_verified and arch == "deepseek_v2")
                if arch == "gpt-oss":
                    inference_status = "UNSUPPORTED_MODEL"
                    qng64_status = "UNSUPPORTED_MODEL" if precision_role else "UNSUPPORTED_ROLE"
                    mutation_mode = "RESTART_REQUIRED"
                    supported_n = []
                else:
                    inference_status = "VERIFIED" if target_verified else "IMPLEMENTED_UNVERIFIED"
                    qng64_status = (
                        "VERIFIED" if precision_role and target_verified
                        else ("IMPLEMENTED_UNVERIFIED" if precision_role else "UNSUPPORTED_ROLE")
                    )
                    mutation_mode = "HOT_REBIND_CANDIDATE" if precision_role else "RESTART_REQUIRED"
                    supported_n = MLX_QNG64_WIDTHS if precision_role else []
            rows.append({
                "target_key": target,
                "role": role,
                "layer": node.get("layer"),
                "expert_id": node.get("expert_id"),
                "backend": backend,
                "inference_status": inference_status,
                "qng64_status": qng64_status,
                "supported_n": supported_n,
                "mutation_mode": mutation_mode,
                "validation_required": inference_status != "VERIFIED" or qng64_status != "VERIFIED",
                "reason_code": None,
                "evidence_refs": [],
                "verification_source": (
                    "explicit_runtime_evidence" if inference_status == "VERIFIED"
                    else "inspection_only"
                ),
            })
    matrix = {
        "schema": "beglin-backend-capability-v1",
        "requested_backend": requested_backend,
        "rows": sorted(rows, key=lambda r: (r["target_key"], r["backend"])),
    }
    matrix["backend_capability_sha256"] = stable_identity_sha256(matrix)
    return matrix


def build_quant_capability_matrix(
    tensor_graph: Mapping[str, Any],
    backend_matrix: Mapping[str, Any],
) -> dict:
    by_key_backend = {
        (r["target_key"], r["backend"]): r
        for r in backend_matrix.get("rows", [])
    }
    rows = []
    for node in tensor_graph.get("nodes", []):
        role = str(node["role"])
        if role == "IGNORE_NON_PRECISION":
            continue
        for backend in sorted({r["backend"] for r in backend_matrix.get("rows", [])}):
            cap = by_key_backend.get((node["canonical_target_key"], backend))
            if cap is None:
                continue
            rows.append({
                "target_key": node["canonical_target_key"],
                "backend": backend,
                "source_precision": node.get("source_quant_format"),
                "supported_n": list(cap.get("supported_n", [])),
                "qng64_status": cap.get("qng64_status"),
                "int4": role in PRECISION_ROLES,
                "int8": role in PRECISION_ROLES,
                "fp32": role in PRECISION_ROLES or role in {"EMBEDDING", "LM_HEAD"},
                "transcode_required": str(node.get("source_quant_format")) not in {"Q4_0", "Q4_K", "Q5_0", "Q5_K", "Q6_K", "Q8_0"},
                "cacheable": role in PRECISION_ROLES,
            })
    matrix = {
        "schema": "beglin-quant-capability-v1",
        "rows": rows,
    }
    matrix["quant_capability_sha256"] = stable_identity_sha256(matrix)
    return matrix


def build_runtime_mutation_matrix(backend_matrix: Mapping[str, Any]) -> dict:
    rows = []
    for cap in backend_matrix.get("rows", []):
        mode = str(cap["mutation_mode"])
        if mode == "HOT_REBIND_CANDIDATE":
            resolved_mode = (
                "HOT_REBIND_SINGLE"
                if cap["qng64_status"] == "VERIFIED"
                else "IMPLEMENTED_UNVERIFIED"
            )
        else:
            resolved_mode = mode
        rows.append({
            "target_key": cap["target_key"],
            "backend": cap["backend"],
            "mutation_mode": resolved_mode,
            "allowed_target_precisions": cap.get("supported_n", []),
            "max_atomic_targets": 8 if cap["backend"] == "mlx_metal" else 0,
            "requires_quiesce": cap["backend"] == "mlx_metal" and bool(cap.get("supported_n")),
            "requires_snapshot": cap["backend"] == "mlx_metal" and bool(cap.get("supported_n")),
            "epoch_increment": 1 if cap.get("supported_n") else 0,
            "rollback_supported": bool(cap.get("supported_n")),
            "policy_shape_change_allowed": False,
        })
    matrix = {
        "schema": "beglin-runtime-mutation-v1",
        "rows": rows,
    }
    matrix["runtime_mutation_sha256"] = stable_identity_sha256(matrix)
    return matrix


def build_precision_search_space(
    quant_matrix: Mapping[str, Any],
    mutation_matrix: Mapping[str, Any],
) -> list[dict]:
    mutation = {
        (r["target_key"], r["backend"]): r
        for r in mutation_matrix.get("rows", [])
    }
    out = []
    for row in quant_matrix.get("rows", []):
        if not row.get("supported_n"):
            continue
        m = mutation.get((row["target_key"], row["backend"]))
        if not m:
            continue
        out.append({
            "target_key": row["target_key"],
            "backend": row["backend"],
            "supported_n": row["supported_n"],
            "mutation_mode": m["mutation_mode"],
            "requires_validation": row["qng64_status"] != "VERIFIED",
        })
    return sorted(out, key=lambda r: (r["backend"], r["target_key"]))


def pipeline_eligibility(
    descriptor: Mapping[str, Any],
    tensor_graph: Mapping[str, Any],
    operator_graph: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    loader: Mapping[str, Any],
    search_space: list[dict],
) -> dict:
    reasons = []
    if descriptor.get("status") != "VERIFIED_ADAPTER":
        reasons.append("ARCHITECTURE_UNSUPPORTED")
    if operator_graph.get("unsupported_primitives"):
        reasons.append("UNSUPPORTED_OPERATOR")
    if loader.get("status") == "UNSUPPORTED":
        reasons.append("LOADER_UNSUPPORTED")
    if not search_space:
        reasons.append("NO_PRECISION_TARGETS")
    if reasons:
        status = "DENIED"
    else:
        partial = (
            tensor_graph.get("unmapped_tensor_count", 0) > 0
            or tokenizer.get("status") in {"UNSUPPORTED", "IMPLEMENTED_UNVERIFIED"}
            or loader.get("status") != "VERIFIED"
            or any(row.get("requires_validation") for row in search_space)
        )
        status = "PARTIAL" if partial else "FULL"
        if tensor_graph.get("unmapped_tensor_count", 0) > 0:
            reasons.append("UNMAPPED_TENSORS")
        if tokenizer.get("status") in {"UNSUPPORTED", "IMPLEMENTED_UNVERIFIED"}:
            reasons.append("TOKENIZER_NOT_FULLY_VERIFIED")
        if loader.get("status") != "VERIFIED":
            reasons.append("LOADER_NOT_VERIFIED")
        if any(row.get("requires_validation") for row in search_space):
            reasons.append("BACKEND_OR_QNG64_REQUIRES_VALIDATION")
    return {
        "schema": "beglin-pipeline-eligibility-v1",
        "status": status,
        "reasons": sorted(set(reasons)),
        "p8_allowed": status in {"FULL", "PARTIAL"},
        "p9_allowed": status in {"FULL", "PARTIAL"},
        "p10_allowed": status in {"FULL", "PARTIAL"},
        "p11_allowed": status == "FULL",
        "automatic_live_promotion": False,
    }


def build_validation_plan(
    descriptor: Mapping[str, Any],
    tensor_graph: Mapping[str, Any],
    operator_graph: Mapping[str, Any],
    eligibility: Mapping[str, Any],
) -> dict:
    structural_ok = (
        descriptor.get("status") == "VERIFIED_ADAPTER"
        and not operator_graph.get("unsupported_primitives")
    )
    stages = [
        {
            "stage": "V0_STRUCTURAL",
            "status": (
                "PASS" if structural_ok and tensor_graph.get("unmapped_tensor_count", 0) == 0
                else ("PARTIAL" if structural_ok else "FAIL")
            ),
            "required_evidence": ["model_source", "model_skeleton", "tensor_role_graph", "operator_graph"],
        },
        {"stage": "V1_FINITE", "status": "NOT_RUN", "required_evidence": ["finite_logits"]},
        {"stage": "V2_REFERENCE", "status": "NOT_RUN", "required_evidence": ["trusted_reference"]},
        {"stage": "V3_PRECISION", "status": "NOT_RUN", "required_evidence": ["precision_numeric_gate"]},
        {"stage": "V4_RUNTIME_MUTATION", "status": "NOT_RUN", "required_evidence": ["apply_ack_rollback"]},
        {"stage": "V5_PRODUCTION", "status": "NOT_RUN", "required_evidence": ["p8_p11_evidence"]},
    ]
    plan = {
        "schema": "beglin-validation-plan-v1",
        "status": "READY_FOR_VALIDATION" if eligibility.get("status") != "DENIED" else "DENIED",
        "stages": stages,
        "automatic_live_promotion": False,
    }
    plan["validation_plan_sha256"] = stable_identity_sha256(plan)
    return plan


def compile_model_capabilities(
    path: str | Path,
    *,
    backend: str | None = None,
    cpu_runtime_verified: bool = False,
    mlx_runtime_verified: bool = False,
) -> dict:
    source = inspect_model_source(path)
    descriptor = build_architecture_descriptor(source)
    tensor_graph = build_tensor_role_graph(source, descriptor)
    operator_graph = build_operator_graph(descriptor)
    tokenizer = build_tokenizer_contract(source, descriptor)
    loader = build_loader_contract(source, descriptor)
    skeleton = build_model_skeleton(
        source, descriptor, tensor_graph, operator_graph, tokenizer, loader
    )
    backend_matrix = build_backend_capability_matrix(
        tensor_graph, descriptor,
        requested_backend=backend,
        cpu_runtime_verified=cpu_runtime_verified,
        mlx_runtime_verified=mlx_runtime_verified,
    )
    quant_matrix = build_quant_capability_matrix(tensor_graph, backend_matrix)
    mutation_matrix = build_runtime_mutation_matrix(backend_matrix)
    search_space = build_precision_search_space(quant_matrix, mutation_matrix)
    eligibility = pipeline_eligibility(
        descriptor, tensor_graph, operator_graph, tokenizer, loader, search_space
    )
    validation_plan = build_validation_plan(
        descriptor, tensor_graph, operator_graph, eligibility
    )

    bundle = {
        "schema": "beglin-model-capability-bundle-v1",
        "model_id": source["model_id"],
        "checkpoint_identity_sha256": source["checkpoint_identity_sha256"],
        "source_manifest": source,
        "architecture_descriptor": descriptor,
        "model_skeleton": skeleton,
        "tensor_role_graph": tensor_graph,
        "operator_graph": operator_graph,
        "tokenizer_contract": tokenizer,
        "loader_contract": loader,
        "backend_capability_matrix": backend_matrix,
        "quant_capability_matrix": quant_matrix,
        "runtime_mutation_matrix": mutation_matrix,
        "precision_search_targets": search_space,
        "unsupported_targets": tensor_graph.get("unmapped_tensors", []),
        "p8_p11_eligibility": eligibility,
        "validation_plan": validation_plan,
        "automatic_live_promotion": False,
        "production_write_allowed": False,
    }
    bundle["bundle_sha256"] = stable_identity_sha256(bundle)
    return bundle


def capability_summary(bundle: Mapping[str, Any]) -> dict:
    descriptor = bundle["architecture_descriptor"]
    graph = bundle["tensor_role_graph"]
    tokenizer = bundle["tokenizer_contract"]
    loader = bundle["loader_contract"]
    search = bundle["precision_search_targets"]
    counts: dict[str, int] = {}
    for row in search:
        counts[row["backend"]] = counts.get(row["backend"], 0) + 1
    return {
        "model_id": bundle["model_id"],
        "architecture": descriptor["architecture_id"],
        "architecture_status": descriptor["status"],
        "source_format": bundle["source_manifest"]["source_format"],
        "tensor_count": graph["tensor_count"],
        "mapped_tensor_count": graph["mapped_tensor_count"],
        "unmapped_tensor_count": graph["unmapped_tensor_count"],
        "mapping_coverage": graph["mapping_coverage"],
        "tokenizer_status": tokenizer["status"],
        "loader_status": loader["status"],
        "precision_search_target_counts": counts,
        "p8_p11_eligibility": bundle["p8_p11_eligibility"],
        "bundle_sha256": bundle["bundle_sha256"],
    }
