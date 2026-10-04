#!/usr/bin/env python3
"""Header-only GGUF v2/v3 metadata and tensor descriptor inspector.

The implementation mirrors Beglin's existing gguf_load.c container contract:
all reads are bounds checked, nested arrays are refused, and tensor payloads are
never read. Unknown GGML tensor type ids are preserved as explicit unknowns.
"""
from __future__ import annotations

import os
from pathlib import Path
import struct
from typing import Any


class GgufInspectionError(RuntimeError):
    pass


_VTYPE = {
    0: ("UINT8", 1, "<B"),
    1: ("INT8", 1, "<b"),
    2: ("UINT16", 2, "<H"),
    3: ("INT16", 2, "<h"),
    4: ("UINT32", 4, "<I"),
    5: ("INT32", 4, "<i"),
    6: ("FLOAT32", 4, "<f"),
    7: ("BOOL", 1, "<?"),
    8: ("STRING", None, None),
    9: ("ARRAY", None, None),
    10: ("UINT64", 8, "<Q"),
    11: ("INT64", 8, "<q"),
    12: ("FLOAT64", 8, "<d"),
}

_GGML_TYPE_NAMES = {
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

_MAX_STRING_BYTES = 64 * 1024 * 1024
_MAX_ARRAY_ITEMS = 100_000_000
_MAX_TENSORS = 10_000_000
_MAX_KV = 10_000_000


class _Reader:
    def __init__(self, path: Path):
        self.path = path
        self.size = path.stat().st_size
        self.handle = path.open("rb")

    def close(self):
        self.handle.close()

    @property
    def pos(self) -> int:
        return self.handle.tell()

    def read(self, n: int) -> bytes:
        if n < 0 or self.pos + n > self.size:
            raise GgufInspectionError(
                f"truncated GGUF: need {n} bytes at {self.pos}, size={self.size}"
            )
        raw = self.handle.read(n)
        if len(raw) != n:
            raise GgufInspectionError("short GGUF read")
        return raw

    def skip(self, n: int) -> None:
        if n < 0 or self.pos + n > self.size:
            raise GgufInspectionError(
                f"GGUF skip out of bounds: {n} bytes at {self.pos}"
            )
        self.handle.seek(n, os.SEEK_CUR)

    def scalar(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))[0]

    def u32(self) -> int:
        return int(self.scalar("<I"))

    def u64(self) -> int:
        return int(self.scalar("<Q"))

    def string(self) -> str:
        n = self.u64()
        if n > _MAX_STRING_BYTES:
            raise GgufInspectionError(f"GGUF string is unreasonably large: {n}")
        raw = self.read(n)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GgufInspectionError("GGUF string is not UTF-8") from exc


def _read_scalar(reader: _Reader, type_id: int) -> Any:
    info = _VTYPE.get(type_id)
    if info is None:
        raise GgufInspectionError(f"unknown GGUF value type id: {type_id}")
    name, width, fmt = info
    if name == "STRING":
        return reader.string()
    if name == "ARRAY":
        raise GgufInspectionError("nested/standalone ARRAY scalar is invalid")
    assert width is not None and fmt is not None
    return reader.scalar(fmt)


def _read_or_skip_value(reader: _Reader, type_id: int, *, materialize: bool) -> dict:
    info = _VTYPE.get(type_id)
    if info is None:
        raise GgufInspectionError(f"unknown GGUF value type id: {type_id}")
    type_name, width, fmt = info

    if type_name != "ARRAY":
        if materialize:
            return {"type": type_name, "is_array": False, "value": _read_scalar(reader, type_id)}
        if type_name == "STRING":
            n = reader.u64()
            if n > _MAX_STRING_BYTES:
                raise GgufInspectionError(f"GGUF string is unreasonably large: {n}")
            reader.skip(n)
        else:
            assert width is not None
            reader.skip(width)
        return {"type": type_name, "is_array": False}

    elem_type = reader.u32()
    elem_info = _VTYPE.get(elem_type)
    if elem_info is None:
        raise GgufInspectionError(f"unknown GGUF array element type id: {elem_type}")
    elem_name, elem_width, _ = elem_info
    if elem_name == "ARRAY":
        raise GgufInspectionError("nested GGUF arrays are unsupported")
    count = reader.u64()
    if count > _MAX_ARRAY_ITEMS:
        raise GgufInspectionError(f"GGUF array item count is unreasonably large: {count}")

    # Array bodies are summarized only. This is enough for tokenizer vocab size
    # and source capability inspection without materializing millions of tokens.
    if elem_name == "STRING":
        for _ in range(count):
            n = reader.u64()
            if n > _MAX_STRING_BYTES:
                raise GgufInspectionError(f"GGUF string array element too large: {n}")
            reader.skip(n)
    else:
        assert elem_width is not None
        reader.skip(count * elem_width)
    return {
        "type": elem_name,
        "is_array": True,
        "array_length": int(count),
    }


_MATERIALIZE_KEYS = {
    "general.architecture",
    "general.name",
    "general.alignment",
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
}


def inspect_gguf_header(path: str | Path) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise GgufInspectionError(f"GGUF path is not a file: {p}")

    r = _Reader(p)
    try:
        if r.read(4) != b"GGUF":
            raise GgufInspectionError("bad GGUF magic")
        version = r.u32()
        if version not in {2, 3}:
            raise GgufInspectionError(f"unsupported GGUF version: {version}")
        tensor_count = r.u64()
        kv_count = r.u64()
        if tensor_count > _MAX_TENSORS:
            raise GgufInspectionError(f"unreasonable tensor_count={tensor_count}")
        if kv_count > _MAX_KV:
            raise GgufInspectionError(f"unreasonable kv_count={kv_count}")

        metadata: dict[str, dict] = {}
        scalar_metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = r.string()
            type_id = r.u32()
            materialize = key in _MATERIALIZE_KEYS or (
                key.endswith((
                    ".context_length",
                    ".embedding_length",
                    ".block_count",
                    ".feed_forward_length",
                    ".attention.head_count",
                    ".attention.head_count_kv",
                    ".expert_count",
                    ".expert_used_count",
                    ".expert_shared_count",
                    ".attention.sliding_window",
                ))
            )
            value = _read_or_skip_value(r, type_id, materialize=materialize)
            metadata[key] = value
            if materialize and not value.get("is_array") and "value" in value:
                scalar_metadata[key] = value["value"]

        tensors = []
        seen = set()
        for _ in range(tensor_count):
            name = r.string()
            if name in seen:
                raise GgufInspectionError(f"duplicate GGUF tensor name: {name}")
            seen.add(name)
            n_dims = r.u32()
            if n_dims < 1 or n_dims > 4:
                raise GgufInspectionError(
                    f"tensor {name!r} has invalid n_dims={n_dims}"
                )
            shape = [r.u64() for _ in range(n_dims)]
            type_id = r.u32()
            rel_offset = r.u64()
            tensors.append({
                "name": name,
                "shape": [int(v) for v in shape],
                "ggml_type_id": int(type_id),
                "ggml_type": _GGML_TYPE_NAMES.get(type_id, f"UNKNOWN_{type_id}"),
                "relative_data_offset": int(rel_offset),
            })

        return {
            "schema": "beglin-gguf-header-inventory-v1",
            "version": int(version),
            "tensor_count": int(tensor_count),
            "kv_count": int(kv_count),
            "architecture": scalar_metadata.get("general.architecture"),
            "name": scalar_metadata.get("general.name"),
            "metadata": metadata,
            "scalar_metadata": scalar_metadata,
            "tensors": tensors,
            "header_end_offset": r.pos,
            "file_size": r.size,
        }
    finally:
        r.close()


def architecture_facts_from_gguf(inventory: dict) -> dict:
    architecture = str(inventory.get("architecture") or "")
    meta = inventory.get("scalar_metadata") or {}

    def first(*suffixes):
        for suffix in suffixes:
            key = f"{architecture}.{suffix}"
            if key in meta:
                return meta[key]
        return None

    facts = {}
    candidates = {
        "hidden_size": first("embedding_length"),
        "intermediate_size": first("feed_forward_length"),
        "num_hidden_layers": first("block_count"),
        "num_attention_heads": first("attention.head_count"),
        "num_key_value_heads": first("attention.head_count_kv"),
        "max_position_embeddings": first("context_length"),
        "num_experts": first("expert_count"),
        "num_experts_per_tok": first("expert_used_count"),
        "n_shared_experts": first("expert_shared_count"),
        "sliding_window": first("attention.sliding_window"),
    }
    for key, value in candidates.items():
        if value is not None:
            facts[key] = value

    tokens = (inventory.get("metadata") or {}).get("tokenizer.ggml.tokens")
    if isinstance(tokens, dict) and tokens.get("is_array"):
        facts["vocab_size"] = int(tokens["array_length"])
    return facts
