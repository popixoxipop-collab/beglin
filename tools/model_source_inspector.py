#!/usr/bin/env python3
"""P12 read-only model-source inspector.

Supports identity discovery for GGUF and HuggingFace safetensors layouts.
GGUF architecture facts come from the file's own metadata, never its filename.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import gguf_metadata_inspector as gguf
import model_capability as mc


class SourceInspectionError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _row(root: Path, path: Path, kind: str) -> dict:
    return {
        "logical_path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "kind": kind,
    }


def _read_config(root: Path) -> dict[str, Any]:
    path = root / "config.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise SourceInspectionError("config.json is not valid JSON") from exc
    if not isinstance(value, dict):
        raise SourceInspectionError("config.json root must be an object")
    return value


def architecture_name_from_config(config: dict[str, Any]) -> str | None:
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type:
        return model_type
    archs = config.get("architectures")
    if isinstance(archs, list) and archs and isinstance(archs[0], str):
        name = archs[0].lower()
        aliases = {
            "deepseekv2forcausallm": "deepseek_v2",
            "qwen3moeforcausallm": "qwen3_moe",
            "olmoeforcausallm": "olmoe",
            "llamaforcausallm": "llama",
            "qwen2forcausallm": "qwen2",
        }
        return aliases.get(name, name)
    return None


def inspect_model_source(path: str | Path, *, model_id: str | None = None, model_revision: str = "local") -> dict:
    source = Path(path).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != ".gguf":
            raise SourceInspectionError("single-file inspection currently accepts GGUF only")
        root = source.parent
        manifest = mc.build_model_source_manifest(
            model_id=model_id or source.stem,
            model_revision=model_revision,
            source_format="GGUF",
            files=[_row(root, source, "gguf")],
            root_path=str(root),
        )
        try:
            inventory = gguf.inspect_gguf_header(source)
        except gguf.GgufInspectionError as exc:
            raise SourceInspectionError(str(exc)) from exc
        return {
            "manifest": manifest,
            "architecture_source_name": inventory.get("architecture"),
            "config": gguf.architecture_facts_from_gguf(inventory),
            "gguf_inventory": inventory,
        }

    if not source.is_dir():
        raise SourceInspectionError(f"model path does not exist: {source}")

    config = _read_config(source)
    files = []
    if (source / "config.json").is_file():
        files.append(_row(source, source / "config.json", "config"))

    index = source / "model.safetensors.index.json"
    if index.is_file():
        try:
            idx = json.loads(index.read_text())
            weight_map = idx["weight_map"]
        except Exception as exc:
            raise SourceInspectionError("invalid model.safetensors.index.json") from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise SourceInspectionError("safetensors weight_map must be a non-empty object")
        shard_names = sorted(set(str(v) for v in weight_map.values()))
        files.append(_row(source, index, "safetensors_index"))
        for shard in shard_names:
            if "/" in shard or "\\" in shard or shard.startswith("."):
                raise SourceInspectionError(f"unsafe shard name: {shard!r}")
            shard_path = source / shard
            if not shard_path.is_file():
                raise SourceInspectionError(f"missing safetensors shard: {shard}")
            files.append(_row(source, shard_path, "safetensors_shard"))
        fmt = "SAFETENSORS_SHARDED"
    else:
        shards = sorted(source.glob("*.safetensors"))
        if len(shards) == 1:
            files.append(_row(source, shards[0], "safetensors"))
            fmt = "SAFETENSORS_SINGLE"
        elif len(shards) > 1:
            raise SourceInspectionError("multiple safetensors files require model.safetensors.index.json")
        else:
            raise SourceInspectionError("no supported model weight source found")

    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        p = source / name
        if p.is_file():
            files.append(_row(source, p, "tokenizer"))

    manifest = mc.build_model_source_manifest(
        model_id=model_id or source.name,
        model_revision=model_revision,
        source_format=fmt,
        files=files,
        root_path=str(source),
    )
    return {
        "manifest": manifest,
        "architecture_source_name": architecture_name_from_config(config),
        "config": config,
        "gguf_inventory": None,
    }
