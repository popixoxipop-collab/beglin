#!/usr/bin/env python3
"""P12 loader/quant-format capability resolver.

The registry reports what loader code paths exist. It does not claim a
checkpoint is numerically VERIFIED without explicit evidence.
"""
from __future__ import annotations

from typing import Any, Mapping

import model_capability as mc


GGUF_FORMATS = (
    "F32", "F16", "BF16", "Q4_0", "Q5_0", "Q8_0",
    "Q3_K", "Q4_K", "Q5_K", "Q6_K", "MXFP4",
)
GGUF_UNSUPPORTED = ("Q2_K", "IQ_SERIES")
SAFETENSORS_FORMATS = ("F32", "F16", "BF16")


def resolve_loader_contract(
    *,
    source_format: str,
    architecture_id: str,
    source_files: list[Mapping[str, Any]],
    verified_evidence: list[Mapping[str, Any]] | None = None,
) -> dict:
    evidence = list(verified_evidence or [])
    fmt = str(source_format)
    arch = str(architecture_id)

    if fmt == "GGUF":
        supported = list(GGUF_FORMATS)
        unsupported = list(GGUF_UNSUPPORTED)
        status = "VERIFIED" if evidence else "IMPLEMENTED_UNVERIFIED"
        strategy = "mmap+architecture_adapter"
    elif fmt in {"SAFETENSORS_SINGLE", "SAFETENSORS_SHARDED"}:
        supported = list(SAFETENSORS_FORMATS)
        unsupported = []
        known = arch in {"qwen2", "llama", "deepseek_v2", "qwen3_moe", "olmoe", "gpt-oss"}
        status = ("VERIFIED" if evidence else "IMPLEMENTED_UNVERIFIED") if known else "PARTIAL"
        strategy = "indexed_shards" if fmt == "SAFETENSORS_SHARDED" else "single_mmap"
    elif fmt == "LEGACY_BEG_LIN":
        supported = ["BEGLIN_LEGACY"]
        unsupported = []
        status = "VERIFIED" if evidence else "IMPLEMENTED_UNVERIFIED"
        strategy = "legacy_manifest"
    else:
        supported = []
        unsupported = [fmt]
        status = "UNSUPPORTED"
        strategy = "none"

    out = mc.build_loader_contract(
        source_format=fmt,
        status=status,
        supported_formats=supported,
        unsupported_formats=unsupported,
        evidence_refs=evidence,
    )
    out["architecture_id"] = arch
    out["strategy"] = strategy
    out["source_file_count"] = len(source_files)
    out["transcode_required"] = fmt == "GGUF"
    out["silent_dense_fallback_allowed"] = False
    return out
