#!/usr/bin/env python3
"""P12 tokenizer capability resolver.

Resolution is intentionally conservative. Architecture support means an adapter
exists; without checkpoint-bound evidence the resolver does not label a
checkpoint VERIFIED.
"""
from __future__ import annotations

from typing import Any, Mapping

import model_capability as mc


_ARCH_TOKENIZERS = {
    "qwen2": {"family": "QWEN2_BPE", "implementation": "bpe_tokenizer"},
    "llama": {"family": "LLAMA_BPE", "implementation": "bpe_tokenizer"},
    "olmoe": {"family": "OLMO_BPE", "implementation": "bpe_tokenizer"},
    "qwen3_moe": {"family": "QWEN2_BPE", "implementation": "bpe_tokenizer"},
    "gpt-oss": {"family": "O200K_HARMONY", "implementation": "external_tiktoken"},
    "deepseek_v2": {"family": "DEEPSEEK_TOKENIZER", "implementation": "unimplemented"},
}


def resolve_tokenizer_contract(
    *,
    model_id: str,
    architecture_id: str,
    source_files: list[Mapping[str, Any]],
    vocab_size: int | None = None,
    verified_evidence: list[Mapping[str, Any]] | None = None,
) -> dict:
    profile = _ARCH_TOKENIZERS.get(str(architecture_id))
    evidence = list(verified_evidence or [])
    tokenizer_files = sorted(
        str(row["logical_path"])
        for row in source_files
        if str(row.get("kind")) == "tokenizer"
    )

    if profile is None:
        family = "UNKNOWN"
        status = "UNSUPPORTED"
    elif profile["implementation"] == "unimplemented":
        family = profile["family"]
        status = "UNSUPPORTED"
    elif evidence:
        # Only the caller that owns real checkpoint-bound evidence may elevate.
        status = "EXTERNAL_VERIFIED" if profile["implementation"] == "external_tiktoken" else "IN_ENGINE_VERIFIED"
        family = profile["family"]
    else:
        family = profile["family"]
        status = "IMPLEMENTED_UNVERIFIED"

    out = mc.build_tokenizer_contract(
        tokenizer_id=f"{model_id}:{family.lower()}",
        tokenizer_family=family,
        status=status,
        vocab_size=vocab_size,
        evidence_refs=evidence,
    )
    out["implementation"] = profile["implementation"] if profile else "none"
    out["source_files"] = tokenizer_files
    out["text_io_ready"] = status in {"IN_ENGINE_VERIFIED", "EXTERNAL_VERIFIED"}
    out["requires_external_runtime"] = bool(profile and profile["implementation"] == "external_tiktoken")
    return out


def tokenizer_architectures() -> list[str]:
    return sorted(_ARCH_TOKENIZERS)
