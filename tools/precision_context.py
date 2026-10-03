#!/usr/bin/env python3
"""Backend-scoped immutable execution context and policy hashing for precision v3."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, Mapping


SUPPORTED_BACKENDS = {"cpu", "mlx_metal"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def normalize_policy(policy: Mapping[tuple[str, int], int] | list[dict]) -> list[dict]:
    if isinstance(policy, list):
        rows = [
            {"role": str(r["role"]), "layer": int(r["layer"]), "n": int(r["n"])}
            for r in policy
        ]
    else:
        rows = [
            {"role": str(role), "layer": int(layer), "n": int(n)}
            for (role, layer), n in policy.items()
        ]
    return sorted(rows, key=lambda r: (r["role"], r["layer"], r["n"]))


def policy_hash(policy: Mapping[tuple[str, int], int] | list[dict]) -> str:
    return sha256_json(normalize_policy(policy))


@dataclass(frozen=True)
class ExecutionContext:
    schema: str
    model_id: str
    architecture: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    base_artifact_sha256: str
    backend: str
    device_fingerprint: str
    binary_sha256: str
    build_manifest_sha256: str
    kernel_revision: str
    execution_mode: str
    runtime_config_sha256: str
    quant_format: str
    group_size: int
    correction_mode: str

    def __post_init__(self):
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported backend: {self.backend}")
        for name in (
            "checkpoint_sha256", "tokenizer_sha256", "base_artifact_sha256",
            "binary_sha256", "build_manifest_sha256", "runtime_config_sha256",
        ):
            value = getattr(self, name)
            if value and (
                len(value) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in value)
            ):
                raise ValueError(f"{name} must be a SHA-256 hex string")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")

    def payload(self) -> dict:
        return asdict(self)

    @property
    def context_hash(self) -> str:
        return sha256_json(self.payload())


def require_same_context(expected_hash: str, actual: ExecutionContext) -> None:
    if actual.context_hash != expected_hash:
        raise RuntimeError(
            f"execution context mismatch: expected={expected_hash} "
            f"actual={actual.context_hash}"
        )


def require_backend(context: ExecutionContext, backend: str) -> None:
    if context.backend != backend:
        raise RuntimeError(
            f"backend evidence mismatch: evidence={context.backend} requested={backend}"
        )
