#!/usr/bin/env python3
"""Backend adapter contract for precision automation G1."""

from dataclasses import dataclass
from pathlib import Path

import precision_context as pc


@dataclass(frozen=True)
class BackendAdapter:
    name: str
    state_subdir: str
    requires_gpu: bool
    promotion_file_env: str
    promotion_source_env: str

    def state_dir(self, root, model_revision_id):
        return pc.backend_state_dir(root, model_revision_id, self.state_subdir)

    def validate_context(self, context):
        cid = pc.context_id(context)
        backend = context["execution"]["backend"]
        if backend != self.name:
            raise pc.ContextMismatch(
                f"adapter {self.name!r} cannot consume context for {backend!r}"
            )
        return cid

    def validate_evidence(self, evidence, context, *, preimage_policy=None):
        self.validate_context(context)
        return pc.assert_evidence_compatible(
            evidence,
            context,
            preimage_policy=preimage_policy,
        )


CPU = BackendAdapter(
    name="cpu",
    state_subdir="cpu",
    requires_gpu=False,
    promotion_file_env="QWEN_MOE_PROMOTION_FILE_NQ",
    promotion_source_env="QWEN_MOE_PROMOTION_SAFETENSORS",
)

MLX_METAL = BackendAdapter(
    name="mlx_metal",
    state_subdir="mlx_metal",
    requires_gpu=True,
    promotion_file_env="QWEN_MOE_PROMOTION_FILE_NQ",
    promotion_source_env="QWEN_MOE_PROMOTION_SAFETENSORS",
)

_ADAPTERS = {
    CPU.name: CPU,
    MLX_METAL.name: MLX_METAL,
}


def get_backend_adapter(name):
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown backend {name!r}") from exc


def backend_scoped_paths(root, model_revision_id, backend):
    adapter = get_backend_adapter(backend)
    base = adapter.state_dir(root, model_revision_id)
    return {
        "root": str(base),
        "desired_policy": str(base / "desired_policy.json"),
        "applied_state": str(base / "applied_state.json"),
        "quarantine": str(base / "quarantine.json"),
        "journal": str(base / "journal"),
        "runs": str(base / "runs"),
    }


def admission_evidence_ok(evidence, context, *, preimage_policy=None):
    """Fail closed; only exact backend/context evidence is admission-compatible."""
    adapter = get_backend_adapter(context["execution"]["backend"])
    try:
        adapter.validate_evidence(
            evidence,
            context,
            preimage_policy=preimage_policy,
        )
    except (pc.ContextMismatch, ValueError, KeyError):
        return False
    return (
        evidence.get("status") in {"passed", "qualified_for_scope"}
        and evidence.get("pass") is True
    )


def assert_backend_state_path(path, root, model_revision_id, backend):
    expected = Path(backend_scoped_paths(root, model_revision_id, backend)["root"]).resolve()
    actual = Path(path).resolve()
    if actual != expected and expected not in actual.parents:
        raise pc.ContextMismatch(
            f"path {actual} escapes backend state root {expected}"
        )
    return actual


class BackendCapabilityError(ValueError):
    pass


QUANT_CAPABILITIES = {
    "cpu": {
        "q4g64": {"native": {4}},
        "q8g64": {"native": {8}},
        "qng64_g64_ef_v1": {"native": {2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
        "dense": {"native": {16, 32}},
    },
    "mlx_metal": {
        "q4g64": {"mlx_native": {4}},
        "q8g64": {"mlx_native_repack": {8}},
        "qng64_g64_ef_v1": {
            "mlx_native_repack": {2, 3, 5, 6},
            "custom_metal_bitplane": {7, 9, 10, 11, 12, 13, 14, 15},
        },
        "dense": {"mlx_dense": {16, 32}},
    },
}


def normalize_backend_name(name):
    value = str(name).strip().lower()
    aliases = {"mlx": "mlx_metal", "metal": "mlx_metal", "gpu": "mlx_metal"}
    value = aliases.get(value, value)
    if value not in pc.VALID_BACKENDS:
        raise BackendCapabilityError(f"unsupported backend: {name!r}")
    return value


def quant_path(backend, quant_format, n):
    backend = normalize_backend_name(backend)
    n = int(n)
    formats = QUANT_CAPABILITIES[backend]
    if quant_format not in formats:
        raise BackendCapabilityError(
            f"{backend} does not implement quant_format={quant_format}"
        )
    for path, widths in formats[quant_format].items():
        if n in widths:
            return path
    supported = sorted({w for widths in formats[quant_format].values() for w in widths})
    raise BackendCapabilityError(
        f"{backend} does not support {quant_format} n={n}; supported={supported}"
    )


def validate_candidate(backend, quant_format, n):
    return {
        "backend": normalize_backend_name(backend),
        "quant_format": str(quant_format),
        "n": int(n),
        "path": quant_path(backend, quant_format, n),
    }


def initial_auto_ladder(backend, quant_format="qng64_g64_ef_v1"):
    backend = normalize_backend_name(backend)
    out = []
    for n in (5, 6, 7):
        try:
            quant_path(backend, quant_format, n)
        except BackendCapabilityError:
            continue
        out.append(n)
    return out
