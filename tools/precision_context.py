#!/usr/bin/env python3
"""Canonical backend-scoped precision context contract (G1/v3).

The context hash is deliberately stricter than a policy hash. A PASS from a
different backend, binary, device, kernel revision, checkpoint, or runtime
configuration is not compatible evidence.
"""

import hashlib
import json
from pathlib import Path
import uuid


SCHEMA_VERSION = "precision-context-v3"
VALID_BACKENDS = {"cpu", "mlx_metal"}


class ContextMismatch(RuntimeError):
    pass


def normalize_backend(value):
    backend = _nonempty("backend", value).strip().lower()
    aliases = {"mlx": "mlx_metal", "metal": "mlx_metal", "gpu": "mlx_metal"}
    backend = aliases.get(backend, backend)
    if backend not in VALID_BACKENDS:
        raise ValueError(f"unsupported backend {backend!r}")
    return backend


def _nonempty(name, value):
    if value is None or str(value).strip() == "":
        raise ValueError(f"{name} must be non-empty")
    return str(value)


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_policy(policy):
    """Return a stable list of role/layer/n rows from dict or row-list input."""
    rows = []
    if isinstance(policy, dict):
        for key, n in policy.items():
            if isinstance(key, tuple) and len(key) == 2:
                role, layer = key
            elif isinstance(key, str) and "/" in key:
                role, layer = key.rsplit("/", 1)
                layer = layer[1:] if layer.startswith("L") else layer
            else:
                raise ValueError(f"unsupported policy key: {key!r}")
            rows.append({
                "role": str(role),
                "layer": int(layer),
                "n": int(n),
            })
    else:
        for row in policy:
            rows.append({
                "role": str(row["role"]),
                "layer": int(row["layer"]),
                "n": int(row["n"]),
            })
    rows.sort(key=lambda r: (r["role"], r["layer"], r["n"]))
    return rows


def policy_hash(policy):
    return sha256_json({
        "schema_version": "precision-policy-v1",
        "rows": normalize_policy(policy),
    })


def build_context(
    *,
    model_id,
    architecture,
    checkpoint_sha256,
    tokenizer_sha256,
    base_artifact_sha256,
    backend,
    device_fingerprint,
    binary_sha256,
    build_manifest_sha256,
    kernel_version,
    precision_mode,
    quant_format,
    group_size,
    encoder_version,
    execution_mode,
    runtime_config,
):
    backend = _nonempty("backend", backend)
    if backend not in VALID_BACKENDS:
        raise ValueError(f"unsupported backend {backend!r}")
    body = {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "model_id": _nonempty("model_id", model_id),
            "architecture": _nonempty("architecture", architecture),
            "checkpoint_sha256": _nonempty("checkpoint_sha256", checkpoint_sha256),
            "tokenizer_sha256": _nonempty("tokenizer_sha256", tokenizer_sha256),
            "base_artifact_sha256": _nonempty("base_artifact_sha256", base_artifact_sha256),
        },
        "execution": {
            "backend": backend,
            "device_fingerprint": _nonempty("device_fingerprint", device_fingerprint),
            "binary_sha256": _nonempty("binary_sha256", binary_sha256),
            "build_manifest_sha256": _nonempty("build_manifest_sha256", build_manifest_sha256),
            "kernel_version": _nonempty("kernel_version", kernel_version),
            "execution_mode": _nonempty("execution_mode", execution_mode),
        },
        "numeric": {
            "precision_mode": _nonempty("precision_mode", precision_mode),
            "quant_format": _nonempty("quant_format", quant_format),
            "group_size": int(group_size),
            "encoder_version": _nonempty("encoder_version", encoder_version),
        },
        "runtime_config": runtime_config,
    }
    if body["numeric"]["group_size"] <= 0:
        raise ValueError("group_size must be positive")
    body["context_id"] = sha256_json(body)
    return body


def context_id(context):
    supplied = context.get("context_id")
    body = dict(context)
    body.pop("context_id", None)
    calculated = sha256_json(body)
    if supplied is not None and supplied != calculated:
        raise ContextMismatch(
            f"context_id does not match canonical context: {supplied} != {calculated}"
        )
    return calculated


def evidence_scope(
    *,
    context,
    run_id,
    role,
    layer,
    n,
    preimage_policy,
    candidate_policy,
):
    return {
        "context_id": context_id(context),
        "backend": context["execution"]["backend"],
        "run_id": _nonempty("run_id", run_id),
        "role": _nonempty("role", role),
        "layer": int(layer),
        "n": int(n),
        "preimage_policy_sha256": policy_hash(preimage_policy),
        "candidate_policy_sha256": policy_hash(candidate_policy),
    }


def assert_evidence_compatible(evidence, context, *, preimage_policy=None):
    expected_context_id = context_id(context)
    expected_backend = context["execution"]["backend"]
    got_context_id = evidence.get("context_id")
    got_backend = evidence.get("backend")
    if got_context_id != expected_context_id:
        raise ContextMismatch(
            f"context mismatch: evidence={got_context_id!r}, expected={expected_context_id!r}"
        )
    if got_backend != expected_backend:
        raise ContextMismatch(
            f"backend mismatch: evidence={got_backend!r}, expected={expected_backend!r}"
        )
    if preimage_policy is not None:
        expected_preimage = policy_hash(preimage_policy)
        got_preimage = evidence.get("preimage_policy_sha256")
        if got_preimage != expected_preimage:
            raise ContextMismatch(
                f"preimage policy mismatch: evidence={got_preimage!r}, expected={expected_preimage!r}"
            )
    return True


def backend_state_dir(root, model_revision_id, backend):
    backend = _nonempty("backend", backend)
    if backend not in VALID_BACKENDS:
        raise ValueError(f"unsupported backend {backend!r}")
    return Path(root) / _nonempty("model_revision_id", model_revision_id) / backend



def validate_context(context):
    context_id(context)
    normalize_backend(context["execution"]["backend"])
    return context


def new_run_id(backend, prefix="precision"):
    backend = normalize_backend(backend)
    return f"{prefix}_{backend}_{uuid.uuid4().hex}"


ContextError = ContextMismatch


def require_evidence_match(evidence, *, backend, context_hash):
    """Compatibility shim: context_hash is the canonical v3 context_id."""
    got_context = evidence.get("context_id")
    got_backend = evidence.get("backend")
    if got_context != context_hash:
        raise ContextMismatch(
            f"context mismatch: evidence={got_context!r}, expected={context_hash!r}"
        )
    if got_backend != normalize_backend(backend):
        raise ContextMismatch(
            f"backend mismatch: evidence={got_backend!r}, expected={backend!r}"
        )
    return True
