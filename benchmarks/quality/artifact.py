from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

RUN_SCHEMA = "beglin-quality-run/1"
COMPARABLE_IDENTITY_FIELDS = (
    "source_commit",
    "source_tree",
    "binary_sha256",
    "checkpoint_sha256",
    "model_id",
    "hardware_id",
    "backend",
    "seed",
    "prompt_corpus_sha256",
    "runtime_config_hash",
)
SUPPORTED_PRECISIONS = {"bf16", "fp16", "safe_baseline", "n4", "n5", "n6", "n7"}


class QualityArtifactError(ValueError):
    pass


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(stable_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_hex(value: str, lengths: tuple[int, ...]) -> bool:
    return len(value) in lengths and all(ch in "0123456789abcdef" for ch in value)


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QualityArtifactError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise QualityArtifactError(f"{name} must be an array")
    return value


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualityArtifactError(f"{name} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QualityArtifactError(f"{name} must be a non-negative integer")
    return value


def _finite_optional(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualityArtifactError(f"{name} must be numeric or null")
    out = float(value)
    if not math.isfinite(out):
        raise QualityArtifactError(f"{name} must be finite")
    return out


def validate_identity(raw: Any) -> dict[str, Any]:
    identity = _require_dict(raw, "identity")
    required = set(COMPARABLE_IDENTITY_FIELDS)
    missing = sorted(required - set(identity))
    if missing:
        raise QualityArtifactError(f"identity missing required fields: {missing}")

    out = dict(identity)
    out["source_commit"] = _require_str(out["source_commit"], "identity.source_commit")
    if not _is_hex(out["source_commit"], (40, 64)):
        raise QualityArtifactError("identity.source_commit must be lowercase 40/64-hex")

    out["source_tree"] = _require_str(out["source_tree"], "identity.source_tree")
    if not _is_hex(out["source_tree"], (40, 64)):
        raise QualityArtifactError("identity.source_tree must be lowercase 40/64-hex")

    for key in ("binary_sha256", "checkpoint_sha256", "prompt_corpus_sha256", "runtime_config_hash"):
        value = _require_str(out[key], f"identity.{key}")
        if not _is_hex(value, (64,)):
            raise QualityArtifactError(f"identity.{key} must be lowercase 64-hex")
        out[key] = value

    for key in ("model_id", "hardware_id", "backend"):
        out[key] = _require_str(out[key], f"identity.{key}")

    out["seed"] = _require_nonnegative_int(out["seed"], "identity.seed")

    policy_hash = out.get("policy_hash")
    if policy_hash is not None:
        policy_hash = _require_str(policy_hash, "identity.policy_hash")
        if not _is_hex(policy_hash, (64,)):
            raise QualityArtifactError("identity.policy_hash must be lowercase 64-hex")
    out["policy_hash"] = policy_hash
    return out


def validate_variant(raw: Any) -> dict[str, Any]:
    variant = _require_dict(raw, "variant")
    kind = variant.get("kind")
    if kind not in {"reference", "candidate"}:
        raise QualityArtifactError("variant.kind must be reference or candidate")
    precision = variant.get("precision")
    if precision not in SUPPORTED_PRECISIONS:
        raise QualityArtifactError(f"unsupported precision: {precision!r}")
    n = variant.get("n")
    if kind == "reference":
        if n is not None:
            raise QualityArtifactError("reference variant.n must be null")
    else:
        if n not in {4, 5, 6, 7}:
            raise QualityArtifactError("candidate variant.n must be one of 4,5,6,7")
        if precision != f"n{n}":
            raise QualityArtifactError("candidate precision must match n")
    return {"kind": kind, "precision": precision, "n": n}


def _validate_activation(raw: Any, prefix: str) -> dict[str, Any]:
    item = _require_dict(raw, prefix)
    role = _require_str(item.get("role"), f"{prefix}.role")
    layer = _require_nonnegative_int(item.get("layer"), f"{prefix}.layer")
    expert_id = item.get("expert_id")
    if expert_id is not None:
        expert_id = _require_nonnegative_int(expert_id, f"{prefix}.expert_id")
    mean_abs = _finite_optional(item.get("mean_abs"), f"{prefix}.mean_abs")
    rms = _finite_optional(item.get("rms"), f"{prefix}.rms")
    if mean_abs is not None and mean_abs < 0:
        raise QualityArtifactError(f"{prefix}.mean_abs must be >= 0")
    if rms is not None and rms < 0:
        raise QualityArtifactError(f"{prefix}.rms must be >= 0")
    return {
        "role": role,
        "layer": layer,
        "expert_id": expert_id,
        "mean_abs": mean_abs,
        "rms": rms,
    }


def _validate_router(raw: Any, prefix: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    item = _require_dict(raw, prefix)
    near_tie_count = item.get("near_tie_count")
    decision_count = item.get("decision_count")
    near_tie_rate = item.get("near_tie_rate")
    if near_tie_count is not None:
        near_tie_count = _require_nonnegative_int(near_tie_count, f"{prefix}.near_tie_count")
    if decision_count is not None:
        decision_count = _require_nonnegative_int(decision_count, f"{prefix}.decision_count")
    near_tie_rate = _finite_optional(near_tie_rate, f"{prefix}.near_tie_rate")
    if near_tie_rate is not None and not 0 <= near_tie_rate <= 1:
        raise QualityArtifactError(f"{prefix}.near_tie_rate must be within [0,1]")
    if near_tie_count is not None and decision_count is not None and near_tie_count > decision_count:
        raise QualityArtifactError(f"{prefix}.near_tie_count exceeds decision_count")
    return {
        "near_tie_count": near_tie_count,
        "decision_count": decision_count,
        "near_tie_rate": near_tie_rate,
    }


def validate_request(raw: Any, index: int) -> dict[str, Any]:
    prefix = f"requests[{index}]"
    item = _require_dict(raw, prefix)
    request_id = _require_str(item.get("request_id"), f"{prefix}.request_id")
    prompt_id = _require_str(item.get("prompt_id"), f"{prefix}.prompt_id")
    repetition = _require_nonnegative_int(item.get("repetition"), f"{prefix}.repetition")
    context_tokens = _require_nonnegative_int(item.get("context_tokens"), f"{prefix}.context_tokens")

    output_token_ids = item.get("output_token_ids")
    if output_token_ids is not None:
        output_token_ids = _require_list(output_token_ids, f"{prefix}.output_token_ids")
        for token_index, token in enumerate(output_token_ids):
            _require_nonnegative_int(token, f"{prefix}.output_token_ids[{token_index}]")

    output_text = item.get("output_text")
    if output_text is not None and not isinstance(output_text, str):
        raise QualityArtifactError(f"{prefix}.output_text must be string or null")

    finite_logits = item.get("finite_logits")
    if finite_logits is not None and not isinstance(finite_logits, bool):
        raise QualityArtifactError(f"{prefix}.finite_logits must be boolean or null")

    token_logprobs = item.get("token_logprobs")
    if token_logprobs is not None:
        token_logprobs = _require_list(token_logprobs, f"{prefix}.token_logprobs")
        converted = []
        for i, value in enumerate(token_logprobs):
            parsed = _finite_optional(value, f"{prefix}.token_logprobs[{i}]")
            if parsed is None:
                raise QualityArtifactError(f"{prefix}.token_logprobs cannot contain null")
            converted.append(parsed)
        token_logprobs = converted

    nll_sum = _finite_optional(item.get("nll_sum"), f"{prefix}.nll_sum")
    nll_token_count = item.get("nll_token_count")
    if nll_token_count is not None:
        nll_token_count = _require_nonnegative_int(nll_token_count, f"{prefix}.nll_token_count")
        if nll_token_count == 0:
            raise QualityArtifactError(f"{prefix}.nll_token_count must be > 0 when present")
    if (nll_sum is None) != (nll_token_count is None):
        raise QualityArtifactError(f"{prefix} must provide both nll_sum and nll_token_count")
    if nll_sum is not None and nll_sum < 0:
        raise QualityArtifactError(f"{prefix}.nll_sum must be >= 0")

    corrections = item.get("corrections")
    if corrections is not None:
        corrections = _require_nonnegative_int(corrections, f"{prefix}.corrections")
    promotions = item.get("promotions")
    if promotions is not None:
        promotions = _require_nonnegative_int(promotions, f"{prefix}.promotions")

    quality_score = _finite_optional(item.get("quality_score"), f"{prefix}.quality_score")
    if quality_score is not None and not 0 <= quality_score <= 1:
        raise QualityArtifactError(f"{prefix}.quality_score must be within [0,1]")

    activations = [
        _validate_activation(value, f"{prefix}.activation_summaries[{i}]")
        for i, value in enumerate(
            _require_list(item.get("activation_summaries", []), f"{prefix}.activation_summaries")
        )
    ]
    router = _validate_router(item.get("router"), f"{prefix}.router")

    return {
        "request_id": request_id,
        "prompt_id": prompt_id,
        "repetition": repetition,
        "context_tokens": context_tokens,
        "output_token_ids": output_token_ids,
        "output_text": output_text,
        "finite_logits": finite_logits,
        "token_logprobs": token_logprobs,
        "nll_sum": nll_sum,
        "nll_token_count": nll_token_count,
        "corrections": corrections,
        "promotions": promotions,
        "activation_summaries": activations,
        "router": router,
        "quality_score": quality_score,
    }


def validate_run_artifact(raw: Any) -> dict[str, Any]:
    artifact = _require_dict(raw, "artifact")
    if artifact.get("schema") != RUN_SCHEMA:
        raise QualityArtifactError(f"artifact.schema must be {RUN_SCHEMA}")

    run_id = _require_str(artifact.get("run_id"), "run_id")
    variant = validate_variant(artifact.get("variant"))
    identity = validate_identity(artifact.get("identity"))
    requests_raw = _require_list(artifact.get("requests"), "requests")
    if not requests_raw:
        raise QualityArtifactError("requests must not be empty")
    requests = [validate_request(value, i) for i, value in enumerate(requests_raw)]

    seen = set()
    for item in requests:
        key = (item["prompt_id"], item["repetition"])
        if key in seen:
            raise QualityArtifactError(f"duplicate prompt/repetition: {key}")
        seen.add(key)

    canonical = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "variant": variant,
        "identity": identity,
        "requests": requests,
    }
    canonical["artifact_hash"] = sha256_json(canonical)
    return canonical


def load_run_artifact(path: Path) -> dict[str, Any]:
    return validate_run_artifact(json.loads(path.read_text(encoding="utf-8")))


def compare_identity(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left = reference["identity"]
    right = candidate["identity"]
    differing = [
        field for field in COMPARABLE_IDENTITY_FIELDS
        if left.get(field) != right.get(field)
    ]
    return {
        "compatible": not differing,
        "differing_fields": differing,
        "reference": {field: left.get(field) for field in COMPARABLE_IDENTITY_FIELDS},
        "candidate": {field: right.get(field) for field in COMPARABLE_IDENTITY_FIELDS},
    }
