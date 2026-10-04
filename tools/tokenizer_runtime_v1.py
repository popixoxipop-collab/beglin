#!/usr/bin/env python3
"""P12 tokenizer runtime planning and explicit external adapter execution.

This module closes the gap between TokenizerContract capability metadata and
an executable tokenizer path without introducing silent fallbacks.

Safety properties:
- unsupported contracts fail closed
- EXTERNAL_VERIFIED is required before an external adapter can execute
- external executables must be absolute, regular, non-symlink files
- executable SHA-256 is bound into the runtime plan
- subprocess execution never uses a shell
- stdin/stdout use one narrow JSON protocol
- in-engine BPE remains an engine text-path capability, not a hidden Python fallback
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXTERNAL_BACKENDS = {
    "tiktoken_o200k_harmony",
    "external_deepseek_reference",
    "sentencepiece_external",
}
IN_ENGINE_BACKENDS = {"beglin_bpe"}
SUPPORTED_OPERATIONS = {"encode", "decode"}


class TokenizerRuntimeError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_sha(name: str, value: Any) -> str:
    out = str(value or "").lower()
    if not HEX64.fullmatch(out):
        raise TokenizerRuntimeError(f"{name} must be sha256 hex")
    return out


def _stable_identity(value: Mapping[str, Any], self_hash_field: str | None = None) -> str:
    payload = dict(value)
    if self_hash_field:
        payload.pop(self_hash_field, None)
    return sha256_json(payload)


def _validate_contract(contract: Mapping[str, Any]) -> dict:
    value = dict(contract)
    if value.get("schema") != "beglin-tokenizer-contract-v1":
        raise TokenizerRuntimeError("unsupported tokenizer contract schema")
    backend = value.get("encode_backend")
    status = str(value.get("status") or "")
    if backend is None or status == "UNSUPPORTED":
        raise TokenizerRuntimeError("tokenizer contract is unsupported")
    if backend not in IN_ENGINE_BACKENDS | EXTERNAL_BACKENDS:
        raise TokenizerRuntimeError(f"unregistered tokenizer backend: {backend!r}")
    contract_sha = _require_sha(
        "tokenizer_contract_sha256", value.get("tokenizer_contract_sha256")
    )
    if contract_sha != _stable_identity(value, "tokenizer_contract_sha256"):
        raise TokenizerRuntimeError("tokenizer contract self-hash mismatch")
    return value


def compile_runtime_plan(
    contract: Mapping[str, Any],
    *,
    external_executable: str | None = None,
    external_executable_sha256: str | None = None,
) -> dict:
    """Compile one deterministic execution plan from a tokenizer contract."""
    value = _validate_contract(contract)
    backend = str(value["encode_backend"])
    status = str(value["status"])

    if backend in IN_ENGINE_BACKENDS:
        if external_executable is not None or external_executable_sha256 is not None:
            raise TokenizerRuntimeError(
                "in-engine tokenizer plan may not bind an external executable"
            )
        plan_status = (
            "READY_IN_ENGINE"
            if status == "IN_ENGINE_VERIFIED"
            else "VALIDATION_REQUIRED"
        )
        plan = {
            "schema": "beglin-tokenizer-runtime-plan-v1",
            "status": plan_status,
            "execution_mode": "ENGINE_TEXT_PATH",
            "tokenizer_contract_sha256": value["tokenizer_contract_sha256"],
            "architecture_id": value["architecture_id"],
            "tokenizer_family": value["tokenizer_family"],
            "encode_backend": backend,
            "decode_backend": str(value.get("decode_backend") or backend),
            "text_io_mode": value.get("text_io_mode"),
            "external_executable": None,
            "external_executable_sha256": None,
            "protocol": None,
            "silent_fallback_allowed": False,
        }
    else:
        if status != "EXTERNAL_VERIFIED":
            raise TokenizerRuntimeError(
                f"external tokenizer backend requires EXTERNAL_VERIFIED, got {status}"
            )
        evidence = value.get("verification_evidence")
        if not isinstance(evidence, Mapping) or not evidence:
            raise TokenizerRuntimeError(
                "external tokenizer requires explicit verification evidence"
            )
        if not external_executable:
            raise TokenizerRuntimeError("external tokenizer executable is required")
        path = Path(external_executable)
        if not path.is_absolute():
            raise TokenizerRuntimeError("external tokenizer executable must be absolute")
        if path.is_symlink():
            raise TokenizerRuntimeError("external tokenizer executable may not be a symlink")
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise TokenizerRuntimeError("external tokenizer executable is missing") from exc
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise TokenizerRuntimeError(
                "external tokenizer executable must be a regular executable file"
            )
        expected_sha = _require_sha(
            "external_executable_sha256", external_executable_sha256
        )
        actual_sha = file_sha256(resolved)
        if actual_sha != expected_sha:
            raise TokenizerRuntimeError(
                "external tokenizer executable SHA mismatch: "
                f"expected={expected_sha} actual={actual_sha}"
            )
        plan = {
            "schema": "beglin-tokenizer-runtime-plan-v1",
            "status": "READY_EXTERNAL",
            "execution_mode": "EXTERNAL_JSON_STDIN",
            "tokenizer_contract_sha256": value["tokenizer_contract_sha256"],
            "architecture_id": value["architecture_id"],
            "tokenizer_family": value["tokenizer_family"],
            "encode_backend": backend,
            "decode_backend": str(value.get("decode_backend") or backend),
            "text_io_mode": value.get("text_io_mode"),
            "external_executable": str(resolved),
            "external_executable_sha256": actual_sha,
            "protocol": "BEGLIN_TOKENIZER_JSON_STDIN_V1",
            "silent_fallback_allowed": False,
        }

    plan["runtime_plan_sha256"] = _stable_identity(plan, "runtime_plan_sha256")
    return plan


def verify_runtime_plan(plan: Mapping[str, Any]) -> dict:
    value = dict(plan)
    if value.get("schema") != "beglin-tokenizer-runtime-plan-v1":
        raise TokenizerRuntimeError("unsupported tokenizer runtime plan schema")
    got = _require_sha("runtime_plan_sha256", value.get("runtime_plan_sha256"))
    actual = _stable_identity(value, "runtime_plan_sha256")
    if got != actual:
        raise TokenizerRuntimeError(
            f"tokenizer runtime plan hash mismatch: expected={got} actual={actual}"
        )
    if value.get("silent_fallback_allowed") is not False:
        raise TokenizerRuntimeError("silent tokenizer fallback must remain disabled")
    return value


def run_external(
    plan: Mapping[str, Any],
    *,
    operation: str,
    text: str | None = None,
    tokens: list[int] | None = None,
    timeout: int = 30,
) -> dict:
    """Execute an explicitly-bound external tokenizer adapter.

    Protocol request:
      {"schema":"beglin-tokenizer-request-v1","operation":"encode","text":"..."}
      {"schema":"beglin-tokenizer-request-v1","operation":"decode","tokens":[...]}

    Protocol response:
      {"schema":"beglin-tokenizer-response-v1","tokens":[...]}
      {"schema":"beglin-tokenizer-response-v1","text":"..."}
    """
    value = verify_runtime_plan(plan)
    if value.get("execution_mode") != "EXTERNAL_JSON_STDIN":
        raise TokenizerRuntimeError(
            "runtime plan is not an executable external tokenizer plan"
        )
    if operation not in SUPPORTED_OPERATIONS:
        raise TokenizerRuntimeError(f"unsupported tokenizer operation: {operation}")
    executable = Path(str(value.get("external_executable") or ""))
    if file_sha256(executable) != value.get("external_executable_sha256"):
        raise TokenizerRuntimeError("external tokenizer executable changed after planning")

    if operation == "encode":
        if not isinstance(text, str) or tokens is not None:
            raise TokenizerRuntimeError("encode requires text and forbids tokens")
        payload = {
            "schema": "beglin-tokenizer-request-v1",
            "operation": "encode",
            "text": text,
        }
    else:
        if text is not None or not isinstance(tokens, list):
            raise TokenizerRuntimeError("decode requires tokens and forbids text")
        normalized = []
        for token in tokens:
            if isinstance(token, bool):
                raise TokenizerRuntimeError("token IDs must be integers")
            try:
                token_id = int(token)
            except (TypeError, ValueError) as exc:
                raise TokenizerRuntimeError("token IDs must be integers") from exc
            if token_id < 0:
                raise TokenizerRuntimeError("token IDs must be non-negative")
            normalized.append(token_id)
        payload = {
            "schema": "beglin-tokenizer-request-v1",
            "operation": "decode",
            "tokens": normalized,
        }

    proc = subprocess.run(
        [str(executable)],
        input=canonical_json(payload),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=max(1, int(timeout)),
    )
    if proc.returncode != 0:
        raise TokenizerRuntimeError(
            "external tokenizer failed with non-zero exit status"
        )
    try:
        response = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise TokenizerRuntimeError("external tokenizer returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise TokenizerRuntimeError("external tokenizer response must be an object")
    if response.get("schema") != "beglin-tokenizer-response-v1":
        raise TokenizerRuntimeError("external tokenizer response schema mismatch")

    if operation == "encode":
        raw_tokens = response.get("tokens")
        if not isinstance(raw_tokens, list):
            raise TokenizerRuntimeError("encode response is missing tokens")
        normalized = []
        for token in raw_tokens:
            if isinstance(token, bool):
                raise TokenizerRuntimeError("encode response token is not an integer")
            try:
                token_id = int(token)
            except (TypeError, ValueError) as exc:
                raise TokenizerRuntimeError(
                    "encode response token is not an integer"
                ) from exc
            if token_id < 0:
                raise TokenizerRuntimeError(
                    "encode response contains a negative token"
                )
            normalized.append(token_id)
        result = {"operation": "encode", "tokens": normalized}
    else:
        decoded = response.get("text")
        if not isinstance(decoded, str):
            raise TokenizerRuntimeError("decode response is missing text")
        result = {"operation": "decode", "text": decoded}

    result.update(
        {
            "schema": "beglin-tokenizer-runtime-result-v1",
            "runtime_plan_sha256": value["runtime_plan_sha256"],
            "tokenizer_contract_sha256": value["tokenizer_contract_sha256"],
            "encode_backend": value["encode_backend"],
        }
    )
    result["result_sha256"] = sha256_json(result)
    return result


def plan_from_bundle(
    bundle: Mapping[str, Any],
    *,
    external_executable: str | None = None,
    external_executable_sha256: str | None = None,
) -> dict:
    contract = bundle.get("tokenizer_contract")
    if not isinstance(contract, Mapping):
        raise TokenizerRuntimeError("model capability bundle has no tokenizer contract")
    return compile_runtime_plan(
        contract,
        external_executable=external_executable,
        external_executable_sha256=external_executable_sha256,
    )
