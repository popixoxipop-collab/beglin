#!/usr/bin/env python3
"""Manual one-target canary approval contract (dry-run only).

This module validates that a human-issued approval is bound to one exact
proposal/evidence/budget context.  It intentionally contains no production
mutation path.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Mapping


HEX64 = re.compile(r"^[0-9a-f]{64}$")
DRY_RUN_MODE = "dry_run"
DRY_RUN_SIGNATURE = "DRY_RUN_TEST_ONLY"
PRODUCTION_SIGNATURE = "TRUSTED_PRODUCTION_SIGNATURE"


class ManualCanaryContractError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def parse_time(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManualCanaryContractError(f"invalid timestamp: {value!r}") from exc
    if dt.tzinfo is None:
        raise ManualCanaryContractError("timestamps must include a timezone")
    return dt.astimezone(timezone.utc)


def require_hex64(name: str, value: str) -> str:
    value = str(value).lower()
    if not HEX64.fullmatch(value):
        raise ManualCanaryContractError(f"{name} must be 64 lowercase hex chars")
    return value


def _positive_int(name: str, value: Any) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ManualCanaryContractError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ManualCanaryContractError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class TargetDelta:
    role: str
    layer: int
    before_n: int
    after_n: int

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "TargetDelta":
        try:
            target = cls(
                role=str(value["role"]),
                layer=int(value["layer"]),
                before_n=int(value["before_n"]),
                after_n=int(value["after_n"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManualCanaryContractError("invalid single_target") from exc
        if not target.role:
            raise ManualCanaryContractError("target role must be non-empty")
        if target.layer < 0:
            raise ManualCanaryContractError("target layer must be non-negative")
        if target.before_n <= 0 or target.after_n <= 0:
            raise ManualCanaryContractError("target n values must be positive")
        if target.before_n == target.after_n:
            raise ManualCanaryContractError("target before_n and after_n must differ")
        return target

    def payload(self) -> dict:
        return {
            "role": self.role,
            "layer": self.layer,
            "before_n": self.before_n,
            "after_n": self.after_n,
        }


def normalize_budget(value: Mapping[str, Any]) -> dict:
    return {
        "max_requests": _positive_int("max_requests", value.get("max_requests")),
        "max_tokens": _positive_int("max_tokens", value.get("max_tokens")),
        "max_duration_ms": _positive_int("max_duration_ms", value.get("max_duration_ms")),
        "max_memory_bytes": _positive_int("max_memory_bytes", value.get("max_memory_bytes")),
    }


def normalize_evidence_refs(value: list[Mapping[str, Any]]) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ManualCanaryContractError("evidence_refs must be a non-empty list")
    out = []
    for row in value:
        try:
            kind = str(row["kind"])
            run_id = str(row["run_id"])
            sha256 = require_hex64("evidence sha256", row["sha256"])
        except (KeyError, TypeError) as exc:
            raise ManualCanaryContractError("invalid evidence_refs row") from exc
        if kind not in {"G4_A_B_R", "G6_RESTART_CANARY"}:
            raise ManualCanaryContractError(f"unsupported evidence kind: {kind}")
        if not run_id:
            raise ManualCanaryContractError("evidence run_id must be non-empty")
        out.append({"kind": kind, "run_id": run_id, "sha256": sha256})
    kinds = {row["kind"] for row in out}
    if kinds != {"G4_A_B_R", "G6_RESTART_CANARY"}:
        raise ManualCanaryContractError("approval requires both G4 and G6 evidence")
    return sorted(out, key=lambda r: (r["kind"], r["run_id"], r["sha256"]))


def normalize_proposal(value: Mapping[str, Any]) -> dict:
    if value.get("mode") != DRY_RUN_MODE:
        raise ManualCanaryContractError("manual canary implementation is dry-run only")
    if value.get("production_write_allowed") is not False:
        raise ManualCanaryContractError("production_write_allowed must be false")

    proposal = {
        "schema": "manual-canary-proposal-v1",
        "mode": DRY_RUN_MODE,
        "production_write_allowed": False,
        "proposal_id": str(value.get("proposal_id", "")),
        "proposer": str(value.get("proposer", "")),
        "environment_id": str(value.get("environment_id", "")),
        "model_revision": str(value.get("model_revision", "")),
        "backend": str(value.get("backend", "")),
        "architecture": str(value.get("architecture", "")),
        "source_commit": str(value.get("source_commit", "")),
        "binary_sha256": require_hex64("binary_sha256", value.get("binary_sha256", "")),
        "checkpoint_sha256": require_hex64("checkpoint_sha256", value.get("checkpoint_sha256", "")),
        "baseline_policy_hash": require_hex64("baseline_policy_hash", value.get("baseline_policy_hash", "")),
        "candidate_policy_hash": require_hex64("candidate_policy_hash", value.get("candidate_policy_hash", "")),
        "single_target": TargetDelta.from_value(value.get("single_target", {})).payload(),
        "evidence_refs": normalize_evidence_refs(value.get("evidence_refs")),
        "budget": normalize_budget(value.get("budget", {})),
        "expected_epoch": int(value.get("expected_epoch")),
        "restart_instance_id": str(value.get("restart_instance_id", "")),
        "kill_switch_scope": str(value.get("kill_switch_scope", "")),
        "rollback_plan": str(value.get("rollback_plan", "")),
    }
    for name in (
        "proposal_id", "proposer", "environment_id", "model_revision", "backend",
        "architecture", "source_commit", "restart_instance_id", "kill_switch_scope",
        "rollback_plan",
    ):
        if not proposal[name]:
            raise ManualCanaryContractError(f"{name} must be non-empty")
    if proposal["expected_epoch"] < 0:
        raise ManualCanaryContractError("expected_epoch must be non-negative")
    if proposal["baseline_policy_hash"] == proposal["candidate_policy_hash"]:
        raise ManualCanaryContractError("baseline and candidate policy hashes must differ")
    return proposal


def proposal_digest(value: Mapping[str, Any]) -> str:
    return sha256_json(normalize_proposal(value))


def normalize_approval(value: Mapping[str, Any]) -> dict:
    approval = {
        "schema": "manual-canary-approval-v1",
        "approval_id": str(value.get("approval_id", "")),
        "proposal_digest": require_hex64("proposal_digest", value.get("proposal_digest", "")),
        "issuer": str(value.get("issuer", "")),
        "issued_at": str(value.get("issued_at", "")),
        "expires_at": str(value.get("expires_at", "")),
        "nonce": str(value.get("nonce", "")),
        "mode": str(value.get("mode", "")),
        "signature_status": str(value.get("signature_status", "")),
        "production_write_allowed": value.get("production_write_allowed"),
    }
    for name in ("approval_id", "issuer", "issued_at", "expires_at", "nonce"):
        if not approval[name]:
            raise ManualCanaryContractError(f"{name} must be non-empty")
    if approval["mode"] != DRY_RUN_MODE:
        raise ManualCanaryContractError("only dry_run approvals are accepted")
    if approval["production_write_allowed"] is not False:
        raise ManualCanaryContractError("approval production_write_allowed must be false")
    if approval["signature_status"] != DRY_RUN_SIGNATURE:
        raise ManualCanaryContractError(
            "dry-run approvals require signature_status=DRY_RUN_TEST_ONLY; "
            "this contract is untrusted for production"
        )
    issued = parse_time(approval["issued_at"])
    expires = parse_time(approval["expires_at"])
    if expires <= issued:
        raise ManualCanaryContractError("approval expires_at must be after issued_at")
    return approval


def validate_approval(
    *,
    proposal: Mapping[str, Any],
    approval: Mapping[str, Any],
    now: datetime,
    consumed_nonces: set[str] | None = None,
) -> tuple[dict, dict]:
    p = normalize_proposal(proposal)
    a = normalize_approval(approval)
    digest = sha256_json(p)
    if a["proposal_digest"] != digest:
        raise ManualCanaryContractError("approval proposal_digest mismatch")
    if a["issuer"] == p["proposer"]:
        raise ManualCanaryContractError("self-approval is not allowed")
    now = now.astimezone(timezone.utc)
    issued = parse_time(a["issued_at"])
    expires = parse_time(a["expires_at"])
    if now < issued:
        raise ManualCanaryContractError("approval is not valid yet")
    if now >= expires:
        raise ManualCanaryContractError("approval has expired")
    if consumed_nonces is not None and a["nonce"] in consumed_nonces:
        raise ManualCanaryContractError("approval nonce was already consumed")
    return p, a
