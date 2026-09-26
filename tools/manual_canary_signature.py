#!/usr/bin/env python3
"""Trusted human approval signature verification for Agent E.

The verifier uses OpenSSH detached signatures and an allowed-signers trust
anchor. It verifies authorization metadata for the already-normalized dry-run
proposal, but it does not enable production mutation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

import manual_canary_contract as mc


SSH_NAMESPACE = "beglin-manual-canary-v1"
DEFAULT_SSH_KEYGEN = Path("/usr/bin/ssh-keygen")


class HumanSignatureError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _parse_time(value: str) -> datetime:
    return mc.parse_time(value)


def normalize_human_approval_metadata(value: Mapping[str, Any]) -> dict:
    out = {
        "schema": "manual-canary-human-approval-v1",
        "approval_id": str(value.get("approval_id", "")),
        "issuer": str(value.get("issuer", "")),
        "principal": str(value.get("principal", "")),
        "issued_at": str(value.get("issued_at", "")),
        "expires_at": str(value.get("expires_at", "")),
        "nonce": str(value.get("nonce", "")),
        "mode": str(value.get("mode", "")),
        "production_write_allowed": value.get("production_write_allowed"),
    }
    for key in ("approval_id", "issuer", "principal", "issued_at", "expires_at", "nonce"):
        if not out[key]:
            raise HumanSignatureError(f"{key} must be non-empty")
    if out["issuer"] != out["principal"]:
        raise HumanSignatureError("issuer must equal the trusted SSH principal")
    if out["mode"] != mc.DRY_RUN_MODE:
        raise HumanSignatureError("trusted signature layer currently accepts dry_run only")
    if out["production_write_allowed"] is not False:
        raise HumanSignatureError("production_write_allowed must remain false")
    issued = _parse_time(out["issued_at"])
    expires = _parse_time(out["expires_at"])
    if expires <= issued:
        raise HumanSignatureError("expires_at must be after issued_at")
    return out


def signing_payload(proposal: Mapping[str, Any], approval_metadata: Mapping[str, Any]) -> bytes:
    p = mc.normalize_proposal(proposal)
    a = normalize_human_approval_metadata(approval_metadata)
    if a["issuer"] == p["proposer"]:
        raise HumanSignatureError("self-approval is not allowed")
    payload = {
        "schema": "manual-canary-human-signing-payload-v1",
        "proposal_digest": mc.sha256_json(p),
        "approval": a,
    }
    return canonical_json(payload).encode()


def _validate_allowed_signers(path: Path) -> Path:
    path = path.expanduser()
    if path.is_symlink():
        raise HumanSignatureError("allowed-signers file may not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise HumanSignatureError("allowed-signers trust anchor is not a regular file")
    if resolved.stat().st_size <= 0:
        raise HumanSignatureError("allowed-signers trust anchor is empty")
    return resolved


def _validate_ssh_keygen(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise HumanSignatureError("ssh-keygen path must be absolute")
    if path.name != "ssh-keygen":
        raise HumanSignatureError("verification executable must be ssh-keygen")
    if not path.is_file() or not os.access(path, os.X_OK):
        raise HumanSignatureError(f"ssh-keygen is not executable: {path}")
    return path


def verify_human_signature(
    *,
    proposal: Mapping[str, Any],
    approval_metadata: Mapping[str, Any],
    signature_text: str,
    allowed_signers_path: str | Path,
    now: datetime,
    consumed_nonces: set[str] | None = None,
    ssh_keygen_path: str | Path = DEFAULT_SSH_KEYGEN,
) -> dict:
    p = mc.normalize_proposal(proposal)
    a = normalize_human_approval_metadata(approval_metadata)
    payload = signing_payload(p, a)
    now = now.astimezone(timezone.utc)
    issued = _parse_time(a["issued_at"])
    expires = _parse_time(a["expires_at"])
    if now < issued:
        raise HumanSignatureError("approval is not valid yet")
    if now >= expires:
        raise HumanSignatureError("approval has expired")
    if consumed_nonces is not None and a["nonce"] in consumed_nonces:
        raise HumanSignatureError("approval nonce was already consumed")
    if "BEGIN SSH SIGNATURE" not in str(signature_text):
        raise HumanSignatureError("signature is not an armored OpenSSH signature")

    trust = _validate_allowed_signers(Path(allowed_signers_path))
    ssh_keygen = _validate_ssh_keygen(Path(ssh_keygen_path))

    with tempfile.TemporaryDirectory(prefix="beglin-human-signature-") as td:
        sig = Path(td) / "approval.sig"
        sig.write_text(str(signature_text))
        proc = subprocess.run(
            [
                str(ssh_keygen),
                "-Y", "verify",
                "-f", str(trust),
                "-I", a["principal"],
                "-n", SSH_NAMESPACE,
                "-s", str(sig),
            ],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if proc.returncode != 0:
        raise HumanSignatureError(
            "OpenSSH signature verification failed"
        )

    return {
        "schema": "manual-canary-human-signature-verification-v1",
        "status": "VERIFIED",
        "production_write_allowed": False,
        "proposal_digest": mc.sha256_json(p),
        "approval_id": a["approval_id"],
        "issuer": a["issuer"],
        "principal": a["principal"],
        "nonce": a["nonce"],
        "issued_at": a["issued_at"],
        "expires_at": a["expires_at"],
        "namespace": SSH_NAMESPACE,
        "signed_payload_sha256": sha256_bytes(payload),
        "allowed_signers_sha256": sha256_file(trust),
        "signature_sha256": sha256_bytes(str(signature_text).encode()),
    }
