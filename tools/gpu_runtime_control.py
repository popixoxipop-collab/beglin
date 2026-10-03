#!/usr/bin/env python3
"""Durable file bridge for the in-process MLX precision transaction protocol.

The C runtime owns the actual quiescent transition.  This controller only emits
one-target commands after proving that the runtime ACK still matches the
planner's expected GPU epoch, policy preimage and target precision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import precision_context as pc


ACK_SCHEMA = "gpu-precision-applied-v1"
BACKEND = "mlx_metal"
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")
TERMINAL_STATUSES = {
    "PROMOTION_APPLIED",
    "ROLLBACK_APPLIED",
    "REBIND_APPLIED",
    "REBIND_FAILED",
    "STALE_COMMAND",
}
SUPPORTED_QNG64 = {2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15}


class RuntimeControlError(RuntimeError):
    pass


class StaleRuntimeState(RuntimeControlError):
    pass


def _read_json(path: str | os.PathLike) -> dict:
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise RuntimeControlError("runtime ACK must be a JSON object")
    return value


def _fsync_parent(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    fd = os.open(str(path.parent), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_text(path: str | os.PathLike, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        data = text.encode()
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            if n <= 0:
                raise RuntimeControlError("short write while publishing GPU txn")
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    _fsync_parent(path)


def _validate_sha256(value: str, field: str) -> str:
    value = str(value)
    if len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise RuntimeControlError(f"{field} must be a SHA-256 hex string")
    return value.lower()


def normalize_runtime_ack(value: dict) -> dict:
    """Validate one runtime ACK object and attach deterministic evidence hashes."""
    if not isinstance(value, dict):
        raise RuntimeControlError("runtime ACK must be a JSON object")
    ack = dict(value)
    raw_ack_sha256 = hashlib.sha256(
        pc.canonical_json(ack).encode()
    ).hexdigest()
    if ack.get("schema") != ACK_SCHEMA:
        raise RuntimeControlError(
            f"unexpected ACK schema: {ack.get('schema')!r}"
        )
    if ack.get("backend") != BACKEND:
        raise RuntimeControlError(
            f"unexpected ACK backend: {ack.get('backend')!r}"
        )
    correction_mode = ack.get("correction_mode")
    if correction_mode not in {"off", "on"}:
        raise RuntimeControlError(
            f"unexpected ACK correction_mode: {correction_mode!r}"
        )
    try:
        epoch = int(ack["weight_epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeControlError("ACK weight_epoch is missing/invalid") from exc
    if epoch < 0:
        raise RuntimeControlError("ACK weight_epoch must be non-negative")

    policy = ack.get("active_policy")
    if not isinstance(policy, list):
        raise RuntimeControlError("ACK active_policy must be a list")
    try:
        normalized = pc.normalize_policy(policy)
    except Exception as exc:
        raise RuntimeControlError(f"invalid ACK active_policy: {exc}") from exc

    # One active row per role/layer.  Duplicates make the runtime state
    # ambiguous even if canonical hashing would otherwise be deterministic.
    keys = [(r["role"], r["layer"]) for r in normalized]
    if len(keys) != len(set(keys)):
        raise RuntimeControlError("ACK active_policy contains duplicate role/layer")

    return {
        **ack,
        "weight_epoch": epoch,
        "active_policy": normalized,
        "active_policy_hash": pc.policy_hash(normalized),
        "ack_sha256": raw_ack_sha256,
    }


def read_runtime_ack(path: str | os.PathLike) -> dict:
    return normalize_runtime_ack(_read_json(path))


def _target_n(policy: list[dict], role: str, layer: int) -> int | None:
    for row in policy:
        if row["role"] == role and int(row["layer"]) == int(layer):
            return int(row["n"])
    return None


def prepare_demote(
    *,
    ack_path: str | os.PathLike,
    txn_path: str | os.PathLike,
    txn_id: str,
    expected_epoch: int,
    expected_policy_hash: str,
    role: str,
    layer: int,
    expected_n: int,
) -> dict:
    if not SAFE_TOKEN.fullmatch(str(txn_id)):
        raise RuntimeControlError("txn_id contains unsupported characters")
    if not SAFE_TOKEN.fullmatch(str(role)):
        raise RuntimeControlError("role contains unsupported characters")
    expected_policy_hash = _validate_sha256(
        expected_policy_hash, "expected_policy_hash"
    )
    expected_epoch = int(expected_epoch)
    layer = int(layer)
    expected_n = int(expected_n)
    if expected_epoch < 0 or layer < 0 or expected_n <= 0:
        raise RuntimeControlError("epoch/layer/n values are out of range")

    ack = read_runtime_ack(ack_path)
    if ack["weight_epoch"] != expected_epoch:
        raise StaleRuntimeState(
            f"runtime epoch changed: expected={expected_epoch} "
            f"actual={ack['weight_epoch']}"
        )
    if ack["active_policy_hash"] != expected_policy_hash:
        raise StaleRuntimeState(
            "runtime policy preimage changed: "
            f"expected={expected_policy_hash} "
            f"actual={ack['active_policy_hash']}"
        )
    actual_n = _target_n(ack["active_policy"], role, layer)
    if actual_n != expected_n:
        raise StaleRuntimeState(
            f"runtime target changed: {role}/L{layer} "
            f"expected_n={expected_n} actual_n={actual_n}"
        )

    line = (
        f"DEMOTE {txn_id} {expected_epoch} {expected_n} "
        f"{role} {layer} {expected_policy_hash}\n"
    )
    _atomic_text(txn_path, line)
    return {
        "status": "REQUESTED",
        "backend": BACKEND,
        "txn_id": txn_id,
        "expected_epoch": expected_epoch,
        "expected_policy_hash": expected_policy_hash,
        "role": role,
        "layer": layer,
        "expected_n": expected_n,
        "command": line.rstrip(),
    }


def prepare_rebind(
    *,
    ack_path: str | os.PathLike,
    txn_path: str | os.PathLike,
    txn_id: str,
    expected_epoch: int,
    expected_policy_hash: str,
    role: str,
    layer: int,
    expected_n: int,
    target_n: int,
) -> dict:
    if not SAFE_TOKEN.fullmatch(str(txn_id)):
        raise RuntimeControlError("txn_id contains unsupported characters")
    if not SAFE_TOKEN.fullmatch(str(role)):
        raise RuntimeControlError("role contains unsupported characters")
    expected_policy_hash = _validate_sha256(
        expected_policy_hash, "expected_policy_hash"
    )
    expected_epoch = int(expected_epoch)
    layer = int(layer)
    expected_n = int(expected_n)
    target_n = int(target_n)
    if expected_epoch < 0 or layer < 0:
        raise RuntimeControlError("epoch/layer values are out of range")
    if expected_n not in SUPPORTED_QNG64:
        raise RuntimeControlError(f"expected_n={expected_n} is not supported qNg64")
    if target_n not in SUPPORTED_QNG64:
        raise RuntimeControlError(f"target_n={target_n} is not supported qNg64")
    if target_n == expected_n:
        raise RuntimeControlError("target_n must differ from expected_n")

    ack = read_runtime_ack(ack_path)
    if ack["weight_epoch"] != expected_epoch:
        raise StaleRuntimeState(
            f"runtime epoch changed: expected={expected_epoch} "
            f"actual={ack['weight_epoch']}"
        )
    if ack["active_policy_hash"] != expected_policy_hash:
        raise StaleRuntimeState(
            "runtime policy preimage changed: "
            f"expected={expected_policy_hash} "
            f"actual={ack['active_policy_hash']}"
        )
    actual_n = _target_n(ack["active_policy"], role, layer)
    if actual_n != expected_n:
        raise StaleRuntimeState(
            f"runtime target changed: {role}/L{layer} "
            f"expected_n={expected_n} actual_n={actual_n}"
        )

    line = (
        f"REBIND {txn_id} {expected_epoch} {expected_n} {target_n} "
        f"{role} {layer} {expected_policy_hash}\n"
    )
    _atomic_text(txn_path, line)
    return {
        "status": "REQUESTED",
        "backend": BACKEND,
        "txn_id": txn_id,
        "expected_epoch": expected_epoch,
        "expected_policy_hash": expected_policy_hash,
        "role": role,
        "layer": layer,
        "expected_n": expected_n,
        "target_n": target_n,
        "command": line.rstrip(),
    }


def verify_terminal_ack(
    *,
    ack_path: str | os.PathLike,
    txn_id: str,
    allowed_statuses: set[str] | None = None,
) -> dict:
    ack = read_runtime_ack(ack_path)
    allowed = TERMINAL_STATUSES if allowed_statuses is None else set(allowed_statuses)
    if ack.get("txn_id") != txn_id:
        raise RuntimeControlError(
            f"ACK txn mismatch: expected={txn_id!r} actual={ack.get('txn_id')!r}"
        )
    if ack.get("status") not in allowed:
        raise RuntimeControlError(
            f"ACK status is not terminal/allowed: {ack.get('status')!r}"
        )
    return ack


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-demote")
    p.add_argument("--ack", required=True)
    p.add_argument("--txn-file", required=True)
    p.add_argument("--txn-id", required=True)
    p.add_argument("--expected-epoch", required=True, type=int)
    p.add_argument("--expected-policy-hash", required=True)
    p.add_argument("--role", required=True)
    p.add_argument("--layer", required=True, type=int)
    p.add_argument("--expected-n", required=True, type=int)

    r = sub.add_parser("prepare-rebind")
    r.add_argument("--ack", required=True)
    r.add_argument("--txn-file", required=True)
    r.add_argument("--txn-id", required=True)
    r.add_argument("--expected-epoch", required=True, type=int)
    r.add_argument("--expected-policy-hash", required=True)
    r.add_argument("--role", required=True)
    r.add_argument("--layer", required=True, type=int)
    r.add_argument("--expected-n", required=True, type=int)
    r.add_argument("--target-n", required=True, type=int)

    v = sub.add_parser("verify-ack")
    v.add_argument("--ack", required=True)
    v.add_argument("--txn-id", required=True)

    args = ap.parse_args()
    try:
        if args.command == "prepare-demote":
            result = prepare_demote(
                ack_path=args.ack,
                txn_path=args.txn_file,
                txn_id=args.txn_id,
                expected_epoch=args.expected_epoch,
                expected_policy_hash=args.expected_policy_hash,
                role=args.role,
                layer=args.layer,
                expected_n=args.expected_n,
            )
        elif args.command == "prepare-rebind":
            result = prepare_rebind(
                ack_path=args.ack,
                txn_path=args.txn_file,
                txn_id=args.txn_id,
                expected_epoch=args.expected_epoch,
                expected_policy_hash=args.expected_policy_hash,
                role=args.role,
                layer=args.layer,
                expected_n=args.expected_n,
                target_n=args.target_n,
            )
        else:
            result = verify_terminal_ack(
                ack_path=args.ack,
                txn_id=args.txn_id,
            )
    except RuntimeControlError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
