#!/usr/bin/env python3
"""Content-bound, repeatable checkpoint identity verification.

This verifier is deliberately independent from the production-shadow cycle.
It reads a safetensors index (or one safetensors file), hashes every referenced
shard with stable pre/post stat checks, repeats the full content pass, and
requires both identity manifests to match.

The final checkpoint identity is compatible with the existing
checkpoint-identity-v1 manifest used by gpu_shadow_materialize.py.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Callable


IDENTITY_SCHEMA = "checkpoint-identity-v1"
VERIFICATION_SCHEMA = "checkpoint-verification-v1"
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024


class CheckpointIdentityError(RuntimeError):
    pass


class InputChangedError(CheckpointIdentityError):
    pass


@dataclass(frozen=True)
class FileStat:
    dev: int
    ino: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _stat(path: Path) -> FileStat:
    st = path.stat()
    return FileStat(
        dev=int(st.st_dev),
        ino=int(st.st_ino),
        size=int(st.st_size),
        mtime_ns=int(st.st_mtime_ns),
        ctime_ns=int(st.st_ctime_ns),
    )


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _identity_hash(manifest: dict) -> str:
    return hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _hash_file_stable(
    path: Path,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> tuple[str, FileStat]:
    before = _stat(path)
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
            total += len(chunk)
    after = _stat(path)
    if before != after:
        raise InputChangedError(
            f"file changed while hashing: {path.name}; "
            f"before={asdict(before)} after={asdict(after)}"
        )
    if total != before.size:
        raise InputChangedError(
            f"file size changed or short read while hashing: {path.name}; "
            f"expected={before.size} read={total}"
        )
    return h.hexdigest(), after


def _read_index_stable(path: Path) -> tuple[bytes, FileStat]:
    before = _stat(path)
    data = path.read_bytes()
    after = _stat(path)
    if before != after:
        raise InputChangedError(
            f"index changed while reading: {path.name}; "
            f"before={asdict(before)} after={asdict(after)}"
        )
    if len(data) != before.size:
        raise InputChangedError(
            f"index size changed or short read: {path.name}; "
            f"expected={before.size} read={len(data)}"
        )
    return data, after


def _resolve_shard(
    index_path: Path,
    name: str,
    *,
    allowed_root: Path,
) -> Path:
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts:
        raise CheckpointIdentityError(
            f"unsafe shard path in safetensors index: {name!r}"
        )
    raw = index_path.parent.joinpath(*rel.parts)
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CheckpointIdentityError(
            f"safetensors shard referenced by index is missing: {raw}"
        ) from exc
    if not resolved.is_file():
        raise CheckpointIdentityError(
            f"safetensors shard is not a regular file: {raw}"
        )
    if not _is_within(resolved, allowed_root):
        raise CheckpointIdentityError(
            f"safetensors shard resolves outside approved root: "
            f"name={name!r} resolved={resolved} root={allowed_root}"
        )
    return resolved


def checkpoint_manifest_once(
    checkpoint: str | os.PathLike,
    *,
    allowed_root: str | os.PathLike | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> tuple[str, dict, dict]:
    """Build one content identity pass plus non-hashed verification metadata."""
    requested = Path(checkpoint).expanduser()
    try:
        path = requested.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CheckpointIdentityError(
            f"checkpoint artifact does not exist: {requested}"
        ) from exc
    if not path.is_file():
        raise CheckpointIdentityError(f"checkpoint artifact is not a file: {path}")

    approved = (
        Path(allowed_root).expanduser().resolve(strict=True)
        if allowed_root is not None
        else path.parent.resolve(strict=True)
    )
    if not approved.is_dir():
        raise CheckpointIdentityError(
            f"approved checkpoint root is not a directory: {approved}"
        )
    if not _is_within(path, approved):
        raise CheckpointIdentityError(
            f"checkpoint artifact resolves outside approved root: "
            f"path={path} root={approved}"
        )

    observed_files: list[dict] = []

    if path.name.endswith(".index.json"):
        index_bytes, index_stat = _read_index_stable(path)
        try:
            obj = json.loads(index_bytes)
        except json.JSONDecodeError as exc:
            raise CheckpointIdentityError(
                f"invalid safetensors index JSON: {path}: {exc}"
            ) from exc
        weight_map = obj.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointIdentityError(
                "safetensors index must contain a non-empty weight_map"
            )
        shard_names = []
        for tensor, value in weight_map.items():
            if not isinstance(tensor, str) or not isinstance(value, str) or not value:
                raise CheckpointIdentityError(
                    "safetensors weight_map keys and shard names must be non-empty strings"
                )
            shard_names.append(value)
        shard_names = sorted(set(shard_names))

        index_sha = hashlib.sha256(index_bytes).hexdigest()
        shards = []
        observed_files.append(
            {
                "name": path.name,
                "role": "index",
                "stat": asdict(index_stat),
                "sha256": index_sha,
            }
        )
        for name in shard_names:
            shard = _resolve_shard(path, name, allowed_root=approved)
            shard_sha, shard_stat = _hash_file_stable(
                shard,
                chunk_bytes=chunk_bytes,
            )
            shards.append(
                {
                    "name": name,
                    "size_bytes": shard_stat.size,
                    "sha256": shard_sha,
                }
            )
            observed_files.append(
                {
                    "name": name,
                    "role": "shard",
                    "stat": asdict(shard_stat),
                    "sha256": shard_sha,
                }
            )

        # Re-read the index after all shards to ensure the source map itself
        # did not change during the pass.
        index_bytes_after, index_stat_after = _read_index_stable(path)
        if index_bytes_after != index_bytes or index_stat_after != index_stat:
            raise InputChangedError(
                f"safetensors index changed during checkpoint pass: {path.name}"
            )

        manifest = {
            "schema": IDENTITY_SCHEMA,
            "kind": "safetensors-index",
            "index": {
                "name": path.name,
                "size_bytes": index_stat.size,
                "sha256": index_sha,
            },
            "shards": shards,
        }
    else:
        file_sha, file_stat = _hash_file_stable(
            path,
            chunk_bytes=chunk_bytes,
        )
        manifest = {
            "schema": IDENTITY_SCHEMA,
            "kind": "single-file",
            "file": {
                "name": path.name,
                "size_bytes": file_stat.size,
                "sha256": file_sha,
            },
        }
        observed_files.append(
            {
                "name": path.name,
                "role": "file",
                "stat": asdict(file_stat),
                "sha256": file_sha,
            }
        )

    identity = _identity_hash(manifest)
    audit = {
        "checkpoint_path": str(path),
        "approved_root": str(approved),
        "files": observed_files,
        "bytes_per_pass": sum(
            int(row["stat"]["size"]) for row in observed_files
        ),
    }
    return identity, manifest, audit


def verify_checkpoint_identity(
    checkpoint: str | os.PathLike,
    *,
    output: str | os.PathLike | None = None,
    allowed_root: str | os.PathLike | None = None,
    passes: int = 2,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    between_passes: Callable[[int], None] | None = None,
) -> dict:
    if passes < 2:
        raise CheckpointIdentityError("verification requires at least two full passes")
    if passes > 3:
        raise CheckpointIdentityError("verification passes are bounded to at most 3")
    if chunk_bytes < 4096:
        raise CheckpointIdentityError("chunk_bytes must be at least 4096")

    pass_rows = []
    first_identity = None
    first_manifest = None
    first_files = None

    for pass_index in range(passes):
        identity, manifest, audit = checkpoint_manifest_once(
            checkpoint,
            allowed_root=allowed_root,
            chunk_bytes=chunk_bytes,
        )
        pass_rows.append(
            {
                "pass": pass_index + 1,
                "checkpoint_sha256": identity,
                "bytes_hashed": audit["bytes_per_pass"],
                "files": audit["files"],
            }
        )
        if first_identity is None:
            first_identity = identity
            first_manifest = manifest
            first_files = audit["files"]
        else:
            if identity != first_identity or manifest != first_manifest:
                raise InputChangedError(
                    "checkpoint identity changed between verification passes"
                )
            if audit["files"] != first_files:
                raise InputChangedError(
                    "checkpoint file metadata/content changed between verification passes"
                )

        if between_passes is not None and pass_index + 1 < passes:
            between_passes(pass_index + 1)

    result = {
        "schema": VERIFICATION_SCHEMA,
        "status": "VERIFIED",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": {
            "file_hash": "sha256",
            "identity_manifest_schema": IDENTITY_SCHEMA,
            "identity_manifest_encoding": "canonical-json-sort-keys-ascii",
            "passes": passes,
            "chunk_bytes": chunk_bytes,
        },
        "checkpoint_sha256": first_identity,
        "identity_manifest": first_manifest,
        "verification_passes": pass_rows,
        "production_write_allowed": False,
    }
    if output is not None:
        _atomic_json(Path(output).expanduser(), result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--allowed-root")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    args = ap.parse_args()

    try:
        result = verify_checkpoint_identity(
            args.checkpoint,
            output=args.output,
            allowed_root=args.allowed_root,
            passes=args.passes,
            chunk_bytes=args.chunk_bytes,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": VERIFICATION_SCHEMA,
                    "status": "VERIFY_ERROR",
                    "production_write_allowed": False,
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2

    print(
        json.dumps(
            {
                "schema": VERIFICATION_SCHEMA,
                "status": result["status"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "passes": len(result["verification_passes"]),
                "output": str(Path(args.output).expanduser()),
                "production_write_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
