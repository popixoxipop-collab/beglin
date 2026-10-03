#!/usr/bin/env python3
"""Materialize a READY shadow discovery row into a gpu_shadow_runner spec.

This tool is local-filesystem only. It never SSHes to a production host and
never writes production state. Source paths must already be mirrored/readable
locally through explicit prefix mappings. Generated manifests/specs live only
under the requested shadow input directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex


SCHEMA = "gpu-shadow-spec-v1"


class ShadowMaterializeError(RuntimeError):
    pass


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_json(path: Path, value) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def checkpoint_identity_from_safetensors(safetensors: str):
    """Return a content-bound checkpoint identity and its audit manifest.

    For a Hugging Face safetensors index, the identity binds the index bytes
    plus every distinct shard referenced by weight_map.  For a single
    safetensors file it binds that file directly.  Index entries must be
    relative paths without '..'; symlinked shard files are allowed because
    common model caches use them, but the index itself controls only the
    relative filename.
    """
    path = Path(safetensors).expanduser().resolve(strict=False)
    if not path.is_file():
        raise ShadowMaterializeError(
            f"safetensors checkpoint artifact does not exist: {path}"
        )

    if path.name.endswith(".index.json"):
        try:
            obj = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ShadowMaterializeError(
                f"invalid safetensors index JSON: {path}: {exc}"
            ) from exc
        weight_map = obj.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ShadowMaterializeError(
                "safetensors index must contain a non-empty weight_map"
            )

        shard_names = sorted({str(v) for v in weight_map.values()})
        shards = []
        for name in shard_names:
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ShadowMaterializeError(
                    f"unsafe shard path in safetensors index: {name!r}"
                )
            shard = path.parent.joinpath(*rel.parts)
            if not shard.is_file():
                raise ShadowMaterializeError(
                    f"safetensors shard referenced by index is missing: {shard}"
                )
            shards.append({
                "name": name,
                "size_bytes": shard.stat().st_size,
                "sha256": _sha256_file(shard),
            })

        manifest = {
            "schema": "checkpoint-identity-v1",
            "kind": "safetensors-index",
            "index": {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            },
            "shards": shards,
        }
    else:
        manifest = {
            "schema": "checkpoint-identity-v1",
            "kind": "single-file",
            "file": {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            },
        }

    identity = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
    return identity, manifest


def load_json(path):
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ShadowMaterializeError(f"{path} must contain a JSON object")
    return value


def parse_maps(values):
    mappings = []
    for item in values:
        if "=" not in item:
            raise ShadowMaterializeError(
                f"path map must be FROM=TO, got {item!r}"
            )
        source_raw, dest_raw = item.split("=", 1)
        source = PurePosixPath(source_raw)
        if not source.is_absolute():
            raise ShadowMaterializeError(
                f"path-map source must be an absolute POSIX path: {source_raw!r}"
            )
        dest = Path(dest_raw).expanduser().resolve(strict=False)
        mappings.append((source, dest))
    return sorted(mappings, key=lambda pair: len(str(pair[0])), reverse=True)


def map_path(path: str, mappings) -> Path:
    value = PurePosixPath(path)
    if not value.is_absolute():
        raise ShadowMaterializeError(
            f"source path must be absolute before mapping: {path!r}"
        )
    for source, dest_root in mappings:
        try:
            rel = value.relative_to(source)
        except ValueError:
            continue
        mapped = (dest_root / Path(*rel.parts)).resolve(strict=False)
        if not (mapped == dest_root or _is_within(mapped, dest_root)):
            raise ShadowMaterializeError(
                f"mapped path escapes local mirror root {dest_root}: {path!r}"
            )
        return mapped
    raise ShadowMaterializeError(
        f"no explicit local path mapping for source path {path!r}"
    )


def select_ready(discovery: dict, candidate_id: str | None = None) -> dict:
    if discovery.get("schema") != "gpu-shadow-discovery-v1":
        raise ShadowMaterializeError("unexpected discovery schema")
    ready = discovery.get("ready")
    if not isinstance(ready, list):
        raise ShadowMaterializeError("discovery ready field must be a list")
    if candidate_id:
        matches = [row for row in ready if row.get("candidate_id") == candidate_id]
        if len(matches) != 1:
            raise ShadowMaterializeError(
                f"candidate_id {candidate_id!r} matched {len(matches)} READY rows"
            )
        return matches[0]
    if len(ready) != 1:
        raise ShadowMaterializeError(
            "candidate_id is required unless discovery contains exactly one READY row"
        )
    return ready[0]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or _is_within(a, b) or _is_within(b, a)


def _manifest_entry(path: Path):
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != 1:
        raise ShadowMaterializeError(
            f"G4 source manifest must contain exactly one active entry, got {len(lines)}"
        )
    parts = shlex.split(lines[0])
    if len(parts) != 2:
        raise ShadowMaterializeError(
            "manifest entry must be '<raw-i32-token-file> <max_new_tokens>'"
        )
    try:
        max_new_tokens = int(parts[1])
    except ValueError as exc:
        raise ShadowMaterializeError("manifest max_new_tokens must be integer") from exc
    if max_new_tokens <= 0:
        raise ShadowMaterializeError("manifest max_new_tokens must be positive")
    return parts[0], max_new_tokens


def materialize(
    discovery: dict,
    *,
    candidate_id: str | None,
    path_mappings,
    output_dir: str,
    cwd: str,
    binary: str,
    checkpoint_sha256: str,
    moe_base: str,
    safetensors: str,
    g6_repeats: int = 64,
) -> dict:
    row = select_ready(discovery, candidate_id)
    source_manifest = row.get("provenance", {}).get("manifest")
    if not source_manifest:
        raise ShadowMaterializeError("READY row lacks provenance manifest")

    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    cwd_path = Path(cwd).expanduser().resolve(strict=False)
    if _overlaps(output_dir, cwd_path):
        raise ShadowMaterializeError(
            "shadow input output_dir must not overlap the engine checkout/worktree"
        )

    local_manifest = map_path(source_manifest, path_mappings).resolve(strict=False)
    if not local_manifest.is_file():
        raise ShadowMaterializeError(
            f"mapped source manifest does not exist: {local_manifest}"
        )
    source_token, max_new_tokens = _manifest_entry(local_manifest)
    source_token_path = PurePosixPath(source_token)
    if not source_token_path.is_absolute():
        source_token_path = PurePosixPath(source_manifest).parent / source_token_path
    local_token = map_path(str(source_token_path), path_mappings).resolve(strict=False)
    if not local_token.is_file():
        raise ShadowMaterializeError(
            f"mapped raw token file does not exist: {local_token}"
        )
    token_bytes = local_token.stat().st_size
    if token_bytes <= 0 or token_bytes % 4:
        raise ShadowMaterializeError(
            f"raw token file must be non-empty int32 data, size={token_bytes}"
        )
    prompt_len = token_bytes // 4

    event = row.get("event") or {}
    pos = int(event.get("pos", -1))
    if pos < 0:
        raise ShadowMaterializeError("READY row has invalid event position")
    if g6_repeats <= 0 or g6_repeats > 100:
        raise ShadowMaterializeError("g6_repeats must be in [1,100]")

    binary_path = Path(binary).expanduser().resolve(strict=False)
    if not binary_path.is_file():
        raise ShadowMaterializeError(f"GPU binary does not exist: {binary_path}")
    binary_sha256 = _sha256_file(binary_path)
    checkpoint_input = str(checkpoint_sha256).strip().lower()
    checkpoint_identity = None
    if checkpoint_input == "auto":
        checkpoint_sha256, checkpoint_identity = checkpoint_identity_from_safetensors(
            safetensors
        )
    else:
        checkpoint_sha256 = checkpoint_input
        if len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256):
            raise ShadowMaterializeError(
                'checkpoint_sha256 must be 64 lowercase hex chars or "auto"'
            )
        checkpoint_identity = {
            "schema": "checkpoint-identity-v1",
            "kind": "manual",
            "sha256": checkpoint_sha256,
        }

    candidate_dir = output_dir / str(row["candidate_id"])
    g4_path = candidate_dir / "g4_manifest.txt"
    g6_path = candidate_dir / "g6_manifest.txt"
    spec_path = candidate_dir / "candidate_spec.json"

    entry = f"{local_token} {max_new_tokens}\n"
    _atomic_text(g4_path, entry)
    _atomic_text(g6_path, entry * int(g6_repeats))

    spec = {
        "schema": SCHEMA,
        "production_write_allowed": False,
        "candidate_id": row["candidate_id"],
        "role": row["role"],
        "layer": int(row["layer"]),
        "n": int(row["n"]),
        "event": {
            "orig_token": int(event["orig_token"]),
            "corrected_token": int(event["corrected_token"]),
            "pos": pos,
        },
        "reference": {
            "emitted_token": int(row["reference"]["emitted_token"]),
        },
        "prompt_len": int(prompt_len),
        "g4_manifest": str(g4_path),
        "g6_manifest": str(g6_path),
        "cwd": str(cwd_path),
        "binary": str(binary_path),
        "binary_sha256": binary_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "moe_base": str(Path(moe_base).expanduser().resolve(strict=False)),
        "safetensors": str(Path(safetensors).expanduser().resolve(strict=False)),
        "source": {
            "discovery_payload_sha256": discovery.get("payload_sha256"),
            "provenance": row.get("provenance"),
            "mapped_source_manifest": str(local_manifest),
            "mapped_raw_token_file": str(local_token),
            "raw_token_sha256": _sha256_file(local_token),
            "checkpoint_identity": checkpoint_identity,
            "g6_repeats": int(g6_repeats),
        },
    }
    _atomic_json(spec_path, spec)
    return {
        "status": "SPEC_READY",
        "production_write_allowed": False,
        "candidate_id": row["candidate_id"],
        "candidate_spec": str(spec_path),
        "g4_manifest": str(g4_path),
        "g6_manifest": str(g6_path),
        "prompt_len": prompt_len,
        "binary_sha256": binary_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--discovery", required=True)
    ap.add_argument("--candidate-id")
    ap.add_argument("--path-map", action="append", default=[], metavar="FROM=TO")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument(
        "--checkpoint-sha256",
        required=True,
        help='64-hex checkpoint identity, or "auto" to hash the safetensors index and all referenced shards',
    )
    ap.add_argument("--moe-base", required=True)
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--g6-repeats", type=int, default=64)
    args = ap.parse_args()

    try:
        result = materialize(
            load_json(args.discovery),
            candidate_id=args.candidate_id,
            path_mappings=parse_maps(args.path_map),
            output_dir=args.output_dir,
            cwd=args.cwd,
            binary=args.binary,
            checkpoint_sha256=args.checkpoint_sha256,
            moe_base=args.moe_base,
            safetensors=args.safetensors,
            g6_repeats=args.g6_repeats,
        )
    except Exception as exc:
        print(json.dumps({
            "status": "SPEC_ERROR",
            "production_write_allowed": False,
            "error": str(exc),
        }, sort_keys=True))
        return 2

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
