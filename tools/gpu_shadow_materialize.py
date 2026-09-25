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
from pathlib import Path
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
        source, dest = item.split("=", 1)
        source = str(Path(source).expanduser())
        dest = str(Path(dest).expanduser())
        if not source or not dest:
            raise ShadowMaterializeError("path map FROM and TO must be non-empty")
        mappings.append((source.rstrip("/"), dest.rstrip("/")))
    return sorted(mappings, key=lambda pair: len(pair[0]), reverse=True)


def map_path(path: str, mappings) -> Path:
    value = str(Path(path).expanduser())
    for source, dest in mappings:
        if value == source:
            return Path(dest)
        prefix = source + "/"
        if value.startswith(prefix):
            return Path(dest + value[len(source):])
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
    local_token = map_path(source_token, path_mappings).resolve(strict=False)
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
    checkpoint_sha256 = str(checkpoint_sha256).lower()
    if len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256):
        raise ShadowMaterializeError("checkpoint_sha256 must be 64 lowercase hex chars")

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
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--discovery", required=True)
    ap.add_argument("--candidate-id")
    ap.add_argument("--path-map", action="append", default=[], metavar="FROM=TO")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
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
