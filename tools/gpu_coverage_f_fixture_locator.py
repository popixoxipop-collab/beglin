#!/usr/bin/env python3
"""Locate the existing kv_a/L11 replay fixture in the fixed XOX read-only mirror.

This tool only reads files under /Users/xox/vdsp_shadow_mirror/vdsp_p5_pre and
hashes the current certified GPU binary. It does not contact bob, Supabase, or
production control state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCHEMA = "gpu-coverage-f-fixture-v1"
MIRROR = Path("/Users/xox/vdsp_shadow_mirror/vdsp_p5_pre")
REPO = Path("/Users/xox/vdsp-engine-gpu-precision")
BINARY = REPO / "build-gpu-precision/qwen_infer_gpu"
MOE_BASE = Path("/Users/xox/vdsp_local_data/moe_base_deepseek")
SAFETENSORS = Path(
    "/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors/"
    "model.safetensors.index.json"
)
CHECKPOINT_SHA = "1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898"

TARGET = {
    "role": "kv_a_proj_with_mqa",
    "layer": 11,
    "orig_token": 8713,
    "corrected_token": 4794,
    "pos": 14,
}


class FixtureError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _iter_text_files(root: Path):
    allowed = {".jsonl", ".json", ".txt", ".manifest"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed:
            yield p


def _event_match(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    role = obj.get("role")
    layer = obj.get("layer")
    pos = obj.get("pos")
    orig = obj.get("orig_argmax", obj.get("orig_token"))
    corrected = obj.get("corrected_argmax", obj.get("corrected_token"))
    try:
        return (
            role == TARGET["role"]
            and int(layer) == TARGET["layer"]
            and int(pos) == TARGET["pos"]
            and int(orig) == TARGET["orig_token"]
            and int(corrected) == TARGET["corrected_token"]
        )
    except (TypeError, ValueError):
        return False


def _search_event(root: Path):
    matches = []
    for path in _iter_text_files(root):
        try:
            text = path.read_text(errors="strict")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _event_match(obj):
                matches.append((path, lineno, obj))
    return matches


def _candidate_manifest_values(obj: dict):
    for key in (
        "manifest",
        "manifest_path",
        "replay_manifest",
        "source_manifest",
    ):
        value = obj.get(key)
        if isinstance(value, str) and value:
            yield value


def _resolve_manifest(value: str, source: Path, root: Path) -> Path | None:
    p = Path(value).expanduser()
    candidates = []
    if p.is_absolute():
        # Historical provenance may point to bob. Only map the vdsp_p5_pre
        # suffix into the fixed local mirror; never read the remote path.
        marker = "/vdsp_p5_pre/"
        s = str(p)
        if marker in s:
            suffix = s.split(marker, 1)[1]
            candidates.append(root / suffix)
    else:
        candidates.extend((source.parent / p, root / p))
    for c in candidates:
        r = c.resolve(strict=False)
        if _within(r, root) and r.is_file():
            return r
    return None


def _parse_manifest(path: Path):
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise FixtureError(f"manifest has no active entries: {path}")

    parsed = []
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            raise FixtureError(f"bad manifest line: {line!r}")
        raw, max_new = parts
        try:
            max_new = int(max_new)
        except ValueError as exc:
            raise FixtureError("manifest max_new_tokens must be integer") from exc
        token = Path(raw).expanduser()
        if token.is_absolute():
            marker = "/vdsp_p5_pre/"
            s = str(token)
            if marker not in s:
                raise FixtureError(f"token path is outside mirror namespace: {token}")
            token = MIRROR / s.split(marker, 1)[1]
        else:
            token = path.parent / token
        token = token.resolve(strict=False)
        if not _within(token, MIRROR) or not token.is_file():
            raise FixtureError(f"mapped token file is missing/outside mirror: {token}")
        size = token.stat().st_size
        if size <= 0 or size % 4:
            raise FixtureError(f"token file is not non-empty int32 data: {token}")
        parsed.append({
            "token_file": str(token),
            "prompt_len": size // 4,
            "max_new_tokens": max_new,
            "token_sha256": sha256_file(token),
        })
    return parsed


def locate() -> dict:
    for p in (MIRROR, REPO, MOE_BASE):
        if not p.exists():
            raise FixtureError(f"required path missing: {p}")
    for p in (BINARY, SAFETENSORS):
        if not p.is_file():
            raise FixtureError(f"required file missing: {p}")

    matches = _search_event(MIRROR)
    if not matches:
        raise FixtureError("kv_a/L11 8713->4794 event not found in local mirror")

    candidates = []
    for source, lineno, obj in matches:
        for value in _candidate_manifest_values(obj):
            manifest = _resolve_manifest(value, source, MIRROR)
            if manifest is None:
                continue
            try:
                entries = _parse_manifest(manifest)
            except FixtureError:
                continue
            candidates.append({
                "source_file": str(source),
                "source_line": lineno,
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "entries": entries,
            })

    if not candidates:
        # Fallback: find a one-entry manifest in the same discovery directory
        # whose token bytes are valid. This remains read-only and is reported
        # as a fallback rather than silently claiming exact provenance.
        seen_dirs = sorted({source.parent for source, _, _ in matches})
        for d in seen_dirs:
            for manifest in sorted(d.glob("*.txt")):
                try:
                    entries = _parse_manifest(manifest)
                except FixtureError:
                    continue
                if len(entries) == 1:
                    candidates.append({
                        "source_file": str(matches[0][0]),
                        "source_line": matches[0][1],
                        "manifest": str(manifest.resolve()),
                        "manifest_sha256": sha256_file(manifest),
                        "entries": entries,
                        "fallback_manifest_search": True,
                    })

    if not candidates:
        raise FixtureError("event found but no usable local replay manifest resolved")

    # Prefer explicit provenance and then the smallest manifest.
    candidates.sort(key=lambda x: (
        bool(x.get("fallback_manifest_search")),
        len(x["entries"]),
        x["manifest"],
    ))
    chosen = candidates[0]
    prompt_lens = {entry["prompt_len"] for entry in chosen["entries"]}
    if len(prompt_lens) != 1:
        raise FixtureError("chosen manifest entries have inconsistent prompt lengths")

    result = {
        "schema": SCHEMA,
        "production_write_allowed": False,
        "target": dict(TARGET),
        "reference": {"emitted_token": TARGET["corrected_token"]},
        "checkpoint_sha256": CHECKPOINT_SHA,
        "binary": str(BINARY),
        "binary_sha256": sha256_file(BINARY),
        "moe_base": str(MOE_BASE),
        "safetensors": str(SAFETENSORS),
        "g4_manifest": chosen["manifest"],
        "g6_manifest": chosen["manifest"],
        "prompt_len": next(iter(prompt_lens)),
        "manifest_sha256": chosen["manifest_sha256"],
        "source_file": chosen["source_file"],
        "source_line": chosen["source_line"],
        "fallback_manifest_search": bool(chosen.get("fallback_manifest_search")),
        "candidate_count": len(candidates),
        "mirror_root": str(MIRROR),
    }
    return result


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    try:
        value = locate()
        atomic_json(Path(args.output), value)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "ok": True,
        "output": args.output,
        "binary_sha256": value["binary_sha256"],
        "manifest": value["g4_manifest"],
        "prompt_len": value["prompt_len"],
        "fallback_manifest_search": value["fallback_manifest_search"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
