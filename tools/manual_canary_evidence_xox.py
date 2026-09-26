#!/usr/bin/env python3
"""Read-only exporter for Agent-E binding of existing XOX G4/G6 scratch evidence.

No path arguments are accepted. The CLI is intentionally pinned to the
certified worktree's existing g_autopilot_test directory. It never writes,
never follows symlinks, never reads environment variables, and never emits raw
log contents. It reports file identity plus a small allowlist of JSON fields.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "g_autopilot_test"
SUBROOTS = ("g4", "g6_pre", "g6_post", "state")
MAX_FILES = 128
MAX_JSON_BYTES = 1024 * 1024

SAFE_JSON_KEYS = {
    "schema",
    "status",
    "action",
    "final_status",
    "decision",
    "rollback_required",
    "context_hash",
    "active_policy_hash",
    "policy_hash",
    "weight_epoch",
    "ack_sha256",
    "worker_instance_id",
    "baseline_emitted_token",
    "candidate_emitted_token",
    "reference_emitted_token",
    "requests_completed",
    "tokens_evaluated",
    "target_replay_pass",
    "txn_id",
    "role",
    "layer",
    "n",
}


class EvidenceExportError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _safe_json_projection(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    out = {}
    for key in sorted(SAFE_JSON_KEYS):
        if key not in value:
            continue
        v = value[key]
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
        elif key == "status" and isinstance(v, dict):
            out[key] = {
                str(k): x
                for k, x in v.items()
                if isinstance(x, (str, int, float, bool)) or x is None
            }
    return out


def _root_checked(root: Path) -> Path:
    if root.is_symlink():
        raise EvidenceExportError(f"evidence root may not be a symlink: {root}")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise EvidenceExportError(f"evidence root is not a directory: {root}")
    return resolved


def _collect_one(root: Path, path: Path) -> dict:
    if path.is_symlink():
        raise EvidenceExportError(f"symlink evidence entry denied: {path}")
    resolved = path.resolve(strict=True)
    if root != resolved and root not in resolved.parents:
        raise EvidenceExportError(f"evidence path escaped fixed root: {path}")
    if not resolved.is_file():
        raise EvidenceExportError(f"non-file evidence entry denied: {path}")
    rel = resolved.relative_to(root).as_posix()
    row = {
        "path": rel,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if resolved.suffix.lower() == ".json" and row["size_bytes"] <= MAX_JSON_BYTES:
        try:
            parsed = json.loads(resolved.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            row["json_projection"] = None
        else:
            row["json_projection"] = _safe_json_projection(parsed)
    return row


def collect(root: Path = EVIDENCE_ROOT) -> dict:
    root = _root_checked(root)
    files = []
    groups = {}
    for name in SUBROOTS:
        sub = root / name
        if not sub.exists():
            groups[name] = {"status": "MISSING", "files": []}
            continue
        if sub.is_symlink():
            raise EvidenceExportError(f"symlink subroot denied: {sub}")
        resolved_sub = sub.resolve(strict=True)
        if root not in resolved_sub.parents:
            raise EvidenceExportError(f"subroot escaped evidence root: {sub}")
        if not resolved_sub.is_dir():
            raise EvidenceExportError(f"subroot is not a directory: {sub}")
        group_files = []
        for path in sorted(resolved_sub.rglob("*")):
            if path.is_dir():
                if path.is_symlink():
                    raise EvidenceExportError(f"symlink directory denied: {path}")
                continue
            group_files.append(_collect_one(root, path))
            if len(files) + len(group_files) > MAX_FILES:
                raise EvidenceExportError("evidence file limit exceeded")
        files.extend(group_files)
        groups[name] = {
            "status": "PRESENT",
            "file_count": len(group_files),
            "bundle_sha256": canonical_sha256(
                [{"path": x["path"], "size_bytes": x["size_bytes"], "sha256": x["sha256"]}
                 for x in group_files]
            ),
            "files": group_files,
        }

    g4 = groups.get("g4", {})
    g6_rows = []
    for name in ("g6_pre", "g6_post"):
        for row in groups.get(name, {}).get("files", []):
            g6_rows.append(
                {"path": row["path"], "size_bytes": row["size_bytes"], "sha256": row["sha256"]}
            )
    g6_bundle = canonical_sha256(g6_rows) if g6_rows else None

    return {
        "schema": "manual-canary-raw-evidence-export-v1",
        "production_write_allowed": False,
        "source": "fixed-g_autopilot_test-read-only",
        "root_name": root.name,
        "file_count": len(files),
        "groups": groups,
        "g4_bundle_sha256": g4.get("bundle_sha256"),
        "g6_bundle_sha256": g6_bundle,
    }


def main() -> int:
    if len(sys.argv) != 1:
        print(json.dumps({
            "schema": "manual-canary-raw-evidence-export-v1",
            "status": "ARGUMENTS_DENIED",
            "production_write_allowed": False,
        }, sort_keys=True))
        return 2
    try:
        result = collect()
    except Exception as exc:
        print(json.dumps({
            "schema": "manual-canary-raw-evidence-export-v1",
            "status": "ERROR",
            "production_write_allowed": False,
            "error": str(exc),
        }, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
