#!/usr/bin/env python3
"""Detached XOX verifier for the certified DeepSeek-V2-Lite checkpoint.

Default invocation launches one detached two-pass verifier. --status reports
only non-secret verification state. The --worker entry point is private to the
local launcher and is not intended for the Tailnet exec allowlist.

No model files are modified and no production control/history files are read or
written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from checkpoint_identity import verify_checkpoint_identity


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(
    "/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors/"
    "model.safetensors.index.json"
)
APPROVED_ROOT = CHECKPOINT.parent
OUTPUT_ROOT = Path("/Users/xox/vdsp_shadow_runs/checkpoint_identity")
RESULT_FILE = OUTPUT_ROOT / "checkpoint_identity.json"
STATUS_FILE = OUTPUT_ROOT / "status.json"
LOG_FILE = OUTPUT_ROOT / "worker.log"


class XoxCheckpointVerifyError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pid_alive(pid) -> bool:
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _minimal_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _load_status() -> dict | None:
    if not STATUS_FILE.is_file():
        return None
    try:
        value = json.loads(STATUS_FILE.read_text())
    except Exception as exc:
        raise XoxCheckpointVerifyError(
            f"checkpoint verifier status is unreadable: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise XoxCheckpointVerifyError("checkpoint verifier status must be an object")
    return value


def launch() -> dict:
    old = _load_status()
    if old and old.get("status") in {"PREPARING", "RUNNING"} and _pid_alive(old.get("pid")):
        raise XoxCheckpointVerifyError(
            f"checkpoint verification already running with pid={old.get('pid')}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        STATUS_FILE,
        {
            "schema": "xox-checkpoint-verifier-v1",
            "status": "PREPARING",
            "production_write_allowed": False,
            "started_at": _now(),
        },
    )
    log = open(LOG_FILE, "ab", buffering=0)
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        cwd=str(REPO_ROOT),
        env=_minimal_env(),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    return {
        "schema": "xox-checkpoint-verifier-v1",
        "status": "LAUNCHED",
        "production_write_allowed": False,
        "worker_pid": proc.pid,
        "status_file": str(STATUS_FILE),
        "result_file": str(RESULT_FILE),
    }


def worker_once(
    *,
    checkpoint: Path = CHECKPOINT,
    approved_root: Path = APPROVED_ROOT,
    output: Path = RESULT_FILE,
) -> dict:
    return verify_checkpoint_identity(
        checkpoint,
        output=output,
        allowed_root=approved_root,
        passes=2,
    )


def worker() -> int:
    pid = os.getpid()
    _atomic_json(
        STATUS_FILE,
        {
            "schema": "xox-checkpoint-verifier-v1",
            "status": "RUNNING",
            "production_write_allowed": False,
            "pid": pid,
            "started_at": _now(),
        },
    )
    try:
        result = worker_once()
        result_sha = _sha256_file(RESULT_FILE)
        passes = result["verification_passes"]
        file_count = len(passes[0]["files"]) if passes else 0
        bytes_per_pass = passes[0]["bytes_hashed"] if passes else 0
        final = {
            "schema": "xox-checkpoint-verifier-v1",
            "status": "COMPLETE",
            "production_write_allowed": False,
            "pid": pid,
            "finished_at": _now(),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "verification_passes": len(passes),
            "file_count": file_count,
            "bytes_per_pass": bytes_per_pass,
            "result_sha256": result_sha,
            "result_file": str(RESULT_FILE),
        }
        _atomic_json(STATUS_FILE, final)
        print(json.dumps(final, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": "xox-checkpoint-verifier-v1",
            "status": "FAILED",
            "production_write_allowed": False,
            "pid": pid,
            "finished_at": _now(),
            "error": str(exc),
        }
        _atomic_json(STATUS_FILE, failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


def status() -> dict:
    value = _load_status()
    if value is None:
        return {
            "schema": "xox-checkpoint-verifier-v1",
            "status": "NOT_STARTED",
            "production_write_allowed": False,
        }
    result = dict(value)
    if result.get("status") in {"PREPARING", "RUNNING"}:
        result["pid_alive"] = _pid_alive(result.get("pid") or result.get("worker_pid"))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.status and args.worker:
        print(json.dumps({"status": "ERROR", "error": "conflicting mode flags"}))
        return 2
    try:
        if args.worker:
            return worker()
        result = status() if args.status else launch()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": "xox-checkpoint-verifier-v1",
                    "status": "ERROR",
                    "production_write_allowed": False,
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
