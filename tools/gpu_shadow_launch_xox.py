#!/usr/bin/env python3
"""Safely launch one detached XOX production-shadow cycle.

The MCP exec environment is intentionally sanitized and therefore does not
carry QWEN_SUPABASE_URL/QWEN_SUPABASE_KEY.  This launcher does NOT shell-source
an env file.  It reads exactly those two keys from the fixed XOX main-worktree
.env file, passes them only to a detached shadow-pipeline worker, and never
prints or persists their values.

Default invocation (the only form intended for the Tailnet exact allowlist):

    python3 tools/gpu_shadow_launch_xox.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


REQUIRED_KEYS = ("QWEN_SUPABASE_URL", "QWEN_SUPABASE_KEY")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/gpu_shadow_xox_certified.json"
DEFAULT_ENV_FILE = Path("/Users/xox/vdsp-engine/.env")
DEFAULT_SHADOW_ROOT = Path("/Users/xox/vdsp_shadow_runs")
STATUS_FILE = DEFAULT_SHADOW_ROOT / "launcher_status.json"
LOG_FILE = DEFAULT_SHADOW_ROOT / "cycle.log"


class ShadowLaunchError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _pid_alive(pid) -> bool:
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def load_credentials(path: str | Path) -> dict[str, str]:
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ShadowLaunchError(f"credential env file may not be a symlink: {path}")
    if not path.is_file():
        raise ShadowLaunchError(f"credential env file is missing: {path}")
    st = path.stat()
    if st.st_uid != os.geteuid():
        raise ShadowLaunchError(
            f"credential env file owner uid mismatch: expected={os.geteuid()} actual={st.st_uid}"
        )

    found: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in REQUIRED_KEYS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not value or "\x00" in value or "\n" in value or "\r" in value:
            raise ShadowLaunchError(f"invalid/empty credential value for {key}")
        if key in found and found[key] != value:
            raise ShadowLaunchError(f"duplicate conflicting value for {key}")
        found[key] = value

    missing = [key for key in REQUIRED_KEYS if not found.get(key)]
    if missing:
        raise ShadowLaunchError(
            "credential env file is missing required key(s): " + ", ".join(missing)
        )
    return found


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise ShadowLaunchError(f"shadow config is missing: {path}")
    with open(path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ShadowLaunchError("shadow config must be a JSON object")
    if Path(cfg.get("cwd", "")).resolve(strict=False) != REPO_ROOT:
        raise ShadowLaunchError("shadow config cwd must be this certified worktree")
    if Path(cfg.get("shadow_root", "")).resolve(strict=False) != DEFAULT_SHADOW_ROOT:
        raise ShadowLaunchError(
            f"shadow config root must be exactly {DEFAULT_SHADOW_ROOT}"
        )
    if str(cfg.get("checkpoint_sha256", "")).lower() != "auto":
        raise ShadowLaunchError('XOX shadow config must use checkpoint_sha256="auto"')
    return cfg


def _safe_child_env(credentials: dict[str, str]) -> dict[str, str]:
    env = {}
    for key in ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    env.update(credentials)
    return env


def launch(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    env_file: str | Path = DEFAULT_ENV_FILE,
    python_bin: str = sys.executable,
) -> dict:
    load_config(config_path)
    credentials = load_credentials(env_file)

    if STATUS_FILE.exists():
        try:
            old = json.loads(STATUS_FILE.read_text())
        except Exception:
            old = {}
        if old.get("status") in {"PREPARING", "RUNNING"} and _pid_alive(old.get("pid")):
            raise ShadowLaunchError(
                f"shadow cycle already running with pid={old.get('pid')}"
            )

    DEFAULT_SHADOW_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_json(STATUS_FILE, {
        "schema": "gpu-shadow-xox-launch-v1",
        "status": "PREPARING",
        "production_write_allowed": False,
        "started_at": _now(),
        "credential_keys_loaded": sorted(REQUIRED_KEYS),
    })

    env = _safe_child_env(credentials)
    log = open(LOG_FILE, "ab", buffering=0)
    worker = subprocess.Popen(
        [
            python_bin,
            str(Path(__file__).resolve()),
            "--worker",
            "--config",
            str(Path(config_path).resolve(strict=False)),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    return {
        "schema": "gpu-shadow-xox-launch-v1",
        "status": "LAUNCHED",
        "production_write_allowed": False,
        "worker_pid": worker.pid,
        "status_file": str(STATUS_FILE),
        "log_file": str(LOG_FILE),
        "credential_keys_loaded": sorted(REQUIRED_KEYS),
    }


def worker(config_path: str, python_bin: str = sys.executable) -> int:
    pid = os.getpid()
    _atomic_json(STATUS_FILE, {
        "schema": "gpu-shadow-xox-launch-v1",
        "status": "RUNNING",
        "production_write_allowed": False,
        "pid": pid,
        "started_at": _now(),
        "credential_keys_loaded": sorted(REQUIRED_KEYS),
    })

    pipeline = REPO_ROOT / "tools/gpu_shadow_pipeline.py"
    proc = subprocess.run(
        [python_bin, str(pipeline), "--config", str(config_path)],
        cwd=str(REPO_ROOT),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
    )
    with open(LOG_FILE, "ab") as f:
        if proc.stdout:
            f.write(proc.stdout.encode())
        if proc.stderr:
            f.write(proc.stderr.encode())

    last_cycle = DEFAULT_SHADOW_ROOT / "last_cycle.json"
    summary = None
    if last_cycle.is_file():
        try:
            obj = json.loads(last_cycle.read_text())
            summary = {
                "status": obj.get("status"),
                "selected_candidate_id": obj.get("selected_candidate_id"),
                "ready_count": obj.get("ready_count"),
                "shadow_status": (obj.get("shadow_run") or {}).get("shadow_status"),
            }
        except Exception:
            summary = {"status": "UNREADABLE_LAST_CYCLE"}

    _atomic_json(STATUS_FILE, {
        "schema": "gpu-shadow-xox-launch-v1",
        "status": "COMPLETE" if proc.returncode == 0 else "FAILED",
        "production_write_allowed": False,
        "pid": pid,
        "returncode": int(proc.returncode),
        "finished_at": _now(),
        "last_cycle": summary,
        "credential_keys_loaded": sorted(REQUIRED_KEYS),
    })
    return int(proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help=argparse.SUPPRESS)
    args = ap.parse_args()

    try:
        if args.worker:
            return worker(args.config)
        result = launch(config_path=args.config)
    except Exception as exc:
        print(json.dumps({
            "schema": "gpu-shadow-xox-launch-v1",
            "status": "LAUNCH_ERROR",
            "production_write_allowed": False,
            "error": str(exc),
        }, sort_keys=True))
        return 2

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
