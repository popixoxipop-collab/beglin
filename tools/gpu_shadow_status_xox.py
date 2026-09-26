#!/usr/bin/env python3
"""Report the XOX detached GPU shadow-cycle state without exposing credentials."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path("/Users/xox/vdsp_shadow_runs")
STATUS = ROOT / "launcher_status.json"
LAST = ROOT / "last_cycle.json"


def _pid_alive(pid) -> bool:
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def main() -> int:
    if not STATUS.is_file():
        print(json.dumps({
            "schema": "gpu-shadow-xox-status-v1",
            "status": "NOT_STARTED",
            "production_write_allowed": False,
        }, sort_keys=True))
        return 0

    try:
        launch = json.loads(STATUS.read_text())
    except Exception as exc:
        print(json.dumps({
            "schema": "gpu-shadow-xox-status-v1",
            "status": "STATUS_ERROR",
            "production_write_allowed": False,
            "error": str(exc),
        }, sort_keys=True))
        return 2

    result = {
        "schema": "gpu-shadow-xox-status-v1",
        "status": launch.get("status"),
        "launch_id": launch.get("launch_id"),
        "cycle_id": launch.get("cycle_id"),
        "shadow_run_id": launch.get("shadow_run_id"),
        "production_write_allowed": False,
        "pid": launch.get("pid") or launch.get("worker_pid"),
        "pid_alive": _pid_alive(launch.get("pid") or launch.get("worker_pid")),
        "returncode": launch.get("returncode"),
        "started_at": launch.get("started_at"),
        "finished_at": launch.get("finished_at"),
        "credential_keys_loaded": launch.get("credential_keys_loaded", []),
        "launcher_last_cycle": launch.get("last_cycle"),
        "last_cycle_exists": LAST.is_file(),
    }
    if LAST.is_file():
        try:
            obj = json.loads(LAST.read_text())
            cycle_launch_id = obj.get("launch_id")
            launch_id = launch.get("launch_id")
            relation = (
                "CURRENT"
                if launch_id and cycle_launch_id == launch_id
                else (
                    "LEGACY_UNLINKED"
                    if not cycle_launch_id
                    else "PREVIOUS"
                )
            )
            result["last_cycle"] = {
                "relation": relation,
                "status": obj.get("status"),
                "launch_id": cycle_launch_id,
                "cycle_id": obj.get("cycle_id"),
                "shadow_run_id": obj.get("shadow_run_id")
                    or (obj.get("shadow_run") or {}).get("run_id"),
                "selected_candidate_id": obj.get("selected_candidate_id"),
                "ready_count": obj.get("ready_count"),
                "shadow_status": (obj.get("shadow_run") or {}).get("shadow_status"),
                "production_write_allowed": obj.get("production_write_allowed"),
            }
        except Exception:
            result["last_cycle"] = {"status": "UNREADABLE"}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
