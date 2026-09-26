#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

SOURCE_COMMIT = "330954b27f146b8a17db2cb353c3e620968bad5e"
BINARY_SHA256 = "6e6258d6d4e402033c303c8c4b622db5a088354a0de79e687eb0fad4190c4392"
CHECKPOINT_SHA256 = "1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-root", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--dashboard-url", default="http://eoe:8790")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--worker-instance-id", required=True)
    ap.add_argument("--follow", action="store_true")
    args = ap.parse_args()

    bridge_root = Path(args.bridge_root).expanduser().resolve()
    bridge = bridge_root / "run_bridge.py"
    log = Path(args.log).expanduser().resolve()

    if not bridge.is_file():
        raise SystemExit(f"run_bridge.py missing: {bridge}")
    if log.is_symlink() or not log.is_file():
        raise SystemExit(f"invalid log: {log}")

    with urllib.request.urlopen(args.dashboard_url.rstrip("/") + "/health", timeout=3) as r:
        if r.status != 200:
            raise SystemExit(f"dashboard health HTTP {r.status}")

    argv = [
        sys.executable,
        str(bridge),
        "--log", str(log),
        "--dashboard-url", args.dashboard_url,
        "--experiment-id", args.experiment_id,
        "--session-id", args.session_id,
        "--source-commit", SOURCE_COMMIT,
        "--binary-sha256", BINARY_SHA256,
        "--checkpoint-sha256", CHECKPOINT_SHA256,
        "--worker-instance-id", args.worker_instance_id,
    ]
    if args.follow:
        argv.append("--follow")
    return subprocess.run(argv, cwd=bridge_root, env=os.environ.copy()).returncode


if __name__ == "__main__":
    raise SystemExit(main())
