#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / ".build"


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )


def main() -> int:
    mlx_cmake_dir = os.environ.get("MLX_CMAKE_DIR")
    if not mlx_cmake_dir:
        candidates = [
            "/Users/xox/Library/Python/3.14/lib/python/site-packages/mlx/share/cmake",
            "/Users/xox/Library/Python/3.9/lib/python/site-packages/mlx/share/cmake",
        ]
        mlx_cmake_dir = next((p for p in candidates if Path(p).is_dir()), None)
    if not mlx_cmake_dir:
        print(json.dumps({"status": "BLOCKED", "reason": "MLX_CMAKE_DIR_NOT_FOUND"}))
        return 2

    configure = run([
        "cmake",
        "-S",
        str(HERE),
        "-B",
        str(BUILD),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DMLX_CMAKE_DIR={mlx_cmake_dir}",
    ])
    if configure.returncode != 0:
        print(json.dumps({
            "status": "FAIL",
            "stage": "configure",
            "stdout": configure.stdout,
            "stderr": configure.stderr,
        }, sort_keys=True))
        return configure.returncode or 1

    build = run(["cmake", "--build", str(BUILD), "-j2"])
    if build.returncode != 0:
        print(json.dumps({
            "status": "FAIL",
            "stage": "build",
            "stdout": build.stdout,
            "stderr": build.stderr,
        }, sort_keys=True))
        return build.returncode or 1

    probe = run([str(BUILD / "binding_runtime_probe")])
    payload = {
        "status": "PASS" if probe.returncode == 0 else "FAIL",
        "stage": "runtime",
        "mlx_cmake_dir": mlx_cmake_dir,
        "exit_code": probe.returncode,
        "stdout": probe.stdout,
        "stderr": probe.stderr,
    }
    print(json.dumps(payload, sort_keys=True))
    return probe.returncode


if __name__ == "__main__":
    raise SystemExit(main())
