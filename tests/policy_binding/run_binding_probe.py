#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / ".build"

EXPECTED_SHA256 = {
    ROOT / "mlx_moe.cpp": "67e3c191cf7dbaead40050b6c8d09ab4318b6fa794351fe0976061a4fa6f762f",
    ROOT / "mlx_moe.h": "753a1eab8542123fda8ad6834e890b88e4a4fdc894f59fac9e1d7cead95a71a6",
    HERE / "binding_runtime_probe.cpp": "9d11ab9d1456588a354edb803e4769b03bd9634a975036714e84a587b156bf2a",
    HERE / "CMakeLists.txt": "49c4fa4c41936432a49a4339839ce9d5936ed82fc15232c9129a793933c36b67",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs() -> tuple[bool, dict[str, dict[str, str]]]:
    details: dict[str, dict[str, str]] = {}
    ok = True
    for path, expected in EXPECTED_SHA256.items():
        observed = sha256_file(path) if path.is_file() else "MISSING"
        details[str(path.relative_to(ROOT))] = {
            "expected": expected,
            "observed": observed,
        }
        ok = ok and observed == expected
    return ok, details


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    inputs_ok, input_hashes = verify_inputs()
    if not inputs_ok:
        print(json.dumps({
            "status": "BLOCKED",
            "reason": "SOURCE_SHA_MISMATCH",
            "inputs": input_hashes,
        }, sort_keys=True))
        return 2

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

    if args.self_test:
        print(json.dumps({
            "status": "PASS",
            "mode": "self-test",
            "mlx_cmake_dir": mlx_cmake_dir,
            "inputs": input_hashes,
        }, sort_keys=True))
        return 0

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
        "inputs": input_hashes,
    }
    print(json.dumps(payload, sort_keys=True))
    return probe.returncode


if __name__ == "__main__":
    raise SystemExit(main())
