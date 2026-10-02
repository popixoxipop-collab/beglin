#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / ".build"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def run(argv):
    return subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=300, check=False)

def git_head() -> str:
    p = run(["git", "rev-parse", "HEAD"])
    if p.returncode != 0:
        return "UNKNOWN"
    return p.stdout.strip()

def source_identity():
    paths = [
        ROOT / "mlx_moe.cpp",
        ROOT / "mlx_moe.h",
        HERE / "binding_runtime_probe.cpp",
        HERE / "CMakeLists.txt",
        HERE / "test_binding_contract.py",
    ]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    mlx_cmake_dir = os.environ.get("MLX_CMAKE_DIR")
    if not mlx_cmake_dir:
        candidates = [
            "/Users/xox/.venv-vllm-metal/lib/python3.12/site-packages/mlx/share/cmake",
            "/Users/xox/Library/Python/3.14/lib/python/site-packages/mlx/share/cmake",
            "/Users/xox/Library/Python/3.9/lib/python/site-packages/mlx/share/cmake",
        ]
        mlx_cmake_dir = next((p for p in candidates if Path(p).is_dir()), None)

    identity = {"git_head": git_head(), "sha256": source_identity()}
    if args.self_test:
        print(json.dumps({
            "status": "PASS",
            "mode": "self-test",
            "mlx_cmake_dir": mlx_cmake_dir,
            "identity": identity,
        }, sort_keys=True))
        return 0
    if not mlx_cmake_dir:
        print(json.dumps({"status":"BLOCKED","reason":"MLX_CMAKE_DIR_NOT_FOUND","identity":identity}, sort_keys=True))
        return 2

    configure = run([
        "cmake", "-S", str(HERE), "-B", str(BUILD),
        "-DCMAKE_BUILD_TYPE=Release", f"-DMLX_CMAKE_DIR={mlx_cmake_dir}",
    ])
    if configure.returncode != 0:
        print(json.dumps({"status":"FAIL","stage":"configure","identity":identity,
                          "stdout":configure.stdout,"stderr":configure.stderr}, sort_keys=True))
        return configure.returncode or 1
    build = run(["cmake", "--build", str(BUILD), "-j2"])
    if build.returncode != 0:
        print(json.dumps({"status":"FAIL","stage":"build","identity":identity,
                          "stdout":build.stdout,"stderr":build.stderr}, sort_keys=True))
        return build.returncode or 1
    p = run([str(BUILD / "binding_runtime_probe")])
    print(json.dumps({
        "status":"PASS" if p.returncode == 0 else "FAIL",
        "stage":"runtime",
        "exit_code":p.returncode,
        "identity":identity,
        "stdout":p.stdout,
        "stderr":p.stderr,
    }, sort_keys=True))
    return p.returncode

if __name__ == "__main__":
    raise SystemExit(main())
