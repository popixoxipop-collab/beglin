#!/usr/bin/env python3
import importlib.metadata
import json
import os
import platform
import sys

try:
    import mlx
    mlx_path = list(mlx.__path__)[0]
    try:
        version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        version = None
    payload = {
        "ok": True,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "mlx_version": version,
        "mlx_path": mlx_path,
        "cmake_dir": os.path.join(mlx_path, "share", "cmake"),
        "cmake_dir_exists": os.path.isdir(os.path.join(mlx_path, "share", "cmake")),
    }
except Exception as e:
    payload = {
        "ok": False,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "error_type": type(e).__name__,
        "error": str(e),
    }

print(json.dumps(payload, sort_keys=True))

