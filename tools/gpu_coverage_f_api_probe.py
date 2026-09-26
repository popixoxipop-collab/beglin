#!/usr/bin/env python3
"""Read-only Agent-F API probe for certified G5/G6 GPU control modules.

The probe imports local Python modules and records only:
- module file path and SHA-256,
- selected callable existence/signatures.

It never calls the probed functions and never writes runtime control state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Iterable


SCHEMA = "gpu-coverage-f-api-probe-v1"

CONTRACT = {
    "gpu_isolated_preflight": (
        "run_ab_preflight",
        "to_planner_evidence",
    ),
    "gpu_observer_control": (
        "request_regression_rollback",
        "complete_regression_rollback",
    ),
    "gpu_restart_canary": (
        "build_restart_canary",
        "evaluate_restart_canary",
    ),
    "gpu_runtime_control": (
        "read_runtime_ack",
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_module(module: ModuleType, names: Iterable[str]) -> dict:
    file_value = getattr(module, "__file__", None)
    file_path = Path(file_value).resolve(strict=False) if file_value else None
    callables = {}
    missing = []
    for name in names:
        value = getattr(module, name, None)
        if not callable(value):
            missing.append(name)
            callables[name] = {
                "exists": False,
                "signature": None,
            }
            continue
        try:
            sig = str(inspect.signature(value))
        except (TypeError, ValueError):
            sig = None
        callables[name] = {
            "exists": True,
            "signature": sig,
            "module": getattr(value, "__module__", None),
            "qualname": getattr(value, "__qualname__", None),
        }

    return {
        "module": module.__name__,
        "file": str(file_path) if file_path else None,
        "file_sha256": (
            _sha256_file(file_path)
            if file_path and file_path.is_file()
            else None
        ),
        "callables": callables,
        "missing": missing,
        "ok": not missing,
    }


def run_probe() -> dict:
    rows = []
    import_errors = []
    for module_name, names in CONTRACT.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            import_errors.append({
                "module": module_name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        rows.append(probe_module(module, names))

    missing = [
        {"module": row["module"], "names": row["missing"]}
        for row in rows
        if row["missing"]
    ]
    return {
        "schema": SCHEMA,
        "generated_at": _now(),
        "read_only": True,
        "production_write_allowed": False,
        "modules": rows,
        "import_errors": import_errors,
        "missing": missing,
        "ok": not import_errors and not missing and len(rows) == len(CONTRACT),
    }


def atomic_json(path: Path, value: dict) -> None:
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
    value = run_probe()
    atomic_json(Path(args.output), value)
    print(json.dumps({
        "ok": value["ok"],
        "output": args.output,
        "import_error_count": len(value["import_errors"]),
        "missing_count": sum(len(x["names"]) for x in value["missing"]),
    }, sort_keys=True))
    return 0 if value["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
