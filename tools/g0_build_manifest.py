#!/usr/bin/env python3
"""Build a conservative G0 build/evidence manifest without invoking a compiler."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(relpath):
    p = ROOT / relpath
    return {
        "path": str(p),
        "exists": p.is_file(),
        "size_bytes": p.stat().st_size if p.is_file() else None,
        "sha256": sha256_file(p) if p.is_file() else None,
    }


def candidate_binaries():
    paths = [
        ROOT / "qwen_infer_gpu",
        ROOT / "build" / "qwen_infer_gpu",
    ]
    paths.extend(ROOT.glob("build*/qwen_infer_gpu"))
    paths.extend(ROOT.glob("cmake-build*/qwen_infer_gpu"))
    seen = set()
    out = []
    for p in paths:
        p = p.resolve()
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        out.append({
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "executable": os.access(p, os.X_OK),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default=str(ROOT / "results" / "precision_parity" / "g0_build_manifest.json"),
    )
    ap.add_argument(
        "--runtime-note",
        default="runtime build/run not executed by this manifest generator",
    )
    args = ap.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    cmake_path = ROOT / "CMakeLists.txt"
    migration_path = ROOT / "supabase_migration_d4_retry_idempotency.sql"
    cmake = cmake_path.read_text() if cmake_path.is_file() else ""
    migration = migration_path.read_text() if migration_path.is_file() else ""

    source_names = [
        "CMakeLists.txt",
        "mlx_moe.cpp",
        "mlx_moe.h",
        "qwen_infer.c",
        "d4_supabase_push.py",
        "tools/autopilot_live_preflight.py",
        "tools/backend_capabilities.py",
        "tools/test_pr_a_regressions.py",
        "supabase_migration_d4_retry_idempotency.sql",
    ]
    binaries = candidate_binaries()

    capability_path = ROOT / "results" / "precision_parity" / "g0_capabilities.json"
    prior_capability = None
    if capability_path.is_file():
        prior_capability = json.loads(capability_path.read_text())

    payload = {
        "schema_version": "precision-build-manifest-g0-v1",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(ROOT),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "source_files": {name: file_record(name) for name in source_names},
        "binaries": binaries,
        "gpu_binary_state": (
            "DISCOVERED_UNVERIFIED" if binaries else "NO_LOCAL_GPU_BINARY_DISCOVERED"
        ),
        "build_recipe": {
            "target": "qwen_infer_gpu",
            "target_declared": "add_executable(qwen_infer_gpu" in cmake,
            "gpu_define_declared": (
                "target_compile_definitions(qwen_infer_obj PRIVATE QWEN_GPU_MLX)" in cmake
            ),
            "mlx_link_declared": "target_link_libraries(qwen_infer_gpu PRIVATE mlx" in cmake,
            "mlx_cmake_discovery_declared": (
                "find_package(MLX CONFIG REQUIRED" in cmake and "MLX_CMAKE_DIR" in cmake
            ),
        },
        "retry_idempotency_contract": {
            "migration_present": migration_path.is_file(),
            "event_source_id_unique": (
                "moe_neartie_events_source_event_id_uidx" in migration
                and "source_event_id" in migration
            ),
            "attribution_receipt_table": "moe_role_precision_hit_receipts" in migration,
            "idempotent_rpc": (
                "increment_role_precision_idempotent" in migration
                and "on conflict (attribution_id) do nothing" in migration
            ),
            "migration_applied_to_remote_db": "NOT_VERIFIED",
        },
        "runtime_verification": {
            "executed": False,
            "note": args.runtime_note,
        },
        "prior_capability_artifact": {
            "path": str(capability_path),
            "exists": capability_path.is_file(),
            "sha256": sha256_file(capability_path) if capability_path.is_file() else None,
            "status": (
                prior_capability.get("capability", {}).get("status")
                if isinstance(prior_capability, dict)
                else None
            ),
        },
    }

    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(str(output))
    print(json.dumps({
        "gpu_binary_state": payload["gpu_binary_state"],
        "build_recipe": payload["build_recipe"],
        "retry_idempotency_contract": payload["retry_idempotency_contract"],
        "runtime_verification": payload["runtime_verification"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

