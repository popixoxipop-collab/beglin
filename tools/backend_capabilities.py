#!/usr/bin/env python3
"""Emit a conservative G0 backend capability/build inventory.

This probe does not claim runtime verification. It records what the current
source tree implements, whether the PR-A safety invariants are present, and
the hashes of any already-built GPU binaries it can find.
"""

import argparse
import hashlib
import importlib.metadata
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


def git_head(repo):
    dotgit = repo / ".git"
    if dotgit.is_file():
        text = dotgit.read_text().strip()
        if text.startswith("gitdir:"):
            dotgit = (repo / text.split(":", 1)[1].strip()).resolve()
    head = dotgit / "HEAD"
    if not head.exists():
        return None
    value = head.read_text().strip()
    if value.startswith("ref:"):
        ref = value.split(":", 1)[1].strip()
        ref_path = dotgit / ref
        if ref_path.exists():
            return ref_path.read_text().strip()
        packed = dotgit / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line and not line.startswith("#") and not line.startswith("^"):
                    commit, name = line.split(" ", 1)
                    if name == ref:
                        return commit
        return value
    return value


def mlx_version():
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        return None


def candidate_binaries(repo):
    seen = set()
    paths = [
        repo / "qwen_infer_gpu",
        repo / "build" / "qwen_infer_gpu",
    ]
    paths.extend(repo.glob("build*/qwen_infer_gpu"))
    paths.extend(repo.glob("cmake-build*/qwen_infer_gpu"))
    out = []
    for p in paths:
        try:
            p = p.resolve()
        except FileNotFoundError:
            continue
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


def source_inventory(repo):
    names = [
        "mlx_moe.cpp",
        "mlx_moe.h",
        "qwen_infer.c",
        "d4_supabase_push.py",
        "tools/autopilot_live_preflight.py",
    ]
    result = {}
    for name in names:
        p = repo / name
        result[name] = {
            "exists": p.is_file(),
            "sha256": sha256_file(p) if p.is_file() else None,
        }
    return result


def source_capabilities(repo):
    mlx = (repo / "mlx_moe.cpp").read_text()
    preflight = (repo / "tools" / "autopilot_live_preflight.py").read_text()
    pusher = (repo / "d4_supabase_push.py").read_text()
    qwen = (repo / "qwen_infer.c").read_text()

    return {
        "backend": "mlx_metal",
        "status": "IMPLEMENTED_UNVERIFIED",
        "quantization_paths": {
            "mlx_native_quantized": {
                "bits": [2, 3, 4, 5, 6, 8],
                "status": "IMPLEMENTED_UNVERIFIED",
            },
            "custom_qng64_metal": {
                "bits": [7, 9, 10, 11, 12, 13, 14, 15],
                "status": "IMPLEMENTED_UNVERIFIED",
            },
            "dense": {
                "bits": [16, 32],
                "status": "IMPLEMENTED_UNVERIFIED",
            },
        },
        "pr_a_invariants": {
            "qng64_erased_before_dense_rebind": (
                "g_tensors.erase(std::string(name));\n"
                "            g_qng64_tensors.erase(std::string(name));"
            ) in mlx,
            "qng64_erased_before_native_rebind": (
                "g_qng64_tensors.erase(std::string(name));\n"
                "        g_tensors.insert_or_assign"
            ) in mlx,
            "preflight_flip_scoped_to_req_pos": (
                'r"correct req=(\\d+) pos=(\\d+) REAL FLIP orig=(\\d+) corrected=(\\d+)"'
                in preflight
                and "fpos == int(pos)" in preflight
            ),
            "pusher_credential_separate_from_event_key": (
                'api_key = os.environ.get("QWEN_SUPABASE_KEY"' in pusher
                and "event_key = (row" in pusher
            ),
            "pusher_cursor_fail_closed": (
                "if not all_ok:" in pusher
                and "cursor remains at {offset}" in pusher
                and 'pending.rfind(b"\\n")' in pusher
            ),
            "promotion_source_decoupled_from_correction": (
                'getenv("QWEN_MOE_PROMOTION_SAFETENSORS")' in qwen
                and 'moe_promotion_source_ensure_open("QWEN_MOE_PROMOTION_FILE_NQ");' in qwen
                and 'moe_promotion_source_ensure_open("QWEN_MOE_PROMOTION_FILE_NQ gpu");' in qwen
            ),
            "candidate_preflight_correction_off": (
                "correction_enabled=False" in preflight
                and '"QWEN_MOE_NEARTIE_CORRECT": "1" if correction_enabled else "0"' in preflight
                and '"QWEN_MOE_PROMOTION_SAFETENSORS": safetensors_index' in preflight
            ),
        },
        "runtime_verification": {
            "executed": False,
            "reason": "current Tailnet Commander exec policy denied the CMake build command",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default=str(ROOT / "results" / "precision_parity" / "g0_capabilities.json"),
    )
    args = ap.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "precision-capability-g0-v1",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(ROOT),
        "git_head": git_head(ROOT),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "mlx_python_version": mlx_version(),
        },
        "source_files": source_inventory(ROOT),
        "binaries": candidate_binaries(ROOT),
        "capability": source_capabilities(ROOT),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(str(output))
    print(json.dumps(payload["capability"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
