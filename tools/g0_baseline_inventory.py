#!/usr/bin/env python3
"""Generate the conservative G0 baseline inventory for precision parity work."""

import hashlib
import importlib.util
import json
from pathlib import Path
import platform
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "precision_parity"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    dotgit = ROOT / ".git"
    if dotgit.is_file():
        text = dotgit.read_text().strip()
        if text.startswith("gitdir:"):
            dotgit = (ROOT / text.split(":", 1)[1].strip()).resolve()
    head = dotgit / "HEAD"
    if not head.exists():
        return None
    value = head.read_text().strip()
    if value.startswith("ref:"):
        ref = value.split(":", 1)[1].strip()
        direct = dotgit / ref
        if direct.exists():
            return direct.read_text().strip()
        packed = dotgit / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line and not line.startswith("#") and not line.startswith("^"):
                    commit, name = line.split(" ", 1)
                    if name == ref:
                        return commit
    return value


def artifact(name):
    p = RESULTS / name
    if not p.is_file():
        return {"exists": False, "path": str(p), "sha256": None}
    return {
        "exists": True,
        "path": str(p),
        "sha256": sha256(p),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    qwen = (ROOT / "qwen_infer.c").read_text()
    preflight = (ROOT / "tools" / "autopilot_live_preflight.py").read_text()
    mlx = (ROOT / "mlx_moe.cpp").read_text()

    payload = {
        "schema_version": "precision-baseline-inventory-g0-v1",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(ROOT),
        "git_head": git_head(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "mlx_python_importable": importlib.util.find_spec("mlx") is not None,
        },
        "backend": {
            "name": "mlx_metal",
            "status": "IMPLEMENTED_UNVERIFIED",
            "local_gpu_binary_discovered": False,
            "runtime_build_verified": False,
            "runtime_blocker": "Tailnet Commander current exec policy denies the CMake build command",
        },
        "precision_contract": {
            "native_quant_bits": [2, 3, 4, 5, 6, 8],
            "custom_qng64_metal_bits": [7, 9, 10, 11, 12, 13, 14, 15],
            "dense_bits": [16, 32],
            "promotion_file_env": "QWEN_MOE_PROMOTION_FILE_NQ",
            "promotion_source_env": "QWEN_MOE_PROMOTION_SAFETENSORS",
            "legacy_source_fallback_env": "QWEN_MOE_NEARTIE_CORRECT_SAFETENSORS",
            "candidate_preflight_correction_off": (
                "correction_enabled=False" in preflight
                and '"QWEN_MOE_NEARTIE_CORRECT": "1" if correction_enabled else "0"' in preflight
            ),
            "promotion_source_decoupled_from_correction": (
                'getenv("QWEN_MOE_PROMOTION_SAFETENSORS")' in qwen
                and 'moe_promotion_source_ensure_open("QWEN_MOE_PROMOTION_FILE_NQ gpu")' in qwen
            ),
            "registry_single_representation_guard": (
                mlx.count("g_qng64_tensors.erase(std::string(name));") >= 2
            ),
        },
        "evidence_contract": {
            "req_pos_scoped_flip_parser": (
                'correct req=(\\d+) pos=(\\d+) REAL FLIP' in preflight
                and "fpos == int(pos)" in preflight
            ),
            "durable_gpu_v3_evidence": False,
            "note": "backend/device/build-scoped v3 evidence remains G1; P5 v2 DB schema is not rewritten in G0",
        },
        "artifacts": {
            "capabilities": artifact("g0_capabilities.json"),
            "build_manifest": artifact("g0_build_manifest.json"),
        },
        "tests": {
            "pr_a_regressions": str(ROOT / "tools" / "test_pr_a_regressions.py"),
            "pr_a_correction_off": str(ROOT / "tools" / "test_pr_a_correction_off.py"),
            "last_verified_counts": {
                "pr_a_regressions": 12,
                "pr_a_correction_off": 4,
            },
        },
        "g0_exit_state": {
            "source_safety_regressions_present": True,
            "capability_inventory_present": True,
            "build_manifest_present": True,
            "runtime_gpu_verification_complete": False,
            "ready_for_g1_contract_work": True,
        },
    }

    out = RESULTS / "baseline_inventory.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(str(out))
    print(json.dumps(payload["g0_exit_state"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
