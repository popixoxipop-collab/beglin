#!/usr/bin/env python3
"""BEGLIN Qv20 read-only EOE volume/weight access preflight.

Checks the already SHA-frozen Qv20 input files, original Q4/n8 reference binary,
and the *resolved* Qwen BF16 safetensors model path before reserving any GPU/CPU
capacity or running the expensive original C inference.

Never edits permissions, TCC, extended attributes, file ownership, model weights,
locked prompts, generated scoring receipts, storage pools, or production releases.
Does not read secret values, and reads only one byte per file.
"""
import errno
import json
import os
from pathlib import Path

ROOT = Path("/Volumes/D50/TailnetCommander/work/beglin-qv20-prompt-aware-20261024")
HF = Path(
    "/Users/eoe/.cache/huggingface/hub/"
    "models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/"
    "snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89/"
    "model.safetensors"
)
QWEN_C = Path(
    "/Users/eoe/mcp-sandbox/tailnet-commander/"
    "beglin-qv5-fisher-20261022/lab_local/"
    "qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_mixed"
)
PROBES = {
    "original_Qwen_BF16_3GB": HF,
    "Qv20_SHA_frozen_public_manifest": ROOT / "Qv20_PREEXEC_PUBLIC.json",
    "Qv20_train32_original_input_manifest": ROOT / "cal_arc_train/frozen_inputs_private.json",
    "Qv20_original_source_Qv8_receipt": ROOT / "native_original_C_static/cal_arc_train/static_source/PUBLIC_AGGREGATE.json",
    "Qv20_original_random73_mask": ROOT / "masks/static_random.txt",
    "Qv20_unchanged_original_C_binary": QWEN_C,
}


def probe(name, path):
    value = {"name": name, "exists": path.exists(), "is_symlink": path.is_symlink()}
    try:
        resolved = path.resolve(strict=True)
        value["resolved_on_D50"] = str(resolved).startswith("/Volumes/D50/")
        if not path.is_file():
            value["status"] = "NOT_A_REGULAR_FILE"
            return value
        with path.open("rb") as stream:
            stream.read(1)
        value["status"] = "READABLE"
    except OSError as err:
        value["status"] = "BLOCKED" if err.errno in (errno.EPERM, errno.EACCES) else "ERROR"
        value["error_code"] = errno.errorcode.get(err.errno, "UNKNOWN")
        value["error_type"] = type(err).__name__
        # Deliberately avoid printing individual prompt contents.
    return value


def main():
    results = [probe(name, path) for name, path in PROBES.items()]
    blocked = [item["name"] for item in results if item["status"] != "READABLE"]
    response = {
        "schema": "BEGLIN-Qv20-EOE-readonly-storage-and-model-prerequisites/1",
        "status": "ACCESS_READY" if not blocked else "NOT_READY_EXTERNAL_STORAGE_ACCESS",
        "uid": os.getuid(),
        "readable_required_paths": len(results) - len(blocked),
        "total_required_paths": len(results),
        "blocked_requirements": blocked,
        "results": results,
        "read_only": True,
        "does_not_prove_CPU_scheduler_storage_scratch_write_ready": True,
        "does_not_modify_SIP_TCC_provenance_chmod_xattr_or_services": True,
        "actual_random73_calibration_not_claimed": True,
    }
    print(json.dumps(response, sort_keys=True, ensure_ascii=False))
    return 0 if not blocked else 4


if __name__ == "__main__":
    raise SystemExit(main())
