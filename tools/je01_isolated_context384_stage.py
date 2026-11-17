#!/usr/bin/env python3
"""JE01 isolated native 384-position BEGLIN source preparation; no build/execute.

Transforms *a local checkout copy* into a dedicated single-slot research-only
binary recipe. Never edits production source, serving routes, or original model
weights. Calling it does not establish short-prefix logit parity or real gains.
All expected anchors are exact and unique; unknown source revisions fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

CONTRACT = "beglin-je01-isolated-context384-stage-v1"
CONTEXT = 384
SLOTS = 1
ARCH_LAYERS = 32
HEADS = 16
QHEAD = 192
VHEAD = 128
FLOAT_BYTES = 4
MAX_ALLOWED_KV_BYTES = 1024 * 1024 * 1024  # isolated B=1, 1 GiB KV cap
REQUIRED_BINARY_NAME = "qwen_infer_gpu_je01_context384"
EXPECTED_MODEL_SHA = "36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc"
EXPECTED_DATASET_MANIFEST_SHA = "a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147"


def kv_bytes(context=CONTEXT, slots=SLOTS):
    if type(context) is not int or type(slots) is not int or context <= 0 or slots <= 0:
        raise ValueError("JE01_KV_BAD_SHAPE")
    return ARCH_LAYERS * HEADS * (QHEAD + VHEAD) * FLOAT_BYTES * context * slots


def exactly_one(text, original, replacement, *, where):
    n = text.count(original)
    if n != 1:
        raise ValueError(f"JE01_UNPINNED_SOURCE_ANCHOR:{where}:count={n}:{original}")
    return text.replace(original, replacement, 1)


def patch_sources(qwen, mlx, cmake):
    if not all(isinstance(t, str) and t.strip() for t in (qwen, mlx, cmake)):
        raise ValueError("JE01_SOURCE_EMPTY")
    if kv_bytes(CONTEXT, SLOTS) > MAX_ALLOWED_KV_BYTES:
        raise ValueError("JE01_SINGLE_SLOT_KV_LIMIT")
    if kv_bytes(CONTEXT, 64) <= MAX_ALLOWED_KV_BYTES:
        raise ValueError("JE01_INVALID_MULTISLOT_NEGATIVE_CONTROL")
    # Source-change scope strictly limited to three sizing constants, the
    # default cbatch slot cap, the MLA GPU B/A guards, and scratch output name.
    # DeepSeek/MLA: other GQA architecture constants stay absolutely unchanged.
    qwen = exactly_one(qwen, "#define MOE_MAXPOS 160",
                        "#define MOE_MAXPOS 384", where="qwen.MOE_MAXPOS")
    qwen = exactly_one(qwen, "#define MOE_CBATCH_MAXPOS 32",
                        "#define MOE_CBATCH_MAXPOS 384", where="qwen.MOE_CBATCH_MAXPOS")
    qwen = exactly_one(qwen,
                        "static int g_moe_cb_slots_cap = MOE_BATCH_MAX;",
                        "static int g_moe_cb_slots_cap = 1; // JE01 isolated B=1; no serving",
                        where="qwen.slot_default")
    qwen = exactly_one(qwen,
                        'if (B < 1 || B > MOE_BATCH_MAX) { fprintf(stderr, "FATAL: QWEN_MOE_CB_SLOTS',
                        'if (B != 1) { fprintf(stderr, "FATAL: QWEN_MOE_CB_SLOTS',
                        where="qwen.cbatch_launch_fence")
    # GPU fixed attention full-window shape. Only MLA (DeepSeek) is touched;
    # GQA config is intentionally preserved, avoiding invalid cross-architecture claims.
    mlx = exactly_one(mlx, "#define MLA_L0_MAXPOS 32",
                      "#define MLA_L0_MAXPOS 384", where="mlx.MLA_L0_MAXPOS")
    mlx = exactly_one(mlx,
                      "if (B < 1 || B > 64) return 0;",
                      "if (B != 1) return 0; // JE01 isolated single-slot only",
                      where="mlx.batch_fence")
    # Ragged step cannot accidentally operate on slot 1..63 or A>1.
    mlx = exactly_one(mlx,
                      "if (A < 1 || A > 64 || l < 0 || l >= MLA_MAXLAYERS) return 0;",
                      "if (A != 1 || !slot || slot[0] != 0 || !spos || "
                      "l < 0 || l >= MLA_MAXLAYERS) return 0; // JE01 single-slot",
                      where="mlx.ragged_fence")
    cmake = exactly_one(cmake,
                        "add_executable(qwen_infer_gpu\n",
                        "add_executable(qwen_infer_gpu\n",
                        where="cmake.original")
    cmake = exactly_one(cmake,
                        "target_link_libraries(qwen_infer_gpu PRIVATE mlx",
                        "set_target_properties(qwen_infer_gpu PROPERTIES "
                        "OUTPUT_NAME qwen_infer_gpu_je01_context384)\n"
                        "target_link_libraries(qwen_infer_gpu PRIVATE mlx",
                        where="cmake.output_isolation")
    return {"qwen_infer.c": qwen, "mlx_moe.cpp": mlx, "CMakeLists.txt": cmake}


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def stage(*, checkout, scratch, expected_commit):
    source = Path(checkout).resolve(strict=True)
    target = Path(scratch).absolute()
    if source == target or source in target.parents:
        raise ValueError("JE01_SOURCE_DESTINATION_CONFLICT")
    if not source.is_dir() or target.exists() or target.is_symlink():
        raise ValueError("JE01_DESTINATION_MUST_BE_NEW")
    if len(expected_commit) != 40 or any(x not in "0123456789abcdef" for x in expected_commit):
        raise ValueError("JE01_EXPECTED_GIT_SHA_REQUIRED")
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True, timeout=15).stdout.strip()
    if commit != expected_commit:
        raise ValueError("JE01_CHECKOUT_HEAD_DRIFT")
    if (source / ".git").is_file():  # worktree file is allowed if its HEAD is exact
        pass
    versions = {}
    for name in ("qwen_infer.c", "mlx_moe.cpp", "CMakeLists.txt"):
        p = source / name
        if not p.is_file() or p.is_symlink():
            raise ValueError("JE01_REQUIRED_SOURCE_MISSING_OR_SYMLINK")
        raw = p.read_bytes()
        versions[name] = {"original_sha256": sha256(raw)}
    patches = patch_sources(
        (source / "qwen_infer.c").read_text(),
        (source / "mlx_moe.cpp").read_text(),
        (source / "CMakeLists.txt").read_text(),
    )
    # Preserve the exact committed tree only. Never copy untracked checkpoints,
    # credentials, symlinks into external files, or an uncommitted dirty worktree.
    for git_args in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        if subprocess.run(["git", "-C", str(source), *git_args],
                          timeout=15).returncode != 0:
            raise ValueError("JE01_CHECKOUT_NOT_CLEAN")
    tracked = subprocess.run(["git", "-C", str(source), "ls-files", "-z"],
                             check=True, capture_output=True, timeout=20).stdout
    names = [x.decode("utf-8") for x in tracked.split(b"\x00") if x]
    if not names:
        raise ValueError("JE01_NO_TRACKED_FILES")
    target.mkdir(parents=True)
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError("JE01_TRACKED_PATH_TRAVERSAL")
        old = source / rel
        if old.is_symlink() or not old.is_file():
            raise ValueError("JE01_TRACKED_SYMLINK_OR_MISSING_FILE")
        new = target / rel
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, new)
    for name, new_text in patches.items():
        f = target / name
        if not f.is_file():
            raise ValueError("JE01_STAGED_SOURCE_MISSING")
        f.write_text(new_text, encoding="utf-8")
        versions[name]["isolated_sha256"] = sha256(f.read_bytes())
        versions[name]["modified"] = True
    out = {
        "schema": CONTRACT, "source_git_commit": commit,
        "stage_status": "SOURCE_STAGED_NOT_COMPILED",
        "isolated_binary_name": REQUIRED_BINARY_NAME,
        "target_positions": CONTEXT, "allowed_slots": SLOTS,
        "native_backend": "BEGLIN_MLX_METAL_ONLY",
        "per_slot_kv_bytes": kv_bytes(), "forbidden_64slot_kv_bytes": kv_bytes(slots=64),
        "max_allowed_kv_bytes": MAX_ALLOWED_KV_BYTES,
        "sha_pinned_source_files": versions,
        "model_identity_sha256": EXPECTED_MODEL_SHA,
        "input_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA,
        "weight_or_production_edits": False,
        "production_install_allowed": False,
        "native_model_forward": "NOT_RUN",
        "short_prefix_logit_parity": "NOT_TESTED",
        "standalone_launch_env": {"QWEN_MOE_CB_SLOTS": "1"},
    }
    # Receipt is returned to the runner; source and weights are never promoted.
    out["receipt_sha256"] = sha256(
        json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
    )
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkout", required=True)
    p.add_argument("--scratch", required=True)
    p.add_argument("--expected-commit", required=True)
    a = p.parse_args()
    try:
        print(json.dumps(stage(checkout=a.checkout, scratch=a.scratch,
                               expected_commit=a.expected_commit), sort_keys=True))
    except Exception as exc:
        print(json.dumps({"schema": CONTRACT, "stage_status": "BLOCKED",
                          "error": str(exc)[:240],
                          "native_model_forward": "NOT_RUN"}))
        raise SystemExit(2)
