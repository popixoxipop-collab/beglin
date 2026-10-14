#!/usr/bin/env python3
"""P6 research-only capture of one GPU prefill last-position logits vector.

Accepts ONLY the P5 source after the verified C/MLX 128-position patch.
Adds one fail-closed float32 capture to the isolated persistent worker.
Never edits the tracked C source, main, deployed binary, or active route.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib

P5_C128_SHA256 = "bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612"
ANCHOR = """                rq_out[r][rq_nout[r]++] = am; rq_t_first[r] = temit;"""
HOOK = r"""                /* P6: scratch-worker first generated token logits; never production. */
                if (pass == 1 && persistent_mode) {
                    const char *p6_dest = getenv("BEGLIN_HCP_P6_LOGITS_OUTPUT");
                    if (p6_dest && p6_dest[0]) {
                        const char *p6_root = getenv("BEGLIN_HCP_P6_CAPTURE_ROOT");
                        size_t p6_prefix = p6_root ? strlen(p6_root) : 0;
                        if (!p6_root || p6_prefix < 10 ||
                            strncmp(p6_dest, p6_root, p6_prefix) != 0 ||
                            p6_dest[p6_prefix] != '/' || strstr(p6_dest, "..")) {
                            fprintf(stderr, "FATAL: P6 capture path outside run-owned scratch\n");
                            exit(1);
                        }
                        FILE *p6_fp = fopen(p6_dest, "wb");
                        if (!p6_fp) { perror("P6 logit open"); exit(1); }
                        size_t p6_written = fwrite(lg, sizeof(float), (size_t)MOE_VOCAB, p6_fp);
                        int p6_closed = fclose(p6_fp);
                        if (p6_written != (size_t)MOE_VOCAB || p6_closed != 0) {
                            fprintf(stderr, "FATAL: P6 logit write incomplete\n");
                            exit(1);
                        }
                    }
                }
"""

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def patch_text(source: str) -> str:
    if sha_bytes(source.encode("utf-8")) != P5_C128_SHA256:
        raise ValueError("P6 only accepts exact SHA-pinned P5 128-position C source")
    if source.count(ANCHOR) != 1 or source.count("REBIND_SET_APPLIED") < 1:
        raise ValueError("P6 capture anchor or REBIND_SET invariant absent")
    if "#define MOE_CBATCH_MAXPOS 128" not in source or "if (pass == 1)" not in source:
        raise ValueError("P6 not applied to proven CB128 native worker")
    return source.replace(ANCHOR, HOOK + ANCHOR, 1)

def apply(root: pathlib.Path) -> dict:
    root = root.resolve(strict=True)
    path = root / "qwen_infer.c"
    if not path.is_file() or path.is_symlink():
        raise ValueError("P6 source not an owned regular scratch file")
    original = path.read_text()
    amended = patch_text(original)
    path.write_text(amended)
    result = {
        "schema": "beglin-hcp-p6-first-logit-isolated-hook-v1",
        "p5_c128_source_sha256": P5_C128_SHA256,
        "hooked_source_sha256": sha_bytes(amended.encode()),
        "capture_scope": "single_last_prefill_position_float32_per_request",
        "model_math_unchanged": True,
        "ack_unchanged": True,
        "production_touched": False,
        "automatic_live_promotion": False,
    }
    print("P6_CAPTURE_HOOK_PATCH", json.dumps(result, sort_keys=True), flush=True)
    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-source", type=pathlib.Path, required=True)
    a = ap.parse_args()
    apply(a.scratch_source)
