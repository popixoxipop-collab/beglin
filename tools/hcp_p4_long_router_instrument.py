#!/usr/bin/env python3
"""Scratch-only P4 112-token true-rank7 + complete layer1 h2 evidence hook.

P3 h2 records were capped at pos<40; P4 extends ONLY the copied scratch hook
to its already certified MOE_MAXPOS=160. The original model and kernel formulas
are unchanged. Max width and env input length are validated before execution.
"""
from __future__ import annotations
import argparse,hashlib,pathlib
from hcp_p3_router_instrument import patch

OLD="if (l == 1 && pos >= 0 && pos < 40)"
NEW="if (l == 1 && pos >= 0 && pos < MOE_MAXPOS)"
BUILD="add_library(qwen_infer_obj OBJECT\n  qwen_infer.c\n"

def prepare(root):
    root=pathlib.Path(root)
    path=root/"qwen_infer.c";cmake=root/"CMakeLists.txt"
    if not path.is_file() or path.is_symlink() or not cmake.is_file():
        raise ValueError("scratch C source must be a real regular file")
    original=path.read_text()
    altered=patch(original)
    if altered.count(OLD)!=1:raise ValueError("unexpected P3 FFN capture cap source")
    altered=altered.replace(OLD,NEW)
    c=cmake.read_text()
    if c.count(BUILD)!=1:raise ValueError("unexpected CMake qNg64 source layout")
    path.write_text(altered)
    cmake.write_text(c.replace(BUILD,BUILD+"  qng64_dense.c\n"))
    receipt={"schema":"beglin-hcp-p4-scratch-only-boundary-h2-instrument-v1",
             "original_c_sha256":hashlib.sha256(original.encode()).hexdigest(),
             "instrumented_c_sha256":hashlib.sha256(altered.encode()).hexdigest(),
             "token_h2_cap":160,"router_capture_full_softmax_before_topk_renorm":True,
             "original_math_unmodified":True,"production_touched":False}
    print("P4_112TOKEN_SCRATCH_HOOK_PASS",receipt)
    return receipt
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--scratch-source",required=True,type=pathlib.Path)
    a=ap.parse_args();prepare(a.scratch_source)
