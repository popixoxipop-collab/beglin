#!/usr/bin/env python3
"""Bounded isolated 128-position native C cache extension; source patch ONLY.

Do not modify main, old stage binary, deployed worker, or persistent route.
Preconditions: current code has exactly one cbatch32 capacity definition and
single-sequence MOE_MAXPOS160. Every existing C use of MOE_CBATCH_MAXPOS grows
from 32 to 128 (KV arrays, stack scores, request input/output buffers, guard).
Slot count is separately limited to one in the scratch worker.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib,re

SCHEMA="beglin-hcp-p5-scratch-capacity128-source-v1"
NEW_CB_MAX=128
ORIGINAL_CB_MAX=32
MOE_SINGLE_MAX=160
ALLOWED_SOURCE_HASH="dd73b7c7f997aa66d0a2217f9b0f3ab74c5db28b8dba1b586d4f182352bbcee1"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def patch(source:str)->str:
    if digest(source.encode())!=ALLOWED_SOURCE_HASH:
        raise ValueError("native C file does not match reviewed P5 research source SHA")
    if source.count("#define MOE_CBATCH_MAXPOS 32")!=1 or source.count("#define MOE_MAXPOS 160")!=1:
        raise ValueError("original bound definitions not unique")
    markers=["moe_cb4b_admit_guard(","moe_mla_attention_ragged(","moe_cK_row(","moe_cV_row(",
      "MOE_CBATCH_MAXPOS + 1","REBIND_SET_APPLIED","moe_gpu_rebind_set_apply_quiescent"]
    if any(m not in source for m in markers):
        raise ValueError("original native CB/cache/rebind invariants missing")
    if not (NEW_CB_MAX < MOE_SINGLE_MAX and NEW_CB_MAX >=112+4):
        raise ValueError("new CB capacity outside proven single-sequence bound")
    patched=source.replace("#define MOE_CBATCH_MAXPOS 32",
                           "#define MOE_CBATCH_MAXPOS 128")
    assert patched.count("#define MOE_CBATCH_MAXPOS 128")==1
    return patched

def apply(root):
    root=pathlib.Path(root).resolve(strict=True)
    c=root/"qwen_infer.c";cmake=root/"CMakeLists.txt"
    if (not c.is_file() or c.is_symlink() or
        not cmake.is_file() or cmake.is_symlink()):
        raise ValueError("source only regular files under owned scratch")
    before=c.read_text()
    patched=patch(before)
    build=cmake.read_text()
    anchor="add_library(qwen_infer_obj OBJECT\n  qwen_infer.c\n"
    if build.count(anchor)!=1:
        raise ValueError("expected CMake source list not found")
    c.write_text(patched)
    cmake.write_text(build.replace(anchor,anchor+"  qng64_dense.c\n"))
    receipt={"schema":SCHEMA,
        "status":"SCRATCH_ONLY_C_AND_CMAKE_PATCH",
        "original_qwen_sha256":digest(before.encode()),
        "patched_qwen_sha256":digest(patched.encode()),
        "native_cbatch_maxpos_from":ORIGINAL_CB_MAX,
        "native_cbatch_maxpos_to":NEW_CB_MAX,
        "native_single_sequence_maxpos_unchanged":MOE_SINGLE_MAX,
        "bounds_checks_not_removed":True,
        "REBIND_SET_remains_compiled":True,
        "C_math_and_router_logic_unchanged":True,
        "production_touched":False,"auto_live_promotion":False}
    print("P5_SCRATCH_CAPACITY128_PATCH",json.dumps(receipt,sort_keys=True))
    return receipt

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source-root",required=True,type=pathlib.Path)
    a=p.parse_args()
    apply(a.source_root)
