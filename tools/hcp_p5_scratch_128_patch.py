#!/usr/bin/env python3
"""P5 scratch-only 128-token cbatch KV pilot, NEVER touching fixed stage code.

Main verifier has MOE_MAXPOS=160. Online native serving is limited to 32 by
the *different* MOE_CBATCH_MAXPOS bound; changing it is nontrivial because
ragged cK/cV, near-tie side caches, output lists, stack attention scores and
per-request input matrices all scale together.

This script patches ONLY a GitHub Actions RUNNER_TEMP source copy after
asserting all known dimensions remain coupled to MOE_CBATCH_MAXPOS, and fixes
missing qng64_dense.c in the *copied* CMake target. No deployed model policy.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,re

LIMIT=128
EXPECTED_OLD=32
SOURCE_OLD="#define MOE_CBATCH_MAXPOS 32"
SOURCE_NEW="#define MOE_CBATCH_MAXPOS 128"
CMAKE_START="add_library(qwen_infer_obj OBJECT\n  qwen_infer.c\n"
SCHEMA="beglin-hcp-p5-scratch-native-kv128-source-v1"

def digest(src):
    return hashlib.sha256(src).hexdigest()

def patch(root):
    root=pathlib.Path(root).resolve(strict=True)
    runner=pathlib.Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    if root==runner or runner not in root.parents or not root.name=="src":
        raise ValueError("P5 patch allowed only in per-run scratch src")
    c=root/"qwen_infer.c"
    cpp=root/"mlx_moe.cpp"
    cmake=root/"CMakeLists.txt"
    if (not c.is_file() or not cmake.is_file() or not cpp.is_file()
        or c.is_symlink() or cmake.is_symlink() or cpp.is_symlink()):
        raise ValueError("scratch C/C++/CMake source missing/symlinked")
    text=c.read_text()
    mlxc=cpp.read_text()
    if (mlxc.count("#define MLA_L0_MAXPOS 32")!=1 or
        mlxc.count("#define GQA_L0_MAXPOS 32")!=1):
        raise ValueError("MLX GPU fused positional caps are unreviewed")
    gpu_bound_paths=(
       "if (spos[i] < 0 || spos[i] >= MLA_L0_MAXPOS)",
       "mask_bytes[(size_t)m * MLA_L0_MAXPOS + j]",
       "g_fused_K.push_back(mx::zeros({B, H, MLA_L0_MAXPOS, QHD}",
       "g_fused_V.push_back(mx::zeros({B, H, MLA_L0_MAXPOS, VHD}",
       "if (spos[i] < 0 || spos[i] >= GQA_L0_MAXPOS)",
       "mask_bytes[(size_t)m * GQA_L0_MAXPOS + j]",
       "g_fused_gqa_cK.push_back(mx::zeros({B, KVH, GQA_L0_MAXPOS, HD}",
       "g_fused_gqa_cV.push_back(mx::zeros({B, KVH, GQA_L0_MAXPOS, HD}",
    )
    absent=[v for v in gpu_bound_paths if v not in mlxc]
    if absent:
        raise ValueError("GPU native positional cache contract changed: "+repr(absent))
    if text.count(SOURCE_OLD)!=1 or "#define MOE_MAXPOS 160" not in text:
        raise ValueError("original max positions do not match reviewed C")
    mandatory=(
       "g_moe_cb_slots_cap*MOE_CBATCH_MAXPOS*MOE_KROW",
       "g_moe_cb_slots_cap*MOE_CBATCH_MAXPOS*MOE_VROW",
       "MOE_CB4C_LANES*MOE_CBATCH_MAXPOS*MOE_KROW",
       "MOE_CB4C_LANES*MOE_CBATCH_MAXPOS*MOE_VROW",
       "MOE_NEARTIE_LANES*MOE_CBATCH_MAXPOS*MOE_KROW",
       "MOE_NEARTIE_LANES*MOE_CBATCH_MAXPOS*MOE_VROW",
       "static int mf_ids[MOE_CB4B_RMAX][MOE_CBATCH_MAXPOS]",
       "rq_out[MOE_CB4B_RMAX][MOE_CBATCH_MAXPOS]",
       "if (plen[r] >= MOE_CBATCH_MAXPOS)",
       "if (plen[r] + maxnew[r] > MOE_CBATCH_MAXPOS)",
       "float scores[MOE_CBATCH_MAXPOS]",
       "REBIND_SET_APPLIED",
    )
    missing=[v for v in mandatory if v not in text]
    if missing:raise ValueError("unreviewed long KV / ACK paths: "+repr(missing))
    if text.count("#define MOE_CB4C_LANES 8")!=1:
        raise ValueError("near-tie mirror lane layout changed")
    if text.count("#define MOE_NEARTIE_LANES 8")!=1:
        raise ValueError("additional near-tie temporary layout changed")
    cm=cmake.read_text()
    if cm.count(CMAKE_START)!=1 or not (root/"qng64_dense.c").is_file():
        raise ValueError("scratch build symbol link contract changed")
    if LIMIT>160 or 112+4>LIMIT:
        raise ValueError("unsafe original P4 request length under cap")
    changed=text.replace(SOURCE_OLD,SOURCE_NEW)
    gpu_changed=(mlxc.replace("#define MLA_L0_MAXPOS 32","#define MLA_L0_MAXPOS 128")
                   .replace("#define GQA_L0_MAXPOS 32","#define GQA_L0_MAXPOS 128"))
    if gpu_changed==mlxc:
        raise ValueError("GPU source capacity unchanged")
    c.write_text(changed)
    cpp.write_text(gpu_changed)
    cmake.write_text(cm.replace(CMAKE_START,CMAKE_START+"  qng64_dense.c\n"))
    # Upper memory estimate: all cKV, side and near-tie cache lanes (4+8+8)
    # at 64 allocated layer slots, KROW+VROW<=8,192 float32.
    max_bytes=64*(4+8+8)*LIMIT*8192*4
    evidence={"schema":SCHEMA,"status":"SOURCE_KV128_PATCHED_SCRATCH_ONLY",
       "original_C_sha256":digest(text.encode()),
       "revised_C_sha256":digest(changed.encode()),
       "original_MLX_cpp_sha256":digest(mlxc.encode()),
       "revised_MLX_cpp_sha256":digest(gpu_changed.encode()),
       "mlx_native_MLA_KV_pos_limit":LIMIT,
       "mlx_native_GQA_KV_pos_limit":LIMIT,
       "scratch_max_position":LIMIT,
       "known_26_routed_layers":True,
       "C_allocated_max_layer_slots":64,
       "scratch_GPU_slot_bound":4,
       "scratch_KV_and_mirror_memory_upper_bytes":max_bytes,
       "native_original_stage_mutated":False,
       "original_single_sequence_max_pos":160,
       "all_reviewed_positioned_KV_buffers_use_same_cap":True,
       "all_reviewed_MLX_device_KV_and_attention_masks_use_same_cap":True,
       "precision_epoch_runtime_source_has_REBIND_SET":True,
       "production_touched":False,"automatic_live_promotion":False}
    print("P5_KV128_SCRATCH_CONTRACT",json.dumps(evidence,sort_keys=True))
    return evidence

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--scratch-src",required=True,type=pathlib.Path)
    a=p.parse_args();patch(a.scratch_src)
