#!/usr/bin/env python3
"""Build a separate BEGLIN Qwen CPU n8 SIMD whole-model test executable.

NOT the previously blocked per-tensor sensitivity workflow. This changes
only native CPU n8 matvec decoding in a SHA-pinned research clone; the model
checkpoint and the operation/production release remain untouched.
"""
from __future__ import annotations

import hashlib, json, os, pathlib, subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2]
PREVIOUS=ROOT/"lab_local/q_full_model_20261021_v2"
TARGET=ROOT/"lab_local/qng64_n8_neon_engine_20261022"
FIX=ROOT/"lab_local/q_apple_cpu_20261021_v2"
SOURCE_ORIGINAL="cb985ec48cfa1f1f9ebe8411d12da5d3f73720c8acdcad51fc279a2b25ff8956"
SOURCE_LAB="c03099aed130a5fd9222f0ba4d0fdf00d3e9bd1b1c203bf145ccb8290a0ad131"
PREFIX="BEGLIN_Q_LAB_PRECISION"
MODELREV="4ae77cb209f06199b8df1c94e21ff341332a3a89"

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def private_write(dest,blob):
    with dest.open("xb") as f:
        f.write(blob);f.flush();os.fsync(f.fileno())
    dest.chmod(0o600)

def cmd(args):
    subprocess.run(args,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

def main():
    if TARGET.exists():raise FileExistsError("will not override previous native performance experiment")
    c=ROOT/"qng64_dense.c"
    labc=PREVIOUS/"qwen_infer_q_lab.c"
    if digest(c)!=SOURCE_ORIGINAL or digest(labc)!=SOURCE_LAB:
        raise RuntimeError("source original or SHA-bound Qwen research lab drift")
    objects=list((ROOT/".build").glob("*.o"))
    if len(objects)!=22:
        raise RuntimeError("missing original native compiled 22 object files")
    os.umask(0o077)
    TARGET.mkdir(parents=True,mode=0o700)
    expected=PREVIOUS/"n8"
    target_inputs=TARGET/"n8"
    target_inputs.mkdir(mode=0o700)
    (target_inputs/"results").mkdir(mode=0o700)
    for n in ("prompt_ids.i32",):
        src=expected/n
        if src.stat().st_mode&0o777!=0o600:raise RuntimeError("old input source not secured")
        private_write(target_inputs/n,src.read_bytes())
    original_aliased=TARGET/"original_scalar_aliased.o"
    experimental=TARGET/"neon_kernel.o"
    wrapper=TARGET/"dispatch_wrapper.o"
    include=["-I",str(ROOT)]
    cmd(["clang","-O3","-w","-c",str(c),"-Dqng64_matvec_f32=beglin_original_scalar_matvec_f32",
         "-o",str(original_aliased)])
    cmd(["clang","-std=c11","-O3","-c",
         str(ROOT/"research/oml_acs/qng64_n8_neon_experimental.c"),
         "-o",str(experimental)]+include)
    cmd(["clang","-std=c11","-O3","-c",
         str(ROOT/"research/oml_acs/qng64_n8_neon_lab_wrapper.c"),
         "-o",str(wrapper)]+include)
    preserved=[p for p in objects if p.name not in ("qwen_infer.o","qng64_dense.o")]
    if len(preserved)!=20:raise RuntimeError("unexpected base object count")
    lab_qwen_obj=PREVIOUS/"qwen_infer_q_lab.o"
    if not lab_qwen_obj.is_file():raise RuntimeError("previous approved experiment object missing")
    exe=TARGET/"beglin_qwen_n8_simd_lab"
    cmd(["clang","-O3",str(lab_qwen_obj)]+[str(x) for x in preserved]+
        [str(original_aliased),str(experimental),str(wrapper),
         "-o",str(exe),"-framework","Accelerate","-lpthread","-lm"])
    exe.chmod(0o700)
    report={
      "schema":"beglin-native-n8-simd-full-model-isolated-build/1",
      "status":"SHA_PINNED_EXPERIMENTAL_BUILD_ONLY",
      "model":"Qwen/Qwen2.5-1.5B-Instruct",
      "model_local_BF16_revision":MODELREV,
      "source_original_scalar_c_sha256":SOURCE_ORIGINAL,
      "source_existing_lab_c_sha256":SOURCE_LAB,
      "source_new_neon_c_sha256":digest(ROOT/"research/oml_acs/qng64_n8_neon_experimental.c"),
      "source_lab_wrapper_c_sha256":digest(ROOT/"research/oml_acs/qng64_n8_neon_lab_wrapper.c"),
      "native_binary_sha256":digest(exe),
      "mode":"n8 only, BEGLIN CPU Qwen original full model, 28 blocks",
      "no_live_promotion":True,"main_or_production_touched":False,
      "n4_or_other_width_fallback_unchanged":True,
      "batched_matmul_path_remains_original_scalar":True,
      "no_labels_or_final_v6_data":True,
    }
    private_write(TARGET/"build_receipt.json",(json.dumps(report,indent=2,sort_keys=True)+"\n").encode())
    print("Q8_NATIVE_NEON_MODEL_RESEARCH_BUILD",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    main()
