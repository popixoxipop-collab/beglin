#!/usr/bin/env python3
"""Isolated full-model previously-frozen BEGLIN n8/q4 mixed policies with NEON n8.

Two original 28/196 mixed masks are already source-selected (weight proxy
and seed73 role-matched random), no new sensitivity selection or labels.
Only link original model source against the research n8 GEMV decoder.
Do not touch production, original engine, model files, or other experiments.
"""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
MIX=ROOT/"lab_local/q_full_model_mixed_20261021"
N8=ROOT/"lab_local/qng64_n8_neon_engine_20261022"
TARGET=ROOT/"lab_local/qng64_n8_neon_mixed_20261022"
PROXY_SHA="f65f50421cd9e9bac32bc780805942d68c3d503075ce962ce25315fdb5914cb9"
RANDOM_SHA="72065ac7a4ae5d68b1c7944dc4a67e40ac9c3c825db54a1d441ca3002ec015c5"
MIX_C_SHA="1671afa05644d4d79d1932fca4de2554a4a324222a13f34c72c7f64f889df590"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def put(p,blob):
    with p.open("xb") as f:
        f.write(blob);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)

def prepare():
    if TARGET.exists():raise FileExistsError("mixed NEON experiment exists; refuse overwrite")
    if sha(MIX/"qwen_infer_q_mixed_lab.c")!=MIX_C_SHA:
        raise RuntimeError("original mixed research source changed")
    if sha(MIX/"policy_proxy.txt")!=PROXY_SHA or sha(MIX/"policy_random.txt")!=RANDOM_SHA:
        raise RuntimeError("original role quotas/policies changed")
    baseline=json.loads((N8/"build_receipt.json").read_text())
    if baseline["status"]!="SHA_PINNED_EXPERIMENTAL_BUILD_ONLY":
        raise RuntimeError("n8 vectorized kernel build not source-verified")
    objs=list((ROOT/".build").glob("*.o"))
    preserved=[f for f in objs if f.name not in ("qwen_infer.o","qng64_dense.o")]
    if len(preserved)!=20:raise RuntimeError("original link closure incomplete")
    dependencies=[
        MIX/"mixed_lab.o",
        N8/"original_scalar_aliased.o",
        N8/"neon_kernel.o",
        N8/"dispatch_wrapper.o",
    ]
    if any(not p.is_file() for p in dependencies):
        raise RuntimeError("missing SHA-verified n8 or mixed experiment object")
    os.umask(0o077)
    TARGET.mkdir(mode=0o700,parents=True)
    for mode in ("proxy","random"):
        p=MIX/f"policy_{mode}.txt"
        put(TARGET/f"policy_{mode}.txt",p.read_bytes())
        original=MIX/f"mixed_{mode}"
        work=TARGET/f"mixed_{mode}"
        work.mkdir(mode=0o700)
        (work/"results").mkdir(mode=0o700)
        (work/"eval").mkdir(mode=0o700)
        (work/"eval/ppl_work").mkdir(mode=0o700)
        for rel in ("prompt_ids.i32","eval/ppl_work/w0.i32","eval/ppl_work/w1.i32"):
            file=original/rel
            if file.stat().st_mode&0o777!=0o600:
                raise RuntimeError("existing public input not secured")
            put(work/rel,file.read_bytes())
    binary=TARGET/"beglin_qwen_mixed_n8_neon"
    subprocess.run(["clang","-O3"]+[str(x) for x in dependencies]+
                   [str(x) for x in preserved]+["-o",str(binary),
                   "-framework","Accelerate","-lpthread","-lm"],
                   check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    binary.chmod(0o700)
    receipt={
        "schema":"beglin-native-mixed-q4-n8-neon-lab/1",
        "status":"SOURCE_PINNED_NATIVE_BUILD_NOT_RUN",
        "model":"Qwen/Qwen2.5-1.5B-Instruct",
        "scope":"28 full blocks, 168 BEGLIN q4 + 28 BEGLIN n8, original tied F32 head",
        "previous_native_mixed_source_sha256":MIX_C_SHA,
        "source_original_sensitivity_policies_unchanged":True,
        "proxy_policy_sha256":PROXY_SHA,"random_policy_sha256":RANDOM_SHA,
        "NEON_source_sha256":baseline["source_new_neon_c_sha256"],
        "NEON_wrapper_source_sha256":baseline["source_lab_wrapper_c_sha256"],
        "native_binary_sha256":sha(binary),
        "source_was_original_BF16_not_q4_upsampled":True,
        "experiment_only":True,"production_touched":False,"main_touched":False,
    }
    put(TARGET/"build_receipt.json",(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode())
    print("MIX_N8_NEON_BIN_READY",json.dumps(receipt,sort_keys=True),flush=True)
if __name__=="__main__":
    prepare()
