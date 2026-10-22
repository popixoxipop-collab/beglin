#!/usr/bin/env python3
"""Source-SHA-pinned BEGLIN Qv2 selective native async stage lab.

Changes only a SHA-verified COPY of the previously audited native Qwen C
source. Fixed source-only Q4G64/n8 tensor policy, tied head and all Qwen
weights unchanged. Runtime opt-in stage mask independently toggles QKV and
Gate/Up pthread dispatch, and chooses 1t vs 6t Q4 pools.

No production source, original model, immutable operator, C1D or V6 access.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,subprocess

ROOT=pathlib.Path(__file__).resolve().parents[2]
BASE=ROOT/"lab_local/q_v2_actual_native_async_retry1_20261022"
SRC=BASE/"qwen_Qv2_native_actual_async.c"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
PRELOCK=ROOT/"configs/research/BEGLIN_QV2_SELECTIVE_NATIVE_STAGE_ABBA_PRELOCK_20261022.json"
TARGET=ROOT/"lab_local/q_v2_selective_native_async_20261022"
CORE=ROOT/"lab_local/qng64_n8_repacked_full_20261022"
EXPECTED_ORIGINAL_SHA="767ae7df399fd6fcd6dbe42da843fbbee5ede717dcb4b1693ca0782e0e3ea598"
EXPECTED_PROTOCOL_SHA="140d65b63aa3cd5dfdfae4b0edd7bb13de7157a7600e3dfd485965de208621d6"
EXPECTED_MASK_SHA="9324c5ec891d638f2713d3bbd4187399e1aae8a9a74df129dfabc4e8ea870212"
EXPECTED_BASE_BINARY_SHA="fb947372495962ed67bb00f91199030b2e41d50ab283f4a5bd81ab2e51cdc054"
MASK=BASE/"frozen_selected_28_roles.txt"
MODE_MAP=("frozen","qkv6","gu6","both6","qkv1","gu1","both1")
INCLUDE='#include "research/oml_acs/q_v2_native_async_stages.h"'
ANCHOR_QKV="qv2_dispatch3(qv2_qkv_mode_per_layer[l],"
ANCHOR_GU="qv2_dispatch2(qv2_gu_mode_per_layer[l],"

FUNCTION=r"""
/*
 * Separate lab-only selective dispatch. Master BEGLIN_QV2_NATIVE_ASYNC
 * still controls the original off path; this mask is examined only for
 * enable=1. The QKV and Gate/Up joins remain in the original stage header.
 * A "6" mode uses per-role Q4 6-thread pools, 1 uses independent 1-thread
 * pools; n8 repacked GEMV is already serial per independent role.
 *
 * Source-only tensor precision policy is not changed here.
 */
static int qv2_selective_stage_mode(int frozen,int kind) {
    static int selection=-1;
    if(selection<0) {
        const char *s=getenv("BEGLIN_QV2_NATIVE_STAGE_POLICY");
        if(!s || !strcmp(s,"frozen")) selection=0;
        else if(!strcmp(s,"qkv6")) selection=1;
        else if(!strcmp(s,"gu6")) selection=2;
        else if(!strcmp(s,"both6")) selection=3;
        else if(!strcmp(s,"qkv1")) selection=4;
        else if(!strcmp(s,"gu1")) selection=5;
        else if(!strcmp(s,"both1")) selection=6;
        else {
            fprintf(stderr,"FATAL: unrecognized isolated research native async stage policy\n");
            abort();
        }
    }
    if(kind!=2 && kind!=3)abort();
    if(selection==0)return frozen;
    if(selection==1)return kind==3 ? 6 : 0;
    if(selection==2)return kind==2 ? 6 : 0;
    if(selection==3)return 6;
    if(selection==4)return kind==3 ? 1 : 0;
    if(selection==5)return kind==2 ? 1 : 0;
    if(selection==6)return 1;
    abort();
}
"""

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def put(path:pathlib.Path,blob:bytes):
    with path.open("xb") as fp:
        fp.write(blob);fp.flush();os.fsync(fp.fileno())
    path.chmod(0o600)

def build():
    if TARGET.exists():raise FileExistsError("no overwrite already-created selective async lab")
    if sha(SRC.read_bytes())!=EXPECTED_ORIGINAL_SHA or sha(PRELOCK.read_bytes())!=EXPECTED_PROTOCOL_SHA:
        raise RuntimeError("native Qwen source or preregistered mode controls changed")
    if sha(MASK.read_bytes())!=EXPECTED_MASK_SHA:
        raise RuntimeError("frozen source activation-aware MILP precision tensor bitmask changed")
    current=BASE/"beglin_Qv2_native_async"
    if sha(current.read_bytes())!=EXPECTED_BASE_BINARY_SHA:
        raise RuntimeError("previous native full-model asynchronous benchmark executable drift")
    if not (SNAP/"model.safetensors").is_file():
        raise RuntimeError("pinned source BF16 Qwen checkpoint unavailable")
    orig=SRC.read_text()
    for token in (INCLUDE,ANCHOR_QKV,ANCHOR_GU):
        if orig.count(token)!=1:raise RuntimeError("not a unique native forward C patch site")
    result=orig.replace(INCLUDE,INCLUDE+"\n"+FUNCTION,1)
    result=result.replace(ANCHOR_QKV,
          "qv2_dispatch3(qv2_selective_stage_mode(qv2_qkv_mode_per_layer[l],3),",1)
    result=result.replace(ANCHOR_GU,
          "qv2_dispatch2(qv2_selective_stage_mode(qv2_gu_mode_per_layer[l],2),",1)
    if result.count("qv2_selective_stage_mode")!=3 or result.count(ANCHOR_QKV) or result.count(ANCHOR_GU):
        raise RuntimeError("research stage patch not correctly isolated")
    rest=sorted(x for x in (ROOT/".build").glob("*.o") if x.name not in ("qwen_infer.o","qng64_dense.o"))
    needed=[CORE/"qng64_original_aliased.o",CORE/"n8_repacked_simd.o",CORE/"n8_repacked_dispatch.o"]
    if len(rest)!=20 or any(not x.is_file() for x in needed):
        raise RuntimeError("canonical native Qwen link closure incomplete")
    os.umask(0o077);TARGET.mkdir(parents=True,mode=0o700)
    file=TARGET/"qwen_qv2_selective_stage.c"
    put(file,result.encode())
    obj=TARGET/"qwen_qv2_selective_stage.o"
    subprocess.run(["clang","-O3","-w","-I",str(ROOT),"-c",str(file),"-o",str(obj)],check=True)
    binary=TARGET/"beglin_qv2_selective_stage"
    subprocess.run(["clang","-O3",str(obj)]+[str(x) for x in needed+rest]+
            ["-o",str(binary),"-framework","Accelerate","-lpthread","-lm"],check=True)
    binary.chmod(0o700)
    lab=TARGET/"eval"
    lab.mkdir(mode=0o700)
    for sub in ("results","eval","eval/ppl_work"):
        (lab/sub).mkdir(mode=0o700,exist_ok=True)
    old=BASE/"isolated"
    for name in ("prompt_ids.i32",)+tuple(f"eval/ppl_work/w{k}.i32" for k in range(8)):
        f=old/name
        if not f.is_file() or f.stat().st_mode&0o777!=0o600:
            raise RuntimeError("original public synthetic eval inputs unavailable/insecure")
        put(lab/name,f.read_bytes())
    put(TARGET/"frozen_28_n8_policy.txt",MASK.read_bytes())
    receipt={
      "schema":"BEGLIN-Qv2-selective-native-QKV-GU-stage-lab-build/1",
      "status":"COMPILED_SOURCE_SHA_PINNED_NO_TIMING_YET",
      "original_Qwen_native_C_sha256":EXPECTED_ORIGINAL_SHA,
      "isolated_new_C_sha256":sha(file.read_bytes()),
      "new_native_binary_sha256":sha(binary.read_bytes()),
      "source_mode_prelock_sha256":EXPECTED_PROTOCOL_SHA,
      "source_bitmask_sha256":EXPECTED_MASK_SHA,
      "mode_test_family":MODE_MAP,
      "source_BF16_cache_revision":SNAP.name,
      "same_8_public_ppl_windows_copied":True,
      "native_both_stage_joins_preserved":True,
      "native_q4_pools_remain_per_role_owned":True,
      "main_or_production_touched":False,
      "prior_blocked_single_tensor_full_logit_intervention_not_retried":True,
      "scientific_quality_verdict":"NO_VERDICT"
    }
    put(TARGET/"build_receipt.json",(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode())
    print("BEGLIN_QV2_SELECTIVE_STAGE_BUILD_PASS",json.dumps(receipt,sort_keys=True),flush=True)
if __name__=="__main__":build()
