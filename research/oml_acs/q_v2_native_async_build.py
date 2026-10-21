#!/usr/bin/env python3
"""BEGLIN Q v2 actual native CPU Q/K/V + Gate/Up stage asynchronous build.

Build a new lab-only binary from original SHA-pinned existing research C copy.
Original BEGLIN CPU engine, main, immutable operator, model weights and
previous PPL/bench remain unchanged. The *source-only* per-layer 28 stage
executor modes are read from the previously frozen MILP receipt, not inferred
from new/native evaluation results.

Stage executor is bounded: reusable 2 pthread workers + caller, 3 separately
owned Q4G64 1t/6t pools, all n8 via existing in-place-repacked NEON dispatch.
Join is mandatory BEFORE reading Q/K/V or Gate/Up outputs. No speculative
cross-layer execution, no KV races. Unrecognized tensor kind fails closed.

Runtime on/off toggle in new binary:
 BEGLIN_QV2_NATIVE_ASYNC=1 -> opt-in parallel sibling stages;
 otherwise -> exact prior sequential path.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2]
OLD=ROOT/"lab_local/q_kernel_aware_native_20261022"
CORE=ROOT/"lab_local/qng64_n8_repacked_full_20261022"
SOURCE=OLD/"qwen_kernel_aware_copy.c"
SCHEDULE=ROOT/"configs/research/BEGLIN_Q_V2_FROZEN_PARALLEL_STAGE_SCHEDULE_20261022.json"
MASK=ROOT/"configs/research/BEGLIN_Q_V2_SELECTED_28_N8_ROLE_LAYER_POLICY_20261022.txt"
HEADER=ROOT/"research/oml_acs/q_v2_native_async_stages.h"
RESULT=ROOT/"lab_local/q_v2_actual_native_async_20261022"
EVAL=ROOT/"lab_local/q_v2_8public_evaluation_20261022/v2_activation_async_MILP"
EXPECTED_SOURCE="d8e149092c73d858dfb7a066f697f2b95e0668ffa747b9d66ddc4ee3617f447a"
EXPECTED_MASK="9324c5ec891d638f2713d3bbd4187399e1aae8a9a74df129dfabc4e8ea870212"
EXPECTED_MILP_PRIVATE="749d8648478b1a4b8c521de56b50821d224f85f8f247859a1d5c7d16994e068e"
TOKENS=8
STAGE_MODES={"serial_q4_6t":0,"parallel_q4_1t":1,"parallel_q4_6t":6}
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def create_file(p:pathlib.Path,raw:bytes):
    with p.open("xb") as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
def validate():
    if RESULT.exists():raise FileExistsError("this native async research run already exists")
    if sha(SOURCE.read_bytes())!=EXPECTED_SOURCE:
        raise RuntimeError("previous approved original n8-repacked native Qwen C source changed")
    j=json.loads(SCHEDULE.read_text())
    if j["source_only_MILP_private_receipt_sha256"]!=EXPECTED_MILP_PRIVATE or j["policy_sha256"]!=EXPECTED_MASK:
        raise RuntimeError("async schedule not bound to prior source-only policy")
    if sha(MASK.read_bytes())!=EXPECTED_MASK:
        raise RuntimeError("source-only Qv2 selected 28 tensor bits drift")
    if len(j["qkv_mode_by_layer"])!=28 or len(j["gate_up_mode_by_layer"])!=28:
        raise RuntimeError("must have 28 frozen layer modes")
    for name in ("qkv_mode_by_layer","gate_up_mode_by_layer"):
        if any(mode not in STAGE_MODES for mode in j[name]):
            raise RuntimeError("unrecognized native Q4 pool concurrency mode in source-only plan")
    if not HEADER.is_file() or not EVAL.is_dir():
        raise RuntimeError("research stage header or already-verified 8 public eval windows missing")
    for rel in ("prompt_ids.i32",)+tuple(f"eval/ppl_work/w{k}.i32" for k in range(TOKENS)):
        file=EVAL/rel
        if not file.is_file() or (file.stat().st_mode&0o777)!=0o600:
            raise RuntimeError("unsecured/unverified eval window "+str(file))
    return j
def patch(source,reg):
    before="static void rmsnorm(const float *x, const float *g, float *y, int n) {"
    if source.count(before)!=1:raise RuntimeError("native C header insertion anchor changed")
    codes=[
      "static const int qv2_qkv_mode_per_layer[28]={"+",".join(str(STAGE_MODES[x]) for x in reg["qkv_mode_by_layer"])+"};",
      "static const int qv2_gu_mode_per_layer[28]={"+",".join(str(STAGE_MODES[x]) for x in reg["gate_up_mode_by_layer"])+"};",
      '#include "research/oml_acs/q_v2_native_async_stages.h"',
    ]
    source=source.replace(before,"\n".join(codes)+"\n\n"+before,1)
    f=source.index("static void forward_token(")
    start=source.index("        if (prof_on()) tp0 = nowt();\n        if (fused_dispatch() & 1) matvec_t(",f)
    end=source.index("        if (prof_on()) tp0 = nowt();\n        for(int h=0;h<g_cfg.nh;h++) rope_apply",start)
    old=source[start:end]
    if old.count("matvec_t(")!=3 or len(old)>1800:
        raise RuntimeError("original Q/K/V calculation source patch drift")
    insert="""        if (qv2_async_enabled()) {
            if (fused_dispatch()!=0) {
                fprintf(stderr,"FATAL: Qv2 async source requires non-fused qkv/gateup roles\\n");
                abort();
            }
            qv2_dispatch3(qv2_qkv_mode_per_layer[l],
                  g_role_wt[ROLE_ATTN_Q][l],g_role_wt[ROLE_ATTN_K][l],
                  g_role_wt[ROLE_ATTN_V][l],hbuf,
                  g_qbias_l[l],g_kbias_l[l],g_vbias_l[l],q,kv_k,kv_v);
        } else {
"""+old+"""        }
"""
    source=source[:start]+insert+source[end:]
    start=source.index("        if (prof_on()) tp0 = nowt();\n        if (fused_dispatch() & 2) matvec_t(",f)
    end=source.index("        if (prof_on()) tp0 = nowt();\n        swiglu(gate,up",start)
    old=source[start:end]
    if old.count("matvec_t(")!=2 or len(old)>1500:
        raise RuntimeError("original Gate/Up source patch drift")
    insert="""        if (qv2_async_enabled()) {
            if (fused_dispatch()!=0) {
                fprintf(stderr,"FATAL: Qv2 async source requires non-fused qkv/gateup roles\\n");
                abort();
            }
            qv2_dispatch2(qv2_gu_mode_per_layer[l],
                g_role_wt[ROLE_MLP_GATE][l],g_role_wt[ROLE_MLP_UP][l],
                hbuf,gate,up);
        } else {
"""+old+"""        }
"""
    source=source[:start]+insert+source[end:]
    if source.count("qv2_dispatch3(")!=1 or source.count("qv2_dispatch2(")!=1:
        raise RuntimeError("not unique patched QKV/GU native dispatch")
    return source
def build():
    sch=validate()
    os.umask(0o077)
    RESULT.mkdir(parents=True,mode=0o700)
    text=patch(SOURCE.read_text(),sch)
    s=RESULT/"qwen_Qv2_native_actual_async.c"
    create_file(s,text.encode("utf8"))
    obj=RESULT/"qwen_Qv2_native_actual_async.o"
    subprocess.run(["clang","-O3","-w","-I",str(ROOT),"-c",str(s),"-o",str(obj)],check=True)
    others=sorted(f for f in (ROOT/".build").glob("*.o") if f.name not in ("qwen_infer.o","qng64_dense.o"))
    links=[CORE/"qng64_original_aliased.o",CORE/"n8_repacked_simd.o",CORE/"n8_repacked_dispatch.o"]
    if len(others)!=20 or any(not p.is_file() for p in links):
        raise RuntimeError("original/native BEGLIN qng64 link dependencies incomplete")
    binary=RESULT/"beglin_Qv2_native_async"
    subprocess.run(["clang","-O3",str(obj)]+[str(p) for p in links]+[str(p) for p in others]+
         ["-o",str(binary),"-framework","Accelerate","-lpthread","-lm"],check=True)
    binary.chmod(0o700)
    where=RESULT/"isolated"
    where.mkdir(mode=0o700)
    for sub in ("results","eval","eval/ppl_work","logs"):
        (where/sub).mkdir(exist_ok=True,mode=0o700)
    for rel in ("prompt_ids.i32",)+tuple(f"eval/ppl_work/w{k}.i32" for k in range(TOKENS)):
        create_file(where/rel,(EVAL/rel).read_bytes())
    create_file(RESULT/"frozen_selected_28_roles.txt",MASK.read_bytes())
    receipt=dict(schema="BEGLIN-Qv2-actual-pthread-QKV-GU-native-lab-build/1",
       status="COMPILED_NOT_YET_EVALUATED",
       original_BEG_source_c_SHA256=EXPECTED_SOURCE,
       isolated_experimental_C_SHA256=sha(s.read_bytes()),
       experimental_header_SHA256=sha(HEADER.read_bytes()),
       schedule_source_sha256=sha(SCHEDULE.read_bytes()),
       originally_frozen_MILP_private_SHA256=EXPECTED_MILP_PRIVATE,
       original_q4_n8_tensor_policy_SHA256=EXPECTED_MASK,
       source_model_BF16_revision="4ae77cb209f06199b8df1c94e21ff341332a3a89",
       native_binary_SHA256=sha(binary.read_bytes()),
       stage3_modes=sch["qkv_mode_by_layer"],stage2_modes=sch["gate_up_mode_by_layer"],
       threadpool_model="persistent pthread 2 worker+caller; each role owns distinct 1/6 thread Q4 pools",
       safe_join_before_attention_and_gate_swiglu=True,
       native_CPU_activation_source_not_modified=True,
       original_live_engine_or_main_modified=False,
       no_evaluation_used_to_choose_modes=True,
       full_model_async_latency_still_unverified=True,
       original_C1D_and_V6_24_unopened=True)
    create_file(RESULT/"build_receipt.json",(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode())
    print("BEGLIN_QV2_ACTUAL_NATIVE_ASYNC_COMPILER_PASS",json.dumps({
      "binary":receipt["native_binary_SHA256"],"source":receipt["isolated_experimental_C_SHA256"],
      "qkv_modes":dict((x,sch["qkv_mode_by_layer"].count(x)) for x in STAGE_MODES),
      "gateup_modes":dict((x,sch["gate_up_mode_by_layer"].count(x)) for x in STAGE_MODES)
    },sort_keys=True),flush=True)
if __name__=="__main__":build()
