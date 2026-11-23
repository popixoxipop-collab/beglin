#!/usr/bin/env python3
"""Original BEGLIN AF: capture actual 26x64 real GPU routes at 16 fixed CAL inputs.

Single NEW research-only XOX local-MLX build. Intrumentation is limited to
one code path: 384-position B1 GPU cbatch, only the final prompt position of
each of 16 frozen train calibration prompts, never gold/heldout/OOD data.

Every real native 102400-way final logit must byte-match the prior sealed
UNINSTRUMENTED original AF 384 full-vocab SHA, before a router trace is accepted.
Never modifies production source, workers, model weights, release or PR merge.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
from urllib.request import urlopen

SCHEMA="beglin-je01-original-AF-real-router26-cal16-logit-parity/1"
BASE=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original384_20261117")
RUN=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original_router16_20261118_v1")
SOURCES=BASE/"src_xox_mlxabi384"
WEIGHTS=Path("/Users/xox/vdsp_local_data/moe_base_deepseek")
MLXLIB=Path("/Users/xox/Library/Python/3.14/lib/python/site-packages/mlx/lib")
MLXCMAKE=Path("/Users/xox/Library/Python/3.14/lib/python/site-packages/mlx/share/cmake")
PREVIOUS=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_logits32_384_20261117_v1")
PREV_RAW_SHA="428d2f985d1eca80ab299a7ab8cbf3d1330391efe46791f87d013a670f5d387a"
PREV_RECEIPT_SHA="34da06b2441bd1cfa8c8593ff20e8c2d3e40850e1c0ef6d6c8a127d70cb75d14"
MANIFEST_SHA="44961869cecdacdd03ac6f38c45fada44780a083e0331b18b85d53c1fe00dcc2"
REAL512_SHA="0aef144a8d549aa9e33c3a9fd9e899e0301e826bd563ceb92ad541cff1bbf768"
SOURCE_SHA={"qwen_infer.c":"4e91092f7f860a446d97e5d871f249faeab0693d43c0293622a9cf551755622e",
 "mlx_moe.cpp":"c8b92e3ce5832b012713d41551f604855a89043ca316931501305ef501754782",
 "mlx_moe.h":None,
 "CMakeLists.txt":"f9bb057869f3fbc6a0a8ae55ab0d66247d27b211b092aa7cdfb8ab601fc023de"}
EXPERTS=64
ROUTED_LAYERS=26
TOPK=6
CASES=16
MAGIC=b"JE01_REAL_ORIGINAL_ROUTER_L26_T6_V1\n"
LOGMAGIC=b"JE01_NATIVE_FULLLOGITS_F32_V1\n"

def h(p):
    d=hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):d.update(c)
    return d.hexdigest()

def ensure(x,cond,msg):
    if not cond:raise ValueError(msg)
    return x

def replace(s,a,b,count=1,tag="ANCHOR"):
    if s.count(a)!=count:raise ValueError("JE01_SOURCE_ANCHOR_CHANGED_"+tag+":"+str(s.count(a)))
    return s.replace(a,b)

def canonical(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),
                        allow_nan=False).encode()).hexdigest()

def health():
    with urlopen("http://127.0.0.1:18765/healthz",timeout=5) as f:
        j=json.load(f)
    if j.get("status")!="ok" or j.get("route_generation")!=6 or \
       j.get("active_route",{}).get("route_id")!="beglin-candidate-shared-up-l3-n6":
        raise ValueError("JE01_PRODUCTION_HEALTH_NOT_OK")
    workers=j.get("workers",{})
    if len(workers)!=2 or any(not x.get("alive") for x in workers.values()):
        raise ValueError("JE01_PRODUCTION_PID_UNSAFE")
    ids={k:x.get("pid") for k,x in workers.items()}
    if set(ids.values())!={35481,35493}:raise ValueError("JE01_PRODUCTION_PID_CHANGED")
    return ids

def memfree():
    o=subprocess.check_output(["/usr/bin/memory_pressure","-Q"],text=True,timeout=7)
    m=re.search(r"System-wide memory free percentage:\s*(\d+)%",o)
    if not m:raise ValueError("JE01_MISSING_MEMORY_PRESSURE")
    return int(m.group(1))

def assert_source():
    for k,sha in SOURCE_SHA.items():
        p=SOURCES/k
        if not p.is_file() or (sha and h(p)!=sha):
            raise ValueError("JE01_XOX_ORIGINAL_SOURCE_DRIFT_"+k)
    if h(BASE/"manifests_xox/calibration16.manifest")!=MANIFEST_SHA:
        raise ValueError("JE01_CAL16_FROZEN_INPUT_MANIFEST_DRIFT")
    idx=json.loads((BASE/"manifests_xox/receipt.json").read_text())
    if idx.get("receipt_sha256")!=REAL512_SHA or idx.get("count_by_cohort",{}).get("calibration")!=256:
        raise ValueError("JE01_ORIGINAL_512_INPUTS_SOURCE_DRIFT")
    prev=json.loads((PREVIOUS/"JE01_SHADOW_LOGITS_32_384_FINAL.json").read_text())
    sig=prev.pop("receipt_sha256")
    if sig!=PREV_RECEIPT_SHA or canonical(prev)!=sig:
        raise ValueError("JE01_PRIOR_NATIVE_LOGITS_RECEIPT_DRIFT")
    if h(PREVIOUS/"run384/raw_full_logits.bin")!=PREV_RAW_SHA:
        raise ValueError("JE01_PRIOR_REAL_384_LOGITS_BYTES_DRIFT")
    for p in (MLXLIB/"libmlx.dylib",):
        if not p.is_file():raise ValueError("JE01_XOX_MLX_LOCAL_LIBRARY_MISSING")

def patch_mlxc(cpp):
    old="static int g_cbatch_layers_done = 0;\n"
    addition=r'''
// JE01 isolated instrumentation, never in installed MLX/serving source.
// Captures all 64-way *real model GPU softmax* and exact native Top-6; it
// neither alters the selection nor any model parameters.
static bool g_je01_real_route_capture = false;
static std::vector<mx::array> g_je01_real_route_scores;
static std::vector<mx::array> g_je01_real_route_top;
int mlx_gpu_je01_original_router_trace_begin(void) {
    if (g_je01_real_route_capture) return 0;
    g_je01_real_route_scores.clear();
    g_je01_real_route_top.clear();
    g_je01_real_route_capture = true;
    return 1;
}
int mlx_gpu_je01_original_router_trace_read(float *scores_out, int32_t *top6_out) {
    if (!g_je01_real_route_capture || !scores_out || !top6_out ||
        g_je01_real_route_scores.size() != 26 ||
        g_je01_real_route_top.size() != 26) return 0;
    try {
        std::vector<mx::array> all;
        for (size_t k=0; k<26; k++) {
            all.push_back(g_je01_real_route_scores[k]);
            all.push_back(g_je01_real_route_top[k]);
        }
        mx::eval(all);
        for (size_t k=0; k<26; k++) {
            const mx::array &a=g_je01_real_route_scores[k];
            const mx::array &b=g_je01_real_route_top[k];
            if (a.shape() != mx::Shape{1,64} || b.shape() != mx::Shape{1,6}) return 0;
            std::memcpy(scores_out+k*64, a.data<float>(), 64*sizeof(float));
            std::memcpy(top6_out+k*6, b.data<int32_t>(), 6*sizeof(int32_t));
        }
        g_je01_real_route_scores.clear();
        g_je01_real_route_top.clear();
        g_je01_real_route_capture = false;
        return 1;
    } catch (...) { return 0; }
}
'''
    cpp=replace(cpp,old,old+addition,1,"CPP_SHADOW_GLOBALS")
    first=cpp.index("int mlx_gpu_cbatch_layer_step_lazy(")
    end=cpp.index("int mlx_gpu_cbatch_forward_finalize(",first)
    section=cpp[first:end]
    anchor="mx::array top_idx = mx::astype(top_idx_u, mx::int32);\n"
    extra=r'''
            if (g_je01_real_route_capture) {
                if (A != 1 || NE != 64 || TOPK != 6 ||
                    (int)g_je01_real_route_scores.size()!=l-1)
                    throw std::runtime_error("JE01_ACTUAL_ROUTE_LAYERS_OR_SHAPE_CHANGED");
                g_je01_real_route_scores.push_back(scores);
                g_je01_real_route_top.push_back(top_idx);
            }
'''
    section=replace(section,anchor,anchor+extra,1,"CPP_REAL_CBATCH_TOP6")
    cpp=cpp[:first]+section+cpp[end:]
    return cpp

def patch_c(src):
    first=src.index("static int run_moe_gpu_cbatch_online_gate(")
    end=src.index("// V5i Phase B:",first)
    block=src[first:end]
    anchor='    int persistent_warmed = 0;\n'
    extra=r'''
    const char *je01_route_path = getenv("JE01_REAL_ROUTER_TRACE_OUTPUT");
    const char *je01_logit_path = getenv("JE01_REAL_ROUTER_LOGITS_OUTPUT");
    FILE *je01_route_file = NULL, *je01_logit_file = NULL;
    if (je01_route_path && je01_route_path[0]) {
        if (!je01_logit_path || !je01_logit_path[0] ||
            B != 1 || R != 16 || pfB != 1 || persistent_mode)
            { fprintf(stderr,"FATAL: JE01 isolated trace only B1/R16/pfB1\n"); exit(1); }
        int rr=open(je01_route_path,O_CREAT|O_EXCL|O_WRONLY,0600);
        int ll=open(je01_logit_path,O_CREAT|O_EXCL|O_WRONLY,0600);
        if (rr<0 || ll<0) { fprintf(stderr,"FATAL: JE01 scratch output already exists\n"); exit(1); }
        je01_route_file=fdopen(rr,"wb"); je01_logit_file=fdopen(ll,"wb");
        if (!je01_route_file || !je01_logit_file) exit(1);
        const char a[]="JE01_REAL_ORIGINAL_ROUTER_L26_T6_V1\n";
        const char b[]="JE01_NATIVE_FULLLOGITS_F32_V1\n";
        if (fwrite(a,1,sizeof(a)-1,je01_route_file)!=sizeof(a)-1 ||
            fwrite(b,1,sizeof(b)-1,je01_logit_file)!=sizeof(b)-1) exit(1);
    }
'''
    block=replace(block,anchor,anchor+extra,1,"C_TRACE_OUTPUT_OPEN")
    anchor2='            (void)want_logits;   // GPU forward_finalize always computes logits (no early-return path)\n'
    new2=r'''
            int je01_capture_this_step = je01_route_file && pass==1 &&
                A==1 && ndec==0 && mcb_active[slot_arr[0]]==2 &&
                spos_arr[0]==rq_plen[mcb_req[slot_arr[0]]]-1;
            if (je01_capture_this_step &&
                !mlx_gpu_je01_original_router_trace_begin()) {
                fprintf(stderr,"FATAL: JE01 real router begin error\n"); exit(1);
            }
'''
    block=replace(block,anchor2,anchor2+new2,1,"C_TRACE_BEGIN")
    anchor3='            if (validation_on && pass == 1) {\n'
    new3=r'''
            if (je01_capture_this_step) {
                float je01_64way[26*64];
                int32_t je01_top6[26*6];
                if (!mlx_gpu_je01_original_router_trace_read(je01_64way,je01_top6)) {
                    fprintf(stderr,"FATAL: JE01 native GPU 26 router trace read error\n"); exit(1);
                }
                uint32_t meta[3]={(uint32_t)mcb_req[slot_arr[0]],
                    (uint32_t)spos_arr[0],26};
                if (fwrite(meta,sizeof(uint32_t),3,je01_route_file)!=3 ||
                    fwrite(je01_64way,sizeof(float),26*64,je01_route_file)!=(size_t)(26*64) ||
                    fwrite(je01_top6,sizeof(int32_t),26*6,je01_route_file)!=(size_t)(26*6))
                    {fprintf(stderr,"FATAL: JE01 real router write error\n"); exit(1);}
                uint32_t head[4]={meta[0],meta[1],102400,
                                  (uint32_t)rq_plen[mcb_req[slot_arr[0]]]};
                if (fwrite(head,sizeof(uint32_t),4,je01_logit_file)!=4 ||
                    fwrite(gpu_logits,sizeof(float),102400,je01_logit_file)!=102400)
                    {fprintf(stderr,"FATAL: JE01 real full-logit write error\n"); exit(1);}
            }
'''
    block=replace(block,anchor3,new3+anchor3,1,"C_TRACE_AFTER_GPU_FINISH")
    anchor4='    free(x_embed); free(gpu_logits);\n    return 1;\n'
    fin=r'''
    if (je01_route_file) {
        if (fflush(je01_route_file)!=0 || fsync(fileno(je01_route_file))!=0 ||
            fclose(je01_route_file)!=0 ||
            fflush(je01_logit_file)!=0 || fsync(fileno(je01_logit_file))!=0 ||
            fclose(je01_logit_file)!=0)
            {fprintf(stderr,"FATAL: JE01 scratch flush failed\n"); exit(1);}
    }
'''
    block=replace(block,anchor4,fin+anchor4,1,"C_TRACE_FINISH")
    return src[:first]+block+src[end:]

def patch_header(hdr):
    anchor="#endif\n"
    # append to existing exported C ABI block before the closing extern C
    at=hdr.rfind("#ifdef __cplusplus")
    if at<0:raise ValueError("JE01_MLX_HEADER_EXTERN_C_END_MISSING")
    add="int mlx_gpu_je01_original_router_trace_begin(void);\nint mlx_gpu_je01_original_router_trace_read(float *scores_out,int32_t *top6_out);\n"
    return hdr[:at]+add+hdr[at:]

def build():
    if RUN.exists() or RUN.is_symlink():raise ValueError("JE01_SHADOW_OUTPUT_ROOT_ALREADY_EXISTS")
    RUN.mkdir(mode=0o700)
    dest=RUN/"src"
    shutil.copytree(SOURCES,dest)
    q=patch_c((dest/"qwen_infer.c").read_text())
    c=patch_mlxc((dest/"mlx_moe.cpp").read_text())
    hd=patch_header((dest/"mlx_moe.h").read_text())
    cm=replace((dest/"CMakeLists.txt").read_text(),
      "OUTPUT_NAME qwen_infer_gpu_je01_context384_xoxabi",
      "OUTPUT_NAME je01_real_original_router_trace384",1,"CMAKE_BINARY_NAME")
    for k,v in [("qwen_infer.c",q),("mlx_moe.cpp",c),("mlx_moe.h",hd),("CMakeLists.txt",cm)]:
        (dest/k).write_text(v)
    print(json.dumps({"event":"SHADOW_SOURCE_CREATED",
                      "source_sha256":{k:h(dest/k) for k in
                      ("qwen_infer.c","mlx_moe.cpp","mlx_moe.h","CMakeLists.txt")}}),flush=True)
    output=RUN/"build"
    p=subprocess.run(["/opt/homebrew/bin/cmake","-S",str(dest),"-B",str(output),
       "-DCMAKE_BUILD_TYPE=Release","-DMLX_CMAKE_DIR="+str(MLXCMAKE)],
       text=True,capture_output=True,timeout=80)
    if p.returncode:raise RuntimeError("JE01_CMAKE_CONFIG_FAILED:"+p.stderr[-900:])
    p=subprocess.run(["/opt/homebrew/bin/cmake","--build",str(output),
       "--target","qwen_infer_gpu","--parallel","2"],
       capture_output=True,text=True,timeout=145)
    if p.returncode:raise RuntimeError("JE01_CMAKE_BUILD_FAILED:"+p.stderr[-2000:]+p.stdout[-700:])
    exe=output/"je01_real_original_router_trace384"
    if not exe.is_file():raise RuntimeError("JE01_TRACE_EXECUTABLE_MISSING")
    print(json.dumps({"event":"SHADOW_NATIVE_BINARY_BUILT","binary_sha256":h(exe)}),flush=True)
    return exe

def verify_outputs(out,logits):
    expected=PREV_RAW_SHA
    if h(logits)!=expected:raise ValueError("JE01_INSTRUMENTATION_CHANGED_BASELINE_FULL102400_LOGITS")
    raw=out.read_bytes()
    if not raw.startswith(MAGIC):raise ValueError("JE01_TRACE_MAGIC_MISSING")
    offset=len(MAGIC)
    cases=[]
    freq=[[0]*64 for _ in range(26)]
    top6_by_layer={str(l):{} for l in range(1,27)}
    for req in range(CASES):
        head=raw[offset:offset+12]
        if len(head)!=12:raise ValueError("JE01_TRUNCATED_REAL_ROUTE_HEADER")
        rid,pos,nlayer=struct.unpack("<III",head)
        offset+=12
        if rid!=req or nlayer!=26:raise ValueError("JE01_ROUTE_ROW_ORDER_OR_LAYER_MISMATCH")
        prob=struct.unpack("<"+str(26*64)+"f",raw[offset:offset+26*64*4])
        offset+=26*64*4
        sel=struct.unpack("<"+str(26*6)+"i",raw[offset:offset+26*6*4])
        offset+=26*6*4
        event={"request":req,"native_final_prefill_position":pos,
               "per_layer_top6":[],"min_top6_margin":None}
        for li in range(26):
            r=prob[li*64:(li+1)*64]
            ids=sel[li*6:(li+1)*6]
            if len(set(ids))!=6 or any(x<0 or x>=64 for x in ids):
                raise ValueError("JE01_INVALID_NATIVE_TOP6_IDS")
            if any(not math.isfinite(x) or x<0 for x in r):
                raise ValueError("JE01_INVALID_NATIVE_ROUTER_SOFTMAX")
            if abs(math.fsum(r)-1)>1e-4:
                raise ValueError("JE01_NATIVE_SOFTMAX_SUM_DRIFT")
            gap=min(r[j] for j in ids)-max(r[j] for j in range(64) if j not in ids)
            if gap<0:raise ValueError("JE01_ROUTER_RECORDED_TOP6_IS_NOT_ARGMAX")
            for expert in ids:freq[li][expert]+=1
            event["min_top6_margin"]=gap if event["min_top6_margin"] is None else min(event["min_top6_margin"],gap)
            event["per_layer_top6"].append({"layer":li+1,"expert_ids":list(ids),
                 "min_6_vs_7_probability_gap":gap})
        cases.append(event)
    if offset!=len(raw):raise ValueError("JE01_NATIVE_ROUTER_TRACE_TRAILING_BYTES")
    rankings=sorted([{"layer":l+1,"expert":e,"calls":freq[l][e]}
                     for l in range(26) for e in range(64)],
                    key=lambda v:(-v["calls"],v["layer"],v["expert"]))
    return {"raw_router_sha256":h(out),"raw_102400_full_logits_sha256":h(logits),
      "real_route_events":CASES*ROUTED_LAYERS,"real_expert_decisions":CASES*ROUTED_LAYERS*TOPK,
      "real_router_scores":CASES*ROUTED_LAYERS*EXPERTS,
      "real_pilot_top8_occupancy_only_not_frozen_for_256":rankings[:8],
      "all_event_sums_valid":True,"all_native_top6_boundaries_nonnegative":True,
      "events":cases}

def run(exe,baseline):
    dest=RUN/"native_run"
    dest.mkdir(mode=0o700)
    out=dest/"router26_raw.bin"
    logits=dest/"logits16_102400.bin"
    stderr=dest/"stderr.txt"
    stdout=dest/"stdout.txt"
    env={"PATH":"/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
      "HOME":"/Users/xox","TMPDIR":str(dest),"DYLD_LIBRARY_PATH":str(MLXLIB),
      "QWEN_MOE_BASE":str(WEIGHTS),"QWEN_MOE_GPU_CBATCH_ONLINE":"1",
      "QWEN_MOE_CB_SLOTS":"1","QWEN_MOE_CB_REQS":"16",
      "QWEN_MOE_CB_PREFILL_BUDGET":"1","QWEN_MOE_CB_PROMPT_MANIFEST":str(BASE/"manifests_xox/calibration16.manifest"),
      "QWEN_MOE_GPU_SORT_THRESHOLD":"64","QWEN_MOE_CBATCH_BUDGET":"0",
      "JE01_REAL_ROUTER_TRACE_OUTPUT":str(out),
      "JE01_REAL_ROUTER_LOGITS_OUTPUT":str(logits)}
    minfree=memfree();start=time.monotonic();killed=None
    with stdout.open("xb") as o,stderr.open("xb") as e:
        proc=subprocess.Popen([str(exe)],cwd=str(dest),env=env,stdout=o,stderr=e)
        while proc.poll() is None:
            if time.monotonic()-start>115:killed="TIMEOUT";proc.terminate();break
            try:
                free=memfree();minfree=min(minfree,free)
                if free<35:killed="LOW_MEMORY";proc.terminate();break
                if health()!=baseline:killed="PRODUCTION_IDENTITY_DRIFT";proc.terminate();break
                ps=subprocess.run(["/bin/ps","-o","rss=","-p",str(proc.pid)],
                    capture_output=True,text=True,timeout=4).stdout.strip()
                if ps.isdigit() and int(ps)>16*1024*1024:
                    killed="RSS_EXCEEDS_16GB";proc.terminate();break
            except Exception as er:killed="MONITOR_"+type(er).__name__;proc.terminate();break
            time.sleep(.5)
        if killed and proc.poll() is None:
            try:proc.wait(timeout=8)
            except subprocess.TimeoutExpired:proc.kill()
        rc=proc.wait()
    error=stderr.read_text(errors="replace")
    if rc or killed or "FATAL:" in error or \
       "RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16" not in error:
        raise RuntimeError("JE01_REAL_ROUTER_NATIVE_EXEC_BLOCKED:"+str(killed)+"/"+str(rc)+" "+error[-1700:])
    if health()!=baseline:raise ValueError("JE01_PRODUCTION_CHANGED")
    result=verify_outputs(out,logits)
    return result,{"seconds":round(time.monotonic()-start,3),
                   "min_mem_free_percent":minfree,"exit_code":rc,
                   "native_stderr_sha256":h(stderr)}

def main():
    baseline=health()
    if memfree()<53:raise ValueError("JE01_INSUFFICIENT_REAL_FREE_MEMORY")
    assert_source()
    exe=build()
    if health()!=baseline:raise ValueError("JE01_PRODUCTION_DRIFT_AFTER_SCRATCH_BUILD")
    result,details=run(exe,baseline)
    receipt={"schema":SCHEMA,"status":"PASS_REAL_ORIGINAL_GPU_26_LAYER_16_CAL_ROUTER_PARITY",
       "actual_native_original_model_Af_weights":True,
       "source_512_manifest_sha256":REAL512_SHA,
       "calibration16_manifest_sha256":MANIFEST_SHA,
       "source_research_binary_sha256":h(exe),
       "previous_uninstrumented_full102400_sha256":PREV_RAW_SHA,
       "source_previous_full102400_receipt_sha256":PREV_RECEIPT_SHA,
       "real_calibration_only_unlabeled_router":True,
       "original_256_calibration_slot_freeze":"NOT_PERFORMED",
       "actual_bayesian_expert_NLL_effect":"NOT_MEASURED",
       "paired_heldout_quality":"NOT_MEASURED",
       "production_before":baseline,"production_after":health(),
       "source_original_weights_mutated":False,
       "production_route_mutated":False,"production_promotion_allowed":False,
       **details,**result}
    if receipt["production_before"]!=receipt["production_after"]:
        raise ValueError("JE01_PRODUCTION_CHANGED_AFTER_MODEL")
    receipt["receipt_sha256"]=canonical(receipt)
    with (RUN/"JE01_ORIGINAL_ROUTER16_REAL_FULL64_RECEIPT.json").open("x") as f:
        f.write(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"event":"FINAL_ROUTER_RECEIPT",
      "status":receipt["status"],"full_logit_sha":receipt["raw_102400_full_logits_sha256"],
      "router_events":receipt["real_route_events"],"router_decisions":receipt["real_expert_decisions"],
      "top8_pilot_only":receipt["real_pilot_top8_occupancy_only_not_frozen_for_256"],
      "min_memory_percent":receipt["min_mem_free_percent"],
      "receipt_sha256":receipt["receipt_sha256"],
      "bayesian_NLL":"NOT_MEASURED"},sort_keys=True),flush=True)

if __name__=="__main__":
    try:main()
    except BaseException as e:
        print(json.dumps({"event":"BLOCKED","type":type(e).__name__,
           "detail":str(e)[:1700],"production_promotion":False}),flush=True)
        sys.exit(2)
