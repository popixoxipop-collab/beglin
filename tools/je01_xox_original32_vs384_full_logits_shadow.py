#!/usr/bin/env python3
"""JE01 original DeepSeek 32 vs 384 full-logit native shadow, XOX ONLY.

A separate, new scratch-only C/MLX source tree is built for each context bound.
Both share the same original BEGLIN compressed AF weights and copied source,
except the *isolated* 32 vs 384 context constants. A debug-only C output hook
captures all 102400 next-token float32 logits for 16 prespecified, real frozen
calibration prompts. Rejected if production route, worker PIDs, original
sha-pinned source/binary or memory-safety gates drift. Does not modify
production worker, source code, weights, release, or trained Expert precision.
"""
from __future__ import annotations
from array import array
import hashlib
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

SCHEMA="BEGLIN-JE01-ORIGINAL-AF-FULL102400-LOGITS-32-VS384/1"
ROOT=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original384_20261117")
SOURCES=ROOT/"src_xox_mlxabi384"
RUNROOT=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_logits32_384_20261117_v1")
MLXLIB=Path("/Users/xox/Library/Python/3.14/lib/python/site-packages/mlx/lib")
MLX_CMAKE_DIR=Path("/Users/xox/Library/Python/3.14/lib/python/site-packages/mlx/share/cmake")
WEIGHTS=Path("/Users/xox/vdsp_local_data/moe_base_deepseek")
ROUTE="beglin-candidate-shared-up-l3-n6"
MIN_FREE=35
SOURCE_SHAS={
 "qwen_infer.c":"4e91092f7f860a446d97e5d871f249faeab0693d43c0293622a9cf551755622e",
 "mlx_moe.cpp":"c8b92e3ce5832b012713d41551f604855a89043ca316931501305ef501754782",
 "CMakeLists.txt":"f9bb057869f3fbc6a0a8ae55ab0d66247d27b211b092aa7cdfb8ab601fc023de"}
ORIGINAL_16_NEXT=(11,279,2238,31460,330,72672,185,1501,
                  245,11,457,285,327,20171,889,68307)
RAW_MAGIC=b"JE01_NATIVE_FULLLOGITS_F32_V1\n"
VOCAB=102400
R=16

def emit(tag,**values):
    print(json.dumps({"stage":tag,**values},sort_keys=True),flush=True)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as fd:
        for chunk in iter(lambda:fd.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def precisely(s,old,new,count,label):
    if s.count(old)!=count:raise ValueError("JE01_SOURCE_ANCHOR_DRIFT:"+label+":"+str(s.count(old)))
    return s.replace(old,new)

def health():
    with urlopen("http://127.0.0.1:18765/healthz",timeout=6) as f:
        doc=json.load(f)
    if doc.get("status")!="ok" or doc.get("route_generation")!=6 or \
       doc.get("active_route",{}).get("route_id")!=ROUTE:
        raise ValueError("JE01_PRODUCTION_SERVICE_CHANGED")
    workers=doc.get("workers",{})
    if len(workers)!=2 or not all(w.get("alive") for w in workers.values()):
        raise ValueError("JE01_ORIGINAL_PRODUCTION_WORKERS_CHANGED")
    return {"generation":6,"route":ROUTE,
            "worker_pids":{key:w["pid"] for key,w in workers.items()}}

def free_pct():
    result=subprocess.run(["/usr/bin/memory_pressure","-Q"],
                          text=True,capture_output=True,timeout=8,check=True)
    m=re.search(r"System-wide memory free percentage:\s*(\d+)%",result.stdout)
    if not m:raise ValueError("JE01_HOST_PRESSURE_SENSOR_MISSING")
    return int(m.group(1))

def patch_source(qwen,mlx,cmake,arm):
    if arm not in (32,384):raise ValueError("JE01_ARM_IS_NOT_32_OR_384")
    assert qwen.find("JE01_NATIVE_FULLLOGITS_F32_V1")<0
    qwen=precisely(qwen,
      '    int persistent_mode = persistent_dir && persistent_dir[0];\n',
      '    int persistent_mode = persistent_dir && persistent_dir[0];\n'
      '    // JE01 only: new scratch binary, exclusive O_EXCL full-vocab logits output.\n'
      '    const char *je01_debug_path = getenv("JE01_SHADOW_FULL_LOGITS_PATH");\n'
      '    FILE *je01_full_logit_file = NULL;\n'
      '    if (je01_debug_path && je01_debug_path[0]) {\n'
      '        if (B != 1 || R != 16 || pfB != 1 || persistent_mode) {\n'
      '            fprintf(stderr,"FATAL: JE01 logit output requires B=1 R=16 pfB=1 nonpersistent\\n"); exit(1);\n'
      '        }\n'
      '        int je01_fd = open(je01_debug_path, O_WRONLY|O_CREAT|O_EXCL, 0600);\n'
      '        if (je01_fd < 0) { fprintf(stderr,"FATAL: JE01 cannot exclusively create logit output\\n"); exit(1); }\n'
      '        je01_full_logit_file = fdopen(je01_fd,"wb");\n'
      '        if (!je01_full_logit_file) { close(je01_fd); exit(1); }\n'
      '        const char je01_magic[] = "JE01_NATIVE_FULLLOGITS_F32_V1\\n";\n'
      '        if (fwrite(je01_magic,1,sizeof(je01_magic)-1,je01_full_logit_file)!=sizeof(je01_magic)-1) exit(1);\n'
      '    }\n',
      1,"enable_shadow_logits")
    anchor='                float *lg = gpu_logits + (size_t)m * MOE_VOCAB;\n' \
           '                int am = moe_gpu_argmax_finite(lg, MOE_VOCAB, "moe gpu cb online", step, r);\n'
    repl='                float *lg = gpu_logits + (size_t)m * MOE_VOCAB;\n' \
         '                if (je01_full_logit_file && pass == 1) {\n' \
         '                    if (MOE_VOCAB != 102400) { fprintf(stderr,"FATAL: JE01 wrong vocab\\n"); exit(1); }\n' \
         '                    uint32_t record_head[4] = {(uint32_t)r,(uint32_t)spos_arr[m],(uint32_t)MOE_VOCAB,(uint32_t)rq_plen[r]};\n' \
         '                    if (fwrite(record_head,sizeof(uint32_t),4,je01_full_logit_file)!=4 ||\n' \
         '                        fwrite(lg,sizeof(float),MOE_VOCAB,je01_full_logit_file)!=(size_t)MOE_VOCAB) {\n' \
         '                        fprintf(stderr,"FATAL: JE01 full logits write failed\\n"); exit(1);\n' \
         '                    }\n' \
         '                }\n' \
         '                int am = moe_gpu_argmax_finite(lg, MOE_VOCAB, "moe gpu cb online", step, r);\n'
    # This anchor occurs twice (decode + prefill); patch LAST/second only,
    # and insist no silent change in ordinary decode branch.
    if qwen.count(anchor)!=2:
        raise ValueError("JE01_EXPECTED_TWO_ARGMAX_LOCATIONS:"+str(qwen.count(anchor)))
    loc=qwen.rfind(anchor)
    qwen=qwen[:loc]+qwen[loc:].replace(anchor,repl,1)
    # The same cleanup text appears in eight *other* MoE gate helpers.
    # Scope the exact change to the unique GPU online gate, fail closed.
    begin=qwen.index("static int run_moe_gpu_cbatch_online_gate(")
    end=qwen.index("// V5i Phase B:",begin)
    isolated=qwen[begin:end]
    isolated=precisely(isolated,
      '    free(x_embed); free(gpu_logits);\n    return 1;\n',
      '    if (je01_full_logit_file) {\n'
      '        if (fflush(je01_full_logit_file)!=0 || fsync(fileno(je01_full_logit_file))!=0) exit(1);\n'
      '        if (fclose(je01_full_logit_file)!=0) exit(1);\n'
      '    }\n'
      '    free(x_embed); free(gpu_logits);\n    return 1;\n',1,"shadow_logits_flush_online_only")
    qwen=qwen[:begin]+isolated+qwen[end:]
    if arm==32:
        qwen=precisely(qwen,"#define MOE_MAXPOS 384","#define MOE_MAXPOS 160",1,"CPU_32_BASE")
        qwen=precisely(qwen,"#define MOE_CBATCH_MAXPOS 384","#define MOE_CBATCH_MAXPOS 32",1,"GPU_32_MAX")
        mlx=precisely(mlx,"#define MLA_L0_MAXPOS 384","#define MLA_L0_MAXPOS 32",1,"MLA_32_MAX")
    cmake=precisely(cmake,
        "OUTPUT_NAME qwen_infer_gpu_je01_context384_xoxabi",
        "OUTPUT_NAME je01_original_shadow_full_logits_context"+str(arm),
        1,"scratch_only_executable_name")
    return {"qwen_infer.c":qwen,"mlx_moe.cpp":mlx,"CMakeLists.txt":cmake}

def build(arm,before):
    folder=RUNROOT/("src"+str(arm))
    outdir=RUNROOT/("build"+str(arm))
    if folder.exists() or outdir.exists():raise ValueError("JE01_SHADOW_BUILD_ALREADY_EXISTS")
    shutil.copytree(SOURCES,folder,symlinks=False)
    patched=patch_source(*[(folder/k).read_text() for k in SOURCE_SHAS],arm=arm)
    for name,body in patched.items():
        (folder/name).write_text(body)
    subprocess.run(["/opt/homebrew/bin/cmake","-S",str(folder),"-B",str(outdir),
                    "-DCMAKE_BUILD_TYPE=Release","-DMLX_CMAKE_DIR="+str(MLX_CMAKE_DIR)],
                   check=True,capture_output=True,text=True,timeout=75)
    emit("SCRATCH_CMAKE_CONFIGURED",arm=arm)
    p=subprocess.run(["/opt/homebrew/bin/cmake","--build",str(outdir),
                      "--target","qwen_infer_gpu","--parallel","2"],
                      capture_output=True,text=True,timeout=145)
    if p.returncode!=0:
        raise RuntimeError("JE01_SCRATCH_BUILD_FAILED:"+p.stderr[-1300:]+p.stdout[-1300:])
    binary=outdir/("je01_original_shadow_full_logits_context"+str(arm))
    if not binary.is_file():raise ValueError("JE01_EXPECTED_BINARY_MISSING")
    if health()!=before:raise ValueError("JE01_PRODUCTION_CHANGED_DURING_BUILD")
    free=free_pct()
    if free<45:raise RuntimeError("JE01_MEMORY_PRESSURE_LOW_DURING_BUILD")
    result={"arm":arm,"sha256":sha(binary),
            "patched_source_sha256":{k:sha(folder/k) for k in patched},
            "context_limit":arm,"exit_code":0,"system_free_percent":free}
    emit("SHADOW_BINARY_BUILT",**result)
    return binary,result

def run_arm(arm,binary,baseline):
    folder=RUNROOT/("run"+str(arm))
    if folder.exists():raise ValueError("JE01_SHADOW_RUN_ALREADY_EXISTS")
    folder.mkdir(mode=0o700)
    dump=folder/"raw_full_logits.bin"
    stdout=folder/"stdout.txt";stderr=folder/"stderr.txt"
    env={"PATH":"/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
         "HOME":"/Users/xox","TMPDIR":str(folder),
         "DYLD_LIBRARY_PATH":str(MLXLIB),
         "QWEN_MOE_BASE":str(WEIGHTS),
         "QWEN_MOE_GPU_CBATCH_ONLINE":"1",
         "QWEN_MOE_CB_SLOTS":"1","QWEN_MOE_CB_REQS":"16",
         "QWEN_MOE_CB_PREFILL_BUDGET":"1",
         "QWEN_MOE_CB_PROMPT_MANIFEST":str(ROOT/"manifests_xox"/"calibration16.manifest"),
         "QWEN_MOE_GPU_SORT_THRESHOLD":"64",
         "QWEN_MOE_GPU_CBATCH_BUDGET":"0",
         "JE01_SHADOW_FULL_LOGITS_PATH":str(dump)}
    lowest=free_pct();start=time.monotonic();killed=None
    emit("SHADOW_ARM_STARTED",arm=arm,initial_free=lowest)
    with stdout.open("xb") as output,stderr.open("xb") as errors:
        proc=subprocess.Popen([str(binary)],cwd=folder,env=env,stdout=output,stderr=errors)
        while proc.poll() is None:
            if time.monotonic()-start>95:
                killed="TIMEOUT";proc.terminate();break
            try:
                free=free_pct();lowest=min(free,lowest)
                if free<MIN_FREE:killed="LOW_MEMORY";proc.terminate();break
                if health()!=baseline:killed="PRODUCTION_DRIFT";proc.terminate();break
                result=subprocess.run(["/bin/ps","-o","rss=","-p",str(proc.pid)],
                                      capture_output=True,text=True,timeout=4)
                ram=result.stdout.strip()
                if ram.isdigit() and int(ram)>16*1024*1024:
                    killed="RSS_BOUND_EXCEEDED";proc.terminate();break
            except BaseException as error:
                killed="MONITOR_ERROR_"+type(error).__name__;proc.terminate();break
            time.sleep(.5)
        if killed and proc.poll() is None:
            try:proc.wait(timeout=8)
            except subprocess.TimeoutExpired:proc.kill()
        rc=proc.wait()
    er=stderr.read_text(errors="replace")
    if (killed or rc or "FATAL:" in er or
        "RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16" not in er):
        raise RuntimeError("JE01_NATIVE_PROBE_FAIL:"+str(killed)+":"+str(rc)+":"+er[-1200:])
    if health()!=baseline:raise ValueError("JE01_POST_ARM_PRODUCTION_DRIFT")
    if not dump.is_file():raise ValueError("JE01_NATIVE_LOGITS_DUMP_MISSING")
    duration=round(time.monotonic()-start,3)
    result={"arm":arm,"exit_code":rc,"seconds":duration,"free_min":lowest,
            "raw_logits_sha256":sha(dump),"log_bytes":dump.stat().st_size,
            "stderr_sha256":sha(stderr),
            "production_pids_unchanged":True}
    emit("ARM_REAL_FULL_LOGITS_CAPTURED",**result)
    return dump,result

def parse(path):
    with path.open("rb") as fp:
        if fp.readline()!=RAW_MAGIC:raise ValueError("JE01_WRONG_NATIVE_LOGIT_HEADER")
        rows=[]
        for i in range(R):
            head=fp.read(16)
            if len(head)!=16:raise ValueError("JE01_MISSING_FULLLOGIT_RECORD")
            r,pos,vocab,promptlen=struct.unpack("<4I",head)
            if r!=i or pos!=promptlen-1 or vocab!=VOCAB or promptlen>31:
                raise ValueError("JE01_WRONG_RECORD_POSITION_OR_VOCAB")
            data=fp.read(VOCAB*4)
            if len(data)!=VOCAB*4:raise ValueError("JE01_TRUNCATED_FULLLOGIT_ARRAY")
            vec=array("f");vec.frombytes(data)
            if len(vec)!=VOCAB or any(not math.isfinite(t) for t in vec):
                raise ValueError("JE01_NONFINITE_NATIVE_LOGITS")
            top=max(range(VOCAB),key=vec.__getitem__)
            if top!=ORIGINAL_16_NEXT[i]:
                raise ValueError("JE01_NATIVE_LOGITS_DIFFER_FROM_FROZEN_GREEDY:"+str(i))
            rows.append({"request":r,"position":pos,"prompt_tokens":promptlen,
                         "argmax":top,"float32":vec,"record_sha256":hashlib.sha256(head+data).hexdigest()})
        if fp.read():raise ValueError("JE01_TRAILING_RECORDS")
    return rows

def evaluate(a,b):
    ar=parse(a);br=parse(b);maxdiff=0.0;sumdiff=0.0;different=0
    scored=[]
    for aa,bb in zip(ar,br):
        if aa["request"]!=bb["request"] or aa["position"]!=bb["position"]:
            raise ValueError("JE01_DIFFERENT_CASE_ORDER")
        va,vb=aa["float32"],bb["float32"]
        dmax=0.0;dmean=0.0
        for x,y in zip(va,vb):
            diff=abs(x-y)
            dmax=max(dmax,diff);dmean+=diff
            if diff: different+=1
        maxdiff=max(maxdiff,dmax);sumdiff+=dmean
        scored.append({"request":aa["request"],"input_positions":aa["prompt_tokens"],
                       "argmax_old32":aa["argmax"],"argmax_new384":bb["argmax"],
                       "max_logit_abs_delta":dmax,
                       "mean_abs_logit_delta":dmean/VOCAB,
                       "source32_full102400_sha256":aa["record_sha256"],
                       "source384_full102400_sha256":bb["record_sha256"]})
    return {"sample_count":16,"vocab_count":VOCAB,"all_logits_compared":16*VOCAB,
            "max_abs_logit_difference":maxdiff,
            "mean_abs_logit_difference":sumdiff/(16*VOCAB),
            "count_exact_different_float32_values":different,
            "all_greedy_token_ids_match":True,
            "cases":scored}

def main():
    baseline=health()
    free=free_pct()
    if free<55:raise ValueError("JE01_BEFORE_PRODUCTION_MEMORY_FREE_BELOW_55")
    if RUNROOT.exists() or RUNROOT.is_symlink():raise ValueError("JE01_EXPERIMENT_OUTPUT_ALREADY_EXISTS")
    receipt=json.loads((ROOT/"manifests_xox/receipt.json").read_text())
    if receipt.get("receipt_sha256")!="0aef144a8d549aa9e33c3a9fd9e899e0301e826bd563ceb92ad541cff1bbf768":
        raise ValueError("JE01_REAL512_INPUT_RECEIPT_DRIFT")
    for n,h in SOURCE_SHAS.items():
        if sha(SOURCES/n)!=h:raise ValueError("JE01_ORIGINAL384_SOURCE_SHA_CHANGED:"+n)
    for n in ("arch_config_moe.txt","layout_af.txt","layout_f32.txt",
              "deepseek_moe_af.bin","deepseek_moe_f32.bin"):
        p=WEIGHTS/"weights_moe"/n
        if not p.is_file():raise ValueError("JE01_ORIGINAL_AF_SOURCE_MISSING")
    if not (MLXLIB/"libmlx.dylib").is_file():raise ValueError("JE01_MLX_ABI_LIBRARY_MISSING")
    RUNROOT.mkdir(mode=0o700)
    output={"schema":SCHEMA,"status":"RUNNING","source_original384_sha":SOURCE_SHAS,
            "same_original_AF_weights_no_mutation":True,
            "actual_full_model_27_layers":True,"same_source_real_calibration_manifest_sha256":
            sha(ROOT/"manifests_xox/calibration16.manifest"),
            "production_before":baseline,
            "native_original_512_gold_NLL":"NOT_MEASURED",
            "production_route_modified":False,"release_modified":False}
    bins={}
    for arm in (32,384):
        b,r=build(arm,baseline)
        bins[arm]=b
        output["build_"+str(arm)]=r
    dumps={}
    for arm in (32,384):
        d,r=run_arm(arm,bins[arm],baseline)
        dumps[arm]=d
        output["run_"+str(arm)]=r
    output["paired_full_logits"]=evaluate(dumps[32],dumps[384])
    output["production_after"]=health()
    if output["production_after"]!=baseline:raise ValueError("JE01_EXIT_PRODUCTION_DRIFT")
    output["min_free_pct_after"]=free_pct()
    output["status"]="PASS_FULL_102400_LOGIT_PAIRED_COMPARISON"
    output["production_promotion_allowed"]=False
    output["receipt_sha256"]=hashlib.sha256(json.dumps(output,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    with (RUNROOT/"JE01_SHADOW_LOGITS_32_384_FINAL.json").open("x") as f:
        f.write(json.dumps(output,sort_keys=True,indent=2)+"\n")
    emit("EXPERIMENT_FINAL",status=output["status"],
         receipt_sha256=output["receipt_sha256"],
         native_model_pretrained=True,full_logit_comparison=output["paired_full_logits"],
         production_same=True)

if __name__=="__main__":
    try:main()
    except BaseException as error:
        emit("EXPERIMENT_FAILED_OR_BLOCKED",type=type(error).__name__,detail=str(error)[:1300],
             production_write_allowed=False)
        sys.exit(2)
