#!/usr/bin/env python3
"""Research-only BEGLIN Qv2 full CPU async stage-selective A/B executable.

Do not change main, original source, existing actual async experimental lab,
model, exact source-locked 28n8/168q4 policy, evaluation windows or live release.
This fork keeps identical 3 separate worker/pool ownership, but permits
QKV-only or FFN-only async execution to measure oversubscription overhead.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2]
SRC=ROOT/"lab_local/q_v2_actual_full_cpu_async_20261022/qwen_research_actual_async_copy.c"
BASE=ROOT/"lab_local/q_v2_actual_full_cpu_async_20261022"
OUT=ROOT/"lab_local/q_v2_hybrid_full_cpu_async_20261022"
BEFORE_SHA="ea6e8db6f01ac90505754e0883f2396e5903ff7a6a6c043d33c0558684f60018"
MODEL="4ae77cb209f06199b8df1c94e21ff341332a3a89"
MASK_SHA="cb9268c9177d47705e548c2a90a6b22772569b3f2bb54dfef2767ca53ffa3e76"
ENABLED="        if (qv2_async_on()) {"
STAGE_DECL=r"""
static int qv2_async_stage_enabled(int group) {
    const char *e=getenv("BEGLIN_QV2_ASYNC_STAGE_SPEC");
    if(!e || !strcmp(e,"both"))return 1;
    if(!strcmp(e,"qkv"))return group==1;
    if(!strcmp(e,"ffn"))return group==2;
    fprintf(stderr,"FATAL: invalid research async stage selection\\n");
    abort();
}
"""
STAGE_DECL=STAGE_DECL.replace(chr(92)+chr(92)+"n",chr(92)+"n")
def sha(data):return hashlib.sha256(data).hexdigest()
def put(path,raw):
    with path.open("xb") as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def run():
    if OUT.exists():raise FileExistsError("hybrid stage experiment already exists")
    if sha(SRC.read_bytes())!=BEFORE_SHA:
        raise RuntimeError("existing actual native async C source differs from tested reference")
    if not BASE.is_dir() or sha((BASE/"policy.txt").read_bytes())!=MASK_SHA:
        raise RuntimeError("not original preselected v2 source-only 28/196 policy")
    s=SRC.read_text()
    if s.count(ENABLED)!=2 or s.count("static void rmsnorm(")!=1:
        raise RuntimeError("source patch sites nonunique")
    s=s.replace("static void rmsnorm(",STAGE_DECL+"\nstatic void rmsnorm(",1)
    s=s.replace(ENABLED,"        if (qv2_async_on() && qv2_async_stage_enabled(1)) {",1)
    s=s.replace(ENABLED,"        if (qv2_async_on() && qv2_async_stage_enabled(2)) {",1)
    if s.count("qv2_async_stage_enabled(")!=3:
        raise RuntimeError("incorrect dependency-stage function reference count")
    regular=sorted(p for p in (ROOT/".build").glob("*.o") if p.name not in ("qwen_infer.o","qng64_dense.o"))
    deps=[
       ROOT/"lab_local/qng64_n8_repacked_full_20261022/qng64_original_aliased.o",
       ROOT/"lab_local/qng64_n8_repacked_full_20261022/n8_repacked_simd.o",
       ROOT/"lab_local/qng64_n8_repacked_full_20261022/n8_repacked_dispatch.o"]
    if len(regular)!=20 or any(not x.is_file() for x in deps):
        raise RuntimeError("original compiled BEGLIN source link closure drift")
    os.umask(0o077);OUT.mkdir(parents=True,mode=0o700)
    newc=OUT/"qwen_v2_async_stage_selective_copy.c"
    put(newc,s.encode())
    obj=OUT/"qwen_v2_async_stage_selective.o"
    subprocess.run(["clang","-O3","-w","-I",str(ROOT),"-c",str(newc),"-o",str(obj)],check=True)
    exe=OUT/"beglin_qwen_stage_selective_native"
    subprocess.run(["clang","-O3",str(obj)]+[str(x) for x in deps+regular]+[
        "-o",str(exe),"-framework","Accelerate","-lpthread","-lm"],check=True)
    exe.chmod(0o700)
    for name in ("qkv_only","ffn_only"):
        f=OUT/name
        f.mkdir(mode=0o700)
        for sub in ("results","eval","eval/ppl_work"):
            (f/sub).mkdir(exist_ok=True,mode=0o700)
        original=BASE/"serial"
        for rel in ["prompt_ids.i32"]+[f"eval/ppl_work/w{i}.i32" for i in range(8)]:
            old=original/rel
            if not old.is_file() or (old.stat().st_mode&0o777)!=0o600:
                raise RuntimeError("prior native public fixed eval bytes not private")
            put(f/rel,old.read_bytes())
    put(OUT/"policy.txt",(BASE/"policy.txt").read_bytes())
    result=dict(schema="BEGLIN-Qv2-native-full-async-selective-stages/1",
        status="EXPERIMENTAL_HYBRID_BINARIES_ONLY",
        source_prior_async_C_sha256=BEFORE_SHA,
        source_hybrid_C_sha256=sha(newc.read_bytes()),
        native_hybrid_bin_sha256=sha(exe.read_bytes()),
        pinned_BF16_rev=MODEL,
        role_policy_SHA256=MASK_SHA,
        modes=["qkv_only","ffn_only"],
        async_stage_microbench_separate_from_full_model=True,
        no_quality_policy_hyperparam_change=True,
        original_v2_PPL_failure_unchanged=True,
        production_or_main_unchanged=True)
    put(OUT/"build_receipt.json",(json.dumps(result,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_QV2_HYBRID_ASYNC_STAGES_BIN_READY",json.dumps(result,sort_keys=True),flush=True)
if __name__=="__main__":run()
