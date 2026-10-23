#!/usr/bin/env python3
"""Qv16 explicit-startup preparation of alternative quantized tensor buffers.
Derived ONLY from pinned Qv15 role-paired C. Same original Apple CPU Qwen28
forward kernels/scorer. Research only; no original C, checkpoint or release edits.
"""
import pathlib,hashlib,os,subprocess,json,difflib
SRC=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv15-bounded-cache-20261023/cached_original_C_native_build_paired_v2/qv15_original_C_paired_cache_v2.c")
PIN="cf4a3bafa1e20887d97f44b66bcc29dccaef6b93027dbb9471f873e0ba219d7f"
QW=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023")
DEST=LAB/"derived_startup_prewarm_C_v1"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for q in iter(lambda:f.read(262144),b""):h.update(q)
 return h.hexdigest()
def save(p,b):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
HELPER=r'''
/* Qv16: optional explicit startup warmup, no scoring or forward during warmup.
   For EACH changing role-layer: materialize alternative from pinned BF16 using
   Qv15's unchanged live WT re-quantizer, then switch back to original Qv8
   (a guaranteed cache hit). Keep ONLY inactive alternative, not second full
   quantized model. Fail closed if predicted inactive packed+scales > cap.
   Dedicated precompute wall time and extra residency are always published. */
static int qv16_prewarm_enabled=0;
static double qv16_prewarm_ms=0;
static int qv16_precomputed_tensors=0;
static void qv16_precompute_startup(void){
    const char *enabled=getenv("BEGLIN_QV16_PREWARM");
    if(!enabled||!strcmp(enabled,"0"))return;
    if(strcmp(enabled,"1")){fprintf(stderr,"FATAL: Qv16 PREWARM must be 0 or 1\n");exit(1);}
    if(qv15_max_bytes != (size_t)96*1048576){
        fprintf(stderr,"FATAL: Qv16 explicit prewarm requires bound 96MiB inactive cache\n");exit(1);
    }
    size_t projected_cache=0;
    int changing=0;
    for(int r=0;r<7;r++)for(int l=0;l<28;l++){
        int role=qv14_roles_id[r];
        if(qv14_mask_source[role][l]==qv14_mask_random[role][l])continue;
        char name[96];snprintf(name,sizeof name,ROLE_PATTERN_HF[role],l);
        WT *w=wt(name);
        int target=(int)qv14_mask_random[role][l];
        size_t size=qv15_variant_bytes(w,target?8:4);
        if(size>qv15_max_bytes-projected_cache){
            fprintf(stderr,"FATAL: Qv16 prewarm alt bytes exceed 96MiB cap\n");exit(1);
        }
        projected_cache+=size;changing++;
    }
    if(changing!=40){fprintf(stderr,"FATAL: Qv16 exactly 40 changing role-layers expected, got %d\n",changing);exit(1);}
    double start=qv14_clock_ms();
    for(int r=0;r<7;r++)for(int l=0;l<28;l++){
        int role=qv14_roles_id[r];
        int original=(int)qv14_mask_source[role][l];
        int target=(int)qv14_mask_random[role][l];
        if(original==target)continue;
        qv14_repack_one(role,l,target);
        qv14_repack_one(role,l,original);
        qv16_precomputed_tensors++;
    }
    qv16_prewarm_ms=qv14_clock_ms()-start;
    qv14_check_state(qv14_mask_source);
    if(qv16_precomputed_tensors!=40||qv15_cache_bytes!=projected_cache ||
       qv15_hits!=40||qv15_misses!=40||qv15_drops!=0){
        fprintf(stderr,"FATAL: Qv16 preload cache count/reconstruction mismatch\n");exit(1);
    }
    fprintf(stderr,"[QV16-PREWARM] prepared %d startup_ms %.5f inactive_cache_bytes %zu cap_bytes %zu "
                   "startup_native_requantizations %lu startup_swap_hits %lu\n",
       qv16_precomputed_tensors,qv16_prewarm_ms,qv15_cache_bytes,qv15_max_bytes,
       qv15_misses,qv15_hits);
    /* START OF MEASURED REQUEST PERIOD -- precompute isn't counted as a request hit. */
    qv15_hits=0;qv15_misses=0;qv15_drops=0;
    qv16_prewarm_enabled=1;
}
'''
def run():
 if DEST.exists():raise FileExistsError("Qv16 isolated C build already exists")
 if sha(SRC)!=PIN:raise RuntimeError("pinned Qv15 original C changed")
 old=SRC.read_text()
 marker='static void qv14_prepare_profiles(void){'
 if old.count(marker)!=1:raise RuntimeError("Qv15 startup anchor drift")
 new=old.replace(marker,HELPER+'\n'+marker,1)
 init='    qv15_cache_init();\n    qv14_profile=0;'
 if new.count(init)!=1:raise RuntimeError("Qv15 init anchor drift")
 new=new.replace(init,'    qv15_cache_init();\n    qv16_precompute_startup();\n    qv14_profile=0;',1)
 # No other changed math or original scoring fields.
 b=old.index('            for(int p=0;p<n;p++)forward_token(ids[p],p,xtmp,0);')
 e=old.index('        if(qv14_active)fprintf(stderr,"[QV14-DYNAMIC-TOTAL]',b)
 bb=new.index('            for(int p=0;p<n;p++)forward_token(ids[p],p,xtmp,0);')
 ee=new.index('        if(qv14_active)fprintf(stderr,"[QV14-DYNAMIC-TOTAL]',bb)
 if old[b:e]!=new[bb:ee]:raise RuntimeError("source original Qwen28 math changed")
 diff=list(difflib.unified_diff(old.splitlines(),new.splitlines(),fromfile="Qv15-paired",tofile="Qv16-prewarm",n=2))
 if sum(x.startswith("@@") for x in diff)!=2:raise RuntimeError("startup patch should only have 2 C hunks")
 os.umask(0o077);DEST.mkdir(parents=True,exist_ok=False,mode=0o700)
 code=DEST/"qv16_from_original_Qv15_paired.c";save(code,new.encode())
 save(DEST/"pinned_source_diff.patch",("\n".join(diff)+"\n").encode())
 obj=DEST/"qv16_prewarm_source.o";exe=DEST/"beglin_qv16_bounded_cache_prewarm"
 c=subprocess.run(["clang","-O3","-w","-I",str(QW),"-c",str(code),"-o",str(obj)],
   capture_output=True,text=True,timeout=180)
 save(DEST/"compile_log.txt",(c.stdout+"\n"+c.stderr).encode())
 if c.returncode:raise RuntimeError("isolated native C compile failed "+c.stderr[-3000:])
 objs=[QW/"lab_local/qng64_n8_repacked_full_20261022"/n for n in
   ("qng64_original_aliased.o","n8_repacked_simd.o","n8_repacked_dispatch.o")]
 others=sorted(x for x in (QW/".build").glob("*.o") if x.name not in ("qwen_infer.o","qng64_dense.o"))
 if len(others)!=20 or any(not f.is_file() for f in objs):raise RuntimeError("original 23 object closure drift")
 link=subprocess.run(["clang","-O3",str(obj)]+[str(f) for f in objs+others]+["-o",str(exe),"-framework","Accelerate","-lpthread","-lm"],
   capture_output=True,text=True,timeout=180)
 save(DEST/"link_log.txt",(link.stdout+"\n"+link.stderr).encode())
 if link.returncode:raise RuntimeError("original kernel link failed "+link.stderr[-3000:])
 exe.chmod(0o700)
 receipt={"schema":"BEGLIN-Qv16-original-Qwen28-C-explicit-startup-prewarm-isolated-build/1",
 "state":"EXACT_PINNED_QV15_SOURCE_WITH_OPTIONAL_PREWARM_COMPILED",
 "pinned_parent_C_SHA256":PIN,"derived_source_SHA256":sha(code),
 "derived_native_CPU_binary_SHA256":sha(exe),
 "original_unchanged_NEON_and_Q4_linked_object_SHA256":[{"file":str(f),"sha256":sha(f)} for f in objs+others],
 "source_diff_hunks":2,"no_forward_NLL_scorer_math_changed":True,
 "startup_prewarm_only_if_env":"BEGLIN_QV16_PREWARM=1",
 "explicit_96MiB_only_for_prewarm":True,
 "prewarm_sets_final_active_mask_source":"Qv8",
 "prewarm_quantizes_40_inactive_tensor_variants_from_source_BF16_then_swaps_back":True,
 "request_latency_performance_not_yet_measured":True,
 "startup_computation_time_not_removed":True,
 "no_original_model_or_operating_release_change":True}
 save(DEST/"BUILD_RECEIPT.json",(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode())
 print("QV16_ISOLATED_PREWARM_CPU_BUILT",json.dumps({"original":PIN,"derived_C_sha256":sha(code),
 "binary_sha256":sha(exe),"linked_original_objects":23,"path":str(exe)},sort_keys=True),flush=True)
if __name__=="__main__":run()
