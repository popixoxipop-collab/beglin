#!/usr/bin/env python3
"""BEGLIN Qv15 isolated native Apple CPU build, bounded inactive-precision tensor cache.
Qv14 v3 live WT switch is the EXACT pinned source; original inference kernels untouched.
Cache is one alternate *quantized tensor* variant per live projection tensor, NOT a
second whole-model checkpoint. Cap counts packed bytes + scales and is hard enforced.
"""
import pathlib,hashlib,subprocess,json,os
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv15-bounded-cache-20261023")
BUILD=LAB/"cached_original_C_native_build"
Q14=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv14-inprocess-20261023/native_original_C_dynamic_repack_v3")
QWEN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
SRC=Q14/"qv14_from_original_qv5_mixed_dynamic.c"
PIN_SRC="07a69e0a67cac0c448d6c7f6ec23cc506d95cf9a3bfa905acbb1d41abbbd7702"
PIN_ORIGINAL="375b3dfb02ac2c0fc90f3f500718578fe62d38cdcac541099fb896797470dbeb"
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(262144),b""):h.update(b)
 return h.hexdigest()
def put(p,contents):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open("xb") as f:f.write(contents);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
SLOT= r'''
/* QV15 bounded in-process precision cache. Research only.
   A cache slot is ALWAYS the inactive format for one tensor (when present).
   The live WT owns its active packed/scales. Cached buffers are never mapped to
   different tensors; a slot uses its exact role-layer and its original dimensions.
   Memory budget counts INACTIVE packed+scales; total process RSS is separately measured.
   Startup requires BEGLIN_QV15_CACHE_MIB exactly 0, 32, 64 or 96.
*/
typedef struct {
    uint8_t *packed;
    float *scales;
    int bits;
    size_t bytes;
} Qv15CacheSlot;
static Qv15CacheSlot qv15_slots[N_LAYER_ROLES][64];
static size_t qv15_max_bytes=0,qv15_cache_bytes=0,qv15_peak_cache_bytes=0;
static unsigned long qv15_hits=0,qv15_misses=0,qv15_drops=0;
static void qv15_cache_init(void){
    const char *v=getenv("BEGLIN_QV15_CACHE_MIB");
    if(!v||(strcmp(v,"0") && strcmp(v,"32") && strcmp(v,"64") && strcmp(v,"96"))){
        fprintf(stderr,"FATAL: Qv15 cache MiB must be 0,32,64,96\n");exit(1);
    }
    long mb=strtol(v,NULL,10);
    if(mb!=0&&mb!=32&&mb!=64&&mb!=96){fprintf(stderr,"FATAL: Qv15 cap range\n");exit(1);}
    qv15_max_bytes=(size_t)mb*1048576u;
    fprintf(stderr,"[QV15-CACHE-INIT] capacity_MiB %ld capacity_bytes %zu\n",mb,qv15_max_bytes);
}
static size_t qv15_variant_bytes(const WT *w,int bits){
    size_t n=(size_t)w->out*(size_t)w->in;
    return (bits==8?qng64_packed_bytes(w->out,w->in,8):n/2)+
        (size_t)w->out*(size_t)w->ng*sizeof(float);
}
static void qv15_cache_save_old(Qv15CacheSlot *slot,WT *w,
                               uint8_t *oldpack,float *oldscale,int oldbits){
    size_t size=qv15_variant_bytes(w,oldbits);
    if(slot->packed){fprintf(stderr,"FATAL: Qv15 slot must be empty before saving\n");exit(1);}
    if(qv15_cache_bytes<=qv15_max_bytes && size<=qv15_max_bytes-qv15_cache_bytes){
        slot->packed=oldpack;slot->scales=oldscale;slot->bits=oldbits;
        slot->bytes=size;qv15_cache_bytes+=size;
        if(qv15_cache_bytes>qv15_peak_cache_bytes)qv15_peak_cache_bytes=qv15_cache_bytes;
    }else{
        free(oldpack);free(oldscale);qv15_drops++;
    }
    if(qv15_cache_bytes>qv15_max_bytes){
        fprintf(stderr,"FATAL: Qv15 hard inactive cache cap exceeded\n");exit(1);
    }
}
'''
# Important: keep C source helper fully opt-in and only exchange originals for same shape.
NEW_FUNC=r'''static void qv14_repack_one(int rid,int layer,int target_n8){
    char name[96];snprintf(name,sizeof name,ROLE_PATTERN_HF[rid],layer);
    WT *w=wt(name);
    if(w->kai_rhs || w->kai_rhs_bytes || w->sub || w->f32){
        fprintf(stderr,"FATAL: Qv15 refuses aliased/non-owning WT %s\n",name);exit(1);
    }
    int currently_n8=(w->kind==K_QNG64&&w->bits==8);
    if(w->kind!=K_Q4G64&&!currently_n8){
        fprintf(stderr,"FATAL: Qv15 not native packed %s\n",name);exit(1);
    }
    if(currently_n8==target_n8)return;
    Qv15CacheSlot *slot=&qv15_slots[rid][layer];
    int oldbits=currently_n8?8:4;
    uint8_t *oldpack=(uint8_t *)w->packed;
    float *oldscale=(float *)w->scales;
    if(!oldpack||!oldscale){fprintf(stderr,"FATAL: Qv15 packed pointer missing\n");exit(1);}
    if(slot->packed && slot->bits==(target_n8?8:4)){
        uint8_t *newpack=slot->packed;float *newscale=slot->scales;
        if(slot->bytes>qv15_cache_bytes){fprintf(stderr,"FATAL: Qv15 cache accounting underflow\n");exit(1);}
        qv15_cache_bytes-=slot->bytes;
        memset(slot,0,sizeof *slot);
        w->packed=newpack;w->scales=newscale;
        w->kind=target_n8?K_QNG64:K_Q4G64;w->bits=target_n8?8:4;
        w->kai_rhs=NULL;w->kai_rhs_bytes=0;w->kai_lazy_failed=0;
        qv15_cache_save_old(slot,w,oldpack,oldscale,oldbits);
        qv15_hits++;
        return;
    }
    if(slot->packed){
        fprintf(stderr,"FATAL: Qv15 cached variant not opposite for %s\n",name);exit(1);
    }
    /* Cache miss: original Qv14 numerical path, unchanged. */
    SafetensorsFile *shard=NULL;
    const SafetensorsInfo *t=safetensors_multi_find_tensor(g_st,name,&shard);
    if(!t||!shard||t->n_dims!=2||!safetensors_dequant_supported(t->dtype)
       ||(long long)t->shape[0]!=w->out||(long long)t->shape[1]!=w->in
       ||t->n_elements!=(uint64_t)w->out*(uint64_t)w->in||w->in%64){
        fprintf(stderr,"FATAL: Qv15 original BF16 tensor unavailable/shape mismatch %s\n",name);exit(1);
    }
    size_t count=(size_t)w->out*(size_t)w->in;
    float *deq=malloc(sizeof(float)*count);
    size_t nb=target_n8?qng64_packed_bytes(w->out,w->in,8):count/2;
    uint8_t *packed=malloc(nb);
    float *scales=malloc(sizeof(float)*(size_t)w->out*(size_t)w->ng);
    if(!deq||!packed||!scales){
        fprintf(stderr,"FATAL: Qv15 memory allocation %s\n",name);exit(1);
    }
    safetensors_dequant_row(t->dtype,safetensors_tensor_data(shard,t),deq,t->n_elements);
    if(target_n8){
        if(qng64_quantize_f32(deq,w->out,w->in,8,packed,scales)!=0 ||
           beglin_qng64_n8_repack_inplace(packed,w->out,w->in)!=0){
            fprintf(stderr,"FATAL: Qv15 native n8 quant/repack %s\n",name);exit(1);
        }
    }else{
        gguf_quantize_q4g64_error_feedback(deq,w->out,w->in,packed,scales);
    }
    free(deq);
    w->packed=packed;w->scales=scales;w->kind=target_n8?K_QNG64:K_Q4G64;
    w->bits=target_n8?8:4;
    w->kai_rhs=NULL;w->kai_rhs_bytes=0;w->kai_lazy_failed=0;
    qv15_cache_save_old(slot,w,oldpack,oldscale,oldbits);
    qv15_misses++;
}
'''
def main():
 if BUILD.exists():raise FileExistsError("Qv15 derived cache C source already built; no overwrite")
 if digest(SRC)!=PIN_SRC:raise RuntimeError("Qv14 verified pinned v3 original source drift")
 old=SRC.read_text()
 begin="static void qv14_repack_one(int rid,int layer,int target_n8){"
 end="static double qv14_clock_ms(void){"
 if old.count(begin)!=1 or old.count(end)!=1:raise RuntimeError("source splice anchors ambiguous")
 b=old.index(begin);e=old.index(end,b)
 new=old[:b]+SLOT+"\n"+NEW_FUNC+old[e:]
 a='    qv14_profile=0;\n}'
 if new.count(a)!=1:raise RuntimeError("qv14 init anchor drift")
 new=new.replace(a,'    qv15_cache_init();\n    qv14_profile=0;\n}',1)
 stats='    int changed=0;double start=qv14_clock_ms();'
 if new.count(stats)!=1:raise RuntimeError("switch counter anchor drift")
 new=new.replace(stats,'    unsigned long prevhits=qv15_hits,prevmisses=qv15_misses;\n'+stats,1)
 end_snip='''      window,prefix,target==0?"source_qv8":"random73",changed,elapsed);
}'''
 if new.count(end_snip)!=1:raise RuntimeError("native real switch receipt anchor drift")
 new=new.replace(end_snip,'''      window,prefix,target==0?"source_qv8":"random73",changed,elapsed);
    fprintf(stderr,"[QV15-CACHE] window %d hits %lu misses %lu drops_total %lu cache_bytes %zu peak_cache_bytes %zu cache_cap_bytes %zu\\n",
        window,qv15_hits-prevhits,qv15_misses-prevmisses,qv15_drops,
        qv15_cache_bytes,qv15_peak_cache_bytes,qv15_max_bytes);
}''',1)
 # Append total stats without touching original qv5 token scoring.
 orig='''        if(qv14_active)fprintf(stderr,"[QV14-DYNAMIC-TOTAL] switches %d tensors_repacked %d switch_ms %.5f\\n",
            qv14_switches_total,qv14_weight_repacked,qv14_switch_ms_total);'''
 if new.count(orig)!=1:raise RuntimeError("native total marker drift")
 new=new.replace(orig,orig+'''
        if(qv14_active)fprintf(stderr,"[QV15-CACHE-TOTAL] hits %lu misses %lu drops %lu bytes %zu peak_bytes %zu cap_bytes %zu\\n",
           qv15_hits,qv15_misses,qv15_drops,qv15_cache_bytes,qv15_peak_cache_bytes,qv15_max_bytes);''',1)
 os.umask(0o077);BUILD.mkdir(parents=True,exist_ok=False,mode=0o700)
 p=BUILD/"qv15_from_pinned_original_Qv14_v3.c";put(p,new.encode())
 obj=BUILD/"qv15_source.o"; exe=BUILD/"beglin_qv15_bounded_cache"
 c=subprocess.run(["clang","-O3","-w","-I",str(QWEN),"-c",str(p),"-o",str(obj)],capture_output=True,text=True,timeout=180)
 put(BUILD/"compile_log.txt",(c.stdout+"\n"+c.stderr).encode())
 if c.returncode:raise RuntimeError("C compile failed "+c.stderr[-2000:])
 # Match original source v3 build-link provenance. 23 original object files.
 first=[QWEN/"lab_local/qng64_n8_repacked_full_20261022"/f for f in
     ("qng64_original_aliased.o","n8_repacked_simd.o","n8_repacked_dispatch.o")]
 other=sorted(x for x in (QWEN/".build").glob("*.o") if x.name not in ("qwen_infer.o","qng64_dense.o"))
 if len(other)!=20 or any(not x.is_file() for x in first):raise RuntimeError("original NEON object link closure changed")
 link=subprocess.run(["clang","-O3",str(obj)]+[str(o) for o in first+other]+["-o",str(exe),"-framework","Accelerate","-lpthread","-lm"],capture_output=True,text=True,timeout=180)
 put(BUILD/"link_log.txt",(link.stdout+"\n"+link.stderr).encode())
 if link.returncode:raise RuntimeError("C link failed "+link.stderr[-2000:])
 exe.chmod(0o700)
 receipt={"schema":"BEGLIN-Qv15-original-28layer-inprocess-bounded-cache-C-build/1",
 "status":"ISOLATED_C_Q4_N8_BOUNDED_INACTIVE_VARIANT_CACHE_COMPILED_UNTESTED",
 "original_Qv14_v3_source_sha256":PIN_SRC,"derived_C_sha256":digest(p),
 "derived_native_executable_sha256":digest(exe),
 "compile_log_sha256":digest(BUILD/"compile_log.txt"),
 "link_log_sha256":digest(BUILD/"link_log.txt"),
 "original_quantized_kernel_object_sha256":[{"file":str(x),"sha256":digest(x)} for x in first+other],
 "cache_allowed_MiB":[0,32,64,96],"same_original_model_and_quantization_math":True,
 "memory_budget_applies_to":"inactive quantized packed bytes + scales, counted but additional model BF16 mmap/transient dequant buffers and runtime RSS are separately measured",
 "cache_policy":"one inactive variant per changing tensor; no whole duplicated model, no silent unbounded cache, cache refusal falls back to BF16 read and original quant kernel",
 "real_time_or_general_quality_unverified":True,"gpu_execution_unverified":True,
 "no_operating_release_change":True}
 put(BUILD/"BUILD_RECEIPT.json",(json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode())
 print("Qv15_COMPILED",json.dumps({"exe":str(exe),"binary_sha256":digest(exe),"derived_source_sha256":digest(p),
 "pinned_v3_sha256":PIN_SRC,"native_kernel_objects":len(first+other)},sort_keys=True),flush=True)
if __name__=="__main__":main()
