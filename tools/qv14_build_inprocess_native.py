#!/usr/bin/env python3
"""Build ISOLATED original 28-layer Qwen C in-process mixed-mask re-quantization lab.
Original Qv5 source, precision codecs, weights, immutable release are untouched.
"""
import hashlib,json,os,pathlib,subprocess,time,difflib
R=pathlib.Path(__file__).resolve().parents[1]
SOURCE=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin/qv5_mixed.c")
E=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv14-inprocess-20261023")
BUILD=LAB/"native_original_C_dynamic_repack_v2"
ORIGSHA="375b3dfb02ac2c0fc90f3f500718578fe62d38cdcac541099fb896797470dbeb"
ORIGEXESHA="f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221"
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(p,data):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def main():
 if BUILD.exists():raise FileExistsError("immutable Qv14 research native binary already prepared")
 if sha(SOURCE)!=ORIGSHA:raise RuntimeError("original mixed loader source changed")
 oldbin=SOURCE.parent/"beglin_qv5_mixed"
 if sha(oldbin)!=ORIGEXESHA:raise RuntimeError("original 28 n8 binary changed")
 src=SOURCE.read_text()
 helper=r'''
/* QV14 isolated in-process experiment: exact-original BF16 -> Q4 or n8
 * re-quantization on a live WT. Never holds both full-model bit masks resident.
 * Select mask from TOKEN LENGTH ONLY, never from expected answers or logits. */
static const char *qv14_roles[7]={"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"};
static const int qv14_roles_id[7]={ROLE_ATTN_Q,ROLE_ATTN_K,ROLE_ATTN_V,ROLE_ATTN_O,ROLE_MLP_GATE,ROLE_MLP_UP,ROLE_MLP_DOWN};
static int qv14_read_mask(const char *file,unsigned char bits[N_LAYER_ROLES][64]){
    if(!file||!file[0]||g_cfg.nl!=28)return -1;
    FILE *f=fopen(file,"rb");if(!f)return -2;
    memset(bits,0,sizeof(unsigned char)*N_LAYER_ROLES*64);
    char name[40],extra=0;int layer=0,counts[7]={0},n=0;
    while(1){
        int got=fscanf(f," %39s %d",name,&layer);
        if(got==EOF)break;
        if(got!=2){fclose(f);return -3;}
        int j=-1;
        for(int k=0;k<7;k++)if(!strcmp(name,qv14_roles[k])){j=k;break;}
        if(j<0||layer<0||layer>=28||bits[qv14_roles_id[j]][layer]){
            fclose(f);return -4;
        }
        bits[qv14_roles_id[j]][layer]=1;counts[j]++;n++;
    }
    fclose(f);if(n!=28)return -5;
    return 0;
}
static void qv14_check_state(const unsigned char bits[N_LAYER_ROLES][64]){
    int n8=0,q4=0;
    for(int r=0;r<7;r++)for(int l=0;l<28;l++){
        char name[96];snprintf(name,sizeof name,ROLE_PATTERN_HF[qv14_roles_id[r]],l);
        WT *w=wt(name);int want=bits[qv14_roles_id[r]][l];
        if(want){
            if(w->kind!=K_QNG64 || w->bits!=8){fprintf(stderr,"FATAL: Qv14 n8 role mismatch %s\n",name);exit(1);}
            n8++;
        }else{
            if(w->kind!=K_Q4G64){fprintf(stderr,"FATAL: Qv14 Q4 role mismatch %s\n",name);exit(1);}
            q4++;
        }
    }
    if(n8!=28||q4!=168){fprintf(stderr,"FATAL: Qv14 role totals %d %d\n",n8,q4);exit(1);}
}
/* The existing WT packed/scales belong to the original safetensors registration
 * malloc, not an mmap. We only mutate 196 projection WTs in THIS isolated binary.
 * Commit the new WTs only after successful quant/repack; no second full model. */
static void qv14_repack_one(int rid,int layer,int target_n8){
    char name[96];snprintf(name,sizeof name,ROLE_PATTERN_HF[rid],layer);
    WT *w=wt(name);
    if(w->kai_rhs || w->kai_rhs_bytes || w->sub || w->f32){
        fprintf(stderr,"FATAL: Qv14 cannot retune aliased/non-owning WT %s\n",name);exit(1);
    }
    int currently_n8=(w->kind==K_QNG64&&w->bits==8);
    if(w->kind!=K_Q4G64&&!currently_n8){fprintf(stderr,"FATAL: Qv14 not native packed %s\n",name);exit(1);}
    if(currently_n8==target_n8)return;
    SafetensorsFile *shard=NULL;
    const SafetensorsInfo *t=safetensors_multi_find_tensor(g_st,name,&shard);
    if(!t||!shard||t->n_dims!=2||!safetensors_dequant_supported(t->dtype)
       ||(long long)t->shape[0]!=w->out||(long long)t->shape[1]!=w->in
       ||t->n_elements!=(uint64_t)w->out*(uint64_t)w->in||w->in%64){
        fprintf(stderr,"FATAL: Qv14 original BF16 tensor unavailable/shape mismatch %s\n",name);exit(1);
    }
    size_t count=(size_t)w->out*(size_t)w->in;
    float *deq=malloc(sizeof(float)*count);
    size_t nb=target_n8?qng64_packed_bytes(w->out,w->in,8):count/2;
    uint8_t *packed=malloc(nb);
    float *scales=malloc(sizeof(float)*(size_t)w->out*(size_t)w->ng);
    if(!deq||!packed||!scales){fprintf(stderr,"FATAL: Qv14 memory allocation %s\n",name);exit(1);}
    safetensors_dequant_row(t->dtype,safetensors_tensor_data(shard,t),deq,t->n_elements);
    if(target_n8){
        if(qng64_quantize_f32(deq,w->out,w->in,8,packed,scales)!=0 ||
           beglin_qng64_n8_repack_inplace(packed,w->out,w->in)!=0){
            fprintf(stderr,"FATAL: Qv14 n8 native quant/repack %s\n",name);exit(1);
        }
    } else {
        gguf_quantize_q4g64_error_feedback(deq,w->out,w->in,packed,scales);
    }
    free(deq);
    free((void *)w->packed);free((void *)w->scales);
    w->packed=packed;w->scales=scales;w->kind=target_n8?K_QNG64:K_Q4G64;
    w->bits=target_n8?8:4;
    w->kai_rhs=NULL;w->kai_rhs_bytes=0;w->kai_lazy_failed=0;
}
static double qv14_clock_ms(void){
    struct timespec v;clock_gettime(CLOCK_MONOTONIC,&v);
    return (double)v.tv_sec*1000.0+(double)v.tv_nsec/1000000.0;
}
static unsigned char qv14_mask_source[N_LAYER_ROLES][64];
static unsigned char qv14_mask_random[N_LAYER_ROLES][64];
static int qv14_profile=-1;
static int qv14_switches_total=0,qv14_weight_repacked=0;
static double qv14_switch_ms_total=0;
static void qv14_prepare_profiles(void){
    const char *source=getenv("BEGLIN_QV14_MASK_SOURCE");
    const char *random=getenv("BEGLIN_QV14_MASK_RANDOM");
    const char *original=getenv("BEGLIN_Q_LAB_N8_ROLES_FILE");
    if(!source||!random||!original||strcmp(original,source) ||
       qv14_read_mask(source,qv14_mask_source)!=0 ||
       qv14_read_mask(random,qv14_mask_random)!=0){
        fprintf(stderr,"FATAL: Qv14 requires frozen source and random masks; original mixed model must start at source\n");exit(1);
    }
    qv14_check_state(qv14_mask_source);
    qv14_profile=0;
}
static void qv14_apply_profile(int target,int window,int prefix){
    if(target!=0&&target!=1){fprintf(stderr,"FATAL: Qv14 bad requested profile\n");exit(1);}
    const unsigned char (*desired)[64] = target==0 ? qv14_mask_source:qv14_mask_random;
    int changed=0;double start=qv14_clock_ms();
    if(target!=qv14_profile){
        /* Downgrade before upgrade so resident packed bytes cannot exceed
           the fixed current/budget by more than ONE transient tensor buffer. */
        for(int r=0;r<7;r++)for(int l=0;l<28;l++){
            int role=qv14_roles_id[r];
            if(!desired[role][l]){
                char name[96];snprintf(name,sizeof name,ROLE_PATTERN_HF[role],l);
                WT *w=wt(name);
                if(w->kind==K_QNG64&&w->bits==8){qv14_repack_one(role,l,0);changed++;}
            }
        }
        for(int r=0;r<7;r++)for(int l=0;l<28;l++){
            int role=qv14_roles_id[r];
            if(desired[role][l]){
                char name[96];snprintf(name,sizeof name,ROLE_PATTERN_HF[role],l);
                WT *w=wt(name);
                if(w->kind==K_Q4G64){qv14_repack_one(role,l,1);changed++;}
            }
        }
        qv14_profile=target;qv14_switches_total++;qv14_weight_repacked+=changed;
    }
    qv14_check_state(desired);
    double elapsed=qv14_clock_ms()-start;qv14_switch_ms_total+=elapsed;
    fprintf(stderr,"[QV14-DYNAMIC-SWITCH] window %d prefix %d profile %s tensors_repacked %d n8 28 q4 168 switch_ms %.5f\\n",
      window,prefix,target==0?"source_qv8":"random73",changed,elapsed);
}
'''
 marker="static int dense_qng64_override_n(const char *name) {"
 if src.count(marker)!=1:raise RuntimeError("source helper insertion point ambiguous")
 src=src.replace(marker,helper+"\n"+marker,1)
 old='} else if (!strcmp(mode,"masked_choice_qv5")) {'
 new='} else if (!strcmp(mode,"masked_choice_qv5") || !strcmp(mode,"qv14_dynamic_choice")) {\n        int qv14_active=!strcmp(mode,"qv14_dynamic_choice");\n        if(qv14_active)qv14_prepare_profiles();'
 if src.count(old)!=1:raise RuntimeError("original real scoring branch missing")
 src=src.replace(old,new,1)
 before='''            for(int p=0;p<n;p++)forward_token(ids[p],p,xtmp,0);
            final_logits(xtmp,logits);'''
 after='''            if(qv14_active)qv14_apply_profile(n>=85?1:0,wi,n);
            for(int p=0;p<n;p++)forward_token(ids[p],p,xtmp,0);
            final_logits(xtmp,logits);'''
 if src.count(before)!=1:raise RuntimeError("Qv5 original tokenizer/scoring insertion ambiguous "+str(src.count(before)))
 src=src.replace(before,after,1)
 summary_before='''        fprintf(stdout,"qv5_choice NLL %.17g items %d correct %d mean %.17g ppl %.17g\\n",'''
 summary_after='''        if(qv14_active)fprintf(stderr,"[QV14-DYNAMIC-TOTAL] switches %d tensors_repacked %d switch_ms %.5f\\n",
            qv14_switches_total,qv14_weight_repacked,qv14_switch_ms_total);
        fprintf(stdout,"qv5_choice NLL %.17g items %d correct %d mean %.17g ppl %.17g\\n",'''
 if src.count(summary_before)!=1:raise RuntimeError("Qv5 summary insertion ambiguous")
 src=src.replace(summary_before,summary_after,1)
 os.umask(0o077);BUILD.mkdir(parents=True,mode=0o700)
 cp=BUILD/"qv14_from_original_qv5_mixed_dynamic.c"
 put(cp,src.encode())
 obj=BUILD/"qv14_source.o";exe=BUILD/"beglin_qv14_inprocess_dynamic"
 cc=subprocess.run(["clang","-O3","-w","-I",str(E),"-c",str(cp),"-o",str(obj)],
     capture_output=True,text=True,timeout=180)
 put(BUILD/"clang_compile_log.txt",(cc.stdout+"\n"+cc.stderr).encode())
 if cc.returncode:raise RuntimeError("original 28-layer C clone compile failed "+cc.stderr[-2500:])
 parts=[E/"lab_local/qng64_n8_repacked_full_20261022"/x for x in
 ("qng64_original_aliased.o","n8_repacked_simd.o","n8_repacked_dispatch.o")]
 extra=sorted(x for x in (E/".build").glob("*.o") if x.name not in ("qwen_infer.o","qng64_dense.o"))
 if len(extra)!=20 or not all(p.is_file() for p in parts):raise RuntimeError("original CPU binary link closure changed")
 link=subprocess.run(["clang","-O3",str(obj)]+[str(x) for x in parts+extra]+
    ["-o",str(exe),"-framework","Accelerate","-lpthread","-lm"],capture_output=True,text=True,timeout=180)
 put(BUILD/"clang_link_log.txt",(link.stdout+"\n"+link.stderr).encode())
 if link.returncode:raise RuntimeError("original NEON+Q4 linker failed "+link.stderr[-2500:])
 exe.chmod(0o700)
 receipt={"schema":"BEGLIN-Qv14-original-Qwen28-C-inprocess-precision-repack-native-build/1",
 "status":"ISOLATED_NATIVE_CPU_Q4_N8_DYNAMIC_REAL_WT_REPACK_COMPILED_NOT_YET_EVALUATED",
 "original_Qv5_mixed_source_SHA256":ORIGSHA,
 "original_Qv5_mixed_binary_SHA256":ORIGEXESHA,
 "derived_C_SHA256":sha(cp),"derived_binary_SHA256":sha(exe),
 "compile_log_SHA256":sha(BUILD/"clang_compile_log.txt"),
 "link_log_SHA256":sha(BUILD/"clang_link_log.txt"),
 "unchanged_original_N8_NEON_and_Q4_kernel_objects":True,
 "new_command":"qv14_dynamic_choice",
 "precision_switch_mechanism":"live WT in-process, source BF16 dequant per changing tensor then Q4G64 or qNg64 n8 transcode; old packed buffer free, no dual whole-model retained",
 "source_only_dispatch":"prefix_len >=85 random73 else source_qv8",
 "real_switch_vs_static_numeric_accuracy_not_yet_verified":True,
 "gpu_not_supported":True,
 "no_main_merge_or_operator_release_change":True}
 put(BUILD/"BUILD_RECEIPT.json",(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode())
 print("QV14_NATIVE_DERIVED_C_BINARY_BUILT",json.dumps({"source_sha":receipt["derived_C_SHA256"],
   "exe_sha":receipt["derived_binary_SHA256"],"base_sha":ORIGSHA,
   "executable":str(exe),"compiled_using_original_objects":len(parts+extra)},sort_keys=True),flush=True)
if __name__=="__main__":main()
