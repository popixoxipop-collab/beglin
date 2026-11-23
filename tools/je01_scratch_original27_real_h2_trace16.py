#!/usr/bin/env python3
"""JE01 scratch-only actual original DeepSeek B1 train16 last-prompt hidden trace.

Records TRUE 2048-dim normalized FFN input h2 and original Router TOP6
probabilities for 8 frozen (layer,Expert) slots, on *last prompt position only*
of 16 training inputs. No labels, Fisher, gradient or Hessian. Source only.
Requires existing isolated original384 patch, fullgold patch, and rejects any
input source drift; output file created exclusively in runner-controlled D50.
Crucial: verify scored native gold NLL output SHA against original uninstrumented
before calling captured h2 valid for unperturbed path.
"""
import argparse,hashlib,json
from pathlib import Path
C_START="static int run_moe_gpu_cbatch_online_gate(int argc, char **argv) {"
C_END="// V5i Phase B:"
CPP_START="int mlx_gpu_cbatch_layer_step_lazy("
CPP_END="int mlx_gpu_cbatch_forward_finalize("
HEADER_ANCHOR="#ifdef __cplusplus\n}\n#endif"
START_ANCHOR="    mlx_gpu_set_sort_threshold(sort_thr);\n"
LOOP_ANCHOR="            for (int l = 0; l < MOE_NL; l++) {\n                MoeLayerTensors *t = &g_moe_lt[l];"
END_ANCHOR='    fprintf(stderr, "RESULT: MoE GPU V5h online cbatch gate complete, B=%d R=%d\\n", B, R);'
ACT_AFTER="            top_wgt = moe_topk_renorm_rows(top_wgt);"
CPP_FUNCTIONS=r'''
// JE01 actual original pretrained h2 pilot capture; confined to scratch.
static FILE *g_je01_h2_file=nullptr;
static int g_je01_h2_req=-1,g_je01_h2_pos=-1,g_je01_h2_last=-1,g_je01_h2_pass=0;
static int g_je01_h2_count=0;
int mlx_gpu_je01_h2_open(const char *p) {
    if(!p || !*p || g_je01_h2_file || g_je01_h2_count) return 0;
    g_je01_h2_file=std::fopen(p,"wbx");  // no overwrite
    if(!g_je01_h2_file)return 0;
    const char hdr[]="JE01_ACTUAL_REAL_AF4_TRAIN16_ROUTED_H2_V1\n";
    if(std::fwrite(hdr,1,sizeof(hdr)-1,g_je01_h2_file)!=sizeof(hdr)-1)return 0;
    return 1;
}
int mlx_gpu_je01_h2_context(int req,int pos,int last,int pass) {
    if(!g_je01_h2_file || req<0 || req>=16 || last<0 || last>=384 ||
       pos<0 || pos>=384 || (pass!=0 && pass!=1))return 0;
    g_je01_h2_req=req;g_je01_h2_pos=pos;g_je01_h2_last=last;g_je01_h2_pass=pass;
    return 1;
}
int mlx_gpu_je01_h2_finish(void) {
    if(!g_je01_h2_file)return 0;
    int rc=std::fflush(g_je01_h2_file)==0 && std::fclose(g_je01_h2_file)==0;
    g_je01_h2_file=nullptr;
    std::fprintf(stderr,"JE01_ORIGINAL_ACTUAL_ROUTED_H2_TRAIN16_RECORDS=%d\n",g_je01_h2_count);
    return rc && g_je01_h2_count>0;
}
'''
CPP_HOOK=r'''
            // Actual observed h2 only at last ORIGINAL TRAIN prompt token.
            // The observed Router TOP6 and data<float>() are *model outputs*.
            if(g_je01_h2_file && g_je01_h2_pass==1 &&
               g_je01_h2_pos==g_je01_h2_last && A==1 && HIDDEN==2048 &&
               (l==12 || l==13 || l==14 || l==20 || l==21 || l==24)) {
                mx::array hv=mx::contiguous(h2);
                mx::array ii=mx::contiguous(top_idx);
                mx::array ww=mx::contiguous(top_wgt);
                mx::eval(std::vector<mx::array>{hv,ii,ww});
                const float *v=hv.data<float>();
                const int32_t *ids=ii.data<int32_t>();
                const float *weights=ww.data<float>();
                for(int k=0;k<6;k++) {
                    const int e=ids[k];
                    const bool chosen=(l==12&&e==13)||(l==13&&(e==22||e==28))||
                        (l==14&&e==63)||(l==20&&e==17)||(l==21&&e==61)||
                        (l==24&&(e==26||e==60));
                    if(!chosen)continue;
                    if(!std::isfinite(weights[k]) || weights[k]<0 || weights[k]>1)
                        throw std::runtime_error("JE01_H2_REAL_ROUTER_WEIGHT_INVALID");
                    for(int j=0;j<2048;j++)if(!std::isfinite(v[j]))
                        throw std::runtime_error("JE01_H2_REAL_HIDDEN_NONFINITE");
                    uint32_t tags[4]={(uint32_t)g_je01_h2_req,
                      (uint32_t)l,(uint32_t)e,(uint32_t)g_je01_h2_pos};
                    if(std::fwrite(tags,sizeof(uint32_t),4,g_je01_h2_file)!=4 ||
                       std::fwrite(&weights[k],sizeof(float),1,g_je01_h2_file)!=1 ||
                       std::fwrite(v,sizeof(float),2048,g_je01_h2_file)!=2048)
                        throw std::runtime_error("JE01_H2_ACTUAL_RECORD_WRITE_FAILED");
                    g_je01_h2_count++;
                }
            }
'''
C_INIT=r'''
    const char *je01_h2_path=getenv("JE01_ROUTED_LAST_H2_OUT");
    const int je01_h2_on=je01_h2_path && *je01_h2_path;
    int je01_h2_prompt_last[16]={0};
    if(je01_h2_on) {
        if(B!=1 || R!=16 || pfB!=1) {
            fprintf(stderr,"FATAL: JE01_H2_REAL_B1_R16_PFB1_REQUIRED\n");exit(1);
        }
        // Real original input includes the full teacher-forced target text.
        // True prompt LAST pos = first_gold_start-1, never rq_plen-1.
        const char *gm=getenv("JE01_ORIGINAL_GOLD_META");
        FILE *goldmeta=gm && *gm ? fopen(gm,"rb") : NULL;
        if(!goldmeta) {fprintf(stderr,"FATAL: JE01_H2_ORIGINAL_GOLDMETA_REQUIRED\n");exit(1);}
        static const char magic[]="JE01_ORIGINAL_FULL_GOLD_TARGETS_META_V1\n";
        char marker[sizeof(magic)-1];
        if(fread(marker,1,sizeof(marker),goldmeta)!=sizeof(marker) ||
           memcmp(marker,magic,sizeof(marker))!=0)
            {fprintf(stderr,"FATAL: JE01_H2_ORIGINAL_GOLDMETA_MAGIC_DRIFT\n");exit(1);}
        for(int rr=0;rr<16;rr++) {
            uint32_t hdr[2];int32_t ids[384];
            if(fread(hdr,sizeof(uint32_t),2,goldmeta)!=2 ||
               fread(ids,sizeof(int32_t),384,goldmeta)!=384 ||
               hdr[0]<2 || hdr[0]>=384 || hdr[1]<1 || hdr[0]+hdr[1]>384)
                {fprintf(stderr,"FATAL: JE01_H2_ORIGINAL_GOLDMETA_ROW_DRIFT\n");exit(1);}
            je01_h2_prompt_last[rr]=(int)hdr[0]-1;
        }
        if(fgetc(goldmeta)!=EOF || fclose(goldmeta)!=0 ||
           !mlx_gpu_je01_h2_open(je01_h2_path)) {
            fprintf(stderr,"FATAL: JE01_H2_REAL_GOLDMETA_OR_OUTPUT_OPEN_FAILED\n");exit(1);
        }
    }
'''
C_STEP=r'''
            if(je01_h2_on) {
                if(A!=1 || slot_arr[0]!=0 || mcb_req[0]<0 || mcb_req[0]>=16 ||
                   !mlx_gpu_je01_h2_context(mcb_req[0],spos_arr[0],
                                              je01_h2_prompt_last[mcb_req[0]],pass)) {
                    fprintf(stderr,"FATAL: JE01_ACTUAL_LAST_H2_CONTEXT_WRONG\n");exit(1);
                }
            }
'''
C_ENDHOOK=r'''
    if(je01_h2_on && !mlx_gpu_je01_h2_finish()) {
        fprintf(stderr,"FATAL: JE01_ACTUAL_LAST_H2_CLOSE_OR_EMPTY\n");exit(1);
    }
'''
HEADER=r'''
// JE01 trace is only in disposable original384 research build.
int mlx_gpu_je01_h2_open(const char *path);
int mlx_gpu_je01_h2_context(int req,int pos,int last,int pass);
int mlx_gpu_je01_h2_finish(void);
'''
def once(s,a,b,name):
    if s.count(a)!=1:raise ValueError("JE01_H2_SOURCE_EXACT_ANCHOR_"+name+"_COUNT_"+str(s.count(a)))
    return s.replace(a,b,1)
def patch(c,cpp,header):
    start=c.index(C_START);end=c.index(C_END,start)
    cmain=c[start:end]
    cmain=once(cmain,START_ANCHOR,START_ANCHOR+C_INIT,"C_SOURCE_INIT")
    cmain=once(cmain,LOOP_ANCHOR,C_STEP+LOOP_ANCHOR,"C_NATIVE_CBAT_CONTEXT")
    cmain=once(cmain,END_ANCHOR,C_ENDHOOK+END_ANCHOR,"C_NATIVE_CLOSE")
    c=c[:start]+cmain+c[end:]
    if cpp.count(CPP_START)!=1 or cpp.count(CPP_END)!=1:
        raise ValueError("JE01_NATIVE_REAL_CBAT_GPU_SOURCE_BOUNDARIES_CHANGED")
    lo=cpp.index(CPP_START);hi=cpp.index(CPP_END,lo)
    scope=cpp[lo:hi]
    scope=once(scope,ACT_AFTER,ACT_AFTER+CPP_HOOK,"CXX_ACTUAL_H2_AFTER_ROUTER")
    cpp=cpp[:lo]+scope+cpp[hi:]
    cpp=once(cpp,CPP_START,CPP_FUNCTIONS+"\n"+CPP_START,"CXX_ACTUAL_TRACE_FUNCTIONS")
    header=once(header,HEADER_ANCHOR,HEADER+"\n"+HEADER_ANCHOR,"HEADER_EXCLUSIVE_REAL_TRACE")
    if c.count("JE01_ACTUAL_LAST_H2_CLOSE_OR_EMPTY")!=1 or \
       cpp.count("JE01_ACTUAL_REAL_AF4_TRAIN16_ROUTED_H2_V1")!=1:
        raise ValueError("JE01_NATIVE_H2_TRACE_NOT_SINGLETON")
    return c,cpp,header
def main(root):
    if not root.is_dir() or root.is_symlink():raise ValueError("JE01_RESEARCH_ONLY_SCRATCH_DIR")
    names=["qwen_infer.c","mlx_moe.cpp","mlx_moe.h"]
    paths=[root/name for name in names]
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError("JE01_REAL_ORIGINAL_SOURCE_MISSING")
    originals=[p.read_text() for p in paths]
    modified=patch(*originals)
    for p,s in zip(paths,modified):p.write_text(s)
    report={"schema":"beglin-je01-actual-original27-16-train-lasttoken-routed-hidden2048-instrument/1",
       "status":"JE01_ISOLATED_ORIGINAL27_LASTPROMPT_ROUTED_H2_TRACE_SOURCE_STAGED",
       "source_sha256":{p.name:hashlib.sha256(s.encode()).hexdigest() for p,s in zip(paths,originals)},
       "instrumented_sha256":{p.name:hashlib.sha256(s.encode()).hexdigest() for p,s in zip(paths,modified)},
       "trained_real_forward":"NOT_RUN","original_gold_NLL_parity":"NOT_TESTED",
       "heldout_OOD_labels_accessed":False,"true_Fisher_or_Hessian":"NOT_MEASURED",
       "root_production_modified":False}
    print(json.dumps(report,sort_keys=True),flush=True)
    return report
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--scratch",type=Path,required=True)
    try:main(ap.parse_args().scratch)
    except Exception as exc:
        print(json.dumps({"status":"BLOCKED","error":str(exc)[:340]},sort_keys=True),flush=True)
        raise SystemExit(2)
