#!/usr/bin/env python3
"""JE01 scratch-only original27 native Metal true 256-training FFN h2 + SwiGLU.

Extend the existing SHA-pinned true h2 trace (proven 16/656 original-gold
parity) to the immutable original 256 TRAIN PROMPTS only. Capture at the
last input prompt position for every top-6 routed selected eight Expert:
- 2048 actual post-attention normalized FFN h2 floats for gate/up;
- 1408 actual native original AF4 SwiGLU floats for down;
- original exact selected router probability, layer/expert/case/position.
Records are raw model activations, not source-weight or fake Fisher.
Do not touch training gold, same-source validation or OOD labels, weights,
optimizer, production C/C++ files, or serving model.
"""
import argparse,hashlib,importlib.util,json
from pathlib import Path
SRC=Path(__file__).resolve().parents[1]/"tools/je01_scratch_original27_real_h2_trace16.py"
SRC_GIT="39370fefa262a982fa5ac267f6adf6c01fd23ab1"
MAGIC="JE01_REAL_ORIGINAL_AF4_CAL256_ACTUAL_H2_SWIGLU_V1\n"
ADD_HOOK=r'''
            // JE01 original-only TRUE train-prompt FFN input activation,
            // captured after actual original AF4 gate/up SwiGLU evaluation.
            // The B1 unsorted branch is mandatory: A*TOPK=6 <sort_thr=64.
            if (g_je01_h2_file && g_je01_h2_pass==1 &&
                g_je01_h2_pos==g_je01_h2_last && A==1 && HIDDEN==2048 &&
                IM==1408 && TOPK==6 && NE==64 &&
                (l==12 || l==13 || l==14 || l==20 || l==21 || l==24)) {
                mx::array hv=mx::contiguous(h2);
                mx::array sw=mx::contiguous(swiglu_3d);
                mx::array ids=mx::contiguous(top_idx);
                mx::array pw=mx::contiguous(top_wgt);
                mx::eval(std::vector<mx::array>{hv,sw,ids,pw});
                const float *v=hv.data<float>();
                const float *down=sw.data<float>();
                const int32_t *idx=ids.data<int32_t>();
                const float *probs=pw.data<float>();
                for(int j=0;j<2048;j++)
                    if(!std::isfinite(v[j]))
                        throw std::runtime_error("JE01_CAL256_ORIGINAL_H2_NONFINITE");
                for(int k=0;k<TOPK;k++) {
                    int e=idx[k];
                    bool frozen=(l==12&&e==13)||
                        (l==13&&(e==22||e==28))||
                        (l==14&&e==63)||(l==20&&e==17)||
                        (l==21&&e==61)||(l==24&&(e==26||e==60));
                    if(!frozen)continue;
                    float prob=probs[k];
                    if(!std::isfinite(prob) || prob<=0 || prob>1)
                        throw std::runtime_error("JE01_CAL256_ORIGINAL_ROUTER_PROB_INVALID");
                    const float *z=down+(size_t)k*IM;
                    for(int j=0;j<IM;j++)
                        if(!std::isfinite(z[j]))
                            throw std::runtime_error("JE01_CAL256_ORIGINAL_SWIGLU_NONFINITE");
                    uint32_t tags[4]={(uint32_t)g_je01_h2_req,
                                     (uint32_t)l,(uint32_t)e,(uint32_t)g_je01_h2_pos};
                    if(std::fwrite(tags,sizeof(uint32_t),4,g_je01_h2_file)!=4 ||
                       std::fwrite(&prob,sizeof(float),1,g_je01_h2_file)!=1 ||
                       std::fwrite(v,sizeof(float),2048,g_je01_h2_file)!=2048 ||
                       std::fwrite(z,sizeof(float),1408,g_je01_h2_file)!=1408)
                        throw std::runtime_error("JE01_CAL256_TRUE_NATIVE_ACTIVATION_WRITE_FAILED");
                    g_je01_h2_count++;
                }
            }
'''
C_INIT=r'''
    // JE01 source256 last-prompt activation: requires NO gold/labels.
    const char *je01_h2_path=getenv("JE01_ROUTED_LAST_H2_OUT");
    const int je01_h2_on=je01_h2_path && *je01_h2_path;
    if(je01_h2_on) {
        if(B!=1 || R!=16 || pfB!=1 || sort_thr<=6 ||
           !mlx_gpu_je01_h2_open(je01_h2_path)) {
            fprintf(stderr,"FATAL: JE01_REAL_TRAIN256_ACTIVATION_B1_R16_LASTPROMPT_GUARD\n");
            exit(1);
        }
    }
'''
C_STEP=r'''
            // Actual original input source prompt length, NOT a gold label.
            if(je01_h2_on) {
                if(A!=1 || slot_arr[0]!=0 || mcb_req[0]<0 ||
                   mcb_req[0]>=16 || rq_plen[mcb_req[0]]<2 ||
                   rq_plen[mcb_req[0]]>=384 ||
                   !mlx_gpu_je01_h2_context(
                       mcb_req[0],spos_arr[0],rq_plen[mcb_req[0]]-1,pass)) {
                    fprintf(stderr,"FATAL: JE01_REAL_CAL256_PROMPT_H2_CONTEXT_INVALID\n");
                    exit(1);
                }
            }
'''
def gitblob(p):
    b=p.read_bytes()
    return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def unique(s,old,new,name):
    n=s.count(old)
    if n!=1:raise ValueError("JE01_CAL256_H2_SWIGLU_SOURCE_ANCHOR_"+name+"_COUNT_"+str(n))
    return s.replace(old,new,1)
def patch(c,cpp,h):
    if SRC.is_symlink() or not SRC.is_file() or gitblob(SRC)!=SRC_GIT:
        raise ValueError("JE01_PROVEN_ACTUAL_H2_SOURCE_SHA_DRIFT")
    spec=importlib.util.spec_from_file_location("je01_previous_original_h2",SRC)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    c,cpp,h=m.patch(c,cpp,h)
    c=unique(c,m.C_INIT,C_INIT,"GOLDLESS_FROZEN_ORIGINAL_PROMPTS")
    c=unique(c,m.C_STEP,C_STEP,"TRUE_ORIGINAL_PROMPT_LAST_POS")
    cpp=unique(cpp,m.CPP_HOOK,"","REMOVE_INCOMPLETE_GATEUP_ONLY_CAPTURE")
    cpp=unique(cpp,"JE01_ACTUAL_REAL_AF4_TRAIN16_ROUTED_H2_V1\\n",
               MAGIC.replace("\n","\\n"),"TRACE_SCHEMA_V1_TO_CAL256")
    a=cpp.index(m.CPP_START);b=cpp.index(m.CPP_END,a)
    target="mx::array swiglu_3d = mx::reshape(lazy_silu(gate_all) * up_all, {A * TOPK, 1, IM});"
    cb=unique(cpp[a:b],target,target+"\n"+ADD_HOOK,"TRUE_NATIVE_UNSORTED_SWIGLU")
    cpp=cpp[:a]+cb+cpp[b:]
    if cpp.count("JE01_CAL256_TRUE_NATIVE_ACTIVATION_WRITE_FAILED")!=1 or \
       cpp.count(MAGIC.replace("\n","\\n"))!=1 or \
       c.count("JE01_REAL_CAL256_PROMPT_H2_CONTEXT_INVALID")!=1:
        raise ValueError("JE01_TRUE_CAL256_H2_SWIGLU_STAGE_NOT_UNIQUE")
    return c,cpp,h
def run(root):
    if not root.is_dir() or root.is_symlink():raise ValueError("JE01_SCRATCH_ONLY_ROOT")
    files=[root/"qwen_infer.c",root/"mlx_moe.cpp",root/"mlx_moe.h"]
    if any(not p.is_file() or p.is_symlink() for p in files):
        raise ValueError("JE01_NATIVE_REAL_ORIGINAL27_SOURCE_MISSING")
    old=[p.read_text() for p in files]
    changed=patch(*old)
    for p,t in zip(files,changed):p.write_text(t)
    report={"schema":"beglin-je01-real-original27-256-calibration-h2-sw-nogold-instrument/1",
        "status":"JE01_ORIGINAL27_ACTUAL_CAL256_TRAIN_ONLY_H2_SWIGLU_SCRATCH_PATCHED",
        "source_previous_proven_original16_trace_blob":SRC_GIT,
        "source_sha256":{p.name:hashlib.sha256(t.encode()).hexdigest() for p,t in zip(files,old)},
        "instrumented_sha256":{p.name:hashlib.sha256(t.encode()).hexdigest() for p,t in zip(files,changed)},
        "input":"ORIGINAL_256_TRAIN_PROMPTS_LAST_TOKEN_ONLY",
        "observed_original_FFN_vectors":"H2_2048_AND_DOWN_SWIGLU_1408",
        "full_27layer_NLL":"NOT_MEASURED",
        "true_loss_Fisher_Hessian":"NOT_MEASURED",
        "heldout_OOD_labels_used":False,"production_modified":False}
    print(json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--scratch",type=Path,required=True)
    args=p.parse_args()
    try:run(args.scratch)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","type":type(e).__name__,
            "reason":str(e)[:390]},sort_keys=True),flush=True)
        raise SystemExit(2)
