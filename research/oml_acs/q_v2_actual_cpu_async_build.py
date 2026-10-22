#!/usr/bin/env python3
"""Construct a fail-closed research-only native Qwen QKV/FFN asynchronous CPU fork.

Original exact SHA-bound mixed-precision BEGLIN Qwen model C source is left
untouched. Three background pthread workers own INDEPENDENT Q4G64 pools
(K, V, up); main performs Q and gate on original BEGLIN global pool.
Each background reads identical pre-norm input and writes independent outputs.
After the join barrier, the original RoPE/attention/SwiGLU executes unchanged.

The same experimental C binary can run with BEGLIN_QV2_FULL_ASYNC=0 (serial)
or =1 (actual 3-worker async), under QWEN_FUSED_DISPATCH=0. No external
inputs, output sensitivity experiments, model edits, main branch or release.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2]
BASE=ROOT/"lab_local/q_kernel_aware_native_20261022/qwen_kernel_aware_copy.c"
BASE_SHA="d8e149092c73d858dfb7a066f697f2b95e0668ffa747b9d66ddc4ee3617f447a"
PREVIOUS=ROOT/"lab_local/qng64_n8_repacked_full_20261022"
EVAL=ROOT/"lab_local/q_v2_activation_async_native_eval_20261022"
OUT=ROOT/"lab_local/q_v2_actual_full_cpu_async_20261022"
EXPECTED_MASK_SHA="cb9268c9177d47705e548c2a90a6b22772569b3f2bb54dfef2767ca53ffa3e76"
EXPECTED_PUBLIC_RECEIPT_SHA="1dfb72b95b11262bbde56e2e47fa52965c96a4fe6c21f089b999ba1c816c9185"
PRELOCK=ROOT/"configs/research/BEGLIN_QV2_ACTUAL_FULL_CPU_ASYNC_PRELOCK_20261022.json"

ASYNC_NATIVE_CODE=r"""
/* BEGLIN Qv2 local research-only async Q/K/V and FFN gate/up FORWARD-TOKEN. 
 * Three persistent worker threads each own a separate Q4G64 pool. Only a
 * small isolated native CPU build includes this code. Never compiled live.
 */
#include <pthread.h>
#include <stdint.h>
typedef struct {
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t ready;
    pthread_cond_t finished_cv;
    q4pool pool6;
    uint64_t submitted, completed;
    int stop;
    const WT *weights;
    const float *input,*bias;
    float *output;
    uint64_t calls_q4, calls_n8;
} QV2AsyncWorker;
static QV2AsyncWorker g_qv2_workers[3];
static int g_qv2_inited=0;
static uint64_t g_qv2_qkv_tokens=0,g_qv2_ffn_tokens=0;
static void *qv2_worker_main(void *opaque) {
    QV2AsyncWorker *w=(QV2AsyncWorker *)opaque;
    pthread_mutex_lock(&w->mu);
    for (;;) {
        while (!w->stop && w->submitted==w->completed)
            pthread_cond_wait(&w->ready,&w->mu);
        if (w->stop)break;
        const WT *t=w->weights;
        const float *input=w->input,*bias=w->bias;
        float *output=w->output;
        uint64_t seq=w->submitted;
        pthread_mutex_unlock(&w->mu);
        if (t->kind==K_Q4G64) {
            // SAME Q4G64 kernel as original matvec_t; different q4pool to
            // prevent shared job/cv mutations between active projections.
            gemv_q4g64_mt(&w->pool6,t->packed,t->scales,input,bias,output,t->out,t->in);
            w->calls_q4++;
        } else if (t->kind==K_QNG64 && t->bits==8) {
            // Original full-model loader already repacked this n8 tensor
            // IN PLACE; same source-bound native wrapper as serial baseline.
            qng64_matvec_f32(t->packed,t->scales,8,input,bias,output,t->out,t->in);
            w->calls_n8++;
        } else {
            fprintf(stderr,"FATAL: async research unrecognized precision kind %d\\n",t->kind);
            abort();
        }
        pthread_mutex_lock(&w->mu);
        w->completed=seq;
        pthread_cond_signal(&w->finished_cv);
    }
    pthread_mutex_unlock(&w->mu);
    return NULL;
}
static void qv2_async_shutdown(void) {
    if (!g_qv2_inited)return;
    for (int i=0;i<3;i++) {
        QV2AsyncWorker *w=&g_qv2_workers[i];
        pthread_mutex_lock(&w->mu);
        if(w->submitted!=w->completed) {
            fprintf(stderr,"FATAL: research async shutdown with unfinished work\\n");
            abort();
        }
        w->stop=1;
        pthread_cond_signal(&w->ready);
        pthread_mutex_unlock(&w->mu);
    }
    for(int i=0;i<3;i++) {
        QV2AsyncWorker *w=&g_qv2_workers[i];
        pthread_join(w->thread,NULL);
        q4pool_destroy(&w->pool6);
        pthread_cond_destroy(&w->ready);
        pthread_cond_destroy(&w->finished_cv);
        pthread_mutex_destroy(&w->mu);
    }
    fprintf(stderr,"[BEGLIN-QV2-FULL-CPU-ASYNC] qkv_tokens=%llu ffn_tokens=%llu K_q4=%llu K_n8=%llu V_q4=%llu V_n8=%llu U_q4=%llu U_n8=%llu threads=3 per_worker_Q4pool=6 done=1\\n",
        (unsigned long long)g_qv2_qkv_tokens,
        (unsigned long long)g_qv2_ffn_tokens,
        (unsigned long long)g_qv2_workers[0].calls_q4,
        (unsigned long long)g_qv2_workers[0].calls_n8,
        (unsigned long long)g_qv2_workers[1].calls_q4,
        (unsigned long long)g_qv2_workers[1].calls_n8,
        (unsigned long long)g_qv2_workers[2].calls_q4,
        (unsigned long long)g_qv2_workers[2].calls_n8);
    g_qv2_inited=0;
}
static int qv2_async_on(void) {
    static int mode=-1;
    if (mode<0) {
        const char *e=getenv("BEGLIN_QV2_FULL_ASYNC");
        if (!e || !strcmp(e,"0"))mode=0;
        else if (!strcmp(e,"1"))mode=1;
        else {
            fprintf(stderr,"FATAL: BEGLIN_QV2_FULL_ASYNC must be literal 0/1\\n");abort();
        }
        if (mode) {
            const char *f=getenv("QWEN_FUSED_DISPATCH");
            if (!f || strcmp(f,"0")) {
                fprintf(stderr,"FATAL: research async requires explicit QWEN_FUSED_DISPATCH=0\\n");abort();
            }
            const char *p=getenv("QWEN_PROF");
            if(p && strcmp(p,"0")) {
                fprintf(stderr,"FATAL: QWEN_PROF unsupported under async research instrumentation\\n");abort();
            }
            if(g_cfg.im<=0 || g_cfg.im>32768) {
                fprintf(stderr,"FATAL: unexpected hidden native model IM\\n");abort();
            }
            for(int i=0;i<3;i++) {
                QV2AsyncWorker *w=&g_qv2_workers[i];
                pthread_mutex_init(&w->mu,NULL);
                pthread_cond_init(&w->ready,NULL);
                pthread_cond_init(&w->finished_cv,NULL);
                q4pool_init(&w->pool6,6,g_cfg.im);
                q4pool_start(&w->pool6);
                if(pthread_create(&w->thread,NULL,qv2_worker_main,w)) {
                    fprintf(stderr,"FATAL: research background pthread launch failed\\n");abort();
                }
            }
            g_qv2_inited=1;
            if(atexit(qv2_async_shutdown))abort();
        }
    }
    return mode;
}
static void qv2_async_submit(int lane,const WT *weight,const float *x,
                              const float *bias,float *y) {
    if(lane<0||lane>=3||!weight||!x||!y)abort();
    QV2AsyncWorker *w=&g_qv2_workers[lane];
    pthread_mutex_lock(&w->mu);
    if(w->submitted!=w->completed) {
        fprintf(stderr,"FATAL: overlapping async role submission\\n");abort();
    }
    w->weights=weight;w->input=x;w->bias=bias;w->output=y;
    w->submitted++;
    pthread_cond_signal(&w->ready);
    pthread_mutex_unlock(&w->mu);
}
static void qv2_async_wait(int lane) {
    QV2AsyncWorker *w=&g_qv2_workers[lane];
    pthread_mutex_lock(&w->mu);
    while(w->completed!=w->submitted)
        pthread_cond_wait(&w->finished_cv,&w->mu);
    pthread_mutex_unlock(&w->mu);
}
"""
# Raw string above uses double-backslash for C \n; convert back to one C escape.
ASYNC_NATIVE_CODE=ASYNC_NATIVE_CODE.replace(chr(92)+chr(92)+"n",chr(92)+"n")

ORIGINAL_QKV="""        if (prof_on()) tp0 = nowt();
        if (fused_dispatch() & 1) matvec_t(&g_qkv_fused[l], hbuf, qkv_fused_bias_row(l), q);
        else matvec_t(g_role_wt[ROLE_ATTN_Q][l], hbuf, g_qbias_l[l], q);
        if (prof_on()) g_t_q += nowt()-tp0;

        if (prof_on()) tp0 = nowt();
        if (!(fused_dispatch() & 1)) matvec_t(g_role_wt[ROLE_ATTN_K][l], hbuf, g_kbias_l[l], kv_k);
        if (prof_on()) g_t_k += nowt()-tp0;

        if (prof_on()) tp0 = nowt();
        if (!(fused_dispatch() & 1)) matvec_t(g_role_wt[ROLE_ATTN_V][l], hbuf, g_vbias_l[l], kv_v);
        if (prof_on()) g_t_v += nowt()-tp0;

"""
QKV_ASYNC="""        if (qv2_async_on()) {
            // Same pre-norm input, disjoint Q/K/V outputs, explicit barrier
            // before RoPE/KV/attention; no next-layer cross-dependency abuse.
            qv2_async_submit(0,g_role_wt[ROLE_ATTN_K][l],hbuf,g_kbias_l[l],kv_k);
            qv2_async_submit(1,g_role_wt[ROLE_ATTN_V][l],hbuf,g_vbias_l[l],kv_v);
            matvec_t(g_role_wt[ROLE_ATTN_Q][l],hbuf,g_qbias_l[l],q);
            qv2_async_wait(0);
            qv2_async_wait(1);
            g_qv2_qkv_tokens++;
        } else {
"""+ORIGINAL_QKV+"""        }

"""
ORIGINAL_GU="""        if (prof_on()) tp0 = nowt();
        if (fused_dispatch() & 2) matvec_t(&g_gu_fused[l], hbuf, NULL, gate);
        else matvec_t(g_role_wt[ROLE_MLP_GATE][l], hbuf, NULL, gate);
        if (prof_on()) g_t_gate += nowt()-tp0;

        if (prof_on()) tp0 = nowt();
        if (!(fused_dispatch() & 2)) matvec_t(g_role_wt[ROLE_MLP_UP][l], hbuf, NULL, up);
        if (prof_on()) g_t_up += nowt()-tp0;

"""
GU_ASYNC="""        if (qv2_async_on()) {
            // Up and Gate read the same post-attn normalized input; join
            // before SwiGLU and MLP down.
            qv2_async_submit(2,g_role_wt[ROLE_MLP_UP][l],hbuf,NULL,up);
            matvec_t(g_role_wt[ROLE_MLP_GATE][l],hbuf,NULL,gate);
            qv2_async_wait(2);
            g_qv2_ffn_tokens++;
        } else {
"""+ORIGINAL_GU+"""        }

"""
def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()
def put(path:pathlib.Path,data:bytes):
    if path.exists():raise FileExistsError("refuse to overwrite native async model")
    with path.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def build():
    if OUT.exists():raise FileExistsError("async native Qwen lab exists")
    if sha(BASE.read_bytes())!=BASE_SHA:raise RuntimeError("source baseline changed")
    if not PRELOCK.exists():raise RuntimeError("safety/precision prelock missing")
    prereg=json.loads(PRELOCK.read_text())
    if prereg["original_source_lab_C_sha256"]!=BASE_SHA or prereg["policy_sha256"]!=EXPECTED_MASK_SHA:
        raise RuntimeError("protocol source or original fixed mask mismatch")
    old=EVAL/"v2_mask.txt"
    if not old.is_file() or (old.stat().st_mode&0o777)!=0o600 or sha(old.read_bytes())!=EXPECTED_MASK_SHA:
        raise RuntimeError("original source-selected 28 tensor mask drift")
    # Original Qwen C model and the previously verified native n8 lab are unchanged.
    original=BASE.read_text()
    for target in ("static void rmsnorm(",ORIGINAL_QKV,ORIGINAL_GU):
        if original.count(target)!=1:
            raise RuntimeError("original native forward cutpoint not unique")
    copy=original.replace("static void rmsnorm(",ASYNC_NATIVE_CODE+"\nstatic void rmsnorm(",1)
    copy=copy.replace(ORIGINAL_QKV,QKV_ASYNC,1).replace(ORIGINAL_GU,GU_ASYNC,1)
    if copy.count("qv2_async_submit(0")!=1 or copy.count("qv2_async_submit(2")!=1:
        raise RuntimeError("safety patch did not match source")
    base_objs=sorted((ROOT/".build").glob("*.o"))
    regular=[p for p in base_objs if p.name not in ("qwen_infer.o","qng64_dense.o")]
    linkobjects=[
        PREVIOUS/"qng64_original_aliased.o",
        PREVIOUS/"n8_repacked_simd.o",
        PREVIOUS/"n8_repacked_dispatch.o"]
    if len(regular)!=20 or any(not p.is_file() for p in linkobjects):
        raise RuntimeError("original source link objects missing")
    os.umask(0o077);OUT.mkdir(mode=0o700,parents=True)
    src=OUT/"qwen_research_actual_async_copy.c"
    put(src,copy.encode("utf-8"))
    obj=OUT/"qwen_research_actual_async_copy.o"
    subprocess.run(["clang","-O3","-w","-I",str(ROOT),"-c",str(src),"-o",str(obj)],check=True)
    binary=OUT/"beglin_qwen_actual_async"
    subprocess.run(["clang","-O3",str(obj)]+[str(p) for p in linkobjects+regular]+[
        "-o",str(binary),"-framework","Accelerate","-lpthread","-lm"],check=True)
    binary.chmod(0o700)
    for mode in ("serial","async"):
        base=OUT/mode
        base.mkdir(mode=0o700)
        for sub in ("results","eval","eval/ppl_work"):
            (base/sub).mkdir(exist_ok=True,mode=0o700)
        prior=EVAL/"v2_async_activation"
        paths=["prompt_ids.i32"]+[f"eval/ppl_work/w{i}.i32" for i in range(8)]
        for rel in paths:
            f=prior/rel
            if not f.is_file() or (f.stat().st_mode&0o777)!=0o600:
                raise RuntimeError("original 131 public synthetic eval positions missing")
            put(base/rel,f.read_bytes())
    put(OUT/"policy.txt",old.read_bytes())
    meta=dict(schema="BEGLIN-Qv2-real-full-CPU-async-build/1",
        status="EXPERIMENTAL_ASYNC_QWEN_NATIVE_BI_MODE_BINARY_BUILT",
        model_original_BF16_rev="4ae77cb209f06199b8df1c94e21ff341332a3a89",
        prior_native_source_C_sha256=BASE_SHA,
        actual_async_source_copy_sha256=sha(src.read_bytes()),
        native_binary_sha256=sha(binary.read_bytes()),
        selected_role_mask_SHA256=EXPECTED_MASK_SHA,
        source_protocol_SHA256=sha(PRELOCK.read_bytes()),
        concurrency_worker_count=3,
        native_q4pool_per_background_role_separate=True,
        same_binary_serial_or_async_env_toggle=True,
        same_131_evaluation_positions=True,
        output_original_main_unchanged=True,production_unchanged=True,
        full_forward_tokens_batch_async=False,
        model_weight_files_read_only=True,
        blocked_full_logit_intervention_not_retried=True)
    put(OUT/"build_receipt.json",(json.dumps(meta,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_QV2_ACTUAL_ASYNC_NATIVE_BUILD",json.dumps(meta,sort_keys=True),flush=True)
if __name__=="__main__":build()
