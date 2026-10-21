/*
 * BEGLIN Q v2 isolated research-only real CPU asynchronous sibling projection
 * executor. NOT linked to main/production engine or immutable operator.
 *
 * Q/K/V forward projections read the same layer-normalized H, write independent
 * Q/K/V destination buffers; Gate/Up read same H after attention, write
 * independent buffers. Attention, O, SwiGLU, Down and residual remain strictly
 * ordered, with the stage join BEFORE any consumer touches the outputs.
 *
 * Workers persist for all 28 layers/tokens. Each role-worker owns its own
 * q4pool (1t or 6t), so NO concurrent access to global g_pool or a shared
 * native threadpool job; n8 uses independently tested repacked GEMV wrapper.
 *
 * Shared work source and mask copied from frozen source-only policy. This is
 * used ONLY by SHA-pinned research clone including this header after matvec_t.
 */
#ifndef BEGLIN_RESEARCH_QV2_NATIVE_ASYNC_STAGES_H
#define BEGLIN_RESEARCH_QV2_NATIVE_ASYNC_STAGES_H
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

typedef struct {
    const WT *W;
    const float *input;
    const float *bias;
    float *output;
    int mode;
} QV2AsyncProjection;

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv_work;
    pthread_cond_t cv_done;
    pthread_t workers[2];
    q4pool q4_single[3];
    q4pool q4_six[3];
    int worker_index[2];
    int generation;
    int pending;
    int finished;
    int stop;
    int started;
    QV2AsyncProjection task[2];
} QV2AsyncPool;
static QV2AsyncPool qv2_async_pool={0};
static atomic_ullong qv2_stage3_calls=0, qv2_stage2_calls=0;
static atomic_ullong qv2_serial_stages=0;

static int qv2_async_enabled(void) {
    const char *p=getenv("BEGLIN_QV2_NATIVE_ASYNC");
    return p && !strcmp(p,"1");
}
static void qv2_eval_proj(int slot,const QV2AsyncProjection *job) {
    const WT *W=job->W;
    if(!W || !job->input || !job->output || slot<0 || slot>2)abort();
    if(W->kind==K_Q4G64) {
        if(job->mode==1) {
            gemv_q4g64_mt(&qv2_async_pool.q4_single[slot],
                W->packed,W->scales,job->input,job->bias,
                job->output,W->out,W->in);
        } else if(job->mode==6) {
            gemv_q4g64_mt(&qv2_async_pool.q4_six[slot],
                W->packed,W->scales,job->input,job->bias,
                job->output,W->out,W->in);
        } else abort();
    } else if(W->kind==K_QNG64 && W->bits==8) {
        // Research link dispatch expects ORIGINAL BF16 -> native qNg64 n8
        // -> in-place packed-byte-to-signed-code conversion at load.
        qng64_matvec_f32(W->packed,W->scales,8,job->input,job->bias,
                        job->output,W->out,W->in);
    } else {
        fprintf(stderr,"FATAL Qv2 concurrent projection unsupported kind=%d bits=%d\n",
                        (int)W->kind,W->bits);
        abort();
    }
}
static void *qv2_async_worker(void *arg) {
    int index=*(int *)arg;
    int observed=0;
    QV2AsyncPool *p=&qv2_async_pool;
    for(;;) {
        pthread_mutex_lock(&p->mu);
        while(!p->stop && p->generation==observed)
            pthread_cond_wait(&p->cv_work,&p->mu);
        if(p->stop) {
            pthread_mutex_unlock(&p->mu);
            return NULL;
        }
        observed=p->generation;
        int work=(index<=p->pending);
        QV2AsyncProjection task={0};
        if(work)task=p->task[index-1];
        pthread_mutex_unlock(&p->mu);

        if(work)qv2_eval_proj(index,&task);

        if(work) {
            pthread_mutex_lock(&p->mu);
            p->finished++;
            if(p->finished==p->pending)
                pthread_cond_signal(&p->cv_done);
            pthread_mutex_unlock(&p->mu);
        }
    }
}
static void qv2_async_shutdown(void) {
    QV2AsyncPool *p=&qv2_async_pool;
    if(!p->started)return;
    pthread_mutex_lock(&p->mu);
    p->stop=1;
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mu);
    for(int i=0;i<2;i++)pthread_join(p->workers[i],NULL);
    for(int i=0;i<3;i++) {
        q4pool_destroy(&p->q4_six[i]);
        q4pool_destroy(&p->q4_single[i]);
    }
    pthread_cond_destroy(&p->cv_work);
    pthread_cond_destroy(&p->cv_done);
    pthread_mutex_destroy(&p->mu);
    p->started=0;
    fprintf(stderr,"[BEGLIN-QV2-NATIVE-ASYNC] qkv_joined=%llu gateup_joined=%llu serial_stage=%llu "
                   "pooled_q4_independent=1 production_changed=0\n",
        (unsigned long long)atomic_load(&qv2_stage3_calls),
        (unsigned long long)atomic_load(&qv2_stage2_calls),
        (unsigned long long)atomic_load(&qv2_serial_stages));
}
static void qv2_async_start(void) {
    QV2AsyncPool *p=&qv2_async_pool;
    if(p->started)return;
    pthread_mutex_init(&p->mu,NULL);
    pthread_cond_init(&p->cv_work,NULL);
    pthread_cond_init(&p->cv_done,NULL);
    for(int i=0;i<3;i++) {
        q4pool_init(&p->q4_single[i],1,g_cfg.im);
        q4pool_start(&p->q4_single[i]);
        q4pool_init(&p->q4_six[i],6,g_cfg.im);
        q4pool_start(&p->q4_six[i]);
    }
    p->stop=0;p->generation=0;p->pending=0;p->finished=0;
    for(int i=0;i<2;i++) {
        p->worker_index[i]=i+1;
        if(pthread_create(&p->workers[i],NULL,qv2_async_worker,
                          &p->worker_index[i])!=0)abort();
    }
    p->started=1;
    atexit(qv2_async_shutdown);
}
static void qv2_async_joined(int mode,QV2AsyncProjection *tasks,int num) {
    if(mode!=1 && mode!=6)abort();
    if(num!=2 && num!=3)abort();
    qv2_async_start();
    QV2AsyncPool *p=&qv2_async_pool;
    pthread_mutex_lock(&p->mu);
    p->pending=num-1;
    p->finished=0;
    for(int idx=1;idx<num;idx++) {
        p->task[idx-1]=tasks[idx];
        p->task[idx-1].mode=mode;
    }
    p->generation++;
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mu);

    tasks[0].mode=mode;
    qv2_eval_proj(0,&tasks[0]);

    pthread_mutex_lock(&p->mu);
    while(p->finished<p->pending)
        pthread_cond_wait(&p->cv_done,&p->mu);
    pthread_mutex_unlock(&p->mu);
    if(num==3)atomic_fetch_add(&qv2_stage3_calls,1);
    else atomic_fetch_add(&qv2_stage2_calls,1);
}
static void qv2_dispatch3(int mode,const WT *Wq,const WT *Wk,const WT *Wv,
                          const float *h,const float *bq,const float *bk,const float *bv,
                          float *q,float *k,float *v) {
    if(mode==0) {
        matvec_t(Wq,h,bq,q);matvec_t(Wk,h,bk,k);matvec_t(Wv,h,bv,v);
        atomic_fetch_add(&qv2_serial_stages,1);return;
    }
    QV2AsyncProjection jobs[3]={
        {Wq,h,bq,q,mode},{Wk,h,bk,k,mode},{Wv,h,bv,v,mode}
    };
    qv2_async_joined(mode,jobs,3);
}
static void qv2_dispatch2(int mode,const WT *Wg,const WT *Wu,
                          const float *h,float *gate,float *up) {
    if(mode==0) {
        matvec_t(Wg,h,NULL,gate);matvec_t(Wu,h,NULL,up);
        atomic_fetch_add(&qv2_serial_stages,1);return;
    }
    QV2AsyncProjection jobs[2]={
        {Wg,h,NULL,gate,mode},{Wu,h,NULL,up,mode}
    };
    qv2_async_joined(mode,jobs,2);
}
#endif
