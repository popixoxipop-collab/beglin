/* BEGLIN Qv19 research ONLY. Injected immediately before Qv17's main.
 * ALL mutable original model state (live Q4/n8 WT, attention KV, activation
 * scratch and logits) is touched only by qv19_run_request callbacks.
 * Reader requests are SERIALIZED: not true parallel native forward.
 */
#include "qv19_request_lease.h"

typedef struct qv19_native_server qv19_native_server;
typedef struct {
    qv19_native_server *server;
    int request_index, prefix_len, target;
    int *tokens;
    int choices[5];
    int best, answer;
    uint64_t epoch_observed_before, epoch_observed_after;
} qv19_native_call;

struct qv19_native_server {
    qv19_model_lease lease;
    pthread_mutex_t barrier_mutex;
    pthread_cond_t barrier_cond;
    int ready, request_count, done_count[2], failed;
    const char *input_root;
    float *shared_xtmp, *shared_logits;
    qv19_request_receipt receipts[128];
    double losses[128];
    int gold[128], predicted[128], prefixes[128], workers[128];
};

typedef struct {
    qv19_native_server *server;
    int worker_index;
} qv19_native_worker;

static void qv19_native_fatal(const char *reason) {
    fprintf(stderr,"FATAL: Qv19 guarded original C %s\n",reason);
    exit(2);
}
static void qv19_native_need(int condition,const char *reason) {
    if (!condition) qv19_native_fatal(reason);
}
static int qv19_native_apply(void *opaque,int target) {
    qv19_native_call *c=(qv19_native_call*)opaque;
    if (target!=0 && target!=1) return -1;
    qv14_apply_profile(target,c->request_index,c->prefix_len);
    return qv14_profile==target?0:-1;
}
static int qv19_native_forward_score(void *opaque,double *gold_nll) {
    qv19_native_call *c=(qv19_native_call*)opaque;
    qv19_native_server *s=c->server;
    c->epoch_observed_before=s->lease.epoch;
    if (qv14_profile!=c->target || c->prefix_len<2 || c->prefix_len>208)
        return -1;
    for(int p=0;p<c->prefix_len;p++)
        forward_token(c->tokens[p],p,s->shared_xtmp,0);
    final_logits(s->shared_xtmp,s->shared_logits);
    float mx;
    vDSP_maxv(s->shared_logits,1,&mx,g_cfg.vocab);
    double sum=0;
    for(int j=0;j<g_cfg.vocab;j++)
        sum+=exp((double)(s->shared_logits[j]-mx));
    int answer=c->choices[4],best=0;
    for(int j=1;j<4;j++)
        if(s->shared_logits[c->choices[j]]>s->shared_logits[c->choices[best]])
            best=j;
    double nll=-(double)s->shared_logits[c->choices[answer]]+mx+log(sum);
    if(!isfinite(nll)||nll<0||qv14_profile!=c->target||
       s->lease.epoch!=c->epoch_observed_before)
        return -1;
    *gold_nll=nll;
    c->best=best;c->answer=answer;
    c->epoch_observed_after=s->lease.epoch;
    return 0;
}
static void *qv19_native_worker_run(void *opaque) {
    qv19_native_worker *w=(qv19_native_worker*)opaque;
    qv19_native_server *s=w->server;
    qv19_native_need(pthread_mutex_lock(&s->barrier_mutex)==0,"barrier lock");
    s->ready++;
    pthread_cond_broadcast(&s->barrier_cond);
    while(s->ready<2)
        qv19_native_need(pthread_cond_wait(&s->barrier_cond,&s->barrier_mutex)==0,
                         "barrier wait");
    qv19_native_need(pthread_mutex_unlock(&s->barrier_mutex)==0,"barrier unlock");

    for(int wi=w->worker_index;wi<s->request_count;wi+=2) {
        /* Parsing is independent and thread-local; no global model state. */
        int ids[g_cfg.maxseq];
        char filename[4096],junk=0;
        int choice[5]={-1,-1,-1,-1,-1};
        snprintf(filename,sizeof filename,"%s/eval/ppl_work/w%d.i32",s->input_root,wi);
        int n=load_ids(filename,ids,g_cfg.maxseq);
        qv19_native_need(n>=2&&n<=208,"prefix input invalid");
        snprintf(filename,sizeof filename,"%s/eval/ppl_work/w%d.choices",s->input_root,wi);
        FILE *fp=fopen(filename,"rb");
        qv19_native_need(fp!=NULL,"original choice input absent");
        int scanned=fscanf(fp,"%d %d %d %d %d %c",&choice[0],&choice[1],
                    &choice[2],&choice[3],&choice[4],&junk);
        fclose(fp);
        qv19_native_need(scanned==5&&choice[4]>=0&&choice[4]<=3,
                         "source gold option invalid");
        for(int i=0;i<4;i++) {
            qv19_native_need(choice[i]>=0&&choice[i]<g_cfg.vocab,
                             "source choice token invalid");
            for(int j=i+1;j<4;j++)
                qv19_native_need(choice[i]!=choice[j],"duplicate choice token");
        }
        qv19_native_call call={0};
        call.server=s;call.request_index=wi;call.prefix_len=n;
        call.target=(n>=85?1:0);call.tokens=ids;
        for(int i=0;i<5;i++)call.choices[i]=choice[i];
        qv19_request_receipt receipt;
        qv19_result status=qv19_run_request(&s->lease,call.target,
            qv19_native_apply,qv19_native_forward_score,&call,0,&receipt);
        qv19_native_need(status==QV19_OK,"lease failed or poisoned");
        qv19_native_need(receipt.epoch_enter==receipt.epoch_exit &&
            receipt.epoch_enter==call.epoch_observed_before &&
            receipt.epoch_exit==call.epoch_observed_after,
            "in-flight precision snapshot changed");
        s->receipts[wi]=receipt;
        s->losses[wi]=receipt.gold_nll;
        s->gold[wi]=call.answer;s->predicted[wi]=call.best;
        s->prefixes[wi]=n;s->workers[wi]=w->worker_index;
        s->done_count[w->worker_index]++;
        fprintf(stderr,"[qv5-choice] window %d NLL %.17g expected %d predicted %d prefix %d\n",
            wi,receipt.gold_nll,call.answer,call.best,n);
        fprintf(stderr,"[QV19-LEASE] window %d worker %d epoch_enter %llu "
            "epoch_exit %llu profile %s changed %d wait_ns %llu guarded_ns %llu "
            "snapshot_stable 1\n",
            wi,w->worker_index,(unsigned long long)receipt.epoch_enter,
            (unsigned long long)receipt.epoch_exit,call.target?"random73":"source_qv8",
            receipt.profile_changed,(unsigned long long)receipt.lock_wait_ns,
            (unsigned long long)receipt.guarded_duration_ns);
    }
    return NULL;
}

static void qv19_native_guarded_real_C(const char *base,
                                       float *xtmp,float *logits,int n) {
    const char *gate=getenv("BEGLIN_QV19_ENABLE_EXCLUSIVE_MODEL_REQUESTS");
    const char *frozen=getenv("BEGLIN_QV5_FROZEN_ARC_CHALLENGE");
    qv19_native_need(gate && !strcmp(gate,"1") && frozen && !strcmp(frozen,"1"),
                     "requires explicit original C research gate");
    qv19_native_need(n>=2&&n<=48,"invalid C experiment request count");
    qv14_prepare_profiles();
    qv19_native_server s={0};
    s.request_count=n;s.input_root=base;s.shared_xtmp=xtmp;s.shared_logits=logits;
    qv19_native_need(qv19_model_lease_init(&s.lease,0)==QV19_OK,"lease init");
    qv19_native_need(pthread_mutex_init(&s.barrier_mutex,NULL)==0,"barrier init");
    qv19_native_need(pthread_cond_init(&s.barrier_cond,NULL)==0,"barrier condition init");
    pthread_t threads[2];qv19_native_worker workers[2];
    for(int i=0;i<2;i++) {
        workers[i].server=&s;workers[i].worker_index=i;
        qv19_native_need(pthread_create(&threads[i],NULL,
                         qv19_native_worker_run,&workers[i])==0,"thread start");
    }
    for(int i=0;i<2;i++)
        qv19_native_need(pthread_join(threads[i],NULL)==0,"join workers");
    qv19_lease_snapshot snapshot;
    qv19_native_need(qv19_lease_get_snapshot(&s.lease,&snapshot)==QV19_OK,
                     "stable final snapshot");
    qv19_native_need(snapshot.peak_active==1&&snapshot.active==0&&
        snapshot.completed==(uint64_t)n&&snapshot.epoch==(uint64_t)n&&
        !snapshot.poisoned&&s.ready==2&&s.done_count[0]>0&&s.done_count[1]>0&&
        s.done_count[0]+s.done_count[1]==n,
        "full guarded original C inference invariants");
    double total=0;int correct=0;
    for(int i=0;i<n;i++) {
        qv19_native_need(s.receipts[i].epoch_enter==s.receipts[i].epoch_exit &&
            s.receipts[i].epoch_enter>0,"per request epoch stability");
        total+=s.losses[i];correct+=(s.gold[i]==s.predicted[i]);
    }
    fprintf(stderr,"[QV19-MODEL-LEASE-TOTAL] requests %d workers 2 worker0 %d worker1 %d "
        "max_active %u epoch %llu poisoned %d guarded_forward_serial 1\n",
        n,s.done_count[0],s.done_count[1],snapshot.peak_active,
        (unsigned long long)snapshot.epoch,snapshot.poisoned);
    fprintf(stderr,"[QV15-CACHE-TOTAL] hits %lu misses %lu drops %lu bytes %zu "
        "peak_bytes %zu cap_bytes %zu\n",qv15_hits,qv15_misses,qv15_drops,
        qv15_cache_bytes,qv15_peak_cache_bytes,qv15_max_bytes);
    fprintf(stdout,"qv5_choice NLL %.17g items %d correct %d mean %.17g ppl %.17g\n",
        total,n,correct,total/n,exp(total/n));
    qv19_native_need(qv19_model_lease_destroy(&s.lease)==QV19_OK,
                     "shutdown after workers joined");
    qv19_native_need(pthread_cond_destroy(&s.barrier_cond)==0,"condition destroy");
    qv19_native_need(pthread_mutex_destroy(&s.barrier_mutex)==0,"barrier destroy");
}
