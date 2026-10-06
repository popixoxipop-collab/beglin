#!/usr/bin/env python3
from pathlib import Path
import subprocess
p=Path("qwen_infer.c");s=p.read_text()
a='static double (*g_moe_row_fn)(const uint8_t *, MoeAFTensor *, long, long, const float *) = moe_matvec_af_row;\n'
ins=r'''
// QT mixed-cell experiment adapter. Side registry keeps MoeAFTensor ABI unchanged.
typedef double (*MoeMixedRowFn)(const uint8_t *, MoeAFTensor *, long, long, const float *, const void *);
typedef struct { MoeAFTensor *tensor; MoeMixedRowFn row_fn; const void *ctx; } MoeMixedOverride;
#define MOE_MIXED_OVERRIDE_MAX 128
static MoeMixedOverride g_moe_mixed_overrides[MOE_MIXED_OVERRIDE_MAX];
static int g_moe_mixed_override_count = 0;
static const MoeMixedOverride *moe_mixed_override_find(const MoeAFTensor *t) {
    for (int i=0;i<g_moe_mixed_override_count;i++) if (g_moe_mixed_overrides[i].tensor==t) return &g_moe_mixed_overrides[i];
    return NULL;
}
static void moe_mixed_override_register(MoeAFTensor *t, MoeMixedRowFn fn, const void *ctx) {
    if (!t || !fn || moe_mixed_override_find(t) || g_moe_mixed_override_count>=MOE_MIXED_OVERRIDE_MAX) {
        fprintf(stderr,"FATAL: invalid QT mixed override registration\n"); exit(1);
    }
    g_moe_mixed_overrides[g_moe_mixed_override_count++] = (MoeMixedOverride){t,fn,ctx};
}
static inline double moe_row_dispatch(const uint8_t *blob,MoeAFTensor *t,long e,long row,const float *x) {
    const MoeMixedOverride *ov=moe_mixed_override_find(t);
    return ov ? ov->row_fn(blob,t,e,row,x,ov->ctx) : g_moe_row_fn(blob,t,e,row,x);
}
'''
if s.count(a)!=1: raise SystemExit("row-fn anchor mismatch")
s=s.replace(a,a+ins,1)
old='for (long r = r0; r < r1; r++) j->y[r] = (float)g_moe_row_fn(j->blob, j->t, j->e, r, j->x);'
new='for (long r = r0; r < r1; r++) j->y[r] = (float)moe_row_dispatch(j->blob, j->t, j->e, r, j->x);'
if s.count(old)!=1: raise SystemExit("dispatch anchor mismatch")

# Phase 2 is intentionally correctness-first: the side-registry plumbing above is
# independently compile-gated before this loader is promoted. Packed Metal remains separate.
anchor='typedef struct { char name[128]; long off, numel; } MoeF32Tensor;\n'
loader=r'''
typedef struct {
    int out, in, ng;
    float *dequant;
    uint8_t *bits;
} MoeMixedDenseCtx;

static double moe_mixed_dense_row(const uint8_t *blob, MoeAFTensor *t, long e, long row,
                                  const float *x, const void *opaque) {
    (void)blob; (void)t; (void)e;
    const MoeMixedDenseCtx *m=(const MoeMixedDenseCtx *)opaque;
    const float *w=m->dequant+(size_t)row*m->in;
    double acc=0.0;
    for (int i=0;i<m->in;i++) acc+=(double)w[i]*x[i];
    return acc;
}
static void moe_mixed_dense_register_dequant(MoeAFTensor *t, const float *weights,
                                             const uint8_t *bits, int out, int in) {
    if (!t || t->out!=out || t->in!=in || in%64) {
        fprintf(stderr,"FATAL: QT mixed dense shape mismatch\n"); exit(1);
    }
    int ng=in/64; size_t cells=(size_t)out*ng;
    MoeMixedDenseCtx *m=calloc(1,sizeof(*m));
    m->out=out; m->in=in; m->ng=ng;
    m->bits=malloc(cells); m->dequant=malloc((size_t)out*in*sizeof(float));
    if (!m->bits || !m->dequant) { fprintf(stderr,"FATAL: QT mixed alloc\n"); exit(1); }
    memcpy(m->bits,bits,cells);
    for (int r=0;r<out;r++) for (int g=0;g<ng;g++) {
        int nb=m->bits[(size_t)r*ng+g];
        if (nb<4 || nb>8) { fprintf(stderr,"FATAL: QT mixed bits[%d,%d]=%d\n",r,g,nb); exit(1); }
        int qmax=(1<<(nb-1))-1, qmin=-(1<<(nb-1));
        const float *src=weights+(size_t)r*in+g*64;
        float mx=0.f; for(int i=0;i<64;i++){float a=fabsf(src[i]);if(a>mx)mx=a;}
        float sc=mx>1e-12f?mx/qmax:1.f, inv=1.f/sc, residual=0.f;
        float *dst=m->dequant+(size_t)r*in+g*64;
        for(int i=0;i<64;i++){
            float z=src[i]+residual; int q=(int)lrintf(z*inv);
            if(q<qmin)q=qmin;if(q>qmax)q=qmax;
            dst[i]=q*sc; residual=z-dst[i];
        }
    }
    moe_mixed_override_register(t,moe_mixed_dense_row,m);
}
'''
if s.count(anchor)!=1: raise SystemExit("loader anchor mismatch")
s=s.replace(anchor,loader+anchor,1)


# Dense Qwen correctness hook: mark st_register_q4g64_as() as the only legal
# source point for mixed-cell construction, before canonical Q4 transcode.
dense_anchor='static WT *st_register_q4g64_as(const char *name) {\n'
dense_ins=r'''
static int qt_dense_mixed_target(const char *name) {
    const char *m=getenv("QWEN_QT_MIXED_MANIFEST");
    if (!m || !m[0]) return 0;
    int layer=-1; char role[16]={0};
    if (sscanf(name,"model.layers.%d.self_attn.%15[^.].weight",&layer,role)!=2) return 0;
    if (layer<0) return 0;
    if (strcmp(role,"q_proj") && strcmp(role,"k_proj") && strcmp(role,"v_proj") && strcmp(role,"o_proj")) return 0;
    char bp[1024],safe[256]; size_t n=strlen(name); if(n>=sizeof safe) return 0;
    for(size_t i=0;i<=n;i++) safe[i]=(name[i]=='.')?'_':name[i];
    snprintf(bp,sizeof bp,"%s/%s.bits",m,safe);
    FILE *f=fopen(bp,"rb"); if(!f) return 0; fclose(f); return 1;
}
'''
if s.count(dense_anchor)!=1: raise SystemExit("dense register anchor mismatch")
s=s.replace(dense_anchor,dense_ins+dense_anchor,1)
# Fail closed for now unless the actual original-float registration path is wired;
# this proves the env gate reaches exactly the intended two tensors without silently
# double-quantizing. The next hunk replaces this guard with the real loader.
probe='    int out = (int)t->shape[0], in = (int)t->shape[1];\n'
guard=r'''    if (qt_dense_mixed_target(name)) {
        fprintf(stderr,"[qt mixed dense] target=%s source=original-safetensors\n",name);
    }
'''
if s.count(probe)<1: raise SystemExit("dense shape anchor mismatch")
pos=s.index(dense_anchor); tail=s[pos:]; 
if tail.count(probe)<1: raise SystemExit("dense local shape anchor mismatch")
tail=tail.replace(probe,probe+guard,1);s=s[:pos]+tail


# Dense correctness implementation: when QWEN_QT_MIXED_MANIFEST points to a
# directory containing <sanitized_tensor>.bits, build a K_F32 WT from the SAME
# original safetensors dequant buffer after per-cell n4..n8 EF quant/dequant.
dense_quant_anchor='    int ng = in / 64;\n    uint8_t *packed = malloc((size_t)out * (in / 2));\n'
dense_quant=r'''    int ng = in / 64;
    if (qt_dense_mixed_target(name)) {
        const char *dir=getenv("QWEN_QT_MIXED_MANIFEST");
        char bp[1024], safe[256]; size_t sn=strlen(name);
        if (sn>=sizeof safe) { fprintf(stderr,"FATAL: QT mixed tensor name too long\n"); exit(1); }
        for(size_t i=0;i<=sn;i++) safe[i]=(name[i]=='.')?'_':name[i];
        snprintf(bp,sizeof bp,"%s/%s.bits",dir,safe);
        FILE *bf=fopen(bp,"rb"); if(!bf){perror(bp);exit(1);}
        size_t cells=(size_t)out*ng; uint8_t *bits=malloc(cells);
        if(!bits || fread(bits,1,cells,bf)!=cells || fgetc(bf)!=EOF){fprintf(stderr,"FATAL: QT mixed bits size %s\n",bp);exit(1);}
        fclose(bf);
        float *mix=malloc(sizeof(float)*(size_t)out*in); if(!mix){fprintf(stderr,"FATAL: QT mixed alloc\n");exit(1);}
        const char *pass=getenv("QWEN_QT_FP32_PASSTHROUGH");
        if(pass && pass[0] && strcmp(pass,"0")){
            memcpy(mix,deq,sizeof(float)*(size_t)out*in); free(bits); free(deq);
            WT *w=&g_wt[g_nwt++]; snprintf(w->name,sizeof w->name,"%s",name);
            w->kind=K_F32;w->in=in;w->out=out;w->ng=ng;w->f32=mix;w->packed=NULL;w->scales=NULL;w->sub=NULL;
            w->kai_rhs=NULL;w->kai_rhs_bytes=0;w->kai_lazy_failed=0;
            fprintf(stderr,"[qt mixed dense] fp32-passthrough %s source=original-safetensors\\n",name);
            return w;
        }
        for(int r=0;r<out;r++) for(int g=0;g<ng;g++){
            int nb=bits[(size_t)r*ng+g]; if(nb<4||nb>8){fprintf(stderr,"FATAL: QT mixed bit=%d\n",nb);exit(1);}
            int qmax=(1<<(nb-1))-1,qmin=-(1<<(nb-1)); const float *src=deq+(size_t)r*in+g*64;
            float mx=0.f;for(int i=0;i<64;i++){float a=fabsf(src[i]);if(a>mx)mx=a;}
            float sc=mx>1e-12f?mx/qmax:1.f,inv=1.f/sc,res=0.f;float *dst=mix+(size_t)r*in+g*64;
            for(int i=0;i<64;i++){float z=src[i]+res;int q=(int)lrintf(z*inv);if(q<qmin)q=qmin;if(q>qmax)q=qmax;dst[i]=q*sc;res=z-dst[i];}
        }
        free(bits); free(deq);
        WT *w=&g_wt[g_nwt++]; snprintf(w->name,sizeof w->name,"%s",name);
        w->kind=K_F32;w->in=in;w->out=out;w->ng=ng;w->f32=mix;w->packed=NULL;w->scales=NULL;w->sub=NULL;
        w->kai_rhs=NULL;w->kai_rhs_bytes=0;w->kai_lazy_failed=0;
        fprintf(stderr,"[qt mixed dense] registered %s cells=%zu source=original-safetensors\n",name,cells);
        return w;
    }
    uint8_t *packed = malloc((size_t)out * (in / 2));
'''
# replace only inside st_register_q4g64_as by selecting after its function anchor
pos=s.index('static WT *st_register_q4g64_as(const char *name) {')
tail=s[pos:]
if tail.count(dense_quant_anchor)<1: raise SystemExit("dense quant anchor mismatch")
tail=tail.replace(dense_quant_anchor,dense_quant,1);s=s[:pos]+tail

p.write_text(s.replace(old,new,1))
subprocess.run(["python3","tools/patch_dense_layerdump.py"],check=True)
