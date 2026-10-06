#!/usr/bin/env python3
from pathlib import Path
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

p.write_text(s.replace(old,new,1))
