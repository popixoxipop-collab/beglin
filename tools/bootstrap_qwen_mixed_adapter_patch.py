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
p.write_text(s.replace(old,new,1))
