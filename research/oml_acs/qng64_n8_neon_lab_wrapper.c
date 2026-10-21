/*
 * Isolated BEGLIN CPU qNg64 n8 GEMV dispatch for research.
 * Builds original qng64_dense.c with qng64_matvec_f32 renamed to
 * beglin_original_scalar_matvec_f32; only this link target uses wrapper.
 * Research NEON SIMD path is used only for n=8 dense M=1 GEMV.
 * Batched qng64_matmul_f32 continues to call original scalar reference.
 * Live engine and production source are never modified.
 */
#include "qng64_dense.h"
#include "research/oml_acs/qng64_n8_neon_experimental.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void beglin_original_scalar_matvec_f32(const uint8_t *packed,
    const float *scales,int n,const float *x,const float *bias,
    float *y,int out,int in);

static atomic_ullong fast_calls=0;
static atomic_ullong fallback_calls=0;

static void report(void) {
    fprintf(stderr,"[BEGLIN-N8-NEON-RESEARCH] fast_calls=%llu fallback_calls=%llu full-model production=false\n",
       (unsigned long long)atomic_load(&fast_calls),
       (unsigned long long)atomic_load(&fallback_calls));
}
__attribute__((constructor))
static void add_research_exit_report(void) {
    (void)atexit(report);
}

void qng64_matvec_f32(const uint8_t *packed,
    const float *scales,int n,const float *x,const float *bias,
    float *y,int out,int in) {
    if(n==8 && beglin_qng64_n8_matvec_neon(
          packed,scales,x,bias,y,out,in)==0) {
        atomic_fetch_add(&fast_calls,1);
        return;
    }
    atomic_fetch_add(&fallback_calls,1);
    beglin_original_scalar_matvec_f32(packed,scales,n,x,bias,y,out,in);
}
