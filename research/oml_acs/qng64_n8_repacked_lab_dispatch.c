/*
 * ONLY for isolated BEGLIN native-Q laboratory binary. Source bits in HF
 * checkpoint and on-disk qNg64 n8 bitplanes never change. Loader has already
 * repacked all n8 buffers IN PLACE to contiguous signed int8 codes.
 *
 * Explicitly exports both matvec and batched matmul; thus the original scalar
 * batch reference never reads repacked memory as bitplanes. For other qNg64
 * widths, forward the unchanged original BEGLIN scalar entrypoints.
 *
 * Original qng64_dense.c is compiled only in this experiment with symbol
 * aliases so the production C files remain unmodified.
 */
#include "qng64_dense.h"
#include "research/oml_acs/qng64_n8_repacked_neon.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdatomic.h>

void beglin_original_scalar_matvec_f32(const uint8_t*,const float*,int,
                                       const float*,const float*,float*,int,int);
void beglin_original_scalar_matmul_f32(const uint8_t*,const float*,int,
                                       const float*,const float*,float*,int,int,int);
static atomic_ullong n8_fused_gemv_calls=0;
static atomic_ullong n8_batched_gemv_calls=0;
static atomic_ullong original_other_bit_calls=0;
static atomic_ullong original_other_bit_batches=0;
static void report(void) {
    fprintf(stderr,"[BEGLIN-N8-REPACKED-LAB] fast_gemv=%llu fast_batch_rows=%llu other_bit_gemv=%llu other_bit_batches=%llu source_unchanged=1\n",
      (unsigned long long)atomic_load(&n8_fused_gemv_calls),
      (unsigned long long)atomic_load(&n8_batched_gemv_calls),
      (unsigned long long)atomic_load(&original_other_bit_calls),
      (unsigned long long)atomic_load(&original_other_bit_batches));
}
__attribute__((constructor))
static void register_experiment_report(void) {
    (void)atexit(report);
}
void qng64_matvec_f32(const uint8_t *packed,const float *scales,int n,
                      const float *x,const float *bias,float *y,int out,int in) {
    if(n==8) {
        if(beglin_qng64_n8_repacked_matvec_neon((const int8_t*)packed,scales,x,bias,y,out,in)!=0) {
            fprintf(stderr,"FATAL: isolated repacked n8 invalid tensor shape\n");abort();
        }
        atomic_fetch_add_explicit(&n8_fused_gemv_calls,1,memory_order_relaxed);
        return;
    }
    atomic_fetch_add_explicit(&original_other_bit_calls,1,memory_order_relaxed);
    beglin_original_scalar_matvec_f32(packed,scales,n,x,bias,y,out,in);
}
void qng64_matmul_f32(const uint8_t *packed,const float *scales,int n,
                      const float *x,const float *bias,float *y,int out,int in,int M) {
    if(n==8) {
        if(!x || !y || !packed || !scales || out<=0 || in<=0 || M<=0 || (in&63)) {
            fprintf(stderr,"FATAL: isolated repacked n8 batch invalid geometry\n");abort();
        }
        for(int i=0;i<M;i++) {
            if(beglin_qng64_n8_repacked_matvec_neon((const int8_t*)packed,scales,x+(size_t)i*in,
                   bias,y+(size_t)i*out,out,in)!=0) {
                fprintf(stderr,"FATAL: isolated repacked n8 batch failed\n");abort();
            }
        }
        atomic_fetch_add_explicit(&n8_batched_gemv_calls,(unsigned long long)M,memory_order_relaxed);
        return;
    }
    atomic_fetch_add_explicit(&original_other_bit_batches,1,memory_order_relaxed);
    beglin_original_scalar_matmul_f32(packed,scales,n,x,bias,y,out,in,M);
}
