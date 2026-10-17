/* Native BEGLIN qNg64 group-64 baseline vs tiled Accelerate GEMM C API test.
 * Deterministic synthetic F32 matrices, no model weights/serving.
 *
 * Two builds:
 * clang -O3 -std=c11 -DACCELERATE_NEW_LAPACK -I. qng64_dense.c \
 *       benchmarks/qng64_tiled_accelerate_poc.c \
 *       tests/test_qng64_tiled_accelerate_poc.c -framework Accelerate -lm -o /tmp/qngt
 * clang ... -DBEGLIN_FORCE_SCALAR_UNPACK=1 ... -o /tmp/qngt_scalar
 */
#include "qng64_dense.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int beglin_qng64_tiled_cblas_matmul(
    const uint8_t *, const float *, int, const float *,
    const float *, float *, int, int, int, int);

static void check(int cond, const char *msg) {
    if (!cond) { fprintf(stderr,"BEGLIN_QNG64_TILED_TEST_FAIL: %s\n",msg);exit(1); }
}

static double sse_l2(const float *a, const float *b, int count, double *max_abs) {
    double num=0.0,den=0.0,mx=0.0;
    for (int i=0;i<count;++i) {
        check(isfinite(a[i])&&isfinite(b[i]),"nonfinite result");
        const double d=(double)a[i]-b[i];
        num+=d*d;den+=(double)a[i]*a[i];
        if (fabs(d)>mx) mx=fabs(d);
    }
    if (max_abs) *max_abs=mx;
    return sqrt(num/(den>0.0?den:1.0));
}

int main(void) {
    const int widths[]={2,3,5,6,7,8,9,12,15};
    const int shapes[3][4]={{64,128,1,16},{65,512,3,16},{67,128,13,64}};
    const int cases=3;
    int completed=0;
    double max_rel=0.0,max_abs=0.0;
    for (unsigned wi=0;wi<sizeof(widths)/sizeof(widths[0]);++wi) {
        const int n=widths[wi];
        for (int caseid=0;caseid<cases;++caseid) {
            const int out=shapes[caseid][0], in=shapes[caseid][1];
            const int M=shapes[caseid][2], tile=shapes[caseid][3];
            const size_t count=(size_t)out*(size_t)in;
            const size_t codes=qng64_packed_bytes(out,in,n);
            const size_t ng=(size_t)out*(size_t)(in/64);
            check(codes>0 && codes==ng*(size_t)n*8u,"invalid native packed byte count");
            float *W=malloc(count*sizeof(float));
            float *X=malloc((size_t)M*(size_t)in*sizeof(float));
            float *scales=malloc(ng*sizeof(float));
            float *ref=malloc((size_t)M*(size_t)out*sizeof(float));
            float *fast=malloc((size_t)M*(size_t)out*sizeof(float));
            uint8_t *packed=calloc(codes,1);
            check(W&&X&&scales&&ref&&fast&&packed,"allocation");
            for (size_t i=0;i<count;++i)
                W[i]=sinf((float)i*.0611f)*.11f+cosf((float)i*.0379f)*.035f;
            for (int i=0;i<M*in;++i)
                X[i]=cosf((float)i*.021f)*.13f+sinf((float)i*.049f)*.02f;
            check(qng64_quantize_f32(W,out,in,n,packed,scales)==0,"native encoder");
            qng64_matmul_f32(packed,scales,n,X,NULL,ref,out,in,M);
            check(beglin_qng64_tiled_cblas_matmul(
                packed,scales,n,X,NULL,fast,out,in,M,tile)==0,"tiled GEMM returned error");
            double mx=0.0;
            const double rel=sse_l2(ref,fast,M*out,&mx);
            check(rel<8e-6 && mx<3e-4,"tiled output differs from original C");
            if (rel>max_rel) max_rel=rel;
            if (mx>max_abs) max_abs=mx;
            ++completed;
            free(W);free(X);free(scales);free(ref);free(fast);free(packed);
        }
    }
    /* The no-write on invalid arguments policy is checked with intact buffers. */
    float x[128]={0.f},out[64];
    float scales[2]={1.f,1.f};
    uint8_t packed[2*7*8]={0};
    for (int i=0;i<64;++i)out[i]=123.f;
    const int wrong[4][2]={{4,16},{7,0},{7,65},{16,16}};
    for (int j=0;j<4;++j) {
        const int rc=beglin_qng64_tiled_cblas_matmul(
          packed,scales,wrong[j][0],x,NULL,out,64,128,1,wrong[j][1]);
        check(rc==-1,"invalid format/tile accepted");
        for (int i=0;i<64;++i)check(out[i]==123.f,"invalid call changed output");
    }
#ifdef BEGLIN_FORCE_SCALAR_UNPACK
    const char *decoder="scalar_fallback";
#elif defined(__ARM_NEON)
    const char *decoder="NEON";
#else
    const char *decoder="scalar_fallback_no_NEON";
#endif
    check(completed==27,"expected case count mismatch");
    printf("BEGLIN_QNG64_TILED_ACCELERATE_PASS cases=%d rejects=%d decoder=%s max_relative_L2=%.9g max_abs=%.9g\n",
           completed,4,decoder,max_rel,max_abs);
    return 0;
}
