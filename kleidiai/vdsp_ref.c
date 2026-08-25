// vdsp_ref.c -- computes the vdsp reference (gemm_qXg64_sdot_mt) result, compiled WITHOUT
// +sve2 (plain -mcpu=apple-m4 only) to avoid the autovectorizer emitting non-streaming SVE
// instructions into this NEON-only code (M4 has SME/streaming-SVE only, no base SVE --
// confirmed by direct SIGILL when kai_test.c's +sve2 flag leaked into this function).
#include "q4gemv.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now2(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

// out params: y_ref [out*M] col-major (m*out+row), dt_sdot = per-call ms
void vdsp_ref_compute(const uint8_t *packed, const float *wscale, const float *act_f32,
                       const float *bias, int out, int in, int M, int bl,
                       float *y_ref, double *dt_sdot_ms) {
    int ng = in / bl;
    int8_t *xq_nat = malloc((size_t)M*in), *xq_split = malloc((size_t)M*in);
    float *ascale = malloc((size_t)M*ng*4);
    for (int m = 0; m < M; m++) {
        for (int gi = 0; gi < ng; gi++) {
            float mx = 1e-8f;
            for (int k = 0; k < bl; k++) { float v = fabsf(act_f32[(size_t)m*in+gi*bl+k]); if (v > mx) mx = v; }
            float sc = mx / 127.0f; ascale[(size_t)m*ng+gi] = sc;
            for (int k = 0; k < bl; k++) {
                int q = (int)lroundf(act_f32[(size_t)m*in+gi*bl+k] / sc);
                if (q > 127) q = 127; if (q < -127) q = -127;
                xq_nat[(size_t)m*in+gi*bl+k] = (int8_t)q;
            }
        }
        q4_split_act(xq_nat + (size_t)m*in, xq_split + (size_t)m*in, in);
    }
    q4pool pool; q4pool_init(&pool, 1, in); q4pool_start(&pool);
    gemm_qXg64_sdot_mt(&pool, 4, packed, wscale, xq_split, ascale, bias, y_ref, out, in, M);

    double t0 = now2(); int N = out >= 4096 ? 30 : 200;
    float wscale0_bak = wscale[0];
    for (int rep = 0; rep < N; rep++) {
        ((float*)wscale)[0] += 1e-9f;
        gemm_qXg64_sdot_mt(&pool, 4, packed, wscale, xq_split, ascale, bias, y_ref, out, in, M);
    }
    ((float*)wscale)[0] = wscale0_bak;
    *dt_sdot_ms = (now2()-t0)/N*1e3;

    q4pool_destroy(&pool);
    free(xq_nat); free(xq_split); free(ascale);
}
