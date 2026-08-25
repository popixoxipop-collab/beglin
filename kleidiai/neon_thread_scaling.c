// neon_thread_scaling.c  -- Phase 0 / Task E, NEON control arm
//
// Direct counterpart of kai_thread_scaling.c: same shapes, same M, same thread
// sweep, same timing protocol (best-of-reps after 3 discarded warmups, wall clock
// via CLOCK_MONOTONIC), same GOP accounting (2*M*N*K/1e9) and same table layout --
// but driving vdsp's PRODUCTION NEON int8-SDOT batched GEMM
//   gemm_qXg64_sdot_mt(q4pool*, bits=4, ...)   [engine/q4gemv.h:779]
// through its production thread pool (q4pool), instead of the KleidiAI SME2
// micro-kernel sharded by an ad-hoc pool.
//
// This is the go/no-go comparison: if q4pool already scales near-linearly to the
// M4's 10 cores, the SME2 kernel's ~1.5-1.77x thread ceiling cannot win.
//
// q4gemv.h is included UNMODIFIED (byte-identical to the engine copy on D50).
// Nothing else in ~/vdsp_m4_bench/kleidiai/ is touched by this file.
//
// Activation staging is production-faithful (qwen_infer.c:1469-1476): each of the
// M int8 columns is produced in natural order and then q4_split_act()'d into the
// [lo16][hi16] chunk order the q4 batched kernels require.
//
// RESIDUAL/ERROR-FEEDBACK NOTE: this benchmark quantizes no model weights at all --
// codes/scales/activations below are synthetic operands that exist only to give the
// kernel something legal to time. There is therefore no quantization error for a
// residual/error-feedback term to carry; the engine's real residual policy stays
// upstream at quantization time (q4gemv.h file-top note).
//
// Pool configuration mirrors PRODUCTION (qwen_infer.c:1971-1973): Q4_POOL_SPIN
// defaults to 1 there, i.e. the engine runs the spin-wait barrier, NOT q4pool_init's
// own spin=0 condvar default. qos defaults to 0 in both. Both are settable here so
// the barrier's contribution is visible rather than assumed.
//
// Usage: neon_thread_scaling [model] [M] [reps] [spin] [qos]   model = llama|qwen|all

#include "q4gemv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/sysctl.h>

static uint64_t rng = 0x243F6A8885A308D3ULL;
static uint32_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 32); }

static double nowms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

typedef struct { const char *label; int out, in; } Shape;

static const Shape LLAMA[] = {
    { "llama q_proj/o_proj  (N=4096,  K=4096) ", 4096, 4096 },
    { "llama k_proj/v_proj  (N=1024,  K=4096) ", 1024, 4096 },
    { "llama gate/up_proj   (N=14336, K=4096) ", 14336, 4096 },
    { "llama down_proj      (N=4096,  K=14336)", 4096, 14336 },
};
static const Shape QWEN[] = {
    { "qwen  q_proj/o_proj  (N=1536,  K=1536) ", 1536, 1536 },
    { "qwen  k_proj/v_proj  (N=256,   K=1536) ", 256, 1536 },
    { "qwen  gate/up_proj   (N=8960,  K=1536) ", 8960, 1536 },
    { "qwen  down_proj      (N=1536,  K=8960) ", 1536, 8960 },
};

static const int THREADS[] = { 1, 2, 4, 8, 10 };
#define NTHR (int)(sizeof(THREADS) / sizeof(THREADS[0]))

#define POOL_MAX_IN 16384

static int g_spin = 1, g_qos = 0;   // production defaults (qwen_infer.c:1971-1973)

static void bench_shape(const Shape *sh, int M, int reps) {
    const int bl = 64;                      // q4g64: group size is fixed at 64
    size_t out = (size_t)sh->out, in = (size_t)sh->in;
    size_t ng = in / bl;
    size_t row_bytes = in / 2;

    // ---- synthetic operands (kbench_sdot.c fixture style, randomized scales) ----
    uint8_t *packed = (uint8_t *)malloc(out * row_bytes);
    float   *wscale = (float *)malloc(out * ng * sizeof(float));
    int8_t  *xq_nat = (int8_t *)malloc((size_t)M * in);      // natural order staging
    int8_t  *xq     = (int8_t *)malloc((size_t)M * in);      // split order (kernel input)
    float   *ascale = (float *)malloc((size_t)M * ng * sizeof(float));
    float   *y      = (float *)calloc(out * (size_t)M, sizeof(float));
    float   *ref    = (float *)calloc(out * (size_t)M, sizeof(float));
    if (!packed || !wscale || !xq_nat || !xq || !ascale || !y || !ref) {
        fprintf(stderr, "FATAL: alloc failed for %s\n", sh->label); exit(1);
    }

    for (size_t i = 0; i < out * row_bytes; i++) packed[i] = (uint8_t)(xr() & 0xFF);
    for (size_t i = 0; i < out * ng; i++)
        wscale[i] = 0.001f + 0.02f * ((float)(xr() & 0xFFFF) / 65535.0f);
    for (size_t i = 0; i < (size_t)M * in; i++) xq_nat[i] = (int8_t)((int)(xr() & 255) - 128);
    for (int m = 0; m < M; m++) q4_split_act(xq_nat + (size_t)m * in, xq + (size_t)m * in, (int)in);
    for (size_t i = 0; i < (size_t)M * ng; i++)
        ascale[i] = 0.005f + 0.03f * ((float)(xr() & 0xFFFF) / 65535.0f);

    double gflop = 2.0 * (double)out * (double)in * (double)M / 1e9;

    // ---- baseline: pool with T=1. gemm_qXg64_sdot_mt takes its nthreads==1 branch,
    //      which is a plain q4_run_range(p,0) on the calling thread -- no worker
    //      threads are ever spawned (q4pool_start loops i=1..nthreads-1), so this IS
    //      the direct single-threaded call, matching kai's "single-thread" row. ----
    q4pool base;
    q4pool_init(&base, 1, POOL_MAX_IN);
    base.spin = g_spin; base.qos = g_qos;   // set BEFORE q4pool_start (M29 contract)
    q4pool_start(&base);
    double t1_best = 1e30, t1_sum = 0;
    for (int r = 0; r < reps + 3; r++) {
        memset(y, 0, out * (size_t)M * sizeof(float));
        double t0 = nowms();
        gemm_qXg64_sdot_mt(&base, 4, packed, wscale, xq, ascale, NULL, y, (int)out, (int)in, M);
        double dt = nowms() - t0;
        if (r >= 3) { if (dt < t1_best) t1_best = dt; t1_sum += dt; }
    }
    memcpy(ref, y, out * (size_t)M * sizeof(float));
    q4pool_destroy(&base);

    printf("  %s  bl=%d M=%d reps=%d  work=%.3f GOP\n", sh->label, bl, M, reps, gflop);
    printf("    %-30s %10s %10s %8s %7s %8s %s\n",
           "config", "best_ms", "mean_ms", "speedup", "eff_%", "GOP/s", "bitexact_vs_1T");
    printf("    %-30s %10.4f %10.4f %8.3f %7.1f %8.1f %s\n", "single-thread (direct call)",
           t1_best, t1_sum / reps, 1.0, 100.0, gflop / (t1_best / 1e3), "-");

    for (int ti = 0; ti < NTHR; ti++) {
        int T = THREADS[ti];
        q4pool pool;
        q4pool_init(&pool, T, POOL_MAX_IN);     // fresh pool per thread count
        pool.spin = g_spin; pool.qos = g_qos;   // set BEFORE q4pool_start (M29 contract)
        q4pool_start(&pool);

        // rows are split evenly by q4_run_range: per = ceil(out/T), worker wi owns
        // [wi*per, min((wi+1)*per, out)) -- count the shards that get real work.
        int per = ((int)out + T - 1) / T, active = 0;
        for (int t = 0; t < T; t++) { int r0 = t * per; if (r0 < (int)out) active++; }

        double best = 1e30, sum = 0;
        int exact = 1;
        for (int r = 0; r < reps + 3; r++) {
            memset(y, 0, out * (size_t)M * sizeof(float));
            double t0 = nowms();
            gemm_qXg64_sdot_mt(&pool, 4, packed, wscale, xq, ascale, NULL, y, (int)out, (int)in, M);
            double dt = nowms() - t0;
            if (r >= 3) { if (dt < best) best = dt; sum += dt; }
            if (memcmp(y, ref, out * (size_t)M * sizeof(float)) != 0) exact = 0;
        }
        q4pool_destroy(&pool);

        char cfg[64]; snprintf(cfg, sizeof cfg, "pool T=%d (%d active shards)", T, active);
        printf("    %-30s %10.4f %10.4f %8.3f %7.1f %8.1f %s\n", cfg, best, sum / reps,
               t1_best / best, 100.0 * (t1_best / best) / T, gflop / (best / 1e3), exact ? "yes" : "NO");
    }
    printf("\n");

    free(packed); free(wscale); free(xq_nat); free(xq); free(ascale); free(y); free(ref);
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : "llama";
    int M = argc > 2 ? atoi(argv[2]) : 16;
    int reps = argc > 3 ? atoi(argv[3]) : 40;
    g_spin = argc > 4 ? atoi(argv[4]) : 1;     // production default
    g_qos  = argc > 5 ? atoi(argv[5]) : 0;     // production default

    char brand[128] = {0}; size_t bs = sizeof brand;
    sysctlbyname("machdep.cpu.brand_string", brand, &bs, NULL, 0);
    int ncpu = 0, nperf = 0, neff = 0; size_t sz = sizeof(int);
    sysctlbyname("hw.ncpu", &ncpu, &sz, NULL, 0); sz = sizeof(int);
    sysctlbyname("hw.perflevel0.logicalcpu", &nperf, &sz, NULL, 0); sz = sizeof(int);
    sysctlbyname("hw.perflevel1.logicalcpu", &neff, &sz, NULL, 0);
    printf("host cpu   : %s  ncpu=%d (P=%d E=%d)\n", brand, ncpu, nperf, neff);
    printf("kernel     : vdsp production NEON int8-SDOT batched GEMM "
           "gemm_qXg64_sdot_mt(bits=4) via q4pool\n");
    printf("params     : M=%d bl=64 reps=%d model=%s Q4_SDOT_BMAX=%d  pool.spin=%d pool.qos=%d%s\n\n",
           M, reps, model, Q4_SDOT_BMAX, g_spin, g_qos,
           (g_spin == 1 && g_qos == 0) ? "  [PRODUCTION default]" : "  [non-default]");

    if (M < 1 || M > Q4_SDOT_BMAX) { fprintf(stderr, "FATAL: M=%d out of [1,%d]\n", M, Q4_SDOT_BMAX); return 1; }

    if (!strcmp(model, "llama") || !strcmp(model, "all"))
        for (size_t i = 0; i < sizeof(LLAMA) / sizeof(LLAMA[0]); i++) bench_shape(&LLAMA[i], M, reps);
    if (!strcmp(model, "qwen") || !strcmp(model, "all"))
        for (size_t i = 0; i < sizeof(QWEN) / sizeof(QWEN[0]); i++) bench_shape(&QWEN[i], M, reps);
    return 0;
}
