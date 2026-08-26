// Phase 3 sub-step 5 (PLAN_general_purpose_loader.md, D-gen-3's M-threshold policy table):
// kai_route() currently gates SME2 dispatch on ONE number, kai_sme2_min_m() -- the SME2
// kernel's own hardware row-tile minimum (mr=16 on M4/SME2). That is a CORRECTNESS floor
// (below it the kernel's tiling assumption isn't met), not necessarily a PERFORMANCE
// threshold -- Phase 0's shape_soak.c deliberately measured SME2's own absolute timing curve
// only and left "NEON-comparison timing is a separate, later step" for exactly this
// benchmark to do (see shape_soak.c's own header comment).
//
// IMPORTANT correction found while writing this: an earlier draft of this bench measured the
// f16p-LHS kernel family (kai_sme2_gemm_f16lhs/kai_sme2_repack_q4g64_f16lhs) -- but
// kai_route()'s two real call sites (matmul_t/matmul_sdot in qwen_infer.c, confirmed by
// reading the file directly) dispatch to the INT8-LHS kernel (kai_sme2_gemm_f32/
// kai_sme2_repack_q4g64, no "_f16lhs" suffix), the same kernel kai_sme2_min_m() itself
// queries. This file measures the int8-LHS variant, matching what kai_route() actually uses.
//
// SECOND correction, found while investigating a real contradiction: sme2_kai.h's own comment
// on kai_sme2_min_m() says "verified 2026-08-16: M=1 decode is NOT a target" -- but this file's
// FIRST version showed SME2 beating NEON at M=1 across 4 shapes, by 2.3x-3.9x. Read
// qwen_infer.c's two kai_route() call sites again to resolve this: matmul_t's NEON fallback is
// gemm_qXg64_mt (PLAIN fp32-activation NEON), but matmul_sdot's NEON fallback is
// gemm_qXg64_sdot_mt (int8-SDOT NEON -- the activation is ALSO dynamically quantized to int8,
// same information-loss trade the SME2 int8-LHS kernel itself makes). This file's first version
// only measured the plain-fp32 comparator, so it was answering "does SME2 beat matmul_t's NEON
// fallback" -- not "does SME2 beat matmul_sdot's NEON fallback," which is the harder, and for a
// production dense-projection model, more commonly-hit comparison (matmul_sdot is the serve/
// batched-decode path). The ORIGINAL "M16 near-zero, M64 +37%" dense-projection threshold
// finding (vdsp_general_serving_engine_goal.md project memory, predates this repo) was measured
// on the production dense-projection shape family through matmul_sdot's real call path -- i.e.
// against gemm_qXg64_sdot_mt, not gemm_qXg64_mt. This file now measures BOTH NEON comparators on
// the same shapes so the two prior findings can be reconciled instead of guessed at.
//
// Same isolated-benchmark caveat this project has hit before (the "M29 default promotion"
// note in qwen_infer.c explicitly used REAL end-to-end engine runs, not an isolated
// microbenchmark, for exactly this reason): a tight repeated-call loop over the SAME (out,in)
// shape benefits from warm caches/branch prediction/repack-amortization in ways a real
// forward pass (many DIFFERENT tensor shapes per layer, each touched once per token) may not
// fully share. This file's numbers are the FIRST signal, not the final word -- see RESULTS.md
// for the real end-to-end serve-mode validation this file's finding was checked against
// before being wired into kai_route().

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "sme2_kai.h"
#include "q4gemv.h"

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec*1e-9; }

static void pack_row_q4g64(uint8_t *packed_row, const int *nibbles, int in) {
    for (int b = 0; b < in / 2; b++)
        packed_row[b] = (uint8_t)((nibbles[2*b] & 0xF) | ((nibbles[2*b+1] & 0xF) << 4));
}

static q4pool g_pool;

static void bench_shape(const char *label, int out, int in, int T) {
    if (!kai_sme2_available()) { fprintf(stderr, "[kai_route_bench] SME2 unavailable, aborting\n"); exit(1); }
    if (!kai_sme2_shape_ok(out, in)) { printf("SKIP %s out=%d in=%d shape_ok=0\n", label, out, in); return; }
    int ng = in / 64;
    srand(20260826 + out*1000003 + in*97);

    int *nibbles = malloc(sizeof(int) * (size_t)out * in);
    float *scales = malloc(sizeof(float) * (size_t)out * ng);
    for (long r = 0; r < out; r++) {
        for (int c = 0; c < in; c++) nibbles[r*(long)in+c] = rand() % 16;
        for (int g = 0; g < ng; g++) scales[r*(long)ng+g] = 0.01f + (rand() % 1000) / 5000.0f;
    }
    uint8_t *packed = malloc((size_t)out * (in/2));
    for (long r = 0; r < out; r++) pack_row_q4g64(packed + (size_t)r*(in/2), nibbles + r*(long)in, in);

    q4pool_init(&g_pool, T, in);
    q4pool_start(&g_pool);

    size_t rhs_bytes = kai_sme2_rhs_packed_bytes(out, in);
    void *rhs_packed = aligned_alloc(64, (rhs_bytes + 63) & ~(size_t)63);
    if (kai_sme2_repack_q4g64(out, in, packed, scales, rhs_packed, rhs_bytes) != 0) {
        fprintf(stderr, "FATAL: repack failed for %s\n", label); exit(1);
    }

    int m_values[] = {1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64};
    int n_m = sizeof(m_values)/sizeof(m_values[0]);
    for (int mi = 0; mi < n_m; mi++) {
        int M = m_values[mi];
        float *x = malloc(sizeof(float) * (size_t)M * in);
        for (long i = 0; i < (long)M*in; i++) x[i] = ((rand() % 2000) - 1000) / 500.0f;
        float *y_neon = malloc(sizeof(float) * (size_t)M * out);
        float *y_sme2 = malloc(sizeof(float) * (size_t)M * out);
        size_t scratch_bytes = kai_sme2_lhs_scratch_bytes(M, in);
        void *scratch = aligned_alloc(64, (scratch_bytes + 63) & ~(size_t)63);

        gemm_qXg64_mt(&g_pool, 4, packed, scales, x, NULL, y_neon, out, in, M);           // warmup
        kai_sme2_gemm_f32(M, out, in, x, rhs_packed, NULL, y_sme2, scratch);              // warmup

        double neon_times[7], sme2_times[7];
        for (int rep = 0; rep < 7; rep++) {
            double t0 = now_s();
            gemm_qXg64_mt(&g_pool, 4, packed, scales, x, NULL, y_neon, out, in, M);
            neon_times[rep] = now_s() - t0;
            t0 = now_s();
            kai_sme2_gemm_f32(M, out, in, x, rhs_packed, NULL, y_sme2, scratch);
            sme2_times[rep] = now_s() - t0;
        }
        for (int i=1;i<7;i++){double kv=neon_times[i];int j=i-1;while(j>=0&&neon_times[j]>kv){neon_times[j+1]=neon_times[j];j--;}neon_times[j+1]=kv;}
        for (int i=1;i<7;i++){double kv=sme2_times[i];int j=i-1;while(j>=0&&sme2_times[j]>kv){sme2_times[j+1]=sme2_times[j];j--;}sme2_times[j+1]=kv;}
        double neon_med = neon_times[3], sme2_med = sme2_times[3];
        printf("RESULT shape=%s out=%d in=%d M=%d neon_ms=%.4f sme2_ms=%.4f sme2_faster=%d ratio=%.3f\n",
               label, out, in, M, neon_med*1000, sme2_med*1000, sme2_med < neon_med, neon_med/sme2_med);

        // Second comparator: gemm_qXg64_sdot_mt (int8-SDOT NEON), matmul_sdot's REAL fallback --
        // matches the mixture of matmul_t (compared above) vs matmul_sdot (compared here) as the
        // engine's two actual kai_route() call sites. Quantization mirrors matmul_sdot's own
        // per-row q4_quant_act_i8+q4_split_act loop (qwen_infer.c ~line 1810) exactly.
        int8_t *xq_nat = malloc((size_t)M * in);
        int8_t *xq_split = malloc((size_t)M * in);
        float *ascale = malloc(sizeof(float) * (size_t)M * ng);
        float *y_sdot = malloc(sizeof(float) * (size_t)M * out);
        for (int m = 0; m < M; m++) {
            q4_quant_act_i8(x + (size_t)m*in, in, xq_nat + (size_t)m*in, ascale + (size_t)m*ng);
            q4_split_act(xq_nat + (size_t)m*in, xq_split + (size_t)m*in, in);
        }
        gemm_qXg64_sdot_mt(&g_pool, 4, packed, scales, xq_split, ascale, NULL, y_sdot, out, in, M);  // warmup
        double sdot_times[7];
        for (int rep = 0; rep < 7; rep++) {
            double t0 = now_s();
            gemm_qXg64_sdot_mt(&g_pool, 4, packed, scales, xq_split, ascale, NULL, y_sdot, out, in, M);
            sdot_times[rep] = now_s() - t0;
        }
        for (int i=1;i<7;i++){double kv=sdot_times[i];int j=i-1;while(j>=0&&sdot_times[j]>kv){sdot_times[j+1]=sdot_times[j];j--;}sdot_times[j+1]=kv;}
        double sdot_med = sdot_times[3];
        printf("RESULT2 shape=%s out=%d in=%d M=%d sdotneon_ms=%.4f sme2_ms=%.4f sme2_faster=%d ratio=%.3f\n",
               label, out, in, M, sdot_med*1000, sme2_med*1000, sme2_med < sdot_med, sdot_med/sme2_med);
        free(xq_nat); free(xq_split); free(ascale); free(y_sdot);

        free(x); free(y_neon); free(y_sme2); free(scratch);
    }

    free(nibbles); free(scales); free(packed); free(rhs_packed);
    q4pool_destroy(&g_pool);
}

int main(int argc, char **argv) {
    int T = argc > 1 ? atoi(argv[1]) : 10;
    fprintf(stderr, "[kai_route_bench] T=%d kai_sme2_available=%d kai_sme2_min_m=%d\n",
            T, kai_sme2_available(), kai_sme2_min_m());

    bench_shape("square-4096", 4096, 4096, T);
    bench_shape("mlp-11008x2048", 11008, 2048, T);
    bench_shape("small-2048x2048", 2048, 2048, T);
    bench_shape("small-4864x896", 4864, 896, T);   // ballpark Qwen2.5-0.5B FFN-like shape

    fprintf(stderr, "[kai_route_bench] DONE\n");
    return 0;
}
