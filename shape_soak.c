// shape_soak.c -- Phase 0 "Shape Soak" (PLAN_general_purpose_loader.md).
//
// Question this answers: is kai_sme2_shape_ok()'s lack of an `out` constraint (only `in % 64
// == 0` is checked) actually safe at out values that are NOT a multiple of 64 -- a case this
// engine has literally never exercised (every N it has ever shipped: 576, 1408, 2048, 3072,
// 4096, 10944, 102400, 1536, 8960, 14336, 128256, 151936 -- all %64==0)?
//
// Method mirrors f16lhs_bench.c exactly (same isolated-correctness pattern, already validated
// this session): for each (out, in, M) triple, build random K_Q4G64 weights + random fp32
// activations, run the real vendored repack+GEMM kernel, and compare against a reference that
// applies the SAME lossy LHS rounding the kernel itself does (fp16 round for f16p-LHS,
// per-64-group symmetric int8 round for int8-LHS) before a double-precision dot product --
// isolating "does the kernel handle this shape correctly" from "how much does LHS precision
// cost" (a question already answered this session for the shapes actually shipped).
//
// Pass bar (pre-declared, per plan Phase 0): no crash/SIGILL at any shape; f16p-LHS rel_l2
// stays in the same ~1e-3 band f16lhs_bench.c already established at known-good shapes: a
// shape-specific bug would show as a spike relative to the other shapes' rel_l2, not a uniform
// small residual. Every shape is tried at 3 M values; timing recorded but M-threshold policy is
// a secondary output here, not the primary question.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "sme2_kai.h"

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec*1e-9; }

// vdsp K_Q4G64 packing (verbatim from f16lhs_bench.c): row stride in/2 bytes, byte b holds
// {low nibble -> col 2b, high nibble -> col 2b+1}, value = (nibble - 8) * scale[row, col/64].
static void pack_row_q4g64(uint8_t *packed_row, const int *nibbles, int in) {
    for (int b = 0; b < in / 2; b++) {
        packed_row[b] = (uint8_t)((nibbles[2*b] & 0xF) | ((nibbles[2*b+1] & 0xF) << 4));
    }
}

typedef struct { int out, in, M; } Shape;

// Standard symmetric per-group int8 quantization (scale = max|x|/127), applied to one group of
// 64 activations -- matches what kai_run_lhs_quant_pack_qsi8d32p_f32_neon does internally
// (group size confirmed 64 == SME2_KAI_BL via sme2_kai.h's own doc comment, not guessed).
static void quant_int8_round_group(const float *x_group, double *xq_group, int gsize) {
    float maxabs = 0.0f;
    for (int i = 0; i < gsize; i++) { float a = fabsf(x_group[i]); if (a > maxabs) maxabs = a; }
    float scale = maxabs > 0.0f ? maxabs / 127.0f : 1.0f;
    for (int i = 0; i < gsize; i++) {
        int q = (int)lrintf(x_group[i] / scale);
        if (q > 127) q = 127; if (q < -127) q = -127;
        xq_group[i] = (double)q * (double)scale;
    }
}

static int run_shape(int out, int in, int M, int use_f16lhs, int verbose) {
    int avail = use_f16lhs ? kai_sme2_f16lhs_available() : kai_sme2_available();
    if (!avail) { fprintf(stderr, "[shape_soak] %s unavailable on this host -- ABORT\n", use_f16lhs?"f16lhs":"int8lhs"); return -1; }
    int ok = use_f16lhs ? kai_sme2_f16lhs_shape_ok(out, in) : kai_sme2_shape_ok(out, in);
    if (!ok) {
        printf("SKIP variant=%s out=%d in=%d M=%d shape_ok=0\n", use_f16lhs?"f16lhs":"int8lhs", out, in, M);
        return 0;
    }
    int ng = in / 64;
    srand(20260825 + out*1000003 + in*97 + M);

    int *nibbles = malloc(sizeof(int) * (size_t)out * in);
    float *scales = malloc(sizeof(float) * (size_t)out * ng);
    for (long r = 0; r < out; r++) {
        for (int c = 0; c < in; c++) nibbles[r*(long)in+c] = rand() % 16;
        for (int g = 0; g < ng; g++) scales[r*(long)ng+g] = 0.01f + (rand() % 1000) / 5000.0f;
    }
    uint8_t *packed = malloc((size_t)out * (in/2));
    for (long r = 0; r < out; r++) pack_row_q4g64(packed + (size_t)r*(in/2), nibbles + r*(long)in, in);

    float *x = malloc(sizeof(float) * (size_t)M * in);
    for (long i = 0; i < (long)M*in; i++) x[i] = ((rand() % 2000) - 1000) / 500.0f;

    // Reference: apply the SAME lossy LHS rounding the kernel applies, then exact double dot.
    double *y_ref = malloc(sizeof(double) * (size_t)M * out);
    double *xq_scratch = use_f16lhs ? NULL : malloc(sizeof(double) * in);
    for (int m = 0; m < M; m++) {
        if (!use_f16lhs) {
            for (int g = 0; g < ng; g++) quant_int8_round_group(x + (long)m*in + g*64, xq_scratch + g*64, 64);
        }
        for (long r = 0; r < out; r++) {
            double acc = 0.0;
            for (int c = 0; c < in; c++) {
                double xv;
                if (use_f16lhs) { __fp16 xh = (__fp16)x[(long)m*in+c]; xv = (double)(float)xh; }
                else { xv = xq_scratch[c]; }
                int g = c / 64;
                double wv = (double)(nibbles[r*(long)in+c] - 8) * (double)scales[r*(long)ng+g];
                acc += xv * wv;
            }
            y_ref[(long)m*out+r] = acc;
        }
    }

    size_t rhs_bytes = use_f16lhs ? kai_sme2_f16lhs_rhs_packed_bytes(out, in) : kai_sme2_rhs_packed_bytes(out, in);
    void *rhs_packed = aligned_alloc(64, (rhs_bytes + 63) & ~(size_t)63);
    int rc = use_f16lhs
        ? kai_sme2_repack_q4g64_f16lhs(out, in, packed, scales, rhs_packed, rhs_bytes)
        : kai_sme2_repack_q4g64(out, in, packed, scales, rhs_packed, rhs_bytes);
    if (rc != 0) {
        printf("FAIL variant=%s out=%d in=%d M=%d stage=repack rc=%d\n", use_f16lhs?"f16lhs":"int8lhs", out, in, M, rc);
        free(nibbles); free(scales); free(packed); free(x); free(y_ref); free(rhs_packed); free(xq_scratch);
        return 1;
    }
    size_t scratch_bytes = use_f16lhs ? kai_sme2_f16lhs_lhs_scratch_bytes(M, in) : kai_sme2_lhs_scratch_bytes(M, in);
    void *scratch = aligned_alloc(64, (scratch_bytes + 63) & ~(size_t)63);
    float *y_test = malloc(sizeof(float) * (size_t)M * out);

    double t0 = now_s();
    if (use_f16lhs) kai_sme2_gemm_f16lhs(M, out, in, x, rhs_packed, NULL, y_test, scratch);
    else            kai_sme2_gemm_f32(M, out, in, x, rhs_packed, NULL, y_test, scratch);
    double t1 = now_s();

    double num = 0.0, den = 0.0;
    for (long i = 0; i < (long)M*out; i++) {
        double d = (double)y_test[i] - y_ref[i];
        num += d*d; den += y_ref[i]*y_ref[i];
    }
    double rel_l2 = den > 0 ? sqrt(num/den) : sqrt(num);
    printf("RESULT variant=%s out=%d in=%d M=%d N_mod64=%d rel_l2=%.6e time_ms=%.4f\n",
           use_f16lhs?"f16lhs":"int8lhs", out, in, M, out%64, rel_l2, (t1-t0)*1000.0);
    if (verbose) fflush(stdout);

    free(nibbles); free(scales); free(packed); free(x); free(y_ref); free(rhs_packed); free(scratch); free(y_test); free(xq_scratch);
    return 0;
}

int main(void) {
    fprintf(stderr, "[shape_soak] kai_sme2_available=%d kai_sme2_f16lhs_available=%d kai_sme2_min_m=%d\n",
            kai_sme2_available(), kai_sme2_f16lhs_available(), kai_sme2_min_m());

    int any_fail = 0;

    // --- Sweep 1: out (N) genericity -- THE core question. Fixed in=2048 (real shipped K),
    // M=8 (moderate). Includes never-shipped non-%64 values AND known-good baselines together
    // so the comparison is apples-to-apples in the same run.
    int n_values[] = {63, 65, 127, 129, 1000, 1408, 2048, 4096, 32000, 102400};
    int n_count = sizeof(n_values)/sizeof(n_values[0]);
    for (int v = 0; v < 2; v++) {
        int use_f16 = (v == 0);
        for (int i = 0; i < n_count; i++) {
            int rc = run_shape(n_values[i], 2048, 8, use_f16, 1);
            if (rc > 0) any_fail = 1;
        }
    }

    // --- Sweep 2: in (K) genericity spot-check -- all %64==0 by hard kernel constraint, this
    // just confirms no silent corruption across a range of K, not testing an unknown.
    int k_values[] = {64, 128, 512, 1536, 2048, 4096, 8960, 14336};
    int k_count = sizeof(k_values)/sizeof(k_values[0]);
    for (int v = 0; v < 2; v++) {
        int use_f16 = (v == 0);
        for (int i = 0; i < k_count; i++) {
            int rc = run_shape(128, k_values[i], 8, use_f16, 1);
            if (rc > 0) any_fail = 1;
        }
    }

    // --- Sweep 3: M sweep at two NEW shape families (not the already-characterized DeepSeek
    // expert shape) -- feeds the D-gen-3 M-threshold policy table. Correctness + timing both
    // recorded; NEON-comparison timing is a separate, later step (this sweep establishes SME2's
    // own absolute cost curve first).
    int m_values[] = {1, 4, 8, 15, 16, 17, 32, 64};
    int m_count = sizeof(m_values)/sizeof(m_values[0]);
    // Family A: "square-ish" attention-adjacent shape (Mistral/Llama-class hidden dim).
    // Family B: "MLP-like" shape (Qwen2.5-3B-class intermediate dim).
    int shape_families[][2] = {{4096, 4096}, {11008, 2048}};
    for (int f = 0; f < 2; f++) {
        for (int v = 0; v < 2; v++) {
            int use_f16 = (v == 0);
            for (int i = 0; i < m_count; i++) {
                int rc = run_shape(shape_families[f][0], shape_families[f][1], m_values[i], use_f16, 1);
                if (rc > 0) any_fail = 1;
            }
        }
    }

    fprintf(stderr, "[shape_soak] DONE any_fail=%d\n", any_fail);
    return any_fail;
}
