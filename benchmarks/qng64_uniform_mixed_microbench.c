/* BEGLIN qNg64 actual CPU GEMV microbenchmark, synthetic tensors ONLY.
 * Reuses native qng64_dense.c quantizer and bitplane matvec, no Python proxy.
 * NOT an LLM token/s, BF16 logits, Metal GPU, or live-memory benchmark.
 * Build: cc -std=c11 -O3 -Wall -Wextra -Werror -I. qng64_dense.c
 *        benchmarks/qng64_uniform_mixed_microbench.c -lm -o /tmp/qngbench
 */
#define _POSIX_C_SOURCE 200809L
#include "qng64_dense.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { NROLE = 6, NOUT = 64, NIN = 512, GROUP = 64,
       WARMUP = 8, REPEATS = 3, ITERATIONS = 120 };
static volatile double sink_value;

typedef struct {
    int n;
    float *weights;
    uint8_t *packed;
    float *scales;
    float *outputs;
    size_t code_bytes;
    size_t group_count;
} Tensor;

static double monotonic_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) {
        perror("clock_gettime");
        exit(2);
    }
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void fatal(const char *message) {
    fprintf(stderr, "BEGLIN_QNG64_BENCH_FAIL: %s\n", message);
    exit(2);
}

static void init_arm(Tensor *t, const int widths[NROLE]) {
    for (int role = 0; role < NROLE; ++role) {
        t[role].n = widths[role];
        t[role].group_count = NOUT * (NIN / GROUP);
        const size_t count = NOUT * NIN;
        t[role].code_bytes = qng64_packed_bytes(NOUT, NIN, widths[role]);
        if (!t[role].code_bytes) fatal("unsupported native qNg64 width");
        t[role].weights = calloc(count, sizeof(float));
        t[role].packed = calloc(t[role].code_bytes, 1);
        t[role].scales = calloc(t[role].group_count, sizeof(float));
        t[role].outputs = calloc(NOUT, sizeof(float));
        if (!t[role].weights || !t[role].packed || !t[role].scales ||
            !t[role].outputs) fatal("allocation failed");

        for (int i = 0; i < NOUT * NIN; ++i)
            t[role].weights[i] =
                0.065f * sinf((float)(i + 7 * role) * 0.0132f) +
                0.026f * cosf((float)(i + 3 * role) * 0.0729f);
        if (qng64_quantize_f32(t[role].weights, NOUT, NIN, t[role].n,
                               t[role].packed, t[role].scales) != 0)
            fatal("native qng64 quantization failed");
    }
}

static void free_arm(Tensor *t) {
    for (int r = 0; r < NROLE; ++r) {
        free(t[r].weights);
        free(t[r].packed);
        free(t[r].scales);
        free(t[r].outputs);
    }
}

static void one_sweep(Tensor *t, const float x[NIN]) {
    for (int role = 0; role < NROLE; ++role)
        qng64_matvec_f32(t[role].packed, t[role].scales, t[role].n,
                          x, NULL, t[role].outputs, NOUT, NIN);
    sink_value += t[0].outputs[0] + t[NROLE - 1].outputs[NOUT - 1];
}

static double matvec_error(const Tensor *t, const float x[NIN]) {
    double diff_sq = 0.0, ref_sq = 0.0;
    for (int role = 0; role < NROLE; ++role) {
        for (int row = 0; row < NOUT; ++row) {
            double dense = 0.0;
            for (int i = 0; i < NIN; ++i)
                dense += (double)t[role].weights[row * NIN + i] * x[i];
            const double diff = dense - t[role].outputs[row];
            diff_sq += diff * diff;
            ref_sq += dense * dense;
        }
    }
    return ref_sq ? sqrt(diff_sq / ref_sq) : 0.0;
}

static double median_three(double a, double b, double c) {
    if (a > b) { double tmp = a; a = b; b = tmp; }
    if (b > c) { double tmp = b; b = c; c = tmp; }
    if (a > b) { double tmp = a; a = b; b = tmp; }
    return b;
}

static size_t run_arm(const char *name, const int widths[NROLE],
                      const float x[NIN], double *median_sps, double *error) {
    Tensor t[NROLE] = {{0}};
    init_arm(t, widths);
    size_t total = 0;
    for (int role = 0; role < NROLE; ++role)
        total += t[role].code_bytes + t[role].group_count * sizeof(float);

    double durations[REPEATS];
    for (int rep = 0; rep < REPEATS; ++rep) {
        for (int warm = 0; warm < WARMUP; ++warm) one_sweep(t, x);
        const double begin = monotonic_seconds();
        for (int iter = 0; iter < ITERATIONS; ++iter) one_sweep(t, x);
        const double duration = monotonic_seconds() - begin;
        if (!(duration > 0) || !isfinite(duration))
            fatal("invalid monotonic measurement");
        durations[rep] = ITERATIONS / duration;
    }
    *median_sps = median_three(durations[0], durations[1], durations[2]);
    *error = matvec_error(t, x);
    if (!isfinite(*error) || !isfinite(*median_sps)) fatal("nonfinite metrics");

    printf("{\"scope\":\"SYNTHETIC_NATIVE_CPU_QNG64_GEMV_ONLY\","
           "\"arm\":\"%s\",\"widths\":[%d,%d,%d,%d,%d,%d],"
           "\"serialized_weight_bytes\":%zu,\"dense_fp32_output_rel_l2\":%.9g,"
           "\"median_sweeps_per_second\":%.9g,\"gemvs_per_sweep\":%d,"
           "\"measure_iters\":%d,\"repeats\":%d}\n",
           name, widths[0], widths[1], widths[2], widths[3], widths[4],
           widths[5], total, *error, *median_sps, NROLE, ITERATIONS, REPEATS);
    free_arm(t);
    return total;
}

int main(int argc, char **argv) {
    const int uniform[NROLE] = {7, 7, 7, 7, 7, 7};
    int candidate[NROLE] = {7, 7, 6, 9, 7, 6};
    const char *label = "mixed_7_7_6_9_7_6";
    if (argc != 1 && argc != NROLE + 1)
        fatal("usage: microbench [n0 n1 n2 n3 n4 n5]");
    if (argc == NROLE + 1) {
        label = "native_output_calibrated";
        for (int i = 0; i < NROLE; ++i) {
            char *end = NULL;
            const long n = strtol(argv[i + 1], &end, 10);
            if (end == argv[i + 1] || *end || n < 2 || n > 15 || n == 4)
                fatal("invalid qNg64 width argument; q4g64 is separate");
            candidate[i] = (int)n;
        }
    }
    float x[NIN];
    for (int i = 0; i < NIN; ++i)
        x[i] = 0.08f * sinf((float)i * 0.0452f) +
               0.04f * cosf((float)i * 0.0119f);

    double sps_a, sps_b, err_a, err_b;
    const size_t bytes_a = run_arm("uniform_n7", uniform, x, &sps_a, &err_a);
    const size_t bytes_b = run_arm(label, candidate, x, &sps_b, &err_b);
    if (argc == 1 && bytes_a != bytes_b)
        fatal("default mixed policy must use same weight budget");
    if (bytes_b > bytes_a)
        fatal("selected policy exceeds uniform 7-bit weight budget");
    if (bytes_a != (size_t)184320)
        fatal("wrong native code+scale accounting");
    if (!(err_a > 0.0) || !(sps_a > 0.0) || !(sps_b > 0.0))
        fatal("invalid denominator");
    printf("{\"scope\":\"SYNTHETIC_NATIVE_CPU_QNG64_GEMV_ONLY\","
           "\"uniform_and_mixed_weight_bytes_equal\":%s,"
           "\"selected_within_uniform_budget\":true,"
           "\"selected_to_uniform_output_L2_ratio\":%.10g,"
           "\"selected_to_uniform_GEMV_throughput_ratio\":%.10g,"
           "\"reported_bf16_bytes\":%d,\"actual_native_cpu_test_only\":true,"
           "\"not_llm_tokens_per_second\":true,\"not_metal_gpu\":true,"
           "\"not_model_quality_certified\":true,"
           "\"automatic_live_promotion\":false}\n",
           bytes_a == bytes_b ? "true" : "false", err_b / err_a,
           sps_b / sps_a, NROLE * NOUT * NIN * 2);
    return 0;
}
