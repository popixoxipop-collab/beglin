// kai_test_correct.c
// Corrected re-verification: uses the ACTUAL ARM-documented pack/kernel pairing for
// kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa, confirmed via
// ARM's own official test file (test/tests/matmul_clamp_f32_qsi8d32p_qsi4c32p_test.cpp):
//   LHS pack: kai_run_lhs_quant_pack_qsi8d32p_f32_neon
//   RHS pack: kai_run_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon
// (This project's earlier prototype used kai_lhs_quant_pack_qsi8d32p_f32 (no _neon) and
// kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0 -- a DIFFERENT, incompatible pack pair for this
// specific kernel. That mismatch, not an ARM kernel bug, is the root cause chased across
// this whole investigation.)
//
// Build: clang -march=armv9-a+sve2+sme2 -O2 -w -I. -o kai_test_correct kai_test_correct.c \
//   kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa.c \
//   kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa_asm.S \
//   kai_lhs_quant_pack_qsi8d32p_f32_neon.c kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon.c \
//   kai_common_sme_asm.S -framework Accelerate -lm

#include "kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa.h"
#include "kai_lhs_quant_pack_qsi8d32p_f32_neon.h"
#include "kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon.h"
#include "kai/kai_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

static uint64_t rng = 0x243F6A8885A308D3ULL;
static uint32_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 32); }

static uint16_t f32_to_f16_bits(float f) {
    __fp16 h = (__fp16)f;
    uint16_t b;
    memcpy(&b, &h, 2);
    return b;
}
static float f16_bits_to_f32(uint16_t b) {
    __fp16 h;
    memcpy(&h, &b, 2);
    return (float)h;
}

int main(void) {
    const int out = 64, in = 64, M = 16, bl = 64;
    const size_t num_blocks = in / bl;  // 1

    // Real weight codes (-8..7) and per-(row,block) scale, random.
    int8_t *code = malloc((size_t)out * in);
    float *wscale = malloc((size_t)out * num_blocks * sizeof(float));
    for (size_t i = 0; i < (size_t)out * in; i++) code[i] = (int8_t)((int)(xr() % 16) - 8);
    for (size_t i = 0; i < (size_t)out * num_blocks; i++) wscale[i] = 0.001f + 0.02f * ((float)(xr() & 0xFFFF) / 65535.0f);

    float *bias = calloc((size_t)out, sizeof(float));

    // Real activation, random.
    float *act = malloc((size_t)M * in * sizeof(float));
    for (size_t i = 0; i < (size_t)M * in; i++) act[i] = (float)((int)(xr() % 256) - 128) / 32.0f;

    // ---- Build the RHS "unpacked" input buffer exactly per ARM's own test fixture
    //      (make_s4s0_rhs_with_scales): per row, per block: [fp16 scale][bl/2 bytes],
    //      byte[idx] = (high<<4)|low, low=nibble(K=idx), high=nibble(K=idx+bl/2),
    //      nibble(k) = (code[row,k]+8)&0xF (same qsu4 offset-by-8 convention as D27's
    //      pack_nibbles -- confirmed correct here since kai_get_qsu4 in ARM's fixture is
    //      just a raw 0..15 value, and the packer's internal XOR-0x88 converts it to
    //      genuine signed two's complement).
    size_t num_bytes_per_block = (size_t)(bl / 2) + sizeof(uint16_t);
    size_t rhs_stride = num_blocks * num_bytes_per_block;
    uint8_t *rhs_unpacked = malloc((size_t)out * rhs_stride);
    for (int row = 0; row < out; row++) {
        for (size_t b = 0; b < num_blocks; b++) {
            uint8_t *block = rhs_unpacked + (size_t)row * rhs_stride + b * num_bytes_per_block;
            uint16_t sbits = f32_to_f16_bits(wscale[(size_t)row * num_blocks + b]);
            memcpy(block, &sbits, 2);
            uint8_t *values = block + 2;
            int k0 = (int)b * bl;
            for (int idx = 0; idx < bl / 2; idx++) {
                int8_t c_low = code[(size_t)row * in + k0 + idx];
                int8_t c_high = code[(size_t)row * in + k0 + idx + bl / 2];
                uint8_t low = (uint8_t)((c_low + 8) & 0xF);
                uint8_t high = (uint8_t)((c_high + 8) & 0xF);
                values[idx] = (uint8_t)((high << 4) | low);
            }
        }
    }

    // ---- Kernel/pack shape parameters (unchanged -- same kernel, only pack fn differs).
    size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    size_t kr = kai_get_kr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    size_t sr = kai_get_sr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    printf("mr=%zu nr=%zu kr=%zu sr=%zu\n", mr, nr, kr, sr);

    size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon(out, in, nr, kr, bl);
    void *rhs_packed = calloc(1, rhs_packed_size);
    struct kai_rhs_pack_qs4cxs1s0_param rhs_params = { .lhs_zero_point = 1, .rhs_zero_point = 8 };
    kai_run_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon(
        1, out, in, nr, kr, sr, bl, rhs_unpacked, NULL, rhs_packed, 0, &rhs_params);

    size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32_neon(M, in, bl, mr, kr, sr);
    void *lhs_packed = calloc(1, lhs_packed_size);
    kai_run_lhs_quant_pack_qsi8d32p_f32_neon(M, in, bl, mr, kr, sr, 0, act, in * sizeof(float), lhs_packed);

    float *y_kai = calloc((size_t)out * M, sizeof(float));
    kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(
        M, out, in, bl, lhs_packed, rhs_packed, y_kai, out * sizeof(float), sizeof(float), -FLT_MAX, FLT_MAX);
    for (int m = 0; m < M; m++)
        for (int r = 0; r < out; r++) y_kai[(size_t)m * out + r] += bias[r];

    // ---- Direct scalar double-precision reference computed straight from code/wscale/act.
    double worst_rel = 0.0, worst_abs = 0.0, typical = 0.0;
    int any_nan = 0;
    float *y_ref = malloc((size_t)out * M * sizeof(float));
    for (int m = 0; m < M; m++) {
        for (int r = 0; r < out; r++) {
            double acc = 0.0;
            for (int k = 0; k < in; k++) {
                acc += (double)code[(size_t)r * in + k] * (double)act[(size_t)m * in + k];
            }
            double y = acc * (double)wscale[r] + (double)bias[r];
            y_ref[(size_t)m * out + r] = (float)y;
        }
    }
    for (int i = 0; i < out * M; i++) typical += fabs(y_ref[i]);
    typical /= (out * M);
    int worst_abs_i = -1, worst_rel_i = -1;
    for (int i = 0; i < out * M; i++) {
        if (isnan(y_kai[i])) any_nan = 1;
        double a = fabs((double)y_kai[i] - (double)y_ref[i]);
        double rel = a / (fabs((double)y_ref[i]) + 1e-9);
        if (isnan(rel)) continue;
        if (a > worst_abs) { worst_abs = a; worst_abs_i = i; }
        if (rel > worst_rel) { worst_rel = rel; worst_rel_i = i; }
    }
    printf("first 8 y_ref: "); for (int i = 0; i < 8; i++) printf("%.4g ", y_ref[i]); printf("\n");
    printf("first 8 y_kai: "); for (int i = 0; i < 8; i++) printf("%.4g ", y_kai[i]); printf("\n");
    printf("worst_abs at i=%d: ref=%.6g kai=%.6g\n", worst_abs_i, y_ref[worst_abs_i], y_kai[worst_abs_i]);
    printf("worst_rel at i=%d: ref=%.6g kai=%.6g\n", worst_rel_i, y_ref[worst_rel_i], y_kai[worst_rel_i]);
    printf("any_nan=%d worst_rel=%.3e worst_abs=%.3e typical=%.3e worst_abs/typical=%.3e -> %s\n",
           any_nan, worst_rel, worst_abs, typical, worst_abs / (typical + 1e-9),
           (!any_nan && worst_abs / (typical + 1e-9) < 5e-2) ? "PASS(loose)" : "FAIL");
    return 0;
}
