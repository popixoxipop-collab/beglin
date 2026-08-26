// gguf_quants.c -- dequantization for GGUF tensor types.
//
// VENDORED from ggml (https://github.com/ggml-org/llama.cpp, MIT License, Copyright (c)
// 2023-2026 The ggml authors), commit d83f72d463287ab9c50b4bc18ee332104a963889
// (~/llamacpp_kleidi_build on bob, 2026-08-17). Block struct layouts and dequant algorithms
// below are ported from ggml/src/ggml-common.h and ggml/src/ggml-quants.c's
// dequantize_row_q4_0 / dequantize_row_q8_0 / dequantize_row_q4_K / dequantize_row_q6_K /
// get_scale_min_k4, with types/names adapted to this project's naming (Ggml* prefix kept for
// the struct/enum names already established in gguf_load.h; dequant_row_* function bodies are
// otherwise a direct, unmodified port -- see VENDOR.md for the exact diff description).
// Full ggml-quants.c/ggml-common.h are NOT vendored -- only the pieces this project's Phase 1
// scope needs (see PLAN_general_purpose_loader.md D-gen-1's WHY for not vendoring the whole
// C++ gguf.cpp reader; the same reasoning applies here: take only what's needed, keep it in
// plain C, keep the bounds-checking discipline this project already enforces elsewhere).
//
// NOT vendored, and why: ggml's own F16<->F32 conversion (ggml_compute_fp16_to_fp32) exists to
// support compilers/platforms without native fp16 hardware support. This project's target is
// Apple clang on Apple Silicon, which has native __fp16 hardware conversion already relied on
// elsewhere in this codebase (see f16lhs_bench.c's `__fp16 xh = (__fp16)x[...]`, and
// sme2_kai.c's f16p-LHS path) -- reinterpreting the raw uint16 bits as __fp16 and letting the
// compiler emit the native conversion is simpler AND more obviously correct than porting a
// portability fallback this project doesn't need. BF16->F32 needs no vendoring either: bf16 is
// exactly the truncated upper 16 bits of an IEEE754 float32 (lossless widening, no rounding
// table, no chip-specific fact to get wrong) -- a left-shift-by-16 bit reinterpretation is the
// whole algorithm.

#include "gguf_quants.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define QK_K 256
#define K_SCALE_SIZE 12
#define QK4_0 32
#define QK8_0 32
#define QK5_0 32

typedef uint16_t ggml_half;

#pragma pack(push, 1)
typedef struct {
    ggml_half d;            // delta
    uint8_t qs[QK4_0 / 2];  // nibbles
} GgmlBlockQ4_0;

typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} GgmlBlockQ8_0;

typedef struct {
    ggml_half d;            // delta
    uint8_t qh[4];          // 5th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} GgmlBlockQ5_0;

typedef struct {
    ggml_half d;                   // super-block scale for quantized scales
    ggml_half dmin;                // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE];  // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K / 2];          // 4-bit quants
} GgmlBlockQ4_K;

typedef struct {
    uint8_t ql[QK_K / 2];      // quants, lower 4 bits
    uint8_t qh[QK_K / 4];      // quants, upper 2 bits
    int8_t  scales[QK_K / 16]; // scales, quantized with 8 bits
    ggml_half d;                // super-block scale
} GgmlBlockQ6_K;
#pragma pack(pop)

static float fp16_to_fp32(ggml_half raw) {
    union { uint16_t u; __fp16 f; } cvt;
    cvt.u = raw;
    return (float)cvt.f;
}
static float bf16_to_fp32(uint16_t raw) {
    union { uint32_t u; float f; } cvt;
    cvt.u = ((uint32_t)raw) << 16;
    return cvt.f;
}

// Ported verbatim from ggml-quants.c's get_scale_min_k4 (see file header for provenance).
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Ported from ggml-quants.c's dequantize_row_q4_0 (see file header for provenance).
static void dequant_row_q4_0(const void *src, float *y, int64_t n) {
    const GgmlBlockQ4_0 *x = (const GgmlBlockQ4_0 *)src;
    const int nb = (int)(n / QK4_0);
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK4_0/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;
            y[i*QK4_0 + j + 0]        = x0*d;
            y[i*QK4_0 + j + QK4_0/2]  = x1*d;
        }
    }
}

// Ported from ggml-quants.c's dequantize_row_q8_0 (see file header for provenance).
static void dequant_row_q8_0(const void *src, float *y, int64_t n) {
    const GgmlBlockQ8_0 *x = (const GgmlBlockQ8_0 *)src;
    const int nb = (int)(n / QK8_0);
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; ++j) y[i*QK8_0 + j] = x[i].qs[j]*d;
    }
}

// Ported from ggml-quants.c's dequantize_row_q5_0 (see file header for provenance). Found
// needed 2026-08-26 (Phase 3 sub-step 1, Qwen2.5-0.5B-Instruct GGUF): small models' Q4_K_M
// recipe drops several tensor kinds (ffn_gate/ffn_up/token_embd) to Q5_0 rather than Q4_K --
// a diagnostic gap, not a math bug, per the general-purpose-loader plan's own Phase 3
// expectation.
static void dequant_row_q5_0(const void *src, float *y, int64_t n) {
    const GgmlBlockQ5_0 *x = (const GgmlBlockQ5_0 *)src;
    const int nb = (int)(n / QK5_0);
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));
        for (int j = 0; j < QK5_0/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;
            y[i*QK5_0 + j + 0]         = x0*d;
            y[i*QK5_0 + j + QK5_0/2]   = x1*d;
        }
    }
}

// Ported from ggml-quants.c's dequantize_row_q4_K (see file header for provenance).
static void dequant_row_q4_k(const void *src, float *y, int64_t n) {
    const GgmlBlockQ4_K *x = (const GgmlBlockQ4_K *)src;
    const int nb = (int)(n / QK_K);
    for (int i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        const float d   = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

// Ported from ggml-quants.c's dequantize_row_q6_K (see file header for provenance).
static void dequant_row_q6_k(const void *src, float *y, int64_t n) {
    const GgmlBlockQ6_K *x = (const GgmlBlockQ6_K *)src;
    const int64_t nb = n / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;
        for (int nn = 0; nn < QK_K; nn += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

static void dequant_row_f16(const void *src, float *y, int64_t n) {
    const uint16_t *x = (const uint16_t *)src;
    for (int64_t i = 0; i < n; i++) y[i] = fp16_to_fp32(x[i]);
}
static void dequant_row_bf16(const void *src, float *y, int64_t n) {
    const uint16_t *x = (const uint16_t *)src;
    for (int64_t i = 0; i < n; i++) y[i] = bf16_to_fp32(x[i]);
}
static void dequant_row_f32(const void *src, float *y, int64_t n) {
    memcpy(y, src, (size_t)n * sizeof(float));
}

int gguf_dequant_supported(GgmlType type) {
    switch (type) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0: case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q6_K:
            return 1;
        default:
            return 0;
    }
}

void gguf_dequant_row(GgmlType type, const void *src, float *dst, int64_t n_elements) {
    switch (type) {
        case GGML_TYPE_F32:  dequant_row_f32(src, dst, n_elements); return;
        case GGML_TYPE_F16:  dequant_row_f16(src, dst, n_elements); return;
        case GGML_TYPE_BF16: dequant_row_bf16(src, dst, n_elements); return;
        case GGML_TYPE_Q4_0: dequant_row_q4_0(src, dst, n_elements); return;
        case GGML_TYPE_Q8_0: dequant_row_q8_0(src, dst, n_elements); return;
        case GGML_TYPE_Q5_0: dequant_row_q5_0(src, dst, n_elements); return;
        case GGML_TYPE_Q4_K: dequant_row_q4_k(src, dst, n_elements); return;
        case GGML_TYPE_Q6_K: dequant_row_q6_k(src, dst, n_elements); return;
        default:
            fprintf(stderr, "FATAL: gguf_dequant_row: unsupported ggml type id %d (see gguf_dequant_supported())\n", (int)type);
            exit(1);
    }
}
