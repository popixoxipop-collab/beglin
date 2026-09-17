// gguf_write_quants.c -- see gguf_write_quants.h. Own translation unit, same reason as every
// other file in this project's GGUF surface.

#include "gguf_write_quants.h"
#include <math.h>
#include <string.h>

#define QK4_0 32
#define QK8_0 32
typedef uint16_t ggml_half;

#pragma pack(push, 1)
typedef struct { ggml_half d; uint8_t qs[QK4_0 / 2]; } WBlockQ4_0;
typedef struct { ggml_half d; int8_t  qs[QK8_0];     } WBlockQ8_0;
#pragma pack(pop)

// Same __fp16-based conversion convention gguf_quants.c's own fp16_to_fp32() uses (compiler/
// hardware-native IEEE754 half conversion, not a hand-rolled bit-twiddling reimplementation).
static ggml_half fp32_to_fp16(float v) {
    union { uint16_t u; __fp16 f; } cvt;
    cvt.f = (__fp16)v;
    return cvt.u;
}

size_t gguf_w_q4_0_nbytes(int64_t n) { return (size_t)(n / QK4_0) * sizeof(WBlockQ4_0); }
size_t gguf_w_q8_0_nbytes(int64_t n) { return (size_t)(n / QK8_0) * sizeof(WBlockQ8_0); }

// Ported from gguf_transcode.c's gguf_quantize_q4g64_error_feedback() -- same error-feedback
// diffusion (D-export-2's own reasoning: the GGUF container doesn't care how codes were
// derived), re-blocked to GGUF's real 32-element block with an inline fp16 scale, and
// re-packed split-half (byte j's low nibble = element j, high nibble = element j+16) instead
// of q4g64's own consecutive-pair layout -- a real, deliberate difference from that function,
// matching real GGML_TYPE_Q4_0's own dequant convention (gguf_quants.c's dequant_row_q4_0()).
void gguf_w_quantize_q4_0(const float *w, int64_t n, uint8_t *out) {
    int64_t nb = n / QK4_0;
    WBlockQ4_0 *blocks = (WBlockQ4_0 *)out;
    for (int64_t b = 0; b < nb; b++) {
        const float *grp = w + b * QK4_0;
        float maxabs = 0.0f;
        for (int p = 0; p < QK4_0; p++) { float a = fabsf(grp[p]); if (a > maxabs) maxabs = a; }
        float scale = maxabs / 7.0f;
        if (scale < 1e-12f) scale = 1.0f;
        blocks[b].d = fp32_to_fp16(scale);
        float inv = 1.0f / scale;
        float err = 0.0f;
        int8_t codes[QK4_0];
        for (int p = 0; p < QK4_0; p++) {
            float x = grp[p] + err;
            float qf = rintf(x * inv);   // round-half-to-even, matches q4g64's own rintf() choice
            if (qf > 7.0f) qf = 7.0f;
            if (qf < -8.0f) qf = -8.0f;
            float deq = qf * scale;
            err = x - deq;
            codes[p] = (int8_t)qf;
        }
        for (int j = 0; j < QK4_0 / 2; j++) {
            uint8_t lo = (uint8_t)((codes[j] + 8) & 0x0F);
            uint8_t hi = (uint8_t)((codes[j + QK4_0/2] + 8) & 0x0F);
            blocks[b].qs[j] = (uint8_t)(lo | (hi << 4));
        }
    }
}

// Ported from gguf_transcode.c's gguf_quantize_q8g64() -- same plain-RTN direct-division
// (no error feedback, matching that function's own established choice), re-blocked to GGUF's
// real 32-element block with an inline fp16 scale instead of q8g64's out-of-band fp32 array.
void gguf_w_quantize_q8_0(const float *w, int64_t n, uint8_t *out) {
    int64_t nb = n / QK8_0;
    WBlockQ8_0 *blocks = (WBlockQ8_0 *)out;
    for (int64_t b = 0; b < nb; b++) {
        const float *grp = w + b * QK8_0;
        float maxabs = 0.0f;
        for (int p = 0; p < QK8_0; p++) { float a = fabsf(grp[p]); if (a > maxabs) maxabs = a; }
        float scale = maxabs / 127.0f;
        if (scale < 1e-12f) scale = 1.0f;
        blocks[b].d = fp32_to_fp16(scale);
        for (int p = 0; p < QK8_0; p++) {
            float qf = rintf(grp[p] / scale);   // direct division, not reciprocal -- matches q8g64's own R4-oracle-found requirement
            if (qf > 127.0f) qf = 127.0f;
            if (qf < -127.0f) qf = -127.0f;
            blocks[b].qs[p] = (int8_t)qf;
        }
    }
}
