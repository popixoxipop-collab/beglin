#ifndef BEGLIN_QNG64_DENSE_H
#define BEGLIN_QNG64_DENSE_H

#include <stddef.h>
#include <stdint.h>

// Dense arbitrary-n group-64 symmetric RTN reference primitive.
// Layout is the canonical production A2 qNg64 format: n bit-planes of 8 bytes each
// per group, little-endian element bits, BIASED signed codes, plus one fp32 scale.
// Quantization uses the same per-group error-feedback recurrence as gguf_quantize_qNg64().
size_t qng64_group_bytes(int n);
size_t qng64_packed_bytes(int out, int in, int n);

int qng64_quantize_f32(
    const float *src, int out, int in, int n,
    uint8_t *packed, float *scales);

float qng64_decode_code(const uint8_t *group, int n, int index);

void qng64_matmul_f32(\n    const uint8_t *packed, const float *scales, int n,\n    const float *x, const float *bias, float *y,\n    int out, int in, int M);\n\nvoid qng64_matvec_f32(
    const uint8_t *packed, const float *scales, int n,
    const float *x, const float *bias, float *y,
    int out, int in);

#endif
