#ifndef BEGLIN_QNG64_DENSE_H
#define BEGLIN_QNG64_DENSE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t qng64_group_bytes(int n);
size_t qng64_packed_bytes(int out, int in, int n);

int qng64_quantize_f32(
    const float *src, int out, int in, int n,
    uint8_t *packed, float *scales);

float qng64_decode_code(const uint8_t *group, int n, int index);

void qng64_matmul_f32(
    const uint8_t *packed, const float *scales, int n,
    const float *x, const float *bias, float *y,
    int out, int in, int M);

void qng64_matvec_f32(
    const uint8_t *packed, const float *scales, int n,
    const float *x, const float *bias, float *y,
    int out, int in);

#ifdef __cplusplus
}
#endif

#endif
