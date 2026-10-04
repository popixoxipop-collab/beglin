#include "qng64_dense.h"

#include <math.h>
#include <string.h>

#define QNG64_GROUP 64

size_t qng64_group_bytes(int n) {
    if (n < 2 || n > 16) return 0;
    return (size_t)(QNG64_GROUP * n + 7) / 8;
}

size_t qng64_packed_bytes(int out, int in, int n) {
    if (out <= 0 || in <= 0 || (in % QNG64_GROUP) != 0) return 0;
    size_t gb = qng64_group_bytes(n);
    return gb ? (size_t)out * (size_t)(in / QNG64_GROUP) * gb : 0;
}

static void put_bits(uint8_t *dst, size_t bitoff, unsigned n, uint32_t value) {
    for (unsigned b = 0; b < n; ++b) {
        size_t p = bitoff + b;
        uint8_t mask = (uint8_t)(1u << (p & 7u));
        if ((value >> b) & 1u) dst[p >> 3] |= mask;
        else dst[p >> 3] &= (uint8_t)~mask;
    }
}

static uint32_t get_bits(const uint8_t *src, size_t bitoff, unsigned n) {
    uint32_t v = 0;
    for (unsigned b = 0; b < n; ++b) {
        size_t p = bitoff + b;
        v |= (uint32_t)((src[p >> 3] >> (p & 7u)) & 1u) << b;
    }
    return v;
}

float qng64_decode_code(const uint8_t *group, int n, int index) {
    if (!group || n < 2 || n > 16 || index < 0 || index >= QNG64_GROUP) return 0.0f;
    uint32_t raw = get_bits(group, (size_t)index * (unsigned)n, (unsigned)n);
    uint32_t sign = 1u << (n - 1);
    int32_t q = (raw & sign) ? (int32_t)(raw | (~0u << n)) : (int32_t)raw;
    return (float)q;
}

int qng64_quantize_f32(
    const float *src, int out, int in, int n,
    uint8_t *packed, float *scales) {
    if (!src || !packed || !scales || out <= 0 || in <= 0 ||
        (in % QNG64_GROUP) != 0 || n < 2 || n > 16) return -1;
    const int ng = in / QNG64_GROUP;
    const int qmax = (1 << (n - 1)) - 1;
    const int qmin = -(1 << (n - 1));
    const size_t gb = qng64_group_bytes(n);
    memset(packed, 0, qng64_packed_bytes(out, in, n));
    for (int r = 0; r < out; ++r) {
        for (int g = 0; g < ng; ++g) {
            const float *grp = src + (size_t)r * in + g * QNG64_GROUP;
            float maxabs = 0.0f;
            for (int p = 0; p < QNG64_GROUP; ++p) {
                float a = fabsf(grp[p]);
                if (a > maxabs) maxabs = a;
            }
            float scale = maxabs > 1e-12f ? maxabs / (float)qmax : 1.0f;
            scales[(size_t)r * ng + g] = scale;
            uint8_t *dst = packed + ((size_t)r * ng + g) * gb;
            for (int p = 0; p < QNG64_GROUP; ++p) {
                long qi = lrintf(grp[p] / scale);
                if (qi > qmax) qi = qmax;
                if (qi < qmin) qi = qmin;
                uint32_t raw = (uint32_t)((int32_t)qi) & ((1u << n) - 1u);
                put_bits(dst, (size_t)p * n, (unsigned)n, raw);
            }
        }
    }
    return 0;
}

void qng64_matvec_f32(
    const uint8_t *packed, const float *scales, int n,
    const float *x, const float *bias, float *y,
    int out, int in) {
    if (!packed || !scales || !x || !y || out <= 0 || in <= 0 ||
        (in % QNG64_GROUP) != 0 || n < 2 || n > 16) return;
    const int ng = in / QNG64_GROUP;
    const size_t gb = qng64_group_bytes(n);
    for (int r = 0; r < out; ++r) {
        double sum = bias ? bias[r] : 0.0;
        for (int g = 0; g < ng; ++g) {
            const uint8_t *grp = packed + ((size_t)r * ng + g) * gb;
            float scale = scales[(size_t)r * ng + g];
            const float *xg = x + g * QNG64_GROUP;
            for (int p = 0; p < QNG64_GROUP; ++p)
                sum += (double)(qng64_decode_code(grp, n, p) * scale) * xg[p];
        }
        y[r] = (float)sum;
    }
}
