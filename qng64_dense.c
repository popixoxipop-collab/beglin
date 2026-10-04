#include "qng64_dense.h"

#include <math.h>
#include <string.h>

#define QNG64_GROUP 64

size_t qng64_group_bytes(int n) {
    if (n < 2 || n > 15 || n == 4) return 0;
    return (size_t)n * 8;
}

size_t qng64_packed_bytes(int out, int in, int n) {
    if (out <= 0 || in <= 0 || (in % QNG64_GROUP) != 0) return 0;
    size_t gb = qng64_group_bytes(n);
    return gb ? (size_t)out * (size_t)(in / QNG64_GROUP) * gb : 0;
}

float qng64_decode_code(const uint8_t *group, int n, int index) {
    if (!group || !qng64_group_bytes(n) || index < 0 || index >= QNG64_GROUP) return 0.0f;
    int u = 0;
    const int byte_in_plane = index >> 3;
    const int bit_in_byte = index & 7;
    for (int j = 0; j < n; ++j)
        if ((group[(size_t)j * 8 + byte_in_plane] >> bit_in_byte) & 1u) u |= 1 << j;
    return (float)(u - (1 << (n - 1)));
}

int qng64_quantize_f32(
    const float *src, int out, int in, int n,
    uint8_t *packed, float *scales) {
    if (!src || !packed || !scales || out <= 0 || in <= 0 ||
        (in % QNG64_GROUP) != 0 || !qng64_group_bytes(n)) return -1;
    const int ng = in / QNG64_GROUP;
    const int qmax = (1 << (n - 1)) - 1;
    const int qmin = -(1 << (n - 1));
    const int bias = 1 << (n - 1);
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
            float inv = 1.0f / scale;
            scales[(size_t)r * ng + g] = scale;
            uint8_t *dst = packed + ((size_t)r * ng + g) * gb;
            float err = 0.0f;
            for (int p = 0; p < QNG64_GROUP; ++p) {
                float v = grp[p] + err;
                long qi = lrintf(v * inv);
                if (qi > qmax) qi = qmax;
                if (qi < qmin) qi = qmin;
                err = v - (float)qi * scale;
                unsigned u = ((unsigned)((int)qi + bias)) & ((1u << n) - 1u);
                const int byte_in_plane = p >> 3;
                const int bit_in_byte = p & 7;
                for (int j = 0; j < n; ++j)
                    if ((u >> j) & 1u) dst[(size_t)j * 8 + byte_in_plane] |= (uint8_t)(1u << bit_in_byte);
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
        (in % QNG64_GROUP) != 0 || !qng64_group_bytes(n)) return;
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
