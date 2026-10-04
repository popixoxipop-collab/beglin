#include "qng64_dense.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int closef(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

int main(void) {
    const int out = 3, in = 64, n = 5;
    float src[out * in], x[in], y[out], ref[out];
    for (int i = 0; i < out * in; ++i) src[i] = ((i * 37) % 101 - 50) / 23.0f;
    for (int i = 0; i < in; ++i) x[i] = ((i * 13) % 31 - 15) / 19.0f;
    size_t bytes = qng64_packed_bytes(out, in, n);
    uint8_t *packed = calloc(bytes, 1);
    float *scales = calloc((size_t)out, sizeof(float));
    if (!packed || !scales) return 2;
    if (qng64_quantize_f32(src, out, in, n, packed, scales) != 0) return 3;
    qng64_matvec_f32(packed, scales, n, x, NULL, y, out, in);
    size_t gb = qng64_group_bytes(n);
    if (gb != 40) return 4;
    for (int r = 0; r < out; ++r) {
        double sum = 0.0;
        for (int p = 0; p < in; ++p) {
            float q = qng64_decode_code(packed + (size_t)r * gb, n, p);
            sum += (double)(q * scales[r]) * x[p];
        }
        ref[r] = (float)sum;
        if (!closef(y[r], ref[r], 1e-5f)) return 5;
    }
    printf("PASS dense qNg64 n=%d group_bytes=%zu packed_bytes=%zu\n", n, gb, bytes);
    free(packed); free(scales);
    return 0;
}
