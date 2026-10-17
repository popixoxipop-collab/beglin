/* Cross-platform smoke test for BEGLIN's real qng64_dense.c CPU encoder/GEMV.
 * Uses synthetic deterministic weights ONLY; no checkpoint or serving state.
 * Compile: cc -std=c11 -O2 -Wall -Wextra -Werror -I. \
 *    qng64_dense.c tests/test_qng64_dense_contract.c -lm -o qng64_dense_contract
 */
#include "qng64_dense.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GROUP 64
#define INPUT 128
#define OUTPUT 2
#define MAX_BYTES (OUTPUT * (INPUT / GROUP) * 15 * 8)

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        return 1; \
    } \
} while (0)

static int reference_code(const unsigned char *group, int n, int index) {
    int u = 0;
    for (int j = 0; j < n; ++j) {
        const unsigned char byte = group[(size_t)j * 8u + ((unsigned)index >> 3)];
        if (byte & (1u << (index & 7))) u |= 1 << j;
    }
    return u - (1 << (n - 1));
}

static float abs_error(const float *src, const unsigned char *packed,
                       const float *scales, int n) {
    const size_t group_bytes = qng64_group_bytes(n);
    double sum_sq = 0.0, ref_sq = 0.0;
    for (int row = 0; row < OUTPUT; ++row) {
        for (int g = 0; g < INPUT / GROUP; ++g) {
            const size_t group_index = (size_t)row * (INPUT / GROUP) + g;
            const unsigned char *p = packed + group_index * group_bytes;
            for (int i = 0; i < GROUP; ++i) {
                const float original = src[row * INPUT + g * GROUP + i];
                const float reconstructed = (float)reference_code(p, n, i) * scales[group_index];
                const double diff = (double)original - reconstructed;
                sum_sq += diff * diff;
                ref_sq += (double)original * original;
            }
        }
    }
    return ref_sq > 0 ? (float)sqrt(sum_sq / ref_sq) : 0.f;
}

int main(void) {
    float weights[OUTPUT * INPUT], x[2 * INPUT], bias[OUTPUT] = {0.125f, -0.15f};
    float scales[OUTPUT * (INPUT / GROUP)], y[OUTPUT], z[2 * OUTPUT];
    unsigned char packed[MAX_BYTES];
    float err3 = NAN, err15 = NAN;
    int tested = 0;

    for (int i = 0; i < OUTPUT * INPUT; ++i)
        weights[i] = sinf((float)i * 0.097f) * 0.14f + cosf((float)i * 0.017f) * 0.06f;
    for (int i = 0; i < 2 * INPUT; ++i)
        x[i] = cosf((float)i * 0.021f) * 0.17f;

    /* n=4 is deliberately forbidden for qNg64: use separate q4g64. */
    CHECK(qng64_group_bytes(4) == 0);
    CHECK(qng64_packed_bytes(OUTPUT, INPUT, 4) == 0);
    CHECK(qng64_quantize_f32(weights, OUTPUT, INPUT, 4, packed, scales) == -1);
    CHECK(qng64_group_bytes(1) == 0 && qng64_group_bytes(16) == 0);
    CHECK(qng64_packed_bytes(1, 63, 7) == 0);
    CHECK(qng64_packed_bytes(OUTPUT, INPUT, 7) == 4u * 7u * 8u);

    for (int n = 2; n <= 15; ++n) {
        if (n == 4) continue;
        const size_t code_bytes = qng64_packed_bytes(OUTPUT, INPUT, n);
        CHECK(code_bytes == (size_t)(OUTPUT * INPUT / GROUP) * (size_t)(n * 8));
        CHECK(code_bytes <= sizeof packed);
        memset(scales, 0, sizeof scales);
        memset(packed, 0x55, sizeof packed);
        CHECK(qng64_quantize_f32(weights, OUTPUT, INPUT, n, packed, scales) == 0);
        for (int g = 0; g < OUTPUT * INPUT / GROUP; ++g) {
            CHECK(isfinite(scales[g]) && scales[g] > 0.f);
            const unsigned char *group = packed + (size_t)g * (size_t)(n * 8);
            for (int pos = 0; pos < GROUP; ++pos) {
                const int actual = (int)qng64_decode_code(group, n, pos);
                CHECK(actual == reference_code(group, n, pos));
            }
        }
        /* Byte count INCLUDING the separate one-float32-per-group scales. */
        CHECK(code_bytes + sizeof scales == (size_t)(OUTPUT * INPUT / GROUP) * (size_t)(n * 8 + 4));

        qng64_matvec_f32(packed, scales, n, x, bias, y, OUTPUT, INPUT);
        for (int row = 0; row < OUTPUT; ++row) {
            double expected = bias[row];
            for (int group = 0; group < INPUT / GROUP; ++group) {
                const size_t group_index = (size_t)row * (INPUT / GROUP) + group;
                const unsigned char *bytes = packed + group_index * (size_t)(n * 8);
                for (int i = 0; i < GROUP; ++i)
                    expected += (double)((float)reference_code(bytes, n, i) * scales[group_index]) *
                                (double)x[group * GROUP + i];
            }
            CHECK(isfinite(y[row]));
            CHECK(fabs((double)y[row] - expected) < 1e-5 * (1.0 + fabs(expected)));
        }

        qng64_matmul_f32(packed, scales, n, x, bias, z, OUTPUT, INPUT, 2);
        for (int row = 0; row < OUTPUT; ++row)
            CHECK(fabsf(z[row] - y[row]) < 1e-6f);
        const float relative_error = abs_error(weights, packed, scales, n);
        CHECK(isfinite(relative_error));
        if (n == 3) err3 = relative_error;
        if (n == 15) err15 = relative_error;
        ++tested;
    }
    CHECK(tested == 13);
    CHECK(isfinite(err3) && isfinite(err15) && err15 < err3);
    CHECK(err15 < 0.001f);
    printf("BEGLIN_NATIVE_QNG64_CPU_PASS widths=%d n4=REJECTED n8=DISTINCT err3=%.8g err15=%.8g\n",
           tested, err3, err15);
    return 0;
}
