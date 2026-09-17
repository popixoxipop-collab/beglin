// D-export Phase 2 oracle test: quantize real-ish F32 data with the new Q4_0/Q8_0 encoders,
// round-trip through this project's OWN existing dequant (gguf_quants.c), and dump the raw
// quantized bytes so an independent Python (gguf-py) script can cross-check too.
#include "gguf_write_quants.h"
#include "gguf_quants.h"
#include "gguf_load.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main(void) {
    // A real-ish, non-trivial test vector: 128 values (4 blocks of 32), varied magnitudes and
    // signs, including near-zero and near-max-magnitude values per block to exercise rounding.
    int64_t n = 128;
    float *w = malloc(n * sizeof(float));
    unsigned seed = 12345;
    for (int64_t i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        float r = ((int)(seed >> 8) % 20000) / 1000.0f - 10.0f;  // roughly [-10, 10)
        w[i] = r * (1.0f + 0.3f * sinf((float)i * 0.37f));
    }

    // Q4_0 round-trip.
    size_t q4bytes = gguf_w_q4_0_nbytes(n);
    uint8_t *q4buf = malloc(q4bytes);
    gguf_w_quantize_q4_0(w, n, q4buf);
    float *q4dec = malloc(n * sizeof(float));
    gguf_dequant_row(GGML_TYPE_Q4_0, q4buf, q4dec, n);
    double q4_maxerr = 0, q4_sumabs = 0;
    for (int64_t i = 0; i < n; i++) {
        double e = fabs(q4dec[i] - w[i]);
        if (e > q4_maxerr) q4_maxerr = e;
        q4_sumabs += fabs(w[i]);
    }
    printf("Q4_0: n=%lld bytes=%zu max_abs_err=%.6f mean_abs_val=%.6f rel=%.4f%%\n",
        (long long)n, q4bytes, q4_maxerr, q4_sumabs/n, 100.0*q4_maxerr/(q4_sumabs/n));

    // Q8_0 round-trip.
    size_t q8bytes = gguf_w_q8_0_nbytes(n);
    uint8_t *q8buf = malloc(q8bytes);
    gguf_w_quantize_q8_0(w, n, q8buf);
    float *q8dec = malloc(n * sizeof(float));
    gguf_dequant_row(GGML_TYPE_Q8_0, q8buf, q8dec, n);
    double q8_maxerr = 0;
    for (int64_t i = 0; i < n; i++) {
        double e = fabs(q8dec[i] - w[i]);
        if (e > q8_maxerr) q8_maxerr = e;
    }
    printf("Q8_0: n=%lld bytes=%zu max_abs_err=%.6f\n", (long long)n, q8bytes, q8_maxerr);

    // Dump inputs + raw quantized bytes for an independent Python cross-check.
    FILE *fw = fopen("/tmp/qtest_w.f32", "wb"); fwrite(w, sizeof(float), n, fw); fclose(fw);
    FILE *fq4 = fopen("/tmp/qtest_q4.bin", "wb"); fwrite(q4buf, 1, q4bytes, fq4); fclose(fq4);
    FILE *fq8 = fopen("/tmp/qtest_q8.bin", "wb"); fwrite(q8buf, 1, q8bytes, fq8); fclose(fq8);
    printf("dumped /tmp/qtest_w.f32 /tmp/qtest_q4.bin /tmp/qtest_q8.bin (n=%lld)\n", (long long)n);

    return 0;
}
