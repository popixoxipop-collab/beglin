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

    // D-export-5: Q5_0 round-trip, reusing the same 128-element vector (32-element blocks,
    // same as Q4_0/Q8_0 -- no new test vector needed).
    size_t q5bytes = gguf_w_q5_0_nbytes(n);
    uint8_t *q5buf = malloc(q5bytes);
    gguf_w_quantize_q5_0(w, n, q5buf);
    float *q5dec = malloc(n * sizeof(float));
    gguf_dequant_row(GGML_TYPE_Q5_0, q5buf, q5dec, n);
    double q5_maxerr = 0;
    for (int64_t i = 0; i < n; i++) {
        double e = fabs(q5dec[i] - w[i]);
        if (e > q5_maxerr) q5_maxerr = e;
    }
    printf("Q5_0: n=%lld bytes=%zu max_abs_err=%.6f\n", (long long)n, q5bytes, q5_maxerr);

    // Dump inputs + raw quantized bytes for an independent Python cross-check.
    FILE *fw = fopen("/tmp/qtest_w.f32", "wb"); fwrite(w, sizeof(float), n, fw); fclose(fw);
    FILE *fq4 = fopen("/tmp/qtest_q4.bin", "wb"); fwrite(q4buf, 1, q4bytes, fq4); fclose(fq4);
    FILE *fq8 = fopen("/tmp/qtest_q8.bin", "wb"); fwrite(q8buf, 1, q8bytes, fq8); fclose(fq8);
    FILE *fq5 = fopen("/tmp/qtest_q5.bin", "wb"); fwrite(q5buf, 1, q5bytes, fq5); fclose(fq5);
    printf("dumped /tmp/qtest_w.f32 /tmp/qtest_q4.bin /tmp/qtest_q8.bin /tmp/qtest_q5.bin (n=%lld)\n", (long long)n);

    // D-export-4: Q4_K round-trip. Needs a real 256-element super-block (n=128 above is too
    // short); build a second, independent test vector at n=1024 (4 super-blocks) covering an
    // all-positive sub-block (exercises the lo-clamped-to-0 path), an all-negative sub-block,
    // and mixed-sign sub-blocks, plus varied magnitude across super-blocks.
    int64_t nk = 1024;
    float *wk = malloc(nk * sizeof(float));
    unsigned seedk = 987654321u;
    for (int64_t i = 0; i < nk; i++) {
        seedk = seedk * 1103515245u + 12345u;
        float r = ((int)(seedk >> 8) % 20000) / 1000.0f - 10.0f;  // roughly [-10, 10)
        int64_t sub = (i / 32) % 8;
        if (sub == 0) r = fabsf(r) + 0.01f;        // all-positive sub-block
        else if (sub == 1) r = -fabsf(r) - 0.01f;  // all-negative sub-block
        wk[i] = r * (1.0f + 0.5f * ((i / 256) % 3)) * (1.0f + 0.2f * sinf((float)i * 0.19f));
    }
    size_t qkbytes = gguf_w_q4_k_nbytes(nk);
    uint8_t *qkbuf = malloc(qkbytes);
    gguf_w_quantize_q4_k(wk, nk, qkbuf);
    float *qkdec = malloc(nk * sizeof(float));
    gguf_dequant_row(GGML_TYPE_Q4_K, qkbuf, qkdec, nk);
    double qk_maxerr = 0, qk_sumabs = 0;
    for (int64_t i = 0; i < nk; i++) {
        double e = fabs(qkdec[i] - wk[i]);
        if (e > qk_maxerr) qk_maxerr = e;
        qk_sumabs += fabs(wk[i]);
    }
    printf("Q4_K: n=%lld bytes=%zu max_abs_err=%.6f mean_abs_val=%.6f rel=%.4f%%\n",
        (long long)nk, qkbytes, qk_maxerr, qk_sumabs/nk, 100.0*qk_maxerr/(qk_sumabs/nk));

    FILE *fwk = fopen("/tmp/qtest_wk.f32", "wb"); fwrite(wk, sizeof(float), nk, fwk); fclose(fwk);
    FILE *fqk = fopen("/tmp/qtest_q4k.bin", "wb"); fwrite(qkbuf, 1, qkbytes, fqk); fclose(fqk);
    printf("dumped /tmp/qtest_wk.f32 /tmp/qtest_q4k.bin (n=%lld)\n", (long long)nk);

    return 0;
}
