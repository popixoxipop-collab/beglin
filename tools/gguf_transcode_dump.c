// R4-oracle dump for Phase 2's transcode quantizer: dequantizes a named GGUF tensor (same path
// Phase 1 already verified against gguf-py), then re-quantizes it via gguf_transcode.c's
// gguf_quantize_q4g64_error_feedback() or gguf_quantize_q8g64(), and writes the packed
// bytes+scales as raw binary so tools/gguf_transcode_oracle.py's independent numpy
// implementation (a straight port of the SAME quant_group_ef/quant_group_int8 this file's
// quantizer itself ports) can be `cmp`'d byte-for-byte against it.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gguf_load.h"
#include "gguf_quants.h"
#include "gguf_transcode.h"

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <file.gguf> <tensor_name> <q4|q8> <out_prefix>\n", argv[0]);
        return 2;
    }
    const char *path = argv[1], *name = argv[2], *mode = argv[3], *prefix = argv[4];
    GgufFile *f = gguf_open(path);
    const GgufTensorInfo *t = gguf_find_tensor(f, name);
    if (!t) { fprintf(stderr, "not found: %s\n", name); return 1; }
    if (t->n_dims != 2) { fprintf(stderr, "expected 2D tensor, got n_dims=%u\n", t->n_dims); return 1; }
    if (!gguf_dequant_supported(t->type)) { fprintf(stderr, "unsupported dequant type\n"); return 1; }

    int in = (int)t->ne[0], out = (int)t->ne[1];
    float *deq = malloc(sizeof(float) * t->n_elements);
    gguf_dequant_row(t->type, gguf_tensor_data(f, t), deq, (int64_t)t->n_elements);

    char path_buf[512];
    if (strcmp(mode, "q4") == 0) {
        int ng = in / 64;
        uint8_t *packed = malloc((size_t)out * (in / 2));
        float *scales = malloc(sizeof(float) * (size_t)out * ng);
        gguf_quantize_q4g64_error_feedback(deq, out, in, packed, scales);
        snprintf(path_buf, sizeof path_buf, "%s.packed.bin", prefix);
        FILE *fp = fopen(path_buf, "wb"); fwrite(packed, 1, (size_t)out * (in / 2), fp); fclose(fp);
        snprintf(path_buf, sizeof path_buf, "%s.scales.bin", prefix);
        fp = fopen(path_buf, "wb"); fwrite(scales, sizeof(float), (size_t)out * ng, fp); fclose(fp);
        printf("q4 out=%d in=%d ng=%d packed_bytes=%zu scales_floats=%d\n",
               out, in, ng, (size_t)out * (in / 2), out * ng);
    } else if (strcmp(mode, "q8") == 0) {
        int ng = in / 64;
        int8_t *codes = malloc((size_t)out * in);
        float *scales = malloc(sizeof(float) * (size_t)out * ng);
        gguf_quantize_q8g64(deq, out, in, codes, scales);
        snprintf(path_buf, sizeof path_buf, "%s.codes.bin", prefix);
        FILE *fp = fopen(path_buf, "wb"); fwrite(codes, 1, (size_t)out * in, fp); fclose(fp);
        snprintf(path_buf, sizeof path_buf, "%s.scales.bin", prefix);
        fp = fopen(path_buf, "wb"); fwrite(scales, sizeof(float), (size_t)out * ng, fp); fclose(fp);
        printf("q8 out=%d in=%d ng=%d codes_bytes=%zu scales_floats=%d\n",
               out, in, ng, (size_t)out * in, out * ng);
    } else if (strcmp(mode, "qN") == 0) {
        // D-qNg64-1: dumps qNg64's own packed planes + scales for the NumPy oracle to
        // cmp against -- `n` is argv[5] (required for this mode only).
        if (argc < 6) { fprintf(stderr, "qN mode needs a 5th arg: n\n"); return 2; }
        int n = atoi(argv[5]);
        int ng = in / 64;
        size_t row_pbytes = (size_t)ng * (size_t)n * 8;
        uint8_t *planes = malloc((size_t)out * row_pbytes);
        float *scales = malloc(sizeof(float) * (size_t)out * ng);
        gguf_quantize_qNg64(deq, out, in, n, planes, scales);
        snprintf(path_buf, sizeof path_buf, "%s.planes.bin", prefix);
        FILE *fp = fopen(path_buf, "wb"); fwrite(planes, 1, (size_t)out * row_pbytes, fp); fclose(fp);
        snprintf(path_buf, sizeof path_buf, "%s.scales.bin", prefix);
        fp = fopen(path_buf, "wb"); fwrite(scales, sizeof(float), (size_t)out * ng, fp); fclose(fp);
        printf("qN n=%d out=%d in=%d ng=%d planes_bytes=%zu scales_floats=%d\n",
               n, out, in, ng, (size_t)out * row_pbytes, out * ng);
    } else if (strcmp(mode, "n4check") == 0) {
        // D-qNg64-1 regression gate: qNg64(n=4) must produce bit-identical CODES and
        // SCALES to gguf_quantize_q4g64_error_feedback() -- the packed BYTE layout
        // differs (nibble vs bit-plane), so this unpacks both back to raw codes before
        // comparing, rather than comparing packed bytes directly.
        int ng = in / 64;
        uint8_t *q4_packed = malloc((size_t)out * (in / 2));
        float *q4_scales = malloc(sizeof(float) * (size_t)out * ng);
        gguf_quantize_q4g64_error_feedback(deq, out, in, q4_packed, q4_scales);

        size_t qn_row_pbytes = (size_t)ng * 4 * 8;
        uint8_t *qn_planes = malloc((size_t)out * qn_row_pbytes);
        float *qn_scales = malloc(sizeof(float) * (size_t)out * ng);
        gguf_quantize_qNg64(deq, out, in, 4, qn_planes, qn_scales);

        long mismatches = 0, scale_mismatches = 0;
        for (int r = 0; r < out; r++) {
            for (int g = 0; g < ng; g++) {
                float s4 = q4_scales[(size_t)r * ng + g];
                float sn = qn_scales[(size_t)r * ng + g];
                if (s4 != sn) scale_mismatches++;
                for (int p = 0; p < 64; p++) {
                    int idx = g * 64 + p;
                    uint8_t byte4 = q4_packed[(size_t)r * (in / 2) + idx / 2];
                    int nib = (idx % 2 == 0) ? (byte4 & 0xF) : ((byte4 >> 4) & 0xF);
                    int code4 = nib - 8;

                    const uint8_t *pgrp = qn_planes + (size_t)r * qn_row_pbytes + (size_t)g * 32;
                    int byte_in_plane = p >> 3, bit_in_byte = p & 7;
                    int u = 0;
                    for (int j = 0; j < 4; j++) {
                        uint8_t pb = pgrp[j * 8 + byte_in_plane];
                        if ((pb >> bit_in_byte) & 1) u |= (1 << j);
                    }
                    int coden = u - 8;
                    if (code4 != coden) mismatches++;
                }
            }
        }
        printf("n4check: out=%d in=%d ng=%d code_mismatches=%ld scale_mismatches=%ld -> %s\n",
               out, in, ng, mismatches, scale_mismatches,
               (mismatches == 0 && scale_mismatches == 0) ? "PASS" : "FAIL");
        if (mismatches != 0 || scale_mismatches != 0) { gguf_close(f); return 1; }
    } else {
        fprintf(stderr, "mode must be q4, q8, qN, or n4check\n"); return 2;
    }
    gguf_close(f);
    return 0;
}
