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
    } else {
        fprintf(stderr, "mode must be q4 or q8\n"); return 2;
    }
    gguf_close(f);
    return 0;
}
