// safetensors_verify.c -- checksum oracle for the new safetensors container parser, same
// pattern as gguf_dequant_checksums.c: a weighted checksum per tensor, diffed against an
// independent Python reference (the real `safetensors` pip package, not a hand-rolled
// re-implementation -- same "independent implementation" discipline as gguf-py's role in
// Phase 4's gates). Only F32/F16/BF16 dequant is implemented (the dtypes this project's dense
// models actually ship in); other dtypes are skipped, matching gguf_dequant_checksums.c's own
// SKIP_UNSUPPORTED_TYPE convention.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "safetensors_load.h"

static float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f; memcpy(&f, &bits, 4);
    return f;
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {
            // subnormal f16 -> normalized f32
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);  // inf/nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, 4);
    return f;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.safetensors>\n", argv[0]); return 1; }
    SafetensorsFile *f = safetensors_open(argv[1]);
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        SafetensorsInfo *t = &f->tensors[i];
        if (t->dtype != ST_TYPE_F32 && t->dtype != ST_TYPE_F16 && t->dtype != ST_TYPE_BF16) {
            printf("%s SKIP_UNSUPPORTED_TYPE(%s)\n", t->name, safetensors_type_name(t->dtype));
            continue;
        }
        const uint8_t *raw = (const uint8_t *)safetensors_tensor_data(f, t);
        double sum = 0.0;
        for (uint64_t j = 0; j < t->n_elements; j++) {
            float v;
            if (t->dtype == ST_TYPE_F32) { memcpy(&v, raw + j * 4, 4); }
            else {
                uint16_t h; memcpy(&h, raw + j * 2, 2);
                v = (t->dtype == ST_TYPE_BF16) ? bf16_to_f32(h) : f16_to_f32(h);
            }
            sum += (double)v * ((double)(j % 97) + 1.0);
        }
        printf("%s checksum=%.9g n=%llu dtype=%s\n", t->name, sum, (unsigned long long)t->n_elements,
               safetensors_type_name(t->dtype));
    }
    safetensors_close(f);
    return 0;
}
