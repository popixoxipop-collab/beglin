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
#include "safetensors_quants.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.safetensors>\n", argv[0]); return 1; }
    SafetensorsFile *f = safetensors_open(argv[1]);
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        SafetensorsInfo *t = &f->tensors[i];
        if (!safetensors_dequant_supported(t->dtype)) {
            printf("%s SKIP_UNSUPPORTED_TYPE(%s)\n", t->name, safetensors_type_name(t->dtype));
            continue;
        }
        const uint8_t *raw = (const uint8_t *)safetensors_tensor_data(f, t);
        float *deq = malloc((size_t)t->n_elements * sizeof(float));
        safetensors_dequant_row(t->dtype, raw, deq, t->n_elements);
        double sum = 0.0;
        for (uint64_t j = 0; j < t->n_elements; j++) sum += (double)deq[j] * ((double)(j % 97) + 1.0);
        free(deq);
        printf("%s checksum=%.9g n=%llu dtype=%s\n", t->name, sum, (unsigned long long)t->n_elements,
               safetensors_type_name(t->dtype));
    }
    safetensors_close(f);
    return 0;
}
