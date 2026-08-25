// Dumps dequantized float values for named tensors from a real GGUF file, for R4-style
// cross-check against gguf-py's independent numpy dequant implementation.
#include <stdio.h>
#include <stdlib.h>
#include "gguf_load.h"
#include "gguf_quants.h"

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <file.gguf> <tensor_name> [tensor_name...]\n", argv[0]); return 2; }
    GgufFile *f = gguf_open(argv[1]);
    for (int a = 2; a < argc; a++) {
        const GgufTensorInfo *t = gguf_find_tensor(f, argv[a]);
        if (!t) { fprintf(stderr, "not found: %s\n", argv[a]); continue; }
        if (!gguf_dequant_supported(t->type)) { fprintf(stderr, "unsupported type for %s\n", argv[a]); continue; }
        float *out = malloc(sizeof(float) * t->n_elements);
        gguf_dequant_row(t->type, gguf_tensor_data(f, t), out, (int64_t)t->n_elements);
        printf("TENSOR %s n=%llu\n", t->name, (unsigned long long)t->n_elements);
        // Print a manageable sample: first 20, last 20, and a checksum over everything (so a
        // subtle mid-array bug isn't hidden by only checking the edges).
        double sum = 0.0;
        for (uint64_t i = 0; i < t->n_elements; i++) sum += (double)out[i] * (i % 97 + 1);
        for (uint64_t i = 0; i < 20 && i < t->n_elements; i++) printf("  [%llu]=%.9g\n", (unsigned long long)i, out[i]);
        printf("  ... checksum=%.9g\n", sum);
        for (uint64_t i = t->n_elements > 20 ? t->n_elements-20 : 0; i < t->n_elements; i++) printf("  [%llu]=%.9g\n", (unsigned long long)i, out[i]);
        free(out);
    }
    gguf_close(f);
    return 0;
}
