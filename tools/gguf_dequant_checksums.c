// Full-sweep sanity net: dequant every tensor in the file, print a checksum per tensor. Used
// alongside gguf_dequant_dump.c's deeper (first/last-20 exact value) check on a representative
// sample of every type actually present -- this catches shape/size edge cases the sample might
// have missed, at the cost of only checking a checksum (not full per-element exactness) per
// tensor.
#include <stdio.h>
#include <stdlib.h>
#include "gguf_load.h"
#include "gguf_quants.h"

int main(int argc, char **argv) { (void)argc;
    GgufFile *f = gguf_open(argv[1]);
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        const GgufTensorInfo *t = &f->tensors[i];
        if (!gguf_dequant_supported(t->type)) { printf("%s SKIP_UNSUPPORTED_TYPE\n", t->name); continue; }
        float *out = malloc(sizeof(float) * t->n_elements);
        gguf_dequant_row(t->type, gguf_tensor_data(f, t), out, (int64_t)t->n_elements);
        double sum = 0.0;
        for (uint64_t j = 0; j < t->n_elements; j++) sum += (double)out[j] * ((j % 97) + 1);
        printf("%s checksum=%.9g n=%llu\n", t->name, sum, (unsigned long long)t->n_elements);
        free(out);
    }
    gguf_close(f);
    return 0;
}
