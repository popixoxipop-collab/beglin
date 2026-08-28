// safetensors_multi_verify.c -- oracle CLI for the SafetensorsMulti shard-aware wrapper
// (safetensors_load.h). Opens a *.safetensors.index.json manifest via safetensors_open_multi(),
// then independently re-reads the SAME manifest via hf_config.h (not through SafetensorsMulti's
// own internals) to enumerate every real tensor name, and for each one calls
// safetensors_multi_find_tensor() and prints dtype/shape/byte-count -- diffable against an
// independent Python `safetensors.safe_open()` read of the same real shards. Same "independent
// implementation as ground truth" discipline as hf_config_verify.c/safetensors_verify.c.
#include <stdio.h>
#include <stdint.h>
#include "safetensors_load.h"
#include "hf_config.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.safetensors.index.json>\n", argv[0]); return 1; }
    const char *path = argv[1];

    SafetensorsMulti *m = safetensors_open_multi(path);

    HfConfig *idx = hf_config_open(path);
    const HfConfig *wm = hf_config_get_object(idx, "weight_map");
    if (!wm) { fprintf(stderr, "FATAL: %s missing weight_map\n", path); return 1; }
    size_t n = hf_config_n_entries(wm);

    for (size_t i = 0; i < n; i++) {
        const char *name = hf_config_entry_key(wm, i);
        SafetensorsFile *shard = NULL;
        const SafetensorsInfo *t = safetensors_multi_find_tensor(m, name, &shard);
        if (!t) { printf("%s MISSING\n", name); continue; }
        printf("%s dtype=%s n_dims=%u shape=[", name, safetensors_type_name(t->dtype), t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++) printf("%s%llu", d ? "," : "", (unsigned long long)t->shape[d]);
        printf("] n_elements=%llu n_bytes=%llu\n", (unsigned long long)t->n_elements, (unsigned long long)t->n_bytes);
    }

    hf_config_close(idx);
    safetensors_multi_close(m);
    return 0;
}
