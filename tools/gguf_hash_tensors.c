#include <stdio.h>
#include <stdint.h>
#include "gguf_load.h"
// Minimal FNV-independent verification: just print offset+first/last bytes + a simple checksum
// (SHA256 isn't worth vendoring for a one-off sanity check -- a byte-range + simple sum that
// matches Python's SHA256-covered range is sufficient to prove "reading from this offset for
// this many bytes gives the identical byte content", which is what data_offset/n_bytes need to
// prove; the earlier oracle diff already proved offset/n_bytes themselves are correct numbers).
int main(int argc, char **argv) {
    GgufFile *f = gguf_open(argv[1]);
    const char *names[] = {"blk.0.attn_norm.weight", "token_embd.weight", "output.weight", "blk.13.ffn_gate.weight"};
    for (int i = 0; i < 4; i++) {
        const GgufTensorInfo *t = gguf_find_tensor(f, names[i]);
        const uint8_t *d = (const uint8_t *)gguf_tensor_data(f, t);
        uint64_t sum = 0;
        for (uint64_t j = 0; j < t->n_bytes; j++) sum = sum * 1000003u + d[j];
        printf("%s n_bytes=%llu checksum=%llu first4=%02x%02x%02x%02x last4=%02x%02x%02x%02x\n",
               names[i], (unsigned long long)t->n_bytes, (unsigned long long)sum,
               d[0], d[1], d[2], d[3], d[t->n_bytes-4], d[t->n_bytes-3], d[t->n_bytes-2], d[t->n_bytes-1]);
    }
    gguf_close(f);
    return 0;
}
