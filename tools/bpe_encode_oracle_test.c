// Reads the prompt from a file (byte-exact, no shell escaping pitfalls) so it can be diffed
// directly against `llama-tokenize -f <file> --ids`. pretok arg: qwen2|llama3|gpt2.
#include "gguf_load.h"
#include "bpe_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <gguf-path> <prompt-file> <qwen2|llama3|gpt2>\n", argv[0]); return 1; }
    BpePretokType pretok;
    if (strcmp(argv[3], "qwen2") == 0) pretok = BPE_PRETOK_QWEN2;
    else if (strcmp(argv[3], "llama3") == 0) pretok = BPE_PRETOK_LLAMA3;
    else if (strcmp(argv[3], "gpt2") == 0) pretok = BPE_PRETOK_GPT2;
    else { fprintf(stderr, "unknown pretok '%s'\n", argv[3]); return 1; }

    GgufFile *f = gguf_open(argv[1]);
    if (!f) { fprintf(stderr, "gguf_open failed\n"); return 1; }
    BpeVocab v;
    if (!bpe_vocab_load(f, pretok, &v)) { fprintf(stderr, "bpe_vocab_load failed\n"); return 1; }

    FILE *pf = fopen(argv[2], "rb");
    if (!pf) { fprintf(stderr, "cannot open prompt file\n"); return 1; }
    char text[65536];
    size_t n = fread(text, 1, sizeof(text), pf);
    fclose(pf);

    int32_t ids[8192];
    int cnt = bpe_encode(&v, text, n, ids, 8192);
    if (cnt < 0) { fprintf(stderr, "encode failed\n"); return 1; }
    printf("[");
    for (int i = 0; i < cnt; i++) printf("%s%d", i ? ", " : "", ids[i]);
    printf("]\n");

    char decoded[65536];
    int dn = bpe_decode(&v, ids, cnt, decoded, sizeof(decoded));
    int ok = (dn == (int)n) && (memcmp(decoded, text, n) == 0);
    fprintf(stderr, "roundtrip: %s\n", ok ? "OK" : "MISMATCH");

    bpe_vocab_free(&v);
    gguf_close(f);
    return 0;
}
