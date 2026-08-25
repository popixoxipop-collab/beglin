// gguf_verify.c -- R4 oracle dump tool (PLAN_general_purpose_loader.md). Prints a canonical,
// line-per-record dump of every KV entry and every tensor's info, in file order, in a format
// designed to diff byte-for-byte against an equivalent Python dump using gguf-py as the
// independent reference implementation. Not part of the shipped engine.
#include <stdio.h>
#include "gguf_load.h"

static const char *vtype_name(GgufValueType t) {
    switch (t) {
        case GGUF_VTYPE_UINT8: return "UINT8"; case GGUF_VTYPE_INT8: return "INT8";
        case GGUF_VTYPE_UINT16: return "UINT16"; case GGUF_VTYPE_INT16: return "INT16";
        case GGUF_VTYPE_UINT32: return "UINT32"; case GGUF_VTYPE_INT32: return "INT32";
        case GGUF_VTYPE_FLOAT32: return "FLOAT32"; case GGUF_VTYPE_BOOL: return "BOOL";
        case GGUF_VTYPE_STRING: return "STRING"; case GGUF_VTYPE_ARRAY: return "ARRAY";
        case GGUF_VTYPE_UINT64: return "UINT64"; case GGUF_VTYPE_INT64: return "INT64";
        case GGUF_VTYPE_FLOAT64: return "FLOAT64"; default: return "?";
    }
}
static const char *gtype_name(GgmlType t) {
    switch (t) {
        case GGML_TYPE_F32: return "F32"; case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_Q4_0: return "Q4_0"; case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0"; case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0"; case GGML_TYPE_Q8_1: return "Q8_1";
        case GGML_TYPE_Q2_K: return "Q2_K"; case GGML_TYPE_Q3_K: return "Q3_K";
        case GGML_TYPE_Q4_K: return "Q4_K"; case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K"; case GGML_TYPE_Q8_K: return "Q8_K";
        case GGML_TYPE_BF16: return "BF16"; default: return "?";
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file.gguf>\n", argv[0]); return 2; }
    GgufFile *f = gguf_open(argv[1]);
    if (!f) { fprintf(stderr, "gguf_open failed\n"); return 1; }

    printf("VERSION %u\n", f->version);
    printf("ALIGNMENT %llu\n", (unsigned long long)f->alignment);
    printf("N_KV %llu\n", (unsigned long long)f->n_kv);
    printf("N_TENSORS %llu\n", (unsigned long long)f->n_tensors);

    for (uint64_t i = 0; i < f->n_kv; i++) {
        GgufKV *kv = &f->kv[i];
        if (kv->is_array) {
            printf("KV %s ARRAY_OF_%s len=%llu\n", kv->key, vtype_name(kv->type), (unsigned long long)kv->arr_len);
        } else if (kv->type == GGUF_VTYPE_STRING) {
            printf("KV %s STRING len=%llu val=%.*s\n", kv->key, (unsigned long long)kv->scalar.str.len,
                   (int)kv->scalar.str.len, kv->scalar.str.ptr);
        } else if (kv->type == GGUF_VTYPE_FLOAT32 || kv->type == GGUF_VTYPE_FLOAT64) {
            printf("KV %s %s val=%.6g\n", kv->key, vtype_name(kv->type), kv->scalar.f);
        } else if (kv->type == GGUF_VTYPE_BOOL) {
            printf("KV %s BOOL val=%d\n", kv->key, kv->scalar.b);
        } else {
            printf("KV %s %s val=%lld\n", kv->key, vtype_name(kv->type), (long long)kv->scalar.i);
        }
    }

    for (uint64_t i = 0; i < f->n_tensors; i++) {
        GgufTensorInfo *t = &f->tensors[i];
        printf("TENSOR %s type=%s n_dims=%u ne=[%llu,%llu,%llu,%llu] n_elements=%llu n_bytes=%llu data_offset=%llu\n",
               t->name, gtype_name(t->type), t->n_dims,
               (unsigned long long)t->ne[0], (unsigned long long)t->ne[1],
               (unsigned long long)t->ne[2], (unsigned long long)t->ne[3],
               (unsigned long long)t->n_elements, (unsigned long long)t->n_bytes,
               (unsigned long long)t->data_offset);
    }

    gguf_close(f);
    return 0;
}
