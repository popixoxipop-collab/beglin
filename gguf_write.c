// gguf_write.c -- see gguf_write.h for the design/ownership contract. Own translation unit,
// same reason as gguf_load.c.

#include "gguf_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *key;
    GgufValueType type;      // scalar type, or element type if is_array
    int is_array;
    uint64_t arr_len;        // 1 if !is_array

    union {
        uint64_t u; int64_t i; double f; int b;
        struct { char *ptr; uint64_t len; } str;   // owned copy (malloc'd), scalar case only
    } scalar;

    const GgufStr *arr_str;   // caller-owned, not copied (array-of-string case)
    const void *arr_fixed;    // caller-owned, not copied (fixed-width array case)
} WKv;

typedef struct {
    char *name;
    GgmlType type;
    uint32_t n_dims;
    uint64_t ne[4];
    const void *data;         // caller-owned, not copied
    uint64_t nbytes;
    uint64_t offset;          // computed in gguf_w_finish(), relative to the data section start
} WTensor;

struct GgufWriter {
    FILE *f;
    uint64_t alignment;       // GGUF default; overridden if the caller sets "general.alignment"
    WKv *kvs; uint64_t n_kv, cap_kv;
    WTensor *tensors; uint64_t n_tensors, cap_tensors;
    int had_error;
};

static WKv *push_kv(GgufWriter *w, const char *key) {
    if (w->n_kv >= w->cap_kv) {
        w->cap_kv = w->cap_kv ? w->cap_kv * 2 : 16;
        w->kvs = realloc(w->kvs, w->cap_kv * sizeof(WKv));
    }
    WKv *kv = &w->kvs[w->n_kv++];
    memset(kv, 0, sizeof(*kv));
    kv->key = strdup(key);
    return kv;
}

GgufWriter *gguf_w_open(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return NULL;
    GgufWriter *w = calloc(1, sizeof(GgufWriter));
    w->f = f;
    w->alignment = 32;   // GGUF spec default, same as gguf_load.c's own read-side default
    return w;
}

static void maybe_update_alignment(GgufWriter *w, const char *key, uint64_t val) {
    if (!strcmp(key, "general.alignment") && val > 0) w->alignment = val;
}

void gguf_w_kv_str(GgufWriter *w, const char *key, const char *val, uint64_t val_len) {
    WKv *kv = push_kv(w, key);
    kv->type = GGUF_VTYPE_STRING;
    kv->scalar.str.ptr = malloc(val_len ? val_len : 1);
    memcpy(kv->scalar.str.ptr, val, val_len);
    kv->scalar.str.len = val_len;
}
void gguf_w_kv_u32(GgufWriter *w, const char *key, uint32_t val) {
    WKv *kv = push_kv(w, key);
    kv->type = GGUF_VTYPE_UINT32;
    kv->scalar.u = val;
    maybe_update_alignment(w, key, val);
}
void gguf_w_kv_u64(GgufWriter *w, const char *key, uint64_t val) {
    WKv *kv = push_kv(w, key);
    kv->type = GGUF_VTYPE_UINT64;
    kv->scalar.u = val;
    maybe_update_alignment(w, key, val);
}
void gguf_w_kv_i64(GgufWriter *w, const char *key, int64_t val) {
    WKv *kv = push_kv(w, key);
    kv->type = GGUF_VTYPE_INT64;
    kv->scalar.i = val;
}
void gguf_w_kv_f32(GgufWriter *w, const char *key, float val) {
    WKv *kv = push_kv(w, key);
    kv->type = GGUF_VTYPE_FLOAT32;
    kv->scalar.f = (double)val;
}
void gguf_w_kv_f64(GgufWriter *w, const char *key, double val) {
    WKv *kv = push_kv(w, key);
    kv->type = GGUF_VTYPE_FLOAT64;
    kv->scalar.f = val;
}
void gguf_w_kv_bool(GgufWriter *w, const char *key, int val) {
    WKv *kv = push_kv(w, key);
    kv->type = GGUF_VTYPE_BOOL;
    kv->scalar.b = val ? 1 : 0;
}

void gguf_w_kv_str_array(GgufWriter *w, const char *key, const GgufStr *arr, uint64_t n) {
    WKv *kv = push_kv(w, key);
    kv->type = GGUF_VTYPE_STRING;
    kv->is_array = 1;
    kv->arr_len = n;
    kv->arr_str = arr;
}
void gguf_w_kv_i32_array(GgufWriter *w, const char *key, const int32_t *arr, uint64_t n) {
    WKv *kv = push_kv(w, key);
    kv->type = GGUF_VTYPE_INT32;
    kv->is_array = 1;
    kv->arr_len = n;
    kv->arr_fixed = arr;
}
void gguf_w_kv_f32_array(GgufWriter *w, const char *key, const float *arr, uint64_t n) {
    WKv *kv = push_kv(w, key);
    kv->type = GGUF_VTYPE_FLOAT32;
    kv->is_array = 1;
    kv->arr_len = n;
    kv->arr_fixed = arr;
}

void gguf_w_add_tensor(GgufWriter *w, const char *name, GgmlType type,
                        uint32_t n_dims, const uint64_t *ne,
                        const void *data, uint64_t nbytes) {
    if (w->n_tensors >= w->cap_tensors) {
        w->cap_tensors = w->cap_tensors ? w->cap_tensors * 2 : 16;
        w->tensors = realloc(w->tensors, w->cap_tensors * sizeof(WTensor));
    }
    WTensor *t = &w->tensors[w->n_tensors++];
    memset(t, 0, sizeof(*t));
    t->name = strdup(name);
    t->type = type;
    t->n_dims = n_dims;
    for (uint32_t d = 0; d < 4; d++) t->ne[d] = (d < n_dims) ? ne[d] : 1;
    t->data = data;
    t->nbytes = nbytes;
}

// ---- Low-level byte writers (mirror gguf_load.c's cur_* readers, in reverse) ----

static void w_bytes(GgufWriter *w, const void *p, size_t n) {
    if (n && fwrite(p, 1, n, w->f) != n) w->had_error = 1;
}
static void w_u32(GgufWriter *w, uint32_t v) { w_bytes(w, &v, 4); }
static void w_u64(GgufWriter *w, uint64_t v) { w_bytes(w, &v, 8); }
static void w_i64(GgufWriter *w, int64_t v)  { w_bytes(w, &v, 8); }
static void w_f32(GgufWriter *w, float v)    { w_bytes(w, &v, 4); }
static void w_f64(GgufWriter *w, double v)   { w_bytes(w, &v, 8); }
static void w_u8(GgufWriter *w, uint8_t v)   { w_bytes(w, &v, 1); }
// GGUF strings: uint64 length prefix, then raw bytes, NOT NUL-terminated -- exact mirror of
// gguf_load.c's own cur_str() read convention.
static void w_str(GgufWriter *w, const char *ptr, uint64_t len) {
    w_u64(w, len);
    w_bytes(w, ptr, (size_t)len);
}

static void w_scalar_value(GgufWriter *w, GgufValueType t, const WKv *kv) {
    switch (t) {
        case GGUF_VTYPE_UINT8:  w_u8(w, (uint8_t)kv->scalar.u); break;
        case GGUF_VTYPE_INT8:   w_u8(w, (uint8_t)(int8_t)kv->scalar.i); break;
        case GGUF_VTYPE_UINT16: { uint16_t v = (uint16_t)kv->scalar.u; w_bytes(w, &v, 2); } break;
        case GGUF_VTYPE_INT16:  { int16_t v = (int16_t)kv->scalar.i; w_bytes(w, &v, 2); } break;
        case GGUF_VTYPE_UINT32: w_u32(w, (uint32_t)kv->scalar.u); break;
        case GGUF_VTYPE_INT32:  w_u32(w, (uint32_t)(int32_t)kv->scalar.i); break;
        case GGUF_VTYPE_FLOAT32:w_f32(w, (float)kv->scalar.f); break;
        case GGUF_VTYPE_BOOL:   w_u8(w, (uint8_t)(kv->scalar.b ? 1 : 0)); break;
        case GGUF_VTYPE_UINT64: w_u64(w, kv->scalar.u); break;
        case GGUF_VTYPE_INT64:  w_i64(w, kv->scalar.i); break;
        case GGUF_VTYPE_FLOAT64:w_f64(w, kv->scalar.f); break;
        case GGUF_VTYPE_STRING: w_str(w, kv->scalar.str.ptr, kv->scalar.str.len); break;
        default:
            fprintf(stderr, "FATAL: gguf_write: w_scalar_value called on non-scalar type %d\n", (int)t);
            exit(1);
    }
}

static uint64_t align_up(uint64_t x, uint64_t a) {
    return (x % a == 0) ? x : x + (a - x % a);
}

int gguf_w_finish(GgufWriter *w) {
    // Magic + version + counts.
    w_bytes(w, "GGUF", 4);
    w_u32(w, 3);
    w_u64(w, w->n_tensors);
    w_u64(w, w->n_kv);

    // KV table.
    for (uint64_t i = 0; i < w->n_kv; i++) {
        WKv *kv = &w->kvs[i];
        w_str(w, kv->key, strlen(kv->key));
        if (kv->is_array) {
            w_u32(w, GGUF_VTYPE_ARRAY);
            w_u32(w, (uint32_t)kv->type);
            w_u64(w, kv->arr_len);
            if (kv->type == GGUF_VTYPE_STRING) {
                for (uint64_t j = 0; j < kv->arr_len; j++) w_str(w, kv->arr_str[j].ptr, kv->arr_str[j].len);
            } else if (kv->type == GGUF_VTYPE_INT32) {
                const int32_t *arr = (const int32_t *)kv->arr_fixed;
                for (uint64_t j = 0; j < kv->arr_len; j++) w_u32(w, (uint32_t)arr[j]);
            } else if (kv->type == GGUF_VTYPE_FLOAT32) {
                const float *arr = (const float *)kv->arr_fixed;
                for (uint64_t j = 0; j < kv->arr_len; j++) w_f32(w, arr[j]);
            } else {
                fprintf(stderr, "FATAL: gguf_write: unsupported array element type %d for key '%s'\n", (int)kv->type, kv->key);
                exit(1);
            }
        } else {
            w_u32(w, (uint32_t)kv->type);
            w_scalar_value(w, kv->type, kv);
        }
    }

    // Tensor-info table -- offsets must be computed first (relative to the data section
    // start), same alignment-padding-between-tensors rule gguf_load.c's read side already
    // applies when computing absolute offsets from these same relative ones.
    uint64_t running = 0;
    for (uint64_t i = 0; i < w->n_tensors; i++) {
        WTensor *t = &w->tensors[i];
        t->offset = running;
        running = align_up(running + t->nbytes, w->alignment);
    }

    for (uint64_t i = 0; i < w->n_tensors; i++) {
        WTensor *t = &w->tensors[i];
        w_str(w, t->name, strlen(t->name));
        w_u32(w, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++) w_u64(w, t->ne[d]);
        w_u32(w, (uint32_t)t->type);
        w_u64(w, t->offset);
    }

    // Data section: starts at the next multiple of `alignment` after the tensor-info table
    // (same rule gguf_load.c's read side applies to locate it), then each tensor's bytes,
    // zero-padded up to the next aligned offset before the following tensor.
    long pos = ftell(w->f);
    if (pos < 0) w->had_error = 1;
    uint64_t data_start = align_up((uint64_t)pos, w->alignment);
    for (uint64_t pad = (uint64_t)pos; pad < data_start; pad++) w_u8(w, 0);

    uint64_t cursor = 0;
    static const uint8_t zeros[64] = {0};
    for (uint64_t i = 0; i < w->n_tensors; i++) {
        WTensor *t = &w->tensors[i];
        w_bytes(w, t->data, t->nbytes);
        cursor += t->nbytes;
        uint64_t next = align_up(cursor, w->alignment);
        while (cursor < next) {
            uint64_t chunk = next - cursor;
            if (chunk > sizeof(zeros)) chunk = sizeof(zeros);
            w_bytes(w, zeros, (size_t)chunk);
            cursor += chunk;
        }
    }

    int ok = !w->had_error;
    if (fclose(w->f) != 0) ok = 0;

    for (uint64_t i = 0; i < w->n_kv; i++) {
        free(w->kvs[i].key);
        if (!w->kvs[i].is_array && w->kvs[i].type == GGUF_VTYPE_STRING) free(w->kvs[i].scalar.str.ptr);
    }
    free(w->kvs);
    for (uint64_t i = 0; i < w->n_tensors; i++) free(w->tensors[i].name);
    free(w->tensors);
    free(w);
    return ok;
}
