// gguf_load.c -- see gguf_load.h for the safety contract and why this is its own TU.
// Compiled PLAIN (no SME/SVE arch flag) -- this file contains no vendor kernel calls at all.

#include "gguf_load.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Block-size/type-size table, ground truth: llama.cpp gguf-py's GGML_QUANT_SIZES, confirmed
// directly from that source rather than derived from memory (see PLAN_general_purpose_loader.md
// Phase 1 notes -- getting this table wrong silently miscomputes every tensor's byte extent,
// which is exactly the "unrecognized kind falls through to a wrong-but-plausible answer" failure
// mode load_int4()'s doctrine exists to prevent). Only entries this project's dequant work
// (Phase 1 follow-on) targets are named; others are still tracked (block/typesize) so the
// parser's byte-accounting stays correct for tensors whose *quantization* isn't supported yet --
// an unsupported dequant is this file's caller's problem, not a reason to also break parsing.
typedef struct { GgmlType type; uint64_t block; uint64_t typesize; } GgmlTypeInfo;
static const GgmlTypeInfo GGML_TYPE_TABLE[] = {
    {GGML_TYPE_F32,  1,   4},
    {GGML_TYPE_F16,  1,   2},
    {GGML_TYPE_Q4_0, 32,  18},
    {GGML_TYPE_Q4_1, 32,  20},
    {GGML_TYPE_Q5_0, 32,  22},
    {GGML_TYPE_Q5_1, 32,  24},
    {GGML_TYPE_Q8_0, 32,  34},
    {GGML_TYPE_Q8_1, 32,  40},
    {GGML_TYPE_Q2_K, 256, 84},
    {GGML_TYPE_Q3_K, 256, 110},
    {GGML_TYPE_Q4_K, 256, 144},
    {GGML_TYPE_Q5_K, 256, 176},
    {GGML_TYPE_Q6_K, 256, 210},
    {GGML_TYPE_Q8_K, 256, 292},
    {GGML_TYPE_BF16, 1,   2},
};
#define N_GGML_TYPES (int)(sizeof(GGML_TYPE_TABLE)/sizeof(GGML_TYPE_TABLE[0]))

static int ggml_type_info(GgmlType t, uint64_t *block, uint64_t *typesize) {
    for (int i = 0; i < N_GGML_TYPES; i++) {
        if (GGML_TYPE_TABLE[i].type == t) { *block = GGML_TYPE_TABLE[i].block; *typesize = GGML_TYPE_TABLE[i].typesize; return 1; }
    }
    return 0;
}

// GGUF value-type fixed byte widths (scalar types only; STRING/ARRAY handled specially).
static int gguf_vtype_width(GgufValueType t, int *out_width) {
    switch (t) {
        case GGUF_VTYPE_UINT8: case GGUF_VTYPE_INT8: case GGUF_VTYPE_BOOL: *out_width = 1; return 1;
        case GGUF_VTYPE_UINT16: case GGUF_VTYPE_INT16: *out_width = 2; return 1;
        case GGUF_VTYPE_UINT32: case GGUF_VTYPE_INT32: case GGUF_VTYPE_FLOAT32: *out_width = 4; return 1;
        case GGUF_VTYPE_UINT64: case GGUF_VTYPE_INT64: case GGUF_VTYPE_FLOAT64: *out_width = 8; return 1;
        default: return 0; // STRING/ARRAY are not fixed-width
    }
}

// ---- Bounds-checked cursor over the mmap'd file ----
typedef struct { const uint8_t *base; size_t size; size_t pos; const char *file_for_errors; } Cur;

static void cur_need(Cur *c, size_t n) {
    if (c->pos + n < c->pos || c->pos + n > c->size) {
        fprintf(stderr, "FATAL: gguf_load: %s: truncated/malformed (need %zu bytes at offset %zu, file is %zu bytes)\n",
                c->file_for_errors, n, c->pos, c->size);
        exit(1);
    }
}
static uint8_t  cur_u8 (Cur *c) { cur_need(c,1); uint8_t  v = c->base[c->pos]; c->pos+=1; return v; }
static uint16_t cur_u16(Cur *c) { cur_need(c,2); uint16_t v; memcpy(&v, c->base+c->pos, 2); c->pos+=2; return v; }
static uint32_t cur_u32(Cur *c) { cur_need(c,4); uint32_t v; memcpy(&v, c->base+c->pos, 4); c->pos+=4; return v; }
static uint64_t cur_u64(Cur *c) { cur_need(c,8); uint64_t v; memcpy(&v, c->base+c->pos, 8); c->pos+=8; return v; }
static float    cur_f32(Cur *c) { cur_need(c,4); float    v; memcpy(&v, c->base+c->pos, 4); c->pos+=4; return v; }
static double   cur_f64(Cur *c) { cur_need(c,8); double   v; memcpy(&v, c->base+c->pos, 8); c->pos+=8; return v; }
// GGUF strings: uint64 length prefix, then that many raw bytes, NOT NUL-terminated in the file.
static void cur_str(Cur *c, const char **out_ptr, uint64_t *out_len) {
    uint64_t len = cur_u64(c);
    cur_need(c, len);
    *out_ptr = (const char *)(c->base + c->pos);
    *out_len = len;
    c->pos += len;
}
static char *dupstr(const char *ptr, uint64_t len) {
    char *s = malloc(len + 1);
    memcpy(s, ptr, len);
    s[len] = '\0';
    return s;
}

// Reads and discards one value of the given type (used for array element skipping and for KV
// entries whose value we don't need to decode, but whose byte extent we must still account for
// to reach the next entry -- this is the "walk over tokenizer arrays correctly without
// materializing them" path described in gguf_load.h).
static void skip_value(Cur *c, GgufValueType t) {
    int width;
    if (gguf_vtype_width(t, &width)) { cur_need(c, width); c->pos += width; return; }
    if (t == GGUF_VTYPE_STRING) { uint64_t len = cur_u64(c); cur_need(c, len); c->pos += len; return; }
    if (t == GGUF_VTYPE_ARRAY) {
        GgufValueType elem_t = (GgufValueType)cur_u32(c);
        uint64_t n = cur_u64(c);
        if (elem_t == GGUF_VTYPE_ARRAY) { fprintf(stderr, "FATAL: gguf_load: nested array KV value not supported\n"); exit(1); }
        for (uint64_t i = 0; i < n; i++) skip_value(c, elem_t);
        return;
    }
    fprintf(stderr, "FATAL: gguf_load: unrecognized GGUF value type %d -- cannot safely skip, refusing to guess a byte width\n", (int)t);
    exit(1);
}

static void decode_scalar_into(Cur *c, GgufValueType t, GgufKV *kv) {
    switch (t) {
        case GGUF_VTYPE_UINT8:  kv->scalar.u = cur_u8(c); break;
        case GGUF_VTYPE_INT8:   kv->scalar.i = (int8_t)cur_u8(c); break;
        case GGUF_VTYPE_UINT16: kv->scalar.u = cur_u16(c); break;
        case GGUF_VTYPE_INT16:  kv->scalar.i = (int16_t)cur_u16(c); break;
        case GGUF_VTYPE_UINT32: kv->scalar.u = cur_u32(c); break;
        case GGUF_VTYPE_INT32:  kv->scalar.i = (int32_t)cur_u32(c); break;
        case GGUF_VTYPE_FLOAT32:kv->scalar.f = (double)cur_f32(c); break;
        case GGUF_VTYPE_BOOL:   kv->scalar.b = cur_u8(c) != 0; break;
        case GGUF_VTYPE_UINT64: kv->scalar.u = cur_u64(c); break;
        case GGUF_VTYPE_INT64:  kv->scalar.i = (int64_t)cur_u64(c); break;
        case GGUF_VTYPE_FLOAT64:kv->scalar.f = cur_f64(c); break;
        case GGUF_VTYPE_STRING: cur_str(c, &kv->scalar.str.ptr, &kv->scalar.str.len); break;
        default:
            fprintf(stderr, "FATAL: gguf_load: decode_scalar_into called on non-scalar type %d\n", (int)t);
            exit(1);
    }
}

GgufFile *gguf_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { return NULL; }
    struct stat sb;
    if (fstat(fd, &sb) != 0) { close(fd); return NULL; }
    void *base = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { close(fd); return NULL; }

    GgufFile *f = calloc(1, sizeof(GgufFile));
    f->fd = fd;
    f->base = (uint8_t *)base;
    f->file_size = (size_t)sb.st_size;
    f->alignment = 32; // GGUF default; overridden below if general.alignment KV is present

    Cur c = { f->base, f->file_size, 0, path };

    // Magic: literal ASCII bytes 'G','G','U','F' (not a byte-order-dependent uint32 read).
    cur_need(&c, 4);
    if (memcmp(c.base + c.pos, "GGUF", 4) != 0) {
        fprintf(stderr, "FATAL: gguf_load: %s: bad magic (not a GGUF file)\n", path);
        exit(1);
    }
    c.pos += 4;

    f->version = cur_u32(&c);
    if (f->version != 2 && f->version != 3) {
        fprintf(stderr, "FATAL: gguf_load: %s: unsupported GGUF version %u (this parser targets v3, v2 is close enough to also work; anything else is refused rather than guessed at)\n", path, f->version);
        exit(1);
    }
    f->n_tensors = cur_u64(&c);
    f->n_kv = cur_u64(&c);

    f->kv = calloc(f->n_kv ? f->n_kv : 1, sizeof(GgufKV));
    for (uint64_t i = 0; i < f->n_kv; i++) {
        const char *kptr; uint64_t klen;
        cur_str(&c, &kptr, &klen);
        GgufKV *kv = &f->kv[i];
        kv->key = dupstr(kptr, klen);
        GgufValueType vtype = (GgufValueType)cur_u32(&c);
        if (vtype == GGUF_VTYPE_ARRAY) {
            GgufValueType elem_t = (GgufValueType)cur_u32(&c);
            uint64_t n = cur_u64(&c);
            kv->type = elem_t;
            kv->is_array = 1;
            kv->arr_len = n;
            // Arrays aren't materialized (see header contract) -- just walk past the payload
            // so the cursor lands correctly on the next KV entry's key.
            for (uint64_t j = 0; j < n; j++) skip_value(&c, elem_t);
        } else {
            kv->type = vtype;
            kv->is_array = 0;
            kv->arr_len = 1;
            decode_scalar_into(&c, vtype, kv);
        }
        if (strcmp(kv->key, "general.alignment") == 0 && !kv->is_array &&
            (kv->type == GGUF_VTYPE_UINT32 || kv->type == GGUF_VTYPE_UINT64 || kv->type == GGUF_VTYPE_INT32)) {
            f->alignment = kv->scalar.u;
        }
    }

    f->tensors = calloc(f->n_tensors ? f->n_tensors : 1, sizeof(GgufTensorInfo));
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        const char *nptr; uint64_t nlen;
        cur_str(&c, &nptr, &nlen);
        GgufTensorInfo *t = &f->tensors[i];
        t->name = dupstr(nptr, nlen);
        t->n_dims = cur_u32(&c);
        if (t->n_dims == 0 || t->n_dims > 4) {
            fprintf(stderr, "FATAL: gguf_load: %s: tensor '%s' has n_dims=%u (expected 1..4)\n", path, t->name, t->n_dims);
            exit(1);
        }
        for (int d = 0; d < 4; d++) t->ne[d] = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) t->ne[d] = cur_u64(&c);
        t->type = (GgmlType)cur_u32(&c);
        uint64_t rel_offset = cur_u64(&c);

        uint64_t block, typesize;
        if (!ggml_type_info(t->type, &block, &typesize)) {
            fprintf(stderr, "FATAL: gguf_load: %s: tensor '%s' has unrecognized ggml type id %d -- refusing to guess its byte size\n", path, t->name, (int)t->type);
            exit(1);
        }
        uint64_t n_elem = 1;
        for (int d = 0; d < 4; d++) n_elem *= t->ne[d];
        if (n_elem % block != 0) {
            fprintf(stderr, "FATAL: gguf_load: %s: tensor '%s' has %llu elements, not a multiple of its type's block size %llu\n",
                    path, t->name, (unsigned long long)n_elem, (unsigned long long)block);
            exit(1);
        }
        t->n_elements = n_elem;
        t->n_bytes = (n_elem / block) * typesize;
        t->data_offset = rel_offset; // absolute-ized below, once the data section start is known
    }

    // Data section starts at the next multiple of `alignment` after the tensor info table.
    uint64_t data_start = c.pos;
    if (f->alignment == 0) { fprintf(stderr, "FATAL: gguf_load: %s: general.alignment=0\n", path); exit(1); }
    if (data_start % f->alignment != 0) data_start += f->alignment - (data_start % f->alignment);

    for (uint64_t i = 0; i < f->n_tensors; i++) {
        GgufTensorInfo *t = &f->tensors[i];
        uint64_t abs_offset = data_start + t->data_offset;
        if (abs_offset + t->n_bytes < abs_offset || abs_offset + t->n_bytes > f->file_size) {
            fprintf(stderr, "FATAL: gguf_load: %s: tensor '%s' data range [%llu, %llu) exceeds file size %zu\n",
                    path, t->name, (unsigned long long)abs_offset, (unsigned long long)(abs_offset + t->n_bytes), f->file_size);
            exit(1);
        }
        t->data_offset = abs_offset;
    }

    return f;
}

void gguf_close(GgufFile *f) {
    if (!f) return;
    for (uint64_t i = 0; i < f->n_kv; i++) free(f->kv[i].key);
    free(f->kv);
    for (uint64_t i = 0; i < f->n_tensors; i++) free(f->tensors[i].name);
    free(f->tensors);
    munmap(f->base, f->file_size);
    close(f->fd);
    free(f);
}

static const GgufKV *find_kv(const GgufFile *f, const char *key) {
    for (uint64_t i = 0; i < f->n_kv; i++) if (strcmp(f->kv[i].key, key) == 0) return &f->kv[i];
    return NULL;
}

int gguf_kv_str(const GgufFile *f, const char *key, const char **out_ptr, uint64_t *out_len) {
    const GgufKV *kv = find_kv(f, key);
    if (!kv || kv->is_array || kv->type != GGUF_VTYPE_STRING) return 0;
    *out_ptr = kv->scalar.str.ptr; *out_len = kv->scalar.str.len; return 1;
}
int gguf_kv_u64(const GgufFile *f, const char *key, uint64_t *out) {
    const GgufKV *kv = find_kv(f, key);
    if (!kv || kv->is_array) return 0;
    switch (kv->type) {
        case GGUF_VTYPE_UINT8: case GGUF_VTYPE_UINT16: case GGUF_VTYPE_UINT32: case GGUF_VTYPE_UINT64:
            *out = kv->scalar.u; return 1;
        case GGUF_VTYPE_INT8: case GGUF_VTYPE_INT16: case GGUF_VTYPE_INT32: case GGUF_VTYPE_INT64:
            if (kv->scalar.i < 0) return 0;
            *out = (uint64_t)kv->scalar.i; return 1;
        default: return 0;
    }
}
int gguf_kv_i64(const GgufFile *f, const char *key, int64_t *out) {
    const GgufKV *kv = find_kv(f, key);
    if (!kv || kv->is_array) return 0;
    switch (kv->type) {
        case GGUF_VTYPE_INT8: case GGUF_VTYPE_INT16: case GGUF_VTYPE_INT32: case GGUF_VTYPE_INT64:
            *out = kv->scalar.i; return 1;
        case GGUF_VTYPE_UINT8: case GGUF_VTYPE_UINT16: case GGUF_VTYPE_UINT32: case GGUF_VTYPE_UINT64:
            *out = (int64_t)kv->scalar.u; return 1;
        default: return 0;
    }
}
int gguf_kv_f64(const GgufFile *f, const char *key, double *out) {
    const GgufKV *kv = find_kv(f, key);
    if (!kv || kv->is_array || (kv->type != GGUF_VTYPE_FLOAT32 && kv->type != GGUF_VTYPE_FLOAT64)) return 0;
    *out = kv->scalar.f; return 1;
}
int gguf_kv_bool(const GgufFile *f, const char *key, int *out) {
    const GgufKV *kv = find_kv(f, key);
    if (!kv || kv->is_array || kv->type != GGUF_VTYPE_BOOL) return 0;
    *out = kv->scalar.b; return 1;
}

const GgufTensorInfo *gguf_find_tensor(const GgufFile *f, const char *name) {
    for (uint64_t i = 0; i < f->n_tensors; i++) if (strcmp(f->tensors[i].name, name) == 0) return &f->tensors[i];
    return NULL;
}
const void *gguf_tensor_data(const GgufFile *f, const GgufTensorInfo *t) {
    return f->base + t->data_offset;
}
