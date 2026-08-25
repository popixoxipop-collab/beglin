// gguf_load.h -- GGUF (v3) container parser. Deliberately its own translation unit, never
// included by qwen_infer.c's build unit (see RESULTS.md's "caller-plain convention": adding
// unrelated code to the plain-compiled top-level TU already once changed clang's
// autovectorization of an unrelated function and produced a SIGILL -- this file's byte-swap/
// bit-walk loops are exactly that code shape).
//
// SAFETY CONTRACT (mirrors qwen_infer.c's load_int4 doctrine, stated there as: "an
// unrecognized kind used to silently fall through ... is silent weight corruption"): every
// read in gguf_load.c is bounds-checked against the mmap'd file length. A truncated or
// malformed file is a FATAL with a specific reason, never a best-effort partial parse -- this
// is a published package that will be pointed at files it did not generate.
//
// No residual/error-feedback logic anywhere in this file, on purpose (same reasoning already
// established for this project's other "decode an already-quantized blob" code --
// kai_sme2_repack_q4g64()'s header comment, f16lhs_bench.c's file comment): GGML_TYPE_Q4_K
// below is a container-format type TAG, not a quantization scheme this file chooses or tunes.
// The bytes it names were already quantized upstream (by whatever produced the GGUF file) and
// arrive frozen; this file's job is reading which type tag a tensor carries, not deciding how
// to quantize anything. A residual/error-feedback term is a training-time technique for a
// quantization scheme with a tunable choice in it -- there is no such choice here to tune.
//
// This header intentionally does NOT expose array-of-string KV values (tokenizer vocab/
// merges) beyond their raw byte-range -- see the project's own D-gen-5 (tokenizer is
// explicitly out of scope through Phase 5). The parser still walks over them correctly to
// reach subsequent KV entries; it just doesn't materialize them.

#ifndef GGUF_LOAD_H
#define GGUF_LOAD_H

#include <stddef.h>
#include <stdint.h>

// GGUF spec value type ids (ground truth: llama.cpp gguf-py's GGUFValueType enum, confirmed
// against the actual fixture this project tests against -- not guessed from memory).
typedef enum {
    GGUF_VTYPE_UINT8   = 0,
    GGUF_VTYPE_INT8    = 1,
    GGUF_VTYPE_UINT16  = 2,
    GGUF_VTYPE_INT16   = 3,
    GGUF_VTYPE_UINT32  = 4,
    GGUF_VTYPE_INT32   = 5,
    GGUF_VTYPE_FLOAT32 = 6,
    GGUF_VTYPE_BOOL    = 7,
    GGUF_VTYPE_STRING  = 8,
    GGUF_VTYPE_ARRAY   = 9,
    GGUF_VTYPE_UINT64  = 10,
    GGUF_VTYPE_INT64   = 11,
    GGUF_VTYPE_FLOAT64 = 12,
} GgufValueType;

// ggml tensor quantization type ids (ground truth: gguf-py's GGMLQuantizationType enum,
// confirmed against the actual fixture -- this project's dequant support (Phase 1 follow-on)
// only covers a subset; the parser itself is type-agnostic, it just records the id).
typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_BF16 = 30,
} GgmlType;

typedef struct {
    char *key;              // NUL-terminated, malloc'd copy
    GgufValueType type;      // scalar type, or element type if is_array
    int is_array;
    uint64_t arr_len;        // 1 if !is_array

    union {
        uint64_t u; int64_t i; double f; int b;
        struct { const char *ptr; uint64_t len; } str;  // points into the mmap, NOT NUL-terminated
    } scalar;
} GgufKV;

typedef struct {
    char *name;              // NUL-terminated, malloc'd copy
    uint32_t n_dims;
    uint64_t ne[4];           // ggml order: ne[0] is the fastest-varying dimension
    GgmlType type;
    uint64_t n_elements;
    uint64_t n_bytes;         // computed from type+n_elements, per-type block size
    uint64_t data_offset;     // absolute byte offset into the mmap'd file
} GgufTensorInfo;

typedef struct {
    int fd;
    uint8_t *base;            // mmap base
    size_t file_size;
    uint32_t version;
    uint64_t alignment;        // from "general.alignment" KV if present, else 32 (GGUF default)
    GgufKV *kv;
    uint64_t n_kv;
    GgufTensorInfo *tensors;
    uint64_t n_tensors;
} GgufFile;

GgufFile *gguf_open(const char *path);
void gguf_close(GgufFile *f);

int gguf_kv_str(const GgufFile *f, const char *key, const char **out_ptr, uint64_t *out_len);
int gguf_kv_u64(const GgufFile *f, const char *key, uint64_t *out);
int gguf_kv_i64(const GgufFile *f, const char *key, int64_t *out);
int gguf_kv_f64(const GgufFile *f, const char *key, double *out);
int gguf_kv_bool(const GgufFile *f, const char *key, int *out);

const GgufTensorInfo *gguf_find_tensor(const GgufFile *f, const char *name);
const void *gguf_tensor_data(const GgufFile *f, const GgufTensorInfo *t);

#endif // GGUF_LOAD_H
