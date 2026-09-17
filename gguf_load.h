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
// D-tok (Phase 6, tokenizer): array-of-string/int32/float32 KV materialization (tokenizer
// vocab/merges/scores/token_type, or any other array KV) IS exposed, via gguf_kv_str_array()/
// gguf_kv_i32_array()/gguf_kv_f32_array() below -- this was the one deliberate gap D-gen-5 left
// (see PLAN_general_purpose_loader.md's D-gen-5 for why it was deferred, not why it's absent;
// it's no longer absent). Fixed-width element arrays are zero-copy (a raw pointer straight into
// the mmap, same "caller casts per kv->type" convention gguf_tensor_data() already uses for
// tensor payloads); string arrays are a malloc'd array of {ptr-into-mmap, len} pairs, since
// string byte offsets aren't a fixed stride and can't be zero-copy-indexed.

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
    // D-gptoss-1: MXFP4, real value/layout confirmed by hand-parsing the actual
    // ggml-org/gpt-oss-20b-GGUF header (gpt-oss-20b-MXFP4.gguf) rather than assumed from
    // llama.cpp source alone -- OCP microscaling FP4: 32 values/block, 17 bytes/block (1 byte
    // shared E8M0 exponent + 16 bytes of packed E2M1 nibbles, 2 values/byte). No residual/
    // error-feedback applies here, same D-gen-9 exemption already used for Q3_K/Q5_K: this is
    // a fixed upstream (OCP/OpenAI) encoding this engine only decodes, not a quantizer this
    // project designs -- there is no requantization step to add error-feedback to.
    GGML_TYPE_MXFP4 = 39,
} GgmlType;

// Shared string-view shape: pointer into the mmap + length, NOT NUL-terminated. Used both for
// a scalar STRING value and for each element of a STRING array (GgufKV.arr_str below).
typedef struct { const char *ptr; uint64_t len; } GgufStr;

typedef struct {
    char *key;              // NUL-terminated, malloc'd copy
    GgufValueType type;      // scalar type, or element type if is_array
    int is_array;
    uint64_t arr_len;        // 1 if !is_array

    union {
        uint64_t u; int64_t i; double f; int b;
        GgufStr str;
    } scalar;

    // Only meaningful when is_array (see gguf_kv_str_array()/gguf_kv_i32_array()/
    // gguf_kv_f32_array() below). For elem type STRING, arr_str is a malloc'd array of
    // arr_len GgufStr. For every other (fixed-width) elem type, arr_fixed points directly
    // into the mmap at the array's first element in the file's own native packed layout
    // (zero-copy) -- caller casts per `type` the same way gguf_tensor_data() callers already
    // cast tensor payloads. NULL/unused when !is_array.
    GgufStr *arr_str;
    const void *arr_fixed;
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

// Array KV accessors (D-tok, Phase 6) -- each returns 0 if the key is missing, not an array,
// or the array's element type doesn't match. out_arr/out_len are only written on success (1).
int gguf_kv_str_array(const GgufFile *f, const char *key, const GgufStr **out_arr, uint64_t *out_len);
int gguf_kv_i32_array(const GgufFile *f, const char *key, const int32_t **out_arr, uint64_t *out_len);
int gguf_kv_f32_array(const GgufFile *f, const char *key, const float **out_arr, uint64_t *out_len);

const GgufTensorInfo *gguf_find_tensor(const GgufFile *f, const char *name);
const void *gguf_tensor_data(const GgufFile *f, const GgufTensorInfo *t);

#endif // GGUF_LOAD_H
