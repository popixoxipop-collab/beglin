// gguf_write.h -- GGUF (v3) container writer. Own translation unit, same "caller-plain
// convention" reasoning as gguf_load.h (see its own header comment) -- this file's byte-packing
// loops are the same code shape that once mis-autovectorized when folded into qwen_infer.c's
// plain-compiled TU.
//
// D-export-3: mirrors gguf_load.h's own type enums (GgufValueType/GgmlType, included from
// there directly) and KV-accessor shape, but for writing -- not gguf_cache.c's `.beglin`/
// BEGLINC2 ad hoc cache format, which uses a different alignment (64 vs GGUF's 32-default),
// fixed-size names (not length-prefixed strings), and no KV table at all. See
// PLAN_general_purpose_loader.md's Phase 5 / RESULTS.md's D-export-N for the real design
// reasoning.
//
// Two-pass by construction: gguf_w_kv_*()/gguf_w_add_tensor() only buffer -- nothing is
// written to disk until gguf_w_finish(), which needs every tensor's final byte size known
// before it can compute the tensor-info table's offsets (which are written before the data
// section they describe).
//
// Ownership: KV *scalar* string values are copied (malloc'd duplicate) at call time -- cheap,
// and callers often pass short-lived local buffers for these. KV *array* values and tensor
// *data* buffers are NOT copied -- the caller must keep them valid until gguf_w_finish()
// returns. This is deliberate: D-export-5's whole point is copying tokenizer arrays straight
// from an already-open source GgufFile's own mmap (kept alive for the writer's whole lifetime
// anyway), and tensor data buffers are typically large (multi-MB/GB) requantized rows a caller
// already owns -- copying them again would double peak memory for no reason.

#ifndef GGUF_WRITE_H
#define GGUF_WRITE_H

#include <stddef.h>
#include <stdint.h>
#include "gguf_load.h"   // reuses GgufValueType/GgmlType/GgufStr -- one definition, not two

typedef struct GgufWriter GgufWriter;

// Opens `path` for writing (truncates if it exists). Returns NULL on open failure (caller
// checks errno). Nothing is written to disk yet -- see the two-pass note above.
GgufWriter *gguf_w_open(const char *path);

// Scalar KV writers. `key` is copied. String values are copied; everything else is by value.
// Calling any of these with key=="general.alignment" (u32 or u64 variant) additionally updates
// this writer's own data-section alignment (default 32, matching GGUF's spec default) -- same
// special-case gguf_load.c's read side already applies to this exact key.
void gguf_w_kv_str(GgufWriter *w, const char *key, const char *val, uint64_t val_len);
void gguf_w_kv_u32(GgufWriter *w, const char *key, uint32_t val);
void gguf_w_kv_u64(GgufWriter *w, const char *key, uint64_t val);
void gguf_w_kv_i64(GgufWriter *w, const char *key, int64_t val);
void gguf_w_kv_f32(GgufWriter *w, const char *key, float val);
void gguf_w_kv_f64(GgufWriter *w, const char *key, double val);
void gguf_w_kv_bool(GgufWriter *w, const char *key, int val);

// D-export-3 (Phase 3): writes a scalar KV under its EXACT original GGUF type tag (UINT8/
// INT8/UINT16/INT16/UINT32/INT32/UINT64/INT64/BOOL only -- FATALs on FLOAT32/FLOAT64/STRING/
// ARRAY, use the dedicated functions above for those). For a passthrough copy of a source
// KV read via gguf_load.h's GgufKV (whose scalar union already upcasts every integer width to
// u/i uint64_t/int64_t, per its own comment), pass `type` from the source KV and `bits` as
// that same union's `.u` (unsigned types) or `.i` reinterpreted as uint64_t (signed types) --
// truncating to the narrower original width on write reproduces the exact original bit
// pattern for both signed and unsigned integers.
void gguf_w_kv_int_raw(GgufWriter *w, const char *key, GgufValueType type, uint64_t bits);

// Array KV writers. `key` is copied; `arr`/`n` are NOT (see ownership note above) -- for
// gguf_w_kv_str_array, each GgufStr's own .ptr must also stay valid until gguf_w_finish().
void gguf_w_kv_str_array(GgufWriter *w, const char *key, const GgufStr *arr, uint64_t n);
void gguf_w_kv_i32_array(GgufWriter *w, const char *key, const int32_t *arr, uint64_t n);
void gguf_w_kv_f32_array(GgufWriter *w, const char *key, const float *arr, uint64_t n);

// Registers one tensor. `name` is copied; `data` is NOT (see ownership note above). `ne` must
// have exactly `n_dims` valid entries (1..4), ggml order (ne[0] = fastest-varying). `nbytes`
// must match `type`'s real block-encoded size for the given element count -- not re-derived
// here (the caller already computed it while requantizing; re-deriving would just be a second
// place for that formula to drift from the first).
void gguf_w_add_tensor(GgufWriter *w, const char *name, GgmlType type,
                        uint32_t n_dims, const uint64_t *ne,
                        const void *data, uint64_t nbytes);

// Writes the whole file (magic/version/counts, KV table, tensor-info table, aligned data
// section, in that order) and closes it. Returns 1 on success, 0 on write failure (caller
// checks errno). `w` is freed either way -- do not use it again after this call.
int gguf_w_finish(GgufWriter *w);

#endif // GGUF_WRITE_H
