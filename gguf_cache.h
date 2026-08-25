// gguf_cache.h -- Phase 2 sub-step 2 (PLAN_general_purpose_loader.md): on-disk transcode cache.
//
// Problem this solves: naive transcode re-dequantizes + re-RTN-quantizes every tensor on every
// process start, and keeps the mmap'd GGUF source, the transcoded K_Q4G64/K_Q8G64 buffers, and
// (once repacked) the SME2 RHS buffers all resident at once -- the "12+GB on a 16GB machine"
// risk the plan itself names. A `<model>.beglin` sidecar written on first load lets every
// subsequent load skip dequant+transcode entirely: the cache is mmap'd and WT fields point
// straight into it, zero malloc, zero recompute.
//
// Own TU for the same reason gguf_load.c/gguf_quants.c/gguf_transcode.c are: file I/O and
// mmap bookkeeping is exactly the kind of code that has no business inside qwen_infer.c's
// plain-compiled top-level TU (see gguf_load.h's header comment for the fuller rationale).

#ifndef GGUF_CACHE_H
#define GGUF_CACHE_H

#include <stdint.h>

typedef struct {
    char name[96];
    int32_t kind, out, in, ng;
    uint64_t data_offset, data_bytes;      // packed nibbles / int8 codes / raw f32, per `kind`
    uint64_t scales_offset, scales_bytes;  // 0/0 when this tensor has no scales (K_F32)
} GgufCacheEntry;

typedef struct GgufCacheFile GgufCacheFile;

// True iff a cache file exists at cache_path AND its recorded source size+mtime match
// src_gguf_path's current stat() -- i.e. the source GGUF hasn't changed since the cache was
// written. False (not FATAL) for "doesn't exist" or "stale" alike -- both mean "(re)build it".
int gguf_cache_is_valid(const char *cache_path, const char *src_gguf_path);

// Opens (mmaps) an already-validated cache. Caller must have checked gguf_cache_is_valid()
// first -- this FATALs on any I/O/format error, since reaching here means the caller already
// believes the file is good.
GgufCacheFile *gguf_cache_open(const char *cache_path);
uint32_t gguf_cache_count(const GgufCacheFile *c);
const GgufCacheEntry *gguf_cache_entry(const GgufCacheFile *c, uint32_t i);
const uint8_t *gguf_cache_base(const GgufCacheFile *c);  // mmap base; entry offsets are absolute file offsets into this

// ---- writer: one gguf_cache_writer_add() call per already-transcoded g_wt[] entry, in any
// order, then a single gguf_cache_writer_close() to finalize. `data`/`scales` are copied
// (fwrite) into the cache file -- no aliasing with the caller's buffers afterward. ----
typedef struct GgufCacheWriter GgufCacheWriter;
GgufCacheWriter *gguf_cache_writer_open(const char *cache_path, const char *src_gguf_path, uint32_t n_tensors);
void gguf_cache_writer_add(GgufCacheWriter *w, const char *name, int kind, int out, int in, int ng,
                            const void *data, uint64_t data_bytes,
                            const void *scales, uint64_t scales_bytes);
void gguf_cache_writer_close(GgufCacheWriter *w);

#endif // GGUF_CACHE_H
