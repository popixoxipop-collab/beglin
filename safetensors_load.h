// safetensors_load.h -- safetensors container parser. Deliberately its own translation unit,
// never included by qwen_infer.c's build unit (same "caller-plain convention" reasoning as
// gguf_load.h -- see that file's own header comment for the full SIGILL-history rationale).
//
// Format (https://github.com/huggingface/safetensors, verified against a real downloaded file
// this session, not assumed from the spec alone -- Qwen/Qwen2.5-0.5B's model.safetensors):
//   bytes[0:8]    little-endian u64 N = header JSON length in bytes
//   bytes[8:8+N]  UTF-8 JSON: {"tensor.name": {"dtype":"BF16","shape":[...],
//                 "data_offsets":[start,end]}, ..., "__metadata__": {...}} -- __metadata__ is
//                 an optional arbitrary string->string map, skipped, not surfaced by this parser
//   bytes[8+N:]   raw tensor data; each tensor's data_offsets are relative to THIS point (i.e.
//                 absolute_offset = 8+N+data_offsets[0]), tensors is NOT required to be
//                 name-sorted or offset-sorted in the header
//
// SAFETY CONTRACT (mirrors gguf_load.h's doctrine verbatim): every read is bounds-checked
// against the mmap'd file length. A truncated/malformed file or an out-of-range offset is a
// FATAL with a specific reason, never a best-effort partial parse -- this reads files it did
// not generate.
//
// No residual/error-feedback logic here, same reasoning as gguf_load.h: this file decodes an
// already-serialized tensor (BF16/F16/F32, already quantized-or-not upstream), it does not
// choose or tune a quantization scheme.
//
// JSON parsing: safetensors headers are machine-generated with a narrow, predictable grammar
// (one level of nesting: {name: {dtype, shape:[int,...], data_offsets:[int,int]}, ...},
// tensor names are plain identifiers, no escaped unicode in practice) -- this file hand-rolls a
// scanner for exactly that grammar rather than vendoring a general JSON library, same
// "hand-parse the exact format actually encountered" discipline gguf_load.c already uses for
// GGUF's binary KV format.
//
// Scope note: this is a container parser only (open/enumerate/get-raw-bytes). Wiring it into
// the engine's dense/MoE weight registration (role-name mapping, dequant-and-transcode,
// arch-config-from-a-companion-config.json) is separate, larger follow-on work -- not done
// here. Tensor names in a real HF safetensors checkpoint already match this engine's existing
// HF-style logical names exactly (confirmed against the real file above -- e.g.
// "model.layers.0.self_attn.q_proj.weight"), so that future step needs no role-mapping table,
// unlike GGUF's "blk.N.attn_q.weight" naming.

#ifndef SAFETENSORS_LOAD_H
#define SAFETENSORS_LOAD_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    ST_TYPE_UNKNOWN = 0,
    ST_TYPE_F64, ST_TYPE_F32, ST_TYPE_F16, ST_TYPE_BF16,
    ST_TYPE_I64, ST_TYPE_I32, ST_TYPE_I16, ST_TYPE_I8, ST_TYPE_U8, ST_TYPE_BOOL,
} SafetensorsType;

#define SAFETENSORS_MAX_DIMS 8

typedef struct {
    char *name;                       // NUL-terminated, malloc'd copy
    SafetensorsType dtype;
    uint32_t n_dims;
    uint64_t shape[SAFETENSORS_MAX_DIMS];
    uint64_t n_elements;              // product of shape[]
    uint64_t data_offset;             // absolute byte offset into the mmap'd file
    uint64_t n_bytes;                 // data_offsets[1] - data_offsets[0], already validated
} SafetensorsInfo;

typedef struct {
    int fd;
    uint8_t *base;                    // mmap base
    size_t file_size;
    uint64_t data_start;              // = 8 + header_len, every tensor's data_offset is relative to this
    SafetensorsInfo *tensors;
    uint64_t n_tensors;
} SafetensorsFile;

SafetensorsFile *safetensors_open(const char *path);
void safetensors_close(SafetensorsFile *f);

const SafetensorsInfo *safetensors_find_tensor(const SafetensorsFile *f, const char *name);
// Returns the raw tensor bytes (still in `dtype`'s on-disk encoding -- e.g. real BF16 bytes,
// not dequantized). Caller dequantizes via safetensors_dequant_row() in safetensors_quants.h
// (a future addition -- not included in this container-parser-only increment).
const void *safetensors_tensor_data(const SafetensorsFile *f, const SafetensorsInfo *t);

// Human-readable dtype name, for logging/error messages -- e.g. "BF16". Returns "UNKNOWN" for
// ST_TYPE_UNKNOWN or an out-of-range value.
const char *safetensors_type_name(SafetensorsType t);

#endif // SAFETENSORS_LOAD_H
