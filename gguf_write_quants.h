// gguf_write_quants.h -- F32 -> real, standard GGUF quant types (Q4_0/Q8_0). Mirrors the
// gguf_load.c/gguf_quants.c split on the read side: gguf_write.c handles the container
// mechanics (KV table, tensor-info table, alignment), this file handles the actual encode
// algorithms -- symmetric responsibilities, own translation unit for the same reason as every
// other file in this project's GGUF surface.
//
// D-export-1: NOT a copy-paste of gguf_transcode.c's gguf_quantize_q4g64_error_feedback()/
// gguf_quantize_q8g64() -- those target this engine's own 64-element-group, out-of-band-scale
// internal formats. Real GGUF Q4_0/Q8_0 use 32-element blocks with an inline fp16 scale at the
// head of each block (see gguf_quants.c's own GgmlBlockQ4_0/GgmlBlockQ8_0 -- the read-side
// spec reference these encoders must produce bytes compatible with). D-export-2: Q4_0 keeps
// this project's own real error-feedback diffusion technique (q4g64's own established
// accuracy edge over plain RTN); Q8_0 stays plain direct-division RTN, matching q8g64's own
// existing convention (which never used error feedback either).

#ifndef GGUF_WRITE_QUANTS_H
#define GGUF_WRITE_QUANTS_H

#include <stdint.h>
#include <stddef.h>

// Packed-byte size for `n` elements at each type (matches real GGUF block/typesize: 32
// elements/block, 18 bytes/block for Q4_0, 34 bytes/block for Q8_0). `n` must be a multiple
// of 32 -- callers needing padding decide that themselves (real row widths in this engine are
// already multiples of 32 for every tensor that would plausibly be exported this way).
size_t gguf_w_q4_0_nbytes(int64_t n);
size_t gguf_w_q8_0_nbytes(int64_t n);

// Quantizes `n` (a multiple of 32) F32 values from `w` into `out` (must be at least
// gguf_w_q4_0_nbytes(n)/gguf_w_q8_0_nbytes(n) bytes), in real GGUF Q4_0/Q8_0 block layout.
void gguf_w_quantize_q4_0(const float *w, int64_t n, uint8_t *out);
void gguf_w_quantize_q8_0(const float *w, int64_t n, uint8_t *out);

// D-export-4 (K-quant follow-up): real GGUF Q4_K, 256-element super-blocks (8 sub-blocks of
// 32, 6-bit-packed per-sub-block scale+min, matching real GGML_TYPE_Q4_K's own container
// exactly -- see gguf_write_quants.c's header comment for the byte layout and the one real
// algorithmic simplification this encoder makes vs. llama.cpp's own optimizer). `n` must be a
// multiple of 256.
size_t gguf_w_q4_k_nbytes(int64_t n);
void gguf_w_quantize_q4_k(const float *w, int64_t n, uint8_t *out);

// D-export-5 (F32-tier follow-up): real GGUF Q5_0, 32-element blocks, inline fp16 scale +
// 5-bit symmetric codes (4-bit nibble in qs[] + 1 high bit in qh[], real GGML_TYPE_Q5_0
// layout). `n` must be a multiple of 32. Used for embed_tokens.weight at export -- matches
// the REAL source Q4_K_M checkpoint's own choice for this exact tensor (verified via gguf-py
// against the real file, not assumed), a deliberately higher-than-4-bit precision for
// embeddings specifically. Same error-feedback (residual) diffusion technique as
// gguf_w_quantize_q4_0() above, applied per 32-element block before the 5-bit code is packed.
size_t gguf_w_q5_0_nbytes(int64_t n);
void gguf_w_quantize_q5_0(const float *w, int64_t n, uint8_t *out);

// D-export-6 (real-recipe-parity follow-up): real GGUF Q6_K, 256-element super-blocks, 16
// sub-groups of 16 with an int8 per-group scale, symmetric (no separate min, unlike Q4_K/
// Q5_K -- ggml's own real Q3_K/Q6_K convention). `n` must be a multiple of 256. Used for
// ffn_down.weight at export -- matches the REAL source Q4_K_M checkpoint's own choice for
// that exact tensor (verified via gguf-py against the real file: type=14=Q6_K, not the Q4_K
// this project's own D-export-4 had assumed/used). Same error-feedback (residual) diffusion
// technique as gguf_w_quantize_q4_0()/_q5_0() above, applied per 16-element sub-group.
size_t gguf_w_q6_k_nbytes(int64_t n);
void gguf_w_quantize_q6_k(const float *w, int64_t n, uint8_t *out);

#endif // GGUF_WRITE_QUANTS_H
