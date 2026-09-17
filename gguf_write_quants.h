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

#endif // GGUF_WRITE_QUANTS_H
