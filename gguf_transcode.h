// gguf_transcode.h -- Phase 2 (PLAN_general_purpose_loader.md, D-gen-2 Path A): re-quantize an
// already-dequantized (F32) GGUF tensor into this engine's own K_Q4G64/K_Q8G64 formats, so it
// can run through the SME2/NEON group-64 kernel families instead of the F32/BLAS fallback.
//
// This is genuinely a different concern from gguf_quants.c (which only ever decodes an
// upstream-frozen encoding, on purpose, per that file's own header comment) -- this file makes
// a quantization CHOICE, with a tunable residual term, so it lives in its own TU.
//
// Algorithm is a direct port of eval/quantize_int4.py's quant_group_ef() / quant_group_int8()
// (the same functions that produced this project's own documented ppl numbers), not a
// reinvention -- see gguf_transcode.c's header comment for the port notes and the oracle
// verification this was checked against.

#ifndef GGUF_TRANSCODE_H
#define GGUF_TRANSCODE_H

#include <stdint.h>

// Group size for both formats -- fixed at 64 throughout this codebase (K_Q4G64/K_Q8G64 naming,
// SME2_KAI_BL), not a tunable here.
#define GGUF_TRANSCODE_GROUP 64

// Symmetric int4 group-64 RTN quantization WITH error feedback (D8 in quantize_int4.py: 1-D
// error_feedback diffusion within each group, carrying err=(x+err)-q*scale forward across the
// group's 64 positions -- the same error_feedback residual term, not omitted). `w` is [out][in]
// row-major fp32 (in % GGUF_TRANSCODE_GROUP == 0, caller-checked). `packed_out` must be
// out*(in/2) bytes (nibble = code+8 in [0,15], byte = low(even col) | high(odd col)<<4 --
// byte-identical layout to kai_sme2_repack_q4g64()'s expected input). `scales_out` must be
// out*(in/GGUF_TRANSCODE_GROUP) floats.
void gguf_quantize_q4g64_error_feedback(const float *w, int out, int in,
                                         uint8_t *packed_out, float *scales_out);

// Symmetric int8 group-64 RTN quantization, no error feedback needed at 8 bits (matches
// quant_group_int8() -- used only for the untied lm_head, per D7/D17's "near-lossless without
// the error_feedback residual term" finding, already measured in this project, not assumed).
// `codes_out` must be out*in int8_t; `scales_out` must be out*(in/GGUF_TRANSCODE_GROUP) floats.
void gguf_quantize_q8g64(const float *w, int out, int in,
                          int8_t *codes_out, float *scales_out);

// qNg64 (D-qNg64-1): arbitrary-n bit-plane group-64 quantization, symmetric, WITH error
// feedback -- a direct generalization of gguf_quantize_q4g64_error_feedback() above, NOT of
// gguf_quantize_q8g64() (q8g64 clamps asymmetrically to [-127,127] and uses plain division
// with no error feedback; qNg64(n=8) would clamp to [-128,127] with reciprocal-multiply and EF
// -- the two are deliberately different at n=8, do not assume interchangeability). Intended for
// n in [2,7] (n=4 and n=8 already have their own dedicated, faster formats above; this family
// exists for the bit-widths that don't). Per-element MSE is measurably HIGHER with error
// feedback than without (~2x, a direct consequence of differencing residual noise) -- the real
// benefit error feedback gives is a ~60x reduction in per-GROUP summed error, which is what a
// group-64 dot product actually accumulates. (The comment on q4g64 above claiming EF simply
// "lowers quantization MSE" is imprecise for exactly this reason -- don't copy that phrasing
// here.)
//
// Code range is [-2^(n-1), 2^(n-1)-1], scale = maxabs/(2^(n-1)-1) (same 1e-12 floor convention
// as both formats above). Rounding is rintf() on a precomputed reciprocal (`x * (1/scale)`),
// the SAME convention q4g64 uses -- NOT q8g64's direct division. Values are stored BIASED
// (u = (code + 2^(n-1)) & (2^n-1), matching q4g64's own `(code+8)&0x0F`) then bit-plane packed:
// for n=4, this is byte-identical in VALUE (not layout) to what gguf_quantize_q4g64_error_
// feedback() computes -- same codes, same scales -- differing only in how those codes are
// packed into bytes.
//
// Packing layout (pin this exactly -- a pack/unpack round-trip test cannot catch a
// self-consistent but wrong bit-order, so any reader porting this to another language must
// match it exactly, not just "something round-trippable"): a group of 64 elements occupies
// exactly 8*n bytes = n "planes" of 8 bytes (64 bits) each, laid out consecutively (plane 0's
// 8 bytes, then plane 1's, ... then plane n-1's) within the group; groups are laid out
// consecutively within a row (row stride = (in/64)*n*8 bytes). Within plane j, byte b holds
// bits for element indices 8b..8b+7 of that group, and element i's bit lives at bit (i & 7) of
// that byte (little-endian within the byte -- element 0 is the LSB). Plane j holds bit j of
// each element's BIASED code u (i.e. bit j of `u`, not of the raw two's-complement `code`).
// `planes_out` must be out*(in/64)*n*8 bytes; `scales_out` must be out*(in/64) floats, same
// shape as the two formats above.
void gguf_quantize_qNg64(const float *w, int out, int in, int n,
                          uint8_t *planes_out, float *scales_out);

#endif // GGUF_TRANSCODE_H
