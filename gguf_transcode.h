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

#endif // GGUF_TRANSCODE_H
