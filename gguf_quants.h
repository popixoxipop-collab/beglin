// gguf_quants.h -- dequantization for GGUF tensor types. See gguf_quants.c for provenance
// (vendored from ggml, MIT license) and VENDOR.md for the full attribution record.
//
// No residual/error-feedback logic anywhere in this file, on purpose (same established
// reasoning as gguf_load.h / kai_sme2_repack_q4g64()): these functions decode an
// already-quantized blob produced upstream (by whatever tool wrote the GGUF file) into fp32.
// There is no quantization *choice* being made here to tune with a residual term -- dequant is
// the inverse of a frozen encoding, not a scheme with a tunable parameter.

#ifndef GGUF_QUANTS_H
#define GGUF_QUANTS_H

#include <stddef.h>
#include <stdint.h>
#include "gguf_load.h"  // GgmlType

// Dequantizes n_elements of `type` starting at `src` into `dst` (already-allocated, n_elements
// floats). FATAL if `type` isn't one of the types this file supports (F32, F16, BF16, Q4_0,
// Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4 -- MXFP4 added D-gptoss-1, real OpenAI GPT-OSS
// checkpoints) -- an unsupported type is a caller bug (should have checked
// gguf_dequant_supported() first), not something to silently skip or approximate.
void gguf_dequant_row(GgmlType type, const void *src, float *dst, int64_t n_elements);

// Whether gguf_dequant_row() supports this type. Callers should check this before attempting a
// dequant of a tensor type they haven't verified is covered yet (see
// PLAN_general_purpose_loader.md Phase 1 -- IQ*/Q2_K/Q2_0/etc. are parsed by gguf_load.c's
// container reader but not yet dequantizable).
int gguf_dequant_supported(GgmlType type);

// D-gptoss-9: public surface for MXFP4's own real E8M0-exponent-halving and E2M1-nibble LUT,
// so qwen_infer.c's zero-copy MoE row-decode branches (moe_decode_af()/moe_matvec_af_row())
// can reuse the exact same constants dequant_row_mxfp4() (this file) uses internally, instead
// of duplicating the table. Not used by anything in this file besides dequant_row_mxfp4()
// itself -- exposed purely for that external reuse.
float gguf_e8m0_to_fp32_half(uint8_t x);
int8_t gguf_mxfp4_nibble(int code);

#endif // GGUF_QUANTS_H
