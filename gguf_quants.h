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
// Q8_0, Q4_K, Q6_K) -- an unsupported type is a caller bug (should have checked
// gguf_dequant_supported() first), not something to silently skip or approximate.
void gguf_dequant_row(GgmlType type, const void *src, float *dst, int64_t n_elements);

// Whether gguf_dequant_row() supports this type. Callers should check this before attempting a
// dequant of a tensor type they haven't verified is covered yet (see
// PLAN_general_purpose_loader.md Phase 1 -- IQ*/Q2_K/Q3_K/Q5_K/Q2_0/etc. are parsed by
// gguf_load.c's container reader but not yet dequantizable).
int gguf_dequant_supported(GgmlType type);

#endif // GGUF_QUANTS_H
