// safetensors_quants.h -- widens a safetensors-sourced tensor row (already-known dtype: F32,
// F16, or BF16 -- the dtypes a real HF checkpoint actually ships weights in) to F32. Mirrors
// gguf_quants.h's dequant_row/dequant_supported split, but for safetensors' dtype set rather than
// GGUF's block-quantized types (safetensors is a raw-tensor container, not a quantization format
// -- there is no block/K-quant decode to port here, unlike gguf_quants.c).
//
// Own translation unit (never #include'd into qwen_infer.c's plain-compiled build unit), matching
// every other format-specific parser in this project -- see gguf_load.h's header comment for the
// full rationale.
//
// This exact filename/function name was anticipated in safetensors_load.h's own header comment
// when the container-parser-only increment shipped: "Caller dequantizes via
// safetensors_dequant_row() in safetensors_quants.h (a future addition)."

#ifndef SAFETENSORS_QUANTS_H
#define SAFETENSORS_QUANTS_H

#include <stdint.h>
#include "safetensors_load.h"

// True for the dtypes this file can widen (F32/F16/BF16). False for everything else
// (I64/I32/I16/I8/U8/BOOL/F64) -- callers should check this before calling
// safetensors_dequant_row() and FATAL themselves with tensor-name context, matching
// gguf_dequant_supported()'s own split (the check and the FATAL live in different places on
// purpose: this file doesn't know the tensor's name, the caller does).
int safetensors_dequant_supported(SafetensorsType dtype);

// Widens `n` elements of `raw` (already known to be dtype `dtype`) into `out` as F32. Caller must
// have checked safetensors_dequant_supported(dtype) first -- FATALs on an unsupported dtype
// rather than silently producing garbage.
void safetensors_dequant_row(SafetensorsType dtype, const void *raw, float *out, uint64_t n);

#endif // SAFETENSORS_QUANTS_H
