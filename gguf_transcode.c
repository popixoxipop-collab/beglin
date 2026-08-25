// gguf_transcode.c -- Phase 2 transcode: F32 (already dequantized by gguf_quants.c) -> this
// engine's own K_Q4G64 / K_Q8G64 formats.
//
// PORTED, not reinvented, from eval/quantize_int4.py's quant_group_ef() / quant_group_int8() --
// the same two functions that produced qwen15b_int4g64.bin, the packed blob this project's own
// documented ppl 12.10 / +13.6% numbers (RESULTS.md) come from. Arithmetic order (per-row,
// per-group, sequential position 0..63 with the error_feedback residual threaded through that
// same order) is kept identical to the Python reference on purpose -- this is what let the
// oracle check (tools/gguf_transcode_oracle.py vs tools/gguf_transcode_dump.c) verify an exact
// bitwise match on real dequantized GGUF tensor data, not just "close enough" (see RESULTS.md
// General-purpose loader -- Phase 2 sub-step 1 for the actual verification run).
//
// error_feedback (D8 in quantize_int4.py): `err` below is exactly that residual term -- carried
// forward WITHIN one (row, group) at a time as p advances 0..63, reset to 0 at the start of
// every new group. This is not omitted or approximated; it is the same 1-D diffusion the
// existing offline pipeline already uses and this project has already measured lowers
// quantization MSE (quantize_int4.py's own header comment, D8).

#include "gguf_transcode.h"
#include <math.h>
#include <stddef.h>

void gguf_quantize_q4g64_error_feedback(const float *w, int out, int in,
                                         uint8_t *packed_out, float *scales_out) {
    int ng = in / GGUF_TRANSCODE_GROUP;
    for (int r = 0; r < out; r++) {
        const float *row = w + (size_t)r * in;
        uint8_t *prow = packed_out + (size_t)r * (in / 2);
        float *srow = scales_out + (size_t)r * ng;
        for (int g = 0; g < ng; g++) {
            const float *grp = row + (size_t)g * GGUF_TRANSCODE_GROUP;
            float maxabs = 0.0f;
            for (int p = 0; p < GGUF_TRANSCODE_GROUP; p++) {
                float a = fabsf(grp[p]);
                if (a > maxabs) maxabs = a;
            }
            float scale = maxabs / 7.0f;
            if (scale < 1e-12f) scale = 1.0f;
            srow[g] = scale;
            float inv = 1.0f / scale;
            float err = 0.0f;
            for (int p = 0; p < GGUF_TRANSCODE_GROUP; p++) {
                float x = grp[p] + err;
                // rintf(), not roundf(): roundf() is C99 round-half-away-from-zero, but
                // numpy's np.round() (the Python reference this ports) is IEEE754
                // round-half-to-even -- rintf() follows the current FP rounding mode, which
                // defaults to round-to-nearest-even, matching numpy. Found via the R4 oracle
                // (exact max-abs-diff of 1 at ~0.1% of positions -- the classic tie-break
                // signature -- not a logic bug) before this fix, zero-diff after.
                float qf = rintf(x * inv);
                if (qf > 7.0f) qf = 7.0f;
                if (qf < -8.0f) qf = -8.0f;
                float deq = qf * scale;
                err = x - deq;
                int code = (int)qf;                 // [-8,7]
                int idx = g * GGUF_TRANSCODE_GROUP + p;
                int nib = (code + 8) & 0x0F;         // [0,15]
                int byte_i = idx / 2;
                if ((idx & 1) == 0) {
                    prow[byte_i] = (uint8_t)nib;                   // low nibble, even col
                } else {
                    prow[byte_i] = (uint8_t)(prow[byte_i] | (nib << 4)); // high nibble, odd col
                }
            }
        }
    }
}

void gguf_quantize_q8g64(const float *w, int out, int in,
                          int8_t *codes_out, float *scales_out) {
    int ng = in / GGUF_TRANSCODE_GROUP;
    for (int r = 0; r < out; r++) {
        const float *row = w + (size_t)r * in;
        int8_t *crow = codes_out + (size_t)r * in;
        float *srow = scales_out + (size_t)r * ng;
        for (int g = 0; g < ng; g++) {
            const float *grp = row + (size_t)g * GGUF_TRANSCODE_GROUP;
            float maxabs = 0.0f;
            for (int p = 0; p < GGUF_TRANSCODE_GROUP; p++) {
                float a = fabsf(grp[p]);
                if (a > maxabs) maxabs = a;
            }
            float scale = maxabs / 127.0f;
            if (scale < 1e-12f) scale = 1.0f;
            srow[g] = scale;
            // Direct division, NOT a precomputed reciprocal multiply: quant_group_int8() in
            // Python divides by `scale` directly (no `inv`, unlike quant_group_ef()'s q4 path
            // above). A reciprocal multiply double-rounds (1/scale rounds once, the product
            // rounds again) and is not bit-identical to a single division -- found via the R4
            // oracle as a second, independent source of off-by-one diffs after the rintf fix
            // alone didn't reach zero-diff.
            for (int p = 0; p < GGUF_TRANSCODE_GROUP; p++) {
                float qf = rintf(grp[p] / scale);
                if (qf > 127.0f) qf = 127.0f;
                if (qf < -127.0f) qf = -127.0f;
                crow[g * GGUF_TRANSCODE_GROUP + p] = (int8_t)qf;
            }
        }
    }
}
