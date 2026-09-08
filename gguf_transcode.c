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
#include <stdio.h>

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


// D-qNg64-1: see gguf_transcode.h's own comment for the full format spec (bias-before-extract,
// LE bit order, reciprocal+rintf rounding, n=4 relationship to q4g64 above). Structurally this
// is q4g64's loop (per-row, per-group, sequential error-feedback over p=0..63) with the
// nibble-pack replaced by a bit-plane pack and the fixed 7/-8 bounds replaced by n-derived ones.
void gguf_quantize_qNg64(const float *w, int out, int in, int n,
                          uint8_t *planes_out, float *scales_out) {
    int ng = in / GGUF_TRANSCODE_GROUP;
    int qmax = (1 << (n - 1)) - 1;
    int qmin = -(1 << (n - 1));
    int bias = 1 << (n - 1);
    int mask = (1 << n) - 1;
    size_t group_pbytes = (size_t)n * 8;
    size_t row_pbytes = (size_t)ng * group_pbytes;
    long saturated = 0;
    for (int r = 0; r < out; r++) {
        const float *row = w + (size_t)r * in;
        uint8_t *prow = planes_out + (size_t)r * row_pbytes;
        float *srow = scales_out + (size_t)r * ng;
        for (int g = 0; g < ng; g++) {
            const float *grp = row + (size_t)g * GGUF_TRANSCODE_GROUP;
            uint8_t *pgrp = prow + (size_t)g * group_pbytes;
            for (size_t i = 0; i < group_pbytes; i++) pgrp[i] = 0;  // only OR'd into below
            float maxabs = 0.0f;
            for (int p = 0; p < GGUF_TRANSCODE_GROUP; p++) {
                float a = fabsf(grp[p]);
                if (a > maxabs) maxabs = a;
            }
            float scale = maxabs / (float)qmax;
            if (scale < 1e-12f) scale = 1.0f;
            srow[g] = scale;
            float inv = 1.0f / scale;
            float err = 0.0f;
            for (int p = 0; p < GGUF_TRANSCODE_GROUP; p++) {
                float x = grp[p] + err;
                float qf = rintf(x * inv);      // reciprocal + rintf -- see header comment
                if (qf > (float)qmax) { qf = (float)qmax; saturated++; }
                if (qf < (float)qmin) { qf = (float)qmin; saturated++; }
                float deq = qf * scale;
                err = x - deq;
                int code = (int)qf;
                int u = (code + bias) & mask;   // bias FIRST, then bit-extract below
                int byte_in_plane = p >> 3;
                int bit_in_byte = p & 7;
                for (int j = 0; j < n; j++) {
                    if ((u >> j) & 1) {
                        pgrp[(size_t)j * 8 + byte_in_plane] |= (uint8_t)(1u << bit_in_byte);
                    }
                }
            }
        }
    }
    // Cheap saturation counter (D-qNg64-1's own design review flagged EF-driven saturation to
    // the extreme code as a theoretical risk -- measured elsewhere at 0/51200 samples, a tail
    // event, not blocked here, just surfaced if it ever actually happens on real data).
    if (saturated > 0) {
        fprintf(stderr, "[gguf_transcode] qNg64(n=%d): %ld/%zu codes saturated to an extreme value\n",
                n, saturated, (size_t)out * (size_t)in);
    }
}
