// gguf_write_quants.c -- see gguf_write_quants.h. Own translation unit, same reason as every
// other file in this project's GGUF surface.

#include "gguf_write_quants.h"
#include <math.h>
#include <string.h>

#define QK4_0 32
#define QK8_0 32
typedef uint16_t ggml_half;

#pragma pack(push, 1)
typedef struct { ggml_half d; uint8_t qs[QK4_0 / 2]; } WBlockQ4_0;
typedef struct { ggml_half d; int8_t  qs[QK8_0];     } WBlockQ8_0;
#pragma pack(pop)

// Same __fp16-based conversion convention gguf_quants.c's own fp16_to_fp32() uses (compiler/
// hardware-native IEEE754 half conversion, not a hand-rolled bit-twiddling reimplementation).
static ggml_half fp32_to_fp16(float v) {
    union { uint16_t u; __fp16 f; } cvt;
    cvt.f = (__fp16)v;
    return cvt.u;
}

size_t gguf_w_q4_0_nbytes(int64_t n) { return (size_t)(n / QK4_0) * sizeof(WBlockQ4_0); }
size_t gguf_w_q8_0_nbytes(int64_t n) { return (size_t)(n / QK8_0) * sizeof(WBlockQ8_0); }

// Ported from gguf_transcode.c's gguf_quantize_q4g64_error_feedback() -- same error-feedback
// diffusion (D-export-2's own reasoning: the GGUF container doesn't care how codes were
// derived), re-blocked to GGUF's real 32-element block with an inline fp16 scale, and
// re-packed split-half (byte j's low nibble = element j, high nibble = element j+16) instead
// of q4g64's own consecutive-pair layout -- a real, deliberate difference from that function,
// matching real GGML_TYPE_Q4_0's own dequant convention (gguf_quants.c's dequant_row_q4_0()).
void gguf_w_quantize_q4_0(const float *w, int64_t n, uint8_t *out) {
    int64_t nb = n / QK4_0;
    WBlockQ4_0 *blocks = (WBlockQ4_0 *)out;
    for (int64_t b = 0; b < nb; b++) {
        const float *grp = w + b * QK4_0;
        float maxabs = 0.0f;
        for (int p = 0; p < QK4_0; p++) { float a = fabsf(grp[p]); if (a > maxabs) maxabs = a; }
        float scale = maxabs / 7.0f;
        if (scale < 1e-12f) scale = 1.0f;
        blocks[b].d = fp32_to_fp16(scale);
        float inv = 1.0f / scale;
        float err = 0.0f;
        int8_t codes[QK4_0];
        for (int p = 0; p < QK4_0; p++) {
            float x = grp[p] + err;
            float qf = rintf(x * inv);   // round-half-to-even, matches q4g64's own rintf() choice
            if (qf > 7.0f) qf = 7.0f;
            if (qf < -8.0f) qf = -8.0f;
            float deq = qf * scale;
            err = x - deq;
            codes[p] = (int8_t)qf;
        }
        for (int j = 0; j < QK4_0 / 2; j++) {
            uint8_t lo = (uint8_t)((codes[j] + 8) & 0x0F);
            uint8_t hi = (uint8_t)((codes[j + QK4_0/2] + 8) & 0x0F);
            blocks[b].qs[j] = (uint8_t)(lo | (hi << 4));
        }
    }
}

// Ported from gguf_transcode.c's gguf_quantize_q8g64() -- same plain-RTN direct-division
// (no error feedback, matching that function's own established choice), re-blocked to GGUF's
// real 32-element block with an inline fp16 scale instead of q8g64's out-of-band fp32 array.
void gguf_w_quantize_q8_0(const float *w, int64_t n, uint8_t *out) {
    int64_t nb = n / QK8_0;
    WBlockQ8_0 *blocks = (WBlockQ8_0 *)out;
    for (int64_t b = 0; b < nb; b++) {
        const float *grp = w + b * QK8_0;
        float maxabs = 0.0f;
        for (int p = 0; p < QK8_0; p++) { float a = fabsf(grp[p]); if (a > maxabs) maxabs = a; }
        float scale = maxabs / 127.0f;
        if (scale < 1e-12f) scale = 1.0f;
        blocks[b].d = fp32_to_fp16(scale);
        for (int p = 0; p < QK8_0; p++) {
            float qf = rintf(grp[p] / scale);   // direct division, not reciprocal -- matches q8g64's own R4-oracle-found requirement
            if (qf > 127.0f) qf = 127.0f;
            if (qf < -127.0f) qf = -127.0f;
            blocks[b].qs[p] = (int8_t)qf;
        }
    }
}

// D-export-4: real GGUF Q4_K. 256-element super-blocks, 8 sub-blocks of 32, asymmetric
// affine per sub-block (value = d*sc[j]*code - dmin*m[j]*1, i.e. code=0 decodes to -dmin*m[j]
// not 0 -- unlike Q4_0's symmetric code*scale). Byte layout ({d,dmin,scales[12],qs[128]},
// 144B/super-block) and the scales[12] 6-bit packing scheme (get_scale_min_k4) are ported
// VERBATIM from ggml's real quantize_row_q4_K_ref()/dequantize_row_q4_K() (ggml-quants.c) --
// this part is the container format itself, not an algorithm choice, so it must match exactly
// for gguf_dequant_row(GGML_TYPE_Q4_K,...)/gguf-py/llama.cpp to decode these bytes correctly.
//
// The ONE deliberate deviation from ggml's own encoder (same D-export-2 stance as Q4_0/Q8_0
// above -- the container doesn't care how the codes were chosen): ggml's reference picks each
// sub-block's scale/min via make_qkx2_quants(), an iterative weighted-error search whose exact
// algorithm was not available to port faithfully. This encoder instead derives sub-block
// scale/min directly from each sub-block's real min/max (clamping min to <=0 so the packed
// "min" term, which must fit an unsigned 6-bit field, is always representable), then applies
// this project's own established error-feedback (residual) diffusion per sub-block when
// choosing the final 4-bit codes -- not a bit-exact replica of ggml's own quantizer, but a
// real, valid, spec-conformant Q4_K encoder any real Q4_K reader decodes correctly. Expected
// consequence: slightly worse fidelity than ggml's own optimizer-driven Q4_K_M at the identical
// file size and bit-width -- a real, known tradeoff, not hidden (see RESULTS.md D-export-4).
#define QK_K 256

#pragma pack(push, 1)
typedef struct { ggml_half d; ggml_half dmin; uint8_t scales[12]; uint8_t qs[QK_K / 2]; } WBlockQ4_K;
#pragma pack(pop)

size_t gguf_w_q4_k_nbytes(int64_t n) { return (size_t)(n / QK_K) * sizeof(WBlockQ4_K); }

// Verbatim port of ggml-quants.c's own static get_scale_min_k4() -- must stay byte-identical
// to that function (and to gguf_quants.c's read-side copy) since this IS the container format.
static void get_scale_min_k4_local(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

void gguf_w_quantize_q4_k(const float *w, int64_t n, uint8_t *out) {
    int64_t nb = n / QK_K;
    WBlockQ4_K *blocks = (WBlockQ4_K *)out;
    for (int64_t b = 0; b < nb; b++) {
        const float *sb = w + b * QK_K;

        // Pass 1: per-sub-block scale/min from real min/max (lo forced <=0 so min_off=-lo>=0).
        float scale[8], min_off[8];
        for (int j = 0; j < 8; j++) {
            const float *grp = sb + j * 32;
            float lo = grp[0], hi = grp[0];
            for (int l = 1; l < 32; l++) { if (grp[l] < lo) lo = grp[l]; if (grp[l] > hi) hi = grp[l]; }
            if (lo > 0.0f) lo = 0.0f;
            float sc = (hi - lo) / 15.0f;
            if (sc < 1e-12f) sc = 1.0f;
            scale[j] = sc;
            min_off[j] = -lo;
        }
        float max_scale = 0.0f, max_min = 0.0f;
        for (int j = 0; j < 8; j++) {
            if (scale[j] > max_scale) max_scale = scale[j];
            if (min_off[j] > max_min) max_min = min_off[j];
        }
        float d = max_scale / 63.0f;
        float dmin = max_min / 63.0f;
        blocks[b].d = fp32_to_fp16(d);
        blocks[b].dmin = fp32_to_fp16(dmin);

        // Pass 2: round each sub-block's scale/min to its 6-bit code, pack into scales[12]
        // (verbatim port of quantize_row_q4_K_ref()'s own packing loop -- container format).
        uint8_t ls[8], lm[8];
        for (int j = 0; j < 8; j++) {
            int lsv = d    > 0.0f ? (int)rintf(scale[j]   / d)    : 0;
            int lmv = dmin > 0.0f ? (int)rintf(min_off[j] / dmin) : 0;
            if (lsv > 63) lsv = 63; if (lsv < 0) lsv = 0;
            if (lmv > 63) lmv = 63; if (lmv < 0) lmv = 0;
            ls[j] = (uint8_t)lsv;
            lm[j] = (uint8_t)lmv;
        }
        memset(blocks[b].scales, 0, 12);
        for (int j = 0; j < 8; j++) {
            if (j < 4) {
                blocks[b].scales[j]     = ls[j];
                blocks[b].scales[j + 4] = lm[j];
            } else {
                blocks[b].scales[j + 4]  = (uint8_t)((ls[j] & 0xF) | ((lm[j] & 0xF) << 4));
                blocks[b].scales[j - 4] |= (uint8_t)((ls[j] >> 4) << 6);
                blocks[b].scales[j - 0] |= (uint8_t)((lm[j] >> 4) << 6);
            }
        }

        // Pass 3: re-derive each sub-block's ACTUAL applied d_j/dm_j from the ROUNDED 6-bit
        // codes just packed (not the original float scale/min_off) -- matches ggml's own
        // two-pass consistency, since decode reconstructs from those same rounded codes.
        // Error-feedback (residual) diffusion within each 32-element sub-block, same
        // technique as gguf_w_quantize_q4_0() above (D-export-2).
        uint8_t L[QK_K];
        for (int j = 0; j < 8; j++) {
            uint8_t sc6, m6;
            get_scale_min_k4_local(j, blocks[b].scales, &sc6, &m6);
            float dj  = d * sc6;
            float dmj = dmin * m6;
            float err_feedback = 0.0f;
            for (int l = 0; l < 32; l++) {
                float x = sb[j * 32 + l] + err_feedback;
                float qf = dj > 0.0f ? rintf((x + dmj) / dj) : 0.0f;
                if (qf > 15.0f) qf = 15.0f;
                if (qf < 0.0f)  qf = 0.0f;
                float deq = dj * qf - dmj;
                err_feedback = x - deq;
                L[j * 32 + l] = (uint8_t)qf;
            }
        }

        // Pack 4-bit codes into qs[128]: for each 64-element mega-group, low nibble = first
        // 32 codes, high nibble = next 32 -- verbatim port of the reference's own qs loop.
        uint8_t *q = blocks[b].qs;
        for (int j2 = 0; j2 < QK_K; j2 += 64) {
            for (int l = 0; l < 32; l++) q[l] = (uint8_t)(L[j2 + l] | (L[j2 + l + 32] << 4));
            q += 32;
        }
    }
}

// D-export-5: real GGUF Q5_0. Block layout {ggml_half d; uint8_t qh[4]; uint8_t qs[16];} (22
// bytes/32 elements, 5.5 bits/element) is real GGML_TYPE_Q5_0 (gguf_quants.c's own
// GgmlBlockQ5_0/dequant_row_q5_0 -- the read-side spec reference). qs[] uses the same
// split-half nibble packing as gguf_w_quantize_q4_0() above (byte j's low nibble = element j,
// high nibble = element j+16); qh[] is a 32-bit bitmask where bit i is element i's 5th
// (highest) bit, confirmed by inspecting dequant_row_q5_0()'s own bit extraction (no
// interleaving despite the split-half nibble packing -- qh addresses elements directly by
// their real index). Symmetric quantization: code = round(x/scale)+16, code range [0,31],
// value = (code-16)*scale -- direct 5-bit analog of gguf_w_quantize_q4_0()'s own 4-bit
// [-8,7]/scale=maxabs/7 scheme (scale=maxabs/15, range [-16,15]), same error-feedback
// (residual) diffusion technique carried over unchanged.
#define QK5_0 32

#pragma pack(push, 1)
typedef struct { ggml_half d; uint8_t qh[4]; uint8_t qs[QK5_0 / 2]; } WBlockQ5_0;
#pragma pack(pop)

size_t gguf_w_q5_0_nbytes(int64_t n) { return (size_t)(n / QK5_0) * sizeof(WBlockQ5_0); }

void gguf_w_quantize_q5_0(const float *w, int64_t n, uint8_t *out) {
    int64_t nb = n / QK5_0;
    WBlockQ5_0 *blocks = (WBlockQ5_0 *)out;
    for (int64_t b = 0; b < nb; b++) {
        const float *grp = w + b * QK5_0;
        float maxabs = 0.0f;
        for (int p = 0; p < QK5_0; p++) { float a = fabsf(grp[p]); if (a > maxabs) maxabs = a; }
        float scale = maxabs / 15.0f;
        if (scale < 1e-12f) scale = 1.0f;
        blocks[b].d = fp32_to_fp16(scale);
        float inv = 1.0f / scale;
        float err_feedback = 0.0f;
        int codes[QK5_0];   // unsigned 5-bit, [0,31]
        for (int p = 0; p < QK5_0; p++) {
            float x = grp[p] + err_feedback;
            float qf = rintf(x * inv);
            if (qf > 15.0f)  qf = 15.0f;
            if (qf < -16.0f) qf = -16.0f;
            float deq = qf * scale;
            err_feedback = x - deq;
            codes[p] = (int)qf + 16;
        }
        uint32_t qh = 0;
        for (int j = 0; j < QK5_0 / 2; j++) {
            int c0 = codes[j];              // element j
            int c1 = codes[j + QK5_0 / 2];  // element j+16
            blocks[b].qs[j] = (uint8_t)((c0 & 0x0F) | ((c1 & 0x0F) << 4));
            if (c0 & 0x10) qh |= (1u << j);
            if (c1 & 0x10) qh |= (1u << (j + QK5_0 / 2));
        }
        memcpy(blocks[b].qh, &qh, sizeof(qh));
    }
}

// D-export-6: real GGUF Q6_K. Field order {ql[128]; qh[64]; scales[16]; d} -- `d` LAST, unlike
// Q4_K/Q5_K where d/dmin come first (confirmed against this project's own already-existing
// GgmlBlockQ6_K/dequant_row_q6_k, gguf_quants.c). Symmetric (no dmin -- one scale per 16-
// element sub-group, code range [-32,31] biased +32 for the unsigned 6-bit field, exact
// analog of gguf_w_quantize_q4_0()'s symmetric scheme at 6 bits instead of 4). 210 bytes/256
// elements = 6.5625 bits/element.
//
// The decoder's own loop (nn in {0,128}, l in 0..31, is=l/16, reading sc[is+0/2/4/6] for the
// 4 elements at offsets {0,32,64,96} from nn+l) LOOKS like an interleaved scale-group mapping
// but isn't: working through the index arithmetic by hand shows sub-group g (0..15) covers
// exactly the CONTIGUOUS elements [16*g, 16*g+16) -- i.e. plain sequential 16-groups, same as
// this encoder's own gscale[]/sc8[] indexing below. Only the qs/qh BIT PACKING is interleaved
// (for SIMD-friendly decode), not the scale grouping -- confirmed by manual derivation before
// writing this encoder, not assumed.
#define QK6_K_G 16   // elements per scale sub-group (QK_K/16 = 16 groups of 16)

#pragma pack(push, 1)
typedef struct { uint8_t ql[QK_K / 2]; uint8_t qh[QK_K / 4]; int8_t scales[QK_K / 16]; ggml_half d; } WBlockQ6_K;
#pragma pack(pop)

size_t gguf_w_q6_k_nbytes(int64_t n) { return (size_t)(n / QK_K) * sizeof(WBlockQ6_K); }

void gguf_w_quantize_q6_k(const float *w, int64_t n, uint8_t *out) {
    int64_t nb = n / QK_K;
    WBlockQ6_K *blocks = (WBlockQ6_K *)out;
    for (int64_t b = 0; b < nb; b++) {
        const float *sb = w + b * QK_K;

        // Pass 1: per-16-element sub-group symmetric scale from real absmax (16 groups).
        float gscale[16];
        for (int g = 0; g < 16; g++) {
            const float *grp = sb + g * QK6_K_G;
            float maxabs = 0.0f;
            for (int l = 0; l < QK6_K_G; l++) { float a = fabsf(grp[l]); if (a > maxabs) maxabs = a; }
            float sc = maxabs / 31.0f;
            if (sc < 1e-12f) sc = 1.0f;
            gscale[g] = sc;
        }
        float maxscale = 0.0f;
        for (int g = 0; g < 16; g++) if (gscale[g] > maxscale) maxscale = gscale[g];
        float d = maxscale / 127.0f;
        if (d < 1e-12f) d = 1.0f;
        blocks[b].d = fp32_to_fp16(d);

        // Pass 2: round each group's scale to its int8 code (non-negative, [0,127] -- this
        // encoder's own min/max-based choice, not a replica of ggml's own signed-scale
        // optimizer, same D-export-2/4 stance every other type in this file already takes).
        int8_t sc8[16];
        for (int g = 0; g < 16; g++) {
            int v = (int)rintf(gscale[g] / d);
            if (v > 127) v = 127;
            if (v < 0)   v = 0;
            sc8[g] = (int8_t)v;
        }
        memcpy(blocks[b].scales, sc8, 16);

        // Pass 3: re-derive each group's ACTUAL applied scale from the rounded int8 code (not
        // the original float gscale), requantize with error-feedback diffusion within the
        // 16-element sub-group -- same consistency reasoning as gguf_w_quantize_q4_k() above.
        int8_t L[QK_K];   // signed code, [-32,31]
        for (int g = 0; g < 16; g++) {
            float dg = d * (float)sc8[g];
            float err_feedback = 0.0f;
            for (int l = 0; l < QK6_K_G; l++) {
                float x = sb[g * QK6_K_G + l] + err_feedback;
                float qf = dg > 0.0f ? rintf(x / dg) : 0.0f;
                if (qf > 31.0f)  qf = 31.0f;
                if (qf < -32.0f) qf = -32.0f;
                float deq = qf * dg;
                err_feedback = x - deq;
                L[g * QK6_K_G + l] = (int8_t)qf;
            }
        }

        // Pack: verbatim inverse of dequant_row_q6_k()'s own loop (derived by hand above,
        // confirmed element->group mapping is contiguous so L[] is already in absolute order).
        for (int nn = 0; nn < QK_K; nn += 128) {
            uint8_t *ql = blocks[b].ql + (nn / 128) * 64;
            uint8_t *qh = blocks[b].qh + (nn / 128) * 32;
            for (int l = 0; l < 32; l++) {
                uint8_t c1 = (uint8_t)(L[nn + l + 0]  + 32);
                uint8_t c2 = (uint8_t)(L[nn + l + 32] + 32);
                uint8_t c3 = (uint8_t)(L[nn + l + 64] + 32);
                uint8_t c4 = (uint8_t)(L[nn + l + 96] + 32);
                ql[l + 0]  = (uint8_t)((c1 & 0x0F) | ((c3 & 0x0F) << 4));
                ql[l + 32] = (uint8_t)((c2 & 0x0F) | ((c4 & 0x0F) << 4));
                qh[l] = (uint8_t)(((c1 >> 4) & 3) | (((c2 >> 4) & 3) << 2) |
                                  (((c3 >> 4) & 3) << 4) | (((c4 >> 4) & 3) << 6));
            }
        }
    }
}
