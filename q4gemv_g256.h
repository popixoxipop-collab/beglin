// q4gemv_g256.h -- Phase 1 kernel microbenchmark variants: group-256 activation
// quantization + group-256 weight scaling (plain and subfold-O4), sitting
// ALONGSIDE q4gemv.h's deployed group-64 kernels (same pattern as q8_dot_sdot_row
// living next to q4_dot_sdot_row today). q4gemv.h itself is NOT modified -- it is
// the shared header included by both qwen_infer.c and qwen_spec.c.
//
// RESIDUAL/ERROR-FEEDBACK NOTE (mandatory-residual policy, matching q4gemv.h's
// own file-top note verbatim): this is a READ-ONLY inference dequant kernel --
// there is no runtime residual here BY DESIGN, because error-feedback/residual
// compensation (if any) is applied UPSTREAM at quantization time, not during
// decode. Every function in this file only RECONSTRUCTS already-quantized int4
// codes and already-computed subfold sub_codes/super-scales; no weight update
// happens during inference, so a runtime residual would be meaningless here,
// same as it is for q4_dot_sdot_row/Q4_SDOT_ROW_BATCH_FIXED in q4gemv.h. This
// file is also Phase-1 scope: synthetic/random weight+scale data only (no real
// quantizer, no calibration) -- residual/error-feedback compensation is
// explicitly a Phase-2-or-later concern (the offline subfold+smoothing
// quantizer emission path), not something this microbenchmark produces or needs.
//
// D1: new functions only, zero edits to q4gemv.h
//   WHY: q4_quant_act_i8/q4_dot_sdot_row/Q4_SDOT_ROW_BATCH_FIXED are the ACTUAL
//        deployed group-64 kernels other code depends on; a bug in an in-place
//        edit would be a correctness regression in the live engine, not just a
//        failed experiment
//   COST: some code duplication of the unpack/dot inner loop (4 variants x the
//         64-wide unpack pattern) -- acceptable for a standalone Phase 1
//         microbench; Phase 2 (if this clears the gate) can factor shared
//         pieces out once the winning variant is known
//   EXIT: none needed, this is the intended structure
//
// D2: subfold-O4 effective per-64 scale = wsuper * (sub_code / 63.0f), sub_code
//     in [0,63] (6-bit). Matches eval/phase0_superblock_scope.py:134-150's
//     subfold_scales() definition exactly (qmax = (1<<6)-1 = 63).
//   WHY: this is the format M33's Phase 0b actually validated for ppl (Arm2b);
//        any other scale semantics would make this measurement irrelevant to
//        that result
//   COST: none, this is just matching the validated definition
//   EXIT: n/a
//
// D3: two finalize strategies (V: vector vmlaq_n_s32 combine, defers horizontal
//     reduce to once per 256; S: scalar sacc += reduced*sub_code, reduce stays
//     per-64 but the FLOAT convert/multiply/add drops to per-256) -- BOTH
//     implemented and measured, not one chosen by argument
//   WHY: V is the maximal version of the finalize-frequency hypothesis (4x
//        fewer horizontal reduces) but has real register-pressure risk at
//        MC=8 (q4gemv.h's own M20-D2 note: MC=8's existing g64 batched kernel
//        already runs ~15 live q-regs; V's batched form needs combined[MC] +
//        ac[MC] live simultaneously, ~2x that). S keeps peak vector-register
//        pressure close to the existing g64 kernel's (only sacc[MC], a plain
//        int32 array, persists across sub-groups) at the cost of NOT reducing
//        horizontal-reduce frequency, only float-finalize frequency.
//   COST: 2x the code for the sf variant; resolved empirically, not analytically
//   EXIT: n/a, this is by design (see Phase 1 plan's own risk #1)
//
// D4: q4_split_act (q4gemv.h:306-311) is reused UNCHANGED, no _g256 variant
//   WHY: it splits already-quantized int8 activation bytes into [lo16][hi16]
//        chunks tied to the 64-wide nibble-unpack sub-step -- this is a byte-
//        layout concern independent of which SCALE grouping produced those
//        int8 values; the packed weight nibble layout is likewise unchanged
//        (subfold/plain-g256 only change how the SCALE is derived and applied,
//        never the packed int4 codes themselves)
//   COST: none
//   EXIT: n/a
#ifndef Q4GEMV_G256_H
#define Q4GEMV_G256_H
#include "q4gemv.h"
#include <assert.h>

// ---- activation quantizer, group=256 (M26 rounding-mode regression guard:
// round-half-away-from-zero copied verbatim from q4gemv.h:195-211, NOT
// torch/vcvtnq round-half-to-even) ----
static inline void q4_quant_act_i8_g256(const float *restrict x, int in,
                                        int8_t *restrict xq, float *restrict ascale) {
    assert(in % 256 == 0);
    int ng = in >> 8;
    for (int gi = 0; gi < ng; gi++) {
        const float *xp = x + (size_t)gi * 256;
        float32x4_t m = vdupq_n_f32(1e-8f);
        for (int b = 0; b < 256; b += 4) m = vmaxq_f32(m, vabsq_f32(vld1q_f32(xp + b)));
        float mx = vmaxvq_f32(m);
        float sc = mx / 127.0f, inv = 1.0f / sc; ascale[gi] = sc;
        float32x4_t iv = vdupq_n_f32(inv);
        const float32x4_t half = vdupq_n_f32(0.5f), mhalf = vdupq_n_f32(-0.5f), zero = vdupq_n_f32(0.0f);
        int8_t *qp = xq + (size_t)gi * 256;
        for (int b = 0; b < 256; b += 16) {
            float32x4_t p0 = vmulq_f32(vld1q_f32(xp + b),      iv);
            float32x4_t p1 = vmulq_f32(vld1q_f32(xp + b + 4),  iv);
            float32x4_t p2 = vmulq_f32(vld1q_f32(xp + b + 8),  iv);
            float32x4_t p3 = vmulq_f32(vld1q_f32(xp + b + 12), iv);
            int32x4_t a0 = vcvtq_s32_f32(vaddq_f32(p0, vbslq_f32(vcgeq_f32(p0, zero), half, mhalf)));
            int32x4_t a1 = vcvtq_s32_f32(vaddq_f32(p1, vbslq_f32(vcgeq_f32(p1, zero), half, mhalf)));
            int32x4_t a2 = vcvtq_s32_f32(vaddq_f32(p2, vbslq_f32(vcgeq_f32(p2, zero), half, mhalf)));
            int32x4_t a3 = vcvtq_s32_f32(vaddq_f32(p3, vbslq_f32(vcgeq_f32(p3, zero), half, mhalf)));
            int16x8_t s0 = vcombine_s16(vqmovn_s32(a0), vqmovn_s32(a1));
            int16x8_t s1 = vcombine_s16(vqmovn_s32(a2), vqmovn_s32(a3));
            vst1q_s8(qp + b, vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)));
        }
    }
}

// ---- row (M=1) kernels ----

// Diagnostic upper bound: plain fp32 per-256 weight scale (NOT subfold, NOT a
// deployable accuracy config on its own -- Phase 0 found W:GPTQ-g256 alone
// fails the ppl gate badly). Isolates how much finalize-frequency reduction
// alone buys; the gap between this and *_g256sf below is subfold's own
// integer-multiply overhead.
static inline float q4_dot_sdot_row_g256plain(const uint8_t *restrict pr, const float *restrict wscale256,
                                              const int8_t *restrict xq, const float *restrict ascale256, int in) {
    const uint8x16_t mask = vdupq_n_u8(0x0F); const int8x16_t eight = vdupq_n_s8(8);
    int ng256 = in >> 8; float acc = 0.0f;
    for (int gs = 0; gs < ng256; gs++) {
        int32x4_t combined = vdupq_n_s32(0);
        for (int sub = 0; sub < 4; sub++) {
            const uint8_t *gp = pr + (size_t)(gs * 4 + sub) * 32;
            int32x4_t iacc = vdupq_n_s32(0);
            for (int bb = 0; bb < 32; bb += 16) {
                uint8x16_t raw = vld1q_u8(gp + bb);
                int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), eight);
                int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
                const int8_t *sp = xq + (size_t)(gs * 4 + sub) * 64 + (size_t)bb * 2;
                iacc = vdotq_s32(iacc, lo, vld1q_s8(sp));
                iacc = vdotq_s32(iacc, hi, vld1q_s8(sp + 16));
            }
            combined = vaddq_s32(combined, iacc);
        }
        acc += (float)vaddvq_s32(combined) * wscale256[gs] * ascale256[gs];
    }
    return acc;
}

// Candidate: subfold-O4 weights (per-64 6-bit sub_code x per-256 fp32 wsuper)
// + activation-256. V-form: defer horizontal reduce to once per 256 elements.
static inline float q4_dot_sdot_row_g256sf_v(const uint8_t *restrict pr, const float *restrict wsuper,
                                             const uint8_t *restrict subcode,
                                             const int8_t *restrict xq, const float *restrict ascale256, int in) {
    const uint8x16_t mask = vdupq_n_u8(0x0F); const int8x16_t eight = vdupq_n_s8(8);
    int ng256 = in >> 8; float acc = 0.0f;
    for (int gs = 0; gs < ng256; gs++) {
        int32x4_t combined = vdupq_n_s32(0);
        for (int sub = 0; sub < 4; sub++) {
            const uint8_t *gp = pr + (size_t)(gs * 4 + sub) * 32;
            int32x4_t iacc = vdupq_n_s32(0);
            for (int bb = 0; bb < 32; bb += 16) {
                uint8x16_t raw = vld1q_u8(gp + bb);
                int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), eight);
                int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
                const int8_t *sp = xq + (size_t)(gs * 4 + sub) * 64 + (size_t)bb * 2;
                iacc = vdotq_s32(iacc, lo, vld1q_s8(sp));
                iacc = vdotq_s32(iacc, hi, vld1q_s8(sp + 16));
            }
            combined = vmlaq_n_s32(combined, iacc, (int32_t)subcode[gs * 4 + sub]);
        }
        acc += (float)vaddvq_s32(combined) * wsuper[gs] * (1.0f / 63.0f) * ascale256[gs];
    }
    return acc;
}

// Candidate: subfold-O4, S-form. Horizontal reduce stays per-64 (like the
// deployed g64 kernel); only the float convert/multiply/add drops to per-256.
static inline float q4_dot_sdot_row_g256sf_s(const uint8_t *restrict pr, const float *restrict wsuper,
                                             const uint8_t *restrict subcode,
                                             const int8_t *restrict xq, const float *restrict ascale256, int in) {
    const uint8x16_t mask = vdupq_n_u8(0x0F); const int8x16_t eight = vdupq_n_s8(8);
    int ng256 = in >> 8; float acc = 0.0f;
    for (int gs = 0; gs < ng256; gs++) {
        int32_t sacc = 0;
        for (int sub = 0; sub < 4; sub++) {
            const uint8_t *gp = pr + (size_t)(gs * 4 + sub) * 32;
            int32x4_t iacc = vdupq_n_s32(0);
            for (int bb = 0; bb < 32; bb += 16) {
                uint8x16_t raw = vld1q_u8(gp + bb);
                int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), eight);
                int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
                const int8_t *sp = xq + (size_t)(gs * 4 + sub) * 64 + (size_t)bb * 2;
                iacc = vdotq_s32(iacc, lo, vld1q_s8(sp));
                iacc = vdotq_s32(iacc, hi, vld1q_s8(sp + 16));
            }
            sacc += vaddvq_s32(iacc) * (int32_t)subcode[gs * 4 + sub];
        }
        acc += (float)sacc * wsuper[gs] * (1.0f / 63.0f) * ascale256[gs];
    }
    return acc;
}

// ---- batched (MC-fixed) kernels, same macro-generation pattern as q4gemv.h's
// Q4_SDOT_ROW_BATCH_FIXED. Overflow note (worst case, re-verified against the
// real code): per-lane sub-64 accumulation <= 16*8*128 = 16,384; V-form's
// combined[] lane after vmlaq_n_s32 by sub_code(<=63): <= 16,384*63 ~= 1.03M;
// summed over 4 sub-groups per lane: ~4.13M; fully horizontally reduced
// (vaddvq_s32, 4 lanes): ~16.5M -- ~130x below INT32_MAX even at this most
// conservative bound. S-form's scalar path has the identical bound (same
// arithmetic, different reduction order). Comfortable margin either way; the
// correctness gates (dequant parity, batch=row bit-consistency) are what
// actually certify this, not the arithmetic estimate alone. ----

#define Q4_SDOT_ROW_BATCH_FIXED_G256PLAIN(MC) \
static void q4_sdot_row_batch_g256plain_##MC(const uint8_t *restrict pr, const float *restrict wscale256, \
                                   const int8_t *restrict xq, const float *restrict ascale256, \
                                   int in, float b, float *restrict y, size_t ystep) { \
    const uint8x16_t mask = vdupq_n_u8(0x0F); const int8x16_t eight = vdupq_n_s8(8); \
    int ng256 = in >> 8; \
    float facc[MC]; \
    for (int c = 0; c < MC; c++) facc[c] = 0.0f; \
    for (int gs = 0; gs < ng256; gs++) { \
        int32x4_t combined[MC]; \
        for (int c = 0; c < MC; c++) combined[c] = vdupq_n_s32(0); \
        for (int sub = 0; sub < 4; sub++) { \
            const uint8_t *gp = pr + (size_t)(gs * 4 + sub) * 32; \
            int32x4_t ac[MC]; \
            for (int c = 0; c < MC; c++) ac[c] = vdupq_n_s32(0); \
            for (int bb = 0; bb < 32; bb += 16) { \
                uint8x16_t raw = vld1q_u8(gp + bb); \
                int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), eight); \
                int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight); \
                for (int c = 0; c < MC; c++) { \
                    const int8_t *sp = xq + (size_t)c * in + (size_t)(gs * 4 + sub) * 64 + (size_t)bb * 2; \
                    ac[c] = vdotq_s32(ac[c], lo, vld1q_s8(sp)); \
                    ac[c] = vdotq_s32(ac[c], hi, vld1q_s8(sp + 16)); \
                } \
            } \
            for (int c = 0; c < MC; c++) combined[c] = vaddq_s32(combined[c], ac[c]); \
        } \
        float ws = wscale256[gs]; \
        for (int c = 0; c < MC; c++) facc[c] += (float)vaddvq_s32(combined[c]) * ws * ascale256[(size_t)c * ng256 + gs]; \
    } \
    for (int c = 0; c < MC; c++) y[(size_t)c * ystep] = b + facc[c]; \
}

#define Q4_SDOT_ROW_BATCH_FIXED_G256SF_V(MC) \
static void q4_sdot_row_batch_g256sf_v_##MC(const uint8_t *restrict pr, const float *restrict wsuper, \
                                   const uint8_t *restrict subcode, \
                                   const int8_t *restrict xq, const float *restrict ascale256, \
                                   int in, float b, float *restrict y, size_t ystep) { \
    const uint8x16_t mask = vdupq_n_u8(0x0F); const int8x16_t eight = vdupq_n_s8(8); \
    int ng256 = in >> 8; \
    float facc[MC]; \
    for (int c = 0; c < MC; c++) facc[c] = 0.0f; \
    for (int gs = 0; gs < ng256; gs++) { \
        int32x4_t combined[MC]; \
        for (int c = 0; c < MC; c++) combined[c] = vdupq_n_s32(0); \
        for (int sub = 0; sub < 4; sub++) { \
            const uint8_t *gp = pr + (size_t)(gs * 4 + sub) * 32; \
            int32x4_t ac[MC]; \
            for (int c = 0; c < MC; c++) ac[c] = vdupq_n_s32(0); \
            for (int bb = 0; bb < 32; bb += 16) { \
                uint8x16_t raw = vld1q_u8(gp + bb); \
                int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), eight); \
                int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight); \
                for (int c = 0; c < MC; c++) { \
                    const int8_t *sp = xq + (size_t)c * in + (size_t)(gs * 4 + sub) * 64 + (size_t)bb * 2; \
                    ac[c] = vdotq_s32(ac[c], lo, vld1q_s8(sp)); \
                    ac[c] = vdotq_s32(ac[c], hi, vld1q_s8(sp + 16)); \
                } \
            } \
            int32_t sc = (int32_t)subcode[gs * 4 + sub]; \
            for (int c = 0; c < MC; c++) combined[c] = vmlaq_n_s32(combined[c], ac[c], sc); \
        } \
        for (int c = 0; c < MC; c++) facc[c] += (float)vaddvq_s32(combined[c]) * wsuper[gs] * (1.0f / 63.0f) * ascale256[(size_t)c * ng256 + gs]; \
    } \
    for (int c = 0; c < MC; c++) y[(size_t)c * ystep] = b + facc[c]; \
}

#define Q4_SDOT_ROW_BATCH_FIXED_G256SF_S(MC) \
static void q4_sdot_row_batch_g256sf_s_##MC(const uint8_t *restrict pr, const float *restrict wsuper, \
                                   const uint8_t *restrict subcode, \
                                   const int8_t *restrict xq, const float *restrict ascale256, \
                                   int in, float b, float *restrict y, size_t ystep) { \
    const uint8x16_t mask = vdupq_n_u8(0x0F); const int8x16_t eight = vdupq_n_s8(8); \
    int ng256 = in >> 8; \
    float facc[MC]; int32_t sacc[MC]; \
    for (int c = 0; c < MC; c++) facc[c] = 0.0f; \
    for (int gs = 0; gs < ng256; gs++) { \
        for (int c = 0; c < MC; c++) sacc[c] = 0; \
        for (int sub = 0; sub < 4; sub++) { \
            const uint8_t *gp = pr + (size_t)(gs * 4 + sub) * 32; \
            int32x4_t ac[MC]; \
            for (int c = 0; c < MC; c++) ac[c] = vdupq_n_s32(0); \
            for (int bb = 0; bb < 32; bb += 16) { \
                uint8x16_t raw = vld1q_u8(gp + bb); \
                int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), eight); \
                int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight); \
                for (int c = 0; c < MC; c++) { \
                    const int8_t *sp = xq + (size_t)c * in + (size_t)(gs * 4 + sub) * 64 + (size_t)bb * 2; \
                    ac[c] = vdotq_s32(ac[c], lo, vld1q_s8(sp)); \
                    ac[c] = vdotq_s32(ac[c], hi, vld1q_s8(sp + 16)); \
                } \
            } \
            int32_t sc = (int32_t)subcode[gs * 4 + sub]; \
            for (int c = 0; c < MC; c++) sacc[c] += vaddvq_s32(ac[c]) * sc; \
        } \
        for (int c = 0; c < MC; c++) facc[c] += (float)sacc[c] * wsuper[gs] * (1.0f / 63.0f) * ascale256[(size_t)c * ng256 + gs]; \
    } \
    for (int c = 0; c < MC; c++) y[(size_t)c * ystep] = b + facc[c]; \
}

Q4_SDOT_ROW_BATCH_FIXED_G256PLAIN(1)
Q4_SDOT_ROW_BATCH_FIXED_G256PLAIN(4)
Q4_SDOT_ROW_BATCH_FIXED_G256PLAIN(8)
Q4_SDOT_ROW_BATCH_FIXED_G256SF_V(1)
Q4_SDOT_ROW_BATCH_FIXED_G256SF_V(4)
Q4_SDOT_ROW_BATCH_FIXED_G256SF_V(8)
Q4_SDOT_ROW_BATCH_FIXED_G256SF_S(1)
Q4_SDOT_ROW_BATCH_FIXED_G256SF_S(4)
Q4_SDOT_ROW_BATCH_FIXED_G256SF_S(8)

// ===================================================================== Phase 2 (M36) ====
// Engine-integration glue: pool-threaded gemv/gemm entries for the g256sf format, mirroring
// gemv_q4g64_sdot_mt / gemm_qXg64_sdot_mt exactly (same guards, same scratch reuse, same
// flag-hygiene resets, same residual policy: READ-ONLY inference, residual/error_feedback
// compensation upstream at quantization time -- see file-top note). Phase 1 sections above
// are untouched.
//
// D5: V-form kernels only are wired into the engine
//   WHY: Phase 1 measured V beating S consistently at MC>=4 and on the T=6 row leg
//        (RESULTS M34; re-verified this session); S stays above as the measured
//        alternative, deliberately not wired
//   COST: at MC=1 tails S occasionally edges V by <=4% on one shape -- noise-level, and
//         MC=1 tails are rare in serve (B=8 default)
//   EXIT: swap the two fn-ptr installs below to the S forms
// D6: MC=2 specialization added (Phase 1 built {1,4,8} for its fixed-MC benches; the
//     deployed greedy tile dispatcher chunks {8,4,2,1} -- q4gemv.h q4_sdot_row_batch)

Q4_SDOT_ROW_BATCH_FIXED_G256SF_V(2)

// greedy fixed-MC tile dispatcher, mirroring q4_sdot_row_batch (q4gemv.h) exactly
static inline void q4_sdot_row_batch_g256sf(const uint8_t *pr, const float *wsuper, const uint8_t *subcode,
                                            const int8_t *xq, const float *ascale, int in, int M, float b,
                                            float *y, size_t ystep) {
    int ng = in >> 8, c0 = 0;
    while (M - c0 >= 8) { q4_sdot_row_batch_g256sf_v_8(pr, wsuper, subcode, xq+(size_t)c0*in, ascale+(size_t)c0*ng, in, b, y+(size_t)c0*ystep, ystep); c0 += 8; }
    while (M - c0 >= 4) { q4_sdot_row_batch_g256sf_v_4(pr, wsuper, subcode, xq+(size_t)c0*in, ascale+(size_t)c0*ng, in, b, y+(size_t)c0*ystep, ystep); c0 += 4; }
    while (M - c0 >= 2) { q4_sdot_row_batch_g256sf_v_2(pr, wsuper, subcode, xq+(size_t)c0*in, ascale+(size_t)c0*ng, in, b, y+(size_t)c0*ystep, ystep); c0 += 2; }
    while (M - c0 >= 1) { q4_sdot_row_batch_g256sf_v_1(pr, wsuper, subcode, xq+(size_t)c0*in, ascale+(size_t)c0*ng, in, b, y+(size_t)c0*ystep, ystep); c0 += 1; }
}

// single-stream W4A8 gemv (M=1): quantize group-256 once, split once (q4_split_act reused
// unchanged, D4 above), dispatch V-form rows through the shared pool.
static inline void gemv_q4g256sf_sdot_mt(q4pool *p, const uint8_t *packed, const float *wsuper,
                                         const uint8_t *subcode, const float *x, const float *bias,
                                         float *y, int out, int in) {
    if (in > p->max_in) { fprintf(stderr, "FATAL: g256sf gemv in=%d > pool max_in=%d\n", in, p->max_in); exit(1); }
    if (in % 256 != 0) { fprintf(stderr, "FATAL: g256sf gemv in=%d %% 256 != 0\n", in); exit(1); }
    q4_quant_act_i8_g256(x, in, p->xq_scratch, p->ascale_scratch);   // in/256 scales fit the in/64-sized scratch
    q4_split_act(p->xq_scratch, p->xq_split, in);
    if (p->nthreads == 1) {
        for (int r = 0; r < out; r++) {
            float dot = q4_dot_sdot_row_g256sf_v(packed + (size_t)r * ((size_t)in / 2),
                                                 wsuper + (size_t)r * (in >> 8),
                                                 subcode + (size_t)r * (in >> 6),
                                                 p->xq_split, p->ascale_scratch, in);
            y[r] = (bias ? bias[r] : 0.0f) + dot;
        }
        return;
    }
    p->job.packed = packed; p->job.scales = wsuper; p->job.subcode = subcode;
    p->job.x = x; p->job.bias = bias; p->job.y = y;
    p->job.out = out; p->job.in = in; p->job.ng = in >> 8; p->job.bits = Q4_BITS_G256SF; p->job.M = 1;
    p->job.row_bytes = (size_t)in / 2; p->job.fused = 0; p->job.sdot = 1; p->job.sdotb = 0;
    p->job.xq = p->xq_split; p->job.ascale = p->ascale_scratch;
    p->job.g256_row = q4_dot_sdot_row_g256sf_v; p->job.g256_tile = q4_sdot_row_batch_g256sf;
    q4pool_go_and_wait(p);
    p->job.sdot = 0;   // leave the shared job flag clean, same convention as gemv_q4g64_sdot_mt
}

// batched serve/cbatch gemm: caller-quantized columns (xq in the SPLIT layout, one
// q4_quant_act_i8_g256 + q4_split_act per column at the call site; ascale=[M][in/256]).
static inline void gemm_q4g256sf_sdot_mt(q4pool *p, const uint8_t *packed, const float *wsuper,
                                         const uint8_t *subcode, const int8_t *xq, const float *ascale,
                                         const float *bias, float *y, int out, int in, int M) {
    if (in > p->max_in) { fprintf(stderr, "FATAL: g256sf gemm in=%d > pool max_in=%d\n", in, p->max_in); exit(1); }
    if (in % 256 != 0) { fprintf(stderr, "FATAL: g256sf gemm in=%d %% 256 != 0\n", in); exit(1); }
    if (M < 1 || M > Q4_SDOT_BMAX) { fprintf(stderr, "FATAL: g256sf gemm M=%d out of [1,%d]\n", M, Q4_SDOT_BMAX); exit(1); }
    p->job.packed = packed; p->job.scales = wsuper; p->job.subcode = subcode;
    p->job.x = NULL; p->job.bias = bias; p->job.y = y;
    p->job.out = out; p->job.in = in; p->job.ng = in >> 8; p->job.bits = Q4_BITS_G256SF;
    p->job.M = M; p->job.row_bytes = (size_t)in / 2;
    p->job.fused = 0; p->job.sdot = 1; p->job.sdotb = 1; p->job.xq = xq; p->job.ascale = ascale;
    p->job.g256_row = q4_dot_sdot_row_g256sf_v; p->job.g256_tile = q4_sdot_row_batch_g256sf;
    if (p->nthreads == 1) { q4_run_range(p, 0); p->job.sdot = 0; p->job.sdotb = 0; return; }
    q4pool_go_and_wait(p);
    p->job.sdot = 0; p->job.sdotb = 0;   // leave the shared job flags clean for the next dispatch
}

#endif // Q4GEMV_G256_H

