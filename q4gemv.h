// q4gemv.h -- inline int4 (group-64) dequant GEMV for the decode path.
// y[out] = W_q4g64[out,in] @ x[in] (+bias). Reads PACKED nibbles + per-group
// fp32 scales directly (never materializes an fp32 weight in DRAM) -> the
// per-token weight traffic drops ~3.7x, the whole point of the task.
//
// D11: fresh inference-only kernel; reuses spectral_trunk's nibble CONVENTION
//      (2 codes/byte, low=even col / high=odd col, value=(nibble-8)*scale)
//      but NOT its per-tensor-scale/training code.
// D13: NEON-unpack a whole row into an L1-resident fp32 tile (per-group scale
//      applied during widening via vst2q_f32 interleave), then ONE
//      vDSP_dotpr(tile,x) per row. Reads packed once from DRAM (the win);
//      dotpr runs from L1. fp32 accumulation -> only summation-order deltas
//      vs the evaluated dq model (gate G1b/G1c).
// D14: row-block threading via a persistent pthread pool; rows independent +
//      each row summed by one thread in fixed order -> bitwise deterministic.
// R7:  every offset/byte index is long/size_t (blob ~1.67e9 B).
//
// RESIDUAL/ERROR-FEEDBACK NOTE (mandatory-residual policy): this is a
// READ-ONLY inference dequant kernel -- there is no residual here BY DESIGN
// because error-feedback is applied UPSTREAM at quantization time
// (eval/quantize_int4.py D8, 1-D error diffusion). The codes this kernel
// reads are already the error-feedback-corrected values; decode only
// reconstructs (nibble-8)*scale. A runtime residual would be meaningless
// (no weight update happens during inference).
#ifndef Q4GEMV_H
#define Q4GEMV_H
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <stdatomic.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Unpack+dequant one row of `in` codes (packed as in/2 bytes) into tile[in],
// applying the per-group scale (group = 64). in % 64 == 0 (asserted at load).
static inline void q4_unpack_row(const uint8_t *restrict pr, const float *restrict scales,
                                 float *restrict tile, int in) {
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t eight = vdupq_n_s8(8);
    int ng = in >> 6;                 // in / 64
    for (int gi = 0; gi < ng; gi++) {
        float s = scales[gi];
        float32x4_t sv = vdupq_n_f32(s);
        const uint8_t *gp = pr + (size_t)gi * 32;   // 64 codes = 32 bytes
        float *tp = tile + (size_t)gi * 64;
        for (int b = 0; b < 32; b += 16) {          // 16 bytes -> 32 codes
            uint8x16_t raw = vld1q_u8(gp + b);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), eight);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
            int16x8_t lo0 = vmovl_s8(vget_low_s8(lo)),  lo1 = vmovl_s8(vget_high_s8(lo));
            int16x8_t hi0 = vmovl_s8(vget_low_s8(hi)),  hi1 = vmovl_s8(vget_high_s8(hi));
            float32x4_t lf[4] = {
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo0))), sv),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo0))), sv),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo1))), sv),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo1))), sv)};
            float32x4_t hf[4] = {
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi0))), sv),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi0))), sv),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi1))), sv),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi1))), sv)};
            float *dp = tp + (size_t)b * 2;         // 32 codes -> 64 tile floats
            for (int k = 0; k < 4; k++) {
                float32x4x2_t pair = { { lf[k], hf[k] } };
                vst2q_f32(dp + (size_t)k * 8, pair);
            }
        }
    }
}

static inline void gemv_q4g64(const uint8_t *restrict packed, const float *restrict scales,
                              const float *restrict x, const float *restrict bias,
                              float *restrict y, int out, int in, float *restrict tile) {
    size_t row_bytes = (size_t)in / 2;
    int ng = in >> 6;
    for (int r = 0; r < out; r++) {
        q4_unpack_row(packed + (size_t)r * row_bytes, scales + (size_t)r * ng, tile, in);
        float dot; vDSP_dotpr(tile, 1, x, 1, &dot, in);
        y[r] = bias ? bias[r] + dot : dot;
    }
}

static inline float q4_dot_fused_row(const uint8_t *restrict pr, const float *restrict scales,
                                      const float *restrict x, int in) {
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t eight = vdupq_n_s8(8);
    int ng = in >> 6;
    float32x4_t acc_e = vdupq_n_f32(0), acc_o = vdupq_n_f32(0);
    for (int gi = 0; gi < ng; gi++) {
        float32x4_t sv = vdupq_n_f32(scales[gi]);
        const uint8_t *gp = pr + (size_t)gi * 32;
        const float *xp = x + (size_t)gi * 64;
        for (int b = 0; b < 32; b += 16) {
            uint8x16_t raw = vld1q_u8(gp + b);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), eight);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
            int16x8_t lo0 = vmovl_s8(vget_low_s8(lo)),  lo1 = vmovl_s8(vget_high_s8(lo));
            int16x8_t hi0 = vmovl_s8(vget_low_s8(hi)),  hi1 = vmovl_s8(vget_high_s8(hi));
            float32x4_t lf[4] = {
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo0))), sv),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo0))), sv),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo1))), sv),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo1))), sv)};
            float32x4_t hf[4] = {
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi0))), sv),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi0))), sv),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi1))), sv),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi1))), sv)};
            const float *dp = xp + (size_t)b * 2;
            for (int k = 0; k < 4; k++) {
                float32x4x2_t xpair = vld2q_f32(dp + (size_t)k * 8);
                acc_e = vfmaq_f32(acc_e, lf[k], xpair.val[0]);
                acc_o = vfmaq_f32(acc_o, hf[k], xpair.val[1]);
            }
        }
    }
    return vaddvq_f32(acc_e) + vaddvq_f32(acc_o);
}

static inline void gemv_q4g64_fused(const uint8_t *restrict packed, const float *restrict scales,
                                     const float *restrict x, const float *restrict bias,
                                     float *restrict y, int out, int in) {
    size_t row_bytes = (size_t)in / 2;
    int ng = in >> 6;
    for (int r = 0; r < out; r++) {
        float dot = q4_dot_fused_row(packed + (size_t)r * row_bytes, scales + (size_t)r * ng, x, in);
        y[r] = bias ? bias[r] + dot : dot;
    }
}

// ---- int8 (group-64) variant for the untied lm_head (near-lossless, 1 byte/code) ----
// value = code*scale (symmetric, no -8 offset). Simpler than int4: codes are already in
// column order, so a plain contiguous widen+store (no vst2 interleave). NOTE: no runtime
// residual here for the same reason as the int4 kernel -- this is read-only inference; the
// weight is a fixed near-lossless int8 encoding produced offline (quantize_int4.py).
static inline void q8_unpack_row(const int8_t *restrict pr, const float *restrict scales,
                                 float *restrict tile, int in) {
    int ng = in >> 6;                 // in / 64
    for (int gi = 0; gi < ng; gi++) {
        float32x4_t sv = vdupq_n_f32(scales[gi]);
        const int8_t *gp = pr + (size_t)gi * 64;
        float *tp = tile + (size_t)gi * 64;
        for (int b = 0; b < 64; b += 16) {          // 16 int8 codes -> 16 floats
            int8x16_t c = vld1q_s8(gp + b);
            int16x8_t c0 = vmovl_s8(vget_low_s8(c)), c1 = vmovl_s8(vget_high_s8(c));
            vst1q_f32(tp + b,      vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(c0))),  sv));
            vst1q_f32(tp + b + 4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(c0))), sv));
            vst1q_f32(tp + b + 8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(c1))),  sv));
            vst1q_f32(tp + b + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(c1))), sv));
        }
    }
}

static inline void gemv_q8g64(const int8_t *restrict packed, const float *restrict scales,
                              const float *restrict x, const float *restrict bias,
                              float *restrict y, int out, int in, float *restrict tile) {
    size_t row_bytes = (size_t)in;     // 1 byte/code
    int ng = in >> 6;
    for (int r = 0; r < out; r++) {
        q8_unpack_row(packed + (size_t)r * row_bytes, scales + (size_t)r * ng, tile, in);
        float dot; vDSP_dotpr(tile, 1, x, 1, &dot, in);
        y[r] = bias ? bias[r] + dot : dot;
    }
}

// ---- persistent thread pool: static contiguous row-block partition ----
// M19: W4A8 int8-SDOT path (int4-residual-guard: inference kernel over already-
// quantized int4 weights, no residual). Activation is dynamically quantized to
// int8 per group-64 (once per GEMV, shared across rows); weights unpack int4->int8
// with AND/shift only (no fp32 convert); vdotq_s32 accumulates int32; scale once
// per group. Reuses the EXISTING packed weight layout; since M27 the q4 activation is
// pre-split once per token (q4_split_act) so the row kernel reads it with plain vld1q_s8.
static inline void q4_quant_act_i8(const float *restrict x, int in,
                                   int8_t *restrict xq, float *restrict ascale) {
    int ng = in >> 6;
    for (int gi = 0; gi < ng; gi++) {
        const float *xp = x + (size_t)gi * 64;
        float32x4_t m = vdupq_n_f32(1e-8f);                 // abs-max over the 64-group
        for (int b = 0; b < 64; b += 4) m = vmaxq_f32(m, vabsq_f32(vld1q_f32(xp + b)));
        float mx = vmaxvq_f32(m);
        // int4-residual-guard: READ-side activation quantizer; residual / error-feedback
        // compensation is applied UPSTREAM at weight-quantization time (quantize_int4.py D8
        // error diffusion), same policy as the file-top note -- no runtime residual by design.
        // M26-D1 (activation-quant rounding regression fix): WHY -- M19's SCALAR quant used
        // round-half-away-from-zero and inv=1/(mx/127); the 0986a59 vectorization switched to
        // vcvtnq (round-half-to-EVEN) + inv=127/mx, gated only as greedy-identical (argmax is
        // robust to the sub-ULP shift) but it silently moved W4A8 ppl 12.0705 -> 12.1213
        // (+0.42%). Restore scalar semantics in NEON so the vectorized path is bit-identical to
        // the scalar quantizer (ppl + greedy), keeping vectorized speed. EXIT: revert to
        // vcvtnq_s32_f32 + inv=127/mx to reproduce the 12.1213 regression.
        float sc = mx / 127.0f, inv = 1.0f / sc; ascale[gi] = sc;
        float32x4_t iv = vdupq_n_f32(inv);
        const float32x4_t half = vdupq_n_f32(0.5f), mhalf = vdupq_n_f32(-0.5f), zero = vdupq_n_f32(0.0f);
        int8_t *qp = xq + (size_t)gi * 64;
        for (int b = 0; b < 64; b += 16) {                  // round-half-away-from-zero (matches M19 scalar), saturating narrow
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
// M27: reads the SPLIT activation layout ([lo16][hi16] per 32-byte chunk, q4_split_act),
// same read pattern as q4_sdot_row_batch_1 -- identical VALUES in identical lanes vs the
// old vld2q_s8 de-interleave, so the dot inputs (and bit-identity) are unchanged.
// M27-D1: WHY -- the old body de-interleaved the activation with vld2q_s8 on EVERY output
// row, but the activation is quantized once per token and reused across all `out` rows;
// splitting it ONCE in the caller (q4_split_act, one pass over `in` bytes) removes the
// redundant per-row zip and lets the hot loop use two plain vld1q_s8, exactly as the
// batched kernels already do (M20). COST: one q4_split_act pass per projection + the
// xq_split pool buffer. EXIT: revert this body to the vld2q_s8 read and drop the
// q4_split_act call in gemv_q4g64_sdot_mt to reproduce the ~20 GB/s baseline.
// (int4-residual-guard: READ-ONLY inference kernel, residual/error-feedback applied
// upstream at quantization time -- same policy as the file-top note.)
static inline float q4_dot_sdot_row(const uint8_t *restrict pr, const float *restrict wscale,
                                    const int8_t *restrict xq, const float *restrict ascale, int in) {
    const uint8x16_t mask = vdupq_n_u8(0x0F); const int8x16_t eight = vdupq_n_s8(8);
    int ng = in >> 6; float acc = 0.0f;
    for (int gi = 0; gi < ng; gi++) {
        const uint8_t *gp = pr + (size_t)gi * 32;
        int32x4_t iacc = vdupq_n_s32(0);
        for (int bb = 0; bb < 32; bb += 16) {
            uint8x16_t raw = vld1q_u8(gp + bb);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), eight);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
            const int8_t *sp = xq + (size_t)gi * 64 + (size_t)bb * 2;
            iacc = vdotq_s32(iacc, lo, vld1q_s8(sp));
            iacc = vdotq_s32(iacc, hi, vld1q_s8(sp + 16));
        }
        acc += (float)vaddvq_s32(iacc) * wscale[gi] * ascale[gi];
    }
    return acc;
}
// M19: W8A8 int8-SDOT row (int8 weights need no nibble unpack and no interleave --
// straight vld1q_s8 weight and activation, vdotq_s32, scale once per group).
static inline float q8_dot_sdot_row(const int8_t *restrict pr, const float *restrict wscale,
                                    const int8_t *restrict xq, const float *restrict ascale, int in) {
    int ng = in >> 6; float acc = 0.0f;
    for (int gi = 0; gi < ng; gi++) {
        const int8_t *gp = pr + (size_t)gi * 64; const int8_t *xp = xq + (size_t)gi * 64;
        int32x4_t iacc = vdupq_n_s32(0);
        for (int b = 0; b < 64; b += 16)
            iacc = vdotq_s32(iacc, vld1q_s8(gp + b), vld1q_s8(xp + b));
        acc += (float)vaddvq_s32(iacc) * wscale[gi] * ascale[gi];
    }
    return acc;
}

// ---- M20 (serve): request-batched int8-SDOT row kernels ----
// int4-residual-guard: READ-ONLY inference kernels over already-quantized int4 weights --
// error-feedback/residual compensation was applied UPSTREAM at quantization time
// (eval/quantize_int4.py D8 error diffusion); no runtime residual by design, same policy
// as the file-top RESIDUAL/ERROR-FEEDBACK NOTE and the M19 sdot kernels.
// M20-D2 (register tiling / B-cap): WHY -- batch_roofline.c's pattern (unpack each int4
// weight chunk to int8 in registers ONCE, dot against all M activation columns, M int32x4
// accumulators) but with M as a COMPILE-TIME constant per specialization: a runtime-M
// column loop forces the accumulator array onto the stack (clang cannot registerize an
// indexed array with unknown trip count), and that stack round-trip per vdot measured
// ~4-5x slower at M=1 than q4_dot_sdot_row -- it silently flattened the whole batching
// win inside the engine (roofline's 6.7x was vs its own stack-degraded B=1 baseline).
// Serve's B-cap stays Q4_SDOT_BMAX=16 (one 16-wide tile; larger B would chunk and
// re-read weights per 16 columns -- dispatcher below already supports it, cap is about
// serve's static buffers). COST: 5 specializations x2 kernels of generated code; odd M
// decomposes into 8/4/2/1 chunks that re-read the weight row per chunk. EXIT: bump
// SRV_BMAX/Q4_SDOT_BMAX and let the greedy dispatcher chunk, or add MC=32.
//
// Numerical contract: per column this is EXACTLY q4_dot_sdot_row's chain -- int32 group
// accumulation is exact (no fp order sensitivity), and the float accumulation
// facc += (float)group_sum * wscale[gi] * ascale[gi] runs in the same group order with the
// same left-associative multiply chain -> bit-identical per column to the M=1 kernel.
// M23: 16 -> 32, exercising this block's documented EXIT ("bump SRV_BMAX/Q4_SDOT_BMAX
// and let the greedy dispatcher chunk"): M=32 decomposes as 8+8+8+8 column tiles, and
// per-column arithmetic is identical at every decomposition (bit-identity note below).
// The check in gemm_qXg64_sdot_mt is a caller-scratch-capacity guard, not a kernel limit.
// M24: 32 -> 64, same EXIT exercised once more for the int4-KV B-cap raise (M24-D6 in
// qwen_infer.c). Scratch-capacity only; per-column arithmetic is identical at every
// tile decomposition (bit-identity note above), so B<=32 workloads are unchanged.
#define Q4_SDOT_BMAX 64
// Fixed-M specializations: MC is a compile-time constant so clang fully unrolls the
// column loops and keeps all MC int32x4 accumulators REGISTER-resident. The first
// (runtime-M) implementation of this kernel kept ac[] on the stack -- every vdot did a
// load+store round-trip -- and measured ~4-5x slower at M=1 than q4_dot_sdot_row; the
// batch_roofline.c "6.7x" was relative to that degraded baseline (post-mortem in the
// M20 bench notes). Register pressure: MC=8 needs ~15 live q-regs (8 acc + lo/hi/raw +
// mask/eight + loads); a 16-wide variant spilled and lost to 8+8 chunking (see the
// dispatcher note below), so MC tops out at 8.
// M20 activation pre-split: the M=1 kernel deinterleaves the activation in the hot loop
// (vld2q_s8) to match the lo/hi nibble order; in the batched kernel that zip runs
// out*ng*2*M times per GEMM. Instead the CALLER splits each quantized column ONCE into
// [lo16][hi16] chunk order (q4_split_act below, one pass over M*in bytes), and the fixed
// kernels read it with two plain vld1q_s8 -- identical VALUES in identical lanes, so the
// dot inputs (and thus bit-identity) are unchanged; only the load form differs.
// NOTE: q4 batched kernels REQUIRE split activation; q8 batched kernels keep natural
// order (int8 weights never interleave). The single-token gemv_* paths are untouched.
static inline void q4_split_act(const int8_t *restrict nat, int8_t *restrict split, int in) {
    for (int off = 0; off < in; off += 32) {          // in % 64 == 0 (asserted at load)
        int8x16x2_t xd = vld2q_s8(nat + off);
        vst1q_s8(split + off, xd.val[0]);
        vst1q_s8(split + off + 16, xd.val[1]);
    }
}
#define Q4_SDOT_ROW_BATCH_FIXED(MC) \
static void q4_sdot_row_batch_##MC(const uint8_t *restrict pr, const float *restrict wscale, \
                                   const int8_t *restrict xq, const float *restrict ascale, \
                                   int in, float b, float *restrict y, size_t ystep) { \
    const uint8x16_t mask = vdupq_n_u8(0x0F); const int8x16_t eight = vdupq_n_s8(8); \
    int ng = in >> 6; \
    float facc[MC]; \
    for (int c = 0; c < MC; c++) facc[c] = 0.0f; \
    for (int gi = 0; gi < ng; gi++) { \
        const uint8_t *gp = pr + (size_t)gi * 32; \
        int32x4_t ac[MC]; \
        for (int c = 0; c < MC; c++) ac[c] = vdupq_n_s32(0); \
        for (int bb = 0; bb < 32; bb += 16) { \
            uint8x16_t raw = vld1q_u8(gp + bb); \
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), eight); \
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight); \
            for (int c = 0; c < MC; c++) { \
                const int8_t *sp = xq + (size_t)c * in + (size_t)gi * 64 + (size_t)bb * 2; \
                ac[c] = vdotq_s32(ac[c], lo, vld1q_s8(sp)); \
                ac[c] = vdotq_s32(ac[c], hi, vld1q_s8(sp + 16)); \
            } \
        } \
        float ws = wscale[gi]; \
        for (int c = 0; c < MC; c++) facc[c] += (float)vaddvq_s32(ac[c]) * ws * ascale[(size_t)c * ng + gi]; \
    } \
    for (int c = 0; c < MC; c++) y[(size_t)c * ystep] = b + facc[c]; \
}
Q4_SDOT_ROW_BATCH_FIXED(1)
Q4_SDOT_ROW_BATCH_FIXED(2)
Q4_SDOT_ROW_BATCH_FIXED(4)
Q4_SDOT_ROW_BATCH_FIXED(8)
#define Q8_SDOT_ROW_BATCH_FIXED(MC) \
static void q8_sdot_row_batch_##MC(const int8_t *restrict pr, const float *restrict wscale, \
                                   const int8_t *restrict xq, const float *restrict ascale, \
                                   int in, float b, float *restrict y, size_t ystep) { \
    int ng = in >> 6; \
    float facc[MC]; \
    for (int c = 0; c < MC; c++) facc[c] = 0.0f; \
    for (int gi = 0; gi < ng; gi++) { \
        const int8_t *gp = pr + (size_t)gi * 64; \
        int32x4_t ac[MC]; \
        for (int c = 0; c < MC; c++) ac[c] = vdupq_n_s32(0); \
        for (int bb = 0; bb < 64; bb += 16) { \
            int8x16_t wv = vld1q_s8(gp + bb); \
            for (int c = 0; c < MC; c++) \
                ac[c] = vdotq_s32(ac[c], wv, vld1q_s8(xq + (size_t)c * in + (size_t)gi * 64 + bb)); \
        } \
        float ws = wscale[gi]; \
        for (int c = 0; c < MC; c++) facc[c] += (float)vaddvq_s32(ac[c]) * ws * ascale[(size_t)c * ng + gi]; \
    } \
    for (int c = 0; c < MC; c++) y[(size_t)c * ystep] = b + facc[c]; \
}
Q8_SDOT_ROW_BATCH_FIXED(1)
Q8_SDOT_ROW_BATCH_FIXED(2)
Q8_SDOT_ROW_BATCH_FIXED(4)
Q8_SDOT_ROW_BATCH_FIXED(8)
// Column-chunk dispatchers: greedy 8/4/2/1 WHILE-loop tiles over the M columns (while,
// not if: an if-chain silently DROPPED columns for M with repeated tile sizes -- M=16
// decomposed as 8+4+2 and never computed the last 2 columns; caught by the serve
// all-streams-identical gate during the tile-width A/B). Max tile is 8: a 16-wide
// specialization was measured SLOWER than 8+8 chunking (127 vs 148 agg tok/s at B=16 --
// 16 int32x4 accumulators + split-load addressing spill past the register file), so the
// 16-tile was removed. Each chunk re-runs the weight-row unpack (weight re-read per
// chunk), but every chunk's per-column arithmetic is IDENTICAL to the fixed kernel at
// that offset -> bit-identity per column is preserved for every M decomposition.
static inline void q4_sdot_row_batch(const uint8_t *restrict pr, const float *restrict wscale,
                                     const int8_t *restrict xq, const float *restrict ascale,
                                     int in, int M, float b, float *restrict y, size_t ystep) {
    int ng = in >> 6, c0 = 0;
    while (M - c0 >= 8) { q4_sdot_row_batch_8(pr, wscale, xq+(size_t)c0*in, ascale+(size_t)c0*ng, in, b, y+(size_t)c0*ystep, ystep); c0 += 8; }
    while (M - c0 >= 4) { q4_sdot_row_batch_4(pr, wscale, xq+(size_t)c0*in, ascale+(size_t)c0*ng, in, b, y+(size_t)c0*ystep, ystep); c0 += 4; }
    while (M - c0 >= 2) { q4_sdot_row_batch_2(pr, wscale, xq+(size_t)c0*in, ascale+(size_t)c0*ng, in, b, y+(size_t)c0*ystep, ystep); c0 += 2; }
    while (M - c0 >= 1) { q4_sdot_row_batch_1(pr, wscale, xq+(size_t)c0*in, ascale+(size_t)c0*ng, in, b, y+(size_t)c0*ystep, ystep); c0 += 1; }
}
static inline void q8_sdot_row_batch(const int8_t *restrict pr, const float *restrict wscale,
                                     const int8_t *restrict xq, const float *restrict ascale,
                                     int in, int M, float b, float *restrict y, size_t ystep) {
    int ng = in >> 6, c0 = 0;
    while (M - c0 >= 8) { q8_sdot_row_batch_8(pr, wscale, xq+(size_t)c0*in, ascale+(size_t)c0*ng, in, b, y+(size_t)c0*ystep, ystep); c0 += 8; }
    while (M - c0 >= 4) { q8_sdot_row_batch_4(pr, wscale, xq+(size_t)c0*in, ascale+(size_t)c0*ng, in, b, y+(size_t)c0*ystep, ystep); c0 += 4; }
    while (M - c0 >= 2) { q8_sdot_row_batch_2(pr, wscale, xq+(size_t)c0*in, ascale+(size_t)c0*ng, in, b, y+(size_t)c0*ystep, ystep); c0 += 2; }
    while (M - c0 >= 1) { q8_sdot_row_batch_1(pr, wscale, xq+(size_t)c0*in, ascale+(size_t)c0*ng, in, b, y+(size_t)c0*ystep, ystep); c0 += 1; }
}

// Phase 2 (M36, g256sf): bits sentinel for the group-256/subfold-O4 format. A sentinel on the
// ALWAYS-caller-set bits field (never 4/8) instead of a separate format flag, because q4job is
// a persistent pool member whose fields are set per-call by many existing entry points -- a new
// standalone flag would go stale between g256sf and legacy dispatches unless every existing
// entry were edited to clear it (exactly the class of shared-state hazard the sdot=0 resets
// below already guard against). Residual/error-feedback policy is unchanged by this dispatch
// extension: these are READ-ONLY inference paths; residual compensation stays upstream at
// quantization time (GPTQ error_feedback in the offline emitter), per the file-top note.
// EXIT: promote to a proper wfmt field if a third format appears.
#define Q4_BITS_G256SF 246

typedef struct {
    const uint8_t *packed; const float *scales, *x, *bias; float *y;
    int out, in, ng, bits, M; size_t row_bytes;   // bits: 4|8|Q4_BITS_G256SF; M: #token columns (>=1)
    int fused;   // M15 Phase C1: 1 = q4_dot_fused_row path (M=1 only), 0 = today's tile+dotpr path
    int sdot;              // M19: 1 = W4A8 int8-SDOT path (M=1 int4 only)
    int sdotb;             // M20: 1 = batched sdot path (fixed-M kernels; q4 expects SPLIT activation)
    const int8_t *xq;      // M19: pre-quantized int8 activation (sdot path)
    const float *ascale;   // M19: per-group activation scale (sdot path)
    // Phase 2 (M36, g256sf): only read when bits==Q4_BITS_G256SF (scales then holds wsuper
    // [out][in/256] and ng==in>>8). Function pointers, not direct calls, so this shared header
    // needs no symbol from q4gemv_g256.h (which is included only by TUs that opt in) -- TUs
    // that never dispatch g256sf (qwen_spec.c, test_q4gemv.c, batch_roofline.c) compile
    // unchanged. COST: one indirect call per output row, amortized over an entire row dot
    // (>=0.4us of work) -- noise. EXIT: hard-wire direct calls here if the indirection ever
    // shows up in a profile (requires moving the g256 kernels into this header).
    const uint8_t *subcode;   // [out][in/64] 6-bit sub-codes, one byte each
    float (*g256_row)(const uint8_t *, const float *, const uint8_t *, const int8_t *, const float *, int);
    void (*g256_tile)(const uint8_t *, const float *, const uint8_t *, const int8_t *, const float *, int, int, float, float *, size_t);
} q4job;

typedef struct q4pool q4pool;
typedef struct { q4pool *p; int wi; } q4warg;   // per-worker arg (was a global array;
                                                //  now pool-owned so 2 pools never collide)
struct q4pool {
    int nthreads, max_in;                       // max_in: tile capacity, guards in>max_in
    pthread_t th[16];
    pthread_mutex_t mtx;
    pthread_cond_t cv_go, cv_done;
    _Atomic int gen, done_count, stop;   // M29: atomic so the spin path can touch them
                                         // without the mutex; the condvar path uses
                                         // _explicit(..., memory_order_relaxed) uniformly --
                                         // the mutex alone supplies that path's happens-
                                         // before edge, so plain seq_cst (what bare ++/==
                                         // would silently upgrade to on an _Atomic-qualified
                                         // object) would only add cost there.
    int spin;              // M29: 0=condvar (default), 1=spin-wait instead of condvar wake
    int spin_iters;        // M29: bound on the busy-wait before sched_yield() (see M29 sweep)
    int qos;               // M29: 0=default (no QoS hint), 1=QOS_CLASS_USER_INTERACTIVE on workers
    q4job job;
    float *tiles[16];
    q4warg wargs[16];
    int8_t *xq_scratch;    // M19: int8 activation scratch (sized max_in)
    int8_t *xq_split;      // M27: pre-split ([lo16][hi16]) q4 activation scratch (sized max_in)
    float *ascale_scratch; // M19: per-group activation scale scratch (sized max_in/64)
};

static void *q4_worker(void *arg);

static inline void q4_run_range(q4pool *p, int wi) {
    q4job *j = &p->job;
    int per = (j->out + p->nthreads - 1) / p->nthreads;
    int r0 = wi * per, r1 = r0 + per; if (r1 > j->out) r1 = j->out;
    float *tile = p->tiles[wi];
    int M = j->M, in = j->in, out = j->out;
    if (j->sdot) {   // M19: int-SDOT path, M=1. bits==8 -> W8A8 (lm_head), else W4A8.
        if (j->bits == Q4_BITS_G256SF) {   // Phase 2 (M36): g256sf via the installed fn ptrs
            // (read-only inference dispatch; residual/error_feedback compensation is upstream
            // at quantization time, file-top note). scales==wsuper stride j->ng==in>>8;
            // subcode stride in>>6. Placed BEFORE the bits==8 checks so the sentinel can
            // never fall through into a g64-scale interpretation of wsuper.
            int ngsub = in >> 6;
            if (j->sdotb) {
                for (int r = r0; r < r1; r++) {
                    float b = j->bias ? j->bias[r] : 0.0f;
                    j->g256_tile(j->packed + (size_t)r * j->row_bytes, j->scales + (size_t)r * j->ng,
                                 j->subcode + (size_t)r * ngsub, j->xq, j->ascale, in, M, b,
                                 j->y + r, (size_t)out);
                }
            } else {
                for (int r = r0; r < r1; r++) {
                    float dot = j->g256_row(j->packed + (size_t)r * j->row_bytes,
                                            j->scales + (size_t)r * j->ng,
                                            j->subcode + (size_t)r * ngsub, j->xq, j->ascale, in);
                    j->y[r] = (j->bias ? j->bias[r] : 0.0f) + dot;
                }
            }
            return;
        }
        if (j->sdotb) {   // M20: batched serve path (any M>=1) -- fixed-M register-tiled kernels, y=[M][out]

            for (int r = r0; r < r1; r++) {
                float b = j->bias ? j->bias[r] : 0.0f;
                if (j->bits == 8)
                    q8_sdot_row_batch((const int8_t *)(j->packed + (size_t)r * j->row_bytes),
                                      j->scales + (size_t)r * j->ng, j->xq, j->ascale, in, M, b,
                                      j->y + r, (size_t)out);
                else
                    q4_sdot_row_batch(j->packed + (size_t)r * j->row_bytes,
                                      j->scales + (size_t)r * j->ng, j->xq, j->ascale, in, M, b,
                                      j->y + r, (size_t)out);
            }
            return;
        }
        if (j->bits == 8) {
            for (int r = r0; r < r1; r++) {
                float dot = q8_dot_sdot_row((const int8_t *)(j->packed + (size_t)r * j->row_bytes),
                                            j->scales + (size_t)r * j->ng, j->xq, j->ascale, in);
                j->y[r] = (j->bias ? j->bias[r] : 0.0f) + dot;
            }
        } else {
            for (int r = r0; r < r1; r++) {
                float dot = q4_dot_sdot_row(j->packed + (size_t)r * j->row_bytes,
                                            j->scales + (size_t)r * j->ng, j->xq, j->ascale, in);
                j->y[r] = (j->bias ? j->bias[r] : 0.0f) + dot;
            }
        }
        return;
    }
    if (j->fused) {   // M15 Phase C1: M=1 int4 only (asserted by the caller, gemv_q4g64_mt_fused)
        for (int r = r0; r < r1; r++) {
            float dot = q4_dot_fused_row(j->packed + (size_t)r * j->row_bytes,
                                          j->scales + (size_t)r * j->ng, j->x, in);
            j->y[r] = (j->bias ? j->bias[r] : 0.0f) + dot;
        }
        return;
    }
    for (int r = r0; r < r1; r++) {
        if (j->bits == 8)
            q8_unpack_row((const int8_t *)(j->packed + (size_t)r * j->row_bytes),
                          j->scales + (size_t)r * j->ng, tile, in);           // unpack ONCE
        else
            q4_unpack_row(j->packed + (size_t)r * j->row_bytes, j->scales + (size_t)r * j->ng,
                          tile, in);
        float b = j->bias ? j->bias[r] : 0.0f;
        for (int m = 0; m < M; m++) {                                          // reuse across M tokens
            float dot; vDSP_dotpr(tile, 1, j->x + (size_t)m * in, 1, &dot, in);
            j->y[(size_t)m * out + r] = b + dot;
        }
    }
}

#define Q4_SPIN_ITERS_DEFAULT 4000   // M29: bound on the busy-wait before sched_yield();
                                              // empirically swept via test_q4gemv.c's B2a probe
                                              // (RESULTS_QWEN_VDSP.md M29), overridable per-pool
                                              // via p->spin_iters (Q4_POOL_SPIN_ITERS env var).

static void *q4_worker(void *arg) {
    q4warg *wa = (q4warg *)arg; q4pool *p = wa->p; int wi = wa->wi;
    int last = 0;
    // M29: opt-in QoS hint so the scheduler prefers a P-core for this pool worker --
    // tests the H3 hypothesis (RESULTS_QWEN_VDSP.md M28) that an unpinned worker landing
    // on an E-core drags out the barrier join. Failure is non-fatal (logged, continues
    // unpinned) -- this is a scheduling hint, not a correctness requirement. Safe to read
    // here unsynchronized: p->qos is fixed by the caller before q4pool_start() spawns this
    // thread (see q4pool_start) and never written again -- pthread_create's own happens-
    // before edge covers it, no atomic needed.
    if (p->qos && pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) != 0)
        fprintf(stderr, "[engine] q4_worker %d: pthread_set_qos_class_self_np failed, continuing unpinned\n", wi);
    if (p->spin) {   // M29: spin-wait path -- p->spin is fixed-before-spawn, same argument as p->qos above
        for (;;) {
            int spins = 0, g;
            while ((g = atomic_load_explicit(&p->gen, memory_order_acquire)) == last) {
                if (atomic_load_explicit(&p->stop, memory_order_relaxed)) return NULL;
                if (++spins > p->spin_iters) { sched_yield(); spins = 0; }
            }
            last = g;
            q4_run_range(p, wi);
            atomic_fetch_add_explicit(&p->done_count, 1, memory_order_release);
        }
    }
    for (;;) {
        pthread_mutex_lock(&p->mtx);
        while (atomic_load_explicit(&p->gen, memory_order_relaxed) == last &&
               !atomic_load_explicit(&p->stop, memory_order_relaxed))
            pthread_cond_wait(&p->cv_go, &p->mtx);
        if (atomic_load_explicit(&p->stop, memory_order_relaxed)) { pthread_mutex_unlock(&p->mtx); return NULL; }
        last = atomic_load_explicit(&p->gen, memory_order_relaxed);
        pthread_mutex_unlock(&p->mtx);
        q4_run_range(p, wi);
        pthread_mutex_lock(&p->mtx);
        int dc = atomic_fetch_add_explicit(&p->done_count, 1, memory_order_relaxed) + 1;
        if (dc == p->nthreads) pthread_cond_signal(&p->cv_done);
        pthread_mutex_unlock(&p->mtx);
    }
}

static inline void q4pool_init(q4pool *p, int nthreads, int max_in) {
    if (nthreads < 1) nthreads = 1; if (nthreads > 16) nthreads = 16;
    p->nthreads = nthreads; p->max_in = max_in;
    atomic_init(&p->gen, 0); atomic_init(&p->done_count, 0); atomic_init(&p->stop, 0);
    p->spin = 0; p->spin_iters = Q4_SPIN_ITERS_DEFAULT; p->qos = 0;
    memset(&p->job, 0, sizeof(p->job));   // M19: deterministic job.sdot=0 etc. before first dispatch
    pthread_mutex_init(&p->mtx, NULL);
    pthread_cond_init(&p->cv_go, NULL); pthread_cond_init(&p->cv_done, NULL);
    size_t tbytes = (size_t)((max_in + 15) & ~15) * sizeof(float);
    for (int i = 0; i < nthreads; i++) {
        p->tiles[i] = (float *)aligned_alloc(64, tbytes);
        if (!p->tiles[i]) { fprintf(stderr, "FATAL: q4pool tile alloc failed\n"); exit(1); }
    }
    size_t xqsz = (size_t)((max_in + 63) & ~63);                       // multiple of 64
    size_t assz = (((size_t)((max_in / 64) + 16) * sizeof(float)) + 63) & ~63;   // round to 64
    p->xq_scratch = (int8_t *)aligned_alloc(64, xqsz);
    p->xq_split = (int8_t *)aligned_alloc(64, xqsz);   // M27: split q4 activation scratch
    p->ascale_scratch = (float *)aligned_alloc(64, assz);
    if (!p->xq_scratch || !p->xq_split || !p->ascale_scratch) { fprintf(stderr, "FATAL: q4pool sdot scratch alloc failed\n"); exit(1); }
    // M29: worker threads are NOT spawned here anymore -- call q4pool_start(p) after
    // setting p->spin/p->spin_iters/p->qos (if non-default), so those fields are fully
    // visible before any worker's first read. Spawning here, then setting fields after
    // (the original M29 draft), was a real data race -- in the spin case, a protocol-
    // mismatch deadlock (dispatcher bumps gen expecting spin-poll, worker still waits on
    // cv_go expecting a broadcast that never comes). See RESULTS_QWEN_VDSP.md M29.
}

// M29: spawns the pool's worker threads. Caller must set p->spin/p->spin_iters/p->qos
// (if non-default) BEFORE calling this -- q4pool_init() above no longer spawns threads.
static inline void q4pool_start(q4pool *p) {
    for (int i = 1; i < p->nthreads; i++) {
        p->wargs[i].p = p; p->wargs[i].wi = i;
        if (pthread_create(&p->th[i], NULL, q4_worker, &p->wargs[i]) != 0) {
            fprintf(stderr, "FATAL: q4pool pthread_create failed\n"); exit(1);
        }
    }
}

// Stop+join workers, free tiles, destroy sync primitives (library-reuse hygiene).
static inline void q4pool_destroy(q4pool *p) {
    if (p->spin) {
        atomic_store_explicit(&p->stop, 1, memory_order_release);   // spin workers poll stop themselves
    } else {
        pthread_mutex_lock(&p->mtx);
        atomic_store_explicit(&p->stop, 1, memory_order_relaxed);
        pthread_cond_broadcast(&p->cv_go);
        pthread_mutex_unlock(&p->mtx);
    }
    for (int i = 1; i < p->nthreads; i++) pthread_join(p->th[i], NULL);
    for (int i = 0; i < p->nthreads; i++) free(p->tiles[i]);
    free(p->xq_scratch); free(p->xq_split); free(p->ascale_scratch);
    pthread_mutex_destroy(&p->mtx);
    pthread_cond_destroy(&p->cv_go); pthread_cond_destroy(&p->cv_done);
}

// M29: the dispatch+wait pattern shared by all 7 gemv/gemm dispatch functions below,
// factored into one place so the spin-wait alternative is written once, not risk-
// duplicated 7x. Scope is deliberately narrow: lock->enqueue/broadcast->unlock->
// q4_run_range(p,0)->lock->wait->unlock, and NOTHING else -- callers still do their own
// job-field setup before calling this, and their own post-wait cleanup (e.g. clearing
// job.sdot/job.sdotb) after it returns.
static inline void q4pool_go_and_wait(q4pool *p) {
    if (p->spin) {
        atomic_store_explicit(&p->done_count, 1, memory_order_relaxed);   // safe: precedes
                                                                           // the gen release-
                                                                           // store below, same
                                                                           // thread -- do not
                                                                           // reorder
        atomic_fetch_add_explicit(&p->gen, 1, memory_order_release);
        q4_run_range(p, 0);
        int spins = 0;
        while (atomic_load_explicit(&p->done_count, memory_order_acquire) < p->nthreads) {
            if (++spins > p->spin_iters) { sched_yield(); spins = 0; }
        }
        return;
    }
    pthread_mutex_lock(&p->mtx);
    atomic_store_explicit(&p->done_count, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->gen, 1, memory_order_relaxed);
    pthread_cond_broadcast(&p->cv_go);
    pthread_mutex_unlock(&p->mtx);
    q4_run_range(p, 0);
    pthread_mutex_lock(&p->mtx);
    while (atomic_load_explicit(&p->done_count, memory_order_relaxed) < p->nthreads)
        pthread_cond_wait(&p->cv_done, &p->mtx);
    pthread_mutex_unlock(&p->mtx);
}

static inline void gemv_q4g64_mt(q4pool *p, const uint8_t *packed, const float *scales,
                                 const float *x, const float *bias, float *y,
                                 int out, int in) {
    if (in > p->max_in) { fprintf(stderr, "FATAL: gemv in=%d > pool max_in=%d\n", in, p->max_in); exit(1); }
    if (p->nthreads == 1) { gemv_q4g64(packed, scales, x, bias, y, out, in, p->tiles[0]); return; }
    p->job.packed = packed; p->job.scales = scales; p->job.x = x; p->job.bias = bias;
    p->job.y = y; p->job.out = out; p->job.in = in; p->job.ng = in >> 6; p->job.bits = 4; p->job.M = 1;
    p->job.row_bytes = (size_t)in / 2; p->job.fused = 0; p->job.sdot = 0;
    q4pool_go_and_wait(p);
}

// M15 Phase C1: sibling of gemv_q4g64_mt using the fused dequant-dot row kernel instead of
// tile+dotpr. M=1 (single-token) only -- gemm_qXg64_mt (batched verify) is untouched and
// never sets job.fused, so it always takes the original tile+dotpr path (M15-D3).
static inline void gemv_q4g64_mt_fused(q4pool *p, const uint8_t *packed, const float *scales,
                                        const float *x, const float *bias, float *y,
                                        int out, int in) {
    if (in > p->max_in) { fprintf(stderr, "FATAL: gemv-fused in=%d > pool max_in=%d\n", in, p->max_in); exit(1); }
    if (p->nthreads == 1) { gemv_q4g64_fused(packed, scales, x, bias, y, out, in); return; }
    p->job.packed = packed; p->job.scales = scales; p->job.x = x; p->job.bias = bias;
    p->job.y = y; p->job.out = out; p->job.in = in; p->job.ng = in >> 6; p->job.bits = 4; p->job.M = 1;
    p->job.row_bytes = (size_t)in / 2; p->job.fused = 1;
    q4pool_go_and_wait(p);
}

// M19: W4A8 int8-SDOT sibling of gemv_q4g64_mt. Quantizes the activation to int8
// ONCE (shared across all rows/threads), then dispatches the sdot row kernel. M=1
// single-token only. NOT bit-exact vs the fp32 path -- the activation is int8
// (gated on ppl, M19 step 1: +0.2%), the weight path is unchanged.
static inline void gemv_q4g64_sdot_mt(q4pool *p, const uint8_t *packed, const float *scales,
                                      const float *x, const float *bias, float *y,
                                      int out, int in) {
    if (in > p->max_in) { fprintf(stderr, "FATAL: sdot gemv in=%d > pool max_in=%d\n", in, p->max_in); exit(1); }
    q4_quant_act_i8(x, in, p->xq_scratch, p->ascale_scratch);
    q4_split_act(p->xq_scratch, p->xq_split, in);   // M27: split ONCE per token, reused by all rows
    if (p->nthreads == 1) {
        for (int r = 0; r < out; r++) {
            float dot = q4_dot_sdot_row(packed + (size_t)r * ((size_t)in / 2),
                                        scales + (size_t)r * (in >> 6),
                                        p->xq_split, p->ascale_scratch, in);
            y[r] = (bias ? bias[r] : 0.0f) + dot;
        }
        return;
    }
    p->job.packed = packed; p->job.scales = scales; p->job.x = x; p->job.bias = bias;
    p->job.y = y; p->job.out = out; p->job.in = in; p->job.ng = in >> 6; p->job.bits = 4; p->job.M = 1;
    p->job.row_bytes = (size_t)in / 2; p->job.fused = 0; p->job.sdot = 1;
    p->job.xq = p->xq_split; p->job.ascale = p->ascale_scratch;   // M27: q4 workers read the split layout
    q4pool_go_and_wait(p);
    p->job.sdot = 0;   // leave the shared job flag clean for the next non-sdot dispatch
}

// M19: W8A8 int8-SDOT sibling of gemv_q8g64_mt (lm_head). Same activation-int8
// quantize-once then dispatch; int8 weights, no unpack.
static inline void gemv_q8g64_sdot_mt(q4pool *p, const uint8_t *packed, const float *scales,
                                      const float *x, const float *bias, float *y,
                                      int out, int in) {
    if (in > p->max_in) { fprintf(stderr, "FATAL: sdot8 gemv in=%d > pool max_in=%d\n", in, p->max_in); exit(1); }
    q4_quant_act_i8(x, in, p->xq_scratch, p->ascale_scratch);
    if (p->nthreads == 1) {
        for (int r = 0; r < out; r++) {
            float dot = q8_dot_sdot_row((const int8_t *)packed + (size_t)r * in,
                                        scales + (size_t)r * (in >> 6),
                                        p->xq_scratch, p->ascale_scratch, in);
            y[r] = (bias ? bias[r] : 0.0f) + dot;
        }
        return;
    }
    p->job.packed = packed; p->job.scales = scales; p->job.x = x; p->job.bias = bias;
    p->job.y = y; p->job.out = out; p->job.in = in; p->job.ng = in >> 6; p->job.bits = 8; p->job.M = 1;
    p->job.row_bytes = (size_t)in; p->job.fused = 0; p->job.sdot = 1;
    p->job.xq = p->xq_scratch; p->job.ascale = p->ascale_scratch;
    q4pool_go_and_wait(p);
    p->job.sdot = 0;
}

static inline void gemv_q8g64_mt(q4pool *p, const uint8_t *packed, const float *scales,
                                 const float *x, const float *bias, float *y,
                                 int out, int in) {
    if (in > p->max_in) { fprintf(stderr, "FATAL: q8 gemv in=%d > pool max_in=%d\n", in, p->max_in); exit(1); }
    if (p->nthreads == 1) { gemv_q8g64((const int8_t *)packed, scales, x, bias, y, out, in, p->tiles[0]); return; }
    p->job.packed = packed; p->job.scales = scales; p->job.x = x; p->job.bias = bias;
    p->job.y = y; p->job.out = out; p->job.in = in; p->job.ng = in >> 6; p->job.bits = 8; p->job.M = 1;
    p->job.row_bytes = (size_t)in; p->job.fused = 0;
    q4pool_go_and_wait(p);
}

// ---- batched (M token columns) variants: unpack each weight row ONCE, dot with all M tokens.
// The whole point of speculative decode: verify M tokens for ~1 token's weight traffic.
static inline void gemm_qXg64_mt(q4pool *p, int bits, const uint8_t *packed, const float *scales,
                                 const float *x, const float *bias, float *y, int out, int in, int M) {
    if (in > p->max_in) { fprintf(stderr, "FATAL: gemm in=%d > pool max_in=%d\n", in, p->max_in); exit(1); }
    p->job.packed = packed; p->job.scales = scales; p->job.x = x; p->job.bias = bias;
    p->job.y = y; p->job.out = out; p->job.in = in; p->job.ng = in >> 6; p->job.bits = bits;
    p->job.M = M; p->job.row_bytes = (bits == 8) ? (size_t)in : (size_t)in / 2; p->job.fused = 0;
    if (p->nthreads == 1) { q4_run_range(p, 0); return; }
    q4pool_go_and_wait(p);
}

// M20 (serve): batched int8-SDOT GEMM entry. xq=[M][in] int8 activation columns,
// ascale=[M][in/64] per-group scales -- quantized by the CALLER (q4_quant_act_i8 per
// column), NOT via the pool's M=1 scratch, so one quantization of a layer's normed
// activation can be shared across the q/k/v (resp. gate/up) projections. y=[M][out].
// Threads over output rows via the existing pool; each row is computed fully by one
// thread in fixed order -> bitwise deterministic like every other pool kernel here.
// int4-residual-guard: inference-only, residual/error-feedback applied upstream at
// quantization time (see file-top note).
static inline void gemm_qXg64_sdot_mt(q4pool *p, int bits, const uint8_t *packed, const float *scales,
                                      const int8_t *xq, const float *ascale, const float *bias,
                                      float *y, int out, int in, int M) {
    if (in > p->max_in) { fprintf(stderr, "FATAL: sdot gemm in=%d > pool max_in=%d\n", in, p->max_in); exit(1); }
    if (M < 1 || M > Q4_SDOT_BMAX) { fprintf(stderr, "FATAL: sdot gemm M=%d out of [1,%d]\n", M, Q4_SDOT_BMAX); exit(1); }
    p->job.packed = packed; p->job.scales = scales; p->job.x = NULL; p->job.bias = bias;
    p->job.y = y; p->job.out = out; p->job.in = in; p->job.ng = in >> 6; p->job.bits = bits;
    p->job.M = M; p->job.row_bytes = (bits == 8) ? (size_t)in : (size_t)in / 2;
    p->job.fused = 0; p->job.sdot = 1; p->job.sdotb = 1; p->job.xq = xq; p->job.ascale = ascale;
    if (p->nthreads == 1) { q4_run_range(p, 0); p->job.sdot = 0; p->job.sdotb = 0; return; }
    q4pool_go_and_wait(p);
    p->job.sdot = 0; p->job.sdotb = 0;   // leave the shared job flags clean for the next dispatch
}

#endif
