// attn_neon.h -- hand-written NEON attention (QK^T + weighted-V-sum), no BLAS.
//
// M13 background: M11 tried strided cblas_sgemv (lda=KVD, no reorg) -- ~29% slower
// than the naive per-position vDSP_dotpr/vDSP_vsma loop. M12 tried the "fix" (KV
// cache reorganized for lda==HD contiguity) -- STILL slower, worse than M11 (mean
// 38.6%). Conclusion from both: memory layout was never the bottleneck; cblas_sgemv
// itself has a per-call floor higher than vDSP's specialized small-vector kernels,
// for matrix sizes in this range (few hundred to ~1500 rows x 128 cols). This header
// stops going through general BLAS entirely and hand-writes the same shape vDSP_dotpr/
// vDSP_vsma already specialize for, testing whether inlined NEON can match or beat
// them without BLAS's dispatch cost.
//
// M13-D2: reads the ORIGINAL kslot/vslot cache (position-major, KVD stride) -- NOT
//   M12's kv-head-major mirror. M12 already proved layout/stride wasn't the issue,
//   and 128-float rows at a fixed stride are already prefetcher-friendly; the mirror
//   would only add memory and write traffic for nothing this kernel needs.
//   WHY: isolate the "inlined NEON vs BLAS dispatch" variable cleanly.
//   COST: none identified -- the mirror bought nothing measurable in M12 either.
//   EXIT: delete this header and its call sites; kslot/vslot and the naive loop are
//   untouched by anything here.
//
// Two independent mechanisms, tested separately so a win or loss can be attributed:
//
//   Variant A (attn_qk_neon / attn_wsum_neon) -- per-head drop-in. Same loop nesting
//     as naive (one head at a time), but the O(pos) dylib calls become one inlined
//     function with 4-position register blocking (amortizes the q/score reload and
//     gives 4 independent FMA chains to hide latency). Isolates "per-call overhead
//     elimination" as its own effect.
//
//   Variant B (attn_qk_group_neon / attn_wsum_group_neon) -- fused per-KV-group.
//     GROUP=6 query heads share one KV head's data (GQA); naive and variant A both
//     re-stream that KV-head's K/V slice once PER QUERY HEAD (6x re-reads at ctx 1536
//     that's 768 KB re-streamed 6 times, from L2 at best). This variant streams K and
//     V ONCE per KV-head and applies each row to all 6 heads' accumulators in the same
//     pass -- the one mechanism neither M11 nor M12 (nor variant A) tested.
//
// Exactness note: both variants preserve the naive loop's summation ORDER (positions
// processed t=0,1,2,...,n-1 in program order; each fixed output element is read-
// accumulated-written every position, not summed in a different tree order) -- the
// only reduction-order divergence from vDSP_dotpr/vDSP_vsma is WITHIN one position's
// 128-wide dot product (32-lane NEON tree vs vDSP's internal order), the same
// same-class-not-bit-identical fp32 noise this project has already documented and
// accepted since M6/M7 and reconfirmed in M11/M12 (~1e-5 max-abs logit class).
//
// AHD/AGROUP are private constants (not the includer's HD/GROUP macros) so this
// header is includable standalone by test_attn_neon.c without pulling in qwen_infer.c.
// qwen_infer.c asserts HD==AHD and GROUP==AGROUP at its own include site to catch any
// future model-config drift.

#ifndef ATTN_NEON_H
#define ATTN_NEON_H
#include <arm_neon.h>

#define AHD 128     // head dim -- exactly 32 float32x4_t lanes, no column tail ever
#define AGROUP 6    // query heads per KV head for this model (NH=12, NKV=2)

// ---- Variant A: per-head drop-in (isolates per-call-overhead elimination) ----

// scores[t] = scale * dot(qh, K[t]) for t in [0, n). kbase points at position 0's
// K row for this (layer, kv_head); consecutive positions are `stridef` floats apart
// (stridef == KVD for the original kslot() layout).
static inline void attn_qk_neon(const float *restrict qh, const float *restrict kbase,
                                 long stridef, int n, float scale, float *restrict scores) {
    int t = 0;
    for (; t + 4 <= n; t += 4) {
        const float *k0 = kbase + (long)(t + 0) * stridef;
        const float *k1 = kbase + (long)(t + 1) * stridef;
        const float *k2 = kbase + (long)(t + 2) * stridef;
        const float *k3 = kbase + (long)(t + 3) * stridef;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
        for (int d = 0; d < AHD; d += 4) {
            float32x4_t qv = vld1q_f32(qh + d);
            acc0 = vfmaq_f32(acc0, qv, vld1q_f32(k0 + d));
            acc1 = vfmaq_f32(acc1, qv, vld1q_f32(k1 + d));
            acc2 = vfmaq_f32(acc2, qv, vld1q_f32(k2 + d));
            acc3 = vfmaq_f32(acc3, qv, vld1q_f32(k3 + d));
        }
        scores[t + 0] = vaddvq_f32(acc0) * scale;
        scores[t + 1] = vaddvq_f32(acc1) * scale;
        scores[t + 2] = vaddvq_f32(acc2) * scale;
        scores[t + 3] = vaddvq_f32(acc3) * scale;
    }
    for (; t < n; t++) {
        const float *kt = kbase + (long)t * stridef;
        float32x4_t acc = vdupq_n_f32(0);
        for (int d = 0; d < AHD; d += 4) acc = vfmaq_f32(acc, vld1q_f32(qh + d), vld1q_f32(kt + d));
        scores[t] = vaddvq_f32(acc) * scale;
    }
}

// ah[d] = sum_t scores[t] * V[t][d], t in [0, n). ah zeroed internally (replaces the
// caller's memset). vbase/stridef same convention as attn_qk_neon.
static inline void attn_wsum_neon(const float *restrict scores, const float *restrict vbase,
                                   long stridef, int n, float *restrict ah) {
    for (int d = 0; d < AHD; d += 4) vst1q_f32(ah + d, vdupq_n_f32(0));
    int t = 0;
    for (; t + 4 <= n; t += 4) {
        const float *v0 = vbase + (long)(t + 0) * stridef;
        const float *v1 = vbase + (long)(t + 1) * stridef;
        const float *v2 = vbase + (long)(t + 2) * stridef;
        const float *v3 = vbase + (long)(t + 3) * stridef;
        float s0 = scores[t + 0], s1 = scores[t + 1], s2 = scores[t + 2], s3 = scores[t + 3];
        for (int d = 0; d < AHD; d += 4) {
            float32x4_t acc = vld1q_f32(ah + d);
            acc = vfmaq_n_f32(acc, vld1q_f32(v0 + d), s0);
            acc = vfmaq_n_f32(acc, vld1q_f32(v1 + d), s1);
            acc = vfmaq_n_f32(acc, vld1q_f32(v2 + d), s2);
            acc = vfmaq_n_f32(acc, vld1q_f32(v3 + d), s3);
            vst1q_f32(ah + d, acc);
        }
    }
    for (; t < n; t++) {
        const float *vt = vbase + (long)t * stridef;
        float s = scores[t];
        for (int d = 0; d < AHD; d += 4) {
            float32x4_t acc = vld1q_f32(ah + d);
            acc = vfmaq_n_f32(acc, vld1q_f32(vt + d), s);
            vst1q_f32(ah + d, acc);
        }
    }
}

// ---- Variant B: fused per-KV-group (isolates K/V-reread elimination across GQA) ----

// qg points at AGROUP*AHD contiguous floats (all 6 query heads sharing this KV head --
// already contiguous in q[] by construction: heads kvh*GROUP..kvh*GROUP+GROUP-1).
// scores[g] (g in [0,AGROUP)) are caller-owned row buffers, each holding this head's
// scores[0..n). Streams K once; applies each row to all 6 heads' accumulators.
static inline void attn_qk_group_neon(const float *restrict qg, const float *restrict kbase,
                                       long stridef, int n, float scale,
                                       float *scores[AGROUP]) {
    for (int t = 0; t < n; t++) {
        const float *kt = kbase + (long)t * stridef;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0), acc2 = vdupq_n_f32(0);
        float32x4_t acc3 = vdupq_n_f32(0), acc4 = vdupq_n_f32(0), acc5 = vdupq_n_f32(0);
        for (int d = 0; d < AHD; d += 4) {
            float32x4_t kv = vld1q_f32(kt + d);
            acc0 = vfmaq_f32(acc0, kv, vld1q_f32(qg + 0 * AHD + d));
            acc1 = vfmaq_f32(acc1, kv, vld1q_f32(qg + 1 * AHD + d));
            acc2 = vfmaq_f32(acc2, kv, vld1q_f32(qg + 2 * AHD + d));
            acc3 = vfmaq_f32(acc3, kv, vld1q_f32(qg + 3 * AHD + d));
            acc4 = vfmaq_f32(acc4, kv, vld1q_f32(qg + 4 * AHD + d));
            acc5 = vfmaq_f32(acc5, kv, vld1q_f32(qg + 5 * AHD + d));
        }
        scores[0][t] = vaddvq_f32(acc0) * scale;
        scores[1][t] = vaddvq_f32(acc1) * scale;
        scores[2][t] = vaddvq_f32(acc2) * scale;
        scores[3][t] = vaddvq_f32(acc3) * scale;
        scores[4][t] = vaddvq_f32(acc4) * scale;
        scores[5][t] = vaddvq_f32(acc5) * scale;
    }
}

// ag: AGROUP*AHD contiguous floats out (already contiguous in attn[] by construction,
// same layout as qg). Streams V once; RMW's all 6 heads' accumulators per position
// (6*AHD*4=3KB stays L1-hot across the whole n loop). Zeroed internally.
static inline void attn_wsum_group_neon(float *scores[AGROUP], const float *restrict vbase,
                                         long stridef, int n, float *restrict ag) {
    for (int i = 0; i < AGROUP * AHD; i += 4) vst1q_f32(ag + i, vdupq_n_f32(0));
    for (int t = 0; t < n; t++) {
        const float *vt = vbase + (long)t * stridef;
        float s0 = scores[0][t], s1 = scores[1][t], s2 = scores[2][t];
        float s3 = scores[3][t], s4 = scores[4][t], s5 = scores[5][t];
        for (int d = 0; d < AHD; d += 4) {
            float32x4_t vv = vld1q_f32(vt + d);
            float32x4_t a0 = vld1q_f32(ag + 0 * AHD + d); a0 = vfmaq_n_f32(a0, vv, s0); vst1q_f32(ag + 0 * AHD + d, a0);
            float32x4_t a1 = vld1q_f32(ag + 1 * AHD + d); a1 = vfmaq_n_f32(a1, vv, s1); vst1q_f32(ag + 1 * AHD + d, a1);
            float32x4_t a2 = vld1q_f32(ag + 2 * AHD + d); a2 = vfmaq_n_f32(a2, vv, s2); vst1q_f32(ag + 2 * AHD + d, a2);
            float32x4_t a3 = vld1q_f32(ag + 3 * AHD + d); a3 = vfmaq_n_f32(a3, vv, s3); vst1q_f32(ag + 3 * AHD + d, a3);
            float32x4_t a4 = vld1q_f32(ag + 4 * AHD + d); a4 = vfmaq_n_f32(a4, vv, s4); vst1q_f32(ag + 4 * AHD + d, a4);
            float32x4_t a5 = vld1q_f32(ag + 5 * AHD + d); a5 = vfmaq_n_f32(a5, vv, s5); vst1q_f32(ag + 5 * AHD + d, a5);
        }
    }
}


// ---- M23: int8-KV attention (QWEN_KV_INT8=1) ----
//
// The KV cache stores K and V as symmetric int8 with one fp32 scale per group-64
// along HD (2 scales per head per position; quantized by q4_quant_act_i8, the same
// function the W4A8 activation axis already uses). These kernels are the group-fused
// (variant-B) shape only -- when KV-int8 is on the engine collapses every attn_mode
// to this one implementation (M23-D4 in qwen_infer.c).
//
// attn_qk_group_i8: scores via int8 x int8 vdotq_s32 (the SDOT thesis applied to the
//   last fp32 hot-path operand). Per position the two 64-wide group sums are
//   accumulated in int32 EXACTLY, then scaled: score = (g0*ks0*qs0 + g1*ks1*qs1)*scale.
//   Same combined-scale structure as q8_dot_sdot_row, fused 6-wide across the GQA
//   group so K streams once per KV head (variant B's mechanism, kept).
//
// attn_wsum_group_i8: weighted-V keeps the fp32 softmax probs (no prob quantization
//   -- M23-D3: probs span [0,1] with mass concentrated on few positions; int8-izing
//   them is a second, uncontrolled error axis and there is no SDOT win available
//   because the reduction runs over t, not over the contiguous d dim in this
//   position-major layout). Each V row is dequantized ONCE per position into a
//   stack row (int8->int16->int32->fp32 widen, x its group scale), then applied to
//   all 6 heads' accumulators -- the exact attn_wsum_group_neon RMW pattern on a
//   dequantized row. V streams once per KV head at 1/4 the bytes.
//
// int4-residual-guard note (residual / error_feedback / compensat): these are
// READ-ONLY inference kernels over an already-quantized int8 cache; residual /
// error_feedback compensation is intentionally NOT applied at runtime, the same
// policy as the M19/M20 SDOT kernels. The KV quant here is a per-token dynamic
// quant: each (pos, group) is quantized exactly once and never revisited, so there
// is no accumulation loop for an error-feedback term to close over.

// qg: AGROUP*AHD contiguous int8 (the 6 query heads sharing this KV head, quantized
// per group-64); qsc: AGROUP*2 fp32 scales (2 per head, contiguous). kqbase points at
// position 0's int8 K row for this (layer, kv_head); consecutive positions are
// stride_b BYTES apart. kscbase: position 0's 2 K scales for this kv_head; consecutive
// positions are sstride floats apart.
static inline void attn_qk_group_i8(const int8_t *restrict qg, const float *restrict qsc,
                                    const int8_t *restrict kqbase, long stride_b,
                                    const float *restrict kscbase, long sstride,
                                    int n, float scale, float *scores[AGROUP]) {
    for (int t = 0; t < n; t++) {
        const int8_t *kt = kqbase + (long)t * stride_b;
        float ks0 = kscbase[(long)t * sstride], ks1 = kscbase[(long)t * sstride + 1];
        int32x4_t a0l = vdupq_n_s32(0), a1l = vdupq_n_s32(0), a2l = vdupq_n_s32(0);
        int32x4_t a3l = vdupq_n_s32(0), a4l = vdupq_n_s32(0), a5l = vdupq_n_s32(0);
        int32x4_t a0h = vdupq_n_s32(0), a1h = vdupq_n_s32(0), a2h = vdupq_n_s32(0);
        int32x4_t a3h = vdupq_n_s32(0), a4h = vdupq_n_s32(0), a5h = vdupq_n_s32(0);
        for (int b = 0; b < 64; b += 16) {          // group 0: dims 0..63
            int8x16_t kv = vld1q_s8(kt + b);
            a0l = vdotq_s32(a0l, kv, vld1q_s8(qg + 0 * AHD + b));
            a1l = vdotq_s32(a1l, kv, vld1q_s8(qg + 1 * AHD + b));
            a2l = vdotq_s32(a2l, kv, vld1q_s8(qg + 2 * AHD + b));
            a3l = vdotq_s32(a3l, kv, vld1q_s8(qg + 3 * AHD + b));
            a4l = vdotq_s32(a4l, kv, vld1q_s8(qg + 4 * AHD + b));
            a5l = vdotq_s32(a5l, kv, vld1q_s8(qg + 5 * AHD + b));
        }
        for (int b = 64; b < AHD; b += 16) {        // group 1: dims 64..127
            int8x16_t kv = vld1q_s8(kt + b);
            a0h = vdotq_s32(a0h, kv, vld1q_s8(qg + 0 * AHD + b));
            a1h = vdotq_s32(a1h, kv, vld1q_s8(qg + 1 * AHD + b));
            a2h = vdotq_s32(a2h, kv, vld1q_s8(qg + 2 * AHD + b));
            a3h = vdotq_s32(a3h, kv, vld1q_s8(qg + 3 * AHD + b));
            a4h = vdotq_s32(a4h, kv, vld1q_s8(qg + 4 * AHD + b));
            a5h = vdotq_s32(a5h, kv, vld1q_s8(qg + 5 * AHD + b));
        }
        scores[0][t] = ((float)vaddvq_s32(a0l) * ks0 * qsc[0]  + (float)vaddvq_s32(a0h) * ks1 * qsc[1])  * scale;
        scores[1][t] = ((float)vaddvq_s32(a1l) * ks0 * qsc[2]  + (float)vaddvq_s32(a1h) * ks1 * qsc[3])  * scale;
        scores[2][t] = ((float)vaddvq_s32(a2l) * ks0 * qsc[4]  + (float)vaddvq_s32(a2h) * ks1 * qsc[5])  * scale;
        scores[3][t] = ((float)vaddvq_s32(a3l) * ks0 * qsc[6]  + (float)vaddvq_s32(a3h) * ks1 * qsc[7])  * scale;
        scores[4][t] = ((float)vaddvq_s32(a4l) * ks0 * qsc[8]  + (float)vaddvq_s32(a4h) * ks1 * qsc[9])  * scale;
        scores[5][t] = ((float)vaddvq_s32(a5l) * ks0 * qsc[10] + (float)vaddvq_s32(a5h) * ks1 * qsc[11]) * scale;
    }
}

// ag: AGROUP*AHD contiguous fp32 out (zeroed internally). vqbase/stride_b/vscbase/
// sstride: same convention as attn_qk_group_i8 but for the int8 V cache.
static inline void attn_wsum_group_i8(float *scores[AGROUP], const int8_t *restrict vqbase,
                                      long stride_b, const float *restrict vscbase, long sstride,
                                      int n, float *restrict ag) {
    for (int i = 0; i < AGROUP * AHD; i += 4) vst1q_f32(ag + i, vdupq_n_f32(0));
    float vrow[AHD];
    for (int t = 0; t < n; t++) {
        const int8_t *vt = vqbase + (long)t * stride_b;
        float vs0 = vscbase[(long)t * sstride], vs1 = vscbase[(long)t * sstride + 1];
        for (int d = 0; d < AHD; d += 16) {         // dequant the row once (16 | 64: no group straddle)
            float32x4_t vs = vdupq_n_f32(d < 64 ? vs0 : vs1);
            int8x16_t b = vld1q_s8(vt + d);
            int16x8_t l16 = vmovl_s8(vget_low_s8(b)), h16 = vmovl_s8(vget_high_s8(b));
            vst1q_f32(vrow + d + 0,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(l16))),  vs));
            vst1q_f32(vrow + d + 4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(l16))), vs));
            vst1q_f32(vrow + d + 8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(h16))),  vs));
            vst1q_f32(vrow + d + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(h16))), vs));
        }
        float s0 = scores[0][t], s1 = scores[1][t], s2 = scores[2][t];
        float s3 = scores[3][t], s4 = scores[4][t], s5 = scores[5][t];
        for (int d = 0; d < AHD; d += 4) {          // same RMW pattern as attn_wsum_group_neon
            float32x4_t vv = vld1q_f32(vrow + d);
            float32x4_t a0 = vld1q_f32(ag + 0 * AHD + d); a0 = vfmaq_n_f32(a0, vv, s0); vst1q_f32(ag + 0 * AHD + d, a0);
            float32x4_t a1 = vld1q_f32(ag + 1 * AHD + d); a1 = vfmaq_n_f32(a1, vv, s1); vst1q_f32(ag + 1 * AHD + d, a1);
            float32x4_t a2 = vld1q_f32(ag + 2 * AHD + d); a2 = vfmaq_n_f32(a2, vv, s2); vst1q_f32(ag + 2 * AHD + d, a2);
            float32x4_t a3 = vld1q_f32(ag + 3 * AHD + d); a3 = vfmaq_n_f32(a3, vv, s3); vst1q_f32(ag + 3 * AHD + d, a3);
            float32x4_t a4 = vld1q_f32(ag + 4 * AHD + d); a4 = vfmaq_n_f32(a4, vv, s4); vst1q_f32(ag + 4 * AHD + d, a4);
            float32x4_t a5 = vld1q_f32(ag + 5 * AHD + d); a5 = vfmaq_n_f32(a5, vv, s5); vst1q_f32(ag + 5 * AHD + d, a5);
        }
    }
}

// M23 mode 2 (QWEN_KV_INT8=2): int8 STORAGE, fp32 COMPUTE -- dequantize each K row once
// per position and run the exact fp32 group dot on it (fp32 query, no query quantization).
// Purpose: attribution + quality fallback. Comparing mode 1 (full int8-SDOT scores) against
// mode 2 (same stored bytes, fp32 scores) isolates how much ppl loss comes from the score
// compute path (Q quant + int8 dot rounding) vs the cache storage rounding itself.
static inline void attn_qk_group_i8f(const float *restrict qg, const int8_t *restrict kqbase,
                                     long stride_b, const float *restrict kscbase, long sstride,
                                     int n, float scale, float *scores[AGROUP]) {
    float krow[AHD];
    for (int t = 0; t < n; t++) {
        const int8_t *kt = kqbase + (long)t * stride_b;
        float ks0 = kscbase[(long)t * sstride], ks1 = kscbase[(long)t * sstride + 1];
        for (int d = 0; d < AHD; d += 16) {
            float32x4_t ks = vdupq_n_f32(d < 64 ? ks0 : ks1);
            int8x16_t b = vld1q_s8(kt + d);
            int16x8_t l16 = vmovl_s8(vget_low_s8(b)), h16 = vmovl_s8(vget_high_s8(b));
            vst1q_f32(krow + d + 0,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(l16))),  ks));
            vst1q_f32(krow + d + 4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(l16))), ks));
            vst1q_f32(krow + d + 8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(h16))),  ks));
            vst1q_f32(krow + d + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(h16))), ks));
        }
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0), acc2 = vdupq_n_f32(0);
        float32x4_t acc3 = vdupq_n_f32(0), acc4 = vdupq_n_f32(0), acc5 = vdupq_n_f32(0);
        for (int d = 0; d < AHD; d += 4) {
            float32x4_t kv = vld1q_f32(krow + d);
            acc0 = vfmaq_f32(acc0, kv, vld1q_f32(qg + 0 * AHD + d));
            acc1 = vfmaq_f32(acc1, kv, vld1q_f32(qg + 1 * AHD + d));
            acc2 = vfmaq_f32(acc2, kv, vld1q_f32(qg + 2 * AHD + d));
            acc3 = vfmaq_f32(acc3, kv, vld1q_f32(qg + 3 * AHD + d));
            acc4 = vfmaq_f32(acc4, kv, vld1q_f32(qg + 4 * AHD + d));
            acc5 = vfmaq_f32(acc5, kv, vld1q_f32(qg + 5 * AHD + d));
        }
        scores[0][t] = vaddvq_f32(acc0) * scale;
        scores[1][t] = vaddvq_f32(acc1) * scale;
        scores[2][t] = vaddvq_f32(acc2) * scale;
        scores[3][t] = vaddvq_f32(acc3) * scale;
        scores[4][t] = vaddvq_f32(acc4) * scale;
        scores[5][t] = vaddvq_f32(acc5) * scale;
    }
}

// ---- M14 Phase 3a: draft-model shape (HD_D=64, GROUP_D=7) ----
//
// qwen_spec.c's draft model (Qwen2.5-0.5B) has NH_D=14, NKV_D=2, HD_D=64 -- genuinely
// different from the target's HD=128/GROUP=6 that AHD/AGROUP above are hardcoded for.
// M14-D2: purpose-built duplicate, not a parameterized/generic version of variant
//   A/B above -- matches this project's established style (q4gemv.h already has a
//   group-64-specific kernel alongside its general one; same precedent applies here).
//   WHY: a generic stridef/n-only kernel would need AHD/AGROUP as runtime params,
//   losing the compile-time-constant loop bounds the NEON codegen above relies on.
//   COST: ~2x source duplication in this header (acceptable, small, and each half is
//   independently readable/testable -- test_attn_neon.c covers both shapes).
//   EXIT: if the draft kill gate (G-M14c, test_attn_neon.c) doesn't clear >=10% faster
//   than naive-at-HD64 at n>=512, delete this section and its call sites; the target-
//   side kernels above are untouched either way.
//
// Register-budget check (why GROUP=7 is still safe): variant B's fused form keeps
// AGROUP7 live float32x4_t accumulators across the inner d-loop (7, one per query
// head sharing this KV head) plus one transient for the streamed K/V row -- 8 live
// vector registers at any point, well inside AArch64's 32-register NEON file (v0-v31).
// HD_D=64 halves the inner d-loop trip count vs AHD=128 (16 lanes of 4 instead of 32);
// this has no effect on register pressure, only on how many FMA chains run per row.

#define AHD64 64     // draft model head dim -- 16 float32x4_t lanes, no column tail
#define AGROUP7 7    // draft model: NH_D=14, NKV_D=2 -> 7 query heads per KV head

// ---- Variant A (h64): per-head drop-in, draft shape ----

static inline void attn_qk_neon_h64(const float *restrict qh, const float *restrict kbase,
                                     long stridef, int n, float scale, float *restrict scores) {
    int t = 0;
    for (; t + 4 <= n; t += 4) {
        const float *k0 = kbase + (long)(t + 0) * stridef;
        const float *k1 = kbase + (long)(t + 1) * stridef;
        const float *k2 = kbase + (long)(t + 2) * stridef;
        const float *k3 = kbase + (long)(t + 3) * stridef;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
        for (int d = 0; d < AHD64; d += 4) {
            float32x4_t qv = vld1q_f32(qh + d);
            acc0 = vfmaq_f32(acc0, qv, vld1q_f32(k0 + d));
            acc1 = vfmaq_f32(acc1, qv, vld1q_f32(k1 + d));
            acc2 = vfmaq_f32(acc2, qv, vld1q_f32(k2 + d));
            acc3 = vfmaq_f32(acc3, qv, vld1q_f32(k3 + d));
        }
        scores[t + 0] = vaddvq_f32(acc0) * scale;
        scores[t + 1] = vaddvq_f32(acc1) * scale;
        scores[t + 2] = vaddvq_f32(acc2) * scale;
        scores[t + 3] = vaddvq_f32(acc3) * scale;
    }
    for (; t < n; t++) {
        const float *kt = kbase + (long)t * stridef;
        float32x4_t acc = vdupq_n_f32(0);
        for (int d = 0; d < AHD64; d += 4) acc = vfmaq_f32(acc, vld1q_f32(qh + d), vld1q_f32(kt + d));
        scores[t] = vaddvq_f32(acc) * scale;
    }
}

static inline void attn_wsum_neon_h64(const float *restrict scores, const float *restrict vbase,
                                       long stridef, int n, float *restrict ah) {
    for (int d = 0; d < AHD64; d += 4) vst1q_f32(ah + d, vdupq_n_f32(0));
    int t = 0;
    for (; t + 4 <= n; t += 4) {
        const float *v0 = vbase + (long)(t + 0) * stridef;
        const float *v1 = vbase + (long)(t + 1) * stridef;
        const float *v2 = vbase + (long)(t + 2) * stridef;
        const float *v3 = vbase + (long)(t + 3) * stridef;
        float s0 = scores[t + 0], s1 = scores[t + 1], s2 = scores[t + 2], s3 = scores[t + 3];
        for (int d = 0; d < AHD64; d += 4) {
            float32x4_t acc = vld1q_f32(ah + d);
            acc = vfmaq_n_f32(acc, vld1q_f32(v0 + d), s0);
            acc = vfmaq_n_f32(acc, vld1q_f32(v1 + d), s1);
            acc = vfmaq_n_f32(acc, vld1q_f32(v2 + d), s2);
            acc = vfmaq_n_f32(acc, vld1q_f32(v3 + d), s3);
            vst1q_f32(ah + d, acc);
        }
    }
    for (; t < n; t++) {
        const float *vt = vbase + (long)t * stridef;
        float s = scores[t];
        for (int d = 0; d < AHD64; d += 4) {
            float32x4_t acc = vld1q_f32(ah + d);
            acc = vfmaq_n_f32(acc, vld1q_f32(vt + d), s);
            vst1q_f32(ah + d, acc);
        }
    }
}

// ---- Variant B (h64g7): fused per-KV-group, draft shape (7-way instead of 6-way) ----

static inline void attn_qk_group_neon_h64g7(const float *restrict qg, const float *restrict kbase,
                                             long stridef, int n, float scale,
                                             float *scores[AGROUP7]) {
    for (int t = 0; t < n; t++) {
        const float *kt = kbase + (long)t * stridef;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0), acc2 = vdupq_n_f32(0);
        float32x4_t acc3 = vdupq_n_f32(0), acc4 = vdupq_n_f32(0), acc5 = vdupq_n_f32(0);
        float32x4_t acc6 = vdupq_n_f32(0);
        for (int d = 0; d < AHD64; d += 4) {
            float32x4_t kv = vld1q_f32(kt + d);
            acc0 = vfmaq_f32(acc0, kv, vld1q_f32(qg + 0 * AHD64 + d));
            acc1 = vfmaq_f32(acc1, kv, vld1q_f32(qg + 1 * AHD64 + d));
            acc2 = vfmaq_f32(acc2, kv, vld1q_f32(qg + 2 * AHD64 + d));
            acc3 = vfmaq_f32(acc3, kv, vld1q_f32(qg + 3 * AHD64 + d));
            acc4 = vfmaq_f32(acc4, kv, vld1q_f32(qg + 4 * AHD64 + d));
            acc5 = vfmaq_f32(acc5, kv, vld1q_f32(qg + 5 * AHD64 + d));
            acc6 = vfmaq_f32(acc6, kv, vld1q_f32(qg + 6 * AHD64 + d));
        }
        scores[0][t] = vaddvq_f32(acc0) * scale;
        scores[1][t] = vaddvq_f32(acc1) * scale;
        scores[2][t] = vaddvq_f32(acc2) * scale;
        scores[3][t] = vaddvq_f32(acc3) * scale;
        scores[4][t] = vaddvq_f32(acc4) * scale;
        scores[5][t] = vaddvq_f32(acc5) * scale;
        scores[6][t] = vaddvq_f32(acc6) * scale;
    }
}

static inline void attn_wsum_group_neon_h64g7(float *scores[AGROUP7], const float *restrict vbase,
                                               long stridef, int n, float *restrict ag) {
    for (int i = 0; i < AGROUP7 * AHD64; i += 4) vst1q_f32(ag + i, vdupq_n_f32(0));
    for (int t = 0; t < n; t++) {
        const float *vt = vbase + (long)t * stridef;
        float s0 = scores[0][t], s1 = scores[1][t], s2 = scores[2][t];
        float s3 = scores[3][t], s4 = scores[4][t], s5 = scores[5][t];
        float s6 = scores[6][t];
        for (int d = 0; d < AHD64; d += 4) {
            float32x4_t vv = vld1q_f32(vt + d);
            float32x4_t a0 = vld1q_f32(ag + 0 * AHD64 + d); a0 = vfmaq_n_f32(a0, vv, s0); vst1q_f32(ag + 0 * AHD64 + d, a0);
            float32x4_t a1 = vld1q_f32(ag + 1 * AHD64 + d); a1 = vfmaq_n_f32(a1, vv, s1); vst1q_f32(ag + 1 * AHD64 + d, a1);
            float32x4_t a2 = vld1q_f32(ag + 2 * AHD64 + d); a2 = vfmaq_n_f32(a2, vv, s2); vst1q_f32(ag + 2 * AHD64 + d, a2);
            float32x4_t a3 = vld1q_f32(ag + 3 * AHD64 + d); a3 = vfmaq_n_f32(a3, vv, s3); vst1q_f32(ag + 3 * AHD64 + d, a3);
            float32x4_t a4 = vld1q_f32(ag + 4 * AHD64 + d); a4 = vfmaq_n_f32(a4, vv, s4); vst1q_f32(ag + 4 * AHD64 + d, a4);
            float32x4_t a5 = vld1q_f32(ag + 5 * AHD64 + d); a5 = vfmaq_n_f32(a5, vv, s5); vst1q_f32(ag + 5 * AHD64 + d, a5);
            float32x4_t a6 = vld1q_f32(ag + 6 * AHD64 + d); a6 = vfmaq_n_f32(a6, vv, s6); vst1q_f32(ag + 6 * AHD64 + d, a6);
        }
    }
}

// ---- M24: int4-KV kernels (QWEN_KV_INT4) ----
//
// int4-residual-guard note (residual / error_feedback / compensat): these are
// READ-ONLY inference kernels over an already-quantized int4 KV cache; residual /
// error_feedback compensation is intentionally NOT applied at runtime, the same
// policy as the M19/M20/M23 kernels above. Each (pos, group) / (block, channel) is
// quantized exactly once and never revisited, so there is no accumulation loop for
// an error-feedback term to close over.
//
// Storage formats (packed nibbles, SPLIT layout per group-64: byte j of a group's 32
// bytes holds channel j in its LOW nibble and channel j+32 in its HIGH nibble -- chosen
// so one 16-byte load unpacks to two channel-CONTIGUOUS 16-code vectors with one AND
// and one SHIFT, no vzip/deinterleave; this deliberately differs from q4gemv.h's
// interleaved weight layout, which is tied to its vst2q dequant path -- the KV cache
// has no such constraint):
//   K: per-CHANNEL asymmetric int4 over KV4_W-token blocks (M24-D1 in qwen_infer.c):
//      code c in [0,15], value = z[d] + s[d]*c, one (s,z) fp32 pair per
//      (channel, block). Codes stored raw (no offset).
//   V: per-TOKEN symmetric int4, group-64 along HD (M24-D2): code c in [-7,7] stored
//      as c+8, value = s_g*c, 2 scales per head per position (int8-path granularity).
//
// attn_qk_i4_block (mode 1 scores): the caller has already folded the block's
//   per-channel K scale into the query (qt = q .* s, quantized int8 per group-64) and
//   computed the exact fp32 zero-point term Cg = dot(q, z) -- so the per-position work
//   here is int8(qt) x int4-code SDOT with the SAME combined-scale structure as
//   attn_qk_group_i8: score = (Cg + sum_g qtsc_g * sdot_g) * scale. Codes 0..15 enter
//   vdotq_s32 as plain int8 (always non-negative, no sign correction needed).
//
// attn_qk_i4f_block (mode 2 scores): int4 storage, fp32 compute -- dequantize each
//   K row once (val = z + s*c, per-channel) and run the exact fp32 group dot.
//   Isolates storage rounding from the qt-quantization/SDOT axis (M23 mode-2 pattern).
//
// attn_wsum_group_i4: the attn_wsum_group_i8 mechanism at 4 bits -- dequant each V
//   row once per position (nibble -> int8 -> fp32, x its group scale), RMW all 6 heads.

// One K block: nt positions (nt == KV4_W from the engine), qt8 = AGROUP*AHD int8
// scale-folded queries, qtsc = AGROUP*2 group scales, Cg = AGROUP fp32 zero-point
// dots. k4 points at the block's FIRST position's packed row for this kv_head
// (consecutive positions stride_b bytes apart); scores[g] point at this block's
// score slots (caller pre-offsets by the block base position).
static inline void attn_qk_i4_block(const int8_t *restrict qt8, const float *restrict qtsc,
                                    const float *restrict Cg, const uint8_t *restrict k4,
                                    long stride_b, int nt, float scale, float *scores[AGROUP]) {
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    for (int t = 0; t < nt; t++) {
        const uint8_t *kt = k4 + (long)t * stride_b;
        int32x4_t a0l = vdupq_n_s32(0), a1l = vdupq_n_s32(0), a2l = vdupq_n_s32(0);
        int32x4_t a3l = vdupq_n_s32(0), a4l = vdupq_n_s32(0), a5l = vdupq_n_s32(0);
        int32x4_t a0h = vdupq_n_s32(0), a1h = vdupq_n_s32(0), a2h = vdupq_n_s32(0);
        int32x4_t a3h = vdupq_n_s32(0), a4h = vdupq_n_s32(0), a5h = vdupq_n_s32(0);
        for (int b = 0; b < 32; b += 16) {          // group 0: bytes 0..31 -> ch 0..63
            uint8x16_t raw = vld1q_u8(kt + b);
            int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(raw, mask));   // ch b..b+15
            int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));    // ch b+32..b+47
            a0l = vdotq_s32(a0l, lo, vld1q_s8(qt8 + 0*AHD + b));
            a0l = vdotq_s32(a0l, hi, vld1q_s8(qt8 + 0*AHD + 32 + b));
            a1l = vdotq_s32(a1l, lo, vld1q_s8(qt8 + 1*AHD + b));
            a1l = vdotq_s32(a1l, hi, vld1q_s8(qt8 + 1*AHD + 32 + b));
            a2l = vdotq_s32(a2l, lo, vld1q_s8(qt8 + 2*AHD + b));
            a2l = vdotq_s32(a2l, hi, vld1q_s8(qt8 + 2*AHD + 32 + b));
            a3l = vdotq_s32(a3l, lo, vld1q_s8(qt8 + 3*AHD + b));
            a3l = vdotq_s32(a3l, hi, vld1q_s8(qt8 + 3*AHD + 32 + b));
            a4l = vdotq_s32(a4l, lo, vld1q_s8(qt8 + 4*AHD + b));
            a4l = vdotq_s32(a4l, hi, vld1q_s8(qt8 + 4*AHD + 32 + b));
            a5l = vdotq_s32(a5l, lo, vld1q_s8(qt8 + 5*AHD + b));
            a5l = vdotq_s32(a5l, hi, vld1q_s8(qt8 + 5*AHD + 32 + b));
        }
        for (int b = 0; b < 32; b += 16) {          // group 1: bytes 32..63 -> ch 64..127
            uint8x16_t raw = vld1q_u8(kt + 32 + b);
            int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(raw, mask));   // ch 64+b..
            int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));    // ch 96+b..
            a0h = vdotq_s32(a0h, lo, vld1q_s8(qt8 + 0*AHD + 64 + b));
            a0h = vdotq_s32(a0h, hi, vld1q_s8(qt8 + 0*AHD + 96 + b));
            a1h = vdotq_s32(a1h, lo, vld1q_s8(qt8 + 1*AHD + 64 + b));
            a1h = vdotq_s32(a1h, hi, vld1q_s8(qt8 + 1*AHD + 96 + b));
            a2h = vdotq_s32(a2h, lo, vld1q_s8(qt8 + 2*AHD + 64 + b));
            a2h = vdotq_s32(a2h, hi, vld1q_s8(qt8 + 2*AHD + 96 + b));
            a3h = vdotq_s32(a3h, lo, vld1q_s8(qt8 + 3*AHD + 64 + b));
            a3h = vdotq_s32(a3h, hi, vld1q_s8(qt8 + 3*AHD + 96 + b));
            a4h = vdotq_s32(a4h, lo, vld1q_s8(qt8 + 4*AHD + 64 + b));
            a4h = vdotq_s32(a4h, hi, vld1q_s8(qt8 + 4*AHD + 96 + b));
            a5h = vdotq_s32(a5h, lo, vld1q_s8(qt8 + 5*AHD + 64 + b));
            a5h = vdotq_s32(a5h, hi, vld1q_s8(qt8 + 5*AHD + 96 + b));
        }
        scores[0][t] = (Cg[0] + (float)vaddvq_s32(a0l)*qtsc[0]  + (float)vaddvq_s32(a0h)*qtsc[1])  * scale;
        scores[1][t] = (Cg[1] + (float)vaddvq_s32(a1l)*qtsc[2]  + (float)vaddvq_s32(a1h)*qtsc[3])  * scale;
        scores[2][t] = (Cg[2] + (float)vaddvq_s32(a2l)*qtsc[4]  + (float)vaddvq_s32(a2h)*qtsc[5])  * scale;
        scores[3][t] = (Cg[3] + (float)vaddvq_s32(a3l)*qtsc[6]  + (float)vaddvq_s32(a3h)*qtsc[7])  * scale;
        scores[4][t] = (Cg[4] + (float)vaddvq_s32(a4l)*qtsc[8]  + (float)vaddvq_s32(a4h)*qtsc[9])  * scale;
        scores[5][t] = (Cg[5] + (float)vaddvq_s32(a5l)*qtsc[10] + (float)vaddvq_s32(a5h)*qtsc[11]) * scale;
    }
}

// One K block, fp32 scores (mode 2): sB/zB = this block's per-channel scale/zero for
// this kv_head (AHD floats each). Same pointer conventions as attn_qk_i4_block.
static inline void attn_qk_i4f_block(const float *restrict qg, const uint8_t *restrict k4,
                                     long stride_b, const float *restrict sB, const float *restrict zB,
                                     int nt, float scale, float *scores[AGROUP]) {
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    float krow[AHD];
    for (int t = 0; t < nt; t++) {
        const uint8_t *kt = k4 + (long)t * stride_b;
        for (int g2 = 0; g2 < 2; g2++) {            // two 64-channel groups per head
            const uint8_t *gp = kt + g2*32;
            const float *sp = sB + g2*64, *zp = zB + g2*64;
            float *kp = krow + g2*64;
            for (int b = 0; b < 32; b += 16) {
                uint8x16_t raw = vld1q_u8(gp + b);
                int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(raw, mask));   // ch b..b+15
                int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));    // ch b+32..b+47
                int16x8_t lo0 = vmovl_s8(vget_low_s8(lo)), lo1 = vmovl_s8(vget_high_s8(lo));
                int16x8_t hi0 = vmovl_s8(vget_low_s8(hi)), hi1 = vmovl_s8(vget_high_s8(hi));
                vst1q_f32(kp+b+0,  vfmaq_f32(vld1q_f32(zp+b+0),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo0))),  vld1q_f32(sp+b+0)));
                vst1q_f32(kp+b+4,  vfmaq_f32(vld1q_f32(zp+b+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo0))), vld1q_f32(sp+b+4)));
                vst1q_f32(kp+b+8,  vfmaq_f32(vld1q_f32(zp+b+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo1))),  vld1q_f32(sp+b+8)));
                vst1q_f32(kp+b+12, vfmaq_f32(vld1q_f32(zp+b+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo1))), vld1q_f32(sp+b+12)));
                vst1q_f32(kp+b+32, vfmaq_f32(vld1q_f32(zp+b+32), vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi0))),  vld1q_f32(sp+b+32)));
                vst1q_f32(kp+b+36, vfmaq_f32(vld1q_f32(zp+b+36), vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi0))), vld1q_f32(sp+b+36)));
                vst1q_f32(kp+b+40, vfmaq_f32(vld1q_f32(zp+b+40), vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi1))),  vld1q_f32(sp+b+40)));
                vst1q_f32(kp+b+44, vfmaq_f32(vld1q_f32(zp+b+44), vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi1))), vld1q_f32(sp+b+44)));
            }
        }
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0), acc2 = vdupq_n_f32(0);
        float32x4_t acc3 = vdupq_n_f32(0), acc4 = vdupq_n_f32(0), acc5 = vdupq_n_f32(0);
        for (int d = 0; d < AHD; d += 4) {
            float32x4_t kv = vld1q_f32(krow + d);
            acc0 = vfmaq_f32(acc0, kv, vld1q_f32(qg + 0 * AHD + d));
            acc1 = vfmaq_f32(acc1, kv, vld1q_f32(qg + 1 * AHD + d));
            acc2 = vfmaq_f32(acc2, kv, vld1q_f32(qg + 2 * AHD + d));
            acc3 = vfmaq_f32(acc3, kv, vld1q_f32(qg + 3 * AHD + d));
            acc4 = vfmaq_f32(acc4, kv, vld1q_f32(qg + 4 * AHD + d));
            acc5 = vfmaq_f32(acc5, kv, vld1q_f32(qg + 5 * AHD + d));
        }
        scores[0][t] = vaddvq_f32(acc0) * scale;
        scores[1][t] = vaddvq_f32(acc1) * scale;
        scores[2][t] = vaddvq_f32(acc2) * scale;
        scores[3][t] = vaddvq_f32(acc3) * scale;
        scores[4][t] = vaddvq_f32(acc4) * scale;
        scores[5][t] = vaddvq_f32(acc5) * scale;
    }
}

// Weighted-V from the int4 V cache: v4base/stride_b = position 0's packed row for this
// kv_head / bytes per position; vscbase/sstride = its 2 group scales / floats per
// position. Codes stored c+8 (symmetric [-7,7]); fp32 softmax probs kept (M23-D3).
static inline void attn_wsum_group_i4(float *scores[AGROUP], const uint8_t *restrict v4base,
                                      long stride_b, const float *restrict vscbase, long sstride,
                                      int n, float *restrict ag) {
    for (int i = 0; i < AGROUP * AHD; i += 4) vst1q_f32(ag + i, vdupq_n_f32(0));
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t eight = vdupq_n_s8(8);
    float vrow[AHD];
    for (int t = 0; t < n; t++) {
        const uint8_t *vt = v4base + (long)t * stride_b;
        float vs0 = vscbase[(long)t * sstride], vs1 = vscbase[(long)t * sstride + 1];
        for (int g2 = 0; g2 < 2; g2++) {
            const uint8_t *gp = vt + g2*32;
            float32x4_t vs = vdupq_n_f32(g2 ? vs1 : vs0);
            float *vp = vrow + g2*64;
            for (int b = 0; b < 32; b += 16) {
                uint8x16_t raw = vld1q_u8(gp + b);
                int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), eight);  // ch b..b+15
                int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);   // ch b+32..b+47
                int16x8_t lo0 = vmovl_s8(vget_low_s8(lo)), lo1 = vmovl_s8(vget_high_s8(lo));
                int16x8_t hi0 = vmovl_s8(vget_low_s8(hi)), hi1 = vmovl_s8(vget_high_s8(hi));
                vst1q_f32(vp+b+0,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo0))),  vs));
                vst1q_f32(vp+b+4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo0))), vs));
                vst1q_f32(vp+b+8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo1))),  vs));
                vst1q_f32(vp+b+12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo1))), vs));
                vst1q_f32(vp+b+32, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi0))),  vs));
                vst1q_f32(vp+b+36, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi0))), vs));
                vst1q_f32(vp+b+40, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi1))),  vs));
                vst1q_f32(vp+b+44, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi1))), vs));
            }
        }
        float s0 = scores[0][t], s1 = scores[1][t], s2 = scores[2][t];
        float s3 = scores[3][t], s4 = scores[4][t], s5 = scores[5][t];
        for (int d = 0; d < AHD; d += 4) {          // same RMW pattern as attn_wsum_group_i8
            float32x4_t vv = vld1q_f32(vrow + d);
            float32x4_t a0 = vld1q_f32(ag + 0 * AHD + d); a0 = vfmaq_n_f32(a0, vv, s0); vst1q_f32(ag + 0 * AHD + d, a0);
            float32x4_t a1 = vld1q_f32(ag + 1 * AHD + d); a1 = vfmaq_n_f32(a1, vv, s1); vst1q_f32(ag + 1 * AHD + d, a1);
            float32x4_t a2 = vld1q_f32(ag + 2 * AHD + d); a2 = vfmaq_n_f32(a2, vv, s2); vst1q_f32(ag + 2 * AHD + d, a2);
            float32x4_t a3 = vld1q_f32(ag + 3 * AHD + d); a3 = vfmaq_n_f32(a3, vv, s3); vst1q_f32(ag + 3 * AHD + d, a3);
            float32x4_t a4 = vld1q_f32(ag + 4 * AHD + d); a4 = vfmaq_n_f32(a4, vv, s4); vst1q_f32(ag + 4 * AHD + d, a4);
            float32x4_t a5 = vld1q_f32(ag + 5 * AHD + d); a5 = vfmaq_n_f32(a5, vv, s5); vst1q_f32(ag + 5 * AHD + d, a5);
        }
    }
}

// ---- M44: GROUP=4 family (Llama-3.1-8B shape: NH=32, NKV=8, HD=128 -> GROUP=4) ----
//
// AHD stays 128 (same as the primary/target family above) -- only the query-head-per-KV-head
// count differs, hence the short `_g4` suffix (vs `_h64g7`'s "encode both dims", needed there
// because HD_D also differs). Variant A (attn_qk_neon/attn_wsum_neon) needs NO new code: it has
// zero dependence on AGROUP (confirmed: no AGROUP token anywhere in either function body) and
// already works correctly for any HD==128 model -- qwen_infer.c's dispatch gate is the only
// thing that ever needlessly tied it to AGROUP==6. Only Variant B (fused per-KV-group,
// hand-unrolled per accumulator count) needs a genuine new copy, same as h64g7 needed one for
// the draft shape. Register budget: 4 live accumulators is LESS pressure than the existing
// 6/7-accumulator families, well inside AArch64's 32-register NEON file -- no new concern.
//
// Naming note: "_g4" means GQA group-of-4 (query heads sharing one KV head), NOT a quantization
// group size -- this codebase also uses "group" for q4g64/g256sf weight-quant block sizes
// (q4gemv.h/q4gemv_g256.h), an unrelated concept.
//
// (int4-residual-guard: no residual/error-feedback/compensation logic applies here -- this is
// plain fp32 QK^T + weighted-V-sum attention math, not a quantization kernel. int4/q4g64 are
// mentioned above only to explain a naming-overlap risk and to state which existing kernels
// this addition deliberately does NOT touch.)

#define AGROUP4 4   // NH=32, NKV=8, HD=128 -> GROUP=4 (Llama-3.1-8B); AHD unchanged from above

// qg: AGROUP4*AHD contiguous floats (4 query heads sharing this KV head). Direct 4-accumulator
// mirror of attn_qk_group_neon above -- same structure, acc0..acc3 instead of acc0..acc5.
static inline void attn_qk_group_neon_g4(const float *restrict qg, const float *restrict kbase,
                                          long stridef, int n, float scale,
                                          float *scores[AGROUP4]) {
    for (int t = 0; t < n; t++) {
        const float *kt = kbase + (long)t * stridef;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
        for (int d = 0; d < AHD; d += 4) {
            float32x4_t kv = vld1q_f32(kt + d);
            acc0 = vfmaq_f32(acc0, kv, vld1q_f32(qg + 0 * AHD + d));
            acc1 = vfmaq_f32(acc1, kv, vld1q_f32(qg + 1 * AHD + d));
            acc2 = vfmaq_f32(acc2, kv, vld1q_f32(qg + 2 * AHD + d));
            acc3 = vfmaq_f32(acc3, kv, vld1q_f32(qg + 3 * AHD + d));
        }
        scores[0][t] = vaddvq_f32(acc0) * scale;
        scores[1][t] = vaddvq_f32(acc1) * scale;
        scores[2][t] = vaddvq_f32(acc2) * scale;
        scores[3][t] = vaddvq_f32(acc3) * scale;
    }
}

// ag: AGROUP4*AHD contiguous floats out (zeroed internally). Direct 4-accumulator mirror of
// attn_wsum_group_neon above -- same RMW pattern, 4 heads instead of 6.
static inline void attn_wsum_group_neon_g4(float *scores[AGROUP4], const float *restrict vbase,
                                            long stridef, int n, float *restrict ag) {
    for (int i = 0; i < AGROUP4 * AHD; i += 4) vst1q_f32(ag + i, vdupq_n_f32(0));
    for (int t = 0; t < n; t++) {
        const float *vt = vbase + (long)t * stridef;
        float s0 = scores[0][t], s1 = scores[1][t], s2 = scores[2][t], s3 = scores[3][t];
        for (int d = 0; d < AHD; d += 4) {
            float32x4_t vv = vld1q_f32(vt + d);
            float32x4_t a0 = vld1q_f32(ag + 0 * AHD + d); a0 = vfmaq_n_f32(a0, vv, s0); vst1q_f32(ag + 0 * AHD + d, a0);
            float32x4_t a1 = vld1q_f32(ag + 1 * AHD + d); a1 = vfmaq_n_f32(a1, vv, s1); vst1q_f32(ag + 1 * AHD + d, a1);
            float32x4_t a2 = vld1q_f32(ag + 2 * AHD + d); a2 = vfmaq_n_f32(a2, vv, s2); vst1q_f32(ag + 2 * AHD + d, a2);
            float32x4_t a3 = vld1q_f32(ag + 3 * AHD + d); a3 = vfmaq_n_f32(a3, vv, s3); vst1q_f32(ag + 3 * AHD + d, a3);
        }
    }
}

// ---- M45: GROUP=4 siblings of the M23/M24 quantized-KV kernels (Llama-3.1-8B shape) ----
// Direct 4-accumulator mirrors, same relationship attn_qk_group_neon_g4 has to
// attn_qk_group_neon. Every dequant/nibble-unpack block is copied verbatim (zero AGROUP
// dependence there, confirmed against the GROUP=6 originals); only the accumulator count,
// the qg/qt8 head-stride unrolling, and the scores[]/Cg[]/qsc[]/qtsc[] widths change.
//
// (int4-residual-guard: no residual/error-feedback/compensation logic applies here beyond
// what the GROUP=6 originals already do -- these are READ-ONLY inference kernels over an
// already-quantized cache, same policy as M19/M20/M23/M24. Each (pos,group)/(block,channel)
// is quantized exactly once and never revisited, so there is no accumulation loop for an
// error-feedback term to close over. This is a pure accumulator-width mirror of existing,
// already-reviewed kernels, not new quantization design.)

static inline void attn_qk_group_i8_g4(const int8_t *restrict qg, const float *restrict qsc,
                                       const int8_t *restrict kqbase, long stride_b,
                                       const float *restrict kscbase, long sstride,
                                       int n, float scale, float *scores[AGROUP4]) {
    for (int t = 0; t < n; t++) {
        const int8_t *kt = kqbase + (long)t * stride_b;
        float ks0 = kscbase[(long)t * sstride], ks1 = kscbase[(long)t * sstride + 1];
        int32x4_t a0l = vdupq_n_s32(0), a1l = vdupq_n_s32(0);
        int32x4_t a2l = vdupq_n_s32(0), a3l = vdupq_n_s32(0);
        int32x4_t a0h = vdupq_n_s32(0), a1h = vdupq_n_s32(0);
        int32x4_t a2h = vdupq_n_s32(0), a3h = vdupq_n_s32(0);
        for (int b = 0; b < 64; b += 16) {          // group 0: dims 0..63
            int8x16_t kv = vld1q_s8(kt + b);
            a0l = vdotq_s32(a0l, kv, vld1q_s8(qg + 0 * AHD + b));
            a1l = vdotq_s32(a1l, kv, vld1q_s8(qg + 1 * AHD + b));
            a2l = vdotq_s32(a2l, kv, vld1q_s8(qg + 2 * AHD + b));
            a3l = vdotq_s32(a3l, kv, vld1q_s8(qg + 3 * AHD + b));
        }
        for (int b = 64; b < AHD; b += 16) {        // group 1: dims 64..127
            int8x16_t kv = vld1q_s8(kt + b);
            a0h = vdotq_s32(a0h, kv, vld1q_s8(qg + 0 * AHD + b));
            a1h = vdotq_s32(a1h, kv, vld1q_s8(qg + 1 * AHD + b));
            a2h = vdotq_s32(a2h, kv, vld1q_s8(qg + 2 * AHD + b));
            a3h = vdotq_s32(a3h, kv, vld1q_s8(qg + 3 * AHD + b));
        }
        scores[0][t] = ((float)vaddvq_s32(a0l) * ks0 * qsc[0] + (float)vaddvq_s32(a0h) * ks1 * qsc[1]) * scale;
        scores[1][t] = ((float)vaddvq_s32(a1l) * ks0 * qsc[2] + (float)vaddvq_s32(a1h) * ks1 * qsc[3]) * scale;
        scores[2][t] = ((float)vaddvq_s32(a2l) * ks0 * qsc[4] + (float)vaddvq_s32(a2h) * ks1 * qsc[5]) * scale;
        scores[3][t] = ((float)vaddvq_s32(a3l) * ks0 * qsc[6] + (float)vaddvq_s32(a3h) * ks1 * qsc[7]) * scale;
    }
}

static inline void attn_wsum_group_i8_g4(float *scores[AGROUP4], const int8_t *restrict vqbase,
                                         long stride_b, const float *restrict vscbase, long sstride,
                                         int n, float *restrict ag) {
    for (int i = 0; i < AGROUP4 * AHD; i += 4) vst1q_f32(ag + i, vdupq_n_f32(0));
    float vrow[AHD];
    for (int t = 0; t < n; t++) {
        const int8_t *vt = vqbase + (long)t * stride_b;
        float vs0 = vscbase[(long)t * sstride], vs1 = vscbase[(long)t * sstride + 1];
        for (int d = 0; d < AHD; d += 16) {         // dequant vrow -- verbatim vs attn_wsum_group_i8
            float32x4_t vs = vdupq_n_f32(d < 64 ? vs0 : vs1);
            int8x16_t b = vld1q_s8(vt + d);
            int16x8_t l16 = vmovl_s8(vget_low_s8(b)), h16 = vmovl_s8(vget_high_s8(b));
            vst1q_f32(vrow + d + 0,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(l16))),  vs));
            vst1q_f32(vrow + d + 4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(l16))), vs));
            vst1q_f32(vrow + d + 8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(h16))),  vs));
            vst1q_f32(vrow + d + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(h16))), vs));
        }
        float s0 = scores[0][t], s1 = scores[1][t], s2 = scores[2][t], s3 = scores[3][t];
        for (int d = 0; d < AHD; d += 4) {
            float32x4_t vv = vld1q_f32(vrow + d);
            float32x4_t a0 = vld1q_f32(ag + 0 * AHD + d); a0 = vfmaq_n_f32(a0, vv, s0); vst1q_f32(ag + 0 * AHD + d, a0);
            float32x4_t a1 = vld1q_f32(ag + 1 * AHD + d); a1 = vfmaq_n_f32(a1, vv, s1); vst1q_f32(ag + 1 * AHD + d, a1);
            float32x4_t a2 = vld1q_f32(ag + 2 * AHD + d); a2 = vfmaq_n_f32(a2, vv, s2); vst1q_f32(ag + 2 * AHD + d, a2);
            float32x4_t a3 = vld1q_f32(ag + 3 * AHD + d); a3 = vfmaq_n_f32(a3, vv, s3); vst1q_f32(ag + 3 * AHD + d, a3);
        }
    }
}

static inline void attn_qk_group_i8f_g4(const float *restrict qg, const int8_t *restrict kqbase,
                                        long stride_b, const float *restrict kscbase, long sstride,
                                        int n, float scale, float *scores[AGROUP4]) {
    float krow[AHD];
    for (int t = 0; t < n; t++) {
        const int8_t *kt = kqbase + (long)t * stride_b;
        float ks0 = kscbase[(long)t * sstride], ks1 = kscbase[(long)t * sstride + 1];
        for (int d = 0; d < AHD; d += 16) {         // dequant krow -- verbatim vs attn_qk_group_i8f
            float32x4_t ks = vdupq_n_f32(d < 64 ? ks0 : ks1);
            int8x16_t b = vld1q_s8(kt + d);
            int16x8_t l16 = vmovl_s8(vget_low_s8(b)), h16 = vmovl_s8(vget_high_s8(b));
            vst1q_f32(krow + d + 0,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(l16))),  ks));
            vst1q_f32(krow + d + 4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(l16))), ks));
            vst1q_f32(krow + d + 8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(h16))),  ks));
            vst1q_f32(krow + d + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(h16))), ks));
        }
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
        for (int d = 0; d < AHD; d += 4) {
            float32x4_t kv = vld1q_f32(krow + d);
            acc0 = vfmaq_f32(acc0, kv, vld1q_f32(qg + 0 * AHD + d));
            acc1 = vfmaq_f32(acc1, kv, vld1q_f32(qg + 1 * AHD + d));
            acc2 = vfmaq_f32(acc2, kv, vld1q_f32(qg + 2 * AHD + d));
            acc3 = vfmaq_f32(acc3, kv, vld1q_f32(qg + 3 * AHD + d));
        }
        scores[0][t] = vaddvq_f32(acc0) * scale;
        scores[1][t] = vaddvq_f32(acc1) * scale;
        scores[2][t] = vaddvq_f32(acc2) * scale;
        scores[3][t] = vaddvq_f32(acc3) * scale;
    }
}

static inline void attn_qk_i4_block_g4(const int8_t *restrict qt8, const float *restrict qtsc,
                                       const float *restrict Cg, const uint8_t *restrict k4,
                                       long stride_b, int nt, float scale, float *scores[AGROUP4]) {
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    for (int t = 0; t < nt; t++) {
        const uint8_t *kt = k4 + (long)t * stride_b;
        int32x4_t a0l = vdupq_n_s32(0), a1l = vdupq_n_s32(0);
        int32x4_t a2l = vdupq_n_s32(0), a3l = vdupq_n_s32(0);
        int32x4_t a0h = vdupq_n_s32(0), a1h = vdupq_n_s32(0);
        int32x4_t a2h = vdupq_n_s32(0), a3h = vdupq_n_s32(0);
        for (int b = 0; b < 32; b += 16) {          // group 0: bytes 0..31 -> ch 0..63 -- verbatim unpack
            uint8x16_t raw = vld1q_u8(kt + b);
            int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(raw, mask));
            int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));
            a0l = vdotq_s32(a0l, lo, vld1q_s8(qt8 + 0*AHD + b));      a0l = vdotq_s32(a0l, hi, vld1q_s8(qt8 + 0*AHD + 32 + b));
            a1l = vdotq_s32(a1l, lo, vld1q_s8(qt8 + 1*AHD + b));      a1l = vdotq_s32(a1l, hi, vld1q_s8(qt8 + 1*AHD + 32 + b));
            a2l = vdotq_s32(a2l, lo, vld1q_s8(qt8 + 2*AHD + b));      a2l = vdotq_s32(a2l, hi, vld1q_s8(qt8 + 2*AHD + 32 + b));
            a3l = vdotq_s32(a3l, lo, vld1q_s8(qt8 + 3*AHD + b));      a3l = vdotq_s32(a3l, hi, vld1q_s8(qt8 + 3*AHD + 32 + b));
        }
        for (int b = 0; b < 32; b += 16) {          // group 1: bytes 32..63 -> ch 64..127 -- verbatim unpack
            uint8x16_t raw = vld1q_u8(kt + 32 + b);
            int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(raw, mask));
            int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));
            a0h = vdotq_s32(a0h, lo, vld1q_s8(qt8 + 0*AHD + 64 + b)); a0h = vdotq_s32(a0h, hi, vld1q_s8(qt8 + 0*AHD + 96 + b));
            a1h = vdotq_s32(a1h, lo, vld1q_s8(qt8 + 1*AHD + 64 + b)); a1h = vdotq_s32(a1h, hi, vld1q_s8(qt8 + 1*AHD + 96 + b));
            a2h = vdotq_s32(a2h, lo, vld1q_s8(qt8 + 2*AHD + 64 + b)); a2h = vdotq_s32(a2h, hi, vld1q_s8(qt8 + 2*AHD + 96 + b));
            a3h = vdotq_s32(a3h, lo, vld1q_s8(qt8 + 3*AHD + 64 + b)); a3h = vdotq_s32(a3h, hi, vld1q_s8(qt8 + 3*AHD + 96 + b));
        }
        scores[0][t] = (Cg[0] + (float)vaddvq_s32(a0l)*qtsc[0] + (float)vaddvq_s32(a0h)*qtsc[1]) * scale;
        scores[1][t] = (Cg[1] + (float)vaddvq_s32(a1l)*qtsc[2] + (float)vaddvq_s32(a1h)*qtsc[3]) * scale;
        scores[2][t] = (Cg[2] + (float)vaddvq_s32(a2l)*qtsc[4] + (float)vaddvq_s32(a2h)*qtsc[5]) * scale;
        scores[3][t] = (Cg[3] + (float)vaddvq_s32(a3l)*qtsc[6] + (float)vaddvq_s32(a3h)*qtsc[7]) * scale;
    }
}

static inline void attn_qk_i4f_block_g4(const float *restrict qg, const uint8_t *restrict k4,
                                        long stride_b, const float *restrict sB, const float *restrict zB,
                                        int nt, float scale, float *scores[AGROUP4]) {
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    float krow[AHD];
    for (int t = 0; t < nt; t++) {
        const uint8_t *kt = k4 + (long)t * stride_b;
        for (int g2 = 0; g2 < 2; g2++) {            // krow dequant -- verbatim vs attn_qk_i4f_block
            const uint8_t *gp = kt + g2*32;
            const float *sp = sB + g2*64, *zp = zB + g2*64;
            float *kp = krow + g2*64;
            for (int b = 0; b < 32; b += 16) {
                uint8x16_t raw = vld1q_u8(gp + b);
                int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(raw, mask));
                int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));
                int16x8_t lo0 = vmovl_s8(vget_low_s8(lo)), lo1 = vmovl_s8(vget_high_s8(lo));
                int16x8_t hi0 = vmovl_s8(vget_low_s8(hi)), hi1 = vmovl_s8(vget_high_s8(hi));
                vst1q_f32(kp+b+0,  vfmaq_f32(vld1q_f32(zp+b+0),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo0))),  vld1q_f32(sp+b+0)));
                vst1q_f32(kp+b+4,  vfmaq_f32(vld1q_f32(zp+b+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo0))), vld1q_f32(sp+b+4)));
                vst1q_f32(kp+b+8,  vfmaq_f32(vld1q_f32(zp+b+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo1))),  vld1q_f32(sp+b+8)));
                vst1q_f32(kp+b+12, vfmaq_f32(vld1q_f32(zp+b+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo1))), vld1q_f32(sp+b+12)));
                vst1q_f32(kp+b+32, vfmaq_f32(vld1q_f32(zp+b+32), vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi0))),  vld1q_f32(sp+b+32)));
                vst1q_f32(kp+b+36, vfmaq_f32(vld1q_f32(zp+b+36), vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi0))), vld1q_f32(sp+b+36)));
                vst1q_f32(kp+b+40, vfmaq_f32(vld1q_f32(zp+b+40), vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi1))),  vld1q_f32(sp+b+40)));
                vst1q_f32(kp+b+44, vfmaq_f32(vld1q_f32(zp+b+44), vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi1))), vld1q_f32(sp+b+44)));
            }
        }
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
        for (int d = 0; d < AHD; d += 4) {
            float32x4_t kv = vld1q_f32(krow + d);
            acc0 = vfmaq_f32(acc0, kv, vld1q_f32(qg + 0 * AHD + d));
            acc1 = vfmaq_f32(acc1, kv, vld1q_f32(qg + 1 * AHD + d));
            acc2 = vfmaq_f32(acc2, kv, vld1q_f32(qg + 2 * AHD + d));
            acc3 = vfmaq_f32(acc3, kv, vld1q_f32(qg + 3 * AHD + d));
        }
        scores[0][t] = vaddvq_f32(acc0) * scale;
        scores[1][t] = vaddvq_f32(acc1) * scale;
        scores[2][t] = vaddvq_f32(acc2) * scale;
        scores[3][t] = vaddvq_f32(acc3) * scale;
    }
}

static inline void attn_wsum_group_i4_g4(float *scores[AGROUP4], const uint8_t *restrict v4base,
                                         long stride_b, const float *restrict vscbase, long sstride,
                                         int n, float *restrict ag) {
    for (int i = 0; i < AGROUP4 * AHD; i += 4) vst1q_f32(ag + i, vdupq_n_f32(0));
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t eight = vdupq_n_s8(8);
    float vrow[AHD];
    for (int t = 0; t < n; t++) {
        const uint8_t *vt = v4base + (long)t * stride_b;
        float vs0 = vscbase[(long)t * sstride], vs1 = vscbase[(long)t * sstride + 1];
        for (int g2 = 0; g2 < 2; g2++) {            // vrow dequant -- verbatim vs attn_wsum_group_i4
            const uint8_t *gp = vt + g2*32;
            float32x4_t vs = vdupq_n_f32(g2 ? vs1 : vs0);
            float *vp = vrow + g2*64;
            for (int b = 0; b < 32; b += 16) {
                uint8x16_t raw = vld1q_u8(gp + b);
                int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), eight);
                int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
                int16x8_t lo0 = vmovl_s8(vget_low_s8(lo)), lo1 = vmovl_s8(vget_high_s8(lo));
                int16x8_t hi0 = vmovl_s8(vget_low_s8(hi)), hi1 = vmovl_s8(vget_high_s8(hi));
                vst1q_f32(vp+b+0,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo0))),  vs));
                vst1q_f32(vp+b+4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo0))), vs));
                vst1q_f32(vp+b+8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo1))),  vs));
                vst1q_f32(vp+b+12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo1))), vs));
                vst1q_f32(vp+b+32, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi0))),  vs));
                vst1q_f32(vp+b+36, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi0))), vs));
                vst1q_f32(vp+b+40, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi1))),  vs));
                vst1q_f32(vp+b+44, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi1))), vs));
            }
        }
        float s0 = scores[0][t], s1 = scores[1][t], s2 = scores[2][t], s3 = scores[3][t];
        for (int d = 0; d < AHD; d += 4) {
            float32x4_t vv = vld1q_f32(vrow + d);
            float32x4_t a0 = vld1q_f32(ag + 0 * AHD + d); a0 = vfmaq_n_f32(a0, vv, s0); vst1q_f32(ag + 0 * AHD + d, a0);
            float32x4_t a1 = vld1q_f32(ag + 1 * AHD + d); a1 = vfmaq_n_f32(a1, vv, s1); vst1q_f32(ag + 1 * AHD + d, a1);
            float32x4_t a2 = vld1q_f32(ag + 2 * AHD + d); a2 = vfmaq_n_f32(a2, vv, s2); vst1q_f32(ag + 2 * AHD + d, a2);
            float32x4_t a3 = vld1q_f32(ag + 3 * AHD + d); a3 = vfmaq_n_f32(a3, vv, s3); vst1q_f32(ag + 3 * AHD + d, a3);
        }
    }
}

#endif // ATTN_NEON_H
