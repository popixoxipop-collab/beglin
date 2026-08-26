// sme2_kai.c -- see sme2_kai.h for the safety contract this file implements.
//
// Phase 2 of the KleidiAI SME2 runtime-detection integration: hardware
// detection + shape/size queries only. Nothing in qwen_infer.c calls any of
// this yet (Phase 4 wires the actual GEMM dispatch) -- this file's object is
// linked into engine/qwen_infer purely to prove that linking it in does not
// change the engine's behavior on non-SME2 hardware (see build_qwen_infer.sh).
//
// This file performs no numeric weight/activation quantization arithmetic
// itself, so there is no quantization error for a residual/error-feedback
// term to carry. Quantization happens inside the vendored kleidiai
// kai_rhs_pack_*/kai_lhs_quant_pack_* functions this file only queries sizes
// from (and, in a later phase, will call) -- this file is pure hardware
// detection and shape/size arithmetic.

#include "sme2_kai.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>

#include "kleidiai/kai/kai_common.h"
#include "kleidiai/kai_lhs_quant_pack_qsi8d32p_f32_neon.h"
#include "kleidiai/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa.h"
#include "kleidiai/kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon.h"
#include "kleidiai/kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa.h"
#include "kleidiai/kai_lhs_pack_f16pmrx2_f32_neon.h"
#include "kleidiai/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"

// vdsp's weight format always groups K in blocks of 64; this kernel's `bl`
// parameter is not a tunable here, it's fixed to match that group size.
#define SME2_KAI_BL 64

static int sysctl_bool(const char *name) {
    int v = 0;
    size_t sz = sizeof v;
    if (sysctlbyname(name, &v, &sz, NULL, 0) != 0) return 0;  // key absent (e.g. M1 Max) -> not available
    return v == 1;
}

int kai_sme2_available(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    // MOPA here is the activation-vs-weight -> int32-accumulate outer-product
    // form, hence both FEAT_SME2 (SME2 present at all) and SME_I8I32 (that
    // specific outer-product datatype support) are required.
    cached = (sysctl_bool("hw.optional.arm.FEAT_SME2") && sysctl_bool("hw.optional.arm.SME_I8I32")) ? 1 : 0;
    return cached;
}

int kai_sme2_min_m(void) {
    if (!kai_sme2_available()) return INT_MAX;
    return (int)kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
}

int kai_sme2_shape_ok(int out, int in) {
    if (!kai_sme2_available()) return 0;
    if (out <= 0 || in <= 0) return 0;
    return (in % SME2_KAI_BL) == 0;
}

size_t kai_sme2_rhs_packed_bytes(int out, int in) {
    if (!kai_sme2_shape_ok(out, in)) return SIZE_MAX;  // covers the !available() case too
    size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    size_t kr = kai_get_kr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon(
        (size_t)out, (size_t)in, nr, kr, SME2_KAI_BL);
}

int kai_sme2_repack_q4g64(int out, int in, const uint8_t *packed, const float *scales, void *dst, size_t dst_bytes) {
    if (!kai_sme2_shape_ok(out, in)) return -1;
    size_t need = kai_sme2_rhs_packed_bytes(out, in);
    if (need == SIZE_MAX || dst_bytes < need) return -1;

    const size_t ng = (size_t)in / SME2_KAI_BL;
    const size_t vdsp_row_bytes = (size_t)in / 2;
    const size_t vdsp_group_bytes = SME2_KAI_BL / 2;       // 32 bytes = 64 codes
    const size_t block_bytes = vdsp_group_bytes + 2;       // + 2-byte fp16 scale header
    const size_t rhs_stride = ng * block_bytes;

    // Fixture-format scratch buffer for kai_run_rhs_pack_nxk_*'s `rhs` input:
    // per row, per group, [fp16 scale][32 permuted nibble bytes] -- this exact
    // layout is what kai_test_correct2.c (this session's from-scratch, bit-
    // exact-verified reference harness) constructs and this kernel consumes
    // correctly; see VENDOR.md for why that's trusted over this pack
    // function's own header doc comment, which describes a K0/K4 pairing that
    // does not match either the working reference harness or the permutation
    // below.
    uint8_t *unpacked = malloc((size_t)out * rhs_stride);
    if (!unpacked) return -1;

    for (int r = 0; r < out; r++) {
        const uint8_t *row_packed = packed + (size_t)r * vdsp_row_bytes;
        for (size_t gi = 0; gi < ng; gi++) {
            float s = scales[(size_t)r * ng + gi];
            if (!isfinite(s) || s == 0.0f) { free(unpacked); return -1; }
            uint8_t *block = unpacked + (size_t)r * rhs_stride + gi * block_bytes;
            __fp16 sh = (__fp16)s;
            memcpy(block, &sh, 2);
            uint8_t *values = block + 2;
            const uint8_t *group_packed = row_packed + gi * vdsp_group_bytes;
            // Pure nibble permutation -- both formats already carry the same
            // +8 zero-point bias (vdsp: q4gemv.h top comment "value=(nibble-8)*
            // scale"; kleidiai: kai_test_correct2.c's (code+8)&0xF), so no
            // arithmetic on values, only which nibble goes where:
            //   vdsp byte b:        low nibble = col(2b),   high nibble = col(2b+1)
            //   kleidiai byte idx:  low nibble = col(idx),  high nibble = col(idx+32)
            // => kleidiai byte idx reads vdsp bytes (idx/2) and (idx/2+16), same
            // even/odd nibble selection for both (idx and idx+32 share parity
            // since 32 is even).
            // D-f16lhs-2: WHY -- 2026-08-25, live lldb backtrace on bob showed this
            // exact loop autovectorized into a raw SVE  instruction that
            // SIGILLs (EXC_BAD_INSTRUCTION) on M4: this CPU only implements SVE inside
            // streaming mode (part of SME2), and nothing in this call chain
            // (moe_matvec_af_group_smart -> here) ever issues smstart -- only the
            // hand-written kernel .S files do that. Same root cause and same fix as
            // kai_sme2_repack_q4g64_f16lhs() below (D-f16lhs-1): this loop's own
            // vectorization got newly triggered once the f16lhs sibling function was
            // added to this file (whole-TU codegen shift at -O2/-O3), even though this
            // function itself is unchanged. COST: negligible, one-time repack at
            // model-load. EXIT: safe to drop if a future compiler stops emitting SVE
            // for this loop shape without the pragma (verify via otool -tv again).
            #pragma clang loop vectorize(disable) interleave(disable)
            for (size_t idx = 0; idx < vdsp_group_bytes; idx++) {
                size_t lo_byte = idx / 2, hi_byte = idx / 2 + 16;
                int odd = idx & 1;
                uint8_t lo_nib = odd ? (group_packed[lo_byte] >> 4) : (group_packed[lo_byte] & 0x0F);
                uint8_t hi_nib = odd ? (group_packed[hi_byte] >> 4) : (group_packed[hi_byte] & 0x0F);
                values[idx] = (uint8_t)((hi_nib << 4) | lo_nib);
            }
        }
    }

    size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    size_t kr = kai_get_kr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    size_t sr = kai_get_sr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    struct kai_rhs_pack_qs4cxs1s0_param params = { .lhs_zero_point = 1, .rhs_zero_point = 8 };
    kai_run_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon(
        1, (size_t)out, (size_t)in, nr, kr, sr, SME2_KAI_BL, unpacked, NULL, dst, 0, &params);

    free(unpacked);
    return 0;
}

size_t kai_sme2_lhs_scratch_bytes(int max_m, int max_in) {
    if (!kai_sme2_available()) return 0;
    if (max_m <= 0 || max_in <= 0) return 0;
    size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    size_t kr = kai_get_kr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    size_t sr = kai_get_sr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    return kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32_neon(
        (size_t)max_m, (size_t)max_in, SME2_KAI_BL, mr, kr, sr);
}

void kai_sme2_gemm_f32(int M, int out, int in, const float *x, const void *rhs_packed,
                       const float *bias, float *y, void *lhs_scratch) {
    if (!kai_sme2_available()) return;   // last-resort safety net, see sme2_kai.h -- callers gate too
    size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    size_t kr = kai_get_kr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    size_t sr = kai_get_sr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();

    kai_run_lhs_quant_pack_qsi8d32p_f32_neon(
        (size_t)M, (size_t)in, SME2_KAI_BL, mr, kr, sr, 0, x, (size_t)in * sizeof(float), lhs_scratch);

    kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(
        (size_t)M, (size_t)out, (size_t)in, SME2_KAI_BL, lhs_scratch, rhs_packed, y,
        (size_t)out * sizeof(float), sizeof(float), -FLT_MAX, FLT_MAX);

    // The kernel itself has no bias parameter (verified against its .h and
    // kai_test_correct2.c's own usage) -- added here, same as every reference
    // harness in this project's KleidiAI work does it.
    //
    // D4 (2026-08-26, found via a real production SIGILL, not a code review):
    //   WHY: this plain scalar bias-add loop, in this SME2-arch-flagged TU, gets
    //     autovectorized into raw SVE instructions (rdvl + z-register ldr/fadd/str) --
    //     the SAME root cause already fixed twice elsewhere in this file (see the
    //     kai_sme2_repack_q4g64_f16lhs() pragma above and its own D-f16lhs-1 note), but
    //     missed here because this loop was added later, after those fixes landed. This
    //     code runs AFTER kai_run_matmul_clamp_f32_...() has already returned -- that
    //     kernel's own .S assembly enters+exits SME2 streaming mode internally, so by
    //     the time this loop runs, streaming mode is OFF again, and Apple Silicon has no
    //     plain FEAT_SVE to fall back to -- any SVE instruction here is unconditionally
    //     illegal. Confirmed root cause via lldb backtrace (real crash, bob/M4) landing
    //     exactly on `rdvl` inside this function, then confirmed by disassembly showing
    //     no smstart anywhere in this function, then confirmed with a minimal repro
    //     (single tensor, bias!=NULL, M=1) that reproduces the SIGILL in isolation.
    //   COST: negligible -- this loop is O(M*out) scalar adds, a tiny fraction of the
    //     GEMM's own cost; disabling vectorization here doesn't measurably change
    //     matmul_t/matmul_sdot's SME2 wall-clock.
    //   EXIT: safe to drop this pragma once/if this bias-add is rewritten to run INSIDE
    //     the kernel's own streaming-mode session instead of after it returns (would
    //     need a KleidiAI kernel variant with native bias support, or manual
    //     smstart/smstop around this loop) -- verify via otool -tV | grep -c 'rdvl|ldr z'
    //     on this function's .o before removing.
    if (bias) {
        // D4 correction: the pragma must sit on the INNER loop (the one actually large
        // enough to vectorize -- `out` is typically 256-151936, `M` is often 1) -- an
        // earlier version of this fix put it on the outer `m` loop by mistake, which
        // Clang's `#pragma clang loop` only ever applies to the loop it immediately
        // precedes, so it silently did nothing and the SVE codegen was unchanged.
        // Verified via otool -tV after moving it here: zero rdvl/z-register instructions
        // in this function's compiled object.
        for (int m = 0; m < M; m++) {
            float *ym = y + (size_t)m * out;
            #pragma clang loop vectorize(disable) interleave(disable)
            for (int r = 0; r < out; r++) ym[r] += bias[r];
        }
    }
}

// ============================================================================================
// f16p-LHS variant -- see sme2_kai.h for the full rationale/contract comment. No residual/
// error-feedback logic here either, same reasoning as kai_sme2_repack_q4g64() above: this is
// inference-time repacking of already-quantized frozen weights into a hardware tile layout via
// vendored KleidiAI kernels, not a learned/tunable quantization scheme.

int kai_sme2_f16lhs_available(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    // This kernel's outer-product datatype is fp16(activations) x int4(weights) -> fp32
    // accumulate, NOT the int8x4->int32 form kai_sme2_available() gates -- confirmed via
    // kleidiai_upstream/test/common/cpu_info.cpp's cpu_has_sme2()+cpu_has_fp16() gate on the
    // matmul_clamp_f32_f16p_qsi4c32p_test.cpp variant list (cpu_check<cpu_has_sme2,
    // cpu_has_fp16>). SME_I8I32 is irrelevant here.
    cached = (sysctl_bool("hw.optional.arm.FEAT_SME2") && sysctl_bool("hw.optional.arm.FEAT_FP16")) ? 1 : 0;
    return cached;
}

int kai_sme2_f16lhs_shape_ok(int out, int in) {
    if (!kai_sme2_f16lhs_available()) return 0;
    if (out <= 0 || in <= 0) return 0;
    return (in % SME2_KAI_BL) == 0;   // same vdsp K_Q4G64 group-64 invariant as the int8-LHS path
}

size_t kai_sme2_f16lhs_rhs_packed_bytes(int out, int in) {
    if (!kai_sme2_f16lhs_shape_ok(out, in)) return SIZE_MAX;
    size_t nr = kai_get_nr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa();
    size_t kr = kai_get_kr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa();
    return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(
        (size_t)out, (size_t)in, nr, kr, SME2_KAI_BL);
}

int kai_sme2_repack_q4g64_f16lhs(int out, int in, const uint8_t *packed, const float *scales, void *dst, size_t dst_bytes) {
    if (!kai_sme2_f16lhs_shape_ok(out, in)) return -1;
    size_t need = kai_sme2_f16lhs_rhs_packed_bytes(out, in);
    if (need == SIZE_MAX || dst_bytes < need) return -1;

    const size_t ng = (size_t)in / SME2_KAI_BL;
    const size_t vdsp_row_bytes = (size_t)in / 2;
    const size_t vdsp_group_bytes = SME2_KAI_BL / 2;
    const size_t block_bytes = vdsp_group_bytes + 2;
    const size_t rhs_stride = ng * block_bytes;

    // Identical unpacked-fixture construction/nibble permutation as kai_sme2_repack_q4g64()
    // above -- verified reusable across both RHS pack schemes (kleidiai_upstream/test/tests/
    // matmul_clamp_f32_qsi8d32p_qsi4c32p_test.cpp feeds the SAME pack_data_scales_interleave_
    // block<UInt4,Float16> reference construction into both the ps1s0 and ps4s0 RHS pack
    // functions; only kai_run_rhs_pack_nxk_*'s own internal OUTPUT tile layout differs, not
    // this function's unpacked INPUT).
    uint8_t *unpacked = malloc((size_t)out * rhs_stride);
    if (!unpacked) return -1;

    for (int r = 0; r < out; r++) {
        const uint8_t *row_packed = packed + (size_t)r * vdsp_row_bytes;
        for (size_t gi = 0; gi < ng; gi++) {
            float s = scales[(size_t)r * ng + gi];
            if (!isfinite(s) || s == 0.0f) { free(unpacked); return -1; }
            uint8_t *block = unpacked + (size_t)r * rhs_stride + gi * block_bytes;
            __fp16 sh = (__fp16)s;
            memcpy(block, &sh, 2);
            uint8_t *values = block + 2;
            const uint8_t *group_packed = row_packed + gi * vdsp_group_bytes;
            // D-f16lhs-1: WHY -- verified via isolated repro (2026-08-25) that Apple clang
            // -O2 -march=armv9.2-a+sme2 autovectorizes this exact nibble-permute loop into
            // code that SIGILLs at runtime, ONLY when `unpacked` later escapes to an opaque
            // external call (kai_run_rhs_pack_nxk_qsi4c32ps1s0scalef16_...) -- confirmed by
            // bisection: -O0 fixes it, and this vectorize(disable) pragma fixes it at full
            // -O2 too, isolated down to exactly this loop. The sibling loop in
            // kai_sme2_repack_q4g64() above (identical pattern, ps4s0 pack target) is NOT
            // touched -- it's been correctness-verified in production for weeks and there is
            // no evidence it hits the same miscompile in that context, so leave it as-is
            // rather than risk it. COST: negligible -- this loop runs once per tensor at
            // model-load-time repack, not per-token. EXIT: if a future compiler version fixes
            // the underlying autovectorizer bug, this pragma becomes a no-op, safe to leave.
            #pragma clang loop vectorize(disable) interleave(disable)
            for (size_t idx = 0; idx < vdsp_group_bytes; idx++) {
                size_t lo_byte = idx / 2, hi_byte = idx / 2 + 16;
                int odd = idx & 1;
                uint8_t lo_nib = odd ? (group_packed[lo_byte] >> 4) : (group_packed[lo_byte] & 0x0F);
                uint8_t hi_nib = odd ? (group_packed[hi_byte] >> 4) : (group_packed[hi_byte] & 0x0F);
                values[idx] = (uint8_t)((hi_nib << 4) | lo_nib);
            }
        }
    }

    size_t nr = kai_get_nr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa();
    size_t kr = kai_get_kr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa();
    size_t sr = kai_get_sr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa();
    struct kai_rhs_pack_qs4cxs1s0_param params = { .lhs_zero_point = 1, .rhs_zero_point = 8 };
    kai_run_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(
        1, (size_t)out, (size_t)in, nr, kr, sr, SME2_KAI_BL, unpacked, NULL, dst, 0, &params);

    free(unpacked);
    return 0;
}

size_t kai_sme2_f16lhs_lhs_scratch_bytes(int max_m, int max_in) {
    if (!kai_sme2_f16lhs_available()) return 0;
    if (max_m <= 0 || max_in <= 0) return 0;
    size_t mr = kai_get_mr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa();
    size_t kr = kai_get_kr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa();
    size_t sr = kai_get_sr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa();
    return kai_get_lhs_packed_size_lhs_pack_f16pmrx2_f32_neon(
        (size_t)max_m, (size_t)max_in, SME2_KAI_BL, mr, kr, sr);
}

void kai_sme2_gemm_f16lhs(int M, int out, int in, const float *x, const void *rhs_packed,
                          const float *bias, float *y, void *lhs_scratch) {
    if (!kai_sme2_f16lhs_available()) return;
    size_t mr = kai_get_mr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa();
    size_t kr = kai_get_kr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa();
    size_t sr = kai_get_sr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa();

    kai_run_lhs_pack_f16pmrx2_f32_neon(
        (size_t)M, (size_t)in, SME2_KAI_BL, mr, kr, sr, 0, x, (size_t)in * sizeof(float), lhs_scratch);

    kai_run_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa(
        (size_t)M, (size_t)out, (size_t)in, SME2_KAI_BL, lhs_scratch, rhs_packed, y,
        (size_t)out * sizeof(float), sizeof(float), -FLT_MAX, FLT_MAX);

    if (bias) {
        // D4 (same root cause + same fix as kai_sme2_gemm_f32's identical bias-add loop
        // above -- see that comment for the full story): pragma on the INNER loop only.
        for (int m = 0; m < M; m++) {
            float *ym = y + (size_t)m * out;
            #pragma clang loop vectorize(disable) interleave(disable)
            for (int r = 0; r < out; r++) ym[r] += bias[r];
        }
    }
}
