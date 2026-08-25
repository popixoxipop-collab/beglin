// d31_fp32_scale_prototype.c
// Phase 3 prototype for the D31 finding (kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa_asm.S):
// the kernel combines LHS activation scale (genuinely FP16, kai_cast_f16_f32()
// in kai_lhs_quant_pack_qsi8d32p_f32.c) and RHS weight scale (genuinely BF16,
// raw memcpy in kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.c, asserts scale_dt ==
// kai_dt_bf16) via `fmlalb`/`fmlalt`. Per ARM's own SVE ISA reference (see D31
// in vdsp_kleidiai_sme2_padding_mimicry.md memory), FMLALB/FMLALT are FP16-only
// widening multiply-add (FPMulAddH, "no BF16 variant is specified") -- so the
// RHS's genuine BF16 bit pattern is misinterpreted as FP16, producing a wrong
// combined dequant scale.
//
// This file does NOT touch or re-verify the SME2 asm itself (that's a separate,
// much higher-risk change -- D32's own conclusion was "issue first, agree on
// direction with maintainers before a fix PR"). It isolates the SCALAR
// scale-combination math on both sides -- the buggy one (bit-reinterpret,
// simulating what fmlalb's hardware actually does) and the proposed fix
// (explicit widen-to-FP32 of both operands, then FP32 fmul) -- to prove the
// fix's correctness in isolation, as supporting evidence for an ARM GitLab
// issue.
//
// NOTE ON THE EXAMPLE VALUES: the weight-scale magnitude (~0.015) and the
// FP16-interpreted activation-scale magnitude (~109) below are REPRESENTATIVE
// of the ranges observed during the D29/D30 LLDB trace (a genuine q4g64
// weight scale and a genuine per-block int8 activation scale), not literal
// re-captured register bytes from that session (ASLR + a fresh process launch
// make raw addresses/bytes non-reproducible across sessions; the point being
// demonstrated -- BF16-as-FP16 misread vs. correct FP32 widen -- does not
// depend on using the exact original bytes).
//
// Build: clang -O2 -march=armv9-a+sve2 -o d31_fp32_scale_prototype d31_fp32_scale_prototype.c

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Exact copy of ARM's own kai_cast_f32_bf16 (kai/kai_common.h) -- bf16 is
// bit-identical to the top 16 bits of an f32, so this upcast is lossless.
static float kai_cast_f32_bf16(uint16_t bf16) {
    const uint32_t i32 = ((uint32_t)bf16 << 16);
    float f32 = 0;
    memcpy(&f32, &i32, sizeof(i32));
    return f32;
}

// Exact inverse of ARM's own kai_cast_f16_f32 (which does f32->f16 via the
// native float16_t cast) -- f16->f32 via the same native hardware path.
static float f16_bits_to_f32(uint16_t f16_bits) {
    __fp16 tmp;
    memcpy(&tmp, &f16_bits, sizeof(tmp));
    return (float)tmp;
}

// What the CURRENT kernel effectively does: fmlalb/fmlalt widen BOTH source
// half-lanes as FP16 (FPMulAddH, D31) regardless of what the bits actually
// encode. For the RHS operand this misreads a genuine BF16 bit pattern as if
// it were FP16.
static float buggy_rhs_scale_as_read_by_fmlalb(uint16_t rhs_bf16_bits) {
    return f16_bits_to_f32(rhs_bf16_bits);  // same bits, wrong interpretation
}

// PROPOSED FIX: explicitly widen each scale through its OWN correct type to
// FP32, then combine via a genuine FP32 fmul. Preserves the existing packed
// ABI (LHS stays FP16-packed, RHS stays BF16-packed) -- only the in-kernel
// combination step changes.
static float fixed_combined_scale_fp32(uint16_t lhs_fp16_bits, uint16_t rhs_bf16_bits) {
    float lhs_scale = f16_bits_to_f32(lhs_fp16_bits);
    float rhs_scale = kai_cast_f32_bf16(rhs_bf16_bits);
    return lhs_scale * rhs_scale;
}

// Buggy path exactly as the kernel currently computes it: BOTH operands
// widened as FP16 by fmlalb, i.e. multiply the two FP16-MISREAD magnitudes.
static float buggy_combined_scale_as_computed_by_kernel(uint16_t lhs_fp16_bits, uint16_t rhs_bf16_bits) {
    float lhs_scale_correct = f16_bits_to_f32(lhs_fp16_bits);   // LHS genuinely IS fp16 -- this part is correct
    float rhs_scale_misread = f16_bits_to_f32(rhs_bf16_bits);   // RHS is bf16, misread as fp16 -- this part is the bug
    return lhs_scale_correct * rhs_scale_misread;
}

static uint16_t f32_to_bf16_bits(float f32) {
    uint32_t i32;
    memcpy(&i32, &f32, sizeof(i32));
    return (uint16_t)(i32 >> 16);
}

static uint16_t f32_to_f16_bits(float f32) {
    __fp16 tmp = (__fp16)f32;
    uint16_t bits;
    memcpy(&bits, &tmp, sizeof(bits));
    return bits;
}

int main(void) {
    // Representative q4g64 weight scale (RHS, genuinely BF16) and per-block
    // int8 activation scale (LHS, genuinely FP16) -- see NOTE above.
    float true_rhs_weight_scale = 0.015625f;   // exact in both bf16 and f32 (power of two)
    float true_lhs_activation_scale = 109.0f;  // representative magnitude from D30's trace

    uint16_t rhs_bf16_bits = f32_to_bf16_bits(true_rhs_weight_scale);
    uint16_t lhs_fp16_bits = f32_to_f16_bits(true_lhs_activation_scale);

    printf("Inputs:\n");
    printf("  true RHS weight scale     = %.6f  (encoded as bf16 bits 0x%04x)\n", true_rhs_weight_scale, rhs_bf16_bits);
    printf("  true LHS activation scale = %.6f  (encoded as fp16 bits 0x%04x)\n", true_lhs_activation_scale, lhs_fp16_bits);

    float expected_combined = true_rhs_weight_scale * true_lhs_activation_scale;
    printf("\nExpected correct combined dequant scale = %.6e\n", expected_combined);

    float rhs_misread = buggy_rhs_scale_as_read_by_fmlalb(rhs_bf16_bits);
    printf("\n[diagnostic] RHS bf16 bits 0x%04x, misread as fp16 by fmlalb  = %.6e  (should be %.6f)\n",
           rhs_bf16_bits, rhs_misread, true_rhs_weight_scale);

    float buggy = buggy_combined_scale_as_computed_by_kernel(lhs_fp16_bits, rhs_bf16_bits);
    float fixed = fixed_combined_scale_fp32(lhs_fp16_bits, rhs_bf16_bits);

    printf("\nResults:\n");
    printf("  CURRENT kernel (fmlalb, RHS misread as fp16) -> combined scale = %.6e   ratio to expected: %.3e x\n",
           buggy, buggy / expected_combined);
    printf("  PROPOSED FIX (explicit widen-to-fp32 both sides, fp32 fmul)    -> combined scale = %.6e   ratio to expected: %.3e x\n",
           fixed, fixed / expected_combined);

    double err_buggy = fabs((double)buggy - (double)expected_combined);
    double err_fixed = fabs((double)fixed - (double)expected_combined);
    printf("\nabs error vs expected: buggy=%.6e  fixed=%.6e\n", err_buggy, err_fixed);
    printf("%s\n", (err_fixed < 1e-9 && err_buggy > 1e-6) ?
           "PASS: fix recovers the exact correct scale; current kernel path does not." :
           "UNEXPECTED -- re-check assumptions.");
    return 0;
}
