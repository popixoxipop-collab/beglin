// sme2_kai.h -- vendor boundary between qwen_infer.c and kleidiai/.
//
// This header deliberately does NOT expose WT or any other qwen_infer.c type:
// keeps the vendored KleidiAI code and its runtime gate reusable/independent
// of the engine's internal weight representation, same spirit as this
// project's existing "sibling engines keep independent headers" convention
// (qwen_spec.c has its own load_arch_cfg() rather than sharing qwen_infer.c's).
//
// SAFETY CONTRACT: kai_sme2_available() is the only function that queries
// hardware. Every other function below re-checks it internally and returns a
// safe/inert value if unavailable -- so a caller that forgets to gate a call
// site still can't reach a real KleidiAI (and therefore SME2) code path. This
// matters because kai_get_sme_vector_length_u8() (used internally by several
// KleidiAI kai_get_* size queries) emits a raw `rdsvl` instruction OUTSIDE any
// __ARM_FEATURE_SME preprocessor guard (see kleidiai/kai_common_sme_asm.S) --
// unlike the actual GEMM kernel entry point, none of KleidiAI's own C wrappers
// guard this for us. On hardware without SME (e.g. this project's production
// machine, Apple M1 Max), executing it is an illegal instruction (SIGILL).

#ifndef SME2_KAI_H
#define SME2_KAI_H

#include <stddef.h>
#include <stdint.h>

// Runtime hardware check: FEAT_SME2 (SME2 present) AND SME_I8I32 (the int8->
// int32 outer-product form this MOPA kernel uses) both required. Cached after
// first call (lazy, like this project's existing detect_q4_threads()/
// w4a8_on() pattern). Always 0 on hosts lacking the sysctl keys entirely
// (sysctlbyname failure treated as "not available", not as an error).
int kai_sme2_available(void);

// Minimum M (batch size) worth routing to this kernel. Equals the kernel's
// own mr (row-tile height) -- below that, the MOPA tile is mostly wasted
// (verified 2026-08-16: M=1 decode is NOT a target for this reason). Returns
// INT_MAX when SME2 is unavailable, so any "M >= kai_sme2_min_m()" caller-side
// check is false by construction rather than by remembering to also check
// kai_sme2_available() -- redundant safety, not the only gate.
int kai_sme2_min_m(void);

// Whether (out, in) is a legal shape for this kernel's fixed group size
// (bl=64, matching vdsp's own K_Q4G64 packing -- this is a vdsp-format
// invariant, not a runtime-configurable value). Returns 0 whenever
// !kai_sme2_available(), independent of the shape.
int kai_sme2_shape_ok(int out, int in);

// Bytes needed for one tensor's KleidiAI-packed RHS buffer, or SIZE_MAX if
// SME2 is unavailable or the shape is invalid (kai_sme2_shape_ok() false) --
// SIZE_MAX rather than 0, so a careless caller that skips the shape/
// availability check doesn't misread "0 bytes" as "a valid empty allocation".
size_t kai_sme2_rhs_packed_bytes(int out, int in);

// One-time repack of a K_Q4G64 tensor's raw packed nibble bytes + per-group
// fp32 scales (vdsp's own layout: row stride in/2 bytes, group gi at byte
// offset gi*32 within a row, byte b = {low nibble -> col 2b, high nibble ->
// col 2b+1}, both formats already carry the same +8 zero-point bias so this
// is a pure nibble permutation, no arithmetic on values) into KleidiAI's
// RHS-packed buffer via kai_run_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon.
// `dst` must be >= kai_sme2_rhs_packed_bytes(out, in) bytes (call that first
// to size the allocation). Returns 0 on success; nonzero if SME2 is
// unavailable, the shape is invalid, dst_bytes is too small, or a scale isn't
// finite and nonzero (never observed on this project's actual model weights --
// see VENDOR.md -- but checked rather than assumed, since a corrupt/future
// weight file could still trip it). On any nonzero return the caller should
// leave that tensor's KleidiAI buffer unset and fall back to the existing
// NEON path for it, not treat the failure as fatal.
int kai_sme2_repack_q4g64(int out, int in, const uint8_t *packed, const float *scales, void *dst, size_t dst_bytes);

// Bytes needed for a scratch buffer big enough to hold kai_run_lhs_quant_pack_*'s
// packed output for up to `max_m` rows of `max_in` columns each (call ONCE
// with the largest M/in this process will ever pass to kai_sme2_gemm_f32,
// reuse the resulting buffer across every call). Returns 0 if SME2 is
// unavailable -- a 0-byte "scratch buffer" is never dereferenced because
// kai_sme2_gemm_f32 re-checks availability itself before touching it.
size_t kai_sme2_lhs_scratch_bytes(int max_m, int max_in);

// Runs one GEMM through the SME2 kernel: y[M][out] = x[M][in] @ rhs_packed^T (+ bias).
// `rhs_packed` must be a buffer kai_sme2_repack_q4g64() already filled for this
// exact (out, in) shape. `lhs_scratch` must be >= kai_sme2_lhs_scratch_bytes(M, in)
// bytes (a single buffer sized for the largest M/in the caller will ever pass
// is fine to reuse call after call -- this function does not retain any state
// in it). `bias` may be NULL (several projections have none). If
// !kai_sme2_available(), this is a silent no-op (does NOT touch `y`) --
// callers must gate on kai_sme2_available() themselves before deciding to call
// this at all; it re-checks only as a last-resort safety net, not as a
// substitute for that gating.
void kai_sme2_gemm_f32(int M, int out, int in, const float *x, const void *rhs_packed,
                       const float *bias, float *y, void *lhs_scratch);

// ============================================================================================
// f16p-LHS variant (2026-08-25 experiment). No residual/error-feedback logic here, same as
// kai_sme2_repack_q4g64() above: this file decodes an already-quantized blob and calls vendored
// KleidiAI pack/GEMM kernels with fixed per-group-64 symmetric quantization -- there is no
// tunable residual term to add (that's a training-time technique for learned quantization
// schemes; this is inference-time repacking of frozen weights into a hardware layout).
//
// Identical safety contract and API shape to the functions above, but routes through
// kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa instead of the
// qsi8d32p1vlx4_.../sme_mopa kernel used above. The difference: LHS (activations) are packed
// as fp16 (kai_run_lhs_pack_f16pmrx2_f32_neon) instead of being dynamically quantized to int8
// -- RHS (int4 weights) quantization is unchanged either way. Motivation: this session measured
// that int8 LHS quantization (not just SME2-vs-scalar reduction-order noise) is the dominant
// source of the ~15-20pp accuracy gap between the SME2 fast path and the scalar reference; f16
// LHS should narrow that gap at the source rather than requiring the Tier1/Tier2 margin-reverify
// mechanism to patch it after the fact. UNVERIFIED -- this is what the isolated microbenchmark
// (f16lhs_bench.c) exists to check before any engine wiring happens.
//
// Distinct feature gate: kai_sme2_available() checks FEAT_SME2+SME_I8I32 (the int8x4->int32
// outer-product datatype). This kernel needs FEAT_SME2+FEAT_FP16 instead (confirmed from
// kleidiai_upstream/test/common/cpu_info.cpp's cpu_has_sme2()+cpu_has_fp16() gate on the
// matmul_clamp_f32_f16p_qsi4c32p_test.cpp variant list) -- SME_I8I32 is irrelevant here since
// nothing is quantized to int8 in this path. Both keys confirmed present on bob (M4) via
// `sysctl hw.optional.arm.FEAT_FP16` = 1.
//
// RHS packed buffers from kai_sme2_repack_q4g64() (above) are NOT valid input to
// kai_sme2_gemm_f16lhs() and vice versa -- different internal tile layout
// (qsi4c32ps1s0scalef16 vs qsi4c32ps4s0sf16), different nr/kr. Callers must keep the two
// repack caches in entirely separate arrays/buffers, never mix them.
int kai_sme2_f16lhs_available(void);
int kai_sme2_f16lhs_shape_ok(int out, int in);
size_t kai_sme2_f16lhs_rhs_packed_bytes(int out, int in);

// Same vdsp K_Q4G64 packed-nibble + per-group-scale input contract as kai_sme2_repack_q4g64()
// (verified reusable: kleidiai_upstream/test/tests/matmul_clamp_f32_qsi8d32p_qsi4c32p_test.cpp
// feeds the SAME pack_data_scales_interleave_block<UInt4,Float16> reference construction into
// both the ps1s0 and ps4s0 RHS pack functions -- only the output tile layout differs, not the
// unpacked input this function builds). Same lhs_zero_point=1/rhs_zero_point=8 params confirmed
// via that test file too.
int kai_sme2_repack_q4g64_f16lhs(int out, int in, const uint8_t *packed, const float *scales, void *dst, size_t dst_bytes);

size_t kai_sme2_f16lhs_lhs_scratch_bytes(int max_m, int max_in);

// Runs one GEMM through the f16p-LHS SME2 kernel. Same call contract as kai_sme2_gemm_f32().
void kai_sme2_gemm_f16lhs(int M, int out, int in, const float *x, const void *rhs_packed,
                          const float *bias, float *y, void *lhs_scratch);

#endif // SME2_KAI_H
