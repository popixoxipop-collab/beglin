//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa.h"

#include <float.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

typedef struct {
    size_t lhs_packed_stride;    // 0x00
    size_t rhs_packed_stride;    // 0x08
    size_t mr;                   // 0x10
    size_t bl;                   // 0x18
    float scalar_min;            // 0x20
    float scalar_max;            // 0x24
    // D27 (bugfix, supersedes the {0}-init attempt below): the asm kernel reads this field via
    //   `ldr x5, [x0, #0x28]` (64-bit) + `cbz x5`, not a 32-bit load. Declaring it uint32_t left
    //   a 4-byte compiler-inserted alignment gap before `m` (0x30) that no C code ever writes to
    //   -- an optimizing compiler is free to (and did, empirically) treat zeroing that gap as
    //   dead code, since nothing observable-in-C depends on it. Widening the field itself to
    //   size_t removes the gap entirely: the C assignment now legitimately fills all 8 bytes the
    //   asm reads, with no reliance on padding surviving optimization. EXIT: revert if upstream
    //   fixes the asm kernel to use a genuine 32-bit load instead.
    size_t is_clamp_valid;       // 0x28
    size_t m;                    // 0x30
    size_t n;                    // 0x38
    size_t k;                    // 0x40
    const void* lhs_packed;      // 0x48
    const void* rhs_packed;      // 0x50
    const uint16_t* lhs_scales;  // 0x58
    const uint16_t* rhs_scales;  // 0x60
    float* dst;                  // 0x68
    size_t dst_stride_row;       // 0x70
} KernelArgs;

// Phase-2 ABI guard (2026-08-16, follows D27): the asm kernel
// (..._asm.S) addresses every field of this struct by a raw `ldr x_, [x0,
// #0xNN]` offset from a KernelArgs* -- there is no shared header or
// compiler-checked ABI between this C struct and the hand-written assembly.
// D27 found is_clamp_valid's offset held (uint32_t left a compiler padding
// gap the asm's 64-bit read silently included), but nothing stopped the
// *next* accidental field reorder/retype from breaking a different offset
// the same way and shipping wrong dequant math instead of a compile error.
// These asserts pin every offset the asm file actually reads (extracted via
// `grep -n '\[x0, #0x' *_asm.S`, cross-checked 2026-08-16) so a mismatch is
// a build failure, not a silent runtime miscalculation.
_Static_assert(offsetof(KernelArgs, lhs_packed_stride) == 0x00, "ABI: asm reads lhs_packed_stride at [x0, #0x00]");
_Static_assert(offsetof(KernelArgs, rhs_packed_stride) == 0x08, "ABI: asm reads rhs_packed_stride at [x0, #0x08]");
_Static_assert(offsetof(KernelArgs, mr) == 0x10, "ABI: asm reads mr at [x0, #0x10]");
_Static_assert(offsetof(KernelArgs, bl) == 0x18, "ABI: asm reads bl at [x0, #0x18]");
_Static_assert(offsetof(KernelArgs, scalar_min) == 0x20, "ABI: asm expects scalar_min at [x0, #0x20]");
_Static_assert(offsetof(KernelArgs, scalar_max) == 0x24, "ABI: asm expects scalar_max at [x0, #0x24]");
_Static_assert(offsetof(KernelArgs, is_clamp_valid) == 0x28, "ABI: asm reads is_clamp_valid at [x0, #0x28] (D27)");
_Static_assert(offsetof(KernelArgs, m) == 0x30, "ABI: asm reads m at [x0, #0x30]");
_Static_assert(offsetof(KernelArgs, n) == 0x38, "ABI: asm reads n at [x0, #0x38]");
_Static_assert(offsetof(KernelArgs, k) == 0x40, "ABI: asm reads k at [x0, #0x40]");
_Static_assert(offsetof(KernelArgs, lhs_packed) == 0x48, "ABI: asm reads lhs_packed at [x0, #0x48]");
_Static_assert(offsetof(KernelArgs, rhs_packed) == 0x50, "ABI: asm reads rhs_packed at [x0, #0x50]");
_Static_assert(offsetof(KernelArgs, lhs_scales) == 0x58, "ABI: asm reads lhs_scales at [x0, #0x58]");
_Static_assert(offsetof(KernelArgs, rhs_scales) == 0x60, "ABI: asm reads rhs_scales at [x0, #0x60]");
_Static_assert(offsetof(KernelArgs, dst) == 0x68, "ABI: asm reads dst at [x0, #0x68]");
_Static_assert(offsetof(KernelArgs, dst_stride_row) == 0x70, "ABI: asm reads dst_stride_row at [x0, #0x70]");
_Static_assert(sizeof(KernelArgs) == 0x78, "ABI: struct total size must match asm's implicit layout (0x70 + 8B dst_stride_row)");
// Width guard: offsetof alone does NOT catch a D27-style regression, because a
// narrower field's compiler-inserted alignment padding keeps every LATER field's
// offset unchanged (verified empirically 2026-08-16: reverting is_clamp_valid to
// uint32_t alone still compiles clean under only the offsetof asserts above --
// m stays at 0x30 either way). Every field the asm reads via a 64-bit `ldr x`
// must independently be pinned to 8 bytes.
_Static_assert(sizeof(((KernelArgs*)0)->lhs_packed_stride) == 8, "ABI: lhs_packed_stride must be 8 bytes (asm 64-bit read)");
_Static_assert(sizeof(((KernelArgs*)0)->rhs_packed_stride) == 8, "ABI: rhs_packed_stride must be 8 bytes (asm 64-bit read)");
_Static_assert(sizeof(((KernelArgs*)0)->mr) == 8, "ABI: mr must be 8 bytes (asm 64-bit read)");
_Static_assert(sizeof(((KernelArgs*)0)->bl) == 8, "ABI: bl must be 8 bytes (asm 64-bit read)");
_Static_assert(sizeof(((KernelArgs*)0)->is_clamp_valid) == 8, "ABI: is_clamp_valid must be 8 bytes (asm 64-bit read, D27)");
_Static_assert(sizeof(((KernelArgs*)0)->m) == 8, "ABI: m must be 8 bytes (asm 64-bit read)");
_Static_assert(sizeof(((KernelArgs*)0)->n) == 8, "ABI: n must be 8 bytes (asm 64-bit read)");
_Static_assert(sizeof(((KernelArgs*)0)->k) == 8, "ABI: k must be 8 bytes (asm 64-bit read)");
_Static_assert(sizeof(((KernelArgs*)0)->lhs_packed) == 8, "ABI: lhs_packed must be 8 bytes (asm 64-bit read)");
_Static_assert(sizeof(((KernelArgs*)0)->rhs_packed) == 8, "ABI: rhs_packed must be 8 bytes (asm 64-bit read)");
_Static_assert(sizeof(((KernelArgs*)0)->lhs_scales) == 8, "ABI: lhs_scales must be 8 bytes (asm 64-bit read)");
_Static_assert(sizeof(((KernelArgs*)0)->rhs_scales) == 8, "ABI: rhs_scales must be 8 bytes (asm 64-bit read)");
_Static_assert(sizeof(((KernelArgs*)0)->dst) == 8, "ABI: dst must be 8 bytes (asm 64-bit read)");
_Static_assert(sizeof(((KernelArgs*)0)->dst_stride_row) == 8, "ABI: dst_stride_row must be 8 bytes (asm 64-bit read)");

extern void kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(KernelArgs* args_ptr);

// Compute args
static const size_t kai_m_step = 1;  // Multiple of vector length
static const size_t kai_n_step = 4;  // Multiple of vector length
// Packing args
static const size_t kai_mr = 1;  // Multiple of vector length
static const size_t kai_nr = 4;  // Multiple of vector length
static const size_t kai_kr = 4;
static const size_t kai_sr = 2;
// LHS format args (num. bytes per value, multiplier)
static const size_t kai_num_bytes_qvalue_lhs = 1;
static const size_t kai_num_bytes_multiplier_lhs = 2;
// RHS format args (num. bytes per value, multiplier)
static const size_t kai_recip_num_bytes_qvalue_rhs = 2;
static const size_t kai_num_bytes_multiplier_rhs = 2;
// DST format args
static const size_t kai_num_bytes_dst_value = 4;
// Extra args
static const size_t kai_bl = 32;

inline static size_t kai_get_num_bytes_per_block_lhs(const size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);
    return (bl * kai_num_bytes_qvalue_lhs) + kai_num_bytes_multiplier_lhs;
}

inline static size_t kai_get_num_bytes_per_block_rhs(const size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);
    const size_t num_bytes_per_block_rhs = (bl / kai_recip_num_bytes_qvalue_rhs) + kai_num_bytes_multiplier_rhs;
    return num_bytes_per_block_rhs;
}

inline static size_t kai_get_num_blocks_per_row(const size_t k, const size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);
    KAI_ASSUME((k % bl) == 0);

    return k / bl;
}

inline static size_t kai_get_lhs_packed_stride(const size_t k, const size_t bl) {
    const size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    return mr * kai_get_num_blocks_per_row(k, bl) * kai_get_num_bytes_per_block_lhs(bl);
}

inline static size_t kai_get_rhs_packed_stride(const size_t k, const size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);
    KAI_ASSUME((k % bl) == 0);

    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block_rhs(bl);
    const size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();

    const size_t rhs_packed_stride = nr * (num_bytes_per_block * num_blocks_per_row);

    return rhs_packed_stride;
}

size_t kai_get_m_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(void) {
    return kai_m_step * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(void) {
    return kai_n_step * kai_get_sme_vector_length_u32();
}

size_t kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_kr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(
    const size_t m_idx, const size_t k, const size_t bl) {
    const size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    KAI_ASSUME((m_idx % mr) == 0);

    return (m_idx / mr) * kai_get_lhs_packed_stride(k, bl);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(
    const size_t n_idx, const size_t k, const size_t bl) {
    const size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();

    KAI_ASSUME((n_idx % nr) == 0);

    return (n_idx / nr) * kai_get_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(
    const size_t m_idx, const size_t n_idx, const size_t dst_stride) {
    const size_t m_step = kai_get_m_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    const size_t n_step = kai_get_n_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    KAI_ASSUME((m_idx % m_step) == 0);
    KAI_ASSUME((n_idx % n_step) == 0);

    return (n_idx * kai_num_bytes_dst_value) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(const size_t m, const size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(
    const size_t m,                   //
    const size_t n,                   //
    const size_t k,                   //
    const size_t bl,                  //
    const void* restrict lhs_packed,  //
    const void* restrict rhs_packed,  //
    float* restrict dst,              // NOLINT(readability-non-const-parameter)
    const size_t dst_stride_row,      //
    const size_t dst_stride_col,      //
    const float scalar_min,           //
    const float scalar_max) {
    KAI_UNUSED(dst_stride_col);
    KAI_ASSUME((bl % kai_bl) == 0);

    // D27 (bugfix, root-caused via LLDB): KernelArgs.is_clamp_valid is declared uint32_t (0x28),
    //   but the hand-written SME2 kernel (kai_kernel_matmul_..._asm.S) reads it as a full 64-bit
    //   value (`ldr x5, [x0, #0x28]; cbz x5, ...`), spanning the field AND the compiler-inserted
    //   4-byte alignment padding before the next size_t field `m` (0x30). An uninitialized `ka`
    //   left that padding as stale stack garbage from whatever ran before this call -- on some
    //   calls the garbage read as zero (clamp correctly skipped), on others nonzero (clamp
    //   spuriously entered, corrupting the result down to NaN). WHY: {0}-init is the minimal fix
    //   that zeroes the whole struct including padding, without touching the asm kernel's ABI.
    //   COST: one extra zero-fill of a small stack struct per call, negligible.
    //   EXIT: if upstream fixes the asm kernel to use a 32-bit load instead, this becomes moot.
    KernelArgs ka = {0};

    const size_t num_blocks = kai_get_num_blocks_per_row(k, bl);

    const size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    const size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();

    ka.mr = mr;
    ka.lhs_packed_stride = kai_get_lhs_packed_stride(k, bl);
    ka.rhs_packed_stride = kai_get_rhs_packed_stride(k, bl);
    ka.bl = bl;
    ka.scalar_min = scalar_min;
    ka.scalar_max = scalar_max;

    const uint8_t* lhs_packed_bytes = (const uint8_t*)lhs_packed;
    const uint8_t* rhs_packed_bytes = (const uint8_t*)rhs_packed;
    const uint16_t* lhs_scales =
        (const uint16_t*)(lhs_packed_bytes + ka.lhs_packed_stride - (mr * num_blocks) * kai_num_bytes_multiplier_lhs);
    const uint16_t* rhs_scales =
        (const uint16_t*)(rhs_packed_bytes + ka.rhs_packed_stride - (nr * num_blocks) * kai_num_bytes_multiplier_rhs);

    ka.is_clamp_valid = (scalar_min > -FLT_MAX) || (scalar_max < FLT_MAX);
    ka.m = m;
    ka.n = n;
    ka.k = k;
    ka.lhs_packed = lhs_packed;
    ka.rhs_packed = rhs_packed;
    ka.lhs_scales = lhs_scales;
    ka.rhs_scales = rhs_scales;
    ka.dst = dst;
    ka.dst_stride_row = dst_stride_row;

    kai_commit_za();

    kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(&ka);
}

#endif  // Architectural features check.
