**RETRACTED (2026-08-16) -- DO NOT SUBMIT. This issue's premise is FALSE.**
**Root cause found**: this project was feeding the kernel data packed by
`kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0` (BF16 scales) instead of the
kernel's ACTUALLY-DOCUMENTED, ARM-test-suite-confirmed companion packer
`kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon` (genuine FP16 scales
throughout, matching what `fmlalb`/`fmlalt` correctly expect). Confirmed via
`kleidiai_upstream/test/tests/matmul_clamp_f32_qsi8d32p_qsi4c32p_test.cpp`'s
own `UKernelVariants` table. With the correct pack pair, `kai_test_correct.c`
(same repo, `kleidiai/`) reproduces zero-NaN, ~2% quantization-noise-level
correctness across 4 random seeds -- the kernel's `fmlalb`/`fmlalt` usage is
NOT a bug. This project's own earlier pack/kernel pairing mistake (picked by
superficial name similarity in an unrecoverable prior session) was the sole
cause of everything documented below. Kept for the historical record only.

~~STATUS: DRAFT ONLY -- NOT SUBMITTED. Requires explicit user approval before
posting to gitlab.arm.com/kleidi/kleidiai (CLAUDE.md §18: write actions
against repos not owned by the user require explicit approval).
Discussion-first per our own conclusion below -- not proposing a specific
asm patch in this issue, since the fix touches hand-written SME2 assembly
and we'd like to agree on direction with maintainers first.~~

---

Title: `fmlalb`/`fmlalt` in the SME2 int4/int8 matmul kernel combine a
genuinely-FP16 LHS scale with a genuinely-BF16 RHS scale, but FMLALB/FMLALT
are FP16-only -- RHS scale is misread

Kernel: `kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa`

## Summary

The kernel dequantizes by combining a per-block LHS (activation) scale and a
per-block RHS (weight) scale, then multiplying against the raw INT8×INT4 dot
product accumulator. The two scales are combined via:

```
fmlalb z8.s, z19.h, z0.h
fmlalb z9.s, z19.h, z1.h
fmlalt z10.s, z19.h, z0.h
fmlalt z11.s, z19.h, z1.h
```

where `z19` carries the LHS activation scale and `z0`/`z1` carry the RHS
weight scale.

We believe there is a type mismatch between what these two operands actually
encode and what `fmlalb`/`fmlalt` expect:

- **LHS (activation) scale is genuinely FP16.** `kai_lhs_quant_pack_qsi8d32p_f32.c`
  stores it via `kai_cast_f16_f32()` (`kai/kai_common.h`), which performs a
  real IEEE-754 half-precision cast.
- **RHS (weight) scale is genuinely BF16.** `kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.c`
  copies the scale bytes via a raw `memcpy` with no conversion, and asserts
  `scale_dt == kai_dt_bf16` on the packing parameters.
- **`FMLALB`/`FMLALT` (vectors, FP16 to FP32)`, per the ARM SVE ISA
  reference, operate on FP16 source lanes exclusively** (`FPMulAddH()`,
  half-precision-specific) -- the reference explicitly notes no BF16 variant
  is specified for this encoding. `BFMLALB`/`BFMLALT` are a separate
  instruction pair with a different encoding for genuine BFloat16 operands.

So the kernel appears to use an FP16-only instruction to consume an operand
(`z0`/`z1`, the RHS weight scale) that the packing code produces as BF16.
Reading a BF16 bit pattern as if it were FP16 gives numerically nonsensical
results, since the two formats have different exponent width/bias (BF16: 8
exponent bits, bias 127, same range as FP32; FP16: 5 exponent bits, bias
15) -- the same 16 bits decode to wildly different magnitudes depending on
which format is assumed.

## Impact observed

Debugging a specific SME2 padding-mimicry integration, we traced a
dequantized output that was off by roughly 6 orders of magnitude (and, for
one degenerate all-equal-activation input, produced outright `NaN`) back to
this combined-scale computation via LLDB register inspection at the SME2
instruction level (single-stepping through all 16 `ZA_LOOP` iterations,
inspecting `z0`/`z8`/`z19`/`z28` directly). We isolated the scale-combination
math into a standalone scalar prototype (attached / linked) that reproduces
the same order-of-magnitude error when the RHS's BF16 bits are
bit-reinterpreted as FP16, and confirms exact recovery of the correct scale
when both operands are explicitly widened to FP32 (through their own
correct source type) before an FP32 multiply.

## What we are NOT proposing here

We are not proposing a specific SME2 assembly patch in this issue. The fix
likely means replacing the `fmlalb`/`fmlalt` combination with an explicit
widen-both-to-FP32-then-FP32-multiply sequence (preserving the existing
packed ABI -- LHS stays FP16-packed, RHS stays BF16-packed, only the
in-kernel combination changes), but we have not attempted that change in the
hand-written SME2 assembly ourselves and would rather confirm our
understanding of the intended scale-combination semantics with the
maintainers first, in case we are missing context about how `z0`/`z19` are
expected to be consumed.

## Questions for maintainers

1. Is our understanding of the LHS-FP16 / RHS-BF16 scale types correct, or
   is there a conversion step we're missing elsewhere in the pipeline that
   would make `fmlalb`/`fmlalt` valid here?
2. If this is confirmed, is an FP32-widen-both-then-multiply the preferred
   fix direction, or is there a reason (perf? existing convention elsewhere
   in the kernel family?) to prefer converting one operand's packed format
   instead (e.g. re-pack RHS scales as FP16 at pack time)?
3. Is there an existing correctness test in this repo's test suite that
   should have caught this, and if so, do you know why it didn't (we found
   and independently fixed an unrelated NaN-comparison bug in our own local
   test harness that was masking similar failures, and want to check
   whether something analogous exists upstream)?

Happy to share the scalar prototype and/or continue in a thread here.
