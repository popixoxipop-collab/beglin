**STATUS: DRAFT ONLY -- NOT SUBMITTED. Requires explicit user approval before
posting to gitlab.arm.com/kleidi/kleidiai (CLAUDE.md §18: write actions
against repos not owned by the user require explicit approval).**

---

Title: `is_clamp_valid` in `KernelArgs` is declared `uint32_t` but the SME2 asm
kernel reads it as a 64-bit value, leaving a nondeterministic
compiler-padding gap

Kernel: `kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa`

## Summary

`KernelArgs.is_clamp_valid` is declared `uint32_t` (4 bytes) in
`kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa.c`, but
the hand-written SME2 assembly kernel
(`kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa_asm.S`)
reads the field with a full 64-bit load:

```
ldr x5, [x0, #0x28]
cbz x5, ...
```

Because `uint32_t is_clamp_valid` at offset `0x28` is followed by `size_t m`
at offset `0x30` (which requires 8-byte alignment), the C compiler inserts a
4-byte padding gap at `0x2C`-`0x2F` that no C code in this translation unit
ever writes to. The asm's 64-bit read at `0x28` spans the real field plus
this padding gap.

If a caller allocates `KernelArgs` on the stack without zero-initializing it
(or even with `KernelArgs ka = {0};` -- see below), the padding bytes are
uninitialized stack garbage. The `cbz x5` branch then nondeterministically
takes the clamp-enabled or clamp-disabled path depending on whatever was
previously on the stack at that address, call to call.

## Why `{0}`-initialization does not reliably fix this

We initially assumed `KernelArgs ka = {0};` would be sufficient, since it
zero-initializes all named fields including any implicit padding per the C
standard. In practice, on the optimizing compiler we tested (Apple
clang/LLVM, `-O2`, AArch64), the zeroing of the *padding byte range itself*
was eliminated as dead code: because no C-visible read ever observes that
padding directly (only the asm does, via a raw offset load the compiler
cannot see through), the store to those bytes has no observable effect from
the compiler's point of view and gets optimized away.

We verified this with LLDB by single-stepping into the kernel call and
inspecting `x5` directly after the `ldr x5, [x0, #0x28]`: even with
`ka = {0}`, `x5`'s upper 32 bits were sometimes non-zero garbage, and the
`cbz` branch outcome varied nondeterministically across otherwise-identical
calls.

## Reproduction

1. Call the kernel with `KernelArgs ka = {0};` (all other fields set
   correctly) across many repeated invocations with the same logical
   inputs.
2. Observe (e.g. via LLDB, breaking at the `cbz` after `ldr x5, [x0, #0x28]`)
   that `x5`'s value is not deterministically `0` across calls, and that
   downstream output occasionally differs between calls that should be
   numerically identical.

## Suggested fix

Widen the field to `size_t` (8 bytes), which matches what the asm actually
reads and removes the padding gap entirely -- the C assignment now
legitimately fills all 8 bytes the asm loads, with no reliance on
implicit-padding zeroing surviving optimization. This costs no extra memory:
the field already occupied space that alignment would have reserved as
padding before `m` at `0x30`.

```diff
-    uint32_t is_clamp_valid;     // 0x28
+    size_t is_clamp_valid;       // 0x28
     size_t m;                     // 0x30
```

We also recommend adding `_Static_assert(offsetof(KernelArgs, field) ==
0xNN, ...)` and `_Static_assert(sizeof(((KernelArgs*)0)->field) == N, ...)`
for every field this asm file addresses via a raw `[x0, #offset]` load, so a
similar width/offset mismatch fails at compile time rather than
silently corrupting results at runtime. (We prototyped this locally and
confirmed offset-only assertions do NOT catch this specific class of bug --
padding before the next 8-byte-aligned field keeps that field's offset
unchanged regardless of the buggy field's width -- so both offset and
per-field `sizeof` assertions are needed.)

## Patch

The one-line struct change above is attached as a minimal patch. Happy to
open an MR if that's the preferred contribution path for this repo (we note
GitHub is a read-only mirror and gitlab.arm.com/kleidi/kleidiai is the
canonical contribution home).
