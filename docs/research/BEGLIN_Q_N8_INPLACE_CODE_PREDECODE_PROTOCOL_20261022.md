# BEGLIN Q — n8 in-place bitplane-to-code research optimization (2026-10-22)

Engineering-only experiment; original BEGLIN native source and immutable runtime unchanged.

The n8 qNg64 bitplane payload is exactly **one byte per weight** at group=64, plus the same F32 scale per group. Each GEMV currently decodes eight bitplanes per 16 codes. Test a **one-time in-place** conversion of each 64-byte bitplane group into the same 64 signed int8 codes, followed by an Apple NEON GEMV that reads contiguous codes. Conversion must finish decoding a full group before overwriting its source to prevent alias corruption.

**Controls:** original BEGLIN scalar GEMV, previously verified on-the-fly NEON bitplane GEMV and new in-place repacked NEON GEMV, all on the same actual Qwen2.5-1.5B source-BF16 group64 n8 packed tensor and 96 source-derived inputs. No task labels, no model output sensitivity, no V6 evaluation set. Compare strict output numerical tolerances (max absolute ≤0.0005, relative L2 ≤1e-5), conversion cost, 7 repeated rounds of 24 vectors with order alternation, and original packed-storage bytes vs repacked in-memory bytes. In-place content is **no longer compatible with the on-disk bitplane interpretation**; it must be tagged distinctly and never passed to the original scalar bitplane decoder, GPU path, or batched matmul. If full-model integration proceeds, use an opt-in isolated lab build and verify three-tier logits/NLL against original scalar n8.

No claims of general quality gain, final Q hypothesis acceptance, production performance or new tensor allocation. Preserve negative results.
