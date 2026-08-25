# Vendored code

## gguf_quants.c

**Upstream**: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), specifically the
`ggml` subproject (`ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`).

**Commit**: `d83f72d463287ab9c50b4bc18ee332104a963889` (2026-08-17), from a local checkout
(`~/llamacpp_kleidi_build` on `bob`).

**License**: MIT (`Copyright (c) 2023-2026 The ggml authors`).

**What was vendored**: block struct layouts (`block_q4_0`, `block_q8_0`, `block_q4_K`,
`block_q6_K` from `ggml-common.h`, renamed `GgmlBlockQ4_0` etc. here) and the dequantization
algorithms for those four types plus the `get_scale_min_k4` helper Q4_K's dequant depends on
(from `ggml-quants.c`'s `dequantize_row_q4_0` / `dequantize_row_q8_0` / `dequantize_row_q4_K` /
`dequantize_row_q6_K`). These are direct, unmodified ports of the algorithm logic — same
operation order, same types — with only naming adapted to this project's conventions. Function
bodies are annotated with "Ported from ggml-quants.c's ... (see file header for provenance)"
comments at each site in `gguf_quants.c`.

**What was NOT vendored, and why**:
- The full `ggml-quants.c`/`ggml-common.h` files (dozens of other quant types, quantization —
  not just dequantization — routines, SIMD-optimized variants) — only the four types this
  project's Phase 1 scope targets (see `PLAN_general_purpose_loader.md`).
- `ggml_compute_fp16_to_fp32`, ggml's portable F16→F32 conversion — this project's target
  (Apple clang on Apple Silicon) has native `__fp16` hardware conversion already relied on
  elsewhere in this codebase (`f16lhs_bench.c`, `sme2_kai.c`'s f16p-LHS path); reinterpreting
  the raw bits as `__fp16` and letting the compiler emit the native conversion is simpler and
  needs no vendoring.
- BF16→F32 — a lossless bit-shift (bf16 is the truncated upper 16 bits of an IEEE754 float32),
  not something with a chip-specific fact to get wrong.
- llama.cpp's own GGUF container reader (`ggml/src/gguf.cpp`) — see `PLAN_general_purpose_loader.md`
  D-gen-1 for why `gguf_load.c` is a from-scratch parser instead (that C++ file pulls in ggml's
  backend/type-trait machinery this project doesn't want, and doesn't have this project's own
  bounds-checking discipline for untrusted input).

**Verification** (not just reviewed — run against real data, see `RESULTS.md`): the ported
dequant functions were checked against `gguf-py`'s independent numpy implementation
(`gguf.quants.dequantize`, a *different* implementation, not a wrapper around the same C code)
on a real 1.1GB production GGUF file. Every one of its 339 tensors (all three quant types
actually present — F32, Q4_K, Q6_K) dequantized to a checksum that matches the Python oracle's
independently-computed checksum to 9 significant digits, with zero mismatches; a deeper
first/last-20-element exact-value check on a representative tensor of each type also showed
zero diff.
