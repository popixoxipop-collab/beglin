# Results

All numbers below are from real execution on Apple Silicon hardware (M4,
unless noted), against real pretrained weights (Qwen2.5-1.5B-Instruct
dense, plus an MoE variant for the batched-serving results). Nothing here
is projected or simulated.

## Correctness vs HuggingFace reference

| Check | Result |
|---|---|
| Per-layer hidden state error (fp32) | ≤ 8e-5 |
| Final logits error (fp32) | 6.3e-5 |
| Greedy decoding, 32 tokens | **32/32 exact match** |
| WikiText-103 validation ppl (fp32) | engine **10.648** vs HF **10.647** (ratio 1.00007) |
| HellaSwag, 200 items, length-normalized (fp32) | engine **60.50** = HF **60.50**, 100% per-item agreement |

## Quantization

| Config | ppl | Notes |
|---|---|---|
| RTN int4 | 14.87 | naive round-to-nearest baseline |
| **GPTQ int4 layers + int8 lm_head (deployment default)** | **12.10** | +13.6% vs fp32, −18.6% vs RTN; decode throughput lossless vs RTN |

## Dense-model decode throughput

Progressive optimization of the single-stream decode path, Qwen2.5-1.5B,
each step measured against the previous on the same hardware:

| Stage | Result |
|---|---|
| int4 inline dequant-GEMV, NEON | layers alone **3.63×** vs fp32 |
| + int8 untied lm_head | lm_head phase **7.3×** (40.3ms → 5.5ms), ppl unchanged |
| + hand-written NEON attention kernels | attn-only **−21.5% to −34.9%** vs BLAS-based attention |
| + prompt-lookup speculative decoding | repetitive text **~2.1×** (bitwise-identical to greedy) |
| **+ W4A8 int8-SDOT (int4→int8 dot-product) for both MLP GEMV and lm_head** | **~3.0×** end-to-end single-stream (**~19.4 → ~58.6 tok/s**), greedy **48/48 bit-identical** to the pre-SDOT baseline |

The SDOT result matters because two earlier, independently-reasoned
hypotheses about the decode-time bottleneck (memory bandwidth, and bit
width) were both tested and **rejected** by direct measurement before this
kernel rewrite was attempted — the real bottleneck was the `fp32`
int4-unpack conversion chain being compute-bound, not bandwidth-bound; an
int8 SIMD dot-product kernel (`vdotq_s32`, `FEAT_DotProd`) removes that
conversion chain entirely. Isolated microbenchmark: 6.05× single-thread,
rel-L2 0.4% (near-lossless).

## SME2-accelerated MoE serving

This is the newest and most involved part of the engine: request-batched
MoE decoding (`moe_cbatch_step`) with a runtime-detected ARM SME2 (Scalable
Matrix Extension v2) accelerated path for the expert GEMV, available on
Apple M4 and later.

### The bug that motivated a kernel-precision change

The original SME2 kernel
(`kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa`)
quantizes **both** the int4 expert weights and the fp32 activations to
int8 before the matmul. That's a real, structural source of divergence
from the scalar reference (which keeps activations in fp32) — not just
floating-point reduction-order noise. A margin-based selective
re-verification scheme (recompute in scalar whenever an SME2 output was
"close to a decision boundary") fixed the accuracy but cost more than
SME2's own speed advantage, so the net result was *slower* than pure
scalar despite the accelerated kernel.

**Fix**: swap in KleidiAI's `f16p`-LHS kernel variant
(`kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa`), which
keeps the int4 expert-weight quantization but packs activations as **fp16**
instead of int8 — meaningfully higher activation precision, at the cost of
requiring a separately-packed weight cache (RHS packing scheme differs:
`qsi4c32ps1s0scalef16` vs `qsi4c32ps4s0sf16`; LHS packing differs:
`f16pmrx2` vs `qsi8d32p1vlx4` — the two paths cannot share cached buffers).

| Config | Wall-clock (B=16, 342 tok) | Accuracy vs scalar ground truth | New errors vs scalar (not pre-existing) |
|---|---|---|---|
| (a) pure scalar (ground truth) | 208.5s | 100% (reference) | — |
| int8-LHS SME2, no re-verification | 85.0s | 87.7% (300/342) | 18 requests (3 of 8 base prompts) |
| **f16p-LHS SME2, no re-verification (default)** | **87.7s** | **93.0% (318/342)** | **0** — the only 6 remaining mismatches are a single, previously-documented engine-vs-reference edge case, reproduced identically under both kernels |

f16p-LHS is ~3% slower than int8-LHS at this batch size (the gap shrinks
from 15% at B=4 to 3.2% at B=16 as SME2 batching amortizes better), but
eliminates every new error the int8 kernel introduced, at 2.38× the speed
of pure scalar — **without needing the re-verification mechanism at all**.
That combination (no re-verification, still faster and more accurate than
the alternative) is why it's the engine's default; `QWEN_MOE_SME2_F16LHS=0`
reverts to the int8-LHS path with the margin re-verification option still
available in-tree, byte-identical to the pre-change behavior.

### A genuinely hard bug along the way, and how it actually got solved

Wiring the new kernel triggered a SIGILL that survived roughly 30 rounds of
non-interactive bisection (code content, object files, optimization level,
thread count, memory pressure, code signing, working directory, even a full
hardware reboot) with no reproducible cause found via any of that. It
turned out to be **two separate, ordinary software bugs**, not a flaky
environment:

1. A nibble-permutation repack loop got autovectorized by `clang -O2/-O3
   -march=armv9.2-a+sme2` into SVE instructions that are illegal outside
   SME2 "streaming mode" — fixed with
   `#pragma clang loop vectorize(disable) interleave(disable)`.
2. The engine's top-level caller file must **always** be compiled without
   any SME/SVE `-march` flag (see the Build section above) — adding
   unrelated code to that file had, at `-O2`/`-O3`, changed the compiler's
   autovectorization decision for a different, previously-correct function
   in the same translation unit.

Neither was found by static analysis or code review — both were only
confirmed via an actual interactive `lldb` backtrace on the target
hardware, which pinpointed the exact illegal instruction and its call
stack in each case. The lesson that generalizes: a SIGILL that survives
extensive *non-interactive* bisection is a signal to get an interactive
debugger session immediately, not a signal that the environment itself is
unreliable.
