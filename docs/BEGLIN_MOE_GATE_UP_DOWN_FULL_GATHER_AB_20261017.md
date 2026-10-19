# BEGLIN P18 — Full Gate/Up/Down packed Metal MoE gather A/B (2026-10-17)

**Research-only engine validation.** This experiment used DeepSeek-V2-Lite **real BF16 tensors from a D50 research copy**, the original source-defined `MoEGate`/first two decoder blocks, and the original packed qNg64 format. It **does not** attest the original XOX model files, run the 27-layer model, or modify serving, production release policy or backend dispatch.

## Executed source-defined workload and weight contract

Three short independently authored text prompts generated **37 hidden token positions**, a real source `topK=6` router, **222 token–expert pairs**, and **57 activated experts** out of the 64 available. Each activated expert has three Gate/Up/Down projection matrices, so **171 routed packed projection matrices** were measured. The always-active shared expert also used three packed GPU matrices.

| Selected source-defined MoE FFN weights only | Bytes |
|---|---:|
| Original BF16 equivalent | 1,020,788,736 |
| qNg64 n7 bitplanes + F32 scales | **478,494,720** |
| Weight payload reduction | **53.125%** |

This selected-experts payload total excludes the rest of the transformer, embedding, KV cache and temporary activation buffers. Source FFN weights used exactly the same BEGLIN C native qNg64 encoder before being resident on the GPU.

## A. Full routed FFN numerical equivalence

Previously only the Gate projection was gathered across experts. The new comparison executed all of the following with real BF16 model hidden states and identical encoded bytes:

1. **171 separate packed Metal projection calls** (57 Gate, 57 Up, 57 Down), followed by BF16 SiLU/multiply and source-defined topK weight reduction.
2. **Three packed Metal gather calls** (one per Gate/Up/Down), with exactly the same BF16 arithmetic boundaries and route aggregation.
3. Shared expert Gate/Up/Down on the same packed GPU data, included in the two-layer output.

Result: **bitwise identical final two-layer BF16 outputs** between 1 and 2. The relative L2 deviation from original, unquantized BF16 two-layer output was **0.009734939014** for either implementation. The original BF16 router/expert ID assignment was unchanged.

The earlier all-active-experts standalone experiment also established real Metal-vs-C maximum numerical relative L2 of **1.322×10⁻⁷** on the actual packed expert matrices.

## B. Interleaved CPU/MLX API-level timing

All five repetitions used the same loaded bitplanes and 222 expert-routing assignments. Measured wall time includes GPU transfers, kernel dispatch/synchronization and CPU Torch BF16 SiLU/route combination, while **excluding source attention/router generation, model loading, weight encoding and one shared-expert output precomputed identically for the comparison**.

| Method | Median time across five rounds | Relative to 171-call case |
|---|---:|---:|
| 171 per-expert packed Metal operations | 53.48 ms | 1.00× |
| **3 multi-expert packed Metal gathers** | **24.31 ms** | **2.20× faster** |

The 2.20× result is **routed MoE FFN projection-and-aggregation API time**, not whole-MoE source layer latency or LLM decoding tokens/s.

Recorded 3-gather source receipt SHA-256: `a395b587e0559fc164ede1e1c7c7f76a3ab5fb4e3342c4dcdbd8af5b2002ffd2`.

## C. Device-resident and fused candidate (negative performance findings)

To reduce host-side intermediate transfers, a research `QNg64MetalRoutedFFN` variant left BF16 SiLU, Down input and router reduction on the GPU. This matched the earlier mixed CPU/GPU output **exactly** but took **30.09 ms**, slower than the **24.37 ms** staged gather in the same experiment.

A further candidate fused SiLU×Up and the BF16 topK/shared/residual reduction into two custom Metal kernels:

| Variant (third experiment) | Median API wall time | Output difference vs staged gather |
|---|---:|---:|
| 171 separate packed expert projection calls | 53.48 ms | zero by independent per-expert comparison |
| **3 gathers + staged BF16 work** | **24.31 ms** | reference |
| GPU-resident unfused | 30.21 ms | exact array agreement |
| Metal-fused BF16 arithmetic | 29.81 ms | relative L2 **2.565×10⁻⁵** |

The fused relative L2 vs original BF16 was **0.009734973**, versus **0.009734939** for staged gather. The fused candidate did not improve speed. **Retain the 3-gather staged path as the research performance baseline**; do not default to fully device-resident/fused simply because it avoids host transfers.

Receipts:
- GPU-resident (v2) SHA-256: `afabbdf33d89f17d873313f3d6e38445a61eb53e4e08cb9718df4db8bf89f7b0`
- Fused (v3) SHA-256: `a5b29cb3b99b0dca0d51443f39aaf1f2dcc037f70e0ce22f501a84ade2cfca76`

## D. Engine implementation and hardware tests

Three opt-in standalone modules are available alongside the already-tested qNg64 Metal gather:

- `research/local_quant_cpu_metal_20261017/qng64_metal_expert_gather_research.py` — identical source algorithm to existing `mlx_moe.cpp::qng64_gather_gemv_kernel`.
- `research/local_quant_cpu_metal_20261017/qng64_metal_routed_ffn_research.py` — routed Gate+Up+Down plus optional shared and residual, one MLX memory context. **4/4 synthetic Apple GPU tests PASS**.
- `research/local_quant_cpu_metal_20261017/qng64_metal_fused_routed_ffn_research.py` — experimental Metal RNE-BF16 rounding/fused SiLU and topK reduction. **3/3 synthetic Apple GPU tests PASS**.

Hardware tests include non-overlapping batch shapes, independent per-expert packed calculations, on-device input equivalence, invalid expert-ID/type/range guards and packed bit mutation (in the basic gather tests). They are not automatically run by the generic GitHub CI where no verified Metal device is present.

In the current C++ BEGLIN engine, 6-bit routed projection weights follow a **different native MLX quantized-matmul/repacking route**; the Python research direct bitplane n6 path requires explicit opt-in and is **not production C++ behavior**.

## Next verification gates

1. A **source-identified XOX engine build** using this packed gather path, without changing live serving. Research Python wrapper equivalence alone is not proof that the live binary has been updated.
2. Robust performance benchmarking with 3-stage GPU ops and alternative activation reduction ordering on multiple prompt lengths/batches, power/thermal isolation and direct GPU-only timings. Do not extrapolate this 2.20× routed FFN API result to LLM tokens/s.
3. Model-wide full **27-layer DeepSeek-V2-Lite** BF16/uniform/mixed paired comparisons at exact total packed memory budget, including logits, NLL, generation stability, true CPU/GPU resident+peak memory, KV cache weight epoch and restart consistency.

All corresponding code remains **Draft PR research**, not merged/deployed. Source authority and server-owned issue certificates were not modified.
