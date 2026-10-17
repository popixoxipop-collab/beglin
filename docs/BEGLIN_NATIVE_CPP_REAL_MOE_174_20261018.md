# BEGLIN P18 — real DeepSeek-V2-Lite MoE through compiled C++ Metal (2026-10-18)

**Status: PASS native C++ hardware operator parity and 2-layer output; NOT full-model C++ / quality / serving certified.**

## New result and why it matters

The earlier model-wide 27/27-layer study used a Python-hosted MLX Metal bitplane kernel, not the compiled BEGLIN C++ dispatcher. The first follow-up verified a true C++ layer-0 Q projection. This follow-up uses **actual source-defined DeepSeek MoEGate top-6 routes**, original BF16 first two decoder layers, and the compiled BEGLIN C++ `mlx_moe.cpp` kernel to compress every active routed/shared expert projection in the **second transformer layer**.

The read-only D50 checkpoint, tokenizer, config and model source are digest-pinned to previous research. All 37 real token positions are from three short, author-written, previously fixed prompts, not an official model-quality dataset.

| Verification dimension | Result |
|---|---:|
| Source-generated positions | 37 |
| Source MoEGate selected experts | 57 of 64 |
| Routed token-expert pairs | 222 (top-6) |
| Routed packed Gate/Up/Down matrices | 171 |
| Shared packed Gate/Up/Down matrices | 3 |
| **Total real compressed model matrices exercised** | **174** |
| Source BF16 equivalent active weights | 1,020,788,736 B |
| Exact qNg64 n7 packed bytes (code + F32 group64 scale) | **478,494,720 B** |
| Actual compiled C++ routed Metal gather calls | **3** |
| Real C++ bound shared projections | **3** |
| BF16 reference vs native-C++-packed 2-layer final hidden relative L2 | **0.00973493901397561** |

Source BF16 experts and router are not replaced; input states and selected expert IDs are recalculated from original source code. Every packed matrix was encoded using BEGLIN C `qng64_dense.c`. Gate, Up and Down route projections execute the actual `mlx_gpu_qng64_gather_probe` kernel in compiled `mlx_moe.cpp` (the same low-level Metal kernel as the engine's `ffn_gather`), not Python `mlx.fast.metal_kernel`. The three shared-expert roles execute `mlx_gpu_bind_af` to `lazy_matvec_e0` via the actual C++ qNg64 Metal bound-batch probe. Native C packed-matrix calculations serve as independent numerical oracles.

The C++ gather probe is a **direct research C ABI**, not the full serving-layer `ffn_gather` reshape/canonicalization and scheduler. The experiment proves the low-level compiled GPU path and its BF16-compatible downstream arithmetic, **not** integrated 27-layer model serving.

## C and Metal numerical checks

On the actual DeepSeek MoE activations and source-selected experts, 8 deterministic token-expert pairs per routed role were compared against independent BEGLIN native C packed matvec:

| Role | Max sampled C++ GPU / native C relative L2 |
|---|---:|
| Gate | 1.31351561e-7 |
| Up | 1.34885224e-7 |
| Down | 1.18543302e-7 |

Four actual token rows per shared role were also verified against native C (12 additional checks). The final two-layer hidden-state L2 is consistent with the earlier Python-owned Metal result, to the recorded digits. This is operator-parity evidence, not a statistical model-quality improvement claim.

## Native C++ guard hardening

The direct qNg64 gather probe now rejects invalid or out-of-range expert IDs, null data pointers, group64 shape mismatches, unsupported direct Metal widths (n6 is a distinct MLX-native repack path), zero batches and provably overflowing packed-shape arithmetic **before any Metal dispatch**. The guard is local to the test entrypoint; full arbitrary checkpoint length/offset validation remains a separate requirement.

- New reusable synthetic C++ hardware test: `tools/qng64_native_cpp_gather_regression.cpp`.
- Valid synthetic expert gather (E=4, 9 routed pairs): n7 and n9 numerical PASS against independent CPU decode.
- Negative tests: **13 of 13** rejected for each width, including bad route IDs, shapes, widths, null pointers and overflow attempts.
- Existing n6/n7/n9 compressed projection regression and snapshot/weight-epoch checks: PASS after hardening.
- The **guarded C++** reran all real 57-expert/222-pair/174-matrix verification and produced the **same 0.00973493901397561** final relative L2 as the pre-guard C++.

## Integrity and private evidence

Source-based hardware execution and append-only research receipts are retained only on EOE, not uploaded with model weights or prompts.

- Initial actual C++ MoE receipt SHA-256: `8202bd7cbc1a2e98f14d262eaa1014509c28fd47f655d6fca015b1119aee9c7f`
- Hardened original-C++ MoE replay SHA-256: `0fdb388008e21a66a7cf95008edc0a3bde563b25f620037adae47daf9b7fc866`
- Hardened `mlx_moe.cpp` source SHA-256: `fdd9f08d4478ede612c4203c663eefb0a9cb4d2522773249dc05031cb0899151`
- Independent, cross-binary numerical and original-checkpoint identity audit: **PASS_CROSS_BINARY_MOE_GUARD_REPLAY**.

Total payload savings for these active expert matrices are **53.125%** relative to BF16 storage. This is *not* whole-model peak allocator memory, KV cache, on-device unified memory, or a benchmark on all 5,181 eligible matrices. Timed local kernels and processes must not be reported as tokens/second.

## Next strictly gated work

1. Validate full source-attested tensor-manifest offsets/lengths/group64/bitwidth before binding whole-model blobs (existing `mlx_gpu_bind_af` has a legacy weak bounds check).
2. Test the source-defined C++ FFN routed entrypoint including `rhs_indices` rank/broadcast and weight-epoch transitions, rather than only the direct gather probe.
3. Verify all 27 source layers through **native BEGLIN C++ model-level** execution and compare 102,400-vocab logits to the pinned BF16 source, with independent held-out NLL, full-vocabulary KL and generated-token accuracy.
4. Benchmark warmed autoregressive prefill/decode tokens/s, KV correctness, actual unified-memory peak, and equal whole-model serialized-weight bytes before selection/promotion.

**Draft PR #63 only. The actual XOX live engine/binary remains untouched. No production model-serving, model-quality certification or automatic_live_promotion is authorized.**
