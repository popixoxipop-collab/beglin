# BEGLIN — native C++ qNg64 Metal bridge (2026-10-18)

**Research status: physical Apple Silicon GPU PASS at real projection; NOT model quality certified; NOT deployed.**

## Scope and engine change

The prior 27/27-layer DeepSeek-V2-Lite logits/NLL comparison ran Python-owned MLX Metal research kernels. That alone did not demonstrate the compiled BEGLIN C++ path. The new research-only exported API, mlx_gpu_qng64_bound_batch_probe(name, x, batch, y), routes already-bound single-expert qNg64 weights through the original lazy_matvec_e0 dispatch in mlx_moe.cpp. This is the path used by BEGLIN attention projection calls.

n6 uses the existing MLX native bit-repacking/quantized-matmul path. n7 and n9 use the existing custom compressed bitplane Metal kernel. No serving, loader, scheduler, automatic promotion or operating release was changed. The previously incomplete mlx_gpu_matvec_probe has also been made to accept the custom qNg64 binding kind instead of returning failure for n7/n9.

## Independent synthetic Apple hardware regression

Real physical Mac, MLX 0.32.1, actual compiled BEGLIN mlx_moe.cpp linked with MLX. The new tools/qng64_native_cpp_bridge_regression.cpp used a three-row batch and independent CPU decode/matmul reference.

| Width | Actual native route | Relative L2 vs CPU |
|---|---|---:|
| n6 | Native MLX repack | 5.36793565e-7 |
| n7 | Custom Metal bitplanes | 1.47234089e-7 |
| n9 | Custom Metal bitplanes | 3.62237828e-8 |

All passed with finite outputs. The test also passed n7 -> n6 -> n9 rebinding, binding snapshot/restore, GPU synchronization and weight-epoch reset, missing/invalid batch controls, and unsupported-width rejection. Existing QT mixed-qNg64 binding regression remained PASS.

## Real D50 DeepSeek-V2-Lite source projection parity

Source: read-only pinned BF16 safetensors research copy, model-original tokenizer and BF16 embedding/RMSNorm. Two new sentences produced 28 actual layer-0 activation positions. Private prompt texts and weights were not uploaded.

Original model tensor: model.layers.0.self_attn.q_proj.weight, shape 3072 x 2048. BEGLIN native C qng64_dense.c produced the packed bitplanes/scales. The actual compiled C++ mlx_gpu_bind_af -> lazy_matvec_e0 -> Metal output was paired with the independent actual C qng64_matmul_f32 on the same encoded bytes.

| Width | Packed model matrix including scales | C++ vs native C relative L2 | C++ vs original BF16 projection relative L2 |
|---|---:|---:|---:|
| n6 | 5,111,808 B | 2.20464487e-7 | 0.0149263 |
| n7 | 5,898,240 B | 6.51453990e-8 | 0.0074991 |
| n9 | 7,471,104 B | 6.62244632e-8 | 0.0024401 |

Real model q-projection C/C++ numerical parity PASS for all three widths. The BF16 projection error drops with increasing bit width but spends more bytes. This **does not** prove better whole-model logits, NLL or quality.

Private receipt SHA-256: ecf76f817b3eb8a99bd6c41f093855e3cffea85506af4d8fedf9bda53d4f9355.
Engine modified-source SHA-256: 64bdac9029eeb8a776167244152e601a98102681797438c36db8d08f9c.
Research C++ binary SHA-256: b819e43a07c5e709e498149caa31aea01bcebbd4d8120cc2c5f050c32fdfab7a.

## Reproduction and safety gates

Compile tools/qng64_native_cpp_bridge_regression.cpp with mlx_moe.cpp using Apple clang++ C++20 and the pinned MLX wheel headers/lib. Require JSON PASS with all 3 widths, restore, epoch, negative-guard checks. Existing tools/qt_mixed_qng64_binding_probe.cpp must still pass. GitHub CI compile/link gates are **compile-only**, not real GPU certification.

Unverified: native C++ 27-layer all-projection inference, real C++ routed MoE, 102,400-logit parity, autoregressive KV cache equivalence, weight-epoch recovery under multiple requests, realistic prefill/decode tokens/s, peak unified memory including KV, original XOX serving binary/source attestation and independent large-scale quality evaluation. The legacy mlx_gpu_bind_af blob_bytes field is not a complete bounds validator; whole-model tensor admission must harden exact offsets/shapes before release use.

## Next isolated gates

1. Fail-closed source-attested whole-model tensor manifest, offsets, dimensions and format rejection tests.
2. Native C++ real routed-MoE Gate/Up/Down and dense attention roles, matched to C and BF16.
3. Source-attested 27-layer C++ logits and incremental KV decode vs BF16 on held-out prompts with NLL, KL, tokens/s and unified-memory peak.

**Draft PR only, automatic_live_promotion=false, no production or release-plane changes.**
