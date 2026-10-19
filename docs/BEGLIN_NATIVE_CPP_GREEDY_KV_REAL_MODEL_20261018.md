# BEGLIN C++ Metal qNg64: Actual One-Token Greedy + KV, Full 27 Layers (2026-10-18)

**Scope: isolated source-attested EOE research; PR remains DRAFT and production is not modified.** This report adds genuinely model-selected token input to the earlier full-27 C++ packed-kernel and teacher-forced cache experiments. The checkpoint is an independent SHA-pinned BF16 DeepSeek-V2-Lite research copy read **only** from D50. The original XOX source-model production authority has **not** been established.

## The actual autoregressive boundary crossed

The following **two separate 27-layer passes** were executed on a physical Apple Silicon Metal device:

1. Prefill an original-source-tokenized **5-token** prompt, one layer at a time, through the original BF16 DeepSeek attention/RMSNorm/MoEGate plus the **compiled BEGLIN C++ qNg64(n7, group64) Metal gather kernel** for all quantized projections. Preserve each layer's actual BF16 expanded K/V.
2. Independently compute the 102,400-vocabulary BF16 lm_head logits of the prefix last token. The source model itself selected **greedy next token ID 429**. The resulting prefix last-position logits had **relative L2 = 0.0** compared with independently archived original-model Python MLX packed-n7 logits.
3. Load token **429** as the **actual newly selected input embedding**, and compute one incremental step through the 27 original decoder layers using their persistent prefill K/V. In the same research process, independently **recompute the full 6-token sequence without a cache** using the same original source and compiled C++ compressed kernels.
4. Compare each layer's last hidden state and, after layer 26, both 102,400-dimensional BF16 lm_head outputs. Both cached and uncached recomputation selected **next top-1 token ID 4281**. Token 4281 was predicted but **not** fed into another generation step.

The earlier teacher-forced 5+2 fixed-token experiment is a **different** protocol and cannot be substituted for this one-token sampled-input result.

## Fresh physical execution: PASS

| Measured or directly verified quantity | Result |
| --- | ---: |
| Original DeepSeek-V2-Lite source decoder layers | **27/27** |
| Genuine greedy-selected tokens **fed back** | **1** |
| Prefix source tokens / real cache length after step | 5 / 6 |
| Source prefix greedy token, embedded as next model input | **429** |
| Post-decode cached versus uncached next top-1 | **4281 / 4281** |
| Last-position logits relative L2, cached vs uncached | **0.002894764345814241** |
| Last-position logits maximum absolute difference | **0.125** |
| Packed native C++ Metal module instances constructed **and actually exercised**, across prefill and comparison/step | **4,347** |
| Largest sampled C++ Metal packed projection vs independent native C relative L2 | **4.177512908940176e-7** |
| Actual source expanded BF16 K/V bytes (27 layers, 6 tokens) | **1,658,880 B** |
| D50 BF16-weight-streaming prefill wall time | **265.19 s** |
| One-step + separate no-cache recomputation **including repeated source streaming and quantization** | **273.21 s** |
| Sum of incremental cached decoder-layer forward times **excluding** source-weight loading and independent recomputation | **8.20 s** |
| Whole research program wall time | **538.72 s** |
| Process peak RSS in this experiment | **1,494,466,560 B** |
| Durable job terminal state | **SUCCEEDED; exit code 0** |

The 4,347 count denotes checked packed-module **instances** across the prefill and step/reference phases; individual module instances may invoke the kernel more than once. It is not a count of independently sampled final tokens.

## Durable private evidence / independent source audit

- Native BEGLIN `mlx_moe.cpp` research source SHA-256: `fdd9f08d4478ede612c4203c663eefb0a9cb4d2522773249dc05031cb0899151`.
- Compiled real hardware C++ research dylib SHA-256: `86283cf903915deb0947d68f5df2cba353f720afa1534139dfa8513815445741`.
- Private isolated EOE greedy experiment source SHA-256: `96ab16952f2c50651e05e4201e1126898e9fd95a5f03357848b066f69c9f2dfa`.
- Actual experiment receipt SHA-256: **`286a9a2f338c4588cba4a7354397e8deba0340f475bfbb4cca15b7d5d97b090b`**.
- Independent source/shape/quantized-module/greedy-top1 audit SHA-256: **`6c100c22f1c8f4b360d8f2815d49f4e4f774bb67e275be55c1805a59c78f2b9e`**, verdict `PASS_AUTHENTIC_ARGMAX_ONE_STEP_CPP_KV_SOURCE_RECEIPTS`. An additional read-only replay of audit assertions confirmed 27 full layers, all 4,347 module instances, original frozen prompt's top1, source SHA, cache-byte accounting and numeric threshold.
- Model-wide source admission C++ real 4-shard/5,291 tensor/5,181 qNg64 matrix and 24 adversarial fixture tests remain separately recorded in `docs/BEGLIN_NATIVE_CPP_FULL27_MLA_KV_SOURCE_BOUNDS_20261018.md`.
- Physical 1-layer and 2-layer isolated C++/KV input-path regression both PASS, with zero hidden difference between KV and full causal recompute on those small-prefix tests.

**No model tensor data, logits array, model files, prompt text, model-checkpoint mount path, environment secrets, or machine-private research scripts are committed to this repository.** Full private receipts and scripts remain solely on the authorized EOE host.

## KV memory interpretation and next engineering gate

The existing original-source expanded BF16 per-head cache stores **16 attention heads × (192 K + 128 V) × 2 BF16 bytes = 10,240 B per token per layer**, or **276,480 B per token for 27 layers**. Therefore the **actual 6-token saved cache** is 1,658,880 B.

The original DeepSeek-V2-Lite config also has a 512-dimensional KV low-rank latent and a 64-dimensional rotary-key component. The **hypothetical** latent BF16 cache would contain (512+64)×2 = 1,152 B per token per layer (186,624 B for six tokens across 27 layers), a **theoretical 88.75% KV-byte reduction**. **No compressed latent MLA KV cache was implemented, evaluated for numerical correctness, or measured for runtime memory here**; these are source configuration sizes only.

Next safe steps: implement and independently verify a latent MLA KV-cache algorithm against the actual original attention on fresh token-by-token inputs; test at least one longer, independent greedy trace; remove per-layer source re-quantization and improve memory residency; measure TTFT, TPOT, p95 and memory **under genuine serving conditions** with original deployment authority. Keep draft PR and GPU auto-promotion disabled until independent quality/latency certification.

## Explicit exclusions

This proof is **not** a full native-C++ LLM forward: source attention, residuals, normalizations, router and lm_head execute in PyTorch BF16 CPU, while the quantized linear projections use native compiled C++ Metal. The test generates **one** token and produces a next-logits prediction; it does not certify long-context decode, two sampled tokens fed back, latent MLA KV, full model checkpoint resident, true sustained end-to-end tokens/s, production inference p99, task quality, or promotion. The reported 273.21-second paired research step includes reloading source weights and performing independent full recomputation and is **not** a legitimate inverse tokens/s estimate.

**Promotion/serving status:** main unmodified, PR remains DRAFT, original XOX serving unmodified, automatic_live_promotion=false.
