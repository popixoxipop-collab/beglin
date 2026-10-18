# BEGLIN Real DeepSeek-V2-Lite: 27-Layer C++ Metal / MLA Latent KV (2026-10-19)

**Research only / Draft PR #63 / production unchanged.** Real Apple Silicon EOE research with a SHA-pinned read-only D50 DeepSeek-V2-Lite BF16 source. The original XOX serving checkpoint's independent production authority has NOT been established. No model checkpoint, tensors, activations, logits, prompts, credentials or host-private scripts are checked into this repository.

## What was implemented (and what was not)

An original-source-compatible BF16 **resident latent MLA KV cache** stores just:
- **512** coordinates per position from the original `kv_a_layernorm(compressed_kv)` (BF16), and
- **64** source-original rotated key-position coordinates (BF16), with one copy rather than 16 identical attention-head copies.

On later decoding steps, a scoped `kv_a_layernorm` capture and source `kv_b_proj` of **previously cached** latent rows transiently reconstruct the prior expanded K/V for the unchanged original DeepSeek BF16 attention. The current position's independently computed source K/V are appended to this ephemeral attention input. The `kv_b_proj` is, in the real full-model research harness, an actual **compiled C++ / MLX Metal qNg64 n7 compressed projection**, not a pre-dequantized dense GPU weight. No full expanded K/V is retained after the attention call.

Critical isolation: before leaving each layer, the cache `unbind()` releases its reference to the underlying attention/Metal weights. `bind()` attaches the same source layer index again for the next decode; `fork()` supports MoE routing preview without altering the resident KV state. This is intentionally **B=1, 27 layers, research max-context=16 tokens** in actual full-model tests, not a generic production KV API.

This is **not** an optimized attention algorithm that avoids transient expanded K/V: the original BF16 attention and its K/V rehydration still run on CPU. It is not an all-C++ decoder and does not establish production TTFT, TPOT, steady-state tokens/s, long-context model quality or reduced process RSS.

## Real physics-backed 27-layer experiments

All quantized linear projections use the SHA-pinned compiled BEGLIN native C++ / Apple Metal qNg64 gather kernel. Original source DeepSeek MoEGate, BF16 attention, RMSNorm, residual, source embedding and BF16 lm_head run through the unchanged PyTorch CPU reference. Only source-routed experts are loaded, and the source checkpoint is streamed from a read-only D50 research copy, never all resident.

| Evidence | Real 1-token autoregressive run | Real **2-token** autoregressive run |
|---|---:|---:|
| Source decoder layers | 27/27 | 27/27 for prefill and both decode steps |
| Actual model-selected tokens **fed back as embeddings** | [429] | **[429, 4281]** |
| Next model greedy prediction (not yet fed back) | 4281 | **6230** |
| Checked compressed native C++ Metal module instances across pass and reference phases | 4,347 | **6,726** |
| Actual expanded-cache-bypass *resident* latent KV bytes | **186,624** (6 positions) | **217,728** (7 positions) |
| Corresponding original-source expanded BF16 KV capacity | 1,658,880 | 1,935,360 |
| **Measured reduction in persistent BF16 KV tensor bytes** | **88.75%** | **88.75%** |
| Source cached vs independent full causal recomputation: final logits relative L2 | 0.00289476435 | **0.00455995009** after second fed-back token |
| Next-token Top-1 cached vs uncached reference | 4281 == 4281 | **6230 == 6230** |
| Maximum native C++ Metal projection vs independent C relative L2 | 4.1775129e-7 | 4.1775129e-7 |
| Compiled source actually executed | yes | yes |
| Full process wall time (source weights streamed and references recomputed) | not a serving number | **813.772 s** |
| Native-cache update / prior latent rehydration, across 27 layers | 54 / 27 | **81 / 54** |

**Interpretation:** 16 original attention heads × (192 key + 128 value) × 2 BF16 bytes = **10,240 B/token/layer** for the original expanded-KV representation. Actual implemented resident normalized-latent + rotated-key stores (512+64) × 2 = **1,152 B/token/layer**. At seven positions across 27 layers, actual resident KV tensors equal **217,728 B**, compared with expanded **1,935,360 B**. This verifies the **cache-tensor storage** difference, NOT a corresponding reduction in peak process RSS, Metal unified-memory working set or end-to-end model size. Transient full K/V matrices are reconstructed per layer, and source BF16 original weights plus n7 packed modules are repeatedly reloaded/encoded in the research path.

The 2-token real run also observed approximately **270.50 s** for its first decoder step plus independent full recomputation and **282.55 s** for its second decoder step plus separate recomputation, while model source weights were reloaded. These times are **not** valid standalone cached-token generation latencies, and their inverse is **not** a certified throughput number. Peak process RSS was **1,567,653,888 B** during the two-token research process; this is not an all-model-resident peak benchmark.

## Source provenance and negative controls

- Original BF16 D50 research model index SHA-256: `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`.
- Frozen 5,181-matrix model-wide qNg64 uniform7 budget manifest SHA-256: `39b98921c30a992f93ea896283336a6b5fe1810c5737443132f2410b474e9612`.
- Actual C++ source `mlx_moe.cpp` SHA-256: `fdd9f08d4478ede612c4203c663eefb0a9cb4d2522773249dc05031cb0899151`.
- Actual EOE compiled research dylib SHA-256: `86283cf903915deb0947d68f5df2cba353f720afa1534139dfa8513815445741`.
- Source-only reusable cache module in this research branch: `research/local_quant_cpu_metal_20261017/latent_mla_kv_source_cache_v2.py`, SHA-256 `587b5371f31a88f07979e27405d26785e54942eccf5fb8c80e08d629eb3eb792`.
- Offline source-cache/invalid-shape/streaming-weight-unbind/rebind regression: **20/20 PASS** via `test_latent_mla_kv_source_cache_v2.py`. A further purely synthetic **64-token BF16 replay**, with six append stages and identical recompressed expanded source K/V, also PASS; this is a synthetic cache test, not a genuine 64-token 27-layer model run.
- Actual 1-token full-27 private research receipt SHA-256: `d5e8133cdbb62cd256af9ac9e6e1e233da93c284a8fd75f6d327249066fe9787`.
- Independent actual 1-token research auditor receipt SHA-256: `5b823964a13db4a8d980df8d3d138345397578ba16faf53f9340fea0db68a5e5`.
- **Actual full-27 two-successive-sampled-token private research receipt SHA-256:** `c3f7f9edfaddcc1ce8035261805e3e38a109a0027bdf31c455c55359bdd31645`.
- **Independent 2-token audit SHA-256:** `d1f1495fe33869467056c4b41a8d5cb7291eea2d08588f1989236fde31a53de8`, with **`PASS_TWO_ACTUAL_GREEDY_TOKENS_NATIVE_CPP_MLA_AND_RESIDENT_KV`**. The independent QA verified actual source greedy IDs, all 27 layers, 6,726 Metal modules, 81 KV source updates, 54 K/V rehydrations, the 7-token resident-byte sum, the BF16 source/packed native checksums, and numerical comparison with source no-cache recompute.
- The earlier expanded-KV and present latent-KV single-token experiments agree on all recorded per-layer hidden-gap metrics and selected expert counts, with identical final-source cached/recomputed logits relative L2.

Portable source/test files contain no absolute private host paths or checkpoint data. Hosted CI is **not a substitute** for these EOE hardware/research-source tests; only the portable module and synthetic tests are committed, while actual checkpoint-dependent scripts/receipts remain private and immutable on EOE.

## Security, release and remaining gates

- No mutation of XOX serving, model checkpoint, model auto-promotion, finance, Supabase DB, operator release or production environment. Research branch remains Draft PR #63; `main` is not merged.
- **Unresolved:** genuinely optimized MLA attention that never reconstructs expanded K/V; long real-model contexts across different independent prompts; additional multi-step generation beyond two fed-back IDs; fixed model resident packed weights/expert scheduling; CPU/GPU route without repeated requantization; independent task-quality, TTFT/TPOT/p95 latency and whole-process peak-memory experiments under serving workloads.
- No 88.75% total GPU-memory reduction, 88.75% whole-process reduction or tokens/sec improvement is claimed.
