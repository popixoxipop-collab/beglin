# BEGLIN Native C++ qNg64 / Full-27 Research Evidence — 2026-10-18

## Scope and source identity

**Research only.** The original BF16 DeepSeek-V2-Lite checkpoint is a pinned, read-only, independently held research copy. Original XOX serving model authority has **not** been attested by this work. No checkpoint tensors, embeddings, hidden states, logits, user prompts, or private machine paths are checked into GitHub.

The original BEGLIN `mlx_moe.cpp` native C++ packed qNg64 Metal kernel was exercised using a **physical Apple Silicon GPU** in an isolated EOE research process. The frozen uniform n7 (64-element groups, uint8 bitplanes and float32 scales) policy is used for exact comparison; 110 excluded BF16 tensors (embedding, norm, routing gate, lm_head and other noneligible tensors) remain BF16.

Compiled source SHA-256: `fdd9f08d4478ede612c4203c663eefb0a9cb4d2522773249dc05031cb0899151`. Compiled research dylib SHA-256: `86283cf903915deb0947d68f5df2cba353f720afa1534139dfa8513815445741`.

## A. Actual source checkpoint header/boundary admission

An independent C++17 validator (`tools/verify_full27_source_bounds_cpp_v1.cpp`) validates a separate, SHA-pinned read-only source-boundary manifest and opens original safetensors files with `O_NOFOLLOW` and `pread`, without reading weight payloads. It checks full shard sizes, raw header digests, tensor name uniqueness, BF16 shape/rank, exact `[begin,end)` byte bounds, partition without gaps/overlaps, expected 27-layer and 64-expert group64 roles, packed bitplane/F32-scale bytes and overflow. It also rejects noncanonical real source directory paths and verifies the frozen index/config/model-source digests.

| Measured source property | Value |
| --- | ---: |
| Safetensors shards | 4 |
| Total BF16 tensors | 5,291 |
| qNg64-eligible matrices | 5,181 |
| Eligible per layer | 7 (layer 0), 199 each (layers 1–26) |
| Source BF16 weight bytes | 31,412,968,448 |
| Uniform n7 compressed eligible weight bytes | 14,328,299,520 |
| Excluded BF16 bytes | 845,929,472 |
| Total model-wide serialized payload (n7 + BF16 exclusions) | 15,174,228,992 |

Source-boundary manifest SHA-256: `274247e626f67579e56256e356f554fd1b1e5e5967df2603ec82bfb902af4963`. The manifest includes only structural source metadata and remains on EOE; the 31GB checkpoint is never copied or packed into the GitHub artifact.

Result: `PASS_CPP_SOURCE_MANIFEST_BOUNDS_NO_WEIGHT_PAYLOAD`.

Independent isolated synthetic QA: **1 valid fixture accepted and 24 invalid-input checks refused** (including malformed header, missing/overlapping tensor spans, symlink, unexpected dtype/bit width, bytes mismatch, overflow attempts, invalid expert index and missing newline). Test source: `tools/test_full27_source_bounds_cpp_v1.py`. GitHub hosted CI may compile and execute these isolated synthetic checks; it does **not** contain the real model.

## B. Actual DeepSeek 27-layer logits via compiled C++ Metal compressed projection

The research harness streams source BF16 decoder layers and routes only selected MoE experts. It executes **2,064 source-selected quantized projections** via the native compiled `mlx_gpu_qng64_gather_probe`, which calls BEGLIN's original Metal gather bitplane kernel. It uses the unchanged source PyTorch BF16 attention, MoEGate, RMSNorm, residuals, embedding and lm_head. This is a real **C++ packed-projection / BF16 CPU hybrid 27-layer forward**, not an all-C++ inference engine.

| Metric | Observed |
| --- | ---: |
| Actual source decoder layers evaluated | 27/27 |
| Compiled C++ Metal source-selected qNg64 matmuls | 2,064 |
| Final logits shape | 5 × 102,400 |
| Logits relative L2 vs independent Python/MLX packed Metal execution | **0.0** |
| Maximum logits absolute difference vs Python/MLX packed | **0.0** |
| Largest sampled native C++ Metal vs independent C reference relative L2 | 4.1775129e-7 |
| Logits relative L2 vs original BF16 source output | 0.0198469275 |
| Prompt-specific self-NLL (nats) | 5.2575125694 |
| First prompt final top-1 token ID | 429 |
| Entire research program wall time, with repeated source quantization/IO | 263.05 s |

Real-source receipt SHA-256: `8974323ad00ee798621341df8377b9ae03ae44092c85f7c8b67061416e3231f9`. The original independent Python/MLX packed logits file SHA-256 equals the native research logits file SHA-256 `6db850be2b069b34b6510d9746df6387078022c9f668e37bf3cb195e65e3621e`.

Status: `PASS_FULL27_CPP_METAL_PROJECTIONS_LOGITS_PARITY` — **one short non-frozen prompt only**, not a full-model task/quality certification or general throughput test.

## C. Original-model BF16 expanded K/V cache, same 27 native C++ Metal quantized layers

Using **five prefix tokens and two predetermined teacher-forced continuation token IDs**, the research harness actually updates and rereads the original DeepSeek attention's expanded BF16 K/V cache. At every layer, the same source qNg64/C++ packed projections execute both cached prefix+decode and uncached full-sequence recomputation, using the same union of routed experts. This does **not** sample/autoregressively generate the two continuation tokens.

| Metric | Observed |
| --- | ---: |
| Validated source layers | 27/27 |
| Actual C++ Metal compressed matrices executed | 2,439 |
| Max per-layer cached vs full recompute hidden relative L2 | 0.0017471374 |
| Final cached vs full recompute logits relative L2 | 0.0050229085 |
| Maximum final logits absolute difference | 0.1875 |
| Last-token top-1 | 4281 in both arms |
| Physical per-layer expanded BF16 KV buffer bytes at 7 tokens | 71,680 |
| Sum of 27 such source-layer KV buffers | 1,935,360 bytes |
| Sum divided by 7 tokens | 276,480 bytes/token |
| Full driver time including weight source streaming and qNg64 packing | 304.71 s |

Note: The 1,935,360-byte total is a sum of **independently observed one-layer KV buffers**, not a measurement of simultaneous GPU residency or optimized compressed MLA KV. Current source writes K/V at 16 attention heads, BF16, rather than persistent low-rank MLA KV. First-call CPU-oracle and quantization costs make these driver timings inappropriate for claiming end-to-end tokens/s, TTFT, TPOT, or p99 latency.

Status: `PASS_SOURCE_DEFINED_CPP_PACKED_KV_VS_FULL_RECOMPUTE`. Local receipt SHA-256: `548ad0fd95b34eac31aa6b657d553fc97022f3181985e52b922eccc30f2eac4b`.

## Promotion boundary and next tests

- Full-model packed **CPU-BF16 / C++-Metal mixed** parity established for one short prompt. **Native all-C++ decoder graph, original `ffn_gather` scheduler pathway and true auto-regressive decode are not certified.**
- A source-boundary validator has been compiled and run independently; its checks are **not yet wired as a mandatory gate into all production model loading paths**.
- No representative task-quality evaluation, stable-model-resident end-to-end tokens/s, TTFT/TPOT/p99, incremental GPU KV peak RSS, or MLX Metal compressed MLA cache memory proof.
- Original XOX live source checkpoint identity/issuer and production promotion gates remain separate.
- GitHub code only provides a C++ header+span admission validator and entirely synthetic fail-closed tests. **No original model data** uploaded. The PR remains Draft/unmerged, and `automatic_live_promotion=false`.

Next: source-manifest admission in the real C++ loader; full `ffn_gather` shape/scheduler parity; compressed MLX-native MLA cache and true GPU-resident autoregressive decode timing; independent larger held-out evaluation before any promotion.
