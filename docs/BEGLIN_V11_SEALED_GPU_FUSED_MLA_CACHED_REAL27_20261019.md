# BEGLIN v11 — actual full27 sealed prequant + GPU-owned fused MLA KV cached greedy

Date: 2026-10-19 Asia/Seoul
Status: **PASS_REAL_FULL27_SEALED_PREQUANT_GPU_FUSED_MLA_TWO_GREEDY_SOURCE_PARITY**
Safety: **RESEARCH-ONLY / NOT_READY FOR PRODUCTION.** Draft PR #63, no merge, no runtime or XOX serving modifications, automatic GPU promotion remains disabled. Keep v10/v11 fallback.

## 1. Model and source authority

- Original real DeepSeek-V2-Lite 27-layer BF16 model, source-defined attention, RMSNorm, shared and 64 routed experts, top-6 MoE router and original BF16 vocabulary head.
- Original checkpoint index SHA-256: d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf
- Original pinned source shards SHA-256, in order: a314b79a30810b1038f9db08345178e16fb5742aa266c4a3dc05054c59800082 / b0bd088feada0af954614da3a5b179c4d4d5d8bb2482ec83194d101f3fe91796 / d31b192e488f0b3a04e007d842c3c9ba5783d944cb1cc27402fe8b8bb837c490 / 304499be0ea06bf8a3937d7a3cc515bc3738dbd288bdd0bbc1b2d3560fce4930
- Full model source: 5,291 tensors, of which 5,181 eligible qNg64 n7/group64 and 110 remain BF16. No policy changes.
- Original uniform7 policy SHA-256: 39b98921c30a992f93ea896283336a6b5fe1810c5737443132f2410b474e9612
- Complete read-only sealed n7 model manifest SHA-256: **c82361f7fbd73884b633bb595334805c0b7c0cf2f82ab8de26311ca78a0142b8**
- SHA-pinned original source, shard, exporter and native C authority receipt SHA-256: 3b13bc1d98c5843f146afa428b6dfff45352f7def698d6a5a156d32d0fae136f
- Verified 5,181 original BF16 -> qNg64 compressed payloads are bitwise identical to independently repeated native-C quantization; qNg64 eligible payload totals 14,328,299,520 bytes. Noneligible BF16 bytes remain additional. These serialized bytes are NOT memory usage.

## 2. New genuine cached model execution

The new independent, original-source-pinned research runner is:

research/n7_multi_projection_resident_v1/run_v11_sealed_full27_cached_mla_v1.py

Its SHA-256 on the audited EOE research checkout: **e15aad397849769bb29af521a4c03c366552bdf12551976071790beff6890057**.
It imports the SHA-pinned existing original source-defined DeepSeek research modules, uses the full 5,181-role sealed weights via mmap for all selected packed linears, and executes those linears through the physical C++/MLX/Metal v11 GPU-computed weight-owner registry.

The preexisting, independently source-pinned native GPU-owned fused MLA module registers each layer's packed kv_b projection exactly once (27 resident handles) and reuses it for all actual decode steps and routed-expert preview forks. The cache retains only BF16 512D normalized latent + 64D rotated RoPE key per token per layer. Original expanded per-head K/V is not persisted. Other attention/FFN/expert qNg64 packed roles are admitted to a bounded 256 MiB / 96-handle LRU GPU registry. The model still loads original BF16 layers / source structures and selected experts for original reference compatibility; do not claim BF16 source reads eliminated or fully GPU-resident weights.

One real five-token prefill, then two greedy model-selected tokens consumed by the SAME persistent MLA caches for 27 layers, completed with the exact original token sequence:

**prefill selects 429; decode consuming 429 selects 4281; decode consuming 4281 predicts 6230.**

No externally imposed teacher-forced token, no extra/replaced source router, no on-the-fly native C compressed weight generation. Original input/prompt text and all model tensors stay private on EOE D50.

## 3. Hardware and numerical evidence

| Measurement, each cold isolated process | Cold process A | Cold process B |
| --- | ---: | ---: |
| Full original decoder layer count in each phase | 27 | 27 |
| Model-selected token inputs | 429, 4281 | 429, 4281 |
| Prediction after second cached token | 6230 | 6230 |
| Ordinary actual Metal compressed GEMVs | 3,525 | 3,525 |
| Fused GPU MLA queries, including router previews | 371 | 371 |
| Device-resident original kv_b layer handles | 27 | 27 |
| kv_b compressed resident byte payload | 53,084,160 | 53,084,160 |
| Actual generic GPU LRU reuse / evictions | 234 / 3,207 | 234 / 3,207 |
| Model compressed role module instances opened | 3,372 | 3,372 |
| Ordinary GPU C-oracle verified role module instances | 3,291 | 3,291 |
| Inference-time original native C re-quantizations | **0** | **0** |
| Persisted 7-token 27-layer BF16 latent KV bytes | **217,728** | **217,728** |
| BF16 1-ULP rounding ambiguous transitions | 1 | 1 |
| Max FP32 native C-vs-GPU relative L2 on BF16 ambiguity | 1.0239116628600527e-7 | same |
| Full research process wall time | 117.000 s | 113.153 s |
| Peak process RSS | 1,860,435,968 B | 1,864,007,680 B |
| GPU handles after cleanup; source cache layer references | **all cleaned** | **all cleaned** |

The full source-derived first prefill logits vs the historical independent uniform7 27L reference had relative L2 0.014482878254655093 in both cold processes and selected original token 429. This is source-compatible fused MLA numerical variation, **not bitwise identical logits**.

More importantly, the new full vocabulary logits hashes for the two subsequent cached decode outputs were EXACTLY the same across both cold processes:

- Decode consuming token 429, predicting 4281: e7f5a625b24fa83ff2174d6b6cddb087cca22278b67427f4b811c410895b3bfb
- Decode consuming token 4281, predicting 6230: db87161e9f5b594856101d11c6f120f5f96992e9cd7089c12376daef35559185

The old v11 cached result and independent new v11 uncached model also produced 429 → 4281 → 6230. Full old-v11-vs-new-v11 logits byte identity was **not** verified. The native FP32 oracle and BF16 one-ULP handling is bounded and does not relax to unconstrained tolerance.

## 4. Immutable private evidence receipts

- Cold cached full27 A receipt SHA-256: 165460894349ed00efbaa7ce96a276a6772d7e9bfa82c89e90c61bc244f21081
- Cold cached full27 B receipt SHA-256: 2287e590049ea0b3e3816700d6055e08896c7d6e6f1f526a6b2d178696f9b637
- Original v11 real 27-layer cached baseline receipt SHA-256: 8c36149cd9b062724efe604495b5f56d14aede59d9b86883ea93757cb8676745
- New 27-layer sealed full-sequence recompute receipt SHA-256: ae29603fec2814c5164bd7598006117a7ba776cd154243e1a215cf5b30d8263e
- New independent cached fused-MLA source/token/cleanup audit receipt SHA-256: **2f23d67da400368ce12487189b355d0a5404062dd044c04ea7ab7abf12f674dc**
- Independent auditor source: research/n7_multi_projection_resident_v1/audit_v11_sealed_cached_mla_independent_v1.py (SHA-256 **1372770216f56344c51cbcfa89f58460c9cda070f6954867759467d417a664a3**).

All these receipts and private original model binary/weights stay on EOE. Only non-sensitive hashes, dimensions, statuses and scope declarations are in this GitHub document.

## 5. What is NOT proven / release blockers

1. Production shipping C++ model prequant loader is NOT integrated into actual XOX vDSP inference serving; this is a standalone EOE research Python orchestration using real native C++ GPU operators.
2. The BF16 original model structural/source/reference reads remain, even though compressed eligible role weights are loaded from mmap and native-C re-quantization is 0. A dedicated serving loader can cut those rereads, but no such production memory or TTFT claim is made.
3. Only one short authored prompt and two cold process repetitions have been checked with true fused cached continuation. Full-vocab logits hashes are identical across new runs, but NOT compared bitwise to original v11 cached arrays.
4. No multi-hour pressure/restart/sleep-resume, multi-request concurrency, GPU in-flight eviction race safety or driver/MLX matrix.
5. No matched A/B p50/p95/p99 TTFT, TPOT, end-to-end tokens/s, full unified memory footprint or official frozen held-out NLL/quality.
6. Frozen source XOX serving binary authority and owner-reviewed promotion gates are still absent. Do not merge/deploy v11, remove fallback v10, or auto-promote GPUs.

For scope, the earlier v11 8GiB host memo research pipeline ran once at 464.119 s / 9.133 GiB RSS. New direct cached runner did not include the same independent full-recomputation work in its timed path. Therefore apparent timing/RSS deltas are NOT apples-to-apples speedup/total-memory claims.

**Conclusion:** source-attested 5,181-matrix prequant loading + true full27 fused GPU-owned MLA latent cache + two model-selected greedy decoding steps are verified in an isolated research environment; production readiness remains NOT_READY.
