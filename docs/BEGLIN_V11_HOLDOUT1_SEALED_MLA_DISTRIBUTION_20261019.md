# BEGLIN v11 — 7-token original-model holdout, sealed n7, fused MLA vs uncached distribution parity

Date: 2026-10-19 (Asia/Seoul).  
Verdict: **PASS_REAL_HOLDOUT1_SEALED_MLA_SOURCE_TOP1_AND_DISTRIBUTION_PARITY** and **PASS_REAL_HOLDOUT1_TWO_COLD_CACHED_MLA_IDENTICAL_MODEL_LOGITS_AND_TOKEN_STREAM**.  
**Research candidate only. PR #63 remains Draft, not merged/deployed, no serving authority, v10 fallback preserved, no automatic GPU promotion.**

## 1. Source and independent holdout

A separate, **previously frozen** original DeepSeek-V2-Lite seven-token prompt ("A numerical method measures errors.") was tested, distinct from the five-token prompt used by the original v11 integration ("The sun rises."). Both prompts are short research fixtures, NOT an official benchmark or sealed production-quality corpus.

- Original real DeepSeek 27 decoder layers, original source MoE/router, BF16 embedding, original attention/RMSNorm, 102,400-token BF16 vocabulary head.
- Original immutable model index SHA-256: d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf.
- Full model sealed n7/group64 5,181-matrix manifest SHA-256: **c82361f7fbd73884b633bb595334805c0b7c0cf2f82ab8de26311ca78a0142b8**. Additional 110 model roles deliberately remain BF16.
- Model original uniform n7 bitwidth policy SHA-256: 39b98921c30a992f93ea896283336a6b5fe1810c5737443132f2410b474e9612.
- Original seven-token prompt SHA-256: 242e4efbf39716e8fe6842507b64638309b3fa0d139ae57b287d2c858d624d49.
- Earlier original-source uniform7 full 7 × 102,400 logits baseline npy SHA-256: **75e36edb9f696af89bf47c7720f2f124ff6a14c47eb3efaef5f2da2c95326f8c**. Its last-position first greedy token is 185.

The reader re-hashes SHA-sealed packed planes and F32 scales; a separate original source run authority SHA pins the original BF16 shards, native C compressor and exporter. The entire 5,181 matrices have independently re-quantized bitwise source parity from the preceding phase. Original BF16 checkpoint files remain unmodified and are never uploaded to GitHub.

## 2. Scope of actual model/hardware paths

The **cached** arm uses original fused GPU-owned Metal MLA with 27 persistent original n7 packed kv_b GPU handles (one per decoder layer), bounded generic packed-weight Metal LRU for selected attention/FFN/MoE linears, and **only BF16 512D normalized latent plus 64D rotated RoPE key per position/layer** persisted as KV. It runs five? No: **seven** genuine source input tokens in prefill, followed by two source-model greedy selections in real cached decode. No teacher-forced token.

The **uncached** arm re-runs all 27 original source decoder layers from the entire 7-, 8-, and 9-token context with the *same* prequant sealed n7 weights and native Metal GEMVs, computing a full 102,400-token output at every step. It does not use an MLA KV cache.

Both arms retain original BF16 source layer/expert structural reads and do **zero inference-time native C qNg64 re-quantizations**. Therefore this is NOT a shipping standalone compressed-weight-serving loader, fully GPU-resident engine, or apples-to-apples production TTFT/TPOT throughput comparison.

| Verified original source execution | GPU-owned fused MLA cache | Full-context recompute |
| --- | ---: | ---: |
| Original model layer passes | 27 × 3 stages | 27 × 3 stages |
| Real model-selected predictions | **185 → 185 → 1679** | **185 → 185 → 1679** |
| Ordinary native packed GPU Metal linears | 4,026 | 8,343 |
| Fused MLA physical GPU attention queries | 477 | n/a |
| Original source packed role instances loaded | 3,873 | 8,031 |
| Source native C re-quantizations during generation | **0** | **0** |
| Persistent BF16 latent KV after 9 positions | **279,936 bytes** | not present |
| GPU weight handles left after shutdown | **0** | **0** |
| First cold-process wall time (research end-to-end protocol) | 140.203 s | 339.785 s |
| First cold-process peak RSS | 1,913,798,656 bytes | 1,992,032,256 bytes |

The different computational work, original BF16-reference reads, weight/LRU histories and JIT/cache conditions invalidate treating this experiment as real serving speedup or memory savings.

## 3. Full vocabulary numerical audit: three exact source prediction steps

The model's **all 102,400 F32 vocabulary logits** were retained as PRIVATE, bounded append-only .npy files on EOE D50, with their SHA-256 recorded in local receipts. They were NOT checked into GitHub and must never be uploaded.

A separate read-only auditor validated each model receipt's full-vocab private-array SHA, shape [1, 102400], F32 dtype, finite output, correct greedy argmax, complete model/source authority and GPU cleanup. Predeclared fail-closed gates were **relative L2 <= 0.075** and **KL(uncached || cached) <= 0.05 nats**, plus exact Top-1 agreement at every stage. No threshold was changed after observing the results.

| Stage | Uncached token | Cached token | Full-vocab relative L2 | KL, nats | JS, nats | Top5 overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7-token prefill | 185 | 185 | 0.0117711993 | 0.0021065726 | 0.0005250276 | 4/5 |
| 8-token after consuming 185 | 185 | 185 | 0.0265395963 | 0.0019639315 | 0.0004935354 | 5/5 |
| 9-token after consuming 185 again | 1679 | 1679 | 0.0234938473 | 0.0096102676 | 0.0023615984 | 4/5 |

**Independent outcome: PASS_REAL_HOLDOUT1_SEALED_MLA_SOURCE_TOP1_AND_DISTRIBUTION_PARITY.**

Source audit receipt (append-only, private EOE) SHA-256: **43f75939b422cfeb9dace216665fa26fc276a0ed13b89e9fa2189b800bb371bc**.

## 4. Independent cold-process reproducibility

The *same* source-pinned seven-token fused MLA cached runner was executed twice in separate cold processes. Both runs completed with exit 0, zero new C quantizations, all native GPU handles released, exactly 4,026 ordinary packed Metal GEMVs, 477 fused MLA queries, and 185 → 185 → 1679.

| Independent cold GPU test | Run A | Run B |
| --- | ---: | ---: |
| Original cached run receipt SHA-256 | 361231835e46099dd5a4fa954860b75f76021a4a71d9dae23628230c9df49f07 | 3f833907da36897c7680b50d407e1a1ecb6921ac3702422b5ea655cefb3df02e |
| Total process wall seconds | 140.203 | 145.768 |
| Peak process RSS, bytes | 1,913,798,656 | 1,977,729,024 |
| Model-selected tokens | 185,185,1679 | identical |
| All 102,400 original-model logits bytes per stage | identical hashes | identical hashes |

Private numpy-file SHA-256 for each stage is IDENTICAL across these separate cold runs:
- Prefill: 5f2fffc2e0b5f7a57e141c98c87fa7dd5abfbd6697501a20d44ab7287a4af1f1
- First decode: fe2505efbda4022c3a81d7201f1c6938d91df52211a067b1501f4003c26a42f1
- Second decode: 5c9cc6adf47637807021f7fb9b6377e1031bab9df81ce5a

These are **byte-identical full vocabulary model logits within the NEW cached arm across two executions**, NOT proof of byte-identical logits versus original v11 or uncached arm.

Separate cold replay independent audit SHA-256: **3c8a299ec24a8baabb04c90e7891689e5a69c47810c5058d9bced786344bd606**.

## 5. Code and CI evidence

New research scripts added to Draft PR #63, existing source unchanged:
- research/n7_multi_projection_resident_v1/run_v11_sealed_full27_cached_mla_holdout1_v1.py (SHA-256: f5129ad58a841e982cfdd94ddbf3c29797b9ff3e6534eb909c75c71d94c08c7a)
- research/n7_multi_projection_resident_v1/run_v11_sealed_full27_recompute_holdout1_v1.py (SHA-256: f4aa596dd29cbcfef928bf53ee51b0382cf7de94e5839c02fde237051871690c)
- research/n7_multi_projection_resident_v1/audit_v11_sealed_holdout1_cached_vs_uncached_v1.py (SHA-256: c1942dbc4814b97137ac60421f451ca917754ec83f6000ccec1359beb2734261)
- research/n7_multi_projection_resident_v1/audit_v11_sealed_holdout1_cold_replay_v1.py (SHA-256: 3fa9a08ccb4fdf65e4b353749a2e014bebb58d590501d7adf75d16044e29dfbe)
- research/n7_multi_projection_resident_v1/test_v11_holdout1_audit_contract_v1.py (9 CPU-only adversarial and deterministic full-vocab tests PASS on physical EOE).

GitHub dedicated prequant workflow: Ubuntu Python 3.11 and 3.13, 15 sealed-mmap negative tests plus 9 full-vocabulary audit negative/mathematical tests, plus new source compile checks. These GitHub jobs do not have access to original model payloads or Apple Metal; all actual model tests were performed on isolated EOE hardware.

## 6. Outstanding blockers — do NOT imply serving ready

1. Only **two short, previously authored original-model prompts** tested. Need a larger frozen, independent held-out corpus and meaningful NLL/quality metrics beyond deterministic output compatibility.
2. End-to-end vDSP serving C++ prequant loader and model-authority path are NOT integrated; original BF16 source structural/model-reference rereads remain.
3. No native serving TTFT/TPOT/tokens-per-second p50/p95/p99 under a matched warm/cold controlled run and no full-system unified-memory/allocator comparative experiment.
4. No sustained multi-hour/model-level multi-request concurrent GPU eviction and stale-handle race test, restart/sleep-resume and crash recovery matrix.
5. No source-attested original XOX shipping backend authority and independent release qualification. Preserve v10 fallback, keep PR #63 Draft and disable automatic live GPU promotion.

**Verdict:** research-verified physical original-model 5,181 n7 prequant + GPU-owned fused MLA cached 27-layer two-token generation across two distinct short source prompts, and held-out full-vocab distribution + cold-reproducibility evidence. NOT_READY_FOR_SERVING.
