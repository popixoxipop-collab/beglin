# BEGLIN — original-source 27L dual GPU + SHA-addressed qNg64 host memo (2026-10-19)

**Decision: VERIFIED ISOLATED RESEARCH; DRAFT ONLY.** This extends the prior DeepSeek-V2-Lite original-model 27-layer / two-generated-token test. The existing research system has single-fused qNg64 Metal MLA attention, 512+64-dimensional BF16 latent KV and a 256 MiB bounded GPU qNg64 general-MoE role cache. The new component is an opt-in **6 GiB CPU packed-weight memo** using SHA-256 of original F32 weight content to avoid repeated n7/group64 native C quantization.

**No model/tensor payloads, hidden states, source prompts, passwords or private machine paths are committed.** D50 BF16 original source remains READ ONLY. XOX serving, release activation and automatic live promotion were not modified.

## Real 27-layer model and numerical evidence

The actual original model selected and consumed greedy tokens **429 → 4281**; it predicted 6230 next, but did not consume 6230. All 27 layers ran for each of prefill, first decode and second decode. The independent no-cache original-model comparison retained last-token logits relative L2 **0.00862746973559613**, exactly the same as the previous no-host-memo run, with unchanged next-token top1. The GPU fused MLA path performed 371 physical attention calls and zero expanded per-head K/V reconstructions.

Real BF16 normalized latent+rotated RoPE KV tensor storage: **217,728 bytes** for 7 tokens/27 layers versus **1,935,360 bytes** expanded BF16, or 88.75% saved *only in persistent KV tensors*. This is not total process memory saving.

## Measured original-model research comparison

| Measure | 27L dual GPU, before host memo | 27L dual GPU + host memo |
| --- | ---: | ---: |
| Original checkpoint source prefill + packing | 264.411 s | 275.737 s |
| First generated-token stage incl. independent no-cache recompute | 280.890 s | **115.704 s** |
| Second generated-token stage incl. independent no-cache recompute | 293.216 s | 305.833 s |
| Whole source-attested research program | 839.196 s | **698.151 s** |
| Peak complete process RSS | 2,092,007,424 B (1.95 GiB) | **9,140,600,832 B (8.51 GiB)** |
| Last logits relative L2 | 0.00862746973559613 | **unchanged** |

Observed **16.81% shorter overall research wall time** for one controlled protocol pair. First decode was 58.81% shorter; prefill and second decode were each slower. This benchmark **includes original source BF16 reads and independent verification** and therefore does not certify TPOT, TTFT, serving TPS or a production performance improvement.

Host native n7 quantizer/compact cache:
- **6,726** source n7 packing requests from original model.
- **4,853** actual native C quantizations, **1,873 (27.85%)** SHA-content memo hits that skipped n7 C packing.
- **6 GiB** RAM cache limit; measured cached uint8 planes+FP32 scales peak **6,442,352,640 B**. 2,563 compact LRU evictions evicted 7,346,012,160 cumulative bytes.
- Measured cumulative original C packing time **453.155 s** across misses and SHA hashing time **33.274 s** across all requests. These substage durations **must not** be subtracted from end-to-end to imply a production TPS gain.
- Every original BF16 source read was still performed. Compact buffers are immutable and keyed by exact original F32 source byte SHA-256 plus rows, columns, n7 bits; original `Native.bind` restored on close.

GPU memory and routing source authority:
- Original fused kv_b direct GPU roles: 27 original qNg64 n7 registered handles, **53,084,160 B**, all freed.
- Original generic qNg64 native GPU role LRU: 256 MiB / 96 handles; **2,367 distinct source roles**, 8,289 physical GEMV calls, **1,566 hits, 6,723 registrations, 6,629 evictions**; CPU compact source buffers retained for safety.
- Native C and original compiled C++/Metal numerical oracles remained active for newly created original-model linear modules. No dense weight fallback.

## Independent private evidence receipts (SHA-256)

| Scope | SHA-256 |
| --- | --- |
| Prior source dual GPU/no-memo result | `aa5ab53a128fdcadc479e54c893d6f29acbe301a4ff68b6eddf1e042d38fb00f` |
| Physical 27L dual GPU + host-memo research result | **`d3aa9327d250d4acfe7243e763ceb8baaba5a7d527b555bdcd1e92141c48cd7e`** |
| Independent source/bytes/token/numerical audit PASS | **`0559c5de5c97b4c5fe247e95a81457707c2ae70c32993ff4fd3f5fdd4d441cc7`** |
| Research driver (kept private on EOE) | `0771fbe60347f40942e7d4b4a4d70fd1100095e89ef346925b64a14952dc6cc7` |
| Portable generic host qNg64 memo source | `e8a90850b09cad67a3f1a6df601267c40914e2c0170d898400a0def56ac2e4d0` |

## Portable source and QA changes

Within `research/n7_multi_projection_resident_v1/`:
- `research_n7_source_pack_memo_v1.py` — opt-in, capacity-bounded source Native.bind memo with F32 byte SHA verification, immutable n7 bitplanes/scales, eviction, source restoring on shutdown.
- `test_n7_host_packed_memo_portable_v1.py` — no original model; real BEGLIN native-C quantizer/matmul tests including content hits, precision/shape rejection, 4 MiB eviction and deterministic original-source repack.
- `build_and_test.sh` — runs existing physical Apple Silicon Metal multi-matrix test **plus** original-C CPU SHA memo regression. Creates a unique per-run QA receipt directory to preserve immutable previous receipts on repeated runs.
- `test_multi_gpu_n7_v10_portable.py` — accepts new `BEGLIN_TEST_RECEIPTS_DIR` for isolated QA output, never silently overwrites earlier receipts.

These portable tests were rebuilt and rerun on physical EOE Apple Silicon. Six GPU-resident synthetic n7 matrices passed independent C parity with max relative L2 **1.2930364961681175e-7**; original native C host memo test passed **6 true quantizations, 1 SHA hit and 3 forced LRU evictions**. No real model checkpoint is shipped or accessed by this package.

## What remains

1. Evaluate bounded host memo policy/budget without unbounded 6 GiB cache memory overhead; repeat source-attested experiments with improved admission/eviction rather than assume larger cache is better. The second sampled-token stage is already slower with this current policy.
2. Removing repeated original BF16 source checkpoint reads requires an authenticated **prepacked artifact store** with frozen SHA provenance, atomic writes, on-disk format and resume verification; not established by RAM memo.
3. Verify longer prospective sampled token traces, independent quality holdouts, model-level prequantized residency, exact CPU+GPU peak usage and TTFT/TPOT p50/p95 under actual serving conditions.
4. Keep PR DRAFT and GPU auto-promotion OFF. Neither full native C++ inference nor production serving throughput is certified by these research results.
