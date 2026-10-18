# BEGLIN v11 — 5,181-role sealed n7, real 27-layer two-token Metal generation

Date: 2026-10-19 (Asia/Seoul)
Verdict: **PASS_REAL_FULL27_TWO_GREEDY_SEALED_N7_METAL_RECOMPUTE_SOURCE_AND_TOKEN_PARITY**
Scope: **research-only**, isolated physical EOE Apple Silicon. **DO NOT PROMOTE/DEPLOY.**

## Objective and materialization

Using the original DeepSeek-V2-Lite 4-shard BF16 checkpoint, the original BEGLIN native C qNg64 n7/group64 codec wrote 5,181 rank-2 eligible matrix bitplanes and positive F32 group scales to a separate D50 research output. 110 source tensors remain original BF16.

- Complete 5,181-role compressed payload: **14,328,299,520 bytes** (planes/scales).
- Full model including nonquantized BF16 exclusions: **15,174,228,992 bytes** (serialized theoretical/source scope only).
- Original full BF16 checkpoint payload: 31,412,968,448 bytes.
- Sealed 5,181-role manifest SHA-256:
  **c82361f7fbd73884b633bb595334805c0b7c0cf2f82ab8de26311ca78a0142b8**
- Materializer RUN_AUTHORITY SHA-256:
  **3b13bc1d98c5843f146afa428b6dfff45352f7def698d6a5a156d32d0fae136f**
- Original index SHA: d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf
- Original native C quantizer SHA: 6d76b36ef23341e7ac90ecda79541e93133e701e5538ab215e757c334526d786
- Original materializer script SHA: ca1872c6ea41861d6db97ed29010e68d2d794c500cb1909e16a5fb9e40b1acf8

After export:
1. All 5,181 role payload pairs were loaded and each planes/scales SHA recomputed on read; all passed.
2. All 5,181 original BF16 eligible tensors were independently re-encoded by native C and byte-compared to their sealed stored representations; **14,328,299,520 / 14,328,299,520 packed bytes identical**, execution exit code 0. This direct independent check covered *every* eligible role, not merely 27 sample Q projections.
3. A full 27-layer fixed Attention/shared FFN native Metal adapter smoke validated **189 sealed roles**, 378 normal physical GPU calls, 189 cache reuse hits, 125 LRU evictions, all handles released; CPU C vs Metal max BF16 relative L2 0.00012388. An independent real BF16 Q projection re-encoding check also matched all 27 layers.

## Full 27-layer original model computation

New opt-in standalone research runner:
- research/n7_multi_projection_resident_v1/run_v11_sealed_full27_recompute_v1.py
- research/n7_multi_projection_resident_v1/audit_v11_sealed_uncached_recompute_v1.py

The runner imports SHA-pinned ORIGINAL DeepSeek-V2-Lite source-defined decoder, MoEGate router, selected 64-way expert modules, RMSNorm, attention and BF16 lm_head. All eligible attention/routed/shared FFN operations that are *actually executed* load the same sealed n7 bitplanes and F32 scales through SHA-checked **read-only mmap**, and dispatch the v11 GPU-computed ownership native C++/MLX Metal kernel using bounded 256 MiB/96-handle LRU.

The run generates two model-selected token inputs and actually feeds both back into the original 27-layer model. It uses **full-context recomputation** for each new token (5-token prefill, 6-token and 7-token contexts), **NOT** MLA KV caching. Every opened packed model module is checked against independent original C qNg64 F32 matmul on first use. BF16 boundary ambiguity is admitted only when the **physical Metal F32 output** differs from C by relative L2 below 1e-5, the BF16 result is at most **one neighboring representable BF16 value (one ULP)** apart from the C-rounded result, and both GPU executions agree deterministically. The gate remains fail-closed on substantive divergence.

| Verified field | Measured |
| --- | ---: |
| Original source model layers | **27**, repeated for **3** complete contexts |
| Model-selected first generated input | **429** |
| Model-selected second generated input | **4281** |
| Next token prediction after second input | **6230** |
| Source-selected packed GPU role module instances | **6,726** |
| Native Metal model forward calls | **7,038** |
| Additional native F32 GPU oracle validation calls | **3** |
| Physical native Metal GPU calls (both types) | **7,041** |
| GPU reuse hits | **312** |
| GPU LRU evictions | **6,632** |
| Peak BF16 GPU vs C relative L2 for first-use modules | **0.000391979** |
| GPU vs C max raw F32 relative L2 on BF16 boundary cases | **1.612433e-7** |
| BF16 one-ULP rounding flips | **3** |
| Stage elapsed (5 / 6 / 7 tokens) | **74.893 / 87.242 / 94.543 seconds** |
| Whole research program runtime | **257.406 seconds** |
| Observed maximum process RSS | **1,938,374,656 bytes** |
| All GPU weight handles after final close | **0** |
| Exit code | **0** |

The first model logits agree with the previous independent 27-layer uniform n7 research logits with **relative L2 0.0** under the reported comparison. Every token prediction matches the earlier GPU-owned v11 baseline and its independent audit.

Immutable private source receipts:
- Prior v11 cached baseline SHA: **8c36149cd9b062724efe604495b5f56d14aede59d9b86883ea93757cb8676745**
- Prior v11 source independent audit SHA: **d4d05d13fe398ec01d1dc4f7e97b05805fcff8dce59d7f5f7664ab2003763132**
- New sealed all27 2-token direct execution SHA: **ae29603fec2814c5164bd7598006117a7ba776cd154243e1a215cf5b30d8263e**
- New independent source/tokens/oracle/bounds audit SHA: **fe6967402a9b6319d845abaebe6c204759413af7f4cbc238b14d933a099f498a**

The independent audit requires exact source prompt and model policy hashes, 3 complete 27-layer runs, logits for each context, [429,4281] actual input and 6230 next prediction, C/Metal numerical gates, packed role counts and handle teardown. Audit status:
**PASS_REAL_FULL27_TWO_GREEDY_SEALED_N7_METAL_RECOMPUTE_SOURCE_AND_TOKEN_PARITY**.

Original model, hidden state, expert-routing IDs per token, full logits arrays, prompts, private host paths, credential data and model weight bytes are **not uploaded to GitHub**. Only SHA-provenance and aggregate measurements are recorded here.

## What this does and does not establish

**Established:**
- Complete actual-original BF16-to-prequantized n7 model storage identity, verified independently on all 5,181 eligible roles.
- Native physical Metal source-defined 27-layer Attention/MoE/lm_head composition on *actually used* packed roles, with two real greedy outputs fed back.
- Actual n7 model role inference from sealed artifact requires **zero new native C weight quantization calls**.
- New v11 GPU-computed packed weight cache releases all handles after inference.
- Source-defined token outputs [429, 4281, 6230] match prior verified v11.

**Not established:**
- The previously tested **fused latent MLA KV cache** was **not connected in this new prequant loader run**. This is recompute, not cached-decode throughput.
- Original BF16 tensor loading for source-defined layer construction/reference still occurs. The inference code does not yet eliminate all BF16 source reads, only the dynamic C re-quantization of eligible packed roles.
- Only routed expert matrices actually used by these contexts were dispatched on Metal; the remaining 5,181-role collection is SHA/bitwise certified in storage, **not** all executed in one inference.
- Raw logits arrays were not separately proven bitwise equal for all three steps; source logits first-step relative L2 0.0 and token IDs match. There is no official 480-item task-quality benchmark.
- Current 257-second run uses 3 full-model recomputes; prior v11 464-second run includes cached MLA **and separate no-cache reference computation** and 8GiB host memo. **The timing and RSS values are not directly comparable; no speedup or memory reduction percentage against that protocol is valid.**
- Long-session memory behavior, concurrent *real model* requests, driver sleep/resume, native in-flight eviction race, p50/p95 TTFT/TPOT, full unified memory pressure and production model-serving integration remain unverified.

Release gate **NOT_READY**. Keep PR #63 Draft/unmerged, v10 fallback, no XOX serving edits, no automatic GPU/live promotion. EOE operating release observed g6-0.9.0-operator.900387, unchanged throughout this experiment.
