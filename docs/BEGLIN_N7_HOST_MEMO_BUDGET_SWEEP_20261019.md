# BEGLIN: real DeepSeek-V2-Lite 27L SHA n7 packed host memo 0/6/8 GiB A/B (2026-10-19)

**Status: PASS — full source-attested real-model hardware experiments and separate independent auditors; Draft PR only.** This is NOT an original XOX serving patch, a complete all-native model or a proven production TTFT/TPOT gain.

## Main engineering conclusion

The **6 GiB host compact-source cache** was below the physical source's compressed live working set, provoking cache-thrash. The 8 GiB cap held all distinct n7 bitplane/FP32 scale arrays for this original model and prompt, eliminating all **CPU host compact-cache evictions**. It did **not** change the 256 MiB generic device GPU cache or its 6,629 GPU LRU evictions.

The frozen n7 policy and original experiment's **2,367 distinct generic GPU source roles** imply **735 distinct routed experts** after subtracting 27×6 fixed non-kv_b roles. Every one of the original 1,664 possible expert records has the same three packed-byte role geometry. Including seven fixed original roles per 27 layers, the compressed bytes for this recorded working set are **6,794,035,200 bytes**. The actual 8 GiB run retained exactly this amount with **zero CPU LRU evictions**, independently matching the derived estimate.

### Physical same-protocol comparison

| Source-attested original DeepSeek27 run | No host memo | 6 GiB host memo | 8 GiB host memo |
| --- | ---: | ---: | ---: |
| Original input token length | 5 | 5 | 5 |
| Actually self-selected and fed-back greedy tokens | 429, 4281 | 429, 4281 | 429, 4281 |
| Next token *predicted, not consumed* | 6230 | 6230 | 6230 |
| n7 C quantizations performed | No separate count; 6,726 compressed module creations | 4,853 | **2,394** |
| SHA source-content cache hits avoiding original C packing | 0 | 1,873 | **4,332** |
| Original source n7 host compact-cache evictions | N/A | 2,563 | **0** |
| Maximum measured original host n7 cached bitplanes/scales bytes | N/A | 6,442,352,640 B | **6,794,035,200 B** |
| Prefill including source BF16 reads and C packing | 264.411 s | 275.737 s | 274.968 s |
| First generated-token step including independent uncached recomputation | 280.890 s | 115.704 s | **98.796 s** |
| Second generated-token step including independent uncached recomputation | 293.216 s | 305.833 s | **89.169 s** |
| Whole research program wall | **839.196 s** | 698.151 s | **463.668 s** |
| Peak process RSS | 2,092,007,424 B | 9,140,600,832 B | 9,315,319,808 B |
| Final cached vs uncached BF16 logits relative L2 | 0.00862746973559613 | Same | **Same** |

**Observed full-program research wall reductions:** 6 GiB vs no memo **16.81%**, 8 GiB vs 6 GiB **33.59%**, 8 GiB vs no memo **44.75%**. This is one physically measured execution per fixed protocol, not randomized trials or serving throughput.

The 8 GiB memo performed exactly **2,394 first-time native C quantizations** for 6,726 requests and skipped **4,332 (64.41%)** repeated source quantizations. The independently measured C quantization time over misses was **224.169 s** and SHA-256 hashing time across requests was **33.479 s**. Existing original BF16 source file reads still occur every decoder stage.

The **8 GiB configured cap did not imply 8 GiB allocated**: only 6.794 GB was actually needed, just ~352 MB above the previous 6 GiB cap. The additional resident process memory over the 6 GiB experiment was ~175 MB even though the second decode greatly accelerated.

### Other verified subsystem invariants

- Original source is a SHA-pinned BF16 DeepSeek-V2-Lite checkpoint, 27 decoder layers, 64 routed experts, top-6 selection; model and pre-existing policies were not rewritten.
- Candidate **single-fused C++ qNg64 Metal MLA attention** directly uses compressed n7 kv_b. The BF16 latent/RoPE KV data is **217,728 B** for 7 tokens across 27 layers versus original expanded **1,935,360 B** (88.75% persistent KV tensor saving). No per-head expanded K/V is stored in the candidate.
- Original model's 27 GPU-resident fused kv_b matrices occupy **53,084,160 B** of compact device data and were all released.
- The separate generic non-kv_b Metal n7 cache remains bounded to **256 MiB/96 GPU handles** and underwent **6,723 registrations, 1,566 hits, 6,629 GPU evictions** over 8,289 GEMV calls. The large CPU memo improves C quantization avoidance, *not* GPU residency hit rate or unified device memory allocation.
- CPU compact bitplane/scale buffers remain immutable and are CPU+GPU dual-backed to prevent prior discovered asynchronous borrowed-memory lifetime corruption.
- **Only the original sampled 5+2-token research prompt** was tested; no general task quality, long-context, concurrency, p95/TPOT/TTFT or real serving benchmarking.

### Authoritative private EOE receipt hashes (SHA-256)

| Experiment or independent proof | SHA-256 |
| --- | --- |
| No-host-memo baseline | `aa5ab53a128fdcadc479e54c893d6f29acbe301a4ff68b6eddf1e042d38fb00f` |
| 6 GiB host memo original-model result | `d3aa9327d250d4acfe7243e763ceb8baaba5a7d527b555bdcd1e92141c48cd7e` |
| 6 GiB independent source auditor | `0559c5de5c97b4c5fe247e95a81457707c2ae70c32993ff4fd3f5fdd4d441cc7` |
| **8 GiB host memo original-model result** | **`79cd095f4dfe69503a85fb7f37ba69495452067b11659d15864e24568a6a55ee`** |
| **8 GiB independent source auditor** | **`dace0352de4638bf088c0ffe982c17b3dc4d618dfaa88598b86b59a4de530cfb`** |
| **Cross-trial policy/working-set/numerical comparison auditor** | **`2a9071f5202d936c7dbffdbfcd887604396a04618ddf859203fd0c285fadb252`** |
| Original source-bench 8 GiB research driver; kept private on EOE | `4b4f90b18728c53b763efa051f3600c8f83caa89e9af89ef8f2cf07df40173f9` |

No original model checkpoint tensors, prompt text, expert routing ID membership or logits arrays are copied to this repository. Only aggregate evidence, SHA digests and portable synthetic source are approved.

## Decision and next certified gates

**For this measured 27L real-model research protocol, 8 GiB is the verified zero-eviction host cache cap.** A calculated 7 GiB cap also exceeds the observed unique packed working set but **was not physically certified** here. Do not hardcode the present budget across different prompts or dynamic expert working sets without checking available system memory and admission policy.

1. Derive host memo budgets from the immutable model policy and observed routed expert live-set, maintaining headroom and memory-pressure caps; verify additional independent prompts with changed expert routing.
2. Replace per-token original BF16 source reads/repacking with a **sealed on-disk prepacked n7 artifact** only after source-tensor SHA, checksum/policy, atomic write, restart recovery and private-backing lifetime tests.
3. Add genuine prequantized resident multi-token serving benchmark with independent task/quality holdouts, TTFT/TPOT p50/p95, GPU/CPU peak RSS including dual backing, and model-level queue/restart correctness. Avoid extrapolating the benchmark's independent uncached verification time into generated tokens/s.
4. Leave current Draft PR unmerged and automatic promotion disabled; XOX serving and the EOE immutable release plane are unchanged.
