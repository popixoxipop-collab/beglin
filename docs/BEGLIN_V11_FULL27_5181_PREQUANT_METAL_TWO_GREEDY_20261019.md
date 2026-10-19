# BEGLIN v11 — Full 5,181-role sealed prequant + real Metal 27L decoding (2026-10-19)

**Research PASS; production NOT_READY.** PR #63 intentionally remains Draft; no XOX serving, BEGLIN production C++ path, model source, or Tailnet Commander release was modified.

## 1. Original source and whole-model serialized contract

- DeepSeek-V2-Lite original BF16 safetensors: 5,291 tensors, 27 decoder layers, 64 MoE routed experts with top-6 routing; 5,181 qNg64 n7/group64 eligible matrices plus 110 BF16 exclusions.
- Exact whole original BF16 serialized payload: 31,412,968,448 bytes.
- Actual n7 packed *eligible* matrices on D50: **14,328,299,520 bytes**.
- Full model theoretical n7 packed + 110 BF16 exclusions: **15,174,228,992 bytes**, ~51.69% below BF16 serialized source. Not actual inference GPU memory usage.
- Checkpoint index SHA256: d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf
- Frozen native source-boundary TSV SHA256: 274247e626f67579e56256e356f554fd1b1e5e5967df2603ec82bfb902af4963
- Frozen uniform7 precision policy SHA256: 39b98921c30a992f93ea896283336a6b5fe1810c5737443132f2410b474e9612
- Original BF16 whole-file shard hashes #1..#4: a314b79a30810b1038f9db08345178e16fb5742aa266c4a3dc05054c59800082; b0bd088feada0af954614da3a5b179c4d4d5d8bb2482ec83194d101f3fe91796; d31b192e488f0b3a04e007d842c3c9ba5783d944cb1cc27402fe8b8bb837c490; 304499be0ea06bf8a3937d7a3cc515bc3738dbd288bdd0bbc1b2d3560fce4930.
- Independently recompiled native C++ source manifest validator PASS for all 5,291. Original research copy and independent 27L teacher checkpoint shard SHA equal 4/4.

## 2. Real source export and independent artifact audit

The optional resumable original-C quantizer materialized the complete 5,181 original-source qNg64 n7 matrix payloads to an isolated D50 SSD research stage in 598.938 seconds (resource-reserved job job_a8be54ec9d17f0227d2d611aed509b28; SUCCEEDED, exit 0). Each matrix has a committed individual SHA/size/source/native-C receipt. Model pack was sealed only after completing all 5,181 source roles.

- **Full sealed pack manifest SHA256: c82361f7fbd73884b633bb595334805c0b7c0cf2f82ab8de26311ca78a0142b8**
- Independent source+artifact audit confirmed all 5,181 qNg64 payload hashes and sizes, full BF16 source shard SHA256, and bit-for-bit native C re-quantization parity of the 27 original Q projection matrices.
- Independent audit status: PASS_REAL_FULL27_5181_SEALED_ALL_HASHES_27_INDEPENDENT_C_REQUANT.
- Independent artifact audit receipt SHA256: **c98140a99a327bd97d90e26cb8cfd27b6957b32d504b6e8bab709c53deef5d6d**.
- BF16 source and packed model payload are local research artifacts; **do not upload** checkpoint, hidden states, model tensors, token context or secrets to GitHub.

## 3. Full real 27-layer C++/Metal MLA greedy two-token model execution

A SHA-gated opt-in overlay patches the exact prior EOE private 742-line original real v11 decoder **in memory only** (original source SHA256 defae1ecb7254e5716b87a0d5b0c99964b0d61d727c708f2942df5010c64eb46). Existing model source and hardware C++/Metal functions remain unchanged. The packed source is loaded on-demand from the SHA-sealed 5,181-role manifest, skipping on-the-fly C re-quantization. The separate CPU BF16 model reference *still* reads original BF16 weights.

| Measured single research run | Old v11 8GiB host memo | v11 sealed prequant |
|---|---:|---:|
| Actual selected/embedded input tokens | 429, 4281 | **429, 4281** |
| Next predicted token | 6230 | **6230** |
| Physical GPU generic packed linears | 8,289 | **8,289** |
| GPU LRU hits / evictions | 1,566 / 6,629 | **1,566 / 6,629** |
| Native C quantize executions in inference | 2,394 | **0** |
| Final compact CPU SHA memo after cleanup | 0 bytes | **0 bytes** |
| BF16 latent MLA KV storage | 217,728 bytes | **217,728 bytes** |
| Second token cached vs uncached logits relative L2 | 0.00862746973559613 | **same** |
| General GPU and fused MLA handles released | PASS | **PASS** |
| Full research process wall time | 464.1193 sec | **225.5128 sec** |
| Peak process RSS | 9,133,473,792 bytes | **2,281,324,544 bytes** |

Observed one-run experimental difference: wall time ~51.41% shorter (~2.06x for this entire research protocol) and peak process RSS ~75.02% smaller. **Not production TTFT, TPOT, TPS or statistically guaranteed memory savings**; cache/swap conditions are uncontrolled, full FFN GPU weights are LRU on-demand, and BF16 original reference still loads checkpoint data. The full sealed model is not wholly GPU resident.

The v11 sealed original-source 27L model called 8,289 actual packed GEMVs, loaded 6,726 role instances from the sealed full 5,181-role manifest, finished two greedy cached decode steps and released every GPU handle. No original model or XOX serving change occurred.

- Full experiment receipt SHA256: **1d60bccdb4ac6080d9be5a0d5fbe1c4c10c3d7551c3d090069ab36390915ae2e**.
- Independent paired audit SHA256: **b90c9e3569a11f8c22a4b617ef3a5f2330652faab7a76a39203849b2f64a40d9**.
- Paired independent audit checks source/C++/precision identities, 429→4281→6230, aggregate logits relative L2/max absolute errors, 0 quantizations, GPU call/eviction counts and GPU cleanup. **Full logits arrays byte-identical were NOT verified**, only aggregate error metrics and selected tokens.

## 4. Cold-process reproducibility

Initial full sealed inference: 225.5128 sec, peak RSS 2,281,324,544 B.
First independent cold-process full27 replay: 232.4134 sec, peak RSS 2,283,945,984 B, genuine token selection 429→4281→6230, 8,289 GPU calls, zero C quantizations and clean GPU teardown.
Cold replay A SHA256: ccf58c6521ecac4b258065dbff8ca879e374884e261b20ef870fdb3498915f89.

Second independent cold-process full27 replay: **214.2065 sec**, peak RSS **2,281,062,400 B**, tokens 429→4281→6230, original C re-quant 0, 8,289 physical packed GEMVs, all GPU handles released. Its append-only receipt SHA256:
**7a8d5bd32b840da6bd8fe2cfa40fd584b2ccadf521384dda9c78c6d5700cc072**.

A separate auditor reopened all **three** cold-process receipts and verified source/model/precision/kernel SHAs, every layer count, tokens, exact scalar relative L2/max abs logits metrics, no C re-quantization, identical Metal call/eviction counts and clean GPU lifetime across processes.

Status: **PASS_THREE_COLD_FULL27_PREQUANT_REAL_TWO_GREEDY_IDENTICAL_METRICS**.

| Full research protocol, individual cold process | Total time | Peak process RSS |
| --- | ---: | ---: |
| Initial | 225.5128 s | 2,281,324,544 B |
| Replay A | 232.4134 s | 2,283,945,984 B |
| Replay B | 214.2065 s | 2,281,062,400 B |
| Median (n=3) | **225.5128 s** | **2,281,324,544 B** |

Three-run SHA receipt: **4fc523b6538253ff4d0f19b850b9806dcc02afdea47202ccffa619c813c1ae0e**.

The original old-v11 reference had only one matched research run (464.1193 sec, 9,133,473,792B peak RSS). Hence the three-process spread strengthens new-v11 reproducibility but **is not** a repeated matched A/B, statistical confidence interval, official throughput/TTFT/TPOT or p95 claim. Full logits arrays were not compared byte-for-byte and only one original-model prompt was repeatedly decoded.

## 5. Delivered code, CI and operational scope

- research/n7_multi_projection_resident_v1/full27_sealed_materializer_v1.py
- research/n7_multi_projection_resident_v1/prequant_n7_sealed_v1.py
- research/n7_multi_projection_resident_v1/prequant_full27_execution_adapter_v1.py
- research/n7_multi_projection_resident_v1/run_original_full27_v11_sealed_overlay_v1.py
- research/n7_multi_projection_resident_v1/audit_full27_sealed_prequant_v1.py
- research/n7_multi_projection_resident_v1/audit_v11_sealed_full27_two_greedy_v1.py
- research/n7_multi_projection_resident_v1/audit_v11_sealed_three_cold_full27_v1.py
- .github/workflows/beglin-v11-prequant-sealed.yml

The GitHub Ubuntu Python 3.11/3.13 CI covers 15 SHA/loader tamper tests and source compilation, **not** real source/model weights or physical Metal, which were checked privately on EOE.

## 6. Still required before serving

(1) Multiple held-out prompts, accurate NLL and quality vs BF16 and v10;
(2) serving loader eliminating independent original BF16 rereads from hot inference path;
(3) actual end-to-end TTFT, TPOT p50/p95/p99, tokens/s, system unified memory/MLX allocation;
(4) sustained multi-hour memory/leak and real concurrent in-flight GPU operations, eviction races, sleep/resume and crash/restart recovery;
(5) original XOX source attestation and reproducible ship-ready C++ serving binary.

Maintain fallback v10, Draft PR #63, no merge, no live deployment, no automatic GPU promotion.
