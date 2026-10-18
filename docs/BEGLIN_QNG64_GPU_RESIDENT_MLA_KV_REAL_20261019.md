# BEGLIN GPU-resident qNg64 MLA/KV: original-source 27-layer validation (2026-10-19)

**Verdict: EOE isolated physical Metal PASS; Draft PR #63 only.** No original XOX model, serving process, release, Supabase data or automatic model promotion was modified. Original DeepSeek-V2-Lite BF16 checkpoint data remained on a private, SHA-attested read-only research copy.

## Implemented

The original compiled qNg64 n7/group64 compressed C++/Metal MLA attention kernel is extended with a process-local registry of **GPU-owned compressed kv_b planes and FP32 group scales**. For each actual original DeepSeek-V2-Lite decoder layer, the 4096 x 512 `kv_b_proj` is registered **once**. The same device handle is reused during the five-token causal prefill and both actual greedy-selected decode steps; SHA of the effective compressed source operator is verified after weight-source reloading. Only 512D normalized BF16 latent and 64D rotated key are retained per token. The candidate Metal attention never reconstructs per-head expanded historical K/V or creates a dense dequantized kv_b GPU matrix.

The original source-defined BF16 attention and routed MoE are separately executed for *independent numerical evaluation*; they do create expanded K/V in the comparison arm. All OTHER model linear roles and selected experts remain source-streamed/repacked per research step. This is not a full native-C++ or fully GPU-resident model.

## Physical real-model result and independent audit

| Evidence | Observed |
| --- | ---: |
| Original DeepSeek-V2-Lite decoder layers | **27 / 27** |
| Original qNg64 device-weight registrations | **27 total; once per layer** |
| GPU-owned compressed kv_b weights + scales | **53,084,160 bytes** |
| MLX active GPU memory while all 27 registered, baseline zero | **53,084,160 bytes** |
| Actual physical fused Metal calls using resident source handles | **371** |
| Original-model tokens sampled **and actually fed back** | **429, 4281** |
| Next token predicted but **not** fed back | **6230** |
| Second generated token: cached vs independent full-recompute logits relative L2 | **0.00862746973559613** |
| Cached vs full-recompute next top-1 | **Identical** |
| Original-source compressed projection instances across three phases, incl. paired reference | **6,726** |
| Real 7-token / 27-layer persistent BF16 latent KV | **217,728 B** |
| Equivalent expanded BF16 original per-head KV | **1,935,360 B** |
| Actual persistent KV tensor bytes reduced | **88.75%** |
| Candidate historical expanded K/V rehydrations | **0** |
| Native GPU device handles explicitly released at end | **All 27** |
| Whole source-streamed research run, incl. multiple no-cache references | **842.90 s** |

Frozen source index SHA-256: `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`.

Private EOE experiment receipt SHA-256: `47043a8573ebd06abc689249343d2a61d1ef23a25a107d1579da7235b064ec83`.

Independent audit receipt SHA-256: `6dfd8c2d4c016ceef35cfc80b67fe6c58a5f1cefd408829996bb1f5115764d2a`, verdict `PASS_REAL_27L_TWO_GREEDY_GPU_RESIDENT_SOURCE_KVB_INDEPENDENT`. The independent auditor checked original source/hash identities, all three source-model phases and per-layer rows, selected tokens, source logits, actual GPU device-memory count, all 371 registered forward calls, same weight digest over rebinding and complete device-handle cleanup.

## Additional separately verified physical tests

The actual original source **layer-0** compressed kv_b was registered GPU-resident and matched the prior independently compiled one-shot fused Metal output with relative L2 **0.0**; against the source BF16 expanded attention the relative L2 was **0.00334545**. Eight repeated same real-source MLA probes measured host-observed median approximately **0.875 ms** for this single attention call only. CPU host copied packed buffers were changed after device registration, without changing outputs; cleanup and stale-handle rejection PASS. Private layer-0 receipt SHA-256: `51fbad36f59ed6db3d2293ad556ee05e0af67d4506d9cb7fed3300b21a932bea`.

The independently portable Mac/MLX source-only package in `research/native_mla_gpu_resident_v1/` rebuilds native C and C++/Metal libraries without a model checkpoint. It verified real Apple Metal at lengths 1/2/5/7/16/32/64 with **0.0 relative L2 difference** between GPU-resident and original one-shot fused qNg64 kernels, host-buffer mutation resistance, malformed source rejection, handle cleanup and repeated query reuse. Synthetic independent receipt SHA-256: `97d64c6ee08df11c30f9889f0e3180e51393fe13a03f59fb8548ac8ff2c9b242`. A separate actual device-malloc/ownership verification returned 1,966,080 bytes for one kv_b and zero registered handles after release. This is real Metal execution, not a mocked numerical test.

## Scope boundaries and next gates

**This does NOT mean all 5,181 quantizable matrices are GPU-resident, nor that entire CPU/GPU process memory shrank 88.75%.** The reported 88.75% concerns stored **KV tensors only**. The real 27-layer study still reads original BF16 source weights and requantizes all nonresident roles per token while running an independent full recomputation; **842.90 seconds is not production TPOT**. The isolated attention-kernel millisecond times exclude all other projection, routing, source/model handling and host/device orchestration. GPU allocator *cached* memory may persist after handle release; the contract proves the 27 active source handles are invalidated, not OS-wide memory reclamation.

Future work: independently quality-gate a budgeted resident registry for additional original qNg64 attention/MoE roles; verify fresh prompts, long-context accuracy and multi-request isolation; time actual source-approved resident serving TTFT/TPOT p50/p95 and GPU+CPU peaks, prove not merely smaller persistent KV data, and retain Draft PR / auto-promotion OFF until reviewed. **No checkpoint shards, user prompts, raw weights, logits arrays or source host paths are included in the repository.**
