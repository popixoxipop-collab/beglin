# BEGLIN v11 — original-model 5,181 sealed qNg64 and Metal validation handoff

Date: **2026-10-19** (Asia/Seoul)
Decision: **PASS original model packed-artifact certification and isolated GPU linear bridge; NOT_READY for complete prequant-loaded greedy model inference/production.**

Repository: popixoxipop-collab/beglin
Research branch/PR: research/local-quant-budget-20261017 / Draft #63.
Original DeepSeek-V2-Lite BF16 source and all compressed model payloads remain local to EOE/D50. Do not upload model shards, hidden states, logits arrays, original prompts, or credentials.

## 1. Completed frozen source authorization

| Object | Confirmed |
| --- | --- |
| Source safetensors index SHA-256 | d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf |
| Source boundary 5,291-tensor C++ manifest SHA-256 | 274247e626f67579e56256e356f554fd1b1e5e5967df2603ec82bfb902af4963 |
| Four source shards | Full-file SHA independently checked; matched frozen materializer authority |
| Original native C qNg64 dylib SHA-256 | 6d76b36ef23341e7ac90ecda79541e93133e701e5538ab215e757c334526d786 |
| Materializer source SHA-256 | ca1872c6ea41861d6db97ed29010e68d2d794c500cb1909e16a5fb9e40b1acf8 |
| Sealed original-source run authority SHA-256 | 3b13bc1d98c5843f146afa428b6dfff45352f7def698d6a5a156d32d0fae136f |
| **Complete sealed n7 model manifest SHA-256** | **c82361f7fbd73884b633bb595334805c0b7c0cf2f82ab8de26311ca78a0142b8** |

An independently rebuilt original C++ model-source boundary validator returned PASS on the EOE original D50 checkpoint: **27 decoder layers, 5,291 tensors, 5,181 n7 group64 eligible role weights and 110 unchanged BF16 roles**.

Eligible packed n7 weights have exact combined bytes **14,328,299,520**, with **5,181 planes + 5,181 scale files + 5,181 committed original-source role receipts**. The remaining BF16 exclusion payload is accounted for separately; projected mixed on-disk model weights are **15,174,228,992 bytes** including exclusions, not physical GPU resident memory.

## 2. Real complete model source prequantization, independent value audit

The bounded EOE resource-scheduled native C materializer completed successfully:

- Scheduler job: job_a8be54ec9d17f0227d2d611aed509b28.
- Result: PASS_SEALED_FULL27_5181_QNG64_N7, process exit **0**.
- Original BF16 shards were opened read-only, and each eligible matrix was encoded ONCE with original qng64_dense.c n7/group64/F32-group-scales.
- Output is on D50 under the research work pool, not the original model or serving directory.
- Final full manifest SHA is the value listed above.

A separate load-test independently reopened **all 5,181** sealed packed roles via SHA-checked read-only mmap, confirming all **14,328,299,520 bytes** and the exact 5,181-role shape/dtype/mapping policy: PASS_ALL_5181_SEALED_ROLE_BYTES_REHASHED_ON_READ.

**Stronger independent numerical audit:** A separate durable job reopened the ORIGINAL BF16 tensors and independently called the original native C quantizer for ALL **5,181** roles, comparing **planes and float32 scales bitwise** against the sealed source:

- Scheduler job: job_cdcbee5481d6ae3707c7bef992a5cba3.
- Terminal: **SUCCEEDED, exit code 0**.
- Result: **PASS_ALL_5181_REAL_ORIGINAL_BF16_TO_SEALED_N7_BITWISE_NATIVE_C_REENCODE**.
- Recomputed and compared: **5,181/5,181 roles; 14,328,299,520/14,328,299,520 bytes**.
- Runtime: approximately **499.936 seconds**.
- Original weights unchanged; the second pass only read checkpoint shards and compressed artifacts.

The result verifies compressed value identity to this frozen source/quantizer. It does NOT demonstrate whole-model language task quality or full-model end-to-end decoding parity.

## 3. Native prequant loader/Metal bridging tests

The research-only prequant_full27_execution_adapter_v1.py constructs original C++ packed-module instances without repeating C quantization, reads exact roles from the sealed manifest, and validates original source RUN_AUTHORITY.json against externally pinned checksum/shard identities/encoder.

Authority negatives **PASS**: altered RUN_AUTHORITY.json rejected and missing materialized packed payload rejected. A revised layer-0 adapter test uses the **full 5,181-role sealed manifest**, not a seven-role preview bypass.

| Real-weight Metal test | Result | Interpretation |
| --- | --- | --- |
| One real layer-0 prequant module test | **7 GPU GEMV**, source payload **75,939,840B**, max BF16 L2 vs C **1.257646824e-6**, handles after teardown **0** | Original layer-0 roles through complete sealed-model adapter |
| Original per-layer Q projections across 27 layers | **54 GPU GEMV**, 27 hits, max BF16 L2 vs C **3.483346973e-6**, handles **0** | Q-projection-only parity across all original layers |
| Independently re-encoded original 27 Q projections | **159,252,480B bitwise identical** to prequant files | True C-weight encoding identity, not just numerical approximation |
| 27 layers × seven attention/shared FFN roles with real sealed source | **189 roles, 378 GPU GEMV, 189 hits, 125 evictions**, **833,126,400B** packed payload, max BF16 L2 vs original C **0.000123880644**, handles **0** | Real per-role Metal/CPU oracle and LRU replacement, not full decoder |

The 189-role case loads all seven original non-router selected fixed roles per layer: layer 0 attention + dense FFN, and layers 1–26 attention + shared FFN. The remaining routed MoE expert matrices ARE sealed and bitwise source-attested but were not all processed by this new 189-role GPU test. Earlier v11 routed MoE tests are independent historical evidence, not results of this complete sealed-loader run.

All GPU tests used synthetic BF16 activation vectors, not original end-to-end model hidden states or model-generated prompts. These GPU micro-tests are not measured TTFT, TPOT or model throughput.

## 4. CI and source management

On research PR #63 SHA d99a9088fb3204a3742bf73cf914041d798958a7, 7/7 GitHub checks were confirmed SUCCESS, including the sealed-prequant reader workflow (Python 3.11/3.13). This is a statement about THAT exact tested commit; subsequent documentation commits require a new CI check.

- Research scripts and test adapters are committed to the Draft PR.
- The isolated D50 research checkout was fast-forwarded, preserving all existing XOX dirty files and original model source.
- v10 fallback and BEGLIN production binaries were not changed.
- EOE immutable operator current/live/runtime last checked g6-0.9.0-operator.900387, converged with no active operation.

## 5. Remaining production blockers — do not overclaim

The attempted existing private original v11 27-layer two-greedy execution **with this newly sealed full-model loader** was rejected by an execution security gate; it was not retried by bypassing that gate. Therefore **there is no newly verified 27-layer model logits/token-generation receipt using the sealed loader**. Earlier v11 results using a different host-compressed-cache route did complete the real 429 → 4281 input sequence and next prediction 6230, but those old results must not be relabeled as a successful new prequant-loaded end-to-end run.

Still required in an authorized, reviewed invocation route:

1. Actual frozen 27-layer prefill + TWO source-chosen token decodes (429, 4281) using this complete sealed loader and its verified original source/run authority.
2. Independently evaluate logits and token identity (next prediction 6230), NLL/quality, and saved latent KV cache; require no fresh native C quantization and no handle leak.
3. Collect TTFT/TPOT p50/p95/p99, full RSS/MLX allocator telemetry and longer/concurrent real-model requests with v10 as an unchanged fallback.
4. Review native GPU in-flight eviction and OS sleep/resume before declaring memory/serving safety.

**Conclusion: Certified full-model n7 prequant data, original-source numerical fidelity, and GPU role-by-role integration; complete prequant-loaded 27-layer autoregressive serving and production readiness remain NOT_READY. Auto-promotion OFF.**
