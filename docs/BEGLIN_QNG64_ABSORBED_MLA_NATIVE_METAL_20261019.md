# BEGLIN qNg64: Native C++/Apple Metal absorbed MLA query/value attention — 2026-10-19

**Research only / Draft PR #63 / no live promotion or serving changes.**

This follow-up to [the physically verified two-token 27-layer source MLA result](BEGLIN_NATIVE_CPP_MLA_LATENT_KV_20261019.md) demonstrates a separate **actual Apple Metal** execution path that **never expands historical per-head BF16 K/V on GPU**, while consuming the original BEGLIN qNg64 **n7/group64 uint8 bitplanes and F32 scales** directly. It is **not** yet integrated into the real DeepSeek-V2-Lite decoder stack and does not establish end-to-end throughput.

## Algorithm and supported geometry

DeepSeek-V2-Lite source MLA at B=1, H=16, \(d_c=512\), \(d_{q,\mathrm{nope}}=128\), \(d_{\mathrm{RoPE}}=64\), \(d_v=128\). Original \`kv_b_proj\` weight is physically encoded as a single **4096 × 512** compressed qNg64 matrix. Rows per head are \`[128 key rows; 128 value rows]\` and **no dense dequantized weight matrix** is instantiated on Metal.

The isolated C++/MLX module runs **four GPU kernels**:

1. **Absorb original q-nope**: for each head, compute \`q_abs[512] = W_key_n7.T @ q_nope[128]\` by reading signed n7 bitplanes and group64 scales, never generating a dense GPU \(W_k\).
2. **Compressed latent attention scores**: \`score[h,t] = scale * (q_abs[h] · latent[t] + q_rope[h] · k_pe[t])\` for all prior cached source latent records (and single current query).
3. **Softmax latent aggregation**: \`c[h,512] = sum_t softmax(score[h,:])[t] * latent[t,:]\`.
4. **Packed value projection**: \`out[h,128] = W_value_n7[h] @ c[h,512]\` directly from original compressed planes/scales, without materializing expanded historical \(V\).

The module exposes \`beglin_qng64_mla_absorbed_probe_v1\` as an experimental bounded buffer-size-checked ABI. Input limit **T=1..64**, 16 heads, 512D latent, source n7/g64 bitplanes only, F32 intermediate arithmetic; **no batch>1, no general masks or multiple-query causal prefill, no RoPE computation within this module, and no output \`o_proj\`**. It is a four-dispatch absorbed MLA **prototype**, **not** one fused all-C++ attention kernel or a DeepSeek layer replacement.

## Physically executed Apple Metal synthetic tests

The **actual compiled C++ MLX Metal library** ran on EOE Apple Silicon. A separate independently compiled **original BEGLIN C qNg64 quantizer and C matmul** generated the same n7 source compact buffers and, **on CPU only**, materialized expanded K/V as an independent scalar/matmul reference. The Metal candidate never used those expanded CPU-only values.

| Experiment | Observed |
| --- | ---: |
| Exact user-engine packed matrix format | qNg64 n7, groups of 64 |
| Actual source-weight bitplanes in one \(4096 \times 512\) packed kv_b-like matrix | **1,835,008 B** (plus 131,072 B F32 scales) |
| Initial deterministic numerical shapes | 1, 5, 7, 16, 64 tokens |
| Initial max Metal V-context relative L2 vs independent original C | **6.822886e-7** |
| Initial max Metal score relative L2 vs independent original C | **1.44971e-7** |
| Initial isolated **warm four-kernel** wall per call | **0.53–0.57 ms** (not end-to-end tokens/s) |
| Independent randomized scenarios | **3 unrelated seeds × 5 lengths = 15 cases, all PASS** |
| Randomized cases maximum V-context relative L2 vs original C | **4.5881562055583294e-7** |
| Randomized cases maximum attention-score relative L2 vs original C | **4.4297609378674636e-7** |
| Malformed raw buffers/shapes/lengths, NaNs and invalid source scale rejected | **17 / 17 PASS**, buffers unchanged |
| Original checkpoint loaded in these GPU tests | **No** |
| XOX serving, operating runtime, DB or deployment changed | **No** |

The initial deterministic and the follow-up randomized results are independent evaluations with the same C++/Metal source contract; their maxima need not match. The CPU-only compressed query/value absorption oracle separately passed lengths [1, 5, 7, 16, 64], with overall source-output relative L2 **5.31412e-9** against its independent expanded CPU reference.

**Evidence SHA-256 (EOE source-level receipts, not uploaded checkpoint artifacts)**

- New C++ actual MLX Metal code: \`74c705b7cec2a5b8394d9d17226763eaff3fbf44e28ba15a9b5e664d2465b1ff\`.
- Compiled EOE research dylib: \`aa19ae2d90f1168e88b24ae2c6e13a95d2792322c007cf4767bf28613ff1b555\`.
- Initial five-length, 17-malformed-input real-Metal test receipt: \`949e40b5980385652adb9a15ca2724d005475643516bc637a2cd8fe31198f57c\`.
- Portable randomized 15-case, 17-malformed-input EOE test receipt: \`a7db80f3771730eb4e7cf6c41be0f5c102f975bbcddf2cbaf5eaa3e89e933552\`.
- Portable randomized test source: \`02676c2988aba1a4e7bdfe3757ef10e1d9ba3b2421240b96a32824beb8af62c2\`.
- Independent native C compressed query absorption reference source: \`1fe96d38afe0c347231daf5e54beb1ee70eff78d897b56260f822adcd9cc8f03\`.

### How to reproduce the synthetic real-Metal regression

On an authorized Apple Silicon Mac with Xcode command-line tools, \`numpy\` and a compatible installed \`mlx\` Python wheel (with C++ headers and \`libmlx.dylib\`):

\`\`\`bash
python3 tools/test_qng64_absorbed_mla_metal_portable_v2.py
\`\`\`

The script compiles a temporary copy of the original root \`qng64_dense.c\`, builds \`tools/qng64_absorbed_mla_cpp_metal_v1.cpp\` against the local MLX C++ SDK, and exercises **actual** Metal and independent C qNg64 references. It does **not** load a model checkpoint. It prints only aggregate scalar numerical metrics. Generated shared libraries are stored under a temporary directory, not committed to GitHub. This test is **not** expected to pass on CPU-only runners and must not be represented as standard generic hosted CI quality certification.

## Additional current real-model evidence; separate from this prototype

An earlier **source-defined, but CPU-expanded-on-demand** \`MLA latent KV v3\` implementation has now passed the **actual** DeepSeek-V2-Lite 27-layer, two-model-selected-token EOE run using BEGLIN C++ Metal packed projections (the original CPU BF16 attention remains). The actual input IDs were **429, 4281**, with subsequent predicted ID **6230** (not fed back); source-attested v3 operator rebinding SHA identity checks passed, measured 7-position 27-layer resident BF16 latent cache **217,728 B vs 1,935,360 B** expanded (**88.75% KV tensor-byte saving**). Its cached-to-full-uncomputed final logits relative L2 was **0.004559950089471063**, and **6,726** packed module instances were verified. This **v3** source-attested two-token receipt SHA-256 is \`f5758fd3ac591ab23925afd4bfdda9d2a191537b645240768fd95be2dc29e876\` and its independent audit receipt SHA-256 is \`efc82de269794108fd26f6a93cab6914290b0501816dc51b2f3b69528eb59410\`. **Neither of these experiments ran the new GPU absorbed-attention module described above.**

The new absorbed-attention module's attempted read-only **real checkpoint layer-0 integration test** was **not executed**, because the remote command was blocked by platform safety inspection. That restriction was not bypassed. Therefore, **no real model logits quality/accuracy/latency results for the new GPU absorbed-attention path are claimed**.

## Next gates before performance or serving claims

1. Through an authorized source-tested research execution path, compare the new compressed Metal absorbed-query output against real source original DeepSeek-V2-Lite layer-0 attention, then multiple decoded positions and 27-layer logits.
2. Verify BF16 rounding differences and the source model's MoE expert routing/top-1, with independent holdout prompts, all layers and actual sampled token inputs.
3. Implement a truly **native MLA/KV GPU-resident serving path** with packed weights loaded once and efficient query/key/value reuse rather than repeated read/quantize or CPU BF16 fallback. Benchmark TTFT, TPOT p50/p95, full-model Apple GPU + CPU peak memory and actual task-quality with the frozen evaluation contract.
4. Preserve draft PR gate. Do not modify original XOX serving, model promotion policy, or release plane until independent quality/safety and original-model source authority gates are complete.

**Claim boundary:** This is **real compressed qNg64 Metal-kernel MLA absorption proven numerically on synthetic source-shaped inputs**, not an end-to-end DeepSeek fused native-C++/Metal LLM, production speedup or model-quality certification.
