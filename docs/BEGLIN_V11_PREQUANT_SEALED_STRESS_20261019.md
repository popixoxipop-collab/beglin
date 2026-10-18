# BEGLIN v11 — sealed prequantized loader and physical Metal stress research

Date: 2026-10-19 (Asia/Seoul)
Decision: **PASS research-only, NOT_READY for production.**
Preserve v10 fallback. PR #63 remains Draft; do not merge, deploy or auto-promote.

## Delivered research code

- research/n7_multi_projection_resident_v1/prequant_n7_sealed_v1.py: SHA-pinned qNg64 n7/group64 read-only mmap loader; one-time export of already quantized bitplanes and FP32 scales; role, shape, byte count, positive finite scales and identity validation. Does not load original BF16 at inference time.
- test_prequant_n7_sealed_v1.py: 15 fail-closed contract/adversarial tests.
- test_v11_prequant_metal_concurrency_v1.py: two independent actual Apple Metal processes, each with three request threads serialized by its v11 cache RLock, using the same sealed synthetic matrices. Tracks native GPU calls, LRU eviction, output parity, memory and handle teardown.
- test_v11_prequant_real_checkpoint_role_v1.py: actual original DeepSeek-V2-Lite BF16 checkpoint layer-0 q_proj prequantize ONCE, seal, load as mmap, 20 hardware Metal calls, independent original C comparison.
- .github/workflows/beglin-v11-prequant-sealed.yml: Ubuntu Python 3.11 and 3.13 CI, no access to confidential model data.

The checkpoint index SHA alone is insufficient to attest shard data; the original single-role hardware experiment **also hashed and checked its full 8GiB source shard**. The general loader does not yet attest every original shard or a model-wide frozen compression policy. A sealed manifest being hash-matched at load time does not prove immunity to malicious filesystem mutations afterward.

## Executed EOE results

| Experiment | Result | Proven scope |
| --- | --- | --- |
| Sealed loader CPU tamper checks | 15/15 PASS on Apple Python 3.11 | Schema, SHA, symlink, size, duplicate, rejected roles |
| Original C++/Metal candidate fresh-start synthetic test | 3/3 PASS, six matrices per process, native C max relative L2 1.293036496e-7 | Physical GPU C++ pointer ownership and teardown |
| Sealed synthetic two-process initial run | 1,086 Metal GEMVs; 0 GPU handles after close | Two-process shared artifact read |
| Two-process 25 epoch × 20 repeat test | 18,006 GEMVs; 7,910 LRU evictions; handle count 0 | Bounded simultaneous GPU access and reuse |
| Two-process 100 epoch × 20 repeat test | **72,006 GEMVs; 31,880 LRU evictions; GPU handle count 0; identical output hash** | Repeated calls and cache lifetime under limited duration |
| Real original DeepSeek-V2-Lite BF16 Q projection (decoder layer 0) | 3072 × 2048, n7 payload 5,898,240 bytes, 20 Metal calls, 19 GPU hits, BF16 vs original native C relative L2 6.458573579e-8 | Real weight, single projection with synthetic activation, not full-model logits |
| Same single real original role | Original BF16 export quantization once, reload stage re-quantization zero, handles after close zero | Prequant materialization/mmap-to-Metal interface |

The 100-epoch experiment lasted **about 26 seconds per worker**. RSS grew by about 7,798,784 bytes and 7,962,624 bytes after the first epoch, respectively. The bounded 64 MiB drift gate passed. This is **not** a multi-hour leak-free claim: MLX cached allocator memory, unified system memory, sustained serving, concurrency with in-flight eviction, sleep/resume, and end-to-end TTFT/TPOT remain untested.

The 3 request threads within a worker are **serialized by its Python RLock**; only the two independent worker processes execute concurrently. Therefore this experiment is not a proof that one native C++ GPU registry can safely accept simultaneous in-flight forwards and evictions.

## Source and evidence authority

| Object | SHA-256 |
| --- | --- |
| BF16 safetensors index | d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf |
| First BF16 original source shard (full contents) | a314b79a30810b1038f9db08345178e16fb5742aa266c4a3dc05054c59800082 |
| Existing v11 C++ Metal source | f10a8b30bfeff95cf0e756d45567ac4aaaa3d2552f5ce62a7746af09000a3201 |
| New prequant reader Python file | ba41cf9cd0317d608742d0f2ec95ed6e5ac21fc9b586b9ca89c042685b9622ab |
| New concurrent Metal stress Python file | ea554e1368d3b044d1b6149f026543d7b6a20c7639504e32c300a9f371d6a508 |
| New original model single-role Python file | 48e3340ad605364777cfe994281ba148072b222e86eb1f950566746f6f9c7b91 |
| 2-process, 1,086 GEMV private research receipt | 22c4a39f32a2522a20060473415e70c26d4dd27ba350273a18746b8ed224ba80 |
| 2-process, 18,006 GEMV private research receipt | 50c4fe24f354f50f6d7897bc3634179f441d0eddc9f3ad82daaa2427703de2a4 |
| 2-process, 72,006 GEMV private research receipt | c5a15d360c60f8a7f68a696e0e7dc56da7f71acd09565c737c9f437f915e4705 |
| Actual model role C-quantized payload hash | 14538af5b9cdf93de5f15be10982da89e3a5dde95562773ed093795b38e3550c |
| Actual model role sealed manifest hash | 520fd9bfd6438c1dcc77353d2f33b7ced0209a21a36d63784d5100081c9688e6 |

The three synthetic stress receipts are stored privately in the EOE isolated research checkout under its build/research_stress_receipts folder and were separately reopened and rehashed. Do not publish raw source checkpoints, hidden states, prompts, logits or client credentials.

## Reproduce without production changes

Run only in a new isolated checkout on a compatible physical Apple Silicon Metal machine with original BEGLIN qng64 native C source and MLX C++ libraries:

~~~bash
export BEGLIN_PYTHON=/opt/homebrew/bin/python3.11
bash research/n7_multi_projection_resident_v1/build_and_test_gpuowned_v11_candidate.sh
python3 -m unittest discover \
  -s research/n7_multi_projection_resident_v1 \
  -p 'test_prequant_n7_sealed_v1.py' -v
BEGLIN_V11_STRESS_EPOCHS=100 BEGLIN_V11_STRESS_REPEATS=20 \
  python3 research/n7_multi_projection_resident_v1/test_v11_prequant_metal_concurrency_v1.py
~~~

Original-model single-role test additionally needs a private local original checkpoint directory and private SSD scratch. It enforces the frozen model index and shard SHA; never transfer checkpoint bytes through CI or GitHub.

## Pending readiness blockers

1. Model-wide prequant output generation and sealed loader for all eligible 5,181 matrices, with all 110 BF16 exclusion roles preserved and compression policy frozen.
2. Independent checks of **every** source shard and the quantization transform provenance; fail closed on incomplete, outdated or swapped material.
3. Full 27-layer prefill and repeated greedy decoding with prequant-loaded tensors, and independently reproduced logits, tokens, NLL and quality on adequate held-out samples.
4. Multi-hour memory/allocator soak, mixed concurrent real requests, native in-flight eviction races, SIGTERM/crash restart, sleep/resume, old vs new macOS/MLX.
5. Actual TTFT, TPOT and end-to-end model tokens/s p50/p95/p99, system-level CPU+GPU memory and sustained throughput against v10.

**No XOX serving binaries, original BF16 data, or Tailnet Commander EOE operating releases were modified in this research.** Last checked EOE current/live/runtime was g6-0.9.0-operator.900387 and converged; auto-promotion remains OFF.
