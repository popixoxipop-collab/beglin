# BEGLIN — n7/group64 bounded multi-matrix Metal residency (research)

This is an isolated, portable Apple Silicon research implementation of the original BEGLIN qNg64 **n7, group64** packed bitplane format. It provides real MLX/Metal GEMV for independently registered `[out,in]` matrices, with **evaluated GPU device allocations**, a process-local opaque handle registry and a hard **256 MiB / 128-handle ceiling**.

The Python `research_n7_device_lru_cache_v2.py` adapter provides a model-side per-role SHA-verified LRU with its own configurable byte/handle caps and `torch.bfloat16` forward wrapper. The actual Metal output is F32 and is converted to BF16 for matching the original DeepSeek source.

**Research only. No production serving or automatic GPU promotion.** Do not treat this as a full-model GPU-resident inference engine.

## Portable package

| File | Purpose |
| --- | --- |
| `n7_multi_projection_gpu_residency_v10_256mb.cpp` | Native C++/MLX Metal compressed GEMV, 256 MiB bounded device registry |
| `research_n7_device_lru_cache_v2.py` | Source SHA/rebinding verification and bounded role-level LRU; requires PyTorch when used |
| `test_multi_gpu_n7_v10_portable.py` | Six source-shaped synthetic matrices, independent BEGLIN C quantize/matmul, negative safety tests |
| `build_and_test.sh` | Compile the original root `qng64_dense.c` and the Metal library; run physical synthetic regression |

The **synthetic** test uses `numpy`, Apple Silicon macOS, Xcode Command Line Tools, and the original MLX Python installation with C++ headers and `libmlx.dylib`. The LRU adapter requires `torch` but is **not** imported in the synthetic test. Nothing requires a model checkpoint, network access or credentials.

From the repository root:

```bash
BEGLIN_PYTHON=python3 bash research/n7_multi_projection_resident_v1/build_and_test.sh
```

The script writes only build artifacts and a local JSON receipt under `build/`. **Do not commit those compiler-specific dylibs, generated receipts, or original model tensors.**

## Physically measured correctness — synthetic

- Six simultaneous registered qNg64 matrices, geometries from **128×512** to **4096×2048**; total physical packed uint8 planes + FP32 scales **15,052,800 bytes**.
- Independent BEGLIN C `qng64_matmul_f32` numerical reference: maximum relative L2 **1.2930364961681175×10⁻⁷** over source-shaped multi-batch comparisons.
- Mutating caller-supplied host buffers after registration did not alter outputs. Repeated GPU queries were bitwise repeatable. Invalid bitwidth/dimensions/scales, stale handles and invalid batch sizes were rejected; all handles freed.
- The portable test was compiled and rerun successfully on the actual EOE Apple Silicon device with no checkpoint input.

## Physically measured correctness — separate private original-model evidence

Real source-verified DeepSeek-V2-Lite weights were read **without modifying** the D50 original BF16 checkpoint. The following private EOE experiments are not shipped in this package:

| Experiment | Result |
| --- | --- |
| Original layer 0: all seven n7 projections GPU-registered at once | **75,939,840 B; 7/7 PASS; max C-relative L2 1.2599042231×10⁻⁷** |
| Original layer 0: five- and six-token *complete original decoder layer* GPU-resident GEMV versus original independently compiled C++ Metal decoder | **BF16 hidden relative L2 0.0 for both** |
| Original layer 0: all seven retained in 96 MiB GPU LRU; four repeated six-token forward passes | **7 registrations, 35 reuse hits, 0 evictions**; one-run median approx **46.95 ms** (one layer/six-token recompute, **not TPOT**) |
| Original layer 0: 32 MiB GPU LRU forcing paging | **13 evictions; outputs still bitwise repeatable** |
| Original layer 1: source router chose **22** distinct experts for five input tokens; seven fixed + 3×22 expert roles | **73 actual n7 matrices**, total compressed payload **207,544,320 B** |
| Original layer 1: 96 MiB/12-handle LRU | **134 evictions, zero reuse hits** on two same-input calls; model BF16 output relative L2 **0.0** |
| Original layer 1: 256 MiB/96-handle LRU | **73 first-time device registrations, 73/73 reuse hits next call, 0 evictions**, model BF16 output relative L2 **0.0** |

The same-input single MoE layer warm wall time was observed at **0.15728 s** under the 96 MiB/12-handle condition and **0.12966 s** under 256 MiB/96 handles. These are *different bounded research process trials*, include CPU BF16 original source behavior, Python SHA checks and CPU/GPU staging, and are not general TPOT or causal proof of sustained improvement.

### Private SHA-verified EOE receipts

- Real layer0 seven standalone projections: `393f2fdc205321ac411659b984b5fb12a8381d7604985306a9cd59a59488028f`
- Real layer0 complete-source forward and GPU LRU: `8f03d8a33fb617418345f95e30691fb22bbbba64e6068a89f7792cbbb923701b`
- Real layer1 96 MiB routed MoE run and independent audit: `a4eb68cacf9b9c655e7a9ea066ca8ced748127f010ccd765c55f82fc772801ef`, `5ae42c401dc8c6a045424d2fab868d12a0a5df6cf1b2a70dd815d0720455e5da`
- Real layer1 256 MiB routed MoE run and independent audit: `0354c4e2281d4bd54a4fecb8e882c84a8134d4c9436185ecc5bbff5d1d551cab`, `d407139dcc6e9e087fb0e2e84374ebcef263ccd195e866b1bb8424b7eb3e44d1`

## Important limitations and next gates

1. **GPU plus CPU dual backing**: the correctness-first C++ registry retains immutable compact host buffers for the lifetime of GPU handles, because a borrowed-memory lifetime problem was found in an earlier version. GPU active allocator bytes count the compact GPU allocation, **not the total host+device footprint**. A future version may remove the CPU source backing only after an independent, pointer-lifetime-adversarial device ownership proof.
2. **Explicit per-request input transfers**: the generic GEMV still receives F32 CPU vectors, stages them to GPU, and copies results to CPU BF16 model inference. Original source attention, router, BF16 norms/residuals, and other operators are not all native GPU.
3. **Model-wide**: all 5,181 eligible source matrices amount to many gigabytes; the 256 MiB registry is an explicitly bounded hot working-set, not full model residency. The realized 73-role case applies to one original routed layer and one five-token source prompt.
4. **Quality / latency**: no statistically powered heldout prompts, full 27-layer multi-token resident-GEMV end-to-end benchmark, TTFT/TPOT p50/p95 or pressure/eviction scheduling certification here. Original qNg64 CPU/GPU kernels may require separate low-bit contract gates.
5. **Safety**: raw C input pointers cannot establish backing buffer lengths. Only pass source SHA/manifest-attested arrays from a trusted, capacity-checking loader. Invalid shapes, bits, budget overruns and handles fail closed. Other host/serving/release state is unmodified.

**Promotion status: DRAFT research only; XOX runtime/serving and BEGLIN automatic live promotion remain untouched.**

## Optional content-addressed original-C host packing memo

The companion `research_n7_source_pack_memo_v1.py` is a strictly opt-in,
bounded original `Native.bind` wrapper. It computes SHA-256 over each
contiguous **original source F32 dense weight matrix** and reuses only an
identical previously produced n7/group64 packed uint8/F32-scales tuple,
else calls the original BEGLIN C quantizer. Its upper bound is 8 GiB
(host compact weight memory; runtime config may set a smaller limit).
The arrays are made read-only, and the original method is restored on close.

`test_n7_host_packed_memo_portable_v1.py` exercises the original native C
quantizer/matmul, byte-identical SHA hits, changed-content rejection,
immutable compact arrays, forced RAM LRU eviction and deterministic
repacking; no original source model or actual weights are shipped. Both
standalone synthetic tests run from `build_and_test.sh`.

**Memory scope:** this is deliberately an additional CPU-side compact
cache. It reduces repeat **quantization**, not BF16 original-source file
reads; it must not be interpreted as a reduction in GPU/CPU overall
process memory, full-model residency, real TTFT/TPOT or production speed.


## Original policy working-set calculator (offline, research only)

The additional `estimate_n7_host_cache_budget_v1.py` is a fail-closed
**metadata-only** planner. It checks source manifest SHA-256 and all
**5,181** frozen n7/group64 model tensor policy entries, verifies all
1,664 MoE expert three-role size contracts, and uses an **observed aggregate**
non-kv_b source-role count to estimate immutable packed host memory.
It does not read model weights, shard payloads, hidden states or expert IDs.

In the 27L original-model trial, 2,367 distinct non-kv_b GPU source roles
imply 735 unique experts and **6,794,035,200 B** compressed working set.
The planner uses 20% headroom (rounded up to whole GiB) and recommends
**8 GiB** as a *research-only admission candidate*. An independent physical
8 GiB model trial retained exactly that many compact bytes and had zero
CPU LRU evictions. The 6 GiB trial had 2,563 CPU LRU evictions.

Use a trusted SHA-pinned policy file and an available-memory measurement:

```bash
python3 research/n7_multi_projection_resident_v1/estimate_n7_host_cache_budget_v1.py   --policy-file /path/to/verified/quant-policy.json   --policy-sha256 YOUR_VERIFIED_64_HEX_SHA256   --observed-nonkvb-roles 2367   --available-ram-bytes TRUSTED_AVAILABLE_MEMORY_BYTES
```

The planner rejects incomplete expert trios, unauthorized n7 policy changes,
nonhomogeneous expert sizes, duplicate names, malformed role counts,
insufficient memory (reserving 2 GiB for other work) and cache cap overflow.
Prompts with different expert routing may require different capacity.
It does not authorize production or remove source/checkpoint attestation.

`test_estimate_n7_host_cache_budget_v1.py` provides 10 no-model synthetic
admission/fail-closed tests and is included in `build_and_test.sh`.
