# BEGLIN qNg64 packed CPU/Metal research (2026-10-17)

This folder contains **isolated engine research**, not production backend changes. No model weights, tokens, checkpoint files, host topology, access keys, or deployment configuration are included.

## Implementations

- `qng64_tiled_accelerate_portable_poc.c`: decodes qNg64 bitplanes into bounded 64-row F32 scratch tiles. Uses Apple Accelerate SGEMM on the transient tile. The qNg64 codes and F32 scales remain authoritative; no permanent dense-weight cache. ARM NEON unpack and `-DBEGLIN_FORCE_SCALAR_UNPACK` fallback are both compiled and tested. **Apple Accelerate is still required**; “portable” describes the bitplane decoder, not an all-platform BLAS backend.
- `qng64_metal_packed_matrix_research.py`: GPU-resident uint8 bitplanes plus F32 group scales, exact two-SIMD-group Metal reduction mirrored from the existing `mlx_moe.cpp` qNg64 kernel, batch `M` supported. No GPU dense-weight buffer.
- `test_qng64_metal_packed_matrix_research.py`: standalone hardware-Metal tests for 6/7/9/12/15-bit synthetic codebooks and independent dense-operand oracles; fail-closed shape/format checks; packed-bit mutation and restoration. Requires `mlx`, `numpy`, and a supported Apple GPU.

## Run only on an approved Apple Silicon research machine

```sh
python3 research/local_quant_cpu_metal_20261017/test_qng64_metal_packed_matrix_research.py

clang -O3 -std=c11 -fPIC -dynamiclib -Wall -Wextra -Werror \
  -DACCELERATE_NEW_LAPACK -I. \
  research/local_quant_cpu_metal_20261017/qng64_tiled_accelerate_portable_poc.c \
  qng64_dense.c -framework Accelerate -lm -o /tmp/beglin_qng64_tiled_poc.dylib
```

The Metal test **requires** a GPU and must not be counted as a CI PASS on runners lacking MLX/Metal.

## Source-matched MoE gather kernel

- `qng64_metal_expert_gather_research.py`: mirrors the BEGLIN `mlx_moe.cpp::qng64_gather_gemv_kernel` two-SIMD-group algorithm to process arbitrary explicit `[token, expert]` pairs in **one real Metal dispatch** using raw qNg64 bitplanes. The adapter rejects invalid host expert IDs and weights without copying a decompressed GPU matrix.
- `test_qng64_metal_expert_gather_research.py`: 4 standalone synthetic tests including a separate Metal per-expert path, independent dense output oracle, negative IDs/formats, and packed-code mutation/recovery. Apple M1 Max MLX hardware: **4/4 PASS**; original native C qNg64 vs gather on real DeepSeek first-2-layer MoE routing: max relative L2 `1.314e-7` for independently sampled pairs.
- Limited **Gate projection** timing from 222 actual BF16 hidden-state token/expert pairs routed across 57 active experts: 57 separate Metal GPU calls median **22.01 ms**, single gather dispatch median **7.88 ms**, **2.79× fewer seconds** in 5 interleaved rounds. This is an API-level kernel+CPU synchronization comparison for Gate projection only, not whole MoE or model tokens/s.

To test on an approved Apple GPU:

```sh
python3 research/local_quant_cpu_metal_20261017/test_qng64_metal_expert_gather_research.py
```

All real BF16 checkpoint outputs and SHA-pinned private receipts stay on the authorized research host. **The public repository contains synthetic, reusable engine code and summary numbers only.**

## Important format boundaries

BEGLIN CPU qNg64 supports widths `2..15` except `4`, which is a distinct `q4g64` representation. The current `mlx_moe.cpp` custom Metal bitplane path supports `7,9..15`. This research adapter permits `6` **only with explicit `experimental_n6=True`**. Existing native MLX n6 dispatch repacks the format into MLX `quantized_matmul`; the custom n6 path here is **not an assertion that live C++ code already dispatches that way**.

## Verification status

These optional probes do not run automatically in the default Linux/Windows engine CI. Native GPU performance and source-authority conclusions depend on real hardware tests and independently pinned engine inputs. Detailed research receipts remain outside this public repository; the aggregate results and limitations are summarized in `docs/BEGLIN_LOCAL_QUANT_CPU_METAL_EVIDENCE_20261017.md`.

**No merge/promotion until a verified XOX engine build, real full-model BF16-vs-uniform-vs-mixed logits/NLL, warm tokens/s, peak unified memory and KV/epoch checks pass.**
