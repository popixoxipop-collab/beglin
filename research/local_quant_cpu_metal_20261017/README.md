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

## Full routed Gate/Up/Down experimental operators

- `qng64_metal_routed_ffn_research.py`: whole routed Gate→BF16 SiLU×Up→Down→topK weighted sum and optional packed shared FFN + residual on MLX GPU, retaining the same packed expert banks; synthetic Apple GPU tests **4/4 PASS** (`test_qng64_metal_routed_ffn_research.py`).
- `qng64_metal_fused_routed_ffn_research.py`: alternative custom Metal kernels for RNE-to-BF16 nonlinear and topK/shared/residual accumulation; synthetic tests **3/3 PASS** (`test_qng64_metal_fused_routed_ffn_research.py`).
- **Do not equate less host transfer with more speed:** at 222 real DeepSeek token–expert pairs across 57 routed experts, the existing staged 3-gather method gave **24.31ms** median for 3 Gate/Up/Down + source-style reduction, vs **30.21ms** unfused GPU-resident and **29.81ms** custom-fused; separate 171 calls gave **53.48ms**. Reuse 24.31ms as research baseline. The fused output differed from staged by only relative L2 `2.565e-5`, but **did not improve latency**.

On approved Apple MLX GPU:

```sh
python3 research/local_quant_cpu_metal_20261017/test_qng64_metal_routed_ffn_research.py
python3 research/local_quant_cpu_metal_20261017/test_qng64_metal_fused_routed_ffn_research.py
```

Details and precise benchmark scope: `docs/BEGLIN_MOE_GATE_UP_DOWN_FULL_GATHER_AB_20261017.md`. These are **not** live C++ backend/serving integrations. Model source weights and private receipts were not committed.

## Full 27-layer BF16 versus selective native qNg64 Q-projection inference

- `qng64_tiled_cpu_linear_research.py`: reusable BF16 PyTorch CPU linear with **actual C qNg64 packed bitplanes + F32 group scales**, bounded 64-row Apple Accelerate transient GEMM. No resident full F32 dense model weight; *research-only*, not production C++.
- `test_qng64_tiled_cpu_linear_research.py`: hardware-compatible synthetic 5-case native C test suite, including exact group/code byte accounting, independent mathematical decoder, invalid input rejection, and packed-code bitflip control. **5/5 EOE Apple Silicon PASS**.
- Build its local test dylib by compiling `qng64_dense.c` with the sibling `qng64_tiled_accelerate_portable_poc.c` as demonstrated above; then run with `BEGLIN_QNG64_RESEARCH_DYLIB=/absolute/path/to/compiled.dylib python3 research/local_quant_cpu_metal_20261017/test_qng64_tiled_cpu_linear_research.py`. Requires PyTorch, NumPy and Apple Accelerate, so NOT part of generic Linux GitHub CI.
- The **27/27-layer** DeepSeek BF16 research model ran in an isolated streaming reference, and its final logits SHA was reproduced byte-exactly when evaluated alongside real compressed **q_proj-only** trajectories. One calibration prompt chose n7 under strict equal 27-q-proj code/scale byte budget; +2% Q-only payload upgraded layers 2/7 to n9. Three independently authored holdout prompts yielded **25/25 top1 agreement** for +2% and average self NLL **5.76494 vs 5.78858 BF16**, but one of three prompt NLLs worsened. No production/task-quality or whole-model-all-weight conclusion.
- Details: `docs/BEGLIN_FULL27_REAL_BF16_QPROJ_LOCAL_QUANT_20261018.md`. Research source proof receipts remain on the authorized workstation and are **not** uploaded.

## Important format boundaries

BEGLIN CPU qNg64 supports widths `2..15` except `4`, which is a distinct `q4g64` representation. The current `mlx_moe.cpp` custom Metal bitplane path supports `7,9..15`. This research adapter permits `6` **only with explicit `experimental_n6=True`**. Existing native MLX n6 dispatch repacks the format into MLX `quantized_matmul`; the custom n6 path here is **not an assertion that live C++ code already dispatches that way**.

## Verification status

These optional probes do not run automatically in the default Linux/Windows engine CI. Native GPU performance and source-authority conclusions depend on real hardware tests and independently pinned engine inputs. Detailed research receipts remain outside this public repository; the aggregate results and limitations are summarized in `docs/BEGLIN_LOCAL_QUANT_CPU_METAL_EVIDENCE_20261017.md`.

**No merge/promotion until a verified XOX engine build, real full-model BF16-vs-uniform-vs-mixed logits/NLL, warm tokens/s, peak unified memory and KV/epoch checks pass.**
