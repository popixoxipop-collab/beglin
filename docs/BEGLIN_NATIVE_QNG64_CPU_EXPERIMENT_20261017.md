# BEGLIN: native qNg64 CPU encoder/GEMV and budgeted policy measurements

Date: 2026-10-17 · **Engine research only** · No production inference, model weights, or automatic promotion.

## Native implementation checked

Compiled real `qng64_dense.c` and invoked its `qng64_quantize_f32`, bitplane decoder, and `qng64_matvec_f32` on GitHub CI runners.

- Kernel contract: 13 supported qNg64 bit widths (`2,3,5,6,7,8,9,10,11,12,13,14,15`) passed; `n=4` is rejected by the native API and uses a *separate* `q4g64` format. CPU qNg64 n=8 is distinct from `q8g64`.
- Per-64-weight encoded payload = `n*8` code bytes + 4 bytes for a separate F32 scale. The C `qng64_packed_bytes` function by itself reports **code bytes only**, whereas the Python optimizer reports code **plus scale**.
- Actual native C numeric roundtrip on deterministic synthetic weights: `n=3` relative weight L2 ≈ `0.206016`, `n=15` ≈ `0.00004102` on macOS. This is not DeepSeek quality.

## Same-byte CPU GEMV experiment: arbitrary sensitivity proxy

On synthetic independent matrices (6 roles, 64×512 per role), native CPU `qng64_matvec_f32` was compared at **identical serialized weight bytes: 184,320B**.

| Runner (CI) | Uniform 7-bit output rel L2 | Arbitrary-weight mixed [7,7,6,9,7,6] rel L2 | Mixed/uniform kernel sweeps/s |
|---|---:|---:|---:|
| macOS Python 3.11 | 0.000106430 | 0.000141115 | 0.855× |
| macOS Python 3.13 | 0.000106430 | 0.000141115 | 0.998× |
| Ubuntu Python 3.11 | 0.000106392 | 0.000141129 | 0.774× |
| Ubuntu Python 3.13 | 0.000106392 | 0.000141129 | 0.800× |

**Important counterexample:** The earlier arbitrary per-role weight reconstruction proxy favored a mixed policy on the same 184,320B budget in a different demonstration, but **actual native GEMV output error is 32.6% worse here**. Different synthetic calibration input/importance and no consistent speed improvement invalidate generalizing a proxy win into model-quality improvement.

Evidence: [CI 39009589246](https://github.com/popixoxipop-collab/beglin/actions/runs/39009589246).

## Closed-loop calibration using ACTUAL native C GEMV outputs

Added `benchmarks/qng64_native_output_calibration.c` to compute each role's `rel_l2` vs dense FP32 GEMV, using `importance = ||W x||²` so:

```
importance_i × relative_output_L2_i² = sum_j (dense_ij - qNg64_ij)²
```

For **independent synthetic GEMVs with fixed x**, minimizing the weighted sum is equivalent to minimizing overall stacked-output squared error. Unlike an arbitrary weight-L2 proxy, this exact objective matches the subsequent kernel evaluation **for this limited workload**.

The pipeline (native C calibration → `precision_budget_search.py` exact Pareto search → native C chosen-policy execution) produced **uniform [7,7,7,7,7,7]** as the optimum on all four OS/Python runners. Both uniform and selected policies occupied 184,320B, relative output L2 ≈ 0.00010643 on macOS and 0.00010639 on Ubuntu, i.e. **0% output error reduction**. The measured repeated-sweep throughput varied with runner activity (optimized candidate was the same policy run twice); do not interpret this variation as a performance improvement or regression.

Evidence: [CI 39009904656](https://github.com/popixoxipop-collab/beglin/actions/runs/39009904656); 4/4 matrix jobs SUCCESS. Existing engine CI `gpu-precision-g1-g3` also SUCCESS.

## Existing XOX real CPU/MPS parity (different experiment)

Read-only pre-existing `beglin_followup(status)` reports G53 `PASS` and G54 `PASS`. The stored G53 source SHA-256 `3f7615466dcf30d6bde5fbf299f098bbe1a40c3b507589c03da431314c1ca721` verifies:
- Six G35/Q4 prompt comparisons (128, 512, 1024 tokens) have CPU=MPS Top-1 tokens.
- Maximum absolute last-logit difference among six comparisons ≈ 7.6294e-5, maximum NLL sum difference ≈ 0.001709.
- This establishes parity for **those previous baseline policies**, not mixed-budget qNg64 quality.

## Unresolved engine work

1. Acquire authorized, independent real DeepSeek per-tensor calibration; the XOX registered Python exec policy has denied previous execution. Do not bypass it with privileged pathways or mutate serving.
2. Compare original BF16, uniform, and mixed qNg64 with identical **model/tokenizer/request/epoch** on genuine CPU and MLX/Metal execution paths.
3. Report model-level next-token NLL, task accuracy, warm decode tokens/s, peak CPU RSS, GPU resident memory, and KV-cache epoch consistency. Current test measures only isolated CPU C GEMV, not full LLM tokens/s or GPU Metal.
4. A real model may have heterogeneous role sensitivity unlike the test's similar synthetic roles. The optimal policy may differ; **no claim of model-level performance gain** is made.

Research PR remains DRAFT. No XOX server routes, model files, or Tailnet releases were changed for these CI experiments.
