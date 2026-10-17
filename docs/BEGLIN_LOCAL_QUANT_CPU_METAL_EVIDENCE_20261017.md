# BEGLIN real packed qNg64 CPU–Metal engine evidence — 2026-10-17

**Scope:** Apple Silicon research implementation of **first DeepSeek-V2-Lite decoder layer only**, using real BF16 model weights from a research copy. All 7 Attention/FFN projection weights are stored as packed qNg64 CPU or GPU buffers. Production serving and immutable releases **unchanged**. No trained model/logit data or internal host paths are in this public document.

## Native packed CPU proof

Exact same full tensor shape, source model and two test inputs; 7 independent linears all call BEGLIN C `qng64_matmul_f32`, including FFN gate/up/down. Training chose one of four candidates before separate holdout validation.

| First-layer held-out condition | Uniform qNg64 7 bits | Mixed [9,9,9,7,6,7,7] |
|---|---:|---:|
| Serialized packed code + scale bytes | 75,939,840 | **75,530,240** |
| BF16-reference first-block output relative L2 | 0.013988 | **0.011798** |

Mixed vs uniform: **15.66% smaller relative L2 output error at 0.54% less packed weight memory** in this tiny 2-prompt CPU experiment. Native C called 84 times. Independent F64 bitplane checks and FFN packed-bit mutation/restore negative control PASS. Original source authority and whole-model task quality NOT certified.

## Apple Accelerate tiled C prototype

CPU qNg64 bitplane decoder keeps compact weight storage; decodes at most 64 rows to a transient F32 tile, then calls Apple Accelerate SGEMM. Native NEON and scalar-unpack fallback each passed **27 numerical cases plus 4 parameter-rejection cases**. Max output relative L2 vs BEGLIN's reference C kernel: **2.153×10⁻⁷**.

For the 7-projection/2-prompt first-block CPU test:
- reference scalar-bitplane C: ~4.336 s
- NEON + Accelerate tiled C: ~0.122 s
- approximate improvement: **35.5× compared with this scalar compressed C**, not vs optimized BF16 CPU and not LLM tokens/s.

## Apple GPU real packed Metal proof

The research adapter follows the existing `mlx_moe.cpp` kernel's exact two-level 64-thread simd-group reduction. True uint8 bitplanes and one F32 scale per group reside on Metal; **dense dequantized GPU weights are never uploaded**. For n=6 it uses explicit experimental decoding, since BEGLIN's currently checked-in C++ n6 dispatch is a separate MLX-native repacked path.

- MLX/Metal synthetic C-vs-GPU parity: **6/6 PASS**, maximum relative L2 **1.343×10⁻⁷**.
- Real DeepSeek BF16 hidden-state inputs, all 7 packed Metal projections: maximum GPU-vs-CPU C relative error **1.256×10⁻⁷**.
- Three previously unseen validation texts; original BF16 first-block output-relative L2: uniform **0.013152**, mixed **0.011568**, **12.04%** reduction.
- Same-process alternating timing, 7 pairs of the same 3 short texts: original BF16 PyTorch CPU median **0.094618 s**, uniform packed Metal+CPU hybrid **0.040735 s**, mixed packed Metal+CPU hybrid **0.040230 s**. This is **2.35× faster than that run's BF16 CPU baseline**; another experiment measured BF16 ~0.044 s, showing context sensitivity. Do not generalize timing to standalone GPU full-model tokens/s.

## Measured MLX GPU weight-buffer residency (three separate processes)

MLX active GPU memory recorded after all 7 real BF16 tensor weights were bound and evaluated. This is a **weight-buffer-only resident-memory** measurement; excludes activations, KV cache and inference peak.

| Weight representation | GPU active resident memory bytes |
|---|---:|
| BF16 | 162,004,992 |
| Uniform qNg64 7-bit | 75,972,608 |
| Mixed 9/9/9/7/6/7/7 | **75,563,008** |

Mixed weights show **~53.36% less active GPU memory than BF16**, ~0.54% less than uniform n7. For the two compressed arms, the measured buffer overhead was 32,768 bytes above the exact serialized code+scale payload.

## Source and release restrictions

- Existing BEGLIN `mlx_moe.cpp` research-branch Git blob SHA: `b6cbc1c98cfedcd0f3c99f437d1f6c5978f2ce5a`.
- Existing C `qng64_dense.c` checked SHA-256: `cb985ec48cfa1f1f9ebe8411d12da5d3f73720c8acdcad51fc279a2b25ff8956`.
- The numerical tests ran on an **isolated research copy**, not verified identical to the running XOX serving binary. No official XOX original-file issuer or source receipt was obtained.
- No 27-layer packed inference, actual final-token logits/NLL, task benchmark, final decode tokens/s, true end-to-end compressed GPU peak memory, or KV/weight epoch evidence.
- Mixed role policy comes from the small first-layer calibration/holdout experiment; do NOT treat it as a production-ready model-wide bit policy.
- This PR is draft and the experimental modules are not connected to BEGLIN serving routes.

## Implementation artifacts

See `research/local_quant_cpu_metal_20261017/` for the reusable packed Metal research adapter, GPU hardware tests, and the Accelerate C prototype. All remain opt-in.
