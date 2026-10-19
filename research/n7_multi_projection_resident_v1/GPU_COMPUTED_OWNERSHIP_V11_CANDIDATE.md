# Experimental BEGLIN n7 GPU-computed ownership (v11)

This is a **research-only candidate** for the Apple Silicon C++/MLX qNg64 n7/group64 multi-matrix GPU registry. It is NOT a production or automatic-live-promotion implementation.

## Purpose

The already-certified multi-matrix v10 registry retained one additional immutable CPU packed bitplane+scale copy **inside the C++ GPU handle object**, because an earlier naïve MLX GPU copy could still reference temporary host buffers after their lifetime ended. The v11 candidate instead makes the compact GPU arrays a **materialized GPU computation** (elementwise addition to GPU-allocated zero tensors) and forces evaluation and round-trip byte comparison before allowing the temporary C++ source vectors to be destroyed.

The C++ object then retains only two MLX compact device arrays, not its own additional source CPU vectors. **The original caller/Python model still owns its separate n7 packed arrays, MLX may cache allocations, and the entire model is not exclusively GPU-resident.** No total CPU RSS or model-wide memory saving is claimed from this change alone.

## Portable files and execution

- `n7_multi_projection_gpu_residency_v11_gpuowned_candidate.cpp` — genuine MLX/Metal compressed qNg64 GEMV with candidate GPU-computed ownership, 256 MiB/128-handle bounded registry.
- `test_n7_GPU_only_ownership_v11_portable.py` — no-checkpoint synthetic negative tests, 6 n7 matrix geometries, original C quantizer/matmul reference, source-host mutation/allocator pressure, cleanup.
- `build_and_test_gpuowned_v11_candidate.sh` — compile original BEGLIN native C and candidate MLX C++, then run the synthetic QA in **three isolated new Python processes**, each with fresh immutable receipts.

From repository root on a real Apple Silicon macOS host with the MLX C++ headers and dylib installed:

```bash
BEGLIN_PYTHON=python3 bash research/n7_multi_projection_resident_v1/build_and_test_gpuowned_v11_candidate.sh
```

This is **opt-in**: the existing conservative v10 dual-backed `build_and_test.sh` remains the baseline. Do not change its default or deploy v11 to serving.

## Measured synthetic physical Metal evidence

Three independent cold-start process tests all returned `PASS_GENERAL_MULTIPLE_QNG64_N7_DEVICE_RESIDENT_GEMV` on EOE. Each evaluated six compressed matrices of geometries 128×512, 512×512, 2048×512, 4096×512, 2048×2048 and 4096×2048.

| Check | Outcome |
| --- | --- |
| Original compressed qNg64 planes + FP32 scales | **15,052,800 bytes** total per process |
| Observed MLX active GPU allocator bytes during six-matrix registration | **15,060,992 B** per process |
| Maximum source packed GPU vs independent native-C relative L2 | **1.2930364961681175×10⁻⁷** |
| Three fresh-process C++ registries, actual Metal dispatch and teardown | **3/3 PASS** |
| Host buffer mutation and allocator churn after registration | **PASS** |
| Wrong bitwidth/shape/scales/batch, unknown/stale handle rejection | **PASS** |
| Registered handles after cleanup | **0** |
| Original model checkpoint in portable tests | **Not accessed** |

Three immutable private-run receipt SHA-256 hashes, from an audited cold-process test batch:
- `f8a8aa48d2c56b76fadfb1cbb5930bbda04e0d6e8ae2cfc53ccbf052ca961822`
- `e8cdd9825870b55a8f91f6fd53e05c664d3fac8317dcdb089d52ce044249d1c1`
- `f04de37144a6356410f34248ad2d8787d7d064ddce47f2fe9cdf40e8d925325b`

## Separate private original-model validation — not part of the portable package

Independent EOE evidence compared this candidate against a separate previously verified v10 CPU+GPU dual-backed implementation using the same source-attested D50 BF16 DeepSeek-V2-Lite checkpoint and frozen original n7 policy.

- Original **layer 0**: seven qNg64 roles registered; source direct-C reference maximum relative L2 ≈ **1.2599×10⁻⁷**; full source decoder output at 5 and 6 tokens remained BF16 bitwise identical to independently compiled original C++/Metal.
- Original **layer 1 MoE**: 22 distinct experts selected from five tokens; **73 original qNg64 projection roles**, **207,544,320 B** compact GPU payload, **73** registrations, **73/73** reuse hits in the second same-input pass, **zero evictions**, original BF16 model output relative L2 **0.0**.
- The source model, tensor arrays, prompt, hidden states and expert identities remain private on EOE. Audit `PASS_REAL_73_ROLE_GPU_HANDLE_SOURCE_WITHOUT_DUPLICATE_CXX_HOST_BACKING` confirmed C++-registry backing semantics, not general model GPU-only residency.

Private original-model SHA-256 receipts: candidate layer-0 `a655e7bd619a566ec8b26e5067c41ec4a4b72e9c8b0991d49934004f1a6a8865`; candidate layer-1 MoE `ef87be8e0169f64abfc3da28f89ae8eab6795c198fe10b34eaee2c647b5be913`; layer-1 independent audit `187b43f927f8c71001102f300d36666229b478acfc8903918593761b06e62683`.

## Release blockers

This is still an **experimental device-buffer-lifetime fix** and does not certify memory safety under arbitrary driver/MLX versions, concurrent registration with GPU in-flight eviction, OS sleep/resume, long sessions, memory pressure or restarts, or multiple original-model prompt distributions. The C ABI requires trusted capacity-verified source pointers. Multi-layer multi-token latency, full CPU+GPU RSS, task quality, TTFT/TPOT and production precision policy are unverified.

Preserve v10 as fallback, keep PR Draft, XOX serving unchanged and automatic GPU/live promotion OFF.
