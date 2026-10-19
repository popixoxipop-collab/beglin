# BEGLIN research: device-resident qNg64 n7 MLA Metal cache

This **isolated, research-only** package verifies that Apple's MLX Metal backend can own a native BEGLIN `qNg64` (n7, group64) packed `kv_b_proj` buffer for repeated absorbed-MLA attention queries without re-reading or re-uploading those *weight buffers* on each query.

It does **not** deploy to serving, change model policies, load private checkpoint data, or establish full-model inference speed.

## Files

| File | Purpose |
| --- | --- |
| `mla_qng64_metal_absorbed_staged_v1.cpp` | Independent baseline staged qNg64 compressed absorption kernels |
| `mla_qng64_metal_single_fused_v1.cpp` | One physical Metal threadgroup kernel per head, no expanded per-head K/V |
| `mla_qng64_resident_weight_handle_v1.cpp` | C++ process-local opaque handle registry for GPU-owned **compressed** qNg64 planes and FP32 group scales |
| `test_qng64_resident_gpu_portable_v1.py` | Real Metal synthetic compressed-weight parity, ownership, failure-mode and reuse regression |
| `build_and_test.sh` | Compiles the original BEGLIN C n7 quantizer plus Metal libraries and runs the test |

The resident source includes the single fused source, which includes the staged source, so only `mla_qng64_resident_weight_handle_v1.cpp` is compiled for the resident library.

## Reproduce

Requires **Apple Silicon macOS**, Xcode CLT, an actual Metal GPU, a compatible `mlx` Python installation that includes its C++ header and `libmlx.dylib`, plus NumPy. Run from a repo containing the original root `qng64_dense.c/.h`:

```bash
BEGLIN_PYTHON=python3 bash research/native_mla_gpu_resident_v1/build_and_test.sh
```

No downloads, network/API keys, model payloads, or original model execution. The build outputs and test receipts are generated under local `build/` and must **not** be committed as prebuilt binaries.

## Physically verified EOE synthetic scope

- Source weight geometry: **4096 × 512** (16 heads, key-nope 128 rows + value 128 rows per head), n7/g64.
- Compressed uint8 bitplanes: **1,835,008 bytes**; FP32 scales: **131,072 bytes**; GPU-owned total **1,966,080 bytes** for this *one* `kv_b` matrix.
- GPU MLX allocator active memory increased from zero by exactly the registered packed weight+scale byte count before temporary query buffers. The source is device-owned via explicit `mx::copy(..., gpu)` and `mx::eval`, with GPU-to-CPU byte verification.
- Seven actual-Metal query lengths **1, 2, 5, 7, 16, 32, 64**: source packed device-resident and original single fused Metal output **bitwise identical**; all output comparisons relative L2 **0.0**.
- Mutating the caller's CPU source planes and scale buffers after registration did not change resident GPU output.
- Incorrect n7 bitwidth/source-scale and unregistered or already-freed handle attempts rejected. Device registry count returned zero after releasing the only handle.
- Eight same 7-token source-shaped **synthetic** GPU-resident calls had median host-observed wall time around **0.879 ms** in one isolated test; this measures only the fused attention probe including dispatch/host boundary, **not** layer or full-model TTFT, TPOT or sustained tokens/s.

## Real model evidence is separate

An independent EOE research run also attested original SHA-verified DeepSeek-V2-Lite **layer-0** BF16 source tensors and packed `kv_b` from the D50 research copy. It registered the genuine n7 matrix GPU-resident and compared results to the original one-shot single Metal fused kernel (relative L2 **0.0**) and BF16 source-expanded attention (relative L2 approx **0.003345**). The original checkpoint, weights and hidden states were not committed.

Any whole 27-layer GPU-resident source experiment is a **separate test**; none of this portable package certifies 5,181 matrices resident or production serving throughput.

## Safety boundary

The opaque C ABI cannot derive raw caller pointer capacities; future integration must admit pointer lengths through a SHA/manifest-verified trusted model loader. This prototype accepts only the exact n7/g64 4096×512 `kv_b` geometry, at most 64 key positions, and single-head-query-per-head F32 intermediates. It does not implement general mask, batch, model-wide device memory management, concurrent production scheduling, GPU MoE routing or a complete native-C++ DeepSeek decoder.

**Promotion/merge remains disabled pending independent real-model multi-layer, multi-token, long-context accuracy, real device-memory peak and throughput tests.**
