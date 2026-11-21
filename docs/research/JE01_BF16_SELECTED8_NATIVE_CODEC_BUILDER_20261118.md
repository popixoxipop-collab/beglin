# JE01 — BF16 selected-8 native 2/6/8 bit research artifact builder

2026-11-18 KST. Research-only source addition; original GPU and serving workers untouched.

## New implementation
- `tools/je01_selected_bf16_native_artifacts.py`: strict 8-Expert/24 role loader, sourced from original pinned DeepSeek-V2-Lite BF16 safetensors; checks model index/config and full SHA of every original shard, using previously sealed source attestation. Uses existing `gguf_transcode.c` **pinned Git blob** `ff739d886f55b5a35843cf82fb55c9cd8f76104f` (qNg64 n2/n6 or signed q8g64 n8) compiled in an isolated scratch directory. Reads actual BF16 tensor bytes, invokes those native C quantizers, independently decodes emitted bit-planes/int8 and calculates per-role BF16 reconstruction RMSE. Writes new-only packed+f32-scale files with physical byte sizes and SHA, emits canonical JSON receipt. No output is allowed outside new `/Volumes/D50/TailnetCommander/work/<new>` scratch directory on EOE.
- `tests/je01_selected_bf16_native_artifacts_integration.py`: independent synthetic input tests for native C 2/6/8 paths, signed q8 format, plane endian/decode semantics, BF16 conversion, sizes and fail-closed errors.
- CI: `.github/workflows/je01-selected-bf16-native-codec.yml` with pinned NumPy and native C compiler.

## Scientific gates, unchanged
The original AF eight selected slots (21,61), (24,26), (13,28), (20,17), (12,13), (24,60), (14,63), (13,22) were selected by original 256-route calibration. The earlier Layer11 BF16 synthetic proxy selected DIFFERENT Experts. New source uses the correct eight (layer,Expert) identities.

Outputs from the builder will be actual BF16-source 2/6/8 quantizer artifacts **only after its physical execution and signed receipt**. The three all-tier byte sums are not an actual matched-byte joint assignment. Matching the original AF source semantics, per-index routed GPU overlay, paired gold-token NLL, heldout/ood and p95 latency all remain required and cannot be inferred from codec roundtrip loss.

Before native execution, verify admission via Tailnet Commander; minimum system memory free 35%; preserve XOX original serving processes, no deployed files or production release changes. At last resource probe EOE D50 storage available but reservable CPU 0; XOX internal scratch below storage reserve. No unadmitted run should be performed.

## Example when EOE CPU admission permits
```sh
python tools/je01_selected_bf16_native_artifacts.py --out-dir /Volumes/D50/TailnetCommander/work/JE01_BF16_SELECTED8_NBIT_20261118
```
Only run from the exact JE01 research checkout with the pinned C source blob and sealed BF16 source present. A new directory is mandatory; never overwrite or use original model files. **No model-level Bayesian improvement claim without independent paired NLL.**

## Independent source race prevention and memory gate
Real builder now uses the exact same open file descriptor for per-shard SHA verification and tensor extraction (eliminates a re-open TOCTOU window), and explicitly checks macOS memory_pressure -Q >=35% before full extraction and before each Expert's three roles. CI includes a synthetic shard descriptor-lifetime regression, malicious hash rejection, and 35/34% sensor tests; these tests do not replace actual EOE execution.
