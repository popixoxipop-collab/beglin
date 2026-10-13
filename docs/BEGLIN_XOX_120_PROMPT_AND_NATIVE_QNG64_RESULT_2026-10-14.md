# Beglin — 120 independent prompt Micro-Expert Router and native qNg64 CPU follow-up

## Result status (2026-10-14)

**XOX ISOLATED EXPERIMENTS: SUCCESS. RESEARCH EVIDENCE ONLY; NOT PROMOTABLE.**

- Training / method selection / blind prompt holdout: **72 / 24 / 24 independently processed text prompts**; respective actual token positions: **945 / 308 / 302**, total **1,555**.
- Capture uses a fresh separate DeepSeek-V2-Lite scratch process per prompt, the local offline tokenizer, actual quantized-AF pre-FFN `h2` vectors, and actual layer-1 macro Top-K routes.
- Offline expert counterfactual is reconstructed with the **BF16 Safetensors expert weights**, not the AF forward weights. Therefore this measures BF16 local macro-FFN reconstruction from actual AF hidden inputs, **not** native engine logits, perplexity, token accuracy, or task scores.
- Holdout micro-router method is locked exclusively on 24 validation prompts. Paired confidence intervals resample final test **prompts**, not individual token positions. No heldout target channels are used in the selection rule.
- Generalized Tupper formula provides an exact NxM *routing-mask* encoding, **not** parameter compression.

## Primary result: layer 1, micro width 16, 50% retained

| Method | Final holdout relative L2 output error | Paired improvement vs static | 95% prompt-bootstrap CI |
|---|---:|---:|---:|
| Fixed/static channel choice | 0.477629 | — | — |
| Learned channel choice | 0.439451 | 7.99% | 5.68–10.30% |
| Learned + rank-8 omitted-output corrector | 0.414971 | **13.12%** | **10.52–15.77%** |
| Learned + full kernel-Ridge omitted-output corrector | **0.411714** | **13.80%** | **11.04–16.56%** |

Validation selected **full Ridge** for this setting, without consulting the final holdout.

The improvement repeats the direction observed in the prior 24-prompt pilot, but the error remains about **41%** for the selected sparse method. This is much too large for serving integration.

### Full configuration sweep (holdout aggregate relative L2)

| Channels | Retention | Static | Learned | Rank-8 | Ridge |
|---:|---:|---:|---:|---:|---:|
| 16 | 25% | 0.645499 | 0.590213 | 0.538218 | **0.531997** |
| 16 | 50% | 0.477629 | 0.439451 | 0.414971 | **0.411714** |
| 16 | 75% | 0.311658 | 0.285272 | 0.275664 | **0.274496** |
| 64 | 25% | 0.687372 | 0.637215 | 0.560709 | **0.552280** |
| 64 | 50% | 0.525849 | 0.485747 | 0.442970 | **0.437958** |
| 64 | 75% | 0.324163 | 0.300383 | 0.283662 | **0.281725** |

- Observed routed expert union: **63/64** for this corpus.
- Full-channel fallback routed pairs: **zero** in the 120-prompt evaluation; channel fraction 50% is idealized and excludes the Router, correction and memory traffic.
- Total FP32 trained-model *estimated* storage, 16-neuron width: learned + rank-8 **99,726,880 bytes** (~99.7 MB decimal), learned + full Ridge **141,880,032 bytes** (~141.9 MB decimal); this does NOT include complete serving runtime buffers or checkpoint storage.
- **Rank-8 reduces the estimate by ~29.7%** vs full Ridge at a modest loss of local FFN accuracy.

### Artifact evidence

- [120-prompt GitHub Actions run #38430695333](https://github.com/popixoxipop-collab/beglin/actions/runs/38430695333): SUCCESS.
- [Original aggregate evidence artifact](https://github.com/popixoxipop-collab/beglin/actions/runs/38430695333/artifacts/11830574763) (raw hidden states and model weights are NOT uploaded).
- Result JSON SHA-256: `f11382df3032144a201ced454a65245dda68f401b7f0e78c7724f0d60382ff1a`. Receipt confirms exact same result SHA, source run commit `1c645b0b2e30b880011f9844625345c42584194b`, 120 unique prompts and final split.
- Checkpoint index SHA-256: `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`.

## Separate native qNg64 CPU projection experiment

The **actual DeepSeek-V2-Lite layer-1 expert-0 gate_proj BF16 weights** (matrix 1408×2048) were run through Beglin `qng64_dense.c` native CPU encoder/decoder and matrix-vector kernel. The matvec input was seeded synthetic, **not actual h2**. No micro-router integration occurred.

| qNg64 bits | Relative L2 vs BF16 projection | CPU median matvec (10 samples) |
|---:|---:|---:|
| 3 | 0.378768 | 5.719 ms |
| 5 | 0.077053 | 7.564 ms |
| 7 | 0.018096 | 9.776 ms |
| 9 | 0.004313 | 12.491 ms |

**Never interpret this as measured per-token inference latency, native AF-vs-BF16 parity, or micro-router accuracy.** It is a single-operator CPU-only result.

### Separate micro-router predictor cost shape study

This study uses 2,048-dim **synthetic feature vectors**, 88 candidate micro channels, 80 train vectors, rank 8. It measures only NumPy CPU-side *prediction* with no Micro-Expert FFN or GPU kernel execution.

| Predictor path | XOX CPU median (250 reps) |
|---|---:|
| Gate alone | 0.0302 ms |
| Gate + rank-8 corrector | 0.0540 ms |
| Gate + full Ridge corrector | 0.0772 ms |

Single-expert corrector resident-array sizes in the shape study: rank-8 1,463,360 bytes, full Ridge 2,637,824 bytes (corrector alone; gating arrays separate). Different values from the 99.7 MB/141.9 MB 120-prompt corpus estimate because those measurements have different train-example counts and expert coverage.

### Evidence

- [Native qNg64 and CPU cost GitHub Actions run #38433066380](https://github.com/popixoxipop-collab/beglin/actions/runs/38433066380): SUCCESS.
- [Aggregated native/cost artifact](https://github.com/popixoxipop-collab/beglin/actions/runs/38433066380/artifacts/11831358129).
- Native qNg64 result JSON SHA-256 `2ea0db8911908d725995b36aec95b14ef0d5b1e543d77515d703aeeb6590ca8b`.
- Rank-8 CPU predictor cost JSON SHA-256 `6d46d85033203bd8ec08675bc37798f2f356ac5a7e61c25377bd24e62170b4a2`.
- Native result receipt SHA match, source `f5aa406eabbe257148f03f5b81acc8910f01d9a8`. No production changes.

## Acceptance gates / next experiment

1. **Not yet eligible** for shadow model-quality acceptance, since local 50%-sparse FFN output error is 0.4117; a provisional screening threshold of 0.05 relative L2 is not met.
2. To claim native token accuracy / GPU speed, integrate sparse channel execution and residual correction into a **scratch-only** end-to-end inference path, preserve original Top-K routing, compare token logits, task metrics, full-e2e per-token latency, memory, and reference/native parity against the **exact same quantized representation**. Prefer explicit stop if the quality gap persists.
3. Check error at retention 90%/95%/99% and whether the corrector can be fused into existing hardware kernels with measured overall benefit. Consider more expressive distillation only after those measurement gates.
4. Keep the repository default main and production untouched; automatic live promotion remains OFF.
