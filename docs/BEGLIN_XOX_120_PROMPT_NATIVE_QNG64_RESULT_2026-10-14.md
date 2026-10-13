# Beglin XOX — 120-prompt learned Micro-Expert + native qNg64 local verification (2026-10-14)

## Overall verdict

**120-prompt offline study: SUCCESS / PROMOTION BLOCKED.**
**Native real-weight CPU qNg64 local projection: SUCCESS / WHOLE-MODEL TOKEN PARITY NOT TESTED.**
Original `main`, deployed serving, precision policies, and checkpoint bytes were not changed.

## Source identities and execution

- Research branch: `research/trainable-micro-router-120prompt-20261014`.
- XOX 120-prompt run: https://github.com/popixoxipop-collab/beglin/actions/runs/38430695333 ; executed commit `1c645b0b2e30b880011f9844625345c42584194b`.
- qNg64/local CPU cost run: https://github.com/popixoxipop-collab/beglin/actions/runs/38433066380 ; executed commit `f5aa406eabbe257148f03f5b81acc8910f01d9a8`.
- Checkpoint index SHA-256 `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`.
- 120-prompt result SHA-256 `f11382df3032144a201ced454a65245dda68f401b7f0e78c7724f0d60382ff1a`.
- qNg64 local projection result SHA-256 `2ea0db8911908d725995b36aec95b14ef0d5b1e543d77515d703aeeb6590ca8b`.
- Both downloaded evidence ZIP archives were independently checked against their respective receipt JSON hashes, PASS.

## Data and evaluation design

120 unique, independently executed prompts from 12 subject domains, balanced by domain across training/validation/holdout: **72 / 24 / 24 prompts**. All 120 were tokenized locally without network and had distinct token ID sequences. Total actual DeepSeek inputs: **1,555 token positions**, broken down as **945 / 308 / 302**. Each prompt ran in a separate isolated XOX process. Actual layer-1 post-attention/pre-FFN `h2` inputs and real macro Top-K routes were used; 63/64 routed experts appeared.

The weights for offline channel FFN evaluation came from BF16 Safetensors, whereas `h2` arose from a quantized AF forward. Thus this is **not** native matched-source token prediction, and the results do **not** establish full-model perplexity, accuracy, throughput, or QoS.

Validation prompts selected among methods, with final holdout never consulted in model training or method selection. Prompt-block paired bootstrap compared holdout improvements to static selection. There were **zero** full-expert fallback pairs in the 3,660 routed expert-token pairs over validation+holdout.

## Real holdout output relative L2 (lower is better)

| Channel width | Retained | Static | Learned selection | Learned + mean correction | Learned + rank-8 | Learned + full Ridge |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 25% | 0.645499 | 0.590213 | 0.563602 | 0.538218 | **0.531997** |
| 16 | 50% | 0.477629 | 0.439451 | 0.427979 | 0.414971 | **0.411714** |
| 16 | 75% | 0.311658 | 0.285272 | 0.281793 | 0.275664 | **0.274496** |
| 64 | 25% | 0.687372 | 0.637215 | 0.600228 | 0.560709 | **0.552280** |
| 64 | 50% | 0.525849 | 0.485747 | 0.467079 | 0.442970 | **0.437958** |
| 64 | 75% | 0.324163 | 0.300383 | 0.294889 | 0.283662 | **0.281725** |

Primary registered configuration: 16-neuron channels, 50% retention. Against static, full Ridge reduced aggregate relative L2 by **13.80%**, rank-8 by **13.12%**; prompt-block paired bootstrap 95% intervals **11.04–16.56%** and **10.52–15.77%**, respectively. Both intervals describe variability of the 24 measured prompt blocks, not general task accuracy or the world population. Full Ridge had the lowest validation error, so it remained the validation-selected winner.

At primary configuration, approximated resident FP32 model storage was **141,880,032 bytes** for full Ridge and **99,726,880 bytes** for rank-8, a **29.71%** reduction with holdout error increasing from 0.411714 to 0.414971. These are analytic FP32 coefficient byte counts, **not actual RSS**. The 50% selected-channel fraction is idealized and excludes selector, residual, memory movement, shared experts, attention, and other model runtime costs.

## Native Beglin qNg64 CPU projection (separate local experiment)

Actual DeepSeek-V2-Lite layer-1 expert-0 `gate_proj` weight matrix, shape 1,408 × 2,048. The CPU probe compiled the repository's actual `qng64_dense.c` into a scratch-only library and evaluated a seeded **synthetic** input vector against the unquantized BF16-file matrix. The qNg64 dense format used in this probe does not implement n=4; the engine has other int4 encodings.

| qNg64 bit width | Relative projection L2 vs source | Median CPU matvec | Packed plus scales |
|---:|---:|---:|---:|
| 3 | 0.378768 | 5.719 ms | 1,261,568 bytes |
| 5 | 0.077053 | 7.564 ms | 1,982,464 bytes |
| 7 | 0.018096 | 9.776 ms | 2,703,360 bytes |
| 9 | 0.004313 | 12.491 ms | 3,424,256 bytes |

This verifies the real native CPU quantization/matvec path for one projection, not CPU/GPU parity, compressed inference across all projections, or any token outputs.

### Separate XOX CPU predictor-only microbenchmark

Representative shape: train examples 80, hidden 2,048, channels 88, rank 8, 250 timing repetitions, synthetic features. Median NumPy CPU prediction time (includes the gate, excludes the actual micro FFN): gate-only **0.03019 ms**, gate + rank8 **0.05400 ms**, gate + full Ridge **0.07721 ms**. The rank8 predictor-only path was **30.06% faster** than full Ridge under this shape-matched synthetic benchmark. Correction model byte counts: full Ridge 2,637,824, rank8 1,463,360; gate 1,367,744. Neither measured real-serving latency nor GPU throughput.

## Safe next step / blockers

**NO GO for production or a micro-MoE serving cutover.** Primary output error **0.4117** is too large for a drop-in expert FFN replacement. The study has no native qNg64-aligned Micro-Router execution, whole-model token-logit/perplexity/task A/B, or measured combined CPU/GPU latency+memory. The current model still requires full expert outputs for training targets; this is an offline supervised proof of concept.

Next meaningful experiment: train a *source-matched*, sensitivity-weighted router/corrector using the exact native quantized expert representation, then evaluate candidate outputs on independent prompts. Only if a materially lower FFN output error and measured E2E cost win are demonstrated should an isolated native-token A/B be implemented and inspected. Keep `automatic_live_promotion=false` throughout.

## Artifacts

- 120-prompt evidence artifact: https://github.com/popixoxipop-collab/beglin/actions/runs/38430695333/artifacts/11830574763
- qNg64 CPU/cost evidence artifact: https://github.com/popixoxipop-collab/beglin/actions/runs/38433066380/artifacts/11831358129
- Preregistered method/split contract: `docs/BEGLIN_MICRO_ROUTER_120_PROMPT_PROTOCOL_20261014.md`.
