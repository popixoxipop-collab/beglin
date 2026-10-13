# Beglin — Trainable Micro-Expert Router and Residual Corrector (2026-10-13)

## Status

**IMPLEMENTED / UNIT CI PASS / REAL PRETRAINED EVALUATION NOT RUN.**
Research only, not production ready. Main and the live inference service remain unchanged.

## Actual earlier XOX baseline

- Scratch run: https://github.com/popixoxipop-collab/beglin/actions/runs/38396401476
- 8 actual layer-1 pre-FFN h2 inputs and actual Top-K routes from DeepSeek-V2-Lite quantized AF forward.
- Within one fixed 8-token verification sequence: four calibration positions, four heldout positions.
- Offline 16-neuron static selection: 25% retained => relative L2 0.784850; 50% => 0.548728; 75% => 0.365897.
- Offline 64-neuron static 50% selection => relative L2 0.586682.
- This compares actual AF hidden inputs with BF16 expert FFN weights, not a matched native qNg64 forward. No task accuracy or speed claim.
- Original actual hidden states were never uploaded, only aggregate receipts and hashes.

## New learning architecture

Code: experiments/hierarchical_tupper_moe/trainable_micro_router.py

Each routed macro-expert is decomposed into neuron-channel micro-experts. Offline dense teacher labels for each channel are the normalized L2 magnitudes on calibration inputs only. A per-expert linear kernel ridge gate (regularization 0.25) predicts channel scores from normalized h2 and selects top-K channels. This is not a privileged full-channel oracle at inference time.

An optional residual corrector learns the omitted channel contribution on calibration positions. It compares constant mean residual with kernel ridge reconstruction (regularization 2.0). Evaluation includes static, static-plus-mean, dynamic, dynamic-plus-mean, dynamic-plus-ridge, full-reference and oracle diagnostics. Tupper NxM encoding is merely lossless mask indexing, not parameter compression.

Teacher calibration computes counterfactual full expert outputs for experts not selected by the actual macro-router; this is an offline training cost, not a claim about serving efficiency.

## Verified tests

- GitHub Actions: https://github.com/popixoxipop-collab/beglin/actions/runs/38417915169 — SUCCESS, five tests.
- Exact complete-grid selection, target-label leakage guards, ridge fit using only train examples, fake Safetensors end-to-end smoke, invalid-parameter rejection.
- Code branch: research/trainable-micro-router-20261013, based on prior research head 6dd9776e6b6dca89ad067895f6f9dd9be87829e3.
- Real XOX learned-gate experiment was NOT executed. The previous raw capture was not exported, and a new workflow registration was blocked by a tool safety check.

## Acceptance requirements (not satisfied)

1. Collect many more actual token activations from independent prompts in an explicitly authorized isolated workflow, with pinned hashes. Raw h2 must stay local unless separately authorized.
2. Split by prompt, not adjacent token positions, into train/validation/holdout. Never tune on final holdout.
3. Compare static, learned, residual-corrected, random, and full-channel references for 25/50/75/100 percent selected channels; attach uncertainty intervals.
4. Eliminate BF16-vs-AF source mismatch before claiming native model quality. Use exact checkpoint and kernel representation.
5. Require token-logit, routing, task-metric, GPU/CPU actual latency and memory evidence. Compute savings must include router and residual overhead.
6. Fail closed: no automatic live precision promotion, no production or main changes.
