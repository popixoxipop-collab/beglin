# OML–ACS–BEGLIN integrated research plan (2026-10-21)

Status: **proposal / preregistration draft; no experiment executed**. Independent verdicts: A, B, C1, C2, Q. Do not reinterpret old V6 negative results or reuse its 24 consumed holdouts as fresh test data.

## Question
Do cross-family anchor-projected behavioral directions transfer, and does source reliability plus correlated error weighting improve transfer and answer combination on unseen tasks, including under quantization?

## Source audit
Read current `README.md` in `popixoxipop-collab/beglin` (main) and inspected `RESULTS.md` metadata. README documents vDSP/NEON CPU, optional MLX/Metal, tensor/expert-specific qNg64 bit precision and MoE serving. **This does not establish hidden-state hook / steering adapter support in every candidate model.** Implement and test a separate Transformers/PyTorch baseline first. Existing V6 audit indicates VERIFIED_NO_HIERARCHICAL_GAIN for 24 holdouts; do not overwrite or rebrand it.

## A — calibrated answer combination
Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct, Phi-4-mini-instruct; candidate-string conditional log-likelihood (fixed multi-token scoring and normalization), not raw vocabulary logits. BoolQ, ARC-Challenge, CommonsenseQA. Baselines: strongest single, uniform, calibrated uniform, weighted, input router. Estimate residual covariance using out-of-fold train predictions; never use held-out labels for weights. Primary: NLL, secondary accuracy/ECE and full compute cost. Draft adoption threshold >=2% relative NLL gain over strongest simple baseline with cluster-resampled 95% interval favoring improvement and accuracy non-inferiority (margin must be frozen).

## B — ACS reproduction
Reference: Su-Hyeon Kim & Yo-Sub Han, *Cross-Family Universality of Behavioral Axes via Anchor-Projected Representations*, arXiv:2605.09875v1. Extract matched anchor activations, center/normalize per paper, learn positive-negative behavioral directions on source models, average in anchor coordinates, project to held-out target using target anchors only. 300 anchors proposed; first sentiment/emotion, later sycophancy. Controls: no intervention, matched-norm random direction, shuffled target anchor order, system prompt, target-supervised oracle upper bound separately. Distinct endpoints: representation alignment, AUROC detection, generation steering.

## C — incremental hypotheses
C1: weighted direction c_b=sum_m w_{m,b}u_{m,b}; compare uniform, best source, reliability-only, reliability+source-error-correlation. No target labels for tuning. C2: output-only vs ACS-only vs combined features; show incremental held-out NLL value beyond output-only. C1 target AUROC +0.02 OR preregistered steering gain; C2 held-out NLL gain.

## Q — quantization robustness
After A/B/C baselines, compare BF16, genuine supported 8-bit and packed 4-bit. Frozen BF16 axis/classifier/weights vs refreshed anchors only (without target behavior labels). Measure alignment, AUROC, generation steering, NLL, wall time, peak memory, tokens/s and total inference cost separately. Draft tolerance: AUROC loss <=0.03, NLL rise <=2%. Fake quantization must not be reported as packed-kernel performance. Dynamic precision allocator and MoE caching are out of scope for first pass.

## Splits and budget (provisional, not frozen)
A pilot 360, train 900, dev 450, calibration 450, IID final 600, OOD final 600; disjoint by semantic/document clusters. B per axis: train 100 contrast pairs, dev 50, final 100. Three-model leave-one-family-out pilot, then SmolLM2 and Granite sources and Gemma-2-2B final target. Freeze model revisions, tokenizers, prompt formats, seeds, layer/hook points, data IDs, anchor IDs, exclusion rules and hashes before final. Pilot accelerator <=8h, subsequent <=40h; no paid API or new hardware budget.

## Gates
G0 freeze source/paper contracts and data manifests; G1 baseline ACS + output averaging with actual generations; G2 reliability/covariance; G3 weighted ACS and combined signals; G4 quantization/BEGLIN adapter if compatible; G5 independent locked evaluation and separate accept/reject/inconclusive decisions. If only AUROC rises without steering, do not claim transfer. If internal features fail output-only baseline, do not claim utility. If full cost erases accuracy gains, do not claim efficiency. Do not infer universal manifold from three models.

## Source/code checks still needed
1. Verify arXiv paper equations and anchor/steering details against actual full text before marking exact reproduction.
2. Inspect actual model-specific hooks and code contracts; README is not sufficient evidence of implementation.
3. Generate immutable data manifests and preregistration hash before execution.
4. Verify V6 holdout identities to exclude overlaps.

This branch is documentation-only. No merge, deployment, release, model evaluation, or production change.
