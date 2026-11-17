# JE01 — Actual routed-MoE Expert Bayesian precision allocation (BEGLIN)

**Preregistration / new independent research, 2026-11-17 KST.**
**Research code only. No native MoE perturbation observations, no checkpoint, Router or production edits.**
**Default decision: HOLD / preserve baseline.**

## Why this is a distinct research branch

The Joint-Expert Bayesian B3/B4 study did NOT demonstrate useful Expert cross-covariance (0/6 supported pairs); B5 independent Expert posterior maps passed synthetic/numerical checks, but were not pretrained-weight posterior distributions. B9/B10 Student-t and conformal calibration tested image-model feature uncertainty, not BEGLIN.

The BEGLIN Qv21 experiment, although actually run in original Qwen28 C, involved **two precision masks** on a dense Qwen2.5-1.5B model, not routed MoE Experts. Its 96-case native 5-fold gate officially FAILED: source NLL 1.679052, random73 NLL 1.808085, dynamic 1.691013, selected severe loss 3/18. See [Qv21 original native negative audit in B11 Draft #93](https://github.com/popixoxipop-collab/beglin/pull/93). JE01 must not reuse its calibration/evaluation as JE01 validation.

BEGLIN's README and main engine document per-(layer, expert_id) precision selection, routed FFN role quantization and real DeepSeek-V2-Lite E=64 top-k gather. **JE01 targets those actual routed Experts**, not Qv21 mask variants.

## Hypothesis and parameterization

For routed Expert e and measured byte tier b, estimate the distribution of per-case signed gold-token NLL perturbation relative to the immutable baseline assignment. Use **one independent Normal-Inverse-Gamma scalar posterior per (e,b)**, with prior mean 0, kappa 1, alpha 2, beta 1. The predictive distribution is Student-t. This is only a posterior over the *local paired NLL effect estimate*, NOT an epistemic posterior over the pretrained neural weights.

Each alternative is scored conservatively with Bayesian location + 2 × Student-t predictive scale; this 2-scale term is a *fixed risk surrogate* (NOT a calibrated 95% confidence limit without independent validation). If a candidate has fewer than 32 independent routed calibration observations, missing paired NLL, nonfinite values, or any observed router flip, mark the tier ineligible.

Optimization is exact-byte multiple-choice DP over at most **8 frozen selected Expert slots**:

- Each Expert chooses one of its actually measured packed-bit tiers;
- sum of measured selected packed bytes must **exactly equal** immutable baseline bytes for those same slots;
- all other model weights, Expert slots and layer routing remain unchanged;
- minimize the sum of independent Expert conservative loss scores; require predicted reduction at least 0.15 NLL (pilot screen only);
- no cross-Expert covariance or additive model quality claim is made until full MoE-forward validation.

The pure planner is in tools/je01_bayesian_expert_allocator.py and is intentionally capped at eight slots. It does not run inference, edit precision maps, or activate code. Decision output is always SHADOW_ONLY_NO_WEIGHT_EDIT.

## P0 — native grounding, not yet executed

Pin exact full-model SHA, tokenizers, BEGLIN engine commit, compiled reference binary, backend (CPU/Metal), rotary/routing configuration, output-logit parity, baseline per-slot packed bytes, and data source hashes. Proposed real model family is BEGLIN-supported DeepSeek-V2-Lite MoE, but **the exact checkpoint artifact has not been pinned in JE01**.

A separate, isolated instrumentation build must collect actual top-K routed Expert IDs, pre-change router score gaps, Expert activation norms and per-question gold-token NLL with no modification to the immutable baseline path. No arbitrary high-level "Expert trust" score is accepted as a native observation. Token/logit parity with original backend must PASS before training candidates.

Freeze the **8 pilot Expert slots by pre-label routing frequency on a new unlabeled calibration split**, not by whichever candidate later improves loss. If any missing pinned source, model, byte measurement, parity receipt or routing trace: STOP.

## P1 — new evidence and split

Proposed, to be frozen before any native scoring:
- Source model inputs: independent clean text prompts from WikiText-103, not Qv20/Qv21 ARC/OpenBookQA cases.
- Calibration for Expert-effect posterior: 256 unique inputs, with each tested (e,b) requiring at least 32 real routed Expert activations. Unmeasured tiers remain ineligible.
- Heldout same-source: 128 different inputs. Cross-source OOD: 128 from HellaSwag validation (scored on compatible gold-token NLL construction).
- Source file bytes and content ID/overlap hashes, tokenizer construction, per-alt exact packed byte receipts, seeds and resulting cohort IDs must be immutable before gold results.
- Independent auditor must verify each paired baseline/candidate native score and that the candidate Expert was actually routed on that prompt. No oracle selection using heldout labels.

These named source datasets and counts are **proposed and NOT yet SHA-frozen**. If the exact source partitions lack enough routed observations, record NOT_FEASIBLE rather than silently adjusting thresholds or using the old Qv21 heldout.

## P2 — full pretrained model validation required

Compare (a) frozen baseline bit map, (b) independent Bayesian DP shadow plan, (c) equal-byte random-permutation control, (d) constant-tier frequency heuristic. Run **actual full pretrained MoE forward**, not an additive surrogate alone, on all heldout and OOD samples. Apply identical question IDs, prompt tokens, backend, generation/scoring mode, CPU/thread count and permitted memory ceiling; measure both full-vocabulary paired gold-token NLL and router top-K flip counts. GPU latency must be measured independently under equivalent concurrent load.

Adoption proposal, ALL required:
1. Every source and checkpoint SHA, sampled slot ID and exact packed byte identity independently audited.
2. End-to-end heldout NLL mean improvement against baseline ≥0.02, paired bootstrap 20,000 repeats lower 95% bound >0.
3. OOD 128 mean NLL non-regressing, paired bootstrap lower 95% bound for gain >=0.
4. Zero observed hard router top-K mismatches relative to baseline, no NaNs/unsupported kernel paths.
5. No worse than 5% p95 token latency under hardware-parity conditions.
6. Predicted gain must not be explained by equal-byte random-permutation placebo; familywise multiple-testing correction for any selection among multiple plans.
7. Independent reviewer certifies measurements BEFORE any production or operating-release change.

If any gate fails, keep unchanged baseline. Never use this pilot planner to perform a production release activation. B12/JE02 cross-Expert covariance is out of scope until co-routing evidence and negative controls support nonzero interaction. This is intentionally a **new preregistration**, separate from the negative Qv21 and the B9/B10 image-model results.

## Current engineering artifact

Research-only implementation: tools/je01_bayesian_expert_allocator.py.
Synthetic contracts: tests/test_je01_bayesian_expert_allocator.py.
This code can produce an exact-byte proposed bit layout with independent Student-t risk scores on synthetic observations. Synthetic CI is **not** native-model evidence.


## 2026-11-17 P0 audit follow-up (observations, not a change to preregistered test gates)

DeepSeek-V2-Lite's official [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json) specifies 64 routed Experts with Top-6, two always-on shared Experts, a dense first layer, and no normalization of selected Top-K probabilities. BEGLIN's native GPU backend is Apple MLX/Metal; an Alienware CUDA evaluation is a separate reference and **not native BEGLIN parity**.

The BEGLIN eager native interface mlx_gpu_layer_step() can return the selected Expert IDs and weights, and mlx_gpu_risk_signal_telemetry_get() returns a risk summary. But these existing outputs alone cannot reconstruct all 64 pre-Top-K scores. A separate **isolated, non-production instrumented reference build** is still required to emit all 64 softmax values, selected IDs, and SHA-bound per-question records, then show baseline-logit parity before any candidate calibration.

New shadow-only source:
- tools/je01_checkpoint_attest.py: hashes indexed safetensors shards, config, tokenizer, binary, and engine git commit with path traversal and symlink-escape checks. Without externally pinned identities it reports discovery only, never native parity PASS.
- tools/je01_router_trace_audit.py: rejects outcome labels, invalid 64-probability normalizers, top-6 mismatches, duplicate/unfrozen prompts, forged alternative traces; freezes eight pilot Expert slots by independent prompt routing counts, no gold outcomes.
- tools/je01_bayesian_expert_allocator.py: limited to direct expert precision tiers 2/3/4/5/6/7/8/16/32, rejecting layer 0 (dense). The BEGLIN shipped routed Expert precision default is int8; baseline=4 in synthetic fixtures is not a real model baseline assertion.

CI: 43/43 offline synthetic tests PASS at [GitHub Actions run 43983912466](https://github.com/popixoxipop-collab/beglin/actions/runs/43983912466). CI is NOT real DeepSeek model inference.

Fresh Tailnet Commander current/live remained g6-0.9.0-operator.900484. BEGLIN-native Metal GPU resource_plan recommended no host: EOE lacked reservable RAM and CPU; XOX lacked RAM, CPU and eligible storage pool. No GPU job was run.

XOX read-only *admitted* metadata job job_900786c808f71b9c2fe86774941785f6 finished SUCCEEDED, exit 0. Its inspected standard HuggingFace hub contained the matching model name models--mlx-community--Qwen2.5-1.5B-Instruct-4bit, but no DeepSeek name in that directory; this does not prove no DeepSeek exists elsewhere. EOE Hugging Face cache file reads were denied due to a protected symlink component; directory search and CPU work were refused by existing fleet admission. Therefore a local DeepSeek checkpoint, entire safetensors SHA and tokenizer are NOT YET PINNED.

Final state: P0_OFFLINE_GUARDS_PASS / P0_ACTUAL_MODEL_SHA_NOT_VERIFIED / P1_NATIVE_ROUTER_TRACES_NOT_RUN / P2_FULL_FORWARD_OOD_NOT_RUN / NO_PROMOTION.

The original JE01 method, controls, target sample counts 256/128/128, thresholds and safety rules remain unchanged. New scoring is blocked until the scheduler authorizes actual model-cache inventory, the real full-checkpoint identity is sealed, and an isolated native route trace is proven parity-equivalent.
