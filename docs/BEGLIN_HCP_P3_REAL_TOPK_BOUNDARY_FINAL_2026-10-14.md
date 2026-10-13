# Beglin HCP P3 — Actual DeepSeek-V2-Lite Top-6/Top-7 Boundary Forecast (2026-10-14)

## Verdict

**REAL XOX EXPERIMENT COMPLETE / TRUE K-K+1 GAPS VERIFIED / NO 2-LEVEL PREDICTIVE GAIN / NATIVE POLICY NOT APPROVED.**

This experiment addresses P2's target mismatch: P2 predicted the normalized entropy *among already selected six Experts* and could not observe their competition with the seventh Expert. P3 measures the genuine **6th selected vs strongest 7th unselected** probability from all 64 router scores immediately after the original full Softmax and TopK selection, but **before** `MOE_NORM_TOPK_PROB` mutates selected probabilities. The actual ranking/forward code is not modified outside a copied scratch research binary.

P3 does NOT reproduce H-JEPA joint world-model training: the study compares static, persistence, flat Ridge, shared rank2 PCA+Ridge, and independent rank2 PCA+Ridge. It is neither learned hierarchical long-term action policy nor proof of whole-model qNg64 inference acceleration.

## Exact execution and audit trail

- Real XOX GitHub Actions [#38452218321](https://github.com/popixoxipop-collab/beglin/actions/runs/38452218321) — **SUCCESS**; executed source commit `3652588efa99eca95de6b5bad26b2ba36b106b91`.
- Original complete model: DeepSeek-V2-Lite quantized AF forward in Beglin scratch, original Top-6 router, frozen 120 real independently tokenized questions across 12 domains, fresh process and KV for each.
- Frozen corpus source commit `1b4606eef596e10f3003f7d9f07b607616e990d6`.
- **120 prompts / 1,555 original token positions / 26 routed layers = 40,430 source K/K+1 observations**. Future-target usable positions train **873**, validation **284**, final holdout **278**. Independent prompt split 72/24/24.
- Model index SHA-256 `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`. Scratch research binary SHA-256 `875e9ea52b116794796e73c18bc5e589b5b01d37de4ca11ea304c52594491a02`.
- **Noninvasiveness check:** first actual 13-token prompt processed twice in a fresh process, with and without the actual router hook; original final-token logits file and original macro-route file SHA-256 matched exactly.
- Hook before TopK renorm verifies p6 >= p7, full-64 softmax sum, chosen/unselected IDs and 26-layer observation coverage. Output raw scores/h2/logits stay in XOX scratch; no weights or raw prompt tokens exported.
- Original aggregate evidence artifact [#11840933672](https://github.com/popixoxipop-collab/beglin/actions/runs/38452218321/artifacts/11840933672), ZIP SHA-256 `873a6e06cac10e017e54a00a964f4ad6828b81c682ba8051775b0e0eb8086bd5`.
- `p3_true_boundary_eval.json` SHA-256 `62100fe837de26e8fb32f7677a1ab504f6cf4b1280a7a2f9bcc15b9702119fc5`; exact match to `receipt.json`, verified again independently after download. The per-run scratch directory was removed after upload.

## Measured target, honest meaning

At token t, after token t finished, build a causal feature from pre-FFN layer1 h2, observed selected-top6 summary, actual 6th/7th relative gap at layer1 and the minimum gap across 26 routed layers, and **the previous token's** final-logit entropy. Slow features are 0.75 previous + 0.25 current. Predict `min_{layer=1..26, future=t+1..t+4} (p6-p7)/max(p6,eps)`.

Training-only 20th percentile of the future risk target defines a near-tie event: **relative gap threshold 0.00035878848**. Final holdout event rate **25.54%**. Near-tie AUC measures ranking of this derived event, **not** a qNg64 bit-policy-safe margin, native logit correctness, or actual task quality. The 26-layer minimum is observed only after processing a token; it cannot change that same token's earlier layer decisions.

## Five-arm final prompt-heldout comparison

Lower RMSE is better; AUROC closer to 1 is better. All data come from the same original 120 independent DeepSeek prompts and 278 holdout future-target positions.

| Predictor | Prompt-block macro RMSE | Token-weighted RMSE | AUROC of train-defined near tie | Average precision |
|---|---:|---:|---:|---:|
| Static target mean | 0.003266621 | 0.003346830 | 0.500000 | 0.274349 |
| Current-gap persistence | 0.006217678 | 0.006282529 | 0.485813 | 0.244801 |
| **Flat single-level Ridge** | **0.003091211** | **0.003177784** | 0.607675 | **0.332427** |
| Shared two-level PCA/Ridge | 0.003200103 | 0.003276195 | 0.606314 | 0.286637 |
| Independent two-level PCA/Ridge | 0.003194285 | 0.003271401 | **0.607879** | 0.294531 |

- **Shared two-level vs flat prompt macro RMSE:** **3.5226% worse**, paired 500-draw prompt-bootstrap *improvement* 95% CI **[−8.0051%, −0.3363%]**.
- **Independent two-level vs flat:** **3.3344% worse**, paired improvement CI **[−7.7589%, −0.2149%]**.
- Independent two-level AUROC is ~0.00020 higher than flat (0.607879 vs 0.607675), an extremely small untested AUC difference; its average precision is **worse** (0.2945 vs 0.3324), so this is not a near-tie detection win.
- Both hierarchical models chose validation `slow_alpha=1.0`, meaning **the fast lower-level prediction was unused**. The trained shared/separate latent pair did not learn a beneficial slow/fast combination in this experimental setup.
- Compared to static mean, the flat model slightly improves numerical prediction and near-tie ranking; AUROC ~0.608 remains a *modest* signal. A low absolute RMSE partly reflects that the target margin is tiny and must not be equated to low model failure rate.

## Independent screening gate

A separate fail-closed `tools/hcp_p3_quality_gate.py` requires a positive lower bound of prompt-block improvement, ≥1% practical gain and nondegrading near-tie AUROC, on top of full source/schema/context restrictions. Neither 2-level model qualifies. Even in a hypothetical pass, the result would be a **proxy-only research candidate**, not an authorization to call `PrecisionEpochScheduler` or to change qNg64 policies.

- P3 regression tests [#38451847786](https://github.com/popixoxipop-collab/beglin/actions/runs/38451847786): **19/19 PASS**, covering hook placement, complete 26-layer measurement format, leakage and fake metadata rejection.
- Followup independent gate regression [#38452870004](https://github.com/popixoxipop-collab/beglin/actions/runs/38452870004): **24/24 PASS**, including significance, AUROC and policy-denial checks.
- Separate independent downloaded-result verification: local audit SHA-256 `bdb2ad7ca42216812a2945f903a4c1126c189da722d3773a179afdacfe2243da` (the audit is a derived summary, not the original authoring source).

## What P3 proves and does NOT prove

**Proves:** actual pre-renorm 6th/7th router margin can be captured without changing original model outputs; train/validation/test split and source SHA retained; low-cost flat model predicts future margin better than the studied two-level encoders in this sample.

**Does not prove:** no useful H-JEPA-style architecture exists; no 2-level nonlinear world model could do better; current margin is an accepted precision-risk signal; token-quality improvement; correct alternate qNg64 safe precision; execution acceleration; performance of a model that actually switches precision.

Original old P7 had 200 operator-injected `low_margin` out of one repeated prompt, and median policy-switch costs on the order of hundreds of milliseconds. P3's 4-token prediction scope cannot establish economical policy lifetimes for those switches.

## Recommended next research choice

1. **Keep P3 flat Ridge as the frozen predictive research baseline.** Do not increase hierarchy depth under this source, training objective or sequence length.
2. If the H-JEPA hypothesis remains important, acquire real **long context, multi-layer, independent task trajectories** and either model actual task accuracy under alternative precision policies or minimize independently measured total-inference cost. Use a trained action-conditioned predictive latent transition and compare against flat/constant-duration planners under equal compute budgets. The present 120 short prompts and 4-token target are insufficient to judge this.
3. Make any native candidate pass Beglin's existing per-context `precision_allocator` + `precision_dynamic_selector` + exact combined-policy evidence before any isolated end-to-end token/logit/latency test. Only then consider deployment after an independent explicit authorization path.
4. Keep `main` and the active inference model/precision policy unchanged. Auto-live-promotion remains OFF.

**Final P3 decision: true full-router risk measurement accomplished; hierarchical forecast advantage rejected for this prototype; native policy change BLOCKED.**
