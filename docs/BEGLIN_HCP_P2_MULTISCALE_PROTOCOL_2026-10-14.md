# Beglin H-JEPA-inspired Compute Planner P2 — Multiscale Research Protocol

## Boundary and scope

**Research-only: no new precision policy applied.** All XOX inference happens with a separate scratch-built Beglin binary. The current macro Expert routing, checkpoint and production model remain untouched. This code is **NOT JEPA training**: it is a PCA+Ridge multiscale surrogate model used to falsify or support a two-timescale planning hypothesis before investing in JEPA-style learned latent state.

Upstream P1: [handoff](BEGLIN_HIERARCHICAL_COMPUTE_PLANNER_P1_2026-10-14.md). P1's ten archived cost snapshots and one injected 600-admission P7 sequence did not show robust overall cost benefit from horizon reuse.

## Actual P7 evidence coverage

- [XOX verified P7 survey #38446448259](https://github.com/popixoxipop-collab/beglin/actions/runs/38446448259): 600 validated SHA-linked admissions, one repeated prompt, 200 operator-injected low_margin activations.
- Numeric observed margin present 200/600; numeric entropy 0/600; routing ambiguity 0/600. The operator schedule creates a period-three artifact, so P7 is a **negative control**, not valid multiscale training examples.
- Existing P7 observed three actual policy hashes; P1 simulated only two. Do not use these records to claim native safety, accuracy or policy improvements.

## Independently captured real DeepSeek inference

- XOX scratch-only source, pinned to Beglin study `research/trainable-micro-router-120prompt-20261014` commit `1b4606eef596e10f3003f7d9f07b607616e990d6`.
- 120 natural-language prompts across 12 topical areas. Each prompt gets a separate process and isolated KV cache.
- Prompt split **train 72 / validation 24 / holdout 24**.
- Pre-FFN true layer1 `h2` features and original selected Top-6 Expert IDs/scores; previous token full-forward logits are an optional causal input.
- Current-token fast vector has six scalars: log hidden norm; log hidden std; selected-top6 normalized entropy; selected top-two gap; selected maximum concentration; previous-token logit entropy.
- Slow vector is exponential average of those six features with 0.25 update rate; no future lookahead.
- Targets are next-1 and average next-4 **selected-top6-score entropy**. These are **proxies**, not the real TopK-versus-K+1 boundary, not policy-certified risk or model-quality labels. The current target cannot approve precision rebind.
- Raw hidden states, logits, prompts and token IDs are never uploaded and are removed from run-local scratch. Uploaded evidence is restricted to numeric results and receipts.

## Four-way ablation

- **Static**: train-only constant mean of 4-step target.
- **Flat**: Ridge regression on the same six fast + six smoothed slow input features.
- **Shared two-level**: 2-dimensional shared PCA basis for both fast/slow levels, with short horizon and 4-token held waypoint.
- **Separate two-level**: separate 2-dimensional PCA bases for fast and slow, same prediction and waypoint contract.
- PCA and Ridge are fit using the training prompts exclusively. Validation selects ridge lambda and upper/lower mixing weight. Final holdout metrics reported once.
- Prompt-block paired bootstrap tests whether either two-level arm reduces heldout macro-per-prompt RMSE relative to the flat arm. Additional report: global RMSE, MAE, prompt-macro RMSE, validation metric, model-state bytes and Python study time (not native inference latency).
- Matched information access: all arms derive features from the **same causal observations**, with no holdout labels or future routes accessible when fitting.
- A two-level arm choosing slow_alpha=0 has effectively disabled its slow model; do not claim a hierarchical win from such a result.

## Evidence authority and fail closed

The new `tools/hcp_p2_policy_guard.py` delegates to the existing `precision_closed_loop.materialize_selected_policy`. It requires the exact execution-context identity, exact combined-policy hash, PASS proof and exact trigger context. High latent-predicted risk alone **cannot** select a new qNg64 policy; the guard emits shadow-only proposals with `native_policy_change_allowed=false`.

[XOX regression + contract tests #38448276361](https://github.com/popixoxipop-collab/beglin/actions/runs/38448276361) ran 15 tests, all PASS.

## Critical time-scale mismatch: representation versus economic policy horizon

P2's 4-token slow target is only a **representation learnability test**. A precision-policy switch in the historical P7 example cost ~615 ms cold A→B and ~307 ms for warm B→A, while policy B saved only ~1.51 ms per 10-token request in that particular archived snapshot. Its illustrative switch+restore break-even is roughly 612 **requests**, not four tokens. Thus a positive 4-token proxy forecasting score cannot by itself justify any live precision switching or cost claim.

To establish a useful H-JEPA-style hierarchical *compute* planner, future data must cover long, independently sourced requests and their genuine condition-bucket labels, with amortization beyond switching overhead and per-token/runtime quality proof. Even a successful P2 slow latent predictor is far short of this contract.

## Quality and deployment gates

Any prospective followup requires substantial real predictive performance beyond flat; positive lower endpoint of the paired prompt-bootstrap confidence interval; multi-layer and multiple independent long sequences; actual native Top-K boundary risk labels rather than selected-top6 entropy proxy; verified condition-specific combined-policy evidence and binary/checkpoint identities; matched native qNg64 full-model token/logit comparisons; actual p50/p95 full inference cost, planner overhead, cache transitions and RSS under same backend.

If the extra upper level fails to outperform flat, **retain the flat controller** and do not automatically increase hierarchy depth, and do not schedule native rebinding.

Current P2 experiment: [XOX real 120-prompt multiscale workflow #38447529097](https://github.com/popixoxipop-collab/beglin/actions/runs/38447529097). Final result must be inserted only after a complete successful run and hash-bound receipt.
