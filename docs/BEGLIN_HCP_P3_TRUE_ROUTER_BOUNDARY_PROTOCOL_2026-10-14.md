# Beglin P3 — Genuine rank-6/rank-7 routing margin, prospective study

## Purpose

P2 predicted the normalized entropy **among six already selected Experts**. That is not a measurement of the full 64-way Top-6 versus Top-7 routing boundary, and it did not show meaningful predictive benefit from added hierarchical levels. P3 records the true sixth selected versus highest unselected rank using the same Softmax scores that Beglin uses to rank all 64 Experts.

This is still an H-JEPA-*inspired* predictive surrogate (PCA + Ridge), **not** trained H-JEPA / JEPA world-model code and not a new native qNg64 controller.

## Exact capture site and safety

The original `moe_forward_token` runs:

```
moe_matvec_f32(w_gate, h2, router_scores, MOE_N_EXPERTS, MOE_HIDDEN);
[optional bias];
moe_softmax_full(router_scores, MOE_N_EXPERTS);
moe_top_k_select(router_scores, MOE_N_EXPERTS, MOE_TOP_K, top_idx);
[SCRATCH-ONLY P3 ROUTER BOUNDARY CAPTURE]
if (MOE_NORM_TOPK_PROB) moe_topk_renorm(router_scores, top_idx, MOE_TOP_K);
```

Read-only P3 hook, explicitly fail-closed for `64 experts / top_k=6`, computes:
- rank 6 smallest *selected* full-Softmax probability `p6`;
- rank 7 largest *unselected* full-Softmax probability `p7`;
- `relative_gap=(p6-p7)/max(p6,eps)`;
- log ratio, full distribution entropy, and selected/excluded Expert IDs.
- exact rank and Softmax normalization consistency checks.

The hook operates **only in a copied per-run C source**, never replaces a production binary or modifies deployed router scores. Before the full collection, the first prompt is run **with and without the env-controlled hook** in fresh processes. The original logits and Top-6 route text SHA-256 must be exactly identical. Otherwise P3 fails and the corpus is not accepted.

## Dataset and causal prediction

- Real DeepSeek-V2-Lite quantized-AF forward, 120 independent natural language questions, 12 domains, fixed 72 train / 24 validation / 24 holdout.
- Every prompt is processed in a separate scratch process with a fresh attention cache. The frozen prompt-bank source commit is `1b4606eef596e10f3003f7d9f07b607616e990d6`.
- Per token: pre-FFN layer1 h2, full original token logits (kept scratch-local), original selected Top-6 route, and **all 26 routed layers' true K/K+1 gaps**.
- Causal fast feature vector (8 scalars): layer1 h2 norm/std, selected6 entropy/gap/concentration, actual layer1 K/K+1 relative gap, observed minimum gap across the current token's 26 routed layers, **previous token** full logits entropy. Slow feature vector: 0.75 old +0.25 current fast.
- After completing token t, predict risk for future t+1 and t+1..t+4. Main label: minimum of actual K/K+1 relative gaps across all routed layers in the future four tokens.
- This **does not** supply a previous-token risk signal prior to executing token t; therefore the discovered predictor cannot alter any in-flight token's earlier-layers precision.
- The event threshold for `near_tie` is the **20th percentile of training prompts' future-four min-gap target only**, never the validation or final holdout.
- Validation alone selects lambda and upper/lower mixing. Final 24-prompt holdout assessed once.

## Comparison arms

1. Static training-target average;
2. observed-risk persistence (use current min rank gap to predict future);
3. flat Ridge with all the same fast+slow causal inputs;
4. shared PCA rank2 fast/slow hierarchical predictor;
5. separate PCA rank2 fast/slow hierarchical predictor.

Report prompt-macro and token-weighted RMSE, MAE, risk AUROC/AP (actual Top-6/Top-7 near-tie event derived from train threshold), prompt-block paired bootstrap CI relative to flat and upper mixing parameter. A hierarchy is *research-worthy* only with ≥1% prompt-RMSE improvement, CI lower bound positive, and AUROC not worse than flat; this is not a native quality acceptance gate.

## Authority and resources

The true 6/7 boundary can indicate potential routing sensitivity but **is not evidence that switching qNg64 precision is safe or useful**. It lacks per-context quantization intervention proof, token accuracy, task success and cost certification. All candidate evaluations set `native_policy_rebind_allowed=false`, `production_touched=false`, `auto_live_promotion=false`.

Archived policy-switch P7 costs remain important: example ~615 ms cold switch, ~307 ms restore and ~1.51 ms/10-token request steady difference. A four-token forecast cannot amortize a 600+ request switch. Do not infer any net runtime acceleration from prediction metrics.

No raw h2, token IDs, logits, full rank6/7 arrays, or model weights may be uploaded. Only signed-off aggregate metrics and SHA-256 receipts leave XOX scratch. Delete only the exact per-run scratch root after upload.

## Provenance

- P2 negative result: [real 120-prompt experiment #38447529097](https://github.com/popixoxipop-collab/beglin/actions/runs/38447529097)
- P3 C-site source inspection: [preflight #38450588108](https://github.com/popixoxipop-collab/beglin/actions/runs/38450588108)
- P3 unit suite: [#38451847786](https://github.com/popixoxipop-collab/beglin/actions/runs/38451847786) — 19 tests PASS
- P3 real experiment: [#38452218321](https://github.com/popixoxipop-collab/beglin/actions/runs/38452218321) — pending until full end-to-end evidence
