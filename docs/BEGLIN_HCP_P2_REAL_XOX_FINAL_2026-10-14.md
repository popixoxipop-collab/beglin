# Beglin HCP P2 — XOX Real Multiscale Prediction and Safety Handoff (2026-10-14)

## Status and bottom line

**IMPLEMENTED / REAL XOX 120-PROMPT EVALUATED / NO SUPPORTED HIERARCHICAL GAIN / NATIVE POLICY REBIND BLOCKED.**

The H-JEPA-inspired 2-time-scale slow/fast prototype is structurally functional, and both temporal encoders can be fitted using real DeepSeek-V2-Lite observables. But after the same independent-prompt split, its best 2-level model improved the flat baseline's **per-prompt macro RMSE by ~0.06% only**, with a paired 95% CI crossing zero. The token-weighted RMSE was actually *slightly worse* than flat. Under a predeclared 1% practical improvement floor and positive paired CI lower bound, **no 2-level candidate qualifies**.

Do not represent this as H-JEPA reproduced, evidence of LLM speedup, or evidence that higher-level planning inherently improves precision selection.

## Implementation and experiment identities

- Repository: `popixoxipop-collab/beglin`.
- Research branch: `research/hierarchical-compute-planner-p2-20261014`; not merged.
- P1 predecessor: `research/hierarchical-compute-planner-20261014` / `e5ce531e64e5400949e846f399aaae436e967e3b`.
- Immutable actual 120-prompt XOX run: [#38447529097](https://github.com/popixoxipop-collab/beglin/actions/runs/38447529097) — SUCCESS, executed commit `a838b8bb4e7e8c5caac7ae6cd8e8018860414231`.
- Pinned frozen real DeepSeek prompt source: `1b4606eef596e10f3003f7d9f07b607616e990d6`.
- 120 questions, 12 domains, unique tokenizer inputs, separate scratch process/KV per prompt. Split train **72 prompts / 945 original positions**, validation **24 / 308**, final holdout **24 / 302** (1,555 original positions).
- Predicting future values consumes only positions with an available later target; effective supervised split: train **873**, validation **284**, holdout **278** prediction positions.
- Actual observations: quantized-AF DeepSeek-V2-Lite layer1 pre-FFN h2, original routed Top6 score distribution, and **previous** token's full-forward logits entropy. The actual current-token h2 and route scores are causal observations at the local FFN, not a certified policy-inference event.
- Targets: next token and average next **four** tokens of normalized entropy among the six **selected** experts. This is **not** the K-versus-K+1 routing boundary margin, BEVAL precision safety label, or actual accuracy ground truth.

## Comparison methods and final independent-prompt results

All transforms are fitted on training prompts only. Validation tunes the Ridge lambda and upper/lower mixing alpha. Final holdout is not used for tuning.

| Arm | Heldout per-prompt macro RMSE | Token-weighted RMSE | MAE | Estimated coefficient bytes |
|---|---:|---:|---:|---:|
| Static train mean | 0.020069545 | 0.020223306 | **0.016885352** | 1 marker, not meaningful for actual storage |
| Flat Ridge, same causal fast+slow inputs | **0.020025210** | **0.020203163** | 0.016929307 | 296 |
| Shared 2-level PCA rank2 + Ridge | 0.020061479 | 0.020279505 | 0.016971496 | 240 |
| Separate 2-level PCA rank2 + Ridge | **0.020013255** | 0.020207995 | 0.016886445 | 336 |

The separate 2-level model gains only `1-(0.020013255/0.020025210)=0.0597%` on prompt-macro RMSE. It is slightly **worse** on token-weighted RMSE and statistically indistinguishable:

- Separate vs flat paired prompt-block improvement: **+0.0597%**; 500-draw 95% interval **−1.3208% to +1.6652%**.
- Shared vs flat paired prompt-block improvement: **−0.1811%**; 95% interval **−1.8463% to +1.6589%**.
- Validation selected `slow_alpha=0.75` and `ridge_lambda=10` for separate, so the upper latent actually contributes. Shared selected `slow_alpha=1.0`; flat selected `ridge_lambda=0.01`.
- Recorded Python batch timing (~0.96ms flat, ~1.76ms separate on a heldout batch of 278 examples) includes model fitting and is **NOT** a native per-token overhead or speedup benchmark. Do not compare these times to qNg64 kernel latency.

**Interpretation:** The current proxy is nearly as predictable with a train-only global mean as with these learned models. The available six features and short 4-token horizon may lack informative separable slow/fast structure. This does not falsify H-JEPA's robot planning result or stronger action-conditioned slow/fast world models; it falsifies an unsubstantiated performance claim for this specific cheap P2 surrogate.

## Provenance and evidence

- XOX original artifact: [Actions artifact #11838629728](https://github.com/popixoxipop-collab/beglin/actions/runs/38447529097/artifacts/11838629728).
- `p2_multiscale_metrics.json` SHA-256: `0b52fe34581af52a57ff99a3f37c51c82444f3f85173071301c8e73d64b44b5c`; directly matches `receipt.json`.
- Receipt SHA-256: `203525f48bd17eef8572b0860bb30e8c4a0d35e85e7e7eb03351b4c1985fdc63`.
- Feature-source SHA-256 (features kept in XOX scratch, **not uploaded**): `9c2dfb404ba117029c588b78b6fa32b3f6f6903be1e9b39b295c97e6a8c6c77a`.
- Prompt manifest SHA-256: `bda25a9e225d7094d5406cfb0149debdbe08900de073943381814d88e3b67db9`.
- Original raw h2 vectors, full token logits and prompt IDs never uploaded; per-run XOX scratch removed on successful completion.
- Redacted derived metrics are committed at `experiments/hcp_p2/observed_120prompt_proxy_summary_20261014.json` and pinned to the original metric file SHA. This is a **subset**, not the original signed artifact; use the original GitHub Actions artifact as the authoritative source.

## Safety gates

- Earlier P7 observed 600 verified admissions, but they are **one repeated prompt**, and 200 low_margin signals are operator-injected at every third request. Margin numeric values only 200/600, entropy and ambiguity 0/600. It is not a genuine natural-risk training dataset: [P7 signal survey #38446448259](https://github.com/popixoxipop-collab/beglin/actions/runs/38446448259).
- `tools/hcp_p2_policy_guard.py` calls Beglin's existing `precision_closed_loop.materialize_selected_policy`, and also checks per-trigger matching execution context and exact combined-policy certification. **Learned future proxy may never grant policy authority**.
- `tools/hcp_p2_complexity_gate.py` only considers added upper levels when paired prompt-bootstrap CI lower bound >0 and relative improvement >=1%; it is a *predictive-depth screening* gate, not a production authority. Actual observed candidates fail.
- Regression on GitHub [#38449760862](https://github.com/popixoxipop-collab/beglin/actions/runs/38449760862): **20/20 PASS** including real observed-depth rejection, missing evidence, false certification, wrong-source, holdout-target leakage and causal previous-logit tests.
- `main` unchanged. The active model, native qNg64 precision policies and persistent worker untouched. Automatic live promotion OFF.

## Engineering followup if proceeding with H-JEPA concept

1. **Improve target validity before deepening models.** Capture the actual router's rank-K/rank-(K+1) margin in an isolated C research binary; distinguish that from entropy of the six selected route weights. Match accuracy/quality with original hardware quantization. Do not infer safe bit precision from a proxy.
2. Obtain long, diverse independent prompt and multiple-layer sequences, preserving prompt-level split, model weights/binary identity, route uncertainty, actual logits and request-scoped cost. Four-token predictions do **not** amortize the historical A→B→A policy-switch cost ~922ms; the old profile had only ~1.51ms/10-token-request steady advantage.
3. Train genuine action-conditioned upper/lower predictive state with an H-JEPA-like latent alignment or self-supervised objective **only after** demonstrating nontrivial predictability and a useful safety-relevant target. Compare static, flat, shared and separate with equal causal inputs and model-complexity/cost budgets.
4. Any future certified controller must reuse exact combined-policy/trigger identity gates, and win in separate XOX scratch **full-model logits, token agreement, p50/p95 latency, RSS and total policy transition cost**. No bypass of normal qNg64 quality or production gates.

**Final decision: P2 proxy model discovery DONE; no evidence for complexity gain; retain flat research baseline and keep native policy promotion BLOCKED.**
