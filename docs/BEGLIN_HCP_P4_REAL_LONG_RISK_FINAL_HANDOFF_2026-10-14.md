# Beglin HCP P4 — 112-token Real Router Risk, 16-step Hierarchical Ablation, Native Precision Blockers

## Final experimental disposition

**ACTUAL 36 PROMPT / 3,933 TOKEN / 26 LAYER RISK CORPUS: PASS.**

**ACTUAL 16-TOKEN-FUTURE PREDICTION REPLAY: PASS, BUT NO STATISTICALLY ESTABLISHED HIERARCHICAL ADVANTAGE.**

**112-TOKEN NATIVE qNg64 A/B QUALITY + COST: NOT COMPLETED. NO QUALITY OR SPEED CLAIM.**

**24-TOKEN COMPATIBILITY A/B: REBIND TRANSACTION ACK BLOCKED. NO NATIVE A/B QUALITY CLAIM.**

Research branch: `research/hcp-p4-longcontext-costquality-20261014`. The branch is not merged. Live `main`, deployed model weights, stage binary and route/policy remain unchanged. Automatic promotion is OFF.

## What was actually measured

The independent single-sequence DeepSeek-V2-Lite verifier was cloned into a run-specific scratch build. Its source supports `MOE_MAXPOS=160`, and all input-ID strings were limited to under 900 bytes to avoid the existing 1,024-byte parser's silent truncation issue. Each context had 104–112 actual DeepSeek tokenizer IDs. Three questions per domain were sourced from the frozen, originally preregistered prompt bank; every prompt ran in a fresh process/KV. Each of 12 domains contributed one prompt to each train/validation/final-holdout split.

- Actual 36 prompts, 3,933 input positions, 26 routed layers, **102,258 true full-Softmax Top-6/7 observations** before selected-topK renormalization.
- Original model full forward final logits and Top-K file SHA-256 remained **byte identical** with the instrumentation enabled versus disabled in the 109-token smoke.
- Dataset contains current causal scalars, EMAs over nominal 4/16 scales and future 4/16/32 relative Top-6/7 gap minima, with **2,781 complete future-32 labels**.
- A parser and the single-sequence verifier shared the original DeepSeek index SHA-256 `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`.
- Raw hidden states, full logits, raw token IDs, complete router probabilities, model weights and generated response IDs were not uploaded. They were deleted in the XOX run-local scratch after experiment failure.

**Authoritative XOX raw-derivation receipt:** [Run #38467501959](https://github.com/popixoxipop-collab/beglin/actions/runs/38467501959), job outcome FAIL because the separate native A/B smoke failed **after** completing the 36 real-routing traces. Original artifact [#11852505311](https://github.com/popixoxipop-collab/beglin/actions/runs/38467501959/artifacts/11852505311) has schema `beglin-hcp-p4-context-costquality-safe-evidence-v1`, status `PARTIAL_BLOCKED_NOT_EVALUATED`. This must **not** be cited as a successful native A/B run.

- `p4_derived_long_features.json` SHA-256 `c3b54b1ebe41d9a85ed94ddd0da87cde0de86ecf0af56d2e621733e439944f53`, exactly matches the original XOX receipt.
- `registry_safe.json` SHA-256 `71e9256e3ce2cd62e2f4203bd4f06eed748dbf0f7337fd9a232f5717b327f6cb`.
- The original worker-local source commit was `524c2998ec2bdbae4770a594abd64c18792b847d`.

## Independently evaluated actual 16-token future-risk ablation

Instead of spending 30 minutes again collecting identical real routing features, a separate hosted GitHub Actions job downloaded the archived SHA-bound result and **refused** any incomplete native A/B certification. The 16-step study used **12 train / 12 validation / 12 independent holdout prompts** and respective usable target positions **1,133 / 1,116 / 1,108**.

| Predictor | Final holdout prompt-macro RMSE | Interpretation |
|---|---:|---|
| Static train mean | 0.0003112167 | Strong cheap baseline |
| Observed-risk persistence | 0.006146925 | Poor predictor of worst next-16 event |
| Flat single-level Ridge | 0.0003149282 | Uncertain advantage over static |
| Shared two-level slow/fast PCA+Ridge | **0.0003085169** | 2.0358% relative improvement vs flat |
| Independent two-level slow/fast PCA+Ridge | 0.0003085461 | 2.0265% relative improvement vs flat |

**Confidence intervals (1,000 draws resampling the 12 prompt blocks):**
- Shared versus flat relative macro-RMSE reduction: **+2.036%**, 95% interval **−1.368% to +4.310%**.
- Independent versus flat: **+2.027%**, 95% interval **−2.084% to +4.479%**.

Both intervals include zero, so **no validated hierarchy advantage**. On the validation split, the static baseline had lower RMSE than either learned two-level model. Both two-level models selected `upper_alpha=1.0`, so the fast/lower predictor was effectively unused. A positive-looking heldout point number must not be used to select an untested runtime policy or relabel this as successful H-JEPA training.

This is a linear/PCA proxy-risk experiment; no H-JEPA action-conditioned joint world model was trained. The actual target is the future minimum Top-6/7 relative gap across 26 layers; it is **not an intervention-tested qNg64 bit-safety margin**, logits correctness, real user task success, or overall model throughput.

- Independent final source-bound hosted run: [GitHub Actions #38486900820](https://github.com/popixoxipop-collab/beglin/actions/runs/38486900820), **SUCCESS**; artifact [#11860267318](https://github.com/popixoxipop-collab/beglin/actions/runs/38486900820/artifacts/11860267318).
- Result `real_16future_predictive_ablation.json` SHA-256 `39f6df187526945d0dc652aa7e9a48035d1abd171fd6ca92d45f311a527d181d`, verified against the independent evaluation receipt, source risk SHA and downloaded JSON.
- Run receipt SHA-256 `f6bdcbd6fb3bb5c9c7af9849ff685d29f3a23c2143ad12bb5a5976cdf1913ebf`, status `REAL_36_LONG_PROMPTS_PREDICTIVE_EVALUATED_NATIVE_AB_BLOCKED`.
- Regression source suite ran 28/28 PASS on [#38467796353](https://github.com/popixoxipop-collab/beglin/actions/runs/38467796353). Independent 16-step holdout/no-leakage tests 4/4 PASS on the hosted evaluation workflow.

## Independent native policy A/B blocker 1 — incompatible context capacities

The C single-sequence verification route supports `#define MOE_MAXPOS 160`, but the native `QWEN_MOE_GPU_CBATCH_ONLINE` persistent worker uses a **distinct** `#define MOE_CBATCH_MAXPOS 32`. In source `moe_cb4b_admit_guard`, inputs `plen >= MOE_CBATCH_MAXPOS` are dropped before admission. The P4 run's 104–112-token contexts cannot be sent to that native worker, irrespective of Python's advertised 4,096-token HTTP ingress guard.

In original XOX [#38467501959](https://github.com/popixoxipop-collab/beglin/actions/runs/38467501959), native policy smoke attempt terminated the scratch worker on its first long admission. The worker's full stderr was not archived, so do not claim exact crash causality from this log alone. The **hard source-code capacity mismatch** is confirmed and is an independent admission blocker. Do not raise cbatch32 to 160 or patch a live worker: this would resize large KV buffers and require separate proof of memory allocation, ragged batching, positional correctness and restart authority.

## Independent native policy A/B blocker 2 — durable transaction ACK

A separate 24-input / 4-generated-token scratch-only A/B compatibility test was constructed to remain below the native cbatch32 limit. The first attempt was blocked by a local scratch-path assertion and was fixed. The second reached the existing Beglin `PrecisionEpochScheduler` but failed with **`gpu_runtime_control.RuntimeControlError: ACK txn mismatch; actual=None`** when verifying a staged transaction. See [Run #38487383453](https://github.com/popixoxipop-collab/beglin/actions/runs/38487383453). No full paired A/B result was emitted or certified; production was not modified.

This is a genuine **unresolved isolated native precision-transition/ACK contract problem**, not permission to bypass transaction verification. Avoid reporting performance comparison while the runtime cannot certify the resulting exact policy. P7 earlier historical cost snapshots remain valid only for their originally tested source binaries, policies and request shapes.

## Safe engineering continuation

1. For cost work, audit the **specific staged worker binary and transaction source** against `prepare_rebind_set`, `QWEN_MOE_GPU_TXN_FILE`, `REBIND_SET_APPLIED`, and durable ACK, collecting only sanitized scratch worker status and exact source/binary identity. Stop if transaction ACK does not exactly match. No live rebind.
2. To compare 112-token requests under native qNg64, prepare a separate **scratch-only** capacity extension with explicit allocation/memory budget, positional verification, CUDA/Metal equivalent implementation, and exact 160-token graph-capture boundary. Treat it as its own experiment, not a trivial constant edit.
3. For H-JEPA-inspired hierarchical prediction, train action-conditioned models on truly long heterogeneous trajectories and real intervention labels, with static/flat/shared/separate controls of equal computational cost. Four/16/32-horizon risk prediction is not proof of action selection or economic switching benefit.
4. Keep all separate prototype source, metrics and negative outcomes. No shadow-policy promotion, no changes to `main` or active route or native precision, and automatic live promotion remains disabled.

**P4 outcome: real medium-context routing data collection SUCCESS; two-level next16 predictive advantage INCONCLUSIVE; native long A/B BLOCKED, native short policy transition ACK BLOCKED.**
