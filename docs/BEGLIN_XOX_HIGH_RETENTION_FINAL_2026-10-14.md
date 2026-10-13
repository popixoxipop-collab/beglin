# Beglin — XOX High-Retention Generalized Tupper Micro-MoE: final evidence, 2026-10-14

## Decision

**EVALUATED / ZERO ACCEPTABLE SPARSE CANDIDATES / NATIVE END-TO-END EXPERIMENT BLOCKED BY PRE-REGISTERED QUALITY GATE.**

The standalone XOX GitHub Actions run completed SUCCESS, including offline tokenizer prompts, read-only real-model trace capture, validation-only method selection, 24-prompt final holdout, discrete count gate, result SHA receipt, artifact upload and per-run scratch cleanup. A successful workflow is **not** evidence of model-quality success.

- Experiment workflow: [#38434359681](https://github.com/popixoxipop-collab/beglin/actions/runs/38434359681), executed immutable commit `ef65095d8724a0e07d10f9c563349219131859bc`.
- Self-hosted XOX, 120 distinct prompts and 1,555 real token positions, train 72/945, validation 24/308, final holdout 24/302. Actual layer-1 h2 from the **quantized AF DeepSeek-V2-Lite forward** and actual Top-K routes; local counterfactual FFN from **BF16 expert Safetensors**, not an exact same-representation runtime comparison.
- Baseline 16-neuron 50% channel retention reproduced **0.411714** relative L2 (original study #38430695333). No full-model logits, token accuracy, task scores, end-to-end wall time or GPU memory measured here.

## Measured final holdout local MoE-FFN error

Every method listed is chosen on validation prompts, not the final holdout. For every nondegenerate row validation selected the learned Router with **full Ridge omitted-output correction**.

| Micro-neuron width | Requested retained | Actual groups kept | Idealized micro-channel compute saved | Locked BF16-local output relative L2 | Prompt-block 95% bootstrap L2 CI | Acceptable? |
|---:|---:|---:|---:|---:|---|---|
| 16 | 50% | 44/88 | 50.00% | 0.411714 | [0.393182, 0.428892] | No |
| 16 | 90% | 80/88 | 9.09% | 0.155328 | [0.149172, 0.160212] | No |
| 16 | 95% | 84/88 | 4.55% | **0.109980** | [0.104026, 0.114911] | No |
| 16 | 99% | 88/88 | **0%** | 0 | [0, 0] | **No-op, not sparse** |
| 64 | 50% | 11/22 | 50.00% | 0.437958 | [0.423652, 0.450544] | No |
| 64 | 90% | 20/22 | 9.09% | 0.168342 | [0.160800, 0.175516] | No |
| 64 | 95% | 21/22 | 4.55% | 0.118314 | [0.111698, 0.125862] | No |
| 64 | 99% | 22/22 | **0%** | 0 | [0, 0] | **No-op, not sparse** |

Pre-registered *local* gate: locked-method upper 95% prompt-block CI <= **0.05**, AND at least one micro-channel genuinely skipped. **0 / 8** configurations met both. The best meaningful candidate, 16-neuron keep95%, still has roughly 11% local output relative error, more than twice the 5% screening threshold, for only 4.55% theoretical sub-FFN channel-compute savings before overhead.

### Learned Router vs alternatives

- 16-channel keep90%: static 0.174509, learned uncorrected 0.158969, Rank-8 0.155604, full Ridge 0.155328.
- 16-channel keep95%: static 0.116476, learned uncorrected 0.111750, Rank-8 0.110064, full Ridge 0.109980.
- 64-channel keep95%: static 0.133525, learned uncorrected 0.123049, Rank-8 0.118465, full Ridge 0.118314.
- 16-channel keep95% paired reduction of Ridge vs static: 5.58%; 95% paired prompt bootstrap interval for relative error reduction: [1.21%, 10.51%]. Though positive, the absolute output error and overhead make it unsuitable for native serving.
- At width16, learned Rank-8 and full Ridge estimated FP32 model storage across observed experts remains ~99.7 MB vs ~141.9 MB, respectively (trained-model parameter estimate, NOT RSS). At 95% they differ by only 0.000084 in local output relative L2; this remains a minor correction to an unacceptable overall quality gap.
- Full-channel fallback count in the 120-prompt evaluation: 0. The selected-channel fraction **does not** imply proportional native GEMV speedup. Expensive per-token Router/corrector and memory traffic remain to be measured.

## Zero-error baseline reporting fix (post-run)

The executed source historically computed an invalid **100% relative improvement** when both baseline and candidate output errors were zero at 99% = full execution. This is mathematically **undefined** (0/0), NOT a performance gain.

Original run artifacts are preserved byte-for-byte with SHA. After examining the result, a research-branch code change reports `relative_l2_reduction: null`, `prompt_block_bootstrap_95ci: null`, status `UNDEFINED_ZERO_BASELINE_ERROR` for this case. This correction does **not** change original measured output errors or the gate decision, and **did not rerun** the 120-prompt XOX experiment.

- Corrective unit workflow: [#38437001599](https://github.com/popixoxipop-collab/beglin/actions/runs/38437001599) **SUCCESS, 10/10**.
- Research branch code: `experiments/hierarchical_tupper_moe/prompt_holdout_micro_eval.py`, `high_retention_gate.py`, and `test_high_retention_gate.py`.

## Provenance, integrity and cleanup

- XOX artifact: [Actions artifact 11833676166](https://github.com/popixoxipop-collab/beglin/actions/runs/38434359681/artifacts/11833676166)
- ZIP SHA-256: `d8035af755c5a8daa53cce1f11d7cc46f1b4a9a94198a949c175c3743bd8c108`.
- Original experiment `prompt_holdout_result.json` SHA-256: `d062ae3ff0f7386751bd5eab59f0f3a03baaee417cc8b8e541d79e98eaeda9d1`.
- `high_retention_gate.json` SHA-256: `64ca28ed3e1454202cf4826c0e8f8bdecea6fd45eec3043e8cf4cfc677cc2d13`.
- Checkpoint index SHA-256: `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`.
- Dataset manifest SHA-256: `bda25a9e225d7094d5406cfb0149debdbe08900de073943381814d88e3b67db9`.
- Independent derived audit summary JSON SHA-256: `8359f6a1dc07582a054ae77ae1a9f25f1922a3d257ac29666a3017f4f115c81e`, preserving the original artifact and marking 99% relative improvement undefined.
- Artifact contains aggregate metrics and receipts only. No raw pre-FFN hidden vectors, prompt token IDs, or model checkpoint uploaded. XOX log confirms per-run scratch directory removed.
- Original `main` and operating model untouched; automatic live promotion remains OFF.

## Next engineering choice (not done)

Do not integrate 90/95% candidates into qNg64 serving or proceed to end-to-end logits, next-token agreement, or throughput trials **under the present pre-registered quality gate**. Further theoretical work should use trainable structured pruning/distillation or explicitly optimize a task-level loss while enforcing exact native qNg64 contract and a measured total-resource budget.

If a different candidate eventually passes the independent *local* gate, only then perform a separately reviewed XOX scratch-only native equivalent-weight trial with token logits, task metrics, full inference wall time and memory. Such native trial is **not reported as completed** by this handoff.
