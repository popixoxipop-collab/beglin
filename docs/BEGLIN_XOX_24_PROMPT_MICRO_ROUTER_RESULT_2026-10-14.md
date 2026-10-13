# XOX DeepSeek-V2-Lite — 24-prompt trainable Micro-Router study

## Execution
- GitHub Actions SUCCESS: https://github.com/popixoxipop-collab/beglin/actions/runs/38421598924
- Immutable executed source: `f2b201dfe4b527f94600138500fa0409a95d5381`.
- Read-only independent inference: 24 distinct text prompts, fresh process per prompt, local tokenizer.
- Data split at PROMPT level: train 12 prompts / 150 tokens, validation 6 / 72, holdout 6 / 72.
- Observed layer-1 macro-expert union: 63 of 64.
- This study uses real pre-FFN h2 and real macro Top-K from the quantized model, but BF16 expert weights for local FFN evaluation.

## Heldout aggregate output relative L2 (lower is better)

| Micro width | Channel fraction | Static | Learned | Learned + Ridge |
|---:|---:|---:|---:|---:|
| 16 | 25% | 0.665356 | 0.642001 | 0.605206 |
| 16 | 50% | 0.525019 | 0.464331 | **0.445049** |
| 16 | 75% | 0.332457 | 0.328482 | 0.321866 |
| 64 | 25% | 0.706512 | 0.669911 | 0.614367 |
| 64 | 50% | 0.561816 | 0.530920 | 0.501351 |
| 64 | 75% | 0.372919 | 0.365666 | 0.356063 |

The learned Ridge-corrected method was selected on the validation split without consulting final holdout for all six configurations. At width16, 50% channel retention, relative error improved by 15.23% versus the static selection baseline.

## Costs and scope
- At width16, 50% selection, approximate active-channel fraction including fallback: **50.64%** of the macro FFN (not a measured speedup).
- 11 of 864 routed expert-token pairs across validation+holdout fell back to full execution for insufficient training support.
- Approximate FP32 router/corrector stored parameters: 15,899,200 bytes at width16, 50%.
- Prompt-bootstrap marginal 95% interval, six final holdout prompts: static [0.4508,0.5963], learned+Ridge [0.3964,0.4824]; no paired improvement interval or broad significance claim.
- No direct qNg64 kernel parity, end-to-end logits, task accuracy, wall-time or energy benefit measured.
- BEVAL/production promotion blocked; the final relative L2 remains high.

## Artifacts and reproduction
- GitHub Action evidence: https://github.com/popixoxipop-collab/beglin/actions/runs/38421598924/artifacts/11827030902
- Result JSON SHA-256: `b968d27becbaccd5a3f40ef2839bcbadb3a9030085e8a3983758b1b997b34f9c`.
- Source paths under `experiments/hierarchical_tupper_moe/`: `capture_prompt_suite.py`, `prompt_holdout_micro_eval.py`, `test_prompt_holdout_micro_eval.py`.
- Workflow: `.github/workflows/beglin-xox-prompt-holdout.yml`.
- Unit tests 4/4 passed locally and on XOX.
- Only aggregate results and receipt uploaded; original hidden vectors, model weights and trained coefficients remain unexported. Scratch files were removed after upload.
- `main` and running model remain unchanged.

**Status: EVALUATED_NOT_PROMOTABLE.**
