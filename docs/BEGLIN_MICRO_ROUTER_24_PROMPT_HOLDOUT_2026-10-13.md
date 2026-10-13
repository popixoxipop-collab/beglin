# Beglin trainable Micro-Expert Router: 24-independent-prompt real-XOX holdout (2026-10-13)

## Evidence and verdict

**EVALUATED_NOT_PROMOTABLE.** Isolated XOX GitHub Actions [run #38421598924](https://github.com/popixoxipop-collab/beglin/actions/runs/38421598924) completed **SUCCESS** on source commit `f2b201dfe4b527f94600138500fa0409a95d5381`. The result is a meaningful OFFLINE improvement of BF16-Expert FFN-output reconstruction on quantized-AF-model hidden activations, *not* evidence of end-to-end LLM accuracy or faster production inference.

- Artifact: [beglin-xox-independent-prompt-micro-router-holdout](https://github.com/popixoxipop-collab/beglin/actions/runs/38421598924/artifacts/11827030902)
- `prompt_holdout_result.json` SHA-256 `b968d27becbaccd5a3f40ef2839bcbadb3a9030085e8a3983758b1b997b34f9c`, matched against the signed-by-content research receipt (not a digital signature).
- Safetensors **index** SHA-256 `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`; full checkpoint content identity was **not** verified within this run.

## Independent data

24 distinct offline tokenized text prompts, **12 train, 6 validation, 6 final holdout**, each with its **own new inference process** to prevent KV-cache contamination. XOX collected 294 actual pre-FFN layer-1 `h2` positions: 150 train, 72 validation, 72 holdout. Macro-Top-K routes were read from the model's actual scratch-run routing trace. Across the observed prompts, 63 distinct routed experts were seen. No raw hidden states, token IDs, model files or prompt-specific traces were uploaded. Scratch state was deleted upon completion.

Existing `trainable_micro_router.py` used a kernel-ridge channel-salience gate (`lambda=0.25`) and an optional omitted-channel ridge output corrector (`lambda=2.0`). **Only validation prompts** selected the method; final holdout prompts were not used to fit or select it. The chosen method in all measured cells was `dynamic_ridge_residual`.

## Actual final holdout relative L2 errors (smaller is better)

| Neurons per micro-channel | Fraction retained | Static | Dynamic only | Dynamic + Ridge correction | Improvement relative to static |
|---:|---:|---:|---:|---:|---:|
| 16 | 25% | 0.665356 | 0.642001 | 0.605206 | 9.04% |
| 16 | 50% | 0.525019 | 0.464331 | **0.445049** | **15.23%** |
| 16 | 75% | 0.332457 | 0.328482 | 0.321866 | 3.19% |
| 64 | 25% | 0.706512 | 0.669911 | 0.614367 | 13.04% |
| 64 | 50% | 0.561816 | 0.530920 | 0.501351 | 10.76% |
| 64 | 75% | 0.372919 | 0.365666 | 0.356063 | 4.52% |

The **50%-retained 16-neuron case** reduced relative L2 error by **8.0 percentage points**, or **15.23% relative**, but its *absolute error remains 44.5%*. One-method-at-a-time prompt-bootstrap (300 draws) 95% intervals: static `[0.450802, 0.596303]`; dynamic+ridge `[0.396374, 0.482414]`. These are **not paired improvement intervals**; with only six holdout prompts, statistical significance of improvement is not established.

For each sparsity/width cell the 72 validation + 72 holdout token positions yield 864 routed expert-position pairs. **853** used trained experts and **11** required conservative full-channel fallback (roughly 1.27% of the routed pairs); these fallback paths are included in the result. Approximate learned-router resident FP32 parameter budget is **15,899,200 bytes** for the width-16 experiment. The budget is a model-side estimate, not measured runtime RSS.

## Essential limitations

1. True layer-1 `h2` originates from quantized AF DeepSeek-V2-Lite; expert FFN weights used for reconstruction are from BF16 Safetensors, **not the exact forward's native quantized weights**. Thus these offline FFN metrics are not exact engine parity.
2. Comparison is of one layer's routed expert FFN outputs. **No full-model logits, perplexity, task success, route stability across downstream layers, or actual inference tok/s** were measured.
3. Channel compute fractions are idealized and exclude gate/compensation overhead, packing, cache, GPU/CPU kernel dispatch and transfer. There is **no demonstrated speedup**.
4. Generalized Tupper N×M masks passed bitwise encode/decode tests, but Tupper coding alone is **not weight compression**.
5. Method search used validation data correctly, but six final holdout prompts are too few for broad statistical/generalization claims.
6. Real hardware was isolated GitHub Actions scratch; **production_touched=false**, **automatic_live_promotion=false** and `main` unchanged.

## Next gated experiment

- Collect a larger, more diverse, **prompt-grouped** holdout, at least dozens of independent final-holdout prompts; report **paired prompt-bootstrap confidence intervals** for (static − dynamic+ridge).
- Reproduce the same-layer FFN reconstruction using **exactly matched quantized weights** and native qNg64/AF kernels; include activation/reference identity hashes.
- Benchmark full end-to-end latency, overhead/RSS and downstream logits/quality. Require correctness and an actual measured speedup before considering any runtime routing.
- Keep all testing scratch-only and immutable source-bound, never automatically promoting a research mask to production.

## Implementation

`experiments/hierarchical_tupper_moe/{capture_prompt_suite.py,prompt_holdout_micro_eval.py,trainable_micro_router.py,test_prompt_holdout_micro_eval.py}` and `.github/workflows/beglin-xox-prompt-holdout.yml`.
