# P9 Normalized Quality Artifact Contract

Schema: `beglin-quality-run/1`

The artifact is the boundary between the engine/runner and Agent D. Agent D does not parse arbitrary engine stdout as quality truth.

## Top-level shape

```json
{
  "schema": "beglin-quality-run/1",
  "run_id": "...",
  "variant": {
    "kind": "reference | candidate",
    "precision": "bf16 | fp16 | safe_baseline | n4 | n5 | n6 | n7",
    "n": null
  },
  "identity": {
    "source_commit": "...",
    "source_tree": "...",
    "binary_sha256": "...",
    "checkpoint_sha256": "...",
    "model_id": "...",
    "hardware_id": "...",
    "backend": "...",
    "seed": 0,
    "prompt_corpus_sha256": "...",
    "runtime_config_hash": "...",
    "policy_hash": null
  },
  "requests": []
}
```

Candidate `n` must be exactly 4, 5, 6, or 7 and must agree with `precision`. Reference `n` is null.

## Request evidence

Each `(prompt_id, repetition)` pair must be unique.

Supported evidence fields:

- `context_tokens`
- `output_token_ids` and/or `output_text`
- `finite_logits`
- `token_logprobs` **or** `nll_sum + nll_token_count`
- `corrections`
- `promotions`
- `quality_score` in [0,1]
- `activation_summaries[]` with role/layer/expert + mean_abs/RMS
- `router` with near-tie count/decision count or near-tie rate

Numeric NaN/Inf values are rejected.

## Perplexity

For corpus entries marked `use_for_perplexity=true`, the harness uses repetition 0 only.

- explicit NLL: `exp(sum(nll_sum) / sum(nll_token_count))`
- log-prob hook: `exp(-sum(token_logprobs) / token_count)`

If any required perplexity prompt lacks evidence, perplexity is null rather than partially extrapolated.

## Quality score

`quality_score` is an adapter hook for a bounded external evaluator. Agent D does not invent an evaluator score. If any required request lacks this field, aggregate quality score is null and P9 PASS is blocked.

## Task score

Task scoring is deterministic and corpus-defined:

- exact match,
- contains-all,
- numeric tolerance,
- none.

This is separate from the external quality score.

## Determinism

Two repetitions per prompt are required by the default matrix. The harness compares canonical token IDs first, falling back to exact text only if token IDs are unavailable.

## Activation drift

The harness aggregates matching `(role, layer, expert_id)` cells independently in reference/candidate artifacts, then reports relative mean-abs and RMS deltas. No unmatched cell is synthesized.

## Long-context stability

For corpus entries tagged `long_context`, a candidate counts as stable only when:

1. reported context length satisfies the pinned minimum,
2. logits are finite,
3. canonical output matches the reference.

Incomplete coverage returns null, not a partial PASS.

## Fail-closed rules

- control identity mismatch → `CONTROL_MISMATCH`
- required metric missing → `INCOMPLETE_MISSING_METRICS`
- policy not frozen → at most `P9_QUALITY_HARNESS_READY`
- only a frozen policy with all required metrics and all rules satisfied can produce `P9_QUALITY_PASS`
