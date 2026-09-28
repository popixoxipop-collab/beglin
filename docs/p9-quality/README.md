# P9 Quality Harness

Owner: Agent D  
Branch: `prod/p9-quality`

## Scope

This lane owns only:

- `benchmarks/quality/**`
- `configs/quality/**`
- `docs/p9-quality/**`

It does **not** modify engine runtime sources, policy-binding sources, A/B/C validation code, performance harnesses, serving code, release manifests, or production policy.

## Current phase

Status: **P9_QUALITY_HARNESS_READY**

Physical BF16/reference vs n=4/5/6/7 execution is deliberately blocked until A0 supplies a valid `beglin-p8-integration/1` evidence file with:

- `P8_RUNTIME_VALID_8_OF_8`
- `P8_POLICY_ATTRIBUTION_8_OF_8`
- `P8_EVIDENCE_INDEPENDENT_PASS`
- overall `P8_INTEGRATION_PASS`

Even after that signal, the current matrix keeps `execution_adapter=null`; A0 or Agent D must bind the integrated runner explicitly. The harness never guesses a binary, checkpoint, policy, or command.

## Measurements

The normalized run contract supports:

1. perplexity from token log-probabilities or explicit NLL sums,
2. external bounded quality score,
3. deterministic task score from the pinned prompt corpus,
4. reference + candidate finite-logits rates,
5. reference + candidate within-run deterministic output rates,
6. exact candidate-vs-reference output match,
7. correction count,
8. promotion count,
9. per-role/layer/expert activation mean-abs/RMS drift,
10. reference/candidate expert-router near-tie rates plus candidate-minus-reference delta,
11. long-context stability against the same reference run.

Missing evidence stays `null`. A required metric that is null makes the candidate `INCOMPLETE_MISSING_METRICS`; it is never silently removed from the PASS decision.

## Control identity

Reference and candidate runs must match on:

- source commit,
- source tree,
- binary SHA-256,
- checkpoint SHA-256,
- model ID,
- hardware ID,
- backend,
- seed,
- prompt-corpus SHA-256,
- non-policy runtime-config hash.

The precision/policy itself is experimental and is represented separately by `variant` and optional `policy_hash`.

Any control mismatch yields `CONTROL_MISMATCH` before quality metrics are interpreted.

## Commands

From repository root:

```bash
python3 -m benchmarks.quality.self_test

python3 -m benchmarks.quality.prepare_matrix \
  --matrix configs/quality/p9_quality_matrix_v1.json \
  --corpus configs/quality/deterministic_corpus_v1.json

python3 -m benchmarks.quality.evaluate \
  --reference reference.json \
  --candidate n4.json --candidate n5.json --candidate n6.json --candidate n7.json \
  --corpus configs/quality/deterministic_corpus_v1.json \
  --policy configs/quality/quality_policy_v1.json
```

Before Policy Freeze, `quality_policy_v1.json` is `UNFROZEN`, has no numeric acceptance rules, and therefore cannot emit `P9_QUALITY_PASS`.

## P7 compatibility

`p7_readiness.py` can inspect the current physical P7 summary. It is intentionally expected to return `P7_INSUFFICIENT_FOR_P9`: P7 proves real activation instrumentation but does not contain the full token-level NLL/output/task/router/long-context evidence required by P9.


## Current validation

EOE validation completed on 2026-09-28:

- Python compile: PASS
- synthetic quality self-test: PASS
- no-P8 matrix planner: PASS / WAITING_P8_INTEGRATION
- current P7 readiness probe: P7_INSUFFICIENT_FOR_P9
- physical BF16/reference vs n=4/5/6/7 quality run: NOT RUN
- numeric acceptance thresholds: UNFROZEN

The current P7 physical run has six real activation cells and five performance cells, but it does not carry the complete token-level NLL/output/finite-logits/promotion/router/task/long-context evidence required by this harness. It is therefore evidence input, not a substitute for P9.
