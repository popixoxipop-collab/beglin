# P9 Quality Harness

Owner: A0 integration
Branch: `prod/p9-integration`

## Scope

The integrated P9 lane now contains:

- `benchmarks/quality/**`
- `configs/quality/**`
- `docs/p9-quality/**`
- the isolated `beglin_p9_quality_fixture.py` physical runner,
- P9-only long-context build and measurement taps in `CMakeLists.txt`, `qwen_infer.c`, and `mlx_moe.*`.

The long-context and compact-KV changes are compile/runtime gated. Default and P8 binaries keep their existing cache bounds and execution path. No serving configuration, release manifest, or production policy is changed.

## Current phase

Status: **P9_PHYSICAL_JOB_RUNNING**

The P8 gate is now satisfied by `P8_INTEGRATION_GATE_2026-09-28.json`, which is hash-bound to the canonical P8 result and independent audit. It supplies:

- `P8_RUNTIME_VALID_8_OF_8`
- `P8_POLICY_ATTRIBUTION_8_OF_8`
- `P8_EVIDENCE_INDEPENDENT_PASS`
- overall `P8_INTEGRATION_PASS`

`P9_EXECUTION_PLAN_2026-09-29.json` binds the explicit `p9-xox-mlx-metal-v1` adapter, pinned token fixture, deterministic reference evaluator, reference+n4+n5+n6+n7 mapping, and XOX runner. Adapter preflight now passes with no blockers. Registration evidence is frozen in `P9_TAILNET_REGISTRATION_EVIDENCE_2026-09-29.json`.

The 15,033-token long prompt requires 15,045 positions including generation. The isolated P9 build provides 16,384 positions and uses full causal attention with a symmetric int8 K/V cache plus per-head/position float16 scales. This avoids the roughly 9 GiB float32 cache and same-slot batch replication that would exceed XOX's Metal memory budget. This cache format is a shared control across all five P9 cells and is not enabled in the default/P8 build.

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

python3 -m benchmarks.quality.execution_adapter preflight

python3 -m benchmarks.quality.execution_adapter materialize \
  --output-dir /tmp/p9-tokens

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

EOE validation completed on 2026-09-29:

- Python compile: PASS
- synthetic quality and physical-log normalization self-test: PASS
- C/C++ default and P9 long-context syntax checks: PASS
- token fixture materialization and all eight byte hashes: PASS
- no-P8 matrix planner: PASS / WAITING_P8_INTEGRATION
- P8 compatibility gate: PASS / P8_INTEGRATION_PASS
- post-P8 matrix planner: PASS / P9_EXECUTION_ADAPTER_READY
- current P7 readiness probe: P7_INSUFFICIENT_FOR_P9
- physical reference vs n=4/5/6/7 quality run: RUNNING / XOX job `job_f640ce1213e392ed07d7d627482aa975`
- numeric acceptance thresholds: UNFROZEN

The current P7 physical run has six real activation cells and five performance cells, but it does not carry the complete token-level NLL/output/finite-logits/promotion/router/task/long-context evidence required by this harness. It is therefore evidence input, not a substitute for P9.
