# P9 A0 execution handoff

Status: **P9_PHYSICAL_JOB_RUNNING**

Branch: `prod/p9-integration`
Integration base: `727d950ef420320455b5071969efd49e3751e8c8`

## Completed

The canonical P8 handoff is accepted with:

- `P8_RUNTIME_VALID_8_OF_8`
- `P8_POLICY_ATTRIBUTION_8_OF_8`
- `P8_EVIDENCE_INDEPENDENT_PASS`
- overall `P8_INTEGRATION_PASS`

P9 now has an explicit integrated adapter instead of a null placeholder:

- adapter: `p9-xox-mlx-metal-v1`
- hardware/backend: XOX / MLX Metal
- corpus: eight pinned prompts, two repetitions each
- cells: safe reference, n4, n5, n6, n7
- external score: deterministic token edit agreement against the reference
- physical runner: `beglin_p9_quality_fixture.py`
- exact Tailnet argv: `["python3","beglin_p9_quality_fixture.py"]`

The generated plan is `P9_EXECUTION_PLAN_2026-09-29.json`. Exact fixture registration is now frozen by `P9_TAILNET_REGISTRATION_EVIDENCE_2026-09-29.json`, and adapter preflight has no blockers.

The first physical run is active as durable XOX job `job_f640ce1213e392ed07d7d627482aa975` with run ID `p9-20260928T182317Z-56784`. Build identity is already frozen; final raw metrics are pending.

## Long-context implementation

The pinned retrieval prompt is 15,033 tokens and requires 15,045 positions with generation. The P9-only build sets a 16,384-position limit.

The original expanded MLA float32 cache would require roughly 9 GiB and the existing ragged `take` path could replicate the same B=1 cache for all 64 prefill rows. The isolated P9 path therefore:

- stores expanded K/V as symmetric int8 with one float16 scale per layer/head/position,
- preserves full causal attention rather than truncating or sliding the context,
- broadcasts the single slot cache across query rows with manual attention matmuls,
- allocates only the configured model layer count,
- leaves the default/P8 build path unchanged.

The registered runner recomputes the compact-cache projection from the pinned model config and fails closed if the projection is unexpectedly at or above 3 GB.

## Required physical evidence

The runner must produce one raw log for each of reference, n4, n5, n6, and n7. Every log must have 16 parsed request records with:

- finite logits across prompt and generation rows,
- complete teacher-forced NLL counts,
- deterministic output token IDs,
- promotion count,
- live activation summaries for `kv_a_proj_with_mqa/L11` and `shared_up_proj/L3`,
- expert-router selection-boundary near-tie counts,
- long-context output evidence.

Raw physical success is not a P9 quality PASS. EOE must still normalize all five artifacts, evaluate the complete metric set, freeze numeric policy only after observing the reference, and perform an independent final audit.

## Current prohibition

`configs/quality/quality_policy_v1.json` is now `FROZEN`, hash-bound to the physical reference artifact and timestamped before any candidate log was inspected. `production_write_allowed=false`; candidate completion and an independent final audit are still required before any merge, production promotion, or go-live decision.

## Next exact action

Wait for the registered job to emit `P9_PHYSICAL_RAW_PASS`, then freeze and normalize the five logs:

```text
reference.log
n4.log
n5.log
n6.log
n7.log
```

Do not modify the frozen numeric policy or authorize production from a RUNNING job.
