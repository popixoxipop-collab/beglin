# Agent D → A0 Handoff: P9 Quality Harness

Status: **P9_EXECUTION_PLAN_READY**

Branch: `prod/p9-quality`
Base: `a0b8ab2c90af70fa8095556033e605477ec186ae`

## Ownership respected

Only these paths are modified:

- `benchmarks/quality/**`
- `configs/quality/**`
- `docs/p9-quality/**`

No Agent A/B/C runtime or policy-binding source is modified. No merge/release action has been performed.

## P8 handoff received

The required P8 integration evidence is now materialized as
`docs/p9-quality/P8_INTEGRATION_GATE_2026-09-28.json`. It is hash-bound to the
canonical P8 result, engine-correctness PASS record, and independent audit, and
has this accepted shape:

```json
{
  "schema": "beglin-p8-integration/1",
  "status": "P8_INTEGRATION_PASS",
  "gates": {
    "runtime": "P8_RUNTIME_VALID_8_OF_8",
    "policy": "P8_POLICY_ATTRIBUTION_8_OF_8",
    "evidence": "P8_EVIDENCE_INDEPENDENT_PASS"
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
    "runtime_config_hash": "..."
  }
}
```

`docs/p9-quality/P9_EXECUTION_PLAN_2026-09-28.json` confirms the gate with no
problems. A0 must still name or bind the approved integrated execution adapter
and external quality evaluator. Agent D will not infer an executable path,
quality score, or production policy.

## Required normalized output

The integrated runner must emit one `beglin-quality-run/1` artifact for:

- reference BF16/FP16/safe baseline,
- n=4,
- n=5,
- n=6,
- n=7.

All five runs must share the same non-policy control identity.

The artifact contract is documented in `docs/p9-quality/ARTIFACT_CONTRACT.md`.

## Required P9 evidence

Every candidate must provide all required metrics:

- perplexity ratio,
- candidate external quality score + delta,
- deterministic task score + delta,
- reference and candidate finite-logits rate,
- reference and candidate deterministic-output rate,
- exact candidate/reference output match,
- correction count,
- promotion count,
- activation RMS drift,
- candidate router near-tie rate + delta from reference,
- long-context stability.

Any missing required metric blocks PASS.

## Current P7 result

The current P7 physical run is intentionally **not** reused as P9 PASS evidence.

It has:
- 8 matrix cells,
- 6 activation cells,
- 5 performance cells,

but lacks complete NLL/output/finite-logits/promotion/router/task/long-context evidence. The frozen readiness result is in `docs/p9-quality/P7_READINESS_PROBE.json`.

## Policy Freeze

`configs/quality/quality_policy_v1.json` remains `UNFROZEN` with no numeric acceptance rules.

This is intentional. Numeric quality thresholds must not be invented before the integrated baseline/reference evidence is observed.

## Validation already complete

- Python compile: PASS
- synthetic metric self-test: PASS
- control mismatch fail-closed: PASS
- missing required metrics fail-closed: PASS
- P8 absence blocks execution: PASS
- P7 insufficiency probe: PASS

After P8 integration, Agent D can proceed directly to physical comparison without changing A/B/C-owned files.
