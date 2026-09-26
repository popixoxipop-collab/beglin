# Agent D Handoff — GPU Shadow Observability

Date: 2026-09-26 KST  
Status: READY_FOR_REVIEW  
Draft PR: #10  
Base: `255ed8c1bddcbc100b6cb79f5bc11e62e0984913`

## What Agent D changed

Agent D did not touch production state. The branch changes only shadow observability/result-integrity behavior.

### 1. Run identity linkage

The shadow path now links:

```text
launch_id -> cycle_id -> shadow_run_id
```

The status/reporter layer can distinguish:

```text
CURRENT
PREVIOUS
LEGACY_UNLINKED
```

An old `last_cycle.json` is no longer treated as the current launch result merely because the launcher is RUNNING.

### 2. Terminal-aware dedupe

Reuse remains context-aware and now also requires a reusable terminal result.

Reusable:

```text
SHADOW_ADMITTED
SHADOW_REJECTED
SHADOW_ROLLBACK_OR_REGRESSION
```

Not permanently reusable:

```text
SHADOW_ERROR
SHADOW_COMPLETED_UNCLASSIFIED
runner exception
```

Same-fingerprint non-reusable outcomes retry with a bounded budget. Default:

```text
max_retry_attempts = 2
```

Budget exhaustion becomes:

```text
MANUAL_REVIEW_REQUIRED
```

instead of infinite retry or silent permanent dedupe.

### 3. Top-level cycle status

Pipeline outcomes are now explicit:

```text
SHADOW_CYCLE_COMPLETE
SHADOW_CYCLE_FAILED
SHADOW_CYCLE_UNCLASSIFIED
```

Failed/unclassified CLI terminals return nonzero.

### 4. Read-only reporter

New:

```text
tools/gpu_shadow_report.py
schemas/gpu-shadow-report-v1.schema.json
docs/GPU_SHADOW_OBSERVABILITY.md
```

The reporter reads only known JSON state under an explicit shadow root and does not read credentials, environment variables, or complete worker logs.

It separates:

```text
NO_READY / NO_NEW_READY / MANUAL_REVIEW  -> not_run
executed without explicit G4/G6          -> unknown
explicit G4/G6 failure                   -> failed
explicit current G4+G6 pass              -> passed / real_gpu
previous launch cycle                    -> stale_previous_cycle
legacy unlinked cycle                    -> legacy_unverified
```

`exit 0`, `COMPLETE`, or `SHADOW_ADMITTED` alone are not classified as a fresh GPU PASS.

Reporter output paths are rejected when they resolve inside the observed shadow root.

## Verification

Latest Agent D branch was materialized into the isolated EOE directory:

```text
/Users/eoe/mcp-sandbox/tailnet-commander/d-shadow-observability
```

Independent latest-head checks:

```text
python3 -m compileall -q d-shadow-observability
PASS

python3 -m unittest discover \
  -s d-shadow-observability \
  -p 'test_gpu_shadow*.py'

72/72 PASS
```

Execution evidence:

```text
compileall request_id:
209159e9-3b42-46e6-b26a-2be8f4fb014f

72-test request_id:
f93516a0-76cb-43c7-bd7f-198159ed046f
```

Earlier focused runs also isolated/fixed:
- macOS `/var` vs `/private/var` output-root normalization
- a missing test import

No production state was changed during these tests.

## Files owned by D

```text
docs/GPU_SHADOW_OBSERVABILITY.md
schemas/gpu-shadow-report-v1.schema.json

tools/gpu_shadow_launch_xox.py
tools/gpu_shadow_pipeline.py
tools/gpu_shadow_report.py
tools/gpu_shadow_status_xox.py

tools/test_gpu_shadow_launch_xox.py
tools/test_gpu_shadow_pipeline.py
tools/test_gpu_shadow_report.py
tools/test_gpu_shadow_status_xox.py
```

## Important limitation

This ChatGPT connector session does not directly expose `vdsp_gpu_precision`.

A gate-aware helper attempt was blocked by the security boundary. No bypass was used.

Therefore Agent D does **not** claim:
- a fresh live-XOX reporter run,
- new real-GPU evidence,
- production verification.

This branch is fixture-verified only.

## A0 integration sequence

1. Rebase/cherry-pick D-owned changes onto the current certified XOX integration source.
2. Preserve unrelated dirty GPU experiment files.
3. Run the complete GPU test suite on the integration SHA.
4. Run `git diff --check`.
5. Through an approved read-only XOX path, run the reporter against the real shadow root.
6. Confirm the report labels the existing dedupe/no-run state as `not_run`, not GPU PASS.
7. If/when new evidence produces a real G4/G6 run, confirm explicit stage status is required for `real_gpu`.
8. Keep production mutation OFF.

## Do not do

Do not:
- delete `candidate_history.json`,
- force a deduped candidate to rerun,
- widen production write access,
- enable canary/auto-promotion,
- merge PR #10 as a substitute for reconciling the newer certified XOX source.

The canonical machine-readable result is:

```text
agent_results/20260925/D/agent_result.json
```

## A0 canonical integration unit

Use:

`75e2a5d32427be8881e8711854eeaf810c31cb83`

Branch:

`agent-d-shadow-observability-squashed`

Verified compare from the D base:
- 1 commit
- 10 D-owned files
- no delivery-only agent_result/HANDOFF files in the squash

Latest Tailnet observation before handoff:
`g6-0.4.0-alpha.27`, preflight `10 OK / 0 WARN / 0 FAIL`.

The current chat schema still does not directly expose `vdsp_gpu_precision`, so fresh live-XOX reporter verification remains an A0/new-schema integration step.
