# GPU Shadow Observability

This document defines the read-only observability contract added by Agent D.

## What this solves

A launcher exit code, `COMPLETE`, or an old `last_cycle.json` must never be
reported as a fresh GPU validation pass.

The observability layer distinguishes:

- `not_run`: no READY target, context-aware dedupe, or manual-review gate
- `unknown`: execution happened but explicit G4/G6 stage status is absent
- `failed`: explicit G4/G6 failure or pipeline failure
- `passed`: explicit G4 **and** G6 pass on the cycle linked to the current launch
- `legacy_unverified`: state predates launch/context linkage
- `stale_previous_cycle`: the newest cycle belongs to another launch

## Run identity

The detached XOX launcher generates a random `launch_id` and passes it only to
its shadow worker. The pipeline generates a `cycle_id`; the shadow runner
already owns a `run_id`.

The relation is:

```text
launch_id -> cycle_id -> shadow_run_id
```

A new launcher does not adopt an old `last_cycle.json` merely because the
previous cycle completed successfully.

## Dedupe and retry semantics

Candidate reuse requires both:

1. the candidate/provenance fingerprint, and
2. the validation-context fingerprint, which includes the certified binary,
   checkpoint identity and control-plane code hashes.

Only these shadow statuses are reusable terminal validation outcomes:

```text
SHADOW_ADMITTED
SHADOW_REJECTED
SHADOW_ROLLBACK_OR_REGRESSION
```

`SHADOW_ERROR` and `SHADOW_COMPLETED_UNCLASSIFIED` are **not** permanently
deduped. The pipeline retries the same fingerprint at most
`max_retry_attempts` (default 2). Once that budget is exhausted it emits
`MANUAL_REVIEW_REQUIRED` instead of retrying forever.

If the runner raises before it returns a result, a `SHADOW_ERROR` attempt is
recorded for the same retry accounting.

## Reporter

`tools/gpu_shadow_report.py` reads only the known JSON state files under an
explicit shadow root:

```text
launcher_status.json
last_cycle.json
candidate_history.json
discovery.json
inputs/<candidate>/candidate_spec.json              (only when referenced)
executions/runs/<run_id>/shadow_result.json         (only when referenced)
```

It does not read:

- credential files
- environment variables
- complete cycle logs
- arbitrary paths supplied by the state documents

An absolute candidate-spec path outside the observed shadow root is rejected as
an artifact source.

Reporter outputs must be outside the observed shadow root so running the
reporter cannot alter the state whose hashes it reports.

## GPU pass classification

A current cycle is labeled `real_gpu` only when the child payload contains
explicit G4 and G6 stage status and those statuses classify as pass/fail.

Examples:

```text
NO_NEW_READY_CANDIDATE                -> not_run
exit=0 + final_status=ADMITTED only   -> unknown
G4=PASS, G6=CANARY_PASS               -> passed / real_gpu
G4=REJECTED_AT_G4                     -> failed / real_gpu
old launch_id with a new launcher     -> stale_previous_cycle
```

The reporter does not invent a G4/G6 status when the current autopilot output
does not provide one.

## Report schema

Machine-readable contract:

```text
schemas/gpu-shadow-report-v1.schema.json
```

Key fields include launcher/cycle/run identity, source freshness, evidence
level, candidate and validation-context fingerprints, binary/checkpoint
identity, explicit G4/G6/rollback state, artifact hashes, warnings and unknown
fields.

The generated report always states:

```json
"production_write_allowed": false
```

## Legacy state

Old history rows without `runtime_identity_sha256` are not silently treated as
context-verified. They produce `LEGACY_CONTEXT_UNVERIFIED`.

Old cycles without a `launch_id` produce `legacy_unlinked`.

No migration, history deletion or forced rerun is performed by the reporter.

## Security and production boundary

This change does not:

- write Supabase
- mutate production precision policy
- enable GPU auto-promotion
- reset candidate history
- source or print credentials
- kill external processes

The reporter and its fixtures are review/evidence tools. Applying them to an
actual XOX shadow root requires a separately authorized read-only execution
path.

## Verification

Agent D fixture verification covers:

- NO_READY and NO_NEW_READY classification
- current vs previous launch/cycle linkage
- explicit G4/G6 pass
- G4 rejection
- exit 0 without explicit stage status
- legacy context warning
- credential sentinel non-leakage
- input-file hash immutability
- retry budget for error/unclassified terminal results
- runner exception accounting
- launch/cycle/shadow-run ID propagation

Live XOX reporter execution is not claimed until it is run through an approved
workspace/argv path.
