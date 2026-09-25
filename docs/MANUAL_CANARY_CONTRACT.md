# Manual Canary Contract — Dry Run Only

## Scope

This module implements the Agent-E contract layer only:

```text
verified shadow evidence
-> explicit manual approval
-> one-target dry-run canary state machine
-> bounded observation
-> kill switch / exact-baseline rollback
```

It does **not** connect to the production MLX adapter, restart a real service,
write Supabase, enable GPU auto-promotion, or accept a production approval.

## Approval binding

A proposal binds:

- environment / model / backend / architecture
- source commit
- binary SHA-256
- checkpoint SHA-256
- exact baseline and candidate policy hashes
- one role/layer delta with exact before/after n
- both G4 A/B/R and G6 restart-canary evidence hashes/run IDs
- request/token/duration/memory budgets
- expected epoch
- restart instance identity
- kill-switch scope and rollback plan

A dry-run approval binds the normalized proposal digest, issuer, nonce,
issued/expiry times and `DRY_RUN_TEST_ONLY` signature status. Self-approval,
expiry, forged digest and nonce reuse fail closed.

There is no trusted production signing implementation in this module.
`TRUSTED_PRODUCTION_SIGNATURE` is not accepted by the dry-run validator.

## State machine

```text
PROPOSED
-> SHADOW_EVIDENCE_VERIFIED
-> AWAITING_MANUAL_APPROVAL
-> APPROVAL_VALIDATED
-> PREPARING_CANARY
-> CANARY_RUNNING
-> CANARY_PASS_REVIEW_REQUIRED
or
ROLLBACK_REQUIRED -> ROLLBACK_PENDING -> ROLLBACK_VERIFIED
or
ISOLATED
```

A pass never auto-expands scope. Production remains disabled.

Rollback restores the exact approved baseline preimage. A wrong policy
self-report, apply/sync/restore failure, unknown restart state, kill request,
budget exceed, regression or inconclusive result never becomes success.

## Integration note

The implementation deliberately uses an injected in-memory `DryRunAdapter`.
Latest XOX-local GPU control files are newer than the remote base branch and are
not modified here. A0 must cherry-pick Agent E onto the certified BASE_SHA and
rerun these tests there before considering an adapter integration.

Any future production adapter bridge requires a separate review and a real
trusted approval-signature mechanism.
