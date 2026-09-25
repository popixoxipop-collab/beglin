# Agent E Handoff — Manual Approval Canary Contract

Status: **READY_FOR_REVIEW**  
Branch: `agent-e-manual-canary-20260926`  
Remote staging base: `018c0147dcb7853642078e359606ca876678d85f`  
Implementation/CI head: `8e824fa7ed728be76ecc4d31a43876707337813e`

## What Agent E implemented

A dry-run-only manual approval canary layer:

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

The proposal binds environment/model/backend/architecture, source commit,
binary/checkpoint SHA-256, exact baseline/candidate policy hashes, exactly one
role/layer before/after-n delta, both G4 and G6 evidence hashes/run IDs,
request/token/duration/memory budgets, expected epoch, restart instance,
kill-switch scope and rollback plan.

The dry-run approval binds the normalized proposal digest, issuer, validity
window and nonce. Self-approval, forged digest, expiry, nonce reuse and any
non-dry-run/production-write approval fail closed. There is deliberately no
trusted production signing implementation in this branch.

## Fail-closed controller behavior

- strict method/state ordering; no start before approval
- exact one-target preimage required
- baseline policy/epoch drift rejected before candidate apply
- kill before apply -> ISOLATED
- kill during canary -> ROLLBACK_REQUIRED
- every request/token/time/memory budget axis is bounded
- regression and inconclusive outcomes require rollback
- apply/synchronize/restore failures -> ISOLATED
- rollback only from ROLLBACK_REQUIRED
- exact approved candidate must be present before rollback
- rollback ACK must match txn id, previous epoch, advancing epoch and exact
  baseline policy hash
- restart reconciliation never invents a successful rollback
- a dry-run pass terminates at CANARY_PASS_REVIEW_REQUIRED; there is no auto
  expansion or production adapter.

## Production isolation

The controller imports no `backend_adapters`, `gpu_runtime_control`,
network/Supabase libraries or subprocess. CI parses the AST and enforces that
isolation. All controller files use `production_write_allowed=false`.

## Verification

GitHub Actions run: **36153245385**  
Job: **108131474709**  
Result: **SUCCESS**

```text
compile                         PASS
production isolation guard      PASS
manual canary unit tests        35/35 PASS
dry-run journal fixture         PASS
artifact upload                 PASS
```

Dry-run evidence artifact:

```text
name   agent-e-manual-canary-evidence
id     10872940087
size   3340 bytes
sha256 ee30d2d352b0c6b62205e13a7354b7813e01148455c9b77455f1653c0194c225
```

Fixture terminal states:

```text
happy    CANARY_PASS_REVIEW_REQUIRED
rollback ROLLBACK_VERIFIED
```

These are **simulated evidence only**, not real GPU or production canary runs.

## Important integration limitation

This branch started from the remote staging branch, not from the latest XOX
certified local HEAD. Agent E therefore does **not** claim integration VERIFIED.

A0 should apply the workflow-excluded squash commit from
`agent-e-manual-canary-squashed` onto the current certified BASE_SHA, rerun
the Agent E suite there, and only then bind actual B/F evidence.

## What remains intentionally absent

- production MLX adapter bridge
- service restart or live mutation
- Supabase write
- GPU auto-promotion
- trusted production approval signature
- automatic scope expansion

Any production-writing controller is a separate future review.
