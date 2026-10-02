# Production Routing Cutover Gate

## Current status

Beglin has a verified MLX/Metal isolated candidate bridge, but this repository
does not currently contain a production ingress/router/supervisor that decides
which worker receives user traffic.

Therefore the production-routing gate is intentionally:

`BLOCKED_NO_VERIFIED_SERVING_ROUTER`

This is not a candidate-quality failure. The approved candidate passed the
isolated XOX canary. The missing component is the traffic-routing layer itself.

## Evidence already closed

- GitHub OIDC/Sigstore operator approval: verified.
- Production-adapter bridge plan: OIDC re-verified.
- XOX isolated candidate:
  - shared_up_proj / layer 3 / n=6
  - promotion ACK: applied
  - finite logits: true
  - reference token 1224: 12/12
  - 12 requests / 120 tokens
  - 5.971 s
  - peak RSS 6,321,586,176 bytes
  - baseline worker untouched
  - candidate worker exited
  - production routing unchanged
- Bridge verdict: `CANARY_PASS_REVIEW_REQUIRED / NO_AUTO_PROMOTION`.

## Cutover controller

`tools/production_routing_cutover.py` provides:

1. exact binding to the verified production-adapter bridge result;
2. explicit baseline and candidate route identities;
3. a verified-router capability contract;
4. generation-based compare-and-swap;
5. atomic/fsync route-manifest publication;
6. explicit cutover approval;
7. bounded health evaluation;
8. rollback by CAS to the exact baseline route.

Automatic cutover is not supported.

## Required serving-router capability

Before real traffic can be changed, the serving supervisor must publish a
capability object with all of the following verified:

- atomic CAS route change;
- request admission/drain semantics;
- post-cutover health observation;
- rollback;
- an actual route-store implementation;
- an explicit router identity.

Until then `configs/production_router_capability_20261002.json` remains
`status=ABSENT`.

## Shadow validation

The same atomic manifest code was executed on the real XOX filesystem without
network traffic:

- generation 1: baseline
- generation 2: shadow candidate
- healthy window: `CUTOVER_HEALTH_PASS_REVIEW_REQUIRED`
- injected reference-parity failure: `ROLLBACK_REQUIRED`
- generation 3: baseline restored
- final route: baseline
- network traffic touched: false

This proves file publication/CAS/rollback semantics only. It does not claim
that any production network ingress has been switched.

## Safe integration point

The next implementation should be a small serving supervisor in front of
long-lived Beglin workers. The supervisor, not the inference kernel, should own:

- request ingress;
- the active-route manifest;
- admission pause/drain;
- atomic active-worker selection;
- health-window accounting;
- rollback to the prior route.

The inference engine remains responsible for model execution and per-worker
precision state.

## Cutover rule

Even after a router exists, one successful isolated candidate does not imply
automatic promotion. A separate explicit cutover authorization must bind the
exact cutover-plan SHA and route IDs. After switching, the health window can
only end in:

- `CUTOVER_HEALTH_PASS_REVIEW_REQUIRED`, or
- `ROLLBACK_REQUIRED`.

There is no automatic expansion path.
