# Precision Soak and Failure Injection v1

P7 validates long-running precision transitions without changing production live.

## Scope
- repeated atomic A<->B precision epochs
- same-PID epoch continuity
- qNg64 cache warm-up and plateau
- worker RSS drift guard
- concurrent admission serialization
- corrupt ACK fail-closed behavior
- tampered lineage detection
- scratch worker crash and clean restart
- append-only lineage rotation and bounded retention

The runner accepts --cycles, so the same harness scales from fast acceptance to 1000+ cycles. Failure injection paths are constrained to scratch worker/evidence roots.

## XOX pre-commit acceptance
24 cycles / 48 precision transitions on scratch PID 24432: PASS.

Measured:
- epoch 2 -> 50 with no discontinuity
- cache misses 2, hits 94
- qNg64 cache materialized 8,650,752 bytes during initial warm-up only
- final resident qNg64 cache 17,301,504 bytes
- RSS start 2,741,420,032 bytes
- RSS end/peak 2,757,738,496 bytes
- RSS delta 16,318,464 bytes
- concurrent B then A admissions serialized at epochs 50->51->52
- ACK corruption rejected before inference
- lineage tamper detected
- forced scratch worker termination detected and fresh worker restarted successfully
- rotation retention held at max 3 rotated files
- production_touched=false

Pre-commit result SHA-256: 65a94901ae497c959d8ba7b7cf6f5c59aea010646429ca10cc0736f85051154b

Production remains generation 6 on the existing live binary and P7 is not enabled there.
