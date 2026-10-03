# Precision Observability v1

## Scope
P6 adds request-level lineage and aggregate observability to the precision closed loop without changing any precision decision, production route, or native inference math.

The lineage connects:
signal -> evidence -> selected target/n -> expected cost -> actual transition/inference cost -> result.

Observability is opt-in per persistent worker. Production remains unconfigured in this stage.

## Lineage contract
Each closed-loop admission records:
- admission ID, timestamp, worker PID and request count
- normalized signal and active triggers
- evidence snapshot, allocator, selector and cost-evidence hashes
- per-target decision status and exact evidence IDs/hashes
- before/after policy hash and weight epoch
- selected resident policy and target n changes
- expected policy/target cost when available
- actual transition wall time, cache hit/miss, cache bytes added and resident cache bytes
- actual inference pass count, engine time and roundtrip time
- finite-logit outcome plus a SHA-256 of responses

Raw prompts are not written to the lineage.
## Integrity
The JSONL lineage is append-only and hash chained.

Each row contains prev_record_sha256 and record_sha256. Before every append the existing chain is revalidated. Duplicate admission IDs, changed records, broken predecessor hashes and malformed JSON fail verification.

A summary JSON is atomically replaced only after the lineage append has been fsync'd.

Strict audit mode is available for acceptance/testing. If strict recording fails after inference, the error is surfaced explicitly; observability does not pretend to roll back an already-completed precision transition.

## Aggregate metrics
The summary exposes:
- admissions and request count
- trigger counts, trigger rate
- transition count/rate and target transition counts
- cache hits/misses/hit rate and materialized bytes
- current/max resident qNg64 cache bytes
- inference-pass histogram and extra-pass rate
- finite-logit rate
- exact-policy residency by admission share
- role/layer/n precision residency by admission share
- evidence-use counts
- expected E2E mean, actual roundtrip mean and mean prediction error
- current lineage head SHA-256
## Production boundary
P6 does not enable observability in the live generation-6 supervisor.
No production binary, route generation, adaptive policy, or auto-promotion setting is changed.

Exact-source XOX acceptance and certified regression are sealed after the implementation commit is fixed.
