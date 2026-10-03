# Precision Observability v1

## Scope
P6 adds request-level lineage and aggregate observability to the precision closed loop without changing any precision decision, production route, or native inference math.

The lineage connects:
signal -> evidence -> selected target/n -> expected cost -> actual transition/inference cost -> result.

Observability is opt-in per persistent worker. Production remains unconfigured in this stage.

The existing adaptive L26 two-pass path is also normalized into the same lineage schema when observability is explicitly configured, so a later cutover does not create a second incompatible telemetry format.

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

## Canonical exact-source sealed acceptance
Implementation commit:
4faab46dc6a44c5f41df2745d03976e16e05b305

Immutable evidence directory:
/Users/xox/vdsp_serving/precision-observability-sealed-4faab46-20261003

Hashes:
- result.json: 9c0c0fd6bf80cb80626462984bac70e416ee7e9410b06c28f78b241cb8d31c0e
- lineage.jsonl: f9368c6dd80321a89002a4724c874f5b87d650058d26f8b29149ec50efd497a2
- observability-summary.json: d8cffefef1ca9a1c0119814f479ad4ba78e311611d3c3f42352c194218782018
- pinned P6 cost evidence: 13cac65d375cf9ef2cb59d17a7f2b8204ce7ca9ae58c24426aac0955cce7c270
- native binary: a9b35e4e62480899962c6d3dc2890f5042c9a893abbae313b8366becb44cb028

The acceptance is source-bound to 4faab46dc6a44c5f41df2745d03976e16e05b305 and reports production_touched=false.

Six sequential admissions ran on one scratch worker PID:
1. startup-policy no-risk closed-loop admission;
2. low-margin closed-loop transition to the exact accepted target policy;
3. no-risk hold on that policy;
4. explicit warm restore to startup policy;
5. legacy adaptive L26 recovery with two inference passes;
6. explicit final restore to startup policy.

Measured aggregate metrics:
- admissions: 6
- low-margin triggered admissions: 2/6 = 33.333%
- transitioned admissions: 4/6 = 66.667%
- cache hits/misses: 4/2, hit rate 66.667%
- cache bytes materialized: 8,650,752
- inference-pass histogram: 1-pass=5, 2-pass=1
- extra-pass rate: 1/6 = 16.667%
- finite-logit rate: 100%
- resident qNg64 cache current/max: 17,301,504 bytes
- startup policy residency: 3/6
- closed-loop target policy residency: 2/6
- adaptive n6/n6 policy residency: 1/6
- target residency: L3 n6=4/6, L3 n5=2/6, L26 n5=3/6, L26 n6=3/6
- evidence use: xox-l26-production-low-margin=1, adaptive-l26-certified=1

Integrity checks:
- all six records are hash chained;
- duplicate admission IDs are rejected before append;
- same-PID before/after policy hashes and weight epochs must be continuous;
- hidden precision mutations therefore fail verification instead of being silently accepted.

Certified successor regression:
- run 37125908837
- head 4faab46dc6a44c5f41df2745d03976e16e05b305
- conclusion SUCCESS

Production remains unconfigured for P6 observability in this stage.
