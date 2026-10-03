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

## Exact-source XOX acceptance
Implementation commit:
4faab46dc6a44c5f41df2745d03976e16e05b305

Acceptance result:
/Users/xox/vdsp_serving/precision-observability-acceptance-4faab46/result.json
SHA-256:
dfcb7e304ef13fd10c28e78b009b027ff7d4571a6c57668991fe183df3952201

Lineage JSONL:
/Users/xox/vdsp_serving/precision-observability-acceptance-4faab46/lineage.jsonl
SHA-256:
a02461d0dbdb7c35906bf68d5fddf7d93e100564f1452064b27064c2a5474ea5

Atomic summary snapshot SHA-256:
ac83cc80528dcf07ee8ee479828a28cc671b12b18584d668ee05fe5659cc7130

Pinned P6 cost evidence SHA-256:
13cac65d375cf9ef2cb59d17a7f2b8204ce7ca9ae58c24426aac0955cce7c270

Native binary SHA-256:
a9b35e4e62480899962c6d3dc2890f5042c9a893abbae313b8366becb44cb028

The exact-source acceptance ran six sequential admissions on the same scratch worker PID and verified policy/epoch continuity across every lineage record:
- A no-risk closed-loop admission
- A -> B low-margin closed-loop transition
- B no-op hold
- B -> A explicit-policy restore
- A -> adaptive L26 recovery with two inference passes
- adaptive policy -> A explicit final restore

Measured aggregate result:
- admissions: 6
- low-margin triggered admissions: 2 / 6 = 33.3%
- precision transitions: 4 / 6 = 66.7%
- cache hits/misses: 4 / 2, hit rate 66.7%
- cache bytes materialized: 8,650,752
- inference-pass histogram: one-pass=5, two-pass=1
- extra-pass rate: 1 / 6 = 16.7%
- finite-logit rate: 100%
- startup policy residency: 3/6
- closed-loop target policy residency: 2/6
- adaptive n6/n6 policy residency: 1/6
- target precision residency: L3 n6=4/6, L3 n5=2/6, L26 n5=3/6, L26 n6=3/6
- evidence use: xox-l26-production-low-margin=1, adaptive-l26-certified=1

The append-only chain also rejects duplicate admission IDs before write and detects same-PID policy/epoch discontinuities, exposing hidden precision mutations rather than silently accepting them.

Certified successor regression:
- run 37125908837
- head 4faab46dc6a44c5f41df2745d03976e16e05b305
- conclusion SUCCESS

## Exact-source XOX acceptance
Implementation commit:
4faab46dc6a44c5f41df2745d03976e16e05b305

Fresh cost evidence:
/Users/xox/vdsp_shadow_runs/precision_e2e_cost/p6-observability-4faab46/result.json
SHA-256:
d0a7f12d58a9b18090717b75a16669364927f7bb39e8724106cb178b400754ff

P6 acceptance:
/Users/xox/vdsp_serving/precision-observability-acceptance-4faab46/result.json
SHA-256:
696315c359b5c0ffce00903d2160c17c84659320fb8a1e261b9d107ac1dca8dc

Observed lineage:
- 6 admissions / 6 requests on one scratch worker lineage
- 6/6 observability records RECORDED
- low_margin trigger count 2, trigger rate 1/3
- 4 transitioned admissions
- cache hits 4, misses 2, hit rate 2/3
- 8,650,752 cache bytes materialized
- inference-pass histogram: 1 pass x5, 2 passes x1
- extra-pass rate 1/6
- finite-logit rate 100%
- exact-policy and role/layer/n residency shares recorded
- adaptive L26 evidence and closed-loop L26 evidence both appear in evidence-use counts
- final scratch state restored to reviewed startup policy
- production_touched=false

Lineage JSONL SHA-256:
14a06b51c2b0ecf3fe955918247bb3a68d305b229b877b5d089d64c5e30f4fa8

Summary snapshot SHA-256:
f6368eb4b4206cb9c80db998224be85d8344ccb642295f247ee5c3788f4e0f08

Certified successor regression:
- run 37125908837
- head 4faab46dc6a44c5f41df2745d03976e16e05b305
- conclusion SUCCESS
