# Beglin Dashboard Implementation Status

## Done
- P0 local bskel/becoder/beval inventory
- P1 FastAPI greenfield scaffold + clean bskel preflight
- REST contract (11 operations)
- SSE contract (1 operation)
- authenticated telemetry contract (3 operations)
- A0 evidence adapters
- controlled vs experimental variable separation
- SQLite append-only event journal with collision-safe idempotent event IDs
- heatmap/coverage/metrics/event-log projectors
- SSE event stream with cursor resume
- becoder generated harness build
- beval immutable contract cases validated against real remote snapshot commit
- custom neon React/Vite dashboard consuming the same REST/SSE contracts
- FastAPI same-origin integrated UI serving
- Agent F historical real-GPU coverage matrix replay pack
- deterministic replay projection into experiment cards, heatmap, coverage, metrics and event log
- XOX real runtime log format inspection
- XOX runtime stdout parser for promotion/request/runtime-summary/finite-logits records
- authenticated telemetry ingest into dashboard journal
- live heatmap activation/promotion updates
- live request coverage cells
- live TTFT / tok/s / finite-logits KPI projection
- UI SSE subscription for inference.request.completed and runtime.summary
- bskel VERIFY PASS for features 001/002/003
- becoder contract check PASS for contracts 001/002/003
- beval corpus validate PASS for cases readonly/stream/telemetry
- GitHub dashboard-contract-v1 snapshot updated and byte-verified at commit 47ae6f8bac2c1f6edd0dfada73b510a962340d1d

## Historical replay
- source: EOE coverage_matrix.json
- source SHA-256: ac263e769ab39464dcb327339cffdbd586e907e90a11bad9af797fd6115a7da2
- raw F matrix SHA-256: 33af3b40e25bdb4ca3f9e840962eebfedd8b8d25eb81ba344988842c8b7c3add
- F run: f-20260926T014343Z-52788
- replay pack: replay/f-20260926T014343Z-52788.json
- replay pack SHA-256: e4d6187c46feb6d7cb5b0b302fb7abc990deb637a9a91a763d661322ca0e809c
- canonical replay events: 20
- timeline semantics: replay_order_not_original_wallclock
- default combined journal on a fresh DB: 22 events (2 A0 + 20 replay)
- experiment carousel: 12 evidence-linked cards
- heatmap: Shared Up/L3 + KV/L11
- cumulative coverage: 17 evidence cells
- known verified request count projected: 72
- known corrected hits projected: 72
- shared_up/L3 n=7 regression is preserved as REGRESSION_DETECTED
- kv_a/L11 aggregate NO_BAD_CANDIDATE_WITHIN_BUDGET is preserved and never relabeled PASS
- decision-only reused-hardware rows do not count as new real-GPU samples

## Replay controls
- default: BEGLIN_REPLAY_AUTOSEED=1, instant idempotent seed for ordinary dashboard startup
- animation/demo mode: start backend with BEGLIN_REPLAY_AUTOSEED=0 on a fresh dashboard DB, then run tools/replay_pack.py with --speed or --instant
- replay IDs are stable, so replay is idempotent
- duplicate event IDs with different payloads fail closed

## Live telemetry
- ingest auth token is runtime-only: BEGLIN_TELEMETRY_TOKEN
- API header: X-Beglin-Telemetry-Token
- POST layer observation -> append-only layer.observed
- POST request completion -> append-only inference.request.completed
- POST runtime summary -> append-only runtime.summary
- parser supports the real XOX qwen_infer_gpu log format observed from g6_drill_sharedup3
- raw generated token arrays are intentionally not forwarded
- live projector updates heatmap, request coverage, event log, TTFT, tok/s and finite-logits KPI
- evidence-only startup still renders missing live values as unobserved, never fabricated

## Current UI
- controlled variables panel
- experimental variables panel
- evidence progress
- animated experiment carousel
- 32-layer heatmap with observed cells only
- cumulative coverage fill animation
- core A0 provenance graph
- KPI cards
- live TTFT and live tok/s cards when telemetry exists
- append-only event log
- provenance links to immutable evidence
- missing model/dataset/hardware fields remain explicit "미수집", never fabricated

## Sidecar E2E smoke
- source log: XOX g6_drill_sharedup3/post/worker.log
- parser result: 13 layer + 12 request + 1 summary + 1 validation = 27 canonical items
- projected live metrics: TTFT mean 92.67 ms, tok/s 185.308, finite_logits=true
- projected target: Shared Up/L3 n=6, PROMOTED
- smoke evidence: /Users/eoe/mcp-sandbox/tailnet-commander/BEGLIN_DASHBOARD_SIDECAR_SMOKE_20260927.json
- smoke DB was isolated and deleted after verification; main journal was not polluted

## Next
- XOX sidecar deployment: parser/bridge byte-identical to repo, runtime token synchronized, Tailnet cross-host path verified
- XOX→EOE sidecar E2E smoke: PASS (27 canonical items / 12 requests from real GPU log into isolated dashboard DB)
- main dashboard runtime: Tailnet http://eoe:8790 (VTuber-owned localhost:8787 left untouched)
- add paced replay demo launcher for browser presentation
- multi-experiment comparison view
- paper/export JSON/CSV/SVG-ready data
- optionally record additional runtime instrumentation if a future paper needs per-layer activation magnitude rather than hit/promotion intensity

## Safety
Dashboard read endpoints remain read-only toward production.
Telemetry write endpoints mutate only the dashboard event journal and require a runtime token.
Persistent mutations continue to live behind the separate signed production bridge.
The dashboard contains no shell/SSH/policy-write endpoint.
Raw tensors and raw generated token arrays are never written to the dashboard event stream.
