# Beglin HCP P9 — Prepared MLX qNg64 Cache, XOX Native Research (2026-10-14)

## Final classification

**Real XOX native MLX research demonstrates a 34.87% lower mean precision-switch admission latency with the opt-in prepared packed-weight cache.** This is scoped evidence for DeepSeek-V2-Lite MLA, the two exact shared-expert tensors at L3/L26, 5/6-bit qNg64, 106–112 input tokens, up to four generated tokens, and one GPU request slot. This is NOT generalized task-quality proof, production readiness or promotion authorization.

All native output sequences match the P6 source SHA; genuine REBIND_SET epoch/txn terminal ACK checks were preserved. Existing live binary, active precision route and stage PIDs remain unchanged. Automatic promotion OFF.

## Source and contract

- Repository: popixoxipop-collab/beglin
- Research branch: research/hcp-p9-mlx-prepared-cache-20261014
- P8 base commit: c5dcfd1153c19fc74da4f3331a33ece1f2cd872b
- Initial P9 source commit: a46154d8d3598ef1c7b0d51cd1d4447090c89ac4
- Original P5 CB128 C SHA-256: bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612
- Original P5 MLX KV128 SHA-256: daded3c0e4bf7cc2a86e0fab8c1f73322b5b0ffea0002d9b2d38857d2e9591ae
- New P9 opt-in cached MLX KV128 SHA-256: ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f
- Native benchmark binary SHA-256: 103549b7f7f4a3d142be883013956b4f1437759c8be76bff31f2451956df37fd
- Original model index SHA-256: d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf

New tools/hcp_p9_cache_patch.py requires the exact original P5 MLX Git blob and generates only a new isolated file; the inserted source sections are exactly reversible. The C code, real ACK, snapshot/restore, GPU KV epoch reset, routing mathematics and base checkpoint are unchanged. Default cache OFF; only the run-owned worker with BEGLIN_HCP_P9_PREPARED_CACHE=1 turns it ON.

Target whitelist: model.layers.3.mlp.shared_experts.up_proj and model.layers.26.mlp.shared_experts.down_proj, n in {5,6}. Only up to four prepared entries, 256MiB total retained maximum, 64MiB per packed entry. A cache hit requires unchanged source pointer, offsets, exact dimensions and bit width; full original packed code bytes and scale bytes are compared using memcmp. The reusable MLX packed array owns its allocation; the current source supplies a fresh scales array and bias expression on every binding. Mismatched source fails before changing an active binding. Hot reload, pointer reuse in foreign processes and simultaneous multi-worker binding are not certified.

## Actual native on/off crossover

Same compiled P9 binary, with only the environment cache switch varied. Four counterbalanced temporal blocks; each block covers 12 original P6 source-matched domains with ABBA or BAAB policy order. Four separate warmup admissions per worker excluded from results. The normal native Python precision scheduler verifies actual policy SHA, PID, two changed targets, txn id, epoch and terminal ACK. No corrective fallback pass allowed.

Total: **384/384 real requests successful**, corresponding to **192 exact cache ON/OFF pairs**. Across each mode, **142 switched** and **50 steady** requests. P6-generation sequence SHA: **384/384 match**.

| Measured admission wall ms | Cache OFF | Cache ON |
|---|---:|---:|
| Switched requests: mean, n=142 each | 874.4442 | 569.5269 |
| Switched p50 | 865.3134 | 566.7592 |
| Switched p95 | 913.2153 | 618.9705 |
| Switched mean native engine time | 853.3559 | 550.8889 |
| Steady requests: mean, n=50 each | 560.0703 | 564.1846 |
| Steady p95 | **581.8237** | **621.8075** |
| Switching RSS p95 (GiB) | 5.1207 | 5.1364 |

Paired switch Cache ON−OFF mean **−304.9172ms**. Relative reduction versus OFF **34.8698%**. Exploratory 4-block, 10,000-draw bootstrap 95% interval **[−314.0155, −290.5590]ms**. This compares real whole Python admissions with native REBIND_SET and ACK, but not HTTP/remote network latency. Four time blocks in one device and reused prompt corpus are not independent untouched datasets. This is not a broad model or production speed certification. **The steady p95 worsened in this sample**, and requires additional investigation.

All four cache-ON run-owned workers stored four cache entries apiece with **300 total cache hits**; disabled workers stored/hit none. There were no logged source mismatches, memory-budget denials, skipped stores or ACK errors. RSS remained below the preregistered 14GiB safety limit.

Before the full experiment, a 2-domain × 2-block smoke produced 32/32 successful admissions with mean switched OFF 861.3325ms and ON 564.7088ms. This is separate from the 384-request benchmark.

## Real failed-transaction rollback / recovery

A separate, **fault-only research binary** intentionally failed one multi-target transaction after its first successful GPU binding. The original P5 snapshot/rollback code then ran, with no mock ACK or bypass.

- Fault-only C SHA: 6611036998baa344a83809a526d241f7752530d89b9a9b9cbb0102f6a2466d2a
- Fault-only binary SHA: da7dd1571f2efd5c03231fd2b5935407314380189c92a0da6b4e3a6d1fdfad0b
- Python surfaced RuntimeControlError as expected; native terminal ACK status was REBIND_SET_FAILED.
- Original policy A hash and runtime epoch survived, the same worker PID remained active.
- Subsequent real B→A→B transactions yielded REBIND_SET_APPLIED and exact P6 generation SHA agreement. The prepared cache still produced a hit after rollback.
- Process exit 0. This does not prove cold-boot, power-loss, concurrent-session or cross-model recovery.

## Validation, provenance and preserved boundaries

P9 new source/fault regression 21/21 PASS; P8 original safety/source/governor regression 48/48 PASS; total **69/69 PASS**. Two distinct actual GPU binaries (main research and fault-only) compiled on XOX. Independent EOE read-only script recalculated all 384 results, 192 matched pairs, per-prompt P6 token SHA, four temporal-block intervals and source hashes. Source and output raw token values, full logits and checkpoint bytes were not committed.

Evidence directory: experiments/hcp_p9/evidence/native_xox_20261014/
Files: smoke.json, full.json, rollback.json, smoke_audit.json, full_audit.json, p9_new_tests.log, p8_existing_tests.log, MANIFEST.json.

Source result SHA-256:
- full.json: d6638759819d76d7eb2acabf08e705849b0c9b0974822a25111d699970be3f0b
- rollback.json: d13a15cd6a88ff2d7a9746c2494e3156f19be98e7eb595c35d078a1f1e7176ef
- P6 quality reference: 2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104
- Old stage, unchanged: 8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9
- Active route, unchanged: c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47
- Old live stage PIDs, unchanged: 35481, 35493

No main branch merge, release activation, live service worker restart, or production precision policy update.

## Next gate, not performed

1. Fresh untouched task-accuracy and BF16 teacher-based quality/noninferiority study, including 4+ generation tokens and multiple model architectures.
2. Multi-slot/concurrent workloads, interleaved policy requests, cache-owner lifetime under model hot reload, memory pressure, long-duration and cold-restart recovery, fault scenarios on cache mismatch and allocator exhaustion.
3. Repeat blocks in separate sessions with independent warm-up, measure steady-state tail regression and overall throughput (not only switched requests), compare fixed AF/A/B and dynamic governors under identical workloads.
4. Immutable release certification and explicit production safety review; any release change requires fresh current/live/runtime, stage+verify, new expected_current CAS. No auto-promotion by this study.

**Research cache speedup VERIFIED for scoped switch admissions; production/general model claims BLOCKED.**
