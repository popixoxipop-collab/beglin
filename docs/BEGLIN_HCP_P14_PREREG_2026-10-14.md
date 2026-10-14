# Beglin HCP P14 — Real Multi-Worker Precision Isolation, Crash Survivor, Full-BF16 Feasibility

Date: 2026-10-14 KST. Research-only, frozen before new native runs.

- Parent Beglin P13 commit: dc4d2dbc999633e75ff5ff0a430642ceefc830b3
- Branch: research/hcp-p14-multiworker-crash-gate-20261014
- Exact P5 C128: bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612
- Exact P9 prepared MLX KV128: ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f
- Exact same P9 GPU binary expected: 103549b7f7f4a3d142be883013956b4f1437759c8be76bff31f2451956df37fd.
- Active serving stage, route, PIDs 35481 / 35493 are immutable.
- No main merge, no release activate, no production precision policy write.

## Gate A: Two simultaneous actual isolated GPU worker processes

On XOX only, run two **independent controller processes**, with two run-owned native GPU worker processes, distinct queues, original real transactional REBIND_SET and ACK, bound resident cache per worker.

Resource eligibility requires CPU/GPU/SSD reservation and remaining live RAM at least 8 GiB before launch; measured worker RSS each below 7GiB, sum below 12GiB. GPU request slots 1 per worker and a total of 2. Abort on abnormal memory/swap pressure, with all run-owned workers closed. Initial warming excluded.

Two controllers execute matched 6 original P6 domains × 4 opposite policy turns each = 48 native admissions, with true simultaneous wall-time interval overlap tracked from independent monotonic clocks. Every result must match frozen P6 generation SHA, match exact epoch and terminal txn ACK in its own worker root. No cross-worker global policy state. Prior exact benchmark corpus is reused, so no quality generalization. No speed comparison claims.

## Gate B: Crash one worker while a second real worker survives

Use a separate exact P11 fault-only C binary, patched with reversible `before_ack` fsynced marker; survivor runs the unmodified P9 GPU binary. On actual paused native fault process, SIGKILL only the owned worker after fsynced marker, verify preimage and absence of the interrupted transaction's terminal ACK. Quarantine the ambiguous command with P11 durable INTENT/VERIFIED, **UNKNOWN and no retry**, and start a fresh worker only after verifying quarantine; new B→A→B requests must match P6 SHA. Concurrent surviving GPU worker must keep identical PID, policy ACK and generation output during and after peer's failure. Need not prove power outage / host-reboot or original crash txn exactly-once.

## Gate C: Whole BF16 reference admission

The original DeepSeek-V2-Lite 4-shard BF16 weights alone total 31,413,626,576 bytes ≈29.26GiB. Current XOX ~23GB available memory and high swap usage, EOE ~22GB, Alienware ~25GB RAM/9.5GB GPU free. Full BF16 teacher may not be resident safely without a verified layer-by-layer streaming implementation. No brute-force load or operational service stop. Probe tensor index/memory/architecture and choose safe streamed route if implementable in scope; **no logits quality PASS without actual teacher forward**. Local BF16 qNg64 code SHA consistency from P13 is already established but cannot substitute full-teacher score.

## Acceptance

Only actual non-mocked bounded work is marked PASS. Original stage, route, live PIDs and release convergence verified after. Independent receipt, SHA/ACK/timing verification; pre-admission and cleanup protected. Promotion stays BLOCKED.
