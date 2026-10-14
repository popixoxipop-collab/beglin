# Beglin P11 — Native In-flight REBIND_SET Crash Reconciliation Preregistration

2026-10-14 Asia/Seoul. Research-only, XOX single-worker failure-injection protocol. No automatic live promotion.

## Source and scope
Based on Beglin P10 commit 645e2eba85680f33f8f2148ce574935c279b07c4; original P5 C CB128 bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612, P9 cache MLX KV128 ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f. DeepSeek-V2-Lite index SHA d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf.

## Deliberate crash windows

1. first_bind: after binding the first target, before second target, inside REBIND_SET.
2. before_ack: after two bindings and native epoch mutation, immediately before durable REBIND_SET_APPLIED ACK.

An exact-SHA reversible C test patch writes/fsyncs one-shot marker, then parks test-only process for at most 15 seconds. Parent SIGKILLs only exact run-owned worker PID after validating marker and txn. Gate expiry fails closed with nonzero exit. No mock ACK.

## Recovery invariants

- Dead worker PID verified. Durable command exists with exactly two permitted targets; old ACK still reports preimage and has no terminal for interrupted txn.
- This crash-ambiguous transaction is not declared failed or committed and must NOT be replayed. Before restart, write/fsync quarantine intent, atomically move exact txn file to run-owned quarantine, fsync paths, then write/fsync hash-bound final reconciliation receipt. Partial receipts and duplicates block restart.
- Only after verification that old txn is absent, start a fresh worker at policy A and new PID. New separately authorized B-A-B admissions are compared to original P6 sequence hashes. Retain original uncommitted command in quarantine.
- No exact-once original request completion claim; original outcome UNKNOWN. This covers no power-cycle, host failover or multi-worker scenario.
- Immutable live stage, active-route, live PIDs, main and release plane.

## BF16

The raw original 29.256 GiB BF16 model exceeds safe currently available memory alongside active workers. A small read-only safetensors slice may be tested as a bounded feasibility probe, not full teacher logits or quality noninferiority. Production promotion remains BLOCKED.
