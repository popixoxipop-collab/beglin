# BEGLIN Qv19 — native integration handoff, not yet native-certified

**Date:** 2026-10-23 KST  
**Branch:** `research/qv19-single-process-lease-20261023`  
**Status:** `STANDALONE_AND_SYNTHETIC_BRIDGE_PASS / REAL_QWEN28_NATIVE_NOT_RUN`

The ChatGPT conversation attachment `BEGLIN_QV19_ORIGINAL_C_INTEGRATION_HANDOFF_20261023.zip` contains: the unchanged prior Qv19 C lease module, derived-source builder, original Qv17 source SHA guard, 23 original object SHA binding, native two-pthread callback bridge, developer Ed25519 derived-binary attestation and runner, independent C-log audit, synthetic model fixture, sanitizer tests, and run receipt.

**Completed locally:** Existing standalone model lease 1,200 requests with two real pthread workers, no overlapping forward, BUSY/poison fail closed, AddressSanitizer and UBSan PASS. New integration bridge compiled against an explicitly **synthetic** C scorer and produced 160/160 deterministic results matching its serial scorer across 20 runs; epoch stability, both thread IDs, max active=1, malformed input denial and opt-in gate PASS; ASan/UBSan PASS.

**Not completed:** Actual original Qwen28 C derived binary compile, actual Apple original-C logits/NLL parity, signed Qv19 executable, same-process original-model concurrency and KV correctness, or Metal GPU.

**Remote boundary:** A research-only integration-boundary note and request-lease header were placed on EOE. Transfer of the new request-lease implementation C was blocked by the connected tool's safety check. Do not bypass that block and do not mark native Qv19 complete.

**Pinned original Qv17 C SHA256:** `0c9d797c0066fdb1f8ca5972bdf7611ae75431050d0d55c0f0fdef0653f6bdaa`. Original forward/logits/gold-NLL source span SHA256 `10c84007027d15a9327c15ec687c2327e19390de65f03bf9d95aee4bd4c9fe72`.

**Required next gate:** When authorized, prepare a *new isolated* original-Qv17 C derived research binary with the packaged source builder; verify all 23 Apple Q4/n8 object SHAs, independently bind that newly derived executable via a local developer Ed25519 attestation (the old Qv17 signed manifest does not attest it), run 8+ frozen Qv16 cases under a single model-lifetime mutex with two pthread worker IDs; independently reparse original C NLL to original Qv16 statics and verify epochs stable, max active forward=1, poison=0, cache bound, no model/operating release changes.

**Safety limit:** This primitive serializes *all* global WT, shared KV, activation scratch and final logits during a request; it does **not** enable multiple simultaneous forwards. Original unguarded commands still exist; use only the research opt-in route until every serving entry path is gated. No production promotion.
