# BEGLIN Qv19 pre-sign source provenance gate v2 — 2026-10-23

**Verdict:** `STANDALONE_AND_SYNTHETIC_AND_ADDITIONAL_PRESIGN_NEGATIVE_PASS / ORIGINAL_QWEN28_NATIVE_NOT_RUN`.

- Re-ran attached Qv19 standalone mutex suite: 2 real pthreads, 1,200 deterministic mock tasks, peak active=1, BUSY and poisoned fail-closed checks, ASan+UBSan PASS.
- Re-ran synthetic C bridge: 160/160 serial NLL matches under two worker threads, epochs unique, ungated mode and malformed options refused, ASan+UBSan PASS. **Synthetic Qwen-shaped fixture, not model outputs.**
- Audited the bundle: all 11 previously recorded SHA-256 values agree with actual files.
- Added `qv19_pre_sign_provenance.py` to the locally produced v2 research handoff. Before a Qv19 executable may be signed, it checks: original Qv17 full C SHA `0c9d797c...`; unchanged original gold-token scorer SHA `10c84007...`; exact parent-to-derived bridge patch; actual lease C/header hashes; original 23 Q4+n8 kernel-object hashes; and Apple arm64 Mach-O derived executable hash.
- Strengthened local dev attest/verify/run launcher to require an existing original Qv17 C source path, Qv17 23-object baseline build receipt and Apple Qwen build root. Do not sign from a generated build receipt alone.
- Added additional **7 synthetic/forgery rejection checks** to the local test suite. These reject fixture impersonation, missing source/object provenance, missing mandatory attestation CLI inputs and ambiguous gold-scoring anchors.

**Evidence boundary:** The EOE research path was inspected read-only: it has only `qv19_request_lease.h`, not the new lease C implementation. The direct upload had been blocked by connected safety controls and was not bypassed. Original Qwen28 native Qv19 compile, actual 8-case gold-NLL parity, same-process live-model concurrency and GPU evidence therefore remain **NOT_READY**. Original Qv17, checkpoints, Tailnet `g6-0.9.0-operator.900387`, GitHub main and operating services were not changed.

Full v2 bundle is available as a conversation attachment `BEGLIN_QV19_ORIGINAL_C_INTEGRATION_HANDOFF_V2_20261023.zip` (SHA-256 `cfffecfc3cf77e62f59402a03c5ba5d6f52cb5f0bcc9e1f708af87decafbebee`). The bundle is **local research handoff**; source files were not routed through a previously blocked remote transfer. Only an authorized host integration can perform the native build and NLL audit.
