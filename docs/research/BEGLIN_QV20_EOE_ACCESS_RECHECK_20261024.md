# BEGLIN Qv20 — EOE access-ready verification (2026-10-24 KST)

Research branch only. No heldout model execution or production change.

## Direct EOE observations
- Runtime identity: `g6-0.9.0-operator.900387`; PID `80584`. Release current/live/runtime equal; converged=true, active_operation=null.
- Read-only `tools/qv20_eoe_readonly_access_preflight.py`: exit code 0; `ACCESS_READY`; **6/6** required source/model/input/mask/receipt paths readable by UID 501. This includes the original Qwen BF16 weights resolved onto D50.
- Prior failed managed job `job_612b2edc1ca096a4af656638363fda73`: still `FAILED`, `storage_prepare_failed`, no child PID or actual score.
- These observations **do not establish** scratch-write readiness or CPU scheduler admission. Subsequent plan/inspection calls timed out, and no new managed scoring job has been confirmed or counted.

## Strict continuation
1. Reconfirm read-only preflight 6/6 and source script hashes.
2. Check the existing random73 calibration output directory before any retry. Preserve all artifacts; use only approved exact-case empty-dir recovery.
3. Obtain a fresh managed CPU/storage resource plan; if and only if ready, submit exactly one pinned original Qwen28 C train32 random73 scorer with new idempotency key. Never switch to an untracked/manual inference process.
4. Verify original C per-item logits, full-vocab gold-token NLL, role counts, frozen hashes, and immutable private/public receipts. Do not treat synthetic selector tests as scoring evidence.
5. After both original train32 static arms pass, run unchanged train-only `tools/qv20_fit_prompt_policy.py` exactly once; require preregistered LOOCV gain >=0.03 and >=4 selections per arm, else static fallback. Freeze policy SHA before any of the 48 sealed validation prompts are scored.
6. After freeze, build isolated signed Qv20 C for ARC validation24 and OpenBookQA validation24 three-arm comparisons, then independent audit.

No permissions, TCC/ACL/xattr, BF16 checkpoint, original C binary, sealed heldout split, `main`, or G6 immutable release modified.
