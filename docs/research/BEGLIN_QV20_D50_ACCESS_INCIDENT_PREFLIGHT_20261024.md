# BEGLIN Qv20 — D50 / Qwen BF16 access incident, verified prerequisite gate

**Observed:** 2026-10-24 KST  
**Branch:** `research/qv20-prompt-aware-equal-byte-20261024`  
**State:** `BLOCKED_ACCESS_EP ERM` (formal preflight: `NOT_READY_EXTERNAL_STORAGE_ACCESS`) / original random73 calibration **NOT SCORED**.

## What was actually verified

- A new *read-only* `tools/qv20_eoe_readonly_access_preflight.py` was committed, read back from GitHub, materialized in the ordinary EOE sandbox workspace and actually executed on Mac Studio EOE. It returned **exit 4** and `NOT_READY_EXTERNAL_STORAGE_ACCESS`.
- The exact six probes: the original unchanged Qwen28 native executable was **READABLE**; all **five** inputs/data paths on D50 were **BLOCKED, `EPERM`**, including Qwen2.5-1.5B original BF16 model safetensors, Qv20 public preexecution manifest, calibration 32 input manifest, completed source Qv8 receipt and frozen random73 role mask. **No bytes from those protected files were read.**
- This is not a defect unique to the Qv20 directory. Read-only probes of the older Qv14/Qv15/Qv16/Qv17/Qv19 owner-only directories also return `PermissionError [Errno 1] Operation not permitted`.
- The HF source model path under `/Users/eoe/.cache/huggingface/.../model.safetensors` is a **symlink** whose resolved target lives under `/Volumes/D50/TailnetCommander/work/EOE_EXISTING_LAYOUT_REBALANCE_20261012/model-cache/...`. Consequently, selecting the registered internal `eoe_internal` scratch pool does not make the original model weights readable: the target is still D50.
- The current process UID is 501 (`eoe`). Directory and file POSIX mode are `0700` / `0600` and ownership is `eoe`, but actual `open()` and `listdir()` still fail with `EPERM`. This is consistent with *app-/OS-level privacy or sandbox access control*, but the precise responsible macOS rule is **not confirmed**.
- The observed process parent chain was `tunnel-client` (under `/Users/eoe/projects/tailnet-commander/vendor/tunnel-client/v0.0.14/extracted/`) → Node v24.19.0 → Xcode Python 3.9 → Homebrew Python 3.11. The privacy entitlement, if missing, may be associated with one of these process identities.
- The previous random73 job `job_cbdfe07861a2fbd0a616c3af9325fa07` ran about 99s but its worker then failed with `PermissionError` trying to create `native_original_C_static/cal_arc_train/static_random/native_original_C_Qwen28.log`. Its C result is **not persisted/verified**.
- The later `job_612b2edc1ca096a4af656638363fda73` has confirmed `FAILED`, `termination_reason=storage_prepare_failed`, `worker_pid=null`, `child_pid=null` (the managed CPU job did not begin). A shared access cause is plausible but not established from these separate errors.
- The EOE scheduler has intermittently reported allocatable CPU again, and `eoe_internal` is a registered scratch pool with only about 0.2 GiB allocatable; CPU/alternate scratch capacity alone **cannot resolve** the blocked 3GB model on D50.

## Access restoration must be authorized through the operating system

On the Mac Studio (EOE), inspect **System Settings → Privacy & Security → Files & Folders** for the process that launches Tailnet Commander, particularly `tunnel-client`, and permit only the required volume/files if macOS offers that choice. If necessary, explicitly review **Full Disk Access** for the correct executable in the parent chain and restart the affected connector process through the approved service control method *after the owner approves the privilege*. Apple's settings are intentionally controlled by the signed-in user.

**Do not** automatically alter `chmod`, owner, ACL, `com.apple.provenance` or SIP/TCC databases; do not use root/broker to bypass a denial; do not route a denied D50 file through an unrelated connector; do not regenerate heldout results as if they were original signed source receipts.

## Exact safe recovery gate and next scientific steps

Run in the authorized EOE sandbox after access is restored:

```bash
/opt/homebrew/bin/python3.11 -B \
  /Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv20-native-github-20261024/tools/qv20_eoe_readonly_access_preflight.py
```

This read-only check must return **`ACCESS_READY` / exit 0** (six of six file probes readable). The script does not prove scheduler scratch write permission, so next perform a managed resource plan/submit through the registered scheduler before actual inference.

Then, in order: rerun only missing `ARC-Challenge train32 random73` original native C equal-byte calibration, verify its original token NLL/receipt; freeze the **train-only** ridge/LOOCV policy; *after that* build signed Qv20 derived original C; then score the two untouched validation splits against static Qv8/random73 controls and perform the precommitted independent paired bootstrap/Holm audit. No comparison claims prior to those receipts.

**Scientific boundary:** Qv20 source Qv8 calibration 19/32 correct, mean full-vocab gold next-token NLL about 1.5304, is from a prior successful real native C job. No authentic random73 calibration score or learned Qv20 policy exists yet. Original Qwen/Qv19/Metal kernels, GitHub main and G6 operator release were not altered.
