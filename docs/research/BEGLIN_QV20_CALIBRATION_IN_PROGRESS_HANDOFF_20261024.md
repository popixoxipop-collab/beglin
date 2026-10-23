# BEGLIN Qv20 — 동일 바이트 예산의 입력 조건부 정밀도 선택 연구 상태 (2026-10-24 KST)

**Repository:** `popixoxipop-collab/beglin`  
**Branch:** `research/qv20-prompt-aware-equal-byte-20261024`  
**Current verdict:** `PREEXEC_SPLIT_FREEZE_PASS / ACTUAL_ARCTRAIN_QV8_32_PASS / PROMPT_FEATURE_SYNTHETIC_PASS / RANDOM73_CALIBRATION_NOT_RUN / REAL_DYNAMIC_POLICY_NOT_TRAINED / HELDOUT_48_UNTOUCHED / NO_METAL_OR_PRODUCTION_PROMOTION`.

## Fixed scientific contract

Purpose: test whether a **prefix-token-only, calibration-trained selector** between two already known static 28n8 Q4/N8 projection masks achieves measurable Qwen28 C original next-token full-vocabulary NLL or choice accuracy gains on **new source-disjoint** data, under identical **projection packed-weight bytes**. This preliminary experiment is NOT unrestricted per-tensor optimization over all layers, NOT a cross-model-generalization proof, and NOT equal **total RAM** because a 96MiB additional inactive alternate packed cache is needed for dynamic switching.

- The two masks `source_qv8` and `random73` have the same 28 n8 role cardinalities and `765,493,248` projection packed bytes. Original role-mask SHA are respectively `9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922` and `5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde`.
- Calibration only: ARC-Challenge **train 32** (short/long 16+16), private input SHA `6bedf97854f140939b248001a491b6f7ae38a078b502a8af9a85d3a47cc99f28`.
- Independent test A: ARC-Challenge **validation 24** (short/long 12+12), exclude Qv14/Qv16 prior **76** ARC validation IDs; SHA `b0a27e22f0c5769f49c3dc95d2200feef9f06747349abef55bd82c29d13e2119`.
- Independent test B: OpenBookQA **validation 24** (short/long 12+12), distinct split from Qv13's OpenBookQA test; SHA `2f80539f7eca065e8dd5effcc1c320e2fbcf74f7e6af0e1e2816a8fe72aeb204`.
- **Preexecution public immutable input lock SHA `2aae8540fef6bce9c15e4c71becd873f62d69da556056583a41bae810ec5da1a`**. Published on GitHub before any Qv20 fresh heldout model forward. Input manifests have been independently validated for 80 unique questions across three cohorts, four unique one-token A/B/C/D target IDs per question, 3-arm binary input byte parity; no cross-cohort duplicate question hashes.
- The heldout Qv20 original per-item model logits, full-vocab NLL and accuracy **must not be read for ridge fitting, feature selection, threshold tuning, or model promotion**.

## Training-only policy (precommitted before second calibration result)

Source `tools/qv20_fit_prompt_policy.py` SHA-256 `10c3276d6e61aac036bcb5bb53b07ca3fd10a43cc0a4759a3dd4dd0cdd687df3`.

- Prefix-only features: `log(prefix_len/85)`, fraction of unique IDs, fraction of IDs >=100000, mean token ID normalized by 151936, and adjacent token repetition fraction.
- Predict winsorized `random73_NLL - Qv8_NLL` using ridge with fixed regularization `alpha=6.0` and differences clamped to ±2.0.
- Select Qv8 if predicted advantage >0.10, random73 if <−0.10; otherwise the best static calibration arm.
- Complete leave-one-out crossvalidation on ARC train 32 must beat the best calibration static NLL by **≥0.03** and select each arm ≥4 times. If either condition fails, the policy must **fall back to a single static mask**. Under no circumstances retroactively weaken thresholds based on test.
- `tools/qv20_prompt_feature_synthetic_test.py` validated **132 synthetic five-dimensional prefix feature vectors**, strict threshold semantics, and zero-information ridge fallback to either requested static control. **No actual model logits or test dataset are involved in this synthetic PASS.**
- The original Qv19 `qv19_guarded_choice` behavior must be preserved in the Qv20 derived C; only opt-in `qv20_prompt_aware_choice` uses the fitted router. The research builder was updated on this branch to guard that mode with `BEGLIN_QV20_POLICY_SHA256`; the guarded model lock must cover source-only precision choice → live Q4/n8 WT swap → entire shared KV forward → final logits/NLL. Note: **code changes are committed but the trained policy-derived C has not yet been built**.

## Actual native calibration receipts

**Completed:** real original Apple Qwen28 C, ARC train 32, fixed Qv8 source mask; managed EOE `job_192e789b14812afb4fb95b2c090b4581` `SUCCEEDED / exit=0`. **19/32 correct, mean gold full-vocab token NLL 1.53042149**, native CPU wall 445.87s, peak process RSS 4,638.42MiB. Aggregate SHA `4696d663f66ae7939553b4e66c748e4cab4e10395d4e70a07e49bd8b94e0e746`, private per-item receipt SHA `e580e5c133755c585097b2457abc0ff33db700d682c12511d3c9aa420137a9ca`. Private original case scores and tokens stay owner-only in D50 and are **not** committed to GitHub.

**Not completed:** actual original C random73 32-case calibration because EOE scheduler rejected requests repeatedly with `NO_CAPACITY: insufficient_free_cpu`. One-core actual C scorer `tools/qv20_actual_static_native_onecore.py` has been published and syntax verified and is explicitly tagged `Q4_THREADS=1` instead of source Qv8's 2; this means any elapsed wall comparison is invalid, even if per-case numeric model outputs are comparable. No successful random73 job ID exists at this handoff. Do not invent results.

**Additional host-access issue:** owner-only `/Volumes/D50/TailnetCommander/work/beglin-qv20-prompt-aware-20261024` and its generated files now sometimes return `PermissionError: [Errno 1] Operation not permitted` in EOE connected **short-read command and read_file**. Their `stat` shows ownership `eoe`, mode 0600, macOS provenance attribute. This tool restriction's cause is **not definitively diagnosed**. Neither permissions/extended attributes nor scheduler policy were altered/bypassed to obtain results. Before resuming, verify that a managed job has legitimate read access; do not rely on a short-read tool's access if it fails.

## Follow-up execution incident (2026-10-24 KST)

After a fresh managed `resource_plan` briefly reported EOE CPU capacity for the one-core `random73` calibration, an authorized `resource_submit` using the pinned `tools/qv20_actual_static_native_onecore.py` returned **`STORAGE_PREPARE_FAILED`** during managed storage scratch preparation, before a worker started. Registered job `job_612b2edc1ca096a4af656638363fda73` was independently confirmed **`FAILED`** with `termination_reason=storage_prepare_failed`, `worker_pid=null`, `child_pid=null`, and no native exit code. It **did not produce an actual random73 model score** and must not count as a completed experiment.

The storage-pool error may be related to, but is not proved caused by, the separately observed macOS `Operation not permitted` when reading owner-only D50 research files in the connected tool. Original signed sources/inputs and their provenance attributes were not modified; no privilege escalation, extended-attribute change, undocumented storage-pool switch or process workaround was attempted.

A pure **synthetic-only** prefix-feature contract test also passed **132 deterministic five-feature vectors**, strict ±0.10 sign-threshold choice, and zero-information ridge fallback to either static policy. It ran without loading any model or heldout cases and was committed as `tools/qv20_prompt_feature_synthetic_test.py`. This is **not** real Qv20 calibration evidence.

The Qv20 research builder in GitHub was further corrected to keep the original Qv19 guarded command's 85-token policy unaffected and enable the Qv20 prompt-conditioned routing only under `qv20_prompt_aware_choice` plus an exact `BEGLIN_QV20_POLICY_SHA256` environment guard. Its matching Qv20 heldout runner was updated. These commits have **not** been reapplied as a signed native Qv20 binary on EOE; a connected source-sync request was blocked by the safety gate. Native compilation/real heldout evaluation remain `NOT_RUN`.

## Next executable sequence (strictly in order)

1. Wait for the managed EOE CPU scheduler to admit a legitimate **1-core/1-thread** or **2-core/2-thread** ARC train32 random73 job, using the existing frozen inputs and exact original C binary SHA `f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221`. Validate per-question original C logs, immutable receipt, role counts and mask SHA. Do not rerun original Qv8 unless a specific numerical mismatch requires it.
2. With **both calibration arms actually present**, run `tools/qv20_fit_prompt_policy.py` once, store frozen private/public policy SHA and LOOCV pass-or-fallback verdict. If two original numerics are not available, it must fail closed and NEVER substitute synthetic results.
3. Only after policy freeze, build isolated Qv20 original 28-layer C via `tools/qv20_build_prompt_selector_native.py`; verify original Qv19 model lease and original C math/23 kernels SHA, exact two-mask activation, calibrated policy digest, real Apple arm64 binary. Explicitly bind new binary to the existing development-only Ed25519 signing flow if required, never mislabel as a production certificate.
4. Actual heldout ARC validation24 and OpenBookQA validation24: run source Qv8 static, matched random73 static, and Qv20 selected masked C on each split; **6 original C jobs total**. Reparse each original C NLL log; assert Qv20 per-case numeric equality with the actual chosen original static arm, verify 2 pthread epoch stability and 96MiB inactive cache cap.
5. Invoke `tools/qv20_independent_heldout_audit.py` only after all jobs, using paired prompt-stratified 30k bootstrap and exploratory Holm for two controls. Require that accuracy, NLL and true RSS/cold+warm CPU latency evidence be reported separately; do not claim significance without supporting intervals and strong evaluation design.

## Product and scientific release policy

This is an **algorithm- and performance-research branch**, not a production auto-routing feature. Source-only feature routing between two known fixed masks is NOT the same as an optimizer capable of assigning independent precision per tensor. Additional per-request cache RAM means neither equal total system-memory Pareto optimality nor Metal GPU correctness has been demonstrated.

BEGLIN original BF16 checkpoints, Qv19 original C binary, original 23 Q4/n8 kernels, production main and immutable G6 `g6-0.9.0-operator.900387` were not changed. Until real calibration random73 and post-freeze heldout runs, **Qv20 cannot claim trained dynamic quality advantage, and MUST remain NOT_READY**.
