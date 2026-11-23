# JE01: 48-direction TRAIN240 follow-up plan

**Date:** 2026-11-23. **Status:** PLAN / SOURCE METADATA VERIFIED; no TRAIN240 derivative or allocation result is claimed here. Original AF4 remains the policy. Review base: PR #94 HEAD `a0bac6efaa11182177e88c3f6561fabbebc2f758`.

## 1. Measured boundary and controlling contract

The three L12/E13 Gate/Up/Down **Q4** native full-prefix JVP directions passed independent raw-output verification on the previously seen **TRAIN16 / 656 gold tokens**. The final audit canonical is `5ce09188e65d5cfcc0f956447b2918fe9c80bf81f14e8486ba97864f7137b87b`. This validates those derivatives under fixed hard routes; it does not establish finite-step natural-route stability, all 48 directions, TRAIN240 fit, or likelihood improvement. [Current evidence and limitations][context]

The [next-gate contract][nextgate] controls the new 48-coordinate objective; the [original preregistration][prereg] retains final heldout, routing, latency, placebo, and independent-review requirements. The earlier TRAIN240 FFN/SSE allocation is training evidence, not an independent holdout and not NLL curvature. Existing A/B/C TRAIN16 results must not select this allocation or tune its thresholds. [CAL240 decision][caldecision]

**Gate order:** independent full48 TRAIN16 audit PASS → source/build/runtime preflight → TRAIN240 collection → independent matrix audit → one frozen policy → paired native development diagnostic → untouched validation128/OOD128. Failure records HOLD; it does not trigger outcome-dependent retries, new epsilon choices, or a different candidate after viewing heldout results.

## 2. Exact roles, coordinates and budget

The immutable bank order is slots below, each in Gate, Up, Down order. Coordinate `j = 2*r + tier_index`, with Q4 index 0 and Q5 index 1. AF4 means both coordinates zero. [Bank source][bank]

| Role indices | Layer / expert |
|---|---|
| 0 / 1 / 2 | L12 / E13 |
| 3 / 4 / 5 | L13 / E22 |
| 6 / 7 / 8 | L13 / E28 |
| 9 / 10 / 11 | L14 / E63 |
| 12 / 13 / 14 | L20 / E17 |
| 15 / 16 / 17 | L21 / E61 |
| 18 / 19 / 20 | L24 / E26 |
| 21 / 22 / 23 | L24 / E60 |

Physical role costs are AF4 **1,802,240 B**, Q4 **1,622,016 B**, Q5 **1,982,464 B**. Enforce **12 AF4 + 6 Q4 + 6 Q5 = 43,253,760 B**, plus one tier per role. Equal total bytes alone is insufficient: the exact counts remain mandatory. The 72-entry bank catalogue is 129,761,280 B; it is not the selected deployment bundle. Preserve native Gate/Up shapes and six separate Down SwiGLU inputs.

## 3. Recovered TRAIN240 source chain and paths

Let `R=/Volumes/D50/TailnetCommander/work/je01_role_directional_20261123_v1` and `A=/Volumes/D50/TailnetCommander/work/je01_real_original256_activation_20261122_v1`.

| Purpose | Exact existing path under these roots | Verified status in this review |
|---|---|---|
| Original TRAIN256 gold and prefix lengths | `R/train_gold_source/JE01_ORIGINAL_REAL_TRAIN256_FULL_GOLD_IDS.json` | All 189,936 bytes read; file SHA and canonical independently recomputed |
| Existing 15-chunk preparation | `R/train240_inputs/JE01_TRAIN240_INPUT_PREPARATION_RECEIPT.json` | All 12,298 bytes and canonical checked; reports preparation only |
| Prepared TRAIN240 tokens | `R/train240_inputs/inputs/<input_id>.bin` | Located through existing manifest contract; raw files not reopened in this review |
| Per-chunk input contract | `R/train240_inputs/chunk01..15/{dataset.json,manifest.txt,gold_meta.bin}` | Chunk01 dataset bytes/canonical checked; all 15 dataset pins recovered from preparation |
| Original BOS + prompt binaries | `A/portable512/je01_inputs/calibration/<input_id>.bin` | Authoritative source mapping recovered; no heldout binary opened |
| Native input provenance metadata | `A/portable512/je01_inputs/JE01_NATIVE_384_INPUTS_RECEIPT.json` | Canonical pin recovered; this review did not open the 512 metadata file |
| Historical full256 native NLL | `R/source/docs/research/JE01_ORIGINAL_AF_REAL256_FULL_TARGET_GOLD_NLL_20261118.json` | Path recovered; pinned report read from PR94 |
| Authoritative existing input helper | `R/tools/je01_role_directional_train240_inputs.py` | 16,054 bytes recovered verbatim; SHA `99f4d6da38880e3eaadcd8a32e9f288a75ae87b7f13c9721fe00109616b7175b` |

Gold canonical: `355a169cfb7c9ab79901ae7848afa21a2f5a576d048a5476f527bca699136fb7`; gold file SHA: `b86e08bb54f65d0fa29dd545801d2fb3ef0bfb45bc1993743f21cd30cde7d7fd`. Preparation canonical: `f0a95e7ee7fbb1ce001c419c8626744e3730e85b235b9a726897ae38f946e477`; file SHA: `4dfb79c79bc564eaf1f7372c33b80fcaa3aba966ff280d0ac66ecb9ace565b50`. The frozen source pin is `a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147`; native-input canonical is `2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34`.

The historical full256 NLL canonical is `f71a4c4397cbadf938c7a9631ba8306547b31e33028dd9a54b556a1a758e3f41`; its per-chunk gold output hashes remain external OFF parity pins. Rows 16..255 have **10,368 gold tokens** and historical weighted baseline NLL approximately **2.471021398661067**. This is derived from the old report, not a new native run. [Baseline][baseline] [Independent original audit][baselineaudit]

Full-prefix native route traces for TRAIN240 must be freshly captured and sealed by the forthcoming native OFF preflight. The 256-prompt activation/router study is a last-prompt-token source and does not substitute for all teacher-forcing prefix routes. Existing TRAIN16 fixed-route SHA must never be applied to TRAIN240.

The helper was absent at the pinned PR path but already existed on EOE. It is now recovered locally at `tools/je01_role_directional_train240_inputs.py`; preserve those bytes. Its `validate_source_linkage(data)` checks original prompt, gold, baseline and chunk identity. Compose it with the existing `je01_role_directional_native_runner.validate_dataset(data)` to rehash actual prepared input/meta bytes; helper validation alone is not the complete binary preflight. Do not rerun its `prepare` CLI against existing data. Eight synthetic tests pass for the recovered helper plus this composition. No original input was changed.

The machine-readable [metadata review](JE01_TRAIN240_INPUT_METADATA_20261123.json) preserves all 15 dataset pins and the explicit verification limits.

## 4. Actual prefix counts and expected cost

Inputs are original BOS + prompt + all gold tokens except the last; there is no retokenization or truncation. TRAIN16 has 2,167 such input tokens. TRAIN240 has **23,432 BOS/prompt tokens + 10,368 gold − 240 = 33,560 input tokens**, a **15.4868481772×** prefix-length ratio.

| Chunk | Original rows | Gold | Native input tokens |
|---:|---|---:|---:|
| 01 | 16–31 | 634 | 2,226 |
| 02 | 32–47 | 702 | 2,193 |
| 03 | 48–63 | 664 | 2,252 |
| 04 | 64–79 | 784 | 2,432 |
| 05 | 80–95 | 791 | 2,489 |
| 06 | 96–111 | 772 | 2,403 |
| 07 | 112–127 | 589 | 1,822 |
| 08 | 128–143 | 646 | 2,142 |
| 09 | 144–159 | 742 | 2,558 |
| 10 | 160–175 | 698 | 2,184 |
| 11 | 176–191 | 660 | 2,154 |
| 12 | 192–207 | 700 | 2,247 |
| 13 | 208–223 | 776 | 2,553 |
| 14 | 224–239 | 617 | 1,998 |
| 15 | 240–255 | 593 | 1,907 |
| **Total** | **240 sentences** | **10,368** | **33,560** |

The measured current TRAIN16 native timings are approximately **153 s per JVP** and **71.45 s per OFF/FD** (run-owner review of the completed three-role receipts). Linear scaling by actual input length predicts **31.59 h for 720 JVP cells + 0.31 h for 15 OFF cells = 31.90 h**, excluding new builds, audits, queueing, contention, cold-start changes, and further diagnostics. This is a capacity estimate, not measured TRAIN240 performance or an SLA. The planned TRAIN16 gate is separately **48 JVP + 30 new FD + 1 OFF = 79 cells**, approximately 2 h 39 min.

The new TRAIN16 FD choices are fixed before results: `(0,Q5),(1,Q5),(2,Q5),(3,Q4),(13,Q5),(23,Q4)`, each alpha-zero and ±0.025/±0.05; prior three Q4 checks remain separately pinned. All48 alpha-zero primals/routes and token alignment still require audit; nine sampled numerical FD directions do not independently establish finite-difference agreement for every direction.

If the current two-pass native counter semantics are unchanged, TRAIN240 totals per direction are **67,120 processed tokens, 480 resets and 1,812,240 layer steps**. Derive expected counters per chunk from source lengths and the sealed execution mode before running; verify them, do not merely copy pilot constants. Resets are not independent sentence counts.

The dense float64 10,368×48 derivative matrix needs **3,981,312 B**; 240 per-case 48×48 outer-product sums need **4,423,680 B**. Raw repeated routes dominate evidence storage: the current 40-byte/26-layer/two-pass format predicts approximately **3.42 GB for 735 route files**, plus other outputs and logs. Retain original raw evidence; reserve capacity before admitting work.

## 5. Collection and independent accumulation

Use 15 immutable 16-case chunks, with a fresh OFF and 48 ordered JVP cells per chunk. Seal source/helper/build/binary/adapter/bank/dataset/role-order/plan hashes externally before the first cell. The original AF4 file remains SHA `9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4`. The current three-role native runner is TRAIN16-specific; a separately reviewed runner must parameterize chunk datasets, route references and source-derived counters without weakening its existing guards.

For every chunk, OFF must exactly match the historical native gold-output SHA. Capture its complete baseline routes, then require all alpha-zero JVP primals and route bytes to match that chunk's OFF. Align records by original case ID, local gold index, target ID and absolute position before combining columns. Reject duplicate/missing tokens, NaNs, source drift, incomplete columns, unsupported shapes or wrong counters. Count distinct routed calibration sentences for each frozen slot against the preregistered ≥32-observation eligibility rule; repeated tokens and the two execution passes do not create additional independent observations. A numerical FAIL is retained as FAIL.

Let `d[t,j] = ∂NLL_t/∂alpha_j`. Compute in float64:

- `g = sum_t d[t] / N`, because score `s = -d`.
- `F = sum_t outer(d[t], d[t]) / N`, retaining every cross-coordinate term.
- Per sentence/cluster retain `N_i`, `a_i = sum_t d[t]`, and `B_i = sum_t outer(d[t],d[t])`, plus raw aligned derivative records.
- Recompute independent totals from all 48 raw columns; check symmetry, diagonal second moments and `lambda_min(F) >= -1e-8 * max(1, trace(F))`. Do not replace F with the outer product of mean gradients, a diagonal-only DP, or a silently clipped matrix.

F is an observed-label empirical Fisher/Gauss–Newton surrogate, not the true Hessian and not a guarantee for the finite alpha=1 quantized policy.

Use sentence/source-row units, never individual tokens, for train folds and resampling. The recovered source provides row IDs but no document IDs. If document membership is available from TRAIN-only provenance, group shared documents; otherwise label the analysis sentence-cluster conditional and do not claim document independence. Freeze five folds by a deterministic hash/order of the chosen cluster IDs before derivative access. Fold diagnostics and 20,000 shared cluster-bootstrap draws (seed 20261123) describe stability; no hyperparameter retuning from TRAIN16 or the diagnostic folds. Bootstrap token-weighted ratios as `sum_i w_i a_i / sum_i w_i N_i` and similarly for B, rather than averaging sentence means.

Run serial cells under fresh resource admission. Resume granularity is one sealed cell; persist receipts, raw hashes, durable job IDs and completed-coordinate coverage. Submit one chunk per durable job when the host's supported duration permits, otherwise fixed bounded cell batches. Verify completed outputs before skipping them; do not rerun completed cells to improve outcomes. Stop at source/capacity/integrity failures and preserve the partial evidence.

## 6. One policy and native diagnostic before heldout

**Proposed deterministic search configuration to seal before TRAIN240 derivatives:** minimize exactly `gᵀx + 0.5 xᵀFx` with the prescribed counts and mutually exclusive tiers; curvature multiplier 1, no tuned regularizer. There are **2,498,640,144** feasible assignments, so ordinary independent-role budget DP is not valid for this full quadratic objective.

Use a deterministic heuristic: 64 feasible starts, NumPy PCG64 seed 20261123, including the lexicographically first assignment; best-improvement swaps and both three-tier cycles; maximum 512 accepted moves per start; stop improvement tolerance 1e-12 and lexicographic ties. Record all start seeds, iteration caps, convergence flags, objective values and the best feasible assignment. Recompute the winning objective independently from raw per-case sufficient statistics. Report **heuristic, global optimum unproven**; do not imply an exact solver certificate. Bootstrap reports uncertainty/stability of the fixed rule; it is not a second policy-selection objective.

Freeze one 24-role policy, source hashes, 12/6/6 counts, physical **43,253,760-byte** bundle SHA, native runner, comparison metrics, bootstrap seed and a fixed-seed same-count placebo before finite-policy native NLL or heldout access. Physical packing must be independently audited; a nominal byte sum is insufficient.

Evaluate the frozen policy and AF4 on the same already-seen TRAIN16 with full teacher-forced native NLL, then compare natural hard top-K routes over the complete prefix. Any fixed-route finite-step run is labelled separately. Preserve paired per-token and per-sentence deltas, p95 latency under matched hardware/load, kernel support and NaN checks. This is a development diagnostic and cannot authorize adoption. Predetermine the screen: HOLD on any integrity/kernel failure, route mismatch, >5% p95 regression, or positive candidate-minus-AF4 token-weighted NLL; do not substitute another candidate in the same sealed attempt.

Only after those gates may the one frozen candidate be evaluated on untouched validation128 and OOD128 under the original adoption contract: validation mean token-NLL gain ≥0.02 with paired 20,000-cluster-bootstrap lower95% >0; OOD lower95% gain ≥0; zero hard top-K mismatches; p95 latency regression ≤5%; placebo and multiplicity requirements; independent review. Failed development or heldout results retain AF4. No production promotion follows automatically.

## 7. Immediate handoff

1. Complete and independently audit the already-authorized full48 TRAIN16 diagnostic.
2. Include the recovered helper (exact SHA above) and its eight synthetic source-chain tests in source closure; preserve existing prepared datasets.
3. Review the new TRAIN240 runner schema, then execute read-only source/runtime preflight and fresh per-chunk OFF gates only under the sealed plan.
4. After the full48 gate and preflight pass, admit the approximately 32-hour, 735-cell TRAIN240 collection in durable resumable chunks. No TRAIN240 derivative was launched by this planning review.

Read-only Tailnet identity/current/live were `g6-0.9.0-operator.900520` and converged. The XOX source path was not accessible through the read-file workspace; no alternative execution was used to bypass that denial. EOE sources supplied the needed provenance. Both task-owned filename searches are terminal; no remote job, process, release change, or heldout access was created here.

[nextgate]: https://github.com/popixoxipop-collab/beglin/blob/a0bac6efaa11182177e88c3f6561fabbebc2f758/docs/research/JE01_ACTUAL27_ABC_NLL_BAYESIAN_FISHER_NEXT_GATE_20261123.json
[prereg]: https://github.com/popixoxipop-collab/beglin/blob/a0bac6efaa11182177e88c3f6561fabbebc2f758/docs/research/JE01_MOE_BAYESIAN_EXPERT_BIT_ALLOCATION_PREREG_20261117.md
[caldecision]: https://github.com/popixoxipop-collab/beglin/blob/a0bac6efaa11182177e88c3f6561fabbebc2f758/docs/research/JE01_CAL240_REAL_FFN_ACTIVATION_DECISION_AND_NEXT_NLL_GATE_20261122.md
[context]: https://github.com/popixoxipop-collab/beglin/blob/a0bac6efaa11182177e88c3f6561fabbebc2f758/docs/research/JE01_FOLLOWUP_CONTEXT_20261123.md
[bank]: https://github.com/popixoxipop-collab/beglin/blob/a0bac6efaa11182177e88c3f6561fabbebc2f758/tools/je01_role_directional_bank.py
[baseline]: https://github.com/popixoxipop-collab/beglin/blob/a0bac6efaa11182177e88c3f6561fabbebc2f758/docs/research/JE01_ORIGINAL_AF_REAL256_FULL_TARGET_GOLD_NLL_20261118.json
[baselineaudit]: https://github.com/popixoxipop-collab/beglin/blob/a0bac6efaa11182177e88c3f6561fabbebc2f758/docs/research/JE01_ORIGINAL_AF_REAL256_FULL_TARGET_INDEPENDENT_AUDIT_20261118.json

