# JE01 full48 TRAIN16 v2: active run and completion protocol

**2026-11-23 KST. The run is active; this document is not a numerical PASS.**

The authorized follow-up is 24 fixed Expert roles × Q4/Q5 on the already observed TRAIN16, followed by TRAIN240 preparation. The original AF4 policy remains selected. Do not access validation128 or OOD128, merge PR #94, alter production weights/services, or activate a Tailnet release.

## Exact active execution

| Field | Value |
|---|---|
| Host | EOE |
| Durable native + independent audit job | `job_cd608db3eb69d616b64e20cca1a9f050` |
| Reservation | `res_b2ac87e0085e4d53b7a0f93665d4bfe1` |
| Idempotency key | `je01-all48-train16-v2-native78-and-independent-audit-260c199a-20261123` |
| Started | 2026-11-23 10:07:10 KST |
| Root | `/Volumes/D50/TailnetCommander/work/je01_native_role_jvp_all48_train16_20261123_v2` |
| Execution timeout | 12,600 seconds; each native cell has a separate 900-second timeout |
| Demand reservation | 4 CPU cores, 24 GB RAM, 0.95 Metal fraction, 1 GB D50 work storage |
| Pre-run Git checkpoint | `423896d94548264aa81cef851cbb64e85eaec3a6` |
| Release at submission | current = live `g6-0.9.0-operator.900520`, converged |

The complete evidence contract is **79 observations = 1 inherited OFF + 78 new native cells**, comprising 48 JVP columns and 30 new FD observations. The six new FD coordinates are `(0,Q5), (1,Q5), (2,Q5), (3,Q4), (13,Q5), (23,Q4)`. Prior FD references for Q4 roles 0,1,2 are preserved, and the new JVP bytes must exactly reproduce their earlier raw outputs. Nine coordinates are FD checked; the other 39 are not individually FD checked.

## Immutable pins

| Artifact | SHA-256 |
|---|---|
| v2 plan canonical | `260c199a18f1b60bf94a6388fd9df306803a950383a199bd2665b88a6d553aba` |
| v2 plan file, 51,203 B | `6cfce61f1c2af96bda87d1ffbfd2b5e3fd4d2138d51425a3179f9903ff7e1852` |
| v2 native runner source | `ad07f04937a79868a67f8eacedee22b6fe091399cf0557e5d2575234ebc50df6` |
| v2 independent auditor source | `a3cf5a2dc8048f51208f8a44ce3038eb51b9f44147ba4f62bb8a65aac0524d23` |
| v2 workflow controller source | `7ba9eac5c3dfd2b1e61b26e66dd1ef6368426d35515d890131a18ae1b1a4006b` |
| Native build canonical | `2cbeb4dc1a7e1ab5642a4abb22395498c10dde43bee051eec3c303c7e53fa008` |
| Unchanged native binary | `110b1978d97bcd7c6bdfd6d42142569e56e9dacffa4ff462dc0786b3128e49f4` |
| Original model, 9,814,343,680 B | `9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4` |
| Zero-cell independent preflight canonical | `3865127e0ab6f8d88fc0addb655d8534feff3c863508e0a1468c76043d3e2997` |
| Zero-cell independent preflight file | `eecf7af66633e39faed31b995d1033b63e2ac20a74659453c209f0469f76990c` |
| V2 inherited-OFF bridge canonical | `cc987936697f392b053609e19b259297277e6e5b5c1f6cd8716c0b2cb6e18d01` |
| Original v1 OFF canonical | `f044aa2e2ee1a22ab305f1854d2cf6c7280a5d72cded594859d786fcc5a0caa9` |

The v2 preflight independently reopened 159 files, including the actual native output-path guard and the v1 recovery chain. It reported integrity PASS, numerical NOT_COMPLETE, 0 new v2 cells, expected exit 2. The preceding v1 run was cancelled before any JVP because its root violated the unchanged native output-path contract; preserve its failure history as described in `JE01_ALL48_EXECUTION_CONTRACT_RECOVERY_20261123.md`.

## First live Q5 checkpoint — runner verdict, independent raw audit pending

At the 2026-11-23 10:15:59 KST observation, five FD cells and the first JVP cell had completed. The inherited OFF brings the evidence count to 7/79; new native execution count is 6/78 and JVP columns 1/48. The runner sealed a passing `(role 0, Q5)` checkpoint and began `(role 1, Q5)` FD observations. The first JVP took **153.521061 seconds**.

| First Q5 Gate comparison | RMS error | Acceptance limit | Error / limit |
|---|---:|---:|---:|
| FD h=.05 versus h=.025 | 0.00004934233615420578 | 0.000559211472601608 | 0.0882355577 |
| JVP versus FD h=.025 | 0.000044198944795986535 | 0.000559211472601608 | 0.0790379793 |

Checkpoint canonical: `b5cfefbc499028df297b6668b86b2dd0e736074a31bd6c845a59cb57382c0be1`; exact file SHA `6d6734ec6a5e91d948289837fbc729a6321e572a130b34a054076c78fde6b9bd`, 43,703 bytes. The downloaded checkpoint's checksum and FD comparison were independently recomputed from its derived arrays; the full independent native raw audit has **not** run yet. This is a live runner checkpoint and not a full48 independent PASS or a policy-quality result.

## Follow-up observation: 10:38 KST

Read-only continuation confirmed the same durable job still RUNNING with a fresh heartbeat at **2026-11-23 10:38:25 KST**. The complete stdout page recorded **21/79 evidence observations = 1 inherited OFF + 20 new native cells**, **5/48 JVP columns**, and **5 provisional runner checkpoints**. Gate/Up/Down Q5 and newly reproduced Gate/Up Q4 checkpoints had passed; Down Q4 was the current cell. These are observed runner results, not the final independent raw audit. The immutable native plan and sources remain unchanged.

The full Up and Down Q5 checkpoint bytes, their checksums, and the bounded progress observations are preserved in `JE01_ALL48_TRAIN16_V2_UP_Q5_CHECKPOINT_20261123.json`, `JE01_ALL48_TRAIN16_V2_DOWN_Q5_CHECKPOINT_20261123.json`, and `JE01_ALL48_TRAIN16_V2_Q5_PROGRESS_OBSERVATION_20261123.json`. The earlier Gate Q5 checkpoint is preserved separately. The accompanying `JE01_ALL48_TRAIN16_V2_Q5_PROGRESS_REVIEW_20261123.md` describes the exact verification scope.

## Completion procedure

1. Read fresh Tailnet current/live release and the exact job status/output. Never start a duplicate job or re-execute successful cells. While this job is RUNNING, its own controller already owns the run lock, native execution and final independent audit. A status query must not race a second auditor or alter its source/plan.
2. On terminal completion, read `WORKFLOW_RECEIPT_0001.json`. Retrieve exact bytes through authorized file tools and validate both full-file SHA and canonical receipt. Confirm exit code, termination reason, and task scratch cleanup from `job_status`.
3. Read `JE01_ALL48_TRAIN16_INDEPENDENT_AUDIT.json` and, for a full result, `JE01_ALL48_TRAIN16_ANALYSIS.json` and `RAW_DERIVATIVE_MATRIX.json`. Preserve the returned exact file SHAs and canonical receipts; do not infer them. The independent auditor is a separate process, even though it runs within the same admitted durable job.
4. A completed PASS requires workflow `NATIVE_ALL48_AND_INDEPENDENT_AUDIT_COMPLETED`, runner and auditor exit 0, audit `INDEPENDENT_AUDIT_CONFIRMED_ALL48_TRAIN16_RAW_AND_NINE_CHECKPOINTS_PASS`, evidence integrity PASS, numerical PASS, 79 total/evidence cells, 78 newly executed cells, 1 inherited OFF, 48 JVP columns, 9 FD coordinates, and the matching analysis receipt and external plan/source pins. The 656×48 matrix must match the analysis/audit file pins. Neither a scheduler success label nor CI tests alone establishes this result.
5. If a checkpoint fails, the runner stops subsequent cells and seals `NUMERICAL_HOLD_RECEIPT.json`. A legitimate HOLD has runner/auditor exit 2 and independent audit `INDEPENDENT_AUDIT_CONFIRMED_NUMERICAL_HOLD`. Publish the actual failed gate; do not change h, thresholds, subset, role order or original numerical evidence, and do not publish g/Gram as a completed full48 result. An execution/source/integrity failure is a different state and must be recorded separately.
6. On a verified result, append the complete evidence and concise report to the existing JE01 branch, preserving all prior files. Read the fresh PR head/tree, build on it, and use `expected_sha` CAS with `force=false`. Keep PR #94 Draft and unmerged. Do not write messages/comments to other people.
7. Update the existing private viewer using its audited exporter and external pins. Do a `--check-only` pass first; then export, preserve prior ordinary/fixed/one-role/three-role results and v1 recovery history, push exact Site source, publish the exact source archive, confirm deployment status, and verify the UI through the available browser surface. Report authentication boundaries honestly.
8. TRAIN240 preparation may consume the independently verified full48 PASS. The preparation tools must verify the 15 original datasets, all original prompts and prepared teacher-forced inputs for 240 cases, manifests, exact gold metadata/padding and original gold source before sealing a manifest. They must preserve the JVP mathematics, and change only the three exact completion constants in two source anchors for each chunk; inverse transformation must recover the exact parent C/C++ SHAs. Do not reuse the TRAIN16 binary's 4334/117018/656 completion counters on TRAIN240. This completion check performs neither native compilation nor model inference. Manifest `compile_commands` are a separate future build plan and must not be executed here.

If a terminal job lacks the expected workflow or final-audit receipt, or any integrity check fails, record an execution/integrity HOLD and preserve the raw files. Do not claim completion, publish a full matrix, or automatically rerun native observations. Report the concrete failure and end completion monitoring instead of indefinitely repeating retrieval.

The target remains repository `popixoxipop-collab/beglin`, PR **94**, branch `research/je01-moe-bayesian-bit-allocation-20261117`. The final native and independent auditor processes share the same durable job: use **`job_cd608db3eb69d616b64e20cca1a9f050` for both exporter `--native-job-id` and `--audit-job-id`**. Do not use the preflight job as the final audit job.

## Bounded completion watch and duplicate prevention

A condition watch was created for this exact active job: `Automation_9f00a98bb6188191b13989ad4c5df77d`, enabled, first check **2026-11-23 11:15 KST**, hourly for at most six checks (last scheduled check 16:15 KST). Normal RUNNING observations produce no repeated user notification. The watch retrieves terminal evidence, publishes the verified outcome and, only after full independent PASS, performs the source-only TRAIN240 preparation described in `JE01_TRAIN240_STATIC_PREPARATION_READINESS_20261123.md`. It then disables only itself. It also disables itself after reporting a concrete terminal failure requiring recovery. The existing older JE01 automation remains unchanged.

Before final publication, look for `docs/research/JE01_ALL48_TRAIN16_V2_COMPLETION_RECORD_20261123.json` on the latest research branch. Record the binding **final audit canonical → research evidence commit → Site source commit → deployment ID → browser verification scope → TRAIN240 source-preparation audit/result**. If the same audit already has a complete record, stop without re-export, re-commit or re-deployment. For a partial completion record, reconcile and reuse exact existing receipts, commits, deployments and jobs, then perform only the remaining steps. Environment selection for the watch is inherited; the creation response did not enumerate attached environments, so future access must be verified at execution time.

## Viewer and tested scope

- Existing owner-only Site: `appgprj_6b03682ea5f88191b76bc263e09c1c2c`.
- HTTPS: https://je01-native-sensitivity-20261123-abdb.sweet-eft-6723.chatgpt.site/
- Site source checkpoint: `5c52849b1edd4491e83d6190d0c0e1d28d439652`.
- Successful running-state deployment: `appgdep_6b0398c08de88191aa6a4cf7707959ea`.
- Local checkout when present: `/workspace/scratch/f05b6fa53004/je01_viewer/.sites-checkout`.
- Source entrypoints: `export_all48_jvp.py`, `record_all48_followup.py`; full commands are in Site `README.md`.

The published progress snapshot is explicitly timestamped **2026-11-23 10:34:38 KST**: 19/79 evidence observations (one inherited OFF plus 18 new native cells), 3/48 JVP columns and 3/9 provisional runner FD comparisons. It is not a live backend feed. The local browser verified these counts, the explanatory Gate/Up/Down Q5 note, the still-hidden full48 matrix/Gram and the independent-audit table at 0/9 registered results. All prior experiment objects and six original source JSON files were confirmed unchanged from Site source `2230a3e6342ed6624056b9101ec4b3dcd85c4a1a`. Production deployment succeeded at 2026-11-23 10:37:15 KST. The HTTPS page reached the expected ChatGPT sign-in screen in the available browser; authenticated production content and physical iPad Safari were not verified. The temporary local preview was stopped and the Site checkout is clean.

V2 source checks comprise 12 runner, 15 independent-auditor and 4 workflow tests. All six existing PR workflows at `423896d94548264aa81cef851cbb64e85eaec3a6` succeeded, including independent shadow-contract run `45092664513`. These are contract/build checks, not full48 scientific evidence. The native full48 duration estimate is about 2 hours 40 minutes, not measured end-to-end runtime.

Interpretation remains fixed-branch observed-label directional sensitivity. The uncentered Gram is not parameter Fisher, expected Fisher or Hessian, and full48 numerical PASS would not establish finite bit-allocation NLL improvement, natural routing stability, latency performance or independent generalization.
