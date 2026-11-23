# JE01 all48: native output-path contract recovery

**2026-11-23 KST. This is an execution-contract correction, not a numerical retry.**

The first all48 plan used research root `je01_native_all48_train16_20261123_v1`. The unchanged, SHA-pinned native executable's C export gate requires the derivative output path to begin with `/Volumes/D50/TailnetCommander/work/je01_native_role_jvp_`. The planned JVP path did not satisfy that gate. The Python plan/schema checks and initial independent preflight did not test this native string constraint. A subsequent source review identified it before the first JVP.

The old native guard is retained. No alias, symlink, installed-code change, counter relaxation, or new numerical threshold is used. The corrected run uses a fresh root named `je01_native_role_jvp_all48_train16_20261123_v2`, with new runner/auditor/controller versions and a separately sealed plan. The original v1 code, plan, logs, raw outputs, and failed attempt remain intact.

## Actual stopped execution

| Item | Observed state |
|---|---|
| EOE job | `job_a1db47a91507e972ce817b96457ebebf` |
| Start | 2026-11-23 09:48:42 KST |
| Stop | 2026-11-23 09:51:03 KST |
| Terminal state | CANCELLED, exit 1, requested stop; 141.045 seconds |
| Completed native observations | OFF control only, 72.3879746674 seconds |
| Interrupted attempt | First Q5 fixed-FD alpha-zero cell; 58.6963315420 seconds, native exit -15 |
| JVP columns executed | 0 |
| FD convergence/comparison verdicts | 0 |
| Task scratch cleanup | Removed, confirmed by job receipt |

The completed OFF control exactly reproduces original gold SHA `b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c` and mean NLL `2.291799702354307`. It is retained as an inherited observation; the corrected workflow does not recompute that successful control. The interrupted FD attempt has a sealed execution-failure receipt and supplies no accepted numerical comparison.

## Preserved evidence

| Evidence | Canonical receipt | Full-file SHA-256 |
|---|---|---|
| v1 sealed plan | `60fdced19fad43cbf2b73801dd68effbbd75c186f8a5f39de9274cad39453419` | `dc4cf87f557ef806d44c02cd9bf66a60dbec893303d74aa89cbf9ef0c5a3454f` |
| Completed v1 OFF | `f044aa2e2ee1a22ab305f1854d2cf6c7280a5d72cded594859d786fcc5a0caa9` | `98d68ca9a5984502bd3858bbb0523653814589abf1114ec5ad28eff39e6c6f6d` |
| Interrupted FD | `51b3795007b69b783d94e22eae5f62fc96d8c73f8b377d3a1d18e5ee7e561129` | `40daee88c55a53c478a17c34b584a7736c54b76fd06142cdd0656bcdc5e74579` |
| Stopped workflow | `694f066501ddf1f1e31d94a746c56fc10239081e5c072311118c2c89c6e13b53` | `1f32a3461859f8c90f485fd8e6fff95553cb872b9723bd59ba3c988f5f63cf0e` |
| Independent stopped-state audit | `b21f0e037e27ce30c47bb5ec4ef8b17a44d09c3b38f18389e33ae2fb62df6d3f` | `66fd0128598c2705ea735ea0876006b8effa2602aa25f455fec9e4e4e90a6e35` |

The unchanged v1 auditor reopened the stopped experiment and **153 files** in separate CPU job `job_0323ffdbbb9a2366d53afe3097f9ae4c`. It confirmed integrity PASS, one completed OFF observation, zero JVP columns, zero FD verdicts, and `INDEPENDENT_AUDIT_CONFIRMED_PARTIAL_NO_PASS`. Its expected exit 2 correctly yields a scheduler FAILED label, with numerical state NOT_COMPLETE; this is not an adverse model-quality result. It performed no model inference. The task scratch was removed.

## Corrected execution contract

The complete experiment still requires 79 native observations: **one inherited, independently checked OFF control plus 78 new native cells**. The new cells are the same 48 alpha-zero JVP columns and the same 30 FD cells at the same six coordinates. The three prior Q4 FD references remain unchanged. All nine FD/JVP comparison gates, alpha values, row/column order, model/bank/binary/runtime hashes, gold/route checks, full-prefix counters, and scientific limitations are retained.

The v2 plan explicitly separates its 79-entry evidence sequence from its 78-entry new native sequence. OFF provenance identifies the original v1 plan, runner, native binary, receipt, raw bytes, and independent audit. Its copied raw bytes remain byte-identical, and its bridge receipt states that no native execution occurred for OFF in v2. The new preflight reads the actual SHA-pinned native C export guard and validates every prospective JVP output path before any native cell.

The original AF4 policy is retained. PR #94 stays Draft. The correction does not authorize TRAIN240, allocation adoption, or heldout/OOD access before their existing gates.

## Actual v2 pre-execution plan

The corrected plan was created on EOE in job `job_0e87e4c5a1b2eca9dca6f158bcff7a13`, which completed successfully, exit 0, in 10.410 seconds at 2026-11-23 10:02:01 KST. Its task scratch was removed. No native inference ran in that job.

- Runner source SHA-256: `ad07f04937a79868a67f8eacedee22b6fe091399cf0557e5d2575234ebc50df6` (66,745 bytes).
- Plan canonical receipt: `260c199a18f1b60bf94a6388fd9df306803a950383a199bd2665b88a6d553aba`.
- Exact plan file SHA-256: `6cfce61f1c2af96bda87d1ffbfd2b5e3fd4d2138d51425a3179f9903ff7e1852` (51,203 bytes).
- Native path contract: actual C source SHA `5e2a6ea76c22c1ec0ff9f2fb6d093b7d3a471679adbe439ed1c4b898a56a52a6`, required absolute prefix `/Volumes/D50/TailnetCommander/work/je01_native_role_jvp_`.
- Evidence cells / newly executed cells / inherited native cells: **79 / 78 / 1**.
- Estimated new native work: **9,581.2704 seconds, about 2 hours 40 minutes**, extrapolated from earlier native observations; this is not a measured full48 completion time.

The v2 runner's 12 targeted contract tests pass, including rejection of the v1 output root by the pinned native C guard, inherited OFF provenance, exact role/tier order, numerical gate failures, raw metadata, receipt immutability, and resume rejection after source or evidence drift. The v2 controller's four focused tests pass for process exit handling, signal cleanup, root/whole-workflow lock, and append-only receipt pins. An independent agent reviewed the controller completion and HOLD conditions. These source-level checks do not establish a new native or numerical PASS.

## Independent v2 preflight

The standalone v2 auditor `a3cf5a2dc8048f51208f8a44ce3038eb51b9f44147ba4f62bb8a65aac0524d23` (93,410 bytes) was then run in separate EOE CPU job `job_9c5233ed3fcd688556e205be2e54a743`. It reopened **159 files**, including the actual native source/output-path guard, baseline raw evidence and the v1 OFF recovery chain. Evidence integrity is **PASS**. There were no v2 cells yet, so its numerical verdict is **NOT_COMPLETE** and status is `INDEPENDENT_AUDIT_CONFIRMED_PARTIAL_NO_PASS`. Its expected exit 2 is represented as FAILED by the job scheduler; no inference ran. Duration was 5.334 seconds and task scratch removal was confirmed.

- Preflight canonical receipt: `3865127e0ab6f8d88fc0addb655d8534feff3c863508e0a1468c76043d3e2997`.
- Preflight exact file SHA: `eecf7af66633e39faed31b995d1033b63e2ac20a74659453c209f0469f76990c` (93,492 bytes).
- Controller source SHA: `7ba9eac5c3dfd2b1e61b26e66dd1ef6368426d35515d890131a18ae1b1a4006b` (11,652 bytes).
- V2 independent auditor contract tests: **15 PASS**, including the actual sealed v2 plan schema and negative inherited-execution provenance controls.

The controller requests exactly 78 new native cells, then starts the auditor as a separate process only after the runner has sealed a full result or a numerical HOLD. Completion requires both receipts to match the external plan/source pins and the same 79 total / 78 new / 1 inherited counts. TRAIN240 remains a separately gated future stage.
