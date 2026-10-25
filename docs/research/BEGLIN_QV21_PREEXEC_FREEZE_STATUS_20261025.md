# BEGLIN Qv21 — preexecution data freeze and managed-run gate (2026-10-25 KST)

The Qv21 tail-risk-aware prefix selector is **research only**. This record reports
real EOE file-freeze actions, not any Qv21 original model score.

## Verified preexecution artifacts

- Research branch: `research/qv21-tailrisk-prefix-selector-preregister-20261025`.
- Immutable machine-readable protocol `qv21_spec.json` SHA-256:
  `408c661e66259784480b18eefc15f75d3bc7d034124194baaf3a218637e2f2d0`.
- Executed EOE freeze script `tools/qv21_freeze_preexec.py` SHA-256:
  `b048e640a9b2d59972465d3a7c41934e0aa93bc89a4b9ebeddc621a11507f793`.
- Actual EOE D50 preexec public receipt SHA-256:
  `eaf4633703f7ebc4139cbe6c970e36076a51693afa24141054d39d9fcaa93e40`.
- Frozen **calibration**: ARC-Challenge train96, private manifest SHA
  `982455c845d4c46ff66811e3ddd38fa70e9ade1cbcbba5880919943c8656fe2f`.
- Independent **study-level evaluation**: ARC-Challenge train48, private SHA
  `c17fa191f40c4dc2facc4c8f2546cc68949593d29bdfbd82144c1cb35e07bb58`;
  OpenBookQA train48, private SHA
  `5f8339b82edddceaae63f2f78a53cc0f64416ad4cd52b151a2bd644d800af78c`.
- Read-only independent audit **PASS**: 192/192 selected rows unique, prior Qv20 ARC train32 excluded,
  frozen per-case input/choice binaries for **576 selected case+arm bundles** verified.
  Neither Qv20 validation24 split was opened by the Qv21 freeze.
- Original C reference scorer `tools/qv21_native_static.py` uploaded to GitHub and copied to
  an isolated EOE sandbox, exact SHA-256 `886615757c80ed6f5ffdce6866ba5c1aede3695b6a95f16f9ea04244b3830daa`.
  Two read-only calibration96 preflight calls `static_source` and `static_random` both exit **0**,
  including original native C SHA `f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221`.

## Execution boundary

The legitimate EOE scheduler reported Qv21 Qv8-cal96 capacity as **eligible** on
2026-10-24T21:09:27Z: about 2.12 available CPU cores, 21.39GB RAM and D50 `d50_work`.
**The actual managed resource_submit call was blocked by an OpenAI tool safety check.**
No retry by another route, direct unmanaged native execution, invented score or success claim.
A subsequent fresh `job_list` found **zero Qv21 jobs**, and the Qv21 native result directory was absent.

Therefore `QV21_PREEXEC_FROZEN_PASS` and `NATIVE_CALIBRATION_NOT_RUN` are the only valid
research states. Qv21 policy training, evaluation96, dynamic quality claim and release
promotion are all **NOT_RUN**. Existing G6 operating release was `g6-0.9.0-operator.900387`
without a change in these steps. Private per-case tokens, labels and losses stay on EOE D50.

**Next authorized step:** obtain a permitted managed execution and original C reference
receipts for both calibration96 arms; independently audit both arms before any policy
training or new evaluation input scoring. Do not open Qv20 validation48.

## 2026-10-25 20:35 KST — current G6 resource emergency: fail-closed

This is **fresh read-only runtime/scheduler evidence**, not a Qv21 model benchmark or an instruction to bypass admission.

- Live EOE G6 release: `g6-0.9.0-operator.900405` from independently fresh `release_status` and `runtime_identity`; current=live=runtime, converged=true, active_operation=null. It supersedes the historical `900387` state above. **This Qv21 investigation did not activate any release.**
- Fresh managed EOE `resource_plan` observed at `2026-10-25T12:05:20.770Z` for the exact one-core / 6GB / D50-work ARC train96 Qv8 native C job returned **recommended_host=null** and EOE ineligible with `offline`, `insufficient_free_ram`, `insufficient_free_cpu`, `no_eligible_storage_pool`. This is scheduler admission, not proof that source data was lost.
- Several ordinary EOE host-bound **read-only** `exec` calls returned `FLEET_ADMISSION_BLOCKED`, which explains why even fresh SHA/preflight checks could not complete. Their diagnostic measurement reported **0 allocatable CPU cores; 0 free job slots; macOS swap 31.97/33GiB (~96.9%); five-minute load ~24.3 on a 10-core host; 124 measured managed processes; ten configured GitHub runners with substantial queue pressure**. The exact GitHub queue counts varied across snapshots and are not asserted as one stable count.
- Fresh first-page EOE job list reported **no Qv21-labeled jobs among 30 recent entries**. It showed an unrelated `OUTCOME_UNKNOWN` job (`b12-native-forward-strict68-20261025-v1`); this was **not touched**.
- **No new `resource_submit` was attempted during this emergency**, no ad hoc local original C model inference, no process stopping/deletion, no weakening of fail-closed admission, no use of Qv20 heldout or newly sealed Qv21 evaluation96. The previously frozen Qv21 public manifest remains `eaf4633703f7ebc4139cbe6c970e36076a51693afa24141054d39d9fcaa93e40` based on historical successful check; its present readback was **blocked by admission**.
- Qv21 preregistered policy fitter, independent 96-case native calibration gate and feature code are already committed in this same GitHub branch: `tools/qv21_policy.py`, `tools/qv21_calibration_gate.py`, `tools/qv21_fit.py`. GitHub Actions **BEGLIN Qv21 Offline Tail-Risk Contracts** run [40156619822](https://github.com/popixoxipop-collab/beglin/actions/runs/40156619822) completed **SUCCESS** (18 synthetic-only checks). These results are code-contract tests, **not native calibration receipts or trained dynamic performance**.

**Strict resume gate:** first restore actual EOE scheduler eligibility through authorized resource management without terminating unrelated processes or altering experimental frozen inputs. Then independently read and confirm the complete Qv21 preexec freeze/mask/native-source SHA and absence of existing original-C output/jobs. Only with a fresh eligible `resource_plan`, submit exactly one registered original C Qv8 ARC train96 job with a distinct idempotency key via the normal managed route, and check its durable terminal receipt before submitting the random73 arm. After both succeed, run the already committed independent per-case native receipt audit; fit/freeze the preregistered policy only if the audit returns PASS. Do not open the 48+48 Qv21 evaluations or the 48 Qv20 sealed validation examples before policy freeze.

**Current scientific verdict:** `QV21_PREEXEC_FROZEN_PASS / OFFLINE_CODE_GUARDS_PASS / EOE_ADMISSION_BLOCKED / QV21_NATIVE_CALIBRATION_NOT_VERIFIED / POLICY_NOT_FIT / EVALUATIONS_UNTOUCHED / NO_PRODUCTION_PROMOTION`.
