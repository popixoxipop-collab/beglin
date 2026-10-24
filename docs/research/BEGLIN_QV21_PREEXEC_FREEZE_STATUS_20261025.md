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
