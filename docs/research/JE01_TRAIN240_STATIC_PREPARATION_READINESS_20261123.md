# JE01 TRAIN240: counter-specific source preparation readiness

**2026-11-23 KST. Preparation code and contract tests are complete. Actual input revalidation, manifest sealing, source staging, compilation and model inference have not run.**

The proven TRAIN16 native source contains exact completion checks for 4,334 processed tokens, 32 case resets, 117,018 layer steps and 656 gold scores. Those checks are preserved in the running full48 experiment. TRAIN240 consists of fifteen fixed 16-case chunks with different lengths; the next preparation stage creates separate source variants with exact source-derived completion counts. It does not remove a native guard or change the JVP mathematics.

## Frozen source and test evidence

| File | SHA-256 / result |
|---|---|
| `tools/je01_native_train240_static_prepare.py` | `99c5cb1909136560ae345abc8edd5d652e986f6eaff5b4cc27715aed4082cfd5`, 31,255 B |
| `tools/je01_train240_static_counter_independent_audit.py` | `62170c0191fe06cfa876e73144fa73996a6ef3b007af4cee2ee07d3c56e4d0b3`, 50,224 B |
| `tests/test_je01_native_train240_static_prepare.py` | 8 targeted tests PASS |
| `tests/test_je01_train240_static_counter_independent_audit.py` | 12 targeted tests PASS |
| Unsealed static specification JSON | `4c8a085e5ab6a6b9024b0c467143c76f98a7cc6e2668cee7f1f5ce04ad320769`, 13,342 B |

The tests include all fifteen exact C++ finish fragments compiled in the local test environment, rejection of incorrect counts and changed non-counter source, partial/HOLD rejection before TRAIN240 input access, raw-input reconstruction on synthetic fixtures, original/global/local row mapping, and byte-budget/source-tree checks. They do not constitute fifteen actual native model builds, validation of the remote TRAIN240 raw files, or a completed preparation audit.

## Completion counters

For chunk input-token sum `S` and gold count `N`, the exact constants are `tokens=2*S`, `cases=32`, `layers=54*S`, `gold=N`. The two passes include warmup and measurement; they do not double the number of independent sentences. The model has 27 layers, but the route dump covers 26 routed layers, so `route_records=52*S`.

Across the fifteen chunks the expected source totals are 240 sentences, 10,368 gold scores, 33,560 native input tokens, 67,120 processed tokens, 480 case resets, 1,812,240 layer steps and 1,745,120 route records. Each of the 48 directional runs must use the same complete target set; these counters are execution coverage checks and not a statistical denominator for g or Gram.

The transformation replaces three integer constants within exactly two full source anchors. Inverting those replacements must recover the exact parent source hashes:

- C++: `6055ca3b261075fe5fcf76e5829f0fb20b8ecdd52599248f4b9b1032cfbeb49f`.
- C: `5e2a6ea76c22c1ec0ff9f2fb6d093b7d3a471679adbe439ed1c4b898a56a52a6`.

Every other C/C++ byte, C_OPEN output-prefix guard, 16-case parser, native math, header, CMake file and source-tree entry must remain identical. The independently reconstructed expected tree is checked for each variant. The preparation tool calculates the actual byte cost of all fifteen copies and requires the remaining cost plus 1 GiB free space. Existing build or run artifacts prevent a source-only receipt from being issued.

## Gate and intended remote root

The only planned destination is `/Volumes/D50/TailnetCommander/work/je01_native_role_jvp_train240_static_20261123_v1`. It has not been created or populated by this preparation work. The root deliberately satisfies the unchanged native C_OPEN prefix contract.

Manifest creation, source staging and independent audit first require the actual completed full48 V2 independent audit, externally pinned by both canonical receipt and file SHA. It must prove the known plan, runner and auditor, 79 evidence cells / 78 new cells / 1 inherited OFF, 48 directions, all 9 checkpoints, original-model hash and raw matrix/analysis linkage. A three-role PASS, partial result, numerical HOLD or self-sealed changed JSON is insufficient.

After that gate, the tools verify the original TRAIN256 gold/preparation receipts, all fifteen datasets, original and prepared raw prompt files for rows 16..255, complete teacher-forcing construction, local/global row identities and exact metadata padding. The independent auditor imports no experiment helper and rejects reuse of TRAIN16 source identities under different global rows.

## Exact four-file closure and execution sequence

Stage these verified source files under the new root's `tools/` only after the full48 PASS and fresh resource admission:

1. `je01_native_train240_static_prepare.py`, SHA `99c5cb1909136560ae345abc8edd5d652e986f6eaff5b4cc27715aed4082cfd5`.
2. Recovered original `je01_role_directional_train240_inputs.py`, SHA `99f4d6da38880e3eaadcd8a32e9f288a75ae87b7f13c9721fe00109616b7175b`.
3. Existing `je01_role_directional_native_runner.py`, SHA `5607c0afef3bd4b1a61bcab6622267acc6b6137fc65ba0e79873e92e31d7e9e6`.
4. Standalone `je01_train240_static_counter_independent_audit.py`, SHA `62170c0191fe06cfa876e73144fa73996a6ef3b007af4cee2ee07d3c56e4d0b3`.

Use the existing EOE Python `/opt/homebrew/opt/python@3.11/bin/python3.11`. The prepare CLI has only `spec`, `manifest` and `stage`; it cannot compile or launch a model. `spec` is an unsealed, input-free description. For `manifest`, supply `--root`, `--expected-all48-audit-receipt` and `--expected-all48-audit-file-sha` using actual retrieved values. Capture the resulting canonical/file pins. Inspect its measured source-copy budget, obtain a corresponding storage reservation, and invoke `stage` with the same arguments plus `--expected-manifest-receipt`.

Then run the standalone auditor with `--root`, `--expected-all48-audit-receipt`, `--expected-all48-audit-file-sha`, `--expected-manifest-receipt`, `--expected-stage-receipt`, `--expected-prepare-sha` and a new `--output ROOT/TRAIN240_STATIC_INDEPENDENT_AUDIT.json`. The only successful meaning is `INDEPENDENT_STATIC_SOURCE_PREPARATION_PASS_NOT_COMPILED`. No actual TRAIN240 derivative, finite allocation or heldout score follows automatically.

Separate later work needs fifteen native build receipts, chunk-specific OFF gold reproduction and fresh fixed routes, a collection runner/auditor, and a sealed training selection/evaluation protocol. `JE01_TRAIN240_SELECTION_AND_HELDOUT_ADDENDUM_DRAFT_20261123.md` records proposed statistical clarifications and unresolved mappings to older preregistration gates; it remains a draft and does not waive them.
