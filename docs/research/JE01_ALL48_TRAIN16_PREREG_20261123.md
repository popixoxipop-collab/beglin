# JE01: 48 native directions on previously seen TRAIN16

**2026-11-23 KST — preregistered execution, not a result.** This continues the independently audited three-role Q4 study on Draft PR #94. The original AF4 model remains the policy. No TRAIN240 fitting, allocation selection, validation128, OOD128, serving change, or release activation is part of this execution.

## Frozen scope

The 24 roles are Gate, Up, and Down for the eight previously frozen `(layer, expert)` slots: `(12,13)`, `(13,22)`, `(13,28)`, `(14,63)`, `(20,17)`, `(21,61)`, `(24,26)`, and `(24,60)`. Coordinate order is role-major, Q4 then Q5, giving 48 columns. All columns are measured on the same 16 previously seen sentences and 656 gold-token targets. They are computational derivatives at alpha zero under the baseline hard expert indices; continuous router scores, selected weights, and the entire prompt/K/V tangent remain dynamic.

The proven native executable is reused without rebuilding. OFF uses the natural baseline path. Each non-OFF cell replays baseline expert indices and must reproduce the original route bytes. Every zero-alpha/JVP primal must reproduce the exact original gold bytes. The expected JVP execution counters are 4,334 processed tokens, 32 resets, 117,018 layer steps, and 656 gold scores. The resets include warmup and measurement; they do not represent 32 independent sentences.

## Execution and numerical gates

There are **79 new native cells**: one OFF control, 48 JVP columns, and 30 fixed-route finite-difference cells. The six new FD coordinates are `(0,Q5)`, `(1,Q5)`, `(2,Q5)`, `(3,Q4)`, `(13,Q5)`, and `(23,Q4)`. Each uses alpha `0`, `+0.05`, `-0.05`, `+0.025`, and `-0.025`. The three prior Q4 FD references for roles 0/1/2 are reopened from their independent audit; the new matching JVP outputs must be byte-identical to those earlier measurements.

The execution order is OFF, then the three new Q5 Gate/Up/Down checkpoint blocks, then the remaining JVP columns in role-major order, inserting the other three FD blocks immediately before their JVPs. The complete ordered sequence is in the sealed plan. Nine columns have individual FD checks; **39 columns do not have individual FD checks**.

Both coarse-versus-fine FD convergence and JVP-versus-fine-FD agreement must satisfy `RMS(error) <= 0.0001 + 0.05 * RMS(fine FD)` across all 656 tokens. A failed checkpoint is sealed as a numerical HOLD and stops further native cells. Thresholds, coordinate selection, and alpha values cannot change after the plan is sealed. Only a separately recorded execution/integrity failure permits an explicit failed-cell retry. Successful cells are reopened and verified, never recomputed to obtain a different outcome.

Only after all 79 cells and nine checkpoints pass may the runner publish the `656 x 48` raw derivative matrix, token-weighted `g = mean(dNLL)`, and the uncentered observed-label directional Gram `G = mean(dNLL dNLL^T)`. It preserves sentence boundaries and reports sums as well as means. A numerical eigenvalue check requires `lambda_min >= -1e-8 * max(1, trace(G))`, with a residual bound from the eigensolver. PSD is expected from the construction; it does not independently establish derivative accuracy.

## Pins and actual preparation evidence

| Object | SHA-256 / canonical receipt |
|---|---|
| Original AF4 weights, 9,814,343,680 bytes | `9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4` |
| Reused native JVP executable | `110b1978d97bcd7c6bdfd6d42142569e56e9dacffa4ff462dc0786b3128e49f4` |
| Reused native build receipt | `2cbeb4dc1a7e1ab5642a4abb22395498c10dde43bee051eec3c303c7e53fa008` |
| All48 runner | `7451c867ebb810cff9fe854322e00937bf8b1807c2d0475006d6703be32dd4de` |
| Prerun plan, canonical | `60fdced19fad43cbf2b73801dd68effbbd75c186f8a5f39de9274cad39453419` |
| Prerun plan, full file | `dc4cf87f557ef806d44c02cd9bf66a60dbec893303d74aa89cbf9ef0c5a3454f` |
| Prior independent three-role audit | `5ce09188e65d5cfcc0f956447b2918fe9c80bf81f14e8486ba97864f7137b87b` |

EOE plan job `job_e971441de87234ee2c72fd71ca33f189` completed `SUCCEEDED`, exit 0, in 10.920 seconds, with no native inference. The source closure, original model/runtime, prior raw evidence, and plan were verified before sealing. The downloaded plan's file and canonical hashes match the remote bytes. This is preparation evidence only.

The separately implemented preflight auditor then verified **147 files** on EOE. Its receipt is `737ef45e9f01d627513823fc88ba0c1caa6dc906b2635821cda252d04e3bfcf7`, full file SHA `99cb9e5d23f48e187b403d9628e6f02a0ef5612d17dae35ad63e0cefdf1517f2`. It correctly reported `INDEPENDENT_AUDIT_CONFIRMED_PARTIAL_NO_PASS`, integrity PASS, zero native cells, and numerical status NOT_COMPLETE. Job `job_cc1893407a481c192206282e1b2a87d7` ended with the intentionally specified exit 2 in 5.400 seconds; the scheduler therefore labels that job FAILED. This is an expected preflight state, not a native numerical failure or a completed 48-direction result. Its task scratch was removed. The final independent audit remains a separate output after native execution.

Research root: `/Volumes/D50/TailnetCommander/work/je01_native_all48_train16_20261123_v1`. The runner uses the seven unchanged, hash-pinned parent Python source files in `tools/`, and copies the sealed CAL240 byte ledger from the immutable parent root. It reads the existing parent native binaries, source, bank, adapter, model, and TRAIN16 inputs; none is replaced.

## Runtime and independent audit

Measured prior times were 73.8836 seconds for OFF, 154.9563 seconds for Gate JVP, 151.2950 seconds for Up JVP, and 71.4456 seconds for a fixed-FD cell. The frozen estimate is **9,655.154 native seconds, approximately 2 hours 41 minutes**, excluding preflight hashing and audit. This estimate is not a runtime guarantee. Each cell has a 900-second ceiling, an initial system memory-free gate of 55%, a running floor of 35%, and an 18 GiB process RSS ceiling. GPU allocation is not separately attributed by RSS, so the system memory gate remains necessary.

`je01_all48_train16_workflow.py` binds the exact plan, runner, auditor, and its own source hashes, runs the native study, and then launches the independent standard-library-only auditor in a separate process. The auditor imports no experiment helper, executes no model, reopens every raw gold/route/JVP record, checks actual native configuration logs, independently computes the 48-column statistics and nine FD comparisons, and rehashes the full original model. Complete, numerical HOLD, partial, and evidence-corruption states remain distinct. The workflow does not dispatch TRAIN240 or a policy search.

The fixed-route matrix is an exploratory local surrogate on seen TRAIN16. It is not a full-parameter Fisher, a Hessian, a finite-step natural-routing stability result, or an allocation-quality result. TRAIN240 source-linked inputs and a document/sentence-level fitting plan are the next prerequisites. A single policy must be frozen before validation128/OOD128 access, and a natural-route finite-change NLL check remains necessary for any candidate allocation.
