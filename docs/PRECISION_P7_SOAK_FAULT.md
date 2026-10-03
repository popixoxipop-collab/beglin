# P7 Precision Soak, Rotation and Fault Boundaries

P7 extends P6 observability without enabling it in production.

## Scope
- accelerated repeated-admission soak on an isolated XOX worker;
- bounded active journal rotation into immutable SHA-256 sealed segments;
- hash-chained segment manifest;
- restart replay across sealed segments plus the active journal;
- duplicate admission detection across archived and active records;
- crash recovery for the window after segment+manifest durability but before active-journal reset;
- fault injection for manifest write failure, active reset failure, manifest tamper and segment tamper.

## Safety boundary
Rotation defaults to unlimited segment retention. Automatic deletion is intentionally not enabled because deleting a sealed audit segment without an externally checkpointed/signed retention contract would weaken the audit chain. A future retention policy must first define a trusted checkpoint/anchor.

The production generation-6 supervisor, route manifest, binary and worker processes are not modified by P7 acceptance.

## P7 gates
1. repeated admissions remain on one isolated worker PID;
2. policy/epoch lineage remains replayable across rotations;
3. cache materialization converges to warm hits rather than repeated cold allocation;
4. adaptive two-pass requests remain finite and observable;
5. restart summary equals the pre-restart aggregate;
6. segment/manifest tamper fails closed;
7. manifest failure leaves the active journal intact;
8. reset failure after a durable seal is recoverable without data loss;
9. production pre/post identity is unchanged.

## Exact-source XOX acceptance
Implementation commit: `4baf28b3745a170fa7dc36a6630b1316ac0970fe`.

- native binary SHA-256: `d03f4748345195500179a6418142e1a73b2259a2e330ddcc0c28b702994d401e`
- cost evidence: `/Users/xox/vdsp_shadow_runs/precision_e2e_cost/p7-soak-4baf28b/result.json`
- cost evidence SHA-256: `8f969bd0b01a20a960bdf659ef9fe683cd953608b5f3286b3da597796edca30e`
- soak result: `/Users/xox/vdsp_serving/precision-p7-soak-4baf28b/result.json`
- soak result SHA-256: `8376c1605f2c4e8e9ac89310efadb7ed8aff396ba3631e24e916bfcaf2ab17bd`

Measured 60-admission accelerated soak:
- 60/60 successful admissions on one isolated worker PID;
- 30 closed-loop, 20 explicit-policy and 10 adaptive admissions;
- 40 precision-transition admissions;
- 20 low-margin trigger admissions;
- cache hits/misses 58/2 = 96.67% hit rate after warm materialization;
- inference passes: 50 one-pass, 10 two-pass;
- finite logits 60/60;
- five sealed 10-record segments retained, manifest chain valid;
- restart summary exactly equals the pre-restart aggregate;
- final policy restored to startup policy.

Fault/rotation tests include manifest write failure, reset failure after seal, manifest tamper and segment tamper. The reset crash window is recovered only when the active journal bytes exactly match the latest sealed segment SHA; other overlap remains fail-closed.

Regression before implementation seal: GPU suite 219/219 PASS, focused precision/supervisor/fault suite 63/63 PASS, native transition 1/1 PASS.

Production after acceptance remains generation 6, route manifest `c2bf7eed2788e115af0c0cb3316fceb061000013b2bfcf57710f66d291ff4221`, baseline/candidate PIDs 35481/35493, adaptive L26 enabled and automatic promotion disabled.

Certified successor regression: run `37127197730` SUCCESS on `f73891411d65be15a5de9e7496828a40d2b05d27`. The only change after that implementation+evidence state is removal of the temporary P7 branch trigger from the workflow.

## Extended 300-admission soak

A longer post-merge isolated soak was run from source `fd05cd328b9436f495eb01401bfb7ec24ff79909` using the same certified P7 native binary and cost evidence. Production was not modified.

Evidence:
- `/Users/xox/vdsp_serving/precision-p7-soak-300-20261003/result.json`
- SHA-256 `ed453333e55c52fdaf727a10eec4fd654ac8b5b78c4a98bac4c2edd50336a749`

Measured result:
- 300/300 successful admissions on one isolated worker PID;
- 150 closed-loop, 100 explicit-policy and 50 adaptive admissions;
- 200 transition admissions, 100 low-margin trigger admissions;
- cache hits/misses 298/2 = 99.33% hit rate after warm materialization;
- cache bytes added stayed bounded at 8,650,752 bytes;
- resident qNg64 cache current/max stayed bounded at 17,301,504 bytes;
- inference passes: 250 one-pass, 50 two-pass;
- finite logits 300/300;
- 29 sealed 10-record segments retained and replayed;
- restart summary exactly equaled the pre-restart aggregate;
- final scratch policy restored to startup policy;
- production pre/post identity unchanged.

This is an accelerated 300-admission soak, not a multi-hour or multi-day endurance claim. The next endurance gate should add wall-clock duration and explicit worker/ACK fault injection in addition to the already-covered journal/manifest faults.
