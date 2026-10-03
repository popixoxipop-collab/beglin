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
