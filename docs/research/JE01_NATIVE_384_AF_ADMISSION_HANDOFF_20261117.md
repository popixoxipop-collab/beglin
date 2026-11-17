# JE01 384-context actual BEGLIN AF model handoff — 2026-11-17 KST

## Verified live source and production safety

The current BEGLIN original **native compressed** DeepSeek-V2-Lite model has five read-only source files on the connected XOX host. Exact metadata from the authorized host filesystem:

| Filename | Source bytes |
|---|---:|
| deepseek_moe_af.bin | 9,814,343,680 |
| deepseek_moe_f32.bin | 14,137,344 |
| layout_af.txt | 25,625 |
| layout_f32.txt | 6,695 |
| arch_config_moe.txt | 372 |
| **Total** | **9,828,513,716** |

Source directory is the already-existing BEGLIN native `moe_base_deepseek/weights_moe`. Actual XOX local production health `status=ok`, route generation 6, route `beglin-candidate-shared-up-l3-n6`, 2 live workers; no production files or workers changed in this work.

EOE→XOX SSH was denied at the initial host-key check. **Host key verification was not disabled**. The public ED25519 fingerprint returned by XOX itself exactly matched the EOE LAN keyscan. An EOE→XOX credential test then failed permission validation (publickey/password/keyboard-interactive), so that direction was rejected without bypassing authentication.

The preexisting **XOX→EOE** `macstudio` SSH alias (public-key auth plus strict known-host validation) successfully executed a read-only `id -un` on EOE and destination capacity/stat checks. This approved **reverse-copy** path can read XOX weights without stopping live services. A source-side fail-closed script was staged only in the isolated XOX sandbox: `JE01_XOX_AUTHENTICATED_MIRROR_TO_EOE_20261117.py`. It checks production health before/after, SHA256-hashes originals before transfer, limits rsync bandwidth to 50,000 KiB/s, transfers exactly the five files, and emits a completion sentinel only on success. **The script has NOT been started for the large transfer**.

## Resource admission gate

- Latest observed Tailnet operating `current/live_release`: **g6-0.9.0-operator.900485**, converged. No release mutation or CAS activation was requested.
- EOE D50 filesystem reported approximately **215 GiB** free and approximately **165 GiB allocatable** under Fleet storage-pool reservations, sufficient for an isolated 11 GiB request.
- EOE Fleet `resource_plan` nevertheless rejected the 1 CPU / 1 GiB RAM / 11 GiB work-pool model-copy SHA job because **allocatable CPU was 0** (separate existing reservations of four CPU cores across two reservations; do not cancel others).
- First bounded queue request `je01-eoe-d50-xox-origin-af-transfer-fullsha-20261117-v1` expired after repeated `NO_CAPACITY` and no job dispatch.
- Second 10-minute request `je01-eoe-d50-xox-original-af-transfer-sealed-v2-20261117` is now confirmed **EXPIRED** after **6 NO_CAPACITY attempts**. Exact `resource_queue` read at 2026-11-17T05:42:49.763Z returned `backend_job_id=null`, selected_host=`scheduler`; **the EOE verifier never started, and no AF file transfer was launched**. A new idempotency key and fresh admission would be required to retry. Do not bypass this gate.

BOB was also checked as a destination for 11 GiB / 11 GiB RAM native Metal inference and rejected on `missing_tag:mlx`, `insufficient_free_ram` (6.6 GiB allocatable), and `no_eligible_storage_pool`. XOX rejected even with zero new requested storage due to its configured storage reserve; not overridden.

## New real input staging source

`tools/je01_original_native384_input_manifest.py` was added, with `tests/test_je01_original_native384_input_manifest.py` (GitHub JE01 suite **82/82 PASS**, run `44025811708`). The generator reconstructs the real frozen 256/128/128 public samples, verifies the SHA-attested original tokenizer and config, canonical 512-row manifest hash, exact previous per-cohort token-ID digest, inserts the original DeepSeek BOS ID 100000, guards input+one target <=384, and emits the existing native BEGLIN `QWEN_MOE_CB_PROMPT_MANIFEST` raw little-endian int32 format. It does not use holdout gold labels in row choice or reselect samples.

**Caution:** Code contract CI success is not a claim that the **512 actual binary prompt files have already been emitted** or that a single native 384-position model forward/NLL was run. No original full-model native logit parity, true routed Expert posterior NLL, packed-byte matched quality, or latency improvements have been measured.

## Next bounded action after capacity admission

1. Recheck latest Fleet current/live release, storage and GPU/CPU admission. Confirm a new unique research destination under EOE D50 is empty.
2. **Only when the EOE managed verifier reports READY**, run the already-written strictly authenticated XOX→EOE source read-only rsync with source and destination SHA cross-check.
3. Download the already real-compiled ARM64 MLX 0.32.1 JE01 384 B1 GitHub Actions artifact (run 44017766019) to EOE scratch, verify binary SHA `bd7d79661d25370d50529b92cddaa3871cd60be6211263dbc29f35aeacd35b6f` for the fused-device build; do not install it to the XOX or EOE serving locations.
4. Use the original-source native 384 binary with the copied, hashed AF bundle to test a short exact-prefix forward against the original frozen binary's output. Then run longer prefixes up to the preregistered 331-token complete-sequence maximum. Do not assert native logits parity before measuring actual logits.
5. Only if all previous gates pass, proceed to original JE01 256 calibration / 128 same-source / 128 OOD gold-token NLL and real exact packed-byte model comparison. The existing persistent worker's greedy token result alone **is not gold-token NLL**, so an audited teacher-forcing logits or likelihood output path remains necessary.

**Current decision:** ORIGINAL_SOURCE_FILES_PRESENT / REVERSE_SSH_AUTH_PASS / NATIVE_384_INPUT_GENERATOR_CONTRACT_82_PASS / EOE_CPU_ADMISSION_BLOCKED / SOURCE_MIRROR_NOT_STARTED / ORIGINAL_NATIVE_MODEL_FORWARD_NOT_RUN / JE01_P2_HOLD / NO_PRODUCTION_PROMOTION.


## Real 512-input artifact and copy-to-new-path verification — 2026-11-17 KST

**The prior input-binary generation blocker is RESOLVED.** A GitHub-hosted Linux Runner reconstructed the actual 512 SHA-frozen original DeepSeek inputs, using immutable DeepSeek-V2-Lite revision 419ef8b6b7a816343af14bc1a4436d4e33affb5d, original tokenizer SHA 41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669, original config SHA f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d, and the unchanged 512-row manifest SHA a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147. All 256 calibration +128 in-source heldout +128 OOD native raw little-endian int32 files were actually written, independently rehashed and verified with BOS 100000, correct source token IDs and a 384-position constraint, **512/512 PASS**. Real native input receipt SHA 2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34. First successful run: https://github.com/popixoxipop-collab/beglin/actions/runs/44065018857 .

**Portable destination repair is also REAL tested.** The first native manifests embedded Linux Runner absolute paths; simply moving this archive would break BEGLIN on EOE. A new fail-closed script, tools/je01_native384_relocate_actual_artifact.py, validates the SHA-bound input index and every original 32bit token file, then writes new absolute native manifest paths for the current destination without altering the 512 prompt bytes. A second GitHub-hosted run physically copied the input files to a different directory and proved **all 512 SHA and rebased manifests PASS**. Relocation receipt SHA 204035fe04b0dbdd9f8850b8e071548f31ebd623a26d8b58c2662c1d706df152, run https://github.com/popixoxipop-collab/beglin/actions/runs/44065613386 , [actual artifact and receipt](https://github.com/popixoxipop-collab/beglin/actions/runs/44065613386/artifacts/14320606795), ZIP SHA db799b975f0edcabe1b30810115685dfcc2fea795729d52f1f91eeef06bbf91f, 358,785 bytes. Artifact expiry: 2026-12-01 10:31:55 UTC. Host-specific relocation must be rerun after placing the artifact on EOE; the hosted path is only a successful test, not an EOE path.

Detailed machine-verifiable provenance: JE01_REAL_512_NATIVE384_PORTABLE_ARTIFACT_PASS_20261117.json. Full JE01 offline source and receipt checks **86/86 PASS** at GitHub Actions run 44065890566.

**Remaining gate:** New EOE resource check still shows 0 allocatable CPU after existing reservations despite D50 storage headroom; XOX has no eligible new scratch pool. Thus original BEGLIN AF weights have NOT been copied into EOE, original 384-position full-model inference has NOT run, original BEGLIN logits parity and teacher-forced gold NLL are NOT MEASURED. The original operating release, route, and weights remain untouched. Do not infer Bayesian precision gains from the data-artifact work.
