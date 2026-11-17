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
- Second 10-minute request `je01-eoe-d50-xox-original-af-transfer-sealed-v2-20261117` was still `WAITING` on the last direct check shortly before expiry (no backend job ID at that time). **No destination directory or transferred AF files are claimed** without a corresponding `resource_queue` DISPATCHED + `job_output` READY receipt. Reconcile the queue by idempotency key before any retry.

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
