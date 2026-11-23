# JE01 full48 TRAIN16 v2: Q5 progress and bounded verification

**2026-11-23 KST — RUNNING. This document does not establish a full48 independent PASS.**

## Confirmed progress

The unchanged durable job `job_cd608db3eb69d616b64e20cca1a9f050` was confirmed RUNNING with a fresh heartbeat at **10:38:25 KST**. The complete stdout page contains **20 newly completed cells plus one inherited OFF = 21/79 evidence observations**, **5/48 JVP columns**, and **5 passing provisional runner checkpoints**. New Gate/Up/Down Q5 comparisons and the repeated Gate/Up Q4 comparisons had passed. Down Q4 was the current JVP cell. No duplicate native job or concurrent full independent auditor was started.

The exact observation, earlier published snapshot, immutable plan binding, source commit and successful deployment receipt are preserved in `JE01_ALL48_TRAIN16_V2_Q5_PROGRESS_OBSERVATION_20261123.json`. Normal execution remains owned by the original controller; final independent audit follows the remaining native observations.

## Three new Q5 checkpoints

All three rows concern layer 12, expert 13, the already observed TRAIN16 (16 sentences, 656 gold tokens), and fixed hard-route selection. The acceptance threshold is `1e-4 + 0.05 * RMS(fine FD)`. FD convergence compares h=0.05 against h=0.025. The JVP comparison uses h=0.025 as its finite-difference reference.

| Role | FD convergence RMS | JVP–fine FD RMS | Acceptance limit | JVP error / limit | Runner verdict |
|---|---:|---:|---:|---:|---|
| Gate Q5 | 4.934233615420578e-05 | 4.419894479598654e-05 | 0.000559211472601608 | 0.0790379793 | Provisional PASS |
| Up Q5 | 4.939453533290846e-05 | 4.447416913678203e-05 | 0.0006347045530395977 | 0.0700706635 | Provisional PASS |
| Down Q5 | 5.068618450929481e-05 | 4.531739637901179e-05 | 0.0006808570842775403 | 0.0665593374 | Provisional PASS |

| Checkpoint | File bytes | File SHA-256 | Canonical receipt SHA-256 |
|---|---:|---|---|
| Gate Q5 | 43,703 | `6d6734ec6a5e91d948289837fbc729a6321e572a130b34a054076c78fde6b9bd` | `b5cfefbc499028df297b6668b86b2dd0e736074a31bd6c845a59cb57382c0be1` |
| Up Q5 | 43,640 | `761ae02ac8d00a6ec05d6c2a7612d2dcecf8c5a795bcfaf4aba6ef4f7bd47c99` | `1cf4ca3dac55e8b6db4fba12f4029403b412026aff70a5cf35dfdaa610bdaaf8` |
| Down Q5 | 43,616 | `483705f274370d4ebec130398eb02a119a25a23320a9020407ba931713100e1e` | `ed6a77fc02767368bc6b54fab173fb4d1d0a36838c3f67c99dbbbca5495da49f` |

The exact Gate checkpoint was already preserved as `JE01_ALL48_TRAIN16_V2_GATE_Q5_CHECKPOINT_20261123.json`. The new Up and Down checkpoint files are preserved verbatim beside this document. Root independently confirmed full-file checksums, canonical hashes, the fixed v2 plan and native/model identity fields, and the numerical values shown above. A checkpoint checksum proves integrity against the observed bytes, not a completed independent raw-source audit.

## Limited independent recheck of Up/Down Q5

A separate reviewer read the two completed JVP native-output files, their exact RUN receipts, the pinned gold metadata and a gold output matching the baseline SHA through authorized bounded file-download calls. No remote CPU/native job or full auditor was started.

For each of the two directions, the reviewer independently decoded **656 raw native records**, verified the full-file checksum, format magic and length, `(case, index, target, position)` alignment and finite numbers. The primal NLL embedded in each JVP record was bitwise equal as float64 to the baseline gold NLL. The RUN receipt's three derivative arrays were exactly equal to the native raw values.

The reviewer then independently recalculated JVP versus fine FD, FD convergence, means, RMS values, thresholds and cosines. For both directions, the reported checkpoint metric fields and recomputed values agreed exactly (maximum absolute difference **0.0**). Up cosine was `0.9999914634178091`; Down cosine was `0.9999924938648919`.

This narrower recheck does not replace the scheduled full independent audit. The +/- FD native gold outputs were not decoded again; the FD convergence recheck used checkpoint coarse/fine arrays. Routes, stderr, current whole-model and native-executable hashes, the other 46 JVP columns and the full Gram were outside this limited check. The Gate Q5 check in this continuation is confined to exact checkpoint integrity and reported metrics. The table above therefore retains **provisional runner** labels for all three rows.

The limited decoder used Python standard-library `struct` with 656 `<4I4d` native records and did not import the experiment runner or auditor. It also checked nonnegative logit RMS. The bounded downloads reached EOF and the local verification process exited 0. Exact additional input pins:

| Input | Bytes | Full-file SHA-256 | Canonical receipt, where applicable |
|---|---:|---|---|
| Up Q5 JVP raw | 31,523 | `794b6aa697281c8c8c0ea731547c11b0f0a8c1a549a86fe7995280393673393e` | — |
| Down Q5 JVP raw | 31,523 | `b3fafd1ba14e0a6925b9d7e25b27865826f0d8683296e67f612f6be8b2994653` | — |
| Up Q5 RUN receipt | 100,827 | `2f05072b714ec04c112def153910b7b550cc3df68c9ce31b0bb9d8724507d0bc` | `c952259bd1e7e2200a56c33d1a22867961c37ebb89c29a4059d368bdf2183897` |
| Down Q5 RUN receipt | 100,612 | `cba43b3776e5e0f079c6bd36dea31035f50d0db7f93c533c0bdb56511a6c3ba2` | `ce4c3cbac5b6e53e4f6a50ef62bfb0730d4173146ccdedb1fa895e1e2a4fbab5` |
| Gold metadata | 24,744 | `be0dab12624efb4fae17538b4c5e6ef76bd09aebf4c09b1b428fc1d8f0fff406` | — |
| Gold output identical to baseline | 21,026 | `b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c` | — |

## Published viewer

- URL: https://je01-native-sensitivity-20261123-abdb.sweet-eft-6723.chatgpt.site/
- Source: `5c52849b1edd4491e83d6190d0c0e1d28d439652`.
- Deployment: `appgdep_6b0398c08de88191aa6a4cf7707959ea`, succeeded at 10:37:15 KST.
- Timestamped snapshot: **10:34:38 KST**, **19/79 evidence**, **3/48 JVP**, **3/9 provisional checks**. It is a static snapshot, so it intentionally predates the later 10:38 observation above.

The existing snapshot recorder changed only observation metadata and the explanatory note. All earlier experiment objects, fixed v2 plan/coordinates, v1 recovery history, and six original source JSON files were verified identical to source `2230a3e6342ed6624056b9101ec4b3dcd85c4a1a`. `audit_available`, `available` and `matrix_available` remain false; the final independent table remains 0/9 and no full48 matrix, g or Gram is exposed.

The exact local preview visibly showed the new counts and Q5 note. Production HTTPS reached the ChatGPT sign-in page; authenticated production contents and physical iPad Safari were not verified. The Site remained owner-only (custom access with one owner and zero external visitors). The temporary preview was stopped and the source checkout was clean.

## Continuation and scientific limits

The existing bounded completion watch remains active and unchanged: first check 11:15 KST, hourly for at most six checks. The durable native job and its final independent auditor are intentionally kept running. Full completion still requires 79 evidence observations (78 new native cells plus one inherited OFF), 48 JVP columns, all nine preregistered FD checks, PSD and matrix/analysis integrity, plus matching native and audit exit 0. The other 39 directions are not individually FD checked.

Only a full independent PASS permits the already prepared TRAIN240 source-only manifest, staging and separate source audit. This continuation has not staged or compiled TRAIN240 sources, collected TRAIN240 derivatives, selected an allocation, accessed validation/OOD, changed the AF4 policy, merged the draft PR or activated a release.

Observed-label directional sensitivity is not a finite bit-allocation improvement result. An uncentered Gram is not the model Hessian or expected Fisher. The statistical-selection addendum remains a draft. See `JE01_ALL48_TRAIN16_V2_RUN_HANDOFF_20261123.md` for the exact completion protocol.
