# Beglin HCP P5 — Actual XOX Native REBIND_SET ACK and Scratch KV128 (2026-10-14)

## Definitive result

**BOTH P4 BLOCKERS SOLVED IN TWO FULLY ISOLATED XOX RESEARCH BINARIES. NO DEPLOYMENT OR MAIN CHANGE.**

1. **Legacy native staged executable / new Python epoch scheduler mismatch:** the old immutable persistent binary's original source `8b46cdd38f5b39dcaeaa9d1c5dfbd9642601a30a` does not implement `REBIND_SET`. Current Beglin Python scheduler submits `REBIND_SET` and requires a durable `REBIND_SET_APPLIED` ACK. A new scratch binary compiled from the reviewed current GitHub C source supports both and passed true policy epoch transitions without monkeypatching ACK validation.
2. **Separate GPU capacity bound:** original native C `MOE_CBATCH_MAXPOS=32` and C++ MLX backend `MLA_L0_MAXPOS=32` both prevent 104–112-token persistent MLA inference. Changing C alone failed at GPU layer 0 step 2, as expected from the independent C++ position check. A second **scratch-only** binary that changes **both** to 128 now successfully runs actual 106–112-token inputs, while keeping `MOE_MAXPOS=160`, `GQA_L0_MAXPOS=32`, source mathematics, Top-K routing, and all ACK / OOB checks untouched.

The immutable old stage and original active-route file remain unchanged. Automatic live promotion stays OFF. These are **research-path validations**, not a licensed production deploy or proof of arbitrary-length context.

## Source and exact-build boundaries

- Repo `popixoxipop-collab/beglin`, research branch `research/hcp-p5-scratch-rebind-set-20261014`, **not merged into main**.
- Original current native C SHA-256 `dd73b7c7f997aa66d0a2217f9b0f3ab74c5db28b8dba1b586d4f182352bbcee1`; original GPU C++ Git blob SHA `b6cbc1c98cfedcd0f3c99f437d1f6c5978f2ce5a`.
- Unmodified C / updated CMake scratch binary for 24-token REBIND_SET: SHA-256 `b1b763136a6f71289814602ad46b8c5bd77de692af0939add95f992f56651ec6`.
- Separate bounded scratch C++ backend: C SHA-256 `bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612` and MLX C++ SHA-256 `daded3c0e4bf7cc2a86e0fab8c1f73322b5b0ffea0002d9b2d38857d2e9591ae`; compiled CB128 binary SHA-256 `ca376b07827471243d28cec3ca356fd71cdc2f8aa980f14a153602335dae644e`.
- Old persistent stage binary (unchanged) SHA-256 `8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9`.
- DeepSeek-V2-Lite index SHA-256 `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`.
- Both script paths override the stage binary **only as a variable inside an isolated test Python process** and point to a run-owned binary. Python `PrecisionEpochScheduler` and `gpu_runtime_control.verify_terminal_ack` are unchanged. Original stage SHA and active route SHA are verified before/after.
- Scratch KV128 worker requests at most 112 input + 4 new tokens per prompt, one GPU slot, max 14GiB observed process RSS, >=40GiB physical memory guard, disk guard >=3GiB, and guard for exact sources/bounds. Only reviewed DeepSeek MLA is covered. No live workers are restarted or modified; each run's compiled binary, raw prompt IDs and local worker queue/logs are deleted.

## Actual 24+4 token REBIND_SET proof

[GitHub Actions #38496229099](https://github.com/popixoxipop-collab/beglin/actions/runs/38496229099) **SUCCESS**, executed `a922d7c2fbf59992ef87d65cdf709c1c0d6370a7`.

- Initial two-prompt probe: true native REBIND_SET applied with runtime ACK PASS; no mocking or bypass.
- Full separate scratch native 12-domain A/B: **12/12 first generated token agreement**, **11/12 exact 4-token sequence agreement**.
- Engine time p50: policy A **310.783ms**, B **311.439ms**. Extra model/transition overhead measured but no statistically justified model speedup.
- Exact A startup hash `dbee11614bb74073e0751bbf9fad0f67f0b99396256cb88589df80701f363ef1`. Exact B hash `e471dfb075ca86cd0c5341b972258df7991847bb3f1557b69447c8b6168042ae`.
- Artifact [#11863622660](https://github.com/popixoxipop-collab/beglin/actions/runs/38496229099/artifacts/11863622660). 12-domain result SHA `9cade884e2fa31dfc6bc19fdb1d099667df9a52c95a50fd2f76854468a517f97`.

## Actual 104–112 token scratch CB128 / MLA128 proof

[GitHub Actions #38497591870](https://github.com/popixoxipop-collab/beglin/actions/runs/38497591870) **SUCCESS**, executed `d0ab5c58cf7e8d4846be69d68946e640da03b424`.

- First two independent 109-token DeepSeek prompts: real A/B outputs, exact multi-target epoch and `REBIND_SET_APPLIED` ACK, both pass.
- Follow-on 12 distinct domains: actual input lengths **106–112 tokens**, all matched by original 36-prompt AF source token hashes, a new worker with GPU KV128; 26-layer network fully executes to output.
- **First token:** A/B agree **12/12 (100%)**. **Four-token sequences:** **11/12 (91.67%)**; the mathematics-domain 111-token prompt diverged after the first token. This is a genuine output difference.
- Measured per-admission model engine time p50 (includes transition workload when the admission switches policy): A **698.5865ms**, B **694.8490ms**. p95: A **860.9573ms**, B **993.6007ms**.
- AB/BA alternation cancels much of the policy-switch bias but does not give a robust steady-state benchmark. Across 12 paired domains, *B minus A* mean engine time is **+26.2745ms** with domain bootstrap 95% CI roughly **[−155.28ms, +226.75ms]**: no established B speed benefit. Removing recorded transition duration as an exploratory (not authoritative) residual gives B minus A mean **+2.82ms**, CI roughly **[−9.76ms, +15.57ms]**.
- Real worker RSS p95 A/B both **6,342,121,882 bytes (~6.34GB)**, well below the 14GiB fail-closed guard; independent 64GiB XOX host, 1 GPU request slot. This does **not** size multi-user/large-batch production demand.
- Artifact [#11864107325](https://github.com/popixoxipop-collab/beglin/actions/runs/38497591870/artifacts/11864107325). Original ZIP SHA-256 `8dc730feb1cadbc20844e3f4e76604d86b7d47f9956200804b4352094d262eb1`. Source JSON SHA-256 `6d3899f7cd224b1430e4140e89a8baabc35471ebbecfc2dec76cac351be8519a`; matches exact source's receipt. Receipt SHA-256 `3b5c2f9c8d37d82f80cd4be5097fdea0e5273734081c96e2ccc53ba4b5c5a064`.
- Source previously measured actual 36 longer AF traces [P4 Run #38467501959](https://github.com/popixoxipop-collab/beglin/actions/runs/38467501959): feature SHA `c3b54b1ebe41d9a85ed94ddd0da87cde0de86ecf0af56d2e621733e439944f53`. Every P5 long A/B input was matched to that P4 SHA-bound tokenizer input, **but the original AF logits argmax is not present in this P5 result**.
- Local P5 source/dual GPU-C boundary tests [#38497516087](https://github.com/popixoxipop-collab/beglin/actions/runs/38497516087): **15/15 PASS**.

## Negative / unresolved evidence

- **No AF-model logit agreement measured for the exact P5 long A/B generation:** the original P4 raw logit bytes were correctly removed; matching the prompt hash does not reconstruct their values.
- **No supervised task quality, perplexity, BF16 teacher accuracy, or actual H-JEPA-trained policy gains** measured. First-token A/B agreement is NOT correctness. In the 12-domain comparison there is one real sequence mismatch.
- B is **not demonstrated faster**. Transition times (hundreds of milliseconds at cold/warm epochs) and input context/history dominate small differences, and paired CI crosses zero widely.
- C/MLX scratch CB128 is **not** certified for production, other MoE architectures (GQA still 32), multiple GPU slots, >112-token contexts, long-lived/restart recovery, concurrent admissions, thermal conditions or persistent deployment.
- Existing `main`, operator/release-plane, old stage and every production precision policy remain unchanged. Explicit separate release authority and genuine task-quality/full-context comparison are required before considering migration.

## Recommended next research gate (not performed)

A separate sealed XOX **same-input AF-versus-qNg64 logit and token comparison** for both policies across full 112-token prompts, followed by multiple temporal blocks/seeds, p50/p95 steady-state throughput inclusive of switching and planner cost, concurrent slots, memory ceilings, cold-boot replay, and immutable release provenance. The H-JEPA-inspired 16-step slow/fast predictor from P4 remains statistically inconclusive and is not connected to a real policy control action.

**P5 outcome: ACK blocker SOLVED in scratch; both C and MLX capacity blocker SOLVED in scratch; production promotion remains BLOCKED.**
