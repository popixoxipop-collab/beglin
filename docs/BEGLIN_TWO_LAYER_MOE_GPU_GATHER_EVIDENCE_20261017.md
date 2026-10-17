# BEGLIN P18 — real 2-layer DeepSeek MoE packed GPU and source-matched gather (2026-10-17)

**Engine research only.** No serving changes, no checkpoint uploads, no model-weight blobs in this repository. The original DeepSeek-V2-Lite BF16 files were read-only from a D50 **research copy**, which is NOT an attested XOX original. This document distinguishes implemented/tested boundaries from full-model readiness.

## A. Two source-defined BF16 decoder layers: MoE routing proof

- Original DeepSeek source `MoEGate` and `DeepseekV2MoE` classes executed as pinned source-defined Python; no synthetic router.
- The model has **64 routed experts, top-k 6 per token** at layer 1.
- Three short independently authored strings produced **37 token positions**, **222 top-k token/expert assignments**, using **57 of the 64 experts**. Non-routed experts were never loaded into RAM.
- Real BF16 2-layer layer-output computation PASS. Independently implemented weighted sum over selected source experts vs source MoE output aggregate relative L2: **1.2864e-5** (BF16 operation-order difference).
- BF16 research source bytes read 1,048,736,768; peak research process RSS ~2.29 GB.

## B. Mixed local precision is NOT universally better in routed MoE

The source-defined MoE routing was held fixed for this experiment. One always-on shared expert and one most-used routed expert were replaced with true packed Metal qNg64 Gate/Up/Down. Each of four per-expert policies uses the **same code+scale bytes**:
`(7,7,7)`, `(9,6,6)`, `(6,9,6)`, `(6,6,9)`.

The 16 cross-group policies were ranked by a separate 2-prompt BF16 layer-output SSE training set, then the selected policy tested on a third held-out prompt.

- **Winner: uniform 7 bits** for both expert groups; heldout first-2-layer output relative L2 vs BF16 = **0.003725245**.
- All 16 policies used the same packed Gate/Up/Down two-group budget (~24.33 MB).
- GPU vs source C maximum bitplane numerical relative L2 was **1.174e-7**.

This negative result is important: **per-tensor precision benefits cannot be inferred from the attention/dense-layer result to each MoE expert.** The n6 research custom Metal code is **not** the existing C++ production n6 dispatch (which repacks for MLX native).

## C. All active routed experts' Gate/Up/Down compressed on real Metal

In an independently executed experiment on the same three authored strings, the source BF16 first decoder layer and source BF16 layer-1 Attention/Gate were retained. All **57 routed experts actually invoked**, plus the **1 shared expert**, had their entire Gate/Up/Down projection weights independently encoded with BEGLIN C qNg64 n7, then their packed uint8 planes and F32 group scales uploaded to **real Apple MLX Metal**. 58 groups × 3 projections = **174 actual packed GPU module calls** (at least once each).

| Selected expert FFN weights only | Bytes |
|---|---:|
| Original BF16 equivalent | 1,020,788,736 |
| Native n7 code + F32 scales | **478,494,720** |
| Weight payload saving | **53.125%** |

First-2-layer final output relative L2 against original BF16: **0.009734939**; maximum per-module CPU C vs actual Metal numerical relative L2 **1.322e-7**. This is a real first-2-layer hidden-state test, not a full 27-layer logits/NLL or model task evaluation. GPU active memory after binding was measured for this group but **does not represent end-to-end peak memory** while CPU BF16 source and other modules remain present.

Read-only research proof SHA-256: `63c89d80afef78b6c99a601a1faf40089f59794298c1e52ece48bc000c6afc03`.

## D. Exact original BEGLIN source-matched multi-expert Metal gather

Instead of dozens of expert-specific GPU kernel dispatches, we reused the **two-level SIMD-group Metal bitplane source** from `mlx_moe.cpp::qng64_gather_gemv_kernel` and validated its explicit `(token,expert)` pairing behavior on GPU. Research adapter and standalone hardware tests:

- `research/local_quant_cpu_metal_20261017/qng64_metal_expert_gather_research.py`
- `research/local_quant_cpu_metal_20261017/test_qng64_metal_expert_gather_research.py`

Synthetic hardware checks: **4/4 PASS** for shuffled expert assignments, on-device activation input, invalid IDs/formats, and a packed sign-bit negative control. Tests compare against independent dense arithmetic and independent single-expert packed Metal execution.

**Real DeepSeek gate projection** test: all **222** actual source BF16 routing pairs over **57** active experts, real D50 BF16 expert Gate matrices encoded n7 in the source qNg64 bitplane format. A single gathered Metal dispatch matched the outputs from **57 separate packed expert Metal calls with zero per-element mismatch**, and sampled CPU C GEMV pairs with maximum relative L2 **1.314e-7**.

| Same 222 routed gate-projection pairs | Median wall time (five interleaved rounds) |
|---|---:|
| 57 separate packed Metal expert calls | 22.006 ms |
| **One packed Metal gather call** | **7.879 ms** |
| Measured API-level acceleration | **2.793×** |

Timings include Python/MLX dispatch and synchronization and differ in transfer overhead across the two calling conventions; not a pure shader-only speedup. **Only the Gate projection** was evaluated in this gather throughput test — routed Up/Down, shared FFN, router combination, and 27-layer decode tokens/s are NOT included in the 2.793× result.

Research receipt SHA-256: `b4f38d44bea595acac67b5214fd2a52b45807b0825a609dbdd4c2d0bf574d9bf`.

## E. What remains unproven

1. **Full 27-layer DeepSeek-V2-Lite packed inference** (all attention/shared/routed FFN plus embedding and final logits). Two layers verified so far; the MoE-only and Gate-only subtests have distinct boundaries.
2. **Model quality:** NLL, held-out tasks, decoding errors, top-1 agreement and generation under BF16/uniform/mixed at the *same model-wide* memory budget.
3. **Performance:** real end-to-end prefill/decode tokens/s, controllable GPU warmups, peak GPU/CPU/KV residency and power over multiple independent trials.
4. **Native engine integration:** research Python Metal wrappers reproduce the existing C++ source kernel, but the currently serving XOX BEGLIN binary was **not** rebuilt, activated or source-attested. Model source security gates remain independent and unchanged.
5. **Model-wide adaptive precision:** the earlier first-block n9/n6 policy is a local experiment; MoE equal-byte testing favored uniform n7 in the tested groups.

**Release state:** Draft PR #63 on the `research/local-quant-budget-20261017` branch, no merge or operational promotion. This document is only an engine evidence/implementation handoff.
