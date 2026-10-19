# BEGLIN MLA Latent KV — Rebinding Operator Authority Guard (v3, 2026-10-19)

**Scope:** Research-only follow-on to [the physically verified two-token 27-layer MLA experiment](BEGLIN_NATIVE_CPP_MLA_LATENT_KV_20261019.md). Not merged or deployed.

## Why v3 is separate

The original **v2** source-defined BF16 resident latent KV cache reconstructed past keys and values by reapplying the currently attached layer's `kv_b_proj` to previously stored normalized latent data. The original v2 `bind()` correctly validated layer ID, attention shapes and architecture but did **not** compare the *actual operator weights* on a detached model layer after reloading it.

Thus, if a new `kv_b_proj` with identical input/output shapes but different weights were silently substituted between prefill and decode, the old latent vectors could be rehydrated using the wrong operator. Merely matching a layer ID and shape would not establish numerical identity.

## New fail-closed research-only binding contract

The `research/local_quant_cpu_metal_20261017/latent_mla_kv_source_cache_v3.py` candidate (not used in the real 27-layer two-token result) retains a fixed-length SHA-256 fingerprint of the **effective kv_b transformation** at initial construction:

- With the native BEGLIN qNg64 packed layer: verify n7/g64 expected operator geometry and hash the *actual* compressed uint8 planes and FP32 group scales.
- With standalone synthetic BF16 reference layers: hash the original BF16 contiguous matrix bytes and require exactly the DeepSeek-V2-Lite kv_b geometry.
- On later `bind()`, refuse a different operator digest even when source layer index and dimensions match. `unbind()` still releases the underlying weight/attention references; only the compact fingerprint and latent state persist.

This is a **source fidelity guard**, not a new precision policy. There is no automatic layer/weight replacement, no switch to dense dequantized GPU matrices, and no model/serving/runtime mutation.

## Executed EOE isolated QA — **not** a real 27-layer v3 model test

| Test | Observed result |
| --- | --- |
| v3 synthetic shape/latency-independent cache lifecycle and malformed input cases | **22 / 22 PASS** |
| Extra negative case: same layer, identical kv_b shape, but different BF16 projection values | **REJECTED** |
| Reconnect original source operator with unchanged digest | **ACCEPTED** |
| 16 consecutive synthetic BF16 token states and original projected expanded K/V | **16 / 16 BITWISE PARITY** |
| Actual saved latent BF16 per-layer KV at 16 tokens | **18,432 B** |
| Original expanded BF16 per-layer KV at 16 tokens | **163,840 B** |
| Reduction in resident KV **tensor bytes** | **88.75%** |

The 22-case suite uses synthetic BF16 CPU weights and no real source model checkpoint. The 16-token parity suite also tests detach/rebind and native tensor growth without reading the D50 checkpoint. The real 27-layer physical experiments (two tokens selected and actually fed back, `[429,4281]`, next prediction `6230`) and cache reduction are **valid v2 measurements**, not proof that v3 has already passed on real model data. No real-v3 performance or final logits comparison is claimed.

## Source and evidence identity

- Portable proposed v3 cache algorithm SHA-256: `2d96a2d6b2bbac169a76c91bbea7fed9e36e44e56429e73103adbcf8e2c35650`.
- Portable 22-case negative regression SHA-256: `9097f4156c151a13e9e1941039c5100c7600f941ee49baf3dd17a90c63be188e`.
- Portable 16-step BF16 parity regression SHA-256: `d69e82730f2e2ddcaffc89fc81386d0837b96e8789e5c83f49e443e6ed2d6f03`.
- Original 27-layer MLA v2 source SHA-256: `587b5371f31a88f07979e27405d26785e54942eccf5fb8c80e08d629eb3eb792`.
- Original real 27-layer two-token v2 receipt SHA-256: `c3f7f9edfaddcc1ce8035261805e3e38a109a0027bdf31c455c55359bdd31645`.
- Confirmed separate independent audit receipt (this EOE session): `5ec343f79b3445a601e842d59bab3b819b9e29da465bef5a045eb88d69d63b66`.

The portable research source and synthetic tests contain **no private machine paths, model weights, checkpoint data, logits, activations, keys, user prompts or secrets**. Original checkpoint-dependent execution and immutable receipts remain on EOE.

## Next required gates

Before claiming v3 works on the real model, execute the v3 code on a SHA-pinned independent real BF16 source as a new isolated 1–2 layer and then 27-layer two-token test; require parity against v2 and uncached original model, exact native-C versus Metal numeric checks, same expert router outcomes, a verified operator digest over each rebound kv_b tensor, and unchanged memory accounting. Source fingerprinting has a computational cost whose effect on cold-streaming and fully resident throughput is unmeasured.

Afterward optimize **native fused MLA attention without transient expanded K/V**, add actual persistent prequantized model weights, long generated traces, CPU/GPU memory and fresh quality holdouts, and calibrated TTFT/TPOT/p95. Until then PR #63 remains DRAFT and automatic live promotion stays OFF.
