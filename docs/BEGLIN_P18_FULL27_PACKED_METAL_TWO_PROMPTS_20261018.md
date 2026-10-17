# BEGLIN P18 — 27-layer real DeepSeek-V2-Lite packed Metal logits (2026-10-18)

**Scope: engine research, NOT a release qualification.** Source weights came from an isolated D50 **research copy**, not an attested original-XOX issuer. No production serving, routing configuration, engine release plane, or infrastructure was modified.

## Completed model execution

Source-defined DeepSeek-V2-Lite attention, MoE router, and decoder layers were executed for **all 27 layers**, with the original tokenizer and model embedding. BEGLIN native C qNg64 encoder produced true packed uint8 bitplanes plus one F32 group scale per 64 weights. A research **Apple MLX Metal custom bitplane kernel** executed every eligible linear projection actually needed by the model's MoE selections. Original BF16 embedding, MoE Gate, norms, and final `lm_head` remained BF16.

The frozen global manifest contained **5,181 eligible rank-2 qNg64 matrices** across 27 layers; **110 tensors remain BF16**. Matrices in the manifest are the model-wide storage budget; **only currently routed experts are actually loaded/processed per prompt**, so the number of executed matrices is smaller than 5,181 and varies between prompts.

Independent original safetensors shard header/index audit checked the format, dimensions, group64 alignment and exact per-matrix code+scale bytes for all 5,181 entries, plus all BF16 exclusions. Four mutation tests (width, packed byte count, tensor shape, repeated tensor ID) failed closed.

| Whole-source weight payload | Bytes |
|---|---:|
| Original BF16 checkpoint | 31,412,968,448 |
| Uniform n7 (5181 qNg64 + 110 BF16 exclusions) | 15,174,228,992 |
| Global mixed n6/n7/n9, same eligible matrices | **15,173,819,392** |

Calculated model-wide serialized weight saving for global mixed vs BF16: **51.6957%**. Mixed saves just **409,600 bytes (~0.0027%)** over uniform n7 across the **entire model**. These are **not** live GPU allocator peak or KV-cache values.

## Model-wide logits and NLL: two short authored prompts

Each prompt was passed through the entire 27-layer DeepSeek model independently in BF16, uniform qNg64 n7 Metal and frozen `global_mixed` Metal. All **102,400 vocabulary logits per position** were compared with float64 normalized distribution calculations. The global mixed policy uses an **experimental direct bitplane n6 custom Metal path**, NOT the existing C++ production n6 repacked path.

| Statistic | BF16 | Uniform n7 | Global mixed |
|---|---:|---:|---:|
| Prompt 0 (5 positions): self-NLL (nats) | 5.29747 | 5.25755 | **5.14487** |
| Prompt 1 (7 positions): self-NLL (nats) | 7.59231 | 7.64463 | **7.57869** |
| **10 predicted next-token labels**, mean self-NLL | 6.67437 | 6.68980 | **6.60516** |
| Across 12 positions, Top-1 agrees with BF16 | 12/12 | 12/12 | 12/12 |
| All-position mean BF16-to-quantized vocab KL (nats) | 0 | **0.00648** | 0.01430 |

**Critical negative finding:** Global mixed showed lower self-NLL on these exact two sentences, but **worse BF16 full-distribution fidelity** (KL more than 2× uniform n7). This illustrates why lower self-NLL in a tiny trial must never be called a general accuracy improvement. No PPL/accuracy benchmark or 480-item evaluation was run.

The first experiment executed 2,064 packed GPU linears in uniform n7 and 2,067 in global mixed; the second independent prompt executed 2,568 global-mixed linears. This is **genuine full-model logits on GPU-packed eligible weights**, but not a production serving build or an autoregressive decode-rate measurement.

A localized policy was also evaluated on Prompt 0. It preserved the same model-wide weight byte budget as global mixed but had Top-1 mismatch at one of five positions, showing that local sensitivity improvements can still change final predictions. Localized Prompt 1 was not executed; it must NOT be filled in or extrapolated.

## Evidence (SHA-256 of append-only EOE research receipts)

- Original 27-layer BF16 prompt0: `3de1973704e0364ebf206ce51e54ed48d2a5906bea735df05b0fb21b661051fc`
- Frozen full-model quantization manifest: `39b98921c30a992f93ea896283336a6b5fe1810c5737443132f2410b474e9612`
- Full 5,181 original shape/format/bytes audit: `53264b124af7359d2cb84c188701701db33c2876b09ce751b5d0b1c25770eac9`
- Real global mixed packed Metal prompt0: `1cd5f1994908b617ef1731d428aae8937847b341cda85b8040353c083a6ead60`
- Real global mixed packed Metal prompt1: `fc3422d80f318bb82d8e13d1578c005e1f6b4eab3df2dac191d28a26684ba3cc`
- BF16/global mixed vocab-logit paired prompt0: `a29069842f8c62cdfb7409bb783ad73c00abaaea8bfc6434d027196a33688fc7`
- BF16/global mixed vocab-logit paired prompt1: `35fd17f84fe4fa3f702fb35d219e74f7de564a0de4e4258f58414ea8e777c5f5`
- Joint source-hash/consistency audit: `311bfac490f7669fd90c4536799fc9886dbe0cff1f9db86c66964a161a96a22f`

No model weights, logit arrays, tokens, hidden states or private machine paths are included in this public report.

## Development verdict

**PASS research numerical execution** for all 27 decoder layers under BF16/uniform/global-mixed eligible-projection modes, using real MLX packed bitplane operations and final vocabulary logits.

**NOT READY** for production engine merge/activation: official original XOX source, current running C++ binary parity, full NLL/task dataset with independent holdouts, true autoregressive prefill/decode tokens/s, KV/weight epoch transitions and GPU peak memory remain unverified. In particular, no n6 inference from the research custom Metal specialization may be attributed to the C++ live dispatcher.

This remains a **Draft research PR**, no merge and `automatic_live_promotion=false`.
