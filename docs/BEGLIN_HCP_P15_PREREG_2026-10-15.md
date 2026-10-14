# Beglin HCP P15 — Full original BF16 teacher by bounded layer streaming

Date: 2026-10-15 KST. Parent P14 commit a73d84d45f036e120fd74229f0d93f82e7d36226.
Research branch: research/hcp-p15-streamed-full-teacher-20261015.

## Preregistered scope

Run the COMPLETE original DeepSeek-V2-Lite decoder (27 layers, 5,291 checkpoint tensors, vocabulary 102,400) on XOX CPU. At most ONE layer's weights resident at once. No full-model materialization, quantization, attention approximation, KV truncation, GPU work, production worker stop, or live release modification.

Checkpoint index SHA256 d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf.
Configuration SHA256 f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d.
Original modeling_deepseek.py SHA256 7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c.

Only 18 explicit original torch functions/classes are extracted by AST without changing their bodies. This avoids deprecated Transformers wrapper imports while preserving original attention scaling (including YaRN mscale_all_dim), rotary permutation, expert routing and MoE computation. The source is validated BEFORE compilation. A meta decoder layer is assigned exactly the original parameter names and shapes, computes in evaluation mode, then is released. Pure positional buffers are reinitialized by the ORIGINAL method from the unchanged config.

Two explicit numerical modes: original BF16 weights with BF16 CPU arithmetic; the same weights widened exactly to FP32 with FP32 CPU arithmetic. These are not asserted bit-identical. Neither is the native quantized Beglin engine; this is an independent reference path.

## Gate 1 — numerical and loader validation

A small 3-layer model made with the original functions is materialized in both resident and streamed forms. Require EXACT full-vocabulary logit equality within EACH arithmetic dtype, including dense layer, routed/shared experts, longer and one-token inputs. Test malformed source/hash/shape, missing tensors, out-of-range tokens, unsupported batching, and RoPE mscale. This validates streaming mechanics, not full-model task accuracy.

## Gate 2 — real full-vocabulary teacher output

Freeze tokenization before inference for these engineering probes:
1. `The result of seven times eight is`
2. `Complete the sentence: The capital of France is`

Run one unpadded prompt at a time, <=128 tokens, no KV cache, no generation fallback. Require all 27 layer outputs finite, all 5,291 source tensors consumed exactly once, complete final norm+LM head, and a finite 102,400-element next-token logit vector. Save full logits ONLY in research artifact directories, with SHA and per-layer/value provenance. Inference probes are NOT a statistically independent quality benchmark.

Hard watchdog: own process RSS <=6 GiB, XOX free disk >=20GB, stop on material new swap pressure or insufficient available memory. Two CPU threads, low priority, source model read-only, <220MB new disk use. Watchdog terminates only this run-owned process and records failure. No service shutdown or cache deletion.

## Acceptance and claims

A real full teacher forward is marked complete only after the actual 27-layer loop and final full-vocabulary logits exist and independently validate. This is distinct from P14 header-only feasibility or P12/P13 local-matrix checks. Success does NOT by itself certify native quantized A/B noninferiority, perplexity, multi-token output quality, throughput, or operational promotion. Production promotion stays OFF.

Sources consulted: official Hugging Face safetensors usage (safe_open/get_tensor) and Accelerate big-model-inference documentation (load/use/release by layer). Existing model computation comes from the exact local checkpoint source, not an unpinned downloaded implementation.
