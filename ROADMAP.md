# Roadmap

## Where beglin is today (v0.1.0)

Despite the SME2/NEON kernel work being genuinely hardware-generic, **beglin
is currently a purpose-built engine for a small, validated set of models** —
not a general-purpose loader:

- Validated architectures: Qwen2.5-1.5B-Instruct, Llama-3.1-8B, and one MoE
  model (DeepSeek-V2-Lite-class, MLA attention + routed experts).
- Weights are **not** GGUF. They're a custom binary format
  (`weights/*.bin` + `manifest.json` + `arch_config.txt`) specific to this
  engine's memory layout.
- The scripts that produce that format (`eval/quantize_int4.py`,
  `eval/gptq_quantize.py`, the GGUF-adjacent conversion tooling) are
  **not part of this package** — they're architecture-specific and live in
  the internal dev tree, hand-fit to the 3 validated models.

This is the honest gap vs. tools like llama.cpp: point llama.cpp at almost
any GGUF file and it runs. Point beglin at an arbitrary model today and it
doesn't — there's no metadata-driven architecture discovery, no generic
tensor-name mapping, no public conversion path.

## D-roadmap-1: pursue general-purpose model loading (GGUF-class)

**WHY**: beglin's actual differentiator — a correct, measured SME2
integration with a documented root-cause story for where a naive
integration goes wrong (int8-LHS activation quantization) — is orthogonal
to "which specific model." Right now that differentiator is locked behind
a narrow, hand-maintained model list. Generalizing the loader is what
turns this from a research artifact into something people can actually
adopt the way they adopt llama.cpp, without needing us to hand-port every
new model.

**COST**: this is large, multi-session engineering work, not an
incremental patch:
- A metadata-driven loader (GGUF or an equivalent) has to replace the
  current `arch_config.txt`-per-model, hand-written-per-architecture
  parsing in `qwen_infer.c`.
- The SME2 kernels are currently fit to the exact tensor shapes of the 3
  validated models. Genuinely arbitrary shapes need either (a) a
  dispatch table of shape-specialized fast paths selected at load time,
  or (b) a correct generic fallback (NEON/scalar) for unrecognized shapes
  with SME2 reserved for known-good ones. Which of these — and how much
  of the current hand-written-kernel identity survives it — is an open
  design question, not decided here.
- The export/quantization pipeline needs the same generalization: ingest
  an arbitrary HF checkpoint, not just the 3 models it currently knows
  about.

**EXIT**: if this doesn't pan out (e.g. the shape-generalization cost is
too high relative to adoption benefit), beglin stays a narrow,
high-confidence engine for a curated model list — which is still a
legitimate, honest product; it just doesn't compete with llama.cpp on
scope, only on measured CPU throughput for the models it does support.

## Two work streams (not yet planned in detail)

1. **Generic model loading**: GGUF or equivalent. Open question: write a
   minimal from-scratch GGUF parser, or vendor just the format-parsing
   portion of llama.cpp's `ggml`/`gguf.c` (MIT-licensed) without pulling
   in its compute graph — keeping beglin's own hand-written kernels as the
   actual compute path rather than becoming a ggml wrapper.
2. **Generic quantization/export pipeline**: generalize
   `eval/quantize_int4.py` / `gptq_quantize.py` from "hardcoded tensor
   names for 3 known models" into something that discovers tensor
   structure from an arbitrary HF checkpoint's config, the way llama.cpp's
   `convert_hf_to_gguf.py` does per-architecture conversion classes.

## Explicitly not decided yet

- Priority order for architectures beyond the current 3 (pick by
  tensor-layout similarity to minimize integration cost per model, not
  by popularity alone).
- Whether Homebrew distribution makes sense before or after this work —
  current read: **after**, since brew's value is "point at any model and
  go," which isn't true yet (see current README).
- No implementation should start on this without a dedicated planning
  pass (this is 3+ file, architecture-defining work) — this file records
  intent and constraints, not a plan.
