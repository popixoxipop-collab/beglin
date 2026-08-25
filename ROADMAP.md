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
incremental patch. **Correction (2026-08-25, after a code-reading planning
pass — see [`PLAN_general_purpose_loader.md`](PLAN_general_purpose_loader.md)):**
the paragraph below originally claimed the SME2 kernels are shape-fit to
the 3 validated models. That's wrong. `sme2_kai.c`'s shape gate
(`kai_sme2_shape_ok`) only constrains `in % 64 == 0`; `out` is
unconstrained in both the wrapper and the vendored KleidiAI kernel
(arbitrary N is handled by pad-at-pack). Every N this engine has ever
actually passed happens to be a multiple of 64 — so the real situation is
**"probably already shape-generic, but the `out`-arbitrary path has never
been exercised,"** not "hand-fit, needs a dispatch table." The dense
engine (`load_arch_cfg`/`alloc_arch_buffers`) is already metadata-driven;
it's specifically the **MoE path** that's dimension-hardcoded to
DeepSeek-V2-Lite (stack arrays sized to its exact dims, already caused one
real stack smash). See the plan doc for the full corrected picture and a
day-one experiment (a shape sweep on real M4 hardware) that resolves the
`out`-genericity unknown before anything else is built on top of it.
- A metadata-driven loader (GGUF or an equivalent) has to replace the
  current `arch_config.txt`-per-model, hand-written-per-architecture
  parsing in `qwen_infer.c`.
- The MoE path's DeepSeek-specific stack arrays need converting to
  heap-allocated, `g_cfg`-style sizing before a second MoE topology can
  be supported — this turns out to be the largest single chunk of the
  work (see plan Phase 4), not the SME2 kernels themselves.
- The export/quantization pipeline needs generalizing to ingest an
  arbitrary HF checkpoint — turns out to be a smaller job than expected;
  the existing scripts are already closer to architecture-generic than
  this file originally assumed (3 hardcoded name literals in one script,
  not a rewrite).

**EXIT**: if the `out`-genericity experiment (plan Phase 0) fails even
with padding, beglin stays a narrow, high-confidence engine for a curated
model list — which is still a legitimate, honest product; it just doesn't
compete with llama.cpp on scope, only on measured CPU throughput for the
models it does support.

## Implementation plan

A phased implementation plan exists:
[`PLAN_general_purpose_loader.md`](PLAN_general_purpose_loader.md)
(produced by a planning pass that read this repo's actual code, the
existing export scripts, and bob's dev environment — not written from
memory). It resolves the open questions below with real positions, not
just restating them.

## Two work streams (see the plan doc for the resolved detail)

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
