# beglin

A from-scratch CPU LLM inference engine for Apple Silicon — no PyTorch, no
MLX, no llama.cpp. Hand-written C using Apple's Accelerate/vDSP framework and
NEON/SME2 intrinsics, running real pretrained models (Qwen2.5-1.5B-Instruct,
Llama-3.1-8B, and MoE variants) end-to-end: RMSNorm, RoPE (incl. Llama-3
NTK scaling), GQA attention, SwiGLU, KV cache, int4/int8 quantized GEMV,
speculative decoding, and request-batched MoE serving.

📜 **License**: [AGPL-3.0-or-later](LICENSE) (open source) or a
[commercial license](COMMERCIAL-LICENSE.md) if you don't want AGPL's
copyleft/source-disclosure obligations.

## Install

```sh
npm install beglin
```

Requires macOS on Apple Silicon + the Xcode Command Line Tools (`clang`).
`postinstall` compiles the engine from the source in this package — there's
no prebuilt binary yet, see `scripts/postinstall-build.js` for why.

```js
const { runEngine, spawnEngine } = require("beglin");

// promise-based, collects output:
const { code, stdout } = await runEngine("greedy", 32, {
  env: { QWEN_BASE: "/path/to/your/exported/weights" },
});

// or get the raw ChildProcess for streaming:
const child = spawnEngine("bench", 64, { env: { QWEN_BASE: "..." } });
```

Or from the command line: `npx beglin greedy 32` (same argv the C
binary itself takes).

## What's actually novel here

**A per-role, per-expert mixed-precision MoE engine — genuinely per
individual tensor, not per model, per layer, or even per tensor
category.** For every MoE checkpoint this engine loads, each of
`q_proj`/`k_proj`/`v_proj`/`o_proj` (or MLA's
`q_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`/`o_proj`), each of the one real
dense layer's internal `gate_proj`/`up_proj`/`down_proj`, each of
shared-experts' internal `gate_proj`/`up_proj`/`down_proj`, and each
individual routed expert by `(layer, expert_id)`, all take their own
int4/int8/F32 precision independently — see
[Mixed-precision MoE configuration](#mixed-precision-moe-configuration).

This isn't a knob built for its own sake. Running OLMoE's real numeric
gate against a genuine MLX (bf16-forced-to-fp32) reference surfaced a
reproducible failure mode: int8 quantization noise in the hidden state
accumulates layer over layer until it flips a *borderline* top-k routing
decision — a real example found this round: layer 13, a router-score gap
of just 1.87e-05 between the correct expert and the one substituted in
its place. Once flipped, an entirely different expert's output stands in
for the intended one, and that perturbation amplifies through every later
layer. Selectively promoting only the four attention projections to F32
(`QWEN_MOE_ROLE_BITS`: `q_proj -1 32`, `k_proj -1 32`, `v_proj -1 32`,
`o_proj -1 32` — nothing else touched) suppressed it directly: the one
hard router mismatch this gate had disappeared entirely, and the
worst-affected position's logit-level error dropped from 4.8e-2 to
6.5e-3 — from a hard failure to comfortably inside tolerance. The
divergence remaining at a few other positions traces to the same
mechanism reached through routed-expert quantization noise instead of
attention noise — same phenomenon, same fix shape, just a different set
of tensors to target next. See `RESULTS.md` for the full investigation,
including the real per-layer hidden-state dumps that localized exactly
where the divergence originates, not just aggregate before/after numbers.

Most "hand-rolled CPU inference" projects also stop at dense-model int4
GEMV with one fixed precision for the whole model. This engine
additionally integrates ARM KleidiAI's **SME2** (Scalable Matrix
Extension v2) hardware-accelerated kernel — a real, non-trivial
integration because SME2 is only reachable inside a special "streaming
mode" on Apple M4, and naively calling into it (or letting the compiler
autovectorize into it) outside that mode is an illegal instruction, not a
compile error. See [`RESULTS.md`](RESULTS.md) for the full measured story,
including the two independent bugs that caused this and how each was
root-caused via interactive `lldb`.

## Measured results (real hardware, not projected)

| | |
|---|---|
| Correctness vs HuggingFace reference (fp32) | WikiText-103 ppl **10.648 vs 10.647**, HellaSwag **60.50 = 60.50** (200 items, 100% per-item match) |
| Quantized (GPTQ int4 + int8 lm_head, deployment default) | ppl **12.10** (+13.6% vs fp32), decode throughput lossless |
| W4A8 int8-SDOT dense decode | **~3.0×** single-stream throughput (~19.4 → ~58.6 tok/s), greedy output **bit-identical** to fp32 baseline |
| MoE batched serving, SME2 f16p-LHS path (default) | **2.38×** faster than pure-scalar baseline at B=16, accuracy **93.0%** token-match vs scalar ground truth, **zero new errors** introduced vs the scalar reference (remaining mismatches are pre-existing, documented engine-vs-reference edge cases) |
| MoE per-role precision override, real numeric gate (OLMoE) | Selectively promoting only `q/k/v/o_proj` to F32 eliminated the gate's one router hard-mismatch (**1 → 0**) and cut the worst-position logit rel-L2 from **4.8e-2 → 6.5e-3** (hard fail → inside tolerance) — a targeted fix aimed at the tensors actually responsible, not a blanket F32 upgrade |

Full methodology, the int8-LHS vs f16p-LHS root-cause story, and every raw
number: [`RESULTS.md`](RESULTS.md).

## Build

```sh
# Main engine: plain compile, no arch flags. This is load-bearing, not a
# style choice — see RESULTS.md "caller-plain convention" for why calling
# it with an SME/SVE -march flag can SIGILL on real hardware.
clang -O3 -w -c qwen_infer.c -o qwen_infer.o

# SME2 kernel wrapper (also plain -- it dispatches through function
# pointers, no SME/SVE code of its own):
clang -O2 -march=armv9.2-a+sme2 -I. -c sme2_kai.c -o sme2_kai.o

# KleidiAI vendored kernels + assembly: only these carry the arch flag.
clang -O2 -march=armv9.2-a+sme2 -I. -c \
  kleidiai/kai_common_sme_asm.S \
  kleidiai/kai_lhs_pack_f16pmrx2_f32_neon.c \
  kleidiai/kai_lhs_quant_pack_qsi8d32p_f32_neon.c \
  kleidiai/kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa.c \
  kleidiai/kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa_asm.S \
  kleidiai/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa.c \
  kleidiai/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa_asm.S \
  kleidiai/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.c \
  kleidiai/kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon.c

# Link
clang -O3 qwen_infer.o sme2_kai.o kai_*.o \
  -o qwen_infer -framework Accelerate -lpthread

# Verify no SVE/SME instruction leaked into the plain-compiled caller
# (this must always print nothing):
otool -tV qwen_infer.o | grep -iE 'sve|sme|addvl'
```

Runs on any Apple Silicon Mac; the SME2-accelerated MoE path additionally
requires an M4-or-later chip (`FEAT_SME2`) — the engine detects this at
runtime and falls back to NEON-only kernels otherwise (same numerical
output, just slower).

Weights aren't included (see `.gitignore`) — point `QWEN_ARCH_CONFIG` and
the weight-directory environment variables at your own exported/quantized
model. Export/quantization tooling isn't part of this package; open an
issue if that's something you need.

## Mixed-precision MoE configuration

This is the mechanism behind the near-tie-routing-flip fix described
above. The MoE safetensors loader (`QWEN_MOE_SAFETENSORS=<path>` --
see [Scope, honestly](#scope-honestly)) doesn't force one precision on
the whole model. Two independent, opt-in overrides let you pick
int4/int8/F32 per individual tensor, not per bundled category:

```sh
# QWEN_MOE_ROLE_BITS: every individually-registered non-expert tensor --
# q_proj/k_proj/v_proj/o_proj (or MLA's q_proj/kv_a_proj_with_mqa/
# kv_b_proj/o_proj), the dense layer's gate/up/down, shared-experts'
# gate/up/down, embed_tokens, lm_head. One "<role> <layer> <bits>" line
# per override; layer=-1 is a wildcard (every layer, or the whole tensor
# for embed_tokens/lm_head, which have no real layer index). bits is 4,
# 8, or 32 (F32 -- valid for attention roles and embed_tokens/lm_head;
# dense-layer/shared-experts internal gate/up/down stay int4/int8 only).
cat > role_bits.txt <<'EOF'
q_proj -1 32           # every layer's q_proj -> F32 (this is the real fix
k_proj -1 32           # from "What's actually novel here" above: promoting
v_proj -1 32           # just these four suppressed a router near-tie flip)
o_proj -1 32
o_proj 5 4             # ONLY layer 5's o_proj overridden back down to int4
embed_tokens -1 8      # embed_tokens -> int8 (default is F32)
EOF

# QWEN_MOE_EXPERT_BITS: routed experts specifically, individually, by
# (layer, expert_id) rather than by role -- "<layer> <expert_id>" per
# promoted-to-int8 line; every other expert defaults to int4. Typically
# generated from real router traffic (see moe_st_expert_profiler.py in
# RESULTS.md's per-expert mixed-precision writeup), not hand-written.
cat > expert_bits.txt <<'EOF'
11 52
11 13
EOF

QWEN_MOE_SAFETENSORS=/path/to/model.safetensors.index.json \
QWEN_MOE_ROLE_BITS=role_bits.txt \
QWEN_MOE_EXPERT_BITS=expert_bits.txt \
./qwen_infer
```

Both are unset by default (every tensor keeps its shipped default -- int8
for attention/dense/shared-experts, F32 for embed_tokens/lm_head, int8 for
routed experts), and confirmed byte-identical to the un-configured binary
when left unset. See `RESULTS.md`'s "Full per-role precision engine" and
"Per-individual-expert mixed precision" entries for the full design
rationale, the two hybrid attempts that motivated per-individual rather
than per-category granularity, and `PLAN_general_purpose_loader.md`'s
`D-gen-6` for why this is a plain text format rather than JSON/YAML.

## The open question this engine exists to make experimentable

Every fix documented above works the same way: promote a specific tensor's
precision until a specific divergence closes. That points at a bigger,
unanswered question this repo deliberately does not try to answer itself:
for a given architecture, is there a *minimal* set of tensors -- which
`(layer, expert_id)` pairs, which attention projections, which layers'
dense/shared-expert internals -- that has to stay above int4/int8 to keep
the model's output within tolerance, with everything else free to drop as
low as the hardware budget demands? Ship-of-Theseus terms fit: how much of
a model can be replaced with a cheaper part before its numerical identity
changes, and what's the smallest set of parts that has to stay original
for that not to happen?

That's a real combinatorial search, not a one-shot measurement, and it's
architecture-specific -- not a single answer to transplant across models.
For OLMoE alone, "which experts can drop precision" is a search over
`C(64, k)` candidate subsets *per layer* (16 layers), crossed with however
many precision tiers are on the table (int4/int8/F32); brute force isn't
viable for any real model, and this round already found the "obvious"
heuristic doesn't transfer cleanly -- OLMoE's top-8-by-router-frequency
subset alone reproduced almost none of what combining it with attention
F32 achieved (see `RESULTS.md`'s `D-expert-promo-1`), so frequency-ranked
selection isn't sufficient on its own, and a different expert count,
sparsity pattern, or hidden size would need its own search from scratch,
not this one's answer copy-pasted in.

Running that search -- heuristic-guided, most likely by an agent trying
candidate assignments and scoring each one, since the space is too large
to hand-enumerate -- is future work, and deliberately not this repo's job.
What this repo *is*: the substrate that makes each candidate assignment
actually testable in an afternoon instead of a rewrite -- independently
addressable precision per attention projection, per dense-layer internal,
per shared-expert internal, and per individual `(layer, expert_id)`, plus
a real numeric-gate harness (teacher-forced comparison against a genuine
MLX reference, the same protocol `compare_moe_st_olmoe.py` already runs)
to score any candidate honestly rather than by intuition. Whoever runs
that search next -- another researcher, another LLM agent, not
necessarily this project -- inherits a place to plug a `(role/tensor,
bits)` assignment in and get a real pass/fail number back, not a rewrite
of the loader first.

## Scope, honestly

This is **not** a general-purpose loader like llama.cpp yet — it's
currently validated for a small set of specific models (Qwen2.5-1.5B,
Llama-3.1-8B, one MoE model), with a custom weight format, not GGUF. See
[`ROADMAP.md`](ROADMAP.md) for the plan to change that.

## Repository contents

```
qwen_infer.c              # the engine: single translation unit, plain-compiled
sme2_kai.h/.c              # SME2 dispatch wrapper (int8-LHS + f16p-LHS paths, runtime HW gate)
q4gemv.h                  # NEON int4/int8 dequant-GEMV + threaded batch GEMM
q4gemv_g256.h             # alternate group-256 kernel variant
attn_neon.h                # hand-written NEON GQA attention kernels
rope_llama3_scale.h        # Llama-3 RoPE NTK scaling
kleidiai/                  # vendored ARM KleidiAI kernels (Apache-2.0, SPDX headers preserved)
                            # + this project's correctness harnesses and repack derivation notes
RESULTS.md                 # full measured results, methodology, and the SIGILL root-cause story
LICENSE                    # AGPL-3.0-or-later
COMMERCIAL-LICENSE.md      # dual-licensing offer + contact
COMMERCIAL-LICENSE-AGREEMENT.md  # the actual commercial contract text (template, per-licensee terms TBD)
```

## Provenance

`kleidiai/` vendors ARM KleidiAI (upstream commit
`6787251d9cc2f38a3a6024b11fd7ace10cde4cd9`, Apache-2.0). Everything else is
original work.
