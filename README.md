# beglin

A from-scratch LLM inference engine for Apple Silicon — no PyTorch, no
llama.cpp. Hand-written C using Apple's Accelerate/vDSP framework and
NEON/SME2 intrinsics for the CPU path (zero external ML dependency there),
plus an optional GPU backend built on MLX, running real pretrained models
(Qwen2.5-1.5B-Instruct, Llama-3.1-8B, and MoE variants) end-to-end:
RMSNorm, RoPE (incl. Llama-3 NTK scaling), GQA/MLA attention, SwiGLU, KV
cache, arbitrary-bit-width quantized GEMV, speculative decoding, and
ragged continuous-batched MoE serving.

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

Two things, and most "hand-rolled CPU inference" projects have neither:
**precision that adjusts per individual tensor, all the way down to
arbitrary bit-widths**, and **a serving engine that adapts to its
hardware and its load instead of running one fixed code path**.

### Adjustable precision, down to a single tensor and an arbitrary bit-width

Most quantized inference engines pick one precision for the whole model,
or at best one per layer. This one goes further in two independent
directions:

**Per-tensor granularity.** For every MoE checkpoint this engine loads,
each of `q_proj`/`k_proj`/`v_proj`/`o_proj` (or MLA's
`q_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`/`o_proj`), each of the one real
dense layer's internal `gate_proj`/`up_proj`/`down_proj`, each of
shared-experts' internal `gate_proj`/`up_proj`/`down_proj`, and each
individual routed expert by `(layer, expert_id)`, all take their own
precision independently — see
[Mixed-precision MoE configuration](#mixed-precision-moe-configuration).

**Arbitrary bit-width, not just int4/int8/F32.** Beyond the three fixed
tiers, a real symmetric, error-feedback bit-plane format (`qNg64`) covers
every bit-width from 2 to 15 as a genuine, independently-verified hardware
encoding — not four tiers with gaps in between. n=8 here is its own
distinct encoding, not a rename of the existing int8 format (different
clamp, different rounding, verified numerically different on purpose).
Confirmed with a real round-trip test across the full n=2..15 range
(monotonically decreasing reconstruction error, no discontinuity at any
bit-width boundary) and a real end-to-end promotion + generation run
against production DeepSeek-V2-Lite weights. See `RESULTS.md`'s
`D-qNg64-18` for the full verification.

This range is CPU-only by default because MLX's own native GPU kernels
only cover bits in {2,3,4,5,6,8} — but the full n=7,9-15 range now has
a real GPU path too, through a custom Metal kernel
(`mx.fast.metal_kernel`) that decodes the same compressed bit-plane
bytes directly on GPU, not a dequantize-to-dense fallback. Two call
shapes, both real, both on the same compressed representation:

- **Single-expert (attention roles).** Verified with a real end-to-end
  GPU generation on production DeepSeek-V2-Lite weights **at every one
  of n=7,9,10,11,12,13,14,15**, each producing output identical to the
  CPU path at the same prompt. See `RESULTS.md`'s `D-metal-4` and
  `D-metal-5`.
- **Routed MoE FFN (per-token top-K expert gather).** A second custom
  kernel extends the same decode logic to a real `E`-expert gather —
  `gate_proj`/`up_proj`/`down_proj`, real E=64, dispatched through the
  same `gather_qmm`-equivalent hot path the native bit-widths already
  use. Verified with synthetic multi-expert data at the kernel level
  (every (pair, expert) coordinate independently re-decoded, not just
  pair 0), then against real DeepSeek-V2-Lite weights: an exact-match
  decode spot-check across experts spanning the full E=64 range, and
  real end-to-end generation **at n=7,9,10,11,12,13** (token-identical
  across all six). n=14 and n=15 hit real, repeated memory contention
  from a concurrently-running benchmark on the same 16GB test machine
  — not a correctness failure, not yet retried under isolated
  conditions. See `RESULTS.md`'s `D-metal-7-1` through `D-metal-7-3`.

Both paths are correctness-verified, not yet performance-measured.

**Why this granularity exists, not just because it's possible.** Running
OLMoE's real numeric gate against a genuine MLX (bf16-forced-to-fp32)
reference surfaced a reproducible failure mode: int8 quantization noise
in the hidden state accumulates layer over layer until it flips a
*borderline* top-k routing decision — a real example found this round:
layer 13, a router-score gap of just 1.87e-05 between the correct expert
and the one substituted in its place. Once flipped, an entirely different
expert's output stands in for the intended one, and that perturbation
amplifies through every later layer. Selectively promoting only the four
attention projections to F32 (`QWEN_MOE_ROLE_BITS`: `q_proj -1 32`,
`k_proj -1 32`, `v_proj -1 32`, `o_proj -1 32` — nothing else touched)
suppressed it directly: the one hard router mismatch this gate had
disappeared entirely, and the worst-affected position's logit-level error
dropped from 4.8e-2 to 6.5e-3 — from a hard failure to comfortably inside
tolerance. See `RESULTS.md` for the full investigation, including the
real per-layer hidden-state dumps that localized exactly where the
divergence originates, not just aggregate before/after numbers.

One thing this granularity does *not* make safe to assume: that accuracy
degrades smoothly as bit-width drops. A real sweep across the full n=2..16
range, one tensor at a time, against real near-tie routing/attention
decisions, found 13-83% of tested targets **non-monotonic** — passing at
n=3, failing at n=4, passing again at n=5+, depending on the tensor,
corpus, and even which hardware executed the quantization. Bisection-style
search (test n=8, pass → try n=4) is not a safe shortcut here; exhaustive
per-n testing is the only default this project has found safe so far. See
[The open question this engine exists to make experimentable](#the-open-question-this-engine-exists-to-make-experimentable).

### A serving engine that adapts, not one fixed code path

**Adapts to the chip it's running on.** SME2 (Scalable Matrix Extension
v2) is a real, non-trivial hardware integration — it's only reachable
inside a special "streaming mode" on Apple M4+, and naively calling into
it (or letting the compiler autovectorize into it) outside that mode is
an illegal instruction, not a compile error. The engine checks
`hw.optional.arm.FEAT_SME2` at runtime and dispatches to the accelerated
KleidiAI kernel when present, falling back to plain NEON otherwise — same
binary, same numerical output, no recompile needed to run on older
Apple Silicon. See [`RESULTS.md`](RESULTS.md) for the full measured
story, including the two independent bugs that caused illegal
instructions and how each was root-caused via interactive `lldb`.

**Adapts to available compute — CPU or GPU, same weights.** Beyond the
CPU/SME2 path, this engine has its own GPU backend through MLX, reaching
**52.91 tok/s** on real DeepSeek-V2-Lite generation — 109% of
llama.cpp+Metal's own bar on the same hardware. See
[vs llama.cpp / MLX](#vs-llamacpp--mlx-same-hardware-same-model) for the
full, honest comparison — including where the CPU path is genuinely
*not* yet competitive, reported plainly rather than only showing the
favorable number.

**Adapts to concurrent load.** The serving loop is ragged continuous
batching, not a fixed-size batch loop: requests are admitted into free
slots and evicted on completion every step, not queued until a batch
fills — the same pattern production LLM servers (vLLM and similar) use,
not the toy fixed-batch loop most from-scratch engines stop at.

**Adapts precision live, without a restart.** A running server can pick
up a new precision decision for a specific `(role, layer)` mid-session —
polled once per request admission from a promotion file, applied as a
single pointer swap, permanent for the process's lifetime with no
restart required. The engine's job here is deliberately scoped to
*applying* a promotion it's told about, not deciding when one is needed —
that decision comes from this project's own attribution tooling (which
tensors are actually causing near-tie routing flips, measured against a
real reference), external and swappable, not baked into the serving hot
path. See `RESULTS.md`'s `D-roadmap-4` Phase 6 for the full design.

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

## vs llama.cpp / MLX (same hardware, same model)

Apple M4 (4P+6E cores, 16GB RAM), DeepSeek-V2-Lite (15.71B, MoE), ~4-bit
quantization on every path (llama.cpp `Q4_K_M` 9.65GiB, this engine's own
`q4g64` int4 AF-blob ~9.8GB). `tok/s` is generation-phase throughput
(excludes prompt processing).

| Path | tok/s |
|---|---|
| llama.cpp, Metal GPU | **53.11 ± 1.01** |
| **This engine, GPU (own MLX backend)** | **52.91** (109% of llama.cpp+Metal's original 48.34 bar) |
| llama.cpp, CPU-only (8 threads) | 22.69 ± 6.32 |
| This engine, CPU/SME2, steady-state (warm) | **~5.65** |
| This engine, CPU/SME2, cold start (first request) | ~1.34 |

**A server isn't perpetually cold, so "real serving" means the warm
number.** SME2 repacks each `(layer, expert, projection)` weight slot the
first time it's touched, then reuses the packed form for the rest of the
process's life. The first request through a freshly-started process pays
that setup cost and measures ~1.34 tok/s; a second request whose routing
overlaps already-warmed experts (measured directly, per-step timing
instrumentation, not estimated) measures **~5.65 tok/s** — a real 4.2×
difference between "cold" and "warm," not noise. This is this engine's
own known limitation, not a new one — it's the exact gap that motivated
building the MLX GPU backend in the first place, which is where the
competitive GPU number above comes from.

**Does batching close the gap? Swept it — no.**

| Batch size | This engine, CPU/SME2 (warm) | llama.cpp, CPU | ratio |
|---|---|---|---|
| 1 | 5.80 | 16.88 | 2.91x |
| 2 | 7.17 | 20.74 | 2.89x |
| 4 | 8.84 | 32.68 | 3.70x |
| 8 | 10.20 | 36.68 | 3.60x |
| 16 | 11.10 | 46.69 | 4.21x |

Both engines' CPU throughput grows with batch size — a generic
memory-bandwidth-amortization effect, not unique to either implementation
— but the *ratio* between them holds roughly flat around 3-4x and if
anything drifts slightly wider, not narrower, as batch size grows. No
crossover point was found in this range; the honest conclusion is that
this sweep found none, not that none exists at some larger, unmeasured
batch size. (B=32 was attempted but came back inconclusive for a real,
specific reason — the warm/cold measurement technique itself breaks down
at that scale — see `RESULTS.md` "D-bench-3" for exactly why, rather than
reporting a number that would have been an artifact.)

Warm CPU/SME2 is still ~3-4x slower than llama.cpp's CPU path and ~9.4x
slower than either GPU path — a real, substantial gap that batching does
not close, at least not by B=16. A real B=64 re-measurement (this
engine's actual target serving scale) remains open, memory-constrained on
the only SME2-capable test machine available this round. Full
methodology, exact commands, and the honest cost/scope notes:
[`RESULTS.md`](RESULTS.md) ("D-bench-1", "D-bench-2", "D-bench-3").

## Build

```sh
# Main engine: plain compile, no arch flags. This is load-bearing, not a
# style choice — see RESULTS.md "caller-plain convention" for why calling
# it with an SME/SVE -march flag can SIGILL on real hardware.
clang -O3 -w -c qwen_infer.c -o qwen_infer.o

# GGUF/safetensors loader sources (also plain -- qwen_infer.c calls into
# these for GGUF transcoding and safetensors checkpoint loading; the
# built-in `postinstall` build compiles these too, see
# scripts/postinstall-build.js):
clang -O3 -w -c gguf_cache.c -o gguf_cache.o
clang -O3 -w -c gguf_load.c -o gguf_load.o
clang -O3 -w -c gguf_quants.c -o gguf_quants.o
clang -O3 -w -c gguf_transcode.c -o gguf_transcode.o
clang -O3 -w -c hf_config.c -o hf_config.o
clang -O3 -w -c safetensors_load.c -o safetensors_load.o
clang -O3 -w -c safetensors_quants.c -o safetensors_quants.o

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
clang -O3 qwen_infer.o gguf_cache.o gguf_load.o gguf_quants.o gguf_transcode.o \
  hf_config.o safetensors_load.o safetensors_quants.o sme2_kai.o kai_*.o \
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

For the arbitrary-bit-width case (n=2,3,5,6,7,8..15 -- see [Adjustable
precision](#whats-actually-novel-here) above), the same "<role> <layer>"
line format takes a third column instead of a fixed bits value:

```sh
cat > promotion_nq.txt <<'EOF'
shared_gate_proj 5 10   # promote to qNg64(n=10)
kv_b_proj 9 13          # promote to qNg64(n=13)
EOF

QWEN_MOE_PROMOTION_FILE_NQ=promotion_nq.txt ./qwen_infer
```

Polled once at startup from a real, already-open checkpoint (requires
`QWEN_MOE_NEARTIE_CORRECT=1` + `QWEN_MOE_NEARTIE_CORRECT_SAFETENSORS=<path>`
to have opened one) -- see `RESULTS.md`'s `D-qNg64-3`/`D-qNg64-18` for the
full mechanism and `D-roadmap-4` Phase 6 for the separate, live (no
restart needed) `QWEN_MOE_PROMOTION_FILE` promotion path this one shares
its polling convention with.

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

One shortcut that would make that search cheap turns out not to be safe:
bisecting on bit-width (test n=8, pass -> try n=4; fail -> try n=16) only
works if accuracy is monotonic in n, and it usually isn't. A real
arbitrary-bit-width sweep (n=2..16, one tensor at a time, against real
near-tie routing/attention decisions on DeepSeek-V2-Lite) found 13-83%
of tested targets non-monotonic depending on the sample -- passing at
n=3, failing at n=4, passing again at n=5+, with no single factor
(tensor identity, corpus, specific event, or even the *quantization
algorithm itself*, confirmed by comparing a CPU RTN simulation against
real native hardware quantization on GPU) explaining which targets will
misbehave. Exhaustive per-n testing is the only safe default found so
far; see `RESULTS.md`'s ROI-G Phase 1/2 sections for the full search
tool (`tools/quant_search_n.py`) and every real data point behind this.

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

## Research notes not in RESULTS.md

Some investigations end in a finding worth keeping but don't belong in the
measured-results narrative (negative results, or results that don't ship as
code). These live in `.claude/memory/` instead — see
[`.claude/memory/MEMORY.md`](.claude/memory/MEMORY.md) for the index. Notably:
[`turboquant_lut_router_sensitivity.md`](.claude/memory/turboquant_lut_router_sensitivity.md) —
a TurboQuant-style non-uniform codebook improves q4g64 weight-reconstruction
error by 8-12% at zero extra storage (confirmed real, generalizes across all
27 layers), but does **not** safely improve real forward-pass accuracy for
this MoE architecture: even a numerically-better per-weight perturbation is
enough to flip close top-k router ties, and one flip cascades through the
rest of the layer stack and the autoregressive sequence. Weight
reconstruction error alone is not a sufficient proxy for MoE forward-pass
fidelity -- any future quantization-scheme change needs the same real
forward-pass gate this repo already uses for precision-tier promotion, not
just an offline reconstruction-error number.

## Scope, honestly

**Real GGUF loading exists and works end-to-end** (`QWEN_GGUF=<path>`),
token-exact-verified against upstream llama.cpp/MLX on real multi-GB
checkpoints — but it is **not yet a general-purpose loader like
llama.cpp**, in two specific, honest ways:

- **Architecture coverage is an allowlist, not open discovery.**
  `qwen2` and `llama` (dense) plus `qwen3moe` (MoE) are recognized;
  anything else FATALs rather than guessing. This project's own two
  flagship MoE architectures (DeepSeek-V2-Lite/MLA, OLMoE) are validated
  through the separate safetensors loader (`QWEN_MOE_SAFETENSORS`), not
  GGUF — "loads a beglin-supported checkpoint" is currently broader than
  "loads an arbitrary GGUF file." Gemma, Phi-3, and the Mamba/Jamba/RWKV
  family are deliberately out of scope so far (each needs real numeric
  work this project hasn't done yet — softcapping, LongRoPE, or no
  attention path at all).
- **Quantization coverage is real but partial**: F32/F16/BF16, Q4_0/
  Q5_0/Q8_0, and Q3_K/Q4_K/Q5_K/Q6_K dequantize correctly (Q3_K/Q5_K
  added this round — see `RESULTS.md`'s `D-gen-9`). Q2_K and the
  IQ-series (lattice/codebook quantization, not simple affine
  scale+min) are not yet supported and FATAL on load.

One more real gap, not a quantization or architecture one: there is no
real tokenizer in this engine. `QWEN_GGUF`/`QWEN_MOE_SAFETENSORS` load
pre-tokenized raw int32 files only — no BPE, no `tokenizer.ggml.*`
metadata consumption. See [`ROADMAP.md`](ROADMAP.md) and
`PLAN_general_purpose_loader.md` for what's planned next on each axis.

## Repository contents

```
qwen_infer.c              # the engine: single translation unit, plain-compiled
gguf_load.h/.c            # GGUF container parser
gguf_quants.h/.c          # GGUF quantization format dequant kernels
gguf_transcode.h/.c        # GGUF -> engine's own K_Q4G64/K_Q8G64/qNg64 transcode
gguf_cache.h/.c           # on-disk transcode cache (.beglin), lazy-repack
hf_config.h/.c             # HuggingFace config.json / arch_config_moe.txt parsing
safetensors_load.h/.c      # safetensors container parser + checkpoint loader
safetensors_quants.h/.c    # safetensors-side quantization helpers
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
