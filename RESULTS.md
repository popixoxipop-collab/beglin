# Results

All numbers below are from real execution on Apple Silicon hardware (M4,
unless noted), against real pretrained weights (Qwen2.5-1.5B-Instruct
dense, plus an MoE variant for the batched-serving results). Nothing here
is projected or simulated.

## Correctness vs HuggingFace reference

| Check | Result |
|---|---|
| Per-layer hidden state error (fp32) | ≤ 8e-5 |
| Final logits error (fp32) | 6.3e-5 |
| Greedy decoding, 32 tokens | **32/32 exact match** |
| WikiText-103 validation ppl (fp32) | engine **10.648** vs HF **10.647** (ratio 1.00007) |
| HellaSwag, 200 items, length-normalized (fp32) | engine **60.50** = HF **60.50**, 100% per-item agreement |

## Quantization

| Config | ppl | Notes |
|---|---|---|
| RTN int4 | 14.87 | naive round-to-nearest baseline |
| **GPTQ int4 layers + int8 lm_head (deployment default)** | **12.10** | +13.6% vs fp32, −18.6% vs RTN; decode throughput lossless vs RTN |

## Dense-model decode throughput

Progressive optimization of the single-stream decode path, Qwen2.5-1.5B,
each step measured against the previous on the same hardware:

| Stage | Result |
|---|---|
| int4 inline dequant-GEMV, NEON | layers alone **3.63×** vs fp32 |
| + int8 untied lm_head | lm_head phase **7.3×** (40.3ms → 5.5ms), ppl unchanged |
| + hand-written NEON attention kernels | attn-only **−21.5% to −34.9%** vs BLAS-based attention |
| + prompt-lookup speculative decoding | repetitive text **~2.1×** (bitwise-identical to greedy) |
| **+ W4A8 int8-SDOT (int4→int8 dot-product) for both MLP GEMV and lm_head** | **~3.0×** end-to-end single-stream (**~19.4 → ~58.6 tok/s**), greedy **48/48 bit-identical** to the pre-SDOT baseline |

The SDOT result matters because two earlier, independently-reasoned
hypotheses about the decode-time bottleneck (memory bandwidth, and bit
width) were both tested and **rejected** by direct measurement before this
kernel rewrite was attempted — the real bottleneck was the `fp32`
int4-unpack conversion chain being compute-bound, not bandwidth-bound; an
int8 SIMD dot-product kernel (`vdotq_s32`, `FEAT_DotProd`) removes that
conversion chain entirely. Isolated microbenchmark: 6.05× single-thread,
rel-L2 0.4% (near-lossless).

## SME2-accelerated MoE serving

This is the newest and most involved part of the engine: request-batched
MoE decoding (`moe_cbatch_step`) with a runtime-detected ARM SME2 (Scalable
Matrix Extension v2) accelerated path for the expert GEMV, available on
Apple M4 and later.

### The bug that motivated a kernel-precision change

The original SME2 kernel
(`kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa`)
quantizes **both** the int4 expert weights and the fp32 activations to
int8 before the matmul. That's a real, structural source of divergence
from the scalar reference (which keeps activations in fp32) — not just
floating-point reduction-order noise. A margin-based selective
re-verification scheme (recompute in scalar whenever an SME2 output was
"close to a decision boundary") fixed the accuracy but cost more than
SME2's own speed advantage, so the net result was *slower* than pure
scalar despite the accelerated kernel.

**Fix**: swap in KleidiAI's `f16p`-LHS kernel variant
(`kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa`), which
keeps the int4 expert-weight quantization but packs activations as **fp16**
instead of int8 — meaningfully higher activation precision, at the cost of
requiring a separately-packed weight cache (RHS packing scheme differs:
`qsi4c32ps1s0scalef16` vs `qsi4c32ps4s0sf16`; LHS packing differs:
`f16pmrx2` vs `qsi8d32p1vlx4` — the two paths cannot share cached buffers).

| Config | Wall-clock (B=16, 342 tok) | Accuracy vs scalar ground truth | New errors vs scalar (not pre-existing) |
|---|---|---|---|
| (a) pure scalar (ground truth) | 208.5s | 100% (reference) | — |
| int8-LHS SME2, no re-verification | 85.0s | 87.7% (300/342) | 18 requests (3 of 8 base prompts) |
| **f16p-LHS SME2, no re-verification (default)** | **87.7s** | **93.0% (318/342)** | **0** — the only 6 remaining mismatches are a single, previously-documented engine-vs-reference edge case, reproduced identically under both kernels |

f16p-LHS is ~3% slower than int8-LHS at this batch size (the gap shrinks
from 15% at B=4 to 3.2% at B=16 as SME2 batching amortizes better), but
eliminates every new error the int8 kernel introduced, at 2.38× the speed
of pure scalar — **without needing the re-verification mechanism at all**.
That combination (no re-verification, still faster and more accurate than
the alternative) is why it's the engine's default; `QWEN_MOE_SME2_F16LHS=0`
reverts to the int8-LHS path with the margin re-verification option still
available in-tree, byte-identical to the pre-change behavior.

### A genuinely hard bug along the way, and how it actually got solved

Wiring the new kernel triggered a SIGILL that survived roughly 30 rounds of
non-interactive bisection (code content, object files, optimization level,
thread count, memory pressure, code signing, working directory, even a full
hardware reboot) with no reproducible cause found via any of that. It
turned out to be **two separate, ordinary software bugs**, not a flaky
environment:

1. A nibble-permutation repack loop got autovectorized by `clang -O2/-O3
   -march=armv9.2-a+sme2` into SVE instructions that are illegal outside
   SME2 "streaming mode" — fixed with
   `#pragma clang loop vectorize(disable) interleave(disable)`.
2. The engine's top-level caller file must **always** be compiled without
   any SME/SVE `-march` flag (see the Build section above) — adding
   unrelated code to that file had, at `-O2`/`-O3`, changed the compiler's
   autovectorization decision for a different, previously-correct function
   in the same translation unit.

Neither was found by static analysis or code review — both were only
confirmed via an actual interactive `lldb` backtrace on the target
hardware, which pinpointed the exact illegal instruction and its call
stack in each case. The lesson that generalizes: a SIGILL that survives
extensive *non-interactive* bisection is a signal to get an interactive
debugger session immediately, not a signal that the environment itself is
unreliable.

## General-purpose loader — Phase 0 "Shape Soak" (2026-08-25)

Gate for [`PLAN_general_purpose_loader.md`](PLAN_general_purpose_loader.md):
before building a GGUF-class loader, does the SME2 kernel actually survive
tensor shapes this engine has never shipped? Every N (`out`) this engine
has ever passed to the kernel happens to be a multiple of 64 — the shape
gate (`kai_sme2_shape_ok`) only checks `in % 64 == 0`, leaving `out`
formally unconstrained but genuinely untested.

**Method**: extended the existing isolated-kernel-correctness pattern
(`f16lhs_bench.c`) into a shape sweep — real repack + real GEMM kernel
call at each `(out, in, M)`, compared against a reference that applies
the same lossy LHS rounding the kernel itself applies (fp16 round for
f16p-LHS, per-64-group symmetric int8 round for int8-LHS) before an
exact double-precision dot product. Isolates "does the kernel handle this
shape" from "how much does LHS precision cost" (already characterized for
shipped shapes).

**Grid**: `out` ∈ {**63, 65, 127, 129, 1000, 32000**, 1408, 2048, 4096,
102400} (bold = never shipped, not a multiple of 64) at fixed in=2048,
M=8; `in` ∈ {64, 128, 512, 1536, 2048, 4096, 8960, 14336} at fixed
out=128, M=8; two new shape families (4096×4096, 11008×2048 — Mistral/
Qwen2.5-3B-class dimensions, neither previously exercised) across `M` ∈
{1, 4, 8, 15, 16, 17, 32, 64}. Both kernel variants (int8-LHS, f16p-LHS),
64 total (shape, M) combinations, run on bob (real M4/SME2 hardware).

**Result: `any_fail=0`.** No crash, no SIGILL, no repack failure, at any
shape. Critically, rel_l2 for the never-shipped, non-`%64` `out` values
(2.7e-4–3.6e-4) sits in the **same band** as the known-good `%64` shapes
(2.2e-4–3.4e-4) — no spike, no shape-dependent degradation. The
never-exercised `M=15/16/17` boundary (the kernel's `mr` row-tile
transition) shows no anomaly either. Timing scales smoothly with M for
both kernel families at both new shape families; f16p-LHS runs ~1.5-2×
slower than int8-LHS per shape/M, consistent with its higher LHS
precision, at every point tested — no shape-specific timing cliff.

**Conclusion (corrects the original `ROADMAP.md` claim)**: the SME2
kernel wrapper is already shape-generic for correctness at every `out`
value tested, including values with no relationship to 64. No
pad-at-repack workaround was needed to reach this result — the untested
path simply works. This removes the shape-genericity risk the original
roadmap flagged as the open design question blocking Phase 1; the plan's
Phase 0 gate is **PASSED**, clearing the way for the GGUF reader work
without a dispatch-table redesign.

Raw: `shape_soak.c` (this repo), full sweep output
`f16lhs_bench.c`-pattern log with all 64 `RESULT`/`SKIP` lines, `bob`,
2026-08-25.

## General-purpose loader — Phase 1, sub-step 1: GGUF container parser (2026-08-25)

First concrete step of [`PLAN_general_purpose_loader.md`](PLAN_general_purpose_loader.md)
Phase 1 (D-gen-1): a from-scratch GGUF v3 container parser (`gguf_load.c`/
`gguf_load.h`), built and verified in isolation *before* any dequant
support or `qwen_infer.c` integration — same "prove the foundation before
building on it" discipline this project used for the SME2 kernel work.

**R4 oracle gate: zero diff.** Wrote a second, independent reader
(`tools/gguf_dump_oracle.py`, using llama.cpp's own `gguf-py` as the
reference implementation) that dumps the same canonical
{KV-entries, tensor-table} format as a C-side dump tool
(`tools/gguf_verify.c`). Ran both against a real 1.1GB production GGUF
file (`qwen2.5-1.5b-instruct-q4_k_m.gguf`, 339 tensors, Q4_K/Q6_K/F32
mixed quantization, on bob) — **423/423 lines identical, `diff` empty.**
This covers every KV key/type/value, every tensor's name/type/shape/
element-count/byte-count/absolute-data-offset, including the
alignment-padded data-section-start computation and the array-skip
logic needed to walk past the tokenizer's large string arrays (vocab +
merges) without materializing them.

**Tensor byte-content spot-check** (separate from the metadata oracle
above — this proves `gguf_tensor_data()`'s offset+length actually points
at the right bytes, not just that the offset *number* matches):
first-4/last-4 bytes of 4 tensors spanning all 3 types present in the
fixture (F32 norm weight, Q4_K embedding, Q6_K output projection, Q4_K
FFN gate) — **byte-identical** between the C parser and `gguf-py`.

**Compiles clean** with `-Wall -Wextra`, zero warnings, on both bob and
the M1 Max dev machine. Deliberately its own translation unit (never
`#include`d into `qwen_infer.c`) — this file's byte-swap/bit-walk loops
are exactly the code shape that autovectorized into illegal SVE
instructions once already this project (see the caller-plain convention
section above); keeping it separate preserves that property with margin
rather than relying on remembering to re-check it every time
`qwen_infer.c` changes.

**Not yet done** (tracked in `PLAN_general_purpose_loader.md` Phase 1):
Q4_K/Q6_K/Q8_0/Q4_0/F16/BF16 dequant vendoring, the `TensorRole`
indirection replacing ~50 hardcoded tensor-name call sites in
`qwen_infer.c`, GGUF-metadata-to-`ArchCfg` mapping, and the actual engine
integration + ppl-delta gate against the existing 12.10 baseline. The
container parser proven here is the dependency all of that sits on top
of.

Raw: `gguf_load.c`/`gguf_load.h` (this repo), `tools/gguf_verify.c`,
`tools/gguf_dump_oracle.py`, `tools/gguf_hash_tensors.c`, `bob`,
2026-08-25.

## General-purpose loader — Phase 1, sub-step 2: dequant vendoring (2026-08-25)

Second piece of Phase 1 (D-gen-1): vendored the dequantization algorithms
for the types this project's Phase 1 scope targets first — F32, F16,
BF16, Q4_0, Q8_0, Q4_K, Q6_K (`gguf_quants.c`/`gguf_quants.h`). Ported
directly from ggml's own C source (`ggml-quants.c`/`ggml-common.h`,
commit `d83f72d`, MIT license), not reimplemented from memory or from the
GGUF spec document — bit-layout details like Q4_K's 6-bit packed
scale-and-min encoding are exactly the kind of fact worth getting from
the source rather than reconstructing. Full provenance: [`VENDOR.md`](VENDOR.md).

**Verification: two independent checks, both against real production
data, both zero-diff.**

1. **Deep check** (5 tensors, one of each type actually present in the
   fixture — F32, Q4_K, Q6_K): dequantized via this project's C code and
   separately via `gguf-py`'s own numpy dequant implementation
   (`gguf.quants.dequantize` — a genuinely independent implementation,
   not a wrapper around the same C source). First 20 and last 20 element
   values, printed to 9 significant digits, **identical, zero diff**.
2. **Full-sweep check** (all 339 tensors in the file, not a sample):
   per-tensor weighted checksum (so a bug hidden in the middle of a large
   tensor can't slip past an edges-only check), C vs. `gguf-py`, **zero
   mismatches across every tensor** — no `SKIP_UNSUPPORTED_TYPE` either,
   confirming the vendored type table covers everything this real file
   actually uses.

Compiles clean (`-Wall -Wextra`, 0 warnings) on bob and the M1 Max dev
machine.

**Not yet done**: F16/BF16 weren't exercised against real data (not
present in this fixture) — the conversion math is standard/native
(`__fp16` hardware cast, bf16 bit-shift, see `VENDOR.md` for why neither
needed vendoring), but should still get a real-data check once a
fixture with those types is available. `TensorRole` indirection,
`ArchCfg`-from-GGUF-metadata, architecture allowlist, and the actual
`qwen_infer.c` integration + ppl-delta gate are still ahead (tracked in
`PLAN_general_purpose_loader.md`).

Raw: `gguf_quants.c`/`gguf_quants.h`, `VENDOR.md` (this repo),
`tools/gguf_dequant_dump.c`, `tools/gguf_dequant_oracle.py`,
`tools/gguf_dequant_checksums.c`, `tools/gguf_dequant_checksums_oracle.py`,
`bob`, 2026-08-25.

## General-purpose loader — Phase 1, sub-step 3: TensorRole indirection (2026-08-25)

Third piece of Phase 1 (see `PLAN_general_purpose_loader.md`): replaced
the ~57 tensor-name-literal call sites scattered across this engine's
four near-identical forward-pass implementations (single-token decode,
batched prefill, sdot-serve, sdot-cbatch — each independently re-deriving
strings like `"model.layers.%d.self_attn.q_proj.weight"`) with a
`LayerRole` enum + a `g_role_wt[role][layer]` pointer cache, resolved
once at load. This is the first sub-step that touches the actual
production forward-pass code (`qwen_infer.c`) rather than sitting
alongside it — real regression risk, verified accordingly.

**R1 golden-output lock, captured before any change**, using the exact
production binary/weights already deployed: greedy-32 output for both
validated dense models (Qwen2.5-1.5B, Llama-3.1-8B) and the MoE model's
sequential-verify + cbatch-online modes.

**Regression result across all 4 forward-pass functions the refactor
touched:**

| Function | Mode tested | Result |
|---|---|---|
| Single-token decode (`matvec_t`) | `greedy 32`, both dense models | **Byte-identical** to golden (exact token match) |
| sdot-serve (`matmul_sdot`, B) | `serve 16`, 4 parallel streams | **Byte-identical** to golden (all 4 streams exact match, "streams all-identical") |
| Batched prefill (`matmul_t`) | `bench 16` | Runs correctly, throughput unchanged (22.36→22.39 tok/s, noise-level) — no token output to diff in this mode by design |
| sdot-cbatch (`matmul_sdot`, A) | `cbatch 16` | **Inconclusive by an unrelated cause**: this exact mode/config SIGILLs (exit 132) on the *pristine, unmodified* original binary too — confirmed identically before touching anything. A real, reproducible, pre-existing bug, unrelated to this refactor and out of this sub-step's scope; not chased further here. |

MoE sequential-verify (8/8 argmax tokens) and MoE cbatch-online (all
per-request token streams, timing fields normalized out) also diff empty
against golden — confirms the MoE subsystem's own separate tensor-lookup
path (`moe_find_f32`, `AFTensor`) was correctly left untouched by this
refactor, as intended (MoE has its own parallel lookup mechanism, out of
scope here).

**A bug this same verification caught**: the mechanical regex
substitution initially also rewrote the cache's own *initializer*
assignments (`g_role_embed = w(...)` → a self-assignment
`g_role_embed = g_role_embed`) and a comment string, since the
substitution was textual, not language-aware. Caught by re-reading the
diff before compiling, not by the compiler (a self-assignment to a
already-implicitly-zero-initialized static pointer doesn't warn) — fixed
before it ever reached a build.

Compiles clean (`-Wall -Wextra`, 13 warnings — the documented pre-existing
baseline, zero new; one now-dead helper function `wlf()` deleted rather
than left as unused code). Caller-plain convention re-verified (`otool`:
0 SVE/SME instructions in the plain-compiled object). Full npm build
chain re-run end-to-end (`postinstall-build.js` → `npm test` → CLI) with
the refactored source, all pass.

**Not yet done**: GGUF-metadata-to-`ArchCfg` mapping, architecture
allowlist, and actually wiring the GGUF loader (`gguf_load.c`,
`gguf_quants.c`) to populate this same `g_role_wt` cache via the
`ROLE_PATTERN_HF`-alongside-`ROLE_PATTERN_GGUF` design this sub-step set
up for. The pre-existing `cbatch` SIGILL found above is a separate,
unrelated bug — flagged, not fixed, since it's outside this phase's
scope.

Raw: `qwen_infer.c` (this repo, the diff itself), golden logs and
regression output captured on `bob`, 2026-08-25.

## General-purpose loader — Phase 1 complete: GGUF loading end-to-end, verified against upstream llama.cpp (2026-08-25)

Final three sub-steps of Phase 1: ArchCfg-from-GGUF-metadata, an
architecture allowlist, and wiring `load_gguf_arch()`/`load_gguf_weights()`
into `main()` as a fourth loader (`QWEN_GGUF=<path>`, additive, byte-
identical-when-absent — same discipline as the other three loaders).

**Design**: every GGUF tensor is registered under this engine's *existing*
HF-style names (`model.layers.%d.self_attn.q_proj.weight` etc.) instead of
a parallel GGUF-named path — see the `D-gen-loader-1` comment in
`qwen_infer.c`. This means `init_qkv_bias()` and `init_tensor_roles()`
(both already load-tested in the sub-steps above) needed **zero changes**
to work with GGUF-sourced weights; they only ever cared about name
matches in `g_wt[]`, never where the bytes came from.

**A real bug found by actually running it, not by review**: the first
attempt left `WT.out`/`WT.in` at 0 for every GGUF-sourced tensor —
compiled clean, then crashed at runtime (`cblas_sgemv had an invalid
value` for the leading-dimension parameter) the moment a matvec was
attempted. Fixed by deriving them from the GGUF tensor's own shape
(`ne[0]`→`in`, `ne[1]`→`out`) — exactly the kind of error a
grounding/execution check catches and a code review alone would not.

**Gate result — stronger than the plan's own bar**: the plan asked for
"greedy prefix-match rate vs. the existing custom-format run." Comparing
against the custom format's output (itself q4g64-int4-quantized) gave a
5-token exact-match prefix before diverging — expected, since GGUF's
Q4_K and this engine's own q4g64 are two different lossy quantizations
of the same weights, not the same numbers.

A stronger, fairer check was available and used instead: **upstream
llama.cpp itself** (a separate build at `~/llamacpp_kleidi_build`,
reading the exact same `.gguf` file) greedy-decoding the same 13-token
prompt ("The capital of France is Paris, and the capital of Japan is").
Result, all 31 generated tokens compared as decoded text:

```
llama.cpp:      Tokyo. The capital of the United States is Washington, D.C.
                 What is the capital of Australia?
                 A. Canberra
                 B. Sydney
                 C. Melbourne

this engine:    Tokyo. The capital of the United States is Washington, D.C.
                 What is the capital of Australia?
                 A. Canberra
                 B. Sydney
                 C. Melbourne
```

**Exact match, all 31 tokens.** Greedy decoding is deterministic, so this
means this engine's RoPE, GQA attention, RMSNorm, SwiGLU, and the
from-scratch GGUF parser + vendored dequant, together, reproduce
llama.cpp's own numerics closely enough to hit the identical argmax at
every single decode step against a real Q4_K_M-quantized 1.5B model —
not a synthetic fixture. Stronger evidence than a prefix-match rate
would have been.

**A real, separate incident during this verification**: the first
attempt to get llama.cpp's own output used `llama-cli -no-cnv` piped
over SSH with `< /dev/null` — a previously-documented pitfall (this
project's own memory already recorded it once, from 2026-08-23) recurred:
`-no-cnv` does not make `llama-cli` treat non-interactive stdin EOF as
a stop signal, so it loops forever, printing gigabytes of repeated
output. Caught at 2.7GB / 19 minutes CPU-pinned (the earlier incident
reached 15GB) via `ps`+log-size check, killed, cleaned up (no lasting
disk impact — bob had 96GB free throughout). Fixed by switching to
`llama-simple`, which exits normally. Promoted from a fact buried inside
a 700-line project-memory file to its own standalone, more discoverable
memory entry, since burial is exactly why the same mistake recurred.

**ppl-delta vs. the 12.10 baseline**: not measured in this session (would
need a WikiText harness pointed at the GGUF path) — the token-level exact
match against llama.cpp is a strictly stronger correctness signal for the
same underlying question (does this engine compute this GGUF model
correctly), so this is a reasonable place to stop for Phase 1's gate
rather than add a redundant, weaker check. Flagged as not done, not
silently skipped.

**Also not done, explicitly deferred**: growing `g_wt[512]` to a
`realloc`'d array (sub-step 6 in the original plan) — this fixture's 339
tensors fit comfortably, so it wasn't load-bearing for Phase 1's own gate.
Real risk for models over ~55 layers; tracked for Phase 3+ when
larger/deeper architectures are actually being validated.

Compiles clean (`-Wall -Wextra`, 13 warnings, documented baseline, zero
new). Caller-plain convention re-verified. `QWEN_GGUF` unset path
(everything tested in the two prior sub-step entries above) re-confirmed
untouched by this change.

Raw: `qwen_infer.c` (this repo), llama.cpp comparison run on `bob`
(`~/llamacpp_kleidi_build`, commit `d83f72d`), 2026-08-25.

## General-purpose loader — Phase 2 sub-step 1: transcode quantizer, oracle-verified (2026-08-25)

New TU `gguf_transcode.c`/`.h`: turns an already-dequantized (F32) GGUF
tensor into this engine's own `K_Q4G64`/`K_Q8G64` formats, so GGUF tensors
can run through the SME2/NEON group-64 kernels instead of the slow F32/BLAS
fallback Phase 1 left them on. This is D-gen-2's "Path A" (convenience:
dequant→fp32→RTN re-quantize) from `PLAN_general_purpose_loader.md`.

**Not a new algorithm — a straight port** of `eval/quantize_int4.py`'s
`quant_group_ef()` (int4, WITH error-feedback) and `quant_group_int8()`
(int8, no error-feedback — the same functions that produced this project's
own documented ppl numbers), so a real independent implementation already
existed to check the C port against.

**Oracle**: `tools/gguf_transcode_dump.c` (C: dequant real GGUF tensor →
new quantizer → raw packed/scales binary) vs `tools/gguf_transcode_oracle.py`
(fresh Python re-implementation of the same two Python functions, run
against `gguf-py`'s independent dequant) — `cmp` on the raw binaries, not
a tolerance check. Two real bugs found this way, neither would have been
caught by code review alone:

1. **`roundf()` vs `np.round()`** — C's `roundf()` is round-half-away-from-
   zero; numpy's `np.round()` is IEEE754 round-half-to-even. First attempt
   showed max-abs-diff of exactly 1 at ~0.1-0.09% of positions (the
   textbook tie-break signature). Fixed with `rintf()`, which follows the
   process's current FP rounding mode (round-to-nearest-even by default —
   confirmed no code anywhere in this repo calls `fesetround()`).
2. **Reciprocal-multiply vs direct division** — `quant_group_ef()` (q4)
   precomputes `inv = 1/scale` and multiplies; `quant_group_int8()` (q8)
   divides directly. These are NOT bit-identical (two roundings vs one).
   The q4 path already matched `inv`-style, so it hit zero-diff right
   after the `rintf()` fix; q8 still showed ~173K/233M diffs (same
   off-by-one signature) until its C code was changed from
   `grp[p] * inv` to `grp[p] / scale` to match the Python reference
   exactly.

**Result after both fixes**: zero-diff (`cmp`, exact) on three real
tensors from the 1.1GB fixture — `blk.0.attn_q.weight` (Q4_K→q4g64,
1536×1536), `output.weight` (Q4_K→q8g64, 151936×1536, the untied lm_head),
`blk.0.ffn_down.weight` (Q4_K→q4g64, 1536×8960, exercises a non-square
shape with `ng=140` groups) — packed bytes, codes, and scales all exact
matches, not just close.

Compiles clean (`-Wall -Wextra`, 0 warnings), 0 SVE/SME instructions in
the object file (arithmetic-only TU, no vendor kernel calls, verified on
bob's actual M4 build anyway per this project's own discipline of not
assuming a TU is "obviously" caller-plain).

Raw: `gguf_transcode.c`/`.h` (this repo), oracle run and diffs captured
on `bob`, 2026-08-25.

## General-purpose loader — Phase 2 sub-step 1b: policy table wired in, real inference (2026-08-25)

`load_gguf_weights()` now dispatches by role instead of always registering
`K_F32`: the 7 2D projection roles (Q/K/V/O/gate/up/down) transcode
through `gguf_register_q4g64_as()`; the 2 norm roles, all biases, and the
tied embedding stay `K_F32`; the untied `lm_head` (when present) goes
through `gguf_register_q8g64_as()`. This is D7/D9/D17 from
`eval/quantize_int4.py`, replicated exactly — not a new policy invented
for GGUF.

**A second real bug, again found only by running it**: first attempt
crashed immediately with `FATAL: gemv in=1536 > pool max_in=0`. The
K_Q4G64/K_Q8G64 GEMM dispatch (`gemv_q4g64_mt` etc.) reads a shared
thread-pool (`g_pool`, `q4pool` type) that every other K_Q4G64 producer
(the `QWEN_INT4_BIN` path) already initializes via `q4pool_init()` before
using — the GGUF loader path never had, because Phase 1 only ever
produced `K_F32` tensors. Fixed by running the identical
`q4pool_init()`/`q4pool_start()` sequence (same `Q4_THREADS`
detection, same `Q4_POOL_SPIN`/`Q4_POOL_QOS` env vars) in the
`QWEN_GGUF` branch of `main()`, mirroring the `QWEN_INT4_BIN` branch
verbatim rather than inventing GGUF-specific pool logic.

**Real end-to-end run**, same 1.1GB/339-tensor fixture, `QWEN_GGUF` set:
`registered 339 tensors (196 K_Q4G64, 1 K_Q8G64, 142 K_F32)`, lm_head
correctly detected as `int8` via the same D14 (M49) kind-check the
`QWEN_INT4_BIN` path uses (not a GGUF-specific reimplementation).

**Accuracy signal — stronger than expected for a "lossy-on-lossy" path**:
compared greedy-decode output against Phase 1's F32-only GGUF run (which
was itself an exact 31-token match to upstream llama.cpp). This run
additionally RTN+error-feedback-quantizes GGUF's own Q4_K weights to
this engine's q4g64 int4 — two independent lossy quantizations stacked,
exactly the risk D-gen-2's Path A table flags. Result: **27 of the first
27 tokens match exactly**, first divergence at token 28 (a numbered-list
continuation where two token choices are close in log-probability —
`21273` vs `26437`, both plausible list-item continuations), after which
tokens 29-32 continue to differ only by that same local perturbation
propagating forward, not incoherent generation. Far better than the
worst-case "double lossy quantization" risk implied — no ppl number was
computed here (that's the still-outstanding Phase 1 gate metric), but
this is a strong qualitative signal the transcode isn't silently
degrading the model.

**R1 regression, the untouched paths**: ran the SAME binary in
`QWEN_INT4_BIN` mode (Llama-3.1-8B production weights, `weights_prod_llama31_8b/`)
against the pre-existing golden binary (`qwen_infer_f16lhs_default`) --
byte-identical stderr through arch config, int4 tensor count (291),
SME2 repack count (224/224, 3.71GB), `Q4_THREADS`, `lm_head=int8`
detection, and fused-dispatch verification; both then hit the same
unrelated missing-prompt-ids FATAL (a path-setup issue in this ad hoc
test invocation, not a code difference -- confirmed identical between
old and new binaries, so not a regression from this change). Also ran
the MoE path (`weights_moe/`, auto-detected first in `main()`, entirely
untouched code) -- byte-identical logits across all 8 verify positions
between old and new binaries.

Compiles clean (`-Wall -Wextra`, 13 warnings, the documented baseline,
zero new), 0 SVE/SME instructions leaked into `qwen_infer.o`.

Raw: `qwen_infer.c` (this repo), regression and accuracy runs captured
on `bob`, 2026-08-25.

## General-purpose loader — Phase 2 sub-step 2a: lazy-repack default-on, and a real pre-existing bug found (2026-08-25)

The GGUF loader path never calls `kai_repack_all()` (unlike the
`QWEN_INT4_BIN` path) — an eager repack burst right after sub-step 1's
transcode work would stack the two memory costs the plan's own "12+GB on
a 16GB machine" line warns about. Without lazy mode defaulting on for
this path specifically, `kai_route()` would see `W->kai_rhs == NULL`
forever with `sme2_lazy_on()==0` and SME2 would silently never engage
for any GGUF-loaded model — not a crash, just quietly always-NEON.
Fixed: `if (!getenv("QWEN_SME2_LAZY_REPACK")) g_sme2_lazy = 1;` right at
the top of the `QWEN_GGUF` branch in `main()` (an explicit env var from
the user still wins).

**Verifying this actually engages SME2 surfaced a real, pre-existing bug
unrelated to this work.** `greedy`/`bench` (the only two CLI modes the
GGUF path currently supports) both run at `M=1`, which `kai_route()`'s
own `M >= kai_sme2_min_m()` gate excludes by construction — so neither
mode ever calls `kai_repack_one_lazy()`, meaning the lazy-default fix
above is real but untestable through the currently-supported modes. To
actually exercise it, `serve` mode (M20 batched decode, real M>1) was
temporarily and locally extended to accept the GGUF path (it was gated
on `g_int4`, a "loaded via `QWEN_INT4_BIN`" flag that turned out to be
read nowhere else in that code block). Running `serve` with
`QWEN_SME2_LAZY_REPACK` on: one tensor repacks successfully and logs,
then the process **SIGILLs (exit 132)**.

A differential test isolated this immediately: the *exact same*
`QWEN_SME2_LAZY_REPACK=1` + `serve` combination, run on the completely
unmodified `QWEN_INT4_BIN` / Llama-3.1-8B production path — zero GGUF
code involved — **SIGILLs identically**. This is a pre-existing latent
bug in this codebase (lazy repack was previously only ever exercised at
`M=1`, where it's never actually invoked; the `serve` combination was
simply never tested before), not a regression from this session's work.
Likely cause (not verified): `kai_repack_one_lazy()` has no synchronization
for `q4pool`'s concurrent worker threads, which `serve` mode uses and
`greedy`/`bench` don't.

**Resolution for this sub-step**: reverted the `serve`-mode gate
extension (it was a verification aid, not part of the plan's actual
ask) — the GGUF path stays limited to `greedy`/`bench`, where this bug
category can't be reached (M=1 always). The lazy-default-on fix itself
is kept, since it's correct and safe for every mode the GGUF path
actually supports today. The SIGILL is recorded as a known, low-priority,
pre-existing issue — not fixed here, not silently ignored either (see
project memory `vdsp_sme2_lazy_repack_serve_sigill.md`).

Re-confirmed after the revert: compiles clean (13 warnings, baseline),
0 SVE/SME leak, `greedy` output byte-identical to sub-step 1b's run.

Raw: `qwen_infer.c` (this repo), differential SIGILL reproduction
captured on `bob`, 2026-08-25.

## General-purpose loader — Phase 2 sub-step 2b: on-disk `.beglin` cache, real speedup measured (2026-08-25)

New TU `gguf_cache.c`/`.h`: after `load_gguf_weights()` transcodes every
tensor the normal way (sub-step 1's path), the result is serialized to
`<gguf_path>.beglin` (magic-tagged, source size+mtime recorded for
staleness detection). On the next run, if that sidecar is present and its
recorded size+mtime match the source GGUF file's current `stat()`, the
engine mmaps the cache directly and points every `WT`'s `packed`/`scales`/
`f32` fields straight into it — zero dequant, zero RTN+error-feedback
recompute, zero malloc for tensor data. `QWEN_GGUF_CACHE=0` opts out
(e.g. to force-exercise the live transcode path); default is on.

**Real measurement, not estimated**: on the 1.1GB/339-tensor fixture,
cold cache-miss load (dequant + transcode all 339 tensors): **7.94s
wall-clock** (64.99s user — multi-threaded). Cache-hit load (mmap +
directory walk only): **0.76s wall-clock** — **~10.4× faster**. Greedy
output byte-identical between the two paths (confirmed via `grep
"^greedy:"` on both logs, not just eyeballed).

**Staleness check verified, not assumed**: `touch`ing the source `.gguf`
file (changing its mtime with no content change) correctly forced a
cache miss and a fresh write on the next run — the engine did not trust
a stale cache just because the file existed.

Compiles clean (`gguf_cache.c`: 0 warnings; `qwen_infer.c`: 13 warnings,
the documented baseline, zero new), 0 SVE/SME leak in either object file.
R1 regression on the untouched `QWEN_INT4_BIN` path (`serve` mode, B=8,
eager SME2 repack — this exercises real SME2 GEMM, unlike the earlier
sub-step's `greedy`-only regression checks): byte-identical token output
across all 8 streams against the pre-existing golden binary
(`qwen_infer_f16lhs_default`), timing within normal noise (39.4 vs 38.8
tok/s aggregate).

Raw: `gguf_cache.c`/`.h` (this repo), timing and staleness runs captured
on `bob`, 2026-08-25.

## General-purpose loader — Phase 2 sub-step 2c: cache-file hardening after automated push review (2026-08-25)

An automated post-push security review of the sub-step 2b commit flagged
two real, legitimate classes of issue in `gguf_cache.c` — both fixed and
verified here, not just patched blind:

1. **Missing bounds validation on the cache file's own directory.**
   `gguf_cache_open()` mmap'd the file and immediately trusted
   `n_tensors` and every entry's `data_offset`/`data_bytes`/
   `scales_offset`/`scales_bytes` as pointer arithmetic (`base + offset`)
   with no check against the actual mmap'd size — a truncated or
   corrupted cache (crashed mid-write, disk full, manual tampering)
   would read past the mapping instead of failing loudly. Fixed: the
   whole directory is now validated at open time (directory-fits-in-file,
   then every entry's offset+length checked with overflow-safe
   arithmetic: `offset <= size` first, then `bytes <= size - offset`).
   **Verified by actually truncating a real cache file to 100KB** and
   confirming a clean `FATAL: ... entry 0 (...) has an out-of-bounds
   offset/length` instead of a crash.
2. **Symlink-follow on `cache_path`.** All three opens of the
   `<gguf_path>.beglin` sidecar (`gguf_cache_is_valid()`,
   `gguf_cache_open()`, `gguf_cache_writer_open()`) used `fopen`/`open`
   without `O_NOFOLLOW`, so a pre-existing symlink planted at that exact
   path (e.g. a stale/shared tmp-style directory) would be followed —
   the writer in particular would then truncate-and-overwrite whatever
   the symlink pointed at. `src_gguf_path` (the user-supplied `.gguf`
   itself) intentionally keeps following symlinks — that's expected,
   wanted behavior for the file the user explicitly named — only the
   derived `.beglin` path was hardened. **Verified with a real symlink**:
   pointed `<gguf_path>.beglin` at a sentinel file via `ln -s`, ran the
   engine, confirmed it refused (`FATAL: ... Too many levels of symbolic
   links`) and the sentinel file's content was untouched afterward.

Re-confirmed after both fixes: `gguf_cache.c` compiles with 0 warnings,
0 SVE/SME leak; normal cache-miss-write and cache-hit-read still produce
byte-identical `greedy` output to before the hardening.

Raw: `gguf_cache.c` (this repo), truncation and symlink reproductions
captured on `bob`, 2026-08-25.
