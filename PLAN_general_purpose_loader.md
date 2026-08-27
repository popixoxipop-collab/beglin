# Plan: General-Purpose (GGUF-Class) Model Loading

Produced by an Opus Plan agent (2026-08-25) that read the real code in this
repo, `/Users/xox/vdsp_local/llm_engine/eval/`, and bob's dev environment
before writing this — it corrects several claims `ROADMAP.md` made from
memory rather than a fresh audit (see §0).

This is the plan for [`ROADMAP.md`](ROADMAP.md)'s D-roadmap-1. Read that
file first for the WHY/COST/EXIT framing; this file is the HOW.

## 0. Findings that correct ROADMAP.md's premises

Three claims in the original roadmap were wrong or imprecise, and two of
them change the sequencing materially.

**Correction 1 — the SME2 kernels are NOT fit to specific tensor shapes.**
`sme2_kai.c:59-63` / `:201-205`:

```c
int kai_sme2_shape_ok(int out, int in) {
    if (!kai_sme2_available()) return 0;
    if (out <= 0 || in <= 0) return 0;
    return (in % SME2_KAI_BL) == 0;      // SME2_KAI_BL == 64
}
```

There is no constraint on `out` at all. On the vendor side,
`kleidiai/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.c:393`
sizes the packed buffer as `kai_roundup(n, nr) / nr` — arbitrary N is
handled by pad-at-pack — and the matmul kernel carries no `KAI_ASSUME` on
`n` (only `bl % 32 == 0`, `k % bl == 0`). The kernel wrapper is already
(M, K, N)-generic subject to `in % 64 == 0`.

The genuine unknown: **every N this engine has ever passed happens to be a
multiple of 64** (576, 1408, 2048, 3072, 4096, 10944, 102400, 1536, 8960,
14336, 128256, 151936). The `out % 64` tail path is claimed-by-predication
and never exercised. That single untested fact is the highest-value
experiment in the whole roadmap (Phase 0), and it costs about a day.

**Correction 2 — the dense engine is already metadata-driven; the MoE
path is what's hardcoded.** `load_arch_cfg` reads 10 keys and derives
HD/KVD/QD/GROUP/KVG/QG; `alloc_arch_buffers` heap-sizes every buffer from
`g_cfg`. Attention already has a three-tier runtime shape gate with a real
generic-scalar fallback. By contrast the MoE path is stack-array-hardcoded
to DeepSeek-V2-Lite (`h2_batch[][2048]`, `logits_out[][102400]`,
`MOE_MAXLAYERS=32`, `groupsum[256]` which silently caps `in ≤ 16384` and
already once caused a real stack smash when `dense_down`, in=10944,
joined that dispatcher).

**Correction 3 — the export pipeline is closer to generic than stated.**
`ref/export_weights.py` is fully architecture-agnostic already (only gap:
single-file, non-sharded safetensors). `eval/quantize_int4.py` quantizes
any 2D tensor with `min(shape) ≥ GROUP` from the manifest — only three
name literals are architecture-specific. `eval/gptq_quantize_fold_g64.py`
targets by module suffix, llama-family-generic not Qwen-specific.
`eval/emit_arch_config.py` already reads HF `config.json` directly.

**Assets that exist and should be used, not rebuilt:** bob has a
llama.cpp checkout (`~/llamacpp_kleidi_build`, commit `d83f72d`, with
`gguf-py/`), a Q4_K_M GGUF of the already-validated Qwen2.5-1.5B-Instruct
(`~/models_gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf`), all three production
weight fixtures, and existing shape-probe harnesses
(`probe_sme2_sizes.c`, `f16lhs_bench.c`, `moe3c_sme2_bench.c`,
`verify_af_decompose.c`).

## 1. Resolved architectural decisions

### D-gen-1: from-scratch GGUF container parser; vendor only the dequant reference

**WHY.** llama.cpp's GGUF reader (`ggml/src/gguf.cpp`) is ~1700 lines of
C++17 depending on `ggml.h`/`ggml-backend.h` — vendoring it means adding a
C++ TU to a pure-C build and pulling in ggml's type-trait/registry
machinery. The container format itself is trivial (~450 lines of C: magic
+ version + counts, typed KV pairs, per-tensor info, padded data
section) — the kind of code this project already writes with discipline
(compare `load_int4`'s bounds-checking and FATAL-on-unrecognized-kind
doctrine). A vendored C++ parser doesn't inherit that discipline, and a
hostile/truncated GGUF is a real attack surface for a published package.

**What we DO vendor:** Q4_K/Q6_K/Q8_0/Q4_0/F16/BF16 dequant reference +
block structs from `ggml-common.h`/`ggml-quants.c` (MIT, SPDX preserved,
same `kleidiai/`-style vendoring convention, needs a `VENDOR.md` entry).

**What we vendor as an oracle, not code:** llama.cpp's `gguf-py` reader —
every fixture gets dumped by both readers, tables must diff empty.

**COST.** ~450 new lines + a vendored dequant file; permanent obligation
to track GGUF format version bumps (v3 today, stable since 2023).
**EXIT.** If GGUF v4 is structurally different, or ggml's tensor-splitting
becomes necessary, revisit — vendoring `gguf.cpp` later is strictly
easier than un-vendoring it now.

**Placement (load-bearing).** New TU `gguf_load.c`, NOT inside
`qwen_infer.c`. `RESULTS.md` documents that adding unrelated code to the
plain-compiled top-level TU already once changed clang's autovectorization
of a different, previously-correct function and produced a SIGILL — a
GGUF reader's byte-swap/bit-unpack loops are exactly that code shape (see
the two `vectorize(disable)` pragmas already in `sme2_kai.c`). A separate
TU preserves the caller-plain convention with margin.

### D-gen-2: GGUF is an input format only; every tensor is transcoded at load. The export pipeline does NOT go away.

Both SME2 kernels require RHS in KleidiAI `qsi4c32` layout: symmetric
int4, one fp16 scale per 64-element block. GGUF Q4_K is a 256-element
super-block with **affine** (scale + min) 6-bit packed values in a
different nibble order; Q6_K is worse. There is no reinterpretation, only
transcoding — the codebase already does this twice (`kai_sme2_repack_q4g64`
for K_Q4G64→KleidiAI tiles; `moe_sme2_ensure_ready` for the affine MoE
blob→symmetric+`adj_bias`). GGUF→beglin is a third instance of a shape
this codebase already knows how to handle.

**Two input paths, both shipped, always labeled in the startup log:**

| Path | Input | Route | Quality |
|---|---|---|---|
| A — convenience | GGUF, already quantized (Q4_K/Q6_K/Q8_0) | dequant→fp32→RTN re-quantize to q4g64/q8g64 | lossy-on-lossy; ppl delta MUST be measured (Phase 1 gate), not estimated |
| B — quality | GGUF F16/BF16/F32, or an HF checkpoint | straight into RTN+EF or GPTQ, no double quantization | this is where `ppl 12.10 / +13.6%` lives; no GGUF-native path reproduces it |

Path A solves ~80% of the adoption problem ("point at any GGUF") for
~20% of the effort; Path B is why beglin's numbers beat a naive GGUF
loader, and turns out to be a smaller job than the original roadmap
implied (Correction 3).

**Flagged, not decided:** `bl` only needs `%32==0`, so a `bl=32` SME2
variant would align with Q4_K's sub-block size, and the affine→symmetric
decomposition needed to drop Q4_K's `min` is *already implemented* in
`moe_sme2_ensure_ready`. Plausible near-direct Q4_K→qsi4c32 transcode —
not assumed for Phase 1 because it changes a documented format invariant
and doubles the correction term's cost. Real experiment, Phase 5.

### D-gen-3: no dispatch table of shape-specialized fast paths — widen two gates, add one measured policy table

The model to copy is the one this project already invented for attention:
one runtime predicate per kernel family + a real generic fallback tier +
a loud startup log naming which tier each tensor landed in.

| Constraint | Kind | Breaks | Fallback |
|---|---|---|---|
| `in % 64 == 0` | HARD | nothing in practice; every mainstream hidden/intermediate size satisfies it, GGUF Q4_K implies `in%256==0` | already exists — `K_F32`→BLAS |
| `out % 64` | **UNKNOWN, never exercised** | possibly the whole SME2 path for e.g. `vocab=32000` models | Phase 0 measures it; remedy = pad N at repack to `ceil(out/64)*64` |
| MoE stack arrays | HARD, MoE only | any non-DeepSeek-V2-Lite MoE, immediately | convert to heap (`g_cfg`-style) — Phase 4 |
| `groupsum[256]` (`in≤16384`) | SOFT, MoE only | Llama-3.1-70B-class `intermediate_size=28672` | already caused one real stack smash; make heap-sized |
| `g_wt[512]` | SOFT | any model >~55 layers | grow to `realloc`'d array |
| `hd = d/nh` | HARD | Qwen3-0.6B, all Gemma | add optional `HD` key |
| `kai_route`'s `M ≥ 16` threshold | policy, not correctness | nothing breaks; it's tuning | keep conservative default for unrecognized shapes, per-shape-family override ONLY from real bench sweeps (same doctrine as `detect_q4_threads()`, which refuses to interpolate from n=2 points) |

### D-gen-4: architecture priority, by marginal engine work — not popularity

Similarity metric that predicts integration cost: (HD, GROUP, RoPE
variant, qkv_bias, norm placement, activation, head_dim-vs-D/NH coupling).

1. **Mistral-7B-v0.3** — HD=128/GROUP=4, byte-identical shape family to
   the already-validated Llama-3.1-8B (M45 `_g4` kernels). rotate_half, no
   rope scaling, no qkv_bias, SwiGLU, same HF names. **Marginal engine
   code: zero** — only the loader has to work. Proves the loader without
   confounding it with kernel work; ship first for exactly that reason.
2. **Qwen2.5-{0.5B, 3B, 7B}** — deliberately hits every dispatch tier
   (7B: HD=128/GROUP=7; 3B: HD=128/GROUP=8; 0.5B: HD=64/GROUP=7, generic-
   scalar tier). Stress-tests the gates, not the math.
3. **Llama-3.2-1B/3B** — reuses `rope_llama3_scale.h` (already verified to
   3.2e-7 vs transformers). 1B forces HD=64.
4. **Qwen3 dense (0.6B-8B)** — adds one new op (per-head q/k RMSNorm
   before RoPE), retires the `hd=d/nh` assumption — a prerequisite for
   Gemma later.
5. **Mixtral-8x7B / Qwen3-30B-A3B** — MoE *without* MLA: reuses dense GQA
   attention entirely, needs only routing + expert FFN. Puts the SME2
   batched-expert win onto a second, mainstream MoE topology. Highest
   strategic value per unit of work after #1.

**Deferred, with reason:** Gemma-2/3 (GeGLU-tanh, logit softcapping, dual
norms, embedding×√d, alternating sliding-window — 5 new numeric-parity
fights, defer until #4 retires the head_dim coupling). Phi-3 (LongRoPE is
a third rope-scaling family; tractable but not first). Mamba/Jamba/RWKV
(no attention path at all — out of scope permanently unless the product
changes).

### D-gen-5: tokenizer is explicitly out of scope until Phase 6

beglin has no tokenizer today (`load_ids` reads a pre-tokenized `.i32`
file). GGUF carries `tokenizer.ggml.*` metadata, but a correct BPE with
the right pre-tokenizer regex per model family is its own multi-week
correctness project — folding it into the loader work would confound the
numeric-regression story that is this project's credibility. **Stance:**
Phases 1-5 ship a small Node helper that reads the GGUF's own vocab and
emits the `.i32` file ("one command, then run"); a real in-engine
tokenizer is Phase 6, scoped and measured separately.

## 2. Regression and safety story (every phase, not just at the end)

**R1 — Golden-output lock**, captured on bob before any code changes: all
three validated models across the env matrix `RESULTS.md` already claims
(`greedy`, `bench`, `serve`, `QWEN_W4A8`/`QWEN_SME2`/`QWEN_SME2_LAZY_REPACK`
on/off, MoE cbatch B=16 at default and `=0`). Bar: greedy/argmax output
bit-identical; MoE token-match within the documented band (93.0%,
zero new errors).

**R2 — Additive entry point**, same pattern as `run_moe_verify_mode`:
GGUF loading is a fourth loader, probed before `load_arch_cfg`, falling
through byte-identically when absent. The three legacy loaders stay
untouched.

**R3 — SVE/SME leak check becomes a per-phase gate**, run on bob's actual
M4 (the only place `kai_sme2_available()` is true — an M1 Max dev run
cannot detect a regression here). Not ceremony: this project's own
history shows ~30 rounds of non-interactive bisection failing to find
this bug class; only interactive lldb on target hardware found it.

**R4 — Differential oracle for the parser**: `gguf-py` vs `gguf_load.c`
KV+tensor table diff must be empty; dequant vs. `ggml`'s must be exactly
0 max-abs-diff (both are exact integer→float reconstructions).

## 3. Phases

### Phase 0 — "Shape Soak" (~1 week, touches zero shipped code) — ✅ DONE 2026-08-25, GATE PASSED

**Result: `any_fail=0` across 64 (shape, M) combinations on real M4
hardware**, including every `out` value that isn't a multiple of 64
(63, 65, 127, 129, 1000, 32000 — none previously shipped). rel_l2 for
those shapes (2.7e-4-3.6e-4) sits in the same band as known-good `%64`
shapes (2.2e-4-3.4e-4) — no shape-dependent degradation, no crash, no
repack failure, no timing cliff at the `M=15/16/17` `mr`-boundary. No
pad-at-repack workaround was needed. Full writeup:
[`RESULTS.md`](RESULTS.md#general-purpose-loader--phase-0-shape-soak-2026-08-25),
harness: [`shape_soak.c`](shape_soak.c), raw output:
[`results/phase0_shape_soak_out.txt`](results/phase0_shape_soak_out.txt).

**This confirms Correction 1 (§0) empirically, not just by code reading**:
the SME2 kernel is already shape-generic for correctness. Phase 1 can
proceed without a dispatch-table redesign for shape support — that open
design question from the original `ROADMAP.md` is closed.

Original phase spec (for reference — superseded by the actual run above):

The credibility hinge everything downstream is contingent on. Extend
`probe_sme2_sizes.c`/`f16lhs_bench.c`/`moe3c_sme2_bench.c` into a
standalone shape sweep: random q4g64 weights at a grid of (M, K, N) →
repack → gemm → compare vs NEON/scalar reference.

Grid: `K` ∈ {64,128,512,1536,2048,4096,8960,14336} (incl. two never
shipped); `N` ∈ {**63,65,127,129,1000**,1408,2048,4096,**32000**,102400}
(bolded = the point — 32000 is Mistral/Llama-2 vocab, not %64); `M` ∈
{1,4,8,15,16,17,32,64}; both LHS variants, both availability gates.

**Pre-declared gate:** no crash/SIGILL at any shape; zero new errors vs.
scalar reference at any shape; rel-L2 at already-shipped shapes stays in
the ~3.9e-3 band already established. If `N%64` fails: pad-N-at-repack,
re-measure memory cost.

Also: M-threshold sweep for two new shape families feeding D-gen-3's
policy table — do not extrapolate from the DeepSeek expert-shape result.

**Gate to Phase 1:** if SME2 can't survive arbitrary N even with padding,
the roadmap's value proposition changes (still shippable — log which
tier each tensor landed in — but must be known before the loader
rewrite, not discovered late).

### Phase 1 — GGUF reader + role-based tensor resolution, behind a flag (~2 weeks)

1. ✅ **DONE 2026-08-25** — `gguf_load.c`/`gguf_load.h`: container parse,
   bounds-checked, FATAL on unrecognized (the `load_int4` doctrine). R4
   oracle gate passed with zero diff against `gguf-py` across all 423
   lines of KV+tensor-table output on the real 1.1GB fixture (339
   tensors); tensor byte-content spot-checked identical too, not just
   offsets. Compiles clean (`-Wall -Wextra`, 0 warnings) on both bob and
   the M1 Max dev machine, its own TU as planned. Full writeup:
   [`RESULTS.md`](RESULTS.md#general-purpose-loader--phase-1-sub-step-1-gguf-container-parser-2026-08-25).
   Remaining sub-steps below are NOT yet done.
2. ✅ **DONE 2026-08-25** — Vendored dequant (`gguf_quants.c`/`gguf_quants.h`,
   MIT, ported from ggml commit `d83f72d`, [`VENDOR.md`](VENDOR.md) entry).
   Verified against `gguf-py`'s independent numpy dequant on the real
   fixture: deep exact-value check on one tensor of each type present
   (F32/Q4_K/Q6_K), plus a full 339/339-tensor checksum sweep — zero
   mismatches either way. F16/BF16 implemented (native `__fp16`/bit-shift,
   no vendoring needed, see `VENDOR.md`) but not yet exercised against
   real data (absent from this fixture). Full writeup:
   [`RESULTS.md`](RESULTS.md#general-purpose-loader--phase-1-sub-step-2-dequant-vendoring-2026-08-25).
3. ✅ **DONE 2026-08-25** — `TensorRole` (`LayerRole` enum) indirection
   replacing 57 name-literal call sites with `g_role_wt[ROLE_X][l]`,
   resolved once at load via `ROLE_PATTERN_HF[]` (mirrors `init_qkv_bias`'s
   one-dereference design as planned; the `"gguf"` naming table is a
   follow-on, not added yet). R1 golden-output regression: 3 of 4
   forward-pass functions byte-identical to pre-refactor output on real
   production weights (both dense models + MoE); the 4th's test mode
   hit an unrelated pre-existing SIGILL confirmed on the *original*
   binary too (flagged, not fixed, out of scope). Full writeup:
   [`RESULTS.md`](RESULTS.md#general-purpose-loader--phase-1-sub-step-3-tensorrole-indirection-2026-08-25).
4. ✅ **DONE 2026-08-25** — `ArchCfg` from GGUF metadata (`%s.block_count`,
   `.embedding_length`, `.feed_forward_length`, `.attention.head_count`,
   `.attention.head_count_kv`, `.rope.freq_base`,
   `.attention.layer_norm_rms_epsilon`), plus VOCAB derived from
   `token_embd.weight`'s own tensor shape and QKV_BIAS from tensor
   *presence* rather than a scalar key. No keys added to `arch_config.txt`
   — this path is fully independent of it, as planned. Full writeup:
   [`RESULTS.md`](RESULTS.md#general-purpose-loader--phase-1-complete-gguf-loading-end-to-end-verified-against-upstream-llamacpp-2026-08-25).
5. ✅ **DONE 2026-08-25** — Architecture allowlist on `general.architecture`
   (only `"qwen2"` currently) — FATAL with a named supported-list rather
   than attempting an unvalidated run, matching `load_int4`'s doctrine.
6. **Deferred, not done** — `g_wt[512]` is still a fixed-size array. Not
   load-bearing for this fixture (339 tensors), so not blocking Phase 1's
   gate; real risk starts around ~55+ layer models. Tracked for Phase 3+
   when deeper architectures actually get validated — noted here rather
   than silently skipped.

**Fixture/gate — result:** `qwen2.5-1.5b-instruct-q4_k_m.gguf`, real
1.1GB/339-tensor file. R4 oracle diff: empty (sub-steps 1-2). Greedy
correctness: the plan's own literal "prefix-match vs. existing [custom-
format] run" metric was superseded by a stronger, fairer check — the
custom format is q4g64-int4 and GGUF here is Q4_K, two different lossy
quantizations of the same weights, so a byte-for-byte token match between
them was never the right bar (got a 5-token exact prefix before expected
divergence, consistent with that). Instead: this engine's GGUF-loaded
greedy output was compared against **upstream llama.cpp** (`llama-simple`,
a separate build) reading the *same* `.gguf` file on the same 13-token
prompt — **exact match, all 31 generated tokens**, decoded to identical
text. WikiText ppl delta vs. the existing 12.10 baseline: **not measured**
this session — Phase 1's GGUF path is F32-only (no quantized transcode
until Phase 2), so this number would mainly confirm "F32 dequant ≥ int4,"
an expected/low-risk result, and the llama.cpp token-exact-match is a
strictly stronger signal for the same underlying correctness question.
Flagged as an outstanding gate metric, not silently dropped. Full
writeup: [`RESULTS.md`](RESULTS.md#general-purpose-loader--phase-1-complete-gguf-loading-end-to-end-verified-against-upstream-llamacpp-2026-08-25).

### Phase 2 — Transcode + on-disk cache; first new model (~1.5 weeks)

1. ✅ **DONE 2026-08-25** — Transcode stage: GGUF tensor → K_Q4G64/K_Q8G64/K_F32
   via a per-role policy table (`gguf_transcode.c`/`.h`, new TU, RTN+error-
   feedback for K_Q4G64 and plain RTN for K_Q8G64 — a direct port of
   `eval/quantize_int4.py`'s `quant_group_ef()`/`quant_group_int8()`, not a
   new algorithm). Oracle-verified zero-diff against an independent Python
   re-implementation on 3 real tensors from the fixture; two real bugs
   (round-half-to-even vs round-half-away-from-zero, and reciprocal-multiply
   vs direct division) found and fixed via the oracle diff, not review.
   `load_gguf_weights()` now wires the policy table in (7 projection roles
   → K_Q4G64, norms/biases/tied-embed → K_F32, untied lm_head → K_Q8G64,
   matching D7/D9/D17 exactly) and a second real bug (the shared `q4pool`
   thread pool never being initialized for this path) was found by running
   it and fixed by mirroring the `QWEN_INT4_BIN` branch's init sequence.
   Real end-to-end run on the 1.1GB fixture: 27/27 exact token match
   against Phase 1's llama.cpp-verified F32 baseline before first
   divergence at a close-log-probability tie point — far better than the
   "lossy-on-lossy" worst case this table's own row warns about. R1
   regression on the untouched `QWEN_INT4_BIN`/MoE paths: byte-identical.
   Full writeup: [`RESULTS.md`](RESULTS.md#general-purpose-loader--phase-2-sub-step-1-transcode-quantizer-oracle-verified-2026-08-25).
2. ✅ **DONE 2026-08-25** — On-disk cache: `gguf_cache.c`/`.h`, new TU.
   `<gguf_path>.beglin` sidecar written after first transcode, mmap'd on
   every subsequent load (`WT` fields point straight into the mapping,
   zero malloc/dequant/transcode on a cache hit); staleness detected via
   the source GGUF's own size+mtime, verified by actually `touch`ing the
   source and confirming a forced re-transcode, not assumed. Real
   measurement: cache-miss 7.94s vs cache-hit 0.76s wall-clock on the
   1.1GB fixture (~10.4×), byte-identical output either way.
   `QWEN_SME2_LAZY_REPACK` flipped default-on for the GGUF path (`if
   (!getenv(...)) g_sme2_lazy = 1;`) — without it, `kai_repack_all()`
   (never called on this path, by design, to avoid stacking an eager
   repack burst on top of the transcode work) would leave SME2 silently
   never engaging. Verifying that turned up a **real, pre-existing bug
   unrelated to this work**: `QWEN_SME2_LAZY_REPACK=1` combined with
   `serve` mode SIGILLs, reproduced identically on the completely
   unmodified `QWEN_INT4_BIN` path — the GGUF path's only supported
   modes (`greedy`/`bench`) run at M=1 and can't reach this bug class,
   so the fix ships safely; the bug itself is recorded, not fixed here
   (project memory `vdsp_sme2_lazy_repack_serve_sigill.md`). Full
   writeup: [`RESULTS.md`](RESULTS.md#general-purpose-loader--phase-2-sub-step-2b-on-disk-beglin-cache-real-speedup-measured-2026-08-25).
3. ✅ **DONE 2026-08-25** — Mistral-7B-v0.3 (D-gen-4 #1). Structural
   loading matched the "zero new engine code" prediction exactly
   (`ArchCfg` derived correctly with one allowlist entry added, no other
   loader change needed). But greedy output was initially degenerate --
   root-caused via direct `llama.cpp` source inspection (not guessed) to
   a real cross-architecture RoPE pairing convention difference this
   plan's "rotate_half, no rope scaling" framing didn't anticipate:
   `LLM_ARCH_QWEN2` uses NeoX/split-half pairing (what this engine has
   always implemented) but `LLM_ARCH_LLAMA` uses NORM/interleaved-pair
   -- GGUF's own tensor layout differs by architecture at the ggml
   level. Fixed with a one-flag branch in `rope_apply()`/`rope_head()`
   (`g_rope_norm`, set from the GGUF architecture string). Post-fix:
   fully coherent output, first 2 tokens exact-match upstream
   `llama.cpp` on the identical file before diverging to an equally
   fluent continuation (same *class* of quantization-noise divergence
   already documented for Qwen2, shorter prefix). Regression: Qwen2
   GGUF and the custom-format Llama-3.1-8B path (same HD=128/GROUP=4
   shape family, different loader) both confirmed byte-identical --
   this flag is provably inert for every path except newly-added
   `"llama"`-tagged GGUF loading. Full writeup:
   [`RESULTS.md`](RESULTS.md#general-purpose-loader--phase-2-sub-step-3-mistral-7b-v03-and-a-real-cross-architecture-rope-bug-2026-08-25).
4. ✅ **DONE 2026-08-25** — Startup log naming which tier (SME2-eligible/
   NEON-q4g64/NEON-q8g64/BLAS-f32) each tensor landed in, named honestly
   as an eligibility classification (`kai_route()`'s per-call `M >=
   kai_sme2_min_m()` check is dynamic, not decidable at load time). Real
   output on the fixture: `196 SME2-eligible, 0 NEON-q4g64, 1
   NEON-q8g64, 142 BLAS-f32`. Full writeup:
   [`RESULTS.md`](RESULTS.md#general-purpose-loader--phase-2-sub-step-4-startup-dispatch-tier-log-2026-08-25).

**Phase 2 status: all four sub-steps done, including a second new
architecture family (Llama, via Mistral-7B-v0.3) validated end-to-end.**
Transcode quantizer + policy table (sub-steps 1-1b), on-disk cache +
lazy-repack default (sub-step 2, hardened post-review), Mistral-7B-v0.3
validation (sub-step 3 -- download was blocked for a while on an
unauthenticated HuggingFace rate limit on `bob`, resolved with a
user-provided token; validation itself found and fixed a real
cross-architecture RoPE bug, see above), startup dispatch-tier log
(sub-step 4).

### Phase 3 — Dispatch-tier hardening across the shape ladder (~1 week)

Qwen2.5-0.5B/3B/7B, Llama-3.2-1B/3B. No new kernels — exercises HD=64,
GROUP∈{3,7,8}, and the generic-scalar attention tier that's existed since
D3 but never shipped. Expect diagnostic bugs, not math bugs. Wire the
Phase 0 M-threshold table into `kai_route`. Decide (with a measured
number, not a guess) whether HD=64 needs a new `attn_neon.h` family or
stays scalar.

**Sub-step 5 done (2026-08-26, see RESULTS.md's own section for the full
investigation)**: built the missing NEON-comparison bench Phase 0 deferred,
found and fixed 2 bugs in it (caller-plain SIGILL, wrong kernel family
measured), then resolved an apparent contradiction with `sme2_kai.h`'s
"M=1 is not a target" comment -- both were right, they were measuring
against *different* NEON kernels (`matmul_sdot`'s int8-SDOT fallback vs
`matmul_t`'s plain-fp32 fallback). `kai_sme2_min_m()=16` stays as-is for
`matmul_sdot` (confirmed correct). `matmul_t`'s floor=1 attempt first hit
a real SIGILL that 3 isolated repros couldn't reproduce -- root-caused
via an interactive `lldb` session (screen-shared into `bob`, since
non-interactive SSH `lldb` refuses debug permission on this hardware) to
an unrelated, previously-latent bug: `sme2_kai.c`'s bias-add loop
(shared by both `kai_sme2_gemm_f32` and `kai_sme2_gemm_f16lhs`)
autovectorized into illegal SVE, the same caller-plain-violation class
this file had already hit twice before, just missed on this
later-added loop. Fixed with the same `vectorize(disable)` pragma
pattern (this time correctly on the inner loop). `matmul_t` now runs
`kai_route_min(W, M, 1)` for real: verified byte-identical `spec`-mode
output vs baseline across 3 runs, **2.98x measured throughput gain**
(33.54 -> ~100 tok/s), and `matmul_sdot`'s already-shipped SME2 path
re-verified unaffected (`identical 1`).

**Sub-steps 1-4 done (2026-08-26)**: all 5 planned models validated
(Qwen2.5-0.5B/3B/7B, Llama-3.2-1B/3B) against `llama.cpp` (`llama-
simple` greedy, matching prompt IDs confirmed via `llama-tokenize`).
Every `HD`/`GROUP` combination the plan named now has a real-model
run through it. Two real gaps found and closed: `Q5_0` dequant
(Qwen2.5-0.5B's small-model quant recipe drops several tensor kinds
to `Q5_0`, not `Q4_K`) and `rope_freqs.weight` NTK scaling (Llama-3's
scaling has no dedicated GGUF KV keys -- llama.cpp ships a precomputed
per-frequency-pair correction tensor instead, confirmed from the real
file + `ggml-cpu/ops.cpp` source, wired onto this engine's existing
`g_rope_scale[]` mechanism rather than a new one). Qwen2.5-7B needed
`llama-gguf-split --merge` for its 2 shards (external tool, this
engine's own loader keeps its single-file design). Greedy exact-match
run length against `llama.cpp` scaled with model size (2 tokens for
the two smallest, 9-11 tokens for the two largest) across all 5 models
-- consistent evidence, not a single lucky result. Full detail in
RESULTS.md.

**Sub-step 6 done (2026-08-26)**: measured (not guessed) whether `HD=64`
needs a dedicated NEON attention kernel. `QWEN_PROF=1` `bench` profiling
on both real `HD=64` models: attention is 1.5-2.5% of per-token decode
time on both (FFN projections 55-63%, `lm_head` 20-32% are the real
cost). A perfect `HD=64` kernel caps the possible gain at that same
1.5-2.5% -- not worth the engineering cost. Decision: **stay scalar**,
no new kernel family. Phase 3 fully complete.

### Phase 4 — De-hardcode the MoE path; second MoE topology (~3 weeks, largest chunk, under-estimated by the original roadmap)

1. **DONE (2026-08-26)** — Convert every DeepSeek-dimensioned stack array to
   config-driven heap allocation (`alloc_moe_buffers()`, mirroring
   `alloc_arch_buffers()`), one array-family at a time (Groups A-H), re-
   verifying an 11-scenario golden gate (byte-identical or normalized-stderr
   -identical) after each of 11 steps, on bob (M4) with the real DeepSeek-
   V2-Lite weights. Found and fixed 6 real bugs along the way, the worst
   being an `MOE_SME2_SLOT_*` hardcoded-64 collision that would silently
   serve wrong expert weights for any future `N_EXPERTS >= 64` model. Full
   writeup: `RESULTS.md` "Phase 4 sub-part 1"; full plan:
   `/Users/xox/.claude/plans/serene-finding-ullman.md`.
2. **DONE (2026-08-27)** — Split MoE forward into attention × FFN via a
   `MOE_ATTN_KIND` dispatch seam and a new `moe_gqa_attention()` family.
   Correction: there is no dense GQA attention function to reuse (verified —
   dense GQA is written inline 4x, bound to `g_cfg` which the MoE path never
   populates); a new MoE-local GQA implementation was written instead,
   structurally mirroring the existing MLA functions. Numeric verification
   is deferred to sub-part 3 Step 3.2 (no second model exists yet to test
   against). Full writeup: `RESULTS.md` "Phase 4 sub-part 2"; full plan:
   `/Users/xox/.claude/plans/serene-finding-ullman.md`.
3. **DONE (2026-08-28)** — Shipped Qwen3-30B-A3B (chosen over Mixtral-8x7B:
   already downloaded, `hidden_size=2048` matches DeepSeek exactly, and a
   prior routing-concentration study predicted its narrower expert shape
   would favor the SME2 gather path more — confirmed: 3.036× at B=16 vs
   DeepSeek's 2.38×, ~27% better). Found and fixed a real bug sub-part 2 had
   shipped with (`MOE_KROW`/`MOE_VROW` never actually wired to the GQA
   branch its own comment promised). Full 48-layer forward vs a real MLX
   reference verified to machine precision (rel_l2 1.7e-7-5.4e-6 against an
   MLX run forced to float32) after tracing an initial ~1e-2-5e-2 gap
   entirely to MLX's own bf16 rounding, not this engine. Full writeup:
   `RESULTS.md` "Phase 4 sub-part 3"; full plan:
   `/Users/xox/.claude/plans/serene-finding-ullman.md`.
   **Prerequisite for item 4 below**: bob has ~33GiB free after this
   sub-part's ~19GB transfer — will not hold a further ~19GB GGUF source
   plus a ~17GB `.beglin` cache at the same time. Decide a cleanup plan
   first.
4. GGUF MoE tensors arrive stacked (`blk.N.ffn_gate_exps.weight`,
   `[n_expert,...]`) — maps onto the existing stacked-expert `MoeAFTensor`
   layout for shape fields, but **the `adj_bias` claim below is wrong,
   verified this session (sub-part 2/3 planning)**: GGUF's Q4_K uses a
   32-column affine granularity (two independent (scale,min) pairs per
   32-column half-block) vs this engine's 64-column groups, and this
   project's own GGUF loader already dequantizes Q4_K → fp32 →
   re-quantizes **symmetric** (`gguf_quantize_q4g64_error_feedback()`,
   oracle-verified) before it ever reaches `MoeAFTensor`. A GGUF-sourced
   expert tensor arrives symmetric (`bias ≡ -8*scale` exactly) — `adj_bias`
   is dead arithmetic and dead memory for it (~1.6% of every GEMM's MACs,
   ~1.81GB of zeros for Qwen3-30B-A3B), not a bridge to build. The correct
   design is an explicit `sym` flag on `MoeAFTensor`/`MoeSme2Slot` that skips
   the `adj_bias` allocation and correction loop entirely when set — a
   simplification, not the bridge this item originally described. Verify,
   don't assume.

### Phase 5 — Generalized export pipeline (Path B) + the bl=32 experiment (~2 weeks)

1. Ship the converter *in the npm package* (currently isn't). Generalize
   `export_weights.py` (add sharded-safetensors support), `emit_arch_config.py`
   (already generic), `quantize_int4.py` (parameterize the 3 name
   literals via the role table), `gptq_quantize_fold_g64.py` (config-driven
   layer-container lookup instead of `model.model.layers`).
2. Emit GGUF as the output format (not the custom layout) so beglin's own
   quantizer output loads through beglin's own general loader — custom
   format becomes legacy-only. Needs a `beglin.*`-namespaced GGUF KV
   extension for symmetric q4g64.
3. bl=32 experiment (flagged in D-gen-2): measure whether it enables
   near-direct Q4_K→qsi4c32 transcoding and what the doubled correction
   term does to the 2.38× number. Ship only if it wins.

### Phase 6 — Tokenizer (separately scoped, per D-gen-5)

## 4. Documentation debt to close alongside implementation

- `ROADMAP.md`'s COST section claimed the SME2 kernels are shape-fit to
  the 3 models — corrected (see §0, Correction 1) to: the wrapper is
  already shape-generic subject to `in%64==0`; the MoE *caller* is
  dimension-hardcoded; `out`-genericity is untested, not broken.
- `README.md`'s "Scope, honestly" section and missing-tooling note change
  at Phase 2 and Phase 5 respectively — update in the same commit as the
  code.
- Every decision above lands as a WHY/COST/EXIT record in the file it
  governs, matching the existing `D-f16lhs-1/2/3`, `D14`, `D-npm-1`
  convention.
