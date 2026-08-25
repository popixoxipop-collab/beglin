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

### Phase 0 — "Shape Soak" (~1 week, touches zero shipped code)

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

1. `gguf_load.c`/`gguf_load.h` — container parse, bounds-checked, FATAL
   on unrecognized (the `load_int4` doctrine).
2. Vendored dequant (`gguf_quants.c`, MIT, `VENDOR.md` entry).
3. `TensorRole` indirection replacing ~14 name literals / ~50 call sites
   with `wt_role(ROLE_ATTN_Q, l)`, resolved once at load via a
   `role → name pattern` table with `"hf"` and `"gguf"` naming
   conventions (mirrors `init_qkv_bias`'s existing one-dereference design).
4. `ArchCfg` from GGUF metadata (`%s.block_count`, `.embedding_length`,
   etc.) — do NOT add keys to `arch_config.txt` (qwen_score.c/qwen_spec.c
   each FATAL on unrecognized keys; RoPE scaling already lives in a
   separate optional sidecar for exactly this reason).
5. Architecture allowlist on `general.architecture` — FATAL with a named
   supported-list rather than attempting an unvalidated run.
6. Grow `g_wt[512]` to a `realloc`'d array.

**Fixture/gate:** `qwen2.5-1.5b-instruct-q4_k_m.gguf` — same model already
validated in the custom format, clean A/B. Metrics: R4 oracle diff empty;
greedy prefix-match vs. existing run; WikiText ppl delta vs. the existing
12.10 (tokens won't be bit-identical — different quantizer — so the ppl
delta IS the Path-A measured number).

### Phase 2 — Transcode + on-disk cache; first new model (~1.5 weeks)

1. Transcode stage: GGUF tensor → K_Q4G64/K_Q8G64/K_F32 via a per-role
   policy table.
2. **On-disk cache, decided now, not deferred**: naive transcode is a
   memory disaster (8B Q4_K = ~4.5GB mmap + ~4.5GB transcoded + up to
   ~3.7GB SME2 repack = 12+GB on a 16GB machine). Write a `<model>.beglin`
   sidecar on first load, mmap on subsequent loads. Flip
   `QWEN_SME2_LAZY_REPACK` to default-on for the GGUF path.
3. Ship Mistral-7B-v0.3 (D-gen-4 #1). Zero new engine code expected — if
   anything beyond the loader is needed, that's a Phase-1 defect, not a
   Phase-2 feature.
4. Startup log naming which tier (SME2/NEON-q4g64/NEON-q8g64/BLAS-f32)
   each tensor group landed in.

### Phase 3 — Dispatch-tier hardening across the shape ladder (~1 week)

Qwen2.5-0.5B/3B/7B, Llama-3.2-1B/3B. No new kernels — exercises HD=64,
GROUP∈{3,7,8}, and the generic-scalar attention tier that's existed since
D3 but never shipped. Expect diagnostic bugs, not math bugs. Wire the
Phase 0 M-threshold table into `kai_route`. Decide (with a measured
number, not a guess) whether HD=64 needs a new `attn_neon.h` family or
stays scalar.

### Phase 4 — De-hardcode the MoE path; second MoE topology (~3 weeks, largest chunk, under-estimated by the original roadmap)

1. Convert every DeepSeek-dimensioned stack array to `g_cfg`-style heap
   allocation, one at a time, re-verifying the R1 MoE golden output after
   each (this code has already stack-smashed once).
2. Split MoE forward into attention × FFN so non-MLA MoE (Mixtral,
   Qwen3-MoE) reuses dense GQA attention and supplies only routing +
   experts.
3. Ship Mixtral-8x7B or Qwen3-30B-A3B — puts the 2.38×-at-B=16 SME2
   result on a second, mainstream topology (this is what makes the
   *differentiator* generalize, not just the loader).
4. GGUF MoE tensors arrive stacked (`blk.N.ffn_gate_exps.weight`,
   `[n_expert,...]`) — maps onto the existing stacked-expert `MoeAFTensor`
   layout, and both are affine, so the existing `adj_bias` decomposition
   is the bridge. Verify, don't assume.

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
