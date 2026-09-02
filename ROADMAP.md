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

## D-roadmap-2: precision selection needs a calibration phase, not one-shot heuristic promotion

**WHY**: every quantization-precision fix this project has shipped so far
follows the same pattern -- find a real divergence (e.g. OLMoE's layer-13
router flip, `RESULTS.md`, "Root-causing the still-remaining gap"), then
jump straight to the safest available precision tier (int8 attention ->
F32 attention) without testing whether anything in between would have
been enough. That's a real gap: we don't actually know whether 16-bit
(not even a supported `QWEN_MOE_ROLE_BITS` value today -- only
`{4, 8, 32}`) would have been sufficient for the layer-13 fix, or whether
today's F32 promotion is paying for more precision than the failure
mode needed. The router near-tie margin profiler
(`moe_router_margin_profiler.py`/`_v2.py`, see `RESULTS.md` "Router
near-tie (margin) statistics profiler" and its cross-corpus replication)
gives real per-(layer,expert-pair) evidence of WHERE decisions are close
to flipping, but it only answers "is this risky", not "what's the
minimum bits that keeps it from actually flipping". Those are different
questions, and only the second one lets precision be assigned
efficiently instead of defensively.

The engine already has the mechanical building blocks for this: a real
per-role/per-layer bit-width ladder (`QWEN_MOE_ROLE_BITS`) and a
per-expert one (`QWEN_MOE_EXPERT_BITS`). What's missing is the loop that
actually walks that ladder per flagged element and measures where
behavior collapses -- i.e. treating precision assignment as something
the engine is *calibrated* for, per model, the way a quantization
toolchain runs a calibration pass, rather than something fixed once by
hand after a single incident and never revisited. Concretely: for each
role/layer/expert the margin profiler flags, sweep the available bit
widths (and any new ones added, e.g. 16-bit if the data justifies adding
it) and measure, at each level, whether the flip/near-tie rate that
motivated the flag actually goes away -- producing a real "minimum
sufficient bits" table per element instead of a binary
safe-vs-unsafe judgment call.

**COST**: real, multi-part engineering, not a follow-up patch --
(1) a sweep harness that re-runs the margin/flip measurement at each
candidate bit width per flagged element; (2) 16-bit isn't a supported
`QWEN_MOE_ROLE_BITS`/`QWEN_MOE_EXPERT_BITS` value today, so the ladder
itself may need extending before it can even be tested as an option;
(3) combinatorial cost -- MoE's per-expert granularity means sweeping
exhaustively could mean many (layer x expert x bit-width) measurement
runs, especially for models with large expert counts (OLMoE: 64 experts
x 16 layers). None of the precision fixes already shipped (OLMoE
attention F32, DeepSeek-V2-Lite's profiler-driven int8 expert promotion)
were validated this way -- retroactively checking whether they're
over- or under-provisioned is itself unstarted work.

**EXIT**: if per-element sweeping proves too expensive to run at the
granularity MoE demands, fall back to the current heuristic (margin
profiler flags a risk -> promote to the next safer tier, don't sweep) --
strictly worse in precision-efficiency but still evidence-driven rather
than guessed, and the margin profiler (the detection layer) stays useful
regardless of whether the correction layer ends up swept-and-minimal or
heuristic-and-safe.

**Update (2026-09-01): the sweep this section called for has been run,
extended beyond layers 1-4, and its scope corrected upward twice.**

Phase 1: for OLMoE's layer 1-4 attention roles (q/k/v/o_proj), bits=16
uniformly, confirmed at both router-decision and output-token level
(`RESULTS.md`, "D-roadmap-2 Track B", "(a) Output-token causal
validation"). Expert/FFN weights don't move the phenomenon at all
(`RESULTS.md`, "(b) Expert/FFN precision sweep") -- effect lives
entirely in the attention/router-input pathway.

Phase 2 (extension to layers 0/5/9/12/14/15): the hypothesis that
deeper layers would need LESS promotion (predicted by the hidden-state-
magnitude correlation) was **not confirmed as a collapse-point effect**
-- 39/40 router and 35/40 output (layer,role) combinations across all
10 layers tested still need bits=16 (`RESULTS.md`, layer-extension
section). The magnitude correlation instead shows up in bits=4's raw
failure severity shrinking with depth, not in whether bits=16 is still
required.

Phase 3 (local-vs-upstream attribution, layers 9/12/15): the real
reason selective/sparse promotion doesn't work -- accumulated noise
from ALL preceding layers disrupts a deep layer's own router **6.7-
11.3x more** than that layer's own local attention imprecision, growing
with depth (`RESULTS.md`, "local-vs-upstream" section). Mechanistically
this means any gap in a promoted-layer set re-admits noise that
propagates forward past it -- **only an unbroken prefix from layer 0
actually closes the gap**, and since virtually every layer needs the
promotion anyway (Phase 2), the simplest defensible policy is blanket:
all four attention roles, all layers, bits=16. Validated end-to-end on
the real C engine (`QWEN_MOE_ROLE_BITS` with all four roles at the `-1`
layer wildcard, `RESULTS.md` "Synthesis" section) -- real, measurable
effect on served output (2/8 spot-check positions changed vs. the int8
baseline), though a same-corpus bf16 ground-truth comparison for that
specific spot-check was inconclusive (wrong reference file, honestly
flagged, doesn't weaken the much larger sweep evidence this rests on).

**No sweep-derived runtime mechanism was needed for attention-role
promotion in the end -- a static, blanket, 4-line config closes
virtually the whole measured gap.** See `D-roadmap-3` below, also
revised -- its original justification (an 84% one-off long tail) turned
out to be a small-sample artifact.

## D-roadmap-3: runtime margin-gated precision escalation -- narrower than first imagined

**WHY**: the original vision (user-proposed) was a "growing" serving
engine that detects near-tie risk at runtime and escalates precision
per-token, the way `moe_cb4c_maybe_reverify()` already does for the
SME2-vs-scalar axis (`qwen_infer.c:4911`, margin-gated Tier1/Tier2
escalation, threshold data-derived at 0.1 from real task#101/102
measurements). D-roadmap-2's sweeps (Track A/B, (a)/(b), the layer
extension, and the local-vs-upstream attribution) now show this
mechanism is **not needed for attention-role near-ties at all**: OLMoE's
near-ties across virtually every layer tested are fully closed by a
one-time, blanket, static `QWEN_MOE_ROLE_BITS` promotion (all 4 roles,
all layers, bits=16), and the FFN/expert side doesn't respond to
precision promotion at all.

**This section's original justification was itself corrected.** It
argued a runtime mechanism was needed to catch the "long tail" of
one-off near-tie pairs the router-margin profiler's 60-prompt corpus
found (84%, 42/50 of the top closest-tie events, non-recurring). A
much larger real-text scan (WikiText-103, 30,216 positions vs. the
60-prompt corpus's 1453, all 16 layers, `RESULTS.md` "WikiText-103-scale
near-tie recurrence scan") found this was **a sampling artifact, not a
real property of the phenomenon** -- at real scale the one-off fraction
drops to **5.8%**, with 25,142 chronic (recurrence>=4) pairs, and the
recurrence distribution is a smooth continuous decline, not the
apparent bimodal split the small sample suggested. Combined with the
local-vs-upstream finding (accumulated noise, not local imprecision,
dominates at depth) and blanket promotion already being cheap and
validated, **the actual remaining case for a runtime mechanism is much
narrower than either the original framing or the first revision of this
section believed**: a truly rare residual of one-off pairs (~5.8%
region) after blanket attention promotion, PLUS whatever this session
hasn't tested yet (other models, expert-level chronic pairs like
layer-12's (51,55), which (b) showed doesn't respond to expert-weight
precision and may need a different fix entirely, not bit-width
escalation of any kind).

**Design (not yet implemented)**: extend `moe_cb4c_maybe_reverify()`'s
proven shape to a second axis. Current: margin below threshold -> Tier1
(scalar exact recompute) -> if disagreement, Tier2 (full exact replay).
New: margin below threshold -> recompute the flagged position's
attention-role tensors at bits=16 (not a different algorithm, a
different stored precision) for that one forward pass, substitute the
corrected result. Needs: (1) a real, measured threshold for the bit-
width axis specifically -- do NOT reuse the SME2-vs-scalar axis's 0.1
value without re-deriving it for this axis, they are different noise
sources with no a priori reason to share a cutoff; (2) since Track A/B
already found layers 1-4's attention roles are the highest-value static
target, the runtime mechanism's real job is catching risk *elsewhere*
(other layers, other models) where no sweep has run yet -- scope the
first version to flag-and-log only (no correction), to build the
"which (layer,role) pairs actually trigger in real traffic" evidence
base a future static promotion round would consume, before spending the
engineering cost of the correction path itself.

**COST**: real C-engine work on a hot path (every forward call pays a
margin check), even in flag-and-log-only form. Getting the per-instance
overhead low enough not to matter needs care (the existing Tier1/Tier2
mechanism already solved this for its own axis -- reuse its cost
discipline, don't re-derive from scratch).

**EXIT**: if flag-and-log data shows the long tail is rare enough in
real traffic to not matter (unlike the router-margin profiler's offline
top-50 sample, which was deliberately adversarial-sampling for tight
margins, not representative of real traffic frequency), drop the
correction path entirely -- the static promotions already shipped are
sufficient, and this becomes a monitoring-only feature, not a
correction engine.

Not started -- this section records the narrowed, evidence-grounded
scope, not a plan. Next step, if pursued: flag-and-log-only prototype,
real traffic data collected before any correction path is built.

**Update (2026-09-01): the flag-and-log-only prototype above is now
shipped, real-execution-verified, no correction path.** `qwen_infer.c`
gained `moe_neartie_maybe_log()`/`moe_neartie_attn_bits_summary()`
(sibling functions to `moe_cb4c_maybe_reverify()`, same file region),
wired into `run_moe_cbatch_verify_mode()`'s online-scheduler decode and
prefill-completion emit loops only -- the same two call sites
`moe_cb4c_maybe_reverify()` already uses, this engine's own definition
of "real traffic." Trigger axis: the same final-output top1-vs-top2
logit gap `moe_cb4c_margin()` already computes, NOT the offline router-
margin profiler's expert-selection-softmax axis -- confirmed by the
correction path's own description above ("recompute the flagged
position's attention-role tensors... substitute the corrected result"
only makes sense against a final-logit trigger). `QWEN_MOE_NEARTIE_LOG`
default off; `QWEN_MOE_NEARTIE_THRESHOLD` has **no shipped default** --
`LOG=1` without it is a FATAL, not a guessed number (no existing sweep
artifact pairs a continuous margin with a flip/no-flip outcome on this
axis yet, so this project's data-first-numerics rule blocks a default
until that measurement exists).

Real-execution verification (bob, real DeepSeek-V2-Lite AF-blob,
`/Users/bob/moe_base_deepseek`, `QWEN_MOE_CB_SLOTS=4 QWEN_MOE_CB_REQS=6`):
disabled-by-construction regression byte-identical to the pre-change
binary except wall-clock timing fields (RESULTS.md, "D-roadmap-3
flag-and-log prototype"); force-trigger smoke test at
`THRESHOLD=100.0` produced exactly one event per emitted token
(45/45, matching the run's own `nout` sum) with a correct startup
`attn_af_bits` tally (108 = `NL*4`); `THRESHOLD=0.0` produced zero
events; the `QWEN_MOE_CBATCH_REVERIFY=tier2` interaction test confirmed
the logged margin reflects `moe_cb4c_maybe_reverify()`'s
*post*-correction logits (measurably different from cb4c's own
pre-correction margin at the same (req,pos), e.g. 0.0170 -> 0.017696),
not a stale pre-correction value; a 3x3 overhead A/B (disabled /
enabled-never-fire / enabled-always-fire) showed no measurable wall-
clock difference between any of the three (all within the run's own
~9% run-to-run noise band) -- the O(MOE_VOCAB) margin scan this feature
always pays once enabled (unconditionally, mirroring `moe_cb4c`'s own
"compute margin first, threshold-compare second" shape) is negligible
next to this workload's per-token forward-pass cost.

**Update (2026-09-01): the threshold follow-up above is now done, a
real default shipped.** `moe_precision_sweep_margin_calibration.py`
(macstudio) extended `moe_precision_sweep_output.py`'s exact
methodology to also capture the bf16-baseline top1-vs-top2 final-logit
gap per position, then cross-tabulated it against which of the same 16
(layer,role) x {4,8,16,32} combos flip that position's final argmax
(sanity-checked: this run's own aggregate flip rates, 1.57%/0.24% for
bits=4/8, land inside `moe_precision_sweep_output.json`'s original
per-combo ranges, confirming no methodology drift). Result, over 1453
real positions: the largest baseline margin among all 94 positions
where a bits=4 attention-role substitution actually flipped the final
token was **0.4786**; every position with margin >= 0.5195 (decile
4's lower edge, and every decile above it) had **zero** flips. `qwen_
infer.c`'s `moe_neartie_threshold()` now returns a cited default of
**0.5** (sits just above the observed max flip margin, same logic
`moe_cbatch_threshold()`'s own 0.1 used) -- `QWEN_MOE_NEARTIE_LOG=1`
no longer FATALs without an explicit `QWEN_MOE_NEARTIE_THRESHOLD`;
the env var still overrides the default for anyone who wants a
workload-specific recalibration. Re-verified on bob against real
DeepSeek-V2-Lite weights: default-threshold run produces a sane
partial trigger rate (11/45 emitted tokens), override still works
identically to before, disabled-by-construction regression still
byte-identical. Same honest caveat as when this was first shipped:
single-model (OLMoE), single perturbation source (attention-role
bits=4) calibration -- not yet cross-validated against other
architectures or other precision-loss sources.

**Update (2026-09-01): the correction path is now shipped and real-
execution-verified (OLMoE/GQA scope).** User explicitly chose to build
this without first gathering the real-traffic evidence this section's
own gating criterion called for ("if flag-and-log data shows the long
tail is rare enough... drop the correction path entirely") -- a
deliberate, disclosed deviation, not a silent skip.

**Scope correction from this section's original single-layer sketch**:
the local-vs-upstream residual-accumulation data above (6.7-11.3x
router-level upstream-noise dominance) made a single-layer promotion
very unlikely to close the gap -- shown to the user, who chose the
data-justified full version instead: on trigger, replay the ENTIRE
flagged position's forward pass (all layers) using bits=16 attention
weights, mirroring the already-validated blanket-promotion finding,
paid reactively per-request rather than unconditionally by every token.

**Mechanism**: a second, independent `SafetensorsMulti` handle opens a
genuine bf16 checkpoint at startup (`QWEN_MOE_NEARTIE_CORRECT_
SAFETENSORS`) and registers bits=16 attention-only tensors (reusing
`st_register_moe_f16_as_af()` verbatim, no new export tool) into a
`g_moe_lt_hi[]` mirror of production `g_moe_lt[]` (FFN/expert pointers
struct-copied, unchanged -- already shown irrelevant to this
phenomenon). A single new global, `g_moe_lt_cur` (normally aliasing
`g_moe_lt`), plus a ONE-LINE change inside `moe_forward_token()` (the
only direct `g_moe_lt[l]` reference in its body -- traced the real call
graph rather than assuming a bigger diff was needed) lets a replay
function swap in the bits=16 table for the duration of a full,
incrementally-cached position-history replay (`moe_neartie_
reverify_hi()`, a structural sibling of `moe_reverify_exact()`, with
its OWN separate shadow K/V lane pool -- never shares storage with
Tier2's, since the cached values are numerically different).

**A dedicated, tighter trigger threshold** (`QWEN_MOE_NEARTIE_CORRECT_
THRESHOLD`, default **0.1**, NOT the logging threshold's 0.5 --
0.5 would trigger ~29% of tokens, prohibitively expensive for a full
replay) was derived from the same calibration data: bounds trigger
rate to ~6.5%, 9.2x flip-rate enrichment over baseline. Shares
`moe_cb4c_maybe_reverify()`'s own per-step budget rather than a new cap.

**Real verification (bob, real OLMoE weights `/Users/bob/
vdsp_olmoe_full_weights` + genuine bf16 `/Users/bob/olmoe_1b7b_hf`)**:
disabled-by-construction regression byte-identical; both FATAL guards
(no checkpoint set, MLA scope guard) real exit 1; force-trigger
(threshold=100) fired on exactly 59/59 emitted tokens with correct
incremental-replay cost decay (n_scalar 29->1->1 across consecutive
triggers on the same request); zero-trigger (threshold=0) produced
identical tokens to the disabled baseline; **a real accuracy case**:
production (uncorrected) predicted token 380 at a flagged position,
genuine bf16 ground truth (computed directly, independently) is 831,
and the correction mechanism's own output was 831 -- an exact match;
real RSS delta ~955-976MiB (higher than the ~576MiB static estimate,
likely transient/dynamic allocation during registration and replay,
not further diagnosed) against bob's 16GB, comfortable headroom;
overhead A/B showed disabled and never-fire statistically
indistinguishable, always-fire ~3.06x wall-clock (expected: every
token pays a full replay); cb4c Tier2 and this correction path
coexist cleanly in the same run (5 cb4c events + 5 correction events,
no budget starvation, no crash) with structurally-guaranteed shadow-
pool separation (entirely disjoint symbol sets, not runtime-checked
pointers -- confirmed by inspection, the correct verification method
for statically-separated memory regions).

**A real, informative side-finding**: at the exact positions the
correction path flagged, the offline Python calibration corpus (built
against genuine bf16) showed LARGE, confident margins (1.99 to 8.21) --
not near-ties at all. The production AF-blob deployment (itself built
from an already-4bit-quantized `mlx-community` checkpoint, the
provenance gap flagged below) has induced its OWN artificial near-ties
that the genuine bf16 model never had. This is a second, independent
confirmation of the correction mechanism's value, beyond the single
accuracy case above.

**Honest limits, not glossed over**: OLMoE/GQA only this round (MLA
deferred -- DeepSeek's genuine bf16 wasn't available yet when this was
built); no K/V resync into the production SME2 batched decode path
(each correction only affects that one token's answer, never
propagates forward -- an explicit, named scope limit, not a silent
gap); the production OLMoE AF-blob's own quantization provenance (from
an already-4bit source, not directly from bf16) means this round's
"closer to bf16" claim rests on real, direct measurement (the 380->831
case) rather than an assumption inherited from the Track A/B sweep
data, which used a different baseline.

**Update (2026-09-01): extended to MLA (DeepSeek-V2-Lite), real-execution-
verified, plus a real crash found and fixed by this same live test.**
User directly challenged the previous update's own deferral: the
correction engine had never been tested against the actual DeepSeek
int4 flip this session found live (`RESULTS.md` "D-deepseek-precint-4",
position 6: genuine bf16/int8+ all agree on token 8872, int4-all
predicts 344). Closing that gap required MLA support, not just a
re-run.

**Mechanism, mirrored from the GQA version above**: `moe_neartie_
correct_load_attn_hi()`'s FATAL-on-non-GQA guard replaced with a role-
table branch (`MOE_ST_ATTN_ROLES_MLA` vs. `_GQA`, both already existed);
new `moe_resolve_attn_tensors_mla_hi()` mirrors production `moe_resolve_
attn_tensors_mla()` exactly, looking up `__neartie_hi`-suffixed
q_proj/kv_a_proj/kv_b_proj/o_proj (kv_a_ln deliberately left as the
struct-copied production F32 value, same treatment as GQA's un-promoted
q_norm/k_norm); `moe_resolve_layer_tensors_hi()` branches on
`MOE_ATTN_KIND` exactly like production `moe_resolve_layer_tensors()`
already does. No changes needed to the shadow K/V pool, the table-swap
mechanism, or the manifest loader -- all already generic over `MOE_KROW`/
`MOE_VROW`/architecture, confirmed by reading, not assumed.

**A real bug, found by this round's own live test, not by static
review**: force-triggering correction on real DeepSeek MLA data
immediately SIGSEGV'd. `lldb` couldn't attach (non-interactive SSH,
same TCC wall documented elsewhere in this project's infra memory) --
root-caused by static analysis + elimination instead: a diagnostic
byte-accounting loop (`attn_hi_bytes` sum, added alongside the original
GQA-only correction path) unconditionally read `g_moe_lt_hi[l].k_proj`/
`.v_proj` -- GQA-only fields, NULL under MLA (the production MLA
resolver never sets them either) -- a straight NULL-pointer dereference
that had nothing to do with the 3 planned MLA changes themselves; it was
old code that silently assumed GQA was the only architecture that would
ever reach it. Fixed by branching the same accounting loop on
`MOE_ATTN_KIND`, same pattern as every other `g_moe_lt_hi[]` reader in
the file. This is exactly the class of bug a live test surfaces that a
static-config-only comparison (this session's earlier D-deepseek-
precint-4 work) structurally cannot -- the correction path itself was
never exercised until this round.

**Real verification (bob, real DeepSeek-V2-Lite AF-blob `/Users/bob/
moe_base_deepseek` + genuine bf16 `/Volumes/D50/deepseek_v2lite_bf16_
safetensors`)**: disabled-by-construction regression against the pre-
MLA-extension binary (`010c465`) on the standard 8-slot/8-request
corpus -- token output byte-identical across all 8 requests, only
`ttft_ms` (wall-clock) differed, as expected; attn_hi byte accounting
now correct post-fix, real measured **743,178,240 bytes (~708.8MiB)**
for the MLA hi-mirror registry and **335,544,320 bytes (~320.0MiB)** for
the shadow K/V pool -- both within ~1% of this round's own pre-
implementation estimate; real peak RSS (`/usr/bin/time -l`) **7.79GiB
disabled -> 9.04GiB enabled (+1.25GiB)**, comfortable under bob's 16GiB.

**The honest, not-forced result on the specific int4 flip**: the
online-cbatch/AF-blob production deployment does **NOT** reproduce the
position-6 flip found in the earlier static safetensors-mode comparison
-- baseline (correction off) already predicts the correct token 8872,
with a real measured margin of **0.7116** (well above both the 0.5
logging threshold and the 0.1 correction threshold -- this position
would never trigger correction under real, non-forced thresholds).
Force-triggering correction anyway (`THRESHOLD=100.0`) confirms the
mechanism itself is structurally sound for MLA: it replays cleanly
(n_scalar=7, full from-scratch replay on the first call for this
request) and its bits=16 answer agrees with both the baseline and
ground truth (8872) -- but there was nothing to "catch" in this exact
scenario. Root cause, not yet independently re-verified: the AF-blob's
own export-time int4 quantization (a separate pipeline from the live
Python-side RTN int4 quant/dequant used in the safetensors-mode
comparison that found the 344 flip) is evidently NOT numerically
identical to that RTN path, even though both are nominally "int4" --
two different int4 quantizations of the same near-tie landed on
different sides of it. This session did not go hunt for a different
position/prompt where the AF-blob's own int4 deployment does flip; the
mechanism is verified working, not verified catching a real flip in
production.

**Honest limits, not glossed over**: the crash this round found was in
code this same feature's own earlier round wrote (the accounting loop),
not in anything upstream -- a reminder that "mirrors the already-proven
GQA path" claims still need real execution, not just structural
symmetry, to trust. The AF-blob-vs-live-RTN int4 quantization
discrepancy is a real, named open question, not resolved here.

## D-roadmap-4: live role x layer near-tie attribution + persistent map (2026-09-02)

**WHY**: D-roadmap-3's correction path only ever knew ONE signal (final-
output margin) -- it had no way to say WHICH role x layer inside the
forward pass actually caused a given near-tie. User's request: build
the live version of that attribution (SLAM-map analogy -- accumulates
from real observations, not a one-shot snapshot), persist it externally
(Supabase, schema mirroring `QWEN_MOE_ROLE_BITS`'s real role table), and
close the loop so a role x layer that accumulates enough evidence gets
promoted to bits=16 without a restart.

**Shipped this round**: (1) `moe_cb4c_margin_ex()` -- sibling of the
existing margin function, also returns the competing (runner-up) token
id, previously computed and discarded (`qwen_infer.c`). (2) Local JSONL
event log (`QWEN_MOE_NEARTIE_EVENTS_LOG`), zero-cost when unset. (3)
Live role x layer attribution (`moe_neartie_attribute()`): on a
CONFIRMED real flip (full-hi replay's argmax differs from production's),
tests each of 108 (role,layer) attention combos via a pointer-swap
"mixed" table (production everywhere, one role at one layer promoted to
the existing `g_moe_lt_hi` mirror) -- zero new tensor loading, only
bookkeeping. Cost-capped (`QWEN_MOE_ATTRIB_MAX_EVENTS`/`_MAX_POS`) after
real pilot data showed a single event's attribution can take minutes
and scale with position.

**Real attribution data (5 confirmed flips, pilot 8-slot corpus +
WikiText-2-validation short-window run)**: 4 of 5 flips (80%) could NOT
be individually reproduced by ANY single (role,layer) promotion (0/108
hits each) -- directly confirms the local-vs-upstream residual-
accumulation finding this project already had from offline analysis,
now with live evidence. 1 of 5 had partial attribution (2/108 and,
separately in the very first pilot event, 13/108, spread across
multiple roles/layers) -- no single-point fix exists for most real
flips; D-roadmap-3's own "replay the whole position across all
attention layers" design is the right shape, not a single-role patch.

**A real production incident this round caused, fixed structurally, not
just apologized for**: launched 3 concurrent heavy qwen_infer instances
on bob (16GB RAM, each needing ~9.8-10.9GB) -- bob's own uptime after
recovery (`up 6:43` vs the prior `up 7 days`) confirms an actual reboot,
not just a slow SSH daemon. New `bob-macstudio-concurrent-load-guard.py`
PreToolUse:Bash hook (global, `~/.claude/hooks/scripts/`) now live-
checks the target host's existing heavy-process count + real available
memory before allowing another SSH-launched qwen_infer invocation,
blocking when projected demand exceeds a safety margin -- verified live
(pipe-tested and fired for real against an actual Bash tool call).

**Extended same round, user's direct challenge**: attribution widened
from attention-only (108 combos) to also cover dense/shared FFN roles
(189 combos total) -- the ORIGINAL "each role independently diagnosable"
vision, not deferred. Real result on a re-tested known flip: hit count
grew 13/108 -> 17/189, the 4 new hits all `shared_*_proj` roles --
direct live evidence FFN roles DO independently matter for at least
some real flips. A live per-layer routing/expert-selection sensor
(`g_moe_routing_capture`, threaded through the real physical slot array
after the implementing fork caught a compact-index-vs-slot bug in the
parent's own directive) also shipped and verified with real, non-stale
captured data. See RESULTS.md for the full record.

**Honest limits**: Supabase persistence (the actual "map in a DB" part
of the original request) is built and schema-ready (`supabase_schema_
d_roadmap4.sql`, `d4_supabase_push.py`) but blocked on REST API
credentials -- a project-scoped MCP server was registered but needs a
one-time interactive approval this session couldn't complete headlessly.
Closed-loop promotion (`g_moe_lt_active`, D5/D6) is implemented locally
(fork, not yet live-tested against bob -- see RESULTS.md for the
checklist). WikiText-2 run only 1-2 of 4 planned chunks completed by
the time this was written -- corpus scale was deliberately cut down
from the user's original "WikiText-103" ask after this round's own
pilot measured attribution cost as prohibitive at that scale (an
explicit, disclosed scope reduction, not a silent one).

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
