// qwen_infer.c -- from-scratch Qwen2.5-1.5B inference on Apple Accelerate/vDSP.
// No PyTorch / MLX / llama.cpp at inference. fp32 path (default) uses cblas_sgemv;
// with QWEN_INT4_BIN set, the 7 per-layer projection matvecs read PACKED int4
// (group-64) weights via q4gemv.h's NEON dequant-in-tile GEMV -- ~3.7x less
// per-token weight traffic (the storage->throughput task). embed/lm_head/norms/
// biases stay fp32 in both modes. RoPE=HF rotate_half, GQA 12q/2kv, KV cache.
//
// D5: RoPE = HF Qwen2 rotate_half, NON-interleaved (see prior comment).
// D12: int4 mode maps ONLY weights/qwen15b_int4g64.bin (1.67GB) indexed by
//      layout_int4.txt (name kind doff out in ng soff). fp32 mode maps
//      qwen15b_fp32.bin via layout.txt -- byte-identical to the pre-int4 engine.
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include "q4gemv.h"
#include "q4gemv_g256.h"   // Phase 2 (M36): g256sf kernels + pool glue (additive header)
#include "sme2_kai.h"      // SME2 integration Phase 3: runtime detection + one-time repack (additive header)
#include <arm_neon.h>

#include "attn_neon.h"
#include "rope_llama3_scale.h"   // M43: Llama-3.1-style RoPE NTK scaling (additive header)
#include "gguf_load.h"     // general-purpose-loader Phase 1: GGUF container parser (own TU --
                            // see gguf_load.h's own header comment for why it's never allowed
                            // to become part of this plain-compiled caller TU's actual logic,
                            // only called into)
#include "gguf_quants.h"   // general-purpose-loader Phase 1: vendored dequant (own TU, same reason)
#include "gguf_transcode.h" // general-purpose-loader Phase 2: RTN(+EF) transcode to K_Q4G64/K_Q8G64 (own TU, same reason)
#include "gguf_cache.h"     // general-purpose-loader Phase 2 sub-step 2: on-disk transcode cache (own TU, same reason)

// D1/D2 (structural generalization): NL/NH/NKV/D/HD/KVD/QD/IM/VOCAB/THETA/EPS/MAXSEQ/GROUP/
// KVG/QG used to be compile-time #defines for Qwen2.5-1.5B only. They are now loaded once at
// startup from a config sidecar (weights/arch_config.txt by default, QWEN_ARCH_CONFIG env
// override) into this struct by load_arch_cfg() below. HD/KVD/QD/GROUP/KVG/QG are DERIVED
// from the sidecar's NH/NKV/D fields (not stored redundantly in the file). attn_neon.h's
// AHD/AGROUP stay compile-time constants -- its kernels are hand-unrolled for exactly
// (128,6), not generic -- see the D3 dispatch gate (fast_attn_shape_ok(), used at every
// attn_neon.h fast-path call site) which replaces the old _Static_assert(HD==AHD,...) /
// _Static_assert(GROUP==AGROUP,...) compile-time checks with a runtime one, falling back to
// this file's own pre-existing generic scalar attn_mode()==0 path when the shapes don't match.
typedef struct {
    int nl, nh, nkv, d, hd, kvd, qd, group, im, vocab, maxseq, kvg, qg;
    float theta, eps;
    int qkv_bias;
} ArchCfg;
static ArchCfg g_cfg;
static inline int fast_attn_shape_ok(void) { return g_cfg.hd == AHD && g_cfg.group == AGROUP; }
// M44: split predicates for the GROUP=4 family. fast_attn_shape_ok() above stays byte-for-byte
// unchanged -- the quantized-KV FATAL guard (see main()) depends on its exact current meaning
// and none of the int8/int4-KV kernels have GROUP=4 counterparts (deliberately out of scope).
// fast_attn_hd_ok(): Variant A (attn_qk_neon/attn_wsum_neon, per-head) has zero dependence on
// GROUP at all -- it only ever needed HD to match. Splitting this out unlocks it for GROUP=4
// (and any future HD==128 shape) with no new kernel code, a gate-only fix.
// fast_attn_shape_ok_g4(): gates the new attn_qk_group_neon_g4/attn_wsum_group_neon_g4 pair
// (attn_neon.h), the genuine new kernel work -- Variant B is hand-unrolled per accumulator
// count, so GROUP must match exactly one of the families this file has kernels for.
static inline int fast_attn_hd_ok(void) { return g_cfg.hd == AHD; }
static inline int fast_attn_shape_ok_g4(void) { return g_cfg.hd == AHD && g_cfg.group == AGROUP4; }

// D1: parses "KEY VALUE\n" lines (fscanf loop, same style as load_int4() below). Default path
// is %base/weights/arch_config.txt (mirrors load_fp32's layout.txt / load_int4's layout_int4.txt
// convention); QWEN_ARCH_CONFIG overrides, mirroring QWEN_INT4_LAYOUT. FATALs loudly on a
// missing file or missing/malformed field -- same philosophy as load_int4's "unknown kind"
// hardening (a silently-wrong architecture config is worse than a crash).
static void load_arch_cfg(const char *base) {
    char path[512];
    const char *ov = getenv("QWEN_ARCH_CONFIG");
    if (ov && ov[0]) snprintf(path,sizeof path,"%s",ov);
    else snprintf(path,sizeof path,"%s/weights/arch_config.txt",base);
    FILE *f = fopen(path,"r");
    if (!f) { perror("arch_config"); fprintf(stderr,"FATAL: could not open arch config %s (set QWEN_ARCH_CONFIG or run eval/emit_arch_config.py)\n", path); exit(1); }
    char key[32]; double val;
    int have_nl=0,have_nh=0,have_nkv=0,have_d=0,have_im=0,have_vocab=0,have_theta=0,have_eps=0,have_maxseq=0,have_qkv=0;
    while (fscanf(f,"%31s %lf",key,&val)==2) {
        if (!strcmp(key,"NL")) { g_cfg.nl=(int)val; have_nl=1; }
        else if (!strcmp(key,"NH")) { g_cfg.nh=(int)val; have_nh=1; }
        else if (!strcmp(key,"NKV")) { g_cfg.nkv=(int)val; have_nkv=1; }
        else if (!strcmp(key,"D")) { g_cfg.d=(int)val; have_d=1; }
        else if (!strcmp(key,"IM")) { g_cfg.im=(int)val; have_im=1; }
        else if (!strcmp(key,"VOCAB")) { g_cfg.vocab=(int)val; have_vocab=1; }
        else if (!strcmp(key,"THETA")) { g_cfg.theta=(float)val; have_theta=1; }
        else if (!strcmp(key,"EPS")) { g_cfg.eps=(float)val; have_eps=1; }
        else if (!strcmp(key,"MAXSEQ")) { g_cfg.maxseq=(int)val; have_maxseq=1; }
        else if (!strcmp(key,"QKV_BIAS")) { g_cfg.qkv_bias=(int)val; have_qkv=1; }
        else { fprintf(stderr,"FATAL: arch_config %s has unrecognized key '%s'\n",path,key); exit(1); }
    }
    fclose(f);
    if (!(have_nl&&have_nh&&have_nkv&&have_d&&have_im&&have_vocab&&have_theta&&have_eps&&have_maxseq&&have_qkv)) {
        fprintf(stderr,"FATAL: arch_config %s missing required field(s) (need NL NH NKV D IM VOCAB THETA EPS MAXSEQ QKV_BIAS)\n",path); exit(1);
    }
    if (g_cfg.nh <= 0 || g_cfg.nkv <= 0 || g_cfg.nh % g_cfg.nkv != 0) {
        fprintf(stderr,"FATAL: arch_config NH=%d not a positive multiple of NKV=%d\n",g_cfg.nh,g_cfg.nkv); exit(1); }
    if (g_cfg.d <= 0 || g_cfg.d % g_cfg.nh != 0) {
        fprintf(stderr,"FATAL: arch_config D=%d not evenly divisible by NH=%d (cannot derive HD)\n",g_cfg.d,g_cfg.nh); exit(1); }
    g_cfg.hd = g_cfg.d / g_cfg.nh;
    if (g_cfg.hd % 2 != 0) { fprintf(stderr,"FATAL: arch_config HD=%d (D/NH) is odd -- RoPE rotate_half requires even head_dim\n",g_cfg.hd); exit(1); }
    g_cfg.kvd = g_cfg.nkv * g_cfg.hd;
    g_cfg.qd  = g_cfg.nh  * g_cfg.hd;
    g_cfg.group = g_cfg.nh / g_cfg.nkv;
    if (g_cfg.kvd % 64 != 0) { fprintf(stderr,"FATAL: arch_config KVD=%d (NKV*HD) not a multiple of 64 (group-64 KV kernels require this)\n",g_cfg.kvd); exit(1); }
    if (g_cfg.qd  % 64 != 0) { fprintf(stderr,"FATAL: arch_config QD=%d (NH*HD) not a multiple of 64 (group-64 kernels require this)\n",g_cfg.qd); exit(1); }
    g_cfg.kvg = g_cfg.kvd / 64;   // M23: int8-KV scale groups per position (group-64 along HD; 2 per head)
    g_cfg.qg  = g_cfg.qd  / 64;   // M23: int8 query scale groups per token
    if (g_cfg.nl <= 0 || g_cfg.maxseq <= 0) { fprintf(stderr,"FATAL: arch_config NL=%d MAXSEQ=%d must be positive\n",g_cfg.nl,g_cfg.maxseq); exit(1); }
    // M44: this ternary is now a 4-way ladder, not 2 -- fast_attn_shape_ok()/fast_attn_shape_ok_g4()
    // are defined above (right after fast_attn_shape_ok()), so no forward-declaration issue.
    // Caught during synthetic-model verification: leaving this a 2-way check would have silently
    // printed "generic scalar fallback" even while the GROUP=4 kernel path was actually live --
    // a diagnostic lie, not a correctness bug, but one this project's own culture treats as a
    // real defect to close, not leave for later.
    const char *attn_path_desc =
        fast_attn_shape_ok()    ? "(attn_neon.h fast path: GROUP=6 family)" :
        fast_attn_shape_ok_g4() ? "(attn_neon.h fast path: GROUP=4 family)" :
        fast_attn_hd_ok()       ? "(attn_neon.h Variant-A-only fast path -- no group-fused kernel for this GROUP)" :
                                   "(generic scalar attn fallback -- no specialized NEON kernel for this HD/GROUP)";
    fprintf(stderr,"[engine] arch config (%s): NL=%d NH=%d NKV=%d D=%d HD=%d IM=%d VOCAB=%d THETA=%.1f EPS=%g MAXSEQ=%d QKV_BIAS=%d GROUP=%d %s\n",
        path, g_cfg.nl,g_cfg.nh,g_cfg.nkv,g_cfg.d,g_cfg.hd,g_cfg.im,g_cfg.vocab,g_cfg.theta,g_cfg.eps,g_cfg.maxseq,g_cfg.qkv_bias,g_cfg.group,
        attn_path_desc);
}

// M43: Llama-3.1-style RoPE NTK scaling ("rope_scaling": {"rope_type": "llama3", ...} in the HF
// config -- Qwen2.5 has none). Separate, OPTIONAL sidecar (weights/rope_scaling.txt by default,
// QWEN_ROPE_SCALING_CONFIG overrides) rather than new keys in arch_config.txt: qwen_score.c/
// qwen_spec.c each have their own independent load_arch_cfg() that FATALs on any unrecognized
// key, so new keys in the file they already read would break both siblings even though RoPE
// scaling isn't implemented there (out of scope for this milestone, deliberately).
//
// Presence semantics are deliberately ASYMMETRIC vs. load_arch_cfg's always-mandatory file:
// default path absent -> not fatal (RoPE stays unscaled -- the deployed Qwen2.5-1.5B case);
// QWEN_ROPE_SCALING_CONFIG explicitly set but unopenable -> FATAL (an explicit override
// pointing nowhere is a near-certain typo, same "silently-wrong is worse than a crash"
// philosophy load_arch_cfg already states above).
typedef struct {
    int enabled;
    float factor, low_freq_factor, high_freq_factor, orig_max_pos;
} RopeScaleCfg;
static RopeScaleCfg g_rope_cfg;

static void load_rope_scale_cfg(const char *base) {
    char path[512];
    const char *ov = getenv("QWEN_ROPE_SCALING_CONFIG");
    int explicit_override = (ov && ov[0]);
    if (explicit_override) snprintf(path,sizeof path,"%s",ov);
    else snprintf(path,sizeof path,"%s/weights/rope_scaling.txt",base);
    FILE *f = fopen(path,"r");
    if (!f) {
        if (explicit_override) { perror("rope_scaling_config"); fprintf(stderr,"FATAL: QWEN_ROPE_SCALING_CONFIG=%s set but could not open it\n", path); exit(1); }
        g_rope_cfg.enabled = 0;   // default path absent -- not fatal, RoPE stays unscaled
        return;
    }
    double rope_type=0, factor=0, low_freq_factor=0, high_freq_factor=0, orig_max_pos=0;
    int have_type=0,have_factor=0,have_low=0,have_high=0,have_omp=0;
    char key[32]; double val;
    while (fscanf(f,"%31s %lf",key,&val)==2) {
        if (!strcmp(key,"ROPE_TYPE")) { rope_type=val; have_type=1; }
        else if (!strcmp(key,"FACTOR")) { factor=val; have_factor=1; }
        else if (!strcmp(key,"LOW_FREQ_FACTOR")) { low_freq_factor=val; have_low=1; }
        else if (!strcmp(key,"HIGH_FREQ_FACTOR")) { high_freq_factor=val; have_high=1; }
        else if (!strcmp(key,"ORIG_MAX_POS")) { orig_max_pos=val; have_omp=1; }
        else { fprintf(stderr,"FATAL: rope_scaling config %s has unrecognized key '%s'\n",path,key); exit(1); }
    }
    fclose(f);
    if (!(have_type&&have_factor&&have_low&&have_high&&have_omp)) {
        fprintf(stderr,"FATAL: rope_scaling config %s missing required field(s) (need ROPE_TYPE FACTOR LOW_FREQ_FACTOR HIGH_FREQ_FACTOR ORIG_MAX_POS)\n",path); exit(1);
    }
    if ((int)rope_type != 1) {
        fprintf(stderr,"FATAL: rope_scaling config %s has ROPE_TYPE=%d, only 1 (llama3) is implemented\n",path,(int)rope_type); exit(1);
    }
    if (high_freq_factor == low_freq_factor) {
        fprintf(stderr,"FATAL: rope_scaling config %s has HIGH_FREQ_FACTOR==LOW_FREQ_FACTOR (%.6g) -- the smooth-blend denominator would be zero\n",path,low_freq_factor); exit(1);
    }
    if (factor <= 0 || orig_max_pos <= 0) {
        fprintf(stderr,"FATAL: rope_scaling config %s has non-positive FACTOR=%.6g or ORIG_MAX_POS=%.6g\n",path,factor,orig_max_pos); exit(1);
    }
    g_rope_cfg.enabled = 1;
    g_rope_cfg.factor = (float)factor;
    g_rope_cfg.low_freq_factor = (float)low_freq_factor;
    g_rope_cfg.high_freq_factor = (float)high_freq_factor;
    g_rope_cfg.orig_max_pos = (float)orig_max_pos;
    fprintf(stderr,"[engine] rope_scaling (%s): type=llama3 factor=%g low_freq_factor=%g high_freq_factor=%g orig_max_pos=%g\n",
        path, g_rope_cfg.factor, g_rope_cfg.low_freq_factor, g_rope_cfg.high_freq_factor, g_rope_cfg.orig_max_pos);
}

enum { K_F32 = 0, K_Q4G64 = 1, K_Q8G64 = 2, K_Q4G256SF = 3 };   // Phase 2 (M36): g256sf format
typedef struct {
    char name[96]; int kind;
    const float *f32;        // K_F32
    const uint8_t *packed;   // K_Q4G64 / K_Q4G256SF (same nibble layout)
    const float *scales;     // K_Q4G64 per-64 scales; K_Q4G256SF wsuper [out][in/256]
    const uint8_t *sub;      // K_Q4G256SF only: 6-bit sub-codes [out][in/64], one byte each
    int out, in, ng;         // ng: in/64 for g64 kinds, in/256 for K_Q4G256SF
    const void *kai_rhs;     // SME2 Phase 3: KleidiAI-repacked RHS, K_Q4G64 only. NULL means
                              // this tensor wasn't repacked (SME2 unavailable/off/shape/failure)
                              // -> callers must fall back to the existing NEON path for it.
    size_t kai_rhs_bytes;    // byte length of *kai_rhs (0 when kai_rhs is NULL)
    int kai_lazy_failed;     // QWEN_SME2_LAZY_REPACK prototype: sticky "repack tried and
                              // failed for this tensor" marker, so kai_route() doesn't
                              // retry a deterministically-failing shape on every call.
} WT;

static WT g_wt[512]; static int g_nwt = 0;
static uint8_t *g_bytes = NULL;      // mmap base
static q4pool g_pool; static int g_int4 = 0; static int g_int8_head = 0;

static WT *wt(const char *name) {
    for (int i = 0; i < g_nwt; i++) if (!strcmp(g_wt[i].name, name)) return &g_wt[i];
    fprintf(stderr, "FATAL: tensor not found: %s\n", name); exit(1);
}
static WT *wtl(const char *fmt, int layer) { char b[96]; snprintf(b,sizeof b,fmt,layer); return wt(b); }
static WT *wt_opt(const char *name) {   // NULL if absent (for the optional int8 lm_head)
    for (int i = 0; i < g_nwt; i++) if (!strcmp(g_wt[i].name, name)) return &g_wt[i];
    return NULL;
}
static const float *w(const char *name) {   // fp32 accessor (embed/norm/bias)
    WT *t = wt(name);
    if (t->kind != K_F32) { fprintf(stderr, "FATAL: %s not f32\n", name); exit(1); }
    return t->f32;
}
// D-gen-tensorrole-1 (general-purpose-loader Phase 1, PLAN_general_purpose_loader.md): centralizes
// the ~50 duplicated tensor-name-literal call sites that were scattered across this file's four
// near-identical forward-pass implementations (single-token decode, batched prefill, sdot-serve,
// sdot-cbatch each re-deriving "model.layers.%d.self_attn.q_proj.weight" etc. on every call) into
// one role->pointer cache, resolved once at load instead of on every call.
//   WHY: (1) safety -- a role's tensor-name PATTERN now has exactly one place to get right,
//        not 4 (and, once the GGUF loader lands, an eventual 2 conventions x N call sites
//        collapses to 2 table entries instead). (2) speed, as a side effect, not the point:
//        wtl()'s snprintf+linear-scan over g_wt[] previously ran on every token/every layer in
//        the hot decode path; this replaces every one of those with a single array index
//        resolved once at load. Mirrors this file's own established init_qkv_bias() pattern
//        (resolve once at load into a g_cfg.nl-sized array, every call site becomes one
//        dereference) rather than inventing a new convention.
//   COST: one more init-time pass over every layer (negligible -- runs once per process, not
//        per token); the cache arrays add g_cfg.nl * N_LAYER_ROLES pointers of memory (trivial,
//        a few KB even at NL=32).
//   EXIT: call sites read g_role_wt[ROLE_X][l] (or ->f32 for the two norm roles) instead of
//        wtl("...", l)/wlf("...", l) -- reverting means restoring those literal calls, no
//        weight-format or loading-mechanism change either way (this sits ONLY on top of the
//        existing wt()/wtl()/w() lookups, which still do the actual name->tensor resolution).
typedef enum {
    ROLE_ATTN_Q, ROLE_ATTN_K, ROLE_ATTN_V, ROLE_ATTN_O,
    ROLE_MLP_GATE, ROLE_MLP_UP, ROLE_MLP_DOWN,
    ROLE_INPUT_LN, ROLE_POST_ATTN_LN,
    N_LAYER_ROLES
} LayerRole;

// "hf" naming convention: matches this engine's existing custom weight-format tensor names,
// themselves copied verbatim from the source HF checkpoint's own tensor names by
// export_weights.py. A "gguf" convention table (blk.%d.attn_q.weight etc.) joins this one
// alongside, not replacing it, once the GGUF loader (PLAN_general_purpose_loader.md Phase 1
// remainder) needs to populate this same cache from a different naming source -- keeping that
// as a per-role table swap instead of another call-site hunt is the whole point of this cache
// existing before that work starts, not just for its own sake now.
static const char *ROLE_PATTERN_HF[N_LAYER_ROLES] = {
    [ROLE_ATTN_Q]       = "model.layers.%d.self_attn.q_proj.weight",
    [ROLE_ATTN_K]       = "model.layers.%d.self_attn.k_proj.weight",
    [ROLE_ATTN_V]       = "model.layers.%d.self_attn.v_proj.weight",
    [ROLE_ATTN_O]       = "model.layers.%d.self_attn.o_proj.weight",
    [ROLE_MLP_GATE]     = "model.layers.%d.mlp.gate_proj.weight",
    [ROLE_MLP_UP]       = "model.layers.%d.mlp.up_proj.weight",
    [ROLE_MLP_DOWN]     = "model.layers.%d.mlp.down_proj.weight",
    [ROLE_INPUT_LN]     = "model.layers.%d.input_layernorm.weight",
    [ROLE_POST_ATTN_LN] = "model.layers.%d.post_attention_layernorm.weight",
};

static WT **g_role_wt[N_LAYER_ROLES];   // each malloc'd to g_cfg.nl entries in init_tensor_roles()

// Singletons (not per-layer): resolved once instead of the 4x-duplicated lm_head fallback
// logic and repeated w("model.embed_tokens.weight")/w("model.norm.weight") calls that were
// previously re-run, identically, in every one of the four forward-pass functions.
static const float *g_role_embed;
static const float *g_role_final_norm;
static WT *g_role_lm_head;

static void init_tensor_roles(void) {
    for (int r = 0; r < N_LAYER_ROLES; r++) {
        g_role_wt[r] = malloc((size_t)g_cfg.nl * sizeof(WT*));
        if (!g_role_wt[r]) { fprintf(stderr, "FATAL: init_tensor_roles alloc failed\n"); exit(1); }
        for (int l = 0; l < g_cfg.nl; l++) {
            g_role_wt[r][l] = wtl(ROLE_PATTERN_HF[r], l);
            if ((r == ROLE_INPUT_LN || r == ROLE_POST_ATTN_LN) && g_role_wt[r][l]->kind != K_F32) {
                // Same check w()/wlf() already did for these two roles before this cache
                // existed -- preserved here so a bad tensor kind still FATALs at the same
                // point in startup, not later as a silent misread of ->f32 on non-f32 data.
                fprintf(stderr, "FATAL: %s not f32\n", g_role_wt[r][l]->name); exit(1);
            }
        }
    }
    g_role_embed = w("model.embed_tokens.weight");
    g_role_final_norm = w("model.norm.weight");
    // Verbatim port of the lm_head resolution logic that was previously duplicated at each of
    // the 4 forward-pass functions' end (D14's kind==K_Q8G64 check applies to g_int8_head
    // itself, set earlier in main() -- this only reproduces the existing fallback-to-embedding
    // logic those 4 sites shared).
    if (g_int8_head) g_role_lm_head = wt("lm_head.weight");
    else { WT *h = wt_opt("lm_head.weight"); g_role_lm_head = (h && h->kind == K_F32) ? h : wt("model.embed_tokens.weight"); }
}

static uint8_t *mmap_bytes(const char *path, long *out_bytes) {
    int fd = open(path, O_RDONLY); if (fd < 0) { perror("open bin"); exit(1); }
    struct stat sb; fstat(fd, &sb);
    void *p = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    close(fd); *out_bytes = sb.st_size; return (uint8_t *)p;
}

// D14 (M49, 2026-08-15): FATAL on a duplicate tensor name instead of silently letting wt()'s
// first-match linear scan pick one arbitrarily. WHY: a real bug in quantize_int4.py (fixed
// same milestone, see its own D14) emitted "lm_head.weight" twice into layout_int4_foldg64.txt
// -- once as q4g64 (from the generic per-tensor loop, which forgot to exclude the untied
// lm_head the way it already excluded the tied embedding), once as q8g64 (from the dedicated
// near-lossless int8-head block). wt() returned the first (q4g64) match on every call, so the
// engine silently served the coarse int4 copy on every forward pass while its own log claimed
// "lm_head=int8" (the int8-head detection only checked presence-by-name via wt_opt(), never
// t->kind -- also hardened below). This produced a real, reproducible +3.50% ppl gap against
// the Python-side reference that took a full milestone (M49) to root-cause. COST: a genuinely
// malformed/duplicated layout now hard-aborts instead of silently degrading -- there is no
// legitimate reason for two entries with the same tensor name, so there's no case this rejects
// that should have been allowed. EXIT: if a future format intentionally wants multiple
// candidate encodings for one logical tensor, that needs an explicit selection mechanism (e.g.
// a suffix or a preference order), not reliance on scan order -- change this check accordingly.
static void check_no_dup_name(const char *nm) {
    for (int i = 0; i < g_nwt; i++)
        if (!strcmp(g_wt[i].name, nm)) {
            fprintf(stderr, "FATAL: duplicate tensor name in layout: %s\n", nm); exit(1);
        }
}

// fp32 mode loader: layout.txt (name off numel ndim dims...)
static void load_fp32(const char *base) {
    char path[512]; long wb;
    const char *ov = getenv("QWEN_FP32_BIN");   // G1c: point fp32 BLAS path at the dq blob
    if (ov && ov[0]) { g_bytes = mmap_bytes(ov,&wb); }
    else { snprintf(path,sizeof path,"%s/weights/qwen15b_fp32.bin",base); g_bytes = mmap_bytes(path,&wb); }
    snprintf(path,sizeof path,"%s/weights/layout.txt",base);
    FILE *f=fopen(path,"r"); if(!f){perror("layout");exit(1);}
    char nm[96]; long off,numel; int nd;
    while (fscanf(f,"%95s %ld %ld %d",nm,&off,&numel,&nd)==4) {
        long dims[4]={0,0,0,0}; for(int k=0;k<nd&&k<4;k++) if(fscanf(f,"%ld",&dims[k])!=1) break;
        for(int k=4;k<nd;k++){long d;if(fscanf(f,"%ld",&d)!=1)break;}
        if (g_nwt >= 512) { fprintf(stderr,"FATAL: >512 tensors in layout\n"); exit(1); }
        if (off < 0 || (size_t)off + (size_t)numel*4 > (size_t)wb) {
            fprintf(stderr,"FATAL: %s extent past blob (off=%ld numel=%ld wb=%ld)\n",nm,off,numel,wb); exit(1); }
        check_no_dup_name(nm);
        WT *t=&g_wt[g_nwt++]; strcpy(t->name,nm); t->kind=K_F32;
        t->f32=(const float*)(g_bytes+off); t->out=(int)dims[0];
        t->in=(nd>=2)?(int)dims[1]:0; t->ng=0;
    }
    fclose(f);
    fprintf(stderr,"[engine] fp32 mode: %d tensors, %.2f GB\n", g_nwt, wb/1e9);
}
// SME2 Phase 5 (promoted default, end-to-end serve/cbatch measurement on real
// SME2 hardware showed no regression at any tested (model, batch) point and
// up to +37% aggregate throughput at large batch -- see
// results/RESULTS_SME2_INTEGRATION.md): when QWEN_SME2_REPACK is UNSET,
// default to kai_sme2_available() (a hardware fact) rather than a hardcoded
// 0/1 -- SME2-capable hosts (e.g. bob, Apple M4) now repack by default,
// non-SME2 hosts (e.g. this repo's actual production machine, Apple M1 Max)
// still don't, automatically, with zero code-path difference from before this
// promotion (kai_sme2_available() was already the gate kai_repack_all() checks
// first; only what an UNSET env var resolves to has changed). An explicit
// QWEN_SME2_REPACK=0 or =1 always wins over the hardware default either way.
static inline int sme2_repack_on(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("QWEN_SME2_REPACK");
        cached = (e && e[0]) ? (atoi(e) ? 1 : 0) : kai_sme2_available();
    }
    return cached;
}

// SME2 Phase 5 (promoted default, same reasoning as sme2_repack_on() above):
// QWEN_SME2 unset now also defaults to kai_sme2_available() rather than a
// hardcoded 0. Kept as a SEPARATE flag from QWEN_SME2_REPACK (not merged into
// one) so Phase 3's repack-only smoke-testing invariant still holds exactly
// as verified (QWEN_SME2_REPACK=1 + QWEN_SME2=0 still repacks-but-never-
// dispatches) -- promoting both flags' defaults together is what actually
// changes end-to-end behavior on SME2 hardware, but they remain two
// independently overridable knobs. Declared here (rather than next to
// kai_route(), which is defined after load_int4()) because kai_repack_all()'s
// summary log below reads it too.
static int g_sme2_gemm = -1;
static inline int sme2_gemm_on(void) {
    if (g_sme2_gemm < 0) {
        const char *e = getenv("QWEN_SME2");
        g_sme2_gemm = (e && e[0]) ? (atoi(e) ? 1 : 0) : kai_sme2_available();
    }
    return g_sme2_gemm;
}

// QWEN_SME2_LAZY_REPACK=1 -- memory-tradeoff prototype (default OFF, unset behaves
// byte-identical to before this flag existed): skip kai_repack_all()'s eager
// load-time repack of every K_Q4G64 tensor, and instead repack each tensor
// individually, on demand, inside kai_route() the first time THAT tensor is
// actually dispatched at M >= kai_sme2_min_m(). A run that never exercises the
// batch/prefill path (e.g. pure M=1 decode, which never routes to SME2 by
// construction -- see kai_sme2_min_m()) holds ZERO SME2 RHS-packed bytes
// instead of the full eager total (~3.7 GB measured on Llama-3.1-8B, see vdsp
// SME2 Ledger). Same "recompute instead of always holding" tradeoff shape as
// gradient checkpointing, but amortized per-tensor-per-process rather than
// per-step: repack cost (O(out*in) nibble permutation) is paid once for a
// tensor the first time it's actually used at qualifying M, then cached in
// kai_rhs exactly like the eager path caches it -- never repeated on later
// calls to that same tensor. Byte-identical either way: this changes ONLY
// when/whether a tensor's kai_rhs buffer gets populated, never what
// kai_sme2_repack_q4g64() puts in it.
static int g_sme2_lazy = -1;
static inline int sme2_lazy_on(void) {
    if (g_sme2_lazy < 0) {
        const char *e = getenv("QWEN_SME2_LAZY_REPACK");
        g_sme2_lazy = (e && atoi(e)) ? 1 : 0;
    }
    return g_sme2_lazy;
}

// SME2 Phase 4: scratch buffer for kai_sme2_gemm_f32()'s LHS-pack output,
// allocated once below (right after a nonempty repack pass) and reused by
// every kai_route()'d matmul_t/matmul_sdot call for the rest of the process.
// Declared here (not next to kai_route() in matvec_t's block, which is
// defined AFTER load_int4()/kai_repack_all() in this file) so kai_repack_all()
// can populate it without a forward declaration.
static void *g_kai_lhs_scratch = NULL;

// Bytes needed for g_kai_lhs_scratch, sized once for the largest M/in this
// process will ever pass through kai_route() (see kai_sme2_gemm_f32()'s
// contract in sme2_kai.h). Shared by both the eager path (kai_repack_all())
// and the QWEN_SME2_LAZY_REPACK path (kai_repack_one_lazy(), below) so
// there's exactly one scratch buffer for the whole process either way, just
// allocated at a different point in time. Idempotent: a no-op if already
// allocated (or if a prior attempt already failed and left it NULL -- retried
// on the next call in that case, same as the eager path's original behavior).
static void kai_ensure_lhs_scratch(void) {
    if (g_kai_lhs_scratch) return;
    int max_in = g_cfg.d > g_cfg.im ? g_cfg.d : g_cfg.im;
    size_t lhs_bytes = kai_sme2_lhs_scratch_bytes(Q4_SDOT_BMAX, max_in);
    if (lhs_bytes > 0) g_kai_lhs_scratch = aligned_alloc(64, (lhs_bytes + 63) & ~(size_t)63);
}

// QWEN_SME2_LAZY_REPACK companion to kai_repack_all(): when lazy mode is on,
// kai_repack_all() does nothing at load (see its early return below) and this
// function repacks exactly ONE tensor, called from kai_route() the first time
// that tensor is actually dispatched at M >= kai_sme2_min_m(). Mirrors
// kai_repack_all()'s per-tensor body exactly (same shape/size queries, same
// kai_sme2_repack_q4g64() call) -- produces byte-identical kai_rhs contents,
// only WHEN it's produced differs. Sticky-fails via kai_lazy_failed so a
// shape that will deterministically never repack isn't retried on every
// subsequent call to this tensor (matches the eager path's existing
// per-tensor-silent-fallback contract, just spread across calls instead of
// one load-time loop).
static void kai_repack_one_lazy(WT *W) {
    if (W->kai_rhs != NULL || W->kai_lazy_failed) return;
    size_t nb = kai_sme2_rhs_packed_bytes(W->out, W->in);
    if (nb == (size_t)-1) { W->kai_lazy_failed = 1; return; }
    kai_ensure_lhs_scratch();
    if (!g_kai_lhs_scratch) { W->kai_lazy_failed = 1; return; }
    void *dst = aligned_alloc(64, (nb + 63) & ~(size_t)63);
    if (!dst) { W->kai_lazy_failed = 1; return; }
    if (kai_sme2_repack_q4g64(W->out, W->in, W->packed, W->scales, dst, nb) != 0) {
        free(dst); W->kai_lazy_failed = 1; return;
    }
    W->kai_rhs = dst; W->kai_rhs_bytes = nb;
    if (getenv("QWEN_SME2_LAZY_LOG"))
        fprintf(stderr, "[engine] SME2 lazy: repacked %s on first use (%.3f MB)\n", W->name, nb/1e6);
}

// SME2 Phase 3/4: one-time post-load repack of every K_Q4G64 tensor into
// KleidiAI's RHS-packed format, when hardware supports it and the opt-in flag
// is set, plus (Phase 4) the matching LHS scratch buffer matmul_t/matmul_sdot's
// kai_route() branch will reuse on every subsequent call. Failure of any kind
// (shape rejected, allocation failure, repack failure) is per-tensor and
// silent-but-logged, not FATAL: a tensor left with kai_rhs==NULL simply keeps
// using the existing, already-correct NEON path -- there is always a working
// fallback, unlike an unrecognized layout `kind` (which has no safe
// interpretation at all and must FATAL, see the Phase 2 (M36) hardening note
// below). Defined before load_int4() (which calls it at the very end, after
// g_wt is fully populated) so no forward declaration is needed.
static void kai_repack_all(void) {
    if (!kai_sme2_available()) return;
    if (!sme2_repack_on()) return;
    if (sme2_lazy_on()) {
        fprintf(stderr, "[engine] SME2: QWEN_SME2_LAZY_REPACK=1 -- skipping eager load-time "
                         "repack, tensors repack individually on first qualifying dispatch\n");
        return;
    }
    size_t total_bytes = 0; int n_ok = 0, n_skip = 0;
    for (int i = 0; i < g_nwt; i++) {
        WT *t = &g_wt[i];
        if (t->kind != K_Q4G64) continue;
        size_t nb = kai_sme2_rhs_packed_bytes(t->out, t->in);
        if (nb == (size_t)-1) { n_skip++; continue; }
        void *dst = aligned_alloc(64, (nb + 63) & ~(size_t)63);
        if (!dst) { n_skip++; continue; }
        if (kai_sme2_repack_q4g64(t->out, t->in, t->packed, t->scales, dst, nb) != 0) {
            free(dst); n_skip++; continue;
        }
        t->kai_rhs = dst; t->kai_rhs_bytes = nb; total_bytes += nb; n_ok++;
    }
    if (n_ok > 0) {
        // Sized for the largest M/in this process can ever pass through
        // kai_route(): M up to Q4_SDOT_BMAX (matmul_sdot's B cap, larger than
        // matmul_t's MAXSPEC -- both defined later in this file, but
        // Q4_SDOT_BMAX itself comes from q4gemv.h, included at file-top, so
        // it's already visible here); in up to max(hidden dim, intermediate
        // dim) since those are the only two K values any K_Q4G64 tensor uses
        // (q/k/v/o/gate/up read g_cfg.d, down_proj reads g_cfg.im). See
        // kai_ensure_lhs_scratch() above -- same sizing, shared with the lazy path.
        kai_ensure_lhs_scratch();
        if (!g_kai_lhs_scratch) {
            // Scratch alloc failed -- every repacked tensor's kai_rhs must be
            // dropped back to NULL (not just left set), or kai_route() would
            // pass a NULL lhs_scratch into kai_sme2_gemm_f32(). No FATAL: the
            // NEON path never needed this buffer and stays fully available.
            for (int i = 0; i < g_nwt; i++) g_wt[i].kai_rhs = NULL;
            fprintf(stderr, "[engine] SME2: repacked %d/%d q4g64 tensors but LHS scratch alloc "
                             "failed -- reverting all to NEON\n", n_ok, n_ok + n_skip);
            return;
        }
    }
    fprintf(stderr, "[engine] SME2: repacked %d/%d q4g64 tensors (%.2f GB), %d skipped -> NEON. "
                     "GEMM dispatch is %s (QWEN_SME2=%s)\n",
            n_ok, n_ok + n_skip, total_bytes / 1e9, n_skip,
            (n_ok > 0 && sme2_gemm_on()) ? "ACTIVE" : "off", sme2_gemm_on() ? "1" : "unset/0");
}

// int4 mode loader: layout file (default %base/weights/layout_int4.txt, overridable via
// QWEN_INT4_LAYOUT so a blob and ITS layout are selected together -- Phase 2 (M36): the blob
// path was already env-selected (QWEN_INT4_BIN) but the layout path was hardcoded, so two
// coexisting blobs (g64 rollback + g256sf) could silently pair with the wrong manifest).
// Line format: name kind doff out in ng soff [soff2] -- the 8th field exists only for
// kind=q4g256sf (subcode array offset); the pre-existing 7-field kinds are unchanged.
static void load_int4(const char *base, const char *bin) {
    char path[512]; long wb;
    g_bytes = mmap_bytes(bin,&wb);
    const char *lay = getenv("QWEN_INT4_LAYOUT");
    if (lay && lay[0]) snprintf(path,sizeof path,"%s",lay);
    else snprintf(path,sizeof path,"%s/weights/layout_int4.txt",base);
    FILE *f=fopen(path,"r"); if(!f){perror("layout_int4");exit(1);}
    char nm[96],kind[16]; long doff,soff; int out,in,ng;   // kind[16]: "q4g256sf" is 8 chars + NUL
    while (fscanf(f,"%95s %15s %ld %d %d %d %ld",nm,kind,&doff,&out,&in,&ng,&soff)==7) {
        if (g_nwt >= 512) { fprintf(stderr,"FATAL: >512 tensors in layout_int4\n"); exit(1); }
        if (out < 0 || in < 0 || doff < 0) { fprintf(stderr,"FATAL %s bad dims/off\n",nm); exit(1); }
        check_no_dup_name(nm);
        WT *t=&g_wt[g_nwt++]; strcpy(t->name,nm); t->out=out; t->in=in; t->ng=ng;
        if (!strcmp(kind,"q4g256sf")) {
            // Phase 2 (M36): group-256/subfold-O4 -- packed nibbles (g64-identical layout) +
            // wsuper fp32 [out][in/256] at soff + sub-codes u8 [out][in/64] at soff2 (8th field).
            long soff2;
            if (fscanf(f," %ld",&soff2)!=1) { fprintf(stderr,"FATAL %s q4g256sf missing soff2 (8th field)\n",nm); exit(1); }
            if (in % 256 != 0) { fprintf(stderr,"FATAL %s in%%256!=0\n",nm); exit(1); }
            if (ng != in/256) { fprintf(stderr,"FATAL %s ng=%d != in/256=%d (kernel recomputes in/256)\n",nm,ng,in/256); exit(1); }
            if (doff % 64 || soff % 64 || soff2 % 64) { fprintf(stderr,"FATAL %s misaligned\n",nm); exit(1); }
            size_t pbytes=(size_t)out*(in/2), sbytes=(size_t)out*ng*4, cbytes=(size_t)out*(in/64);
            if (soff < 0 || soff2 < 0 || (size_t)doff+pbytes > (size_t)wb ||
                (size_t)soff+sbytes > (size_t)wb || (size_t)soff2+cbytes > (size_t)wb) {
                fprintf(stderr,"FATAL %s q4g256sf extent past blob (wb=%ld)\n",nm,wb); exit(1); }
            t->kind=K_Q4G256SF; t->packed=g_bytes+doff; t->scales=(const float*)(g_bytes+soff);
            t->sub=g_bytes+soff2;
        } else if (!strcmp(kind,"q4g64")) {

            if (in % 64 != 0) { fprintf(stderr,"FATAL %s in%%64!=0\n",nm); exit(1); }
            if (ng != in/64) { fprintf(stderr,"FATAL %s ng=%d != in/64=%d (kernel recomputes in/64)\n",nm,ng,in/64); exit(1); }
            if (doff % 64 || soff % 64) { fprintf(stderr,"FATAL %s misaligned\n",nm); exit(1); }
            size_t pbytes=(size_t)out*(in/2), sbytes=(size_t)out*ng*4;
            if (soff < 0 || (size_t)doff+pbytes > (size_t)wb || (size_t)soff+sbytes > (size_t)wb) {
                fprintf(stderr,"FATAL %s q4 extent past blob (wb=%ld)\n",nm,wb); exit(1); }
            t->kind=K_Q4G64; t->packed=g_bytes+doff; t->scales=(const float*)(g_bytes+soff);
        } else if (!strcmp(kind,"q8g64")) {
            if (in % 64 != 0) { fprintf(stderr,"FATAL %s in%%64!=0\n",nm); exit(1); }
            if (ng != in/64) { fprintf(stderr,"FATAL %s ng=%d != in/64=%d (kernel recomputes in/64)\n",nm,ng,in/64); exit(1); }
            size_t cbytes=(size_t)out*in, sbytes=(size_t)out*ng*4;   // int8: 1 byte/code
            if (soff < 0 || (size_t)doff+cbytes > (size_t)wb || (size_t)soff+sbytes > (size_t)wb) {
                fprintf(stderr,"FATAL %s q8 extent past blob (wb=%ld)\n",nm,wb); exit(1); }
            t->kind=K_Q8G64; t->packed=g_bytes+doff; t->scales=(const float*)(g_bytes+soff);
        } else if (!strcmp(kind,"f32")) {
            size_t numel=(in>0)?(size_t)out*in:(size_t)out;
            if ((size_t)doff+numel*4 > (size_t)wb) {
                fprintf(stderr,"FATAL %s f32 extent past blob (wb=%ld)\n",nm,wb); exit(1); }
            t->kind=K_F32; t->f32=(const float*)(g_bytes+doff);
        } else {
            // Phase 2 (M36) hardening: an unrecognized kind used to silently fall through to
            // the f32 branch, gated only by a byte-length bounds check -- for a tensor whose
            // fp32-interpreted extent happens to fit inside the blob that is silent weight
            // corruption, not a loud failure (independently confirmed against this code during
            // the Phase 2 review). Every kind must now be explicit; behavior for all
            // previously-valid layouts (q4g64/q8g64/f32) is unchanged.
            fprintf(stderr,"FATAL %s unknown kind '%s' in %s (recognized: q4g64 q8g64 q4g256sf f32)\n",nm,kind,path); exit(1);
        }
    }
    fclose(f);
    fprintf(stderr,"[engine] int4 mode: %d tensors, %.2f GB packed (layout %s)\n", g_nwt, wb/1e9, path);
    kai_repack_all();
}


// M19: QWEN_W4A8=1 routes the int4 projections to the int8-SDOT kernel (~6x the
// fp32-convert GEMV in microbench; ppl cost +0.2%, M19 step 1). Lazy-cached getenv,
// same pattern as attn_mode()/fused_dispatch(). Routes EVERY K_Q4G64/K_Q8G64 call
// through matvec_t to int8-SDOT -- including the QWEN_FUSED_DISPATCH qkv/gate+up
// blobs, which are themselves K_Q4G64 WTs dispatched via matvec_t. So W4A8 and
// fused-dispatch compose automatically (measured +30.8% together vs SDOT alone,
// M19 headroom). Deployed default is opt-in (unset); fast config = both env set.
static int g_w4a8 = -1;
static inline int w4a8_on(void) {
    if (g_w4a8 < 0) { const char *e = getenv("QWEN_W4A8"); g_w4a8 = (e && atoi(e)) ? 1 : 0; }
    return g_w4a8;
}

// sme2_gemm_on()/g_kai_lhs_scratch are declared earlier (right after
// sme2_repack_on(), before kai_repack_all() -- which populates the scratch
// buffer and logs sme2_gemm_on()'s value) so no forward declaration is
// needed there. kai_route() below is the only thing actually new at this
// point in the file.
//
// Five conditions, ALL required (AND) -- any one false means the existing,
// already-correct NEON path runs unchanged:
//   1. env flag (QWEN_SME2, defaults to condition 2's own value when unset --
//      see sme2_gemm_on() -- still independently forceable to 0 or 1)
//   2. hardware present (always false on macstudio/M1 Max)
//   3. format this repack/dispatch pair actually supports
//   4. this SPECIFIC tensor was successfully repacked (per-tensor fallback --
//      a shape rejection or repack failure on one tensor never blocks others)
//   5. M large enough to fill the kernel's row tile (mr) -- excludes M=1
//      decode by construction, not by remembering to special-case matvec_t
//
// QWEN_SME2_LAZY_REPACK addendum: when conditions 1-3 and 5 hold but kai_rhs
// is still NULL, that no longer means "give up and use NEON" by itself --
// under lazy mode it first means "repack this one tensor right now, then
// check again" (kai_repack_one_lazy() populates kai_rhs synchronously before
// this function returns). const is cast away deliberately: g_wt[] itself is
// never const, only this parameter is -- see kai_repack_one_lazy()'s own
// comment for why mutating kai_rhs here is safe.
static inline int kai_route(const WT *W, int M) {
    if (!(sme2_gemm_on() && kai_sme2_available() && W->kind == K_Q4G64 && M >= kai_sme2_min_m()))
        return 0;
    if (W->kai_rhs == NULL && sme2_lazy_on() && !W->kai_lazy_failed)
        kai_repack_one_lazy((WT *)W);
    return W->kai_rhs != NULL;
}

static void matvec_t(const WT *W, const float *x, const float *bias, float *y) {
    if (W->kind == K_F32) {
        if (bias) memcpy(y, bias, W->out*sizeof(float));
        cblas_sgemv(CblasRowMajor,CblasNoTrans,W->out,W->in,1.0f,W->f32,W->in,x,1,
                    bias?1.0f:0.0f,y,1);
    } else if (W->kind == K_Q8G64) {
        if (w4a8_on()) gemv_q8g64_sdot_mt(&g_pool, W->packed, W->scales, x, bias, y, W->out, W->in);
        else           gemv_q8g64_mt(&g_pool, W->packed, W->scales, x, bias, y, W->out, W->in);
    } else if (W->kind == K_Q4G256SF) {
        // Phase 2 (M36): W4A8-only by design -- no fp32-activation fallback kernel was built
        // for this format, so refuse loudly instead of inventing a slow path nobody validated
        // (the g64 blob + unset envs stays the live rollback for every non-W4A8 use).
        if (!w4a8_on()) { fprintf(stderr,"FATAL: K_Q4G256SF (%s) requires QWEN_W4A8=1 -- no fp32-activation path exists for g256sf\n", W->name); exit(1); }
        gemv_q4g256sf_sdot_mt(&g_pool, W->packed, W->scales, W->sub, x, bias, y, W->out, W->in);
    } else if (w4a8_on()) {
        gemv_q4g64_sdot_mt(&g_pool, W->packed, W->scales, x, bias, y, W->out, W->in);
    } else {
        gemv_q4g64_mt(&g_pool, W->packed, W->scales, x, bias, y, W->out, W->in);
    }
}


static void rmsnorm(const float *x, const float *g, float *y, int n) {
    float ss=0.0f; vDSP_svesq(x,1,&ss,n); float inv=1.0f/sqrtf(ss/n+g_cfg.eps);
    for(int i=0;i<n;i++) y[i]=x[i]*inv*g[i];
}
static void softmax_inplace(float *x, int n) {
    float mx; vDSP_maxv(x,1,&mx,n); float neg=-mx; vDSP_vsadd(x,1,&neg,x,1,n);
    int nn=n; vvexpf(x,x,&nn); float s; vDSP_sve(x,1,&s,n); float inv=1.0f/s; vDSP_vsmul(x,1,&inv,x,1,n);
}
// M15-C2: vectorized SwiGLU via vvexpf (same library call softmax_inplace already uses),
// replacing the scalar expf loop. Same-class fp32 noise vs scalar (ULP-level, same contract
// this project already accepted for softmax/attention reduction-order changes since M6/M7).
// Always called with n=IM. D2 bug fix: this used to be a static array hardcoded to the
// literal 8960 (Qwen2.5-1.5B's IM) instead of the IM symbol it was documented to track --
// silently correct only by coincidence, and a guaranteed overflow the instant IM becomes a
// runtime value >8960 (e.g. Llama-3.1-8B's IM=14336). Now heap-allocated from g_cfg.im.
static float *swiglu_tmp1, *swiglu_tmp2;
static void swiglu(const float *g, const float *u, float *out, int n) {
    vDSP_vneg(g, 1, swiglu_tmp1, 1, n);
    int nn = n; vvexpf(swiglu_tmp1, swiglu_tmp1, &nn);
    float one = 1.0f; vDSP_vsadd(swiglu_tmp1, 1, &one, swiglu_tmp1, 1, n);
    vDSP_vdiv(swiglu_tmp1, 1, g, 1, swiglu_tmp2, 1, n);   // vDSP_vdiv(divisor, dividend, result)
    vDSP_vmul(swiglu_tmp2, 1, u, 1, out, 1, n);
}
// M43: g_cfg.hd/2 floats, the multiplicative ratio applied to inv=theta^(-2i/hd) in rope_head/
// rope_precompute below. Populated once at startup by init_rope_scale() (called from main()
// after alloc_arch_buffers()) -- position-independent (pure function of index i and g_rope_cfg),
// so unlike rc/rs it is computed once for the whole program run, not once per token.
// Phase 2 sub-step 3 (Mistral-7B-v0.3 validation): two RoPE pairing conventions exist across
// architectures this project supports, verified by reading llama.cpp's own rope-type switch
// (src/llama-model.cpp) directly, not guessed: LLM_ARCH_QWEN2 -> LLAMA_ROPE_TYPE_NEOX (split-
// half pairing, v[i] with v[i+hd/2] -- what this file has always implemented, and what every
// model shipped before Llama-family GGUF support used); LLM_ARCH_LLAMA -> LLAMA_ROPE_TYPE_NORM
// (interleaved-pair, v[2i] with v[2i+1]). GGUF's own Q/K tensor bytes are laid out to match
// whichever convention ggml uses for that architecture, so loading a "llama"-arch GGUF and
// applying split-half rotation produces exactly what was observed: real computation, not a
// crash, but positionally scrambled -> degenerate repetitive greedy output. Root-caused via
// A/B against upstream llama.cpp on the identical file (see RESULTS.md), not assumed.
// Default 0 (NEOX): the custom binary-format path and QWEN2 GGUF loads never set this, so
// their behavior is byte-identical to before this flag existed.
static int g_rope_norm = 0;  // 0 = NEOX (split-half, existing), 1 = NORM (interleaved-pair)

static float *g_rope_scale;
static void init_rope_scale(void) {
    int half = g_cfg.hd/2;
    for (int i=0;i<half;i++)
        g_rope_scale[i] = g_rope_cfg.enabled
            ? rope_llama3_scale(g_cfg.theta, g_cfg.hd, g_rope_cfg.factor, g_rope_cfg.low_freq_factor,
                                 g_rope_cfg.high_freq_factor, g_rope_cfg.orig_max_pos, i)
            : 1.0f;
}
static void rope_head(float *v, int pos) {
    if (g_rope_norm) {
        for(int i=0;i<g_cfg.hd/2;i++){ float inv=powf(g_cfg.theta,-(2.0f*i)/g_cfg.hd); inv*=g_rope_scale[i]; float ang=pos*inv;
            float c=cosf(ang),s=sinf(ang); float a=v[2*i],b=v[2*i+1];
            v[2*i]=a*c-b*s; v[2*i+1]=b*c+a*s; }
        return;
    }
    for(int i=0;i<g_cfg.hd/2;i++){ float inv=powf(g_cfg.theta,-(2.0f*i)/g_cfg.hd); inv*=g_rope_scale[i]; float ang=pos*inv;
        float c=cosf(ang),s=sinf(ang); float a=v[i],b=v[i+g_cfg.hd/2];
        v[i]=a*c-b*s; v[i+g_cfg.hd/2]=b*c+a*s; }
}
// M15-C2: forward_token calls this 392x/token (14 heads x 28 layers) with the SAME pos every
// time -- HD/THETA are compile-time constants and pos is fixed for the whole call, so the 64
// (cos,sin) pairs are identical across every layer and every head. Precompute once per token
// (rope_precompute below), apply here with plain multiplies -- bit-identical by construction
// (same cosf/sinf inputs, just computed once instead of 392 times). rope_head (above) is kept
// untouched for forward_tokens (the batched path), not yet propagated here -- separate decision.
static inline void rope_apply(float *v, const float *rc, const float *rs) {
    if (g_rope_norm) {
        for(int i=0;i<g_cfg.hd/2;i++){ float c=rc[i],s=rs[i]; float a=v[2*i],b=v[2*i+1];
            v[2*i]=a*c-b*s; v[2*i+1]=b*c+a*s; }
        return;
    }
    for(int i=0;i<g_cfg.hd/2;i++){ float c=rc[i],s=rs[i]; float a=v[i],b=v[i+g_cfg.hd/2];
        v[i]=a*c-b*s; v[i+g_cfg.hd/2]=b*c+a*s; }
}
static inline void rope_precompute(int pos, float *rc, float *rs) {
    for(int i=0;i<g_cfg.hd/2;i++){ float inv=powf(g_cfg.theta,-(2.0f*i)/g_cfg.hd); inv*=g_rope_scale[i]; float ang=pos*inv;
        rc[i]=cosf(ang); rs[i]=sinf(ang); }
}

static float *g_kcache, *g_vcache;
static inline float *kslot(int l,int pos){ return g_kcache+((long)l*g_cfg.maxseq+pos)*g_cfg.kvd; }
static inline float *vslot(int l,int pos){ return g_vcache+((long)l*g_cfg.maxseq+pos)*g_cfg.kvd; }

// M12: mirrored cache, laid out [layer][kv_head][position][HD] instead of
// [layer][position][kv_head*HD]. For a fixed (l,kvh) consecutive positions are HD floats
// apart here (truly contiguous) vs KVD apart in kslot()/vslot() -- this makes lda==HD==n_cols
// for a BLAS GEMV, which M11's kslot()-based attn_head_fast (lda=KVD, strided) could not offer.
// Written alongside the original cache every step (small extra memcpy, both stay in sync) so
// both layouts remain available for A/B comparison without a recompile.
static float *g_kcache2, *g_vcache2;
static inline float *kslot2(int l,int kvh,int pos){ return g_kcache2+(((long)l*g_cfg.nkv+kvh)*g_cfg.maxseq+pos)*g_cfg.hd; }
static inline float *vslot2(int l,int kvh,int pos){ return g_vcache2+(((long)l*g_cfg.nkv+kvh)*g_cfg.maxseq+pos)*g_cfg.hd; }

// M23: QWEN_KV_INT8=1 -- int8 KV cache + int8-SDOT attention (default off). Completes the
// decode thesis: the KV cache and attention compute were the last fp32-domain hot-path
// component after M19/M20 moved every projection to int8-SDOT.
// M23-D1 (granularity): per-(position, group-64-along-HD) symmetric int8 -- 2 scales per
//   head per position for K and for V, produced by q4_quant_act_i8, the SAME quantizer the
//   W4A8 activation axis validated at +0.2% ppl. WHY: reuses a verified kernel verbatim
//   (zero new quantization code), is strictly finer than per-(head,position) (quality >=,
//   same storage class), and its 64-group layout is exactly the shape q8-style SDOT
//   scaling consumes. COST: 16 scale bytes/position/cache (KVD+KVG*4 = 272 B vs 1024 B
//   fp32 -> 3.76x smaller). EXIT: coarser per-(head,pos) scales would save 8 B/pos for
//   measurable quality risk -- not worth it.
// M23-D2 (RoPE ordering): K is quantized AFTER RoPE -- the rotation must act on the
//   real-valued K, so the cache stores quantized post-RoPE K exactly where the fp32 cache
//   stored post-RoPE K. V has no RoPE. The query is likewise quantized post-RoPE.
// M23-D5 (toggle default): off. Unset => every allocation, write, and attention read is
//   byte-identical to HEAD; the int8 caches are never allocated.
static int g_kv_int8 = -1;
static inline int kv_int8_on(void) {   // 0 off | 1 int8 storage + int8-SDOT scores | 2 int8 storage + fp32 scores
    if (g_kv_int8 < 0) { const char *e = getenv("QWEN_KV_INT8"); int m = e ? atoi(e) : 0;
        g_kv_int8 = (m >= 0 && m <= 2) ? m : 1; }
    return g_kv_int8;
}
static int8_t *g_kq, *g_vq;        // [NL][MAXSEQ][KVD] int8 codes
static float  *g_ksc, *g_vsc;      // [NL][MAXSEQ][KVG] fp32 scales
static inline int8_t *kqslot(int l,int pos){ return g_kq +((long)l*g_cfg.maxseq+pos)*g_cfg.kvd; }
static inline int8_t *vqslot(int l,int pos){ return g_vq +((long)l*g_cfg.maxseq+pos)*g_cfg.kvd; }
static inline float  *kscslot(int l,int pos){ return g_ksc+((long)l*g_cfg.maxseq+pos)*g_cfg.kvg; }
static inline float  *vscslot(int l,int pos){ return g_vsc+((long)l*g_cfg.maxseq+pos)*g_cfg.kvg; }
static int8_t *g_qq8; static float *g_qsc8;   // M23: per-token post-RoPE int8 query (single-stream)

// M23-D7 (bias-split KV quantization): measured root cause of the first-cut +2.7% storage
// ppl loss: Qwen2.5 k_proj.bias has per-channel outliers up to |316| (L0) / |101| (L1) vs
// median |b|~0.7 (read from the HF safetensors) -- one such channel poisons the abs-max
// scale of its whole 64-group, destroying the other 63 channels' precision. Both biases
// are KNOWN at load time and enter linearly, so they are removed EXACTLY, not clipped:
//   K: RoPE is linear => K_cached = RoPE(Wx+b) = RoPE(Wx) + RoPE_pos(b). The cache stores
//      int8(RoPE(Wx)) (outlier-free payload); the score adds the exact fp32 term
//      q . RoPE_t(b) back, computed by the EXISTING fp32 group kernel over a precomputed
//      rotated-bias table g_rbias[NL][MAXSEQ][KVD] (58.7 MB fp32). The table is sequence-
//      independent (shared across all serve/cbatch slots), so its cost does not scale
//      with B. COST: one extra fp32 QK group pass (attention is ~0.5-1.5% of decode).
//   V: no RoPE => V_cached = Wx + b_v. The cache stores int8(Wx); since softmax probs sum
//      to 1, sum_t p_t (V_t + b_v) = wsum_int8 + b_v -- the add-back is one vadd, exact
//      up to the |1 - sum p| ~ 1e-7 softmax normalization noise. Zero per-position cost.
// EXIT: free g_rbias + delete the two vsub/vadd lines; the pre-D7 with-bias behavior
// returns (kept measurable via git history, not a runtime toggle -- with-bias quant is
// strictly worse on both quality and speed axes, so no A/B value remains).
// D2: was static float g_scores_grp[GROUP][MAXSEQ] -- GROUP/MAXSEQ are runtime now, so file-scope
// static 2D arrays (illegal VLA) become one flat buffer + a row-address helper; every call site's
// old g_scores_grp[g] indexing becomes scores_grp_row(g), same float* row-pointer semantics.
static float *g_scores_grp_flat;  // M13 mode 3 (fused per-kv-group NEON) score rows, flat [group][maxseq]
static inline float *scores_grp_row(int g){ return g_scores_grp_flat + (long)g*g_cfg.maxseq; }
static float *g_rbias = NULL;                       // [NL][MAXSEQ][KVD] RoPE_pos(k_bias)
static const float **g_vbias_l;                  // per-layer v_proj.bias (fp32, mmap'd)
static const float **g_qbias_l, **g_kbias_l;      // D4: same pattern, q_proj/k_proj bias
static float *g_zero_bias;                        // D4: shared zero fill when QKV_BIAS=0
static inline float *rbslot(int l,int pos){ return g_rbias+((long)l*g_cfg.maxseq+pos)*g_cfg.kvd; }
static float *g_scores_grp2_flat;          // K-bias score-correction rows, flat [group][maxseq]
static inline float *scores_grp2_row(int g){ return g_scores_grp2_flat + (long)g*g_cfg.maxseq; }
static void kv_i8_init_bias(void) {
    // D4: g_kbias_l[l]/g_vbias_l[l] are already populated by init_qkv_bias() (called from main()
    // before this) -- real per-layer tensors when QKV_BIAS=1, or a shared zero buffer when
    // QKV_BIAS=0 (in which case rotating/adding zero is exactly correct "no bias" behavior, no
    // special-casing needed here). Was: a redundant local wlf() re-fetch of the same k_proj.bias
    // data and a self-assignment of g_vbias_l -- both replaced by reusing the shared arrays.
    g_rbias = malloc((long)g_cfg.nl*g_cfg.maxseq*g_cfg.kvd*sizeof(float));
    if (!g_rbias) { fprintf(stderr,"FATAL: M23-D7 rotated-bias table alloc failed\n"); exit(1); }
    float rc[g_cfg.hd/2], rs[g_cfg.hd/2];
    for (int pos=0;pos<g_cfg.maxseq;pos++) {
        rope_precompute(pos, rc, rs);               // same fn as runtime -> bit-identical rotation
        for (int l=0;l<g_cfg.nl;l++) {
            float *row = rbslot(l,pos);
            memcpy(row, g_kbias_l[l], g_cfg.kvd*sizeof(float));
            for (int h=0;h<g_cfg.nkv;h++) rope_apply(row+h*g_cfg.hd, rc, rs);
        }
    }
    fprintf(stderr,"[engine] M23-D7 rotated k-bias table: %.1f MB\n", (double)g_cfg.nl*g_cfg.maxseq*g_cfg.kvd*4/1e6);
}
// Shared write/attend helpers so all four sites (forward_token / forward_tokens / serve /
// cbatch) run ONE implementation. kq/ksc/vq/vsc: this position's cache slots. kqb/kscb/
// vqb/vscb: position-0 base pointers of this sequence's cache (strides KVD bytes / KVG
// floats). qf: fp32 post-RoPE query; qq/qsc: its int8 form (mode 1 only, else NULL).
static float *kv_i8_knb, *kv_i8_vnb;        // bias-free staging (single-threaded use)
static inline void kv_i8_write(int l, int pos, const float *kv_k_, const float *kv_v_,
                               int8_t *kq, float *ksc, int8_t *vq, float *vsc) {
    vDSP_vsub(rbslot(l,pos),1, kv_k_,1, kv_i8_knb,1, g_cfg.kvd);   // K - RoPE_pos(k_bias)
    vDSP_vsub(g_vbias_l[l],1, kv_v_,1, kv_i8_vnb,1, g_cfg.kvd);    // V - v_bias
    q4_quant_act_i8(kv_i8_knb, g_cfg.kvd, kq, ksc);
    q4_quant_act_i8(kv_i8_vnb, g_cfg.kvd, vq, vsc);
}
static inline void kv_i8_attn(int l, int n, const float *qf, const int8_t *qq, const float *qsc,
                              const int8_t *kqb, const float *kscb,
                              const int8_t *vqb, const float *vscb, float *ab, float scale) {
    for (int kvh=0;kvh<g_cfg.nkv;kvh++) {
        float *sc_ptrs[g_cfg.group], *sc2[g_cfg.group];
        for (int g=0;g<g_cfg.group;g++){ sc_ptrs[g]=scores_grp_row(g); sc2[g]=scores_grp2_row(g); }
        // M45: leaf-level GROUP=4 branch at each kernel call -- sc_ptrs/sc2 are already correctly
        // VLA-sized by g_cfg.group either way; only which function gets CALLED needs to branch
        // (float*[AGROUP] decays to float**, nothing type-enforced at the call boundary).
        if (kv_int8_on() == 1) {
            if (g_cfg.group == AGROUP4)
                attn_qk_group_i8_g4(qq+kvh*g_cfg.group*g_cfg.hd, qsc+kvh*g_cfg.group*2, kqb+kvh*g_cfg.hd, g_cfg.kvd,
                                    kscb+kvh*2, g_cfg.kvg, n, scale, sc_ptrs);
            else
                attn_qk_group_i8(qq+kvh*g_cfg.group*g_cfg.hd, qsc+kvh*g_cfg.group*2, kqb+kvh*g_cfg.hd, g_cfg.kvd,
                                 kscb+kvh*2, g_cfg.kvg, n, scale, sc_ptrs);
        } else {
            if (g_cfg.group == AGROUP4)
                attn_qk_group_i8f_g4(qf+kvh*g_cfg.group*g_cfg.hd, kqb+kvh*g_cfg.hd, g_cfg.kvd, kscb+kvh*2, g_cfg.kvg, n, scale, sc_ptrs);
            else
                attn_qk_group_i8f(qf+kvh*g_cfg.group*g_cfg.hd, kqb+kvh*g_cfg.hd, g_cfg.kvd, kscb+kvh*2, g_cfg.kvg, n, scale, sc_ptrs);
        }
        // M23-D7: exact K-bias score term q . RoPE_t(k_bias) (fp32 group kernel over the table)
        if (g_cfg.group == AGROUP4)
            attn_qk_group_neon_g4(qf+kvh*g_cfg.group*g_cfg.hd, rbslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, n, scale, sc2);
        else
            attn_qk_group_neon(qf+kvh*g_cfg.group*g_cfg.hd, rbslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, n, scale, sc2);
        for (int g=0;g<g_cfg.group;g++) vDSP_vadd(sc_ptrs[g],1, sc2[g],1, sc_ptrs[g],1, n);
        for (int g=0;g<g_cfg.group;g++) softmax_inplace(sc_ptrs[g], n);
        if (g_cfg.group == AGROUP4)
            attn_wsum_group_i8_g4(sc_ptrs, vqb+kvh*g_cfg.hd, g_cfg.kvd, vscb+kvh*2, g_cfg.kvg, n, ab+kvh*g_cfg.group*g_cfg.hd);
        else
            attn_wsum_group_i8(sc_ptrs, vqb+kvh*g_cfg.hd, g_cfg.kvd, vscb+kvh*2, g_cfg.kvg, n, ab+kvh*g_cfg.group*g_cfg.hd);
        // M23-D7: V-bias add-back (softmax probs sum to 1)
        for (int g=0;g<g_cfg.group;g++)
            vDSP_vadd(ab+kvh*g_cfg.group*g_cfg.hd+g*g_cfg.hd,1, g_vbias_l[l]+kvh*g_cfg.hd,1, ab+kvh*g_cfg.group*g_cfg.hd+g*g_cfg.hd,1, g_cfg.hd);
    }
}

// M24: QWEN_KV_INT4=1 -- int4 KV cache (default off; unset => byte-identical to HEAD,
// and the M23 int8 path under QWEN_KV_INT8 is untouched). Pushes the M23 low-bit-KV
// thesis to 4 bits. int4 K is the known-hard part: per-token abs-max scales are
// poisoned by K's per-CHANNEL outliers -- M23-D7's bias-split removed the *bias*
// outliers exactly, but the remaining RoPE(Wx) payload still has channel structure
// that 15 levels cannot absorb the way 255 levels could.
// M24-D1 (K axis): per-CHANNEL asymmetric int4 (KIVI-style), one (scale, zero) fp32
//   pair per (channel, KV4_W-token block) from the block's per-channel min/max.
//   WHY: channel outliers hit every token at the same coordinate, so grouping ALONG
//   TOKENS puts all copies of an outlier under one scale instead of letting one
//   outlier flatten 63 well-behaved channels' precision. The M23-D7 bias-split is
//   KEPT: the cache stores the bias-free RoPE(Wx) payload; scores add the exact fp32
//   q.RoPE_t(b) term. Asymmetric (min zero-point) because per-channel distributions
//   need not be zero-centered within a block; symmetric would clip skewed channels.
//   COST: KVD (scale,zero) fp32 pairs per block = 32 B/pos amortized at KV4_W=64
//   (vs 16 B/pos int8 scales), plus the D3 staging ring. EXIT: per-token int4 K
//   (delete the block machinery) -- rejected upfront by the M23 outlier evidence.
// M24-D2 (V axis): per-TOKEN symmetric int4, group-64 along HD. V has no RoPE and no
//   channel-outlier pathology (M23 evidence: V needed no bias gymnastics beyond the
//   exact vbias add-back), and per-token quant needs no staging. Same granularity as
//   the int8 V path at half the code bytes. EXIT: fold V into the block machinery.
// M24-D3 (staging ring): the newest (pos % KV4_W) K rows of each (seq,layer) are not
//   yet block-quantized; they are held bias-free fp32 in a ring of KV4_RING=2*KV4_W
//   rows and scored by the EXISTING exact fp32 group kernel. WHY 2W not W: batched
//   writers (spec batch / chunked prefill, <=16 consecutive positions) write ALL
//   columns' KV before ANY column's attention runs; with a W-deep ring, writing
//   position p would clobber slot (p-W) while an earlier column's fp32 tail can
//   still legitimately need it (reachable at block-phase >= 49 with 16 columns).
//   With 2W the clobbered position is p-2W, provably below every live tail start.
//   Tail/causality rule: a column at position pos reads blocks only up to
//   nfull = floor((pos+1)/W)*W -- i.e. only blocks whose LAST member is <= pos --
//   so a block quantized by a later same-batch column (whose min/max embeds tokens
//   after pos) is never read by an earlier column. COST: 2W*KVD fp32 per (seq,layer)
//   staging (3.7 MB/seq) + <=W-1 tail positions at fp32 score cost per read.
//   EXIT: W-deep ring with write fencing, or int8-staged tail (adds a requant hop).
// M24-D4 (score path): QWEN_KV_INT4=1 -- int8xint4 SDOT: per block the per-channel K
//   scale folds into the QUERY (qt = q .* s, re-quantized int8 per group-64 ONCE per
//   block per head, amortized over KV4_W positions) and the zero-point enters as the
//   exact fp32 constant dot(q, z); per-position work is then the same combined-scale
//   SDOT structure as M23's attn_qk_group_i8. QWEN_KV_INT4=2 -- same stored bytes,
//   fp32-dequant scores (no qt quantization): isolates storage rounding from the
//   score-compute axis (M23 mode-2 attribution pattern). COST of mode 1: qt
//   re-quantization is a NEW error axis (q .* s spans the block's combined dynamic
//   range) -- measured head-to-head in the M24 gates. MEASURED (final format, D50
//   12x256, fp32-KV baseline 12.1213): mode 2 = 12.2040 (+0.68%, inside the ~1%
//   quality gate) vs mode 1 = 12.2555 (+1.11%, marginally outside). Mode 2 is the
//   RECOMMENDED setting (attention is ~0.5-1.5% of decode, so fp32 scores cost
//   nothing measurable: bench 50.6 tok/s mode 2 vs 48.6 mode 1 vs 45.8 int8 at
//   ctx~235, same-class); invalid env values fall back to mode 2 for this reason.
//   EXIT: mode 1 stays one env value away for the SDOT-score A/B.
// M24-D5 (toggle default + precedence): default off (unset => fp32 KV, bit-identical
//   to HEAD). If both QWEN_KV_INT4 and QWEN_KV_INT8 are set, int4 wins with a stderr
//   warning (checked once in main) -- running both is meaningless and the caches
//   never coexist.
// M24-D7 (V fp32 recent window -- the make-or-break lever, measured): first-cut int4
//   (K per-channel blocks + V per-token sym g64, mode 2 scores) measured +2.39% ppl
//   (12.4109 vs 12.1213, D50 12x256). Side-cache attribution isolated the axes:
//   int4-K + int8-V = 12.1510 (+0.25%, K nearly free) vs int8-K + int4-V = 12.3551
//   (+1.93%) -- V storage rounding dominates, NOT K and NOT the SDOT path (mode 1 -
//   mode 2 = +0.13%). V-format candidates, measured: asym g64 12.3429 (+1.83%),
//   sym g32 12.3154 (+1.60%), asym g32 12.3231 (+1.67%) -- grouping/asymmetry
//   insensitivity says the loss is spread 4-bit noise, not V outliers. What works is
//   KIVI's remaining ingredient, a full-precision RECENT WINDOW for V: keep the last
//   (pos % KV4_W .. up to ring depth) V rows bias-free fp32 in a second KV4_RING and
//   read them fp32 for t >= nfull (same causal horizon as K's staged tail), int4 for
//   t < nfull => 12.2040 (+0.68%), inside the ~1% gate. WHY it works: attention mass
//   concentrates on recent positions, so the positions carrying most probability are
//   served at fp32 while the long tail of old positions -- where int4 noise is
//   averaged over many small probs -- stays 4-bit. COST: one more 2W*KVD fp32 ring
//   (3.7 MB/seq) + <=2W-1 recent positions' V read at fp32; V codes+scales are still
//   written for every position (the window is a read-side policy, storage unchanged).
//   EXIT: read V int4 from position 0 (delete the ring + tail loop) and re-accept
//   +2.4%, or shrink the window and re-measure.
#define KV4_W 64                    // K block width along tokens (M24-D1/D3) -- engine-internal
                                     // algorithm constant, NOT a model architecture field; stays
                                     // compile-time (out of D1's sidecar scope, see plan D2)
#define KV4_RING (2*KV4_W)          // fp32 staging ring depth (M24-D3)
#define KV4_NB (g_cfg.maxseq/KV4_W) // blocks per (layer,seq) -- MAXSEQ is now runtime (D1); the
                                     // MAXSEQ%KV4_W==0 guard (bug fix 2 in the plan) lives at
                                     // KV4_NB's first use site (alloc_arch_buffers(), below)
static int g_kv_int4 = -1;
static inline int kv_int4_on(void) {   // 0 off | 1 int4 storage + SDOT scores | 2 int4 storage + fp32 scores
    if (g_kv_int4 < 0) { const char *e = getenv("QWEN_KV_INT4"); int m = e ? atoi(e) : 0;
        g_kv_int4 = (m >= 0 && m <= 2) ? m : 2; }   // invalid -> mode 2 (M24-D4)
    return g_kv_int4;
}
// M26-D2: error-feedback KV quantization (M25) is DEFAULT OFF. The M25 EF "benefit"
// was a regression artifact -- it was cancelling the systematic bias of the 0986a59
// activation-quant rounding regression; once M26-D1 restores correct rounding, EF has
// no bias to cancel and slightly HURTS ppl (int4 mode2 no-EF 12.1606 -> +EF 12.1686;
// mode1 12.1980 -> 12.2177). Deployed default is plain RTN int4-KV (no EF). Set
// QWEN_KV_EF=1 to re-enable the M25 residual carry (kept for the ablation).
static int g_kv_ef = -1;
static inline int kv_ef_on(void) {
    if (g_kv_ef < 0) { const char *e = getenv("QWEN_KV_EF"); g_kv_ef = (e && atoi(e)) ? 1 : 0; }
    return g_kv_ef;
}
static uint8_t *g_k4, *g_v4;        // [NL][MAXSEQ][KVD/2] packed nibble codes
static float *g_k4s, *g_k4z;        // [NL][KV4_NB][KVD] per-(block,channel) K scale/zero
static float *g_v4sc;               // [NL][MAXSEQ][KVG] per-token V group scales
static float *g_k4stg;              // [NL][KV4_RING][KVD] bias-free fp32 K staging ring
static float *g_v4stg;              // [NL][KV4_RING][KVD] bias-free fp32 V recent-window ring (M24-D7)
static inline uint8_t *k4slot(int l,int pos){ return g_k4+((long)l*g_cfg.maxseq+pos)*(g_cfg.kvd/2); }
static inline uint8_t *v4slot(int l,int pos){ return g_v4+((long)l*g_cfg.maxseq+pos)*(g_cfg.kvd/2); }
static inline float *k4s_l(int l){ return g_k4s+(long)l*KV4_NB*g_cfg.kvd; }
static inline float *k4z_l(int l){ return g_k4z+(long)l*KV4_NB*g_cfg.kvd; }
static inline float *v4sc_slot(int l,int pos){ return g_v4sc+((long)l*g_cfg.maxseq+pos)*g_cfg.kvg; }
static inline float *k4stg_l(int l){ return g_k4stg+(long)l*KV4_RING*g_cfg.kvd; }
static inline float *v4stg_l(int l){ return g_v4stg+(long)l*KV4_RING*g_cfg.kvd; }

// V per-token symmetric int4 quantizer (M24-D2): group-64 abs-max, c = round(x/s)
// clamped to [-7,7], stored c+8 in the split nibble layout (byte j of a group's 32
// bytes: lo = ch j, hi = ch j+32 -- see attn_neon.h M24 note). Scalar: KV-write was
// measured at <1% of decode (M15 prof), not worth NEON here.
static inline void kv4_quant_tok(const float *x, uint8_t *packed, float *sc) {
    // M25-D2 (V HD-axis error-feedback): choose codes sequentially along the 64 lanes
    // of each group, carrying the fp32 rounding residual r into the next lane
    // (q_j = round((x_j + r)/s), r += x_j - q_j*s). WHY: cancels the first-order
    // systematic component of the per-group quantization error so the weighted-V sum's
    // bias shrinks (the diagnostic showed greedy flips are a low-margin phenomenon;
    // reducing bias is the only lever that keeps the memory win). COST: write-time
    // only (<1% of decode, M15 prof); stored layout/scales/read path unchanged.
    // EXIT: if ppl doesn't improve vs M24, drop back to independent rounding.
    for (int gi = 0; gi < g_cfg.kvg; gi++) {
        const float *xp = x + gi*64; uint8_t *pp = packed + gi*32;
        float mx = 1e-8f; for (int j = 0; j < 64; j++){ float a = fabsf(xp[j]); if (a > mx) mx = a; }
        float s = mx/7.0f, inv = 7.0f/mx; sc[gi] = s;
        int c[64]; float r = 0.0f; int ef = kv_ef_on();   // M26-D2: EF default off
        for (int j = 0; j < 64; j++) {
            int q = (int)lrintf((xp[j] + (ef ? r : 0.0f))*inv);
            if (q < -7) q = -7; if (q > 7) q = 7;
            if (ef) r += xp[j] - (float)q*s;
            c[j] = q;
        }
        for (int j = 0; j < 32; j++)
            pp[j] = (uint8_t)((c[j]+8) | ((c[j+32]+8) << 4));
    }
}
// int4-KV write (all four sites): subtract the exact rotated-K/V biases (M23-D7 kept),
// quantize V per-token immediately, stage the bias-free K row, and when a KV4_W block
// completes, per-channel min/max quantize the whole block out of the staging ring.
// stg/k4b/k4sb/k4zb are this (seq,layer)'s staging ring / codes base / block scale &
// zero bases; v4row/v4scrow are this position's V slots.
static inline void kv_i4_write(int l, int pos, const float *kv_k_, const float *kv_v_,
                               float *stg, float *vstg, uint8_t *k4b, float *k4sb, float *k4zb,
                               uint8_t *v4row, float *v4scrow) {
    vDSP_vsub(rbslot(l,pos),1, kv_k_,1, kv_i8_knb,1, g_cfg.kvd);   // K - RoPE_pos(k_bias)
    vDSP_vsub(g_vbias_l[l],1, kv_v_,1, kv_i8_vnb,1, g_cfg.kvd);    // V - v_bias
    kv4_quant_tok(kv_i8_vnb, v4row, v4scrow);
    memcpy(stg + (long)(pos % KV4_RING)*g_cfg.kvd, kv_i8_knb, g_cfg.kvd*sizeof(float));
    memcpy(vstg + (long)(pos % KV4_RING)*g_cfg.kvd, kv_i8_vnb, g_cfg.kvd*sizeof(float));   // M24-D7 V window
    if ((pos % KV4_W) == KV4_W-1) {              // block complete -> per-channel quant (M24-D1)
        int b = pos / KV4_W, p0 = pos - KV4_W + 1;
        float *sB = k4sb + (long)b*g_cfg.kvd, *zB = k4zb + (long)b*g_cfg.kvd;
        for (int d = 0; d < g_cfg.kvd; d++) {
            float mn = 1e30f, mx = -1e30f;
            for (int t = 0; t < KV4_W; t++) {
                float v = stg[(long)((p0+t) % KV4_RING)*g_cfg.kvd + d];
                if (v < mn) mn = v; if (v > mx) mx = v;
            }
            sB[d] = (mx - mn) / 15.0f; zB[d] = mn;   // s==0 (constant channel) is valid: c=0, val=z
        }
        // M25-D1 (K token-axis error-feedback): per channel, quantize the block's
        // KV4_W token values in token order carrying the fp32 rounding residual
        // rK[d] into the next token (q = round((x_t + r - z)/s), r += x_t - xhat).
        // WHY: per-channel rounding bias is systematic across a block's tokens and
        // adds coherently into every q.K score; feeding the residual forward cancels
        // its first-order component while keeping byte layout, scales, and both
        // score paths (mode 1 SDOT / mode 2 fp32-dequant) untouched. Block
        // finalization already sees all KV4_W staged fp32 rows, so no cross-call
        // state is needed. COST: one KVD-float residual array at block close;
        // write-time only. EXIT: if ppl doesn't improve vs M24, revert to
        // independent rounding (honest null is acceptable). MEASURED (D50 12x256,
        // with M25-D2): mode2 12.2040 -> 12.1936, mode1 12.2555 -> 12.2439
        // (fp32-KV 12.1213) -- ~13%/9% of the int4 ppl gap closed. Varied greedy
        // first-flip did NOT move later (mode2 idx 4 -> 3, mode1 4 -> 4),
        // consistent with the flip being a low-margin phenomenon, not bias.
        float rK[g_cfg.kvd]; memset(rK, 0, sizeof rK); int ef = kv_ef_on();   // M26-D2: EF default off
        for (int t = 0; t < KV4_W; t++) {
            const float *row = stg + (long)((p0+t) % KV4_RING)*g_cfg.kvd;
            uint8_t *out = k4b + (long)(p0+t)*(g_cfg.kvd/2);
            for (int gi = 0; gi < g_cfg.kvg; gi++) {
                const float *sp = sB+gi*64, *zp = zB+gi*64, *xp = row+gi*64;
                float *rp = rK + gi*64; uint8_t *pp = out+gi*32;
                for (int j = 0; j < 32; j++) {
                    int lo = 0, hi = 0;
                    if (sp[j] > 0.0f) {
                        lo = (int)lrintf((xp[j] + (ef ? rp[j] : 0.0f) - zp[j]) / sp[j]);
                        if (lo < 0) lo = 0; if (lo > 15) lo = 15;
                        if (ef) rp[j] += xp[j] - ((float)lo*sp[j] + zp[j]);
                    }
                    if (sp[j+32] > 0.0f) {
                        hi = (int)lrintf((xp[j+32] + (ef ? rp[j+32] : 0.0f) - zp[j+32]) / sp[j+32]);
                        if (hi < 0) hi = 0; if (hi > 15) hi = 15;
                        if (ef) rp[j+32] += xp[j+32] - ((float)hi*sp[j+32] + zp[j+32]);
                    }
                    pp[j] = (uint8_t)(lo | (hi << 4));
                }
            }
        }
    }
}
// int4-KV attention (all four sites, ONE implementation -- M23-D4 policy carried
// over as M24-D4): full blocks from int4 (SDOT mode 1 / fp32-dequant mode 2), the
// <KV4_W-position tail from the fp32 staging ring (ring slots for the tail are
// contiguous because KV4_RING=2*KV4_W and block starts are multiples of KV4_W),
// exact fp32 K-bias score term + V-bias add-back exactly as the int8 path (M23-D7).
static int8_t *kv4_qt8; static float *kv4_qtsc, *kv4_qtf, *kv4_Cg;
static inline void kv_i4_attn(int l, int n, const float *qf,
                              const uint8_t *k4b, const float *k4sb, const float *k4zb,
                              const float *stg, const float *vstg,
                              const uint8_t *v4b, const float *v4scb,
                              float *ab, float scale) {
    int nfull = (n / KV4_W) * KV4_W;             // causal block horizon (M24-D3)
    for (int kvh = 0; kvh < g_cfg.nkv; kvh++) {
        float *sc_ptrs[g_cfg.group], *sc2[g_cfg.group];
        for (int g=0;g<g_cfg.group;g++){ sc_ptrs[g]=scores_grp_row(g); sc2[g]=scores_grp2_row(g); }
        const float *qh = qf + kvh*g_cfg.group*g_cfg.hd;
        for (int b0 = 0; b0 < nfull; b0 += KV4_W) {
            const float *sB = k4sb + (long)(b0/KV4_W)*g_cfg.kvd + kvh*g_cfg.hd;
            const float *zB = k4zb + (long)(b0/KV4_W)*g_cfg.kvd + kvh*g_cfg.hd;
            const uint8_t *kB = k4b + (long)b0*(g_cfg.kvd/2) + kvh*(g_cfg.hd/2);
            float *bp[g_cfg.group]; for (int g=0;g<g_cfg.group;g++) bp[g] = sc_ptrs[g] + b0;
            if (kv_int4_on() == 1) {             // M24-D4 mode 1: fold s into q, SDOT codes
                for (int g=0;g<g_cfg.group;g++) {      // shared fold loop -- unchanged, not duplicated
                    vDSP_vmul(qh+g*g_cfg.hd,1, sB,1, kv4_qtf,1, g_cfg.hd);
                    q4_quant_act_i8(kv4_qtf, g_cfg.hd, kv4_qt8+g*g_cfg.hd, kv4_qtsc+g*2);
                    vDSP_dotpr(qh+g*g_cfg.hd,1, zB,1, &kv4_Cg[g], g_cfg.hd);   // exact zero-point term
                }
                if (g_cfg.group == AGROUP4)          // M45: GROUP=4 branch
                    attn_qk_i4_block_g4(kv4_qt8, kv4_qtsc, kv4_Cg, kB, g_cfg.kvd/2, KV4_W, scale, bp);
                else
                    attn_qk_i4_block(kv4_qt8, kv4_qtsc, kv4_Cg, kB, g_cfg.kvd/2, KV4_W, scale, bp);
            } else {                             // M24-D4 mode 2: fp32-dequant scores
                if (g_cfg.group == AGROUP4)          // M45: GROUP=4 branch
                    attn_qk_i4f_block_g4(qh, kB, g_cfg.kvd/2, sB, zB, KV4_W, scale, bp);
                else
                    attn_qk_i4f_block(qh, kB, g_cfg.kvd/2, sB, zB, KV4_W, scale, bp);
            }
        }
        if (n > nfull) {                         // fp32 staging tail (M24-D3)
            float *tp[g_cfg.group]; for (int g=0;g<g_cfg.group;g++) tp[g] = sc_ptrs[g] + nfull;
            if (g_cfg.group == AGROUP4)              // M45: GROUP=4 branch, reuses M44's kernel
                attn_qk_group_neon_g4(qh, stg + (long)(nfull % KV4_RING)*g_cfg.kvd + kvh*g_cfg.hd, g_cfg.kvd,
                                      n-nfull, scale, tp);
            else
                attn_qk_group_neon(qh, stg + (long)(nfull % KV4_RING)*g_cfg.kvd + kvh*g_cfg.hd, g_cfg.kvd,
                                   n-nfull, scale, tp);
        }
        // M23-D7: exact K-bias score term q . RoPE_t(k_bias) over the shared table
        if (g_cfg.group == AGROUP4)                  // M45: GROUP=4 branch, reuses M44's kernel
            attn_qk_group_neon_g4(qh, rbslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, n, scale, sc2);
        else
            attn_qk_group_neon(qh, rbslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, n, scale, sc2);
        for (int g=0;g<g_cfg.group;g++) vDSP_vadd(sc_ptrs[g],1, sc2[g],1, sc_ptrs[g],1, n);
        for (int g=0;g<g_cfg.group;g++) softmax_inplace(sc_ptrs[g], n);
        if (g_cfg.group == AGROUP4)                  // M45: GROUP=4 branch
            attn_wsum_group_i4_g4(sc_ptrs, v4b + kvh*(g_cfg.hd/2), g_cfg.kvd/2, v4scb + kvh*2, g_cfg.kvg, nfull,
                                  ab+kvh*g_cfg.group*g_cfg.hd);
        else
            attn_wsum_group_i4(sc_ptrs, v4b + kvh*(g_cfg.hd/2), g_cfg.kvd/2, v4scb + kvh*2, g_cfg.kvg, nfull,
                               ab+kvh*g_cfg.group*g_cfg.hd);
        for (int t = nfull; t < n; t++) {        // M24-D7: fp32 recent-window V read
            const float *vt = vstg + (long)(t % KV4_RING)*g_cfg.kvd + kvh*g_cfg.hd;
            for (int g=0;g<g_cfg.group;g++){ float pw = sc_ptrs[g][t];
                vDSP_vsma(vt,1,&pw,ab+kvh*g_cfg.group*g_cfg.hd+g*g_cfg.hd,1,ab+kvh*g_cfg.group*g_cfg.hd+g*g_cfg.hd,1,g_cfg.hd); }
        }
        for (int g=0;g<g_cfg.group;g++)                // M23-D7: V-bias add-back
            vDSP_vadd(ab+kvh*g_cfg.group*g_cfg.hd+g*g_cfg.hd,1, g_vbias_l[l]+kvh*g_cfg.hd,1, ab+kvh*g_cfg.group*g_cfg.hd+g*g_cfg.hd,1, g_cfg.hd);
    }
}

static float *xbuf,*hbuf,*attn,*obuf;
static float *mlpact,*mlpout,*scores;
// M16-C: q/kv_k/kv_v/gate/up become pointers, resolved ONCE at startup (init_fused_dispatch(),
// called from main() after weight load) to either their own per-tensor backing store (unfused,
// default) or into a fused contiguous blob (QWEN_FUSED_DISPATCH bits 1/2) built by repacking the
// mmap'd int4 weights at load time. Every downstream site (RoPE apply, kslot/vslot writes,
// attention reads, swiglu call) dereferences whatever these currently point to, unchanged --
// preserves pointer identity at every call site below.
static float *q_solo, *kv_k_solo, *kv_v_solo;
static float *gate_solo, *up_solo;
static float *qkv_fused_buf;
static float *gu_fused_buf;
static float *q, *kv_k, *kv_v, *gate, *up;
static double g_t_layers=0, g_t_logits=0, g_t_attn=0;   // bench phase timers (g_t_attn: attention-loop-only, diagnostic for long-context)
// M15 Phase A: fine-grained phase timers, gated by QWEN_FAST_ATTN-style eager-cached toggle so
// the default (unprofiled) path pays only one cached-int branch per phase, not a syscall.
static double g_t_emb=0, g_t_rms=0, g_t_q=0, g_t_k=0, g_t_v=0, g_t_rope=0, g_t_kvwrite=0,
              g_t_o=0, g_t_resid=0, g_t_gate=0, g_t_up=0, g_t_swiglu=0, g_t_down=0,
              g_t_headrms=0, g_t_headgemv=0, g_t_argmax=0;
static int g_prof = -1;
static inline int prof_on(void) {
    if (g_prof < 0) { const char *e = getenv("QWEN_PROF"); g_prof = (e && atoi(e)) ? 1 : 0; }
    return g_prof;
}
static inline double nowt(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

// Fast attention: replace O(pos) per-position vDSP_dotpr/vDSP_vsma calls with 2 cblas_sgemv
// calls per head, exploiting BLAS's leading-dimension (lda) parameter to read the EXISTING
// KV cache layout (position-major, KVD stride) as a strided [pos+1, HD] matrix -- no KV cache
// reorganization needed. Toggle via QWEN_FAST_ATTN=1 (default off; kept for A/B verification
// before flipping the default). D-ATTN: WHY -- attention is O(pos) per token while every other
// per-token cost is O(1); measured 1.7%->20.7% of layer time as context grows 64->1536 tokens,
// linearly, so at Qwen's supported 32K context this would dominate decode entirely. COST: BLAS's
// internal reduction order can differ from vDSP_dotpr's row-by-row scalar accumulation, a
// same-class (not bit-identical) floating-point difference, same as this project's established
// HF-vs-engine parity class. EXIT: set QWEN_FAST_ATTN=0 (or unset) to revert to the byte-identical
// original per-position loop.
// M12 update: M11 measured the above (lda=KVD, strided) as ~29% SLOWER than the naive loop,
// 5/5 reps, and hypothesized the real fix was a KV cache reorganized for per-kv-head contiguity
// (lda==HD, no stride). QWEN_FAST_ATTN=1 now dispatches to attn_head_fast() over the kslot2()/
// vslot2() mirror below (lda=HD) instead of the old kslot()/vslot() view (lda=KVD) -- testing
// that hypothesis directly. Re-measure before concluding either way; the untouched naive loop
// (QWEN_FAST_ATTN=0, default) remains the exactness reference.
// M13-D1: QWEN_FAST_ATTN is multi-valued (0 naive / 1 M12-BLAS-mirror / 2 NEON-per-head /
// 3 NEON-fused-group), not a new env var. WHY: one toggle namespace keeps the whole A/B/C/D
// matrix enumerable and keeps M11/M12's documented negative results reproducible at "=1";
// unknown/out-of-range values fall back to naive (0), so old scripts using booleans are safe.
// COST: every gate on this function must stay in sync with which modes actually need the
// M12 mirror cache (mode 1 only) -- audited below at every call site.
// EXIT: explicit QWEN_FAST_ATTN=0 is still byte-identical naive at every point in this file;
// deleting modes 1/2/3 (and attn_neon.h) has zero effect on that explicit-0 path.
// Post-M14 default promotion: after M13 (single-token) and M14 (batched + qwen_spec.c
// draft/target + spec_pipe concurrency) all independently validated mode 3 (fused per-KV-group
// NEON) with zero regressions across every code path this project has, the UNSET default moved
// from 0 (naive) to 3 (mode 3 was the stronger variant everywhere it was measured). Naive remains
// one explicit env var away (QWEN_FAST_ATTN=0) for exactness-reference/debugging.
static int g_attn_mode = -1;   // -1 = unresolved, else cached 0-3
static inline int attn_mode(void) {
    if (g_attn_mode < 0) {
        const char *e = getenv("QWEN_FAST_ATTN");
        int m = e ? atoi(e) : 3;
        g_attn_mode = (m >= 0 && m <= 3) ? m : 3;
    }
    return g_attn_mode;
}

// M16-C: QWEN_FUSED_DISPATCH -- load-time contiguous repack of q/k/v (bit 1) and/or gate/up
// (bit 2) int4 projections into one fused blob each, so their per-token GEMV becomes ONE
// gemv_q4g64_mt call instead of 3 (qkv) or 2 (gate+up). WHY: M16-C Phase 1 isolated per-call
// dispatch overhead (thread-pool wake/sync + dequant setup inside gemv_q4g64_mt) via a
// null-dispatch probe and a fused-shape simulation, independent of any new kernel code, and
// measured S = 28*(dqkv+dgu)/ms_tok in the pre-registered 3-5% band (RESULTS_QWEN_VDSP.md M16-C
// Phase 1). COST: pure repack, not a new kernel -- the per-row int4 dequant-dot is unchanged,
// just invoked with a bigger `out`; expected BIT-IDENTICAL (not same-class), unlike the BLAS/NEON
// reduction-order changes elsewhere in this file (verified by G-M16a). EXIT: QWEN_FUSED_DISPATCH=0
// (default) leaves every buffer/dispatch path byte-identical to pre-M16-C. Lazy-cached like
// attn_mode()/prof_on() -- qwen_infer.c's forward_token() is single-threaded at the toggle-read
// level (no background pthread, unlike qwen_spec.c's spec_pipe mode), so this is safe.
// Post-M27 default promotion: composing QWEN_FUSED_DISPATCH=3 with the M27 kernel fix was
// gated bit-identical (greedy diff-0, both test prompts x {fp32,int8,int4}-KV, 6/6; ppl
// 12.0705 exact under both configs -- extending M16-C's original fp32-KV-only gate) and
// measured a real, stable +19.7--20.5% speedup across two context lengths (interleaved,
// best-of-3/4, RESULTS_QWEN_VDSP.md M28). The UNSET default moves from 0 (unfused) to 3
// (both qkv and gate+up fused -- the only combination measured/gated for this promotion).
// Cost: +462MB RSS (M16-C, measured trivial: 0.7% of 64GB). Unfused remains one explicit
// env var away (QWEN_FUSED_DISPATCH=0) for exactness-reference/debugging.
static int g_fused_dispatch = -1;
static inline int fused_dispatch(void) {
    if (g_fused_dispatch < 0) {
        const char *e = getenv("QWEN_FUSED_DISPATCH");
        int m = e ? atoi(e) : 3;
        g_fused_dispatch = (m >= 0 && m <= 3) ? m : 3;
    }
    return g_fused_dispatch;
}
static void fused_fatal_mismatch(const char *what, int layer) {
    fprintf(stderr, "FATAL: M16-C repack byte-mismatch for %s layer %d (load-time exactness check failed)\n", what, layer);
    exit(1);
}
static WT *g_qkv_fused, *g_gu_fused;
static float *g_qkv_fused_bias_flat;
static inline float *qkv_fused_bias_row(int l){ return g_qkv_fused_bias_flat + (long)l*(g_cfg.qd+2*g_cfg.kvd); }
// Concatenate q/k/v (or gate/up) packed+scales rows into ONE fused WT, verifying via memcmp
// immediately after each memcpy that the concatenation is a byte-faithful copy -- not a
// numerical re-derivation -- before this blob is ever handed to a GEMV call.
static void build_fused_qkv(int l) {
    WT *tq = g_role_wt[ROLE_ATTN_Q][l];
    WT *tk = g_role_wt[ROLE_ATTN_K][l];
    WT *tv = g_role_wt[ROLE_ATTN_V][l];
    // Phase 2 (M36): fusion accepts uniform K_Q4G64 OR uniform K_Q4G256SF parts (the fused WT
    // inherits the parts' kind; scales rows are wsuper rows for g256sf, and the per-row
    // sub-code array is concatenated with the same byte-faithful memcpy+memcmp discipline).
    // Mixed kinds within one fused site are structurally invalid -- FATAL.
    int fkind = tq->kind;
    if (!((fkind == K_Q4G64 || fkind == K_Q4G256SF) && tk->kind == fkind && tv->kind == fkind)) {
        fprintf(stderr, "FATAL: M16-C qkv fusion requires uniform K_Q4G64 or K_Q4G256SF parts (layer %d)\n", l); exit(1); }
    if (tq->in != tk->in || tk->in != tv->in || tq->ng != tk->ng || tk->ng != tv->ng) {
        fprintf(stderr, "FATAL: M16-C qkv fusion shape mismatch (layer %d)\n", l); exit(1); }
    int in = tq->in, ng = tq->ng, out = tq->out + tk->out + tv->out;
    if (out != g_cfg.qd+2*g_cfg.kvd) { fprintf(stderr, "FATAL: M16-C qkv fused out=%d != QD+2*KVD=%d (layer %d)\n", out, g_cfg.qd+2*g_cfg.kvd, l); exit(1); }
    size_t row_pbytes = (size_t)(in/2), row_sbytes = (size_t)ng*sizeof(float);
    size_t row_cbytes = (size_t)(in/64);   // g256sf sub-code row stride (one byte per 64 weights)
    uint8_t *packed = aligned_alloc(64, (size_t)out*row_pbytes);
    float   *scales = aligned_alloc(64, (size_t)out*row_sbytes);
    uint8_t *sub = NULL;
    if (fkind == K_Q4G256SF) sub = aligned_alloc(64, (size_t)out*row_cbytes);
    if (!packed || !scales || (fkind == K_Q4G256SF && !sub)) { fprintf(stderr,"FATAL: M16-C qkv fused alloc failed (layer %d)\n",l); exit(1); }
    WT *parts[3] = {tq, tk, tv}; int orow = 0;
    for (int p=0;p<3;p++) {
        WT *t = parts[p];
        memcpy(packed+(size_t)orow*row_pbytes, t->packed, (size_t)t->out*row_pbytes);
        memcpy(scales+(size_t)orow*ng,          t->scales, (size_t)t->out*ng*sizeof(float));
        if (memcmp(packed+(size_t)orow*row_pbytes, t->packed, (size_t)t->out*row_pbytes) != 0) fused_fatal_mismatch("qkv-packed", l);
        if (memcmp(scales+(size_t)orow*ng, t->scales, (size_t)t->out*ng*sizeof(float)) != 0) fused_fatal_mismatch("qkv-scales", l);
        if (fkind == K_Q4G256SF) {
            memcpy(sub+(size_t)orow*row_cbytes, t->sub, (size_t)t->out*row_cbytes);
            if (memcmp(sub+(size_t)orow*row_cbytes, t->sub, (size_t)t->out*row_cbytes) != 0) fused_fatal_mismatch("qkv-subcode", l);
        }
        orow += t->out;
    }

    const float *bq = g_qbias_l[l];
    const float *bk = g_kbias_l[l];
    const float *bv = g_vbias_l[l];
    memcpy(qkv_fused_bias_row(l),                 bq, (size_t)tq->out*sizeof(float));
    memcpy(qkv_fused_bias_row(l)+tq->out,         bk, (size_t)tk->out*sizeof(float));
    memcpy(qkv_fused_bias_row(l)+tq->out+tk->out, bv, (size_t)tv->out*sizeof(float));
    if (memcmp(qkv_fused_bias_row(l), bq, (size_t)tq->out*sizeof(float)) != 0 ||
        memcmp(qkv_fused_bias_row(l)+tq->out, bk, (size_t)tk->out*sizeof(float)) != 0 ||
        memcmp(qkv_fused_bias_row(l)+tq->out+tk->out, bv, (size_t)tv->out*sizeof(float)) != 0) fused_fatal_mismatch("qkv-bias", l);
    g_qkv_fused[l].kind = fkind; g_qkv_fused[l].out = out; g_qkv_fused[l].in = in; g_qkv_fused[l].ng = ng;
    g_qkv_fused[l].packed = packed; g_qkv_fused[l].scales = scales; g_qkv_fused[l].sub = sub;
    snprintf(g_qkv_fused[l].name, sizeof g_qkv_fused[l].name, "qkv_fused.L%d", l);   // Phase 2
                                   // polish: fused WTs previously had an empty name, so any
                                   // FATAL/log mentioning W->name printed "()" -- cosmetic only

}
static void build_fused_gu(int l) {
    WT *tg = g_role_wt[ROLE_MLP_GATE][l];
    WT *tu = g_role_wt[ROLE_MLP_UP][l];
    // Phase 2 (M36): same uniform-kind extension as build_fused_qkv above.
    int fkind = tg->kind;
    if (!((fkind == K_Q4G64 || fkind == K_Q4G256SF) && tu->kind == fkind)) { fprintf(stderr,"FATAL: M16-C gu fusion requires uniform K_Q4G64 or K_Q4G256SF parts (layer %d)\n",l); exit(1); }
    if (tg->in != tu->in || tg->ng != tu->ng || tg->out != tu->out) { fprintf(stderr,"FATAL: M16-C gu fusion shape mismatch (layer %d)\n",l); exit(1); }
    int in = tg->in, ng = tg->ng, out = tg->out + tu->out;
    if (out != 2*g_cfg.im) { fprintf(stderr, "FATAL: M16-C gu fused out=%d != 2*IM=%d (layer %d)\n", out, 2*g_cfg.im, l); exit(1); }
    size_t row_pbytes = (size_t)(in/2), row_sbytes = (size_t)ng*sizeof(float);
    size_t row_cbytes = (size_t)(in/64);
    uint8_t *packed = aligned_alloc(64, (size_t)out*row_pbytes);
    float   *scales = aligned_alloc(64, (size_t)out*row_sbytes);
    uint8_t *sub = NULL;
    if (fkind == K_Q4G256SF) sub = aligned_alloc(64, (size_t)out*row_cbytes);
    if (!packed || !scales || (fkind == K_Q4G256SF && !sub)) { fprintf(stderr,"FATAL: M16-C gu fused alloc failed (layer %d)\n",l); exit(1); }
    memcpy(packed,                            tg->packed, (size_t)tg->out*row_pbytes);
    memcpy(packed+(size_t)tg->out*row_pbytes, tu->packed, (size_t)tu->out*row_pbytes);
    memcpy(scales,                     tg->scales, (size_t)tg->out*ng*sizeof(float));
    memcpy(scales+(size_t)tg->out*ng,  tu->scales, (size_t)tu->out*ng*sizeof(float));
    if (memcmp(packed, tg->packed, (size_t)tg->out*row_pbytes) != 0 ||
        memcmp(packed+(size_t)tg->out*row_pbytes, tu->packed, (size_t)tu->out*row_pbytes) != 0 ||
        memcmp(scales, tg->scales, (size_t)tg->out*ng*sizeof(float)) != 0 ||
        memcmp(scales+(size_t)tg->out*ng, tu->scales, (size_t)tu->out*ng*sizeof(float)) != 0) fused_fatal_mismatch("gu", l);
    if (fkind == K_Q4G256SF) {
        memcpy(sub,                          tg->sub, (size_t)tg->out*row_cbytes);
        memcpy(sub+(size_t)tg->out*row_cbytes, tu->sub, (size_t)tu->out*row_cbytes);
        if (memcmp(sub, tg->sub, (size_t)tg->out*row_cbytes) != 0 ||
            memcmp(sub+(size_t)tg->out*row_cbytes, tu->sub, (size_t)tu->out*row_cbytes) != 0) fused_fatal_mismatch("gu-subcode", l);
    }
    g_gu_fused[l].kind = fkind; g_gu_fused[l].out = out; g_gu_fused[l].in = in; g_gu_fused[l].ng = ng;
    g_gu_fused[l].packed = packed; g_gu_fused[l].scales = scales; g_gu_fused[l].sub = sub;
    snprintf(g_gu_fused[l].name, sizeof g_gu_fused[l].name, "gu_fused.L%d", l);   // Phase 2 polish, see qkv above

}

static void init_fused_dispatch(void) {
    int fd = fused_dispatch();
    if (fd != 0 && !g_int4) {
        const char *e = getenv("QWEN_FUSED_DISPATCH");
        if (e && e[0])   // M28: only warn if the user actually set it -- the promoted fp32
                         // default (fd=3, silently downgraded below) isn't a "request"
            fprintf(stderr, "[engine] QWEN_FUSED_DISPATCH=%d requested but not in int4 mode -- ignoring (fp32 path has no analogous per-call dispatch overhead)\n", fd);
        fd = 0;
        g_fused_dispatch = 0;  // keep the cached value forward_token() reads via fused_dispatch()
                                // in sync with this downgrade -- otherwise it still sees the raw
                                // nonzero env value and dispatches into never-built g_qkv_fused/
                                // g_gu_fused while q/kv_k/kv_v/gate/up point at the solo buffers.
    }
    if (fd & 1) {
        for (int l=0;l<g_cfg.nl;l++) build_fused_qkv(l);
        q = qkv_fused_buf; kv_k = qkv_fused_buf+g_cfg.qd; kv_v = qkv_fused_buf+g_cfg.qd+g_cfg.kvd;
    } else { q = q_solo; kv_k = kv_k_solo; kv_v = kv_v_solo; }
    if (fd & 2) {
        for (int l=0;l<g_cfg.nl;l++) build_fused_gu(l);
        gate = gu_fused_buf; up = gu_fused_buf+g_cfg.im;
    } else { gate = gate_solo; up = up_solo; }
    if (fd) fprintf(stderr, "[engine] QWEN_FUSED_DISPATCH=%d: qkv=%s gu=%s (load-time repack verified byte-identical)\n",
                    fd, (fd&1)?"fused":"solo", (fd&2)?"fused":"solo");
}
static void attn_head_fast(const float *kv_base, int lda, const float *qh, float *scores, int n, int hd,
                            const float *vv_base, float *ah, float scale) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n, hd, scale, kv_base, lda, qh, 1, 0.0f, scores, 1);
    softmax_inplace(scores, n);
    cblas_sgemv(CblasRowMajor, CblasTrans, n, hd, 1.0f, vv_base, lda, scores, 1, 0.0f, ah, 1);
}

static void forward_token(int id, int pos, float *x_out, int dump_layers) {
    double tp0;
    if (prof_on()) tp0 = nowt();
    const float *emb = g_role_embed;
    memcpy(xbuf, emb+(long)id*g_cfg.d, g_cfg.d*sizeof(float));
    if (prof_on()) g_t_emb += nowt()-tp0;
    if (prof_on()) tp0 = nowt();
    float rope_c[g_cfg.hd/2], rope_s[g_cfg.hd/2];
    rope_precompute(pos, rope_c, rope_s);
    if (prof_on()) g_t_rope += nowt()-tp0;
    for (int l=0;l<g_cfg.nl;l++){
        if (prof_on()) tp0 = nowt();
        rmsnorm(xbuf, g_role_wt[ROLE_INPUT_LN][l]->f32, hbuf, g_cfg.d);
        if (prof_on()) g_t_rms += nowt()-tp0;

        if (prof_on()) tp0 = nowt();
        if (fused_dispatch() & 1) matvec_t(&g_qkv_fused[l], hbuf, qkv_fused_bias_row(l), q);
        else matvec_t(g_role_wt[ROLE_ATTN_Q][l], hbuf, g_qbias_l[l], q);
        if (prof_on()) g_t_q += nowt()-tp0;

        if (prof_on()) tp0 = nowt();
        if (!(fused_dispatch() & 1)) matvec_t(g_role_wt[ROLE_ATTN_K][l], hbuf, g_kbias_l[l], kv_k);
        if (prof_on()) g_t_k += nowt()-tp0;

        if (prof_on()) tp0 = nowt();
        if (!(fused_dispatch() & 1)) matvec_t(g_role_wt[ROLE_ATTN_V][l], hbuf, g_vbias_l[l], kv_v);
        if (prof_on()) g_t_v += nowt()-tp0;

        if (prof_on()) tp0 = nowt();
        for(int h=0;h<g_cfg.nh;h++) rope_apply(q+h*g_cfg.hd,rope_c,rope_s);
        for(int h=0;h<g_cfg.nkv;h++) rope_apply(kv_k+h*g_cfg.hd,rope_c,rope_s);
        if (prof_on()) g_t_rope += nowt()-tp0;

        if (prof_on()) tp0 = nowt();
        if (kv_int4_on()) {                       // M24: stage bias-free K / int4-quantize V on write
            kv_i4_write(l, pos, kv_k, kv_v, k4stg_l(l), v4stg_l(l), k4slot(l,0), k4s_l(l), k4z_l(l),
                        v4slot(l,pos), v4sc_slot(l,pos));
        } else if (kv_int8_on()) {                // M23: quantize post-RoPE K (M23-D2) and V on write
            if (kv_int8_on() == 1) q4_quant_act_i8(q, g_cfg.qd, g_qq8, g_qsc8);   // query -> int8 (SDOT scores only)
            kv_i8_write(l, pos, kv_k, kv_v, kqslot(l,pos), kscslot(l,pos), vqslot(l,pos), vscslot(l,pos));
        } else {
            memcpy(kslot(l,pos),kv_k,g_cfg.kvd*sizeof(float));
            memcpy(vslot(l,pos),kv_v,g_cfg.kvd*sizeof(float));
        }
        if (prof_on()) g_t_kvwrite += nowt()-tp0;

        if (attn_mode() == 1 && !kv_int8_on() && !kv_int4_on()) {   // M23/M24: mode-1 mirror is fp32-only
            for(int kvh=0;kvh<g_cfg.nkv;kvh++){
                memcpy(kslot2(l,kvh,pos),kv_k+kvh*g_cfg.hd,g_cfg.hd*sizeof(float));
                memcpy(vslot2(l,kvh,pos),kv_v+kvh*g_cfg.hd,g_cfg.hd*sizeof(float));
            }
        }
        double ta0=nowt();
        float scale=1.0f/sqrtf((float)g_cfg.hd);
        int am = attn_mode();
        if (am == 2 && !fast_attn_hd_ok()) am = 0;                                              // M44: split
        if (am == 3 && !fast_attn_shape_ok() && !fast_attn_shape_ok_g4()) am = 0;                // D3/M44: attn_neon.h's group/per-head
                                     // kernels are hand-unrolled for (AHD,AGROUP) -- fall back to the
                                     // generic scalar path below for any other (hd,group); am==0/1 are
                                     // already dimension-agnostic and pass through unaffected.
        if (kv_int4_on()) {                       // M24-D4: ONE int4 implementation (M23-D4 policy)
            kv_i4_attn(l, pos+1, q, k4slot(l,0), k4s_l(l), k4z_l(l), k4stg_l(l), v4stg_l(l),
                       v4slot(l,0), v4sc_slot(l,0), attn, scale);
        } else if (kv_int8_on()) {
            // M23-D4: ONE int8 implementation for every attn_mode (the group-fused mode-3
            // mechanism). KV-int8 is a new numeric contract; keeping four fp32 mode variants
            // under it would quadruple validation surface for zero A/B value. QWEN_KV_INT8=0
            // restores the full fp32 mode matrix untouched.
            kv_i8_attn(l, pos+1, q, g_qq8, g_qsc8, kqslot(l,0), kscslot(l,0),
                       vqslot(l,0), vscslot(l,0), attn, scale);
        } else if (am == 1) {
            for(int h=0;h<g_cfg.nh;h++){ int kvh=h/g_cfg.group; const float *qh=q+h*g_cfg.hd;
                attn_head_fast(kslot2(l,kvh,0), g_cfg.hd, qh, scores, pos+1, g_cfg.hd, vslot2(l,kvh,0), attn+h*g_cfg.hd, scale);
            }
        } else if (am == 2) {
            // M13 variant A: per-head drop-in NEON (attn_neon.h), reads the ORIGINAL kslot/vslot
            // cache (M13-D2: M12 already proved layout wasn't the bottleneck).
            for(int h=0;h<g_cfg.nh;h++){ int kvh=h/g_cfg.group; const float *qh=q+h*g_cfg.hd;
                attn_qk_neon(qh, kslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, pos+1, scale, scores);
                softmax_inplace(scores, pos+1);
                attn_wsum_neon(scores, vslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, pos+1, attn+h*g_cfg.hd);
            }
        } else if (am == 3) {
            // M13 variant B: fused per-kv-group NEON -- streams each kv-head's K/V once for all
            // GROUP=6 sharing query heads instead of re-streaming per head (the mechanism neither
            // M11, M12, nor variant A tested). M44: GROUP=4 family branches to its own kernel
            // pair -- keep this a strict, exhaustive if/else (not refactored into something less
            // obviously mutually-exclusive later): float*[AGROUP] decays to float** in C, so
            // nothing at the type level stops a miscoded branch from passing a wrongly-sized
            // sc_ptrs into the wrong accumulator-count kernel.
            for(int kvh=0;kvh<g_cfg.nkv;kvh++){
                float *sc_ptrs[g_cfg.group]; for(int g=0;g<g_cfg.group;g++) sc_ptrs[g]=scores_grp_row(g);
                if (g_cfg.group == AGROUP4) {
                    attn_qk_group_neon_g4(q+kvh*g_cfg.group*g_cfg.hd, kslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, pos+1, scale, sc_ptrs);
                    for(int g=0;g<g_cfg.group;g++) softmax_inplace(scores_grp_row(g), pos+1);
                    attn_wsum_group_neon_g4(sc_ptrs, vslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, pos+1, attn+kvh*g_cfg.group*g_cfg.hd);
                } else {
                    attn_qk_group_neon(q+kvh*g_cfg.group*g_cfg.hd, kslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, pos+1, scale, sc_ptrs);
                    for(int g=0;g<g_cfg.group;g++) softmax_inplace(scores_grp_row(g), pos+1);
                    attn_wsum_group_neon(sc_ptrs, vslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, pos+1, attn+kvh*g_cfg.group*g_cfg.hd);
                }
            }
        } else {
            for(int h=0;h<g_cfg.nh;h++){ int kvh=h/g_cfg.group; const float *qh=q+h*g_cfg.hd;
                for(int t=0;t<=pos;t++){ const float *kt=kslot(l,t)+kvh*g_cfg.hd; float dot; vDSP_dotpr(qh,1,kt,1,&dot,g_cfg.hd); scores[t]=dot*scale; }
                softmax_inplace(scores,pos+1);
                float *ah=attn+h*g_cfg.hd; memset(ah,0,g_cfg.hd*sizeof(float));
                for(int t=0;t<=pos;t++){ const float *vt=vslot(l,t)+kvh*g_cfg.hd; float sc=scores[t]; vDSP_vsma(vt,1,&sc,ah,1,ah,1,g_cfg.hd); }
            }
        }
        g_t_attn += nowt()-ta0;

        if (prof_on()) tp0 = nowt();
        matvec_t(g_role_wt[ROLE_ATTN_O][l], attn, NULL, obuf);
        if (prof_on()) g_t_o += nowt()-tp0;

        if (prof_on()) tp0 = nowt();
        vDSP_vadd(xbuf,1,obuf,1,xbuf,1,g_cfg.d);
        if (prof_on()) g_t_resid += nowt()-tp0;

        if (prof_on()) tp0 = nowt();
        rmsnorm(xbuf, g_role_wt[ROLE_POST_ATTN_LN][l]->f32, hbuf, g_cfg.d);
        if (prof_on()) g_t_rms += nowt()-tp0;

        if (prof_on()) tp0 = nowt();
        if (fused_dispatch() & 2) matvec_t(&g_gu_fused[l], hbuf, NULL, gate);
        else matvec_t(g_role_wt[ROLE_MLP_GATE][l], hbuf, NULL, gate);
        if (prof_on()) g_t_gate += nowt()-tp0;

        if (prof_on()) tp0 = nowt();
        if (!(fused_dispatch() & 2)) matvec_t(g_role_wt[ROLE_MLP_UP][l], hbuf, NULL, up);
        if (prof_on()) g_t_up += nowt()-tp0;

        if (prof_on()) tp0 = nowt();
        swiglu(gate,up,mlpact,g_cfg.im);
        if (prof_on()) g_t_swiglu += nowt()-tp0;

        if (prof_on()) tp0 = nowt();
        matvec_t(g_role_wt[ROLE_MLP_DOWN][l], mlpact, NULL, mlpout);
        if (prof_on()) g_t_down += nowt()-tp0;

        if (prof_on()) tp0 = nowt();
        vDSP_vadd(xbuf,1,mlpout,1,xbuf,1,g_cfg.d);
        if (prof_on()) g_t_resid += nowt()-tp0;

        if (dump_layers){ char p[512]; snprintf(p,sizeof p,"/Volumes/D50/vdsp/llm_engine/results/c_layer_%02d.f32",l);
            FILE *o=fopen(p,"wb"); fwrite(xbuf,4,g_cfg.d,o); fclose(o); }
    }
    memcpy(x_out,xbuf,g_cfg.d*sizeof(float));
}
static void final_logits(const float *x, float *logits) {
    double tp0;
    if (prof_on()) tp0 = nowt();
    float normed[g_cfg.d]; rmsnorm(x, g_role_final_norm, normed, g_cfg.d);
    if (prof_on()) g_t_headrms += nowt()-tp0;
    // Head selection: int8 lm_head when active; else a fp32 untied lm_head if the layout has one;
    // else the tied fp32 embed (Qwen2.5-1.5B). The embedding *gather* in forward_token always uses
    // the fp32 embed regardless. (QWEN_FP32_HEAD forces g_int8_head=0 -> our int8 head is skipped
    // by the kind!=K_F32 test and we correctly fall back to the tied embed.)
    WT *head;
    head = g_role_lm_head;
    if (prof_on()) tp0 = nowt();
    matvec_t(head, normed, NULL, logits);
    if (prof_on()) g_t_headgemv += nowt()-tp0;
}

#define MAXSPEC 16
static float *sb_x, *sb_h, *sb_q, *sb_k, *sb_v;
static float *sb_attn, *sb_o, *sb_g, *sb_u, *sb_a;
static float *sb_mo, *sb_norm;
static int8_t *sb_qq; static float *sb_qsc;   // M23: per-column int8 query (spec batch)

// batched projection: x=[M][in], y=[M][out]. q4/q8 unpack each row ONCE, dot with all M.
static void matmul_t(const WT *W, const float *x, const float *bias, float *y, int M) {
    if (W->kind == K_F32) {
        for (int m=0;m<M;m++){ float *ym=y+(size_t)m*W->out; if(bias)memcpy(ym,bias,W->out*sizeof(float));
            cblas_sgemv(CblasRowMajor,CblasNoTrans,W->out,W->in,1.0f,W->f32,W->in,x+(size_t)m*W->in,1,bias?1.0f:0.0f,ym,1); }
    } else if (W->kind == K_Q4G256SF) {
        // Phase 2 (M36) scope boundary: spec mode's fp32-tile batched verify path is NOT wired
        // for g256sf (out of this phase's approved scope) -- refuse loudly, don't misread
        // wsuper as per-64 scales. Spec keeps running on the g64 blob via QWEN_INT4_BIN.
        fprintf(stderr,"FATAL: spec mode is not wired for q4g256sf (%s) -- run spec with the q4g64 blob\n", W->name); exit(1);
    } else if (kai_route(W, M)) {
        // SME2 Phase 4: the only new call site in this function. x here is
        // already fp32 (spec's per-position activation tile) -- exactly what
        // kai_sme2_gemm_f32() wants, no extra quantization step needed here
        // (it does its own LHS int8 quantize+pack internally).
        kai_sme2_gemm_f32(M, W->out, W->in, x, W->kai_rhs, bias, y, g_kai_lhs_scratch);
    } else {
        gemm_qXg64_mt(&g_pool, W->kind==K_Q8G64?8:4, W->packed, W->scales, x, bias, y, W->out, W->in, M);
    }
}


// process n tokens (n<=MAXSPEC) at positions start_pos..start_pos+n-1; fill KV; xout=[n][D].
// causal within the batch (all KV written before attention reads).
static void forward_tokens(const int *ids, int n, int start_pos, float *xout) {
    const float *emb = g_role_embed;
    for (int m=0;m<n;m++) memcpy(sb_x+(size_t)m*g_cfg.d, emb+(long)ids[m]*g_cfg.d, g_cfg.d*sizeof(float));
    for (int l=0;l<g_cfg.nl;l++){
        for (int m=0;m<n;m++) rmsnorm(sb_x+(size_t)m*g_cfg.d, g_role_wt[ROLE_INPUT_LN][l]->f32, sb_h+(size_t)m*g_cfg.d, g_cfg.d);
        matmul_t(g_role_wt[ROLE_ATTN_Q][l], sb_h, g_qbias_l[l], sb_q, n);
        matmul_t(g_role_wt[ROLE_ATTN_K][l], sb_h, g_kbias_l[l], sb_k, n);
        matmul_t(g_role_wt[ROLE_ATTN_V][l], sb_h, g_vbias_l[l], sb_v, n);
        for (int m=0;m<n;m++){ int pos=start_pos+m;
            for(int h=0;h<g_cfg.nh;h++) rope_head(sb_q+(size_t)m*g_cfg.qd+h*g_cfg.hd,pos);
            for(int h=0;h<g_cfg.nkv;h++) rope_head(sb_k+(size_t)m*g_cfg.kvd+h*g_cfg.hd,pos);
            if (kv_int4_on()) {                   // M24 (see forward_token)
                kv_i4_write(l, pos, sb_k+(size_t)m*g_cfg.kvd, sb_v+(size_t)m*g_cfg.kvd,
                            k4stg_l(l), v4stg_l(l), k4slot(l,0), k4s_l(l), k4z_l(l),
                            v4slot(l,pos), v4sc_slot(l,pos));
            } else if (kv_int8_on()) {            // M23 (see forward_token)
                if (kv_int8_on() == 1) q4_quant_act_i8(sb_q+(size_t)m*g_cfg.qd, g_cfg.qd, sb_qq+(size_t)m*g_cfg.qd, sb_qsc+(size_t)m*g_cfg.qg);
                kv_i8_write(l, pos, sb_k+(size_t)m*g_cfg.kvd, sb_v+(size_t)m*g_cfg.kvd,
                            kqslot(l,pos), kscslot(l,pos), vqslot(l,pos), vscslot(l,pos));
            } else {
                memcpy(kslot(l,pos),sb_k+(size_t)m*g_cfg.kvd,g_cfg.kvd*sizeof(float));
                memcpy(vslot(l,pos),sb_v+(size_t)m*g_cfg.kvd,g_cfg.kvd*sizeof(float));
            }
            if (attn_mode() == 1 && !kv_int8_on() && !kv_int4_on()) {   // M23/M24: mode-1 mirror is fp32-only
                for(int kvh=0;kvh<g_cfg.nkv;kvh++){
                    memcpy(kslot2(l,kvh,pos),sb_k+(size_t)m*g_cfg.kvd+kvh*g_cfg.hd,g_cfg.hd*sizeof(float));
                    memcpy(vslot2(l,kvh,pos),sb_v+(size_t)m*g_cfg.kvd+kvh*g_cfg.hd,g_cfg.hd*sizeof(float));
                }
            }
        }
        float scale=1.0f/sqrtf((float)g_cfg.hd);
        // M14-D1: batched attention for modes 2/3 is a per-m loop over the SAME single-token
        // kernels forward_token() already uses -- not a new fused batched kernel. Variant B's win
        // (GQA K/V-streamed-once-per-group) is internal to one token's computation and transfers in
        // full this way; the only thing left on the table is cross-m sharing of the common K/V
        // prefix (positions 0..start_pos), deliberately deferred (register-budget-limited: fusing
        // across m AND the 6-way group needs n*GROUP live accumulators, well past the 32-register
        // NEON file for typical spec batch sizes -- see M14 RESULTS for the full analysis). Mode 1
        // (M12 BLAS mirror) stays naive here -- it is a preserved negative-result record, not
        // extended.
        int am_b = attn_mode();
        if (am_b == 2 && !fast_attn_hd_ok()) am_b = 0;                                          // M44: split
        if (am_b == 3 && !fast_attn_shape_ok() && !fast_attn_shape_ok_g4()) am_b = 0;            // D3/M44, see forward_token
        if (kv_int4_on()) {                       // M24-D4 (see forward_token; per-column nfull
                                                  // keeps block reads causal -- M24-D3)
            for (int m=0;m<n;m++){ int pos=start_pos+m;
                kv_i4_attn(l, pos+1, sb_q+(size_t)m*g_cfg.qd, k4slot(l,0), k4s_l(l), k4z_l(l),
                           k4stg_l(l), v4stg_l(l), v4slot(l,0), v4sc_slot(l,0),
                           sb_attn+(size_t)m*g_cfg.qd, scale);
            }
        } else if (kv_int8_on()) {                // M23-D4 (see forward_token)
            for (int m=0;m<n;m++){ int pos=start_pos+m;
                kv_i8_attn(l, pos+1, sb_q+(size_t)m*g_cfg.qd, sb_qq+(size_t)m*g_cfg.qd, sb_qsc+(size_t)m*g_cfg.qg,
                           kqslot(l,0), kscslot(l,0), vqslot(l,0), vscslot(l,0),
                           sb_attn+(size_t)m*g_cfg.qd, scale);
            }
        } else if (am_b == 2) {
            for (int m=0;m<n;m++){ int pos=start_pos+m; const float *qbase=sb_q+(size_t)m*g_cfg.qd; float *abase=sb_attn+(size_t)m*g_cfg.qd;
                for(int h=0;h<g_cfg.nh;h++){ int kvh=h/g_cfg.group; const float *qh=qbase+h*g_cfg.hd;
                    attn_qk_neon(qh, kslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, pos+1, scale, scores);
                    softmax_inplace(scores, pos+1);
                    attn_wsum_neon(scores, vslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, pos+1, abase+h*g_cfg.hd);
                }
            }
        } else if (am_b == 3) {
            for (int m=0;m<n;m++){ int pos=start_pos+m; const float *qbase=sb_q+(size_t)m*g_cfg.qd; float *abase=sb_attn+(size_t)m*g_cfg.qd;
                for(int kvh=0;kvh<g_cfg.nkv;kvh++){
                    float *sc_ptrs[g_cfg.group]; for(int g=0;g<g_cfg.group;g++) sc_ptrs[g]=scores_grp_row(g);
                    if (g_cfg.group == AGROUP4) {   // M44: GROUP=4 family, see forward_token
                        attn_qk_group_neon_g4(qbase+kvh*g_cfg.group*g_cfg.hd, kslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, pos+1, scale, sc_ptrs);
                        for(int g=0;g<g_cfg.group;g++) softmax_inplace(scores_grp_row(g), pos+1);
                        attn_wsum_group_neon_g4(sc_ptrs, vslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, pos+1, abase+kvh*g_cfg.group*g_cfg.hd);
                    } else {
                        attn_qk_group_neon(qbase+kvh*g_cfg.group*g_cfg.hd, kslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, pos+1, scale, sc_ptrs);
                        for(int g=0;g<g_cfg.group;g++) softmax_inplace(scores_grp_row(g), pos+1);
                        attn_wsum_group_neon(sc_ptrs, vslot(l,0)+kvh*g_cfg.hd, g_cfg.kvd, pos+1, abase+kvh*g_cfg.group*g_cfg.hd);
                    }
                }
            }
        } else {
            for (int m=0;m<n;m++){ int pos=start_pos+m; const float *qbase=sb_q+(size_t)m*g_cfg.qd; float *abase=sb_attn+(size_t)m*g_cfg.qd;
                for(int h=0;h<g_cfg.nh;h++){ int kvh=h/g_cfg.group; const float *qh=qbase+h*g_cfg.hd;
                    for(int t=0;t<=pos;t++){ const float *kt=kslot(l,t)+kvh*g_cfg.hd; float dot; vDSP_dotpr(qh,1,kt,1,&dot,g_cfg.hd); scores[t]=dot*scale; }
                    softmax_inplace(scores,pos+1);
                    float *ah=abase+h*g_cfg.hd; memset(ah,0,g_cfg.hd*sizeof(float));
                    for(int t=0;t<=pos;t++){ const float *vt=vslot(l,t)+kvh*g_cfg.hd; float sc=scores[t]; vDSP_vsma(vt,1,&sc,ah,1,ah,1,g_cfg.hd); }
                }
            }
        }
        matmul_t(g_role_wt[ROLE_ATTN_O][l], sb_attn, NULL, sb_o, n);
        for (int m=0;m<n;m++) vDSP_vadd(sb_x+(size_t)m*g_cfg.d,1,sb_o+(size_t)m*g_cfg.d,1,sb_x+(size_t)m*g_cfg.d,1,g_cfg.d);
        for (int m=0;m<n;m++) rmsnorm(sb_x+(size_t)m*g_cfg.d, g_role_wt[ROLE_POST_ATTN_LN][l]->f32, sb_h+(size_t)m*g_cfg.d, g_cfg.d);
        matmul_t(g_role_wt[ROLE_MLP_GATE][l], sb_h, NULL, sb_g, n);
        matmul_t(g_role_wt[ROLE_MLP_UP][l], sb_h, NULL, sb_u, n);
        for (int m=0;m<n;m++) swiglu(sb_g+(size_t)m*g_cfg.im, sb_u+(size_t)m*g_cfg.im, sb_a+(size_t)m*g_cfg.im, g_cfg.im);
        matmul_t(g_role_wt[ROLE_MLP_DOWN][l], sb_a, NULL, sb_mo, n);
        for (int m=0;m<n;m++) vDSP_vadd(sb_x+(size_t)m*g_cfg.d,1,sb_mo+(size_t)m*g_cfg.d,1,sb_x+(size_t)m*g_cfg.d,1,g_cfg.d);
    }
    for (int m=0;m<n;m++) memcpy(xout+(size_t)m*g_cfg.d, sb_x+(size_t)m*g_cfg.d, g_cfg.d*sizeof(float));
}

// logits for each of n hidden states: L=[n][VOCAB]
static void final_logits_batch(const float *xs, int n, float *L) {
    for (int m=0;m<n;m++) rmsnorm(xs+(size_t)m*g_cfg.d, g_role_final_norm, sb_norm+(size_t)m*g_cfg.d, g_cfg.d);
    WT *head;                                              // same selection as final_logits()
    head = g_role_lm_head;
    matmul_t(head, sb_norm, NULL, L, n);
}
// ---- M20: request-batched decode ("serve" mode) ----
// Decodes B independent sequences in lockstep so every projection weight ROW is read from
// DRAM once per step and reused across B activation columns (batched int8-SDOT GEMM,
// q4gemv.h M20 kernels). Per-sequence work (rmsnorm/RoPE/KV-write/attention/swiglu/argmax)
// stays a serial loop over B -- attention cannot share weights across the batch (each
// sequence attends its OWN KV cache) and was measured at ~0.5-1.5% of short-context decode.
//
// M20-D1 (KV cache layout): WHY -- per-sequence caches as one malloc'd block indexed
// [seq][layer][pos][KVD] (batch is the OUTERMOST stride), so kslot_srv(s,l,0) hands the
// existing NEON attention kernels a base pointer with the exact same inner layout
// ([pos][KVD], stride KVD) as the single-stream g_kcache -- zero kernel changes. vs
// alternatives: interleaving batch innermost ([pos][B][KVD]) would let attention batch
// its DRAM reads across sequences, but the sequences' caches hold DIFFERENT data (no
// reuse win) and it would force new attention kernels. COST: 2*B*NL*MAXSEQ*KVD*4 bytes
// = ~117 MB/seq, ~1.9 GB at B=16 -- acceptable on this 64 GB machine; allocation is
// checked and the error message says to lower QWEN_BATCH. EXIT: shrink the per-seq
// horizon (cap positions at np+N instead of MAXSEQ) or add a per-seq stride parameter.
// M23-D6 (B-cap raise): SRV_BMAX 16 -> 32 (= Q4_SDOT_BMAX, bumped in q4gemv.h) -- the
// documented M20-D2 EXIT ("bump SRV_BMAX/Q4_SDOT_BMAX and let the greedy dispatcher
// chunk"): the 8/4/2/1 column-tile dispatcher decomposes any M, so B=32 runs as 8+8+8+8
// with per-column bit-identity preserved at every decomposition. The KV-memory reason the
// cap was 16 (~117 MB/seq fp32, ~1.9 GB at B=16, ~3.8 GB at B=32) is what the int8 cache
// removes (~31 MB/seq, ~1.0 GB at B=32). WHY the cap is toggle-gated (srv_bcap): with
// fp32 KV the clamp stays 16, so QWEN_KV_INT8-unset behavior -- clamp messages, scheduler
// step patterns, memory footprint -- is identical to HEAD. COST: serve scratch statics
// double (~4 MB). EXIT: revert to 16; nothing else depends on B>16.
#define SRV_BMAX Q4_SDOT_BMAX
#define SRV_PCHUNK 16          // M21-D2 prefill chunk width -- kept at the HEAD value so the
                               // toggle-off cbatch scheduler is step-for-step identical to HEAD
// M24-D6 (B-cap raise): SRV_BMAX 32 -> 64 (= Q4_SDOT_BMAX, bumped again in q4gemv.h;
// same documented EXIT as M23-D6 -- the 8/4/2/1 column-tile dispatcher decomposes any
// M with per-column bit-identity, the cap is caller-scratch capacity only). Toggle-
// gated: int4 KV (~17 MB/seq codes+scales + 3.7 MB/seq staging) affords B=64
// (~1.4 GB); int8 keeps its M23 cap of 32; fp32 KV keeps 16 -- toggle-unset serve/
// cbatch behavior (clamp messages, scheduler steps, memory) is identical to HEAD.
// EXIT: revert to 32.
static inline int srv_bcap(void){ return kv_int4_on() ? SRV_BMAX : kv_int8_on() ? 32 : 16; }
static float *srv_kc = NULL, *srv_vc = NULL;
static inline float *kslot_srv(int s,int l,int pos){ return srv_kc+(((long)s*g_cfg.nl+l)*g_cfg.maxseq+pos)*g_cfg.kvd; }
static inline float *vslot_srv(int s,int l,int pos){ return srv_vc+(((long)s*g_cfg.nl+l)*g_cfg.maxseq+pos)*g_cfg.kvd; }
static int8_t *srv_kq = NULL, *srv_vq = NULL;      // M23: per-seq int8 KV codes [B][NL][MAXSEQ][KVD]
static float  *srv_ksc = NULL, *srv_vsc = NULL;    //      + scales [B][NL][MAXSEQ][KVG]
static inline int8_t *kqslot_srv(int s,int l,int pos){ return srv_kq +(((long)s*g_cfg.nl+l)*g_cfg.maxseq+pos)*g_cfg.kvd; }
static inline int8_t *vqslot_srv(int s,int l,int pos){ return srv_vq +(((long)s*g_cfg.nl+l)*g_cfg.maxseq+pos)*g_cfg.kvd; }
static inline float  *kscslot_srv(int s,int l,int pos){ return srv_ksc+(((long)s*g_cfg.nl+l)*g_cfg.maxseq+pos)*g_cfg.kvg; }
static inline float  *vscslot_srv(int s,int l,int pos){ return srv_vsc+(((long)s*g_cfg.nl+l)*g_cfg.maxseq+pos)*g_cfg.kvg; }
static uint8_t *srv_k4 = NULL, *srv_v4 = NULL;     // M24: per-seq int4 KV codes [B][NL][MAXSEQ][KVD/2]
static float *srv_k4s = NULL, *srv_k4z = NULL;     //      + per-block K scale/zero [B][NL][KV4_NB][KVD]
static float *srv_v4sc = NULL;                     //      + per-token V scales [B][NL][MAXSEQ][KVG]
static float *srv_k4stg = NULL;                    //      + fp32 K staging rings [B][NL][KV4_RING][KVD]
static float *srv_v4stg = NULL;                    //      + fp32 V recent-window rings (M24-D7)
static inline uint8_t *k4slot_srv(int s,int l,int pos){ return srv_k4+(((long)s*g_cfg.nl+l)*g_cfg.maxseq+pos)*(g_cfg.kvd/2); }
static inline uint8_t *v4slot_srv(int s,int l,int pos){ return srv_v4+(((long)s*g_cfg.nl+l)*g_cfg.maxseq+pos)*(g_cfg.kvd/2); }
static inline float *k4s_srv(int s,int l){ return srv_k4s+((long)s*g_cfg.nl+l)*KV4_NB*g_cfg.kvd; }
static inline float *k4z_srv(int s,int l){ return srv_k4z+((long)s*g_cfg.nl+l)*KV4_NB*g_cfg.kvd; }
static inline float *v4sc_slot_srv(int s,int l,int pos){ return srv_v4sc+(((long)s*g_cfg.nl+l)*g_cfg.maxseq+pos)*g_cfg.kvg; }
static inline float *k4stg_srv(int s,int l){ return srv_k4stg+((long)s*g_cfg.nl+l)*KV4_RING*g_cfg.kvd; }
static inline float *v4stg_srv(int s,int l){ return srv_v4stg+((long)s*g_cfg.nl+l)*KV4_RING*g_cfg.kvd; }
static float *srv_x, *srv_h, *srv_q, *srv_k, *srv_v;
static float *srv_attn, *srv_o, *srv_g, *srv_u, *srv_a;
static float *srv_mo, *srv_norm;
static int8_t *srv_xq; static float *srv_as;
static int8_t *srv_xq_nat;   // M20: per-column natural-order quant staging before q4_split_act
static int8_t *srv_qq; static float *srv_qsc;   // M23: per-column int8 query

// M20-D3 (activation-quant placement): WHY -- each batched projection call quantizes its
// B fp32 activation columns to int8 (q4_quant_act_i8, the SAME function the single-token
// W4A8 path runs inside gemv_q4g64_sdot_mt) right here at the call site, into serve-owned
// scratch, then hands pre-quantized columns to gemm_qXg64_sdot_mt. Deterministic function
// of identical inputs -> per-column xq/ascale bit-match the single-stream run. vs
// alternatives: hoisting one quantization per layer for the q/k/v (and gate/up) group
// would save 2+1 redundant D=1536-column quants per layer, but the single-token path
// also re-quantizes per call, and keeping the call structure 1:1 makes the bit-identity
// argument auditable per call site. COST: ~3x redundant quant of the same 1536-float
// column group (microseconds vs the GEMM). EXIT: quantize srv_h once after each rmsnorm
// and pass xq/ascale through -- bit-identical by the same determinism argument.
// M20-D5 (kernel routing): WHY -- serve ALWAYS routes K_Q4G64/K_Q8G64 through the batched
// int8-SDOT kernels regardless of QWEN_W4A8: the mode exists to realize the batched-SDOT
// roofline, and its correctness reference is the QWEN_W4A8=1 single-stream greedy. The
// fp32-tile batched path (gemm_qXg64_mt) is NOT wired here. K_F32 tensors (tied fp32
// lm_head fallback) use per-column cblas_sgemv, identical to matvec_t's fp32 branch.
// COST: no serve A/B against the fp32-activation batched path without new code. EXIT:
// add an env-gated branch to gemm_qXg64_mt (kernel already exists, spec-verify path).
static double g_srv_quant=0, g_srv_gemm=0, g_srv_attn=0, g_srv_step=0;   // serve QWEN_PROF buckets
static void matmul_sdot(const WT *W, const float *x, const float *bias, float *y, int B) {
    if (W->kind == K_F32) {
        for (int m=0;m<B;m++){ float *ym=y+(size_t)m*W->out; if(bias)memcpy(ym,bias,W->out*sizeof(float));
            cblas_sgemv(CblasRowMajor,CblasNoTrans,W->out,W->in,1.0f,W->f32,W->in,x+(size_t)m*W->in,1,bias?1.0f:0.0f,ym,1); }
        return;
    }
    if (kai_route(W, B)) {
        // SME2 Phase 4: placed ABOVE the q4_quant_act_i8/q4_split_act calls
        // below on purpose -- this path does its own LHS int8 quantize+pack
        // inside kai_sme2_gemm_f32(), so those calls would be pure wasted
        // work if reached first. No profiled/non-profiled split needed here
        // (unlike the two existing gemm_qXg64_sdot_mt call sites below):
        // quant+gemm are one opaque call into kai_sme2_gemm_f32(), so the
        // whole thing is charged to g_srv_gemm -- there is no separate
        // "quant" phase to attribute to g_srv_quant from the caller's side.
        double t0; if (prof_on()) t0 = nowt();
        kai_sme2_gemm_f32(B, W->out, W->in, x, W->kai_rhs, bias, y, g_kai_lhs_scratch);
        if (prof_on()) g_srv_gemm += nowt() - t0;
        return;
    }
    // Phase 2 (M36): per-kind activation-scale stride -- g256sf activation groups are 256-wide,
    // so its per-column scale stride is in/256 (fits the in/64-sized srv_as slots).
    int ngg = (W->kind == K_Q4G256SF) ? (W->in >> 8) : (W->in >> 6);
    double tq0; if (prof_on()) tq0=nowt();
    // q4 batched kernels read the activation in pre-SPLIT lo/hi order (q4_split_act);
    // q8 batched kernels read natural order -- see the M20 pre-split note in q4gemv.h.
    if (W->kind == K_Q8G64)
        for (int m=0;m<B;m++) q4_quant_act_i8(x+(size_t)m*W->in, W->in, srv_xq+(size_t)m*W->in, srv_as+(size_t)m*ngg);
    else if (W->kind == K_Q4G256SF)
        for (int m=0;m<B;m++) {
            q4_quant_act_i8_g256(x+(size_t)m*W->in, W->in, srv_xq_nat, srv_as+(size_t)m*ngg);
            q4_split_act(srv_xq_nat, srv_xq+(size_t)m*W->in, W->in);
        }
    else
        for (int m=0;m<B;m++) {
            q4_quant_act_i8(x+(size_t)m*W->in, W->in, srv_xq_nat, srv_as+(size_t)m*ngg);
            q4_split_act(srv_xq_nat, srv_xq+(size_t)m*W->in, W->in);
        }
    if (prof_on()) { double tg0=nowt(); g_srv_quant+=tg0-tq0;
        if (W->kind == K_Q4G256SF)
            gemm_q4g256sf_sdot_mt(&g_pool, W->packed, W->scales, W->sub, srv_xq, srv_as, bias, y, W->out, W->in, B);
        else
            gemm_qXg64_sdot_mt(&g_pool, W->kind==K_Q8G64?8:4, W->packed, W->scales, srv_xq, srv_as, bias, y, W->out, W->in, B);
        g_srv_gemm+=nowt()-tg0; return; }
    if (W->kind == K_Q4G256SF)
        gemm_q4g256sf_sdot_mt(&g_pool, W->packed, W->scales, W->sub, srv_xq, srv_as, bias, y, W->out, W->in, B);
    else
        gemm_qXg64_sdot_mt(&g_pool, W->kind==K_Q8G64?8:4, W->packed, W->scales, srv_xq, srv_as, bias, y, W->out, W->in, B);
}


// One lockstep decode step for B sequences whose current tokens are ids[0..B-1], all at
// the same position pos (same-length prompts; per-seq positions would need a pos array +
// per-seq rope tables -- not needed for the lockstep bench). Fills L=[B][VOCAB].
// M20-D4 (attention loop structure): WHY -- attention is a serial per-sequence loop over
// the SAME single-token NEON kernels forward_token() uses (mode 3 fused per-KV-group by
// default), each sequence reading its own kslot_srv/vslot_srv cache. There is nothing to
// batch: the B caches hold different tensors, so no weight/data reuse exists across
// sequences; only thread-parallelism over B could help, and attention was measured at
// ~0.5-1.5% of short-context decode (M15 prof). Mode 1 (M12 BLAS-mirror, a preserved
// negative result) is NOT extended to serve -- it falls back to naive (mode 0) here
// because the mirror cache has no per-sequence variant. COST: attention/norm/RoPE time
// does not shrink with B, diluting the batched-GEMM speedup (reported honestly in the
// bench). EXIT: thread the per-seq loop over q4pool workers if attention share grows
// (long contexts), or extend kslot2 mirrors per-seq for mode 1.
static void serve_step(const int *ids, int B, int pos, float *L) {
    const float *emb = g_role_embed;
    for (int s=0;s<B;s++) memcpy(srv_x+(size_t)s*g_cfg.d, emb+(long)ids[s]*g_cfg.d, g_cfg.d*sizeof(float));
    float rc[g_cfg.hd/2], rs[g_cfg.hd/2]; rope_precompute(pos, rc, rs);
    float scale=1.0f/sqrtf((float)g_cfg.hd);
    int am = attn_mode(); if (am == 1) am = 0;   // M20-D4: no per-seq mirror cache
    if (am == 2 && !fast_attn_hd_ok()) am = 0;                                              // M44: split
    if (am == 3 && !fast_attn_shape_ok() && !fast_attn_shape_ok_g4()) am = 0;                // D3/M44, see forward_token
    for (int l=0;l<g_cfg.nl;l++){
        for (int s=0;s<B;s++) rmsnorm(srv_x+(size_t)s*g_cfg.d, g_role_wt[ROLE_INPUT_LN][l]->f32, srv_h+(size_t)s*g_cfg.d, g_cfg.d);
        matmul_sdot(g_role_wt[ROLE_ATTN_Q][l], srv_h, g_qbias_l[l], srv_q, B);
        matmul_sdot(g_role_wt[ROLE_ATTN_K][l], srv_h, g_kbias_l[l], srv_k, B);
        matmul_sdot(g_role_wt[ROLE_ATTN_V][l], srv_h, g_vbias_l[l], srv_v, B);
        for (int s=0;s<B;s++){
            for(int h=0;h<g_cfg.nh;h++) rope_apply(srv_q+(size_t)s*g_cfg.qd+h*g_cfg.hd,rc,rs);
            for(int h=0;h<g_cfg.nkv;h++) rope_apply(srv_k+(size_t)s*g_cfg.kvd+h*g_cfg.hd,rc,rs);
            if (kv_int4_on()) {                   // M24 (see forward_token)
                kv_i4_write(l, pos, srv_k+(size_t)s*g_cfg.kvd, srv_v+(size_t)s*g_cfg.kvd,
                            k4stg_srv(s,l), v4stg_srv(s,l), k4slot_srv(s,l,0), k4s_srv(s,l), k4z_srv(s,l),
                            v4slot_srv(s,l,pos), v4sc_slot_srv(s,l,pos));
            } else if (kv_int8_on()) {            // M23 (see forward_token)
                if (kv_int8_on() == 1) q4_quant_act_i8(srv_q+(size_t)s*g_cfg.qd, g_cfg.qd, srv_qq+(size_t)s*g_cfg.qd, srv_qsc+(size_t)s*g_cfg.qg);
                kv_i8_write(l, pos, srv_k+(size_t)s*g_cfg.kvd, srv_v+(size_t)s*g_cfg.kvd,
                            kqslot_srv(s,l,pos), kscslot_srv(s,l,pos), vqslot_srv(s,l,pos), vscslot_srv(s,l,pos));
            } else {
                memcpy(kslot_srv(s,l,pos),srv_k+(size_t)s*g_cfg.kvd,g_cfg.kvd*sizeof(float));
                memcpy(vslot_srv(s,l,pos),srv_v+(size_t)s*g_cfg.kvd,g_cfg.kvd*sizeof(float));
            }
        }
        double tat0; if (prof_on()) tat0=nowt();
        for (int s=0;s<B;s++){
            const float *qb=srv_q+(size_t)s*g_cfg.qd; float *ab=srv_attn+(size_t)s*g_cfg.qd;
            const float *kb=NULL, *vb=NULL;
            if (!kv_int8_on() && !kv_int4_on()) { kb=kslot_srv(s,l,0); vb=vslot_srv(s,l,0); }
            if (kv_int4_on()) {                   // M24-D4 (see forward_token)
                kv_i4_attn(l, pos+1, qb, k4slot_srv(s,l,0), k4s_srv(s,l), k4z_srv(s,l),
                           k4stg_srv(s,l), v4stg_srv(s,l), v4slot_srv(s,l,0), v4sc_slot_srv(s,l,0), ab, scale);
            } else if (kv_int8_on()) {            // M23-D4 (see forward_token)
                kv_i8_attn(l, pos+1, qb, srv_qq+(size_t)s*g_cfg.qd, srv_qsc+(size_t)s*g_cfg.qg,
                           kqslot_srv(s,l,0), kscslot_srv(s,l,0),
                           vqslot_srv(s,l,0), vscslot_srv(s,l,0), ab, scale);
            } else if (am == 3) {
                for(int kvh=0;kvh<g_cfg.nkv;kvh++){
                    float *sc_ptrs[g_cfg.group]; for(int g=0;g<g_cfg.group;g++) sc_ptrs[g]=scores_grp_row(g);
                    if (g_cfg.group == AGROUP4) {   // M44: GROUP=4 family, see forward_token
                        attn_qk_group_neon_g4(qb+kvh*g_cfg.group*g_cfg.hd, kb+kvh*g_cfg.hd, g_cfg.kvd, pos+1, scale, sc_ptrs);
                        for(int g=0;g<g_cfg.group;g++) softmax_inplace(scores_grp_row(g), pos+1);
                        attn_wsum_group_neon_g4(sc_ptrs, vb+kvh*g_cfg.hd, g_cfg.kvd, pos+1, ab+kvh*g_cfg.group*g_cfg.hd);
                    } else {
                        attn_qk_group_neon(qb+kvh*g_cfg.group*g_cfg.hd, kb+kvh*g_cfg.hd, g_cfg.kvd, pos+1, scale, sc_ptrs);
                        for(int g=0;g<g_cfg.group;g++) softmax_inplace(scores_grp_row(g), pos+1);
                        attn_wsum_group_neon(sc_ptrs, vb+kvh*g_cfg.hd, g_cfg.kvd, pos+1, ab+kvh*g_cfg.group*g_cfg.hd);
                    }
                }
            } else if (am == 2) {
                for(int h=0;h<g_cfg.nh;h++){ int kvh=h/g_cfg.group;
                    attn_qk_neon(qb+h*g_cfg.hd, kb+kvh*g_cfg.hd, g_cfg.kvd, pos+1, scale, scores);
                    softmax_inplace(scores, pos+1);
                    attn_wsum_neon(scores, vb+kvh*g_cfg.hd, g_cfg.kvd, pos+1, ab+h*g_cfg.hd);
                }
            } else {
                for(int h=0;h<g_cfg.nh;h++){ int kvh=h/g_cfg.group; const float *qh=qb+h*g_cfg.hd;
                    for(int t=0;t<=pos;t++){ const float *kt=kb+(size_t)t*g_cfg.kvd+kvh*g_cfg.hd; float dot; vDSP_dotpr(qh,1,kt,1,&dot,g_cfg.hd); scores[t]=dot*scale; }
                    softmax_inplace(scores,pos+1);
                    float *ah=ab+h*g_cfg.hd; memset(ah,0,g_cfg.hd*sizeof(float));
                    for(int t=0;t<=pos;t++){ const float *vt=vb+(size_t)t*g_cfg.kvd+kvh*g_cfg.hd; float sc=scores[t]; vDSP_vsma(vt,1,&sc,ah,1,ah,1,g_cfg.hd); }
                }
            }
        }
        if (prof_on()) g_srv_attn+=nowt()-tat0;
        matmul_sdot(g_role_wt[ROLE_ATTN_O][l], srv_attn, NULL, srv_o, B);
        for (int s=0;s<B;s++) vDSP_vadd(srv_x+(size_t)s*g_cfg.d,1,srv_o+(size_t)s*g_cfg.d,1,srv_x+(size_t)s*g_cfg.d,1,g_cfg.d);
        for (int s=0;s<B;s++) rmsnorm(srv_x+(size_t)s*g_cfg.d, g_role_wt[ROLE_POST_ATTN_LN][l]->f32, srv_h+(size_t)s*g_cfg.d, g_cfg.d);
        matmul_sdot(g_role_wt[ROLE_MLP_GATE][l], srv_h, NULL, srv_g, B);
        matmul_sdot(g_role_wt[ROLE_MLP_UP][l], srv_h, NULL, srv_u, B);
        for (int s=0;s<B;s++) swiglu(srv_g+(size_t)s*g_cfg.im, srv_u+(size_t)s*g_cfg.im, srv_a+(size_t)s*g_cfg.im, g_cfg.im);
        matmul_sdot(g_role_wt[ROLE_MLP_DOWN][l], srv_a, NULL, srv_mo, B);
        for (int s=0;s<B;s++) vDSP_vadd(srv_x+(size_t)s*g_cfg.d,1,srv_mo+(size_t)s*g_cfg.d,1,srv_x+(size_t)s*g_cfg.d,1,g_cfg.d);
    }
    for (int s=0;s<B;s++) rmsnorm(srv_x+(size_t)s*g_cfg.d, g_role_final_norm, srv_norm+(size_t)s*g_cfg.d, g_cfg.d);
    WT *head;                                              // same selection as final_logits()
    head = g_role_lm_head;
    matmul_sdot(head, srv_norm, NULL, L, B);
}

// ---- M21: continuous batching ("cbatch" mode) ----
// Removes M20-serve's lockstep restriction: a fixed pool of up to B slots, each running an
// INDEPENDENT request at its OWN position (ragged). Finished slots free immediately and the
// next queued request is admitted; every decode step runs the batched int8-SDOT GEMM over
// however many slots are currently active (1..B).
//
// M21-D1 (slot/position state layout): WHY -- per-slot state is parallel arrays indexed by
// slot id (cb_pos/cb_tok/cb_active/...), and each step builds a COMPACT gather list
// act[0..A-1] -> slot id. The batched-GEMM scratch (srv_x/srv_h/... reused from serve) is
// indexed by the compact index m; only the KV cache and per-slot bookkeeping are indexed by
// the slot id via act[m]. vs alternatives: (a) running the GEMM over all B slots with dead
// columns masked wastes GEMM work on freed slots (the exact lockstep waste this mode removes);
// (b) compacting the KV caches themselves on eviction would need a ~117MB memcpy per finish.
// COST: one extra indirection (act[m]) in the per-slot loops. EXIT: none needed -- the gather
// list is the standard continuous-batching structure (vLLM's "batch" is exactly this).
// M21-D2 (admission & prefill model): WHY -- a newly admitted request is prefilled BY ITSELF
// (single-sequence), but NOT token-by-token: its prompt is fed through the SAME cbatch_step
// in chunks of up to SRV_BMAX=16 COLUMNS OF THE SAME SLOT at consecutive positions. This is
// causally sound with zero extra code because cbatch_step already (a) applies per-column
// RoPE/pos, and (b) writes ALL columns' KV before ANY column's attention runs in each layer,
// and column m only reads cache positions 0..spos[m] -- later columns' (future) KV sits at
// higher positions and is never read. First measurement (token-by-token, A=1) put prefill at
// 77% of wall (2.35s/3.04s, R=6) and lost to the lockstep estimate; chunking recovers the
// batched-GEMM weight reuse for prefill too. The logits head runs only on the final chunk
// and only its LAST column is consumed (head GEMM is weight-bound, so the chunked head costs
// ~= a single-column head). COST: while one request prefills, other active slots stall
// (admission is not overlapped with decode). EXIT (future): vLLM-style mixed chunked prefill
// -- append the new request's next prompt chunk as extra columns of the regular decode step
// (no logits for prefill columns) until consumed; the ragged plumbing (per-column slot, pos,
// rope) already supports mixing, only the scheduler loop changes.
// M21-D3 (eviction/EOS policy): WHY -- a slot finishes when its newly-emitted token is EOS
// (Qwen2.5: 151643 <|endoftext|> or 151645 <|im_end|>; either stops -- HF generation config
// lists both), OR it has emitted max_new_tokens, OR the next feed position would hit MAXSEQ.
// The EOS token itself IS recorded (output = greedy prefix ending at EOS), so the standalone
// greedy diff stays a pure prefix comparison. The slot frees immediately; admission happens
// at the top of the next scheduler iteration. QWEN_CB_STOP_EXTRA=<id> adds one extra stop id
// so the EOS-eviction path can be exercised on workloads whose greedy continuation never
// emits a real EOS (the repetitive bench prompt). COST: none. EXIT: per-request stop lists.
// M21-D4 (ragged active-count -> fixed-M batched GEMM): WHY -- the compact gather hands
// gemm_qXg64_sdot_mt an M equal to the CURRENT active count A (1..B<=16); its dispatcher
// already decomposes any M into fixed register tiles (M=8/4/2/1, 16 via chunking), so ragged
// batch width needs ZERO kernel changes. Each output column's int8-SDOT accumulation is pure
// int32 adds (exact) with per-row fixed-order float group scaling, computed identically at
// every tile width -- so a sequence's logits do not depend on HOW MANY neighbours shared its
// step. That determinism is the entire neighbour-independence claim; gate 1 verifies it
// bit-exactly against A=1 single-stream W4A8 greedy. COST: small M (1-3 active) underutilizes
// the 8-wide tile (throughput, not correctness). EXIT: none needed.
// M21-D5 (queue structure): WHY -- the request queue is a fixed array consumed FIFO by a
// head index (admission order = arrival order, no reordering), because the bench synthesizes
// all R requests up front and FCFS is the honest baseline scheduler. COST: a long-prompt
// request at the queue head briefly stalls decode (see D2). EXIT: priority/shortest-first
// admission is a one-line change on the pick in the admission loop.
// M21-D6 (per-slot RoPE): WHY -- each step recomputes each active slot's (cos,sin) table
// with rope_precompute(cb_pos[slot]) -- the SAME function single-stream greedy uses, so the
// angle values are bit-identical by construction. COST: A x 64 cos/sin per step
// (microseconds). EXIT: cache one table per slot and advance incrementally.
static int cb_act_slot[SRV_BMAX];           // compact m -> slot id (per step)
static float *cb_rc_flat, *cb_rs_flat;
static inline float *cb_rc_row(int m){ return cb_rc_flat + (long)m*(g_cfg.hd/2); }
static inline float *cb_rs_row(int m){ return cb_rs_flat + (long)m*(g_cfg.hd/2); }
// One ragged decode step: A active columns; column m carries token ids[m] for slot slot[m]
// at that slot's own position spos[m]. Fills L[m][VOCAB] when want_logits.
static void cbatch_step(const int *ids, const int *slot, const int *spos, int A,
                        float *L, int want_logits) {
    const float *emb = g_role_embed;
    for (int m=0;m<A;m++) memcpy(srv_x+(size_t)m*g_cfg.d, emb+(long)ids[m]*g_cfg.d, g_cfg.d*sizeof(float));
    for (int m=0;m<A;m++) rope_precompute(spos[m], cb_rc_row(m), cb_rs_row(m));   // M21-D6
    float scale=1.0f/sqrtf((float)g_cfg.hd);
    int am = attn_mode(); if (am == 1) am = 0;   // M20-D4: no per-seq mirror cache
    if (am == 2 && !fast_attn_hd_ok()) am = 0;                                              // M44: split
    if (am == 3 && !fast_attn_shape_ok() && !fast_attn_shape_ok_g4()) am = 0;                // D3/M44, see forward_token
    for (int l=0;l<g_cfg.nl;l++){
        for (int m=0;m<A;m++) rmsnorm(srv_x+(size_t)m*g_cfg.d, g_role_wt[ROLE_INPUT_LN][l]->f32, srv_h+(size_t)m*g_cfg.d, g_cfg.d);
        matmul_sdot(g_role_wt[ROLE_ATTN_Q][l], srv_h, g_qbias_l[l], srv_q, A);
        matmul_sdot(g_role_wt[ROLE_ATTN_K][l], srv_h, g_kbias_l[l], srv_k, A);
        matmul_sdot(g_role_wt[ROLE_ATTN_V][l], srv_h, g_vbias_l[l], srv_v, A);
        for (int m=0;m<A;m++){ int s=slot[m], pos=spos[m];
            for(int h=0;h<g_cfg.nh;h++) rope_apply(srv_q+(size_t)m*g_cfg.qd+h*g_cfg.hd,cb_rc_row(m),cb_rs_row(m));
            for(int h=0;h<g_cfg.nkv;h++) rope_apply(srv_k+(size_t)m*g_cfg.kvd+h*g_cfg.hd,cb_rc_row(m),cb_rs_row(m));
            if (kv_int4_on()) {                   // M24 (see forward_token; per-column causal
                                                  // block horizon per M24-D3)
                kv_i4_write(l, pos, srv_k+(size_t)m*g_cfg.kvd, srv_v+(size_t)m*g_cfg.kvd,
                            k4stg_srv(s,l), v4stg_srv(s,l), k4slot_srv(s,l,0), k4s_srv(s,l), k4z_srv(s,l),
                            v4slot_srv(s,l,pos), v4sc_slot_srv(s,l,pos));
            } else if (kv_int8_on()) {            // M23 (see forward_token)
                if (kv_int8_on() == 1) q4_quant_act_i8(srv_q+(size_t)m*g_cfg.qd, g_cfg.qd, srv_qq+(size_t)m*g_cfg.qd, srv_qsc+(size_t)m*g_cfg.qg);
                kv_i8_write(l, pos, srv_k+(size_t)m*g_cfg.kvd, srv_v+(size_t)m*g_cfg.kvd,
                            kqslot_srv(s,l,pos), kscslot_srv(s,l,pos), vqslot_srv(s,l,pos), vscslot_srv(s,l,pos));
            } else {
                memcpy(kslot_srv(s,l,pos),srv_k+(size_t)m*g_cfg.kvd,g_cfg.kvd*sizeof(float));
                memcpy(vslot_srv(s,l,pos),srv_v+(size_t)m*g_cfg.kvd,g_cfg.kvd*sizeof(float));
            }
        }
        double tat0; if (prof_on()) tat0=nowt();
        for (int m=0;m<A;m++){ int s=slot[m], pos=spos[m];
            const float *qb=srv_q+(size_t)m*g_cfg.qd; float *ab=srv_attn+(size_t)m*g_cfg.qd;
            const float *kb=NULL, *vb=NULL;
            if (!kv_int8_on() && !kv_int4_on()) { kb=kslot_srv(s,l,0); vb=vslot_srv(s,l,0); }
            if (kv_int4_on()) {                   // M24-D4 (see forward_token)
                kv_i4_attn(l, pos+1, qb, k4slot_srv(s,l,0), k4s_srv(s,l), k4z_srv(s,l),
                           k4stg_srv(s,l), v4stg_srv(s,l), v4slot_srv(s,l,0), v4sc_slot_srv(s,l,0), ab, scale);
            } else if (kv_int8_on()) {            // M23-D4 (see forward_token)
                kv_i8_attn(l, pos+1, qb, srv_qq+(size_t)m*g_cfg.qd, srv_qsc+(size_t)m*g_cfg.qg,
                           kqslot_srv(s,l,0), kscslot_srv(s,l,0),
                           vqslot_srv(s,l,0), vscslot_srv(s,l,0), ab, scale);
            } else if (am == 3) {
                for(int kvh=0;kvh<g_cfg.nkv;kvh++){
                    float *sc_ptrs[g_cfg.group]; for(int g=0;g<g_cfg.group;g++) sc_ptrs[g]=scores_grp_row(g);
                    if (g_cfg.group == AGROUP4) {   // M44: GROUP=4 family, see forward_token
                        attn_qk_group_neon_g4(qb+kvh*g_cfg.group*g_cfg.hd, kb+kvh*g_cfg.hd, g_cfg.kvd, pos+1, scale, sc_ptrs);
                        for(int g=0;g<g_cfg.group;g++) softmax_inplace(scores_grp_row(g), pos+1);
                        attn_wsum_group_neon_g4(sc_ptrs, vb+kvh*g_cfg.hd, g_cfg.kvd, pos+1, ab+kvh*g_cfg.group*g_cfg.hd);
                    } else {
                        attn_qk_group_neon(qb+kvh*g_cfg.group*g_cfg.hd, kb+kvh*g_cfg.hd, g_cfg.kvd, pos+1, scale, sc_ptrs);
                        for(int g=0;g<g_cfg.group;g++) softmax_inplace(scores_grp_row(g), pos+1);
                        attn_wsum_group_neon(sc_ptrs, vb+kvh*g_cfg.hd, g_cfg.kvd, pos+1, ab+kvh*g_cfg.group*g_cfg.hd);
                    }
                }
            } else if (am == 2) {
                for(int h=0;h<g_cfg.nh;h++){ int kvh=h/g_cfg.group;
                    attn_qk_neon(qb+h*g_cfg.hd, kb+kvh*g_cfg.hd, g_cfg.kvd, pos+1, scale, scores);
                    softmax_inplace(scores, pos+1);
                    attn_wsum_neon(scores, vb+kvh*g_cfg.hd, g_cfg.kvd, pos+1, ab+h*g_cfg.hd);
                }
            } else {
                for(int h=0;h<g_cfg.nh;h++){ int kvh=h/g_cfg.group; const float *qh=qb+h*g_cfg.hd;
                    for(int t=0;t<=pos;t++){ const float *kt=kb+(size_t)t*g_cfg.kvd+kvh*g_cfg.hd; float dot; vDSP_dotpr(qh,1,kt,1,&dot,g_cfg.hd); scores[t]=dot*scale; }
                    softmax_inplace(scores,pos+1);
                    float *ah=ab+h*g_cfg.hd; memset(ah,0,g_cfg.hd*sizeof(float));
                    for(int t=0;t<=pos;t++){ const float *vt=vb+(size_t)t*g_cfg.kvd+kvh*g_cfg.hd; float sc=scores[t]; vDSP_vsma(vt,1,&sc,ah,1,ah,1,g_cfg.hd); }
                }
            }
        }
        if (prof_on()) g_srv_attn+=nowt()-tat0;
        matmul_sdot(g_role_wt[ROLE_ATTN_O][l], srv_attn, NULL, srv_o, A);
        for (int m=0;m<A;m++) vDSP_vadd(srv_x+(size_t)m*g_cfg.d,1,srv_o+(size_t)m*g_cfg.d,1,srv_x+(size_t)m*g_cfg.d,1,g_cfg.d);
        for (int m=0;m<A;m++) rmsnorm(srv_x+(size_t)m*g_cfg.d, g_role_wt[ROLE_POST_ATTN_LN][l]->f32, srv_h+(size_t)m*g_cfg.d, g_cfg.d);
        matmul_sdot(g_role_wt[ROLE_MLP_GATE][l], srv_h, NULL, srv_g, A);
        matmul_sdot(g_role_wt[ROLE_MLP_UP][l], srv_h, NULL, srv_u, A);
        for (int m=0;m<A;m++) swiglu(srv_g+(size_t)m*g_cfg.im, srv_u+(size_t)m*g_cfg.im, srv_a+(size_t)m*g_cfg.im, g_cfg.im);
        matmul_sdot(g_role_wt[ROLE_MLP_DOWN][l], srv_a, NULL, srv_mo, A);
        for (int m=0;m<A;m++) vDSP_vadd(srv_x+(size_t)m*g_cfg.d,1,srv_mo+(size_t)m*g_cfg.d,1,srv_x+(size_t)m*g_cfg.d,1,g_cfg.d);
    }
    if (!want_logits) return;   // prefill steps before the last prompt token skip the head GEMM
    for (int m=0;m<A;m++) rmsnorm(srv_x+(size_t)m*g_cfg.d, g_role_final_norm, srv_norm+(size_t)m*g_cfg.d, g_cfg.d);
    WT *head;                                              // same selection as final_logits()
    head = g_role_lm_head;
    matmul_sdot(head, srv_norm, NULL, L, A);
}

// M15-C2: vDSP_maxvi returns the index of the FIRST occurrence of the maximum (matches
// the scalar loop's strict > comparison -- ties keep the earlier index), verified by the
// gate as an exact index-equality check, not just a value check.
static inline int argmax_v(const float *v){ float mv; vDSP_Length mi; vDSP_maxvi(v,1,&mv,&mi,g_cfg.vocab); return (int)mi; }

// prompt-lookup: longest suffix (ngram) of hist that occurred earlier -> propose following tokens
static int pl_lookup(const int *hist, int hlen, int *out, int K, int ngram){
    if (hlen < ngram) return 0;
    const int *suf = hist + hlen - ngram;
    for (int i = hlen - ngram - 1; i >= 0; i--){
        int mm=1; for(int j=0;j<ngram;j++) if(hist[i+j]!=suf[j]){mm=0;break;}
        if (mm){ int src=i+ngram, k=0; while(k<K && src+k<hlen){ out[k]=hist[src+k]; k++; } return k; }
    }
    return 0;
}

static int load_ids(const char *path, int *buf, int cap){
    FILE *f=fopen(path,"rb"); if(!f) return -1; int n=0; while(n<cap&&fread(&buf[n],4,1,f)==1)n++; fclose(f); return n;
}

// D2: allocates every buffer that used to be a fixed-size static/global array sized by a
// compile-time architecture constant, now that those constants are runtime (g_cfg, loaded by
// load_arch_cfg() -- must run before this). Follows this file's own existing q4pool_init()/
// aligned_alloc() precedent for runtime-sized allocation; called once from main() right after
// load_arch_cfg(). Uninitialized (not zero-filled) -- every one of these is a write-before-read
// scratch buffer already (same as the static arrays they replace, which were never cleared
// between calls either); the bit-identical verification gate is the empirical check for any
// missed zero-init assumption.
static void alloc_arch_buffers(void) {
    g_qq8  = malloc((size_t)g_cfg.qd*sizeof(int8_t));
    g_qsc8 = malloc((size_t)g_cfg.qg*sizeof(float));
    g_scores_grp_flat  = malloc((size_t)g_cfg.group*g_cfg.maxseq*sizeof(float));
    g_scores_grp2_flat = malloc((size_t)g_cfg.group*g_cfg.maxseq*sizeof(float));
    g_vbias_l = malloc((size_t)g_cfg.nl*sizeof(float*));
    g_qbias_l = malloc((size_t)g_cfg.nl*sizeof(float*));
    g_kbias_l = malloc((size_t)g_cfg.nl*sizeof(float*));
    kv_i8_knb = malloc((size_t)g_cfg.kvd*sizeof(float));
    kv_i8_vnb = malloc((size_t)g_cfg.kvd*sizeof(float));
    kv4_qt8  = malloc((size_t)g_cfg.group*g_cfg.hd*sizeof(int8_t));
    kv4_qtsc = malloc((size_t)g_cfg.group*2*sizeof(float));
    kv4_qtf  = malloc((size_t)g_cfg.hd*sizeof(float));
    kv4_Cg   = malloc((size_t)g_cfg.group*sizeof(float));
    xbuf = malloc((size_t)g_cfg.d*sizeof(float));
    hbuf = malloc((size_t)g_cfg.d*sizeof(float));
    attn = malloc((size_t)g_cfg.qd*sizeof(float));
    obuf = malloc((size_t)g_cfg.d*sizeof(float));
    mlpact = malloc((size_t)g_cfg.im*sizeof(float));
    mlpout = malloc((size_t)g_cfg.d*sizeof(float));
    scores = malloc((size_t)g_cfg.maxseq*sizeof(float));
    swiglu_tmp1 = malloc((size_t)g_cfg.im*sizeof(float));   // D2 bug fix 1 (was hardcoded [8960])
    swiglu_tmp2 = malloc((size_t)g_cfg.im*sizeof(float));
    q_solo = malloc((size_t)g_cfg.qd*sizeof(float));
    kv_k_solo = malloc((size_t)g_cfg.kvd*sizeof(float));
    kv_v_solo = malloc((size_t)g_cfg.kvd*sizeof(float));
    gate_solo = malloc((size_t)g_cfg.im*sizeof(float));
    up_solo = malloc((size_t)g_cfg.im*sizeof(float));
    qkv_fused_buf = malloc((size_t)(g_cfg.qd+2*g_cfg.kvd)*sizeof(float));
    gu_fused_buf  = malloc((size_t)(2*g_cfg.im)*sizeof(float));
    g_qkv_fused = malloc((size_t)g_cfg.nl*sizeof(WT));
    g_gu_fused  = malloc((size_t)g_cfg.nl*sizeof(WT));
    g_qkv_fused_bias_flat = malloc((size_t)g_cfg.nl*(g_cfg.qd+2*g_cfg.kvd)*sizeof(float));

    sb_x = malloc((size_t)MAXSPEC*g_cfg.d*sizeof(float));
    sb_h = malloc((size_t)MAXSPEC*g_cfg.d*sizeof(float));
    sb_q = malloc((size_t)MAXSPEC*g_cfg.qd*sizeof(float));
    sb_k = malloc((size_t)MAXSPEC*g_cfg.kvd*sizeof(float));
    sb_v = malloc((size_t)MAXSPEC*g_cfg.kvd*sizeof(float));
    sb_attn = malloc((size_t)MAXSPEC*g_cfg.qd*sizeof(float));
    sb_o    = malloc((size_t)MAXSPEC*g_cfg.d*sizeof(float));
    sb_g = malloc((size_t)MAXSPEC*g_cfg.im*sizeof(float));
    sb_u = malloc((size_t)MAXSPEC*g_cfg.im*sizeof(float));
    sb_a = malloc((size_t)MAXSPEC*g_cfg.im*sizeof(float));
    sb_mo   = malloc((size_t)MAXSPEC*g_cfg.d*sizeof(float));
    sb_norm = malloc((size_t)MAXSPEC*g_cfg.d*sizeof(float));
    sb_qq  = malloc((size_t)MAXSPEC*g_cfg.qd*sizeof(int8_t));
    sb_qsc = malloc((size_t)MAXSPEC*g_cfg.qg*sizeof(float));

    srv_x = malloc((size_t)SRV_BMAX*g_cfg.d*sizeof(float));
    srv_h = malloc((size_t)SRV_BMAX*g_cfg.d*sizeof(float));
    srv_q = malloc((size_t)SRV_BMAX*g_cfg.qd*sizeof(float));
    srv_k = malloc((size_t)SRV_BMAX*g_cfg.kvd*sizeof(float));
    srv_v = malloc((size_t)SRV_BMAX*g_cfg.kvd*sizeof(float));
    srv_attn = malloc((size_t)SRV_BMAX*g_cfg.qd*sizeof(float));
    srv_o    = malloc((size_t)SRV_BMAX*g_cfg.d*sizeof(float));
    srv_g = malloc((size_t)SRV_BMAX*g_cfg.im*sizeof(float));
    srv_u = malloc((size_t)SRV_BMAX*g_cfg.im*sizeof(float));
    srv_a = malloc((size_t)SRV_BMAX*g_cfg.im*sizeof(float));
    srv_mo   = malloc((size_t)SRV_BMAX*g_cfg.d*sizeof(float));
    srv_norm = malloc((size_t)SRV_BMAX*g_cfg.d*sizeof(float));
    srv_xq = malloc((size_t)SRV_BMAX*g_cfg.im*sizeof(int8_t));
    srv_as = malloc((size_t)SRV_BMAX*(g_cfg.im/64)*sizeof(float));
    srv_xq_nat = malloc((size_t)g_cfg.im*sizeof(int8_t));
    srv_qq  = malloc((size_t)SRV_BMAX*g_cfg.qd*sizeof(int8_t));
    srv_qsc = malloc((size_t)SRV_BMAX*g_cfg.qg*sizeof(float));

    cb_rc_flat = malloc((size_t)SRV_BMAX*(g_cfg.hd/2)*sizeof(float));
    cb_rs_flat = malloc((size_t)SRV_BMAX*(g_cfg.hd/2)*sizeof(float));
    g_rope_scale = malloc((size_t)(g_cfg.hd/2)*sizeof(float));   // M43

    if (!g_qq8||!g_qsc8||!g_scores_grp_flat||!g_scores_grp2_flat||!g_vbias_l||!g_qbias_l||!g_kbias_l||!kv_i8_knb||!kv_i8_vnb||
        !kv4_qt8||!kv4_qtsc||!kv4_qtf||!kv4_Cg||!xbuf||!hbuf||!attn||!obuf||!mlpact||!mlpout||!scores||
        !swiglu_tmp1||!swiglu_tmp2||!q_solo||!kv_k_solo||!kv_v_solo||!gate_solo||!up_solo||!qkv_fused_buf||
        !gu_fused_buf||!g_qkv_fused||!g_gu_fused||!g_qkv_fused_bias_flat||
        !sb_x||!sb_h||!sb_q||!sb_k||!sb_v||!sb_attn||!sb_o||!sb_g||!sb_u||!sb_a||!sb_mo||!sb_norm||!sb_qq||!sb_qsc||
        !srv_x||!srv_h||!srv_q||!srv_k||!srv_v||!srv_attn||!srv_o||!srv_g||!srv_u||!srv_a||!srv_mo||!srv_norm||
        !srv_xq||!srv_as||!srv_xq_nat||!srv_qq||!srv_qsc||!cb_rc_flat||!cb_rs_flat||!g_rope_scale) {
        fprintf(stderr,"FATAL: alloc_arch_buffers: an allocation failed\n"); exit(1);
    }
}

// D4: generalizes q/k/v bias loading from "always present, FATAL if missing" (the old
// unconditional wlf() at every call site) to conditional on g_cfg.qkv_bias, using the same
// wt_opt()-style optional-tensor lookup already established for lm_head.weight. Precomputes
// one pointer per layer per {q,k,v} ONCE here (must run after weight load, i.e. after
// load_fp32()/load_int4() populate g_wt[]) so every downstream +bias call site stays exactly
// what it already was -- a single unconditional pointer dereference (g_qbias_l[l] etc.) --
// instead of adding a live branch at all 15 of them. When qkv_bias is false, every pointer
// aliases one shared zero-filled buffer (sized to the largest bias this model could have, QD
// or KVD) rather than allocating 3*NL empty buffers.
static void init_qkv_bias(void) {
    if (!g_cfg.qkv_bias) {
        int zn = g_cfg.qd > g_cfg.kvd ? g_cfg.qd : g_cfg.kvd;
        g_zero_bias = calloc((size_t)zn, sizeof(float));
        if (!g_zero_bias) { fprintf(stderr,"FATAL: init_qkv_bias zero-buffer alloc failed\n"); exit(1); }
    }
    for (int l=0;l<g_cfg.nl;l++) {
        char nq[96], nk[96], nv[96];
        snprintf(nq,sizeof nq,"model.layers.%d.self_attn.q_proj.bias",l);
        snprintf(nk,sizeof nk,"model.layers.%d.self_attn.k_proj.bias",l);
        snprintf(nv,sizeof nv,"model.layers.%d.self_attn.v_proj.bias",l);
        if (g_cfg.qkv_bias) {
            WT *tq = wt_opt(nq), *tk = wt_opt(nk), *tv = wt_opt(nv);
            if (!tq || !tk || !tv || tq->kind!=K_F32 || tk->kind!=K_F32 || tv->kind!=K_F32) {
                fprintf(stderr,"FATAL: arch_config QKV_BIAS=1 but layer %d is missing a q/k/v bias tensor\n",l); exit(1); }
            g_qbias_l[l] = tq->f32; g_kbias_l[l] = tk->f32; g_vbias_l[l] = tv->f32;
        } else {
            g_qbias_l[l] = g_zero_bias; g_kbias_l[l] = g_zero_bias; g_vbias_l[l] = g_zero_bias;
        }
    }
    fprintf(stderr,"[engine] qkv bias: %s\n", g_cfg.qkv_bias ? "loaded (per-layer tensors)" : "absent (QKV_BIAS=0 -- zero-filled)");
}

// M51 (task: Q4_THREADS auto-detect): "6" was an M1 Max-only empirical result (in-situ
// microbench + full-engine bench-mode sweep) hardcoded as the default -- wrong on other Apple
// Silicon topologies. A cross-machine bench-mode sweep (Q4_THREADS in {2,4,6,8,10}, `qwen_infer
// bench 30`, this project's own median-of-3 methodology; M1 Max run against the real deployed
// Llama-3.1-8B foldg64 weights, an Apple M4 (4P+6E) run against a size-matched zero-filled
// synthetic blob -- bench mode only times memory traffic + compute, weight VALUES don't affect
// throughput, so this is a fair speed-only comparison) found M4's own optimum is T=10 (all
// logical cores), not 6: 4.50 vs 3.06 tok/s, a 32% loss from applying M1 Max's tuned value on
// M4. M1 Max's own optimum (T=6) was reproduced (deterministic fixed-partition threading, two
// runs bit-identical), confirming the original M16-C finding still holds there.
//
// NOT a general core-count formula: M1 Max's optimum (6) is LESS than its own 8 P-cores, while
// M4's optimum (10) is its FULL P+E count -- these don't reduce to one clean rule from n=2
// points, and this project's own data-first-numerics discipline (CLAUDE.md Sec 13) forbids
// guessing one. Instead: an explicit per-chip table of only what's actually been bench-swept,
// exact-matched on `machdep.cpu.brand_string`; anything not in the table falls back to the old
// hardcoded 6 (safe on the most widely-deployed case here) with a stderr note.
// EXIT: bench-sweep a new chip (`qwen_infer bench 30` across a T range spanning its physical
// core count, same methodology as above) and add a table entry -- do not extrapolate one from
// the existing two.
static int detect_q4_threads(void) {
    char brand[128] = {0};
    size_t sz = sizeof(brand) - 1;
    if (sysctlbyname("machdep.cpu.brand_string", brand, &sz, NULL, 0) != 0) {
        fprintf(stderr, "[engine] Q4_THREADS auto-detect: sysctlbyname failed, using untuned default 6\n");
        return 6;
    }
    static const struct { const char *brand; int t; } TUNED[] = {
        {"Apple M1 Max", 6},   // 8P+2E -- M16-C full-engine sweep (this project, pre-M51)
        {"Apple M4",    10},   // 4P+6E -- M51 cross-machine bench sweep, 2026-08-15
    };
    for (size_t i = 0; i < sizeof(TUNED)/sizeof(TUNED[0]); i++) {
        if (!strcmp(brand, TUNED[i].brand)) return TUNED[i].t;
    }
    fprintf(stderr, "[engine] Q4_THREADS auto-detect: '%s' not in the tuned table (only M1 Max "
                     "and M4 bench-swept so far) -- using untuned default 6; run `qwen_infer "
                     "bench 30` across a T range spanning this chip's physical core count to "
                     "find its real optimum and add it to detect_q4_threads()\n", brand);
    return 6;
}

// ============================================================================
// Phase MoE-3a: MLA+MoE forward-pass subsystem, ported from Phase MoE-2b's
// standalone moe2b_verify.c (already gated against MLX's real forward: 8/8
// argmax match, logits rel-L2 1.25e-3..3.91e-3, router 207/208 exact --
// ~/Desktop/vdsp_v2_design/trackb_moe2b_results/RESULTS.md). Purely additive:
// every symbol below is `moe_`/`Moe`/`MOE_`-prefixed and none of it is called
// from the existing GQA dense-model code paths above this point in the file
// -- see run_moe_verify_mode() and its single call site near the top of
// main(). Activates ONLY when weights_moe/arch_config_moe.txt exists
// (QWEN_MOE_BASE overridable, same defensive-parsing convention as
// QWEN_ARCH_CONFIG); the existing GQA model paths are byte-identical to
// before this block existed when that file is absent.
//
// Still scalar (no NEON/SME2), still single-token-at-a-time, still no
// gather/batched dispatch -- this phase's only goal is "does the PRODUCTION
// BINARY reproduce MoE-2b's already-verified numbers", not throughput.
// Gather-based SME2 dispatch is Phase MoE-3b, a separate future session.
//
// No residual/error-feedback logic on purpose: this decodes an
// already-quantized blob and computes a real fp forward pass, it does not
// quantize anything itself.
// ============================================================================

typedef struct {
    char name[128];
    long E, out, in, ng;
    long packed_off, packed_bytes, scale_off, bias_off;
} MoeAFTensor;

static uint8_t *moe_mmap_file(const char *path, long *out_bytes) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    struct stat sb; fstat(fd, &sb);
    void *p = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    close(fd);
    *out_bytes = sb.st_size;
    return (uint8_t *)p;
}

static MoeAFTensor *g_moe_af = NULL;
static int g_moe_naf = 0;
static void moe_load_layout_af(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    g_moe_af = malloc(sizeof(MoeAFTensor) * 512);
    char name[128];
    long E, out, in, ng, po, pb, so, bo;
    while (fscanf(f, "%127s %ld %ld %ld %ld %ld %ld %ld %ld",
                  name, &E, &out, &in, &ng, &po, &pb, &so, &bo) == 9) {
        if (g_moe_naf >= 512) { fprintf(stderr, "FATAL: >512 moe af tensors\n"); exit(1); }
        MoeAFTensor *t = &g_moe_af[g_moe_naf++];
        strncpy(t->name, name, sizeof t->name - 1);
        t->E = E; t->out = out; t->in = in; t->ng = ng;
        t->packed_off = po; t->packed_bytes = pb; t->scale_off = so; t->bias_off = bo;
    }
    fclose(f);
}
static MoeAFTensor *moe_find_af(const char *name) {
    for (int i = 0; i < g_moe_naf; i++) if (!strcmp(g_moe_af[i].name, name)) return &g_moe_af[i];
    fprintf(stderr, "FATAL: moe af tensor not found: %s\n", name); exit(1);
}
static float moe_decode_af(const uint8_t *blob, MoeAFTensor *t, long e, long row, long col) {
    long row_words = t->in / 8;
    long word_idx = col / 8;
    long byte_in_word = (col % 8) / 2;
    long byte_idx = t->packed_off + ((e * t->out + row) * row_words + word_idx) * 4 + byte_in_word;
    uint8_t byte = blob[byte_idx];
    int nib = (col % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
    long group = col / 64;
    long scale_idx = t->scale_off + ((e * t->out + row) * t->ng + group) * 4;
    long bias_idx = t->bias_off + ((e * t->out + row) * t->ng + group) * 4;
    float scale, bias;
    memcpy(&scale, blob + scale_idx, 4);
    memcpy(&bias, blob + bias_idx, 4);
    return (float)nib * scale + bias;
}

// Phase MoE-4c throughput optimization (task#102 follow-up, "다 해봐" pass): moe_decode_af()
// above re-reads (via 2 memcpy calls) the SAME scale/bias pair on every one of the 64 columns
// that share it -- col/64==group is constant across a 64-wide run, so a straight per-element
// loop over moe_decode_af() does 128 redundant memcpy calls per group of 64 instead of 2. This
// hoists the scale/bias load out to once per group, computing the identical row dot product
// (same double accumulation, same left-to-right column order c=0..in-1, bit-for-bit unchanged --
// this is a loop restructuring, not a numeric change) while cutting memcpy calls ~64x. Confirmed
// via real timing instrumentation that moe_decode_af's per-element memcpy overhead, not raw FLOP
// throughput, was the actual dominant cost (this is why scalar-pool threading alone left real
// headroom -- threading parallelizes the same wasteful per-element work across cores, it doesn't
// remove the waste itself).
static double moe_matvec_af_row(const uint8_t *blob, MoeAFTensor *t, long e, long row, const float *x) {
    long row_words = t->in / 8;
    long ng = t->ng;
    long row_base = e * t->out + row;
    double acc = 0.0;
    for (long g = 0; g < ng; g++) {
        long scale_idx = t->scale_off + (row_base * ng + g) * 4;
        long bias_idx = t->bias_off + (row_base * ng + g) * 4;
        float scale, bias;
        memcpy(&scale, blob + scale_idx, 4);
        memcpy(&bias, blob + bias_idx, 4);
        long col0 = g * 64;
        for (long ci = 0; ci < 64; ci++) {
            long col = col0 + ci;
            long word_idx = col / 8;
            long byte_in_word = (col % 8) / 2;
            long byte_idx = t->packed_off + (row_base * row_words + word_idx) * 4 + byte_in_word;
            uint8_t byte = blob[byte_idx];
            int nib = (col % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
            float w = (float)nib * scale + bias;
            acc += (double)w * x[col];
        }
    }
    return acc;
}

// Task#102 follow-up, second optimization pass ("계속 더 파" -- user explicitly asked to push
// into riskier territory after the threading+group-hoist pass hit diminishing returns). This is
// NOT bit-identical to moe_matvec_af_row() above: it still unpacks the same 64 dequantized
// weights per group (identical scale/bias/nibble math), but instead of a serial double-
// accumulate loop over the 64 elements, it hands them to vDSP_dotpr() -- Apple's SIMD-vectorized
// single-precision dot product (Accelerate framework, already linked -- q4gemv.h's own dense-
// model fallback path already uses this exact function for the same purpose). vDSP's internal
// reduction order differs from strict left-to-right double accumulation, so per-group results can
// differ in their last few bits; those float partial sums are then accumulated across groups in
// double (same as before) -- only the WITHIN-group reduction method changes. This is real numeric
// drift, not a pure restructuring like moe_matvec_af_row() was -- kept OPT-IN
// (QWEN_MOE_SCALAR_VDSP=1, default off) and must be validated against real ground truth
// (moe4a_ref_generation.json / mlx_lm), not just compared token-for-token against the exact path,
// before ever being considered for promotion to default.
static double moe_matvec_af_row_vdsp(const uint8_t *blob, MoeAFTensor *t, long e, long row, const float *x) {
    long row_words = t->in / 8;
    long ng = t->ng;
    long row_base = e * t->out + row;
    double acc = 0.0;
    float wbuf[64];
    for (long g = 0; g < ng; g++) {
        long scale_idx = t->scale_off + (row_base * ng + g) * 4;
        long bias_idx = t->bias_off + (row_base * ng + g) * 4;
        float scale, bias;
        memcpy(&scale, blob + scale_idx, 4);
        memcpy(&bias, blob + bias_idx, 4);
        long col0 = g * 64;
        for (long ci = 0; ci < 64; ci++) {
            long col = col0 + ci;
            long word_idx = col / 8;
            long byte_in_word = (col % 8) / 2;
            long byte_idx = t->packed_off + (row_base * row_words + word_idx) * 4 + byte_in_word;
            uint8_t byte = blob[byte_idx];
            int nib = (col % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
            wbuf[ci] = (float)nib * scale + bias;
        }
        float group_dot;
        vDSP_dotpr(wbuf, 1, x + col0, 1, &group_dot, 64);
        acc += (double)group_dot;
    }
    return acc;
}

// Dispatch pointer, set once (moe_scalar_pool_ensure) from QWEN_MOE_SCALAR_VDSP -- both
// moe_matvec_af() (nthreads==1 fallback) and moe_scalar_run_range() (threaded workers) call
// through this so REVERIFY=off's no-op guarantee is untouched (that path never reaches here) and
// both threaded/unthreaded modes stay numerically consistent with each other.
static double (*g_moe_row_fn)(const uint8_t *, MoeAFTensor *, long, long, const float *) = moe_matvec_af_row;

typedef struct { char name[128]; long off, numel; } MoeF32Tensor;
static MoeF32Tensor *g_moe_f32 = NULL;
static int g_moe_nf32 = 0;
static uint8_t *g_moe_f32_blob;
static void moe_load_layout_f32(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    g_moe_f32 = malloc(sizeof(MoeF32Tensor) * 256);
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char name[128]; long off, numel;
        if (sscanf(line, "%127s %ld %ld", name, &off, &numel) != 3) continue;
        if (g_moe_nf32 >= 256) { fprintf(stderr, "FATAL: >256 moe f32 tensors\n"); exit(1); }
        MoeF32Tensor *t = &g_moe_f32[g_moe_nf32++];
        strncpy(t->name, name, sizeof t->name - 1);
        t->off = off; t->numel = numel;
    }
    fclose(f);
}
static MoeF32Tensor *moe_find_f32(const char *name) {
    for (int i = 0; i < g_moe_nf32; i++) if (!strcmp(g_moe_f32[i].name, name)) return &g_moe_f32[i];
    fprintf(stderr, "FATAL: moe f32 tensor not found: %s\n", name); exit(1);
}

static double moe_cfg_get(const char *path, const char *key) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    char line[128], k[64]; double v;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "%63[^=]=%lf", k, &v) == 2 && !strcmp(k, key)) { fclose(f); return v; }
    }
    fclose(f);
    fprintf(stderr, "FATAL: moe arch_config key '%s' not found in %s\n", key, path); exit(1);
}

static int MOE_HIDDEN, MOE_N_HEADS, MOE_KV_LORA_RANK, MOE_QK_ROPE_HD, MOE_QK_NOPE_HD, MOE_V_HD, MOE_Q_HEAD_DIM;
static int MOE_NL, MOE_FIRST_DENSE_LAYERS, MOE_N_EXPERTS, MOE_N_SHARED, MOE_TOP_K, MOE_IM_DIM, MOE_DENSE_IM, MOE_VOCAB;
static double MOE_ROPE_THETA, MOE_YARN_FACTOR, MOE_YARN_BETA_FAST, MOE_YARN_BETA_SLOW, MOE_YARN_MSCALE, MOE_YARN_MSCALE_ALL_DIM, MOE_YARN_ORIG_MAX_POS;
static double g_moe_rope_mscale, g_moe_attn_scale;
#define MOE_MAX_ROPE_PAIRS 64
static double g_moe_yarn_freqs[MOE_MAX_ROPE_PAIRS];

// YaRN table + interleaved RoPE, ported verbatim from Phase MoE-2a's mla_verify.c (the
// angle=pos/freqs[i] DIVISION was that phase's decisive empirically-confirmed finding --
// see trackb_moe2a_results/RESULTS.md -- do not "simplify" back to multiplication).
static double moe_yarn_find_correction_dim(double num_rotations, int dim, double base, double max_pos) {
    return (dim * log(max_pos / (num_rotations * 2.0 * M_PI))) / (2.0 * log(base));
}
static void moe_yarn_find_correction_range(double low_rot, double high_rot, int dim, double base, double max_pos,
                                            double *out_low, double *out_high) {
    double low = floor(moe_yarn_find_correction_dim(low_rot, dim, base, max_pos));
    double high = ceil(moe_yarn_find_correction_dim(high_rot, dim, base, max_pos));
    *out_low = low < 0 ? 0 : low;
    *out_high = high > dim - 1 ? dim - 1 : high;
}
static double moe_yarn_get_mscale(double scale, double mscale) {
    if (scale <= 1.0) return 1.0;
    return 0.1 * mscale * log(scale) + 1.0;
}
static void moe_init_yarn(void) {
    int dim = MOE_QK_ROPE_HD;
    int half = dim / 2;
    double low, high;
    moe_yarn_find_correction_range(MOE_YARN_BETA_FAST, MOE_YARN_BETA_SLOW, dim, MOE_ROPE_THETA, MOE_YARN_ORIG_MAX_POS, &low, &high);
    if (low == high) high += 0.001;
    for (int i = 0; i < half; i++) {
        double freq_extra = pow(MOE_ROPE_THETA, (2.0 * i) / dim);
        double freq_inter = MOE_YARN_FACTOR * pow(MOE_ROPE_THETA, (2.0 * i) / dim);
        double ramp = (i - low) / (high - low);
        if (ramp < 0) ramp = 0; if (ramp > 1) ramp = 1;
        double freq_mask = 1.0 - ramp;
        g_moe_yarn_freqs[i] = (freq_inter * freq_extra) / (freq_inter * freq_mask + freq_extra * (1.0 - freq_mask));
    }
    g_moe_rope_mscale = moe_yarn_get_mscale(MOE_YARN_FACTOR, MOE_YARN_MSCALE) / moe_yarn_get_mscale(MOE_YARN_FACTOR, MOE_YARN_MSCALE_ALL_DIM);
    double mscale2 = moe_yarn_get_mscale(MOE_YARN_FACTOR, MOE_YARN_MSCALE_ALL_DIM);
    g_moe_attn_scale = pow((double)MOE_Q_HEAD_DIM, -0.5) * mscale2 * mscale2;
}
static void moe_rope_traditional_apply(float *v, int dim, int pos) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        double ang = (double)pos / g_moe_yarn_freqs[i];
        double c = cos(ang), s = sin(ang);
        double a = v[2*i], b = v[2*i+1];
        v[2*i]   = (float)(a*c - b*s);
        v[2*i+1] = (float)(a*s + b*c);
    }
}

// Distinct from the file's existing rmsnorm() (which reads g_cfg.eps -- unset/meaningless
// in MoE mode since load_arch_cfg() is never called on this path). DeepSeek's real eps=1e-6.
static void moe_rmsnorm(const float *x, const float *g, float *y, int n) {
    double ss = 0.0; for (int i = 0; i < n; i++) ss += (double)x[i]*x[i];
    double inv = 1.0 / sqrt(ss/n + 1e-6);
    for (int i = 0; i < n; i++) y[i] = (float)(x[i]*inv*g[i]);
}
static void moe_matvec_af(const uint8_t *blob, MoeAFTensor *t, long e, const float *x, float *y) {
    for (long r = 0; r < t->out; r++) y[r] = (float)g_moe_row_fn(blob, t, e, r, x);
}

// ==============================================================================================
// Phase MoE-4c throughput optimization (task#102 follow-up). Real timing instrumentation
// disproved this track's own earlier "Tier2 replay cost grows with position (quadratic)"
// diagnosis: Tier1's single-call cost (~2174-3319ms) and Tier2's per-replayed-token cost
// (~2234-2548ms) came out statistically indistinguishable -- position does NOT make replay
// disproportionately expensive. The real cost driver is moe_matvec_af() above: a completely
// unthreaded, unvectorized scalar double-accumulate loop, called ~550 times per token (dominated
// by lm_head out=102400 and dense_gate/up/down out|in=10944), which is why a single scalar token
// costs ~2450ms measured (not the ~1550ms this track had been citing from a different MoE-4a
// workload/measurement context). moe_decode_af() (used inside the row loop) is a pure function --
// confirmed by reading its body: reads only the read-only blob/t params, no shared mutable state
// -- so splitting the row loop [0,out) across threads (each thread owns a disjoint row range,
// writes disjoint y[r] slots) reproduces the EXACT serial result bit-for-bit: per-row accumulation
// order is unchanged, only the across-row ordering differs, which the math doesn't depend on.
// Modeled on q4gemv.h's q4pool (same mutex+condvar dispatch + persistent workers + row-range
// split + "done_count starts at 1 for the dispatching thread's own q4_run_range(p,0) share"
// pattern, see q4pool_go_and_wait) but with a far simpler job struct suited to the AF blob layout
// -- q4pool's own job struct is tightly coupled to the dense model's packed q4g64 layout.
typedef struct {
    const uint8_t *blob;
    MoeAFTensor *t;
    long e;
    const float *x;
    float *y;
} MoeScalarJob;

// Second optimization pass (task#102, "다 해봐" -- small-call consolidation): moe_forward_token()'s
// MoE-layer branch issues 21 separate moe_matvec_af_mt() dispatches per layer (14 independent
// gate/up calls sharing the SAME input h2, then 7 independent down calls each with its own
// swiglu'd input) -- each dispatch pays a full mutex-lock/broadcast/barrier-wait cycle even
// though the underlying matvecs (out=1408-4096) are modest. This is a pure scheduling change,
// NOT a numeric one: batching N independent matvecs into ONE dispatch (workers split the
// concatenated [0,total_rows) space, look up which item owns their row range, run the identical
// per-row kernel) produces bit-for-bit the same y[] values as calling moe_matvec_af_mt() N times
// -- only the dispatch/barrier overhead is amortized across N items instead of paid N times.
#define MOE_BATCH_MAX_ITEMS 40   // covers the gate+up batch's 2*MOE_TOP_K+2 items at MOE_TOP_K's
                                 // own top_idx[16] upper bound (2*16+2=34) with headroom -- real
                                 // DeepSeek-V2-Lite config has MOE_TOP_K=6 (14 items), this is a
                                 // defensive ceiling, not the expected case.
typedef struct {
    const uint8_t *blob;
    MoeAFTensor *t;
    long e;
    const float *x;
    float *y;
    long out;      // cached t->out, avoids a pointer deref per row in the hot lookup
    long row_off;  // this item's offset in the flattened [0,total_rows) index space
} MoeBatchItem;

typedef struct {
    MoeBatchItem items[MOE_BATCH_MAX_ITEMS];
    int n_items;
    long total_rows;
} MoeBatchJob;

typedef struct MoeScalarPool MoeScalarPool;
typedef struct { MoeScalarPool *p; int wi; } MoeScalarWarg;

#define MOE_SPOOL_MAX_THREADS 64   // D2: raised 32->64 after discovering the post-batch-consolidation
                                   // T=40 sweep point had silently clamped to 32 (same array cap) --
                                   // that "T=40" result was actually a T=32 rerun, not a real
                                   // measurement. WHY: batch consolidation cut dispatch count ~10x,
                                   // changing the overhead/parallelism tradeoff enough to warrant
                                   // re-testing past the old cap. COST: none (pool struct is a few
                                   // more KB, workers only spawned up to whatever nthreads is set).
                                   // EXIT: lower back to 32 (or whatever the next real sweep picks)
                                   // if higher thread counts stop helping.
                                   // task#109 follow-up: was 16 (bob's physical core count), bumped
                                   // to test past the array cap now that group-hoisting made T=16
                                   // still a net win over T=8 -- oversubscription past physical
                                   // cores (bob: 4P+6E=10) may or may not still help, measuring.
struct MoeScalarPool {
    int nthreads;
    pthread_t th[MOE_SPOOL_MAX_THREADS];
    pthread_mutex_t mtx;
    pthread_cond_t cv_go, cv_done;
    _Atomic int gen, done_count, stop;
    int job_kind;       // 0 = single-tensor MoeScalarJob, 1 = MoeBatchJob (multi-item)
    MoeScalarJob job;
    MoeBatchJob batch_job;
    MoeScalarWarg wargs[MOE_SPOOL_MAX_THREADS];
};

static void moe_scalar_run_range(MoeScalarPool *p, int wi) {
    MoeScalarJob *j = &p->job;
    long per = (j->t->out + p->nthreads - 1) / p->nthreads;
    long r0 = (long)wi * per, r1 = r0 + per; if (r1 > j->t->out) r1 = j->t->out;
    for (long r = r0; r < r1; r++) j->y[r] = (float)g_moe_row_fn(j->blob, j->t, j->e, r, j->x);
}

// Batch-job counterpart: workers split the flattened [0,total_rows) space (concatenation of all
// items' row ranges) instead of one tensor's rows. A worker's range can span multiple items or
// sit entirely inside one -- the linear scan below (n_items <= MOE_BATCH_MAX_ITEMS=16, trivial
// cost) finds, for each flattened row index, which item owns it and the corresponding local row.
static void moe_scalar_run_range_batch(MoeScalarPool *p, int wi) {
    MoeBatchJob *bj = &p->batch_job;
    long per = (bj->total_rows + p->nthreads - 1) / p->nthreads;
    long r0 = (long)wi * per, r1 = r0 + per; if (r1 > bj->total_rows) r1 = bj->total_rows;
    int it = 0;
    for (long r = r0; r < r1; r++) {
        while (it + 1 < bj->n_items && r >= bj->items[it + 1].row_off) it++;
        while (it > 0 && r < bj->items[it].row_off) it--;   // defensive; r is monotonic per call so unused in practice
        MoeBatchItem *item = &bj->items[it];
        long local_row = r - item->row_off;
        item->y[local_row] = (float)g_moe_row_fn(item->blob, item->t, item->e, local_row, item->x);
    }
}

static void *moe_scalar_worker(void *arg) {
    MoeScalarWarg *wa = (MoeScalarWarg *)arg; MoeScalarPool *p = wa->p; int wi = wa->wi;
    int last = 0;
    for (;;) {
        pthread_mutex_lock(&p->mtx);
        while (atomic_load_explicit(&p->gen, memory_order_relaxed) == last &&
               !atomic_load_explicit(&p->stop, memory_order_relaxed))
            pthread_cond_wait(&p->cv_go, &p->mtx);
        if (atomic_load_explicit(&p->stop, memory_order_relaxed)) { pthread_mutex_unlock(&p->mtx); return NULL; }
        last = atomic_load_explicit(&p->gen, memory_order_relaxed);
        pthread_mutex_unlock(&p->mtx);
        if (p->job_kind == 1) moe_scalar_run_range_batch(p, wi); else moe_scalar_run_range(p, wi);
        pthread_mutex_lock(&p->mtx);
        int dc = atomic_fetch_add_explicit(&p->done_count, 1, memory_order_relaxed) + 1;
        if (dc == p->nthreads) pthread_cond_signal(&p->cv_done);
        pthread_mutex_unlock(&p->mtx);
    }
}

static MoeScalarPool g_moe_spool;
static int g_moe_spool_ready = 0;

static void moe_scalar_pool_init(int nthreads) {
    if (nthreads < 1) nthreads = 1; if (nthreads > MOE_SPOOL_MAX_THREADS) nthreads = MOE_SPOOL_MAX_THREADS;
    g_moe_spool.nthreads = nthreads;
    atomic_init(&g_moe_spool.gen, 0); atomic_init(&g_moe_spool.done_count, 0); atomic_init(&g_moe_spool.stop, 0);
    pthread_mutex_init(&g_moe_spool.mtx, NULL);
    pthread_cond_init(&g_moe_spool.cv_go, NULL); pthread_cond_init(&g_moe_spool.cv_done, NULL);
    for (int i = 1; i < nthreads; i++) {
        g_moe_spool.wargs[i].p = &g_moe_spool; g_moe_spool.wargs[i].wi = i;
        if (pthread_create(&g_moe_spool.th[i], NULL, moe_scalar_worker, &g_moe_spool.wargs[i]) != 0) {
            fprintf(stderr, "FATAL: moe scalar pool pthread_create failed\n"); exit(1);
        }
    }
    g_moe_spool_ready = 1;
    fprintf(stderr, "[moe scalar pool] nthreads=%d\n", nthreads);
}

// Env var read + lazy pool init on first call -- safe regardless of which of the three MoE entry
// points (run_moe_verify_mode/run_moe_batch_verify_mode/run_moe_cbatch_verify_mode) reaches here
// first, since exactly one of them ever runs per process (main() branches once, near top).
static void moe_scalar_pool_ensure(void) {
    if (g_moe_spool_ready) return;
    const char *env = getenv("QWEN_MOE_SCALAR_THREADS");
    // D3: nthreads default = 64. WHY: measured (task#109, PREFILL_MODE=0, B=4/R=12, all thread
    // counts bit-identical tokens throughout every re-sweep below). Round 1 (pre-optimization):
    // T=1 143.6s, T=2 111.0s, T=4 95.5s, T=8 87.4s. Round 2 (after group-hoisted scale/bias row
    // kernel + MLA attention threading): T=4 86.3s, T=8 78.3s, T=16 70.1s, T=24 68.2s, T=32 65.4s.
    // Round 3 (after small-call batch consolidation, ~10x fewer dispatches/token): T=4 99.4s,
    // T=8 76.7s, T=16 65.8s, T=24 60.8s, T=32 59.9s, T=40 58.2s, T=48 57.2s, T=64 56.4s -- a real
    // T=40 measurement, not the earlier "T=40" sweep point that silently clamped to
    // MOE_SPOOL_MAX_THREADS=32 (caught and fixed, see D2). Diminishing but still net-positive well
    // past bob's physical core count (4P+6E=10) in every round -- plausibly latency/memory-access
    // bound rather than compute-bound. COST: returns from T=48->64 are only -1.4%; stopped there
    // rather than chasing further diminishing gains. EXIT: re-sweep if the workload (B/R/prompt
    // set) changes meaningfully -- these numbers are this specific B=4/R=12 workload's optimum,
    // not a universal constant.
    int nthreads = env && env[0] ? atoi(env) : 64;
    // Opt-in numeric-drift optimization (task#102 2nd pass): default OFF, keeps the exact
    // group-hoisted path as the trusted default. See moe_matvec_af_row_vdsp()'s own comment.
    const char *vdsp_env = getenv("QWEN_MOE_SCALAR_VDSP");
    if (vdsp_env && vdsp_env[0] && atoi(vdsp_env) != 0) g_moe_row_fn = moe_matvec_af_row_vdsp;
    moe_scalar_pool_init(nthreads);
}

static inline void moe_scalar_pool_go_and_wait(MoeScalarPool *p) {
    pthread_mutex_lock(&p->mtx);
    atomic_store_explicit(&p->done_count, 1, memory_order_relaxed);   // this thread's own share,
                                                                       // precedes the gen bump below
    atomic_fetch_add_explicit(&p->gen, 1, memory_order_relaxed);
    pthread_cond_broadcast(&p->cv_go);
    pthread_mutex_unlock(&p->mtx);
    if (p->job_kind == 1) moe_scalar_run_range_batch(p, 0); else moe_scalar_run_range(p, 0);
    pthread_mutex_lock(&p->mtx);
    while (atomic_load_explicit(&p->done_count, memory_order_relaxed) < p->nthreads)
        pthread_cond_wait(&p->cv_done, &p->mtx);
    pthread_mutex_unlock(&p->mtx);
}

// Drop-in threaded replacement for moe_matvec_af() -- bit-identical output (see block comment
// above), only wall-clock differs. moe_forward_token()/moe_cbatch_step_scalar_one() call this
// instead of the plain version, so MoE-3a's verify mode, MoE-3d/e/f's lockstep re-verification
// fallback, MoE-4b/4c's PREFILL_MODE=0 scalar admission burst, and MoE-4c's Tier1/Tier2 all
// benefit uniformly.
static void moe_matvec_af_mt(const uint8_t *blob, MoeAFTensor *t, long e, const float *x, float *y) {
    moe_scalar_pool_ensure();
    if (g_moe_spool.nthreads == 1) { moe_matvec_af(blob, t, e, x, y); return; }
    g_moe_spool.job_kind = 0;
    g_moe_spool.job.blob = blob; g_moe_spool.job.t = t; g_moe_spool.job.e = e;
    g_moe_spool.job.x = x; g_moe_spool.job.y = y;
    moe_scalar_pool_go_and_wait(&g_moe_spool);
}

// Batched dispatch: `items` holds n_items independent matvecs (n_items <= MOE_BATCH_MAX_ITEMS)
// whose row_off/out fields are NOT yet set by the caller -- this function computes them (cumulative
// offsets in the flattened index space) and fires ONE pool dispatch instead of n_items separate
// ones. Falls back to n_items sequential moe_matvec_af_mt() calls at nthreads==1 (same reasoning
// as the single-item fallback: no threading, no benefit from batching the barrier either).
static void moe_matvec_af_batch_mt(MoeBatchItem *items, int n_items) {
    if (n_items > MOE_BATCH_MAX_ITEMS) {
        fprintf(stderr, "FATAL: moe_matvec_af_batch_mt n_items=%d > MOE_BATCH_MAX_ITEMS=%d\n",
                n_items, MOE_BATCH_MAX_ITEMS);
        exit(1);
    }
    moe_scalar_pool_ensure();
    if (g_moe_spool.nthreads == 1) {
        for (int i = 0; i < n_items; i++)
            moe_matvec_af(items[i].blob, items[i].t, items[i].e, items[i].x, items[i].y);
        return;
    }
    MoeBatchJob *bj = &g_moe_spool.batch_job;
    bj->n_items = n_items;
    long off = 0;
    for (int i = 0; i < n_items; i++) {
        bj->items[i] = items[i];
        bj->items[i].out = items[i].t->out;
        bj->items[i].row_off = off;
        off += bj->items[i].out;
    }
    bj->total_rows = off;
    g_moe_spool.job_kind = 1;
    moe_scalar_pool_go_and_wait(&g_moe_spool);
}
// ==============================================================================================

static void moe_matvec_f32(const float *w, const float *x, float *y, int out, int in) {
    for (int r = 0; r < out; r++) {
        double acc = 0.0;
        for (int c = 0; c < in; c++) acc += (double)w[(long)r*in+c] * x[c];
        y[r] = (float)acc;
    }
}
static void moe_softmax_full(float *x, int n) {
    float mx_v = x[0]; for (int i=1;i<n;i++) if (x[i]>mx_v) mx_v=x[i];
    double sum = 0.0;
    for (int i=0;i<n;i++) { x[i] = expf(x[i]-mx_v); sum += x[i]; }
    for (int i=0;i<n;i++) x[i] = (float)(x[i]/sum);
}
// silu(x) = x*sigmoid(x); swiglu(gate,up) = silu(gate)*up -- verbatim from mlx_lm's
// models/activations.py (checked directly in Phase MoE-2b, not assumed).
static void moe_swiglu_inplace(float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        gate[i] = silu * up[i];
    }
}
// Order-invariant repeated-max top-k -- correct regardless of MLX's own argpartition
// tie-break order, since the weighted MoE sum doesn't care about selection order.
static void moe_top_k_select(const float *scores, int n, int k, int *out_idx) {
    int used[128] = {0};
    for (int i = 0; i < k; i++) {
        int best = -1; float bestv = -1e30f;
        for (int j = 0; j < n; j++) if (!used[j] && scores[j] > bestv) { bestv = scores[j]; best = j; }
        used[best] = 1; out_idx[i] = best;
    }
}

typedef struct {
    MoeAFTensor *q_proj, *kv_a_proj, *kv_b_proj, *o_proj;
    MoeF32Tensor *input_ln, *post_attn_ln, *kv_a_ln;
    MoeAFTensor *dense_gate, *dense_up, *dense_down;
    MoeF32Tensor *gate_w;
    MoeAFTensor *shared_gate, *shared_up, *shared_down;
    MoeAFTensor *switch_gate, *switch_up, *switch_down;
} MoeLayerTensors;
#define MOE_MAXLAYERS 32
static MoeLayerTensors g_moe_lt[MOE_MAXLAYERS];

static void moe_resolve_layer_tensors(void) {
    char nm[256];
    for (int l = 0; l < MOE_NL; l++) {
        MoeLayerTensors *t = &g_moe_lt[l];
        snprintf(nm,sizeof nm,"model.layers.%d.self_attn.q_proj",l);            t->q_proj = moe_find_af(nm);
        snprintf(nm,sizeof nm,"model.layers.%d.self_attn.kv_a_proj_with_mqa",l);t->kv_a_proj = moe_find_af(nm);
        snprintf(nm,sizeof nm,"model.layers.%d.self_attn.kv_b_proj",l);         t->kv_b_proj = moe_find_af(nm);
        snprintf(nm,sizeof nm,"model.layers.%d.self_attn.o_proj",l);            t->o_proj = moe_find_af(nm);
        snprintf(nm,sizeof nm,"model.layers.%d.input_layernorm.weight",l);      t->input_ln = moe_find_f32(nm);
        snprintf(nm,sizeof nm,"model.layers.%d.post_attention_layernorm.weight",l); t->post_attn_ln = moe_find_f32(nm);
        snprintf(nm,sizeof nm,"model.layers.%d.self_attn.kv_a_layernorm.weight",l); t->kv_a_ln = moe_find_f32(nm);
        if (l < MOE_FIRST_DENSE_LAYERS) {
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.gate_proj",l); t->dense_gate = moe_find_af(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.up_proj",l);   t->dense_up   = moe_find_af(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.down_proj",l); t->dense_down = moe_find_af(nm);
        } else {
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.gate.weight",l); t->gate_w = moe_find_f32(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.shared_experts.gate_proj",l); t->shared_gate = moe_find_af(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.shared_experts.up_proj",l);   t->shared_up   = moe_find_af(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.shared_experts.down_proj",l); t->shared_down = moe_find_af(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.switch_mlp.gate_proj",l); t->switch_gate = moe_find_af(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.switch_mlp.up_proj",l);   t->switch_up   = moe_find_af(nm);
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.switch_mlp.down_proj",l); t->switch_down = moe_find_af(nm);
        }
    }
}

// Phase MoE-4c: was 16. moe_forward_token()/moe_mla_attention() write g_moe_K/V[l][pos] and a
// stack array float scores[MOE_MAXPOS] (moe_mla_attention(), no bounds check) -- the ragged
// re-verification path (Tier1/Tier2 below) calls these functions at pos up to
// MOE_CBATCH_MAXPOS-1=31 for margin-flagged tokens deep into a sequence, which would silently
// write past a 16-sized array. MoE-1..3f/4a/4b never triggered this (always used pos<16 or
// stayed on the separate, correctly-sized g_moe_cK/cV ragged arrays) -- a real latent bug, not a
// hypothetical one, caught by design review before any re-verification code was written.
#define MOE_MAXPOS 32
static float g_moe_K[MOE_MAXLAYERS][MOE_MAXPOS][16][192];
static float g_moe_V[MOE_MAXLAYERS][MOE_MAXPOS][16][128];

// MLA attention for one layer/position, appending to x_residual in place. Verbatim math
// from Phase MoE-2a's mla_verify.c, per-layer instead of layer-0-only.
static void moe_mla_attention(const uint8_t *af, MoeLayerTensors *t, int l, int pos, const float *h, float *x_residual) {
    float q[16*192], kv_ap[576], kv_b[4096], normed_kv[512], attn_out[16*128], o_out[2048];
    moe_matvec_af_mt(af, t->q_proj, 0, h, q);
    moe_matvec_af_mt(af, t->kv_a_proj, 0, h, kv_ap);
    float *compressed_kv = kv_ap;
    float *k_pe = kv_ap + MOE_KV_LORA_RANK;
    float *w_kvaln = (float *)(g_moe_f32_blob + t->kv_a_ln->off);
    moe_rmsnorm(compressed_kv, w_kvaln, normed_kv, MOE_KV_LORA_RANK);
    moe_matvec_af_mt(af, t->kv_b_proj, 0, normed_kv, kv_b);

    for (int hh = 0; hh < MOE_N_HEADS; hh++) {
        float *q_pe = q + hh*MOE_Q_HEAD_DIM + MOE_QK_NOPE_HD;
        for (int i = 0; i < MOE_QK_ROPE_HD; i++) q_pe[i] *= (float)g_moe_rope_mscale;
    }
    for (int i = 0; i < MOE_QK_ROPE_HD; i++) k_pe[i] *= (float)g_moe_rope_mscale;
    for (int hh = 0; hh < MOE_N_HEADS; hh++) {
        float *q_pe = q + hh*MOE_Q_HEAD_DIM + MOE_QK_NOPE_HD;
        moe_rope_traditional_apply(q_pe, MOE_QK_ROPE_HD, pos);
    }
    moe_rope_traditional_apply(k_pe, MOE_QK_ROPE_HD, pos);

    for (int hh = 0; hh < MOE_N_HEADS; hh++) {
        float *k_nope = kv_b + hh*(MOE_QK_NOPE_HD+MOE_V_HD);
        float *v_h    = kv_b + hh*(MOE_QK_NOPE_HD+MOE_V_HD) + MOE_QK_NOPE_HD;
        memcpy(g_moe_K[l][pos][hh], k_nope, MOE_QK_NOPE_HD*sizeof(float));
        memcpy(g_moe_K[l][pos][hh]+MOE_QK_NOPE_HD, k_pe, MOE_QK_ROPE_HD*sizeof(float));
        memcpy(g_moe_V[l][pos][hh], v_h, MOE_V_HD*sizeof(float));
    }

    for (int hh = 0; hh < MOE_N_HEADS; hh++) {
        float *qh = q + hh*MOE_Q_HEAD_DIM;
        float scores[MOE_MAXPOS];
        for (int j = 0; j <= pos; j++) {
            double dot = 0.0;
            for (int d = 0; d < MOE_Q_HEAD_DIM; d++) dot += (double)qh[d]*g_moe_K[l][j][hh][d];
            scores[j] = (float)(dot * g_moe_attn_scale);
        }
        float mx_s = scores[0]; for (int j=1;j<=pos;j++) if (scores[j]>mx_s) mx_s=scores[j];
        double sum = 0.0;
        for (int j=0;j<=pos;j++) { scores[j] = expf(scores[j]-mx_s); sum += scores[j]; }
        for (int j=0;j<=pos;j++) scores[j] = (float)(scores[j]/sum);
        float *oh = attn_out + hh*MOE_V_HD;
        for (int d = 0; d < MOE_V_HD; d++) {
            double acc = 0.0;
            for (int j = 0; j <= pos; j++) acc += (double)scores[j]*g_moe_V[l][j][hh][d];
            oh[d] = (float)acc;
        }
    }
    moe_matvec_af_mt(af, t->o_proj, 0, attn_out, o_out);
    for (int c = 0; c < MOE_HIDDEN; c++) x_residual[c] += o_out[c];
}

// Full 27-layer MLA+MoE forward for one token, appended to the existing verification-mode
// entry point below -- run_moe_verify_mode() drives this once per position of a fixed
// 8-token prompt (same one used throughout Phase MoE-2a/2b), dumping logits/routing exactly
// like moe2b_verify.c did, so compare_moe2b.py works unmodified against this production
// binary's output.
static void moe_forward_token(const uint8_t *af, MoeAFTensor *t_embed, MoeAFTensor *t_lmhead,
                               float *w_finalnorm, int token_id, int pos, float *logits_out,
                               FILE *routing_out) {
    static float x[2048];   // static: no stack-size concern from the caller's loop
    float h[2048], h2[2048];
    if (pos == 0) memset(x, 0, sizeof x);   // silence-only; overwritten below regardless
    for (int c = 0; c < MOE_HIDDEN; c++) x[c] = moe_decode_af(af, t_embed, 0, token_id, c);

    for (int l = 0; l < MOE_NL; l++) {
        MoeLayerTensors *t = &g_moe_lt[l];
        float *w_inln = (float *)(g_moe_f32_blob + t->input_ln->off);
        moe_rmsnorm(x, w_inln, h, MOE_HIDDEN);
        moe_mla_attention(af, t, l, pos, h, x);

        float *w_postln = (float *)(g_moe_f32_blob + t->post_attn_ln->off);
        moe_rmsnorm(x, w_postln, h2, MOE_HIDDEN);

        float mlp_out[2048];
        if (l < MOE_FIRST_DENSE_LAYERS) {
            float gate_v[16384], up_v[16384];
            moe_matvec_af_mt(af, t->dense_gate, 0, h2, gate_v);
            moe_matvec_af_mt(af, t->dense_up, 0, h2, up_v);
            moe_swiglu_inplace(gate_v, up_v, MOE_DENSE_IM);
            moe_matvec_af_mt(af, t->dense_down, 0, gate_v, mlp_out);
        } else {
            float *w_gate = (float *)(g_moe_f32_blob + t->gate_w->off);
            float router_scores[128];
            moe_matvec_f32(w_gate, h2, router_scores, MOE_N_EXPERTS, MOE_HIDDEN);
            moe_softmax_full(router_scores, MOE_N_EXPERTS);
            int top_idx[16];
            moe_top_k_select(router_scores, MOE_N_EXPERTS, MOE_TOP_K, top_idx);

            for (int c = 0; c < MOE_HIDDEN; c++) mlp_out[c] = 0.0f;
            // Task#102 2nd pass ("다 해봐" -- small-call consolidation): gate_v/up_v (both read
            // the SAME h2) across all MOE_TOP_K routed experts + the shared expert are mutually
            // independent -- batch all 2*MOE_TOP_K+2 into ONE pool dispatch instead of that many
            // separate ones. down_v depends on its own expert's swiglu'd gate/up, but the
            // MOE_TOP_K+1 down projections are independent of EACH OTHER -- second batch. Bit-
            // identical to the old sequential-per-expert version (see MoeBatchJob's own comment);
            // only dispatch/barrier overhead is amortized, per-row math is untouched.
            static float gate_v[16][4096], up_v[16][4096], down_v[16][2048];
            static float sgate_v[4096], sup_v[4096], sdown_v[2048];
            {
                MoeBatchItem items[2 * 16 + 2]; int ni = 0;
                for (int k = 0; k < MOE_TOP_K; k++) {
                    long e = top_idx[k];
                    items[ni++] = (MoeBatchItem){af, t->switch_gate, e, h2, gate_v[k], 0, 0};
                    items[ni++] = (MoeBatchItem){af, t->switch_up,   e, h2, up_v[k],   0, 0};
                }
                items[ni++] = (MoeBatchItem){af, t->shared_gate, 0, h2, sgate_v, 0, 0};
                items[ni++] = (MoeBatchItem){af, t->shared_up,   0, h2, sup_v,   0, 0};
                moe_matvec_af_batch_mt(items, ni);
            }
            {
                MoeBatchItem items[16 + 1]; int ni = 0;
                for (int k = 0; k < MOE_TOP_K; k++) {
                    long e = top_idx[k];
                    moe_swiglu_inplace(gate_v[k], up_v[k], MOE_IM_DIM);
                    items[ni++] = (MoeBatchItem){af, t->switch_down, e, gate_v[k], down_v[k], 0, 0};
                }
                moe_swiglu_inplace(sgate_v, sup_v, MOE_IM_DIM * MOE_N_SHARED);
                items[ni++] = (MoeBatchItem){af, t->shared_down, 0, sgate_v, sdown_v, 0, 0};
                moe_matvec_af_batch_mt(items, ni);
            }
            for (int k = 0; k < MOE_TOP_K; k++) {
                float wgt = router_scores[top_idx[k]];
                for (int c = 0; c < MOE_HIDDEN; c++) mlp_out[c] += wgt * down_v[k][c];
            }
            for (int c = 0; c < MOE_HIDDEN; c++) mlp_out[c] += sdown_v[c];

            if (routing_out) {
                fprintf(routing_out, "pos %d layer %d experts", pos, l);
                for (int k = 0; k < MOE_TOP_K; k++) fprintf(routing_out, " %ld:%.6f", (long)top_idx[k], router_scores[top_idx[k]]);
                fprintf(routing_out, "\n");
            }
        }
        for (int c = 0; c < MOE_HIDDEN; c++) x[c] += mlp_out[c];
    }

    float xn[2048];
    moe_rmsnorm(x, w_finalnorm, xn, MOE_HIDDEN);
    moe_matvec_af_mt(af, t_lmhead, 0, xn, logits_out);
}

// Phase MoE-4c, Tier1: re-verify ONE ragged-batch column's current step fully scalar, reading/
// writing THIS SLOT's current (possibly SME2-drift-contaminated) g_moe_cK/cV. Verbatim port of
// moe_forward_token()'s body just above (still unmodified, still used as-is by MoE-3a/3d's
// lockstep re-verification) with moe_mla_attention() replaced by moe_mla_attention_ragged() so
// it addresses this slot's ragged KV instead of the single-sequence global g_moe_K/V. Local
// (non-static) buffers, since this may be called for different (slot,token) pairs back-to-back
// within one step, unlike moe_forward_token()'s own single always-sequential caller. O(1) cost
// (~1550.84ms measured, MoE-4a), independent of pos -- removes error source (i): this step's own
// SME2 FFN/lm_head noise. Does NOT remove error source (ii): drift already baked into
// g_moe_cK/cV by earlier SME2-approximated steps (e.g. SME2-batched prefill, PREFILL_MODE=1) --
// that needs Tier2 (moe_reverify_exact(), below) instead.
// Forward decl: moe_mla_attention_ragged() itself is defined later (MoE-4a's ragged block, after
// g_moe_cK/cV), but this Tier1 function was placed here to sit textually next to moe_forward_token()
// (which it mirrors) -- needs the declaration up front since C has no implicit function decl in C99+.
static void moe_mla_attention_ragged(const uint8_t *af, MoeLayerTensors *t, int slot, int l, int pos,
                                      const float *h, float *x_residual);

static void moe_cbatch_step_scalar_one(const uint8_t *af, MoeAFTensor *t_embed, MoeAFTensor *t_lmhead,
                                        float *w_finalnorm, int token_id, int slot, int spos,
                                        float *logits_out) {
    float x[2048], h[2048], h2[2048];
    for (int c = 0; c < MOE_HIDDEN; c++) x[c] = moe_decode_af(af, t_embed, 0, token_id, c);

    for (int l = 0; l < MOE_NL; l++) {
        MoeLayerTensors *t = &g_moe_lt[l];
        float *w_inln = (float *)(g_moe_f32_blob + t->input_ln->off);
        moe_rmsnorm(x, w_inln, h, MOE_HIDDEN);
        moe_mla_attention_ragged(af, t, slot, l, spos, h, x);

        float *w_postln = (float *)(g_moe_f32_blob + t->post_attn_ln->off);
        moe_rmsnorm(x, w_postln, h2, MOE_HIDDEN);

        float mlp_out[2048];
        if (l < MOE_FIRST_DENSE_LAYERS) {
            float gate_v[16384], up_v[16384];
            moe_matvec_af_mt(af, t->dense_gate, 0, h2, gate_v);
            moe_matvec_af_mt(af, t->dense_up, 0, h2, up_v);
            moe_swiglu_inplace(gate_v, up_v, MOE_DENSE_IM);
            moe_matvec_af_mt(af, t->dense_down, 0, gate_v, mlp_out);
        } else {
            float *w_gate = (float *)(g_moe_f32_blob + t->gate_w->off);
            float router_scores[128];
            moe_matvec_f32(w_gate, h2, router_scores, MOE_N_EXPERTS, MOE_HIDDEN);
            moe_softmax_full(router_scores, MOE_N_EXPERTS);
            int top_idx[16];
            moe_top_k_select(router_scores, MOE_N_EXPERTS, MOE_TOP_K, top_idx);

            for (int c = 0; c < MOE_HIDDEN; c++) mlp_out[c] = 0.0f;
            // Task#102 2nd pass, same batching as moe_forward_token()'s mirror block above --
            // see that block's comment. Local (non-static) here too, matching this function's own
            // existing convention (it may run different (slot,token) pairs back-to-back).
            float gate_v[16][4096], up_v[16][4096], down_v[16][2048];
            float sgate_v[4096], sup_v[4096], sdown_v[2048];
            {
                MoeBatchItem items[MOE_BATCH_MAX_ITEMS]; int ni = 0;
                for (int k = 0; k < MOE_TOP_K; k++) {
                    long e = top_idx[k];
                    items[ni++] = (MoeBatchItem){af, t->switch_gate, e, h2, gate_v[k], 0, 0};
                    items[ni++] = (MoeBatchItem){af, t->switch_up,   e, h2, up_v[k],   0, 0};
                }
                items[ni++] = (MoeBatchItem){af, t->shared_gate, 0, h2, sgate_v, 0, 0};
                items[ni++] = (MoeBatchItem){af, t->shared_up,   0, h2, sup_v,   0, 0};
                moe_matvec_af_batch_mt(items, ni);
            }
            {
                MoeBatchItem items[MOE_BATCH_MAX_ITEMS]; int ni = 0;
                for (int k = 0; k < MOE_TOP_K; k++) {
                    long e = top_idx[k];
                    moe_swiglu_inplace(gate_v[k], up_v[k], MOE_IM_DIM);
                    items[ni++] = (MoeBatchItem){af, t->switch_down, e, gate_v[k], down_v[k], 0, 0};
                }
                moe_swiglu_inplace(sgate_v, sup_v, MOE_IM_DIM * MOE_N_SHARED);
                items[ni++] = (MoeBatchItem){af, t->shared_down, 0, sgate_v, sdown_v, 0, 0};
                moe_matvec_af_batch_mt(items, ni);
            }
            for (int k = 0; k < MOE_TOP_K; k++) {
                float wgt = router_scores[top_idx[k]];
                for (int c = 0; c < MOE_HIDDEN; c++) mlp_out[c] += wgt * down_v[k][c];
            }
            for (int c = 0; c < MOE_HIDDEN; c++) mlp_out[c] += sdown_v[c];
        }
        for (int c = 0; c < MOE_HIDDEN; c++) x[c] += mlp_out[c];
    }

    float xn[2048];
    moe_rmsnorm(x, w_finalnorm, xn, MOE_HIDDEN);
    moe_matvec_af_mt(af, t_lmhead, 0, xn, logits_out);
}

// Entry point: detects weights_moe/arch_config_moe.txt (QWEN_MOE_BASE overridable), and if
// present, runs the same 8-token fixed-prompt correctness-gate forward as Phase MoE-2b's
// moe2b_verify.c, dumping moe3a_c_logits.bin/moe3a_c_routing.txt for compare_moe2b.py.
// Called from the top of main(), before load_arch_cfg() -- never falls through to any
// GQA-dense-model code below it. Absent the sidecar file, this function is a fast no-op and
// the rest of main() runs 100% as before this block existed.

// ============================================================================
// Phase MoE-3b: batched (B tokens, lockstep, single shared step at seq
// position 0 for every slot) MoE forward with expert-gather dispatch. Scope:
// accuracy + gather-vs-naive throughput ONLY -- still scalar moe_matvec_af
// per gathered token (no SME2/NEON kernel change), still no multi-position
// batching (MLA/RoPE across real positions already verified in MoE-2a/2b/3a;
// this phase isolates router+gather+scatter correctness, the one genuinely
// new piece). See trackb_moe3b_results/RESULTS.md for the affine-tensor
// SME2 decomposition finding that unblocks a FUTURE phase (MoE-3c) --
// deliberately not wired in here, so a gather bug and a kernel-wiring bug
// can never be conflated in the same debugging session.
//
// mirrors serve_step()'s lockstep batching convention (qwen_infer.c, "B
// sequences advance one step together"), not cbatch_step()'s ragged
// continuous batching -- unnecessary complexity for this phase's question.
// ============================================================================

#define MOE_BATCH_MAX 64
static float g_moe_bK[MOE_MAXLAYERS][MOE_BATCH_MAX][16][192];
static float g_moe_bV[MOE_MAXLAYERS][MOE_BATCH_MAX][16][128];

// Phase MoE-4a: per-slot position-indexed KV cache for ragged continuous-batching decode.
// Every prior batched MoE path (MoE-3b~3f, g_moe_bK/bV just above) stores exactly ONE position
// per (layer,slot) -- moe_mla_attention_batched() overwrites it every call and treats attention
// as trivial single-key softmax(=1.0), since those phases only ever ran ONE lockstep step at
// position 0 by design. Real multi-step generation (this phase's actual new capability) needs
// each slot's REAL causal KV history, mirroring the dense model's [slot][layer][pos][KVD]
// layout (kslot_srv/vslot_srv, qwen_infer.c ~L1623). MOE_CBATCH_MAXPOS=32 chosen from a real
// RSS measurement (bob, 2026-08-24): peak observed RSS during a B=64 QWEN_MOE_BATCH run was
// ~5.53GB; this array adds ~1.28GB (32 * 64 slots * 32 layers * 16 heads * (192+128) dims *
// 4B), comfortably within the M4's 17.18GB and the ~9GB free+inactive vm_stat showed
// post-run -- see trackb_moe4a_results/RESULTS.md for the measurement. BSS-allocated (static),
// so unused slot/position entries cost no RSS until actually written.
#define MOE_CBATCH_MAXPOS 32
_Static_assert(MOE_CBATCH_MAXPOS <= MOE_MAXPOS,
    "moe_forward_token()/moe_mla_attention() (used by Tier1/Tier2 ragged re-verification, "
    "Phase MoE-4c) index g_moe_K/V[l][pos] and a stack scores[MOE_MAXPOS] array at pos values "
    "up to MOE_CBATCH_MAXPOS-1 -- MOE_MAXPOS must stay at least that large.");
static float g_moe_cK[MOE_MAXLAYERS][MOE_BATCH_MAX][MOE_CBATCH_MAXPOS][16][192];
static float g_moe_cV[MOE_MAXLAYERS][MOE_BATCH_MAX][MOE_CBATCH_MAXPOS][16][128];

// Verbatim port of moe_mla_attention()'s math (MoE-2a-verified, unchanged formula/RoPE/softmax)
// -- the only difference is reading/writing g_moe_cK/cV[layer][slot][pos] instead of the
// single-sequence global g_moe_K/V[layer][pos], so multiple slots' independent causal histories
// never collide. moe_mla_attention() itself (used by moe_forward_token(), MoE-3a/3d's
// re-verification path) and moe_mla_attention_batched() (MoE-3b~3f's lockstep path) are both
// left completely untouched.
static void moe_mla_attention_ragged(const uint8_t *af, MoeLayerTensors *t, int slot, int l, int pos, const float *h, float *x_residual) {
    float q[16*192], kv_ap[576], kv_b[4096], normed_kv[512], attn_out[16*128], o_out[2048];
    moe_matvec_af_mt(af, t->q_proj, 0, h, q);
    moe_matvec_af_mt(af, t->kv_a_proj, 0, h, kv_ap);
    float *compressed_kv = kv_ap;
    float *k_pe = kv_ap + MOE_KV_LORA_RANK;
    float *w_kvaln = (float *)(g_moe_f32_blob + t->kv_a_ln->off);
    moe_rmsnorm(compressed_kv, w_kvaln, normed_kv, MOE_KV_LORA_RANK);
    moe_matvec_af_mt(af, t->kv_b_proj, 0, normed_kv, kv_b);

    for (int hh = 0; hh < MOE_N_HEADS; hh++) {
        float *q_pe = q + hh*MOE_Q_HEAD_DIM + MOE_QK_NOPE_HD;
        for (int i = 0; i < MOE_QK_ROPE_HD; i++) q_pe[i] *= (float)g_moe_rope_mscale;
    }
    for (int i = 0; i < MOE_QK_ROPE_HD; i++) k_pe[i] *= (float)g_moe_rope_mscale;
    for (int hh = 0; hh < MOE_N_HEADS; hh++) {
        float *q_pe = q + hh*MOE_Q_HEAD_DIM + MOE_QK_NOPE_HD;
        moe_rope_traditional_apply(q_pe, MOE_QK_ROPE_HD, pos);
    }
    moe_rope_traditional_apply(k_pe, MOE_QK_ROPE_HD, pos);

    for (int hh = 0; hh < MOE_N_HEADS; hh++) {
        float *k_nope = kv_b + hh*(MOE_QK_NOPE_HD+MOE_V_HD);
        float *v_h    = kv_b + hh*(MOE_QK_NOPE_HD+MOE_V_HD) + MOE_QK_NOPE_HD;
        memcpy(g_moe_cK[l][slot][pos][hh], k_nope, MOE_QK_NOPE_HD*sizeof(float));
        memcpy(g_moe_cK[l][slot][pos][hh]+MOE_QK_NOPE_HD, k_pe, MOE_QK_ROPE_HD*sizeof(float));
        memcpy(g_moe_cV[l][slot][pos][hh], v_h, MOE_V_HD*sizeof(float));
    }

    for (int hh = 0; hh < MOE_N_HEADS; hh++) {
        float *qh = q + hh*MOE_Q_HEAD_DIM;
        float scores[MOE_CBATCH_MAXPOS];
        for (int j = 0; j <= pos; j++) {
            double dot = 0.0;
            for (int d = 0; d < MOE_Q_HEAD_DIM; d++) dot += (double)qh[d]*g_moe_cK[l][slot][j][hh][d];
            scores[j] = (float)(dot * g_moe_attn_scale);
        }
        float mx_s = scores[0]; for (int j=1;j<=pos;j++) if (scores[j]>mx_s) mx_s=scores[j];
        double sum = 0.0;
        for (int j=0;j<=pos;j++) { scores[j] = expf(scores[j]-mx_s); sum += scores[j]; }
        for (int j=0;j<=pos;j++) scores[j] = (float)(scores[j]/sum);
        float *oh = attn_out + hh*MOE_V_HD;
        for (int d = 0; d < MOE_V_HD; d++) {
            double acc = 0.0;
            for (int j = 0; j <= pos; j++) acc += (double)scores[j]*g_moe_cV[l][slot][j][hh][d];
            oh[d] = (float)acc;
        }
    }
    moe_matvec_af_mt(af, t->o_proj, 0, attn_out, o_out);
    for (int c = 0; c < MOE_HIDDEN; c++) x_residual[c] += o_out[c];
}

// Batched MLA attention, one batch slot at a time (caller loops b=0..B-1).
// Every slot is independently at sequence position 0 (this phase's own first
// token) -- RoPE at pos=0 is the identity rotation by construction
// (rope_traditional_apply's angle=pos/freq=0/freq=0 for every frequency), and
// self-attention over a single key softmaxes to exactly 1.0, not
// approximately -- both already-established facts, not new claims requiring
// re-verification. Otherwise identical math to moe_mla_attention().
static void moe_mla_attention_batched(const uint8_t *af, MoeLayerTensors *t, int l, int b,
                                       const float *h, float *x_residual) {
    float q[16*192], kv_ap[576], kv_b[4096], normed_kv[512], attn_out[16*128], o_out[2048];
    moe_matvec_af(af, t->q_proj, 0, h, q);
    moe_matvec_af(af, t->kv_a_proj, 0, h, kv_ap);
    float *compressed_kv = kv_ap;
    float *k_pe = kv_ap + MOE_KV_LORA_RANK;
    float *w_kvaln = (float *)(g_moe_f32_blob + t->kv_a_ln->off);
    moe_rmsnorm(compressed_kv, w_kvaln, normed_kv, MOE_KV_LORA_RANK);
    moe_matvec_af(af, t->kv_b_proj, 0, normed_kv, kv_b);

    // rope_mscale pre-multiply still applied generally (config-driven, not hardcoded to the
    // 1.0 this specific model happens to have -- same reasoning as moe_mla_attention()).
    for (int hh = 0; hh < MOE_N_HEADS; hh++) {
        float *q_pe = q + hh*MOE_Q_HEAD_DIM + MOE_QK_NOPE_HD;
        for (int i = 0; i < MOE_QK_ROPE_HD; i++) q_pe[i] *= (float)g_moe_rope_mscale;
    }
    for (int i = 0; i < MOE_QK_ROPE_HD; i++) k_pe[i] *= (float)g_moe_rope_mscale;
    // rope_traditional_apply(..., pos=0) intentionally skipped: identity rotation at pos=0.

    for (int hh = 0; hh < MOE_N_HEADS; hh++) {
        float *k_nope = kv_b + hh*(MOE_QK_NOPE_HD+MOE_V_HD);
        float *v_h    = kv_b + hh*(MOE_QK_NOPE_HD+MOE_V_HD) + MOE_QK_NOPE_HD;
        memcpy(g_moe_bK[l][b][hh], k_nope, MOE_QK_NOPE_HD*sizeof(float));
        memcpy(g_moe_bK[l][b][hh]+MOE_QK_NOPE_HD, k_pe, MOE_QK_ROPE_HD*sizeof(float));
        memcpy(g_moe_bV[l][b][hh], v_h, MOE_V_HD*sizeof(float));
    }

    // single-key self-attention: softmax over exactly one score is exactly 1.0 (not merely a
    // reasonable approximation) -- computed via the general formula anyway (not hand-collapsed
    // to "copy V") so this stays a byte-for-byte structural mirror of moe_mla_attention(),
    // minimizing transcription-bug risk against the already-verified function.
    for (int hh = 0; hh < MOE_N_HEADS; hh++) {
        float *qh = q + hh*MOE_Q_HEAD_DIM;
        double dot = 0.0;
        for (int d = 0; d < MOE_Q_HEAD_DIM; d++) dot += (double)qh[d]*g_moe_bK[l][b][hh][d];
        float score = (float)(dot * g_moe_attn_scale);
        (void)score;   // softmax({score}) == 1.0 exactly regardless of value
        float *oh = attn_out + hh*MOE_V_HD;
        for (int d = 0; d < MOE_V_HD; d++) oh[d] = g_moe_bV[l][b][hh][d];
    }
    moe_matvec_af(af, t->o_proj, 0, attn_out, o_out);
    for (int c = 0; c < MOE_HIDDEN; c++) x_residual[c] += o_out[c];
}

// Per-layer expert -> member-token gather table. Rebuilt every layer (routing differs per
// layer). MOE_N_EXPERTS <= 128 fits comfortably; MOE_BATCH_MAX members per expert bucket is
// the worst case (every token in the batch picking the same expert).
typedef struct {
    int member_tok[MOE_BATCH_MAX];    // token indices (into the batch) that picked this expert
    float member_score[MOE_BATCH_MAX];// that token's already-full-softmax weight for this expert
    int n_members;
} MoeExpertBucket;
static MoeExpertBucket g_moe_bucket[128];

// Phase MoE-3c: real SME2 dispatch for routed-expert (switch_mlp) tensors, using the
// symmetric+affine-correction decomposition verified in Phase MoE-3b
// (verify_af_decompose.c/.py) and re-verified against the real production-shaped GEMM path
// (rel_l2 ~3.9e-3, moe3c_sme2_bench.c) before this block was written. Lazy per-(layer,expert,
// projection) repack -- eager repack of all 4992 expert-projection tensors measured at
// ~7.29GB (kai_sme2_rhs_packed_bytes(1408,2048)=1,531,904 bytes x 4992), which real vm_stat
// headroom on this 17.18GB M4 (~6.7GB free+inactive, already sharing the 9.81GB raw AF blob)
// makes a real OOM/swap risk -- not assumed, measured (see RESULTS.md). Purely additive to
// moe_matvec_af() (kept, used as the fallback when SME2 is unavailable or a shape/repack
// fails) and to WT/g_wt[]/K_Q4G64/kai_route() (untouched, as in every prior MoE phase).
typedef struct {
    void *rhs_packed;      // KleidiAI-packed symmetric RHS, NULL until lazily built
    float *adj_bias;       // [out*ng], adj_bias[row*ng+g] = 8*scale[row,g] + bias[row,g]
    int ready;              // 0=not attempted, 1=ready, -1=sticky-failed (shape/hw/alloc issue)
} MoeSme2Slot;
// [layer][cache_slot][0=gate,1=up,2=down] -- routed (switch_mlp) tensors use cache_slot ==
// their real expert index (0..MOE_N_EXPERTS-1, MOE_N_EXPERTS=64 for this model, confirmed via
// arch_config_moe.txt). Phase MoE-3e reserves fixed slots beyond that range for the always-
// M=B tensors, which have exactly one copy in the AF blob (blob address always e=0) but need a
// cache identity distinct from routed expert 0's slot at the same layer (shared_experts runs
// at every layer >=MOE_FIRST_DENSE_LAYERS, where expert 0 may also be routed-to that layer).
#define MOE_SME2_SLOT_DENSE  64
#define MOE_SME2_SLOT_SHARED 65
#define MOE_SME2_SLOT_LMHEAD 66
static MoeSme2Slot g_moe_sme2[MOE_MAXLAYERS][128][3];
static void *g_moe_sme2_lhs_scratch = NULL;
static int g_moe_sme2_scratch_max_in = 0;

// f16p-LHS: separate cache/scratch from the int8-LHS path above -- packed-RHS tile layout
// differs (qsi4c32ps1s0scalef16 vs qsi4c32ps4s0sf16, different nr/kr), so a slot from one path
// is never valid input to the other's GEMM call.
//
// D-f16lhs-3 (2026-08-25): promoted to DEFAULT (was opt-in-only via QWEN_MOE_SME2_F16LHS=1).
//   WHY: real measurement (this session, 2 workload scales) -- REVERIFY=off accuracy int8-LHS
//   80.0-87.7% vs f16p-LHS 93.0-95.3% (matches PREFILL_MODE=0 pure-scalar ceiling exactly; the
//   only remaining mismatch is the pre-existing, reverify-can't-fix-it vdsp-vs-mlx_lm case).
//   Speed: f16p-LHS only 3.2-15% slower than int8-LHS (gap shrinks as batch M grows -- 3.2% at
//   B=16), and both stay 1.25-2.4x faster than pure scalar. f16p-LHS reaches int8-matching
//   speed AND scalar-matching accuracy WITHOUT needing the margin-reverify mechanism at all --
//   strictly dominates the "int8-LHS + Tier1/Tier2 reverify" combination (88.0s, still slower
//   than pure scalar) that this whole MoE-4c phase was built around.
//   COST: f16p-LHS's own feature gate (FEAT_SME2+FEAT_FP16) is checked independently of the
//   int8 path's (FEAT_SME2+SME_I8I32) -- if a future target has SME2 but not FEAT_FP16, this
//   falls all the way back to scalar `moe_matvec_af()` rather than to the still-fast int8-LHS
//   path (same single-fallback-tier design the int8 path itself already has, not a new
//   limitation this decision introduces -- not expected to matter on any Apple Silicon target,
//   both features confirmed present together on M4).
//   EXIT: set QWEN_MOE_SME2_F16LHS=0 to opt back into the int8-LHS path (kept fully intact,
//   byte-identical to its pre-2026-08-25 behavior) -- no code change needed to roll back.
static MoeSme2Slot g_moe_sme2_f16lhs[MOE_MAXLAYERS][128][3];
static void *g_moe_sme2_f16lhs_lhs_scratch = NULL;
static int g_moe_sme2_f16lhs_scratch_max_in = 0;
static int g_moe_sme2_f16lhs_mode_cached = -1;
static int moe_sme2_f16lhs_mode(void) {
    if (g_moe_sme2_f16lhs_mode_cached < 0) {
        const char *e = getenv("QWEN_MOE_SME2_F16LHS");
        // D-f16lhs-3: unset/empty -> default ON now. Only an explicit "0" opts back into int8-LHS.
        g_moe_sme2_f16lhs_mode_cached = (e && e[0] && atoi(e) == 0) ? 0 : 1;
        fprintf(stderr, "[moe sme2] LHS precision: %s%s\n",
                g_moe_sme2_f16lhs_mode_cached ? "f16p (default)" : "int8",
                g_moe_sme2_f16lhs_mode_cached ? "" : " (QWEN_MOE_SME2_F16LHS=0 opt-out)");
    }
    return g_moe_sme2_f16lhs_mode_cached;
}
static void moe_sme2_ensure_scratch_f16lhs(int max_in) {
    if (g_moe_sme2_f16lhs_lhs_scratch && max_in <= g_moe_sme2_f16lhs_scratch_max_in) return;
    size_t nb = kai_sme2_f16lhs_lhs_scratch_bytes(MOE_BATCH_MAX, max_in);
    if (g_moe_sme2_f16lhs_lhs_scratch) { free(g_moe_sme2_f16lhs_lhs_scratch); g_moe_sme2_f16lhs_lhs_scratch = NULL; }
    if (nb > 0) { g_moe_sme2_f16lhs_lhs_scratch = aligned_alloc(64, (nb + 63) & ~(size_t)63); g_moe_sme2_f16lhs_scratch_max_in = max_in; }
}

// Phase MoE-3e: prior to this phase every group_smart caller's `in` was <=2048 (switch_mlp/
// shared_experts gate/up) so a single 2048-sized allocation, never revisited, was safe. Once
// dense_down (in=MOE_DENSE_IM=10944) also dispatches through here, a caller can request a
// larger scratch than what's already allocated -- grow (not just allocate-once) when that
// happens, tracking the largest `in` seen so far.
static void moe_sme2_ensure_scratch(int max_in) {
    if (g_moe_sme2_lhs_scratch && max_in <= g_moe_sme2_scratch_max_in) return;
    size_t nb = kai_sme2_lhs_scratch_bytes(MOE_BATCH_MAX, max_in);
    if (g_moe_sme2_lhs_scratch) { free(g_moe_sme2_lhs_scratch); g_moe_sme2_lhs_scratch = NULL; }
    if (nb > 0) { g_moe_sme2_lhs_scratch = aligned_alloc(64, (nb + 63) & ~(size_t)63); g_moe_sme2_scratch_max_in = max_in; }
}

// Lazy repack: on first dispatch of this (layer,expert,projection), extract the AF blob's
// own packed nibbles + per-group scale into K_Q4G64-layout buffers (byte-identical to what
// kai_sme2_repack_q4g64() already expects -- Phase MoE-1's nibble-layout finding, still
// valid), repack via the UNMODIFIED vendor kernel (bias ignored -- it only ever knew
// symmetric q4g64), and separately build adj_bias directly from the AF blob's real scale+bias
// arrays (no repacking needed for that part, it's just arithmetic on values already read).
// blob_e addresses the AF blob's stacked-expert storage for tsr (real routed expert index for
// switch_mlp; always 0 for the single-copy dense/shared_experts/lm_head tensors). cache_e keys
// g_moe_sme2[layer][cache_e][proj_idx] -- for switch_mlp cache_e==blob_e (unchanged behavior);
// for the always-M=B tensors cache_e is one of the MOE_SME2_SLOT_* reserved constants above, so
// their cache entry never collides with a routed expert's entry at the same layer even though
// blob_e is always 0 for both.
static void moe_sme2_ensure_ready(const uint8_t *af, MoeAFTensor *tsr, long blob_e, long cache_e, int layer, int proj_idx) {
    int f16lhs = moe_sme2_f16lhs_mode();
    MoeSme2Slot *slot = f16lhs ? &g_moe_sme2_f16lhs[layer][cache_e][proj_idx] : &g_moe_sme2[layer][cache_e][proj_idx];
    if (slot->ready != 0) return;
    int out = tsr->out, in = tsr->in, ng = tsr->ng;
    if (f16lhs) {
        if (!kai_sme2_f16lhs_available() || !kai_sme2_f16lhs_shape_ok(out, in)) { slot->ready = -1; return; }
    } else {
        if (!kai_sme2_available() || !kai_sme2_shape_ok(out, in)) { slot->ready = -1; return; }
    }

    size_t row_pbytes = (size_t)(in / 2);
    uint8_t *sym_packed = malloc((size_t)out * row_pbytes);
    float *sym_scales = malloc((size_t)out * ng * sizeof(float));
    float *adj_bias = malloc((size_t)out * ng * sizeof(float));
    long row_words = in / 8;
    for (int row = 0; row < out; row++) {
        const uint8_t *src = af + tsr->packed_off + ((size_t)blob_e * tsr->out + row) * row_words * 4;
        memcpy(sym_packed + (size_t)row * row_pbytes, src, row_pbytes);
        for (int g = 0; g < ng; g++) {
            long sidx = tsr->scale_off + (((size_t)blob_e * tsr->out + row) * ng + g) * 4;
            long bidx = tsr->bias_off + (((size_t)blob_e * tsr->out + row) * ng + g) * 4;
            float scale, bias;
            memcpy(&scale, af + sidx, 4);
            memcpy(&bias, af + bidx, 4);
            sym_scales[row*ng+g] = scale;
            adj_bias[row*ng+g] = 8.0f*scale + bias;
        }
    }
    size_t rhs_bytes = f16lhs ? kai_sme2_f16lhs_rhs_packed_bytes(out, in) : kai_sme2_rhs_packed_bytes(out, in);
    void *rhs = aligned_alloc(64, (rhs_bytes + 63) & ~(size_t)63);
    int rc = f16lhs ? kai_sme2_repack_q4g64_f16lhs(out, in, sym_packed, sym_scales, rhs, rhs_bytes)
                     : kai_sme2_repack_q4g64(out, in, sym_packed, sym_scales, rhs, rhs_bytes);
    free(sym_packed); free(sym_scales);
    if (rc != 0 || !rhs) { free(rhs); free(adj_bias); slot->ready = -1; return; }
    slot->rhs_packed = rhs; slot->adj_bias = adj_bias; slot->ready = 1;
}

// Group GEMM for M already-gathered, contiguous rows (x_group[M][in] -> y_group[M][out]).
// New dispatch gate per the approved plan: SME2 whenever hardware+repack are ready, NO
// M-threshold (V2 measured SME2 winning NEON at every M from 1 to 64 for this exact expert
// shape family -- kai_route()'s existing M>=kai_sme2_min_m()=16 gate is dense-projection-
// tuned and deliberately NOT reused here). Falls back to per-member moe_matvec_af() if SME2
// isn't ready for this tensor (hardware absent, shape illegal, or repack failed).
static void moe_matvec_af_group_smart(const uint8_t *af, MoeAFTensor *tsr, long blob_e, long cache_e, int layer, int proj_idx,
                                       const float *x_group, int M, float *y_group) {
    int f16lhs = moe_sme2_f16lhs_mode();
    moe_sme2_ensure_ready(af, tsr, blob_e, cache_e, layer, proj_idx);
    MoeSme2Slot *slot = f16lhs ? &g_moe_sme2_f16lhs[layer][cache_e][proj_idx] : &g_moe_sme2[layer][cache_e][proj_idx];
    int in = tsr->in, out = tsr->out;
    if (slot->ready != 1) {
        for (int m = 0; m < M; m++) moe_matvec_af(af, tsr, blob_e, x_group + (size_t)m*in, y_group + (size_t)m*out);
        return;
    }
    if (f16lhs) {
        moe_sme2_ensure_scratch_f16lhs(in);
        kai_sme2_gemm_f16lhs(M, out, in, x_group, slot->rhs_packed, NULL, y_group, g_moe_sme2_f16lhs_lhs_scratch);
    } else {
        moe_sme2_ensure_scratch(in);   // grows to fit the largest `in` seen so far (see comment above)
        kai_sme2_gemm_f32(M, out, in, x_group, slot->rhs_packed, NULL, y_group, g_moe_sme2_lhs_scratch);
    }
    int ng = tsr->ng;
    for (int m = 0; m < M; m++) {
        const float *xr = x_group + (size_t)m*in;
        // 256 covers every `in` this dispatcher now sees, including dense_down's
        // MOE_DENSE_IM=10944 (ng=171) -- the prior 64-sized bound was only valid for switch_mlp/
        // shared_experts (in<=2816, ng<=44) and silently stack-smashed once dense joined (real
        // crash caught via a production B=64 run, not by inspection).
        float groupsum[256];
        for (int g = 0; g < ng; g++) {
            double s = 0.0;
            for (int c = g*64; c < g*64+64 && c < in; c++) s += xr[c];
            groupsum[g] = (float)s;
        }
        float *yr = y_group + (size_t)m*out;
        for (int row = 0; row < out; row++) {
            double corr = 0.0;
            for (int g = 0; g < ng; g++) corr += (double)slot->adj_bias[row*ng+g] * groupsum[g];
            yr[row] += (float)corr;
        }
    }
}

// One batched MoE-FFN layer step: routes each of B tokens independently (moe_matvec_f32 +
// moe_softmax_full + moe_top_k_select, unchanged from moe_forward_token()), gathers per
// expert into a real contiguous group buffer (the actual memcpy gather cost V3 measured, not
// simulated), dispatches each expert's WHOLE group through moe_matvec_af_group_smart() (SME2
// when ready, scalar moe_matvec_af() fallback otherwise -- Phase MoE-3c), then scatters
// weighted results back to each token's own accumulator, plus each token's always-applied
// shared_experts pass (never gathered -- every token uses it, nothing to group; also stays on
// moe_matvec_af(), out of this phase's scope per the approved plan).
static void moe_ffn_batched(const uint8_t *af, MoeLayerTensors *t, int l, int B,
                             float h2_batch[][2048], float mlp_out_batch[][2048]) {
    for (int b = 0; b < B; b++) for (int c = 0; c < MOE_HIDDEN; c++) mlp_out_batch[b][c] = 0.0f;

    for (int e = 0; e < MOE_N_EXPERTS; e++) g_moe_bucket[e].n_members = 0;
    float *w_gate = (float *)(g_moe_f32_blob + t->gate_w->off);
    for (int b = 0; b < B; b++) {
        float router_scores[128];
        moe_matvec_f32(w_gate, h2_batch[b], router_scores, MOE_N_EXPERTS, MOE_HIDDEN);
        moe_softmax_full(router_scores, MOE_N_EXPERTS);
        int top_idx[16];
        moe_top_k_select(router_scores, MOE_N_EXPERTS, MOE_TOP_K, top_idx);
        for (int k = 0; k < MOE_TOP_K; k++) {
            int e = top_idx[k];
            MoeExpertBucket *bk = &g_moe_bucket[e];
            bk->member_tok[bk->n_members] = b;
            bk->member_score[bk->n_members] = router_scores[e];
            bk->n_members++;
        }
    }

    // FLAT buffers, tightly packed by each call's actual out/in -- moe_matvec_af_group_smart()
    // (and kai_sme2_gemm_f32() underneath it) both assume row stride == out (writes) / in
    // (reads) with NO padding. A fixed-width 2D array (e.g. float g[MOE_BATCH_MAX][4096])
    // decayed to a flat pointer would silently mismatch that stride against out=MOE_IM_DIM
    // (1408 != 4096) -- caught via a real end-to-end batch run (worst_rel_l2 blew up to 1.7,
    // not noise) before this fix, not by inspection alone. MOE_BATCH_MAX*2048 sized generously
    // (2048 = MOE_HIDDEN, the largest of {MOE_HIDDEN, MOE_IM_DIM} among routed-expert shapes)
    // and indexed with each call's own out/in explicitly, never via a 2D array's own stride.
    static float x_group[MOE_BATCH_MAX*2048];
    static float gate_group[MOE_BATCH_MAX*2048], up_group[MOE_BATCH_MAX*2048], down_group[MOE_BATCH_MAX*2048];
    for (int e = 0; e < MOE_N_EXPERTS; e++) {
        MoeExpertBucket *bk = &g_moe_bucket[e];
        int M = bk->n_members;
        if (M == 0) continue;
        for (int m = 0; m < M; m++) memcpy(x_group + (size_t)m*MOE_HIDDEN, h2_batch[bk->member_tok[m]], MOE_HIDDEN*sizeof(float));
        moe_matvec_af_group_smart(af, t->switch_gate, e, e, l, 0, x_group, M, gate_group);
        moe_matvec_af_group_smart(af, t->switch_up,   e, e, l, 1, x_group, M, up_group);
        for (int m = 0; m < M; m++) moe_swiglu_inplace(gate_group + (size_t)m*MOE_IM_DIM, up_group + (size_t)m*MOE_IM_DIM, MOE_IM_DIM);
        moe_matvec_af_group_smart(af, t->switch_down, e, e, l, 2, gate_group, M, down_group);
        for (int m = 0; m < M; m++) {
            int b = bk->member_tok[m];
            float wgt = bk->member_score[m];
            for (int c = 0; c < MOE_HIDDEN; c++) mlp_out_batch[b][c] += wgt * down_group[(size_t)m*MOE_HIDDEN+c];
        }
    }

    // Phase MoE-3e: shared_experts always runs for every token (no routing/gather needed) --
    // one M=B group GEMM per projection instead of B individual scalar matvecs. h2_batch is a
    // real fixed-width [][2048] array (row stride == MOE_HIDDEN == shared_gate/up's `in`), so
    // it's used directly as x_group with no separate gather memcpy.
    static float sgate_group[MOE_BATCH_MAX*4096], sup_group[MOE_BATCH_MAX*4096], sdown_group[MOE_BATCH_MAX*2048];
    int sh_im = MOE_IM_DIM * MOE_N_SHARED;
    moe_matvec_af_group_smart(af, t->shared_gate, 0, MOE_SME2_SLOT_SHARED, l, 0, &h2_batch[0][0], B, sgate_group);
    moe_matvec_af_group_smart(af, t->shared_up,   0, MOE_SME2_SLOT_SHARED, l, 1, &h2_batch[0][0], B, sup_group);
    for (int b = 0; b < B; b++) moe_swiglu_inplace(sgate_group + (size_t)b*sh_im, sup_group + (size_t)b*sh_im, sh_im);
    moe_matvec_af_group_smart(af, t->shared_down, 0, MOE_SME2_SLOT_SHARED, l, 2, sgate_group, B, sdown_group);
    for (int b = 0; b < B; b++)
        for (int c = 0; c < MOE_HIDDEN; c++) mlp_out_batch[b][c] += sdown_group[(size_t)b*MOE_HIDDEN+c];
}

// Naive (non-gathered) reference for the throughput comparison: same router + same per-token
// switch_mlp/shared_experts math, but dispatched one (token,expert) pair at a time in router-
// selection order, matching the CPU cost shape of Phase MoE-3a's per-token
// moe_forward_token() loop (no gather grouping, no shared contiguous member iteration).
static void moe_ffn_naive_batched(const uint8_t *af, MoeLayerTensors *t, int B,
                                   float h2_batch[][2048], float mlp_out_batch[][2048]) {
    float *w_gate = (float *)(g_moe_f32_blob + t->gate_w->off);
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < MOE_HIDDEN; c++) mlp_out_batch[b][c] = 0.0f;
        float router_scores[128];
        moe_matvec_f32(w_gate, h2_batch[b], router_scores, MOE_N_EXPERTS, MOE_HIDDEN);
        moe_softmax_full(router_scores, MOE_N_EXPERTS);
        int top_idx[16];
        moe_top_k_select(router_scores, MOE_N_EXPERTS, MOE_TOP_K, top_idx);
        for (int k = 0; k < MOE_TOP_K; k++) {
            long e = top_idx[k];
            float wgt = router_scores[e];
            float gate_v[4096], up_v[4096], down_v[2048];
            moe_matvec_af(af, t->switch_gate, e, h2_batch[b], gate_v);
            moe_matvec_af(af, t->switch_up, e, h2_batch[b], up_v);
            moe_swiglu_inplace(gate_v, up_v, MOE_IM_DIM);
            moe_matvec_af(af, t->switch_down, e, gate_v, down_v);
            for (int c = 0; c < MOE_HIDDEN; c++) mlp_out_batch[b][c] += wgt * down_v[c];
        }
        float sgate_v[4096], sup_v[4096], sdown_v[2048];
        moe_matvec_af(af, t->shared_gate, 0, h2_batch[b], sgate_v);
        moe_matvec_af(af, t->shared_up, 0, h2_batch[b], sup_v);
        moe_swiglu_inplace(sgate_v, sup_v, MOE_IM_DIM * MOE_N_SHARED);
        moe_matvec_af(af, t->shared_down, 0, sgate_v, sdown_v);
        for (int c = 0; c < MOE_HIDDEN; c++) mlp_out_batch[b][c] += sdown_v[c];
    }
}

// Full batched 27-layer forward for B tokens, one shared lockstep step at seq position 0 for
// every slot (see moe_mla_attention_batched()'s comment). use_gather selects moe_ffn_batched()
// (gather-grouped dispatch) vs moe_ffn_naive_batched() (per-token dispatch order) for the
// MoE-FFN layers -- the ONLY code-path difference between the two, so a wall-clock comparison
// isolates gather's own effect and nothing else.
static void moe_forward_batch(const uint8_t *af, MoeAFTensor *t_embed, MoeAFTensor *t_lmhead,
                               float *w_finalnorm, const int *token_ids, int B,
                               float logits_out[][102400], int use_gather) {
    static float x[MOE_BATCH_MAX][2048];
    static float h[MOE_BATCH_MAX][2048];
    static float h2[MOE_BATCH_MAX][2048];
    static float mlp_out[MOE_BATCH_MAX][2048];

    for (int b = 0; b < B; b++)
        for (int c = 0; c < MOE_HIDDEN; c++) x[b][c] = moe_decode_af(af, t_embed, 0, token_ids[b], c);

    for (int l = 0; l < MOE_NL; l++) {
        MoeLayerTensors *t = &g_moe_lt[l];
        float *w_inln = (float *)(g_moe_f32_blob + t->input_ln->off);
        float *w_postln = (float *)(g_moe_f32_blob + t->post_attn_ln->off);

        for (int b = 0; b < B; b++) {
            moe_rmsnorm(x[b], w_inln, h[b], MOE_HIDDEN);
            moe_mla_attention_batched(af, t, l, b, h[b], x[b]);
            moe_rmsnorm(x[b], w_postln, h2[b], MOE_HIDDEN);
        }

        if (l < MOE_FIRST_DENSE_LAYERS) {
            if (use_gather) {
                // Phase MoE-3e: layer-0 dense MLP always runs for every token (no routing) --
                // one M=B group GEMM per projection. h2 is a real [][2048] fixed-width array
                // (stride == MOE_HIDDEN == dense_gate/up's `in`), used directly as x_group.
                // Buffers sized to the same 16384 upper bound the prior per-token scalar code
                // already used for MOE_DENSE_IM.
                static float dgate_group[MOE_BATCH_MAX*16384], dup_group[MOE_BATCH_MAX*16384];
                moe_matvec_af_group_smart(af, t->dense_gate, 0, MOE_SME2_SLOT_DENSE, l, 0, &h2[0][0], B, dgate_group);
                moe_matvec_af_group_smart(af, t->dense_up,   0, MOE_SME2_SLOT_DENSE, l, 1, &h2[0][0], B, dup_group);
                for (int b = 0; b < B; b++) moe_swiglu_inplace(dgate_group + (size_t)b*MOE_DENSE_IM, dup_group + (size_t)b*MOE_DENSE_IM, MOE_DENSE_IM);
                moe_matvec_af_group_smart(af, t->dense_down, 0, MOE_SME2_SLOT_DENSE, l, 2, dgate_group, B, &mlp_out[0][0]);
            } else {
                for (int b = 0; b < B; b++) {
                    float gate_v[16384], up_v[16384];
                    moe_matvec_af(af, t->dense_gate, 0, h2[b], gate_v);
                    moe_matvec_af(af, t->dense_up, 0, h2[b], up_v);
                    moe_swiglu_inplace(gate_v, up_v, MOE_DENSE_IM);
                    moe_matvec_af(af, t->dense_down, 0, gate_v, mlp_out[b]);
                }
            }
        } else if (use_gather) {
            moe_ffn_batched(af, t, l, B, h2, mlp_out);
        } else {
            moe_ffn_naive_batched(af, t, B, h2, mlp_out);
        }

        for (int b = 0; b < B; b++)
            for (int c = 0; c < MOE_HIDDEN; c++) x[b][c] += mlp_out[b][c];
    }

    if (use_gather) {
        // Phase MoE-3e: lm_head always runs for every token, one M=B group GEMM. logits_out is
        // a real [][102400] fixed-width array (stride == MOE_VOCAB == t_lmhead's `out`), so the
        // group call writes into it directly with no separate scatter step. layer=0 is reused
        // (lm_head has no real layer concept) -- safe because cache_e=MOE_SME2_SLOT_LMHEAD is
        // already a unique key on its own, distinct from every other slot at layer 0.
        static float xn_group[MOE_BATCH_MAX*2048];
        for (int b = 0; b < B; b++) moe_rmsnorm(x[b], w_finalnorm, xn_group + (size_t)b*MOE_HIDDEN, MOE_HIDDEN);
        moe_matvec_af_group_smart(af, t_lmhead, 0, MOE_SME2_SLOT_LMHEAD, 0, 0, xn_group, B, &logits_out[0][0]);
    } else {
        for (int b = 0; b < B; b++) {
            float xn[2048];
            moe_rmsnorm(x[b], w_finalnorm, xn, MOE_HIDDEN);
            moe_matvec_af(af, t_lmhead, 0, xn, logits_out[b]);
        }
    }
}

// Phase MoE-3f: finalized production margin-threshold policy. Data-derived from MoE-3e's real
// B=8/16/32/64 measurement of "minimal threshold for 100% argmax parity with the naive scalar
// path" (see trackb_moe3e_results/RESULTS.md) -- NOT a formula fit to those 4 points. An
// untested B always inherits the larger (safer, more-reverification) neighboring measured
// threshold rather than interpolating downward, so this stays conservative between breakpoints.
static double moe_baware_threshold(int B) {
    if (B <= 16) return 0.0;    // measured: B=8,16 both reach 100% raw, no reverification needed
    if (B <= 32) return 0.2;    // measured: B=32 needs 0.2 for 100%
    return 0.32;                // measured: B=64 needs 0.32 for 100% (MOE_BATCH_MAX caps B<=64)
}

// Entry point: QWEN_MOE_BATCH=<B> (with weights_moe/ present) runs this phase's batched
// verification instead of MoE-3a's single-sequence mode. Checked at the same point in
// run_moe_verify_mode() -- see the call site edit in main()'s dispatch. Real distinct text
// prompts (P0.2's REAL_TEXTS, reused for consistency -- not synthetic), one token (each
// prompt's own first token) per batch slot, matching this phase's plan.
static int run_moe_batch_verify_mode(int argc, char **argv, const char *dir) {
    (void)argc; (void)argv;
    const char *batch_env = getenv("QWEN_MOE_BATCH");
    if (!batch_env || !batch_env[0]) return 0;
    int B = atoi(batch_env);
    if (B < 1 || B > MOE_BATCH_MAX) { fprintf(stderr, "FATAL: QWEN_MOE_BATCH=%d out of [1,%d]\n", B, MOE_BATCH_MAX); exit(1); }

    fprintf(stderr, "[moe batch] QWEN_MOE_BATCH=%d -- Phase MoE-3b batched verification mode\n", B);
    char moe_dir[900], path[1024];
    snprintf(moe_dir, sizeof moe_dir, "%s/weights_moe", dir);

    // 64 real token ids, generated by actually tokenizing P0.2's REAL_TEXTS corpus with the
    // real DeepSeek tokenizer (sliding-window strided sampling, same anti-duplicate technique
    // P0.2's own script used) -- not synthetic/fabricated placeholders. 51/64 distinct.
    static const int real_first_tokens[MOE_BATCH_MAX] = {
        100000,276,4357,254,3042,254,11,23382,3987,4810,33044,14486,8376,12,21528,54188,
        12,11,7071,8404,11,100000,13,13,10957,317,8666,10616,5532,3164,64625,7195,
        457,207,44274,2018,280,895,37548,10165,285,288,13,285,10988,8909,2577,5559,
        13930,2156,12650,331,245,4941,13,10948,9423,1699,8204,280,100000,56081,895,11
    };
    int token_ids[MOE_BATCH_MAX];
    for (int b = 0; b < B; b++) token_ids[b] = real_first_tokens[b];

    snprintf(path, sizeof path, "%s/layout_af.txt", moe_dir); moe_load_layout_af(path);
    snprintf(path, sizeof path, "%s/deepseek_moe_af.bin", moe_dir);
    long af_bytes; uint8_t *af_blob = moe_mmap_file(path, &af_bytes);
    snprintf(path, sizeof path, "%s/layout_f32.txt", moe_dir); moe_load_layout_f32(path);
    snprintf(path, sizeof path, "%s/deepseek_moe_f32.bin", moe_dir);
    long f32_bytes; g_moe_f32_blob = moe_mmap_file(path, &f32_bytes);
    fprintf(stderr, "[moe batch load] af %ld bytes, f32 %ld bytes\n", af_bytes, f32_bytes);
    moe_resolve_layer_tensors();

    MoeAFTensor *t_embed = moe_find_af("model.embed_tokens");
    MoeAFTensor *t_lmhead = moe_find_af("lm_head");
    MoeF32Tensor *t_finalnorm = moe_find_f32("model.norm.weight");
    float *w_finalnorm = (float *)(g_moe_f32_blob + t_finalnorm->off);

    static float logits_gather[MOE_BATCH_MAX][102400];
    static float logits_naive[MOE_BATCH_MAX][102400];

    struct timespec t0, t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    moe_forward_batch(af_blob, t_embed, t_lmhead, w_finalnorm, token_ids, B, logits_naive, 0);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    moe_forward_batch(af_blob, t_embed, t_lmhead, w_finalnorm, token_ids, B, logits_gather, 1);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double ms_naive  = (t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)/1e6;
    double ms_gather = (t2.tv_sec-t1.tv_sec)*1e3 + (t2.tv_nsec-t1.tv_nsec)/1e6;

    // Accuracy gate: naive-dispatch-order and gather-dispatch-order must match exactly per
    // token (gather only reorders which (token,expert) pair is computed when, never changes
    // the math) -- this IS the phase's real correctness question, checked directly, not
    // inferred from "should be the same by construction".
    double worst_abs = 0.0, worst_rel = 0.0;
    for (int b = 0; b < B; b++) {
        double sse = 0.0, ssref = 0.0;
        for (int v = 0; v < MOE_VOCAB; v++) {
            double d = (double)logits_gather[b][v] - (double)logits_naive[b][v];
            sse += d*d; ssref += (double)logits_naive[b][v]*(double)logits_naive[b][v];
            double ad = fabs(d); if (ad > worst_abs) worst_abs = ad;
        }
        double rel = ssref > 0 ? sqrt(sse/ssref) : (sse == 0 ? 0.0 : 1.0);
        if (rel > worst_rel) worst_rel = rel;
        int am_g = 0; float bm_g = logits_gather[b][0];
        int am_n = 0; float bm_n = logits_naive[b][0];
        for (int v = 1; v < MOE_VOCAB; v++) {
            if (logits_gather[b][v] > bm_g) { bm_g = logits_gather[b][v]; am_g = v; }
            if (logits_naive[b][v]  > bm_n) { bm_n = logits_naive[b][v];  am_n = v; }
        }
        fprintf(stderr, "[moe batch] slot %2d token %6d: naive_argmax=%d gather_argmax=%d %s\n",
                b, token_ids[b], am_n, am_g, am_n == am_g ? "[MATCH]" : "[MISMATCH]");
    }
    fprintf(stderr, "[moe batch] worst_abs_diff=%.6e worst_rel_l2=%.6e (expect ~0.0, gather is a pure dispatch-order change)\n", worst_abs, worst_rel);
    fprintf(stderr, "[moe batch] B=%d naive=%.2fms gather=%.2fms speedup=%.3fx\n", B, ms_naive, ms_gather, ms_naive/ms_gather);

    // Phase MoE-3d: margin-gated selective scalar re-verification. Margin is computed from the
    // GATHER path's OWN logits (top1 vs top2) -- the only signal available at real decision
    // time, not a comparison against the naive answer (which a real deployment wouldn't have).
    // Root cause established before this phase: the SME2-vs-scalar argmax flip is a
    // deterministic per-TOKEN property (verified: identical tokens flip identically regardless
    // of which batch/slot they appear in), NOT a batch-size or group-size effect -- so a
    // margin-based per-token gate is the right lever, not an M-based one (which was considered
    // and correctly discarded before this phase, per the approved plan).
    static float margin_g[MOE_BATCH_MAX];
    static int argmax_g[MOE_BATCH_MAX];
    for (int b = 0; b < B; b++) {
        int am = 0; float bm = logits_gather[b][0];
        for (int v = 1; v < MOE_VOCAB; v++) if (logits_gather[b][v] > bm) { bm = logits_gather[b][v]; am = v; }
        float second = -1e30f;
        for (int v = 0; v < MOE_VOCAB; v++) if (v != am && logits_gather[b][v] > second) second = logits_gather[b][v];
        argmax_g[b] = am;
        margin_g[b] = bm - second;
    }

    const char *thr_env = getenv("QWEN_MOE_MARGIN_THRESHOLD");
    const char *sweep_env = getenv("QWEN_MOE_SWEEP");
    static const double sweep_thresholds[] = {0.0, 0.05, 0.1, 0.2, 0.32, 0.5, 1e9};
    // Phase MoE-3f: with no env vars set at all, the engine now applies the finalized B-aware
    // production policy (moe_baware_threshold(), derived from MoE-3e's real B=8/16/32/64 sweep)
    // as a single threshold, instead of always running the full research sweep. Explicit
    // QWEN_MOE_MARGIN_THRESHOLD still overrides with one manual value (unchanged); explicit
    // QWEN_MOE_SWEEP=1 (only meaningful when threshold is unset) opts back into the old 7-point
    // sweep for research use -- both prior behaviors preserved verbatim, just no longer default.
    int explicit_thr = thr_env && thr_env[0];
    int do_sweep = !explicit_thr && sweep_env && sweep_env[0];
    int n_thr = do_sweep ? (int)(sizeof(sweep_thresholds)/sizeof(sweep_thresholds[0])) : 1;
    double single_thr = explicit_thr ? atof(thr_env) : moe_baware_threshold(B);

    static float logits_hybrid[MOE_BATCH_MAX][102400];
    for (int ti = 0; ti < n_thr; ti++) {
        double threshold = do_sweep ? sweep_thresholds[ti] : single_thr;
        for (int b = 0; b < B; b++) memcpy(logits_hybrid[b], logits_gather[b], sizeof logits_hybrid[b]);

        struct timespec rt0, rt1;
        clock_gettime(CLOCK_MONOTONIC, &rt0);
        int n_reverified = 0;
        for (int b = 0; b < B; b++) {
            if (margin_g[b] < threshold) {
                n_reverified++;
                moe_forward_token(af_blob, t_embed, t_lmhead, w_finalnorm, token_ids[b], 0, logits_hybrid[b], NULL);
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &rt1);
        double ms_reverify = (rt1.tv_sec-rt0.tv_sec)*1e3 + (rt1.tv_nsec-rt0.tv_nsec)/1e6;
        double ms_hybrid_total = ms_gather + ms_reverify;

        int n_match = 0;
        for (int b = 0; b < B; b++) {
            int am_h = 0; float bm_h = logits_hybrid[b][0];
            for (int v = 1; v < MOE_VOCAB; v++) if (logits_hybrid[b][v] > bm_h) { bm_h = logits_hybrid[b][v]; am_h = v; }
            int am_n = 0; float bm_n = logits_naive[b][0];
            for (int v = 1; v < MOE_VOCAB; v++) if (logits_naive[b][v] > bm_n) { bm_n = logits_naive[b][v]; am_n = v; }
            if (am_h == am_n) n_match++;
        }
        fprintf(stderr, "[moe3d] threshold=%.4g reverified=%d/%d argmax_match=%d/%d reverify_ms=%.2f hybrid_total_ms=%.2f speedup_vs_naive=%.3fx\n",
                threshold, n_reverified, B, n_match, B, ms_reverify, ms_hybrid_total, ms_naive/ms_hybrid_total);
    }

    fprintf(stderr, "RESULT: MoE-3d batched forward + margin sweep complete, B=%d\n", B);
    return 1;
}

// Phase MoE-4a: ragged continuous-batching MoE forward -- A active columns, each carrying its
// own slot id and its own real generation position (spos[m]), mirroring the dense model's
// cbatch_step() (qwen_infer.c ~L1882: "slot"/"spos" per column, not a single shared pos like
// moe_forward_batch()'s lockstep B). Attention is genuinely per-column (moe_mla_attention_ragged
// above, reading/writing that column's own KV history) since MLA can't be batched across
// different causal positions -- exactly why dense's cbatch_step() also loops attention
// per-column despite batching every other projection. Everything past attention (dense-layer
// SME2, router+switch_mlp+shared_experts, lm_head) reuses the EXACT same functions MoE-3e/3f
// already verified (moe_ffn_batched(), moe_matvec_af_group_smart()) called with A in place of
// B -- both already take a runtime row count with no lockstep assumption baked in, so no
// change was needed there at all.
static void moe_cbatch_step(const uint8_t *af, MoeAFTensor *t_embed, MoeAFTensor *t_lmhead,
                             float *w_finalnorm, const int *token_ids, const int *slot,
                             const int *spos, int A, float logits_out[][102400],
                             int want_logits) {
    static float x[MOE_BATCH_MAX][2048];
    static float h[MOE_BATCH_MAX][2048];
    static float h2[MOE_BATCH_MAX][2048];
    static float mlp_out[MOE_BATCH_MAX][2048];

    for (int m = 0; m < A; m++)
        for (int c = 0; c < MOE_HIDDEN; c++) x[m][c] = moe_decode_af(af, t_embed, 0, token_ids[m], c);

    for (int l = 0; l < MOE_NL; l++) {
        MoeLayerTensors *t = &g_moe_lt[l];
        float *w_inln = (float *)(g_moe_f32_blob + t->input_ln->off);
        float *w_postln = (float *)(g_moe_f32_blob + t->post_attn_ln->off);

        for (int m = 0; m < A; m++) {
            moe_rmsnorm(x[m], w_inln, h[m], MOE_HIDDEN);
            moe_mla_attention_ragged(af, t, slot[m], l, spos[m], h[m], x[m]);
            moe_rmsnorm(x[m], w_postln, h2[m], MOE_HIDDEN);
        }

        if (l < MOE_FIRST_DENSE_LAYERS) {
            static float dgate_group[MOE_BATCH_MAX*16384], dup_group[MOE_BATCH_MAX*16384];
            moe_matvec_af_group_smart(af, t->dense_gate, 0, MOE_SME2_SLOT_DENSE, l, 0, &h2[0][0], A, dgate_group);
            moe_matvec_af_group_smart(af, t->dense_up,   0, MOE_SME2_SLOT_DENSE, l, 1, &h2[0][0], A, dup_group);
            for (int m = 0; m < A; m++) moe_swiglu_inplace(dgate_group + (size_t)m*MOE_DENSE_IM, dup_group + (size_t)m*MOE_DENSE_IM, MOE_DENSE_IM);
            moe_matvec_af_group_smart(af, t->dense_down, 0, MOE_SME2_SLOT_DENSE, l, 2, dgate_group, A, &mlp_out[0][0]);
        } else {
            moe_ffn_batched(af, t, l, A, h2, mlp_out);
        }

        for (int m = 0; m < A; m++)
            for (int c = 0; c < MOE_HIDDEN; c++) x[m][c] += mlp_out[m][c];
    }

    if (!want_logits) return;   /* Phase MoE-4b: mirrors dense cbatch_step()'s qwen_infer.c L1963 --
                                    pure-prefill steps (no decode column this step) skip the single
                                    largest GEMM in the model (lm_head, out=102400 x in=2048). */
    static float xn_group[MOE_BATCH_MAX*2048];
    for (int m = 0; m < A; m++) moe_rmsnorm(x[m], w_finalnorm, xn_group + (size_t)m*MOE_HIDDEN, MOE_HIDDEN);
    moe_matvec_af_group_smart(af, t_lmhead, 0, MOE_SME2_SLOT_LMHEAD, 0, 0, xn_group, A, &logits_out[0][0]);
}

// Phase MoE-4a entry point: QWEN_MOE_CBATCH=1 runs a STATIC ragged batch -- N real prompts of
// different lengths, all admitted to slots 0..N-1 up front (no online admission, that's
// MoE-4b's scope per the approved plan), each prefilled via the already-verified sequential
// moe_forward_token() (its per-position KV, g_moe_K/V, is then copied into this slot's own
// g_moe_cK/cV entries -- moe_forward_token() itself is completely unmodified, this function
// just reads its output cache afterward), then decoded step by step via moe_cbatch_step() with
// the compact list of still-active slots (A shrinks for real as prompts finish -- genuine
// ragged behavior, not simulated). No margin re-verification wired in this phase (explicitly
// out of scope per the plan: re-verifying a pos>0 token correctly needs that slot's real
// scalar KV history, which moe_forward_token()'s single always-from-pos-0 path doesn't have
// mid-sequence -- a real design question left to a later phase).
#define MOE_CBATCH_N 8
#define MOE_CBATCH_KNEW 12
#define MOE_CBATCH_MAXPLEN 8
// Phase MoE-4b: online admission. QWEN_MOE_CB_ONLINE=1 (default 0, see the branch at the bottom
// of run_moe_cbatch_verify_mode()) replaces MoE-4a's static "admit all N up front" scheduler with
// a real request queue: requests arrive (QWEN_MOE_CB_ARRIVE, step-indexed for determinism -- see
// design doc D10), get admitted into whichever slot is free (including slots freed mid-run by
// eviction -- D8, no KV clearing needed), and prefill either scalar (QWEN_MOE_CB_PREFILL_MODE=0,
// mirrors MoE-4a's admission body exactly) or SME2-batched mixed into the same moe_cbatch_step()
// call as decode columns (=1, the default -- D6: scalar prefill would stall every decode slot for
// ~12s per admitted prompt, a functional regression for online serving, not just a slow one).
#define MOE_CB4B_RMAX 64

// D9 capacity guards: three real unchecked writes exist below MOE_CBATCH_MAXPOS
// (g_moe_cK/cV[l][slot][pos], and moe_mla_attention_ragged()'s own float scores[MOE_CBATCH_MAXPOS]
// stack array) -- MoE-4a stayed safe only because its workload constants happened to fit safely
// under the cap. Online admission can build requests whose plen+maxnew exceeds it, so check here.
static void moe_cb4b_admit_guard(int *plen, int *maxnew, int r) {
    if (plen[r] >= MOE_CBATCH_MAXPOS) {
        fprintf(stderr, "[moe cb4b] WARNING: req %d prompt_len=%d >= MOE_CBATCH_MAXPOS=%d, dropping\n",
                r, plen[r], MOE_CBATCH_MAXPOS);
        plen[r] = -1;   // sentinel: caller skips this request entirely
        return;
    }
    if (plen[r] + maxnew[r] > MOE_CBATCH_MAXPOS) {
        int clamped = MOE_CBATCH_MAXPOS - plen[r];
        fprintf(stderr, "[moe cb4b] WARNING: req %d plen=%d+maxnew=%d > MOE_CBATCH_MAXPOS=%d, "
                "clamping maxnew to %d\n", r, plen[r], maxnew[r], MOE_CBATCH_MAXPOS, clamped);
        maxnew[r] = clamped;
    }
}

// D3 invariant: within one step, columns of the same slot must appear at strictly increasing
// compact index in strictly increasing spos (moe_mla_attention_ragged() writes g_moe_cK/cV[pos]
// then immediately reads j=0..pos in the SAME call, unlike dense's cbatch_step() which splits
// the KV-write and attention loops into two separate passes -- see design doc D3 for the proof).
static void moe_cb4b_assert_invariants(const int *slots, const int *sposs, int A) {
    if (A > MOE_BATCH_MAX) { fprintf(stderr, "[moe cb4b] CHECK FAIL: A=%d > MOE_BATCH_MAX\n", A); exit(1); }
    int last_spos[MOE_BATCH_MAX]; for (int s = 0; s < MOE_BATCH_MAX; s++) last_spos[s] = -1;
    for (int m = 0; m < A; m++) {
        if (sposs[m] < 0 || sposs[m] >= MOE_CBATCH_MAXPOS) {
            fprintf(stderr, "[moe cb4b] CHECK FAIL: column %d slot %d spos=%d out of range\n", m, slots[m], sposs[m]);
            exit(1);
        }
        if (sposs[m] <= last_spos[slots[m]]) {
            fprintf(stderr, "[moe cb4b] CHECK FAIL: slot %d spos not strictly ascending (%d after %d)\n",
                    slots[m], sposs[m], last_spos[slots[m]]);
            exit(1);
        }
        last_spos[slots[m]] = sposs[m];
    }
}

// ==============================================================================================
// Phase MoE-4c: margin-gated re-verification for the ragged path, expanded scope. MoE-4b's own
// real finding motivates this: PREFILL_MODE=1's SME2-batched prefill corrupts g_moe_cK/cV, and
// that corruption surfaces LATER in decode (e.g. prompt 2's real divergence is at output index
// 6, i.e. the 6th DECODE step after prefill -- not at the prefill-completion token itself),
// confirmed via the C0 regression re-run just before this block was written. So re-verification
// must cover both the decode-emit site (step 4) AND the prefill-completion-emit site (step 5) of
// the online scheduler below, via one shared helper.
// ==============================================================================================

// Per-request realized token history (request-indexed, not slot-indexed: MoE-4b slots are reused
// across requests over time, D8 -- a request's own token identity is the stable one Tier2 needs
// to replay). Filled at admission (prompt range, known immediately, independent of
// PREFILL_MODE) and at each emit site (decode/prefill-completion) below.
static int rq_hist[MOE_CB4B_RMAX][MOE_CBATCH_MAXPOS];

// Shadow KV lane pool for Tier2's exact replay cache. Deliberately NOT eagerly seeded at
// admission (MoE-4c's plan-stage correction to the original design): for PREFILL_MODE=0 eager
// seeding would be free (the scalar admission burst already computes this KV, just save it
// instead of discarding it), but for PREFILL_MODE=1 (the default) admission does NO scalar
// compute at all -- eager seeding there would mean paying full scalar prefill cost TWICE (once
// fast via SME2 for real serving, once slow via scalar purely for the shadow), undermining most
// of PREFILL_MODE=1's throughput advantage. Instead: a request's shadow lane is built lazily,
// on its FIRST actual Tier2 escalation, by replaying rq_hist[req][0..pos] from scratch -- the
// same "bounded, not necessarily cheap" worst case the original plan already accepted, just
// moved from admission-time to first-flag-time. MOE_CB4C_LANES=8 lanes * ~21MB/lane ~= 168MB.
#define MOE_CB4C_LANES 8
static float g_moe_sK[MOE_MAXLAYERS][MOE_CB4C_LANES][MOE_CBATCH_MAXPOS][16][192];
static float g_moe_sV[MOE_MAXLAYERS][MOE_CB4C_LANES][MOE_CBATCH_MAXPOS][16][128];
static int   g_moe_sh_req[MOE_CB4C_LANES];     // -1 = free, else owning request id
static int   g_moe_sh_valid[MOE_CB4C_LANES];   // positions 0..valid-1 already replayed+cached
static int   g_moe_sh_next_evict = 0;

static int moe_shadow_valid_peek(int req) {   // read-only: how much of `req`'s history is cached, 0 if no lane
    for (int i = 0; i < MOE_CB4C_LANES; i++) if (g_moe_sh_req[i] == req) return g_moe_sh_valid[i];
    return 0;
}
// Acquires (or allocates, evicting round-robin if the pool is full) a lane for `req`. A request
// whose lane gets evicted under pool pressure just pays a fresh from-scratch replay on its next
// escalation -- a possible extra cost under heavy load, never a correctness issue.
static int moe_shadow_lane_for(int req) {
    for (int i = 0; i < MOE_CB4C_LANES; i++) if (g_moe_sh_req[i] == req) return i;
    int lane = g_moe_sh_next_evict;
    g_moe_sh_next_evict = (g_moe_sh_next_evict + 1) % MOE_CB4C_LANES;
    g_moe_sh_req[lane] = req;
    g_moe_sh_valid[lane] = 0;
    return lane;
}

// Tier2: exact scalar answer for request `req` at position `pos`, via moe_forward_token()
// (unmodified, its own g_moe_K/V) replaying rq_hist[req][0..pos]. Incremental: only replays
// positions beyond what this request's lane already has cached, so a request's TOTAL Tier2
// scalar cost across its whole life is bounded by its final position, never by how many times
// it's flagged (a position is never recomputed once cached).
static void moe_reverify_exact(const uint8_t *af, MoeAFTensor *t_embed, MoeAFTensor *t_lmhead,
                                float *w_finalnorm, int req, int pos, float *logits_out,
                                int *n_scalar_tok_out) {
    int lane = moe_shadow_lane_for(req);
    int start = g_moe_sh_valid[lane];
    if (start > pos) start = 0;   // defensive; a live request's pos only grows, shouldn't trigger

    for (int l = 0; l < MOE_NL; l++)
        for (int p = 0; p < start; p++) {
            memcpy(g_moe_K[l][p], g_moe_sK[l][lane][p], sizeof g_moe_K[l][p]);
            memcpy(g_moe_V[l][p], g_moe_sV[l][lane][p], sizeof g_moe_V[l][p]);
        }

    static float logits_tmp[102400];
    int n_scalar = 0;
    for (int p = start; p <= pos; p++) {
        moe_forward_token(af, t_embed, t_lmhead, w_finalnorm, rq_hist[req][p], p, logits_tmp, NULL);
        n_scalar++;
    }
    memcpy(logits_out, logits_tmp, MOE_VOCAB * sizeof(float));

    for (int l = 0; l < MOE_NL; l++)
        for (int p = start; p <= pos; p++) {
            memcpy(g_moe_sK[l][lane][p], g_moe_K[l][p], sizeof g_moe_K[l][p]);
            memcpy(g_moe_sV[l][lane][p], g_moe_V[l][p], sizeof g_moe_V[l][p]);
        }
    g_moe_sh_valid[lane] = pos + 1;
    if (n_scalar_tok_out) *n_scalar_tok_out = n_scalar;
}

// (A,pos)-aware threshold, mirroring moe_baware_threshold()'s (MoE-3f) discipline: never
// invented, always measured. Task#101 (B=4,R=12, PREFILL_MODE=1, unlimited budget, threshold=1e9
// i.e. always-escalate) captured the FULL margin distribution for this workload: 85 tier1 calls,
// exactly 1 real SME2/scalar disagreement (req6 pos6, margin=0.0817), all 84 agreements had
// margin>=0.1351 except a handful of agreements clustered at 0.03-0.10 that never actually
// flagged a real error. Task#102's own re-run at threshold=0.1 (chosen to sit just above the one
// known real disagreement, 0.0817) reproduced the SAME 95.3% accuracy as unlimited-budget
// threshold=1e9 with far fewer escalations (10 tier1 calls instead of 85) -- confirming 0.1 is a
// real, workload-validated cut point, not a guess. This is still a single-workload measurement
// (not yet an (A,pos)-aware step function like moe_baware_threshold() -- that needs more diverse
// workloads to earn a step function honestly; a flat scalar is what today's data supports).
// QWEN_MOE_CBATCH_THRESHOLD overrides this for targeted experiments.
static double g_moe_cb4c_threshold_override = -1.0;
static double moe_cbatch_threshold(int A, int pos) {
    (void)A; (void)pos;
    if (g_moe_cb4c_threshold_override >= 0.0) return g_moe_cb4c_threshold_override;
    return 0.1;   // measured, task#102: B=4 R=12 PREFILL_MODE=1, 2026-08-24
}

static double moe_cb4c_margin(const float *logits) {
    int am = 0; float bm = logits[0];
    for (int v = 1; v < MOE_VOCAB; v++) if (logits[v] > bm) { bm = logits[v]; am = v; }
    float second = -1e30f;
    for (int v = 0; v < MOE_VOCAB; v++) if (v != am && logits[v] > second) second = logits[v];
    return bm - second;
}

// Shared by step 4 (decode emit) and step 5 (prefill-completion emit) of the online scheduler.
// reverify_mode: 0=off, 1=Tier1-only, 2=Tier1-then-Tier2-if-disagree ("auto" maps to 2 for now).
// Computes margin from logits_inout (the SME2 gather path's own output -- the only signal
// available at real decision time, MoE-3d's principle, unchanged). Below threshold and budget
// allows: Tier1 first (cheap, O(1)); if Tier1's argmax disagrees with the SME2 answer and mode
// requests it, escalate to Tier2 (exact, cost bounded by how much of `req`'s history isn't yet
// shadow-cached). Overwrites logits_inout in place whenever re-verification changes the answer.
static void moe_cb4c_maybe_reverify(const uint8_t *af, MoeAFTensor *t_embed, MoeAFTensor *t_lmhead,
                                     float *w_finalnorm, int slot, int req, int pos, int token_id,
                                     int A, float *logits_inout, int *budget_inout,
                                     int reverify_mode, int resync) {
    if (reverify_mode == 0) return;
    if (moe_cb4c_margin(logits_inout) >= moe_cbatch_threshold(A, pos)) return;
    if (*budget_inout <= 0) return;

    int am_sme2 = 0; float bm = logits_inout[0];
    for (int v = 1; v < MOE_VOCAB; v++) if (logits_inout[v] > bm) { bm = logits_inout[v]; am_sme2 = v; }
    double margin_seen = moe_cb4c_margin(logits_inout);

    static float logits_t1[102400];
    double t1_t0 = nowt();
    moe_cbatch_step_scalar_one(af, t_embed, t_lmhead, w_finalnorm, token_id, slot, pos, logits_t1);
    double t1_ms = (nowt() - t1_t0) * 1000.0;
    *budget_inout -= 1;
    int am_t1 = 0; float bm_t1 = logits_t1[0];
    for (int v = 1; v < MOE_VOCAB; v++) if (logits_t1[v] > bm_t1) { bm_t1 = logits_t1[v]; am_t1 = v; }

    // Task#101 핵심 재현실험용 계측: prompt2류 divergence가 실제로 어느 (req,pos)에서
    // margin이 threshold를 밑돌아 Tier1/Tier2가 개입하는지 실측으로 남긴다 (가정 아님).
    // Task#102b 처리량 재진단용: t1_ms(Tier1 1콜 실제비용)를 함께 남겨 "1550ms/token
    // 상수 가정"이 여전히 유효한지 직접 검증(RESULTS.md의 quadratic 추정은 미검증 주장이었음).
    fprintf(stderr, "[moe cb4c] tier1 req=%d slot=%d pos=%d margin=%.4f sme2=%d t1=%d %s t1_ms=%.2f\n",
            req, slot, pos, margin_seen, am_sme2, am_t1, am_t1 == am_sme2 ? "agree" : "DISAGREE", t1_ms);

    // Task#102 최적화(threshold=0.1 실측에서 Tier2 escalation당 처음부터-replay 비용이
    // baseline보다 느리게 만든다는 걸 확인 -- 234s vs PREFILL_MODE=0 182.3s 목표 미달, 사용자
    // 승인 하에 추가): MLA의 K/V projection은 이번 스텝 hidden state에서 직접 나오므로(과거
    // attention 컨텍스트가 정확하다는 전제 하에) Tier1이 방금 계산한 이번 (req,pos)의 K/V는
    // "과거가 정확했다면" 그 자체로 정확하다. shadow lane의 valid가 정확히 pos와 같을 때
    // (직전까지 GAP 없이 연속 검증된 상태)만 pos+1로 확장 -- gap이 있으면 확장하지 않아
    // Tier2의 exactness(처음부터 replay가 필요할 때는 여전히 처음부터)를 지킨다.
    {
        int lane = moe_shadow_lane_for(req);
        if (g_moe_sh_valid[lane] == pos) {
            for (int l = 0; l < MOE_NL; l++) {
                memcpy(g_moe_sK[l][lane][pos], g_moe_cK[l][slot][pos], sizeof g_moe_sK[l][lane][pos]);
                memcpy(g_moe_sV[l][lane][pos], g_moe_cV[l][slot][pos], sizeof g_moe_sV[l][lane][pos]);
            }
            g_moe_sh_valid[lane] = pos + 1;
        }
    }

    if (am_t1 == am_sme2 || reverify_mode < 2) {
        memcpy(logits_inout, logits_t1, MOE_VOCAB * sizeof(float));
        return;
    }

    int already = moe_shadow_valid_peek(req);
    int cost = pos + 1 - already; if (cost < 0) cost = 0;
    if (*budget_inout < cost) {
        fprintf(stderr, "[moe cb4c] tier2 req=%d pos=%d SKIPPED budget-starved (need=%d have=%d)\n",
                req, pos, cost, *budget_inout);
        memcpy(logits_inout, logits_t1, MOE_VOCAB * sizeof(float));   // budget-starved: keep Tier1
        return;
    }
    static float logits_t2[102400];
    int n_scalar = 0;
    double t2_t0 = nowt();
    moe_reverify_exact(af, t_embed, t_lmhead, w_finalnorm, req, pos, logits_t2, &n_scalar);
    double t2_ms = (nowt() - t2_t0) * 1000.0;
    *budget_inout -= n_scalar;
    int am_t2 = 0; float bm_t2 = logits_t2[0];
    for (int v = 1; v < MOE_VOCAB; v++) if (logits_t2[v] > bm_t2) { bm_t2 = logits_t2[v]; am_t2 = v; }
    fprintf(stderr, "[moe cb4c] tier2 req=%d pos=%d n_scalar=%d t2=%d (was sme2=%d t1=%d) %s t2_ms=%.2f ms_per_tok=%.2f\n",
            req, pos, n_scalar, am_t2, am_sme2, am_t1, am_t2 == am_t1 ? "t1_confirmed" : "t1_WRONG",
            t2_ms, n_scalar > 0 ? t2_ms / n_scalar : 0.0);
    memcpy(logits_inout, logits_t2, MOE_VOCAB * sizeof(float));
    if (resync) {
        int lane = moe_shadow_lane_for(req);   // now resident after moe_reverify_exact() above
        for (int l = 0; l < MOE_NL; l++)
            for (int p = 0; p <= pos; p++) {
                memcpy(g_moe_cK[l][slot][p], g_moe_sK[l][lane][p], sizeof g_moe_cK[l][slot][p]);
                memcpy(g_moe_cV[l][slot][p], g_moe_sV[l][lane][p], sizeof g_moe_cV[l][slot][p]);
            }
    }
}

static int run_moe_cbatch_verify_mode(int argc, char **argv, const char *dir) {
    (void)argc; (void)argv;
    const char *cb_env = getenv("QWEN_MOE_CBATCH");
    if (!cb_env || !cb_env[0]) return 0;
    const char *online_env = getenv("QWEN_MOE_CB_ONLINE");
    int online = online_env && online_env[0] && atoi(online_env) != 0;

    fprintf(stderr, "[moe cbatch] QWEN_MOE_CBATCH=1 -- Phase MoE-4a/4b ragged batch mode (ONLINE=%d)\n", online);
    char moe_dir[900], path[1024];
    snprintf(moe_dir, sizeof moe_dir, "%s/weights_moe", dir);

    // Real prompts (tokenized on macstudio with the real DeepSeek tokenizer,
    // moe4a_reference_capture.py) -- deliberately varied lengths AND varied per-slot generation
    // targets (moe_cbatch_gen below), so slots finish at genuinely different decode steps
    // (varying prompt length alone would NOT do this: every slot starts decoding in the same
    // first step regardless of prompt length, so if every slot generated the same fixed token
    // count they'd all evict on the same step too -- ragged eviction needs staggered stopping
    // points, not just staggered starting positions). Each target is a prefix length of the
    // same slot's already-captured 12-token ground truth (greedy decode is deterministic, so a
    // shorter prefix of a longer captured generation is exactly what a standalone shorter
    // generation would produce).
    static const int prompt_len[MOE_CBATCH_N] = {4,5,6,7,8,5,6,4};
    static const int moe_cbatch_gen[MOE_CBATCH_N] = {4,6,8,10,12,5,9,3};
    static const int prompt_ids[MOE_CBATCH_N][MOE_CBATCH_MAXPLEN] = {
        {100000,549,4345,280},
        {100000,10616,266,75214,1855},
        {100000,549,14471,30925,6230,2577},
        {100000,49099,46756,37926,13930,54188,285},
        {100000,549,56764,9862,438,441,245,2816},
        {100000,10522,3343,9531,3071},
        {100000,11059,4385,278,79386,562},
        {100000,549,17298,3327},
    };

    // Config (MOE_NL/MOE_N_EXPERTS/etc.) already loaded by the caller (run_moe_verify_mode(),
    // which checks this mode alongside run_moe_batch_verify_mode() before its own sequential-
    // mode loading) -- this function loads its own af/f32 blob mmaps exactly like
    // run_moe_batch_verify_mode() does (each mode independently mmaps; cheap, mmap doesn't
    // read the file eagerly).
    snprintf(path, sizeof path, "%s/layout_af.txt", moe_dir); moe_load_layout_af(path);
    snprintf(path, sizeof path, "%s/deepseek_moe_af.bin", moe_dir);
    long af_bytes; uint8_t *af_blob = moe_mmap_file(path, &af_bytes);
    snprintf(path, sizeof path, "%s/layout_f32.txt", moe_dir); moe_load_layout_f32(path);
    snprintf(path, sizeof path, "%s/deepseek_moe_f32.bin", moe_dir);
    long f32_bytes; g_moe_f32_blob = moe_mmap_file(path, &f32_bytes);
    fprintf(stderr, "[moe cbatch load] af %ld bytes, f32 %ld bytes\n", af_bytes, f32_bytes);
    moe_resolve_layer_tensors();

    MoeAFTensor *t_embed = moe_find_af("model.embed_tokens");
    MoeAFTensor *t_lmhead = moe_find_af("lm_head");
    MoeF32Tensor *t_finalnorm = moe_find_f32("model.norm.weight");
    float *w_finalnorm = (float *)(g_moe_f32_blob + t_finalnorm->off);

    if (!online) {
    // ==========================================================================================
    // Phase MoE-4a: static scheduler (all N admitted up front). UNCHANGED from MoE-4a -- this
    // branch's byte-for-byte identity with the pre-MoE-4b binary is Gate 1 of the MoE-4b plan.
    // ==========================================================================================
    static int cb_active[MOE_CBATCH_N];
    static int cb_pos[MOE_CBATCH_N];
    static int cb_nout[MOE_CBATCH_N];
    static int cb_next_tok[MOE_CBATCH_N];
    static int generated[MOE_CBATCH_N][MOE_CBATCH_KNEW];

    struct timespec pt0, pt1; clock_gettime(CLOCK_MONOTONIC, &pt0);
    int n_prefill_tok = 0;
    for (int s = 0; s < MOE_CBATCH_N; s++) {
        static float logits1[102400];
        for (int p = 0; p < prompt_len[s]; p++) {
            moe_forward_token(af_blob, t_embed, t_lmhead, w_finalnorm, prompt_ids[s][p], p, logits1, NULL);
            n_prefill_tok++;
        }
        // Copy this slot's freshly-prefilled single-sequence KV (moe_forward_token()'s own
        // g_moe_K/V, positions 0..prompt_len[s]-1) into its dedicated ragged-cache slot.
        for (int l = 0; l < MOE_NL; l++)
            for (int p = 0; p < prompt_len[s]; p++) {
                memcpy(g_moe_cK[l][s][p], g_moe_K[l][p], sizeof g_moe_K[l][p]);
                memcpy(g_moe_cV[l][s][p], g_moe_V[l][p], sizeof g_moe_V[l][p]);
            }
        int am = 0; float bm = logits1[0];
        for (int v = 1; v < MOE_VOCAB; v++) if (logits1[v] > bm) { bm = logits1[v]; am = v; }
        cb_active[s] = 1; cb_pos[s] = prompt_len[s]; cb_nout[s] = 0; cb_next_tok[s] = am;
    }
    clock_gettime(CLOCK_MONOTONIC, &pt1);
    double ms_prefill = (pt1.tv_sec-pt0.tv_sec)*1e3 + (pt1.tv_nsec-pt0.tv_nsec)/1e6;
    fprintf(stderr, "[moe cbatch] prefill done, %d slots active, %d tokens, %.2fms (%.2fms/token scalar rate)\n",
            MOE_CBATCH_N, n_prefill_tok, ms_prefill, ms_prefill/n_prefill_tok);

    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    static float logits_step[MOE_BATCH_MAX][102400];
    int step = 0;
    while (1) {
        int A = 0;
        static int ids[MOE_CBATCH_N], slots[MOE_CBATCH_N], sposs[MOE_CBATCH_N];
        for (int s = 0; s < MOE_CBATCH_N; s++) if (cb_active[s]) {
            ids[A] = cb_next_tok[s]; slots[A] = s; sposs[A] = cb_pos[s]; A++;
        }
        if (A == 0) break;
        moe_cbatch_step(af_blob, t_embed, t_lmhead, w_finalnorm, ids, slots, sposs, A, logits_step, 1);
        for (int m = 0; m < A; m++) {
            int s = slots[m];
            int am = 0; float bm = logits_step[m][0];
            for (int v = 1; v < MOE_VOCAB; v++) if (logits_step[m][v] > bm) { bm = logits_step[m][v]; am = v; }
            generated[s][cb_nout[s]++] = am;
            cb_pos[s]++;
            cb_next_tok[s] = am;
            if (cb_nout[s] >= moe_cbatch_gen[s] || cb_pos[s] >= MOE_CBATCH_MAXPOS) cb_active[s] = 0;
        }
        step++;
        fprintf(stderr, "[moe cbatch] step %d: A=%d active slots\n", step, A);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms_ragged = (t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)/1e6;

    for (int s = 0; s < MOE_CBATCH_N; s++) {
        fprintf(stderr, "[moe cbatch] slot %d prompt_len=%d target=%d generated:", s, prompt_len[s], moe_cbatch_gen[s]);
        for (int k = 0; k < moe_cbatch_gen[s]; k++) fprintf(stderr, " %d", generated[s][k]);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "[moe cbatch] %d decode steps, wall=%.2fms\n", step, ms_ragged);
    int n_decode_tok = 0;
    for (int s = 0; s < MOE_CBATCH_N; s++) n_decode_tok += moe_cbatch_gen[s];
    double scalar_rate = ms_prefill / n_prefill_tok;
    double ms_naive_equiv = scalar_rate * (n_prefill_tok + n_decode_tok);
    double ms_ragged_total = ms_prefill + ms_ragged;
    fprintf(stderr, "[moe cbatch] naive-equivalent (scalar rate x total tokens)=%.2fms, ragged actual (prefill+decode)=%.2fms, speedup=%.3fx\n",
            ms_naive_equiv, ms_ragged_total, ms_naive_equiv/ms_ragged_total);
    fprintf(stderr, "RESULT: MoE-4a ragged cbatch complete, N=%d\n", MOE_CBATCH_N);
    return 1;
    }  // end if (!online) -- MoE-4a static scheduler

    // ==========================================================================================
    // Phase MoE-4b: online scheduler. Requests arrive per QWEN_MOE_CB_ARRIVE (step-indexed, for
    // determinism -- design doc D10), admitted into whichever slot is free (including slots freed
    // mid-run by eviction -- D8, no KV clearing needed: a fresh request overwrites ascending from
    // pos=0 and moe_mla_attention_ragged() only ever reads j<=pos). Prefill is either scalar
    // (QWEN_MOE_CB_PREFILL_MODE=0, mirrors MoE-4a's admission body exactly, one request at a time)
    // or SME2-batched mixed into the same moe_cbatch_step() call as decode columns (=1, default --
    // D6: scalar prefill would stall every decode slot ~12s per admitted prompt, which would make
    // online admission a functional regression, not just a slow one).
    // ==========================================================================================
    const char *env_slots     = getenv("QWEN_MOE_CB_SLOTS");
    const char *env_reqs      = getenv("QWEN_MOE_CB_REQS");
    const char *env_budget    = getenv("QWEN_MOE_CB_PREFILL_BUDGET");
    const char *env_pfmode    = getenv("QWEN_MOE_CB_PREFILL_MODE");
    const char *env_arrive    = getenv("QWEN_MOE_CB_ARRIVE");
    const char *env_stopextra = getenv("QWEN_MOE_CB_STOP_EXTRA");
    const char *env_check     = getenv("QWEN_MOE_CB_CHECK");

    // Phase MoE-4c: margin 기반 선택적 재검증 (ragged 통합, 범위확장판 -- decode+prefill완료 둘 다).
    // REVERIFY=off가 기본이라 미설정시 MoE-4b 경로와 완전 동일(no-op).
    const char *env_reverify  = getenv("QWEN_MOE_CBATCH_REVERIFY");
    const char *env_threshold = getenv("QWEN_MOE_CBATCH_THRESHOLD");
    const char *env_budget4c  = getenv("QWEN_MOE_CBATCH_BUDGET");
    const char *env_resync    = getenv("QWEN_MOE_CBATCH_RESYNC");

    int B            = env_slots  && env_slots[0]  ? atoi(env_slots)  : 4;
    int R            = env_reqs   && env_reqs[0]   ? atoi(env_reqs)   : 12;
    int pfB          = env_budget && env_budget[0] ? atoi(env_budget) : 16;
    int prefill_mode = env_pfmode && env_pfmode[0] ? atoi(env_pfmode) : 1;
    int stop_extra   = env_stopextra && env_stopextra[0] ? atoi(env_stopextra) : -1;
    int check_on     = env_check  && env_check[0]  && atoi(env_check) != 0;

    // 0=off 1=tier1 2=tier2 (auto도 현재는 tier2로 매핑 -- task#102 실측 후 분기 예정)
    int reverify_mode = 0;
    if (env_reverify && env_reverify[0]) {
        if      (!strcmp(env_reverify, "off"))   reverify_mode = 0;
        else if (!strcmp(env_reverify, "tier1")) reverify_mode = 1;
        else if (!strcmp(env_reverify, "tier2")) reverify_mode = 2;
        else if (!strcmp(env_reverify, "auto"))  reverify_mode = 2;
        else { fprintf(stderr, "FATAL: QWEN_MOE_CBATCH_REVERIFY=%s unknown (off|tier1|tier2|auto)\n", env_reverify); exit(1); }
    }
    g_moe_cb4c_threshold_override = env_threshold && env_threshold[0] ? atof(env_threshold) : -1.0;
    int    cbatch_budget      = env_budget4c  && env_budget4c[0]  ? atoi(env_budget4c)  : MOE_CBATCH_MAXPOS;
    int    resync_on          = env_resync    && env_resync[0]    && atoi(env_resync) != 0;

    if (B < 1 || B > MOE_BATCH_MAX) { fprintf(stderr, "FATAL: QWEN_MOE_CB_SLOTS=%d out of [1,%d]\n", B, MOE_BATCH_MAX); exit(1); }
    if (R < 1 || R > MOE_CB4B_RMAX) { fprintf(stderr, "FATAL: QWEN_MOE_CB_REQS=%d out of [1,%d]\n", R, MOE_CB4B_RMAX); exit(1); }

    static int    rq_plen[MOE_CB4B_RMAX], rq_maxnew[MOE_CB4B_RMAX], rq_arrive[MOE_CB4B_RMAX];
    static int    rq_slot_of[MOE_CB4B_RMAX], rq_admit_step[MOE_CB4B_RMAX];
    static int    rq_out[MOE_CB4B_RMAX][MOE_CBATCH_KNEW], rq_nout[MOE_CB4B_RMAX];
    static double rq_t_admit[MOE_CB4B_RMAX], rq_t_first[MOE_CB4B_RMAX];

    for (int r = 0; r < R; r++) rq_arrive[r] = 0;
    if (env_arrive && env_arrive[0]) {
        const char *p = env_arrive;
        for (int r = 0; r < R && *p; r++) {
            rq_arrive[r] = atoi(p);
            const char *comma = strchr(p, ',');
            if (!comma) break;
            p = comma + 1;
        }
    }
    for (int r = 0; r < R; r++) {
        int sp = r % MOE_CBATCH_N;
        rq_plen[r] = prompt_len[sp]; rq_maxnew[r] = moe_cbatch_gen[sp];
        rq_nout[r] = 0; rq_slot_of[r] = -1; rq_admit_step[r] = -1;
        rq_t_admit[r] = 0.0; rq_t_first[r] = 0.0;
        moe_cb4b_admit_guard(rq_plen, rq_maxnew, r);
    }

    static int mcb_active[MOE_BATCH_MAX];   // 0 free / 1 decode / 2 prefill-in-progress
    static int mcb_req[MOE_BATCH_MAX], mcb_tok[MOE_BATCH_MAX];
    static int mcb_pos[MOE_BATCH_MAX], mcb_pref[MOE_BATCH_MAX];
    static int mcb_freed_before[MOE_BATCH_MAX];
    for (int s = 0; s < B; s++) { mcb_active[s] = 0; mcb_freed_before[s] = 0; }

    // Phase MoE-4c: shadow lane pool은 0-초기화되면 g_moe_sh_req[i]==0이 요청ID 0(유효값)과
    // 충돌해 "free"로 잘못 해석됨 -- 명시적으로 -1(free)로 리셋.
    for (int i = 0; i < MOE_CB4C_LANES; i++) { g_moe_sh_req[i] = -1; g_moe_sh_valid[i] = 0; }
    g_moe_sh_next_evict = 0;
    for (int r = 0; r < R; r++) for (int p = 0; p < MOE_CBATCH_MAXPOS; p++) rq_hist[r][p] = -1;

    int qhead = 0, nact = 0, step = 0;
    long steps_idle = 0, steps_with_idle_slot = 0, admitted_after_evict = 0;
    long queue_wait_events = 0, queue_wait_max_steps = 0, steps_pure_prefill = 0;
    static float logits_step[MOE_BATCH_MAX][102400];

    double t_run0 = nowt();
    while (qhead < R || nact > 0) {
        while (qhead < R && rq_plen[qhead] < 0) qhead++;   // D9: skip requests dropped by the guard

        // 1. admission (D2): occupy free slots whose request has arrived, ZERO forward work in
        //    PREFILL_MODE=1; a full scalar prefill burst (mirrors MoE-4a) in PREFILL_MODE=0.
        for (int s = 0; s < B && qhead < R; s++) {
            if (mcb_active[s]) continue;
            if (rq_arrive[qhead] > step) break;   // D10: FIFO head-of-line block
            int r = qhead++;
            rq_admit_step[r] = step;
            long wait = step - rq_arrive[r];
            if (wait > 0) { queue_wait_events++; if (wait > queue_wait_max_steps) queue_wait_max_steps = wait; }
            rq_t_admit[r] = nowt();
            if (mcb_freed_before[s]) admitted_after_evict++;
            rq_slot_of[r] = s;

            // Phase MoE-4c 조정2: 요청ID로 키잉되는 Tier2 replay가 필요로 하는 실제 토큰열.
            // prompt 구간은 두 PREFILL_MODE 모두 admission 시점에 이미 확정돼있음.
            for (int p = 0; p < rq_plen[r]; p++) rq_hist[r][p] = prompt_ids[r % MOE_CBATCH_N][p];

            if (prefill_mode == 0) {
                static float logits1[102400];
                for (int p = 0; p < rq_plen[r]; p++)
                    moe_forward_token(af_blob, t_embed, t_lmhead, w_finalnorm, prompt_ids[r % MOE_CBATCH_N][p], p, logits1, NULL);
                for (int l = 0; l < MOE_NL; l++)
                    for (int p = 0; p < rq_plen[r]; p++) {
                        memcpy(g_moe_cK[l][s][p], g_moe_K[l][p], sizeof g_moe_K[l][p]);
                        memcpy(g_moe_cV[l][s][p], g_moe_V[l][p], sizeof g_moe_V[l][p]);
                    }
                int am = 0; float bm = logits1[0];
                for (int v = 1; v < MOE_VOCAB; v++) if (logits1[v] > bm) { bm = logits1[v]; am = v; }
                mcb_active[s] = 1; mcb_req[s] = r; mcb_pos[s] = rq_plen[r]; mcb_tok[s] = am;
                rq_t_first[r] = nowt();
                rq_out[r][rq_nout[r]++] = am;
            } else {
                mcb_active[s] = 2; mcb_req[s] = r; mcb_pref[s] = 0; mcb_pos[s] = 0;
            }
            nact++;
        }

        if (nact == 0) { step++; steps_idle++; continue; }
        if (qhead < R) {
            int any_idle = 0; for (int s = 0; s < B; s++) if (!mcb_active[s]) any_idle = 1;
            if (any_idle) steps_with_idle_slot++;
        }

        // 2. packing (D3): decode columns first, compact prefix m < ndec
        int A = 0;
        static int ids[MOE_BATCH_MAX], slots[MOE_BATCH_MAX], sposs[MOE_BATCH_MAX];
        for (int s = 0; s < B; s++) if (mcb_active[s] == 1) { ids[A]=mcb_tok[s]; slots[A]=s; sposs[A]=mcb_pos[s]; A++; }
        int ndec = A;

        // 3. prefill columns under budget (D4), strictly ascending spos per slot (D3) -- mode 1 only
        int want_logits = (ndec > 0);
        if (prefill_mode == 1) {
            int budget = pfB; if (budget > MOE_BATCH_MAX - ndec) budget = MOE_BATCH_MAX - ndec;
            for (int s = 0; s < B && budget > 0; s++) {
                if (mcb_active[s] != 2) continue;
                int r = mcb_req[s];
                int take = rq_plen[r] - mcb_pref[s]; if (take > budget) take = budget;
                for (int i = 0; i < take; i++) {
                    ids[A] = prompt_ids[r % MOE_CBATCH_N][mcb_pref[s]+i];
                    slots[A] = s; sposs[A] = mcb_pref[s]+i; A++;
                }
                mcb_pref[s] += take; budget -= take;
                if (mcb_pref[s] >= rq_plen[r]) want_logits = 1;
            }
        }
        if (A == 0) { step++; continue; }
        if (check_on) moe_cb4b_assert_invariants(slots, sposs, A);
        if (!want_logits) steps_pure_prefill++;   // D5: exercises moe_cbatch_step()'s early return

        moe_cbatch_step(af_blob, t_embed, t_lmhead, w_finalnorm, ids, slots, sposs, A, logits_step, want_logits);
        double temit = nowt();

        int step_budget = cbatch_budget;   // Phase MoE-4c: per-step scalar-token budget (QWEN_MOE_CBATCH_BUDGET)

        // 4. decode columns: emit + evict (EOS/stop_extra/maxnew/MOE_CBATCH_MAXPOS)
        for (int m = 0; m < ndec; m++) {
            int s = slots[m], r = mcb_req[s];
            moe_cb4c_maybe_reverify(af_blob, t_embed, t_lmhead, w_finalnorm, s, r, sposs[m], ids[m],
                                     A, logits_step[m], &step_budget, reverify_mode, resync_on);
            int am = 0; float bm = logits_step[m][0];
            for (int v = 1; v < MOE_VOCAB; v++) if (logits_step[m][v] > bm) { bm = logits_step[m][v]; am = v; }
            rq_out[r][rq_nout[r]++] = am; mcb_pos[s]++;
            if (mcb_pos[s] < MOE_CBATCH_MAXPOS) rq_hist[r][mcb_pos[s]] = am;
            if (am == 100001 || am == stop_extra || rq_nout[r] >= rq_maxnew[r] || mcb_pos[s] >= MOE_CBATCH_MAXPOS)
                { mcb_active[s] = 0; mcb_freed_before[s] = 1; nact--; }
            else mcb_tok[s] = am;
        }
        // 5. prefill columns (mode 1 only): only spos==plen-1 carries a consumable token
        if (prefill_mode == 1) {
            for (int m = ndec; m < A; m++) {
                int s = slots[m], r = mcb_req[s];
                if (sposs[m] != rq_plen[r] - 1) continue;
                moe_cb4c_maybe_reverify(af_blob, t_embed, t_lmhead, w_finalnorm, s, r, sposs[m], ids[m],
                                         A, logits_step[m], &step_budget, reverify_mode, resync_on);
                int am = 0; float bm = logits_step[m][0];
                for (int v = 1; v < MOE_VOCAB; v++) if (logits_step[m][v] > bm) { bm = logits_step[m][v]; am = v; }
                rq_out[r][rq_nout[r]++] = am; rq_t_first[r] = temit;
                if (rq_plen[r] < MOE_CBATCH_MAXPOS) rq_hist[r][rq_plen[r]] = am;
                if (am == 100001 || am == stop_extra || rq_nout[r] >= rq_maxnew[r])
                    { mcb_active[s] = 0; mcb_freed_before[s] = 1; nact--; }
                else { mcb_active[s] = 1; mcb_tok[s] = am; mcb_pos[s] = rq_plen[r]; }
            }
        }
        step++;
    }
    double t_run1 = nowt();

    double ttft_max = 0.0, ttft_sum = 0.0; int ttft_n = 0;
    for (int r = 0; r < R; r++) {
        if (rq_plen[r] < 0) continue;
        double ttft_ms = (rq_t_first[r] - rq_t_admit[r]) * 1000.0;
        if (ttft_ms > ttft_max) ttft_max = ttft_ms;
        ttft_sum += ttft_ms; ttft_n++;
        fprintf(stderr, "[moe cb4b] req %d prompt %d slot %d arrive %d admit_step %d ttft_ms %.2f nout %d tokens:",
                r, r % MOE_CBATCH_N, rq_slot_of[r], rq_arrive[r], rq_admit_step[r], ttft_ms, rq_nout[r]);
        for (int k = 0; k < rq_nout[r]; k++) fprintf(stderr, " %d", rq_out[r][k]);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "[moe cb4b] steps=%d steps_idle=%ld steps_with_idle_slot=%ld admitted_after_evict=%ld "
            "queue_wait_events=%ld queue_wait_max_steps=%ld steps_pure_prefill=%ld ttft_max_ms=%.2f ttft_mean_ms=%.2f wall_ms=%.2f\n",
            step, steps_idle, steps_with_idle_slot, admitted_after_evict, queue_wait_events,
            queue_wait_max_steps, steps_pure_prefill, ttft_max, ttft_n ? ttft_sum/ttft_n : 0.0, (t_run1-t_run0)*1000.0);
    fprintf(stderr, "RESULT: MoE-4b online cbatch complete, B=%d R=%d PREFILL_MODE=%d\n", B, R, prefill_mode);
    return 1;
}

static int run_moe_verify_mode(int argc, char **argv) {
    (void)argc; (void)argv;   // unlike moe2b_verify.c, argv[1] here is the GQA MODE string
                              // (e.g. "greedy"), not a directory -- only QWEN_MOE_BASE selects
                              // the MoE weights dir, defaulting to "." (cwd) when unset.
    const char *override = getenv("QWEN_MOE_BASE");
    const char *dir = (override && override[0]) ? override : ".";
    char path[1024];
    snprintf(path, sizeof path, "%s/weights_moe/arch_config_moe.txt", dir);
    FILE *probe = fopen(path, "r");
    if (!probe) return 0;   // not an MoE model dir -- fall through to normal main()
    fclose(probe);

    fprintf(stderr, "[engine] weights_moe/arch_config_moe.txt found -- MoE-3a verification mode\n");
    char moe_dir[900];
    snprintf(moe_dir, sizeof moe_dir, "%s/weights_moe", dir);

    MOE_HIDDEN = (int)moe_cfg_get(path,"HIDDEN"); MOE_N_HEADS = (int)moe_cfg_get(path,"N_HEADS");
    MOE_KV_LORA_RANK = (int)moe_cfg_get(path,"KV_LORA_RANK"); MOE_QK_ROPE_HD = (int)moe_cfg_get(path,"QK_ROPE_HEAD_DIM");
    MOE_QK_NOPE_HD = (int)moe_cfg_get(path,"QK_NOPE_HEAD_DIM"); MOE_V_HD = (int)moe_cfg_get(path,"V_HEAD_DIM");
    MOE_Q_HEAD_DIM = MOE_QK_ROPE_HD + MOE_QK_NOPE_HD;
    MOE_NL = (int)moe_cfg_get(path,"NL"); MOE_FIRST_DENSE_LAYERS = (int)moe_cfg_get(path,"FIRST_DENSE_LAYERS");
    MOE_N_EXPERTS = (int)moe_cfg_get(path,"N_EXPERTS"); MOE_N_SHARED = (int)moe_cfg_get(path,"N_SHARED");
    MOE_TOP_K = (int)moe_cfg_get(path,"TOP_K"); MOE_IM_DIM = (int)moe_cfg_get(path,"MOE_IM");
    MOE_DENSE_IM = (int)moe_cfg_get(path,"DENSE_IM"); MOE_VOCAB = (int)moe_cfg_get(path,"VOCAB");
    MOE_ROPE_THETA = moe_cfg_get(path,"ROPE_THETA"); MOE_YARN_FACTOR = moe_cfg_get(path,"YARN_FACTOR");
    MOE_YARN_BETA_FAST = moe_cfg_get(path,"YARN_BETA_FAST"); MOE_YARN_BETA_SLOW = moe_cfg_get(path,"YARN_BETA_SLOW");
    MOE_YARN_MSCALE = moe_cfg_get(path,"YARN_MSCALE"); MOE_YARN_MSCALE_ALL_DIM = moe_cfg_get(path,"YARN_MSCALE_ALL_DIM");
    MOE_YARN_ORIG_MAX_POS = moe_cfg_get(path,"YARN_ORIG_MAX_POS");
    moe_init_yarn();
    if (MOE_NL > MOE_MAXLAYERS) { fprintf(stderr,"FATAL: NL=%d > MOE_MAXLAYERS=%d\n",MOE_NL,MOE_MAXLAYERS); exit(1); }
    fprintf(stderr, "[moe cfg] NL=%d FIRST_DENSE=%d N_EXPERTS=%d TOP_K=%d MOE_IM=%d DENSE_IM=%d VOCAB=%d\n",
            MOE_NL,MOE_FIRST_DENSE_LAYERS,MOE_N_EXPERTS,MOE_TOP_K,MOE_IM_DIM,MOE_DENSE_IM,MOE_VOCAB);
    fprintf(stderr, "[moe yarn] rope_mscale=%.10f attn_scale=%.10f\n", g_moe_rope_mscale, g_moe_attn_scale);

    // Phase MoE-3b: QWEN_MOE_BATCH=<B> branches to the batched+gather verification mode
    // instead of MoE-3a's single-sequence mode below -- checked here (config already loaded,
    // needed by both modes) rather than before, so this doesn't duplicate the config-loading
    // block above.
    if (run_moe_batch_verify_mode(argc, argv, dir)) return 1;
    // Phase MoE-4a: QWEN_MOE_CBATCH=1 branches to the ragged continuous-batching mode -- same
    // reasoning as the QWEN_MOE_BATCH check above (config already loaded, needed either way).
    if (run_moe_cbatch_verify_mode(argc, argv, dir)) return 1;

    snprintf(path, sizeof path, "%s/layout_af.txt", moe_dir); moe_load_layout_af(path);
    snprintf(path, sizeof path, "%s/deepseek_moe_af.bin", moe_dir);
    long af_bytes; uint8_t *af_blob = moe_mmap_file(path, &af_bytes);
    fprintf(stderr, "[moe load] af blob %ld bytes, %d tensors\n", af_bytes, g_moe_naf);

    snprintf(path, sizeof path, "%s/layout_f32.txt", moe_dir); moe_load_layout_f32(path);
    snprintf(path, sizeof path, "%s/deepseek_moe_f32.bin", moe_dir);
    long f32_bytes; g_moe_f32_blob = moe_mmap_file(path, &f32_bytes);
    fprintf(stderr, "[moe load] f32 blob %ld bytes, %d tensors\n", f32_bytes, g_moe_nf32);

    moe_resolve_layer_tensors();
    fprintf(stderr, "[moe check] all %d layers' tensors resolved\n", MOE_NL);

    MoeAFTensor *t_embed = moe_find_af("model.embed_tokens");
    MoeAFTensor *t_lmhead = moe_find_af("lm_head");
    MoeF32Tensor *t_finalnorm = moe_find_f32("model.norm.weight");
    float *w_finalnorm = (float *)(g_moe_f32_blob + t_finalnorm->off);
    if (t_lmhead->out != MOE_VOCAB || t_lmhead->in != MOE_HIDDEN) { fprintf(stderr,"FATAL: lm_head shape mismatch\n"); exit(1); }

    int prompt_ids[] = {100000, 549, 4345, 280, 8204, 317, 245, 1234};
    int N = sizeof(prompt_ids) / sizeof(prompt_ids[0]);
    if (N > MOE_MAXPOS) { fprintf(stderr, "FATAL: N=%d > MOE_MAXPOS=%d\n", N, MOE_MAXPOS); exit(1); }

    FILE *logits_out = fopen("moe3a_c_logits.bin", "wb");
    if (!logits_out) { perror("moe3a_c_logits.bin"); exit(1); }
    FILE *routing_out = fopen("moe3a_c_routing.txt", "w");
    if (!routing_out) { perror("moe3a_c_routing.txt"); exit(1); }

    float *logits = malloc((size_t)MOE_VOCAB * sizeof(float));
    for (int pos = 0; pos < N; pos++) {
        moe_forward_token(af_blob, t_embed, t_lmhead, w_finalnorm, prompt_ids[pos], pos, logits, routing_out);
        fwrite(logits, sizeof(float), MOE_VOCAB, logits_out);
        int argmax = 0; float best = logits[0];
        for (int v = 1; v < MOE_VOCAB; v++) if (logits[v] > best) { best = logits[v]; argmax = v; }
        fprintf(stderr, "[moe verify] pos %d token %d -> argmax next-token %d (logit %.4f)\n",
                pos, prompt_ids[pos], argmax, best);
    }
    fclose(logits_out);
    fclose(routing_out);
    fprintf(stderr, "RESULT: MoE-3a production-binary forward complete for %d positions\n", N);
    return 1;   // signal: MoE mode ran, caller should exit without touching GQA main() logic
}

// ============================================================================================
// General-purpose loader, Phase 1 (PLAN_general_purpose_loader.md): GGUF as a fourth loader,
// alongside load_fp32()/load_int4()/the MoE loaders -- same additive, byte-identical-when-absent
// discipline (R2 in the plan): triggered only by QWEN_GGUF being set, touching nothing else.
//
// D-gen-loader-1: register every GGUF tensor under this engine's EXISTING HF-style names
// ("model.layers.%d.self_attn.q_proj.weight" etc, via ROLE_PATTERN_HF -- see the
// D-gen-tensorrole-1 comment above) instead of inventing a parallel GGUF-named code path.
//   WHY: init_qkv_bias() and init_tensor_roles() are already load-tested, byte-verified
//        functions (see RESULTS.md's TensorRole sub-step) that only care about NAME matching
//        in g_wt[] -- they have no idea (and don't need to know) whether the bytes behind a
//        name came from the custom binary format or a dequantized GGUF tensor. Reusing them
//        unmodified for the GGUF path means the two hardest-to-get-right pieces of load-time
//        logic in this file exist exactly once, not twice.
//   COST: an extra small pass renaming each tensor at registration time (negligible -- runs
//        once per process, not per token, same cost class as everything else at load time).
//   EXIT: if GGUF-sourced tensors ever need to diverge from the HF role/name model (e.g. an
//        architecture GGUF's own layout can't be mapped onto ROLE_PATTERN_HF's 9 roles),
//        that's the point to introduce a real GGUF-named parallel path -- not needed yet.
//
// Phase 1 scope note: every tensor is dequantized straight to K_F32 here, regardless of its
// GGUF quant type. This proves the loader itself is correct (which is what Phase 1's own gate
// checks -- greedy-token match + ppl delta vs the existing 12.10 baseline) without also taking
// on quantized-transcode risk in the same step; K_Q4G64 transcoding for real SME2 throughput is
// Phase 2 (D-gen-2 in the plan), a deliberately separate piece of work.
static GgufFile *g_gguf = NULL;

static const char *ROLE_PATTERN_GGUF[N_LAYER_ROLES] = {
    [ROLE_ATTN_Q]       = "blk.%d.attn_q.weight",
    [ROLE_ATTN_K]       = "blk.%d.attn_k.weight",
    [ROLE_ATTN_V]       = "blk.%d.attn_v.weight",
    [ROLE_ATTN_O]       = "blk.%d.attn_output.weight",
    [ROLE_MLP_GATE]     = "blk.%d.ffn_gate.weight",
    [ROLE_MLP_UP]       = "blk.%d.ffn_up.weight",
    [ROLE_MLP_DOWN]     = "blk.%d.ffn_down.weight",
    [ROLE_INPUT_LN]     = "blk.%d.attn_norm.weight",
    [ROLE_POST_ATTN_LN] = "blk.%d.ffn_norm.weight",
};

// Sets every g_cfg.* field load_arch_cfg() would have, from GGUF metadata instead of
// arch_config.txt -- deliberately does NOT touch arch_config.txt or its parser (qwen_score.c/
// qwen_spec.c each FATAL on an unrecognized key in that file; adding GGUF-only concepts to it
// would break both siblings for a feature they don't have, same reasoning load_rope_scale_cfg's
// own comment already gives for keeping RoPE scaling in a separate sidecar file).
static void load_gguf_arch(const char *path) {
    g_gguf = gguf_open(path);
    if (!g_gguf) { perror("gguf_open"); fprintf(stderr, "FATAL: could not open gguf file %s\n", path); exit(1); }

    const char *arch_ptr; uint64_t arch_len;
    if (!gguf_kv_str(g_gguf, "general.architecture", &arch_ptr, &arch_len)) {
        fprintf(stderr, "FATAL: gguf file %s missing general.architecture\n", path); exit(1);
    }
    // Architecture allowlist: FATAL with a named supported-list rather than attempting an
    // unvalidated run -- silently-wrong is worse than a crash (same doctrine as load_int4's
    // unrecognized-kind FATAL). "qwen2" was the Phase 1 fixture; "llama" added for Phase 2
    // sub-step 3 (D-gen-4 #1, Mistral-7B-v0.3 -- GGUF tags it general.architecture="llama",
    // not "mistral"). Extending this list further is exactly the D-gen-4 architecture-priority
    // work.
    static const char *SUPPORTED_ARCH[] = { "qwen2", "llama" };
    int arch_ok = 0;
    for (size_t i = 0; i < sizeof(SUPPORTED_ARCH)/sizeof(SUPPORTED_ARCH[0]); i++) {
        if (arch_len == strlen(SUPPORTED_ARCH[i]) && !memcmp(arch_ptr, SUPPORTED_ARCH[i], arch_len)) { arch_ok = 1; break; }
    }
    if (!arch_ok) {
        fprintf(stderr, "FATAL: gguf architecture '%.*s' not validated by this engine; supported: qwen2, llama\n",
                (int)arch_len, arch_ptr);
        exit(1);
    }
    char arch[64]; snprintf(arch, sizeof arch, "%.*s", (int)arch_len, arch_ptr);

    // g_rope_norm: see its own declaration comment (near rope_head/rope_apply) for the full
    // root-cause writeup. "llama" -> NORM (interleaved-pair); "qwen2" (and anything else that
    // reaches here, since SUPPORTED_ARCH already rejected everything else above) -> NEOX
    // (split-half, this file's original convention -- explicit else, not just leaving the
    // static initializer's 0 to do the work, so a third architecture added later doesn't
    // silently inherit NEOX by omission).
    g_rope_norm = !strcmp(arch, "llama") ? 1 : 0;

    char key[128]; uint64_t u; double d;
    snprintf(key,sizeof key,"%s.block_count",arch);
    if (!gguf_kv_u64(g_gguf,key,&u)) { fprintf(stderr,"FATAL: gguf missing '%s'\n",key); exit(1); } g_cfg.nl=(int)u;
    snprintf(key,sizeof key,"%s.embedding_length",arch);
    if (!gguf_kv_u64(g_gguf,key,&u)) { fprintf(stderr,"FATAL: gguf missing '%s'\n",key); exit(1); } g_cfg.d=(int)u;
    snprintf(key,sizeof key,"%s.feed_forward_length",arch);
    if (!gguf_kv_u64(g_gguf,key,&u)) { fprintf(stderr,"FATAL: gguf missing '%s'\n",key); exit(1); } g_cfg.im=(int)u;
    snprintf(key,sizeof key,"%s.attention.head_count",arch);
    if (!gguf_kv_u64(g_gguf,key,&u)) { fprintf(stderr,"FATAL: gguf missing '%s'\n",key); exit(1); } g_cfg.nh=(int)u;
    snprintf(key,sizeof key,"%s.attention.head_count_kv",arch);
    if (!gguf_kv_u64(g_gguf,key,&u)) { fprintf(stderr,"FATAL: gguf missing '%s'\n",key); exit(1); } g_cfg.nkv=(int)u;
    snprintf(key,sizeof key,"%s.rope.freq_base",arch);
    if (!gguf_kv_f64(g_gguf,key,&d)) { fprintf(stderr,"FATAL: gguf missing '%s'\n",key); exit(1); } g_cfg.theta=(float)d;
    snprintf(key,sizeof key,"%s.attention.layer_norm_rms_epsilon",arch);
    if (!gguf_kv_f64(g_gguf,key,&d)) { fprintf(stderr,"FATAL: gguf missing '%s'\n",key); exit(1); } g_cfg.eps=(float)d;

    // MAXSEQ: deliberately NOT the model's trained context_length metadata (routinely 32K+,
    // which would allocate KV-cache buffers far larger than this Phase 1 correctness check
    // needs) -- a small, override-able cap, same spirit as arch_config.txt's own MAXSEQ.
    const char *ov = getenv("QWEN_GGUF_MAXSEQ");
    g_cfg.maxseq = (ov && ov[0]) ? atoi(ov) : 2048;

    // No single "%s.vocab_size" scalar key in practice -- the embedding table's own shape
    // (ne[1], the slower/row dimension) can't disagree with how many rows it actually has.
    const GgufTensorInfo *embed_t = gguf_find_tensor(g_gguf, "token_embd.weight");
    if (!embed_t) { fprintf(stderr, "FATAL: gguf missing token_embd.weight\n"); exit(1); }
    g_cfg.vocab = (int)embed_t->ne[1];

    // qkv_bias is a presence fact, not a metadata scalar: GGUF simply includes attn_q.bias
    // etc. when the source architecture has them (true for Qwen2, false for Llama-3).
    g_cfg.qkv_bias = gguf_find_tensor(g_gguf, "blk.0.attn_q.bias") ? 1 : 0;

    if (g_cfg.nh <= 0 || g_cfg.nkv <= 0 || g_cfg.nh % g_cfg.nkv != 0) {
        fprintf(stderr,"FATAL: gguf NH=%d not a positive multiple of NKV=%d\n",g_cfg.nh,g_cfg.nkv); exit(1); }
    if (g_cfg.d <= 0 || g_cfg.d % g_cfg.nh != 0) {
        fprintf(stderr,"FATAL: gguf D=%d not evenly divisible by NH=%d (cannot derive HD)\n",g_cfg.d,g_cfg.nh); exit(1); }
    g_cfg.hd = g_cfg.d / g_cfg.nh;
    if (g_cfg.hd % 2 != 0) { fprintf(stderr,"FATAL: gguf HD=%d (D/NH) is odd\n",g_cfg.hd); exit(1); }
    g_cfg.kvd = g_cfg.nkv * g_cfg.hd;
    g_cfg.qd  = g_cfg.nh  * g_cfg.hd;
    g_cfg.group = g_cfg.nh / g_cfg.nkv;
    if (g_cfg.kvd % 64 != 0) { fprintf(stderr,"FATAL: gguf KVD=%d not a multiple of 64\n",g_cfg.kvd); exit(1); }
    if (g_cfg.qd  % 64 != 0) { fprintf(stderr,"FATAL: gguf QD=%d not a multiple of 64\n",g_cfg.qd); exit(1); }
    g_cfg.kvg = g_cfg.kvd / 64;
    g_cfg.qg  = g_cfg.qd  / 64;
    if (g_cfg.nl <= 0 || g_cfg.maxseq <= 0) { fprintf(stderr,"FATAL: gguf NL=%d MAXSEQ=%d must be positive\n",g_cfg.nl,g_cfg.maxseq); exit(1); }

    fprintf(stderr,"[engine] gguf arch config (%s, architecture=%s): NL=%d NH=%d NKV=%d D=%d HD=%d IM=%d VOCAB=%d THETA=%.1f EPS=%g MAXSEQ=%d QKV_BIAS=%d GROUP=%d\n",
        path, arch, g_cfg.nl,g_cfg.nh,g_cfg.nkv,g_cfg.d,g_cfg.hd,g_cfg.im,g_cfg.vocab,g_cfg.theta,g_cfg.eps,g_cfg.maxseq,g_cfg.qkv_bias,g_cfg.group);
}

// Dequantizes one GGUF tensor to fp32 and registers it in g_wt[] under `engine_name` (an
// HF-style name, NOT the raw GGUF tensor name -- see D-gen-loader-1 above). g_gguf must
// already be open (load_gguf_arch() called first).
static WT *gguf_register_f32_as(const char *gguf_name, const char *engine_name) {
    const GgufTensorInfo *t = gguf_find_tensor(g_gguf, gguf_name);
    if (!t) { fprintf(stderr, "FATAL: gguf model missing tensor '%s'\n", gguf_name); exit(1); }
    if (!gguf_dequant_supported(t->type)) {
        fprintf(stderr, "FATAL: gguf tensor '%s' has unsupported quant type id %d (see gguf_dequant_supported())\n",
                gguf_name, (int)t->type);
        exit(1);
    }
    check_no_dup_name(engine_name);
    if (g_nwt >= 512) { fprintf(stderr, "FATAL: >512 tensors loading gguf model\n"); exit(1); }
    WT *w = &g_wt[g_nwt++];
    snprintf(w->name, sizeof w->name, "%s", engine_name);
    w->kind = K_F32; w->ng = 0;
    // GGUF ne[] is fastest-varying-first (ne[0] = row length = this engine's "in"; ne[1] = row
    // count = "out"), matching how load_int4()/load_fp32() already populate these fields for
    // 2D weight matrices. 1D tensors (biases, norms) get ne[1]=1 from gguf_load.c's own
    // unused-dims-default-to-1 convention -- harmless here since those are only ever read via
    // ->f32 directly (rmsnorm/bias-add), never through W->out/W->in in a matvec call.
    w->in = (int)t->ne[0];
    w->out = (int)t->ne[1];
    w->packed = NULL; w->scales = NULL; w->sub = NULL;
    w->kai_rhs = NULL; w->kai_rhs_bytes = 0; w->kai_lazy_failed = 0;
    float *buf = malloc(sizeof(float) * (size_t)t->n_elements);
    if (!buf) { fprintf(stderr, "FATAL: gguf dequant alloc failed for '%s' (%llu elements)\n",
                        gguf_name, (unsigned long long)t->n_elements); exit(1); }
    gguf_dequant_row(t->type, gguf_tensor_data(g_gguf, t), buf, (int64_t)t->n_elements);
    w->f32 = buf;
    return w;
}

// Phase 2 sub-step 1b (D-gen-2 Path A): dequantizes one GGUF tensor, then RTN+error-feedback
// transcodes it to K_Q4G64 via gguf_transcode.c (oracle-verified zero-diff against
// eval/quantize_int4.py's quant_group_ef(), see RESULTS.md). Falls back to a plain K_F32
// registration when `in % 64 != 0` -- the transcode group size's hard requirement, the exact
// same constraint kai_sme2_shape_ok() already enforces downstream (D-gen-3's policy table row:
// "in%64==0, HARD, fallback already exists -- K_F32->BLAS"), not a new one invented here.
static WT *gguf_register_q4g64_as(const char *gguf_name, const char *engine_name) {
    const GgufTensorInfo *t = gguf_find_tensor(g_gguf, gguf_name);
    if (!t) { fprintf(stderr, "FATAL: gguf model missing tensor '%s'\n", gguf_name); exit(1); }
    int in = (int)t->ne[0], out = (int)t->ne[1];
    if (in % 64 != 0) {
        fprintf(stderr, "[engine] gguf: %s in=%d not a multiple of 64 -> K_F32 fallback (not K_Q4G64)\n", engine_name, in);
        return gguf_register_f32_as(gguf_name, engine_name);
    }
    if (!gguf_dequant_supported(t->type)) {
        fprintf(stderr, "FATAL: gguf tensor '%s' has unsupported quant type id %d\n", gguf_name, (int)t->type);
        exit(1);
    }
    check_no_dup_name(engine_name);
    if (g_nwt >= 512) { fprintf(stderr, "FATAL: >512 tensors loading gguf model\n"); exit(1); }
    float *deq = malloc(sizeof(float) * (size_t)t->n_elements);
    if (!deq) { fprintf(stderr, "FATAL: gguf dequant alloc failed for '%s'\n", gguf_name); exit(1); }
    gguf_dequant_row(t->type, gguf_tensor_data(g_gguf, t), deq, (int64_t)t->n_elements);

    int ng = in / 64;
    uint8_t *packed = malloc((size_t)out * (in / 2));
    float *scales = malloc(sizeof(float) * (size_t)out * ng);
    if (!packed || !scales) { fprintf(stderr, "FATAL: gguf transcode alloc failed for '%s'\n", gguf_name); exit(1); }
    gguf_quantize_q4g64_error_feedback(deq, out, in, packed, scales);
    free(deq);

    WT *w = &g_wt[g_nwt++];
    snprintf(w->name, sizeof w->name, "%s", engine_name);
    w->kind = K_Q4G64; w->in = in; w->out = out; w->ng = ng;
    w->f32 = NULL; w->packed = packed; w->scales = scales; w->sub = NULL;
    w->kai_rhs = NULL; w->kai_rhs_bytes = 0; w->kai_lazy_failed = 0;
    return w;
}

// Phase 2 sub-step 1b: symmetric int8 group-64 RTN (no error-feedback -- D7/D17's "near-
// lossless without it" finding), used only for the untied lm_head, matching
// eval/quantize_int4.py's quant_group_int8() policy exactly.
static WT *gguf_register_q8g64_as(const char *gguf_name, const char *engine_name) {
    const GgufTensorInfo *t = gguf_find_tensor(g_gguf, gguf_name);
    if (!t) { fprintf(stderr, "FATAL: gguf model missing tensor '%s'\n", gguf_name); exit(1); }
    int in = (int)t->ne[0], out = (int)t->ne[1];
    if (in % 64 != 0) {
        fprintf(stderr, "[engine] gguf: %s in=%d not a multiple of 64 -> K_F32 fallback (not K_Q8G64)\n", engine_name, in);
        return gguf_register_f32_as(gguf_name, engine_name);
    }
    if (!gguf_dequant_supported(t->type)) {
        fprintf(stderr, "FATAL: gguf tensor '%s' has unsupported quant type id %d\n", gguf_name, (int)t->type);
        exit(1);
    }
    check_no_dup_name(engine_name);
    if (g_nwt >= 512) { fprintf(stderr, "FATAL: >512 tensors loading gguf model\n"); exit(1); }
    float *deq = malloc(sizeof(float) * (size_t)t->n_elements);
    if (!deq) { fprintf(stderr, "FATAL: gguf dequant alloc failed for '%s'\n", gguf_name); exit(1); }
    gguf_dequant_row(t->type, gguf_tensor_data(g_gguf, t), deq, (int64_t)t->n_elements);

    int ng = in / 64;
    int8_t *codes = malloc((size_t)out * in);
    float *scales = malloc(sizeof(float) * (size_t)out * ng);
    if (!codes || !scales) { fprintf(stderr, "FATAL: gguf transcode alloc failed for '%s'\n", gguf_name); exit(1); }
    gguf_quantize_q8g64(deq, out, in, codes, scales);
    free(deq);

    WT *w = &g_wt[g_nwt++];
    snprintf(w->name, sizeof w->name, "%s", engine_name);
    w->kind = K_Q8G64; w->in = in; w->out = out; w->ng = ng;
    w->f32 = NULL; w->packed = (const uint8_t *)codes; w->scales = scales; w->sub = NULL;
    w->kai_rhs = NULL; w->kai_rhs_bytes = 0; w->kai_lazy_failed = 0;
    return w;
}

// Populates g_wt[] with every tensor this dense GQA model needs, under this engine's existing
// HF-style names -- after this returns, init_qkv_bias() and init_tensor_roles() (both already
// defined above, unmodified) work exactly as they do for the custom binary format, because
// from their point of view nothing about where the bytes came from is visible.
//
// Per-role policy (D7 in eval/quantize_int4.py, replicated exactly, not reinvented): the 7
// 2D projection roles (Q/K/V/O/gate/up/down) transcode to K_Q4G64; the 2 norm roles and all
// biases stay K_F32 (quantizing norms/biases was measured, elsewhere in this project's own
// history, to cost accuracy for no compression win worth it at this size); the tied embedding
// stays K_F32 (same D7/D9 finding: int4 on the tied embed cost ppl 10.6->20.4 -- not repeated
// here); the untied lm_head (when present) goes K_Q8G64 (D17: near-lossless, breaks the fp32
// lm_head Amdahl floor).
static void load_gguf_weights(void) {
    int n_q4 = 0, n_f32 = 0;
    for (int r = 0; r < N_LAYER_ROLES; r++) {
        int is_norm = (r == ROLE_INPUT_LN || r == ROLE_POST_ATTN_LN);
        for (int l = 0; l < g_cfg.nl; l++) {
            char gsrc[96], ename[96];
            snprintf(gsrc, sizeof gsrc, ROLE_PATTERN_GGUF[r], l);
            snprintf(ename, sizeof ename, ROLE_PATTERN_HF[r], l);
            if (is_norm) { gguf_register_f32_as(gsrc, ename); n_f32++; }
            else         { gguf_register_q4g64_as(gsrc, ename); n_q4++; }
        }
    }
    if (g_cfg.qkv_bias) {
        for (int l = 0; l < g_cfg.nl; l++) {
            char gq[96], gk[96], gv[96], eq[96], ek[96], ev[96];
            snprintf(gq,sizeof gq,"blk.%d.attn_q.bias",l); snprintf(eq,sizeof eq,"model.layers.%d.self_attn.q_proj.bias",l);
            snprintf(gk,sizeof gk,"blk.%d.attn_k.bias",l); snprintf(ek,sizeof ek,"model.layers.%d.self_attn.k_proj.bias",l);
            snprintf(gv,sizeof gv,"blk.%d.attn_v.bias",l); snprintf(ev,sizeof ev,"model.layers.%d.self_attn.v_proj.bias",l);
            gguf_register_f32_as(gq, eq); n_f32++;
            gguf_register_f32_as(gk, ek); n_f32++;
            gguf_register_f32_as(gv, ev); n_f32++;
        }
    }
    gguf_register_f32_as("token_embd.weight", "model.embed_tokens.weight"); n_f32++;
    gguf_register_f32_as("output_norm.weight", "model.norm.weight"); n_f32++;
    // output.weight present -> untied lm_head (K_Q8G64); absent -> tied embeddings, matching
    // this engine's EXISTING tied-embedding fallback (wt_opt("lm_head.weight") returning NULL
    // is already how init_tensor_roles() detects that case for the custom format).
    int n_q8 = 0;
    if (gguf_find_tensor(g_gguf, "output.weight")) { gguf_register_q8g64_as("output.weight", "lm_head.weight"); n_q8++; }
    fprintf(stderr, "[engine] gguf: registered %d tensors (%d K_Q4G64, %d K_Q8G64, %d K_F32)\n",
            g_nwt, n_q4, n_q8, n_f32);
}

// Phase 2 sub-step 2: on-disk transcode cache (gguf_cache.c). After load_gguf_weights() has
// already populated g_wt[0..g_nwt) the normal way (dequant + transcode, sub-step 1's path),
// serialize every entry to `<gguf_path>.beglin` so the NEXT load can skip straight to mmap --
// zero dequant, zero RTN+error-feedback recompute. Byte sizes are derived purely from each
// WT's own kind/in/out/ng (out defaults to 1 for the 1D norm/bias tensors, per
// gguf_register_f32_as()'s own comment, so out*in*sizeof(float) is correct for those too --
// no separate 1D case needed).
static void gguf_write_cache(const char *cache_path, const char *src_gguf_path) {
    GgufCacheWriter *w = gguf_cache_writer_open(cache_path, src_gguf_path, (uint32_t)g_nwt);
    for (int i = 0; i < g_nwt; i++) {
        WT *t = &g_wt[i];
        uint64_t data_bytes, scales_bytes = 0;
        const void *data_ptr, *scales_ptr = NULL;
        if (t->kind == K_F32) {
            data_ptr = t->f32; data_bytes = (uint64_t)t->out * t->in * sizeof(float);
        } else if (t->kind == K_Q4G64) {
            data_ptr = t->packed; data_bytes = (uint64_t)t->out * (t->in / 2);
            scales_ptr = t->scales; scales_bytes = (uint64_t)t->out * t->ng * sizeof(float);
        } else if (t->kind == K_Q8G64) {
            data_ptr = t->packed; data_bytes = (uint64_t)t->out * t->in;
            scales_ptr = t->scales; scales_bytes = (uint64_t)t->out * t->ng * sizeof(float);
        } else {
            fprintf(stderr, "FATAL: gguf_write_cache: unexpected kind %d for '%s'\n", t->kind, t->name);
            exit(1);
        }
        gguf_cache_writer_add(w, t->name, t->kind, t->out, t->in, t->ng, data_ptr, data_bytes, scales_ptr, scales_bytes);
    }
    gguf_cache_writer_close(w);
    fprintf(stderr, "[engine] gguf cache: wrote %d tensors to %s\n", g_nwt, cache_path);
}

// Phase 2 sub-step 2: cache-hit path. Every WT field points straight into the cache's mmap --
// no malloc, no dequant, no transcode. Byte-identical downstream behavior to the live-transcode
// path (kai_route()/matvec_t() only ever look at kind/in/out/ng/packed/scales/f32, never at
// where the bytes came from -- same "no parallel logic needed" property TensorRole was
// designed around in Phase 1).
static void load_gguf_weights_from_cache(const char *cache_path) {
    GgufCacheFile *c = gguf_cache_open(cache_path);
    uint32_t n = gguf_cache_count(c);
    const uint8_t *base = gguf_cache_base(c);
    for (uint32_t i = 0; i < n; i++) {
        const GgufCacheEntry *e = gguf_cache_entry(c, i);
        check_no_dup_name(e->name);
        if (g_nwt >= 512) { fprintf(stderr, "FATAL: >512 tensors loading gguf cache\n"); exit(1); }
        WT *w = &g_wt[g_nwt++];
        snprintf(w->name, sizeof w->name, "%s", e->name);
        w->kind = e->kind; w->out = e->out; w->in = e->in; w->ng = e->ng;
        w->f32 = NULL; w->packed = NULL; w->scales = NULL; w->sub = NULL;
        w->kai_rhs = NULL; w->kai_rhs_bytes = 0; w->kai_lazy_failed = 0;
        const uint8_t *data_ptr = base + e->data_offset;
        if (e->kind == K_F32) w->f32 = (const float *)data_ptr;
        else w->packed = data_ptr;  // K_Q4G64 nibbles / K_Q8G64 int8 codes, same field reuse as the live path
        if (e->scales_bytes) w->scales = (const float *)(base + e->scales_offset);
    }
    fprintf(stderr, "[engine] gguf cache: loaded %u tensors from %s (mmap, zero transcode)\n", n, cache_path);
}

// Phase 2 sub-step 4: startup log naming which tier each tensor landed in. This is an
// ELIGIBILITY classification, not a guarantee of what actually ran: kai_sme2_shape_ok()
// already internally gates on kai_sme2_available() (see sme2_kai.h's own safety-contract
// comment), so "SME2-eligible" here means kai_route() COULD use SME2 for this tensor -- but
// kai_route()'s M >= kai_sme2_min_m() check is per-call and dynamic (M=1 greedy decode never
// qualifies, batched serve/cbatch calls might), so an SME2-eligible tensor can still run on
// NEON for any individual call. Named honestly as eligibility, not overclaimed as a fact about
// any specific run.
static void log_gguf_dispatch_tiers(void) {
    int n_sme2_eligible = 0, n_neon_q4g64 = 0, n_neon_q8g64 = 0, n_blas_f32 = 0;
    for (int i = 0; i < g_nwt; i++) {
        WT *t = &g_wt[i];
        if (t->kind == K_Q4G64) {
            if (kai_sme2_shape_ok(t->out, t->in)) n_sme2_eligible++;
            else n_neon_q4g64++;
        } else if (t->kind == K_Q8G64) {
            n_neon_q8g64++;
        } else if (t->kind == K_F32) {
            n_blas_f32++;
        }
    }
    fprintf(stderr, "[engine] gguf dispatch tiers: %d SME2-eligible, %d NEON-q4g64, %d NEON-q8g64, %d BLAS-f32 "
            "(SME2-eligible tensors still fall back to NEON per-call below the kernel's row-tile minimum, e.g. M=1 decode)\n",
            n_sme2_eligible, n_neon_q4g64, n_neon_q8g64, n_blas_f32);
}

int main(int argc, char **argv) {
    // Phase MoE-3a: checked FIRST, before load_arch_cfg() or any other GQA-dense-model setup
    // runs -- if weights_moe/arch_config_moe.txt exists, this exits without touching a single
    // line of the code below. Absent that file (every existing Qwen/Llama run), this is one
    // fopen() that immediately fails and falls through -- byte-identical to this file's
    // behavior before Phase MoE-3a existed.
    if (run_moe_verify_mode(argc, argv)) return 0;

    const char *mode = argc>1?argv[1]:"greedy";
    int n_gen = argc>2?atoi(argv[2]):32;
    const char *base_env = getenv("QWEN_BASE");
    const char *base = (base_env && base_env[0]) ? base_env : "/Volumes/D50/vdsp/llm_engine";
    const char *gguf_path = getenv("QWEN_GGUF");   // general-purpose-loader Phase 1: 4th loader,
                                                     // additive, byte-identical-when-absent (R2)
    if (gguf_path && gguf_path[0]) load_gguf_arch(gguf_path);
    else load_arch_cfg(base);      // D1/D2: must run before anything below reads g_cfg
    load_rope_scale_cfg(base);// M43: needs nothing but base; no ordering dependency on g_cfg
    alloc_arch_buffers();     // D2: heap-allocate every buffer g_cfg.* now sizes
    init_rope_scale();        // M43: needs g_cfg.hd/theta (loaded) + g_rope_cfg (loaded) + g_rope_scale (just malloc'd)
    // D2 bug fix 2: KV4_NB = g_cfg.maxseq/KV4_W silently floored with no divisibility guard --
    // exact only by luck for MAXSEQ=2048 (2048/64=32 exactly). A future MAXSEQ not a multiple
    // of KV4_W would under-allocate the int4-KV per-block scale/zero arrays (g_k4s/g_k4z/
    // srv_k4s/srv_k4z, all sized by KV4_NB) by one block. Checked once here, before every path
    // that can reach a KV4_NB-sized allocation (int4-KV single-stream, serve, cbatch).
    if (g_cfg.maxseq % KV4_W != 0) {
        fprintf(stderr,"FATAL: arch_config MAXSEQ=%d not a multiple of KV4_W=%d (int4-KV block math requires this)\n",g_cfg.maxseq,KV4_W); exit(1);
    }
    // D3 scope note (M45 update): unlike the plain fp32-KV attention path (which has a real
    // generic-scalar fallback at attn_mode()==0, wired above), the QWEN_KV_INT8/QWEN_KV_INT4
    // paths (kv_i8_attn/kv_i4_attn) call attn_neon.h's group-fused kernels UNCONDITIONALLY --
    // there never was a scalar reference implementation for quantized-KV attention to fall back
    // to (verified: no such code exists anywhere in this file). As of M45 there are TWO
    // specialized kernel families for this: (HD=128,GROUP=6) [M23/M24 originals] and
    // (HD=128,GROUP=4) [M45 _g4 siblings]. There is STILL no generic scalar fallback for
    // quantized-KV attention at any OTHER (HD,GROUP) -- writing one is new numerical algorithm
    // work, not a kernel-family-extension fix (same category as per-model quantization tuning
    // this project's structural work has repeatedly deferred). FATAL loudly here instead of
    // silently miscomputing or inventing an unvalidated fallback: for any other architecture,
    // plain fp32-KV mode (unset QWEN_KV_INT8/QWEN_KV_INT4) is the correctness-first path this
    // project actually delivers.
    if ((kv_int8_on() || kv_int4_on()) && !fast_attn_shape_ok() && !fast_attn_shape_ok_g4()) {
        fprintf(stderr,"FATAL: QWEN_KV_INT8/QWEN_KV_INT4 require HD=%d GROUP=%d or HD=%d GROUP=%d "
                        "(this config: HD=%d GROUP=%d) -- no generic-scalar fallback exists for "
                        "quantized-KV attention at other shapes; use plain fp32 KV (unset "
                        "QWEN_KV_INT8/QWEN_KV_INT4) for this architecture\n",
                AHD, AGROUP, AHD, AGROUP4, g_cfg.hd, g_cfg.group); exit(1);
    }
    char path[512];
    const char *int4bin = getenv("QWEN_INT4_BIN");
    if (gguf_path && gguf_path[0]) {
        // Phase 2 sub-step 2 (PLAN_general_purpose_loader.md's own text: "flip
        // QWEN_SME2_LAZY_REPACK to default-on for the GGUF path"): this path never calls
        // kai_repack_all() (unlike the QWEN_INT4_BIN branch below), by design -- an eager
        // repack burst here would stack on top of the transcode work sub-step 1 already did,
        // which is exactly the "12+GB on a 16GB machine" risk this plan flagged. Without lazy
        // mode ALSO defaulting on here, kai_route() (below, in matvec_t) would see
        // W->kai_rhs==NULL forever with sme2_lazy_on()==0 and SME2 would silently never
        // engage for any GGUF-loaded model -- not a crash, just quietly always-NEON, exactly
        // the kind of "silently wrong performance" this project's doctrine treats as a bug.
        // An explicit QWEN_SME2_LAZY_REPACK=0/1 from the user still wins; this only fills in
        // the unset case. g_sme2_lazy is this TU's own memoized sme2_lazy_on() cache (-1 =
        // unset), safe to preset directly since nothing reads it before this point in main().
        if (!getenv("QWEN_SME2_LAZY_REPACK")) g_sme2_lazy = 1;
        // Phase 2 sub-step 1b: load_gguf_weights() now registers K_Q4G64/K_Q8G64 tensors (not
        // just K_F32, as in Phase 1), so the same q4pool this engine's other K_Q4G64/K_Q8G64
        // consumer (the QWEN_INT4_BIN path, below) needs must be initialized here too --
        // identical T-detection/spin/qos setup, no GGUF-specific pool logic invented.
        int T = getenv("Q4_THREADS")?atoi(getenv("Q4_THREADS")):detect_q4_threads();
        q4pool_init(&g_pool, T, g_cfg.im);
        { const char *e = getenv("Q4_POOL_SPIN"); int m = e ? atoi(e) : 1; g_pool.spin = m ? 1 : 0; }
        { const char *e = getenv("Q4_POOL_QOS");  g_pool.qos  = (e && atoi(e)) ? 1 : 0; }
        { const char *e = getenv("Q4_POOL_SPIN_ITERS"); if (e && atoi(e) > 0) g_pool.spin_iters = atoi(e); }
        q4pool_start(&g_pool);
        // Phase 2 sub-step 2: on-disk cache. `<gguf_path>.beglin`, default on -- QWEN_GGUF_CACHE=0
        // opts out (e.g. for debugging the live transcode path itself). Staleness is checked by
        // the source GGUF's own size+mtime (gguf_cache_is_valid()), not trusted blindly.
        char cache_path[560];
        snprintf(cache_path, sizeof cache_path, "%s.beglin", gguf_path);
        const char *cache_env = getenv("QWEN_GGUF_CACHE");
        int cache_wanted = !(cache_env && !atoi(cache_env));
        if (cache_wanted && gguf_cache_is_valid(cache_path, gguf_path)) {
            load_gguf_weights_from_cache(cache_path);
        } else {
            load_gguf_weights();   // registers every tensor under the existing HF-style names --
                                    // init_qkv_bias()/init_tensor_roles() below need no changes
            if (cache_wanted) gguf_write_cache(cache_path, gguf_path);
        }
        // Same D14 (M49) kind-check the QWEN_INT4_BIN path below uses, not a GGUF-specific
        // reinvention: wt_opt() returning non-NULL only proves a tensor named "lm_head.weight"
        // exists, not which encoding it's in (untied lm_head may have stayed K_F32 if
        // gguf_register_q8g64_as()'s in%64==0 check fell back).
        const char *fp32h = getenv("QWEN_FP32_HEAD");
        WT *gguf_lmh = wt_opt("lm_head.weight");
        g_int8_head = (gguf_lmh && gguf_lmh->kind == K_Q8G64 && !(fp32h && fp32h[0])) ? 1 : 0;
        fprintf(stderr,"[engine] gguf Q4_THREADS=%d, lm_head=%s\n", T, g_int8_head?"int8":"fp32");
        log_gguf_dispatch_tiers();
    } else if (int4bin && int4bin[0]) {
        g_int4=1; int T = getenv("Q4_THREADS")?atoi(getenv("Q4_THREADS")):detect_q4_threads();  // M51: per-chip tuned table, see detect_q4_threads() above
        q4pool_init(&g_pool, T, g_cfg.im);
        // M29 default promotion: spin-wait measured +21.6--28.5% real decode speedup (TSan-clean,
        // 18/18 bit-identical across both prompts x {fp32,int8,int4}-KV x {spin,qos,spin+qos}),
        // reaching ~78.6% of the T=6 streaming ceiling -- clears this project's own 75-80%
        // "single-stream done" bar (M19/M28). UNSET now resolves to 1 here (qwen_infer.c's
        // single g_pool only -- see below for why qwen_spec.c's two pools are NOT promoted).
        // QoS pinning measured NO benefit alone (~0% vs baseline, both B2a-isolated and
        // end-to-end) -- stays off by default, kept available for future hardware/OS testing.
        // Must be set before q4pool_start() spawns the workers that read them (q4pool_init()
        // above no longer spawns threads itself -- see the M29 init-ordering race note in
        // q4gemv.h's q4pool_init/q4pool_start).
        { const char *e = getenv("Q4_POOL_SPIN"); int m = e ? atoi(e) : 1; g_pool.spin = m ? 1 : 0; }
        { const char *e = getenv("Q4_POOL_QOS");  g_pool.qos  = (e && atoi(e)) ? 1 : 0; }
        { const char *e = getenv("Q4_POOL_SPIN_ITERS"); if (e && atoi(e) > 0) g_pool.spin_iters = atoi(e); }
        q4pool_start(&g_pool);
        load_int4(base, int4bin);
        const char *fp32h = getenv("QWEN_FP32_HEAD");
        // D14 (M49): check kind==K_Q8G64, not just presence-by-name -- WHY: wt_opt() returning
        // non-NULL only proves SOME tensor named "lm_head.weight" was registered; it says
        // nothing about which encoding it is. Before check_no_dup_name() existed, a duplicate
        // layout entry meant this could be true while the FIRST-registered (and therefore
        // actually-dispatched, per wt()'s first-match scan) copy was q4g64 -- the engine logged
        // "lm_head=int8" and silently served int4 on every forward pass. Now that duplicates
        // FATAL at load time this is defense-in-depth rather than the only guard, but it's also
        // a real independent correctness fix on its own: an untied lm_head that's neither
        // quantized (still K_F32) should never have set this flag either.
        WT *lmh = wt_opt("lm_head.weight");
        if (lmh && lmh->kind == K_Q8G64 && !(fp32h && fp32h[0])) g_int8_head = 1;
        fprintf(stderr,"[engine] int4 GEMV, Q4_THREADS=%d, lm_head=%s\n", T, g_int8_head?"int8":"fp32");
    } else load_fp32(base);
    init_qkv_bias();          // D4: must run before init_fused_dispatch() (build_fused_qkv reads
                               // g_qbias_l/g_kbias_l/g_vbias_l) and before any forward pass
    init_tensor_roles();       // D-gen-tensorrole-1: must run after g_int8_head is finalized
                               // (just above) and after weights are loaded, before any forward pass
    init_fused_dispatch();

    if (kv_int4_on() && kv_int8_on()) {           // M24-D5: precedence, checked once
        fprintf(stderr,"[engine] QWEN_KV_INT4 and QWEN_KV_INT8 both set -- int4 takes precedence, int8 disabled\n");
        g_kv_int8 = 0;
    }
    if (kv_int4_on()) {
        // M24: int4 codes + block/group scales + staging ring replace the fp32 cache
        // entirely (the toggle selects the storage; the caches never coexist).
        kv_i8_init_bias();                        // M23-D7 bias-split kept at int4 (M24-D1)
        g_k4 = malloc((long)g_cfg.nl*g_cfg.maxseq*(g_cfg.kvd/2)); g_v4 = malloc((long)g_cfg.nl*g_cfg.maxseq*(g_cfg.kvd/2));
        g_k4s = malloc((long)g_cfg.nl*KV4_NB*g_cfg.kvd*sizeof(float));
        g_k4z = malloc((long)g_cfg.nl*KV4_NB*g_cfg.kvd*sizeof(float));
        g_v4sc = malloc((long)g_cfg.nl*g_cfg.maxseq*g_cfg.kvg*sizeof(float));
        g_k4stg = malloc((long)g_cfg.nl*KV4_RING*g_cfg.kvd*sizeof(float));
        g_v4stg = malloc((long)g_cfg.nl*KV4_RING*g_cfg.kvd*sizeof(float));
        if (!g_k4||!g_v4||!g_k4s||!g_k4z||!g_v4sc||!g_k4stg||!g_v4stg) { fprintf(stderr,"FATAL: int4 KV cache alloc failed\n"); return 1; }
        fprintf(stderr,"[engine] M24 KV int4 (mode %d): %.1f MB codes+scales + %.1f MB fp32 staging (int8 cache: %.1f MB, fp32: %.1f MB)\n",
                kv_int4_on(),
                (2.0*g_cfg.nl*g_cfg.maxseq*(g_cfg.kvd/2) + 2.0*g_cfg.nl*KV4_NB*g_cfg.kvd*4 + 1.0*g_cfg.nl*g_cfg.maxseq*g_cfg.kvg*4)/1e6,
                2.0*g_cfg.nl*KV4_RING*g_cfg.kvd*4/1e6,
                2.0*g_cfg.nl*g_cfg.maxseq*(g_cfg.kvd+g_cfg.kvg*4.0)/1e6, 2.0*g_cfg.nl*g_cfg.maxseq*g_cfg.kvd*4.0/1e6);
    } else if (kv_int8_on()) {
        // M23: int8 codes + per-group scales replace the fp32 cache entirely (the toggle
        // selects the storage; the two never coexist).
        kv_i8_init_bias();                        // M23-D7 rotated k-bias table + v-bias pointers
        g_kq  = malloc((long)g_cfg.nl*g_cfg.maxseq*g_cfg.kvd);  g_vq  = malloc((long)g_cfg.nl*g_cfg.maxseq*g_cfg.kvd);
        g_ksc = malloc((long)g_cfg.nl*g_cfg.maxseq*g_cfg.kvg*sizeof(float));
        g_vsc = malloc((long)g_cfg.nl*g_cfg.maxseq*g_cfg.kvg*sizeof(float));
        if (!g_kq||!g_vq||!g_ksc||!g_vsc) { fprintf(stderr,"FATAL: int8 KV cache alloc failed\n"); return 1; }
        fprintf(stderr,"[engine] M23 KV int8: %.1f MB (fp32 cache would be %.1f MB)\n",
                2.0*g_cfg.nl*g_cfg.maxseq*(g_cfg.kvd+g_cfg.kvg*4.0)/1e6, 2.0*g_cfg.nl*g_cfg.maxseq*g_cfg.kvd*4.0/1e6);
    } else {
        g_kcache=malloc((long)g_cfg.nl*g_cfg.maxseq*g_cfg.kvd*sizeof(float));
        g_vcache=malloc((long)g_cfg.nl*g_cfg.maxseq*g_cfg.kvd*sizeof(float));
    }
    if (attn_mode() == 1 && !kv_int8_on() && !kv_int4_on()) {   // only allocate the mirror cache when it can actually be read;
        g_kcache2=malloc((long)g_cfg.nl*g_cfg.maxseq*g_cfg.kvd*sizeof(float));  // keeps QWEN_FAST_ATTN=0 (default) fully
        g_vcache2=malloc((long)g_cfg.nl*g_cfg.maxseq*g_cfg.kvd*sizeof(float));  // independent of this allocation succeeding
        if (!g_kcache2 || !g_vcache2) { fprintf(stderr,"FATAL: mirror KV cache allocation failed (QWEN_FAST_ATTN=1)\n"); return 1; }
    }
    float *xtmp=malloc((size_t)g_cfg.d*sizeof(float)); float *logits=malloc(g_cfg.vocab*sizeof(float));

    int prompt[g_cfg.maxseq];
    const char *pf = getenv("QWEN_PROMPT");
    if (pf && pf[0]) snprintf(path,sizeof path,"%s",pf);
    else snprintf(path,sizeof path,"%s/ref/prompt_ids.i32",base);
    int np=load_ids(path,prompt,g_cfg.maxseq);
    if (strcmp(mode,"ppl") && np < 1) {
        fprintf(stderr,"FATAL: no prompt ids at %s (load_ids=%d)\n", path, np); return 1; }

    if (!strcmp(mode,"dump")) {
        for(int p=0;p<np;p++) forward_token(prompt[p],p,xtmp,p==np-1);
        final_logits(xtmp,logits);
        snprintf(path,sizeof path,"%s/results/c_logits_last.f32",base);
        FILE *o=fopen(path,"wb"); fwrite(logits,4,g_cfg.vocab,o); fclose(o);
        int am=argmax_v(logits);
        fprintf(stderr,"[engine] dumped, argmax=%d\n",am);
    } else if (!strcmp(mode,"greedy")) {
        for(int p=0;p<np;p++) forward_token(prompt[p],p,xtmp,0);
        final_logits(xtmp,logits);
        int pos=np-1; printf("greedy:");
        for(int g=0;g<n_gen;g++){ int am=argmax_v(logits);
            printf(" %d",am); fflush(stdout);
            if(pos+1>=g_cfg.maxseq) break;                 // KV cache holds MAXSEQ positions
            pos++; forward_token(am,pos,xtmp,0); final_logits(xtmp,logits); }
        printf("\n");
    } else if (!strcmp(mode,"bench")) {
        // decode-only timing: prefill + 8 warmup + N timed tokens, median of 3.
        int N=n_gen; double best=1e9; double bl=0,bg=0,ba=0;
        double bemb=0,brms=0,bq=0,bk=0,bv=0,brope=0,bkvw=0,bo=0,bresid=0,bgate=0,bup=0,bswi=0,bdown=0,bhrms=0,bhgemv=0,bargmax=0;
        if (np + 8 + N >= g_cfg.maxseq) {               // prefill np + 8 warmup + N timed
            int oldN=N; N = g_cfg.maxseq - np - 8 - 1;
            if (N < 1) { fprintf(stderr,"FATAL: prompt too long for bench (np=%d)\n",np); return 1; }
            fprintf(stderr,"[bench] N clamped %d->%d to fit MAXSEQ=%d\n",oldN,N,g_cfg.maxseq);
        }
        for(int rep=0;rep<3;rep++){
        for(int p=0;p<np;p++) forward_token(prompt[p],p,xtmp,0);
            final_logits(xtmp,logits); int pos=np-1;
            for(int g=0;g<8;g++){ int am=argmax_v(logits); pos++; forward_token(am,pos,xtmp,0); final_logits(xtmp,logits);} // warm
            g_t_layers=g_t_logits=g_t_attn=0;
            g_t_emb=g_t_rms=g_t_q=g_t_k=g_t_v=g_t_rope=g_t_kvwrite=g_t_o=g_t_resid=g_t_gate=g_t_up=g_t_swiglu=g_t_down=g_t_headrms=g_t_headgemv=g_t_argmax=0;
            double t0=nowt();
            for(int g=0;g<N;g++){
                double amt0; if(prof_on()) amt0=nowt();
                int am=argmax_v(logits);
                if(prof_on()) g_t_argmax += nowt()-amt0;
                pos++;
                double a=nowt(); forward_token(am,pos,xtmp,0); double b=nowt(); final_logits(xtmp,logits); double c=nowt();
                g_t_layers+=b-a; g_t_logits+=c-b; }
            double dt=nowt()-t0;
            if(dt<best){ best=dt; bl=g_t_layers; bg=g_t_logits; ba=g_t_attn;
                bemb=g_t_emb; brms=g_t_rms; bq=g_t_q; bk=g_t_k; bv=g_t_v; brope=g_t_rope; bkvw=g_t_kvwrite;
                bo=g_t_o; bresid=g_t_resid; bgate=g_t_gate; bup=g_t_up; bswi=g_t_swiglu; bdown=g_t_down;
                bhrms=g_t_headrms; bhgemv=g_t_headgemv; bargmax=g_t_argmax; }
        }
        double ms=best/N*1e3;
        fprintf(stderr,"[bench] %s N=%d pos~%d: %.1f ms/tok = %.2f tok/s  (layers %.1f ms [attn %.1f ms, %.1f%%], logits %.1f ms)\n",
                g_int4?"int4":"fp32", N, np+8, ms, 1000.0/ms, bl/N*1e3, ba/N*1e3, 100.0*ba/bl, bg/N*1e3);
        printf("tok_s %.3f ms_tok %.2f ms_layers %.2f ms_attn %.2f ms_logits %.2f\n", 1000.0/ms, ms, bl/N*1e3, ba/N*1e3, bg/N*1e3);
        if (prof_on()) {
            double total_ms = ms;
            double sum_ms = (bemb+brms+bq+bk+bv+ba+brope+bkvw+bo+bresid+bgate+bup+bswi+bdown)/N*1e3 + bhrms/N*1e3 + bhgemv/N*1e3 + bargmax/N*1e3;
            fprintf(stderr, "[M15-prof] ctx~%d total=%.3fms/tok sum(phases)=%.3fms/tok closure=%.1f%%\n",
                    np+8, total_ms, sum_ms, 100.0*sum_ms/total_ms);
            fprintf(stderr, "[M15-prof]   emb_gather   %.4f ms (%.2f%%)\n", bemb/N*1e3, 100.0*bemb/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   rmsnorm(all) %.4f ms (%.2f%%)\n", brms/N*1e3, 100.0*brms/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   proj_q       %.4f ms (%.2f%%)\n", bq/N*1e3, 100.0*bq/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   proj_k       %.4f ms (%.2f%%)\n", bk/N*1e3, 100.0*bk/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   proj_v       %.4f ms (%.2f%%)\n", bv/N*1e3, 100.0*bv/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   rope         %.4f ms (%.2f%%)\n", brope/N*1e3, 100.0*brope/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   kv_write     %.4f ms (%.2f%%)\n", bkvw/N*1e3, 100.0*bkvw/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   attn         %.4f ms (%.2f%%)\n", ba/N*1e3, 100.0*ba/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   proj_o       %.4f ms (%.2f%%)\n", bo/N*1e3, 100.0*bo/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   residuals    %.4f ms (%.2f%%)\n", bresid/N*1e3, 100.0*bresid/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   proj_gate    %.4f ms (%.2f%%)\n", bgate/N*1e3, 100.0*bgate/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   proj_up      %.4f ms (%.2f%%)\n", bup/N*1e3, 100.0*bup/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   swiglu       %.4f ms (%.2f%%)\n", bswi/N*1e3, 100.0*bswi/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   proj_down    %.4f ms (%.2f%%)\n", bdown/N*1e3, 100.0*bdown/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   head_rms     %.4f ms (%.2f%%)\n", bhrms/N*1e3, 100.0*bhrms/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   head_gemv    %.4f ms (%.2f%%)\n", bhgemv/N*1e3, 100.0*bhgemv/N*1e3/total_ms);
            fprintf(stderr, "[M15-prof]   argmax       %.4f ms (%.2f%%)\n", bargmax/N*1e3, 100.0*bargmax/N*1e3/total_ms);
            printf("M15PROF ctx %d total_ms %.4f emb %.4f rms %.4f q %.4f k %.4f v %.4f rope %.4f kvw %.4f attn %.4f o %.4f resid %.4f gate %.4f up %.4f swiglu %.4f down %.4f headrms %.4f headgemv %.4f argmax %.4f\n",
                   np+8, total_ms, bemb/N*1e3, brms/N*1e3, bq/N*1e3, bk/N*1e3, bv/N*1e3, brope/N*1e3, bkvw/N*1e3, ba/N*1e3, bo/N*1e3, bresid/N*1e3, bgate/N*1e3, bup/N*1e3, bswi/N*1e3, bdown/N*1e3, bhrms/N*1e3, bhgemv/N*1e3, bargmax/N*1e3);
        }
    } else if (!strcmp(mode,"ppl")) {
        // teacher-forced ppl over eval/ppl_work/w{0..NWIN-1}.i32
        int NWIN=argc>2?atoi(argv[2]):12; double tot_nll=0; long tot=0; int ids[g_cfg.maxseq];
        for(int wi=0;wi<NWIN;wi++){
            snprintf(path,sizeof path,"%s/eval/ppl_work/w%d.i32",base,wi);
            int n=load_ids(path,ids,g_cfg.maxseq); if(n<2) continue;
            for(int p=0;p<n;p++){ forward_token(ids[p],p,xtmp,0);
                if(p<n-1){ final_logits(xtmp,logits);
                    float mx; vDSP_maxv(logits,1,&mx,g_cfg.vocab); double se=0; for(int v=0;v<g_cfg.vocab;v++) se+=exp((double)(logits[v]-mx));
                    tot_nll += -((double)logits[ids[p+1]]-mx-log(se)); tot++; } }
            fprintf(stderr,"[ppl] window %d done (%ld toks)\n",wi,tot);
        }
        if (tot == 0) { fprintf(stderr,"FATAL: no ppl tokens (missing eval/ppl_work?)\n"); return 1; }
        double ppl=exp(tot_nll/tot);
        fprintf(stderr,"[ppl] %s ppl %.4f over %ld tokens\n", g_int4?"int4":"fp32", ppl, tot);
        printf("ppl %.4f tokens %ld\n", ppl, tot);
    } else if (!strcmp(mode,"spec")) {
        // prompt-lookup speculative decode: exact-greedy output, up to K+1 tokens per forward.
        int N=n_gen, K=getenv("SPEC_K")?atoi(getenv("SPEC_K")):8, NG=getenv("SPEC_NGRAM")?atoi(getenv("SPEC_NGRAM")):3;
        if (K > MAXSPEC-1) K = MAXSPEC-1;
        int *hist=malloc((size_t)g_cfg.maxseq*sizeof(int)); int hlen=0;
        for(int p=0;p<np && p<g_cfg.maxseq;p++){ forward_token(prompt[p],p,xtmp,0); hist[hlen++]=prompt[p]; }
        final_logits(xtmp,logits);
        int pos=np-1, emitted=0, forwards=0;
        float *xouts=malloc((size_t)MAXSPEC*g_cfg.d*sizeof(float));
        float *L=malloc((size_t)MAXSPEC*g_cfg.vocab*sizeof(float));
        int draft[MAXSPEC], seq[MAXSPEC];
        double t0=nowt();
        printf("spec:");
        while(emitted<N){
            if (pos+1 >= g_cfg.maxseq || hlen >= g_cfg.maxseq) break;  // no room for even the next token (KV/hist)
            int t=argmax_v(logits);
            hist[hlen++]=t;                                // commit t to history BEFORE lookup (safe: hlen<MAXSEQ)
            int k=pl_lookup(hist,hlen,draft,K,NG);         // suffix ends at t -> draft = tokens AFTER t
            int maxk = g_cfg.maxseq-2-pos;                        // KV slots pos+2..pos+1+k must be < MAXSEQ
            if (maxk > g_cfg.maxseq-hlen) maxk = g_cfg.maxseq-hlen;     // hist room for accepted drafts
            if (k > maxk) k = maxk < 0 ? 0 : maxk;         // clamp, don't drop the greedy token
            seq[0]=t; for(int i=0;i<k;i++) seq[1+i]=draft[i]; int slen=1+k;
            forward_tokens(seq,slen,pos+1,xouts); forwards++;
            final_logits_batch(xouts,slen,L);
            printf(" %d",t); pos++; emitted++;
            int na=0;
            for(int j=0;j<k && emitted<N;j++){ int a=argmax_v(L+(size_t)j*g_cfg.vocab);
                if(a==draft[j]){ printf(" %d",draft[j]); hist[hlen++]=draft[j]; pos++; emitted++; na++; } else break; }
            if(getenv("SPEC_DBG")){ int a0=argmax_v(L); fprintf(stderr,"[dbg] round: t=%d k=%d na=%d argmaxL0=%d draft0=%d\n", t,k,na,a0, k>0?draft[0]:-1); }
            memcpy(logits, L+(size_t)na*g_cfg.vocab, g_cfg.vocab*sizeof(float));  // output at last committed pos
        }
        printf("\n");
        double dt=nowt()-t0;
        fprintf(stderr,"[spec] %s N=%d K=%d ng=%d: %d tok / %d forwards = %.2f tok/forward, %.2f tok/s\n",
                g_int4?"int4":"fp32", emitted, K, NG, emitted, forwards, (double)emitted/forwards, emitted/dt);
        printf("tok_s %.3f tok_per_forward %.3f forwards %d\n", emitted/dt, (double)emitted/forwards, forwards);
        free(L);
    } else if (!strcmp(mode,"serve")) {
        // M20: lockstep request-batched greedy decode. B from QWEN_BATCH (default 8, cap
        // SRV_BMAX=16 per M20-D2), N tokens per sequence from argv[2]. All B sequences are
        // initialized with the SAME prompt so each stream is checkable against the
        // single-stream W4A8 greedy (QWEN_W4A8=1 ./qwen_infer greedy N) -- serve always
        // uses the int8-SDOT batched kernels (M20-D5), so THAT is the parity reference.
        if (!g_int4) { fprintf(stderr,"FATAL: serve requires int4 mode (set QWEN_INT4_BIN)\n"); return 1; }
        if (!w4a8_on()) fprintf(stderr,"[serve] note: serve always uses int8-SDOT kernels; parity reference is QWEN_W4A8=1 greedy\n");
        int B = getenv("QWEN_BATCH")?atoi(getenv("QWEN_BATCH")):8;
        if (B < 1) B = 1;
        if (B > srv_bcap()) { fprintf(stderr,"[serve] QWEN_BATCH=%d clamped to %d (%s)\n",B,srv_bcap(),
            kv_int4_on()?"scratch cap, M24-D6":kv_int8_on()?"scratch cap, M23-D6":"register-tile cap, M20-D2"); B = srv_bcap(); }
        int N = n_gen;
        if (np + N >= g_cfg.maxseq) { int oldN=N; N = g_cfg.maxseq-np-1;
            if (N < 1) { fprintf(stderr,"FATAL: prompt too long for serve (np=%d)\n",np); return 1; }
            fprintf(stderr,"[serve] N clamped %d->%d to fit MAXSEQ=%d\n",oldN,N,g_cfg.maxseq); }
        if (kv_int4_on()) {                    // M24: int4 per-seq caches + staging rings
            size_t cb=(size_t)B*g_cfg.nl*g_cfg.maxseq*(g_cfg.kvd/2), bsb=(size_t)B*g_cfg.nl*KV4_NB*g_cfg.kvd*sizeof(float),
                   vsb=(size_t)B*g_cfg.nl*g_cfg.maxseq*g_cfg.kvg*sizeof(float), stb=(size_t)B*g_cfg.nl*KV4_RING*g_cfg.kvd*sizeof(float);
            srv_k4=malloc(cb); srv_v4=malloc(cb); srv_k4s=malloc(bsb); srv_k4z=malloc(bsb);
            srv_v4sc=malloc(vsb); srv_k4stg=malloc(stb); srv_v4stg=malloc(stb);
            if (!srv_k4||!srv_v4||!srv_k4s||!srv_k4z||!srv_v4sc||!srv_k4stg||!srv_v4stg) { fprintf(stderr,"FATAL: serve int4 KV alloc failed -- lower QWEN_BATCH\n"); return 1; }
            fprintf(stderr,"[serve] M24 KV int4: %.0f MB codes+scales + %.0f MB staging at B=%d (int8 would be %.0f MB, fp32 %.0f MB)\n",
                    (2.0*cb+2.0*bsb+vsb)/1e6, 2.0*stb/1e6, B,
                    (double)B*g_cfg.nl*g_cfg.maxseq*2.0*(g_cfg.kvd+g_cfg.kvg*4.0)/1e6, (double)B*g_cfg.nl*g_cfg.maxseq*g_cfg.kvd*8.0/1e6);
        } else if (kv_int8_on()) {             // M23: int8 per-seq caches
            size_t qb=(size_t)B*g_cfg.nl*g_cfg.maxseq*g_cfg.kvd, scb=(size_t)B*g_cfg.nl*g_cfg.maxseq*g_cfg.kvg*sizeof(float);
            srv_kq=malloc(qb); srv_vq=malloc(qb); srv_ksc=malloc(scb); srv_vsc=malloc(scb);
            if (!srv_kq||!srv_vq||!srv_ksc||!srv_vsc) { fprintf(stderr,"FATAL: serve int8 KV alloc failed -- lower QWEN_BATCH\n"); return 1; }
            fprintf(stderr,"[serve] M23 KV int8: %.0f MB total at B=%d (fp32 would be %.0f MB)\n",
                    2.0*(qb+scb)/1e6, B, (size_t)B*g_cfg.nl*g_cfg.maxseq*g_cfg.kvd*8.0/1e6);
        } else {
            size_t kvbytes = (size_t)B*g_cfg.nl*g_cfg.maxseq*g_cfg.kvd*sizeof(float);
            srv_kc = malloc(kvbytes); srv_vc = malloc(kvbytes);
            if (!srv_kc || !srv_vc) { fprintf(stderr,"FATAL: serve KV cache alloc failed (2x %zu MB) -- lower QWEN_BATCH\n",kvbytes>>20); return 1; }
        }
        float *L = malloc((size_t)B*g_cfg.vocab*sizeof(float));
        int *streams = malloc((size_t)B*N*sizeof(int));
        if (!L || !streams) { fprintf(stderr,"FATAL: serve buffer alloc failed\n"); return 1; }
        int ids[SRV_BMAX];
        double tpf0 = nowt();
        for (int p=0;p<np;p++){ for(int s=0;s<B;s++) ids[s]=prompt[p]; serve_step(ids,B,p,L); }
        double tprefill = nowt()-tpf0;
        int pos = np-1;
        double t0 = nowt();
        for (int g=0; g<N; g++){
            for (int s=0;s<B;s++){ ids[s]=argmax_v(L+(size_t)s*g_cfg.vocab); streams[(size_t)s*N+g]=ids[s]; }
            pos++;                                   // np+N < MAXSEQ guaranteed by the clamp above
            serve_step(ids,B,pos,L);
        }
        double dt = nowt()-t0;
        if (prof_on()) fprintf(stderr,"[serve-prof] per-step: quant %.2f ms, gemm %.2f ms, attn %.2f ms, other %.2f ms (of %.2f ms; incl prefill in buckets)\n",
            g_srv_quant/(np+N)*1e3, g_srv_gemm/(np+N)*1e3, g_srv_attn/(np+N)*1e3,
            (tprefill+dt)/(np+N)*1e3 - (g_srv_quant+g_srv_gemm+g_srv_attn)/(np+N)*1e3, (tprefill+dt)/(np+N)*1e3);
        for (int s=0;s<B;s++){
            printf("serve seq%d:",s);
            for (int t=0;t<N;t++) printf(" %d",streams[(size_t)s*N+t]);
            printf("\n");
        }
        int allsame = 1;
        for (int s=1;s<B && allsame;s++) if (memcmp(streams, streams+(size_t)s*N, (size_t)N*sizeof(int))) allsame=0;
        double agg = (double)B*N/dt;
        fprintf(stderr,"[serve] int4 B=%d N=%d (prefill %d toks, %.2fs, batched): %.2f ms/step, aggregate %.2f tok/s (per-seq %.2f), streams %s\n",
                B, N, np, tprefill, dt/N*1e3, agg, agg/B, allsame?"all-identical":"DIVERGED");
        printf("agg_tok_s %.3f per_seq_tok_s %.3f ms_step %.3f B %d N %d identical %d\n",
               agg, agg/B, dt/N*1e3, B, N, allsame);
        free(L); free(streams); free(srv_kc); free(srv_vc); srv_kc=srv_vc=NULL;
        free(srv_kq); free(srv_vq); free(srv_ksc); free(srv_vsc); srv_kq=srv_vq=NULL; srv_ksc=srv_vsc=NULL;
        free(srv_k4); free(srv_v4); free(srv_k4s); free(srv_k4z); free(srv_v4sc); free(srv_k4stg); free(srv_v4stg);
        srv_k4=srv_v4=NULL; srv_k4s=srv_k4z=NULL; srv_v4sc=srv_k4stg=srv_v4stg=NULL;
    } else if (!strcmp(mode,"cbatch")) {
        // M21: continuous batching over a synthesized varied-length workload (see M21-D1..D6
        // above cbatch_step). R requests = truncations of the loaded prompt to a spread of
        // lengths with a spread of max_new_tokens, consumed FIFO by a pool of B slots.
        // argv[2] (n_gen) scales the max_new spread ceiling; QWEN_CB_REQS sets R.
        if (!g_int4) { fprintf(stderr,"FATAL: cbatch requires int4 mode (set QWEN_INT4_BIN)\n"); return 1; }
        int B = getenv("QWEN_BATCH")?atoi(getenv("QWEN_BATCH")):8;
        if (B < 1) B = 1;
        if (B > srv_bcap()) { fprintf(stderr,"[cbatch] QWEN_BATCH=%d clamped to %d (%s)\n",B,srv_bcap(),
            kv_int4_on()?"scratch cap, M24-D6":kv_int8_on()?"scratch cap, M23-D6":"register-tile cap, M20-D2"); B = srv_bcap(); }
        int R = getenv("QWEN_CB_REQS")?atoi(getenv("QWEN_CB_REQS")):12;
        if (R < 1) R = 1;
        int REPS = getenv("QWEN_CB_REPS")?atoi(getenv("QWEN_CB_REPS")):3;   // best-of-N walls
        if (REPS < 1) REPS = 1;
        long stop_extra = getenv("QWEN_CB_STOP_EXTRA")?atol(getenv("QWEN_CB_STOP_EXTRA")):-1;  // M21-D3 test hook
        // ---- M22: mixed chunked-prefill-with-decode (Orca/vLLM/SARATHI-style) ----
        // Removes M21-D2's admission stall: instead of prefilling an admitted request ALONE
        // (stalling every active decode slot for those steps), each scheduler step packs the
        // decode columns of all decode-phase slots AND up to a budget of prefill columns of
        // admitting (prefill-phase) slots into ONE cbatch_step call.
        // M22-D1 (mode selection & admission model): WHY -- mixed scheduling is an env flag
        // (QWEN_CB_MIXED=1) on the existing cbatch mode, not a new mode string: workload
        // synthesis, rep/identity checking, and reporting are shared, so the A/B against the
        // M21 sequential-prefill scheduler runs the same code with one env var flipped, and
        // QWEN_CB_MIXED unset leaves the M21 scheduler untouched. Admission now only OCCUPIES
        // a free slot (phase=prefill, zero forward work at admission time); the prompt is
        // consumed by interleaved chunks inside subsequent mixed steps until done, then the
        // slot flips to decode phase. COST: slot phase becomes tri-state (0 free / 1 decode /
        // 2 prefill) -- the M21 branch only ever writes/reads 0/1, so it is unaffected.
        // EXIT: set QWEN_CB_MIXED=0 (bit-identical M21 behaviour, verified by gate 3).
        // D-promote (SME2-followup track 3): default flipped 0->1 after direct verification
        // (not just trusting the gate-1/gate-3 comments above): cbatch MIXED=1 vs MIXED=0 vs
        // standalone greedy all produced byte-identical per-request token output (4 requests,
        // real Qwen weights), and MIXED=1 measurably helps besides -- worst-case decode
        // inter-token gap (TBT) 165.7ms -> 79.7ms, prefill-caused decode stalls 3 -> 0 on the
        // same synthetic workload. WHY default (not just leaving it opt-in): unlike this
        // session's QWEN_SME2 promotion, this flag is NOT hardware-gated -- it changes cbatch
        // behavior on every machine, including this repo's actual production host (no SME2
        // dependency at all, this is a pure scheduling change). COST: none identified (same
        // shared cbatch_step()/workload/reporting code path per M22-D1, only the admission
        // model differs). EXIT: QWEN_CB_MIXED=0 always available to force the old scheduler.
        // M22-D2 (column packing order): WHY -- decode columns are packed FIRST (slot-scan
        // order), prefill columns after: decode logits rows are then the contiguous prefix
        // m<ndec of L (consumed exactly like M21's decode step), and a prefill column is a
        // slot's LAST prompt token iff spos[m]==plen-1, so completion detection needs no new
        // per-column state. vs alternatives: a per-column role tag adds state for no benefit
        // -- each GEMM output column depends only on its own activation column, so packing
        // order is a bookkeeping choice, not a numerical one. COST: none. EXIT: none needed.
        // M22-D3 (per-step prefill token budget): WHY -- QWEN_CB_PREFILL_BUDGET (default 16)
        // caps how many prefill columns join a step, bounding per-step latency growth for
        // decode slots (SARATHI's chunked-prefill argument); it is further clamped to
        // SRV_BMAX-ndec because the batched-GEMM scratch and register-tile dispatcher cap
        // total columns at Q4_SDOT_BMAX=16. Progress guarantee: a prefill-phase slot does NOT
        // contribute a decode column, so ndec<=B-1<=SRV_BMAX-1 whenever one exists -> clamped
        // budget >= 1 -> every step advances prefill by >=1 token; no starvation, no deadlock.
        // Budget is spent over prefill-phase slots in slot-scan order (deterministic; admission
        // order stays FIFO per M21-D5). COST: a long prompt still takes ~ceil(plen/budget)
        // steps to finish admission -- but decode no longer stalls during them, which is the
        // point. EXIT: tune the env var (1..16).
        // M22-D4 (causal + neighbour correctness of mixed columns): WHY this needs NO kernel
        // or cbatch_step change -- per layer, cbatch_step writes ALL A columns' KV BEFORE any
        // column's attention runs, and column m attends only positions 0..spos[m] of ITS OWN
        // slot's cache. A decode column and a prefill column always live in DIFFERENT slots
        // (disjoint KV caches; per-column RoPE tables per M21-D6); consecutive prefill columns
        // of the SAME slot read each other's just-written KV exactly as causality requires --
        // the identical argument already established for M21-D2's chunked prefill. Neighbour
        // independence is inherited from M21-D4 unchanged: per-column int32-exact SDOT
        // accumulation + fixed per-row scale order mean a column's output does not depend on
        // WHICH or HOW MANY columns share the step -- prefill neighbours included. Gate 1
        // verifies bit-identity vs standalone QWEN_W4A8=1 greedy per request.
        // M22-D5 (logits policy): WHY -- the head GEMM runs (want_logits=1) whenever the step
        // carries >=1 decode column or a completing (last-prompt-token) prefill column; head
        // rows of non-final prefill columns are computed-but-unread on such steps. vs
        // alternatives: a per-column logits mask inside cbatch_step would save those unread
        // rows, but the head GEMM is weight-bound (VOCAB x D weight traffic dominates; extra
        // columns are near-free) and masking would modify the shared step function that gate 3
        // wants untouched. Pure-prefill steps with no completing column still skip the head
        // entirely (want_logits=0), matching M21's non-final-chunk behaviour. COST: a few
        // unread head rows on mixed steps. EXIT: per-column mask in cbatch_step.
        int mixed = getenv("QWEN_CB_MIXED")?atoi(getenv("QWEN_CB_MIXED")):1;
        int pfB = getenv("QWEN_CB_PREFILL_BUDGET")?atoi(getenv("QWEN_CB_PREFILL_BUDGET")):16;
        if (pfB < 1) pfB = 1;
        if (pfB > srv_bcap()) pfB = srv_bcap();   // M23-D6: 16 when fp32 KV (HEAD-identical), 32 when int8
        // Deterministic varied workload: prompt lengths 4..40 (clipped to np), max_new 6..(6+n_gen-1).
        int *plen = malloc(R*sizeof(int)), *maxnew = malloc(R*sizeof(int));
        int mnspread = n_gen > 1 ? n_gen : 27;
        int totmax = 0;
        for (int r=0;r<R;r++){
            plen[r] = 4 + (int)(((long)r*13)%37);            // 4..40, coprime stride -> mixed order
            if (plen[r] > np) plen[r] = np;
            maxnew[r] = 6 + (int)(((long)r*11)%mnspread);    // 6..6+mnspread-1
            if (plen[r]+maxnew[r] >= g_cfg.maxseq) maxnew[r] = g_cfg.maxseq-plen[r]-1;
            totmax += maxnew[r];
        }
        if (kv_int4_on()) {                    // M24: int4 per-seq caches + staging rings
            size_t cb=(size_t)B*g_cfg.nl*g_cfg.maxseq*(g_cfg.kvd/2), bsb=(size_t)B*g_cfg.nl*KV4_NB*g_cfg.kvd*sizeof(float),
                   vsb=(size_t)B*g_cfg.nl*g_cfg.maxseq*g_cfg.kvg*sizeof(float), stb=(size_t)B*g_cfg.nl*KV4_RING*g_cfg.kvd*sizeof(float);
            srv_k4=malloc(cb); srv_v4=malloc(cb); srv_k4s=malloc(bsb); srv_k4z=malloc(bsb);
            srv_v4sc=malloc(vsb); srv_k4stg=malloc(stb); srv_v4stg=malloc(stb);
            if (!srv_k4||!srv_v4||!srv_k4s||!srv_k4z||!srv_v4sc||!srv_k4stg||!srv_v4stg) { fprintf(stderr,"FATAL: cbatch int4 KV alloc failed -- lower QWEN_BATCH\n"); return 1; }
            fprintf(stderr,"[cbatch] M24 KV int4: %.0f MB codes+scales + %.0f MB staging at B=%d (int8 would be %.0f MB, fp32 %.0f MB)\n",
                    (2.0*cb+2.0*bsb+vsb)/1e6, 2.0*stb/1e6, B,
                    (double)B*g_cfg.nl*g_cfg.maxseq*2.0*(g_cfg.kvd+g_cfg.kvg*4.0)/1e6, (double)B*g_cfg.nl*g_cfg.maxseq*g_cfg.kvd*8.0/1e6);
        } else if (kv_int8_on()) {             // M23: int8 per-seq caches
            size_t qb=(size_t)B*g_cfg.nl*g_cfg.maxseq*g_cfg.kvd, scb=(size_t)B*g_cfg.nl*g_cfg.maxseq*g_cfg.kvg*sizeof(float);
            srv_kq=malloc(qb); srv_vq=malloc(qb); srv_ksc=malloc(scb); srv_vsc=malloc(scb);
            if (!srv_kq||!srv_vq||!srv_ksc||!srv_vsc) { fprintf(stderr,"FATAL: cbatch int8 KV alloc failed -- lower QWEN_BATCH\n"); return 1; }
            fprintf(stderr,"[cbatch] M23 KV int8: %.0f MB total at B=%d (fp32 would be %.0f MB)\n",
                    2.0*(qb+scb)/1e6, B, (size_t)B*g_cfg.nl*g_cfg.maxseq*g_cfg.kvd*8.0/1e6);
        } else {
            size_t kvbytes = (size_t)B*g_cfg.nl*g_cfg.maxseq*g_cfg.kvd*sizeof(float);
            srv_kc = malloc(kvbytes); srv_vc = malloc(kvbytes);
            if (!srv_kc || !srv_vc) { fprintf(stderr,"FATAL: cbatch KV cache alloc failed (2x %zu MB) -- lower QWEN_BATCH\n",kvbytes>>20); return 1; }
        }
        float *L = malloc((size_t)SRV_BMAX*g_cfg.vocab*sizeof(float));
        int **out = malloc(R*sizeof(int*)); int *nout = malloc(R*sizeof(int));
        for (int r=0;r<R;r++) out[r] = malloc((size_t)maxnew[r]*sizeof(int));
        if (!L || !out || !nout) { fprintf(stderr,"FATAL: cbatch buffer alloc failed\n"); return 1; }
        // per-slot state (M21-D1; M22-D1 adds cb_pref = next prompt index to feed while phase==2)
        int cb_active[SRV_BMAX], cb_req[SRV_BMAX], cb_tok[SRV_BMAX], cb_pos[SRV_BMAX], cb_pref[SRV_BMAX];
        // M22-D6 (inter-token latency metric): WHY -- on a run-to-drain workload with a fixed
        // token total, sequential-vs-mixed prefill moves the SAME forward work between steps,
        // so aggregate tok/s alone cannot expose the admission stall; the stall is a LATENCY
        // event: an active decode slot emits nothing while another request's prompt prefills
        // alone (M21), vs emitting every step (M22). Measured as the wall-clock gap between
        // consecutive emissions of the same slot (TBT), timestamped once per scheduler step
        // (step granularity is the resolution that matters; per-emission clocks add noise).
        // The clock starts when a slot ENTERS decode phase (its first-token emission), so
        // prefill duration itself (time-to-first-token) is deliberately not counted -- this
        // isolates the neighbour-induced stall. COST: one nowt() + B doubles per step. EXIT:
        // none needed (diagnostic only; printed, never branched on).
        double cb_lastemit[SRV_BMAX];
        double best_wall=1e18, best_decode=0, best_prefill=0;
        long best_dsteps=0, best_asum=0, best_psteps=0; double best_fullB_ms=0; long best_fullB_n=0;
        long best_pstall=0, best_msteps=0, best_pfonly=0, best_pfsum=0;
        double best_maxgap=0, best_sumgap=0; long best_ngap=0;
        int *out_prev = NULL; int reps_identical = 1;
        for (int rep=0; rep<REPS; rep++){
            for (int s=0;s<B;s++) cb_active[s]=0;
            for (int r=0;r<R;r++) nout[r]=0;
            int qhead=0, nact=0;
            long dsteps=0, asum=0, psteps=0; double t_decode=0, t_prefill=0;
            double fullB_ms=0; long fullB_n=0;
            long pstall=0;                       // M21 branch: prefill steps run while decode slots sat idle
            long msteps=0, pfonly=0, pfsum=0;    // M22 branch: mixed steps / prefill-only steps / prefill columns
            double maxgap=0, sumgap=0; long ngap=0;   // M22-D6 decode inter-token gaps (both branches)
            double w0 = nowt();
            while (qhead < R || nact > 0) {
              if (!mixed) {
                // admission: fill every free slot from the FIFO queue (M21-D2/D5)
                for (int s=0;s<B && qhead<R;s++){
                    if (cb_active[s]) continue;
                    int r = qhead++;
                    double tp0 = nowt();
                    int lastA = 0;   // columns in the final chunk (its last column carries the logits)
                    long nch = 0;
                    for (int p=0;p<plen[r];p+=SRV_PCHUNK){   // M21-D2: chunked single-sequence prefill (M23: chunk fixed at 16)
                        int A = plen[r]-p; if (A > SRV_PCHUNK) A = SRV_PCHUNK;
                        int ids_[SRV_BMAX], sl_[SRV_BMAX], sp_[SRV_BMAX];
                        for (int m=0;m<A;m++){ ids_[m]=prompt[p+m]; sl_[m]=s; sp_[m]=p+m; }
                        cbatch_step(ids_,sl_,sp_,A,L, p+A>=plen[r]);
                        psteps++; lastA = A; nch++;
                    }
                    t_prefill += nowt()-tp0;
                    if (nact > 0) pstall += nch;   // M22 gate-2 metric: these steps stalled nact decode slots
                    int t = argmax_v(L+(size_t)(lastA-1)*g_cfg.vocab);
                    out[r][nout[r]++] = t;
                    int stop = (t==151643 || t==151645 || t==stop_extra || nout[r]>=maxnew[r] || plen[r] >= g_cfg.maxseq);
                    if (!stop){ cb_active[s]=1; cb_req[s]=r; cb_tok[s]=t; cb_pos[s]=plen[r]; nact++;
                        cb_lastemit[s]=nowt(); }   // M22-D6: decode-phase TBT clock starts at first token
                    // else: finished during prefill -- slot stays free for the next queued request
                }
                if (nact == 0) continue;   // everything admitted so far finished at prefill
                // ragged gather -> ONE batched step over the A active slots (M21-D1/D4)
                int A=0; int ids[SRV_BMAX], slots[SRV_BMAX], sposs[SRV_BMAX];
                for (int s=0;s<B;s++) if (cb_active[s]){ ids[A]=cb_tok[s]; slots[A]=s; sposs[A]=cb_pos[s]; A++; }
                double td0 = nowt();
                cbatch_step(ids,slots,sposs,A,L,1);
                double td = nowt()-td0;
                t_decode += td; dsteps++; asum += A;
                if (A == B){ fullB_ms += td; fullB_n++; }
                double temit = nowt();   // M22-D6: step-granularity emission timestamp
                for (int m=0;m<A;m++){ int s=slots[m], r=cb_req[s];
                    int t = argmax_v(L+(size_t)m*g_cfg.vocab);
                    out[r][nout[r]++] = t;
                    cb_pos[s]++;
                    { double gp=temit-cb_lastemit[s]; if(gp>maxgap)maxgap=gp; sumgap+=gp; ngap++; cb_lastemit[s]=temit; }
                    if (t==151643 || t==151645 || t==stop_extra || nout[r]>=maxnew[r] || cb_pos[s] >= g_cfg.maxseq){
                        cb_active[s]=0; nact--;      // eviction (M21-D3): slot admits next request next iteration
                    } else cb_tok[s]=t;
                }
              } else {
                // ---- M22 mixed scheduler (see M22-D1..D5 above) ----
                // admission (M22-D1): occupy free slots immediately, zero forward work here
                for (int s=0;s<B && qhead<R;s++){
                    if (cb_active[s]) continue;
                    int r = qhead++;
                    cb_active[s]=2; cb_req[s]=r; cb_pref[s]=0; nact++;
                }
                // column packing (M22-D2): decode columns first ...
                int A=0; int ids[SRV_BMAX], slots[SRV_BMAX], sposs[SRV_BMAX];
                for (int s=0;s<B;s++) if (cb_active[s]==1){ ids[A]=cb_tok[s]; slots[A]=s; sposs[A]=cb_pos[s]; A++; }
                int ndec = A;
                // ... then prefill columns under the per-step budget (M22-D3)
                int budget = pfB; if (budget > srv_bcap()-ndec) budget = srv_bcap()-ndec;   // M23-D6
                int want_logits = (ndec > 0);
                for (int s=0;s<B && budget>0;s++){
                    if (cb_active[s]!=2) continue;
                    int r = cb_req[s];
                    int take = plen[r]-cb_pref[s]; if (take > budget) take = budget;
                    for (int i=0;i<take;i++){ ids[A]=prompt[cb_pref[s]+i]; slots[A]=s; sposs[A]=cb_pref[s]+i; A++; }
                    cb_pref[s] += take; budget -= take;
                    if (cb_pref[s] >= plen[r]) want_logits = 1;   // final prompt column in this step (M22-D5)
                }
                if (A == 0) continue;   // defensive: unreachable while nact>0 (M22-D3 progress guarantee)
                double td0 = nowt();
                cbatch_step(ids,slots,sposs,A,L,want_logits);
                double td = nowt()-td0;
                t_decode += td; dsteps++; asum += ndec; pfsum += A-ndec;
                if (ndec > 0 && A > ndec) msteps++;         // prefill+decode genuinely co-scheduled
                if (ndec == 0) pfonly++;                    // no decode-phase slot existed (startup/drain only)
                if (ndec == B){ fullB_ms += td; fullB_n++; }
                double temit = nowt();   // M22-D6: step-granularity emission timestamp
                // decode columns: emit/evict, same policy as M21-D3
                for (int m=0;m<ndec;m++){ int s=slots[m], r=cb_req[s];
                    int t = argmax_v(L+(size_t)m*g_cfg.vocab);
                    out[r][nout[r]++] = t;
                    cb_pos[s]++;
                    { double gp=temit-cb_lastemit[s]; if(gp>maxgap)maxgap=gp; sumgap+=gp; ngap++; cb_lastemit[s]=temit; }
                    if (t==151643 || t==151645 || t==stop_extra || nout[r]>=maxnew[r] || cb_pos[s] >= g_cfg.maxseq){
                        cb_active[s]=0; nact--;
                    } else cb_tok[s]=t;
                }
                // prefill columns: only a slot's LAST prompt column carries consumable logits
                for (int m=ndec;m<A;m++){ int s=slots[m], r=cb_req[s];
                    if (sposs[m] != plen[r]-1) continue;
                    int t = argmax_v(L+(size_t)m*g_cfg.vocab);
                    out[r][nout[r]++] = t;
                    if (t==151643 || t==151645 || t==stop_extra || nout[r]>=maxnew[r] || plen[r] >= g_cfg.maxseq){
                        cb_active[s]=0; nact--;             // finished at prefill (first token was a stop)
                    } else { cb_active[s]=1; cb_tok[s]=t; cb_pos[s]=plen[r];
                        cb_lastemit[s]=temit; }   // M22-D6: decode-phase TBT clock starts at first token
                }
              }
            }
            double wall = nowt()-w0;
            if (rep==0){ out_prev = malloc(totmax*sizeof(int)); int o=0;
                for(int r=0;r<R;r++) for(int i=0;i<nout[r];i++) out_prev[o++]=out[r][i]; }
            else { int o=0, same=1;
                for(int r=0;r<R;r++) for(int i=0;i<nout[r];i++) if(out_prev[o++]!=out[r][i]) same=0;
                if(!same) reps_identical=0; }
            if (wall < best_wall){ best_wall=wall; best_decode=t_decode; best_prefill=t_prefill;
                best_dsteps=dsteps; best_asum=asum; best_psteps=psteps; best_fullB_ms=fullB_ms; best_fullB_n=fullB_n;
                best_pstall=pstall; best_msteps=msteps; best_pfonly=pfonly; best_pfsum=pfsum;
                best_maxgap=maxgap; best_sumgap=sumgap; best_ngap=ngap; }
        }
        long tottok=0; for(int r=0;r<R;r++) tottok+=nout[r];
        for (int r=0;r<R;r++){
            printf("cbatch req%d plen %d maxnew %d nout %d:",r,plen[r],maxnew[r],nout[r]);
            for (int i=0;i<nout[r];i++) printf(" %d",out[r][i]);
            printf("\n");
        }
        // Lockstep contrast (M20-serve model): FIFO batches of B, every request padded to its
        // batch's max(plen+nout-1) steps at full-B step cost. Estimated with this run's measured
        // full-B decode step time (falls back to the overall mean when no step ever hit A==B).
        long ls_steps=0;
        for (int b0=0;b0<R;b0+=B){ long mx=0;
            for (int r=b0;r<R && r<b0+B;r++){ long st=plen[r]+nout[r]-1; if(st>mx) mx=st; }
            ls_steps += mx; }
        double msB = best_fullB_n ? best_fullB_ms/best_fullB_n*1e3
                                  : (best_dsteps? best_decode/best_dsteps*1e3 : 0);
        double ls_wall_est = ls_steps * msB / 1e3;
        double util = best_dsteps ? (double)best_asum/((double)best_dsteps*B) : 0;
        double ls_util = (double)tottok/((double)ls_steps*B);
        if (mixed)
            fprintf(stderr,"[cbatch] MIXED int4 B=%d R=%d pfbudget=%d tok=%ld: wall %.2fs, %ld steps (mixed pf+dec %ld, prefill-only %ld, decode-stall 0 by construction), agg %.2f tok/s, decode cols %.2f/%d per step (util %.1f%%), prefill cols %ld (%.2f/step), reps=%d %s\n",
                B,R,pfB,tottok,best_wall,best_dsteps,best_msteps,best_pfonly,
                tottok/best_wall,(double)best_asum/(best_dsteps?best_dsteps:1),B,100.0*util,
                best_pfsum,(double)best_pfsum/(best_dsteps?best_dsteps:1),REPS,
                reps_identical?"rep-identical":"REP-DIVERGED");
        else
            fprintf(stderr,"[cbatch] int4 B=%d R=%d tok=%ld: wall %.2fs (prefill %.2fs/%ld steps [%ld stalled active decode slots], decode %.2fs/%ld steps), agg %.2f tok/s, avg_active %.2f/%d (util %.1f%%), reps=%d %s\n",
                B,R,tottok,best_wall,best_prefill,best_psteps,best_pstall,best_decode,best_dsteps,
                tottok/best_wall,(double)best_asum/(best_dsteps?best_dsteps:1),B,100.0*util,REPS,
                reps_identical?"rep-identical":"REP-DIVERGED");
        fprintf(stderr,"[cbatch] lockstep-serve estimate on same workload: %ld full-B steps x %.2f ms = %.2fs, agg %.2f tok/s, useful-slot fraction %.1f%% (vs cbatch %.1f%% of decode slot-steps)\n",
                ls_steps,msB,ls_wall_est,ls_wall_est>0?tottok/ls_wall_est:0,100.0*ls_util,100.0*util);
        fprintf(stderr,"[cbatch] decode inter-token gap (TBT, M22-D6): max %.1f ms, mean %.1f ms over %ld gaps\n",
                best_maxgap*1e3, best_ngap?best_sumgap/best_ngap*1e3:0, best_ngap);
        printf("agg_tok_s %.3f wall_s %.3f decode_s %.3f prefill_s %.3f util %.4f avg_active %.3f B %d R %d tok %ld ls_est_tok_s %.3f ls_util %.4f ms_stepB %.3f reps_identical %d mixed %d steps %ld msteps %ld pfonly %ld pstall %ld pf_cols %ld tbt_max_ms %.3f tbt_mean_ms %.3f\n",
               tottok/best_wall,best_wall,best_decode,best_prefill,util,
               (double)best_asum/(best_dsteps?best_dsteps:1),B,R,tottok,
               ls_wall_est>0?tottok/ls_wall_est:0,ls_util,msB,reps_identical,
               mixed,best_dsteps,best_msteps,best_pfonly,best_pstall,best_pfsum,
               best_maxgap*1e3,best_ngap?best_sumgap/best_ngap*1e3:0);
        for (int r=0;r<R;r++) free(out[r]);
        free(out); free(nout); free(plen); free(maxnew); free(L); free(out_prev);
        free(srv_kc); free(srv_vc); srv_kc=srv_vc=NULL;
        free(srv_kq); free(srv_vq); free(srv_ksc); free(srv_vsc); srv_kq=srv_vq=NULL; srv_ksc=srv_vsc=NULL;
        free(srv_k4); free(srv_v4); free(srv_k4s); free(srv_k4z); free(srv_v4sc); free(srv_k4stg); free(srv_v4stg);
        srv_k4=srv_v4=NULL; srv_k4s=srv_k4z=NULL; srv_v4sc=srv_k4stg=srv_v4stg=NULL;
    }
    if (g_int4) q4pool_destroy(&g_pool);
    return 0;
}
