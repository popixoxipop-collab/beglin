// V5a: MLX GPU backend implementation. C++17, links MLX (D-gpu-1/D-gpu-2).
// Consumes the AF-blob's int4g64 affine bytes directly -- no repack, no
// requantize (D-gpu-3, F-1: max abs diff 0.0 across 3200 sampled coordinates
// against this exact fixture). See mlx_moe.h and
// PLAN_v5_v6_gpu_backend_and_role_device.md for the full design.
//
// This file performs no fp32->intN encoding of its own -- it only binds and
// dequantizes bytes an existing quantizer already produced -- so D18's
// residual-vs-stochastic-rounding tradeoff does not apply here.
#include "mlx_moe.h"
#include "mlx/mlx.h"

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mx = mlx::core;

struct QTensor {
    mx::array w;       // {E, out, in/8} uint32 -- packed int4 codes, row-major per expert
    mx::array scales;  // {E, out, ng} float32
    mx::array biases;  // {E, out, ng} float32
    long E, out, in, ng;
    int bits;
};

static std::unordered_map<std::string, QTensor> g_tensors;
static int g_bound_count = 0;

static void noop_deleter(void *) {
    // Data is owned by qwen_infer.c's mmap of the AF blob (or, for
    // mlx_gpu_matvec_probe's transient x, by the caller's own stack/heap
    // buffer for the duration of this synchronous call) -- MLX must never
    // free it.
}

int mlx_gpu_available(void) {
    static int checked = 0;
    static int available = 0;
    if (checked) return available;
    checked = 1;
    try {
        auto x = mx::sum(mx::ones({4}));
        mx::eval(x);
        available = 1;
    } catch (...) {
        available = 0;
    }
    return available;
}

int mlx_gpu_bind_af(const uint8_t *blob, long blob_bytes, const char *name,
                     long E, long out, long in, long ng,
                     long packed_off, long scale_off, long bias_off, int bits) {
    (void)blob_bytes;  // reserved for a future bounds-check gate, unused for now
    if (!mlx_gpu_available()) return 0;
    // V5a targets this fixture's real bits value only (F-13: every AF-blob
    // tensor is bits==4) -- refusal rather than silently mis-decoding a
    // bit-width this gate hasn't verified.
    if (bits != 4) return 0;
    if (E <= 0 || out <= 0 || in <= 0 || ng <= 0 || (in % 8) != 0) return 0;

    try {
        long row_words = in / 8;
        void *w_ptr = (void *)(blob + packed_off);
        void *scale_ptr = (void *)(blob + scale_off);
        void *bias_ptr = (void *)(blob + bias_off);

        // NOTE (real finding, this round): mx::allocator::can_reuse_alien_buffer()
        // segfaults unconditionally on this host's installed MLX build when called
        // from a plain C++ (non-Python) process -- reproduced in isolation with a
        // trivial malloc'd pointer, both before and after warming up MLX's
        // allocator/Metal device via a real eval(). The raw-pointer mx::array
        // constructor itself works fine without it (also reproduced in isolation);
        // this call was only ever an optional informational check for Gate 5's
        // zero-copy accounting, never required for correctness (mlx_moe.h's own
        // doc comment already said as much) -- so it is skipped entirely here.
        // Gate 5 residency instead relies solely on mlx_gpu_report_memory()'s
        // active/peak/cache counters, which don't go through this function.
        mx::array w(w_ptr, {(int)E, (int)out, (int)row_words}, mx::uint32,
                    noop_deleter);
        mx::array scales(scale_ptr, {(int)E, (int)out, (int)ng}, mx::float32,
                          noop_deleter);
        mx::array biases(bias_ptr, {(int)E, (int)out, (int)ng}, mx::float32,
                          noop_deleter);

        // insert_or_assign, not operator[]= -- QTensor holds mx::array
        // fields with no default constructor, so operator[]'s implicit
        // default-then-assign doesn't compile.
        g_tensors.insert_or_assign(
            std::string(name),
            QTensor{w, scales, biases, E, out, in, ng, bits});
        g_bound_count++;
        return 1;
    } catch (...) {
        return 0;
    }
}

int mlx_gpu_zerocopy_count(int *zero_copy, int *copied, size_t *bytes_copied) {
    // Real finding, this round: can_reuse_alien_buffer() (the only way to
    // classify zero-copy vs copied) segfaults on this host's MLX build --
    // see the comment in mlx_gpu_bind_af(). Reporting an honest "unknown"
    // rather than fabricating a split we can't verify; Gate 5's actual
    // residency evidence comes from mlx_gpu_report_memory() instead.
    if (zero_copy) *zero_copy = -1;
    if (copied) *copied = -1;
    if (bytes_copied) *bytes_copied = 0;
    return g_bound_count;
}

int mlx_gpu_dequant_probe(const char *name, long e, long row, long col0, int ncols,
                           float *out_vals) {
    auto it = g_tensors.find(name);
    if (it == g_tensors.end()) return 0;
    QTensor &t = it->second;
    if (e < 0 || e >= t.E || row < 0 || row >= t.out) return 0;
    if (ncols <= 0 || col0 < 0 || col0 + ncols > t.in) return 0;

    try {
        // mx::dequantize requires >=2 dims (real finding: single-take-per-axis down to a
        // pure 1D row throws "must have at least 2 dimension"). Take the one row we need
        // (cheap -- not the whole tensor), then expand_dims back to {1,row_words}/{1,ng}
        // so dequantize sees a valid 2D shape without paying to dequantize all `out` rows.
        mx::array w_row = mx::expand_dims(
            mx::take(mx::take(t.w, (int)e, 0), (int)row, 0), 0);       // {1, row_words}
        mx::array s_row = mx::expand_dims(
            mx::take(mx::take(t.scales, (int)e, 0), (int)row, 0), 0);  // {1, ng}
        mx::array b_row = mx::expand_dims(
            mx::take(mx::take(t.biases, (int)e, 0), (int)row, 0), 0);  // {1, ng}

        mx::array deq = mx::dequantize(w_row, s_row, b_row,
                                        /*group_size=*/64, /*bits=*/4);  // {1, in}
        mx::eval(deq);
        const float *ptr = deq.data<float>();
        // Real finding (this round): earlier version always read columns [0,ncols) regardless
        // of the caller's col0, silently comparing unrelated columns against the CPU side's
        // moe_decode_af(..., col0+c) -- offsetting by col0 here is the actual fix.
        std::memcpy(out_vals, ptr + col0, sizeof(float) * (size_t)ncols);
        return 1;
    } catch (...) {
        return 0;
    }
}

int mlx_gpu_matvec_probe(const char *name, long e, const float *x, float *y) {
    auto it = g_tensors.find(name);
    if (it == g_tensors.end()) return 0;
    QTensor &t = it->second;
    if (e < 0 || e >= t.E) return 0;

    try {
        mx::array w_e = mx::take(t.w, (int)e, 0);            // {out, in/8}
        mx::array s_e = mx::take(t.scales, (int)e, 0);        // {out, ng}
        mx::array b_e = mx::take(t.biases, (int)e, 0);        // {out, ng}

        mx::array xin((void *)x, {1, (int)t.in}, mx::float32, noop_deleter);

        mx::array yout = mx::quantized_matmul(
            xin, w_e, s_e, b_e, /*transpose=*/true,
            /*group_size=*/64, /*bits=*/4);
        mx::eval(yout);
        const float *ptr = yout.data<float>();
        std::memcpy(y, ptr, sizeof(float) * (size_t)t.out);
        return 1;
    } catch (...) {
        return 0;
    }
}

void mlx_gpu_report_memory(size_t *active, size_t *peak, size_t *cache) {
    if (active) *active = mx::get_active_memory();
    if (peak) *peak = mx::get_peak_memory();
    if (cache) *cache = mx::get_cache_memory();
}

// ---------------------------------------------------------------------------
// V5b: layer-0 MLA attention on GPU. Direct transcription of the CPU
// moe_mla_attention()'s arithmetic (qwen_infer.c, verbatim from Phase MoE-2a's
// mla_verify.c) -- RoPE's traditional=true interleaved pairing and the YaRN
// freqs-as-divisor convention (F-16) are reused via mx::fast::rope's `freqs`
// override parameter, confirmed by isolated probe to match the CPU arithmetic
// to float32 rounding precision (max abs diff 4.77e-07 on a 32-distinct-value
// real YaRN table) before being wired in here -- not assumed from the header.
// mx::fast::rms_norm and mx::fast::scaled_dot_product_attention(mask_mode="")
// were verified the same way (rms_norm: 1.19e-07; sdpa: 1.49e-08, both against
// hand-written CPU references using the exact same arithmetic as this file's
// production CPU counterparts).
static int g_mla_n_heads = 0, g_mla_q_head_dim = 0, g_mla_qk_nope = 0, g_mla_qk_rope = 0,
           g_mla_v_hd = 0, g_mla_kv_lora = 0;
static double g_mla_rope_mscale = 0.0, g_mla_attn_scale = 0.0, g_mla_rms_eps = 1e-6;
static std::vector<float> g_mla_yarn_freqs;   // qk_rope_hd/2 entries
// BUG 1 ROOT CAUSE (found + fixed, throughput round part 2): the fused path's per-call
// freqs_f32 used to wrap a FUNCTION-LOCAL std::vector<float> via noop_deleter -- safe
// only as long as eval() always happened before the function returned (freqs_f still
// alive on the stack at eval time). Once x_out's eval was deferred past the function's
// return (the "one-eval-per-token" design), MLX read freqs_f32 through a pointer into
// ALREADY-FREED stack memory, repeatedly overwritten by the next 26 layers' own local
// variables -- a genuine use-after-free, not an MLX bug. Confirmed by symptom: RoPE's
// rotation angle is pos/freq, which is exactly 0 at pos=0 regardless of the (garbage)
// freq value -- pos=0 stayed correct by coincidence (identity rotation either way)
// while pos>=1 depended on the actual freq value and broke, non-deterministically
// (different stack garbage on different runs) -- exactly what was observed. Fix:
// wrap this PERSISTENT global instead, populated once in mlx_gpu_mla_config().
static std::vector<float> g_mla_yarn_freqs_f32;

// GPU arm's own K/V cache (separate from qwen_infer.c's CPU-side g_moe_K/V --
// D-gpu-2, no shared mutable state across the vendor boundary). Layout
// [layer][head][pos][dim], so the slice needed for one sdpa call (all heads,
// positions 0..pos, one layer) is built by copying H independent contiguous
// runs rather than reordering per-element.
#define MLA_L0_MAXPOS 32
#define MLA_MAXLAYERS 32
static std::vector<float> g_mla_K;  // MLA_MAXLAYERS * H * MLA_L0_MAXPOS * q_head_dim
static std::vector<float> g_mla_V;  // MLA_MAXLAYERS * H * MLA_L0_MAXPOS * v_hd

int mlx_gpu_mla_config(int n_heads, int q_head_dim, int qk_nope_hd, int qk_rope_hd,
                        int v_hd, int kv_lora_rank, double rope_mscale, double attn_scale,
                        const double *yarn_freqs_half, double rms_eps) {
    if (!mlx_gpu_available()) return 0;
    g_mla_n_heads = n_heads;
    g_mla_q_head_dim = q_head_dim;
    g_mla_qk_nope = qk_nope_hd;
    g_mla_qk_rope = qk_rope_hd;
    g_mla_v_hd = v_hd;
    g_mla_kv_lora = kv_lora_rank;
    g_mla_rope_mscale = rope_mscale;
    g_mla_attn_scale = attn_scale;
    g_mla_rms_eps = rms_eps;
    int half = qk_rope_hd / 2;
    g_mla_yarn_freqs.assign(yarn_freqs_half, yarn_freqs_half + half);
    g_mla_yarn_freqs_f32.assign(g_mla_yarn_freqs.begin(), g_mla_yarn_freqs.end());
    g_mla_K.assign((size_t)MLA_MAXLAYERS * n_heads * MLA_L0_MAXPOS * q_head_dim, 0.0f);
    g_mla_V.assign((size_t)MLA_MAXLAYERS * n_heads * MLA_L0_MAXPOS * v_hd, 0.0f);
    return 1;
}

// V5c: same MLA attention arithmetic as mlx_gpu_mla_layer0(), generalized to an arbitrary
// layer index (name prefix + own K/V cache slice) so run_moe_gpu_full_gate() can drive all 27
// layers with one implementation. mlx_gpu_mla_layer0() (V5b, unchanged) is now a thin l=0
// wrapper over this -- same tensors, same cache slot, byte-for-byte identical behavior.
static int mlx_gpu_mla_layer_impl(int l, const float *h, int pos, const float *kv_a_ln_w, float *o_out) {
    if (g_mla_n_heads == 0) return 0;   // mlx_gpu_mla_config() not called
    if (pos < 0 || pos >= MLA_L0_MAXPOS || l < 0 || l >= MLA_MAXLAYERS) return 0;
    char nq[96], nkva[96], nkvb[96], no[96];
    snprintf(nq, sizeof nq, "model.layers.%d.self_attn.q_proj", l);
    snprintf(nkva, sizeof nkva, "model.layers.%d.self_attn.kv_a_proj_with_mqa", l);
    snprintf(nkvb, sizeof nkvb, "model.layers.%d.self_attn.kv_b_proj", l);
    snprintf(no, sizeof no, "model.layers.%d.self_attn.o_proj", l);
    auto itq   = g_tensors.find(nq);
    auto itkva = g_tensors.find(nkva);
    auto itkvb = g_tensors.find(nkvb);
    auto ito   = g_tensors.find(no);
    if (itq == g_tensors.end() || itkva == g_tensors.end() ||
        itkvb == g_tensors.end() || ito == g_tensors.end()) return 0;

    try {
        const int H = g_mla_n_heads, QHD = g_mla_q_head_dim, NOPE = g_mla_qk_nope,
                  ROPE = g_mla_qk_rope, VHD = g_mla_v_hd, KVLORA = g_mla_kv_lora;
        const int hidden = (int)itq->second.in;
        const int qdim = (int)itq->second.out;          // H*QHD
        const int kva_out = (int)itkva->second.out;      // KVLORA+ROPE
        const int kvb_out = (int)itkvb->second.out;      // H*(NOPE+VHD)
        const int attn_out_dim = (int)ito->second.in;    // H*VHD

        // q_proj(h), kv_a_proj(h) -- reuse the already-verified matvec path (Gate 4).
        std::vector<float> q(qdim), kv_ap(kva_out);
        if (!mlx_gpu_matvec_probe(nq, 0, h, q.data())) return 0;
        if (!mlx_gpu_matvec_probe(nkva, 0, h, kv_ap.data())) return 0;

        float *compressed_kv = kv_ap.data();              // first KVLORA
        float *k_pe = kv_ap.data() + KVLORA;               // last ROPE

        // kv_a_layernorm (RMSNorm) on the GPU, verified building block.
        mx::array kvin((void *)compressed_kv, {1, KVLORA}, mx::float32, noop_deleter);
        mx::array wln((void *)kv_a_ln_w, {KVLORA}, mx::float32, noop_deleter);
        mx::array normed = mx::fast::rms_norm(kvin, wln, (float)g_mla_rms_eps);
        mx::eval(normed);
        std::vector<float> normed_kv(KVLORA);
        std::memcpy(normed_kv.data(), normed.data<float>(), sizeof(float) * KVLORA);

        std::vector<float> kv_b(kvb_out);
        if (!mlx_gpu_matvec_probe(nkvb, 0, normed_kv.data(), kv_b.data()))
            return 0;

        // rope_mscale scaling on q_pe (per head) and k_pe -- plain scalar multiply,
        // identical arithmetic whether done here or on GPU; kept on host since it's
        // O(H*ROPE) and not the part this gate is actually verifying (the rotation is).
        for (int hh = 0; hh < H; hh++) {
            float *q_pe = q.data() + hh * QHD + NOPE;
            for (int i = 0; i < ROPE; i++) q_pe[i] *= (float)g_mla_rope_mscale;
        }
        for (int i = 0; i < ROPE; i++) k_pe[i] *= (float)g_mla_rope_mscale;

        // RoPE rotation, GPU, batched across all H heads in one call for q_pe (shape
        // {H,1,ROPE}, offset=pos applied uniformly across the H "batch" entries) and
        // one call for k_pe (shape {1,1,ROPE}, head-independent per MLA's design).
        std::vector<float> q_pe_stage((size_t)H * ROPE);
        for (int hh = 0; hh < H; hh++)
            std::memcpy(q_pe_stage.data() + (size_t)hh * ROPE, q.data() + hh * QHD + NOPE,
                        sizeof(float) * ROPE);
        mx::array freqs_arr((void *)g_mla_yarn_freqs.data(), {(int)g_mla_yarn_freqs.size()},
                             mx::float32, noop_deleter);
        // freqs_arr must be float32 -- g_mla_yarn_freqs is double (matches
        // moe_init_yarn()'s table). Convert once here rather than storing a second
        // float32 copy in config -- this path isn't the throughput-sensitive one yet.
        std::vector<float> freqs_f(g_mla_yarn_freqs.begin(), g_mla_yarn_freqs.end());
        mx::array freqs_f32((void *)freqs_f.data(), {(int)freqs_f.size()}, mx::float32, noop_deleter);

        mx::array q_pe_in((void *)q_pe_stage.data(), {H, 1, ROPE}, mx::float32, noop_deleter);
        mx::array q_pe_out = mx::fast::rope(q_pe_in, ROPE, /*traditional=*/true,
                                             /*base=*/std::nullopt, /*scale=*/1.0f,
                                             /*offset=*/pos, /*freqs=*/freqs_f32);
        mx::array k_pe_in((void *)k_pe, {1, 1, ROPE}, mx::float32, noop_deleter);
        mx::array k_pe_out = mx::fast::rope(k_pe_in, ROPE, /*traditional=*/true,
                                             /*base=*/std::nullopt, /*scale=*/1.0f,
                                             /*offset=*/pos, /*freqs=*/freqs_f32);
        mx::eval(q_pe_out);
        mx::eval(k_pe_out);
        const float *q_pe_rot = q_pe_out.data<float>();
        const float *k_pe_rot = k_pe_out.data<float>();
        for (int hh = 0; hh < H; hh++)
            std::memcpy(q.data() + hh * QHD + NOPE, q_pe_rot + (size_t)hh * ROPE,
                        sizeof(float) * ROPE);
        std::memcpy(k_pe, k_pe_rot, sizeof(float) * ROPE);

        // Write this position's K/V into the layer-0 GPU-arm cache (own storage,
        // D-gpu-2 -- not shared with qwen_infer.c's CPU-side g_moe_K/V).
        for (int hh = 0; hh < H; hh++) {
            const float *k_nope = kv_b.data() + hh * (NOPE + VHD);
            const float *v_h    = kv_b.data() + hh * (NOPE + VHD) + NOPE;
            float *kdst = g_mla_K.data() + (((size_t)l * H + hh) * MLA_L0_MAXPOS + pos) * QHD;
            float *vdst = g_mla_V.data() + (((size_t)l * H + hh) * MLA_L0_MAXPOS + pos) * VHD;
            std::memcpy(kdst, k_nope, sizeof(float) * NOPE);
            std::memcpy(kdst + NOPE, k_pe, sizeof(float) * ROPE);
            std::memcpy(vdst, v_h, sizeof(float) * VHD);
        }

        // sdpa over positions 0..pos -- stage a contiguous [H,pos+1,dim] buffer per
        // side (the per-head cache rows aren't contiguous across heads for pos+1 <
        // MLA_L0_MAXPOS), scale=g_mla_attn_scale, no mask (the K/V we pass is already
        // trimmed to exactly the valid history, verified equivalent to a causal mask
        // for this single-new-query-position case by isolated probe).
        int kv_len = pos + 1;
        std::vector<float> k_stage((size_t)H * kv_len * QHD), v_stage((size_t)H * kv_len * VHD);
        for (int hh = 0; hh < H; hh++) {
            std::memcpy(k_stage.data() + (size_t)hh * kv_len * QHD,
                        g_mla_K.data() + ((size_t)l * H + hh) * MLA_L0_MAXPOS * QHD,
                        sizeof(float) * kv_len * QHD);
            std::memcpy(v_stage.data() + (size_t)hh * kv_len * VHD,
                        g_mla_V.data() + ((size_t)l * H + hh) * MLA_L0_MAXPOS * VHD,
                        sizeof(float) * kv_len * VHD);
        }
        mx::array qa((void *)q.data(), {1, H, 1, QHD}, mx::float32, noop_deleter);
        mx::array ka((void *)k_stage.data(), {1, H, kv_len, QHD}, mx::float32, noop_deleter);
        mx::array va((void *)v_stage.data(), {1, H, kv_len, VHD}, mx::float32, noop_deleter);
        mx::array attn = mx::fast::scaled_dot_product_attention(
            qa, ka, va, (float)g_mla_attn_scale, "");
        mx::eval(attn);
        std::vector<float> attn_out((size_t)H * VHD);
        std::memcpy(attn_out.data(), attn.data<float>(), sizeof(float) * H * VHD);

        (void)attn_out_dim;
        if (!mlx_gpu_matvec_probe(no, 0, attn_out.data(), o_out))
            return 0;
        (void)hidden;
        return 1;
    } catch (...) {
        return 0;
    }
}

// V5b's original entry point, unchanged behavior -- l=0 through the generalized impl above.
int mlx_gpu_mla_layer0(const float *h, int pos, const float *kv_a_ln_w, float *o_out) {
    return mlx_gpu_mla_layer_impl(0, h, pos, kv_a_ln_w, o_out);
}



// ---------------------------------------------------------------------------
// V5c: full transformer block for one layer -- attention + FFN (dense or
// routed+shared), mirroring moe_forward_token()'s per-layer loop body
// (qwen_infer.c) exactly. The router (gate_w matvec + softmax + top-k) runs
// on the host: it's a tiny n_experts-wide op (64x2048 for this fixture), no
// accuracy or performance reason to push it through MLX -- a legitimate
// per-role CPU/GPU split, not a shortcut (the same D-gpu reasoning the user's
// own long-term per-role-device-dispatch goal is built on). gather_qmm
// (verified in isolation: max_abs_diff 4.77e-07 against a hand-decoded int4
// reference at group_size=64/bits=4, the real fixture's config) handles the
// routed switch_gate/switch_up GEMMs in one dispatch each (same h2 input,
// different rhs per selected expert); switch_down loops mlx_gpu_matvec_probe
// per selected expert (each takes a DIFFERENT swiglu'd activation, not a
// shared one, so it doesn't fit gather_qmm's single-shared-x calling
// convention the same way) -- reuses the already Gate-4-verified single-
// expert path rather than risking a new, unverified gather_qmm usage pattern
// under this round's time budget.
static int g_layer_hidden = 0, g_layer_im_dim = 0, g_layer_dense_im = 0,
           g_layer_n_experts = 0, g_layer_n_shared = 0, g_layer_top_k = 0, g_layer_group = 64;

int mlx_gpu_layer_config(int hidden, int im_dim, int dense_im, int n_experts, int n_shared,
                          int top_k, int group_size) {
    if (!mlx_gpu_available()) return 0;
    g_layer_hidden = hidden; g_layer_im_dim = im_dim; g_layer_dense_im = dense_im;
    g_layer_n_experts = n_experts; g_layer_n_shared = n_shared; g_layer_top_k = top_k;
    g_layer_group = group_size;
    return 1;
}

static void host_rmsnorm(const float *x, const float *g, float *y, int n, double eps) {
    double ss = 0.0; for (int i = 0; i < n; i++) ss += (double)x[i]*x[i];
    double inv = 1.0 / sqrt(ss/n + eps);
    for (int i = 0; i < n; i++) y[i] = (float)(x[i]*inv*g[i]);
}
static void host_softmax(float *x, int n) {
    float mx_v = x[0]; for (int i=1;i<n;i++) if (x[i]>mx_v) mx_v=x[i];
    double sum = 0.0;
    for (int i=0;i<n;i++) { x[i] = expf(x[i]-mx_v); sum += x[i]; }
    for (int i=0;i<n;i++) x[i] = (float)(x[i]/sum);
}
// Order-invariant repeated-max top-k -- same algorithm as qwen_infer.c's own
// moe_top_k_select() (duplicated across the D-gpu-1 vendor boundary, not shared).
static void host_top_k_select(const float *scores, int n, int k, int *out_idx) {
    std::vector<int> used(n, 0);
    for (int i = 0; i < k; i++) {
        int best = -1; float bestv = -1e30f;
        for (int j = 0; j < n; j++) if (!used[j] && scores[j] > bestv) { bestv = scores[j]; best = j; }
        used[best] = 1; out_idx[i] = best;
    }
}
static void host_swiglu(float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        gate[i] = silu * up[i];
    }
}

// gather_qmm for one tensor, one token (x shape {1,in}), TOPK selected experts sharing the
// SAME x -- writes out[TOPK*out_dim] contiguous (verified layout: gather_qmm's output for
// x={1,in}, rhs_indices={1,TOPK} is (1,TOPK,1,out_dim), which is exactly TOPK*out_dim
// contiguous floats).
static int gather_qmm_probe(const char *name, const float *x, const int *top_idx, int top_k,
                             float *out) {
    auto it = g_tensors.find(name);
    if (it == g_tensors.end()) return 0;
    QTensor &t = it->second;
    try {
        std::vector<int32_t> idx32(top_idx, top_idx + top_k);
        mx::array xin((void *)x, {1, (int)t.in}, mx::float32, noop_deleter);
        mx::array idxa(idx32.data(), {1, top_k}, mx::int32, noop_deleter);
        mx::array out_arr = mx::gather_qmm(xin, t.w, t.scales, t.biases, std::nullopt, idxa,
                                            /*transpose=*/true, g_layer_group, 4, "affine", false);
        mx::eval(out_arr);
        std::memcpy(out, out_arr.data<float>(), sizeof(float) * (size_t)top_k * t.out);
        return 1;
    } catch (...) {
        return 0;
    }
}

int mlx_gpu_layer_step(int l, int pos, int is_dense,
                        const float *x_in, const float *w_inln, const float *w_postln,
                        const float *w_kvaln, const float *w_gate,
                        float *x_out, int *out_top_idx, float *out_top_wgt) {
    if (g_layer_hidden == 0) return 0;   // mlx_gpu_layer_config() not called
    const int HIDDEN = g_layer_hidden, IM = g_layer_im_dim, DENSE_IM = g_layer_dense_im,
              NE = g_layer_n_experts, NS = g_layer_n_shared, TOPK = g_layer_top_k;
    static const double RMS_EPS = 1e-6;   // matches this fixture's MOE_RMS_EPS default (F-config)

    std::vector<float> h(HIDDEN), x_mid(HIDDEN), h2(HIDDEN), o_attn(HIDDEN), mlp_out(HIDDEN, 0.0f);
    host_rmsnorm(x_in, w_inln, h.data(), HIDDEN, RMS_EPS);
    if (!mlx_gpu_mla_layer_impl(l, h.data(), pos, w_kvaln, o_attn.data())) return 0;
    for (int c = 0; c < HIDDEN; c++) x_mid[c] = x_in[c] + o_attn[c];
    host_rmsnorm(x_mid.data(), w_postln, h2.data(), HIDDEN, RMS_EPS);

    char nm[128];
    if (is_dense) {
        std::vector<float> gate_v(DENSE_IM), up_v(DENSE_IM);
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.gate_proj", l);
        if (!mlx_gpu_matvec_probe(nm, 0, h2.data(), gate_v.data())) return 0;
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.up_proj", l);
        if (!mlx_gpu_matvec_probe(nm, 0, h2.data(), up_v.data())) return 0;
        host_swiglu(gate_v.data(), up_v.data(), DENSE_IM);
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.down_proj", l);
        if (!mlx_gpu_matvec_probe(nm, 0, gate_v.data(), mlp_out.data())) return 0;
    } else {
        std::vector<float> router_scores(NE);
        for (int r = 0; r < NE; r++) {
            double acc = 0.0;
            for (int c = 0; c < HIDDEN; c++) acc += (double)w_gate[(long)r*HIDDEN+c] * h2[c];
            router_scores[r] = (float)acc;
        }
        host_softmax(router_scores.data(), NE);
        std::vector<int> top_idx(TOPK);
        host_top_k_select(router_scores.data(), NE, TOPK, top_idx.data());
        // DeepSeek-V2-Lite: MOE_NORM_TOPK_PROB=0, no renorm of the selected scores (F-config) --
        // this gate is scoped to that fixture, matching qwen_infer.c's own golden-path default.

        std::vector<float> gate_all((size_t)TOPK*IM), up_all((size_t)TOPK*IM);
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.switch_mlp.gate_proj", l);
        if (!gather_qmm_probe(nm, h2.data(), top_idx.data(), TOPK, gate_all.data())) return 0;
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.switch_mlp.up_proj", l);
        if (!gather_qmm_probe(nm, h2.data(), top_idx.data(), TOPK, up_all.data())) return 0;

        std::vector<float> down_v((size_t)TOPK*HIDDEN);
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.switch_mlp.down_proj", l);
        for (int k = 0; k < TOPK; k++) {
            host_swiglu(gate_all.data() + (size_t)k*IM, up_all.data() + (size_t)k*IM, IM);
            if (!mlx_gpu_matvec_probe(nm, top_idx[k], gate_all.data() + (size_t)k*IM,
                                       down_v.data() + (size_t)k*HIDDEN)) return 0;
        }
        for (int k = 0; k < TOPK; k++) {
            float wgt = router_scores[top_idx[k]];
            for (int c = 0; c < HIDDEN; c++) mlp_out[c] += wgt * down_v[(size_t)k*HIDDEN+c];
            if (out_top_idx) out_top_idx[k] = top_idx[k];
            if (out_top_wgt) out_top_wgt[k] = wgt;
        }
        if (NS > 0) {
            std::vector<float> sgate(IM*NS), sup(IM*NS), sdown(HIDDEN);
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.shared_experts.gate_proj", l);
            if (!mlx_gpu_matvec_probe(nm, 0, h2.data(), sgate.data())) return 0;
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.shared_experts.up_proj", l);
            if (!mlx_gpu_matvec_probe(nm, 0, h2.data(), sup.data())) return 0;
            host_swiglu(sgate.data(), sup.data(), IM*NS);
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.shared_experts.down_proj", l);
            if (!mlx_gpu_matvec_probe(nm, 0, sgate.data(), sdown.data())) return 0;
            for (int c = 0; c < HIDDEN; c++) mlp_out[c] += sdown[c];
        }
    }
    for (int c = 0; c < HIDDEN; c++) x_out[c] = x_mid[c] + mlp_out[c];
    return 1;
}

int mlx_gpu_layer_step_dbg(int l, int pos, int is_dense,
                        const float *x_in, const float *w_inln, const float *w_postln,
                        const float *w_kvaln, const float *w_gate,
                        float *x_out, int *out_top_idx, float *out_top_wgt,
                            float *dbg_xmid_out, float *dbg_routed_out) {
    if (g_layer_hidden == 0) return 0;   // mlx_gpu_layer_config() not called
    const int HIDDEN = g_layer_hidden, IM = g_layer_im_dim, DENSE_IM = g_layer_dense_im,
              NE = g_layer_n_experts, NS = g_layer_n_shared, TOPK = g_layer_top_k;
    static const double RMS_EPS = 1e-6;   // matches this fixture's MOE_RMS_EPS default (F-config)

    std::vector<float> h(HIDDEN), x_mid(HIDDEN), h2(HIDDEN), o_attn(HIDDEN), mlp_out(HIDDEN, 0.0f);
    host_rmsnorm(x_in, w_inln, h.data(), HIDDEN, RMS_EPS);
    if (!mlx_gpu_mla_layer_impl(l, h.data(), pos, w_kvaln, o_attn.data())) return 0;
    for (int c = 0; c < HIDDEN; c++) x_mid[c] = x_in[c] + o_attn[c];
    host_rmsnorm(x_mid.data(), w_postln, h2.data(), HIDDEN, RMS_EPS);

    char nm[128];
    if (is_dense) {
        std::vector<float> gate_v(DENSE_IM), up_v(DENSE_IM);
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.gate_proj", l);
        if (!mlx_gpu_matvec_probe(nm, 0, h2.data(), gate_v.data())) return 0;
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.up_proj", l);
        if (!mlx_gpu_matvec_probe(nm, 0, h2.data(), up_v.data())) return 0;
        host_swiglu(gate_v.data(), up_v.data(), DENSE_IM);
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.down_proj", l);
        if (!mlx_gpu_matvec_probe(nm, 0, gate_v.data(), mlp_out.data())) return 0;
    } else {
        std::vector<float> router_scores(NE);
        for (int r = 0; r < NE; r++) {
            double acc = 0.0;
            for (int c = 0; c < HIDDEN; c++) acc += (double)w_gate[(long)r*HIDDEN+c] * h2[c];
            router_scores[r] = (float)acc;
        }
        host_softmax(router_scores.data(), NE);
        std::vector<int> top_idx(TOPK);
        host_top_k_select(router_scores.data(), NE, TOPK, top_idx.data());
        // DeepSeek-V2-Lite: MOE_NORM_TOPK_PROB=0, no renorm of the selected scores (F-config) --
        // this gate is scoped to that fixture, matching qwen_infer.c's own golden-path default.

        std::vector<float> gate_all((size_t)TOPK*IM), up_all((size_t)TOPK*IM);
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.switch_mlp.gate_proj", l);
        if (!gather_qmm_probe(nm, h2.data(), top_idx.data(), TOPK, gate_all.data())) return 0;
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.switch_mlp.up_proj", l);
        if (!gather_qmm_probe(nm, h2.data(), top_idx.data(), TOPK, up_all.data())) return 0;

        std::vector<float> down_v((size_t)TOPK*HIDDEN);
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.switch_mlp.down_proj", l);
        for (int k = 0; k < TOPK; k++) {
            host_swiglu(gate_all.data() + (size_t)k*IM, up_all.data() + (size_t)k*IM, IM);
            if (!mlx_gpu_matvec_probe(nm, top_idx[k], gate_all.data() + (size_t)k*IM,
                                       down_v.data() + (size_t)k*HIDDEN)) return 0;
        }
        for (int k = 0; k < TOPK; k++) {
            float wgt = router_scores[top_idx[k]];
            for (int c = 0; c < HIDDEN; c++) mlp_out[c] += wgt * down_v[(size_t)k*HIDDEN+c];
            if (out_top_idx) out_top_idx[k] = top_idx[k];
            if (out_top_wgt) out_top_wgt[k] = wgt;
        }
        if (NS > 0) {
            std::vector<float> sgate(IM*NS), sup(IM*NS), sdown(HIDDEN);
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.shared_experts.gate_proj", l);
            if (!mlx_gpu_matvec_probe(nm, 0, h2.data(), sgate.data())) return 0;
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.shared_experts.up_proj", l);
            if (!mlx_gpu_matvec_probe(nm, 0, h2.data(), sup.data())) return 0;
            host_swiglu(sgate.data(), sup.data(), IM*NS);
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.shared_experts.down_proj", l);
            if (!mlx_gpu_matvec_probe(nm, 0, sgate.data(), sdown.data())) return 0;
            for (int c = 0; c < HIDDEN; c++) mlp_out[c] += sdown[c];
        }
    }
    if (dbg_xmid_out) std::memcpy(dbg_xmid_out, x_mid.data(), sizeof(float) * HIDDEN);
    if (dbg_routed_out) std::memcpy(dbg_routed_out, mlp_out.data(), sizeof(float) * HIDDEN);
    for (int c = 0; c < HIDDEN; c++) x_out[c] = x_mid[c] + mlp_out[c];
    return 1;
}
// ---------------------------------------------------------------------------
// V5c-fused: one-eval-per-LAYER rewrite. Rebuilds each layer's K/V-history
// handling as a device-side FIXED-SHAPE window (the same principle production
// serving engines use to make GPU decode CUDA-graph-capturable: CUDA Graph
// capture/replay itself requires static shapes+addresses; FreeToken's own MoE
// decode path -- python/freetoken/layers/moe.py -- is explicitly commented
// "device-side with fixed shapes... capture-safe" for the same reason)
// instead of a per-position growing/concatenated array. This replaced an
// earlier version whose K/V history array literally changed shape every
// position (`{1,H,pos,QHD}` concatenated fresh each call) and read from a
// brand-new host allocation each time -- exactly the two properties CUDA
// Graph capture (and, it turned out, this bug) cannot tolerate.
//
// Three real, reproducible bugs were found and fixed getting a correct,
// reproducible one-eval-per-layer design (full history in git log; the first
// two predate this fixed-shape redesign):
//
// Bug 1 (per-TOKEN granularity): building the ENTIRE 27-layer graph lazily
// and calling mx::eval() once at the very end was correct at pos=0, wrong at
// pos>=1, for reasons never isolated -- worked around by evaluating once per
// LAYER instead.
//
// Bug 2 (gather_qmm cross-product + take_along_axis, in the routed-FFN's
// switch_down step): silently returned corrupted data when its eval was
// deferred into a larger combined mx::eval() batch. Originally worked around
// by evaluating it standalone, immediately after computing it -- this cost
// one extra eval() per routed layer. Later (throughput-optimization round)
// ELIMINATED entirely, not just worked around: mlx_lm's own reference
// SwitchLinear/QuantizedSwitchLinear (switch_layers.py) never passes
// lhs_indices for this per-row-distinct-expert pattern in the first place --
// switching to that same lhs_indices=nullopt form (x reshaped {TOPK,1,IM})
// avoids the cross-product entirely, confirmed correct AND safe under
// deferred eval by an isolated probe (probe_down_no_lhs.cpp) before being
// applied to this function, so the standalone eval is no longer needed.
// (No residual/error-feedback compensation applies -- same D18 exemption as
// this file's header: pure dequant-consumption, not requantization.)
//
// Bug 3 (this redesign's own target, ROOT-CAUSED -- not an MLX bug): the
// original per-layer design's `v_new` was built as a BARE `mx::slice()` of
// `kv_b_r` with no follow-up op to force materialization (unlike `k_new`,
// which flows into `mx::concatenate()` afterward and gets compacted as a
// side effect). Confirmed via isolated repro: `mx::eval()` on a bare slice
// does NOT compact it -- `.flags().row_contiguous` stays false, and
// `.data<float>()` returns a pointer into the ORIGINAL parent buffer with the
// slice's true (non-dense) strides. Reading it via naive per-head `hh*VHD`
// indexing (exactly what persisting it into the host K/V cache requires)
// silently interleaves v_new's real data with neighboring `k_nope` bytes --
// this produced a reproducible "every other head correct" corruption
// pattern, confirmed byte-for-byte against a synthetic isolated repro with
// the model's real dimensions (H=16, QHD=192, VHD=128). `mx::copy()` was
// tried first and did NOT fix it (MLX's optimizer elides the copy since the
// strided view is "valid" from its own internal perspective); `mx::contiguous()`
// is the real, confirmed fix -- verified in isolation (row_contiguous flips
// to true, strides become dense, naive reads match exactly) and in the real
// pipeline (layer-0/pos-0 x_mid went from rel-L2 1.39 to 2.96e-07 with this
// one-line fix, no other change).
static mx::array *g_fused_x = nullptr;   // pending residual stream, always ALREADY evaluated between calls
static int g_fused_pos = -1;
static int g_fused_layers_done = 0;

// Throughput round: fused path's OWN persistent K/V cache, as real (lazy-updatable)
// mx::array objects rather than a raw host float buffer (g_mla_K/g_mla_V above, which
// stay untouched -- still used by the eager mlx_gpu_mla_layer_impl()/mlx_gpu_layer_step()
// paths). Storing K/V as device arrays and updating them via mx::slice_update() lets Stage
// A's write and Stage B's windowed read live in the SAME lazy graph, with no forced
// standalone eval() between them -- eliminating the eval() this file's Stage A/B split used
// to require every layer. Verified safe by an isolated probe (probe_slice_update_chain.py)
// before being applied here: chained slice_update() calls, all deferred, evaluated once at
// the end, reproduce a host-buffer reference bit-for-bit (max_abs_diff=0.0), and a masked
// scaled_dot_product_attention read over the still-lazy chained window matches a numpy
// reference to float32 rounding precision (1.19e-07) -- the same class of "does a bare
// slice-derived array stay correct under deferred eval" question Bug 3 raised, now answered
// affirmatively for slice_update specifically (as opposed to bare mx::slice(), which needed
// mx::contiguous() -- slice_update's output is not subject to the same bug, confirmed by
// this probe, not assumed from the analogy).
//
// V5d rescope: gained a leading BATCH dimension B (mlx_gpu_set_batch()). B tokens are
// lockstep -- every sequence in the batch shares the SAME position `pos` every call, which
// is why the attention mask below can stay a single shared {1,1,1,MLA_L0_MAXPOS} array
// (broadcasts across B) instead of needing its own per-sequence copy; this assumption does
// NOT hold for a future ragged (V5e) design and must be revisited there, not reused.
// Verified safe for B>1 by two isolated probes before this generalization was applied:
// probe_batched_slice_update.py (chained slice_update + masked sdpa read, both with a
// leading B axis, still bit-exact/1.79e-07 vs a numpy reference) and
// probe_batched_router_ffn.py (the full router+gather_qmm+switch_down composition at
// B=8, matched a per-row B=1-style loop to 1.4e-06 AND selected identical experts --
// this caught a real gap: gather_qmm with lhs_indices=nullopt does NOT automatically pair
// x's own leading batch dim with rhs_indices' batch dim unless x is first expanded via
// mx::expand_dims(x, {-2,-3}), matching mlx_lm's own SwitchLinear convention exactly --
// omitting that expand silently computes a B x TOPK CROSS PRODUCT instead of the intended
// per-token selection, confirmed via the probe's own shape printout before the fix).
static std::vector<mx::array> g_fused_K;
static std::vector<mx::array> g_fused_V;
static bool g_fused_kv_inited = false;
static int g_fused_kv_inited_B = 0;   // B the current g_fused_K/V were sized for
static int g_fused_B = 1;             // current batch size; mlx_gpu_layer_step_lazy()'s
                                       // x_in_host is B*HIDDEN floats, one row per sequence

// V5d: set the batch size for subsequent mlx_gpu_layer_step_lazy()/mlx_gpu_forward_finalize()
// calls. B=1 (the default) reproduces the exact single-token behavior this file shipped with
// before V5d -- callers that never call this function get byte-identical B=1 behavior.
// Bounded to MOE_BATCH_MAX=64 to match the CPU/SME2 arm's own ceiling (qwen_infer.c).
int mlx_gpu_set_batch(int B) {
    if (!mlx_gpu_available()) return 0;
    if (B < 1 || B > 64) return 0;
    g_fused_B = B;
    return 1;
}

// D-sort-1: adopt mlx_lm's own global flatten+argsort+gather (_gather_sort/_scatter_unsort,
// switch_layers.py) for the routed-FFN's gate/up/down gather_qmm calls, gated on a runtime
// B*TOPK threshold -- NOT mx.gather_qmm's own naive `indices.size >= 64` default, which F-4
// (this project's plan doc) found gives only 2.57x of the real 7.02x available at B=64
// (per-row sort vs a genuine GLOBAL sort). Verified correct via an isolated probe
// (probe_sorted_gather.py, real NE=64/HIDDEN=2048/IM=1408/TOPK=6 dims, B=64) before being
// applied here: sorted-path routed_sum matched the already-shipped unsorted path to 1.0e-05
// (fp32 accumulation noise, not a discrepancy).
//   WHY : the naive/default threshold is wrong for this workload (F-4 measured it costs
//         throughput below B~=16 and underdelivers above it); a runtime-tunable threshold
//         lets the real crossover be measured IN the actual 27-layer streaming pass
//         (mlx_gpu_set_sort_threshold()), not assumed from an isolated, partially
//         cache-warm microbenchmark (F-4's own stated caveat about its own prior number).
//   COST: two extra argsort() + two extra take()-based gathers per routed layer when the
//         sorted branch is taken -- only pays off above the real measured crossover, which
//         is why this stays a runtime branch, not an unconditional replacement.
//   EXIT: mlx_gpu_set_sort_threshold(INT32_MAX) (or never calling it -- this is the default)
//         disables the sorted branch entirely, reverting byte-for-byte to the unsorted code
//         path this file shipped the KILL-GATE with.
static int g_gpu_sort_threshold = 2147483647;   // default: OFF, matches pre-F-4 shipped behavior

int mlx_gpu_set_sort_threshold(int threshold) {
    if (!mlx_gpu_available()) return 0;
    g_gpu_sort_threshold = threshold;
    return 1;
}

static void ensure_fused_kv_init() {
    if (g_fused_kv_inited && g_fused_kv_inited_B == g_fused_B) return;
    const int H = g_mla_n_heads, QHD = g_mla_q_head_dim, VHD = g_mla_v_hd, B = g_fused_B;
    g_fused_K.clear();
    g_fused_V.clear();
    std::vector<mx::array> all;
    for (int l = 0; l < MLA_MAXLAYERS; l++) {
        g_fused_K.push_back(mx::zeros({B, H, MLA_L0_MAXPOS, QHD}, mx::float32));
        g_fused_V.push_back(mx::zeros({B, H, MLA_L0_MAXPOS, VHD}, mx::float32));
        all.push_back(g_fused_K.back());
        all.push_back(g_fused_V.back());
    }
    mx::eval(all);   // one-time materialization at first use, not part of the per-token cost
    g_fused_kv_inited = true;
    g_fused_kv_inited_B = B;
}

static mx::array wrap_host_f32(const float *p, std::initializer_list<int> shape) {
    return mx::array((void *)p, mx::Shape(shape), mx::float32, noop_deleter);
}
static mx::array lazy_silu(const mx::array &x) { return mx::multiply(x, mx::sigmoid(x)); }

// Same shape as mlx_gpu_matvec_probe()'s quantized_matmul call, but returns the
// (unevaluated) array instead of eval()'ing + copying to host. x may carry a leading
// batch dim (B or B*TOPK); this single weight matrix (expert 0, shared/dense role)
// naturally batches via quantized_matmul's own broadcasting -- no expand_dims needed
// here, only for gather_qmm's per-expert-selection calls below.
static mx::array lazy_matvec_e0(const char *name, const mx::array &x) {
    QTensor &t = g_tensors.at(name);
    mx::array w_e = mx::take(t.w, 0, 0);
    mx::array s_e = mx::take(t.scales, 0, 0);
    mx::array b_e = mx::take(t.biases, 0, 0);
    return mx::quantized_matmul(x, w_e, s_e, b_e, /*transpose=*/true, g_layer_group, 4);
}

int mlx_gpu_layer_step_lazy(int l, int pos, int is_dense,
                             const float *x_in_host, const float *w_inln, const float *w_postln,
                             const float *w_kvaln, const float *w_gate) {
    if (g_mla_n_heads == 0 || g_layer_hidden == 0) return 0;
    if (pos < 0 || pos >= MLA_L0_MAXPOS || l < 0 || l >= MLA_MAXLAYERS) return 0;
    try {
        const int H = g_mla_n_heads, QHD = g_mla_q_head_dim, NOPE = g_mla_qk_nope,
                  ROPE = g_mla_qk_rope, VHD = g_mla_v_hd, KVLORA = g_mla_kv_lora;
        const int HIDDEN = g_layer_hidden, IM = g_layer_im_dim, DENSE_IM = g_layer_dense_im,
                  NE = g_layer_n_experts, NS = g_layer_n_shared, TOPK = g_layer_top_k;
        const int B = g_fused_B;

        ensure_fused_kv_init();
        if (l == 0) {
            delete g_fused_x;
            g_fused_x = new mx::array(wrap_host_f32(x_in_host, {B, HIDDEN}));
            g_fused_pos = pos; g_fused_layers_done = 0;
        }
        if (g_fused_pos != pos || g_fused_layers_done != l) return 0;

        char nq[96], nkva[96], nkvb[96], no[96];
        snprintf(nq, sizeof nq, "model.layers.%d.self_attn.q_proj", l);
        snprintf(nkva, sizeof nkva, "model.layers.%d.self_attn.kv_a_proj_with_mqa", l);
        snprintf(nkvb, sizeof nkvb, "model.layers.%d.self_attn.kv_b_proj", l);
        snprintf(no, sizeof no, "model.layers.%d.self_attn.o_proj", l);

        mx::array x = *g_fused_x;
        mx::array h = mx::fast::rms_norm(x, wrap_host_f32(w_inln, {HIDDEN}), (float)g_mla_rms_eps);

        // ---- Stage A: this position's OWN K/V (shape depends only on B/H/QHD/
        // VHD -- constants -- never on pos or on history). Eval + persist
        // immediately, before the fixed-shape attention window is built.
        mx::array q = lazy_matvec_e0(nq, h);
        mx::array kv_ap = lazy_matvec_e0(nkva, h);
        mx::array compressed_kv = mx::slice(kv_ap, {0, 0}, {B, KVLORA});
        mx::array k_pe_raw = mx::slice(kv_ap, {0, KVLORA}, {B, KVLORA + ROPE});
        mx::array normed_kv = mx::fast::rms_norm(compressed_kv, wrap_host_f32(w_kvaln, {KVLORA}),
                                                  (float)g_mla_rms_eps);
        mx::array kv_b = lazy_matvec_e0(nkvb, normed_kv);

        mx::array q_r = mx::reshape(q, {B, H, QHD});
        mx::array q_nope = mx::slice(q_r, {0, 0, 0}, {B, H, NOPE});
        mx::array q_pe_raw = mx::slice(q_r, {0, 0, NOPE}, {B, H, NOPE + ROPE});
        mx::array q_pe_scaled = q_pe_raw * (float)g_mla_rope_mscale;
        mx::array k_pe_scaled = k_pe_raw * (float)g_mla_rope_mscale;

        // Bug 1 fix: wrap the PERSISTENT global, not a function-local std::vector.
        mx::array freqs_f32((void *)g_mla_yarn_freqs_f32.data(),
                             {(int)g_mla_yarn_freqs_f32.size()}, mx::float32, noop_deleter);

        // RoPE's offset=pos is a single scalar, applied identically over every leading
        // "batch-like" axis regardless of how many precede the rotated axis (already true
        // of the B=1 design's own H leading axis; adding B ahead of it changes nothing
        // about how the scalar offset broadcasts).
        mx::array q_pe_hbo = mx::transpose(q_pe_scaled, {1, 0, 2});   // {H,B,ROPE}
        mx::array q_pe_rot_hbo = mx::fast::rope(q_pe_hbo, ROPE, /*traditional=*/true,
                                                 /*base=*/std::nullopt, /*scale=*/1.0f,
                                                 /*offset=*/pos, /*freqs=*/freqs_f32);
        mx::array q_pe_rot = mx::transpose(q_pe_rot_hbo, {1, 0, 2});   // {B,H,ROPE}
        mx::array k_pe_in = mx::reshape(k_pe_scaled, {B, 1, ROPE});
        mx::array k_pe_rot = mx::fast::rope(k_pe_in, ROPE, true, std::nullopt, 1.0f, pos, freqs_f32);

        mx::array q_full = mx::concatenate({q_nope, q_pe_rot}, -1);
        mx::array kv_b_r = mx::reshape(kv_b, {B, H, NOPE + VHD});
        mx::array k_nope = mx::slice(kv_b_r, {0, 0, 0}, {B, H, NOPE});
        // v_new: mx::contiguous() forces genuine materialization of this bare slice --
        // see the Bug 3 header comment above for why this is required (k_nope avoids
        // the same issue only because it flows into concatenate() below, which
        // compacts as a side effect; v_new has no such follow-up op).
        mx::array v_new = mx::contiguous(mx::slice(kv_b_r, {0, 0, NOPE}, {B, H, NOPE + VHD}));
        mx::array k_pe_bcast = mx::broadcast_to(k_pe_rot, {B, H, ROPE});
        mx::array k_new = mx::concatenate({k_nope, k_pe_bcast}, -1);

        // Persist this position's K/V into the FUSED path's own persistent device arrays
        // via slice_update -- NO eval() here (throughput round: this used to be a forced
        // standalone eval + host memcpy into g_mla_K/g_mla_V; verified via
        // probe_slice_update_chain.py, and its B>1 sibling probe_batched_slice_update.py,
        // that deferring this is safe with or without a batch axis). pos_start is built
        // with the VALUE-constructing array(T) ctor (copies at construction time), not a
        // host-pointer wrap, so it stays valid regardless of this function's stack lifetime.
        mx::array k_new_win = mx::reshape(k_new, {B, H, 1, QHD});
        mx::array v_new_win = mx::reshape(v_new, {B, H, 1, VHD});
        mx::array pos_start(pos, mx::int32);
        g_fused_K[l] = mx::slice_update(g_fused_K[l], k_new_win, pos_start, std::vector<int>{2});
        g_fused_V[l] = mx::slice_update(g_fused_V[l], v_new_win, pos_start, std::vector<int>{2});

        // ---- Stage B: fixed-shape attention. k_win/v_win wrap the FULL
        // per-layer window at a CONSTANT pointer and CONSTANT shape
        // {B,H,MLA_L0_MAXPOS,*} on every call, for every position -- the
        // window's per-head blocks are contiguous by construction, so no
        // copy is needed, just the SAME lazy arrays slice_update() just
        // produced above (no host wrap, no separate eval boundary between
        // the write (Stage A) and this windowed read (Stage B) -- both live
        // in one graph, evaluated together below). A boolean causal mask
        // (true for j<=pos, false otherwise) replaces "array length" as the
        // way of encoding how much of the fixed window is valid -- positions
        // beyond `pos` still hold stale/zero data but are masked out of the
        // softmax entirely. The mask stays a SINGLE shared {1,1,1,MLA_L0_MAXPOS}
        // array (not {B,1,1,MLA_L0_MAXPOS}) because V5d's B tokens are
        // lockstep at the same `pos` -- every sequence has identical valid
        // history length this call, so the mask broadcasts correctly across
        // B (verified: probe_batched_slice_update.py's masked-attention test
        // used this exact shared-mask/batched-window combination). This
        // assumption breaks for a future ragged (V5e) design, where different
        // slots can be at different positions simultaneously -- that design
        // will need a per-sequence mask, not this one.
        mx::array k_win = g_fused_K[l];
        mx::array v_win = g_fused_V[l];
        static bool s_mask_buf[MLA_L0_MAXPOS];   // static: fixed address across calls
        for (int j = 0; j < MLA_L0_MAXPOS; j++) s_mask_buf[j] = (j <= pos);
        mx::array mask_arr((void *)s_mask_buf, {1, 1, 1, MLA_L0_MAXPOS}, mx::bool_, noop_deleter);

        mx::array q_full_r = mx::reshape(q_full, {B, H, 1, QHD});
        // mask_mode MUST be "array" (not "") when passing an explicit mask_arr --
        // confirmed against libmlx.dylib's own validation strings ("mask_mode must
        // be 'causal', 'array' or ''"; passing an array with mode "" throws
        // "Invalid mask_arr for mask_mode"). The masking mechanism itself (boolean
        // mask, false beyond `pos`) was independently verified bit-identical to
        // truncating k/v to just the valid positions, both with synthetic data and
        // with this model's real dimensions (H=16, QHD=192, VHD=128) -- it was never
        // the source of Bug 3, despite substantial early suspicion.
        mx::array attn = mx::fast::scaled_dot_product_attention(q_full_r, k_win, v_win,
                                                                  (float)g_mla_attn_scale, "array", mask_arr);
        mx::array attn_flat = mx::reshape(attn, {B, H * VHD});
        mx::array o = lazy_matvec_e0(no, attn_flat);
        mx::array x_mid = x + o;

        mx::array h2 = mx::fast::rms_norm(x_mid, wrap_host_f32(w_postln, {HIDDEN}), (float)g_mla_rms_eps);

        std::optional<mx::array> mlp_out_opt;
        char nm[128];
        if (is_dense) {
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.gate_proj", l);
            mx::array gate_v = lazy_matvec_e0(nm, h2);
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.up_proj", l);
            mx::array up_v = lazy_matvec_e0(nm, h2);
            mx::array sw = lazy_silu(gate_v) * up_v;
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.down_proj", l);
            mlp_out_opt = lazy_matvec_e0(nm, sw);
            (void)DENSE_IM;
        } else {
            mx::array w_gate_arr = wrap_host_f32(w_gate, {NE, HIDDEN});
            mx::array scores_raw = mx::matmul(h2, mx::transpose(w_gate_arr));       // {B,NE}
            mx::array scores = mx::softmax(scores_raw, std::vector<int>{-1}, /*precise=*/true);
            mx::array order = mx::argsort(scores, -1);                              // {B,NE}, per-row
            mx::array top_idx_u = mx::slice(order, {0, NE - TOPK}, {B, NE});         // {B,TOPK}
            mx::array top_idx = mx::astype(top_idx_u, mx::int32);
            mx::array top_wgt = mx::take_along_axis(scores, top_idx, 1);             // {B,TOPK}

            snprintf(nm, sizeof nm, "model.layers.%d.mlp.switch_mlp.gate_proj", l);
            QTensor &tg = g_tensors.at(nm);
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.switch_mlp.up_proj", l);
            QTensor &tu = g_tensors.at(nm);
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.switch_mlp.down_proj", l);
            QTensor &td = g_tensors.at(nm);

            std::optional<mx::array> down_flat_opt;
            if ((long)B * TOPK >= g_gpu_sort_threshold) {
                // D-sort path (F-4): global flatten+argsort+gather, mlx_lm's own
                // _gather_sort/_scatter_unsort composition (switch_layers.py), verified via
                // probe_sorted_gather.py (B=64, real dims) against the unsorted path below
                // (max_abs_diff 1.0e-05, fp32 noise). x_flat3 == what expand_dims(h2,{-2,-3})
                // .flatten(0,-3) reduces to algebraically -- skipping the 4D roundtrip since
                // this branch does its own explicit mx::take() gather instead of relying on
                // gather_qmm's implicit lhs/rhs batch pairing (that pairing is what the
                // UNSORTED branch below needs expand_dims for; this branch doesn't use it).
                mx::array x_flat3 = mx::reshape(h2, {B, 1, HIDDEN});
                mx::array flat_idx = mx::reshape(top_idx, {B * TOPK});
                mx::array order = mx::argsort(flat_idx, 0);
                mx::array inv_order = mx::argsort(order, 0);
                mx::array topk_arr(TOPK, mx::int32);
                mx::array row_sel = mx::floor_divide(order, topk_arr);
                mx::array x_sorted = mx::take(x_flat3, row_sel, 0);      // {B*TOPK,1,HIDDEN}
                mx::array idx_sorted = mx::take(flat_idx, order);        // {B*TOPK}

                mx::array gate_all_s = mx::gather_qmm(x_sorted, tg.w, tg.scales, tg.biases, std::nullopt,
                                                       idx_sorted, true, g_layer_group, 4, "affine", true);
                mx::array up_all_s = mx::gather_qmm(x_sorted, tu.w, tu.scales, tu.biases, std::nullopt,
                                                     idx_sorted, true, g_layer_group, 4, "affine", true);
                mx::array swiglu_3d_s = mx::reshape(lazy_silu(gate_all_s) * up_all_s, {B * TOPK, 1, IM});
                mx::array down_all_s = mx::gather_qmm(swiglu_3d_s, td.w, td.scales, td.biases, std::nullopt,
                                                       idx_sorted, true, g_layer_group, 4, "affine", true);
                mx::array down_flat_s = mx::reshape(down_all_s, {B * TOPK, HIDDEN});
                mx::array down_unsorted = mx::take(down_flat_s, inv_order, 0);   // scatter-unsort
                down_flat_opt = mx::reshape(down_unsorted, {B, TOPK, HIDDEN});
            } else {
                // Unsorted path (unchanged from the KILL-GATE-passing build): gather_qmm's
                // shared-x/per-expert-selection form (gate/up) needs x to carry its OWN
                // leading {B,1,1,...} batch shape via expand_dims so gather_qmm PAIRS it
                // with rhs_indices' {B,TOPK} batch instead of cross-producting the two --
                // confirmed the hard way by probe_batched_router_ffn.py (B=8 without this
                // expand produced shape (8,6,8,1408), a genuine BxTOPKxB cross product, not
                // the intended (8,6,1,1408) per-(batch,expert) selection). This step did not
                // exist in the B=1 code because a bare {1,HIDDEN} x's absent batch dims
                // happened to look identical to correct pairing when there is only one
                // possible pairing -- it was never actually being exercised at B=1.
                mx::array h2_expanded = mx::expand_dims(h2, std::vector<int>{-2, -3});   // {B,1,1,HIDDEN}
                mx::array gate_all = mx::gather_qmm(h2_expanded, tg.w, tg.scales, tg.biases, std::nullopt,
                                                     top_idx, true, g_layer_group, 4, "affine", false);
                mx::array up_all = mx::gather_qmm(h2_expanded, tu.w, tu.scales, tu.biases, std::nullopt,
                                                   top_idx, true, g_layer_group, 4, "affine", false);
                // switch_down: B*TOPK rows, each already belonging to its OWN selected
                // expert (unlike gate/up above, which share one h2 row per batch entry across
                // its TOPK experts) -- flatten to {B*TOPK,1,IM} and omit lhs_indices entirely,
                // mirroring mlx_lm's own SwitchLinear/QuantizedSwitchLinear.__call__
                // (switch_layers.py), which never passes lhs_indices for this exact
                // per-row-distinct-expert case. Verified at B=8 by probe_batched_router_ffn.py
                // against a per-row B=1-style loop (max_abs_diff 1.4e-06, identical expert
                // selection) before being applied here.
                // (No residual/error-feedback compensation applies here either, same as this
                // file's D18 exemption in the header comment above -- this is pure dequant-
                // consumption of an already-quantized weight, not a new requantization.)
                mx::array swiglu_3d = mx::reshape(lazy_silu(gate_all) * up_all, {B * TOPK, 1, IM});
                mx::array top_idx_1d = mx::reshape(top_idx, {B * TOPK});
                mx::array down_all = mx::gather_qmm(swiglu_3d, td.w, td.scales, td.biases, std::nullopt,
                                                     top_idx_1d, true, g_layer_group, 4, "affine", false);
                down_flat_opt = mx::reshape(down_all, {B, TOPK, HIDDEN});
            }
            mx::array down_flat = *down_flat_opt;
            mx::array top_wgt_col = mx::reshape(top_wgt, {B, TOPK, 1});
            mx::array weighted = down_flat * top_wgt_col;
            mx::array routed_sum = mx::sum(weighted, std::vector<int>{1}, false);    // {B,HIDDEN}

            if (NS > 0) {
                snprintf(nm, sizeof nm, "model.layers.%d.mlp.shared_experts.gate_proj", l);
                mx::array sgate = lazy_matvec_e0(nm, h2);
                snprintf(nm, sizeof nm, "model.layers.%d.mlp.shared_experts.up_proj", l);
                mx::array sup = lazy_matvec_e0(nm, h2);
                mx::array sswiglu = lazy_silu(sgate) * sup;
                snprintf(nm, sizeof nm, "model.layers.%d.mlp.shared_experts.down_proj", l);
                mx::array sdown = lazy_matvec_e0(nm, sswiglu);
                mlp_out_opt = routed_sum + sdown;
            } else {
                mlp_out_opt = routed_sum;
            }
        }
        mx::array x_out = x_mid + *mlp_out_opt;
        // Bug 1 FOUND AND FIXED (see g_mla_yarn_freqs_f32's declaration comment above):
        // the earlier attempt at deferring this eval all the way to mlx_gpu_forward_
        // finalize() broke because of an unrelated use-after-free (freqs_f32 wrapping a
        // stack-local buffer), not because of anything wrong with deferring across layer
        // boundaries per se. With that fixed, x_out (and this layer's K/V update) can
        // safely stay lazy here -- evaluated once per BATCH STEP (B tokens together) in
        // finalize(), not once per layer. Verified at B=1: 8/8 argmax, bit-identical
        // gpu_vs_cpu to the old per-layer-eval design, KILL-GATE PASSES (~52.9 tok/s vs
        // the 48.34 bar).
        delete g_fused_x;
        g_fused_x = new mx::array(x_out);   // still LAZY -- NOT evaluated until finalize()
        g_fused_layers_done = l + 1;
        return 1;
    } catch (...) {
        delete g_fused_x; g_fused_x = nullptr;
        g_fused_pos = -1; g_fused_layers_done = 0;
        return 0;
    }
}

int mlx_gpu_forward_finalize(const float *w_finalnorm, float *logits_out) {
    if (!g_fused_x) return 0;
    try {
        const int HIDDEN = g_layer_hidden;
        const int B = g_fused_B;
        mx::array x_final = mx::fast::rms_norm(*g_fused_x, wrap_host_f32(w_finalnorm, {HIDDEN}),
                                                (float)g_mla_rms_eps);
        mx::array logits = lazy_matvec_e0("lm_head", x_final);   // {B,VOCAB}
        // Single eval for the WHOLE batch step: logits plus every layer's K/V update, all
        // still lazy from mlx_gpu_layer_step_lazy() above. Untouched g_fused_K/V slots are
        // already-evaluated arrays from a prior finalize() call, so listing all of them
        // (not just this step's touched layers) is a harmless no-op, not wasted work.
        std::vector<mx::array> all{logits};
        for (auto &a : g_fused_K) all.push_back(a);
        for (auto &a : g_fused_V) all.push_back(a);
        mx::eval(all);
        const int VOCAB = (int)logits.shape().back();
        std::memcpy(logits_out, logits.data<float>(), sizeof(float) * (size_t)B * (size_t)VOCAB);
        delete g_fused_x; g_fused_x = nullptr;
        g_fused_pos = -1; g_fused_layers_done = 0;
        return 1;
    } catch (...) {
        delete g_fused_x; g_fused_x = nullptr;
        g_fused_pos = -1; g_fused_layers_done = 0;
        return 0;
    }
}

// ---------------------------------------------------------------------------
// V5e: ragged multi-step GPU decode -- generalizes V5d's mechanism one more
// step, from "B rows, all sharing one scalar pos (lockstep)" to "A rows,
// each carrying its OWN (slot,pos) pair (ragged)". One call shape covers
// BOTH prefill (a slot advancing one more prompt position) and decode (a
// slot generating a new token) -- unlike the CPU reference (moe_cbatch_step(),
// qwen_infer.c:4639), which hardcodes prefill as a separate scalar
// moe_forward_token() loop. Mirrors moe_cbatch_step()'s own (token_ids, slot,
// spos, A) naming per the standing plan.
//
// Reuses g_fused_K/g_fused_V (V5d's persistent per-slot K/V arrays, shaped
// {N_SLOTS,H,MLA_L0_MAXPOS,*} via mlx_gpu_set_batch(N_SLOTS) -- V5e just
// changes HOW they're written/read: mx::scatter() with 2-axis (slot,pos)
// indices instead of V5d's slice_update() with one shared pos, and
// mx::take(...,slot_arr,0) to gather only this step's active columns' own
// windows instead of using the whole persistent array directly.
//
// Every new MLX composition here was verified via an isolated probe against
// a trusted host-loop/per-row reference BEFORE being used in this function,
// per this file's standing discipline (this codebase's Bug 1 was exactly a
// "looked right at trivial dims, silently wrong at real dims" failure --
// probing at real dims before committing code is how that class of bug is
// caught here, not by inspection):
//   - mx::scatter with 2-axis (slot,pos) indices: probe_ragged_primitives.cpp
//     Probe A, max_abs_diff=0 vs a host-loop reference. Its real convention is
//     a WINDOW scatter (updates.ndim == a.ndim + indices[0].ndim), not a
//     numpy-style scalar-per-index scatter -- updates must be shaped with a
//     leading {A} index-count dim, size-1 at each INDEXED axis, full size at
//     each UNINDEXED axis (confirmed the hard way: the first attempt's naive
//     {A,H,1,QHD} shape threw a dimension-count mismatch).
//   - mx::fast::rope's per-row offset ARRAY overload: probe_ragged_
//     primitives.cpp Probe B (H=1 case, max_abs_diff=0) PLUS a follow-up gap
//     probe (probe_ragged_rope_multihead.cpp) specifically for H>1, since
//     Probe B alone never exercised more than one head and this codebase's
//     own Bug 1 was exactly a "correct at a trivial case, silently wrong at
//     the real multi-dim case" failure. That follow-up found offset must be
//     RANK-1 (a 2D {A,H} offset throws) AND keys off axis -2 as a genuine
//     incrementing "sequence" axis (max_abs_diff 2.54 when H sat at axis -2,
//     since positions became offset[i]+arange(H) -- each head silently got a
//     DIFFERENT rotation instead of sharing the column's one true position).
//     The only verified-correct arrangement: flatten (A,H) into ONE leading
//     axis of size A*H with axis -2 pinned to size 1, and an offset array of
//     rank 1 length A*H with each column's spos value repeated H times in the
//     same row-major order mx::reshape uses (max_abs_diff=0). k_pe has no H
//     axis to begin with (MQA-style, shared across heads before broadcast),
//     so it uses the already-Probe-B-verified {A,1,ROPE}/offset{A} form
//     directly, unchanged in shape convention from what was already probed.
//   - scaled_dot_product_attention's mask_arr with a genuinely PER-ROW
//     (not shared) boolean mask: probe_ragged_primitives.cpp Probe C,
//     max_abs_diff=1.19e-07 (fp32 rounding) vs A independent per-row
//     truncated-K/V references.
//
// slot_arr/pos_arr below use the 3-arg iterator array(It,Shape,Dtype) ctor
// (mlx/array.h array::init() -> allocator::malloc + std::copy), NOT the
// 4-arg pointer-wrap ctor wrap_host_f32() uses -- confirmed via the header
// itself that this ctor COPIES immediately, unlike the no-copy wrap that
// caused Bug 1 (g_mla_yarn_freqs_f32's own header comment). This matters
// specifically here because eval() is deferred all the way to
// mlx_gpu_cbatch_forward_finalize(), well after this function (and the
// caller's own slot[]/spos[] stack buffers) may have returned/been reused.
static mx::array *g_cbatch_x = nullptr;
static int g_cbatch_A = 0;
static int g_cbatch_layers_done = 0;

int mlx_gpu_cbatch_layer_step_lazy(int l, int A, const int *slot, const int *spos, int is_dense,
                                    const float *x_in_host, const float *w_inln,
                                    const float *w_postln, const float *w_kvaln,
                                    const float *w_gate) {
    if (g_mla_n_heads == 0 || g_layer_hidden == 0) return 0;
    if (A < 1 || A > 64 || l < 0 || l >= MLA_MAXLAYERS) return 0;
    try {
        const int H = g_mla_n_heads, QHD = g_mla_q_head_dim, NOPE = g_mla_qk_nope,
                  ROPE = g_mla_qk_rope, VHD = g_mla_v_hd, KVLORA = g_mla_kv_lora;
        const int HIDDEN = g_layer_hidden, IM = g_layer_im_dim, DENSE_IM = g_layer_dense_im,
                  NE = g_layer_n_experts, NS = g_layer_n_shared, TOPK = g_layer_top_k;

        ensure_fused_kv_init();   // sizes g_fused_K/V to {g_fused_B,H,MLA_L0_MAXPOS,*} --
                                   // caller must have set g_fused_B=N_SLOTS via mlx_gpu_set_batch()
                                   // before the first step of a cbatch run (persists across steps,
                                   // unlike V5d's B which is fully rewritten every call in lockstep).
        for (int i = 0; i < A; i++) {
            if (spos[i] < 0 || spos[i] >= MLA_L0_MAXPOS) return 0;
        }
        if (l == 0) {
            delete g_cbatch_x;
            g_cbatch_x = new mx::array(wrap_host_f32(x_in_host, {A, HIDDEN}));
            g_cbatch_A = A; g_cbatch_layers_done = 0;
        }
        if (g_cbatch_A != A || g_cbatch_layers_done != l) return 0;

        // COPYING ctor (see header comment above) -- safe under the deferred eval() this
        // whole fused design relies on.
        mx::array slot_arr(slot, {A}, mx::int32);
        mx::array pos_arr(spos, {A}, mx::int32);

        char nq[96], nkva[96], nkvb[96], no[96];
        snprintf(nq, sizeof nq, "model.layers.%d.self_attn.q_proj", l);
        snprintf(nkva, sizeof nkva, "model.layers.%d.self_attn.kv_a_proj_with_mqa", l);
        snprintf(nkvb, sizeof nkvb, "model.layers.%d.self_attn.kv_b_proj", l);
        snprintf(no, sizeof no, "model.layers.%d.self_attn.o_proj", l);

        mx::array x = *g_cbatch_x;
        mx::array h = mx::fast::rms_norm(x, wrap_host_f32(w_inln, {HIDDEN}), (float)g_mla_rms_eps);

        mx::array q = lazy_matvec_e0(nq, h);
        mx::array kv_ap = lazy_matvec_e0(nkva, h);
        mx::array compressed_kv = mx::slice(kv_ap, {0, 0}, {A, KVLORA});
        mx::array k_pe_raw = mx::slice(kv_ap, {0, KVLORA}, {A, KVLORA + ROPE});
        mx::array normed_kv = mx::fast::rms_norm(compressed_kv, wrap_host_f32(w_kvaln, {KVLORA}),
                                                  (float)g_mla_rms_eps);
        mx::array kv_b = lazy_matvec_e0(nkvb, normed_kv);

        mx::array q_r = mx::reshape(q, {A, H, QHD});
        mx::array q_nope = mx::slice(q_r, {0, 0, 0}, {A, H, NOPE});
        mx::array q_pe_raw = mx::slice(q_r, {0, 0, NOPE}, {A, H, NOPE + ROPE});
        mx::array q_pe_scaled = q_pe_raw * (float)g_mla_rope_mscale;
        mx::array k_pe_scaled = k_pe_raw * (float)g_mla_rope_mscale;

        mx::array freqs_f32((void *)g_mla_yarn_freqs_f32.data(),
                             {(int)g_mla_yarn_freqs_f32.size()}, mx::float32, noop_deleter);

        // Per-row RoPE offset -- pos_arr instead of V5d's scalar pos. q_pe needs the
        // flatten-(A,H)-into-one-axis treatment (see header comment: offset is rank-1 and
        // keys off axis -2 as a real incrementing sequence axis, so H CANNOT sit at axis -2
        // or every head silently gets a different rotation). k_pe has no head axis to begin
        // with, so it reuses the already-probed {A,1,ROPE}/offset{A} form directly.
        std::vector<int32_t> pos_ah((size_t)A * H);
        for (int m = 0; m < A; m++) for (int hh = 0; hh < H; hh++) pos_ah[(size_t)m * H + hh] = spos[m];
        mx::array pos_ah_arr(pos_ah.data(), {A * H}, mx::int32);   // copying ctor, see header comment
        mx::array q_pe_flat = mx::reshape(q_pe_scaled, {A * H, 1, ROPE});
        mx::array q_pe_rot_flat = mx::fast::rope(q_pe_flat, ROPE, /*traditional=*/true,
                                                  /*base=*/std::nullopt, /*scale=*/1.0f,
                                                  /*offset=*/pos_ah_arr, /*freqs=*/freqs_f32);
        mx::array q_pe_rot = mx::reshape(q_pe_rot_flat, {A, H, ROPE});
        mx::array k_pe_in = mx::reshape(k_pe_scaled, {A, 1, ROPE});
        mx::array k_pe_rot = mx::fast::rope(k_pe_in, ROPE, true, std::nullopt, 1.0f, pos_arr, freqs_f32);

        mx::array q_full = mx::concatenate({q_nope, q_pe_rot}, -1);
        mx::array kv_b_r = mx::reshape(kv_b, {A, H, NOPE + VHD});
        mx::array k_nope = mx::slice(kv_b_r, {0, 0, 0}, {A, H, NOPE});
        mx::array v_new = mx::contiguous(mx::slice(kv_b_r, {0, 0, NOPE}, {A, H, NOPE + VHD}));
        mx::array k_pe_bcast = mx::broadcast_to(k_pe_rot, {A, H, ROPE});
        mx::array k_new = mx::concatenate({k_nope, k_pe_bcast}, -1);

        // Scatter this step's K/V into each column's own (slot,pos) -- replaces V5d's
        // slice_update(shared pos). updates shape = {A(index-batch), window@slot=1, H(full),
        // window@pos=1, QHD/VHD(full)} -- scatter's real window-write convention, confirmed
        // via probe_ragged_primitives.cpp Probe A.
        mx::array k_new_win = mx::reshape(k_new, {A, 1, H, 1, QHD});
        mx::array v_new_win = mx::reshape(v_new, {A, 1, H, 1, VHD});
        g_fused_K[l] = mx::scatter(g_fused_K[l], {slot_arr, pos_arr}, k_new_win, {0, 2});
        g_fused_V[l] = mx::scatter(g_fused_V[l], {slot_arr, pos_arr}, v_new_win, {0, 2});

        // Gather THIS step's active columns' own persistent windows (mx::take by slot_arr) --
        // each column reads its OWN slot's full {H,MLA_L0_MAXPOS,*} window, not the whole
        // persistent N_SLOTS array. Per-column mask (row m true for j<=spos[m]) replaces
        // V5d's single shared mask -- verified via Probe C.
        mx::array k_win = mx::take(g_fused_K[l], slot_arr, 0);   // {A,H,MLA_L0_MAXPOS,QHD}
        mx::array v_win = mx::take(g_fused_V[l], slot_arr, 0);   // {A,H,MLA_L0_MAXPOS,VHD}

        std::vector<uint8_t> mask_bytes((size_t)A * MLA_L0_MAXPOS);
        for (int m = 0; m < A; m++)
            for (int j = 0; j < MLA_L0_MAXPOS; j++)
                mask_bytes[(size_t)m * MLA_L0_MAXPOS + j] = (j <= spos[m]) ? 1 : 0;
        mx::array mask_arr(mask_bytes.data(), {A, 1, 1, MLA_L0_MAXPOS}, mx::bool_);   // copying ctor

        mx::array q_full_r = mx::reshape(q_full, {A, H, 1, QHD});
        mx::array attn = mx::fast::scaled_dot_product_attention(q_full_r, k_win, v_win,
                                                                  (float)g_mla_attn_scale, "array", mask_arr);
        mx::array attn_flat = mx::reshape(attn, {A, H * VHD});
        mx::array o = lazy_matvec_e0(no, attn_flat);
        mx::array x_mid = x + o;

        mx::array h2 = mx::fast::rms_norm(x_mid, wrap_host_f32(w_postln, {HIDDEN}), (float)g_mla_rms_eps);

        // Router/FFN block below is IDENTICAL to V5d's mlx_gpu_layer_step_lazy() (including
        // F-4's sort/unsort branch) -- A plays the exact structural role V5d's B did; none of
        // this math has any notion of "slot" or "pos".
        std::optional<mx::array> mlp_out_opt;
        char nm[128];
        if (is_dense) {
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.gate_proj", l);
            mx::array gate_v = lazy_matvec_e0(nm, h2);
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.up_proj", l);
            mx::array up_v = lazy_matvec_e0(nm, h2);
            mx::array sw = lazy_silu(gate_v) * up_v;
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.down_proj", l);
            mlp_out_opt = lazy_matvec_e0(nm, sw);
            (void)DENSE_IM;
        } else {
            mx::array w_gate_arr = wrap_host_f32(w_gate, {NE, HIDDEN});
            mx::array scores_raw = mx::matmul(h2, mx::transpose(w_gate_arr));       // {A,NE}
            mx::array scores = mx::softmax(scores_raw, std::vector<int>{-1}, /*precise=*/true);
            mx::array order = mx::argsort(scores, -1);                              // {A,NE}, per-row
            mx::array top_idx_u = mx::slice(order, {0, NE - TOPK}, {A, NE});         // {A,TOPK}
            mx::array top_idx = mx::astype(top_idx_u, mx::int32);
            mx::array top_wgt = mx::take_along_axis(scores, top_idx, 1);             // {A,TOPK}

            snprintf(nm, sizeof nm, "model.layers.%d.mlp.switch_mlp.gate_proj", l);
            QTensor &tg = g_tensors.at(nm);
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.switch_mlp.up_proj", l);
            QTensor &tu = g_tensors.at(nm);
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.switch_mlp.down_proj", l);
            QTensor &td = g_tensors.at(nm);

            std::optional<mx::array> down_flat_opt;
            if ((long)A * TOPK >= g_gpu_sort_threshold) {
                // D-sort path (F-4), same composition as V5d -- see mlx_gpu_layer_step_lazy()'s
                // own comment for the full derivation/verification history.
                mx::array x_flat3 = mx::reshape(h2, {A, 1, HIDDEN});
                mx::array flat_idx = mx::reshape(top_idx, {A * TOPK});
                mx::array order2 = mx::argsort(flat_idx, 0);
                mx::array inv_order = mx::argsort(order2, 0);
                mx::array topk_arr(TOPK, mx::int32);
                mx::array row_sel = mx::floor_divide(order2, topk_arr);
                mx::array x_sorted = mx::take(x_flat3, row_sel, 0);      // {A*TOPK,1,HIDDEN}
                mx::array idx_sorted = mx::take(flat_idx, order2);       // {A*TOPK}

                mx::array gate_all_s = mx::gather_qmm(x_sorted, tg.w, tg.scales, tg.biases, std::nullopt,
                                                       idx_sorted, true, g_layer_group, 4, "affine", true);
                mx::array up_all_s = mx::gather_qmm(x_sorted, tu.w, tu.scales, tu.biases, std::nullopt,
                                                     idx_sorted, true, g_layer_group, 4, "affine", true);
                mx::array swiglu_3d_s = mx::reshape(lazy_silu(gate_all_s) * up_all_s, {A * TOPK, 1, IM});
                mx::array down_all_s = mx::gather_qmm(swiglu_3d_s, td.w, td.scales, td.biases, std::nullopt,
                                                       idx_sorted, true, g_layer_group, 4, "affine", true);
                mx::array down_flat_s = mx::reshape(down_all_s, {A * TOPK, HIDDEN});
                mx::array down_unsorted = mx::take(down_flat_s, inv_order, 0);   // scatter-unsort
                down_flat_opt = mx::reshape(down_unsorted, {A, TOPK, HIDDEN});
            } else {
                mx::array h2_expanded = mx::expand_dims(h2, std::vector<int>{-2, -3});   // {A,1,1,HIDDEN}
                mx::array gate_all = mx::gather_qmm(h2_expanded, tg.w, tg.scales, tg.biases, std::nullopt,
                                                     top_idx, true, g_layer_group, 4, "affine", false);
                mx::array up_all = mx::gather_qmm(h2_expanded, tu.w, tu.scales, tu.biases, std::nullopt,
                                                   top_idx, true, g_layer_group, 4, "affine", false);
                mx::array swiglu_3d = mx::reshape(lazy_silu(gate_all) * up_all, {A * TOPK, 1, IM});
                mx::array top_idx_1d = mx::reshape(top_idx, {A * TOPK});
                mx::array down_all = mx::gather_qmm(swiglu_3d, td.w, td.scales, td.biases, std::nullopt,
                                                     top_idx_1d, true, g_layer_group, 4, "affine", false);
                down_flat_opt = mx::reshape(down_all, {A, TOPK, HIDDEN});
            }
            mx::array down_flat = *down_flat_opt;
            mx::array top_wgt_col = mx::reshape(top_wgt, {A, TOPK, 1});
            mx::array weighted = down_flat * top_wgt_col;
            mx::array routed_sum = mx::sum(weighted, std::vector<int>{1}, false);    // {A,HIDDEN}

            if (NS > 0) {
                snprintf(nm, sizeof nm, "model.layers.%d.mlp.shared_experts.gate_proj", l);
                mx::array sgate = lazy_matvec_e0(nm, h2);
                snprintf(nm, sizeof nm, "model.layers.%d.mlp.shared_experts.up_proj", l);
                mx::array sup = lazy_matvec_e0(nm, h2);
                mx::array sswiglu = lazy_silu(sgate) * sup;
                snprintf(nm, sizeof nm, "model.layers.%d.mlp.shared_experts.down_proj", l);
                mx::array sdown = lazy_matvec_e0(nm, sswiglu);
                mlp_out_opt = routed_sum + sdown;
            } else {
                mlp_out_opt = routed_sum;
            }
        }
        mx::array x_out = x_mid + *mlp_out_opt;
        delete g_cbatch_x;
        g_cbatch_x = new mx::array(x_out);   // still LAZY -- not evaluated until finalize()
        g_cbatch_layers_done = l + 1;
        return 1;
    } catch (...) {
        delete g_cbatch_x; g_cbatch_x = nullptr;
        g_cbatch_A = 0; g_cbatch_layers_done = 0;
        return 0;
    }
}

int mlx_gpu_cbatch_forward_finalize(const float *w_finalnorm, float *logits_out) {
    if (!g_cbatch_x) return 0;
    try {
        const int HIDDEN = g_layer_hidden;
        const int A = g_cbatch_A;
        mx::array x_final = mx::fast::rms_norm(*g_cbatch_x, wrap_host_f32(w_finalnorm, {HIDDEN}),
                                                (float)g_mla_rms_eps);
        mx::array logits = lazy_matvec_e0("lm_head", x_final);   // {A,VOCAB}
        std::vector<mx::array> all{logits};
        for (auto &a : g_fused_K) all.push_back(a);
        for (auto &a : g_fused_V) all.push_back(a);
        mx::eval(all);
        const int VOCAB = (int)logits.shape().back();
        std::memcpy(logits_out, logits.data<float>(), sizeof(float) * (size_t)A * (size_t)VOCAB);
        delete g_cbatch_x; g_cbatch_x = nullptr;
        g_cbatch_A = 0; g_cbatch_layers_done = 0;
        return 1;
    } catch (...) {
        delete g_cbatch_x; g_cbatch_x = nullptr;
        g_cbatch_A = 0; g_cbatch_layers_done = 0;
        return 0;
    }
}
