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
// V5c-fused: one-eval-per-LAYER rewrite (not per-token -- see below), addressing
// the kill-gate's root cause (~505 eager eval() dispatches/token -> Metal
// command-buffer overhead dominating tiny M=1 GEMMs).
//
// TWO real, reproducible eval-granularity bugs were found and fixed during this
// rewrite, both via the same method: cross-check a lazy intermediate against
// the already-proven eager mlx_gpu_layer_step() using debug peek accessors
// (mlx_gpu_layer_step_dbg() above), added and then removed once each bug was
// isolated and fixed.
//
// Bug 1 (per-TOKEN granularity, ~540-node graph): building the ENTIRE 27-layer
// graph lazily and calling mx::eval() exactly once at the very end (logits +
// every layer's K/V together) gave a CORRECT result at pos=0 but a WRONG one
// for pos>=1, even though every individual sub-computation (attention per
// layer, dense FFN, router selection, routed FFN, and each layer's full
// x_out) checked out correct when read via a forced early eval(). Not
// resolved at the root (MLX's own graph executor was out of scope to debug
// further); worked around by evaluating once per LAYER instead of once per
// token (below).
//
// Bug 2 (per-LAYER granularity, ~15-node graph, found bisecting a REGRESSION
// vs Bug 1's own per-token attempt -- pos=0 was wrong too under naive
// per-layer eval, where it had been correct under per-token): isolated via
// x_mid/mlp_out/top_idx/swiglu_2d/down_flat peeks to the down_proj step
// specifically. down_proj needs a DIFFERENT input row per selected expert
// (switch_down's TOPK rows of swiglu'd activation, one per expert), which
// gather_qmm computes as a full (TOPK,TOPK,out) cross product (verified in
// isolation: NOT a per-row pairing) -- the correct row is its diagonal,
// extracted via mx::take_along_axis(cross, diag_idx, 1) with diag_idx shape
// {TOPK,1,1}. That extraction is numerically correct in ISOLATION (confirmed
// via a synthetic Python mlx.core repro with identical shapes/dtypes) and
// correct when eval'd on its OWN -- but silently returns near-zero/garbled
// data when its eval is deferred and swept into a LARGER combined mx::eval()
// batch alongside unrelated ops (attention, dense-FFN-shaped ops, k_new/v_new).
// Reproduced twice (layer 1 fixed by forcing an early eval, layer 2 identical
// code path still broken without it) before concluding this is inherent to
// gather_qmm(lhs_indices+rhs_indices cross-product)+take_along_axis specifically,
// not a general "big graph" problem -- the per-token version's own bug (Bug 1)
// is graph-depth-related; this one reproduces at a much smaller ~15-node scope
// and is specific to this op combination. Fixed by evaluating down_flat on its
// own, immediately after computing it, before folding it into the rest of the
// layer's (otherwise still-lazy) graph. This fix is CONFIRMED and real: with it,
// pos=0 (no K/V history) matches the golden eager path exactly (rel-L2 5.22e-07).
//
// Bug 3 (UNRESOLVED, pos>=1 only): even with Bug 2 fixed, every position with
// nonempty K/V history (pos>=1) still produces wrong logits (gpu_vs_cpu rel-L2
// 0.44-1.26 across pos=1..7, growing with history length) in the real,
// undebugged pipeline. Extensive bisection was attempted (x_mid/mlp_out/
// top_idx/swiglu_2d/down_flat cross-checks via mlx_gpu_layer_step_dbg, at every
// layer, at every position) and found EVERY individual layer/position combo
// correct -- but this bisection method turned out to be invalid: dbg's eager
// path shares g_mla_K/g_mla_V with the lazy path, so calling it for
// verification silently overwrites (and "self-heals") the very cache slots
// under test, masking whatever the lazy path's real behavior would have been.
// Four different "does this need a standalone eval too" tests (o, attn, x_mid
// as an explicit eval() output, mlp_out_opt as an explicit eval() output, and
// a forced host-sync .data<float>() read on x_out) were tried against the
// CLEAN (non-debug-contaminated) pipeline and produced BIT-IDENTICAL wrong
// output every time -- meaning, unlike Bug 2, this is very unlikely to be an
// eval-timing/graph-composition issue, and is more likely a genuine logic bug
// in the K/V-history read/concatenate path (the only pos-dependent code) that
// was not isolated within the time budget spent. Left for a future session
// with non-cache-sharing debug tooling (e.g. a scratch K/V cache for the dbg
// variant, or a from-scratch hand-computed RoPE+history reference that never
// touches g_mla_K/V at all).
static mx::array *g_fused_x = nullptr;   // pending residual stream, always ALREADY evaluated between calls
static int g_fused_pos = -1;
static int g_fused_layers_done = 0;

static mx::array wrap_host_f32(const float *p, std::initializer_list<int> shape) {
    return mx::array((void *)p, mx::Shape(shape), mx::float32, noop_deleter);
}
static mx::array lazy_silu(const mx::array &x) { return mx::multiply(x, mx::sigmoid(x)); }

// Same shape as mlx_gpu_matvec_probe()'s quantized_matmul call, but returns the
// (unevaluated) array instead of eval()'ing + copying to host.
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
    try {
        const int H = g_mla_n_heads, QHD = g_mla_q_head_dim, NOPE = g_mla_qk_nope,
                  ROPE = g_mla_qk_rope, VHD = g_mla_v_hd, KVLORA = g_mla_kv_lora;
        const int HIDDEN = g_layer_hidden, IM = g_layer_im_dim, DENSE_IM = g_layer_dense_im,
                  NE = g_layer_n_experts, NS = g_layer_n_shared, TOPK = g_layer_top_k;

        if (l == 0) {
            delete g_fused_x;
            g_fused_x = new mx::array(wrap_host_f32(x_in_host, {1, HIDDEN}));
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

        mx::array q = lazy_matvec_e0(nq, h);
        mx::array kv_ap = lazy_matvec_e0(nkva, h);
        mx::array compressed_kv = mx::slice(kv_ap, {0, 0}, {1, KVLORA});
        mx::array k_pe_raw = mx::slice(kv_ap, {0, KVLORA}, {1, KVLORA + ROPE});
        mx::array normed_kv = mx::fast::rms_norm(compressed_kv, wrap_host_f32(w_kvaln, {KVLORA}),
                                                  (float)g_mla_rms_eps);
        mx::array kv_b = lazy_matvec_e0(nkvb, normed_kv);

        mx::array q_r = mx::reshape(q, {1, H, QHD});
        mx::array q_nope = mx::slice(q_r, {0, 0, 0}, {1, H, NOPE});
        mx::array q_pe_raw = mx::slice(q_r, {0, 0, NOPE}, {1, H, NOPE + ROPE});
        mx::array q_pe_scaled = q_pe_raw * (float)g_mla_rope_mscale;
        mx::array k_pe_scaled = k_pe_raw * (float)g_mla_rope_mscale;

        mx::array freqs_arr(g_mla_yarn_freqs.data(), {(int)g_mla_yarn_freqs.size()},
                             mx::float32, noop_deleter);
        std::vector<float> freqs_f(g_mla_yarn_freqs.begin(), g_mla_yarn_freqs.end());
        mx::array freqs_f32((void *)freqs_f.data(), {(int)freqs_f.size()}, mx::float32, noop_deleter);

        mx::array q_pe_hro = mx::transpose(q_pe_scaled, {1, 0, 2});
        mx::array q_pe_rot_hro = mx::fast::rope(q_pe_hro, ROPE, /*traditional=*/true,
                                                 /*base=*/std::nullopt, /*scale=*/1.0f,
                                                 /*offset=*/pos, /*freqs=*/freqs_f32);
        mx::array q_pe_rot = mx::transpose(q_pe_rot_hro, {1, 0, 2});
        mx::array k_pe_in = mx::reshape(k_pe_scaled, {1, 1, ROPE});
        mx::array k_pe_rot = mx::fast::rope(k_pe_in, ROPE, true, std::nullopt, 1.0f, pos, freqs_f32);

        mx::array q_full = mx::concatenate({q_nope, q_pe_rot}, -1);
        mx::array kv_b_r = mx::reshape(kv_b, {1, H, NOPE + VHD});
        mx::array k_nope = mx::slice(kv_b_r, {0, 0, 0}, {1, H, NOPE});
        mx::array v_new = mx::slice(kv_b_r, {0, 0, NOPE}, {1, H, NOPE + VHD});
        mx::array k_pe_bcast = mx::broadcast_to(k_pe_rot, {1, H, ROPE});
        mx::array k_new = mx::concatenate({k_nope, k_pe_bcast}, -1);

        std::optional<mx::array> k_full_opt, v_full_opt;
        std::vector<float> k_stage, v_stage;   // local -- fine, this layer's eval happens before return
        if (pos > 0) {
            k_stage.resize((size_t)H * pos * QHD);
            v_stage.resize((size_t)H * pos * VHD);
            for (int hh = 0; hh < H; hh++) {
                std::memcpy(k_stage.data() + (size_t)hh * pos * QHD,
                            g_mla_K.data() + ((size_t)l * H + hh) * MLA_L0_MAXPOS * QHD,
                            sizeof(float) * (size_t)pos * QHD);
                std::memcpy(v_stage.data() + (size_t)hh * pos * VHD,
                            g_mla_V.data() + ((size_t)l * H + hh) * MLA_L0_MAXPOS * VHD,
                            sizeof(float) * (size_t)pos * VHD);
            }
            mx::array k_hist = wrap_host_f32(k_stage.data(), {1, H, pos, QHD});
            mx::array v_hist = wrap_host_f32(v_stage.data(), {1, H, pos, VHD});
            mx::array k_new_r = mx::reshape(k_new, {1, H, 1, QHD});
            mx::array v_new_r = mx::reshape(v_new, {1, H, 1, VHD});
            k_full_opt = mx::concatenate({k_hist, k_new_r}, 2);
            v_full_opt = mx::concatenate({v_hist, v_new_r}, 2);
        } else {
            k_full_opt = mx::reshape(k_new, {1, H, 1, QHD});
            v_full_opt = mx::reshape(v_new, {1, H, 1, VHD});
        }
        mx::array q_full_r = mx::reshape(q_full, {1, H, 1, QHD});
        mx::array attn = mx::fast::scaled_dot_product_attention(q_full_r, *k_full_opt, *v_full_opt,
                                                                  (float)g_mla_attn_scale, "");
        mx::array attn_flat = mx::reshape(attn, {1, H * VHD});
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
            mx::array scores_raw = mx::matmul(h2, mx::transpose(w_gate_arr));
            mx::array scores = mx::softmax(scores_raw, std::vector<int>{-1}, /*precise=*/true);
            mx::array scores_flat = mx::reshape(scores, {NE});
            mx::array order = mx::argsort(scores_flat, 0);
            mx::array top_idx_u = mx::slice(order, {NE - TOPK}, {NE});
            mx::array top_idx = mx::astype(top_idx_u, mx::int32);
            mx::array top_idx_row = mx::reshape(top_idx, {1, TOPK});
            mx::array top_wgt = mx::take_along_axis(scores, top_idx_row, 1);

            snprintf(nm, sizeof nm, "model.layers.%d.mlp.switch_mlp.gate_proj", l);
            QTensor &tg = g_tensors.at(nm);
            mx::array gate_all = mx::gather_qmm(h2, tg.w, tg.scales, tg.biases, std::nullopt,
                                                 top_idx_row, true, g_layer_group, 4, "affine", false);
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.switch_mlp.up_proj", l);
            QTensor &tu = g_tensors.at(nm);
            mx::array up_all = mx::gather_qmm(h2, tu.w, tu.scales, tu.biases, std::nullopt,
                                               top_idx_row, true, g_layer_group, 4, "affine", false);
            mx::array swiglu_all = lazy_silu(gate_all) * up_all;
            mx::array swiglu_2d = mx::reshape(swiglu_all, {TOPK, IM});

            int32_t lhs_ids[64];
            for (int k = 0; k < TOPK; k++) lhs_ids[k] = k;
            mx::array lhs_idx((void *)lhs_ids, {TOPK}, mx::int32, noop_deleter);
            mx::array top_idx_1d = mx::reshape(top_idx, {TOPK});

            snprintf(nm, sizeof nm, "model.layers.%d.mlp.switch_mlp.down_proj", l);
            QTensor &td = g_tensors.at(nm);
            mx::array down_cross = mx::gather_qmm(swiglu_2d, td.w, td.scales, td.biases, lhs_idx,
                                                   top_idx_1d, true, g_layer_group, 4, "affine", false);
            int32_t diag_ids[64];
            for (int k = 0; k < TOPK; k++) diag_ids[k] = k;
            mx::array diag_idx((void *)diag_ids, {TOPK, 1, 1}, mx::int32, noop_deleter);
            mx::array down_diag = mx::take_along_axis(down_cross, diag_idx, 1);
            mx::array down_flat = mx::reshape(down_diag, {TOPK, HIDDEN});
            // Bug 2 fix (see file header comment): gather_qmm's cross-product +
            // take_along_axis combo silently returns wrong data when its eval is
            // deferred and swept into a later, larger combined mx::eval() batch.
            // Evaluating it here, on its own, is the fix -- confirmed by direct
            // repro (broken when deferred, correct when eval'd standalone here).
            mx::eval(down_flat);
            mx::array top_wgt_col = mx::reshape(top_wgt, {TOPK, 1});
            mx::array weighted = down_flat * top_wgt_col;
            mx::array routed_sum = mx::sum(weighted, std::vector<int>{0}, true);

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

        // ONE eval for the rest of this layer: x_out + this position's new K/V
        // (persisted immediately -- no cross-call deferral).
        std::vector<mx::array> to_eval{x_out, k_new, v_new};
        mx::eval(to_eval);

        const float *kp = k_new.data<float>();
        const float *vp = v_new.data<float>();
        for (int hh = 0; hh < H; hh++) {
            float *kdst = g_mla_K.data() + (((size_t)l * H + hh) * MLA_L0_MAXPOS + pos) * QHD;
            float *vdst = g_mla_V.data() + (((size_t)l * H + hh) * MLA_L0_MAXPOS + pos) * VHD;
            std::memcpy(kdst, kp + (size_t)hh * QHD, sizeof(float) * QHD);
            std::memcpy(vdst, vp + (size_t)hh * VHD, sizeof(float) * VHD);
        }

        delete g_fused_x;
        g_fused_x = new mx::array(x_out);   // already evaluated -- safe to reuse directly
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
        mx::array x_final = mx::fast::rms_norm(*g_fused_x, wrap_host_f32(w_finalnorm, {HIDDEN}),
                                                (float)g_mla_rms_eps);
        mx::array logits = lazy_matvec_e0("lm_head", x_final);
        mx::eval(logits);
        const int VOCAB = (int)logits.shape().back();
        std::memcpy(logits_out, logits.data<float>(), sizeof(float) * (size_t)VOCAB);
        delete g_fused_x; g_fused_x = nullptr;
        g_fused_pos = -1; g_fused_layers_done = 0;
        return 1;
    } catch (...) {
        delete g_fused_x; g_fused_x = nullptr;
        g_fused_pos = -1; g_fused_layers_done = 0;
        return 0;
    }
}
