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

#include <cstring>
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

// Layer-0-only K/V cache, GPU arm's own (separate from qwen_infer.c's CPU-side
// g_moe_K/V -- D-gpu-2, no shared mutable state across the vendor boundary).
// Layout [head][pos][dim], NOT [pos][head][dim], so that the slice needed for
// one sdpa call (all heads, positions 0..pos) is built by copying H
// independent contiguous runs rather than reordering per-element.
#define MLA_L0_MAXPOS 32
static std::vector<float> g_mla_l0_K;  // H * MLA_L0_MAXPOS * q_head_dim
static std::vector<float> g_mla_l0_V;  // H * MLA_L0_MAXPOS * v_hd

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
    g_mla_l0_K.assign((size_t)n_heads * MLA_L0_MAXPOS * q_head_dim, 0.0f);
    g_mla_l0_V.assign((size_t)n_heads * MLA_L0_MAXPOS * v_hd, 0.0f);
    return 1;
}

int mlx_gpu_mla_layer0(const float *h, int pos, const float *kv_a_ln_w, float *o_out) {
    if (g_mla_n_heads == 0) return 0;   // mlx_gpu_mla_config() not called
    if (pos < 0 || pos >= MLA_L0_MAXPOS) return 0;
    auto itq   = g_tensors.find("model.layers.0.self_attn.q_proj");
    auto itkva = g_tensors.find("model.layers.0.self_attn.kv_a_proj_with_mqa");
    auto itkvb = g_tensors.find("model.layers.0.self_attn.kv_b_proj");
    auto ito   = g_tensors.find("model.layers.0.self_attn.o_proj");
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
        if (!mlx_gpu_matvec_probe("model.layers.0.self_attn.q_proj", 0, h, q.data())) return 0;
        if (!mlx_gpu_matvec_probe("model.layers.0.self_attn.kv_a_proj_with_mqa", 0, h, kv_ap.data())) return 0;

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
        if (!mlx_gpu_matvec_probe("model.layers.0.self_attn.kv_b_proj", 0, normed_kv.data(), kv_b.data()))
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
            float *kdst = g_mla_l0_K.data() + ((size_t)hh * MLA_L0_MAXPOS + pos) * QHD;
            float *vdst = g_mla_l0_V.data() + ((size_t)hh * MLA_L0_MAXPOS + pos) * VHD;
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
                        g_mla_l0_K.data() + (size_t)hh * MLA_L0_MAXPOS * QHD,
                        sizeof(float) * kv_len * QHD);
            std::memcpy(v_stage.data() + (size_t)hh * kv_len * VHD,
                        g_mla_l0_V.data() + (size_t)hh * MLA_L0_MAXPOS * VHD,
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
        if (!mlx_gpu_matvec_probe("model.layers.0.self_attn.o_proj", 0, attn_out.data(), o_out))
            return 0;
        (void)hidden;
        return 1;
    } catch (...) {
        return 0;
    }
}
