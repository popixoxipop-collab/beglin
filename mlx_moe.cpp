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
