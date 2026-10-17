// BEGLIN research-only, real mlx_moe.cpp registry -> lazy_matvec_e0 regression.
// No model weights, serving, or policy promotion. Apple MLX hardware required.
#include "mlx_moe.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
constexpr int OUT = 7, IN = 128, NG = IN / 64, BATCH = 3;
const char *NAME = "model.layers.0.self_attn.q_proj";

struct Case {
    int n = 0;
    int kind = 0;
    std::vector<uint8_t> blob; // borrow remains valid while binding/snapshot lives
    std::vector<float> want;
    double rel_l2 = 0.0;
    double max_abs = 0.0;
};

std::vector<float> activation() {
    std::vector<float> x(BATCH * IN);
    for (int b = 0; b < BATCH; ++b)
        for (int j = 0; j < IN; ++j)
            x[(size_t)b * IN + j] = std::sin((b + 1) * (j + 1) * 0.0763f) +
                                   0.3f * std::cos((j + 4) * 0.0385f);
    return x;
}

Case make_case(int n, const std::vector<float> &x) {
    Case c;
    c.n = n;
    c.kind = (n == 6 ? 1 : 3);
    const size_t cells = (size_t)OUT * NG;
    const size_t planes_bytes = cells * n * 8u;
    const size_t scales_bytes = cells * sizeof(float);
    c.blob.assign(planes_bytes + scales_bytes * 2, 0);
    std::vector<double> ref(BATCH * OUT, 0.0);
    for (int row = 0; row < OUT; ++row) {
        for (int g = 0; g < NG; ++g) {
            size_t cell = (size_t)row * NG + g;
            float scale = 0.0007f * (1 + row + 2 * g);
            std::memcpy(c.blob.data() + planes_bytes + cell * sizeof(float), &scale, sizeof(float));
            for (int p = 0; p < 64; ++p) {
                int u = (p * 17 + row * 29 + g * 5 + 3) % (1 << n);
                int code = u - (1 << (n - 1));
                for (int bit = 0; bit < n; ++bit)
                    if ((u >> bit) & 1)
                        c.blob[cell * n * 8u + (size_t)bit * 8u + (p >> 3)] |=
                            (uint8_t)(1u << (p & 7));
                for (int b = 0; b < BATCH; ++b)
                    ref[(size_t)b * OUT + row] +=
                        (double)((float)code * scale) * x[(size_t)b * IN + g * 64 + p];
            }
        }
    }
    c.want.resize(ref.size());
    for (size_t k = 0; k < ref.size(); ++k) c.want[k] = (float)ref[k];
    return c;
}

bool run_case(Case &c, const std::vector<float> &x) {
    const size_t packed_bytes = (size_t)OUT * NG * c.n * 8u;
    const size_t scale_bytes = (size_t)OUT * NG * sizeof(float);
    if (!mlx_gpu_bind_af(c.blob.data(), (long)c.blob.size(), NAME, 1, OUT, IN, NG,
                         0, (long)packed_bytes, (long)(packed_bytes + scale_bytes), c.n))
        return false;
    int bits = -1;
    if (mlx_gpu_binding_kind(NAME, &bits) != c.kind || bits != c.n) return false;
    std::vector<float> y(BATCH * OUT, 0);
    if (!mlx_gpu_qng64_bound_batch_probe(NAME, x.data(), BATCH, y.data())) return false;
    std::vector<float> y0(OUT, 0);
    if (!mlx_gpu_matvec_probe(NAME, 0, x.data(), y0.data())) return false;
    double e2 = 0.0, ref2 = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (!std::isfinite(y[i]) || !std::isfinite(c.want[i])) return false;
        double d = (double)y[i] - c.want[i];
        c.max_abs = std::max(c.max_abs, std::fabs(d));
        e2 += d*d;
        ref2 += (double)c.want[i] * c.want[i];
        if (i < OUT && std::fabs(y[i] - y0[i]) > 0.001) return false;
    }
    c.rel_l2 = std::sqrt(e2 / std::max(ref2, 1e-24));
    return c.rel_l2 < 0.0003 && c.max_abs < 0.002;
}
} // namespace

int main() {
    if (!mlx_gpu_available()) { std::fprintf(stderr, "NO_REAL_APPLE_MLX_GPU\n"); return 2; }
    const auto x = activation();
    Case c7 = make_case(7, x);
    if (!run_case(c7, x)) { std::fprintf(stderr, "FAILED_n7\n"); return 3; }
    uint64_t snapshot = 0;
    if (!mlx_gpu_snapshot_binding(NAME, &snapshot) || snapshot == 0) return 4;
    Case c6 = make_case(6, x);
    if (!run_case(c6, x)) { std::fprintf(stderr, "FAILED_n6\n"); return 5; }
    Case c9 = make_case(9, x);
    if (!run_case(c9, x)) { std::fprintf(stderr, "FAILED_n9\n"); return 6; }
    if (!mlx_gpu_restore_binding_snapshot(snapshot)) return 7;
    int bits = -1;
    if (mlx_gpu_binding_kind(NAME, &bits) != 3 || bits != 7) return 8;
    std::vector<float> restored(BATCH * OUT);
    if (!mlx_gpu_qng64_bound_batch_probe(NAME, x.data(), BATCH, restored.data())) return 9;
    for (size_t k = 0; k < restored.size(); ++k)
        if (std::fabs(restored[k] - c7.want[k]) > 0.002) return 10;
    if (!mlx_gpu_synchronize() || !mlx_gpu_reset_runtime_epoch()) return 11;
    std::vector<float> after_epoch(BATCH * OUT);
    if (!mlx_gpu_qng64_bound_batch_probe(NAME, x.data(), BATCH, after_epoch.data())) return 12;
    for (size_t k = 0; k < after_epoch.size(); ++k)
        if (std::fabs(after_epoch[k] - restored[k]) > 0.0001) return 13;
    if (mlx_gpu_qng64_bound_batch_probe(NAME, x.data(), 0, after_epoch.data()) ||
        mlx_gpu_qng64_bound_batch_probe("missing.tensor", x.data(), BATCH, after_epoch.data()) ||
        mlx_gpu_qng64_bound_batch_probe(nullptr, x.data(), BATCH, after_epoch.data()))
        return 14;
    if (mlx_gpu_bind_af(c7.blob.data(), (long)c7.blob.size(), NAME, 1, OUT, IN, NG,
                        0, OUT * NG * 7 * 8, OUT * NG * 7 * 8 + OUT * NG * 4, 1))
        return 15;
    std::printf(
        "{\"status\":\"PASS\",\"backend\":\"actual BEGLIN mlx_moe.cpp\","
        "\"batch\":3,\"n6_rel_l2\":%.9g,\"n7_rel_l2\":%.9g,\"n9_rel_l2\":%.9g,"
        "\"n6_max_abs\":%.9g,\"n7_max_abs\":%.9g,\"n9_max_abs\":%.9g,"
        "\"snapshot_restore\":true,\"epoch_reset\":true,\"negative_guards\":true}\n",
        c6.rel_l2, c7.rel_l2, c9.rel_l2,
        c6.max_abs, c7.max_abs, c9.max_abs);
    return 0;
}
