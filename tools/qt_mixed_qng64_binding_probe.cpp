#include "mlx_moe.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static void encode_cell(std::vector<uint8_t> &planes, size_t base, int n,
                        const std::vector<int> &codes) {
    const int bias = 1 << (n - 1);
    for (int p = 0; p < 64; ++p) {
        const int u = codes[p] + bias;
        for (int j = 0; j < n; ++j) {
            if ((u >> j) & 1) {
                planes[base + (size_t)j * 8u + (size_t)(p >> 3)] |=
                    (uint8_t)(1u << (p & 7));
            }
        }
    }
}

int main() {
    constexpr long OUT = 5;
    constexpr long IN = 128;
    constexpr long NG = IN / 64;
    constexpr int UNIFORM_N = 5;
    const size_t cells = (size_t)OUT * (size_t)NG;
    const char *name = "qt.mixed.binding";

    std::vector<float> x(IN);
    for (long i = 0; i < IN; ++i) {
        x[(size_t)i] = std::sin((float)(i + 1) * 0.113f)
                     + 0.2f * std::cos((float)(i + 1) * 0.041f);
    }

    // Baseline uniform n5 binding.
    std::vector<uint8_t> uniform_planes(cells * (size_t)UNIFORM_N * 8u, 0);
    std::vector<float> uniform_scales(cells);
    std::vector<float> uniform_cpu(OUT, 0.0f);
    for (long row = 0; row < OUT; ++row) {
        for (long g = 0; g < NG; ++g) {
            const size_t cell = (size_t)row * NG + (size_t)g;
            const float scale = 0.0015f * (float)(1 + cell);
            uniform_scales[cell] = scale;
            std::vector<int> codes(64);
            for (int p = 0; p < 64; ++p) {
                const int levels = 1 << UNIFORM_N;
                const int u = (p * 5 + (int)cell * 9 + 1) % levels;
                codes[p] = u - (1 << (UNIFORM_N - 1));
                uniform_cpu[(size_t)row] += (float)codes[p] * scale *
                    x[(size_t)g * 64u + (size_t)p];
            }
            encode_cell(uniform_planes, cell * (size_t)UNIFORM_N * 8u,
                        UNIFORM_N, codes);
        }
    }

    if (!mlx_gpu_bind_qng64_dense_probe(
            uniform_planes.data(), uniform_scales.data(), name, OUT, IN, UNIFORM_N)) {
        std::fprintf(stderr, "uniform n5 bind failed\n");
        return 2;
    }
    int bits_out = 0;
    if (mlx_gpu_binding_kind(name, &bits_out) != 1 || bits_out != UNIFORM_N) {
        std::fprintf(stderr, "unexpected baseline binding kind/bits\n");
        return 3;
    }
    std::vector<float> baseline_gpu(OUT, 0.0f);
    if (!mlx_gpu_matvec_probe(name, 0, x.data(), baseline_gpu.data())) return 4;
    double baseline_error = 0.0;
    for (long r = 0; r < OUT; ++r) {
        baseline_error = std::max(
            baseline_error,
            std::fabs((double)baseline_gpu[(size_t)r] - (double)uniform_cpu[(size_t)r]));
    }

    uint64_t snapshot_id = 0;
    if (!mlx_gpu_snapshot_binding(name, &snapshot_id) || snapshot_id == 0) return 5;

    // Mixed n3/n4/n5/n6 binding using variable-length cell spans.
    std::vector<uint8_t> mixed_bits(cells);
    std::vector<float> mixed_scales(cells);
    std::vector<uint32_t> mixed_offsets(cells + 1, 0);
    std::vector<uint8_t> mixed_planes;
    std::vector<float> mixed_cpu(OUT, 0.0f);

    for (long row = 0; row < OUT; ++row) {
        for (long g = 0; g < NG; ++g) {
            const size_t cell = (size_t)row * NG + (size_t)g;
            const int n = 3 + (int)(cell % 4);
            const float scale = 0.0009f * (float)(1 + cell);
            mixed_bits[cell] = (uint8_t)n;
            mixed_scales[cell] = scale;
            mixed_offsets[cell] = (uint32_t)mixed_planes.size();
            const size_t base = mixed_planes.size();
            mixed_planes.resize(base + (size_t)n * 8u, 0);
            std::vector<int> codes(64);
            const int levels = 1 << n;
            for (int p = 0; p < 64; ++p) {
                const int u = (p * 7 + (int)cell * 11 + 3) % levels;
                codes[p] = u - (1 << (n - 1));
                mixed_cpu[(size_t)row] += (float)codes[p] * scale *
                    x[(size_t)g * 64u + (size_t)p];
            }
            encode_cell(mixed_planes, base, n, codes);
            mixed_offsets[cell + 1] = (uint32_t)mixed_planes.size();
        }
    }

    const size_t mixed_plane_bytes = mixed_planes.size();
    if (!mlx_gpu_bind_qng64_mixed_dense_probe(
            mixed_planes.data(), (long)mixed_planes.size(),
            mixed_offsets.data(), mixed_bits.data(), mixed_scales.data(),
            name, OUT, IN)) {
        std::fprintf(stderr, "mixed bind failed\n");
        return 6;
    }

    // Prove the binder owns its arrays: destroy all source contents before execution.
    std::fill(mixed_planes.begin(), mixed_planes.end(), 0);
    std::fill(mixed_offsets.begin(), mixed_offsets.end(), 0);
    std::fill(mixed_bits.begin(), mixed_bits.end(), 0);
    std::fill(mixed_scales.begin(), mixed_scales.end(), 0.0f);

    bits_out = -1;
    if (mlx_gpu_binding_kind(name, &bits_out) != 4 || bits_out != 0) {
        std::fprintf(stderr, "unexpected mixed binding kind/bits\n");
        return 7;
    }

    std::vector<float> mixed_gpu(OUT, 0.0f);
    if (!mlx_gpu_matvec_probe(name, 0, x.data(), mixed_gpu.data())) return 8;
    double mixed_error = 0.0;
    for (long r = 0; r < OUT; ++r) {
        mixed_error = std::max(
            mixed_error,
            std::fabs((double)mixed_gpu[(size_t)r] - (double)mixed_cpu[(size_t)r]));
    }

    if (!mlx_gpu_restore_binding_snapshot(snapshot_id)) return 9;
    bits_out = 0;
    if (mlx_gpu_binding_kind(name, &bits_out) != 1 || bits_out != UNIFORM_N) {
        std::fprintf(stderr, "restore binding identity mismatch\n");
        return 10;
    }

    std::vector<float> restored_gpu(OUT, 0.0f);
    if (!mlx_gpu_matvec_probe(name, 0, x.data(), restored_gpu.data())) return 11;
    double restore_diff = 0.0;
    for (long r = 0; r < OUT; ++r) {
        restore_diff = std::max(
            restore_diff,
            std::fabs((double)restored_gpu[(size_t)r] - (double)baseline_gpu[(size_t)r]));
    }

    if (!mlx_gpu_drop_binding_snapshot(snapshot_id)) return 12;
    if (mlx_gpu_binding_snapshot_count() != 0) return 13;

    const bool pass = baseline_error < 5e-4 && mixed_error < 5e-4 && restore_diff == 0.0;
    std::printf(
        "{\"status\":\"%s\",\"baseline_kind\":1,\"baseline_bits\":5,"
        "\"mixed_kind\":4,\"mixed_bits_out\":0,\"restored_kind\":1,"
        "\"restored_bits\":5,\"mixed_planes_bytes\":%zu,"
        "\"baseline_max_abs_error\":%.9g,\"mixed_max_abs_error\":%.9g,"
        "\"restore_max_abs_diff\":%.9g,\"owned_storage_source_destroyed\":true}\n",
        pass ? "PASS" : "FAIL", mixed_plane_bytes,
        baseline_error, mixed_error, restore_diff);
    return pass ? 0 : 14;
}
