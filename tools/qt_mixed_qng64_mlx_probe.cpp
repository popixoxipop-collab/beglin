#include "mlx_moe.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    constexpr long OUT = 5;
    constexpr long IN = 128;
    constexpr long NG = IN / 64;
    constexpr int BATCH = 3;
    const size_t cells = (size_t)OUT * (size_t)NG;

    std::vector<uint8_t> bits(cells);
    std::vector<float> scales(cells);
    std::vector<uint32_t> offsets(cells + 1, 0);
    std::vector<uint8_t> planes;
    std::vector<std::vector<int>> codes(cells, std::vector<int>(64));

    for (size_t cell = 0; cell < cells; ++cell) {
        int n = 3 + (int)(cell % 4); // n3,n4,n5,n6 mixed in one tensor
        bits[cell] = (uint8_t)n;
        scales[cell] = 0.001f * (float)(1 + cell);
        offsets[cell] = (uint32_t)planes.size();
        size_t base = planes.size();
        planes.resize(base + (size_t)n * 8u, 0);
        int levels = 1 << n;
        int bias = 1 << (n - 1);
        for (int p = 0; p < 64; ++p) {
            int u = (p * 7 + (int)cell * 11 + 3) % levels;
            int code = u - bias;
            codes[cell][p] = code;
            for (int j = 0; j < n; ++j) {
                if ((u >> j) & 1) {
                    planes[base + (size_t)j * 8u + (size_t)(p >> 3)] |=
                        (uint8_t)(1u << (p & 7));
                }
            }
        }
        offsets[cell + 1] = (uint32_t)planes.size();
    }

    std::vector<float> x((size_t)BATCH * IN);
    for (int b = 0; b < BATCH; ++b) {
        for (long i = 0; i < IN; ++i) {
            x[(size_t)b * IN + i] =
                std::sin((float)(i + 1) * (0.071f + 0.013f * b)) +
                0.2f * std::cos((float)(i + 1) * (0.037f + 0.005f * b));
        }
    }

    std::vector<float> cpu((size_t)BATCH * OUT, 0.0f);
    for (int b = 0; b < BATCH; ++b) {
        for (long row = 0; row < OUT; ++row) {
            double acc = 0.0;
            for (long g = 0; g < NG; ++g) {
                size_t cell = (size_t)row * NG + (size_t)g;
                for (int p = 0; p < 64; ++p) {
                    float w = (float)codes[cell][p] * scales[cell];
                    acc += (double)w * (double)x[(size_t)b * IN + (size_t)g * 64u + (size_t)p];
                }
            }
            cpu[(size_t)b * OUT + (size_t)row] = (float)acc;
        }
    }

    std::vector<float> gpu((size_t)BATCH * OUT, 0.0f);
    if (!mlx_gpu_qng64_mixed_dense_probe(
            planes.data(), (long)planes.size(), offsets.data(), bits.data(), scales.data(),
            OUT, IN, x.data(), BATCH, gpu.data())) {
        std::fprintf(stderr, "mixed qNg64 MLX probe call failed\n");
        return 2;
    }

    double max_abs = 0.0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs((double)cpu[i] - (double)gpu[i]));
    }
    const bool pass = max_abs < 5e-4;
    std::printf(
        "{\"status\":\"%s\",\"out\":%ld,\"in\":%ld,\"batch\":%d,"
        "\"group_size\":64,\"bits_min\":3,\"bits_max\":6,"
        "\"planes_bytes\":%zu,\"dense_n6_bytes\":%zu,"
        "\"max_abs_error\":%.9g}\n",
        pass ? "PASS" : "FAIL", OUT, IN, BATCH, planes.size(),
        cells * (size_t)6 * 8u, max_abs);
    return pass ? 0 : 3;
}
