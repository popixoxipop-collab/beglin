#include "mlx_moe.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

template <typename T>
static bool read_all(const std::string &path, std::vector<T> &out) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    std::fseek(fp, 0, SEEK_END);
    long bytes = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (bytes < 0 || (bytes % (long)sizeof(T)) != 0) {
        std::fclose(fp);
        return false;
    }
    out.resize((size_t)bytes / sizeof(T));
    const bool ok = out.empty() || std::fread(out.data(), sizeof(T), out.size(), fp) == out.size();
    std::fclose(fp);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s fixture_dir\n", argv[0]);
        return 2;
    }
    constexpr long OUT = 896;
    constexpr long IN = 896;
    constexpr long NG = IN / 64;
    const size_t cells = (size_t)OUT * (size_t)NG;
    const std::string root = argv[1];

    std::vector<uint8_t> planes, bits;
    std::vector<uint32_t> offsets;
    std::vector<float> scales, x, cpu, gpu(OUT, 0.0f);

    if (!read_all(root + "/planes.bin", planes)) return 3;
    if (!read_all(root + "/offsets.bin", offsets)) return 4;
    if (!read_all(root + "/bits.bin", bits)) return 5;
    if (!read_all(root + "/scales.bin", scales)) return 6;
    if (!read_all(root + "/input.bin", x)) return 7;
    if (!read_all(root + "/cpu_expected.bin", cpu)) return 8;

    if (offsets.size() != cells + 1 || bits.size() != cells ||
        scales.size() != cells || x.size() != IN || cpu.size() != OUT) {
        std::fprintf(stderr, "fixture shape mismatch\n");
        return 9;
    }
    if ((size_t)offsets.back() != planes.size()) return 10;

    if (!mlx_gpu_qng64_mixed_dense_probe(
            planes.data(), (long)planes.size(), offsets.data(), bits.data(),
            scales.data(), OUT, IN, x.data(), 1, gpu.data())) {
        std::fprintf(stderr, "mixed real-Qwen Metal probe failed\n");
        return 11;
    }

    double max_abs = 0.0;
    double rms = 0.0;
    for (long i = 0; i < OUT; ++i) {
        double d = (double)gpu[(size_t)i] - (double)cpu[(size_t)i];
        max_abs = std::max(max_abs, std::fabs(d));
        rms += d * d;
    }
    rms = std::sqrt(rms / (double)OUT);

    int min_bits = 255, max_bits = 0;
    size_t counts[16] = {};
    for (uint8_t b : bits) {
        min_bits = std::min(min_bits, (int)b);
        max_bits = std::max(max_bits, (int)b);
        if (b < 16) counts[b]++;
    }

    const bool pass = max_abs < 5e-4;
    std::printf(
        "{\"status\":\"%s\",\"out\":896,\"in\":896,\"group_size\":64,"
        "\"cell_count\":%zu,\"planes_bytes\":%zu,\"bits_min\":%d,\"bits_max\":%d,"
        "\"n3\":%zu,\"n4\":%zu,\"n5\":%zu,\"n6\":%zu,\"n7\":%zu,\"n8\":%zu,"
        "\"n9\":%zu,\"n10\":%zu,\"max_abs_error\":%.9g,\"rms_error\":%.9g}\n",
        pass ? "PASS" : "FAIL", cells, planes.size(), min_bits, max_bits,
        counts[3], counts[4], counts[5], counts[6], counts[7], counts[8], counts[9], counts[10],
        max_abs, rms);
    return pass ? 0 : 12;
}
