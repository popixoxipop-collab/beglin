#include "mlx_moe.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr const char *kName = "g2.synthetic.weight";
constexpr long kE = 1;
constexpr long kOut = 8;
constexpr long kIn = 64;
constexpr long kNg = 1;
constexpr long kPackedOff = 0;
constexpr long kScaleOff = 2048;
constexpr long kBiasOff = 3072;
constexpr size_t kBlobBytes = 4096;

struct Fixture {
    int bits;
    std::vector<uint8_t> blob;

    explicit Fixture(int b) : bits(b), blob(kBlobBytes, 0) {
        // Every quantized representation reads one fp32 scale per row/group.
        // A positive non-zero scale avoids degenerate metadata while registry
        // transition correctness remains independent of numeric weight values.
        if (bits != 16 && bits != 32) {
            for (long row = 0; row < kOut; ++row) {
                float scale = 1.0f;
                std::memcpy(blob.data() + kScaleOff + row * sizeof(float),
                            &scale, sizeof(scale));
                float bias = 0.0f;
                std::memcpy(blob.data() + kBiasOff + row * sizeof(float),
                            &bias, sizeof(bias));
            }
        }
    }
};

int expected_kind(int bits) {
    if (bits == 16 || bits == 32) return 2;
    if (bits == 7 || (bits >= 9 && bits <= 15)) return 3;
    return 1;
}

bool bind_fixture(Fixture &f) {
    if (!mlx_gpu_bind_af(
            f.blob.data(), static_cast<long>(f.blob.size()), kName,
            kE, kOut, kIn, kNg,
            kPackedOff, kScaleOff, kBiasOff, f.bits)) {
        std::fprintf(stderr, "FAIL bind bits=%d\n", f.bits);
        return false;
    }
    return true;
}

bool expect_binding(int bits, const char *where) {
    int actual_bits = 0;
    int kind = mlx_gpu_binding_kind(kName, &actual_bits);
    int want_kind = expected_kind(bits);
    if (kind != want_kind || actual_bits != bits) {
        std::fprintf(
            stderr,
            "FAIL %s: expected kind=%d bits=%d, got kind=%d bits=%d\n",
            where, want_kind, bits, kind, actual_bits);
        return false;
    }
    return true;
}

bool transition(Fixture &from, Fixture &to) {
    if (!bind_fixture(from) || !expect_binding(from.bits, "baseline")) return false;
    if (!mlx_gpu_synchronize()) {
        std::fprintf(stderr, "FAIL synchronize before snapshot %d->%d\n",
                     from.bits, to.bits);
        return false;
    }

    uint64_t sid = 0;
    if (!mlx_gpu_snapshot_binding(kName, &sid) || sid == 0) {
        std::fprintf(stderr, "FAIL snapshot %d->%d\n", from.bits, to.bits);
        return false;
    }
    if (mlx_gpu_binding_snapshot_count() != 1) {
        std::fprintf(stderr, "FAIL snapshot count after create\n");
        return false;
    }

    if (!bind_fixture(to) || !expect_binding(to.bits, "candidate")) return false;
    if (!mlx_gpu_synchronize()) {
        std::fprintf(stderr, "FAIL synchronize before restore %d->%d\n",
                     from.bits, to.bits);
        return false;
    }
    if (!mlx_gpu_restore_binding_snapshot(sid)) {
        std::fprintf(stderr, "FAIL restore %d->%d->%d\n",
                     from.bits, to.bits, from.bits);
        return false;
    }
    if (!expect_binding(from.bits, "restored")) return false;
    if (!mlx_gpu_drop_binding_snapshot(sid)) {
        std::fprintf(stderr, "FAIL snapshot drop\n");
        return false;
    }
    if (mlx_gpu_binding_snapshot_count() != 0) {
        std::fprintf(stderr, "FAIL snapshot leak after restore\n");
        return false;
    }
    std::printf("PASS %d -> %d -> %d\n", from.bits, to.bits, from.bits);
    return true;
}

}  // namespace

int main() {
    if (!mlx_gpu_available()) {
        std::fprintf(stderr, "FAIL MLX/Metal GPU unavailable\n");
        return 2;
    }

    Fixture q4(4);
    Fixture q5(5);
    Fixture q6(6);
    Fixture q7(7);
    Fixture f16(16);

    if (!transition(q4, q5)) return 10;
    if (!transition(q4, q6)) return 11;
    if (!transition(q4, q7)) return 12;
    if (!transition(q7, f16)) return 13;
    if (!transition(q6, q7)) return 14;

    // Exit criterion from the GPU plan: repeated restore must neither drift
    // representation identity nor leak snapshots. Keep all fixture buffers
    // alive for the entire soak because restored mx::array objects may retain
    // zero-copy references to their original host backing storage.
    for (int i = 0; i < 100; ++i) {
        if (!transition(q4, q7)) {
            std::fprintf(stderr, "FAIL soak iteration=%d\n", i);
            return 20;
        }
    }

    if (mlx_gpu_binding_snapshot_count() != 0) {
        std::fprintf(stderr, "FAIL final snapshot leak count=%d\n",
                     mlx_gpu_binding_snapshot_count());
        return 21;
    }
    std::printf("PASS G2 binding transition matrix + 100-cycle restore soak\n");
    return 0;
}
