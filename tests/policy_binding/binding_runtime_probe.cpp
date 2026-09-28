#include "mlx_moe.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<std::vector<unsigned char>> g_backing_storage;

static int expected_representation(int n) {
    return n == 7 ? MLX_GPU_BINDING_QNG64 : MLX_GPU_BINDING_NATIVE_QUANT;
}

static int bind_and_expect(const char *name, int n, int expected_rep) {
    g_backing_storage.emplace_back(1024, 0);
    std::vector<unsigned char> &blob = g_backing_storage.back();
    float scale = 1.0f;
    std::memcpy(blob.data() + 512, &scale, sizeof(scale));

    if (!mlx_gpu_bind_af(
            blob.data(), (long)blob.size(), name,
            1, 1, 64, 1,
            0, 512, 768, n)) {
        std::fprintf(stderr, "bind failed name=%s n=%d\n", name, n);
        return 0;
    }

    int bound_n = -1;
    int representation = -1;
    if (!mlx_gpu_get_binding_state(name, &bound_n, &representation)) {
        std::fprintf(stderr, "binding state missing name=%s n=%d\n", name, n);
        return 0;
    }
    if (bound_n != n || representation != expected_rep) {
        std::fprintf(
            stderr,
            "binding mismatch name=%s requested=%d bound=%d rep=%d expected_rep=%d\n",
            name, n, bound_n, representation, expected_rep);
        return 0;
    }

    int observed_n = -1;
    int observed_rep = -1;
    if (!mlx_gpu_assert_binding(name, n, &observed_n, &observed_rep)) {
        std::fprintf(stderr, "exact binding assertion failed name=%s n=%d\n", name, n);
        return 0;
    }
    if (observed_n != n || observed_rep != expected_rep) return 0;

    const int wrong_n = n == 7 ? 6 : 7;
    if (mlx_gpu_assert_binding(name, wrong_n, &observed_n, &observed_rep)) {
        std::fprintf(
            stderr,
            "mismatch did not fail closed name=%s requested=%d actual=%d\n",
            name, wrong_n, n);
        return 0;
    }
    if (observed_n != n || observed_rep != expected_rep) return 0;
    return 1;
}

static int probe_role(
    const char *role,
    int layer,
    const char *tensor_name,
    int n,
    int expected_rep) {
    if (!bind_and_expect(tensor_name, n, expected_rep)) return 0;

    int bound_n = -1;
    int representation = -1;
    if (!mlx_gpu_get_role_binding_state(
            role, layer, &bound_n, &representation)) {
        std::fprintf(
            stderr,
            "role binding state missing role=%s layer=%d n=%d\n",
            role, layer, n);
        return 0;
    }
    if (bound_n != n || representation != expected_rep) {
        std::fprintf(
            stderr,
            "role binding mismatch role=%s layer=%d requested=%d bound=%d rep=%d expected=%d\n",
            role, layer, n, bound_n, representation, expected_rep);
        return 0;
    }

    if (!mlx_gpu_assert_role_binding(
            role, layer, n, &bound_n, &representation)) {
        std::fprintf(
            stderr,
            "role exact assertion failed role=%s layer=%d n=%d\n",
            role, layer, n);
        return 0;
    }

    const int wrong_n = n == 7 ? 6 : 7;
    if (mlx_gpu_assert_role_binding(
            role, layer, wrong_n, &bound_n, &representation)) {
        std::fprintf(
            stderr,
            "role mismatch did not fail closed role=%s layer=%d requested=%d actual=%d\n",
            role, layer, wrong_n, n);
        return 0;
    }
    if (bound_n != n || representation != expected_rep) return 0;

    std::printf(
        "P8_B_CELL_PASS role=%s layer=%d requested_n=%d bound_n=%d representation=%d\n",
        role, layer, n, bound_n, representation);
    return 1;
}

int main() {
    if (!mlx_gpu_available()) {
        std::fprintf(stderr, "MLX unavailable\n");
        return 77;
    }

    struct Cell {
        const char *role;
        int layer;
        const char *tensor_name;
    };
    const Cell cells[] = {
        {
            "kv_a_proj_with_mqa",
            11,
            "model.layers.11.self_attn.kv_a_proj_with_mqa",
        },
        {
            "shared_up_proj",
            3,
            "model.layers.3.mlp.shared_experts.up_proj",
        },
    };

    int passed = 0;
    for (const Cell &cell : cells) {
        for (int n : {4, 5, 6, 7}) {
            if (!probe_role(
                    cell.role,
                    cell.layer,
                    cell.tensor_name,
                    n,
                    expected_representation(n))) {
                return 1;
            }
            passed++;
        }
    }

    // The last rebind on each canonical tensor is n=7; requesting an older n
    // must not observe a stale native-quant entry.
    int bound_n = -1;
    int rep = -1;
    if (mlx_gpu_assert_role_binding(
            "kv_a_proj_with_mqa", 11, 6, &bound_n, &rep)) return 1;
    if (bound_n != 7 || rep != MLX_GPU_BINDING_QNG64) return 1;
    if (mlx_gpu_assert_role_binding(
            "shared_up_proj", 3, 6, &bound_n, &rep)) return 1;
    if (bound_n != 7 || rep != MLX_GPU_BINDING_QNG64) return 1;

    if (mlx_gpu_get_binding_state(
            "p8b.synthetic.missing", &bound_n, &rep)) return 1;
    if (bound_n != -1 || rep != MLX_GPU_BINDING_NONE) return 1;

    if (passed != 8) return 1;
    std::puts(
        "P8_POLICY_ATTRIBUTION_8_OF_8 "
        "kv_a_L11=4/4 shared_up_L3=4/4 "
        "n4=PASS n5=PASS n6=PASS n7=PASS "
        "role_api=PASS rebind=PASS mismatch=PASS");
    return 0;
}
