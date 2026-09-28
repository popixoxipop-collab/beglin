#include "mlx_moe.h"

#include <cstdio>
#include <cstring>
#include <vector>

static int bind_and_expect(const char *name, int n, int expected_representation) {
    std::vector<unsigned char> blob(1024, 0);
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
    if (bound_n != n || representation != expected_representation) {
        std::fprintf(
            stderr,
            "binding mismatch name=%s requested=%d bound=%d rep=%d expected_rep=%d\n",
            name, n, bound_n, representation, expected_representation);
        return 0;
    }

    int observed_n = -1;
    int observed_rep = -1;
    if (!mlx_gpu_assert_binding(name, n, &observed_n, &observed_rep)) {
        std::fprintf(stderr, "exact binding assertion failed name=%s n=%d\n", name, n);
        return 0;
    }
    if (observed_n != n || observed_rep != expected_representation) return 0;

    if (mlx_gpu_assert_binding(name, n == 7 ? 6 : 7, &observed_n, &observed_rep)) {
        std::fprintf(stderr, "mismatch did not fail closed name=%s n=%d\n", name, n);
        return 0;
    }
    if (observed_n != n || observed_rep != expected_representation) return 0;
    return 1;
}

int main() {
    if (!mlx_gpu_available()) {
        std::fprintf(stderr, "MLX unavailable\n");
        return 77;
    }

    // n=5/6 use MLX-native quantized representation; n=7 uses qNg64.
    if (!bind_and_expect("p8b.synthetic.n5", 5, MLX_GPU_BINDING_NATIVE_QUANT)) return 1;
    if (!bind_and_expect("p8b.synthetic.n6", 6, MLX_GPU_BINDING_NATIVE_QUANT)) return 1;
    if (!bind_and_expect("p8b.synthetic.n7", 7, MLX_GPU_BINDING_QNG64)) return 1;

    // Rebinding the same runtime tensor must switch truth atomically.
    const char *name = "p8b.synthetic.rebind";
    if (!bind_and_expect(name, 5, MLX_GPU_BINDING_NATIVE_QUANT)) return 1;
    if (!bind_and_expect(name, 7, MLX_GPU_BINDING_QNG64)) return 1;
    int bound_n = -1, rep = -1;
    if (mlx_gpu_assert_binding(name, 5, &bound_n, &rep)) return 1;
    if (bound_n != 7 || rep != MLX_GPU_BINDING_QNG64) return 1;
    if (!bind_and_expect(name, 6, MLX_GPU_BINDING_NATIVE_QUANT)) return 1;
    if (mlx_gpu_assert_binding(name, 7, &bound_n, &rep)) return 1;
    if (bound_n != 6 || rep != MLX_GPU_BINDING_NATIVE_QUANT) return 1;

    if (mlx_gpu_get_binding_state("p8b.synthetic.missing", &bound_n, &rep)) return 1;
    if (bound_n != -1 || rep != MLX_GPU_BINDING_NONE) return 1;

    std::puts("P8-B runtime binding probe PASS n5=PASS n6=PASS n7=PASS rebind=PASS mismatch=PASS");
    return 0;
}
