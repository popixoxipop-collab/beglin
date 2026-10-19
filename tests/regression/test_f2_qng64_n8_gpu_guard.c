/* Companion to test_f2_qng64_n8.c (issue #65): the GPU binder (mlx_moe.cpp) only receives `bits`, and its bits==8
 * branch is the raw-int8 q8g64 layout, so a qNg64 n=8 tensor must never reach it with bits==8. The call sites pass
 * moe_gpu_bind_bits(t); rebind/promotion use moe_gpu_qng64_n_supported(n). Both are plain CPU functions, so the guard
 * logic is testable here without MLX/Metal. This does NOT exercise the GPU binder itself (GPU path is unrun on this
 * host); it pins the predicate the binder callers rely on. Separate file because these symbols do not exist on the
 * unfixed base (the decode assertions in test_f2_qng64_n8.c are the ones that fail there). */
#define main qwen_infer_real_main
#include "qwen_infer.c"
#undef main
#include "tests/regression/reg_common.h"

int main(void) {
    reg_init();
    MoeAFTensor *q8n = st_register_moe_experts_qNg64_as("L%d.e%d.w", 0, REG_E, 8, "g_qn8");
    MoeAFTensor *dn8 = st_register_moe_dense_af_qNg64_as("dense.w", 8, "g_dense_qn8");
    MoeAFTensor *q7 = st_register_moe_experts_qNg64_as("L%d.e%d.w", 0, REG_E, 7, "g_qn7");
    MoeAFTensor *q9 = st_register_moe_experts_qNg64_as("L%d.e%d.w", 0, REG_E, 9, "g_qn9");
    MoeAFTensor *q8 = st_register_moe_experts_q8g64_as("L%d.e%d.w", 0, REG_E, "g_q8");
    MoeAFTensor *dq8 = st_register_moe_dense_af_q8g64_as("dense.w", "g_dense_q8");

    REG_CHECK(q8n->qng64 == 1 && dn8->qng64 == 1, "qNg64 n=8 registrars set the bit-plane marker (experts=%d dense=%d)", q8n->qng64, dn8->qng64);
    REG_CHECK(q7->qng64 == 1 && q9->qng64 == 1, "qNg64 n=7 / n=9 are marked too (%d, %d)", q7->qng64, q9->qng64);
    REG_CHECK(q8->qng64 == 0 && dq8->qng64 == 0, "q8g64 registrars leave the marker clear (experts=%d dense=%d)", q8->qng64, dq8->qng64);

    REG_CHECK(moe_gpu_bind_bits(q8n) == 0, "binder bits for qNg64 n=8 experts is refused (0), got %d", moe_gpu_bind_bits(q8n));
    REG_CHECK(moe_gpu_bind_bits(dn8) == 0, "binder bits for dense-AF qNg64 n=8 is refused (0), got %d", moe_gpu_bind_bits(dn8));
    REG_CHECK(moe_gpu_bind_bits(q8) == 8, "binder bits for q8g64 experts stays 8, got %d", moe_gpu_bind_bits(q8));
    REG_CHECK(moe_gpu_bind_bits(dq8) == 8, "binder bits for q8g64 dense stays 8, got %d", moe_gpu_bind_bits(dq8));
    REG_CHECK(moe_gpu_bind_bits(q7) == 7 && moe_gpu_bind_bits(q9) == 9, "binder bits for qNg64 n=7/n=9 unchanged (%d, %d)",
              moe_gpu_bind_bits(q7), moe_gpu_bind_bits(q9));

    int cpu_ok = 1, gpu_ok = 1;
    for (int n = 0; n <= 17; n++) {
        int want_cpu = (n == 2 || n == 3 || (n >= 5 && n <= 15));
        int want_gpu = want_cpu && n != 8;
        if (moe_qng64_n_supported(n) != want_cpu) cpu_ok = 0;
        if (moe_gpu_qng64_n_supported(n) != want_gpu) gpu_ok = 0;
    }
    REG_CHECK(cpu_ok, "CPU qNg64 width set is still {2,3,5..15} (n=8 stays supported on CPU)");
    REG_CHECK(gpu_ok, "GPU qNg64 width set is the CPU set minus 8");
    return reg_fail;
}
