/* Smoke test for the regression scaffold: uniform registrars that are known-good must decode close to the fp32 weights. */
#define main qwen_infer_real_main
#include "qwen_infer.c"
#undef main
#include "tests/regression/reg_common.h"

int main(void) {
    reg_init();
    MoeAFTensor *q8 = st_register_moe_experts_q8g64_as("L%d.e%d.w", 0, REG_E, "scaffold_q8");
    for (int e = 0; e < REG_E; e++) {
        double r = reg_decode_relL2(q8, e, REG_W[e]);
        REG_CHECK(r < 0.05, "uniform q8g64 expert %d decode rel-L2 = %.5f (< 0.05)", e, r);
    }
    MoeAFTensor *q7 = st_register_moe_experts_qNg64_as("L%d.e%d.w", 0, REG_E, 7, "scaffold_q7");
    for (int e = 0; e < REG_E; e++) {
        double r = reg_decode_relL2(q7, e, REG_W[e]);
        REG_CHECK(r < 0.1, "uniform qNg64 n=7 expert %d decode rel-L2 = %.5f (< 0.1)", e, r);
    }
    return reg_fail;
}
