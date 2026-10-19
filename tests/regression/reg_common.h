/* Shared helpers for tests/regression/test_*.c.
 *
 * Every test does:
 *     #define main qwen_infer_real_main
 *     #include "qwen_infer.c"
 *     #undef main
 *     #include "tests/regression/reg_common.h"
 * so the REAL static registrars/decoders of qwen_infer.c run (no copy of the logic is tested).
 * Build/run with tests/regression/run_regression.sh (SME2/KleidiAI symbols stay unresolved; scalar paths only).
 */
#ifndef REG_COMMON_H
#define REG_COMMON_H

#define REG_E 2
#define REG_OUT 16
#define REG_IN 128

static float REG_W[REG_E][REG_OUT][REG_IN];   /* tensors L0.e0.w, L0.e1.w */
static float REG_DW[REG_OUT][REG_IN];         /* tensor dense.w */
static int reg_fail = 0;

/* Allocate the AF tensor table the way the real loaders do (main() owns its lifetime), open the fixture and
 * load the fp32 reference weights. Requires env REG_ST=<path to tiny.safetensors>. */
static void reg_init(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    g_moe_af = calloc(MOE_MAX_AF_TENSORS, sizeof(MoeAFTensor));
    const char *p = getenv("REG_ST");
    if (!p) { fprintf(stderr, "REG_ST not set\n"); exit(2); }
    g_st_moe = safetensors_open_multi(p);
    const char *nm[3] = {"L0.e0.w", "L0.e1.w", "dense.w"};
    for (int i = 0; i < 3; i++) {
        SafetensorsFile *sh = NULL;
        const SafetensorsInfo *t = safetensors_multi_find_tensor(g_st_moe, nm[i], &sh);
        if (!t) { fprintf(stderr, "fixture tensor %s missing\n", nm[i]); exit(2); }
        float *dst = i < 2 ? &REG_W[i][0][0] : &REG_DW[0][0];
        safetensors_dequant_row(t->dtype, safetensors_tensor_data(sh, t), dst, (uint64_t)(REG_OUT * REG_IN));
    }
}

/* Relative L2 of moe_decode_af() for expert e against the fp32 reference rows `ref` ([REG_OUT][REG_IN]). */
static double reg_decode_relL2(MoeAFTensor *t, long e, const float (*ref)[REG_IN]) {
    double num = 0, den = 0;
    for (int r = 0; r < REG_OUT; r++)
        for (int c = 0; c < REG_IN; c++) {
            float d = moe_decode_af(NULL, t, e, r, c), o = ref[r][c];
            num += (double)(d - o) * (d - o);
            den += (double)o * o;
        }
    return sqrt(num / den);
}

/* Relative L2 of moe_matvec_af_row() output for expert e against the fp32 matvec with the fixed probe vector. */
static double reg_matvec_relL2(MoeAFTensor *t, long e, const float (*ref)[REG_IN]) {
    float x[REG_IN];
    for (int c = 0; c < REG_IN; c++) x[c] = (float)((c * 37) % 17 - 8) / 8.0f;
    double num = 0, den = 0;
    for (int r = 0; r < REG_OUT; r++) {
        double yref = 0;
        for (int c = 0; c < REG_IN; c++) yref += (double)ref[r][c] * x[c];
        double y = moe_matvec_af_row(NULL, t, e, r, x);
        num += (y - yref) * (y - yref);
        den += yref * yref;
    }
    return sqrt(num / (den > 0 ? den : 1));
}

#define REG_CHECK(cond, ...) do {                                  \
        if (cond) { printf("  ok:   "); }                         \
        else      { printf("  FAIL: "); reg_fail = 1; }            \
        printf(__VA_ARGS__); printf("\n");                         \
    } while (0)

#endif /* REG_COMMON_H */
