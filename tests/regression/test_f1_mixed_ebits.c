/* F1 (GitHub issue #64) regression: st_register_moe_experts_mixed_as() must pack every per-expert
 * width it accepts in the layout moe_decode_af()/moe_matvec_af_row() read for that width.
 *
 * Defect: ebits[e] in {2,3,5,6,7} was packed with the q4g64 nibble quantizer (in/2 bytes per row)
 * while both decoders read it as a qNg64 bit-plane (ng*n*8 bytes per row) -> silent garbage
 * (n=5 decode rel-L2 6.36 vs 0.12 for a correct 5-bit expert; n=7: 26.1 vs 0.030).
 *
 * Part A: for each w in {2,3,5,6,7} and both expert orders ({w,8} puts the bit-plane expert at
 *         prefix offset 0, {8,w} at a non-zero epacked_off), the mixed tensor's bit-plane expert
 *         must decode like a uniform st_register_moe_experts_qNg64_as(w) tensor (<= 1.5x of its
 *         rel-L2, decode AND matvec) and its 8-bit expert like a uniform q8g64 tensor.
 *         The 1.5x bound alone is too loose for w=2 (the unfixed code measures 1.15x/1.36x there),
 *         so the test also requires the identical quantizer output (the mixed expert is fed the same
 *         floats as the uniform one, so it must decode to the exact same values) and the exact
 *         packed_bytes the bit-plane layout implies.
 * Part B: controls that were already correct and must stay numerically identical:
 *         {4,8}, {8,8}, {16,32}, {4,16}. Pinned to 5 decimals, taken from the unfixed code.
 */
#define main qwen_infer_real_main
#include "qwen_infer.c"
#undef main
#include "tests/regression/reg_common.h"

/* The 4-bit controls below use the engine's real quantizer (residual / error-feedback variant). */
static const char *Q4_QUANTIZER_USED = "gguf_quantize_q4g64_error_feedback";

static MoeAFTensor *reg_mixed(int b0, int b1, const char *engine_name) {
    int eb[REG_E] = {b0, b1};
    return st_register_moe_experts_mixed_as("L%d.e%d.w", 0, REG_E, eb, engine_name);
}

static double dec(MoeAFTensor *t, int e) { return reg_decode_relL2(t, e, REG_W[e]); }
static double mv(MoeAFTensor *t, int e)  { return reg_matvec_relL2(t, e, REG_W[e]); }

/* Part A for one (bit-plane width w, position k of the w-expert in the 2-expert mixed tensor). */
static void check_bitplane_expert(int w, int k, MoeAFTensor *uni, MoeAFTensor *q8) {
    char nm[96];
    snprintf(nm, sizeof nm, "f1_mixed_w%d_k%d", w, k);
    MoeAFTensor *m = k == 0 ? reg_mixed(w, 8, nm) : reg_mixed(8, w, nm);
    int o = 1 - k;   /* the 8-bit expert's index */

    long expect_bytes = (long)REG_OUT * ((long)(REG_IN / 64) * w * 8) + (long)REG_OUT * REG_IN;
    REG_CHECK(m->packed_bytes == expect_bytes,
              "w=%d order=%s: packed_bytes=%ld (expected %ld = bit-plane expert + int8 expert)",
              w, k == 0 ? "{w,8}" : "{8,w}", m->packed_bytes, expect_bytes);

    double ud = dec(uni, k), um = mv(uni, k), md = dec(m, k), mm = mv(m, k);
    printf("        w=%d %s expert%d: decode mixed=%.5f uniform=%.5f | matvec mixed=%.5f uniform=%.5f\n",
           w, k == 0 ? "{w,8}" : "{8,w}", k, md, ud, mm, um);
    REG_CHECK(md <= 1.5 * ud, "w=%d %s bit-plane expert decode rel-L2 %.5f <= 1.5 x uniform %.5f",
              w, k == 0 ? "{w,8}" : "{8,w}", md, ud);
    REG_CHECK(mm <= 1.5 * um, "w=%d %s bit-plane expert matvec rel-L2 %.5f <= 1.5 x uniform %.5f",
              w, k == 0 ? "{w,8}" : "{8,w}", mm, um);
    REG_CHECK(fabs(md - ud) < 1e-9 && fabs(mm - um) < 1e-9,
              "w=%d %s mixed expert decodes identically to the uniform qNg64 tensor (|d decode|=%.2e |d matvec|=%.2e)",
              w, k == 0 ? "{w,8}" : "{8,w}", fabs(md - ud), fabs(mm - um));

    double u8d = dec(q8, o), u8m = mv(q8, o), m8d = dec(m, o), m8m = mv(m, o);
    REG_CHECK(fabs(m8d - u8d) < 1e-9 && fabs(m8m - u8m) < 1e-9,
              "w=%d %s int8 expert%d matches uniform q8g64 (decode %.5f vs %.5f, matvec %.5f vs %.5f)",
              w, k == 0 ? "{w,8}" : "{8,w}", o, m8d, u8d, m8m, u8m);
}

/* A q4g64 nibble expert sharing a tensor with a bit-plane expert: the prefix-sum offsets (epacked_off)
 * must keep the 4-bit expert addressable whether it sits before or after the bit-plane expert. */
static void check_partner_q4(int w, MoeAFTensor *uni, MoeAFTensor *q4) {
    for (int k = 0; k < 2; k++) {
        char nm[96];
        snprintf(nm, sizeof nm, "f1_mixed4_w%d_k%d", w, k);
        MoeAFTensor *m = k == 0 ? reg_mixed(w, 4, nm) : reg_mixed(4, w, nm);
        int o = 1 - k;
        REG_CHECK(fabs(dec(m, k) - dec(uni, k)) < 1e-9 && fabs(mv(m, k) - mv(uni, k)) < 1e-9,
                  "w=%d %s bit-plane expert%d identical to uniform qNg64 (decode %.5f)", w, k == 0 ? "{w,4}" : "{4,w}", k, dec(m, k));
        REG_CHECK(fabs(dec(m, o) - dec(q4, o)) < 1e-9 && fabs(mv(m, o) - mv(q4, o)) < 1e-9,
                  "w=%d %s q4g64 expert%d identical to uniform q4g64 (decode %.5f)", w, k == 0 ? "{w,4}" : "{4,w}", o, dec(m, o));
    }
}

/* Pinned control: both experts' decode and matvec rel-L2 to 5 decimals (values from the unfixed code). */
typedef struct { int b0, b1; double d0, d1, m0, m1; } Control;

static void check_control(const Control *c) {
    char nm[64];
    snprintf(nm, sizeof nm, "f1_ctl_%d_%d", c->b0, c->b1);
    MoeAFTensor *m = reg_mixed(c->b0, c->b1, nm);
    double got[4] = { dec(m, 0), dec(m, 1), mv(m, 0), mv(m, 1) };
    double want[4] = { c->d0, c->d1, c->m0, c->m1 };
    const char *lab[4] = { "decode e0", "decode e1", "matvec e0", "matvec e1" };
    for (int i = 0; i < 4; i++)
        REG_CHECK(fabs(got[i] - want[i]) < 6e-6, "control {%d,%d} %s = %.5f (pinned %.5f)",
                  c->b0, c->b1, lab[i], got[i], want[i]);
}

int main(void) {
    reg_init();
    printf("  q4 quantizer under test: %s\n", Q4_QUANTIZER_USED);

    MoeAFTensor *q8 = st_register_moe_experts_q8g64_as("L%d.e%d.w", 0, REG_E, "f1_uniform_q8");
    MoeAFTensor *q4 = st_register_moe_experts_q4g64_as("L%d.e%d.w", 0, REG_E, "f1_uniform_q4");

    printf("-- Part A: mixed 2/3/5/6/7-bit experts vs uniform qNg64 --\n");
    static const int WS[5] = {2, 3, 5, 6, 7};
    static const double UNIFORM_DECODE_BASELINE[5] = {1.154, 0.551, 0.122, 0.062, 0.030};
    for (int i = 0; i < 5; i++) {
        int w = WS[i];
        char nm[64];
        snprintf(nm, sizeof nm, "f1_uniform_q%d", w);
        MoeAFTensor *uni = st_register_moe_experts_qNg64_as("L%d.e%d.w", 0, REG_E, w, nm);
        double ud = dec(uni, 0);
        REG_CHECK(fabs(ud - UNIFORM_DECODE_BASELINE[i]) < 0.0015,
                  "fixture sanity: uniform qNg64 n=%d expert0 decode rel-L2 %.5f ~ %.3f", w, ud, UNIFORM_DECODE_BASELINE[i]);
        check_bitplane_expert(w, 0, uni, q8);
        check_bitplane_expert(w, 1, uni, q8);
        check_partner_q4(w, uni, q4);
    }

    printf("-- Part B: controls that must not change --\n");
    REG_CHECK(fabs(dec(q4, 0) - 0.27455) < 6e-6, "uniform q4g64 expert0 decode = %.5f (pinned 0.27455)", dec(q4, 0));
    REG_CHECK(fabs(dec(q8, 1) - 0.01121) < 6e-6, "uniform q8g64 expert1 decode = %.5f (pinned 0.01121)", dec(q8, 1));
    {
        MoeAFTensor *m48 = reg_mixed(4, 8, "f1_ctl_4_8_vs_uniform");
        REG_CHECK(fabs(dec(m48, 0) - dec(q4, 0)) < 1e-9 && fabs(mv(m48, 0) - mv(q4, 0)) < 1e-9,
                  "{4,8} expert0 identical to uniform q4g64");
        REG_CHECK(fabs(dec(m48, 1) - dec(q8, 1)) < 1e-9 && fabs(mv(m48, 1) - mv(q8, 1)) < 1e-9,
                  "{4,8} expert1 identical to uniform q8g64");
    }
    static const Control CONTROLS[] = {
        /* b0 b1   decode e0  decode e1  matvec e0  matvec e1 */
        { 4,  8,  0.27455, 0.01121, 0.16776, 0.00924 },
        { 8,  8,  0.01042, 0.01121, 0.01097, 0.00924 },
        { 16, 32, 0.00020, 0.00000, 0.00017, 0.00000 },
        { 4,  16, 0.27455, 0.00020, 0.16776, 0.00032 },
    };
    for (size_t i = 0; i < sizeof CONTROLS / sizeof CONTROLS[0]; i++) check_control(&CONTROLS[i]);
    return reg_fail;
}
