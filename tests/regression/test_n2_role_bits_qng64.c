/* Regression for N2 (issue pending): QWEN_MOE_ROLE_BITS accepts bits in {2,3,4,5,6,7,8,16,32} (moe_load_role_bits()), but
 * st_register_moe_role() only had branches for 32, 16 and 8 plus an `else` that registered q4g64. A requested width of
 * 2/3/5/6/7 was therefore silently registered as 4-bit: a self-consistent tensor (not garbage), but not the precision the
 * user asked for, with no warning. Same family as the first batch's mixed-ebits bug, on the E=1 role path.
 *
 * Asserts, on the real static functions of qwen_infer.c:
 *   - end to end: a role-bits file line "<role> <layer> <bits>" -> moe_load_role_bits() -> moe_role_bits() ->
 *     st_register_moe_role() yields a tensor with bits == requested (and the qNg64 bit-plane marker);
 *   - for every requested n in {2,3,5,6,7}, both attention-style (f32_as_af=1) and embed/lm_head-style (f32_as_af=0)
 *     registration: exactly one tensor is registered, bits == n, sym == 1, qng64 == 1, its decode equals an INDEPENDENT
 *     bit-plane reference read from the packed bytes (layout pinned in gguf_transcode.h), and its decode rel-L2 vs the
 *     fp32 fixture equals the uniform qNg64 value of that width (n=2 1.20871, 3 0.62646, 5 0.13574, 6 0.06342, 7 0.03119);
 *   - controls that must NOT change: bits 4 (q4g64, bits=4, no qNg64 marker), 8 (q8g64 raw int8), 16 (F16 passthrough)
 *     and 32 (F32 passthrough) keep their pre-fix registration and decode values. The control constants were measured
 *     on the unfixed base (fix/integration) with this fixture.
 *
 * Q4_QUANTIZER_USED names the 4-bit quantizer the bits==4 control exercises (error-feedback q4g64). */
#define main qwen_infer_real_main
#include "qwen_infer.c"
#undef main
#include "tests/regression/reg_common.h"

static const char *Q4_QUANTIZER_USED = "gguf_quantize_q4g64_error_feedback";

#define CTRL_TOL 2e-5       /* controls are compared at the 5 decimals recorded from the base run */

/* Print + check one measured value against a pre-fix control. */
static void ctrl(const char *what, double got, double base) {
    REG_CHECK(fabs(got - base) < CTRL_TOL, "%s = %.5f (base control %.5f)", what, got, base);
}

/* Independent bit-plane decode straight from the tensor's packed bytes (layout pinned in gguf_transcode.h: group of 64
 * = n planes of 8 bytes, element i is bit (i&7) of byte (i>>3) of each plane, plane j = bit j of u = code + 2^(n-1)). */
static float ref_planes(const MoeAFTensor *t, long r, long c) {
    int n = t->bits;
    long row_pbytes = t->ng * (long)n * 8;
    const uint8_t *g = t->base + t->packed_off + r * row_pbytes + (c / 64) * (long)n * 8;
    int p = (int)(c % 64), u = 0;
    for (int j = 0; j < n; j++)
        if ((g[j * 8 + (p >> 3)] >> (p & 7)) & 1) u |= 1 << j;
    float sc; memcpy(&sc, t->base + t->scale_off + (r * t->ng + c / 64) * 4, 4);
    return (float)(u - (1 << (n - 1))) * sc;
}

/* Max |engine decode - independent reference| over the whole [REG_OUT][REG_IN] tensor. */
static double ref_diff(MoeAFTensor *t) {
    double mx = 0;
    for (int r = 0; r < REG_OUT; r++)
        for (int c = 0; c < REG_IN; c++) {
            double d = fabs((double)moe_decode_af(NULL, t, 0, r, c) - (double)ref_planes(t, r, c));
            if (d > mx) mx = d;
        }
    return mx;
}

/* Register through the role dispatcher exactly once and return the new tensor (NULL if the count did not move by 1). */
static MoeAFTensor *register_role(const char *engine_name, int bits, const char *role, int f32_as_af) {
    int before = g_moe_naf;
    /* allow_f32 only matters for 16/32 (the F16/F32 passthrough controls); the 2/3/5/6/7 targets are run with it off */
    st_register_moe_role("dense.w", engine_name, bits, role, bits >= 16, f32_as_af);
    return (g_moe_naf == before + 1) ? &g_moe_af[g_moe_naf - 1] : NULL;
}

int main(void) {
    reg_init();
    printf("4-bit control quantizer: %s\n", Q4_QUANTIZER_USED);

    /* uniform qNg64 decode rel-L2 on the dense fixture, as built by st_register_moe_dense_af_qNg64_as() (unchanged by the
     * fix; the role path must reproduce it exactly). The task brief quoted 1.154/0.551/0.122/0.062/0.030, which are the
     * right order of magnitude but not what this fixture measures; the values below are the measured ones. */
    static const int  ns[]   = {2, 3, 5, 6, 7};
    static const double want[] = {1.20871, 0.62646, 0.13574, 0.06342, 0.03119};

    /* --- target: requested widths 2/3/5/6/7 are honoured, both role styles --- */
    for (int i = 0; i < 5; i++) {
        int n = ns[i];
        for (int style = 0; style < 2; style++) {
            char nm[64];
            snprintf(nm, sizeof nm, "n2_role_n%d_%s", n, style ? "attn" : "embed");
            MoeAFTensor *t = register_role(nm, n, style ? "q_proj" : "lm_head", style);
            REG_CHECK(t != NULL, "n=%d %s style: exactly one tensor registered", n, style ? "attn(f32_as_af=1)" : "embed(f32_as_af=0)");
            if (!t) continue;
            REG_CHECK(t->bits == n, "n=%d %s style: registered bits = %d (requested %d)", n, style ? "attn" : "embed", t->bits, n);
            REG_CHECK(t->sym == 1 && t->qng64 == 1, "n=%d %s style: sym=%d qng64=%d (bit-plane qNg64 tensor expected)", n, style ? "attn" : "embed", t->sym, t->qng64);
            double d = reg_decode_relL2(t, 0, REG_DW);
            REG_CHECK(fabs(d - want[i]) < CTRL_TOL, "n=%d %s style: decode rel-L2 = %.5f (uniform qNg64 n=%d = %.5f)", n, style ? "attn" : "embed", d, n, want[i]);
            if (t->bits == n && t->qng64) {
                double x = ref_diff(t);
                REG_CHECK(x == 0.0, "n=%d %s style: decode == independent bit-plane reference (max abs diff %.3g)", n, style ? "attn" : "embed", x);
            }
            /* The role path must be byte-for-byte the SAME tensor the direct qNg64 registrar builds. */
            MoeAFTensor *ref = st_register_moe_dense_af_qNg64_as("dense.w", n, "n2_direct");
            int same = t->bits == ref->bits && t->qng64 == ref->qng64 && t->packed_bytes == ref->packed_bytes && t->ng == ref->ng;
            if (same) same = memcmp(t->base, ref->base, (size_t)ref->packed_bytes + (size_t)(ref->out * ref->ng * 4)) == 0;   /* only compare equally-sized buffers */
            REG_CHECK(same, "n=%d %s style: packed bytes + scales identical to st_register_moe_dense_af_qNg64_as()", n, style ? "attn" : "embed");
        }
    }

    /* --- end to end through the file parser --- */
    {
        char path[512];
        snprintf(path, sizeof path, "%s.role_bits.txt", getenv("REG_ST"));
        FILE *f = fopen(path, "w");
        REG_CHECK(f != NULL, "wrote role-bits fixture file");
        if (f) {
            fputs("q_proj 3 5\nk_proj -1 7\nv_proj 0 2\no_proj 1 3\ndense_up_proj 0 6\n", f);
            fclose(f);
            moe_load_role_bits(path);
            REG_CHECK(g_role_bits_n == 5, "moe_load_role_bits parsed %d entries (expected 5)", g_role_bits_n);
            struct { const char *role; int layer; int bits; } cases[] = {
                {"q_proj", 3, 5}, {"k_proj", 9, 7}, {"v_proj", 0, 2}, {"o_proj", 1, 3}, {"dense_up_proj", 0, 6},
            };
            for (int i = 0; i < 5; i++) {
                int got_bits = moe_role_bits(cases[i].role, cases[i].layer, 8);
                REG_CHECK(got_bits == cases[i].bits, "moe_role_bits(%s, layer %d) = %d (file says %d)", cases[i].role, cases[i].layer, got_bits, cases[i].bits);
                char nm[64]; snprintf(nm, sizeof nm, "n2_e2e_%s", cases[i].role);
                MoeAFTensor *t = register_role(nm, got_bits, cases[i].role, 1);
                REG_CHECK(t && t->bits == cases[i].bits && t->qng64 == 1, "file '%s %d %d' -> registered tensor bits=%d qng64=%d", cases[i].role, cases[i].layer, cases[i].bits, t ? t->bits : -1, t ? t->qng64 : -1);
            }
            /* unlisted (role,layer) still falls through to the call site's own default (unchanged behaviour) */
            REG_CHECK(moe_role_bits("q_proj", 4, 8) == 8, "unlisted (q_proj, layer 4) keeps default bits 8");
        }
    }

    /* --- controls: widths that already worked must be unchanged --- */
    MoeAFTensor *c4 = register_role("n2_ctrl4", 4, "q_proj", 1);
    REG_CHECK(c4 && c4->bits == 4 && c4->sym == 1 && c4->qng64 == 0, "control bits=4: q4g64 tensor (bits=%d sym=%d qng64=%d)", c4 ? c4->bits : -1, c4 ? c4->sym : -1, c4 ? c4->qng64 : -1);
    MoeAFTensor *c8 = register_role("n2_ctrl8", 8, "q_proj", 1);
    REG_CHECK(c8 && c8->bits == 8 && c8->sym == 1 && c8->qng64 == 0, "control bits=8: q8g64 tensor (bits=%d sym=%d qng64=%d)", c8 ? c8->bits : -1, c8 ? c8->sym : -1, c8 ? c8->qng64 : -1);
    MoeAFTensor *c16 = register_role("n2_ctrl16", 16, "q_proj", 1);
    REG_CHECK(c16 && c16->bits == 16, "control bits=16: F16 passthrough tensor (bits=%d)", c16 ? c16->bits : -1);
    MoeAFTensor *c32 = register_role("n2_ctrl32", 32, "q_proj", 1);
    REG_CHECK(c32 && c32->bits == 32, "control bits=32: F32 passthrough tensor (bits=%d)", c32 ? c32->bits : -1);
    if (c4)  ctrl("control bits=4  decode", reg_decode_relL2(c4, 0, REG_DW), 0.29269);
    if (c4)  ctrl("control bits=4  matvec", reg_matvec_relL2(c4, 0, REG_DW), 0.23906);
    if (c8)  ctrl("control bits=8  decode", reg_decode_relL2(c8, 0, REG_DW), 0.01117);
    if (c8)  ctrl("control bits=8  matvec", reg_matvec_relL2(c8, 0, REG_DW), 0.01469);
    if (c16) ctrl("control bits=16 decode", reg_decode_relL2(c16, 0, REG_DW), 0.00019);
    if (c32) ctrl("control bits=32 decode", reg_decode_relL2(c32, 0, REG_DW), 0.0);
    return reg_fail;
}
