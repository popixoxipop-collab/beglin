/* Regression for F2 (issue #65): a qNg64 tensor with n=8 stores bit-plane data with bits=8, sym=1, exactly like
 * q8g64 stores bits=8, sym=1 -- and moe_decode_af()/moe_matvec_af_row() used to take the raw-int8 (q8g64) branch for
 * every bits==8 tensor, so n=8 bit-planes were decoded as int8 garbage (rel-L2 ~3 instead of ~0.015).
 *
 * Asserts, on the real static registrars/decoders of qwen_infer.c:
 *   - uniform qNg64 n=8 (both experts) and dense-AF qNg64 n=8: decode rel-L2 vs fp32 < 0.02 (measured 0.0147..0.0165,
 *     between n=7's ~0.030 and n=9's ~0.0078), decode and matvec EXACTLY equal an independent bit-plane reference
 *     reading the packed bytes per gguf_transcode.h, and matvec rel-L2 vs fp32 < 0.05 (garbage is 2.2..4.2);
 *   - controls stay at their pre-fix values: qNg64 n=7 / n=9 and uniform q8g64 (which must still decode as raw int8,
 *     proving the qNg64 marker did not capture q8g64 tensors).
 * The control constants were measured on the unfixed base (fix/base) with this fixture.
 *
 * Why the n=8 matvec bound is 0.05 and not 0.02: the matvec metric is a 16-row dot product against one fixed probe
 * vector, so it is far noisier than the decode metric (n=7 expert 1 is 0.0454 against a decode of 0.0340; n=8 expert 1
 * is 0.0299 against a decode of 0.0165). That is quantizer noise, not decoder error -- the exact-equality checks
 * against the independent reference carry the correctness claim; the rel-L2 bound is only a garbage detector. */
#define main qwen_infer_real_main
#include "qwen_infer.c"
#undef main
#include "tests/regression/reg_common.h"

#define N8_DECODE_BOUND 0.02
#define N8_MATVEC_BOUND 0.05
#define CTRL_TOL 2e-5       /* controls are compared at the 5 decimals recorded from the base run */

/* Print + check one measured value against a pre-fix control. */
static void ctrl(const char *what, double got, double base) {
    REG_CHECK(fabs(got - base) < CTRL_TOL, "%s = %.5f (base control %.5f)", what, got, base);
}

/* Independent bit-plane decode straight from the tensor's packed bytes (layout pinned in gguf_transcode.h: group of 64
 * = n planes of 8 bytes, element i is bit (i&7) of byte (i>>3) of each plane, plane j = bit j of u = code + 2^(n-1)). */
static float ref_planes(const MoeAFTensor *t, long e, long r, long c) {
    int n = t->bits;
    long row_pbytes = t->ng * (long)n * 8;
    const uint8_t *g = t->base + t->packed_off + (e * t->out + r) * row_pbytes + (c / 64) * (long)n * 8;
    int p = (int)(c % 64), u = 0;
    for (int j = 0; j < n; j++)
        if ((g[j * 8 + (p >> 3)] >> (p & 7)) & 1) u |= 1 << j;
    float sc; memcpy(&sc, t->base + t->scale_off + ((e * t->out + r) * t->ng + c / 64) * 4, 4);
    return (float)(u - (1 << (n - 1))) * sc;
}

/* Max |engine - reference| over every element (decode) and every row (matvec with the reg_common probe vector). */
static void ref_diff(const MoeAFTensor *t, long e, double *max_dec, double *max_mv) {
    float x[REG_IN];
    for (int c = 0; c < REG_IN; c++) x[c] = (float)((c * 37) % 17 - 8) / 8.0f;
    *max_dec = 0; *max_mv = 0;
    for (int r = 0; r < REG_OUT; r++) {
        double ydot = 0;
        for (int c = 0; c < REG_IN; c++) {
            float want = ref_planes(t, e, r, c), got = moe_decode_af(NULL, (MoeAFTensor *)t, e, r, c);
            if (fabs((double)got - want) > *max_dec) *max_dec = fabs((double)got - want);
            ydot += (double)want * x[c];
        }
        double y = moe_matvec_af_row(NULL, (MoeAFTensor *)t, e, r, x);
        if (fabs(y - ydot) > *max_mv) *max_mv = fabs(y - ydot);
    }
}

static void check_n8(const char *what, MoeAFTensor *t, long e, const float (*ref)[REG_IN]) {
    double d = reg_decode_relL2(t, e, ref), m = reg_matvec_relL2(t, e, ref), xd, xm;
    ref_diff(t, e, &xd, &xm);
    REG_CHECK(d < N8_DECODE_BOUND, "%s decode rel-L2 = %.5f (< %.2f)", what, d, N8_DECODE_BOUND);
    REG_CHECK(m < N8_MATVEC_BOUND, "%s matvec rel-L2 = %.5f (< %.2f)", what, m, N8_MATVEC_BOUND);
    REG_CHECK(xd == 0.0, "%s decode == independent bit-plane reference (max abs diff %.3g)", what, xd);
    REG_CHECK(xm < 1e-9, "%s matvec == dot with that reference (max abs diff %.3g)", what, xm);
}

int main(void) {
    reg_init();

    /* --- target: qNg64 n=8 --- */
    MoeAFTensor *q8n = st_register_moe_experts_qNg64_as("L%d.e%d.w", 0, REG_E, 8, "f2_qn8");
    check_n8("uniform qNg64 n=8 expert 0", q8n, 0, REG_W[0]);
    check_n8("uniform qNg64 n=8 expert 1", q8n, 1, REG_W[1]);
    MoeAFTensor *dn8 = st_register_moe_dense_af_qNg64_as("dense.w", 8, "f2_dense_qn8");
    check_n8("dense-AF qNg64 n=8", dn8, 0, REG_DW);

    /* --- controls: neighbours of n=8 (unchanged by the fix) --- */
    MoeAFTensor *q7 = st_register_moe_experts_qNg64_as("L%d.e%d.w", 0, REG_E, 7, "f2_qn7");
    MoeAFTensor *q9 = st_register_moe_experts_qNg64_as("L%d.e%d.w", 0, REG_E, 9, "f2_qn9");
    ctrl("control qNg64 n=7 expert 0 decode", reg_decode_relL2(q7, 0, REG_W[0]), 0.03007);
    ctrl("control qNg64 n=7 expert 1 decode", reg_decode_relL2(q7, 1, REG_W[1]), 0.03396);
    ctrl("control qNg64 n=9 expert 0 decode", reg_decode_relL2(q9, 0, REG_W[0]), 0.00775);
    ctrl("control qNg64 n=9 expert 1 decode", reg_decode_relL2(q9, 1, REG_W[1]), 0.00807);
    ctrl("control qNg64 n=7 expert 0 matvec", reg_matvec_relL2(q7, 0, REG_W[0]), 0.02358);
    ctrl("control qNg64 n=7 expert 1 matvec", reg_matvec_relL2(q7, 1, REG_W[1]), 0.04540);
    ctrl("control qNg64 n=9 expert 0 matvec", reg_matvec_relL2(q9, 0, REG_W[0]), 0.00459);
    ctrl("control qNg64 n=9 expert 1 matvec", reg_matvec_relL2(q9, 1, REG_W[1]), 0.00718);
    MoeAFTensor *d7 = st_register_moe_dense_af_qNg64_as("dense.w", 7, "f2_dense_qn7");
    MoeAFTensor *d9 = st_register_moe_dense_af_qNg64_as("dense.w", 9, "f2_dense_qn9");
    ctrl("control dense-AF qNg64 n=7 decode", reg_decode_relL2(d7, 0, REG_DW), 0.03119);
    ctrl("control dense-AF qNg64 n=7 matvec", reg_matvec_relL2(d7, 0, REG_DW), 0.03868);
    ctrl("control dense-AF qNg64 n=9 decode", reg_decode_relL2(d9, 0, REG_DW), 0.00766);
    ctrl("control dense-AF qNg64 n=9 matvec", reg_matvec_relL2(d9, 0, REG_DW), 0.00611);

    /* --- control: uniform q8g64 must still be the raw-int8 layout (marker must not capture it) --- */
    MoeAFTensor *q8 = st_register_moe_experts_q8g64_as("L%d.e%d.w", 0, REG_E, "f2_q8");
    ctrl("control q8g64 expert 0 decode", reg_decode_relL2(q8, 0, REG_W[0]), 0.01042);
    ctrl("control q8g64 expert 1 decode", reg_decode_relL2(q8, 1, REG_W[1]), 0.01121);
    ctrl("control q8g64 expert 0 matvec", reg_matvec_relL2(q8, 0, REG_W[0]), 0.01097);
    ctrl("control q8g64 expert 1 matvec", reg_matvec_relL2(q8, 1, REG_W[1]), 0.00924);
    MoeAFTensor *dq8 = st_register_moe_dense_af_q8g64_as("dense.w", "f2_dense_q8");
    ctrl("control dense-AF q8g64 decode", reg_decode_relL2(dq8, 0, REG_DW), 0.01117);
    ctrl("control dense-AF q8g64 matvec", reg_matvec_relL2(dq8, 0, REG_DW), 0.01469);
    REG_CHECK(q8->bits == 8 && q8->sym == 1, "q8g64 tensor keeps bits=8 sym=1 (bits=%d sym=%d)", q8->bits, q8->sym);
    REG_CHECK(q8n->bits == 8 && q8n->sym == 1, "qNg64 n=8 tensor also has bits=8 sym=1 (the ambiguity under test)");
    return reg_fail;
}
