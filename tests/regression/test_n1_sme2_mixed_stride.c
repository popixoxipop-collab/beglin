/* N1 (issue pending) regression: moe_sme2_ensure_ready() must address the packed nibbles of a 4-bit expert that lives
 * in a MIXED tensor (tsr->ebits != NULL) through tsr->epacked_off[blob_e], not through the uniform
 * blob_e * out * (in/2) stride.
 *
 * Defect: the SME2 repack read `base + packed_off + (blob_e*out + row) * row_words*4`, i.e. as if every expert
 *         occupied out*in/2 bytes. In a mixed tensor the experts have different sizes (4: in/2 per row, 8: in,
 *         16: 2*in, 32: 4*in, 2/3/5/6/7: ng*n*8) and epacked_off[] is their prefix sum, so a 4-bit expert that
 *         follows ANY non-4-bit expert (e.g. ebits {8,4}, {5,4}, {16,4}) was repacked from the wrong bytes and the
 *         SME2 GEMM then served garbage for it (the scalar paths were right). Expert 0 is coincidentally correct
 *         (epacked_off[0]==0) and so is a 4-bit expert preceded only by 4-bit experts.
 *
 * D-reg-n1: this host (M1 Max) has no SME2, so the kai_* entry points are STUBBED with scalar reference code, as in
 *           test_f3_sme2_slots.c. The stub repack reads the nibble source exactly as the real repack contract does
 *           (row-major, in/2 bytes per row, low nibble = even column, value = (nib-8)*scale), so the REAL
 *           moe_sme2_ensure_ready() / moe_matvec_af_group_smart() address arithmetic is what is under test.
 *   WHY : the defect is source addressing, which is hardware independent.
 *   COST: proves nothing about real SME2 numerics, speed or the real kai_* repack layout.
 *   EXIT: re-run on an SME2 host with sme2_kai.o + kleidiai objects linked instead of these stubs.
 * Both slot arrays are covered: LHS mode 1 = f16p-LHS (default), mode 0 = int8-LHS.
 */
#define main qwen_infer_real_main
#include "qwen_infer.c"
#undef main
#include "tests/regression/reg_common.h"

/* The 4-bit experts under test are NOT quantized here: they come from the engine's own error-feedback (residual-
 * carrying) quantizer through st_register_moe_experts_mixed_as() / st_register_moe_experts_q4g64_as(). */
static const char *Q4_QUANTIZER_USED = "gguf_quantize_q4g64_error_feedback";

static int g_gemm_calls;   /* every call that really went through the (stubbed) SME2 GEMM */

/* ---- scalar stand-ins for the SME2 entry points (rhs "packed" = dequantized float matrix) ---- */
static int stub_repack(int out, int in, const uint8_t *packed, const float *scales, void *dst) {
    float *w = (float *)dst;
    int ng = in / 64;
    for (int r = 0; r < out; r++)
        for (int c = 0; c < in; c++) {
            uint8_t b = packed[(size_t)r * (in / 2) + c / 2];
            int nib = (c % 2 == 0) ? (b & 0xF) : (b >> 4);
            w[(size_t)r * in + c] = ((float)nib - 8.0f) * scales[r * ng + c / 64];
        }
    return 0;
}
static void stub_gemm(int M, int out, int in, const float *x, const void *rhs_packed, float *y) {
    const float *w = (const float *)rhs_packed;
    g_gemm_calls++;
    for (int m = 0; m < M; m++)
        for (int r = 0; r < out; r++) {
            double s = 0;
            for (int c = 0; c < in; c++) s += (double)x[(size_t)m * in + c] * w[(size_t)r * in + c];
            y[(size_t)m * out + r] = (float)s;
        }
}
/* f16p-LHS entry points (default mode) */
int kai_sme2_f16lhs_available(void) { return 1; }
int kai_sme2_f16lhs_shape_ok(int out, int in) { return 1; }
size_t kai_sme2_f16lhs_rhs_packed_bytes(int out, int in) { return (size_t)out * in * sizeof(float); }
size_t kai_sme2_f16lhs_lhs_scratch_bytes(int max_m, int max_in) { return 64; }
int kai_sme2_repack_q4g64_f16lhs(int out, int in, const uint8_t *packed, const float *scales, void *dst, size_t dst_bytes) {
    return stub_repack(out, in, packed, scales, dst);
}
void kai_sme2_gemm_f16lhs(int M, int out, int in, const float *x, const void *rhs_packed, const float *bias, float *y, void *lhs_scratch) {
    stub_gemm(M, out, in, x, rhs_packed, y);
}
/* int8-LHS entry points (QWEN_MOE_SME2_F16LHS=0 mode) */
int kai_sme2_available(void) { return 1; }
int kai_sme2_shape_ok(int out, int in) { return 1; }
size_t kai_sme2_rhs_packed_bytes(int out, int in) { return (size_t)out * in * sizeof(float); }
size_t kai_sme2_lhs_scratch_bytes(int max_m, int max_in) { return 64; }
int kai_sme2_repack_q4g64(int out, int in, const uint8_t *packed, const float *scales, void *dst, size_t dst_bytes) {
    return stub_repack(out, in, packed, scales, dst);
}
void kai_sme2_gemm_f32(int M, int out, int in, const float *x, const void *rhs_packed, const float *bias, float *y, void *lhs_scratch) {
    stub_gemm(M, out, in, x, rhs_packed, y);
}

enum { M = 2 };

static double rel(const float *a, const float *b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { num += (double)(a[i] - b[i]) * (a[i] - b[i]); den += (double)b[i] * b[i]; }
    return sqrt(num / (den > 0 ? den : 1));
}
/* FNV-1a over the raw float bits: printed so before/after runs of an UNCHANGED configuration can be diffed byte for byte. */
static unsigned fnv_bits(const float *a, int n) {
    unsigned h = 2166136261u;
    for (int i = 0; i < n * (int)sizeof(float); i++) { h ^= ((const unsigned char *)a)[i]; h *= 16777619u; }
    return h;
}

static int g_layer;   /* a fresh SME2 slot (layer, e, 0) for every probe: no cross-probe slot reuse */

/* Probe expert e of tensor t through the real SME2 dispatcher in both LHS modes and compare with the scalar truth
 * (moe_matvec_af on the same tensor/expert). want_sme2: the expert is 4-bit, so the slot must really be built and the
 * (stubbed) SME2 GEMM must really run -- otherwise a silent scalar fallback would make the comparison vacuous. */
static void probe(const char *tag, MoeAFTensor *t, long e, int want_sme2) {
    float x[M * REG_IN], y[M * REG_OUT], truth[M * REG_OUT];
    for (int i = 0; i < M * REG_IN; i++) x[i] = (float)((i * 37) % 17 - 8) / 8.0f;
    for (int m = 0; m < M; m++) moe_matvec_af(NULL, t, e, x + m * REG_IN, truth + m * REG_OUT);
    for (int mode = 1; mode >= 0; mode--) {
        g_moe_sme2_f16lhs_mode_cached = mode;   /* normally read once from QWEN_MOE_SME2_F16LHS */
        int layer = g_layer++;
        MoeSme2Slot *s = mode ? moe_sme2_f16lhs_slot(layer, e, 0) : moe_sme2_slot(layer, e, 0);
        int g0 = g_gemm_calls;
        moe_matvec_af_group_smart(NULL, t, e, e, layer, 0, x, M, y);
        double r = rel(y, truth, M * REG_OUT);
        int sme2 = s->ready == 1 && g_gemm_calls == g0 + 1;
        REG_CHECK(sme2 == want_sme2 && r < 1e-6,
                  "%-26s [%s] SME2 path %s (ready=%d gemm+%d), rel-L2 vs scalar truth = %.3g (want < 1e-6)  fnv(y)=%08x fnv(truth)=%08x",
                  tag, mode ? "f16lhs" : "int8-LHS", sme2 ? "taken" : "NOT taken", s->ready, g_gemm_calls - g0, r,
                  fnv_bits(y, M * REG_OUT), fnv_bits(truth, M * REG_OUT));
    }
}

static MoeAFTensor *reg_mixed(int b0, int b1, const char *engine_name) {
    int eb[REG_E] = {b0, b1};
    return st_register_moe_experts_mixed_as("L%d.e%d.w", 0, REG_E, eb, engine_name);
}

int main(void) {
    reg_init();
    MOE_SME2_CACHE_SLOTS = 8;
    MOE_SME2_SLOT_DENSE = 5; MOE_SME2_SLOT_SHARED = 6; MOE_SME2_SLOT_LMHEAD = 7;
    g_moe_sme2 = calloc((size_t)MOE_MAXLAYERS * MOE_SME2_CACHE_SLOTS * 3, sizeof(MoeSme2Slot));
    g_moe_sme2_f16lhs = calloc((size_t)MOE_MAXLAYERS * MOE_SME2_CACHE_SLOTS * 3, sizeof(MoeSme2Slot));
    g_groupsum = malloc(4096 * sizeof(float));
    printf("4-bit data produced by: %s\n", Q4_QUANTIZER_USED);

    /* Part A -- the defect: a 4-bit expert 1 that follows a non-4-bit expert 0 (epacked_off[1] != out*in/2). */
    printf("-- Part A: 4-bit expert 1 preceded by a non-4-bit expert 0 (must repack from epacked_off[1]) --\n");
    static const int PRE[] = {8, 5, 16, 32, 2, 3, 6, 7};
    for (size_t i = 0; i < sizeof PRE / sizeof PRE[0]; i++) {
        char nm[64], tag[64];
        snprintf(nm, sizeof nm, "n1_mixed_%d_4", PRE[i]);
        snprintf(tag, sizeof tag, "{%d,4} expert1", PRE[i]);
        MoeAFTensor *m = reg_mixed(PRE[i], 4, nm);
        REG_CHECK(m->epacked_off[1] != (size_t)REG_OUT * (REG_IN / 2),
                  "%s: epacked_off[1]=%zu differs from the uniform q4 stride %d (the case the bug needs)",
                  tag, m->epacked_off[1], REG_OUT * (REG_IN / 2));
        probe(tag, m, 1, 1);
    }

    /* Part B -- controls that were already correct and must stay numerically identical. */
    printf("-- Part B: controls that must not change --\n");
    {
        MoeAFTensor *m44 = reg_mixed(4, 4, "n1_mixed_4_4");
        REG_CHECK(m44->epacked_off[1] == (size_t)REG_OUT * (REG_IN / 2), "{4,4}: epacked_off[1]=%zu equals the uniform q4 stride (coincidentally right)", m44->epacked_off[1]);
        probe("{4,4} expert0", m44, 0, 1);
        probe("{4,4} expert1", m44, 1, 1);
        static const int POST[] = {8, 16, 32, 5};
        for (size_t i = 0; i < sizeof POST / sizeof POST[0]; i++) {
            char nm[64], tag[64];
            snprintf(nm, sizeof nm, "n1_mixed_4_%d", POST[i]);
            snprintf(tag, sizeof tag, "{4,%d} expert0", POST[i]);
            MoeAFTensor *m = reg_mixed(4, POST[i], nm);
            probe(tag, m, 0, 1);
            snprintf(tag, sizeof tag, "{4,%d} expert1 (non-4-bit)", POST[i]);
            probe(tag, m, 1, 0);   /* scalar fallback verdict must be unchanged */
        }
        for (size_t i = 0; i < sizeof PRE / sizeof PRE[0]; i++) {
            char nm[64], tag[64];
            snprintf(nm, sizeof nm, "n1_mixed_%d_4_e0", PRE[i]);
            snprintf(tag, sizeof tag, "{%d,4} expert0 (non-4-bit)", PRE[i]);
            probe(tag, reg_mixed(PRE[i], 4, nm), 0, 0);
        }
        MoeAFTensor *q4 = st_register_moe_experts_q4g64_as("L%d.e%d.w", 0, REG_E, "n1_uniform_q4");
        probe("uniform q4g64 expert0", q4, 0, 1);
        probe("uniform q4g64 expert1", q4, 1, 1);
    }
    return reg_fail;
}
