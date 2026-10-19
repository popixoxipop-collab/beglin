/* F3 (#66) / F3b (#67) regression: SME2 RHS slot cache vs the tensor bound to a (layer, cache_e, proj) slot.
 *
 * F3 : slot->ready was only ever written 0->1 / 0->-1, and moe_matvec_af_group_smart() served the cached RHS
 *      without checking which tensor/bit-width built it, so a runtime promotion (g_moe_lt_active re-pointed at a
 *      higher-bit tensor) was silently ignored on SME2 hosts, and a demotion never reached SME2.
 * F3b: the q4g64 registrars left MoeAFTensor.bits at calloc 0, so `actual_bits != 4` in moe_sme2_ensure_ready()
 *      sent every registered q4g64 tensor to the scalar path for good.
 *
 * D-reg-f3: this host (M1 Max) has no SME2, so the kai_* entry points are STUBBED with scalar reference code (same
 *           signatures, same q4g64 nibble contract) AFTER including qwen_infer.c. The REAL moe_sme2_ensure_ready()
 *           and moe_matvec_af_group_smart() run unmodified; only "hardware says SME2 is available" is faked.
 *   WHY : the defects are dispatch/caching logic, which is hardware independent.
 *   COST: proves nothing about real SME2 numerics or speed.
 *   EXIT: re-run on an SME2 host with sme2_kai.o + kleidiai objects linked instead of these stubs.
 * Both slot arrays are covered: mode 1 = f16p-LHS (default, g_moe_sme2_f16lhs), mode 0 = int8-LHS (g_moe_sme2),
 * selected by setting the env-derived cache g_moe_sme2_f16lhs_mode_cached directly.
 */
#define main qwen_infer_real_main
#include "qwen_infer.c"
#undef main
#include "tests/regression/reg_common.h"

/* The 4-bit tensors under test are NOT quantized here: they come from the engine's own error-feedback (residual-
 * carrying) quantizer through st_register_moe_experts_q4g64_as(). Recorded so the output says which quantizer ran. */
static const char *Q4_QUANTIZER_USED = "gguf_quantize_q4g64_error_feedback";

static int g_repack_calls;   /* every RHS repack, either LHS mode */
static int g_gemm_calls;     /* every call that really went through the (stubbed) SME2 GEMM */

/* ---- scalar stand-ins for the SME2 entry points (rhs "packed" = dequantized float matrix) ---- */
static int stub_repack(int out, int in, const uint8_t *packed, const float *scales, void *dst) {
    float *w = (float *)dst;
    int ng = in / 64;
    g_repack_calls++;
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
static MoeSme2Slot *slot_of(int f16lhs, int layer, long e, int proj) {
    return f16lhs ? moe_sme2_f16lhs_slot(layer, e, proj) : moe_sme2_slot(layer, e, proj);
}
static void use(MoeAFTensor *t, long e, int layer, int proj, const float *x, float *y) {
    moe_matvec_af_group_smart(NULL, t, e, e, layer, proj, x, M, y);
}

/* Promotion / demotion / no-rebuild scenarios on slots (L, *, *) with q4 = the 4-bit tensor under test. */
static void scenario(const char *tag, int f16lhs, MoeAFTensor *q4, MoeAFTensor *q8, int L) {
    float x[M * REG_IN], y[M * REG_OUT], t4[2][M * REG_OUT], t8[2][M * REG_OUT];
    for (int i = 0; i < M * REG_IN; i++) x[i] = (float)((i * 37) % 17 - 8) / 8.0f;
    for (int e = 0; e < 2; e++)
        for (int m = 0; m < M; m++) {
            moe_matvec_af(NULL, q4, e, x + m * REG_IN, t4[e] + m * REG_OUT);   /* scalar truth, 4-bit  */
            moe_matvec_af(NULL, q8, e, x + m * REG_IN, t8[e] + m * REG_OUT);   /* scalar truth, 8-bit  */
        }
    printf("[%s] scalar sanity: rel-L2(4-bit truth, 8-bit truth) = %.5f\n", tag, rel(t4[0], t8[0], M * REG_OUT));

    /* (first use) a 4-bit tensor must reach the SME2 repack (ready==1) and match the scalar 4-bit truth */
    MoeSme2Slot *s = slot_of(f16lhs, L, 0, 0);
    int g0 = g_gemm_calls;
    use(q4, 0, L, 0, x, y);
    REG_CHECK(s->ready == 1 && g_gemm_calls == g0 + 1, "[%s] (a) first use of the 4-bit tensor builds the SME2 slot and runs the SME2 GEMM: ready=%d gemm+%d (want 1, +1)", tag, s->ready, g_gemm_calls - g0);
    REG_CHECK(rel(y, t4[0], M * REG_OUT) < 1e-6, "[%s] (a) 4-bit slot output == 4-bit truth: rel-L2=%.3g", tag, rel(y, t4[0], M * REG_OUT));
    printf("  info: [%s] fnv(first-use y)=%08x fnv(4-bit truth)=%08x fnv(8-bit truth)=%08x\n", tag, fnv_bits(y, M * REG_OUT), fnv_bits(t4[0], M * REG_OUT), fnv_bits(t8[0], M * REG_OUT));

    /* (b) promotion: same slot, now the 8-bit tensor (what g_moe_lt_active re-pointing does) */
    use(q8, 0, L, 0, x, y);
    REG_CHECK(rel(y, t8[0], M * REG_OUT) < 1e-6, "[%s] (b) promoted slot output == 8-bit truth: rel-L2=%.3g (stale 4-bit would be %.3g)",
              tag, rel(y, t8[0], M * REG_OUT), rel(y, t4[0], M * REG_OUT));
    REG_CHECK(s->ready != 1, "[%s] (b) the slot no longer serves the old 4-bit RHS: ready=%d (want != 1)", tag, s->ready);

    /* (b2) back to the 4-bit tensor: must rebuild, not stay on the 8-bit scalar verdict */
    int c0 = g_repack_calls;
    use(q4, 0, L, 0, x, y);
    REG_CHECK(s->ready == 1 && g_repack_calls == c0 + 1, "[%s] (b2) 8-bit -> 4-bit rebuilds exactly once: ready=%d repacks+%d (want 1, +1)", tag, s->ready, g_repack_calls - c0);
    REG_CHECK(rel(y, t4[0], M * REG_OUT) < 1e-6, "[%s] (b2) rebuilt slot output == 4-bit truth: rel-L2=%.3g", tag, rel(y, t4[0], M * REG_OUT));

    /* (c) demotion: slot first evaluated against the 8-bit tensor (scalar verdict), then given the 4-bit one */
    MoeSme2Slot *d = slot_of(f16lhs, L, 1, 0);
    use(q8, 1, L, 0, x, y);
    REG_CHECK(d->ready != 1 && rel(y, t8[1], M * REG_OUT) < 1e-6, "[%s] (c) slot first used with 8-bit: ready=%d rel-L2(8-bit truth)=%.3g", tag, d->ready, rel(y, t8[1], M * REG_OUT));
    use(q4, 1, L, 0, x, y);
    REG_CHECK(d->ready == 1, "[%s] (c) demotion to the 4-bit tensor reaches SME2: ready=%d (want 1)", tag, d->ready);
    REG_CHECK(rel(y, t4[1], M * REG_OUT) < 1e-6, "[%s] (c) demoted slot output == 4-bit truth: rel-L2=%.3g", tag, rel(y, t4[1], M * REG_OUT));

    /* (d) repeated calls with the same tensor must not rebuild */
    MoeSme2Slot *r = slot_of(f16lhs, L, 1, 1);
    c0 = g_repack_calls;
    for (int i = 0; i < 4; i++) use(q4, 1, L, 1, x, y);
    REG_CHECK(r->ready == 1 && g_repack_calls == c0 + 1, "[%s] (d) 4 calls, same tensor: ready=%d repacks+%d (want 1, +1)", tag, r->ready, g_repack_calls - c0);
    REG_CHECK(rel(y, t4[1], M * REG_OUT) < 1e-6, "[%s] (d) output == 4-bit truth: rel-L2=%.3g", tag, rel(y, t4[1], M * REG_OUT));
}

int main(void) {
    reg_init();
    MOE_SME2_CACHE_SLOTS = 8;
    MOE_SME2_SLOT_DENSE = 5; MOE_SME2_SLOT_SHARED = 6; MOE_SME2_SLOT_LMHEAD = 7;   /* MOE_N_EXPERTS + {0,1,2} for a 5-expert model */
    g_moe_sme2 = calloc((size_t)MOE_MAXLAYERS * MOE_SME2_CACHE_SLOTS * 3, sizeof(MoeSme2Slot));
    g_moe_sme2_f16lhs = calloc((size_t)MOE_MAXLAYERS * MOE_SME2_CACHE_SLOTS * 3, sizeof(MoeSme2Slot));
    g_groupsum = malloc(4096 * sizeof(float));
    printf("4-bit data produced by: %s\n", Q4_QUANTIZER_USED);

    MoeAFTensor *q8 = st_register_moe_experts_q8g64_as("L%d.e%d.w", 0, REG_E, "f3_q8");

    /* F3 alone: a 4-bit tensor whose bits is set explicitly (what the AF-blob loader, moe_load_layout_af(), and the
     * q8/qNg64 registrars do), so the slot-invalidation logic is testable independently of F3b. */
    MoeAFTensor *q4f = st_register_moe_experts_q4g64_as("L%d.e%d.w", 0, REG_E, "f3_q4_bits4");
    q4f->bits = 4;
    for (int mode = 1; mode >= 0; mode--) {
        g_moe_sme2_f16lhs_mode_cached = mode;   /* normally read once from QWEN_MOE_SME2_F16LHS */
        scenario(mode ? "f16lhs, bits=4 set by hand" : "int8-LHS, bits=4 set by hand", mode, q4f, q8, mode ? 1 : 2);
    }

    /* F3b: the tensor exactly as st_register_moe_experts_q4g64_as() returns it (bits NOT touched). */
    MoeAFTensor *q4 = st_register_moe_experts_q4g64_as("L%d.e%d.w", 0, REG_E, "f3_q4_as_registered");
    REG_CHECK(q4->bits == 4, "(F3b) st_register_moe_experts_q4g64_as sets bits: bits=%d (want 4)", q4->bits);
    for (int mode = 1; mode >= 0; mode--) {
        g_moe_sme2_f16lhs_mode_cached = mode;
        scenario(mode ? "f16lhs, q4 as registered" : "int8-LHS, q4 as registered", mode, q4, q8, mode ? 3 : 4);
    }

    /* F3b, E=1 sibling: st_register_moe_dense_af_q4g64_as() (shared/dense/attention roles), as registered. */
    {
        MoeAFTensor *dq4 = st_register_moe_dense_af_q4g64_as("dense.w", "f3_dense_q4_as_registered");
        REG_CHECK(dq4->bits == 4, "(F3b) st_register_moe_dense_af_q4g64_as sets bits: bits=%d (want 4)", dq4->bits);
        float x[M * REG_IN], y[M * REG_OUT], t4[M * REG_OUT];
        for (int i = 0; i < M * REG_IN; i++) x[i] = (float)((i * 37) % 17 - 8) / 8.0f;
        for (int m = 0; m < M; m++) moe_matvec_af(NULL, dq4, 0, x + m * REG_IN, t4 + m * REG_OUT);
        for (int mode = 1; mode >= 0; mode--) {
            g_moe_sme2_f16lhs_mode_cached = mode;
            int g0 = g_gemm_calls;
            moe_matvec_af_group_smart(NULL, dq4, 0, MOE_SME2_SLOT_DENSE, mode ? 5 : 6, 0, x, M, y);
            MoeSme2Slot *s = slot_of(mode, mode ? 5 : 6, MOE_SME2_SLOT_DENSE, 0);
            REG_CHECK(s->ready == 1 && g_gemm_calls == g0 + 1 && rel(y, t4, M * REG_OUT) < 1e-6,
                      "[%s, dense q4 as registered] first use reaches SME2: ready=%d gemm+%d rel-L2=%.3g (want 1, +1, <1e-6)",
                      mode ? "f16lhs" : "int8-LHS", s->ready, g_gemm_calls - g0, rel(y, t4, M * REG_OUT));
        }
    }

    /* F3b, GGUF registrar: gguf_register_moe_q4g64_as() on a one-tensor F32 GGUF [in=128, out=16, E=2] holding the
     * same expert weights, written next to the safetensors fixture (so only under $REG_OUT). */
    {
        char gpath[1024];
        snprintf(gpath, sizeof gpath, "%s.f3.gguf", getenv("REG_ST"));
        GgufWriter *gw = gguf_w_open(gpath);
        uint64_t ne[3] = {REG_IN, REG_OUT, REG_E};
        if (!gw) { fprintf(stderr, "cannot open %s for writing\n", gpath); return 2; }
        gguf_w_kv_str(gw, "general.architecture", "llama", 5);
        gguf_w_add_tensor(gw, "f3.experts.weight", GGML_TYPE_F32, 3, ne, &REG_W[0][0][0], sizeof REG_W);
        if (!gguf_w_finish(gw) || !(g_gguf_moe = gguf_open(gpath))) { fprintf(stderr, "cannot build %s\n", gpath); return 2; }
        MoeAFTensor *gq4 = gguf_register_moe_q4g64_as("f3.experts.weight", "f3_gguf_q4_as_registered");
        REG_CHECK(gq4->bits == 4, "(F3b) gguf_register_moe_q4g64_as sets bits: bits=%d (want 4)", gq4->bits);
        for (int mode = 1; mode >= 0; mode--) {
            g_moe_sme2_f16lhs_mode_cached = mode;
            scenario(mode ? "f16lhs, GGUF q4 as registered" : "int8-LHS, GGUF q4 as registered", mode, gq4, q8, mode ? 7 : 8);
        }
    }
    return reg_fail;
}
