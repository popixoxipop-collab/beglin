/* N3 regression: the Gate 4.2 sym self-test (run_moe_sym_selftest_mode(), QWEN_MOE_SYM_SELFTEST=1) must really
 * compare the SME2 path against itself with and without the adj_bias correction, not scalar against scalar.
 *
 * Defect : the gate builds its two tensors by hand (`MoeAFTensor t_affine = {0}, t_sym = {0}`) and never set `bits`.
 *          moe_sme2_ensure_ready() bails to the scalar path for `actual_bits != 4` (slot->ready = -1), so on an SME2
 *          host BOTH moe_matvec_af_group_smart() calls took the scalar fallback and "RESULT: GATE 4.2 PASS" compared
 *          scalar with scalar. The first batch (F3b) set bits=4 in the three q4g64 registrars; this hand-built pair was
 *          not covered.
 *
 * D-reg-n3: this host (M1 Max) has no SME2, so the kai_* entry points are STUBBED with scalar reference code (same
 *           technique and same stubs as test_f3_sme2_slots.c) AFTER including qwen_infer.c, which makes the gate's own
 *           `if (!kai_sme2_available())` early-out pass. The REAL run_moe_sym_selftest_mode() is then invoked unmodified
 *           (moe_cfg_validate(), alloc_moe_buffers(), the hand-built tensors, both group_smart calls, the memcmp, the
 *           RESULT line); only "hardware says SME2 is available" is faked. Its stderr is captured to a file under
 *           $REG_OUT and re-printed, so the gate's own verdict line is part of the evidence.
 *   WHY : the defect is dispatch logic (which tensors reach the SME2 slot), which is hardware independent.
 *   COST: the stub GEMM proves nothing about real SME2 numerics; the gate was NOT executed on SME2 hardware.
 *   EXIT: re-run `QWEN_MOE_SYM_SELFTEST=1 <engine>` on an SME2 host with the real kleidiai objects linked.
 * Both slot arrays are covered: mode 1 = f16p-LHS (default, g_moe_sme2_f16lhs), mode 0 = int8-LHS (g_moe_sme2).
 */
#define main qwen_infer_real_main
#include "qwen_infer.c"
#undef main
#include "tests/regression/reg_common.h"
#include <fcntl.h>
#include <unistd.h>

/* The gate synthesizes its own raw nibbles ((row*7+col)&0xF); no quantizer runs. Recorded so the output says so. */
static const char *Q4_QUANTIZER_USED = "none: the gate packs raw synthetic nibbles (gguf_quantize_q4g64_error_feedback not involved)";

static int g_repack_calls;   /* every RHS repack, either LHS mode */
static int g_gemm_calls;     /* every call that really went through the (stubbed) SME2 GEMM */

/* ---- scalar stand-ins for the SME2 entry points (rhs "packed" = dequantized float matrix), as in test_f3 ---- */
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

/* Run the real gate once with stderr redirected into a file; returns the gate's return value, text in buf. */
static int run_gate_captured(const char *tag, char *buf, size_t cap) {
    char path[1100];
    snprintf(path, sizeof path, "%s.n3.%s.gate.txt", getenv("REG_ST"), tag);
    fflush(stderr);
    int saved = dup(2);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (saved < 0 || fd < 0) { fprintf(stderr, "cannot capture gate stderr into %s\n", path); exit(2); }
    dup2(fd, 2); close(fd);
    int rc = run_moe_sym_selftest_mode(0, NULL);
    fflush(stderr);
    dup2(saved, 2); close(saved);
    FILE *f = fopen(path, "r");
    size_t n = f ? fread(buf, 1, cap - 1, f) : 0;
    buf[n] = 0;
    if (f) fclose(f);
    return rc;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("4-bit data produced by: %s\n", Q4_QUANTIZER_USED);
    setenv("QWEN_MOE_SYM_SELFTEST", "1", 1);

    for (int mode = 1; mode >= 0; mode--) {
        const char *tag = mode ? "f16lhs" : "int8lhs";
        char out[16384];
        g_moe_sme2_f16lhs_mode_cached = mode;   /* normally read once from QWEN_MOE_SME2_F16LHS */
        int g0 = g_gemm_calls, r0 = g_repack_calls;
        int rc = run_gate_captured(tag, out, sizeof out);
        printf("---- gate stderr [%s] ----\n%s---- end gate stderr ----\n", tag, out);

        /* the gate calls moe_matvec_af_group_smart(NULL, &t_affine, 0, cache_e=0, layer 0, proj 0, ...) and
         * (NULL, &t_sym, 0, cache_e=1, layer 0, proj 0, ...): the slots it filled are read back here */
        MoeSme2Slot *sa = mode ? moe_sme2_f16lhs_slot(0, 0, 0) : moe_sme2_slot(0, 0, 0);
        MoeSme2Slot *ss = mode ? moe_sme2_f16lhs_slot(0, 1, 0) : moe_sme2_slot(0, 1, 0);

        REG_CHECK(rc == 1, "[%s] the real gate ran past its SME2 early-out (returned %d, want 1)", tag, rc);
        REG_CHECK(strstr(out, "SME2 not available") == NULL, "[%s] gate did not skip itself as 'SME2 not available'", tag);
        REG_CHECK(strstr(out, "RESULT: GATE 4.2 PASS") != NULL, "[%s] gate prints its own PASS line", tag);
        /* the point: PASS must come from two SME2 runs, not two scalar fallbacks */
        REG_CHECK(sa->ready == 1, "[%s] affine tensor (sym=0) reached the SME2 slot: ready=%d (want 1; -1 == scalar fallback)", tag, sa->ready);
        REG_CHECK(ss->ready == 1, "[%s] sym tensor (sym=1) reached the SME2 slot: ready=%d (want 1; -1 == scalar fallback)", tag, ss->ready);
        REG_CHECK(g_gemm_calls - g0 == 2, "[%s] both gate calls went through the SME2 GEMM: gemm+%d (want +2; +0 == scalar vs scalar)", tag, g_gemm_calls - g0);
        REG_CHECK(g_repack_calls - r0 == 2, "[%s] both gate tensors were repacked for SME2: repack+%d (want +2)", tag, g_repack_calls - r0);
        /* and the two runs must differ in exactly the way the gate is about: correction loop applied vs skipped */
        REG_CHECK(sa->ready == 1 && sa->adj_bias != NULL, "[%s] affine slot holds an adj_bias (correction loop runs): adj_bias=%s", tag, sa->adj_bias ? "built" : "NULL");
        REG_CHECK(ss->ready == 1 && ss->adj_bias == NULL, "[%s] sym slot has NO adj_bias (correction loop skipped): adj_bias=%s", tag, ss->adj_bias ? "built" : "NULL");
        if (sa->ready == 1 && sa->adj_bias) {
            int nz = 0;
            for (int i = 0; i < 64; i++) if (sa->adj_bias[i] != 0.0f) nz++;
            REG_CHECK(nz == 0, "[%s] affine adj_bias is exactly 0.0 for every group (8*scale + (-8*scale)): nonzero entries=%d", tag, nz);
        }
    }
    return reg_fail;
}
