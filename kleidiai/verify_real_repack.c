// verify_real_repack.c -- Phase 3 end-to-end correctness check for
// kai_sme2_repack_q4g64() against REAL vdsp production weights.
//
// NEW FILE. Reads (never writes) sme2_kai.c/.h, the vendored kleidiai sources,
// and the production weight blob. Modelled on -- but independent of --
// kleidiai/kai_test_correct2.c (which uses synthetic codes; this uses real
// q4g64 tensors straight out of the production blob via mmap).
//
// Per tensor, per seed:
//   ref   = double-precision W@x computed directly from vdsp's own packed
//           nibble format + fp32 scales (the "truth")
//   kai   = kai_sme2_repack_q4g64() -> kai_run_lhs_quant_pack_* ->
//           kai_run_matmul_clamp_*_sme_mopa()
//   xref  = an INDEPENDENT reconstruction of the KleidiAI RHS-packed buffer,
//           built the kai_test_correct2.c way from decoded codes, then
//           memcmp'd byte-for-byte against kai_sme2_repack_q4g64()'s output.
//           This isolates "permutation bug" from "everything else".
//
// Usage:
//   verify_real_repack <layout> <blob> <M> <nseeds> <seed0> <tensor>...
//   verify_real_repack --guard <layout> <blob> <tensor>
//        (mmap guard-page test: dst+need lands exactly on a PROT_NONE page,
//         so any single byte of overrun is a hard SIGSEGV rather than a guess)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* vdsp's OWN production unpack kernel, used here as an independent oracle for
   decode_row() below -- without this, every reference in this file would share
   one reading of the vdsp nibble convention and a misreading would be invisible.
   q4gemv.h is included read-only and only q4_unpack_row() is called.
   NOTE: this file must be compiled WITHOUT +sve2 (see vdsp_ref.c) -- M4 has
   streaming-SVE only, and the autovectorizer would emit non-streaming SVE here. */
#include "../q4gemv.h"

#include "../sme2_kai.h"
#include "kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa.h"
#include "kai_lhs_quant_pack_qsi8d32p_f32_neon.h"
#include "kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon.h"
#include "kai/kai_common.h"

#define BL 64
#define CANARY 0xA5u
#define CANARY_PAD (1u << 20)   /* 1 MiB of canary past the declared size */

/* ------------------------------------------------------------------ RNG */
static uint64_t rng_state;
static void rng_seed(uint64_t s) { rng_state = s ? s : 0x243F6A8885A308D3ULL; }
static uint32_t xr(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

/* --------------------------------------------------------------- layout */
typedef struct {
    char   name[512];
    char   kind[32];
    size_t doff;
    int    out, in, ng;
    size_t soff;
} TensorInfo;

static int find_tensor(const char *layout, const char *want, TensorInfo *ti) {
    FILE *f = fopen(layout, "r");
    if (!f) { fprintf(stderr, "open layout %s: %s\n", layout, strerror(errno)); return -1; }
    char line[2048];
    while (fgets(line, sizeof line, f)) {
        char nm[512], kd[32];
        unsigned long long doff; int o, i, ng; long long soff;
        if (sscanf(line, "%511s %31s %llu %d %d %d %lld", nm, kd, &doff, &o, &i, &ng, &soff) != 7) continue;
        if (strcmp(nm, want) != 0) continue;
        snprintf(ti->name, sizeof ti->name, "%s", nm);
        snprintf(ti->kind, sizeof ti->kind, "%s", kd);
        ti->doff = (size_t)doff; ti->out = o; ti->in = i; ti->ng = ng;
        ti->soff = (size_t)soff;
        fclose(f);
        return 0;
    }
    fclose(f);
    fprintf(stderr, "tensor '%s' not found in %s\n", want, layout);
    return -1;
}

/* ------------------------------------------------------- vdsp decode */
/* group gi occupies byte offset gi*32 within the row; byte b of that group:
   low nibble -> group-relative column 2b, high nibble -> column 2b+1;
   value = (nibble - 8) * scale.  (q4gemv.h / q4_unpack_row) */
static void decode_row(const uint8_t *row_packed, int in, int8_t *code_out) {
    int ng = in / BL;
    for (int gi = 0; gi < ng; gi++) {
        const uint8_t *g = row_packed + (size_t)gi * (BL / 2);
        int8_t *c = code_out + (size_t)gi * BL;
        for (int b = 0; b < BL / 2; b++) {
            uint8_t byte = g[b];
            c[2 * b]     = (int8_t)((int)(byte & 0x0F) - 8);
            c[2 * b + 1] = (int8_t)((int)(byte >> 4)   - 8);
        }
    }
}

/* Oracle: run vdsp's real q4_unpack_row() over sampled rows and require that
   decode_row()'s codes, times the fp32 group scale, reproduce it EXACTLY.
   Returns the number of mismatching elements. */
static size_t oracle_check_decode(const uint8_t *packed, const float *scales,
                                  const int8_t *code, int out, int in, int ng, int nsample) {
    float *tile = malloc((size_t)in * sizeof(float));
    size_t bad = 0;
    for (int s = 0; s < nsample; s++) {
        int r = (nsample == 1) ? 0 : (int)((long)s * (out - 1) / (nsample - 1));
        q4_unpack_row(packed + (size_t)r * (size_t)(in / 2), scales + (size_t)r * ng, tile, in);
        const int8_t *c = code + (size_t)r * in;
        for (int gi = 0; gi < ng; gi++) {
            float sc = scales[(size_t)r * ng + gi];
            for (int k = 0; k < BL; k++) {
                float mine = (float)c[gi * BL + k] * sc;
                float theirs = tile[gi * BL + k];
                if (mine != theirs) bad++;          /* bit-exact equality demanded */
            }
        }
    }
    free(tile);
    return bad;
}

/* -------------------------------------------------- reference (threaded) */
typedef struct {
    int r0, r1;
    int out, in, ng, M;
    const int8_t *code;      /* [out][in] */
    const float  *scales;    /* [out][ng] fp32, vdsp truth */
    const float  *act;       /* [M][in] fp32 activations */
    const int8_t *actq;      /* [M][in] int8, quantized exactly as KleidiAI's LHS pack does */
    const float  *actsf16;   /* [M][ng] fp16-rounded LHS block scales */
    double *y_f32s;          /* [M][out] fp32 w-scales, fp32 acts   = GROUND TRUTH */
    double *y_f16s;          /* [M][out] fp16 w-scales, fp32 acts   = isolates fp16 scale rounding */
    double *y_mod;           /* [M][out] fp16 w-scales, int8 acts   = EXACT MODEL of what the kernel computes */
    double *absum;           /* [M][out] sum of |per-block contribution| -- the magnitude the kernel's
                                fp32 accumulator actually traverses. |y| can be far smaller than this
                                (cancellation), and fp32 rounding scales with THIS, not with |y|. */
} RefJob;

static void *ref_worker(void *p) {
    RefJob *j = (RefJob *)p;
    for (int r = j->r0; r < j->r1; r++) {
        const int8_t *crow = j->code + (size_t)r * j->in;
        for (int m = 0; m < j->M; m++) {
            const float  *a  = j->act  + (size_t)m * j->in;
            const int8_t *aq = j->actq + (size_t)m * j->in;
            double y32 = 0.0, y16 = 0.0, ymod = 0.0, asum = 0.0;
            for (int gi = 0; gi < j->ng; gi++) {
                const int8_t *c = crow + (size_t)gi * BL;
                const float  *aa = a + (size_t)gi * BL;
                const int8_t *ai = aq + (size_t)gi * BL;
                double acc = 0.0;
                int32_t iacc = 0;
                for (int k = 0; k < BL; k++) {
                    acc  += (double)c[k] * (double)aa[k];
                    iacc += (int32_t)c[k] * (int32_t)ai[k];
                }
                float s32 = j->scales[(size_t)r * j->ng + gi];
                float s16 = (float)(__fp16)s32;
                y32  += acc * (double)s32;
                y16  += acc * (double)s16;
                double contrib = (double)iacc * (double)s16 * (double)j->actsf16[(size_t)m * j->ng + gi];
                ymod += contrib;
                asum += fabs(contrib);
            }
            j->y_f32s[(size_t)m * j->out + r] = y32;
            j->y_f16s[(size_t)m * j->out + r] = y16;
            j->y_mod [(size_t)m * j->out + r] = ymod;
            j->absum [(size_t)m * j->out + r] = asum;
        }
    }
    return NULL;
}

/* Replicate kai_run_lhs_quant_pack_qsi8d32p_f32_neon()'s quantization arithmetic
   EXACTLY (read from kai_lhs_quant_pack_qsi8d32p_f32_neon.c):
     amax = max|x| over the block; sf = amax/127; sf_inv = sf ? 1/sf : 0;
     xq = (int8)(int32)roundf(x * sf_inv);  the STORED scale is fp16(sf), while
     the inverse used for quantizing is the fp32 sf -- that asymmetry is real
     and must be reproduced or the model reference will not be exact. */
static void lhs_quant_model(int M, int in, int ng, const float *act, int8_t *actq, float *actsf16) {
    for (int m = 0; m < M; m++) {
        for (int gi = 0; gi < ng; gi++) {
            const float *x = act + (size_t)m * in + (size_t)gi * BL;
            float amax = 0.0f;
            for (int k = 0; k < BL; k++) { float v = fabsf(x[k]); if (v > amax) amax = v; }
            const float sf = amax / (float)((1 << 7) - 1);
            const float sf_inv = sf ? 1.0f / sf : 0.0f;
            int8_t *q = actq + (size_t)m * in + (size_t)gi * BL;
            for (int k = 0; k < BL; k++) q[k] = (int8_t)(int32_t)roundf(x[k] * sf_inv);
            actsf16[(size_t)m * ng + gi] = (float)(__fp16)sf;
        }
    }
}

/* ------------------------------------------------- stats (nan-safe) */
typedef struct {
    double worst_rel;      /* max |k-r|/|r| over ALL elements (blows up near r==0) */
    double worst_rel_big;  /* same, restricted to |r| >= typical -- the meaningful one */
    double worst_abs;
    double rms_abs;
    double typical;        /* mean |ref| */
    double rel_rms;        /* rms_abs / typical */
    int any_nan;
    int worst_rel_i, worst_abs_i;
} Stats;

static Stats compare(const float *kai, const double *ref, size_t n) {
    Stats st; memset(&st, 0, sizeof st);
    st.worst_rel_i = st.worst_abs_i = -1;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += fabs(ref[i]);
    st.typical = sum / (double)n;
    double sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double k = (double)kai[i], r = ref[i];
        /* NaN must be caught EXPLICITLY -- a NaN silently loses every ">" test */
        if (isnan(k) || isnan(r) || isinf(k) || isinf(r)) { st.any_nan = 1; continue; }
        double a = fabs(k - r);
        double rel = a / (fabs(r) + 1e-9);
        if (isnan(a) || isnan(rel)) { st.any_nan = 1; continue; }
        sq += a * a;
        if (a   > st.worst_abs) { st.worst_abs = a;   st.worst_abs_i = (int)i; }
        if (rel > st.worst_rel) { st.worst_rel = rel; st.worst_rel_i = (int)i; }
        if (fabs(r) >= st.typical && rel > st.worst_rel_big) st.worst_rel_big = rel;
    }
    st.rms_abs = sqrt(sq / (double)n);
    st.rel_rms = st.rms_abs / (st.typical + 1e-30);
    return st;
}

/* ------------------------------------------------- negative controls
   A test that only ever passes proves nothing. These build the RHS with a
   DELIBERATELY WRONG nibble permutation and run the same GEMM, so the report
   can show how big the error would have been had kai_sme2_repack_q4g64() got
   the mapping wrong. Modes:
     1 = naive pass-through: kleidiai byte idx <- vdsp byte idx verbatim
         (low=col 2*idx, high=col 2*idx+1). The single most likely bug.
     2 = correct byte pairing but lo/hi nibble halves swapped.
     3 = correct permutation, but group index reversed (gi -> ng-1-gi):
         catches a group-offset/stride error while values stay individually right. */
static void build_fixture_wrong(int mode, const int8_t *code, const float *scales,
                                int out, int in, int ng, uint8_t *fixture) {
    size_t nbpb = (size_t)(BL / 2) + sizeof(uint16_t);
    size_t stride = (size_t)ng * nbpb;
    for (int r = 0; r < out; r++) {
        for (int gi = 0; gi < ng; gi++) {
            int src_gi = (mode == 3) ? (ng - 1 - gi) : gi;
            uint8_t *blk = fixture + (size_t)r * stride + (size_t)gi * nbpb;
            __fp16 h = (__fp16)scales[(size_t)r * ng + src_gi];
            uint16_t sb; memcpy(&sb, &h, 2); memcpy(blk, &sb, 2);
            uint8_t *values = blk + 2;
            const int8_t *c = code + (size_t)r * in + (size_t)src_gi * BL;
            for (int idx = 0; idx < BL / 2; idx++) {
                uint8_t lo, hi;
                if (mode == 1) { lo = (uint8_t)((c[2 * idx] + 8) & 0xF); hi = (uint8_t)((c[2 * idx + 1] + 8) & 0xF); }
                else           { lo = (uint8_t)((c[idx] + 8) & 0xF);     hi = (uint8_t)((c[idx + BL / 2] + 8) & 0xF); }
                if (mode == 2) { uint8_t t = lo; lo = hi; hi = t; }
                values[idx] = (uint8_t)((hi << 4) | lo);
            }
        }
    }
}

/* ------------------------------------------------------- scale sweep
   Multiplying every weight scale by a POWER OF TWO scales the entire pipeline
   exactly: fp16(F*s) == F*fp16(s) bit-for-bit (no mantissa change), the int
   dots are untouched, and every fp32 product/sum just shifts exponent. So a
   numerically sound implementation must report the SAME per-element ulp figure
   at every F. Any F-dependence localises a precision cliff (fp16 subnormal
   range, flush-to-zero, an exponent limit) rather than a permutation bug. */
static void scale_sweep(const TensorInfo *ti, const uint8_t *packed, const float *scales,
                        const int8_t *code, int M, unsigned long long seed,
                        size_t mr, size_t nr, size_t kr, size_t sr);

/* --------------------------------------------------------- guard mode */
static int guard_test(const TensorInfo *ti, const uint8_t *base) {
    int out = ti->out, in = ti->in;
    size_t need = kai_sme2_rhs_packed_bytes(out, in);
    if (need == SIZE_MAX) { printf("GUARD %s: rhs_packed_bytes=SIZE_MAX -> abort\n", ti->name); return 1; }
    long pgl = sysconf(_SC_PAGESIZE);
    size_t pg = (size_t)pgl;
    size_t region = ((need + pg - 1) / pg) * pg;
    uint8_t *map = mmap(NULL, region + pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    if (mprotect(map + region, pg, PROT_NONE) != 0) { perror("mprotect"); return 1; }
    uint8_t *dst = map + region - need;
    size_t slack = 0;
    if (((uintptr_t)dst & 63u) != 0) {
        uint8_t *al = (uint8_t *)((uintptr_t)dst & ~(uintptr_t)63);
        slack = (size_t)(dst - al);
        dst = al;
    }
    printf("GUARD %s out=%d in=%d need=%zu pagesize=%zu dst=%p guard_at=%p slack=%zu\n",
           ti->name, out, in, need, pg, (void *)dst, (void *)(map + region), slack);
    fflush(stdout);
    const uint8_t *packed = base + ti->doff;
    const float   *scales = (const float *)(base + ti->soff);
    int rc = kai_sme2_repack_q4g64(out, in, packed, scales, dst, need);
    printf("GUARD %s: repack rc=%d -- NO SIGSEGV, wrote within [dst, dst+need)%s\n",
           ti->name, rc, slack ? " (minus <=63B alignment slack)" : " exactly");
    return rc;
}

static void scale_sweep(const TensorInfo *ti, const uint8_t *packed, const float *scales,
                        const int8_t *code, int M, unsigned long long seed,
                        size_t mr, size_t nr, size_t kr, size_t sr) {
    int out = ti->out, in = ti->in, ng = ti->ng;
    size_t need = kai_sme2_rhs_packed_bytes(out, in);
    const double u32 = 5.9604644775390625e-8;

    float  *act     = malloc((size_t)M * in * sizeof(float));
    int8_t *actq    = malloc((size_t)M * in);
    float  *actsf16 = malloc((size_t)M * ng * sizeof(float));
    float  *sc      = malloc((size_t)out * ng * sizeof(float));
    uint8_t *dst    = malloc(need);
    float  *y_kai   = malloc((size_t)M * out * sizeof(float));
    double *y_f32s = malloc((size_t)M * out * sizeof(double));
    double *y_f16s = malloc((size_t)M * out * sizeof(double));
    double *y_mod  = malloc((size_t)M * out * sizeof(double));
    double *absum  = malloc((size_t)M * out * sizeof(double));
    size_t lhs_sz = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32_neon((size_t)M, (size_t)in, BL, mr, kr, sr);
    void *lhs_packed = calloc(1, lhs_sz);

    rng_seed(seed);
    for (size_t i = 0; i < (size_t)M * in; i++) act[i] = (float)((int)(xr() % 4096) - 2048) / 512.0f;
    lhs_quant_model(M, in, ng, act, actq, actsf16);
    kai_run_lhs_quant_pack_qsi8d32p_f32_neon((size_t)M, (size_t)in, BL, mr, kr, sr, 0,
                                             act, (size_t)in * sizeof(float), lhs_packed);

    printf("  SCALESWEEP %s out=%d in=%d (F is a power of 2 -> ulp MUST be F-invariant)\n", ti->name, out, in);
    for (int e = -8; e <= 8; e += 2) {
        float F = ldexpf(1.0f, e);
        for (size_t i = 0; i < (size_t)out * ng; i++) sc[i] = scales[i] * F;
        int rc = kai_sme2_repack_q4g64(out, in, packed, sc, dst, need);
        if (rc != 0) { printf("    F=2^%-3d repack rc=%d (scale out of range) -- skipped\n", e, rc); continue; }
        memset(y_kai, 0, (size_t)M * out * sizeof(float));
        kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(
            (size_t)M, (size_t)out, (size_t)in, BL, lhs_packed, dst, y_kai,
            (size_t)out * sizeof(float), sizeof(float), -FLT_MAX, FLT_MAX);
        int T = 8; if (T > out) T = out;
        pthread_t th[16]; RefJob jb[16];
        for (int t = 0; t < T; t++) {
            jb[t] = (RefJob){ .r0 = (int)((long)out * t / T), .r1 = (int)((long)out * (t + 1) / T),
                              .out = out, .in = in, .ng = ng, .M = M, .code = code, .scales = sc,
                              .act = act, .actq = actq, .actsf16 = actsf16,
                              .y_f32s = y_f32s, .y_f16s = y_f16s, .y_mod = y_mod, .absum = absum };
            pthread_create(&th[t], NULL, ref_worker, &jb[t]);
        }
        for (int t = 0; t < T; t++) pthread_join(th[t], NULL);
        double usq = 0.0, umax = 0.0; size_t nz = 0;
        double smin = INFINITY, smax = 0.0;
        for (size_t i = 0; i < (size_t)out * ng; i++) { double v = fabs(sc[i]); if (v < smin) smin = v; if (v > smax) smax = v; }
        for (size_t i = 0; i < (size_t)M * out; i++) {
            double budget = u32 * absum[i];
            if (budget <= 0.0) continue;
            double u = fabs((double)y_kai[i] - y_mod[i]) / budget;
            if (isnan(u)) continue;
            usq += u * u; if (u > umax) umax = u; nz++;
        }
        Stats a = compare(y_kai, y_f32s, (size_t)M * out);
        printf("    F=2^%-3d wscale|min|=%.3e |max|=%.3e  ulp_rms=%9.3f ulp_max=%10.3f  rel_rms_vs_truth=%.4e\n",
               e, smin, smax, sqrt(usq / (double)nz), umax, a.rel_rms);
        fflush(stdout);
    }
    free(act); free(actq); free(actsf16); free(sc); free(dst); free(y_kai);
    free(y_f32s); free(y_f16s); free(y_mod); free(absum); free(lhs_packed);
}

/* ------------------------------------------------------------------ main */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <layout> <blob> <M> <nseeds> <seed0> <tensor>...\n"
            "       %s --guard <layout> <blob> <tensor>\n", argv[0], argv[0]);
        return 2;
    }

    int guard_mode = (strcmp(argv[1], "--guard") == 0);
    int sweep_mode = (strcmp(argv[1], "--sweep") == 0);
    const char *layout, *blobpath;
    int M = 16, nseeds = 1;
    unsigned long long seed0 = 0xC0FFEE;
    int first_tensor;

    if (guard_mode || sweep_mode) {
        if (argc < 5) { fprintf(stderr, "need <layout> <blob> <tensor>\n"); return 2; }
        layout = argv[2]; blobpath = argv[3]; first_tensor = 4;
    } else {
        if (argc < 7) { fprintf(stderr, "need <layout> <blob> <M> <nseeds> <seed0> <tensor>...\n"); return 2; }
        layout = argv[1]; blobpath = argv[2];
        M = atoi(argv[3]); nseeds = atoi(argv[4]);
        seed0 = strtoull(argv[5], NULL, 0);
        first_tensor = 6;
    }

    printf("== kai_sme2_available()=%d  kai_sme2_min_m()=%d\n",
           kai_sme2_available(), kai_sme2_min_m());
    if (!kai_sme2_available()) { fprintf(stderr, "SME2 not available on this host\n"); return 1; }

    int fd = open(blobpath, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open blob: %s\n", strerror(errno)); return 1; }
    struct stat sb;
    if (fstat(fd, &sb) != 0) { perror("fstat"); return 1; }
    uint8_t *base = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap blob"); return 1; }
    printf("== blob %s size=%lld mmap ok\n", blobpath, (long long)sb.st_size);

    size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    size_t kr = kai_get_kr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    size_t sr = kai_get_sr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    printf("== mr=%zu nr=%zu kr=%zu sr=%zu\n", mr, nr, kr, sr);

    int overall_fail = 0;

    for (int ai = first_tensor; ai < argc; ai++) {
        TensorInfo ti;
        if (find_tensor(layout, argv[ai], &ti) != 0) { overall_fail = 1; continue; }
        if (strcmp(ti.kind, "q4g64") != 0) {
            fprintf(stderr, "%s: kind=%s not q4g64, skipping\n", ti.name, ti.kind);
            overall_fail = 1; continue;
        }
        if (ti.ng != ti.in / BL) {
            fprintf(stderr, "%s: ng=%d != in/64=%d\n", ti.name, ti.ng, ti.in / BL);
            overall_fail = 1; continue;
        }
        /* bounds check against the blob */
        size_t data_bytes  = (size_t)ti.out * (size_t)(ti.in / 2);
        size_t scale_bytes = (size_t)ti.out * (size_t)ti.ng * sizeof(float);
        if (ti.doff + data_bytes > (size_t)sb.st_size || ti.soff + scale_bytes > (size_t)sb.st_size) {
            fprintf(stderr, "%s: extends past blob end\n", ti.name); overall_fail = 1; continue;
        }

        const uint8_t *packed = base + ti.doff;
        const float   *scales = (const float *)(base + ti.soff);

        if (guard_mode) { if (guard_test(&ti, base) != 0) overall_fail = 1; continue; }

        int out = ti.out, in = ti.in, ng = ti.ng;
        printf("\n########## %s  out=%d in=%d ng=%d shape_ok=%d\n",
               ti.name, out, in, ng, kai_sme2_shape_ok(out, in));

        /* ---- scale sanity (fp16 range is the one real hazard of the repack) */
        {
            float smin = INFINITY, smax = 0.0f;
            int nonfinite = 0, zero = 0, fp16_of = 0, fp16_uf = 0;
            for (size_t i = 0; i < (size_t)out * ng; i++) {
                float s = scales[i];
                if (!isfinite(s)) { nonfinite++; continue; }
                if (s == 0.0f) { zero++; continue; }
                float as = fabsf(s);
                if (as < smin) smin = as;
                if (as > smax) smax = as;
                if (as > 65504.0f) fp16_of++;
                if (as < 6.103515625e-5f) fp16_uf++;   /* below fp16 min normal */
            }
            printf("  scales: n=%zu |min|=%.6g |max|=%.6g nonfinite=%d zero=%d fp16_overflow=%d fp16_subnormal=%d\n",
                   (size_t)out * ng, smin, smax, nonfinite, zero, fp16_of, fp16_uf);
        }

        /* ---- decode once (reused across seeds) */
        int8_t *code = malloc((size_t)out * in);
        if (!code) { fprintf(stderr, "oom code\n"); return 1; }
        for (int r = 0; r < out; r++)
            decode_row(packed + (size_t)r * (size_t)(in / 2), in, code + (size_t)r * in);

        /* ---- oracle: does decode_row() agree with vdsp's own q4_unpack_row()? */
        {
            int nsample = out < 64 ? out : 64;
            size_t bad = oracle_check_decode(packed, scales, code, out, in, ng, nsample);
            printf("  vdsp oracle (q4_unpack_row, %d sampled rows): mismatching elements=%zu%s\n",
                   nsample, bad, bad ? "  !! DECODE CONVENTION WRONG" : "  -> EXACT");
            if (bad) overall_fail = 1;
        }

        if (sweep_mode) {
            scale_sweep(&ti, packed, scales, code, M, seed0, mr, nr, kr, sr);
            free(code);
            continue;
        }

        /* ---- size query + canary buffer */
        size_t need = kai_sme2_rhs_packed_bytes(out, in);
        size_t need_direct = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon(
                                 (size_t)out, (size_t)in, nr, kr, BL);
        printf("  kai_sme2_rhs_packed_bytes=%zu  kai_get_rhs_packed_size_*=%zu  match=%s\n",
               need, need_direct, need == need_direct ? "YES" : "NO");
        if (need != need_direct) overall_fail = 1;

        uint8_t *dstbuf = malloc(need + CANARY_PAD);
        if (!dstbuf) { fprintf(stderr, "oom dst\n"); return 1; }
        memset(dstbuf, CANARY, need + CANARY_PAD);

        int rc = kai_sme2_repack_q4g64(out, in, packed, scales, dstbuf, need);
        printf("  kai_sme2_repack_q4g64 rc=%d\n", rc);
        if (rc != 0) {
            printf("  !! repack FAILED -- skipping GEMM for this tensor\n");
            overall_fail = 1; free(code); free(dstbuf); continue;
        }

        /* canary: nothing past `need` may have been touched */
        size_t first_bad = (size_t)-1;
        for (size_t i = 0; i < CANARY_PAD; i++)
            if (dstbuf[need + i] != CANARY) { first_bad = i; break; }
        size_t last_written = 0;
        for (size_t i = need; i-- > 0; ) if (dstbuf[i] != CANARY) { last_written = i + 1; break; }
        printf("  canary: overrun=%s (first dirty byte past need: %s%zu)  last_nonCANARY_in_range=%zu/%zu\n",
               first_bad == (size_t)-1 ? "NONE" : "YES",
               first_bad == (size_t)-1 ? "n/a +" : "+", first_bad == (size_t)-1 ? 0 : first_bad,
               last_written, need);
        if (first_bad != (size_t)-1) { printf("  !! BUFFER OVERRUN\n"); overall_fail = 1; }

        /* ---- what does the pack function actually STORE as the fp16 scale?
           The kernel reads rhs scales from the tail of each n-tile stride
           (see kai_matmul_*.c: rhs_packed + rhs_packed_stride - nr*num_blocks*2).
           Print stored-vs-source so any pre-scaling (e.g. /16 for the s4s0
           nibble encoding) is visible instead of assumed -- a stored scale that
           lands in the fp16 SUBNORMAL range loses mantissa bits, which is
           exactly the failure mode the scale sweep localised. */
        {
            size_t nbb_rhs = (size_t)(BL / 2) + 2;
            size_t stride0 = nr * (nbb_rhs * (size_t)ng);
            const uint16_t *sc16 = (const uint16_t *)(dstbuf + stride0 - (nr * (size_t)ng) * 2);
            printf("  stored rhs fp16 scales (n-tile 0): ");
            for (int i = 0; i < 4; i++) {
                __fp16 h; memcpy(&h, &sc16[i], 2);
                printf("[%d]=%.6g ", i, (double)(float)h);
            }
            printf("\n  source scales row0: ");
            for (int i = 0; i < 4; i++) printf("[%d]=%.6g ", i, scales[(size_t)i * ng]);
            printf("\n  ratio stored/source(row i,blk0): ");
            for (int i = 0; i < 4; i++) {
                __fp16 h; memcpy(&h, &sc16[i], 2);
                printf("%.4f ", (double)((float)h / scales[(size_t)i * ng]));
            }
            printf("\n");
        }

        /* ---- independent RHS-packed reconstruction (kai_test_correct2.c method) */
        size_t nbpb = (size_t)(BL / 2) + sizeof(uint16_t);
        size_t rhs_stride = (size_t)ng * nbpb;
        uint8_t *fixture = malloc((size_t)out * rhs_stride);
        if (!fixture) { fprintf(stderr, "oom fixture\n"); return 1; }
        for (int r = 0; r < out; r++) {
            for (int gi = 0; gi < ng; gi++) {
                uint8_t *blk = fixture + (size_t)r * rhs_stride + (size_t)gi * nbpb;
                __fp16 h = (__fp16)scales[(size_t)r * ng + gi];
                uint16_t sbits; memcpy(&sbits, &h, 2);
                memcpy(blk, &sbits, 2);
                uint8_t *values = blk + 2;
                const int8_t *c = code + (size_t)r * in + (size_t)gi * BL;
                for (int idx = 0; idx < BL / 2; idx++) {
                    uint8_t lo = (uint8_t)((c[idx] + 8) & 0xF);
                    uint8_t hi = (uint8_t)((c[idx + BL / 2] + 8) & 0xF);
                    values[idx] = (uint8_t)((hi << 4) | lo);
                }
            }
        }
        uint8_t *xref_packed = calloc(1, need);
        struct kai_rhs_pack_qs4cxs1s0_param rp = { .lhs_zero_point = 1, .rhs_zero_point = 8 };
        kai_run_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon(
            1, (size_t)out, (size_t)in, nr, kr, sr, BL, fixture, NULL, xref_packed, 0, &rp);
        size_t diffcnt = 0, firstdiff = (size_t)-1;
        for (size_t i = 0; i < need; i++)
            if (xref_packed[i] != dstbuf[i]) { if (firstdiff == (size_t)-1) firstdiff = i; diffcnt++; }
        printf("  xref bytewise vs repack: diff_bytes=%zu/%zu%s\n",
               diffcnt, need, diffcnt ? "" : "  -> BIT-IDENTICAL");
        if (diffcnt) printf("  first_diff_at=%zu (repack=0x%02x xref=0x%02x)  !! PERMUTATION/SCALE MISMATCH\n",
                            firstdiff, dstbuf[firstdiff], xref_packed[firstdiff]);
        free(fixture);
        if (diffcnt) overall_fail = 1;

        /* ---- LHS + GEMM + reference, per seed */
        float  *act     = malloc((size_t)M * in * sizeof(float));
        int8_t *actq    = malloc((size_t)M * in);
        float  *actsf16 = malloc((size_t)M * ng * sizeof(float));
        double *y_f32s  = malloc((size_t)M * out * sizeof(double));
        double *y_f16s  = malloc((size_t)M * out * sizeof(double));
        double *y_mod   = malloc((size_t)M * out * sizeof(double));
        double *absum   = malloc((size_t)M * out * sizeof(double));
        float  *y_kai   = malloc((size_t)M * out * sizeof(float));
        if (!act || !actq || !actsf16 || !y_f32s || !y_f16s || !y_mod || !absum || !y_kai) { fprintf(stderr, "oom work\n"); return 1; }

        size_t lhs_sz = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32_neon(
                            (size_t)M, (size_t)in, BL, mr, kr, sr);
        void *lhs_packed = calloc(1, lhs_sz);

        for (int s = 0; s < nseeds; s++) {
            unsigned long long seed = seed0 + (unsigned long long)s * 0x9E3779B97F4A7C15ULL;
            rng_seed(seed);
            for (size_t i = 0; i < (size_t)M * in; i++)
                act[i] = (float)((int)(xr() % 4096) - 2048) / 512.0f;   /* ~U[-4,4) */

            lhs_quant_model(M, in, ng, act, actq, actsf16);

            /* reference, threaded over output rows */
            int T = 8;
            if (T > out) T = out;
            pthread_t th[16]; RefJob jb[16];
            for (int t = 0; t < T; t++) {
                jb[t] = (RefJob){ .r0 = (int)((long)out * t / T), .r1 = (int)((long)out * (t + 1) / T),
                                  .out = out, .in = in, .ng = ng, .M = M,
                                  .code = code, .scales = scales, .act = act,
                                  .actq = actq, .actsf16 = actsf16,
                                  .y_f32s = y_f32s, .y_f16s = y_f16s, .y_mod = y_mod, .absum = absum };
                pthread_create(&th[t], NULL, ref_worker, &jb[t]);
            }
            for (int t = 0; t < T; t++) pthread_join(th[t], NULL);

            memset(lhs_packed, 0, lhs_sz);
            kai_run_lhs_quant_pack_qsi8d32p_f32_neon((size_t)M, (size_t)in, BL, mr, kr, sr, 0,
                                                     act, (size_t)in * sizeof(float), lhs_packed);
            memset(y_kai, 0, (size_t)M * out * sizeof(float));
            kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(
                (size_t)M, (size_t)out, (size_t)in, BL, lhs_packed, dstbuf, y_kai,
                (size_t)out * sizeof(float), sizeof(float), -FLT_MAX, FLT_MAX);

            Stats a = compare(y_kai, y_f32s, (size_t)M * out);  /* vs ground truth */
            Stats b = compare(y_kai, y_f16s, (size_t)M * out);  /* vs fp16-scale truth */
            Stats c = compare(y_kai, y_mod,  (size_t)M * out);  /* vs exact kernel model */
            int nan_ref = 0;
            for (size_t i = 0; i < (size_t)M * out; i++)
                if (isnan(y_f32s[i]) || isinf(y_f32s[i])) { nan_ref = 1; break; }

            double ratio = a.worst_abs / (a.typical + 1e-9);
            /* PASS gate.
               C6 no NaN/Inf anywhere.
               C7 (DECISIVE, repack correctness): the KleidiAI output must agree
                  with the EXACT KERNEL MODEL -- rebuilt independently from
                  vdsp's own decoded nibbles + scales -- at fp32-rounding level.
                  One misplaced nibble, group offset or scale makes this O(1).
               C8 (kernel accuracy, NOT a repack property): rel_rms vs the fp32
                  ground truth must sit in the int8-activation-quantization band.
               worst_abs/typical is reported but is NOT a gate: it is a
               max-over-N/mean statistic, so it grows with output count and
               shrinks with |y|. kai_test_correct2.c's loose 5e-2 bound on it is
               a heuristic that a 229376-element tensor with small mean |y|
               (e.g. up_proj) exceeds even when every byte is provably correct. */
            /* Normalize the model-delta by the magnitude the kernel's fp32
               accumulator actually traverses (sum of |per-block contributions|),
               NOT by |y|. Where blocks cancel, |y| << absum and a |y|-relative
               bound punishes a numerically perfect kernel. u32 = 2^-24. */
            const double u32 = 5.9604644775390625e-8;   /* 2^-24 */
            double mean_absum = 0.0, max_absum = 0.0;
            for (size_t i = 0; i < (size_t)M * out; i++) {
                mean_absum += absum[i];
                if (absum[i] > max_absum) max_absum = absum[i];
            }
            mean_absum /= (double)((size_t)M * out);
            /* PER-ELEMENT normalization: each output's fp32 rounding budget is set
               by ITS OWN accumulated magnitude. Normalizing by the mean instead
               mis-scores heavy-tailed tensors (a few outlier rows carry most of
               the magnitude) -- which is exactly what v_proj turned out to be. */
            double ulp_sq = 0.0, ulp_max = 0.0;
            for (size_t i = 0; i < (size_t)M * out; i++) {
                double budget = u32 * absum[i];
                if (budget <= 0.0) continue;
                double u = fabs((double)y_kai[i] - y_mod[i]) / budget;
                if (isnan(u)) continue;
                ulp_sq += u * u;
                if (u > ulp_max) ulp_max = u;
            }
            double ulp_units = sqrt(ulp_sq / (double)((size_t)M * out));
            double cancel = mean_absum / (a.typical + 1e-300);
            double tail = max_absum / (mean_absum + 1e-300);

            int c6 = (!a.any_nan && !c.any_nan && !nan_ref);
            int c7 = (ulp_units < 256.0);
            int c8 = (a.rel_rms < 1e-2);
            int pass = c6 && c7 && c8;
            printf("  RESULT tensor=%s out=%d in=%d M=%d seed=0x%llx rc=%d\n"
                   "    vs GROUND TRUTH (fp32 scales, fp32 acts): worst_rel=%.4e worst_rel_big=%.4e "
                   "worst_abs=%.4e typical=%.4e worst_abs/typical=%.4e rel_rms=%.4e any_nan=%d\n"
                   "    vs fp16-scale truth                     : worst_rel_big=%.4e worst_abs=%.4e rel_rms=%.4e\n"
                   "    vs EXACT KERNEL MODEL (int8 acts)       : worst_rel_big=%.4e worst_abs=%.4e rel_rms=%.4e any_nan=%d\n"
                   "    accum magnitude: mean_absum=%.4e max/mean=%.1f cancel=absum/|y|=%.3f  "
                   "model_delta per-elem ulp: rms=%.2f max=%.2f\n"
                   "    -> %s  [C6_nan=%s C7_model=%s C8_kernel_acc=%s]\n",
                   ti.name, out, in, M, seed, rc,
                   a.worst_rel, a.worst_rel_big, a.worst_abs, a.typical, ratio, a.rel_rms, a.any_nan,
                   b.worst_rel_big, b.worst_abs, b.rel_rms,
                   c.worst_rel_big, c.worst_abs, c.rel_rms, c.any_nan,
                   mean_absum, tail, cancel, ulp_units, ulp_max,
                   pass ? "PASS" : "FAIL",
                   c6 ? "ok" : "FAIL", c7 ? "ok" : "FAIL", c8 ? "ok" : "FAIL");
            if (!pass) {
                overall_fail = 1;
                if (a.worst_abs_i >= 0)
                    printf("    worst_abs at i=%d (m=%d,r=%d): truth=%.8g model=%.8g kai=%.8g\n",
                           a.worst_abs_i, a.worst_abs_i / out, a.worst_abs_i % out,
                           y_f32s[a.worst_abs_i], y_mod[a.worst_abs_i], y_kai[a.worst_abs_i]);
                if (c.worst_abs_i >= 0)
                    printf("    worst model-delta at i=%d: model=%.8g kai=%.8g\n",
                           c.worst_abs_i, y_mod[c.worst_abs_i], y_kai[c.worst_abs_i]);
                printf("    first 6 truth: "); for (int i = 0; i < 6; i++) printf("%.6g ", y_f32s[i]); printf("\n");
                printf("    first 6 model: "); for (int i = 0; i < 6; i++) printf("%.6g ", y_mod[i]);  printf("\n");
                printf("    first 6 kai  : "); for (int i = 0; i < 6; i++) printf("%.6g ", y_kai[i]);  printf("\n");
            }
            /* ---- negative controls, once per tensor (proves the harness discriminates) */
            if (s == 0) {
                uint8_t *wf = malloc((size_t)out * rhs_stride);
                uint8_t *wp = calloc(1, need);
                float   *y_w = malloc((size_t)M * out * sizeof(float));
                for (int mode = 1; mode <= 3; mode++) {
                    build_fixture_wrong(mode, code, scales, out, in, ng, wf);
                    memset(wp, 0, need);
                    kai_run_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon(
                        1, (size_t)out, (size_t)in, nr, kr, sr, BL, wf, NULL, wp, 0, &rp);
                    size_t nd = 0;
                    for (size_t i = 0; i < need; i++) if (wp[i] != dstbuf[i]) nd++;
                    memset(y_w, 0, (size_t)M * out * sizeof(float));
                    kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(
                        (size_t)M, (size_t)out, (size_t)in, BL, lhs_packed, wp, y_w,
                        (size_t)out * sizeof(float), sizeof(float), -FLT_MAX, FLT_MAX);
                    Stats w = compare(y_w, y_f32s, (size_t)M * out);
                    printf("    NEGCTL mode=%d (%s): rhs_diff_bytes=%zu/%zu  rel_rms=%.4e "
                           "worst_abs/typical=%.4e  -> harness %s\n",
                           mode,
                           mode == 1 ? "naive byte pass-through" :
                           mode == 2 ? "lo/hi nibble halves swapped" : "group order reversed",
                           nd, need, w.rel_rms, w.worst_abs / (w.typical + 1e-9),
                           (w.rel_rms > 1e-2) ? "DETECTS" : "MISSES(!!)");
                }
                free(wf); free(wp); free(y_w);
            }
            fflush(stdout);
        }

        free(lhs_packed); free(act); free(actq); free(actsf16);
        free(y_f32s); free(y_f16s); free(y_mod); free(absum); free(y_kai);
        free(xref_packed); free(dstbuf); free(code);
    }

    printf("\n==== OVERALL: %s\n", overall_fail ? "FAIL" : "PASS");
    return overall_fail ? 1 : 0;
}
