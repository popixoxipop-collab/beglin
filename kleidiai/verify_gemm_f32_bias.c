// verify_gemm_f32_bias.c -- Phase 4 check specifically for kai_sme2_gemm_f32()'s
// bias-add path, which verify_real_repack.c (Phase 3) never exercised (it
// only tested W@x with no bias). Reads a REAL q_proj.weight (q4g64) + its
// REAL q_proj.bias (f32) from the Qwen2.5-1.5B production blob (Qwen has
// qkv_bias=1; Llama-3.1-8B in this project's other weight set does not carry
// attention bias, so this is the only model with a real bias tensor to test
// against) and checks kai_sme2_gemm_f32(M, out, in, x, rhs, bias, y, scratch)
// against a from-scratch double-precision W@x+bias reference built directly
// from vdsp's own packed nibbles/scales.
//
// NEW FILE. Reads (never writes) sme2_kai.c/.h and the production weight blob.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "../q4gemv.h"
#include "../sme2_kai.h"

#define BL 64

static uint64_t rng_state;
static void rng_seed(uint64_t s) { rng_state = s ? s : 0x243F6A8885A308D3ULL; }
static uint32_t xr(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

typedef struct { char name[512]; char kind[32]; size_t off; int out, in, ng; size_t soff; } TI;

static int find_t(const char *layout, const char *want, TI *ti) {
    FILE *f = fopen(layout, "r"); if (!f) { perror("layout"); return -1; }
    char line[2048];
    while (fgets(line, sizeof line, f)) {
        char nm[512], kd[32]; unsigned long long o; int out_, in_, ng_; long long so;
        if (sscanf(line, "%511s %31s %llu %d %d %d %lld", nm, kd, &o, &out_, &in_, &ng_, &so) != 7) continue;
        if (strcmp(nm, want)) continue;
        snprintf(ti->name, sizeof ti->name, "%s", nm); snprintf(ti->kind, sizeof ti->kind, "%s", kd);
        ti->off = (size_t)o; ti->out = out_; ti->in = in_; ti->ng = ng_; ti->soff = (size_t)so;
        fclose(f); return 0;
    }
    fclose(f); fprintf(stderr, "'%s' not found\n", want); return -1;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <layout> <blob> <M>\n", argv[0]); return 2; }
    const char *layout = argv[1], *blobpath = argv[2];
    int M = atoi(argv[3]);

    printf("kai_sme2_available=%d\n", kai_sme2_available());
    if (!kai_sme2_available()) { fprintf(stderr, "no SME2\n"); return 1; }

    int fd = open(blobpath, O_RDONLY); if (fd < 0) { perror("open"); return 1; }
    struct stat sb; fstat(fd, &sb);
    uint8_t *base = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }

    TI w, b;
    if (find_t(layout, "model.layers.0.self_attn.q_proj.weight", &w) != 0) return 1;
    if (find_t(layout, "model.layers.0.self_attn.q_proj.bias", &b) != 0) return 1;
    if (strcmp(w.kind, "q4g64") != 0) { fprintf(stderr, "not q4g64\n"); return 1; }
    if (strcmp(b.kind, "f32") != 0) { fprintf(stderr, "bias not f32\n"); return 1; }
    if (b.out != w.out) { fprintf(stderr, "bias len %d != out %d\n", b.out, w.out); return 1; }

    int out = w.out, in = w.in, ng = w.ng;
    printf("tensor=%s out=%d in=%d ng=%d  bias=%s\n", w.name, out, in, ng, b.name);

    const uint8_t *packed = base + w.off;
    const float   *scales = (const float *)(base + w.soff);
    const float   *bias   = (const float *)(base + b.off);

    // repack RHS
    size_t need = kai_sme2_rhs_packed_bytes(out, in);
    void *rhs = malloc(need);
    int rc = kai_sme2_repack_q4g64(out, in, packed, scales, rhs, need);
    printf("repack rc=%d need=%zu\n", rc, need);
    if (rc != 0) return 1;

    // random activation
    float *x = malloc((size_t)M * in * sizeof(float));
    rng_seed(0xB1A5);
    for (size_t i = 0; i < (size_t)M * in; i++) x[i] = (float)((int)(xr() % 4096) - 2048) / 512.0f;

    // reference: double-precision dequant(packed,scales) @ x + bias, WITH bias
    double *y_ref = malloc((size_t)M * out * sizeof(double));
    float *tile = malloc((size_t)in * sizeof(float));
    for (int r = 0; r < out; r++) {
        q4_unpack_row(packed + (size_t)r * (in / 2), scales + (size_t)r * ng, tile, in);
        for (int m = 0; m < M; m++) {
            const float *xm = x + (size_t)m * in;
            double acc = 0.0;
            for (int k = 0; k < in; k++) acc += (double)tile[k] * (double)xm[k];
            y_ref[(size_t)m * out + r] = acc + (double)bias[r];
        }
    }
    free(tile);

    // WITHOUT bias, for isolating the bias-add step specifically
    double *y_ref_nobias = malloc((size_t)M * out * sizeof(double));
    for (size_t i = 0; i < (size_t)M * out; i++) y_ref_nobias[i] = y_ref[i] - (double)bias[i % out];

    size_t lhs_bytes = kai_sme2_lhs_scratch_bytes(M, in);
    void *scratch = malloc(lhs_bytes);
    printf("lhs_scratch_bytes=%zu\n", lhs_bytes);

    float *y_with_bias = calloc((size_t)M * out, sizeof(float));
    kai_sme2_gemm_f32(M, out, in, x, rhs, bias, y_with_bias, scratch);

    float *y_no_bias = calloc((size_t)M * out, sizeof(float));
    kai_sme2_gemm_f32(M, out, in, x, rhs, NULL, y_no_bias, scratch);

    // worst_rel (max over ALL elements) blows up near ref==0 -- same benign
    // phenomenon Phase 3's verify_real_repack.c separated into worst_rel_big
    // (restricted to |ref|>=typical). rel_rms (rms_abs/typical) is the stable
    // metric used as the actual gate there; reproduced here for the same reason.
    double worst_rel_with = 0, worst_abs_with = 0, typ = 0, worst_rel_no = 0, worst_abs_no = 0;
    double sq_with = 0, sq_no = 0;
    double sum_bias_delta = 0, worst_bias_delta = 0;
    int any_nan = 0;
    for (size_t i = 0; i < (size_t)M * out; i++) typ += fabs(y_ref[i]);
    typ /= (double)((size_t)M * out);
    for (size_t i = 0; i < (size_t)M * out; i++) {
        if (isnan(y_with_bias[i]) || isnan(y_no_bias[i])) { any_nan = 1; continue; }
        double aw = fabs((double)y_with_bias[i] - y_ref[i]);
        double rw = aw / (fabs(y_ref[i]) + 1e-9);
        if (aw > worst_abs_with) worst_abs_with = aw;
        if (rw > worst_rel_with) worst_rel_with = rw;
        sq_with += aw * aw;
        double an = fabs((double)y_no_bias[i] - y_ref_nobias[i]);
        double rn = an / (fabs(y_ref_nobias[i]) + 1e-9);
        if (an > worst_abs_no) worst_abs_no = an;
        if (rn > worst_rel_no) worst_rel_no = rn;
        sq_no += an * an;
        // isolate JUST the bias-add: (with_bias output) - (no_bias output) should equal bias[r] exactly (fp32 add)
        double bd = fabs(((double)y_with_bias[i] - (double)y_no_bias[i]) - (double)bias[i % out]);
        sum_bias_delta += bd;
        if (bd > worst_bias_delta) worst_bias_delta = bd;
    }
    double rel_rms_with = sqrt(sq_with / (double)((size_t)M * out)) / (typ + 1e-30);
    double rel_rms_no   = sqrt(sq_no   / (double)((size_t)M * out)) / (typ + 1e-30);
    printf("vs ground truth (WITH bias): worst_rel(unfiltered,blows up near ref=0)=%.4e worst_abs=%.4e "
           "typical=%.4e rel_rms=%.4e any_nan=%d\n",
           worst_rel_with, worst_abs_with, typ, rel_rms_with, any_nan);
    printf("vs ground truth (NO bias)  : worst_rel(unfiltered)=%.4e worst_abs=%.4e rel_rms=%.4e\n",
           worst_rel_no, worst_abs_no, rel_rms_no);
    printf("bias-add isolation: (with-no)-bias[r] worst=%.4e mean=%.4e  (expect ~0, pure fp32 add)\n",
           worst_bias_delta, sum_bias_delta / (double)((size_t)M * out));

    // rel_rms is the stable gate (matches verify_real_repack.c's C8); the
    // unfiltered worst_rel above is printed for visibility only, same
    // "blows up near ref==0" caveat documented there.
    int pass = !any_nan && rel_rms_with < 1e-2 && worst_bias_delta < 1e-3;
    printf("==== %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
