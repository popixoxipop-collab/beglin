// D-export Phase 1 oracle test: write a synthetic GGUF file exercising every scalar/array KV
// type and F32 tensors, then read it back with gguf_load.c and diff against what was written.
#include "gguf_write.h"
#include "gguf_load.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } } while (0)

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <out.gguf>\n", argv[0]); return 1; }
    const char *path = argv[1];

    // ---- Write ----
    GgufWriter *w = gguf_w_open(path);
    if (!w) { perror("gguf_w_open"); return 1; }

    gguf_w_kv_str(w, "general.architecture", "testarch", strlen("testarch"));
    gguf_w_kv_u32(w, "testarch.block_count", 7);
    gguf_w_kv_u64(w, "testarch.embedding_length", 12345678901234ULL);
    gguf_w_kv_i64(w, "testarch.some_signed", -42);
    gguf_w_kv_f32(w, "testarch.rope_freq_base", 1000000.0f);
    gguf_w_kv_f64(w, "testarch.eps", 1e-6);
    gguf_w_kv_bool(w, "testarch.flag_true", 1);
    gguf_w_kv_bool(w, "testarch.flag_false", 0);

    GgufStr strs[3];
    strs[0].ptr = "hello"; strs[0].len = 5;
    strs[1].ptr = "world!"; strs[1].len = 6;
    strs[2].ptr = ""; strs[2].len = 0;
    gguf_w_kv_str_array(w, "tokenizer.ggml.tokens", strs, 3);

    int32_t i32arr[4] = {1, -2, 3, -4};
    gguf_w_kv_i32_array(w, "tokenizer.ggml.token_type", i32arr, 4);

    float f32arr[3] = {1.5f, -2.25f, 0.0f};
    gguf_w_kv_f32_array(w, "tokenizer.ggml.scores", f32arr, 3);

    // Two F32 tensors, deliberately small and odd-shaped to exercise alignment padding.
    float t0data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};   // shape [2,3]
    uint64_t t0ne[2] = {2, 3};
    gguf_w_add_tensor(w, "test.tensor0", GGML_TYPE_F32, 2, t0ne, t0data, sizeof(t0data));

    float t1data[5] = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f};   // shape [5]
    uint64_t t1ne[1] = {5};
    gguf_w_add_tensor(w, "test.tensor1", GGML_TYPE_F32, 1, t1ne, t1data, sizeof(t1data));

    if (!gguf_w_finish(w)) { fprintf(stderr, "gguf_w_finish failed\n"); return 1; }
    printf("wrote %s\n", path);

    // ---- Read back with gguf_load.c ----
    GgufFile *f = gguf_open(path);
    CHECK(f != NULL, "gguf_open failed on our own written file");
    if (!f) return failures ? 1 : 0;

    const char *sptr; uint64_t slen;
    CHECK(gguf_kv_str(f, "general.architecture", &sptr, &slen) && slen == 8 && !memcmp(sptr, "testarch", 8), "general.architecture");

    uint64_t u;
    CHECK(gguf_kv_u64(f, "testarch.block_count", &u) && u == 7, "block_count");
    CHECK(gguf_kv_u64(f, "testarch.embedding_length", &u) && u == 12345678901234ULL, "embedding_length");
    int64_t iv;
    CHECK(gguf_kv_i64(f, "testarch.some_signed", &iv) && iv == -42, "some_signed");
    double dv;
    CHECK(gguf_kv_f64(f, "testarch.rope_freq_base", &dv) && fabs(dv - 1000000.0) < 1.0, "rope_freq_base");
    CHECK(gguf_kv_f64(f, "testarch.eps", &dv) && fabs(dv - 1e-6) < 1e-12, "eps");
    int bv;
    CHECK(gguf_kv_bool(f, "testarch.flag_true", &bv) && bv == 1, "flag_true");
    CHECK(gguf_kv_bool(f, "testarch.flag_false", &bv) && bv == 0, "flag_false");

    const GgufStr *rstrs; uint64_t rn;
    CHECK(gguf_kv_str_array(f, "tokenizer.ggml.tokens", &rstrs, &rn) && rn == 3, "tokens array len");
    if (rn == 3) {
        CHECK(rstrs[0].len == 5 && !memcmp(rstrs[0].ptr, "hello", 5), "tokens[0]");
        CHECK(rstrs[1].len == 6 && !memcmp(rstrs[1].ptr, "world!", 6), "tokens[1]");
        CHECK(rstrs[2].len == 0, "tokens[2] empty");
    }

    const int32_t *ri32; uint64_t rn2;
    CHECK(gguf_kv_i32_array(f, "tokenizer.ggml.token_type", &ri32, &rn2) && rn2 == 4, "token_type array len");
    if (rn2 == 4) CHECK(ri32[0]==1 && ri32[1]==-2 && ri32[2]==3 && ri32[3]==-4, "token_type values");

    const float *rf32; uint64_t rn3;
    CHECK(gguf_kv_f32_array(f, "tokenizer.ggml.scores", &rf32, &rn3) && rn3 == 3, "scores array len");
    if (rn3 == 3) CHECK(rf32[0]==1.5f && rf32[1]==-2.25f && rf32[2]==0.0f, "scores values");

    const GgufTensorInfo *t0 = gguf_find_tensor(f, "test.tensor0");
    CHECK(t0 != NULL && t0->n_dims==2 && t0->ne[0]==2 && t0->ne[1]==3, "tensor0 shape");
    if (t0) {
        const float *d0 = (const float *)gguf_tensor_data(f, t0);
        CHECK(!memcmp(d0, t0data, sizeof(t0data)), "tensor0 data bytes");
    }
    const GgufTensorInfo *t1 = gguf_find_tensor(f, "test.tensor1");
    CHECK(t1 != NULL && t1->n_dims==1 && t1->ne[0]==5, "tensor1 shape");
    if (t1) {
        const float *d1 = (const float *)gguf_tensor_data(f, t1);
        CHECK(!memcmp(d1, t1data, sizeof(t1data)), "tensor1 data bytes");
    }

    gguf_close(f);

    if (failures) { fprintf(stderr, "%d CHECK(S) FAILED\n", failures); return 1; }
    printf("ALL CHECKS PASSED (self round-trip via gguf_load.c)\n");
    return 0;
}
