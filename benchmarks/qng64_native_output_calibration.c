/* Per-tensor calibration using BEGLIN's REAL qng64_dense.c CPU encoder/GEMV.
 * Synthetic deterministic vectors only. NOT original DeepSeek weights or logits.
 * Measured importance is per-role FP32 output energy at ONE synthetic input.
 * With these independent GEMVs, importance * rel_l2^2 equals output SSE,
 * so exact DP minimizes the actual stacked output L2 error for this fixture.
 */
#define _POSIX_C_SOURCE 200809L
#include "qng64_dense.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ROLES = 6, OUT = 64, IN = 512, GROUP = 64 };
static const char *const role_names[ROLES] = {
    "q_proj", "kv_a_proj_with_mqa", "shared_up_proj",
    "shared_down_proj", "q_proj", "shared_down_proj"
};
static const int layers[ROLES] = {0, 0, 0, 0, 1, 1};

static void die(const char *what) {
    fprintf(stderr, "BEGLIN_CALIBRATION_FAIL: %s\n", what);
    exit(2);
}

int main(void) {
    float x[IN], weights[OUT * IN], scales[OUT * (IN / GROUP)];
    float y[OUT];
    uint8_t packed[OUT * (IN / GROUP) * (15 * 8)];
    double reference[OUT];

    for (int i = 0; i < IN; ++i)
        x[i] = 0.08f * sinf((float)i * 0.0452f) +
               0.04f * cosf((float)i * 0.0119f);

    printf("{\"schema\":\"beglin-qng64-per-tensor-budget-v1\","
           "\"model_id\":\"SYNTHETIC_NATIVE_CPU_GEMV_NOT_LLM\","
           "\"checkpoint_sha256\":\"%064d\","
           "\"backend\":\"cpu\",\"budget_bytes\":184320,"
           "\"synthetic_only\":true,"
           "\"tensors\":[", 0);

    for (int role = 0; role < ROLES; ++role) {
        double reference_energy = 0.0;
        for (int i = 0; i < OUT * IN; ++i)
            weights[i] =
                0.065f * sinf((float)(i + 7 * role) * 0.0132f) +
                0.026f * cosf((float)(i + 3 * role) * 0.0729f);

        for (int row = 0; row < OUT; ++row) {
            double value = 0.0;
            for (int i = 0; i < IN; ++i)
                value += (double)weights[row * IN + i] * (double)x[i];
            reference[row] = value;
            reference_energy += value * value;
        }
        if (!(reference_energy > 0.0) || !isfinite(reference_energy))
            die("nonpositive reference output norm");
        if (role) printf(",");
        printf("{\"name\":\"synthetic.layer.%d.%s.%d\","
               "\"role\":\"%s\",\"layer\":%d,"
               "\"elements\":%d,\"importance\":%.17g,\"candidates\":[",
               layers[role], role_names[role], role, role_names[role],
               layers[role], OUT * IN, reference_energy);

        int first = 1;
        for (int n = 2; n <= 15; ++n) {
            if (n == 4) continue; /* q4g64 is separate, not qNg64(n=4). */
            const size_t bytes = qng64_packed_bytes(OUT, IN, n);
            if (!bytes || bytes > sizeof packed)
                die("unsupported qng64 code width");
            if (qng64_quantize_f32(weights, OUT, IN, n, packed, scales) != 0)
                die("native quantizer returned failure");
            qng64_matvec_f32(packed, scales, n, x, NULL, y, OUT, IN);
            double squared_error = 0.0;
            for (int row = 0; row < OUT; ++row) {
                if (!isfinite(y[row]))
                    die("nonfinite native GEMV output");
                const double delta = reference[row] - (double)y[row];
                squared_error += delta * delta;
            }
            const double rel_l2 = sqrt(squared_error / reference_energy);
            if (!isfinite(rel_l2))
                die("nonfinite calibrated output L2");
            if (!first) printf(",");
            printf("{\"n\":%d,\"rel_l2\":%.17g}", n, rel_l2);
            first = 0;
        }
        printf("]}");
    }
    printf("]}\n");
    return 0;
}
