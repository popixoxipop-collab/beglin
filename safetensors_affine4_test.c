#include "safetensors_quants.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int closef(float a, float b) {
    return fabsf(a - b) < 1e-6f;
}

int main(void) {
    // Exact mx.quantize() oracle from installed MLX, input [-31..32],
    // group_size=64, bits=4, mode=affine.
    const uint32_t codes[8] = {
        4009754623u, 3437092334u, 2864430028u, 2291767722u,
        1719105416u, 1146443110u, 573780804u, 1118498u,
    };
    // BF16 encodings of -4.0 and 32.0 returned by the same oracle.
    const uint16_t scale[1] = { 0xc080u };
    const uint16_t bias[1] = { 0x4200u };
    float out[64] = {0};

    if (!safetensors_affine4_supported(
            ST_TYPE_U32, ST_TYPE_BF16, ST_TYPE_BF16)) {
        fprintf(stderr, "affine4 support probe failed\n");
        return 2;
    }
    safetensors_dequant_affine4_matrix(
        ST_TYPE_U32, codes,
        ST_TYPE_BF16, scale,
        ST_TYPE_BF16, bias,
        out, 1, 64, 64);

    const float expected_first16[16] = {
        -28,-28,-28,-28,-28,-28,-24,-24,
        -24,-24,-20,-20,-20,-20,-16,-16,
    };
    for (int i = 0; i < 16; i++) {
        if (!closef(out[i], expected_first16[i])) {
            fprintf(stderr, "oracle mismatch i=%d got=%g expected=%g\n",
                    i, out[i], expected_first16[i]);
            return 3;
        }
    }
    if (!closef(out[62], 32.0f) || !closef(out[63], 32.0f)) {
        fprintf(stderr, "oracle tail mismatch got=[%g,%g]\n", out[62], out[63]);
        return 4;
    }
    puts("PASS mlx-affine4 oracle");
    return 0;
}
