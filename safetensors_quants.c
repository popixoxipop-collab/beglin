// safetensors_quants.c -- see safetensors_quants.h. bf16_to_f32()/f16_to_f32() are extracted
// verbatim from safetensors_verify.c (already checksum-verified 290/290 exact against a real
// downloaded checkpoint this session) -- this is pure code motion, not new math.
#include "safetensors_quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// BF16 is exactly the top 16 bits of an FP32 value (truncated mantissa) -- no lookup table or
// bit-fiddling needed, unlike true half-precision F16.
static float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f; memcpy(&f, &bits, 4);
    return f;
}

// True half-precision needs real exponent/mantissa rebiasing (subnormal/inf/nan handled
// explicitly), implemented by hand -- verbatim from safetensors_verify.c.
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {
            // subnormal f16 -> normalized f32
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);  // inf/nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, 4);
    return f;
}

int safetensors_dequant_supported(SafetensorsType dtype) {
    return dtype == ST_TYPE_F32 || dtype == ST_TYPE_F16 || dtype == ST_TYPE_BF16;
}

void safetensors_dequant_row(SafetensorsType dtype, const void *raw, float *out, uint64_t n) {
    const uint8_t *r = (const uint8_t *)raw;
    if (dtype == ST_TYPE_F32) {
        memcpy(out, r, (size_t)n * 4);
        return;
    }
    if (dtype == ST_TYPE_F16 || dtype == ST_TYPE_BF16) {
        for (uint64_t i = 0; i < n; i++) {
            uint16_t h; memcpy(&h, r + i * 2, 2);
            out[i] = (dtype == ST_TYPE_BF16) ? bf16_to_f32(h) : f16_to_f32(h);
        }
        return;
    }
    fprintf(stderr, "FATAL: safetensors_dequant_row: unsupported dtype %s\n", safetensors_type_name(dtype));
    exit(1);
}


static float st_scalar_to_f32(SafetensorsType dtype, const uint8_t *raw) {
    if (dtype == ST_TYPE_F32) {
        float v; memcpy(&v, raw, sizeof(v)); return v;
    }
    if (dtype == ST_TYPE_F16) {
        uint16_t h; memcpy(&h, raw, sizeof(h)); return f16_to_f32(h);
    }
    if (dtype == ST_TYPE_BF16) {
        uint16_t h; memcpy(&h, raw, sizeof(h)); return bf16_to_f32(h);
    }
    fprintf(stderr, "FATAL: affine4 metadata dtype %s is not floating-point\n",
            safetensors_type_name(dtype));
    exit(1);
}

static size_t st_scalar_width(SafetensorsType dtype) {
    if (dtype == ST_TYPE_F32) return 4;
    if (dtype == ST_TYPE_F16 || dtype == ST_TYPE_BF16) return 2;
    return 0;
}

int safetensors_affine4_supported(
    SafetensorsType codes_dtype,
    SafetensorsType scales_dtype,
    SafetensorsType biases_dtype) {
    return codes_dtype == ST_TYPE_U32
        && st_scalar_width(scales_dtype) != 0
        && st_scalar_width(biases_dtype) != 0;
}

void safetensors_dequant_affine4_matrix(
    SafetensorsType codes_dtype,
    const void *codes_raw,
    SafetensorsType scales_dtype,
    const void *scales_raw,
    SafetensorsType biases_dtype,
    const void *biases_raw,
    float *out,
    uint64_t rows,
    uint64_t cols,
    uint32_t group_size) {
    if (!safetensors_affine4_supported(
            codes_dtype, scales_dtype, biases_dtype)) {
        fprintf(stderr,
                "FATAL: unsupported MLX affine4 dtypes codes=%s scales=%s biases=%s\n",
                safetensors_type_name(codes_dtype),
                safetensors_type_name(scales_dtype),
                safetensors_type_name(biases_dtype));
        exit(1);
    }
    if (!rows || !cols || !group_size || (group_size % 8) != 0
            || (cols % group_size) != 0) {
        fprintf(stderr,
                "FATAL: invalid MLX affine4 shape rows=%llu cols=%llu group_size=%u\n",
                (unsigned long long)rows,
                (unsigned long long)cols,
                (unsigned)group_size);
        exit(1);
    }

    const uint8_t *codes = (const uint8_t *)codes_raw;
    const uint8_t *scales = (const uint8_t *)scales_raw;
    const uint8_t *biases = (const uint8_t *)biases_raw;
    const size_t sw = st_scalar_width(scales_dtype);
    const size_t bw = st_scalar_width(biases_dtype);
    const uint64_t groups_per_row = cols / group_size;
    const uint64_t words_per_group = group_size / 8;
    const uint64_t words_per_row = cols / 8;

    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t g = 0; g < groups_per_row; g++) {
            const uint64_t meta_index = r * groups_per_row + g;
            const float scale = st_scalar_to_f32(
                scales_dtype, scales + meta_index * sw);
            const float bias = st_scalar_to_f32(
                biases_dtype, biases + meta_index * bw);
            for (uint64_t w = 0; w < words_per_group; w++) {
                uint32_t word;
                const uint64_t word_index =
                    r * words_per_row + g * words_per_group + w;
                memcpy(&word, codes + word_index * sizeof(uint32_t), sizeof(word));
                for (uint64_t nib = 0; nib < 8; nib++) {
                    const uint32_t q = (word >> (4 * nib)) & 0x0fu;
                    const uint64_t c = g * group_size + w * 8 + nib;
                    out[r * cols + c] = scale * (float)q + bias;
                }
            }
        }
    }
}
