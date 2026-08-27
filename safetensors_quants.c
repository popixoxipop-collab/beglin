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
