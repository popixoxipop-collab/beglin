// rope_llama3_scale.h -- M43: Llama-3.1-style RoPE NTK ("rope_type": "llama3") frequency scaling.
// Shared by qwen_infer.c (the shipped code path) and bench/rope_llama3_check.c (the verification
// harness), so the harness tests the literal compiled function, not a hand-copied lookalike.
//
// Formula verified against the actually-installed transformers==5.12.1's
// transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS["llama3"] on Llama-3.1-8B-Instruct's real
// published rope_scaling values -- max relative error 3.2e-7 (float32-rounding level, not a
// formula bug). attention_factor is always 1.0 for this rope_type ("unused" per HF's own code
// comment) -- no cos/sin post-multiply, only the per-index inv_freq multiplier below.
//
// Returns the RATIO to multiply the existing theta^(-2i/hd) by, not the final inv_freq value --
// keeps the call site a single multiply rather than a restructure. Pure arithmetic (divisions,
// one comparison ladder), no logf -- that's YaRN's correction-dimension formula, not needed here.
#ifndef ROPE_LLAMA3_SCALE_H
#define ROPE_LLAMA3_SCALE_H
#include <math.h>

static inline float rope_llama3_scale(float theta, int hd, float factor, float low_freq_factor,
                                       float high_freq_factor, float orig_max_pos, int i) {
    float inv = powf(theta, -(2.0f*i)/hd);
    float wavelen = 2.0f*(float)M_PI/inv;
    float low_freq_wavelen  = orig_max_pos / low_freq_factor;
    float high_freq_wavelen = orig_max_pos / high_freq_factor;
    if (wavelen > low_freq_wavelen) return 1.0f/factor;
    if (wavelen < high_freq_wavelen) return 1.0f;
    float smooth = (orig_max_pos/wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
    return (1.0f-smooth)/factor + smooth;
}

#endif
