// hf_config_verify.c -- dumps every field the safetensors dense loader will read from a real
// config.json, in a form diffable against an independent Python `json.load()` reference. Same
// "independent implementation as ground truth" discipline gguf-py served for the GGUF gates.
#include <stdio.h>
#include "hf_config.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <config.json>\n", argv[0]); return 1; }
    HfConfig *c = hf_config_open(argv[1]);

    int64_t i;
    double f;
    int b;
    const char *s;

    if (hf_config_get_i64(c, "hidden_size", &i)) printf("hidden_size=%lld\n", (long long)i);
    if (hf_config_get_i64(c, "num_hidden_layers", &i)) printf("num_hidden_layers=%lld\n", (long long)i);
    if (hf_config_get_i64(c, "intermediate_size", &i)) printf("intermediate_size=%lld\n", (long long)i);
    if (hf_config_get_i64(c, "num_attention_heads", &i)) printf("num_attention_heads=%lld\n", (long long)i);
    if (hf_config_get_i64(c, "num_key_value_heads", &i)) printf("num_key_value_heads=%lld\n", (long long)i);
    if (hf_config_get_f64(c, "rms_norm_eps", &f)) printf("rms_norm_eps=%.9g\n", f);
    if (hf_config_get_f64(c, "rope_theta", &f)) printf("rope_theta=%.9g\n", f);
    if (hf_config_get_i64(c, "vocab_size", &i)) printf("vocab_size=%lld\n", (long long)i);
    if (hf_config_get_bool(c, "tie_word_embeddings", &b)) printf("tie_word_embeddings=%s\n", b ? "true" : "false");
    if (hf_config_get_str(c, "model_type", &s)) printf("model_type=%s\n", s);
    if (hf_config_get_i64(c, "max_position_embeddings", &i)) printf("max_position_embeddings=%lld\n", (long long)i);
    printf("has_rope_scaling=%s\n", hf_config_has_key(c, "rope_scaling") ? "true" : "false");

    hf_config_close(c);
    return 0;
}
