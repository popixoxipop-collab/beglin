// JE01 isolated 384-pos MLX/M1 smoke probe (no model weights, no inference).
// Proves an Apple Metal device is usable, B=1 accepted, B>1 rejected,
// and host MLA KV setup with actual DeepSeek dimensions succeeds.
// It does NOT prove full-GPU cache materialization or pretrained-logit parity.
#include "mlx_moe.h"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <sys/resource.h>

static long long rss_bytes(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return -1;
#if defined(__APPLE__)
    return (long long)usage.ru_maxrss;  // Apple reports bytes
#else
    return (long long)usage.ru_maxrss * 1024;
#endif
}

int main() {
    const long long planned_kv = 251658240LL;  // 384*32*16*(192+128)*4, B=1
    if (!mlx_gpu_available()) {
        std::puts("JE01_NATIVE_METAL_NOT_AVAILABLE");
        return 2;
    }
    if (mlx_gpu_set_batch(1) != 1) {
        std::puts("JE01_BATCH1_NOT_ACCEPTED");
        return 3;
    }
    if (mlx_gpu_set_batch(2) != 0 || mlx_gpu_set_batch(64) != 0 ||
        mlx_gpu_set_batch(0) != 0) {
        std::puts("JE01_MULTISLOT_FAIL_CLOSED_VIOLATION");
        return 4;
    }
    double yarn_freqs[32];
    for (int i=0; i<32; ++i)
        yarn_freqs[i] = std::pow(10000.0, -(2.0*i)/64.0);
    const long long before = rss_bytes();
    // DeepSeek V2 Lite: heads=16, QHD=192 (128 nope+64 rope),
    // VHD=128, kv_lora_rank=512. Host full-window MLA config only.
    if (mlx_gpu_mla_config(16, 192, 128, 64, 128, 512,
                           0.707, 1.0/std::sqrt(192.0),
                           yarn_freqs, 1e-6) != 1) {
        std::puts("JE01_MLA384_CONFIG_FAIL");
        return 5;
    }
    const long long after = rss_bytes();
    const long long rss_delta = (before >= 0 && after >= 0)
        ? after - before : -1;
    std::printf("{\"schema\":\"beglin-je01-metal384-host-probe-v1\","
                "\"status\":\"PASS_NO_MODEL\","
                "\"mlx_device_available\":true,"
                "\"single_slot_accepted\":true,"
                "\"two_or_more_slots_rejected\":true,"
                "\"mla_host_configured\":true,"
                "\"theoretical_kv_bytes_per_single_slot\":%lld,"
                "\"measured_process_peak_rss_delta_bytes\":%lld,"
                "\"pretrained_model_forward\":\"NOT_RUN\","
                "\"metal_full_fused_kv_materialized\":false,"
                "\"model_weights_loaded\":false,"
                "\"production_modified\":false}\n",
                planned_kv, rss_delta);
    return 0;
}
