# BEGLIN: bounded GPU-resident qNg64 multi-projection and routed MoE (2026-10-19)

**Decision: isolated research PASS; production promotion NOT_READY.** Frozen real DeepSeek-V2-Lite BF16 source, original native-C++/Metal n7 quantized reference and an independently compiled GPU-resident GEMV were compared on EOE Apple Silicon. Original XOX serving and runtime were not modified.

The new reproducible generic source and Apple Silicon synthetic regression are in [`research/n7_multi_projection_resident_v1/`](../research/n7_multi_projection_resident_v1/README.md). This is an independent follow-on to the [single fused MLA GPU attention research](BEGLIN_QNG64_ABSORBED_MLA_NATIVE_METAL_20261019.md) and the [two-token latent KV experiment](BEGLIN_NATIVE_CPP_MLA_LATENT_KV_20261019.md). **Do not combine their timing claims into a single end-to-end TPS figure.**

## Source-attested real hardware observations

| Scope | Actual result |
| --- | --- |
| Physical synthetic generic GEMV | 6 matrices, packed+scale total **15,052,800 bytes**, independent BEGLIN C output max relative L2 **1.29304×10⁻⁷** |
| Real original DeepSeek **decoder layer 0**, 7 distinct n7 linear roles | Original 5- and 6-token BF16 decoder hidden outputs **exactly equal** to original C++ Metal; **75,939,840 bytes** GPU packed/scales |
| Layer 0, 96 MiB GPU budget | 7 first registrations, 35 reuse hits after repeated computation, zero evictions |
| Layer 0, 32 MiB bounded LRU | 13 evictions; same original source BF16 decoder output and bitwise replay |
| Real original DeepSeek **decoder layer 1**, actual top-6 routed experts | 22 distinct experts selected for 5 input positions; 7 fixed + 3×22 = **73 original n7 roles**, **207,544,320 bytes** packed/scales |
| Layer 1, 96 MiB / 12 handle GPU LRU (two identical original model forwards) | **134 evictions**, zero reuse hits across 146 actual GPU linear calls; final BF16 output identical to independent source |
| Layer 1, 256 MiB / 96 handle GPU LRU | First call registered **73 roles**, second call yielded **73 reuse hits**, **zero re-registrations/evictions**; output again identical to independently computed original source BF16 |
| Layer 1, observed second-call full layer wall time | ~0.1573 s (96 MiB) vs ~0.1297 s (256 MiB); per-layer research-only observation, **not** production throughput |

Both real MoE configurations passed independently verified, SHA-pinned original-source numeric tests. Neither stored hidden arrays, real model tensors, prompt text, or the selected expert ID list in the repository.

## Evidence and scope

- Layer0 7-projection real-source numerical receipt SHA-256: `393f2fdc205321ac411659b984b5fb12a8381d7604985306a9cd59a59488028f`
- Layer0 full-source forward / low-budget eviction receipt: `8f03d8a33fb617418345f95e30691fb22bbbba64e6068a89f7792cbbb923701b`
- Layer1 original selected-expert 96 MiB LRU: `a4eb68cacf9b9c655e7a9ea066ca8ced748127f010ccd765c55f82fc772801ef`; independent audit `5ae42c401dc8c6a045424d2fab868d12a0a5df6cf1b2a70dd815d0720455e5da`
- Layer1 same actual source 256 MiB fully retained: `0354c4e2281d4bd54a4fecb8e882c84a8134d4c9436185ecc5bbff5d1d551cab`; independent audit `d407139dcc6e9e087fb0e2e84374ebcef263ccd195e866b1bb8424b7eb3e44d1`
- Original source BF16 safetensors index SHA-256: `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`

The original actual source research receipts stay in the authorized private EOE workspace. The portable package contains only **synthetic source-shaped data**, source code, test driver and instructions. The source is from the user's own research branch; no original checkpoint, hidden/output tensors, source prompts, API keys, pretrained data or platform binaries are committed.

**Important boundary:** The correct GPU LRU implementation is deliberately **dual-owned compact CPU+GPU** because we observed corruption in the earlier GPU registration implementation when a host source buffer was destroyed after registration. More memory optimization must first prove independent GPU backing/lifetime; do not remove host backing on assumption. The proof measures GPU active allocator bytes for registered compact matrices, not total model process memory reduction.

**Next practical gates:** real multi-layer inference with resident n7 roles + fused latent-KV attention using a shared authoritative memory budget, prospective prompt/quality holdouts, longer sampled token traces, exact peak memory/TTFT/TPOT and guard-verified restart/resume. Leave PR Draft and automatic live promotion disabled until those tests pass.
