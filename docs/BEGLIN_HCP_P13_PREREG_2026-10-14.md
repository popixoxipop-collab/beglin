# Beglin HCP P13 — Native P9 packed-qNg64 byte-exact comparison, isolated preregistration

2026-10-14 KST | No live promotion.

Parent P12 commit: `6ae561d1b4bb3cb83d4ba4e43c6cea9b257f94e3`
Research branch: `research/hcp-p13-native-bf16-byte-parity-20261014`.

## Hypothesis and exact references

P12 measured local BF16 5/6-bit `gguf_quantize_qNg64` group64 weight projection errors, but did not prove its packed codes and scales were byte-identical to actual P9 MLX GPU inference cache. P13 instruments the exact P5 C CB128 / P9 MLX KV128 isolated native inference binary, using macOS CommonCrypto SHA-256 only when a research environment flag is set. The C code may not alter quantized weights, MLX binding calls, protocol or generation.

The four frozen reference packed/scales SHA-256 pairs and exact target shape/bitwidth are sealed in `experiments/hcp_p13/P13_REFERENCE_CODES.json` and sourced from P12 immutable result `p12_local_BF16_oracle.json` SHA-256 `104d8112a737515f7bd52ae187735940e5948afebea78601cb61a73e58484de4`. Freeze before native experiment.

- Two tensors: L3 shared_up_proj (2816×2048), L26 shared_down_proj (2048×2816).
- Both n=5 and n=6, E=1, group=64, real original DeepSeek-V2-Lite safetensors index SHA pinned.
- Native trace must show FOUR distinct matching packed and scale hashes, from the same pinned executable. A trace absent or different SHA is FAIL; do not guess equivalence.
- Real A/B/A/B native REBIND_SET terminal ACK and generated token sequence hashes compared with P6 frozen DeepSeek source.
- Cache-enabled research binary ON only within owned XOX scratch; baseline operation untouched; no timing speedup claim with SHA logging enabled.
- Original full BF16 logits / model quality noninferiority remain unmeasured.

## Independent controller / journal scope

The P12 prestart exclusive lock and durable terminal journal are already verified. P13 should additionally show that another controller process cannot take the same lease while one real research native worker holds it, and that an altered previous terminal ACK/INTENT/VERIFIED cannot trick a new controller into replaying the original. These are not full multi-worker deployment or exactly-once crash recovery, which remain BLOCKED.

P13 operational acceptance requires original stage/route SHA, original serving PIDs, source commit, 20GB SSD reserve, source/target proof and no main merge, no release activation.
