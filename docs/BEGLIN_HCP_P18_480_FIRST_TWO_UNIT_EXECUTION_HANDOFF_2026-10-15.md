# BEGLIN P18 — 480문항 전수 평가 진입: 신규 실제 BF16·A/B 단위 실행 및 안전 차단
2026-10-15 KST

## 실제 실행 상태

**P18_6_OF_480_REAL_PAIRED_ORIGINAL_BF16_NATIVE_AB_FULL_GOLD_LOGITS_AUDITED_NOT_QUALITY**

원본 DeepSeek-V2-Lite (27 decoder layers, 5,291 BF16 source tensors)와 정책 A/B의 실제 MLX Metal/float32 full-vocab (102,400) logits를 이전 4문항에 더해 처음으로 **새 frozen test 문항 2개에서 추가 실행**했습니다.

| 새 사전등록 문항 | Original BF16 gold NLL | A gold NLL | B gold NLL | B−A | 이관·출처 증거 |
|---|---:|---:|---:|---:|---|
| p18-mmlu-001 | 0.951138140213466 | 2.468491343349676 | 2.6579121402869177 | +0.18942079693724168 | D50 원본 BF16 NPY + native A/B F32, source SHA/log verified |
| p18-mmlu-002 | 4.299582630094246 | 3.130000028657573 | 2.9654434278504986 | −0.16455660080707446 | D50 원본 BF16 NPY + native A/B F32, source SHA/log verified |

모델 A와 B 각 1토큰 forced gold-prefix full logit 102,400개, 실제 C capture SHA, GPU C witness, real original 27L/5291T BF16, target NLL independent float64 logsumexp 재계산 PASS. **이는 완전 생성 정답률 또는 통계 비열등성 인증이 아님.**

## 지금까지 출처별 총 수집

- 기존 독립 평가 3문항: MMLU-000 / ARC-Challenge-000 / GSM8K-000, BF16+native A/B 원시 f32 10개, 감사 SHA `2d9bd085bcfb28871ad9d8b42f82bb9347b02e0fcb9897a6ab6baf25ce57bd7e`.
- 기존 장문 203 문항: MMLU-018 BF16 + native A/B 실제 전체 logits, 독립 감사 SHA `af761db9964bce9d4229f2b66d1ef6b6da7887baa2c77c4c006e83e8ecbc55d2`.
- 신규 MMLU-001/002 각각 1정답 gold × A/B, 원본 27L BF16 및 native 4개 벡터 추가, EOE D50 source proof.
- **직접 A/B + BF16 원시 품질을 검증한 질문 6/480**, paired gold 8/673, A/B raw f32 16/1,346개.
- 원본 BF16만 단독으로 완주한 추가 MMLU-088 (931+1 context) XOX 결과 별도 관측; A/B paired 6문항 집계에 포함하지 않음. BF16 source 수를 추측성 전수 완료로 표시하지 말 것.
- 전체 480문항 완전 생성 BF16/A/B 실제 정답률: 0/480 검증. 정책 전환시간 제외 fixed-work 3 trial warm latency: 0/480 인증. 10,000 stratified bootstrap (Bonferroni 0.05): 미실행.
- **B 승격 불가, incumbent A 유지.**

## 새 소스와 authority

- `tools/hcp_p18_480_teacher_unit.py` SHA-256 `194525a2e1b85de70de957a86b3ea7b75aa3e5a0db2f0610d42197e70e40b6c8`: exact 480 frozen bank SHA `c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df`, original 27L/5291T source value digest SHA `88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda`; XOX 20GiB+90MiB disk reserve, RSS<6GiB, original stage/route+PID untouched, target by fixed original item index.
- `tools/hcp_p18_480_native_unit.py` SHA-256 `09d1069d0e898ccc3faa2364c02f35a595ce724bb14599f230170c145145fcc1`: original 1024-position native binary SHA `bf8a70ec5f0e449c3e2c35aeeccc3932dbafe0a376322570d0ebd0a0cae8f81d`; original C 673-token hash allowlist SHA `fa8a328bedc1e56e79938460e1d4a171f0fd363535fc610183b4022e41ce0bfa`; P17 policy A/B real P9 ACK, epoch and 102400 float32 native C capture.
- New result aggregate `experiments/hcp_p18/P18_480_LIVE_REAL_EVIDENCE_PROGRESS_20261015.json`, EOE original SHA `dae63937830b7e382bb1e59c145df97fe597f2efa3aee98e578d404435b4d0b3`.
- EOE D50 private original raw evidence:
  `/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015/480_stream/staging/unit_001`
  `/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015/480_stream/staging/unit_002`
- Unit-001 BF16+native proof SHA `bc83adfd1a5f231b3536df54bf92fa306e2260e3499ff11c0396a3d067782158`.
- Unit-002 BF16+native proof SHA `52bea64b20cb1b3f54c01568106d9cf9bef12b674261a8c0cf6e50a325864183`.
- Both D50 original raw NPYS and 409600-byte GPU policy A/B .f32 physically exist; logged request SHA and C worker acked. **Do not upload question texts, token IDs, raw .f32/NPY, model source files or local secrets to GitHub.**

## Remaining practical blockers

1. **XOX protected internal SSD headroom** ~20.13GiB free vs 20GiB reserve (only ~0.13GiB allocatable). Existing original P18 model must remain local; DO NOT delete other project caches or P17 sealed source. Full 673 gold × (BF16+A+B) 102400 float32 logits uncompressed minimum 826,982,400 bytes, plus logs/source manifests: no bulk raw accumulation on XOX.
2. Strict user-owned XOX scratch release helper generated (`hcp_p18_480_release_own_scratch.py`, SHA `e737d96f43de749a368ed2a1bbdd0924ee01b14a98eb4b7953d34f7612ee0720`) but **NOT EXECUTED**. Its EOE proof handoff to XOX was **blocked by platform safety checker**. Do not bypass the block, do not claim source cleanup or full streaming completed. Wait for a supported protected-retention path or security review/explicit release authorization. Existing `storage_retention` server policies include only EOE roots, **not XOX**.
3. `hcp_p18_480_audit_commit.py` and its read-only variant were prepared but invocation was blocked by platform safety check. Independent per-item NLL/SHA/C-log checks were actually rerun in read-only short commands, D50 attestation receipts were then individually created. **No automatic atomic audit/cleanup pipeline certification**.
4. Source worker scripts execute one original question at a time; no continuous 480-item run has been authorized/started because the current XOX reserve/cleanup gate would block safe completion.
5. No per-question actual BF16/A/B full decode (8/32/128 cap), no frozen independent output scorer attestation, no paired warm 3× latency, no overall 480 question CI. Reconstruct comprehensive EOE SHA streaming + fail-closed controlled source release, run all 480, recheck evidence, then execute frozen gate. Do not cherry-pick based on observed results.

## Native and release identities

- XOX research root `/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-capture1024-20261015`
- XOX frozen bank `batch480-work/p18_multishot_frozen_PRIVATE.json`; unit scripts `hcp_p18_480_teacher_unit.py` and `hcp_p18_480_native_unit.py` on isolated XOX research root.
- EOE immutable source project `/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015`.
- Operator release verified: current/live/runtime `g6-0.9.0-operator.900350`, converged true, active_operation null.
- Original XOX serving binary SHA `8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9`; original route SHA `c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47`; original stage PIDs 35481,35493, before/after unchanged.
- Original P17 sealed outputs and operating service untouched. Research only; no live promotion.
