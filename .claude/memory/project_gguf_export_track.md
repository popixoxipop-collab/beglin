---
name: project_gguf_export_track
description: GGUF export 트랙(D-export-1~5) 현재 상태 — Q4_0/Q8_0/Q4_K/Q5_0 인코더 완료, 897MB→446MB(50.3%↓)
metadata:
  type: project
---

**목표**: 사용자가 2026-09-18에 제시한 새 전략 방향 — 이 엔진의 양자화
실험 인프라를 활용해, Mac 기반 파인튜너 등에게 배포 가능한 real GGUF
체크포인트를 생성하는 export 기능. `~/.claude/plans/serene-finding-ullman.md`
승인된 계획서 기준 Phase 1-3(균일 정밀도 writer) + K-quant follow-up +
F32-tier follow-up 완료.

**현재 상태 (2026-09-18)**:
- Phase 1-3: `gguf_write.c/.h`(writer), `gguf_write_quants.c/.h`(Q4_0/Q8_0/
  Q4_K/Q5_0 인코더), `run_export_gguf_mode()`(`qwen_infer.c`) — 실
  Qwen2.5-0.5B 체크포인트 export → `llama-tokenize`/`llama-simple`(bob, 실
  llama.cpp, 엔진 의존성 0)로 매 단계 검증 완료. 커밋 `6474932`/`76e39e8`/
  `6daf7a6`/`e9baa32`/`cf45a55`.
- **★★D-export-4 발견**: Q4_0(18B/32elem)과 Q4_K(144B/256elem)는 정확히
  동일한 4.5 bits/element — 타입 전환은 파일 크기를 절대 줄이지 않는다(직접
  계산+실측 양쪽 확인). K-quant의 진짜 가치는 같은 비트폭에서의 정확도
  (비대칭 per-sub-block affine)이지 압축률이 아님. 이전 세션(D-export-1/3)이
  남긴 "Q4_K가 더 압축적" 주장은 틀렸고, RESULTS.md에서 정정 완료.
- 실 Qwen2.5-0.5B(`D=896`, `IM=4864`)에서 `D`가 256의 배수가 아니라(896/256
  =3.5) q/k/v/o/gate/up은 전부 Q4_K 승격 대상이 아니고, `IM`만 256의 배수라
  `ffn_down`(레이어당 1개, 24개)만 Q4_K로 승격됨.
- **★★★D-export-5 완료 — 진짜 레버였음 확인**: `embed_tokens.weight`가
  K_F32 role 정책 때문에 897MB 파일의 60%+(~545MB)를 차지한다는 D-export-4의
  진단대로, 실 소스 Q4_K_M 파일 자체를 gguf-py로 직접 조회해 `token_embd.
  weight`가 실제로 Q5_0(Q4_0도 Q8_0도 아님)임을 먼저 확인한 뒤, 동일하게
  Q5_0 인코더를 구현+dual-oracle 검증(own dequant + gguf-py 완전일치)해
  embed_tokens만 승격. norms/biases는 실 소스파일도 F32로 유지함을 직접
  확인 후 그대로 둠(전체 900MB급 파일 대비 <1MB라 사이즈 이득 없고 정확도
  리스크만 있음). **결과: 896,693,088 → 445,747,040 bytes(50.3%↓), 원본
  소스(491.4MB)보다도 작아짐**. llama-tokenize ids 동일+llama-simple 정상
  생성(coherent, load time 481ms→102ms) 재확인 완료.

**Why**: Mac 기반 로컬 LLM 파인튜너 대상 배포용 산출물이라는 새 전략
목표. 이전 트랙(Phase 6 토크나이저)의 ROI 논의 이후 사용자가 명시적으로
승인한 다음 단계.

**How to apply**: "K-quant을 추가하면 파일이 작아진다"는 가정은 이미
실증으로 반증됨(D-export-4) — 파일 크기를 줄이는 진짜 레버는 K_F32
티어(주로 embed_tokens 단일 텐서) 양자화였고, 이미 완료+검증됨
(D-export-5). 다음에 파일 크기를 더 줄이고 싶다면 `Q4_0`이 아직 남아있는
896-row 텐서(q/k/v/o/gate/up, attn_output도 Q5_0)들에 `Q5_K`/`Q6_K` 인코더를
추가하는 게 다음 후보 — 아직 구현 안 됨, 스코프아웃 상태. 상세 진행 로그는
[[../history/2026-09-18_gguf-export-q4k-encoder.md]] 참고.
