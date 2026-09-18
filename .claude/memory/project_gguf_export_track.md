---
name: project_gguf_export_track
description: GGUF export 트랙(D-export-1~4) 현재 상태 — Q4_0/Q8_0/Q4_K 인코더 완료, 파일크기 진짜 원인 규명
metadata:
  type: project
---

**목표**: 사용자가 2026-09-18에 제시한 새 전략 방향 — 이 엔진의 양자화
실험 인프라를 활용해, Mac 기반 파인튜너 등에게 배포 가능한 real GGUF
체크포인트를 생성하는 export 기능. `~/.claude/plans/serene-finding-ullman.md`
승인된 계획서 기준 Phase 1-3(균일 정밀도 writer) + K-quant follow-up
완료.

**현재 상태 (2026-09-18)**:
- Phase 1-3: `gguf_write.c/.h`(writer), `gguf_write_quants.c/.h`(Q4_0/Q8_0/
  Q4_K 인코더), `run_export_gguf_mode()`(`qwen_infer.c`) — 실 Qwen2.5-0.5B
  체크포인트 export → `llama-tokenize`/`llama-simple`(bob, 실 llama.cpp,
  엔진 의존성 0)로 검증 완료. 커밋 `6474932`/`76e39e8`/`6daf7a6`/`e9baa32`.
- **★★중요 발견**: Q4_0(18B/32elem)과 Q4_K(144B/256elem)는 정확히 동일한
  4.5 bits/element — 타입 전환은 파일 크기를 절대 줄이지 않는다(직접
  계산+실측 양쪽 확인). K-quant의 진짜 가치는 같은 비트폭에서의 정확도
  (비대칭 per-sub-block affine)이지 압축률이 아님. 이전 세션(D-export-1/3)이
  남긴 "Q4_K가 더 압축적" 주장은 틀렸고, RESULTS.md에서 정정 완료.
- 실 Qwen2.5-0.5B(`D=896`, `IM=4864`)에서 `D`가 256의 배수가 아니라(896/256
  =3.5) q/k/v/o/gate/up은 전부 Q4_K 승격 대상이 아니고, `IM`만 256의 배수라
  `ffn_down`(레이어당 1개, 24개)만 Q4_K로 승격됨.
- **파일크기 격차의 진짜 원인**: 이 엔진의 K_F32 role 정책이 embed_tokens/
  norms/biases를 F32로 유지 — embed_tokens 단독으로 ~545MB(전체 897MB
  파일의 60%+). 원본 Q4_K_M 소스(491.4MB 실측, 이전 세션의 "~397MB"는
  부정확한 추정치였음)와의 격차를 진짜로 좁히려면 K_F32 티어 양자화가
  필요 — 아직 스코프 아웃 상태, 사용자 판단 대기.

**Why**: Mac 기반 로컬 LLM 파인튜너 대상 배포용 산출물이라는 새 전략
목표. 이전 트랙(Phase 6 토크나이저)의 ROI 논의 이후 사용자가 명시적으로
승인한 다음 단계.

**How to apply**: 이 트랙을 이어갈 때 "K-quant을 추가하면 파일이 작아진다"는
가정을 반복하지 말 것 — 이미 실증으로 반증됨. 파일 크기를 실제로 줄이려면
Q5_K/Q6_K가 아니라 K_F32 티어(embed/norm/bias) 양자화가 필요하다는 걸
전제로 다음 대화를 시작할 것. 상세 진행 로그는
[[../history/2026-09-18_gguf-export-q4k-encoder.md]] 참고.
