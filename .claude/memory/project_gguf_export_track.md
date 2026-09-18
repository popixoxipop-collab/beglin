---
name: project_gguf_export_track
description: GGUF export 트랙(D-export-1~6) 완료 — 실 레시피 정밀도 매칭, 원본 대비 2.85% 이내
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

- **★★★D-export-6 완료 — 실 레시피 정밀도 매칭**: "남은 Q4_0 텐서에 Q5_K/
  Q6_K 추가"라는 사용자 요청을 실행하기 전 실 소스파일을 gguf-py로 먼저
  조회 → **Q5_K/Q6_K 둘 다 Q4_K와 동일한 QK_K=256 정렬 제약**이라 D=896인
  q/k/v/o/gate/up 전부에 애초에 적용 불가능함을 사전에 확인(코드로 해결
  불가능한 구조적 사실). `Q5_K`는 이 모델에서 실제 적용 대상이 전혀
  없음(스킵), `Q6_K`는 `ffn_down`(IM=4864, 256 배수)에만 적용되고 실
  소스파일도 거기서 Q6_K를 씀(D-export-4가 Q4_K로 잘못 골랐던 것도 이때
  정정). 실 레이어0 전수조사 결과: attn_q/k/output+ffn_gate/up→Q5_0,
  attn_v→**Q8_0**(실 llama.cpp "M"레시피의 알려진 휴리스틱), ffn_down→Q6_K.
  Q6_K 인코더 신규구현+dual-oracle 검증(own dequant+gguf-py 완전일치,
  rel=8.9% vs Q4_K의 37.8%로 fidelity 개선 확인) 후 나머지는 이미 만들어둔
  Q5_0/Q8_0으로 이름기반 라우팅. **결과: 505,399,136 bytes, 원본
  (491,400,032 bytes) 대비 2.85% 이내** — 텐서타입 구성이 실 레시피와
  정확히 일치(Q4_0 잔존 0개). llama-tokenize ids 동일+llama-simple 정상
  생성 재확인(주관적으로 이전 라운드보다 더 일관된 출력, 단일샘플이라
  과대해석 안 함). **★중요 공개 caveat**: 이 엔진 자체 로더가 lm_head
  제외 모든 양자화 텐서를 이미 로드시점에 K_Q4G64(int4)로 다운캐스트하므로,
  export시 더 높은 비트폭으로 재인코딩해도 원본의 진짜 정밀도를 복구하는
  게 아니라 "두 번째 손실 양자화를 추가로 얹지 않는다"는 좁은 의미의
  이득만 진짜임 — 컨테이너 타입 일치≠수치 충실도 일치, 과장하지 않음.

**Why**: Mac 기반 로컬 LLM 파인튜너 대상 배포용 산출물이라는 새 전략
목표. 이전 트랙(Phase 6 토크나이저)의 ROI 논의 이후 사용자가 명시적으로
승인한 다음 단계.

**How to apply**: 이 트랙(D-export-1~6)은 사실상 완료 상태 — 균일정밀도
writer+K-quant+F32티어+실레시피 매칭까지 전부 실검증됨. 다음에 더 갈
곳이 있다면 (a) 이 엔진 자체 로더의 uniform-int4 정책을 real
per-tensor-precision 보존으로 바꾸는 아키텍처 변경(위 caveat의 진짜
해결책, 훨씬 큰 스코프) (b) MoE 아키텍처 export (c) precision-search
기반 export — 전부 명시적으로 스코프아웃 상태, 아직 요청 없음. 상세
진행 로그는 [[../history/2026-09-18_gguf-export-q4k-encoder.md]] 참고.
