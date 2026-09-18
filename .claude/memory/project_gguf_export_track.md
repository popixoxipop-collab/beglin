---
name: project_gguf_export_track
description: GGUF export 트랙(D-export-1~8) 완료 — dense+MoE(OLMoE/GPT-OSS) 전부 원본과 바이트단위 동일크기
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

- **★★★★D-export-7 완료 — 로더 정밀도 정책 트랙, export-only로 스코프 확정
  후 완료**: 사용자가 "엔진 로더 자체를 uniform-int4에서 per-tensor-precision
  보존으로 바꿔"라고 요청 → Plan Mode 진입해 조사한 결과 두 가지 중요한
  실측 근거 발견: (1) `load_gguf_weights()`의 uniform-int4 정책 자체가
  이 프로젝트 자신의 D7/D9/D17(`eval/quantize_int4.py`) 실측으로 이미
  검증된 선택(tied embed int4=ppl 10.6→20.4, lm_head는 int8이 near-
  lossless) — 실수가 아님. (2) vendored SME2 커널 전체가 int4 가중치
  전용으로 고정(`qsi4c32`, 다른 정밀도 변형 벤더링 안 됨) — 실 추론경로
  정밀도를 소스에 맞추면 현재 SME2 가속되는 168개 텐서 대부분이 가속을
  잃음. AskUserQuestion으로 사용자에게 "Export 전용" vs "실 추론경로도
  변경" 확인 → **"Export 전용(추천)" 선택**받아 스코프 확정. 구현:
  `run_export_gguf_mode()`가 이제 각 텐서의 진짜 원본 GGUF 타입을 g_gguf
  (프로세스 전체수명 동안 mmap 유지 확인됨, gguf_close() 호출 0건)에서
  직접 조회해 소스와 동일 타입으로 재인코딩(`switch(src->type)`) —
  D-export-6의 이름/정렬 기반 추측(`is_v`/`in%256`) 완전 대체.
  load_gguf_weights()/GEMM·SME2 디스패치/실 추론경로는 전혀 안 건드림
  (Explore agent로 blast radius 확인: 전체 33개 `->kind==` 비교 중 진짜
  연산 디스패처는 4개뿐, 이 변경은 그 4개 중 어느 것도 안 건드림).
  **★★부수 발견(과잉일반화 자기교정)**: 24개 레이어 전수조사(gguf-py)
  결과 D-export-6의 "attn_v는 전부 Q8_0, ffn_down은 전부 Q6_K"라는
  결론이 **레이어0 하나만 보고 낸 틀린 일반화**였음을 발견 — 실제로는
  attn_v가 12레이어 Q8_0/12레이어 Q5_0, ffn_down이 12레이어 Q6_K/12레이어
  Q4_K로 레이어별로 갈림(llama.cpp의 실제 알려진 target-bpw 히스토그램
  방식). 새 코드는 패턴을 몰라도 텐서별로 직접 물어보니 자동으로 정확함.
  **결과: 491,400,032 bytes — 원본 소스 파일과 바이트 단위로 완전 동일한
  크기**. gguf-py로 두 파일 텐서셋 diff(0 missing/0 extra)+원본 실값과
  수치 diff까지 확인: Q8_0 소스 텐서는 **max_abs_diff=0.0 완전 비트일치**
  (자체 Q8_0 인코더가 ggml 레퍼런스와 이미 동일한 direct-division RTN),
  Q5_0/Q6_K/Q4_K는 작은 기대된 차이(자체 min/max+에러피드백 스케일 탐색
  vs ggml의 make_qkx2_quants류 옵티마이저, 기존 확립된 공개 트레이드오프).
  llama-tokenize ids 동일+llama-simple 정상생성(실 Metal `kernel_mul_mv_
  q6_K_f32` 커널 로드 확인, Q6_K 태깅이 진짜임을 독립 재확인).

- **★★★★★D-export-8 완료 — MoE 아키텍처(OLMoE+GPT-OSS-20B) export**: dense
  전용이던 export를 완전히 별개 파이프라인(`g_gguf_moe`/`MoeAFTensor`/
  `run_gguf_moe_verify_mode()`)으로 확장. Plan Mode 조사 결과 MoE가
  오히려 dense보다 간단함을 발견: (1) 기존 writer 인코더들이 flat-array
  기반이라 3D expert-stacked 텐서에도 코드 변경 없이 그대로 작동(GGUF
  블록양자화 자체가 shape-agnostic), (2) `GgufFile`이 이미 `.tensors`/
  `.n_tensors`를 노출해 아키텍처별 role table도 이름번역 함수도 전혀
  필요없는 완전 제네릭 순회가 가능(MoE 텐서는 자기 진짜 GGUF 이름 그대로
  재출력). `QWEN_MOE_EXPORT_GGUF` 신설, 전체 weight 등록 루프 실행 전에
  삽입해 불필요한 재양자화 연산 회피. GPT-OSS의 native MXFP4 expert는
  진짜 raw byte passthrough(writer 인코더 자체가 불필요 — 그대로 복사가
  이미 최선).
  **실 검증(OLMoE-1B-7B, 64 experts)**: `195 tensors (81 F32, 113 Q4_0,
  1 Q6_K)`, **3,928,037,440 bytes — 원본과 바이트단위 완전동일**,
  `ffn_gate_exps.weight` shape `[2048,1024,64]`로 3D expert-stacking
  최초 실증(버그없이 한번에 성공), gguf-py 텐서셋diff 0/0, Q4_0텐서는
  기대된 작은 오차(자체 에러피드백 vs ggml RTN, D-export-2와 동일
  트레이드오프), llama-tokenize ids 원본과 재비교해 완전동일, llama-simple
  "Paris" 정답 생성 확인.
  **실 검증(GPT-OSS-20B, 32 experts, native MXFP4)**: `459 tensors
  (289 F32, 98 Q8_0, 72 MXFP4)`. 1차 시도 disk 100% full로 FATAL(이
  세션이 스스로 쌓아둔 ~17GB 스크래치파일 원인, 정리 후 재시도 성공).
  **12,109,566,624 bytes — 원본과 바이트단위 완전동일**(이 세션에서만
  3번째 동일결과). gguf-py로 **샘플 텐서 전부 max_abs_diff=0.0 완전
  비트일치**(MXFP4 expert 포함 — gguf-py 자체 독립 MXFP4 디코더로 재확인),
  이 체크포인트 실레시피가 F32/Q8_0/MXFP4만 써서(K-quant 없음) Q8_0의
  기존 발견(비트일치)이 텐서 전부에 적용됨. llama-tokenize 성공.
  llama-simple 전체생성은 **이 머신의 실 GPU 메모리 한계로 실패**(원본
  파일도 동일하게 크래시 재현 확인 — export 결함 아님, 정직히 보고).

**Why**: Mac 기반 로컬 LLM 파인튜너 대상 배포용 산출물이라는 새 전략
목표. 이전 트랙(Phase 6 토크나이저)의 ROI 논의 이후 사용자가 명시적으로
승인한 다음 단계.

**How to apply**: 이 트랙(D-export-1~8)은 완료 상태 — dense 균일정밀도
writer+K-quant+F32티어+실레시피매칭+source-type-aware export, 그리고
MoE(OLMoE+GPT-OSS) export까지 전부 실검증됨. dense/MoE 양쪽 export
결과물이 전부 원본과 바이트단위로 동일한 크기, GPT-OSS는 텐서 전부
비트일치까지 확인됨. **"로더를 바꿔달라"는 요청은 export-only로
스코프를 좁혀 완료됨** — 실 추론경로(load_gguf_weights/MoE 등록루프,
SME2 디스패치)는 여전히 uniform-int4 그대로, 사용자가 명시적 확인 후
선택한 스코프임. 다음에 더 갈 곳이 있다면 (a) 실 추론경로 정밀도
변경(SME2 가속 상실 감수, 새 근거 필요) (b) Qwen3-MoE export(bob에 실
체크포인트 없어 오라클 검증 불가, 파일 생기면 코드 변경 없이 바로
작동할 것) (c) precision-search 기반 export — 전부 명시적으로 스코프아웃
상태, 아직 요청 없음. 상세 진행 로그는
[[../history/2026-09-18_gguf-export-q4k-encoder.md]] 참고.
