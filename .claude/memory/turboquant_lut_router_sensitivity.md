---
name: turboquant-lut-router-sensitivity
description: TurboQuant(PolarQuant+Lloyd-Max) 스타일 비균등 코드북이 q4g64 가중치 재구성 오차는 실제로 개선하지만(-8~12%, 저장공간 0 추가), MoE top-k 라우팅의 이산적 민감도 때문에 실제 forward-pass에서는 안전하지 않다는 실측 결론
metadata:
  type: project
---

## 배경

`github.com/0xSero/turboquant`(Google TurboQuant의 비공식 재구현, KV 캐시 전용)을
계기로, 그 핵심 알고리즘(PolarQuant 회전 + Lloyd-Max 비균등 스칼라 양자화)이
vdsp의 q4g64 **weight** 양자화(q/k/v/o_proj, dense/shared/routed expert
gate/up/down)에도 적용 가능한지 조사한 세션. TurboQuant 자체는 KV 캐시용이라
직접 이식 대상이 아니었고, 알고리즘만 원용.

## 1단계: 가중치 재구성 오차 실측 (오프라인, 실제 bf16 원본 vs q4g64)

`deepseek-ai/DeepSeek-V2-Lite-Chat`의 실제 bf16 가중치를 HuggingFace safetensors
byte-range fetch로 받아(합성 데이터 아님) 4가지 양자화 방식을 group=64 기준
비교:

| 방식 | int4 rel-L2 | int3 rel-L2 | 저장비용 |
|---|---|---|---|
| A. 현재 프로덕션(균등 affine) | 0.0918(기준) | 0.1969(기준) | 기준 |
| B. 랜덤 Hadamard 회전(RHT)만 | -2.5% | -2.6% | ~0 |
| C. 그룹별 독립 Lloyd-Max 코드북 | **-21.9%** | **-22.1%** | **+140%**(blob 9.1→21.8GB, bob 16GB 예산 초과) |
| **D. 모델 전체 공유 비균등 코드북**(그룹별 scale/bias 그대로, 코드북 하나만 상수로 교체) | **-8.5%** | **-12.3%** | **0 (동일)** |

D를 27레이어 전체(80개 텐서 샘플, layer0-1 16개로만 학습한 코드북을 나머지
25개 레이어에 적용)로 일반화 재확인 — **고정 코드북이 텐서별 로컬 재학습과
사실상 동일한 성능**(차이 0.02%p), 깊이/역할 트렌드 전혀 없음. TurboQuant
원 논문(KV캐시 맥락)은 회전(PolarQuant)이 핵심이었지만, **weight 양자화
맥락에서는 회전의 기여가 미미하고 비균등 코드북(Lloyd-Max) 쪽이
압도적**이었음 — 같은 알고리즘이라도 적용 도메인에 따라 어느 구성요소가
핵심인지 달라질 수 있다는 교훈.

핵심 발견 (D의 코드북, int4): `[0.2138, 1.598, 2.7263, ..., 14.778]` — 균등
간격이 아니라 가운데는 촘촘하고 양끝은 성긴 Lloyd-Max 이론과 정합적인 형태.

## 2단계: 실제 forward-pass 게이트 — 재현 가능한 심각한 버그 1건 + 최종 결론

D를 실제 프로덕션 blob(`deepseek_moe_af.bin`, 9.1GB)에 적용해 8-position
forward-pass 게이트(`run_moe_verify_mode()`, `moe2b_ref_logits.bin` ground
truth)로 검증.

### 재현+수정된 실제 버그: 그룹별 scale/bias 부호 컨벤션이 데이터마다 뒤섞임

이미 4bit로 양자화된 nib 코드에서 재양자화 시도(정보 손실 후이므로 무의미)
→ 실제 bf16 원본에서 재유도해야 함을 확인. 재유도 시 **production 블롭의
실제 scale/bias가 `bias=gmin`(양의 scale)과 `bias=gmax`(음의 scale)를
그룹마다 예측 불가능하게 섞어 쓴다는 사실을 실측으로 발견**(같은 텐서의
row=0은 bias=max, row=5는 bias=min). 재계산 대신 배포된 blob에서 실제
scale/bias를 그대로 읽어 재사용하는 방식으로 수정 — nibble 불일치가
60.7%→0.22%(bf16 원본 소스 간 경계값 반올림 노이즈 수준)로 해소됨. 가중치
레벨 단독 검증(o_proj E=1, switch_mlp E=64-stacked 둘 다)에서는 코드북
방식이 실측대로 정확히 개선됨(-7~9%) 확인.

### 최종 결론: 가중치 재구성 개선이 forward-pass 안전성으로 이어지지 않음

8-position 실제 forward-pass 게이트 결과: **argmax parity baseline 8/8 →
LUT 5/8**, rel-L2 vs truth **baseline 0.0022 → LUT 0.20**(position 0부터
이미 발생, K/V 히스토리 오염 이전). 근본원인을 라우팅 로그 직접 대조로 확정:
**position 0, layer 1부터 이미 top-6 expert 선택이 달라짐**(라우터
score가 근소하게 바뀌며 순위가 뒤바뀜 — 전체 라우팅 로그에서 416줄 차이).

이건 버그가 아니라 **이 프로젝트가 이미 다른 맥락에서 문서화해둔 것과 동일한
메커니즘**: `[[vdsp_general_serving_engine_goal]]`의 MoE-2b가 실측한
"7.76e-06 절대차로 top-6 순위가 뒤집히는 근접동점", MoE-4c가 실측한
"SME2 int8 노이즈로 인한 라우터 플립 1건이 자기회귀 전체 시퀀스를 오염"과
정확히 같은 현상이, 원인만 다르게(SME2 노이즈 대신 양자화 스킴 교체) 재현된
것. **가중치 레벨에서 통계적으로 더 나은 양자화라도, MoE의 이산적 top-k
라우팅이 그 작은 섭동을 증폭시켜 완전히 다른 expert 조합을 선택하게 만들 수
있다.**

**Why**: weight reconstruction error(rel-L2)는 MoE 모델의 실제 forward-pass
정확도를 보장하는 충분조건이 아니다 — 라우터가 있는 아키텍처에서는 반드시
실제 forward-pass 게이트(로딩 게이트와 분리)로 재검증해야 한다는 이
프로젝트의 기존 규율(`[[vdsp_general_serving_engine_goal]]`)이 양자화 스킴
연구에도 그대로 적용됨을 확인.

**How to apply**: 이 LUT 방식을 실제로 채택하려면 MoE-3d/4c가 SME2 노이즈에
대해 이미 구축한 것과 같은 종류의 안전장치(라우터 근접동점 margin 기반
선택적 재검증)가 반드시 필요 — 가중치 재구성 오차 수치만 보고 "더 정확하니
바로 배포 가능"이라고 판단하면 안 됨. 향후 유사한 양자화/근사 스킴을 MoE
라우터가 있는 모델에 적용할 때 이 순서(오프라인 재구성 오차 → 실제
forward-pass 게이트 → 라우팅 로그 직접 대조)를 반드시 거칠 것.

## 재현 관련 참고 (일회성 스크래치 자산, 영속 보장 안 됨)

- 실행 스크립트/blob: `bob:~/vdsp_m4_bench/lut_gate/`(bob 재부팅/청소 시 소실
  가능 — 실제 검증 로직은 이 메모리 본문에 다 기록해둠, 코드 재현은
  `qwen_infer.c`의 `g_moe_lut_enabled`/`moe_lut_apply()`/`QWEN_MOE_LUT_TEST`
  패치를 참고하되, 이 패치는 커밋되지 않았고 연구용으로만 존재)
- 코드북 상수(int4): `[0.2138, 1.598, 2.7263, 3.7191, 4.63, 5.4868, 6.3074,
  7.1074, 7.8975, 8.6968, 9.5149, 10.3663, 11.2735, 12.2693, 13.4003, 14.778]`
- 코드북 상수(int3): `[0.424, 1.5448, 2.3957, 3.1434, 3.8663, 4.618, 5.4646,
  6.5837]`
