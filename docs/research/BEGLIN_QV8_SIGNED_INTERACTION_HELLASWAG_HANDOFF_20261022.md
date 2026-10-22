# BEGLIN Qv8 — 전체 모델 정답 손실 방향의 교차 텐서 signed 민감도 연구

2026-10-22 KST · 연구 브랜치 research/qv8-joint-loss-signed-source-20261022 · 운영 릴리스와 main 변경 없음.

## A. 선행 부정적 근거 및 연구 목적

Qv5·Qv6에서는 각 투영별 경험적 Fisher 대리지표 E[(g_i·δ_i)^2]로 승격할 8비트 텐서를 선택했다. ARC 과학 문제의 OpenBookQA 개발 평가에서는 일부 개선을 관찰했지만 TinyLlama Llama 계열에서는 재현되지 않았다. 높은 소스 민감도 순위 Spearman 자체로는 실제 답변 NLL이 개선되지 않았다.

Qv8은 모든 196개 투영의 실제 오차 방향을 동시에 고려한다. 먼저 원래의 **수정하지 않은 BF16 Qwen**의 정답 A/B/C/D 다음 토큰에 관한 그래디언트에서 **부호를 보존한** 스칼라 a_si=g_si·(Q4(W_i)-W_i), b_si=g_si·(n8(W_i)-W_i)를 캡처한다. 업그레이드 x_i=1인 텐서는 n8, 아니면 Q4다.

수식: r_s(x) = Σ_i [a_si + x_i(b_si−a_si)].
소스 목적함수: J(x) = E_s[r_s(x)^2].

제곱 전 부호를 합산하므로 텐서 간 **선형 1차 오차 상쇄·증폭**을 계산한다. 그러나 이는 고차 Hessian, 실제 양자화 모델의 자유 생성 손실, 진짜 정확도와 같지 않다. 소스 상쇄를 과적합했을 가능성 때문에 검증 데이터는 원래 모델 실행 전 별도 SHA 고정했다. 기존 금지된 단일 텐서 최종 모델 출력 개입은 실행하지 않았다.

- Qwen2.5-1.5B-Instruct 공식 BF16 revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306.
- 보정 원본: ARC-Challenge train 고정 첫 96개, Qv6 이전 SHA 토큰과 동일.
- 새 평가: Rowan/hellaswag revision 218ec52e09a7e7462a5400043bb9a69a41d06b76, validation SHA 정렬 48개. 과학문제 이외의 서술 상황 선택 과제이지만 다른 모든 추론 과제로 일반화된다고 말할 수 없다.
- 정확히 네 개 A/B/C/D 원본 Qwen ChatML next-token 선택지를 모두 한 토큰으로 고정, 문항별 원본 길이 208 토큰 이하.
- 소스 96×196 signed gradient private receipt SHA: e693fdf6798690c86ad2b53bb4a8e90cc208994956944f519841968ce680ed83.

## B. 소스 교차 텐서 목적함수 결과 — 실제 모델 품질이 아님

| 원본 BF16 소스96에서 평가한 정책 | 1차 손실변화 제곱의 경험적 평균 |
|---|---:|
| 전체 Q4 | 4752.271319 |
| 기존 Qv5 Fisher24 | 3870.675600 |
| 기존 Qv6 Fisher64 | 2650.881113 |
| 기존 Qv6 Fisher128 | 2579.764284 |
| **Qv8 signed96 joint** | **87.256952** |

엄격한 실측 BEGLIN CPU 커널 비용 구속 28개 n8/168개 Q4, 가중치 메모리 765,493,248B, Q4 기준 순차 디코드 예측 86.702ms, 측정 stage DAG 합계 49.969ms. 실측 전체 모델 토큰 지연이 아니라 커널 합산 대리값이다.

선택 알고리즘은 HiGHS 전역 QP 최적 증명이 아니라, 여러 소스 출발점에서 **엄격한 제약을 지키는 1-for-1 교환 국소 탐색**이다. 독립 전수 검사에서는 가능한 4,704개 교환 중 메모리/커널 제약에 적합한 1,222개를 재검산했고, 더 낮은 소스 목적함수로 가는 단일 교환이 없었다. 이는 오직 소스 데이터의 국소성 증거다. 모든 출발점·채택 교환·정지 목적함수 기록을 보존했다. 새로운 정책의 전역 최적/동적 요청별 전환 지원은 주장하지 않는다.

새 모델 평가 실행 전에 최적해 출발점 라벨의 메타데이터만 오기였음을 발견해 **마스크·점수 불변을 SHA로 입증한 정오표**를 발행했다. 최초 영수증도 EOE 비공개 영역에 보존했다. 정오표 SHA: 9baf2d951c57ccacfc03626cb70ce110bb52fc16c834161d3933acb9ca2ca735.

## C. 서로 다른 과제에서 실제 28계층 BEGLIN CPU 평가

원본 BF16 HF MPS 대조: 34/48, 정답 단일 글자 NLL 1.227067 (BF16 원본과 BEGLIN의 소스 BF16 가중치 기반 widened F32 추론은 수치 경로가 다름).

Q4 C 프로그램 하나, 나머지 혼합 세 조건은 *동일한 별도 혼합 C 프로그램 하나*에 승격 텐서 SHA만 바꾸었다. 완전히 동일한 48개 새 HellaSwag 원본 프롬프트로 28계층 실행. 정답 NLL은 전체 어휘 확률에서, 정답률은 네 후보의 argmax로 계산했다.

| 네이티브 C 정책 | 정답 수 | 정답률 | 정답 한 글자 NLL ↓ |
|---|---:|---:|---:|
| 전체 Q4G64 | 23/48 | 47.9% | 2.167170 |
| 기존 Fisher64 | 24/48 | 50.0% | 1.864106 |
| 기존 Fisher128 | 23/48 | 47.9% | 1.895563 |
| 신규 교차 텐서 signed96 | 25/48 | 52.1% | 1.954110 |

Qv8 − 전체 Q4 NLL 차이: **-0.213060**. 현재 개발 실험 품질 판정: **새 mask가 전체 Q4보다 낮은 평균 정답 NLL을 보였으나 다른 모델/코호트 일반화는 미입증**.

### 쌍별 통계 — 탐색적, 다중 검정 보정 없음

동일한 48문항의 **신규 joint96 − 기준 정책** NLL 차이. 음수일수록 신규 정책이 좋으며 부트스트랩 5,000회 고정 시드 결과다. 정확도는 McNemar exact 검정(단일 데이터셋)으로 별도 판단.

| 비교 기준 | NLL 차이 (joint−기준) | 95% 쌍별 bootstrap 구간 | 정답 증가 | McNemar 정확 p |
|---|---:|---:|---:|---:|
| 전체 Q4G64 | -0.213060 | [-0.584824, +0.187032] | +2 | 0.6875 |
| 기존 Fisher64 | +0.090004 | [-0.215690, +0.413959] | +1 | 1.0000 |
| 기존 Fisher128 | +0.058547 | [-0.406714, +0.532856] | +2 | 0.7266 |

**새로운 보정/데이터셋 1개 + 모델 1개**만으로 고정된 글로벌 일반화, calibration 최적성 또는 운영 속도·메모리 이득을 주장하지 않는다. 특히 소스 전역 대리 손실이 극단적으로 작더라도 실제 출력을 독립 확인해야 한다.

## D. 독립 감사 및 재현 재료

- 사전등록 SHA: 9b833f5fc2729d13320c87e368baaff952c890ed068586622badfc7da63b4db1
- 새 데이터셋 잠금 공개 SHA: 7e4c9a014bc04799ef8701500c25174aea9449976c67ae93fba3f788bf4ec565
- 소스96 교차 텐서 대리지표 공개 SHA: 0e056473ec9e14a786245969b2c8413abc55fe08f6748f5b990c0f8e9b911982
- 원래 source-only Qv8 allocator 공개 SHA: c8eb0ad2737c3082772b40e8543f8a74d2b615c9d4c8b5e2ce9e29caf3ef0c88
- source independent audit SHA: 6ae7a5d03a104b95b1183c89548e5b545a41ab161d35e572f4e6b3c0a6427736
- independent exhaustive 1swap source certificate SHA: a4cd45da4886131fd5bae7a81d4f565cd9632109171592b20d61d0aef30368fe
- 새 HellaSwag 48×4 CPU 독립 결과 SHA: 84419a1695fc0a4cdcb46616b963f85e02e0f44d4e26c503bcaead3d760f181f
- BF16 원본 모델 독립 대조 SHA: 0e9cc25fcfe061a078e4f2e0768dad3c735b1fbd0235a0b457073eb0450e96f3
- 전체 96×196 BF16 original 정답 손실 그래디언트와 native Q4/n8 양자화 오차의 signed 스칼라 소스 checkpoint 검증, 4×48 모델 C 로그와 원본 모델 바이너리·프롬프트·A/B/C/D 토큰 SHA 검증 완료.
- 원본 gradient·원본 모델 가중치·개별 held 정답·C 원본 로그·작업용 weight 델타는 owner-only EOE lab_local에 보관, GitHub 제외.

## E. 이어서 필요한 연구

1. 새로운 모델군에서 이 signed 전역 목적함수의 재현성 검사. Qwen 소스96 / HellaSwag 개발48은 반복 사용하지 않기.
2. 단순 E[(Σg·δ)^2]가 국소 상쇄를 지나치게 유도할 수 있으므로 signed objective와 per-tensor scale-normalized error, Hessian-vector/GGN 근사의 외삽 실험을 **새 데이터로 사전등록**하기.
3. 안정적인 저부하 Apple CPU에서 같은 decode 계약 ABBA p50/p95/tokens/s/최대 RSS/소비전력과 n8 pack amortization 측정. MPS GPU와 CPU의 자동 승격은 실제 모델 전체 E2E 검증 전 금지.
4. 현재 운영 main, 원본 BF16 checkpoint, Tailnet Commander immutable operator, live auto promotion은 그대로 두기.

결론: SIGNED_BF16_SOURCE_CAPTURE_PASS / STRICT_CPU_MASK_FREEZE_PASS / NATIVE_HELLASWAG_4ARM_PASS / INDEPENDENT_SOURCE_AND_NATIVE_AUDIT_PASS / CROSS_MODEL_GENERALIZATION_NO_VERDICT / LIVE_PROMOTION_DENIED.
