# BEGLIN Qv11 — 실제 단일 텐서 8비트 승격의 네이티브 영향 검증

**연구일:** 2026-10-23 (KST)
**브랜치:** research/qv11-real-one-tensor-intervention-20261023
**판정:** REAL_NATIVE_SINGLETON_EXECUTION_PASS / SEVEN_ARM_AUDIT_PASS / SOURCE_RANK_DIRECTION_2_OF_3 / GENERALIZATION_NOT_ESTABLISHED / AUTO_PROMOTION_OFF

## 1. 연구 질문

Qv5~Qv10에서 원본 BF16 그래디언트, Fisher, signed joint loss, 조건부 출력 Gauss–Newton, Top2+OTHER 출력 KL을 순차적으로 평가했으나, 보정 대리지표가 낮아지는 것이 곧 BEGLIN 전체 모델의 실제 정답 NLL 개선으로 이어진다는 증거는 확보하지 못했다.

Qv11은 기존 목적함수를 또 추가하는 대신, **원본 Q4 대비 실제 C 모델의 정확히 하나의 투영 텐서만 8비트로 승격할 때 전체 출력 NLL이 얼마나 바뀌는가**를 직접 측정한다. Qv6 Fisher128 source 순위가 이 직접 개입 이득을 예측하는지도 검사한다.

## 2. 사전등록 및 원본성

- Qwen/Qwen2.5-1.5B-Instruct HF revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306.
- Source: 이전에 SHA 고정한 ARC-Challenge train128 BF16 그래디언트 Fisher 유사 점수. 역할별 28계층에서 가장 높은 계층 하나와 가장 낮은 계층 하나를 **평가 전** 고정했다.
- 평가: ehovy/race revision 2fec9fd81f1dc971569a9b729c43f2f0e6436637, middle **test** 16문항, 원문 지문이 다른 16건, 질문·보기 무자름, 126~207개의 Qwen ChatML 토큰.
- 사전등록 SHA-256: 2cd480dc4c72b5383014a1325d350e51a0b47b1335ec0013a6d069773b546a39.
- 원본 평가 16문항 비공개 SHA-256: 0de242b608837bd29ed72db99f1e910626753e2b27ea6ada9a24bdf864c96499.
- 원본 Q4 C 실행 바이너리 SHA-256: 408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c.
- 새 연구용 단일 n8 C 실행 바이너리 SHA-256: a054e68789fd6bc20a4bd815da5108b63361b14227b1fc4ab9e68a3b7b9a2711.

새 연구 바이너리는 감사된 기존 Qv5 native mixed C 소스에서 **n8 총량 검증 조건을 28→1로 변경**하고 출력 문구 두 곳만 수정한 복제본을 사용한다. 네이티브 Q4G64·n8 NEON 양자화 커널, 가중치, 28계층, 196개 투영은 기존과 동일하다. 단일 n8 조건마다 정확히 1개 n8 / 195개 Q4를 실제로 등록·실행했다. 원본 운영 엔진이나 모델 가중치는 변경하지 않았다.

## 3. 실제 RACE16 전체 모델 결과

정답률은 A/B/C/D 후보 중 argmax이며, NLL은 **전체 어휘 softmax 아래의 정답 단일 토큰**에 대한 값이다.

| 실험 | source Fisher128 이득 점수 | 실제 정답 | 실제 정답 NLL ↓ | 전체 Q4 대비 NLL 개선 |
|---|---:|---:|---:|---:|
| 원본 HF BF16 / MPS 참고 | 해당 없음 | 11/16 | 1.612267 | 서로 다른 연산 경로 |
| 전체 Q4 | 기준 | 8/16 | 2.432595 | 기준 |
| down_proj 2계층만 n8 | 101.9897 | 7/16 | 2.164660 | +0.267935 |
| down_proj 10계층만 n8 | 2.6998 | 8/16 | **2.070520** | **+0.362075** |
| k_proj 0계층만 n8 | 98.1972 | 8/16 | 2.257447 | +0.175148 |
| k_proj 1계층만 n8 | 0.3733 | 8/16 | 2.437356 | −0.004761 |
| v_proj 20계층만 n8 | 65.9075 | 7/16 | 2.188938 | +0.243657 |
| v_proj 27계층만 n8 | 0.3209 | 8/16 | 2.454870 | −0.022275 |

**다양한 정밀도 효과:** 한 개의 원래 Q4 텐서를 실제 n8로 변경하는 것만으로 네이티브 전체 모델 출력이 변한다. 모든 텐서가 좋은 방향으로 바뀌지는 않으며, NLL 개선과 정확도도 일치하지 않을 수 있다.

특히 down_proj는 source Fisher 점수 높은 2계층(101.99)이 점수 낮은 10계층(2.70)보다 실제 네이티브 NLL이 **0.09414 높았다**. 즉, 표본에서 역할 내 사전 순위가 역전됐다.

## 4. 민감도와 직접 개입 효과의 정합성

원래 BF16 source Fisher 대리지표는 E[(g·δQ4)^2 − (g·δn8)^2]로, 개별 양자화 오차의 방향성 제곱 차이를 나타낼 뿐 전체 네이티브 양자화 모델의 품질 이득이 아니다.

실제 16문항에서 source 점수 높은 텐서와 낮은 텐서를 짝지어 비교한 결과는 다음과 같다. **차이 = 높은 Fisher 점수 개입의 NLL − 낮은 점수 개입의 NLL.** 따라서 음수면 Fisher의 방향 예측과 일치한다.

| 투영 역할 | 높은 계층 / 낮은 계층 | 높은−낮은 실제 NLL | 16문항 paired bootstrap 95% CI | 방향 |
|---|---|---:|---:|---|
| down_proj | 2 / 10 | +0.094140 | [−0.399440, +0.733589] | **역전** |
| k_proj | 0 / 1 | −0.179909 | [−0.519733, +0.069584] | 일치 |
| v_proj | 20 / 27 | −0.265931 | [−0.663735, +0.076759] | 일치 |

- 세 역할의 high/low 순위 방향 중 **2/3 일치**.
- 여섯 텐서의 source Fisher 점수와 **실제 전체 Q4 NLL−1n8 NLL** 개선량의 Spearman ρ: **0.6000**.
- 실제 16개 지문의 B=5,000 재표집에서 ρ의 95% 백분위 구간: **[−0.085714, +0.942857]**.
- 각 역할 내 NLL 차이의 95% 구간도 모두 **0 포함**. 다중 검정 보정 및 전 196개 층의 무작위 대표성 확보 없음.

따라서 Qv11은 모델의 실제 단일 텐서 개입을 측정했고 일부 source 민감도 순위 역전을 관측했지만, **전체 196개 텐서·다른 과제·다른 모델군에 대한 Fisher 점수의 예측력**을 결론 낼 만큼 표본이 크지 않다.

## 5. 감사 증거

- 원본 Qv5 C 소스 계보와 별도 실제 native 1n8 엔진 빌드 공개 검증: docs/research/BEGLIN_QV11_ORIGINAL_NATIVE_1N8_BUILD_PUBLIC_20261023.json (SHA 19deebfcd33866bdfda1376b27a11e5463fca03e6ecdbd483268eb1e76c4f51a).
- 원본 RACE16 SHA·source high/low 고정: docs/research/BEGLIN_QV11_RACE_TEST16_PREMODEL_INPUTS_20261023.json.
- 실제 7조건×16문항 **112개** 네이티브 정답 NLL·정답 여부, 실제 1n8/195Q4 역할 등록, 바이너리·프롬프트·정답 파일 원문 SHA와 집계 독립 감사: docs/research/BEGLIN_QV11_REAL_ONE_TENSOR_NATIVE_CAUSAL_AUDIT_PUBLIC_20261023.json (SHA 026d3454a63168acee908e57696f61a73e3c8f883eab27b453e8b77ddda2fe15).
- 사전등록된 paired bootstrap 및 역할 내 민감도 순위 별도 검산: docs/research/BEGLIN_QV11_ROLE_RANK_PAIRED_BOOTSTRAP_PUBLIC_20261023.json (SHA f86f5a67cc3ec2a6a3c0d9f9a767a8a4e4ad82b853d39e1851ecf7009da22378).
- 원본 BF16/MPS 별도 참고(다른 연산 정밀도): docs/research/BEGLIN_QV11_ORIGINAL_BF16_RACE_TEST16_PUBLIC_20261023.json.

비공개 모델 가중치, 개별 RACE 지문/토큰, 전체 실제 C 로그와 개별 문항 NLL은 EOE 연구용 lab_local 아래 유지하고 GitHub에 올리지 않는다. GitHub 공개 CI는 공개 자료 계약을 검증할 뿐 비공개 모델의 BF16 역전파나 네이티브 추론을 다시 실행하지 않는다.

## 6. 다음 연구 순서

1. **분포가 다른 새 질문·모델에서 원본 source Fisher와 실제 native 단일 텐서 개입의 상관을 독립 검증**할 것. 여섯 극단 텐서 대신 비용·역할을 통제해 무작위 중간 순위 텐서까지 표본에 포함한다.
2. 각 텐서 단독 NLL 개선량이 **28개 n8 예산의 공동 선택 결과**를 얼마나 예측하는지, 두 텐서의 교차 항 및 비가산성을 직접 실험한다. 현재 singleton 설정을 28n8 정책의 효용과 동일시하지 않는다.
3. 충분한 실제 근거가 확보되기 전에는 source 학습지표나 개별 RACE16 결과를 새 할당기의 실측 성능 증명으로 재사용하지 않는다.
4. 저부하 Apple CPU 지속 TPS·p50/p95·RSS, Metal GPU의 모델별 경로 검증 후에만 운영 게이트 논의를 진행한다.

**운영 정책:** main 병합 없음, Tailnet Commander immutable 운영 릴리스 변경 없음, 요청별 동적 bit 자동 승격 없음, 기존 Qv5~Qv10의 긍정/부정 이력 그대로 보존.
