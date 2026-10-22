# BEGLIN Qv12 — 실제 두 텐서 동시 8비트 승격의 비가산성 검증

**날짜:** 2026-10-23 KST
**분리 연구 브랜치:** research/qv12-two-n8-interaction-20261023
**최종 범위:** REAL_NATIVE_C_SINGLE_AND_DOUBLE_N8 / INDEPENDENT_AUDIT / NO_GENERALIZATION_CLAIM / NO_LIVE_PROMOTION.

## A. 동기와 사전 실험 계약

Qv11에서 개별 텐서를 Q4에서 8비트로 바꾸는 실질적인 전체 Qwen28 BEGLIN C 개입을 수행했다. 그러나 28개 텐서를 동시에 승격시키는 혼합 할당이 개별 개선을 선형적으로 더할 수 있는지는 조사되지 않았다. Qv12는 이를 확인하기 위해 **진짜 두 텐서의 동시 개입**을 수행한 후, 둘의 단독 효과 합과 비교한다. 이것은 source Fisher/Gauss-Newton 등 대리점수가 아닌 동일 원본 네이티브 전체 모델의 실측 정답 토큰 NLL 차이이다.

원본 모델 HF revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306. 모델/토크나이저는 고정하고, **기존 Qv11 개발 데이터의 16문항에서 역할별 단독 개선이 가장 컸던** down_proj:10, v_proj:20, k_proj:0을 평가 전에 선정했다. 선별에 기존 결과를 사용했으므로 초기 3텐서 후보 선정은 outcome-driven이며, 새 RACE 평가 뒤에 후보를 바꾸지 않았다.

새 평가: ehovy/race@2fec9fd81f1dc971569a9b729c43f2f0e6436637, middle/test, 서로 다른 원문 지문 16개, 이전 Qv11 RACE16에 쓰인 지문을 모두 제외했다. 질문, 원문 지문, 4지 보기 무자름, 161~208 Qwen ChatML tokens. 전 일곱 모델에서 동일한 A/B/C/D one-token labels, source prompt SHA와 정답 SHA 사용. 그러나 16개 공개 test 문항이 언어모델의 사전학습에서 전혀 노출되지 않았다는 보장은 없다.

사전등록 SHA: a565bdec341d675ce52a37132b06355760c2fef96b8e4329c17508a237e4374f.
새 RACE16 private input SHA: 0650f791937f0d924706db6462b7f5e370fc58c1819b1228e91402d4937ca63e.

## B. 0개, 1개, 2개 n8 원본 C 실행

실험은 같은 Qwen2.5-1.5B 원본 BF16 사전양자화 가중치를 BEGLIN CPU 실기기 C로 로드했다.

- 기준 모델: 196개 Q4, n8 0개.
- 단독 모델 3개: 역할별 1개 n8, 나머지 195개 Q4.
- 두 텐서 조합 3개: 2개 n8, 나머지 194개 Q4.
- 원본 Q4·1n8 연구 바이너리와, 새 2n8 연구용 복제본은 original C source SHA에 묶었다. 2n8 복제본은 total_n8 cardinality assert를 28→2로 바꾸었고, 진단 문구 2개만 고쳤다. 원래 Apple NEON n8/Q4 kernel, 모델 계층 및 weight는 수정하지 않았다.

원본 Q4 binary SHA: 408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c.
원본 1n8 binary SHA: a054e68789fd6bc20a4bd815da5108b63361b14227b1fc4ab9e68a3b7b9a2711.
신규 2n8 binary SHA: d40fd6d42e08b0bc9d5dd3c7f8e0a4f5bcaabb2d9de7310d89f033e02500e961.

아래 투영 저장 크기는 진짜 소스 그룹별 Q4/n8 packed weights + scales를 통한 원가 장부 추정이다. 실제 RSS, GPU 사용량, 지속 실행 tok/s, 온도/전력은 실측한 항목이 아니다. 특히 down_proj n8은 k_proj/v_proj보다 추가 저장 비용이 크다. **2n8은 정식 28n8 정책과 예산이 맞지 않는다.**

## C. 진짜 전체 모델 7조건 ×16문항 결과

정답률은 A/B/C/D 후보 argmax, NLL은 전체 어휘로 정규화된 정답 single-letter next-token negative log likelihood다.
비교용 원본 BF16 PyTorch/MPS 참고 점수: **11/16**, 평균 NLL **1.266511** (네이티브 widened-F32 경로와 별도 산술).

| 실제 native 정밀도 정책 | n8 / Q4 | 정답 수 | mean NLL ↓ | Q4 대비 NLL 개선 ↑ | projection packed bytes |
|---|---:|---:|---:|---:|---:|
| q4 | 0 / 196 | 7/16 | 1.691992 | +0.000000 | 736,985,088 |
| down_proj_10 | 1 / 195 | 7/16 | 1.554375 | +0.137617 | 743,866,368 |
| v_proj_20 | 1 / 195 | 7/16 | 1.520034 | +0.171957 | 737,181,696 |
| k_proj_0 | 1 / 195 | 7/16 | 1.645575 | +0.046416 | 737,181,696 |
| down_proj_10__v_proj_20 | 2 / 194 | 7/16 | 1.398692 | +0.293299 | 744,062,976 |
| down_proj_10__k_proj_0 | 2 / 194 | 7/16 | 1.517718 | +0.174274 | 744,062,976 |
| v_proj_20__k_proj_0 | 2 / 194 | 7/16 | 1.478525 | +0.213466 | 737,378,304 |

단독 개선량은 **전혀 다른 이번 RACE16 문항에서 새로 측정**했고 과거 Qv11의 단독 NLL 절대값을 조합 효과 산술에 재사용하지 않았다.

## D. 실제 단독-동시 정밀도 상호작용

각 동일한 원본 평가 지문 s에 대해 아래 실제 네이티브 NLL 값을 사용한다.

    I_ij(s) = L_pair_ij(s) − L_single_i(s) − L_single_j(s) + L_Q4(s)

**I<0**이면 두 텐서 승격의 NLL 개선이 단독 이득 합보다 큰 초가산적(synergistic) 효과.
**I>0**이면 두 텐서 승격의 개선이 단독 이득 합보다 작은 부분가산적 효과.
**I=0**이면 가산적 효과.

16개 원문 지문을 단위로 5,000회 재표집(seed 20261023)한 쌍별 탐색적 부트스트랩 구간을 계산했다.

| 선택한 두 텐서 | 단독 NLL 이득 합 예상 ↑ | 실제 2n8 NLL 이득 ↑ | 초과 손실 I_ij (음수=시너지) | paired 95% CI | CI 판정 |
|---|---:|---:|---:|---|---|
| down_proj_10__v_proj_20 | +0.309574 | +0.293299 | +0.016275 | [-0.007659, +0.040534] | includes zero |
| down_proj_10__k_proj_0 | +0.184033 | +0.174274 | +0.009759 | [-0.021187, +0.045867] | includes zero |
| v_proj_20__k_proj_0 | +0.218374 | +0.213466 | +0.004907 | [-0.011568, +0.020998] | includes zero |

관측한 각 텐서 조합의 유한한 비가산적 평균 차이가 수치상 0이 아닌 쌍은 **3/3**이고, 부트스트랩 95% 구간이 0을 제외한 쌍은 **0/3**이다. 표본 16개·쌍 3개, 다중비교 보정 없음. 이것만으로 일반적인 196텐서 결합 효과나 범용 동적 할당기 승격을 주장하지 않는다.

## E. 독립 감사, 보안 및 향후 실험

원본 실제 Qwen C 7조건 ×16개 정답 NLL 총 **112개**를 독립 코드로 다시 파싱하고, C 바이너리/새 2n8 C 원본 소스 바이트와 exactly cardinality-only source diff, SHA-locked 평가 원문/프롬프트/one-token 정답, native 기록의 실제 Q4/n8 역할 수를 확인했다. 원본 개별 모델 입력/손실/가중치/로그는 untracked owner-only lab_local에 남겨두며 GitHub에 올리지 않는다.

**공개 독립 감사 SHA:** 89771cd5dafed2b8ee299d328125803ee039aaa2efc9195c5847a2bce83f8bac.
**원본 C 빌드 계보 공개 SHA:** f3c29d086dae8ec79775bc8d2f3f8498bce19dc9029e6156da94fefbe8e88cc2.
**원본 평가 입력 사전 봉인 SHA:** ae3bc62a1343b037d1a6f805eb8c89f30b9eeb1f5a33e99ddbda6071dba5d168.

남은 핵심 게이트는 (1) 다른 모델 계열의 진짜 2n8 개입 재현, (2) 원래 예산인 28n8 정책에서 순차 추가 이득/상호작용, (3) 더 큰 신규 평가 문항에 대한 bootstrap 신뢰도, (4) 저부하 Apple CPU p50/p95, RSS, Metal GPU 별도 검증이다. 이번 새 RACE16에 맞춰 후행 마스크를 조정하거나 같은 데이터에서 새 알고리즘 성능 우위를 주장해서는 안 된다.

**운영:** main 병합 없음, immutable Tailnet release 미변경, 원본 모델 및 커널 미변경, 자동 정밀도 라우팅/승격 금지.

**최종 판정:** REAL_2N8_INTERACTION_EXECUTED / INDEPENDENT_112_NATIVE_AUDIT_PASS / FULL_28N8_POLICY_NOT_PROVEN / NO_LIVE_PROMOTION.
