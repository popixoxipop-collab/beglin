# BEGLIN Q v2 — 활성값 품질 대리 지표 + 커널 인식 비동기 DAG 할당 연구 핸드오프

**날짜:** 2026-10-22 KST  
**저장소/브랜치:** `popixoxipop-collab/beglin`, `research/oml-acs-prereg-20261021`  
**상태:** **알고리즘 구현 및 전체 Qwen2.5-1.5B 실험 완료 · 실제 품질 향상 FAIL · 운영 승격 NOT_READY**.

## 1. 이번 목표와 실제 수행

기존 v1은 Apple Silicon CPU에서 실제 BEGLIN Q4G64 6스레드 및 qNg64 n8 사전 재배열 NEON의 속도/메모리를 측정하고 196개 텐서 중 28개를 승격했다. 다만 가중치 복원 오차만으로는 전체 모델의 NLL을 잘 예측하지 못했다.

이번 v2는 **품질을 고려한 정밀도 선택**에 원본 BF16 *활성값 분포*를 추가하고, 모델 계층 내 독립적으로 계산 가능한 **Q/K/V 및 FFN gate/up의 비동기 병렬 작업 그래프**를 별도 실기기에서 측정했다.

세 구역을 분리했다.

1. **원본 BF16 활성값 수집:** Alienware RTX 5070 Ti에서 Qwen2.5-1.5B 정확한 공식 리비전 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`의 196개 projection 입력에 *forward prehook*을 설치했다. 미리 고정한 공개 보정 문장 12개, 총 **461개 토큰 위치**에 대해 입력 채널 제곱 평균만 산출했다. 기존 v6 최종 문항/기존 C1-D 레이블, 모델 내부 가중치 변경·텐서 개별 출력 교란은 사용하지 않았다.
2. **196개 원본 텐서 품질 대리 점수:** 원본 BEGLIN C 양자화기(Q4G64 및 qNg64 n8)를 사용해 각 텐서의 BF16 원본 및 복원 가중치 차이를 활성값 제곱 평균으로 가중했다. **3개의 제한된 CPU 작업자**가 별도 텐서를 병렬 계산했다. 결과 196/196, SHA 개인 원본 `7b3ad80294275807cb3775601915b18ec3caeeef430849d8cea48389b3623616`, 원본 활성값 NPZ SHA `ee7b177b4dde984a5d83e0cdb5983d2ae00c34ee752fef9113a79568a1d12663`. 이것은 *입력 조건부 채널 대각 국소 출력오차*이지 실제 28계층 최종 출력 또는 Hessian/인과적 민감도가 아니다.
3. **독립 투영 단계의 병렬 비용:** EOE Apple Silicon 실기기에서 원본 Q4G64 1/6스레드와 재배열 n8 커널을 측정했다. 병렬 Q4는 **역할별 독립 q4pool**을 사용하여 공유 job 구조의 경쟁 상태를 피했다. QKV 8비트 조합 8개와 gate/up 4개에 대해 직렬·병렬 1스레드·병렬 6스레드의 실제 벽시계 지연을 비교했다. QKV에서 직렬 대비 **1.13~1.55배**, FFN gate/up에서 **1.06~1.90배** 단계별 병렬 효과를 관측했다. 비동기 작업의 스레드풀 오버헤드·CPU 과구독은 측정치에 포함된다.

**중요:** 전체 BEGLIN 추론기는 이 단계에서 **직렬 그대로**다. 실측 비동기 단계 시간을 전체 토큰/초로 환산해서는 안 된다.

## 2. 정수 최적화 및 동결

사전 동결 문서: `configs/research/BEGLIN_Q_ACTIVATION_ASYNC_PARALLEL_V2_PRELOCK_20261022.json`, `configs/research/BEGLIN_Q_V2_ASYNC_DAG_MILP_PRELOCK_20261022.json`.

- Qwen 28계층 × 네 가지 실행 단계(QKV, O, gate/up, down)의 정밀도·커널 조합: **이진 선택 변수 448개**, 실제 7종 텐서 역할 196개.
- 한 계층의 QKV(Q/K/V)는 동일 입력의 투영 계산에 한해 병렬 후보. attention 결과를 필요로 하는 O는 뒤이어 실행. FFN gate/up은 동일 post-attention LN을 입력으로 병렬 후보, down은 결합 이후 실행. 계층 간 무조건 병렬화하지 않는다.
- 제약: **8비트 28개, 4비트 168개**, 투영 가중치 압축 비용 **830,570,496바이트 이하**, 기존 Q4 기준/역할별 커널 비용에 보정한 **직렬 전체 토큰 예상 100ms 이하**, **실측 독립 단계 DAG 비용 28계층 합계 50ms 이하**(전체 모델 지연 아님).
- 목적: 원본 BF16 활성값 가중 국소 복원오차의 감소 최대화. SciPy HiGHS MILP **optimal, gap 0**로 해를 얻었다.
- **선택된 8비트 역할 수:** `q_proj=1, k_proj=1, v_proj=23, o_proj=1, gate_proj=0, up_proj=2, down_proj=0` (합계 28). 나머지 168개 Q4.
- 투영 압축 가중치 **757,825,536바이트**; 사전 추정 직렬 토큰 시간 **83.31ms**, 독립 병렬 단계 비용 합 **49.70ms**. 세부 텐서 ID/추천 stage dispatch와 SHA는 동결 파일 `configs/research/BEGLIN_QV2_ASYNC_ACTIVATION_PRE_EVAL_FROZEN_POLICY_20261022.json`에 기록했다.
- 정책 **SHA `cb9268c9177d47705e548c2a90a6b22772569b3f2bb54dfef2767ca53ffa3e76`**, 원본 비공개 MILP 영수증 **SHA `7d19591c22846f8d001256f4c4addacfa7babeebf49eae9b3783128e9c464ea4`**. 평가 출력은 정책 동결에 이용하지 않았다.
- 독립 MILP 검증 `tests/research/test_q_v2_async_activation_milp.py` **8/8 PASS**. 실제 448개 후보, 112개 stage 선택의 SHA, 28개 승격/메모리 한도/직렬·DAG 제약, 원본 가중치와 활성값 점수 합산을 검산했다.

## 3. 신규 공개 개발 입력 8개에 대한 실제 native Qwen 실측

미리 고정한 **평가 문장 8개**, 토큰화 후 총 **131개 다음 토큰 예측 위치**. 여섯 정책에 각 파일의 원시 입력 토큰 바이트를 동일하게 복사해 SHA로 검사했다. 공개 합성문장으로서 **독립 최종 IID/OOD 증거는 아니다**.

| 조건 | 동일 131개 위치 PPL ↓ | 토큰 생성 속도 | PPL 실행 최대 RSS |
|---|---:|---:|---:|
| 원본 BF16 가중치를 FP32로 복원 | **348.3639** | 별도 이번 벤치 없음 | 9.273GB |
| 이전 역할 균등 무작위 혼합(28 N8) | **552.7020** | 이전 단일 토큰 약 8.61 tok/s | 4.923GB |
| 기존 v1 커널 인식 가중치오차 정책 | 619.6130 | 이전 단일 토큰 약 12.05 tok/s | 4.861GB |
| 이전 역할 균등 가중치오차 혼합 | 705.3324 | 이전 단일 토큰 약 8.61 tok/s | 4.923GB |
| 전체 Q4G64 | 727.9173 | 이전 단일 토큰 약 13.29 tok/s | 4.829GB |
| **새 v2 활성값+비동기 비용 정책** | **807.3630** | **11.975 tok/s**(83.5ms/token) | **4.852GB** |

속도는 기존 BEGLIN `bench 1`의 워밍업 이후 단일 토큰 최단값을 근거로 하며, 표에서 이전 측정으로 표시된 값은 **같은 시점 새 측정이 아니다**. 지속 처리량·동시 사용자 성능을 주장할 수 없다.

새 정책의 PPL을 NLL로 환산하면 **6.69377 nats/token**, 고정밀 기준은 **5.85325**이며 상대 NLL 증가 **14.36%**다. 이전 무작위 혼합의 NLL은 **6.31482**로 더 양호하다. 동일 개발 샘플에서 새 정책은 전체 Q4보다 **PPL·투영 가중치 저장량·관측 단일 토큰 지연 모두 불리**한 후보다.

**품질 판정: FAILED ON DEVELOPMENT / FINAL Q NO_VERDICT.** 정밀도를 더 올리는 텐서를 골랐어도 국소 재구성 오차만으로 전체 모델 성능을 설명하지 못한다. 선택 정책에 평가 결과를 반영해 수정하는 일은 금지한다.

## 4. fail-closed 게이트 및 검증

- 연구 게이트 `research/oml_acs/q_v2_failclosed_promotion_guard.py`: 실측 증거의 SHA·여섯 정책 비교·원본 대비 NLL 증가·비동기 전체 추론 미구현·테스트 크기를 확인한다. 결론 **`NOT_READY`**, `--require-ready` 실행 시 종료 코드 **2**.
- 게이트 사유: 평가 131 위치 부족, 상대 NLL 증가 2% 기준 초과, 전체 Q4에 관측된 품질·메모리·속도 열세, 전체 비동기 추론 실행 미구현, 독립 IID/OOD 및 실제 작업 AUROC 미검증.
- 연구 검사: MILP 독립 시험 **8/8 PASS**, 6개 native 입력·바이너리·PPL 교차 시험 **8/8 PASS**, fail-closed 게이트 시험 **5/5 PASS** = **21/21 PASS**. 테스트가 통과했다는 것은 **정책 품질이 좋다는 뜻이 아닌, 부정적 결과를 올바르게 보존했다는 의미**다.
- 비공개 native 평가 요약: `lab_local/q_v2_activation_async_native_eval_20261022/qv2_6case_native_evaluation_summary_private.json`, SHA `1dfb72b95b11262bbde56e2e47fa52965c96a4fe6c21f089b999ba1c816c9185`. 사용자 원문/모델 텐서/원시 활성값은 공개 저장소에 올리지 않았다.
- 공개 실측 요약: `docs/research/BEGLIN_QV2_ACTIVATION_ASYNC_6POLICY_REAL_NATIVE_20261022.json` 및 활성값·비동기 측정 JSON.
- 네이티브 연구 실행 코드: `research/oml_acs/q_v2_capture_activation_diag.py`, `q_v2_score_activation_local.py`, `q_v2_async_stage_native.c`, `q_v2_async_stage_benchmark.py`, `q_v2_async_activation_milp.py`, `q_v2_native_policy_eval.py`.

## 5. 다음 연구와 운영 분리

**완료:** 원본 BF16 활성값 통계+196개 텐서 오프라인 혼합 비트/커널 선택+측정된 QKV/FFN 비동기 단계 비용+Apple native 전체 모델 개발 검증. **미완료:** 실제 full BEGLIN 디코더의 안정적인 비동기 실행, 입력별 런타임 동적 비트전환, 모델 행동 민감도 추정의 품질 일반화, 새 데이터 기반 정량 유효성.

우선순위는 **출력 품질 제약을 먼저 지키는 더 나은 민감도/불확실성 모형을 보정 데이터에서 정량 검증**하고, 별도 역할별 안전한 CPU worker pool을 실제 디코더의 의존성 경계에 통합하되 동등 로짓/타임라인/스레드 경합/메모리 핫패스/CPU·Metal 통합을 검증하는 것이다. 기존에 차단된 단일 텐서별 전체 로짓 교란 실험은 우회하거나 수행 완료라고 표시하지 않는다. 원래 V6 사용 24개 문항 출처도 인증되지 않았으므로 최종 holdout 사용 금지.

**변경 금지 준수:** BEGLIN `main`·원본 모델·운영 추론 프로세스·immutable Tailnet 운영 릴리스 변경 없음. 오직 분리된 연구 체크아웃/별도 G5 자원 예약·비공개 입력 디렉터리와 연구 GitHub 브랜치를 사용했다.
