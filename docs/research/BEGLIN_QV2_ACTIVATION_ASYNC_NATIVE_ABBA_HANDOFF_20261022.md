# BEGLIN Q v2 — 활성값·커널 인식 양자화와 실제 비동기 병렬 실행 종합 핸드오프

**일자:** 2026-10-22 KST  
**실기기:** EOE Apple Silicon CPU / Alienware RTX 5070 Ti CUDA  
**GitHub:** `popixoxipop-collab/beglin`, 연구 브랜치 `research/oml-acs-prereg-20261021`  
**보호 범위:** BEGLIN `main`, 기존 모델 체크포인트, Tailnet immutable operator 릴리스, 원래 V6 최종·holdout 및 출처 미상 V6 24개는 변경/접근하지 않음.

## 결론

**활성값을 반영한 196개 텐서의 정밀도 선택과 Q/K/V·FFN 의존성 그래프의 병렬 실행은 모두 실제 코드·실기기로 기술 검증했다.** 하지만 **새 비트 할당은 별도 공개 개발 평가에서 더 좋은 품질을 보이지 않았고**, **실제 전체 모델의 pthread 병렬 Q/K/V·gate/up 실행도 유의미한 속도 향상을 입증하지 못했다.** 연구용 opt-in 구현·테스트·SHA 증거만 유지하고 자동 승격은 차단한다.

## 1. 품질 대리지표와 커널 측정

1. Alienware CUDA에서 원본 Qwen2.5-1.5B BF16 가중치를 그대로 읽어 공개 보정 문장 **12개/총 461개 토큰 위치**에 대해 **28계층 × 7투영 = 196개 투영의 입력 채널별 제곱 평균**을 forward-prehook으로 수집. 원본 hidden 텐서·보정 텍스트 원자료는 개인 연구 디렉터리와 공개 GitHub 간에 이동하지 않고, **196개 집계 점수**만 연구 브랜치에 저장했다. 개인 통계 SHA `ee7b177b4dde984a5d83e0cdb5983d2ae00c34ee752fef9113a79568a1d12663`.
2. 원래 BEGLIN C 인코더 `GGUF Q4G64` / `qNg64 n8`을 별도 CUDA가 아닌 CPU 계산에 사용해 196개 텐서의 **대각 활성값 가중 국소 투영 오차 감소**를 산출. 3개 독립 CPU worker를 bounded 병렬 처리했다. 이 지표는 **전 모델 로짓의 인과적 민감도, 헤시안, NLL, 실제 정답률을 의미하지 않는다.** 과거 차단된 텐서별 전체 모델 로짓 개입 실험은 재시도하지 않았다.
3. EOE Apple CPU에서 실제 4/8비트 조합별 QKV 8개, `gate/up` 4개의 **직렬·독립 1스레드 풀 병렬·독립 6스레드 풀 병렬** 실행을 측정했다. 실제 단계 성능상 더 나은 경우에도 추가 스레드풀·합류 비용과 전체 디코더 경로가 남는다. 실행 가능 경로는 `pre-norm → (Q∥K∥V) → attention → O → norm → (gate∥up) → SwiGLU → down`이며 레이어 간 병렬성은 주장하지 않는다. 원본 Q4 스레드풀의 job/cv 공유 동시 진입은 금지하고 역할별 개별 스레드풀을 사용한다.
4. 별도 MILP는 정확한 **28개 n8 / 168개 Q4**, 최대 **830,570,496바이트 투영 저장량**, 순차 지연 **100ms 예측 한도**, 병렬 DAG 지연 **90ms 예측 한도**를 두고 사전 동결된 source-only 값을 최적화했다. source BF16 rev `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. 기존 계열 내 80ms source-only 역할 top-k 제한 정책은 별도 후보였으며, 전체 196 임의 선택에 대해 최적이라 주장하지 않는다. 사전선택 시 평가 점수/로짓은 사용하지 않았다.

## 2. 정책별 진짜 native Qwen 개발 검증

**모두 같은 별도 공개 입력 8개/131개 다음 토큰 평가. 원래 V6 24개 또는 과거 C1-D 개인 라벨은 쓰지 않았다.**

| 정밀도 배치 | 교사 강제 PPL ↓ | 비고 |
|---|---:|---|
| 원본 BF16 가중치에서 FP32로 복원 | **348.36** | 정확도 기준, native BF16 계산은 아님 |
| 기존 frozen seed73 무작위 28n8 | **552.70** | 28개 혼합 정책 중 더 양호 |
| 전체 4비트 Q4G64 | 727.92 | 빠른/메모리 저렴한 대조군 |
| 활성값+병렬 비용 최적화, source-only 80ms 가정 | 790.69 | 품질 개선 실패 |
| 활성값+병렬 비용 글로벌 MILP, source-only 90ms 가정 | **827.50** | 품질 개선 실패 |

글로벌 MILP 정책은 **8비트 28개/4비트 168개**, 투영 가중치 약 **776.9MB**, source-only 병렬 DAG 모델 예측 **89.71ms/토큰**, 순차 모델 예측 **91.93ms/토큰**이다. 이 수치는 이전에는 예측일 뿐이었다. 이번에 **실제 Q/K/V·gate/up 전체 모델 pthread 병렬 실행**을 구현/검증해 별도로 측정했다.

## 3. 실제 전체 모델 비동기 ON/OFF (동일 바이너리)

**모델:** Qwen2.5-1.5B 전체 28계층; 실제 원래 BF16 가중치에서 생성한 동일 Q4 168개와 n8 28개 텐서, 동일 리비전·계층별 커널 스케줄.  
**native 바이너리 SHA** `fb947372495962ed67bb00f91199030b2e41d50ab283f4a5bd81ab2e51cdc054`.  
**정밀도 마스크 SHA** `9324c5ec891d638f2713d3bbd4187399e1aae8a9a74df129dfabc4e8ea870212`.

- `BEGLIN_QV2_NATIVE_ASYNC=1`: 별도 실험 C 코드가 고정된 레이어별 QKV·gate/up 단계 스케줄을 호출하고 **의존성 경계에서 join한 후만** attention/SwiGLU로 진행. 검증 때 public 평가 경로에서 각 단계의 합류 **3,892회**와 native n8 GEMV 호출 증거를 확인.
- `BEGLIN_QV2_NATIVE_ASYNC=0`: 원래 순차 QKV·gate/up 호출. **실기기 전체 어휘 151,936개 첫 로짓 배열은 ON/OFF의 파일 SHA가 완전히 일치**.
- 8개 공개 평가 131개 위치의 **PPL ON = OFF = 827.4999**. 즉, 현재 작은 평가 범위에서 명령어/실행 순서를 바꿔도 출력 품질이 유지된다.

**동일 네이티브 바이너리·측정 순서를 교차한 decode 속도 (토큰당 ms, 낮을수록 좋음):**

| 실험 | 순차 OFF | 비동기 ON | 결과 |
|---|---:|---:|---|
| 최초 분리 시험쌍 | 92.04ms | 92.66ms | ON이 약 0.7% 느림 |
| 두 번째 ABBA 시험 | 98.21 / 90.61ms | 94.33 / 93.98ms | ON 평균 약 0.27% 빠름 |
| 합산 3회 평균 | **93.62ms** | **93.66ms** | **병렬 속도 우위 미입증** |

원래 stage microbench에서 QKV 1.13~1.55배, gate/up 1.06~1.90배 이득이 있던 조합도 있었다. 그러나 전체 네이티브 LLM은 토큰화/헤드/나머지 직렬 O·down, 스레드풀 시작·join·CPU 경합 때문에 **실측 전체 처리속도 이득이 없다**. 측정치는 BEGLIN 워밍업 뒤 1토큰 best-of-3의 소수 반복값이므로 장시간 TPS/분산을 나타내지 않는다.

실제 PPL 프로세스 최대 RSS는 ON 약 **4.870GB**, OFF 약 **4.869GB**. source-only 투영 바이트 외에 여러 Q4 worker 풀의 스크래치·운영 메모리가 별도로 존재한다.

## 4. Fail-closed 선택 정책

**실험용 read-only eligibility gate** `research/oml_acs/q_v2_actual_async_eligibility.py`를 추가했다. 기기당 실제 ON/OFF speed ratio가 최소 **1.05(5%)**이고 PPL parity 및 대조군 대비 품질 손실이 없어야 *추가 검토 후보*로 인정한다. 5%는 엄격한 엔지니어링 안전 여유로 실측 후 정한 값이며, 사전등록된 과학적 유의성 기준이 아니다.

이번 6회 네이티브 측정의 OFF/ON 평균 속도비는 **0.999608**, 품질도 선행 무작위 혼합·전체 Q4보다 나쁘므로 **두 게이트 모두 불합격**. 따라서 `BEGLIN_QV2_NATIVE_ASYNC=0` 기본값과 기존 승격 차단을 유지한다. 이 결정문서가 실제 환경변수를 바꾸거나 운영 자동 승격을 수행하는 것은 아니다.

## 5. 산출물·검증·향후 작업

| 항목 | 코드/증거 |
|---|---|
| 비동기 독립 stage 커널 측정 | `research/oml_acs/q_v2_async_stage_native.c`, `q_v2_async_stage_benchmark.py` |
| BF16 활성값 수집 | `research/oml_acs/q_v2_capture_activation_diag.py` |
| 196 활성값 가중 오차 | `research/oml_acs/q_v2_score_activation_local.py` |
| 기존 사전동결 MILP | `docs/research/BEGLIN_Q_V2_MILP_FROZEN_ACTIVATION_ASYNC_POLICY_20261022.json` |
| 실제 native async stage runtime | `research/oml_acs/q_v2_native_async_stages.h`, `q_v2_native_async_build.py` |
| 실제 ON/OFF 6회·PPL 재현 | `research/oml_acs/q_v2_actual_async_native_abba_test.py` |
| 동시성 진입·출력·품질 fail-closed 감사 | `research/oml_acs/q_v2_actual_async_eligibility.py`; **회귀 8/8 PASS** |
| 공개 요약 | `docs/research/BEGLIN_QV2_ACTUAL_ASYNC_ABBA_RELEASE_GATE_EVIDENCE_20261022.json` |
| 비공개 실제 ON/OFF SHA | `8f03f93670400c3e2b34f83bdc1d8fc65e1dd0bd915556dafd4f73b0201bab00` |
| 비공개 독립 gate SHA | `6e9ed2794964aa4c3efba307744db61388730ada2aa18c81f4f65838c5da5341` |
| 별도 기존 독립 감사 SHA | `2f326545ef39f02988138f6c0c0d50ea23bc7c528fcb68b23c1b96144b6af438` |

추가 검증 없이 **연구 코드에서 운영에 자동 승격하면 안 된다.** 향후 실제 개선 방향은 전체 모델 품질/지연이 동시에 유리한 새로운 정밀도 선택 목적, QKV 병렬 스레드 경합 억제 및 역할별 CPU 코어 예약, 장시간 연속 디코드·실제 동시 사용자 요청·소비 전력·Apple Metal CPU/GPU 경로, 신뢰할 수 있는 새 IID/OOD 실험이다. 기존 차단된 텐서별 실제 로짓 개입을 우회/재시도하거나 원래 V6 24개를 열지 않는다.

**운영 엔진/파일 접근 경계:** `main`·Qwen 체크포인트·Tailnet 현재 릴리스 변경 없음. 단일 호스트의 0.27% 차이를 논문 성과 또는 과학적 모델 품질 우위로 주장하지 않는다.
