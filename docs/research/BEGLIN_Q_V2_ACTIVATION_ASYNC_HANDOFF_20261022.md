# BEGLIN Q v2 — 활성값·비동기 병렬 처리 고려한 정밀도 할당기 핸드오프

**날짜**: 2026-10-22 KST  
**저장소/연구 브랜치**: `popixoxipop-collab/beglin` / `research/oml-acs-prereg-20261021`  
**실험**: Qwen2.5-1.5B-Instruct, 28계층 × 7개 투영 역할 = 196개 텐서. CUDA BF16 정합 소스 활성값 추출(Alienware) + Apple Silicon EOE CPU 원본 BEGLIN Q4G64/n8 원시 커널·pthread 실제 병렬 실행.  
**상태**: 소스 기반 점수 196/196, 커널 병렬 비용 12조합, 정밀도 할당 정책 동결, 전체 모델 8개 공개 개발 입력·131개 예측 위치 평가, 원본/병렬 실제 비교, 독립 감사 9/9 PASS. **품질 우위 실패, CPU 전체 모델의 비동기 속도 우위 실패, 운영 승격 금지.**

## 1. 목적과 바뀐 설계

기존 v1의 가중치 상대 L2 오차 기반 할당은 역할별 무작위 28개 8비트 승격 정책보다 실제 생성 NLL이 높았다. v2는 **원본 BF16 모델의 입력 활성값 분포**를 반영하고, **커널 실행 비용뿐 아니라 독립 연산의 실제 비동기 겹침**도 구분해 평가했다. 앞서 실행이 차단된 *각 텐서 단독 승격에 대한 전체 모델 최종 로짓 반사실 민감도*는 재시도하거나 우회하지 않았다.

원본 BF16 Qwen2.5-1.5B 정확한 공식 revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Apple 원본 모델 별도 캐시 revision: `4ae77cb209f06199b8df1c94e21ff341332a3a89`. 앞선 실험에서 주요 원본 텐서 샘플의 BF16 바이트 SHA 일치를 확인했으며, 서로 다른 장비의 전체 체크포인트 텐서 100%를 별도 대조했다고 주장하지 않는다.

## 2. 품질 민감도 대리 지표

Alienware RTX 5070 Ti CUDA에서 공식 원본 BF16 모델을 **변경 없이** 불러왔다. 사전 동결된 공개 합성 보정 문장 **12개(전체 461개 토큰 위치)** 를 실행하며 196개 투영의 입력에 forward prehook을 걸었다. 각 텐서의 입력 채널 제곱 평균만 모으고 로짓·정답 레이블·원시 토큰별 활성값은 정책 입력으로 사용하지 않았다.

각 텐서 i의 4비트/8비트 후보별 오차는 다음 **대각 채널 공분산 근사**로 계산한다.

`D_i(b) = [sum_j E[x_{ij}^2] sum_o (W_{i,oj} - Q_b(W_i)_{oj})^2] / [sum_j E[x_{ij}^2] sum_o W_{i,oj}^2]`

8비트 승격 이득은 `G_i = D_i(4) - D_i(8)`. 원래 BEGLIN Q4G64/원래 qNg64 n8 인코더를 이용한 오차이며, 원래 BF16 가중치를 캐시한 CUDA 호스트에서 **동시 CPU 작업자 3개로 196개**를 계산했다. **이 지표는 국소 투영출력의 대리 민감도이며, 실제 정답률·최종 로짓 영향·NLL 인과효과가 아니다.**

- 사전 동결 프로토콜: `configs/research/BEGLIN_Q_ACTIVATION_ASYNC_PARALLEL_V2_PRELOCK_20261022.json`.
- 비공개 활성값 모멘트 SHA-256: `ee7b177b4dde984a5d83e0cdb5983d2ae00c34ee752fef9113a79568a1d12663`. 원자료 `0600`, 사용자 개인 데이터나 이전 V6 최종 시험 입력 없음.
- 비공개 196개 품질 점수 전체 SHA-256: `7b3ad80294275807cb3775601915b18ec3caeeef430849d8cea48389b3623616`. 공개 집계는 `docs/research/BEGLIN_Q_V2_ACTIVATION_DIAG_196_PUBLIC_AGGREGATE_20261022.json`에만 저장했다.

## 3. 비동기 실행 비용 측정

Apple Silicon EOE에서 **원본 BEGLIN CPU Q4G64(스레드풀 1/6개) 및 qNg64 n8 사전 재배열 NEON**에 같은 모델 가중치를 입력했다.

**진짜 데이터 의존관계:**

`RMSNorm → [Q ∥ K ∥ V] JOIN → RoPE/Attention → O → residual/Norm → [Gate ∥ Up] JOIN → SwiGLU → Down → 다음 계층`.

Q4G64의 기존 전역 q4pool은 여러 호출이 공유하면 race 위험이 있으므로, 동시 CPU 검증에서는 각 역할에 **독립적인 네이티브 q4pool**을 갖췄다. QKV **8개 비트 조합**과 GateUp **4개 조합**의 직렬 및 병렬 단계 실제 지연을 각각 측정했다. 최고 비동기 단계 이득은 QKV **1.55배**, GateUp **1.90배**였으나 이는 전체 모델의 속도가 아니다.

비공개 12조합 단계 영수증 SHA-256: `380670d42ccd2f12c4a015600be3d921f672c570a3d3a4a1652c61e217791808`.

## 4. 오프라인 비트+커널 할당 정책

원본 196개 점수·기존 역할별 커널 실측 비용과 12조합 비동기 단계 DAG 비용만 보고, 새 평가 데이터를 열기 전에 정책을 SHA로 고정했다.

- 고정: **28개 8비트, 168개 4비트, 압축 투영 가중치 최대 830,570,496바이트**.
- `serial100`: 원본 단일 토큰 직렬 비용 추정 100ms 이하, 역할마다 활성값 기반 이득 순위 상위 k개를 선택. 역할 개수의 제한된 정수 공간에서 정확하게 탐색했다.
- `parallel80`: 위 직렬 제한에 추가로 원본 QKV·GateUp 비동기 단계 실측으로 구성한 **가상 병렬 DAG**의 예측 지연 80ms 이하. 같은 제한된 역할별 top-k 정책 집합에서 탐색한다. 전역 2^196 비트 패턴에 대한 완전 최적성은 아님.
- 별개로 기록된 `v2_activation_async_MILP`: Qwen 각 계층의 7-bit 역할 마스크 128개 중 하나를 택하는 총 3,584개 이진 후보에 대한 HiGHS 기반 소스 전용 MILP 정책. **이 정책의 글로벌 최적성 기록은 Apple EOE에서 독립 SciPy 재실행하지 않았다.** 원래 선택 정책·실행 근거 SHA는 보존했다.

| 정책 | 8비트 계층 개수 q/k/v/o/gate/up/down | 코드+스케일 저장량 | 비용 예측 |
|---|---|---:|---|
| 기존 역할 균등 | 4/4/4/4/4/4/4 | 830.6 MB | 기존 실측 116.18ms/토큰 |
| v2 `serial100` | **2/0/18/1/0/5/2** | **792.2 MB** | 직렬 98.99ms |
| v2 `parallel80` | **2/1/22/1/0/2/0** | **758.8 MB** | 직렬 83.87ms / 가상 병렬 79.58ms |

두 새 정책의 **실제 전체 모델 단일 토큰 디코드 속도 추가 시험 요청은 안전 검사에 차단되어 수행하지 못했다**. 이 예측값을 실측으로 쓰면 안 된다. 전체 모델 비동기는 뒤의 별도 MILP 정책에서 실제로 실행했다.

정책 동결 SHA-256 `8dd48956820d47073f7f415c7427d07f20e01db154cdd8c895ec5d6d134c38a4`.

## 5. 신규 8개 공개 개발 입력의 실제 전체 Qwen 품질

소스 보정 입력 12개와 분리하여 미리 동결한 **공개 합성 문장 8개, teacher-forced 예측 위치 131개**를 원본 BEGLIN Apple CPU 전체 Qwen에 동일한 바이트 순서로 입력했다. 모든 정책의 Q4G64와 qNg64 n8는 실제 압축 커널을 썼고, 출력 헤드 등은 동일 조건이다.

| 조건 | PPL(낮을수록 좋음) |
|---|---:|
| 원본 BF16 소스→FP32 widened 기준 | **348.3639** |
| 기존 seed73 역할 균등 무작위 혼합 | **552.7020** |
| 전체 Q4G64 | **727.9173** |
| v2 활성값+`parallel80` 정책 | **790.6870** |
| 별개 소스 전용 비동기 MILP 정책 | **827.4999** |
| v2 활성값+`serial100` 정책 | **1084.7481** |

**기존 무작위 혼합을 능가한 새 활성값 기반 정책은 없다.** 오히려 더 작은 메모리와 예측 실행비용으로 만든 정책들이 실모델 PPL에서 열위였다. 이는 데이터가 적은 개발 관찰이다. 활성값 대각 모멘트가 전체 모델 행동/정답 품질을 충분히 설명하지 못한다는 주요 부정적 증거로 보존한다.

- `serial100`/`parallel80` 5조건 실측 SHA-256 `afff3f20a7d258a2639ebe28e3a163eedfa46787503d7a1ff31abcee0ab56f7f`.
- 별개 MILP 8공개 입력 5조건 영수증 SHA-256 `043ee279a2f5a7e0880286aa0c1b5e5cae7692dbf4775bf53c49a27b803383a9`.
- 공개 8문장은 차후 독립 final IID/OOD 시험이나 실제 업무 QA와 대체 가능하지 않다. 원본 V6에서 사용한 24개 입력 ID는 아직 확인하지 않았고, 사용 출처 중복 배제가 완결됐다고 주장하지 않는다.

## 6. 전체 모델의 **진짜** 비동기 pthread 구현

`research/oml_acs/q_v2_native_async_stages.h`는 QKV 3개, GateUp 2개 작업에 대해 재사용하는 pthread 2개 + caller, 각 역할 독립 1/6-thread Q4풀을 사용한다. JOIN 후에만 Attention/SwiGLU가 출력을 읽게 했다. 이전에 동결된 `v2_activation_async_MILP` 정밀도 정책과 단계 스케줄을 source SHA로 확인하고, 동일한 격리 바이너리에서 `BEGLIN_QV2_NATIVE_ASYNC=0`과 `=1`을 직접 비교했다.

| 실제 Qwen 28계층 같은 바이너리 | 직렬 | 비동기 pthread |
|---|---:|---:|
| 워밍업 후 단일토큰 best-of-3 지연 | **92.04ms** | 92.66ms |
| 생성 속도 | **10.865 tok/s** | 10.792 tok/s |
| 같은 공개 131위치 PPL | **827.4999** | **827.4999** |
| PPL 실행 최대 RSS | 4,869,226,496바이트 | 4,869,980,160바이트 |
| 확인된 QKV / GateUp JOIN (PPL 측정) | 0 / 0 | **3892 / 3892** |

- 실제 비동기 실행과 이전 직렬 실행의 **최종 151,936개 로짓 바이트 SHA가 완전히 동일**했다.
- 실제 전체 모델 비동기는 직렬 대비 **0.9933배**, 즉 **약 0.7% 느렸다**. 독립 단계의 병렬 이득은 전체 모델의 추가 스레드/합류 비용에 의해 상쇄된 것으로 판단한다. 이 병렬 스케줄은 **성능 승격 조건 불충족**.
- Native pthread 빌드 실행 바이너리 SHA `fb947372495962ed67bb00f91199030b2e41d50ab283f4a5bd81ab2e51cdc054`. EOE에서 `q_v2_actual_native_async_retry1_20261022`에 격리 저장.
- 비공개 직렬-병렬 paired 영수증 SHA `26096060a809b380cbf5bef749b87f09dcec1a253c3ecc99d6ef20652daec692`.

## 7. 재현성·감사·후속 작업

독립 소스 점수 재계산, 12개 단계 조합 비용·미리 고정한 두 비트 정책·새 평가 입력 동일성·PPL·실제 pthread 실행 횟수·전체 로짓 SHA·부정적 병렬 결과를 `research/oml_acs/q_v2_activation_async_independent_audit.py`에서 검증했다. 비공개 영수증 SHA **`2f326545ef39f02988138f6c0c0d50ea23bc7c528fcb68b23c1b96144b6af438`**. `tests/research/test_q_v2_activation_async.py` **9/9 PASS**.

코드 경로: `q_v2_capture_activation_diag.py`(원본 BF16 CUDA), `q_v2_score_activation_local.py`(원본 BEGLIN encoder·CPU 3작업자), `q_v2_async_stage_native.c`/`q_v2_async_stage_benchmark.py`(role-local 1/6-thread Q4풀), `q_v2_joint_activation_async_allocator.py`(소스 전용 제한 정확 탐색), `q_v2_evaluate_frozen_native.py`(8개 신규 공개 개발 PPL), `q_v2_native_async_stages.h`/`q_v2_native_async_build.py`/`q_v2_native_async_paired_verify.py`(진짜 pthread 전체 모델), `q_v2_activation_async_independent_audit.py`.

**다음 단계(승격 아닌 연구):**

1. 온디바이스 병렬 스케줄의 실제 전체 모델 지연을 다시 줄이려면 QKV/GU별 **병렬 실행 임계치**, persistent q4pool의 **총 스레드 수/oversubscription**, worker barrier 비용을 **별도 속도 보정용 입력으로** 측정한다. 현재 성능 열위인 모든 구간을 무조건 병렬 실행하지 않는다. 최종 성능 평가는 보정과 분리한다.
2. 가중치 국소 출력 오차의 **공분산 교차항, block/activation gradient 또는 관련 행동 불확실도**를 대리지표로 탐구하고, 실제 모델 품질과 독립적으로 비교한다. 이전에 안전 검사로 차단된 per-tensor 전체 최종 로짓 민감도 실험은 우회·재실행하지 않는다.
3. 모델·작업별 대표성, 지속 100+토큰 처리량, M>1 batch, 애플 Metal GPU, 프로덕션 및 독립 IID/OOD, 과거 V6 24개 출처 제외까지 검증하지 않는 한 기존 Q 최종 가설을 승인하지 않는다.
4. 연구 전용으로만 운영하며 `main`, BEGLIN XOX 운영 바이너리, 원본 BF16 모델, Tailnet `g6-0.9.0-operator.900387` immutable live에는 변경을 가하지 않는다. **NO AUTO PROMOTION**.

**현재 정리:** 설계·구현·성능 측정·감사는 완료했지만, **더 좋은 정밀도 정책**과 **전체 모델 병렬 속도 향상**을 발견한 것은 아니다. 실패 결과를 보존하는 것이 다음 가설 설계의 출발점이다.
