# JE01 — 원본 DeepSeek-V2-Lite 256개 전체 정답 구간 NLL 및 실제 Expert 기준선 실증

**실행:** 2026-11-18 KST, XOX M1 Max 격리 연구 디렉터리. **상태:** 원본 pretrained AF 모델 실험 PASS / Bayesian 정밀도 대안 품질 HOLD / 운영 변경 없음.

## 핵심 실측

| 실험 | 실제 관측 |
|---|---|
| 원본 체크포인트 | BEGLIN DeepSeek-V2-Lite AF 5개 파일, 합계 9,828,513,716 bytes, 소스 불변 |
| 네이티브 런타임 | 384위치, 1슬롯, Metal/MLX, 27층 / 실제 모델 269개 AF 바인딩 |
| 원본 데이터 분리 | WikiText train 256개 보정, 독립 WikiText128 / HellaSwag OOD128 봉인 유지 |
| 실제 정답 구간 | **256/256 문장, 11,024/11,024 토큰** teacher forcing PASS |
| 전체 정답 토큰 가중 평균 NLL | **2.460356537197240** |
| 20,000회 문장 재표집 95% 구간 | **2.365813040318407–2.551722621915675** |
| 각 문장의 *첫* 정답 토큰 NLL 별도 측정 | 256건 평균 **3.201974851245297**; 전체 구간 NLL과 구별 |
| 모든 첫 토큰 점수 선행 원장 교차 검증 | 허용오차 1e-5 이하, PASS |
| GPU 실행 메모리 최소 시스템 여유 | **45%**, 자체 중단 임계 35% 상회 |
| 기존 운영 경로 / worker PID | `beglin-candidate-shared-up-l3-n6`, **35481 / 35493**, 변경 없음 |

처음에는 원본 384 B1 격리 실행기에서 **16개 실제 짧은 WikiText 보정 입력의 102,400개 float32 로그잇**을 기존 32위치 원본 런타임과 직접 비교했고, **16×102400 = 1,638,400개 로그잇이 바이트 단위로 완전 동일**했다. 이후 실제 보정 입력256개, 원본 라우팅26층×256개=6,656건과 전체 Expert 확률425,984개를 SHA 봉인했다.

전체 정답 구간 실험은 사전 봉인된 원본 WikiText train 텍스트/토크나이저의 목표 토큰 ID를 사용했다. 각 문장에서 `BOS + prompt + gold[:-1]`을 원본 GPU에 순차 입력하면서 매 정답 토큰에 대해 전체 102,400개 로그잇의 `logsumexp - logit_gold`를 계산했다. 입력 최장 317위치로 384 한도 내에 있었으며 16묶음의 실제 GPU 종료 코드 0과 각 묶음의 SHA, 이전 첫 정답 토큰 NLL이 검증됐다. 모델의 학습이나 weight 수정은 전혀 수행하지 않았다.

**독립 감사:** 16개 원본 C/MLX GPU 바이너리 결과를 별도 Python 프로그램이 재개봉하여 11,024개 `(문장ID, 정답 토큰 위치/ID, NLL)`과 보정 256개 입력의 실제 정답 토큰 원장, 파일 SHA를 비교하고 **동일 평균 2.460356537197240**를 재산출했다. 20,000회 부트스트랩은 256개 **문장**을 재표본 단위로 사용했으며, 위 신뢰구간은 **원본 모델의 train 보정 NLL에 대한 불확실성**이지 다른 정밀도 배분과의 성능 차이 신뢰구간이 아니다.

## Joint Expert 베이지안 옵티마이저 실증 게이트

원본 256개 입력에서 **Expert별 최종 입력 위치의 실제 라우팅 빈도**를 기반으로 8개 (layer, Expert)를 고정했다. 선택: (21,61,97회), (24,26,91회), (13,28,87회), (20,17,86회), (12,13,81회), (24,60,81회), (14,63,80회), (13,22,79회). 모두 독립 보정 관측 32회 이상 조건을 통과한다.

독립된 전후 절반 128/128 라우팅 분석에서는 **28 Expert 쌍 중 17쌍**에 Holm-보정 유의한 공동 선택 연관성이 나왔다. 이는 *라우팅 지표의 공동 발생* 증거일 뿐, Expert 가중치 정밀도 변화에 따른 NLL 효과의 공분산은 **미측정**이다. 상위8 슬롯을 선정할 당시 일부 16개 보정 문장 첫 정답은 앞선 탐색에서 이미 공개됐으므로 '256개 완전 미열람 라벨에서 선정'이라는 엄격한 주장도 하지 않는다.

원본 AF 바이너리의 **실제 packed weights + FP32 scales/biases** 오프셋을 역산하여 선택 Expert 한 개의 gate/up/down 비용 **5,406,720 bytes**, 선택 8개 합계 **43,253,760 bytes**를 확인했다. 여기서의 현물 기준은 **uniform 4bit AF**, 이전 BF16 실험의 'uniform6보다 혼합4/8 proxy MSE가 높았다'는 별도 소규모/합성입력 근거와 동일한 실험이 아니다.

**남은 필수 게이트:** 2/6/8bit 대체 Expert의 *실제로 생성한* mixed-precision 압축 아티팩트와 SHA/정확한 저장 바이트 비용 없음. 따라서 Bayesian NIG posterior에 넣을 **실제 per-(Expert,tier) paired gold NLL 변화량도 없음**, 동일 현물 압축 바이트의 비교도 불가. Bayesian > uniform4 성능 개선, 독립 WikiText128 및 HellaSwag OOD128 품질, 20,000회 **paired gain** bootstrap 및 실제 latency 영향은 모두 **NOT_MEASURED**. 운영 승격·PR 병합 불가.

## 연구 증거와 재현 경로

- [원본 전체 정답 구간 256건 실제 모델 NLL](JE01_ORIGINAL_AF_REAL256_FULL_TARGET_GOLD_NLL_20261118.json) — 원장 SHA `f71a4c4397cbadf938c7a9631ba8306547b31e33028dd9a54b556a1a758e3f41`
- [11,024토큰 독립 원장 재검산 + 20,000회 기준선 bootstrap](JE01_ORIGINAL_AF_REAL256_FULL_TARGET_INDEPENDENT_AUDIT_20261118.json) — 원장 SHA `d00a56221a12a7456b39321f870e9d4a119f3425215a78142cd36c78b7eba2fa`
- [실제 8개 Expert 저장 바이트 원장](JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json) — SHA `f47bbccbc3878a70dccb9f4b7f5f7f5bd43087aeec58c948924ef85bb428ca4c`
- [대체 정밀도 결과 **이전** 8개 슬롯/바이트/대조 실험 조건 고정](JE01_ORIGINAL_AF_EIGHT_EXPERT_PREALTERNATIVE_PILOT_LOCK_20261118.json), 과거 calibration16 첫 정답 공개 사실 명시
- [실제 원본 라우팅256 기록](JE01_ACTUAL_XOX_ORIGINAL_AF_REAL256_ROUTER_FINAL_20261118.json), [분할 교차검증·공동 라우팅 감사](JE01_ACTUAL_ORIGINAL_256_FULL64_ROUTER_CROSSFIT_COVARIANCE_20261118.json)
- 실제 C/MLX 샌드박스 `tools/je01_xox_original256_fulltarget_teacherforced_nll.py`, 별도 재감사 `tools/je01_original256_fulltarget_independent_audit.py`; XOX 연구 디렉터리 `/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original_fulltarget_teacherforce_20261118_v2` (16개 원본 결과와 256개 입력 보존)

**최종:** `SOURCE_SHA_PASS / NATIVE_384_FORWARD_PASS / REAL_ROUTER_256_PASS / ACTUAL_256_FULL_GOLD_NLL_PASS / INDEPENDENT_11024_RECORD_AUDIT_PASS / EXACT_BASELINE_8_EXPERT_BYTES_PASS / BAYESIAN_MATCHED_BYTE_ALTERNATIVE_HOLD / NO_PRODUCTION_PROMOTION`.
