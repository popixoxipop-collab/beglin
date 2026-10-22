# BEGLIN Qv13 — 바이트 예산이 같은 국소 정밀도 할당 검증 계획 (초안)

**일자:** 2026-10-23 KST
**분리 브랜치:** `research/qv13-byte-budget-20261023`
**근거:** Qv12 7조건×16문항 실기기 112 NLL과 독립 감사; 원본 Q4/n8 packed-weight 비용.
**현재 판정:** DESIGN_DRAFT + QV12_EXPLORATORY_COST_PARETO; Qv13 신규 CPU/GPU 추론, 28n8 평가, 신규 holdout 평가는 아직 수행하지 않음.

## 1. 왜 바이트 예산이 핵심인가

Qv12에서 down_proj:10 한 개를 n8로 승격할 때의 packed-weight 추가 비용은 **6,881,280 bytes**, v_proj:20 또는 k_proj:0 한 개는 각각 **196,608 bytes**였다. 따라서 1n8/2n8 개수만 맞춘 비교는 공정한 메모리 예산 비교가 아니다.

- 전체 Q4: 736,985,088 projection packed bytes; 7/16; NLL 1.691992.
- `v_proj:20`: +196,608 bytes; 7/16; NLL 1.520034.
- `v_proj:20+k_proj:0`: +393,216 bytes; 7/16; NLL 1.478525.
- `down_proj:10`: +6,881,280 bytes; 7/16; NLL 1.554375.
- `down_proj:10+v_proj:20`: +7,077,888 bytes; 7/16; NLL 1.398692.

이 지표는 **원본 source shape의 packed-weight + scale 저장비용**이지 peak RSS·지속 추론 지연의 실측이 아니다.

## 2. Qv12에서 허용되는 주장

Qv12 표본 평균에서 cost/NLL Pareto frontier는 `Q4 → v:20 → v:20+k:0 → down:10+v:20`이다. 이 경계는 **기존 16문항에서 계산한 탐색적 경계**이며 새 할당기가 이를 독립 성능으로 재확인한 적은 없다.

Qv12 내부 16개 문항의 NLL을 직접 비교하면 v+k가 down 단독보다 평균 0.075850 좋아도, 개선한 지문은 7개/16개이고 재표집 95% CI는 [-0.1714, +0.3972]이다. **지문별 불확실성과 데이터 의존이 크므로 선정 규칙으로 확정하지 않는다.**

Qv12 전 조건은 7/16 정답이고 맞힌 정확한 문항도 같았다. BF16/MPS 11/16은 다른 산술 경로여서 BEGLIN 네이티브 Q4와 공정한 양자화 ablation으로 환산할 수 없다.

## 3. 연구 가설, 분리 및 대조군

H1: **같은 실제 추가 packed bytes**에서 검증용 소스 집합으로 선택한 n8 텐서는, 원래 고정 28n8 정책과 동일 예산 무작위 할당보다 새 평가 지문의 평균 정답 NLL을 유의하게 낮춘다.

H2: 여러 텐서의 정밀도를 함께 선택할 때 손실 상호작용을 추정하는 접근은 단독 효과/바이트 greedy보다 추가적인 개선을 준다. 단 Qv12 세 pair의 구간이 0을 포함하므로 상호작용항을 0으로 축소하는 기준 모델을 필수로 둔다.

H3: 실제 요청별 정밀도 선택은 **사전에 학습·동결한 정책으로 네이티브 커널에서 mask를 실제 전환하고** 비용까지 측정할 때만 검증한다. 정적인 mask 파일을 여러 개 만들어 놓은 것은 동적 실행이 아니다.

대조군: 원본 all Q4, 원본 28n8, 동일 바이트 제약의 seed 고정 무작위 할당 여러 번, 개발 코호트의 단독 NLL 개선/바이트 greedy, 후보 pair-aware 방법. 28n8 **개수 일치** 비교와 **바이트 일치** 비교를 별도 보고한다.

## 4. 평가 단계와 자료 누출 차단

1. **G0 — 소스 봉인/원가 장부.** Qv12 감사 hash·커널·HF revision·196개 projection마다 실제 n8-Q4 packed bytes를 고정한다. Qv11/Qv12 사용 문항 ID 제외 집합을 생성하고 SHA를 고정한다.
2. **G1 — 개발 데이터에서 정책 학습.** RACE train/validation에서 분리된 문항만으로 후보 텐서와 상호작용 추정을 개발한다. Qv12 평가 16문항을 G1 마스크 선정이나 하이퍼파라미터 변경에 사용하지 않는다. 선택은 holdout 전에 동결.
3. **G2 — 예산 일치 네이티브 실행.** 여러 바이트 예산 및 원래 28n8에 해당하는 `B_28`에서 실제 original C Q4/n8 kernel을 사용한다. role/mask cardinality assert 이외 모델 산술 변경 금지. budget별 유효한 candidate만 평가.
4. **G3 — 신설 holdout.** 이전 평가와 겹치지 않는 RACE middle/test **최소 128개 지문**과 난도 변화용 high/test **최소 128개 지문**을 분리 봉인한다. 질문·지문·4지 보기 무자름, single-token A/B/C/D와 동일 prompt/label SHA 적용. 최종자료 열람 뒤 mask 재선정 금지.
5. **G4 — 독립 비용/성능.** 실제 CPU peak RSS, cold/warm p50/p95, steady tok/s, 정밀도 로딩·스위칭 시간 및 가능한 Metal kernel을 **별도** 확인한다. BF16 PyTorch/MPS와 native widened-F32의 차이를 기록한다.
6. **G5 — 논문 판정.** passage-paired 10,000 bootstrap; 다중 후보 주 비교 Holm 보정. NLL 주 지표, multiple-choice conditional NLL·정답 margin·정확도/비열등성·RSS·지연을 함께 보고. 표본 부족이나 불확실성은 INCONCLUSIVE로 둔다.

## 5. 승인 전 중단 조건

- 모든 196 projection의 packed byte 비용과 Q4/n8 실제 레이블/커널/원본 BF16 source identity가 확정되지 않으면 비교 금지.
- Qv12 테스트 지문이나 그 정답 NLL을 정책 학습에 다시 넣으면 confirmatory 지위 박탈.
- 역할별 크기가 다르므로 28n8 개수만 같은 정책을 비용 일치라고 주장 금지.
- full-vocab correct-token NLL만 낮아지고 **multiple-choice conditional NLL, 정답 margin, 정확도가 유지 또는 악화**된다면 '문제 해결 능력 개선'으로 해석 금지.
- 모델 체크포인트·네이티브 커널·main·immutable Tailnet 릴리스 무변경. 원본 로그, 프롬프트, weights 등은 owner-only `lab_local`에만 보관.

## 6. 지금 생성한 산출물

- `configs/research/BEGLIN_QV13_BYTE_BUDGET_PRELOCK_DRAFT_20261023.json`: **아직 봉인/등록되지 않은 설계 초안**
- `tools/qv12_exploratory_pareto.py`: 공개 Qv12 원가/감사 JSON의 해시를 검증해 탐색적 Pareto를 재계산, 선택적으로 owner-only 문항별 NLL로 진단 구간 재표집
- `docs/research/BEGLIN_QV13_QV12_COST_PARETO_EXPLORATORY_20261023.json`: 재현 산출물 (**재선정 실험 근거 아님**)

**G0 이후 실제 28n8, 신규 holdout, 원본 C 새 실행과 메모리/속도 측정은 별도 수행 필요.**
