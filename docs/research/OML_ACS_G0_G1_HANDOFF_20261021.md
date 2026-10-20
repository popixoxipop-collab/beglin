# BEGLIN OML–ACS — G0/G1 실행 핸드오프 (2026-10-21)

**연구 브랜치**: `research/oml-acs-prereg-20261021` (main 미병합, 운영 엔진/릴리스 미변경). 모든 새 평가는 *탐색·기술 검증*이며 사전등록 연구 A/B/C/Q의 성공/실패가 아니다.

## 실행 증거와 판정

| 단계 | 실제 실행 | 상태 |
|---|---|---|
| G0 수치·계약·누수방지 | EOE Python/NumPy, 12+8+9=29 단위시험 | PASS |
| G0 실제 모델 | Qwen2.5-1.5B BF16, 6 ×1536 활성값, 4앵커 방향, 6토큰 생성 | PASS (스모크) |
| G1 단일모델 통합 | 저장소 `hf_adapter.py` + `acs_core.py` + `g1_smoke.py` 소스 SHA를 WSL에서 재검증하여 CUDA 실측 | PASS (기술) |
| G1 후보 답안 확률 | 부정 리뷰 1개, P(Negative)=0.999447..., 비보정 조건 | 계산 PASS, 정답률/NLL 성능 미판정 |
| G1 같은 모델의 steering | 무개입/자체 행동축 alpha3/무작위축 alpha3의 생성 결과 동일 | 실제 행동 변화 **미입증** |
| G1 Qwen→Mistral | Qwen 12×1536 → Mistral 12×4096 매칭 앵커로 축 역투영, BF16, 실토큰 생성 | 실행 PASS, 효과 미입증 |
| G1 탐색 검출 | ACS 12/16 vs 교란 앵커 15/16 긍부정 쌍 점수 서열 | **대조보다 나쁨**, 유의성 추정 없음 |
| G1 대상 모델 생성 | baseline과 alpha3 결과 문장이 달라짐 | 코드 개입 확인, 감성 개선/정답 향상 미증명 |
| A/B/C/Q 사전등록 연구 판정 | 별도 실험자료·300 앵커·3계열 회전·최종 평가 필요 | NOT_STARTED/NO_VERDICT |

실기기: Alienware WSL, NVIDIA RTX 5070 Ti 16GB. Qwen 스모크 peak CUDA 3025.96 MB / 실험 elapsed 64.04s. Qwen→Mistral peak CUDA 11779.2 MB / 실험 elapsed 159.8s. 후자는 Mistral 모델을 12GiB GPU 메모리 상한으로 배치해 실행했으며 영구 모델 파일을 새로 다운로드하지 않았다. 모델은 하나씩 로딩했다.

## 증거 파일
- `docs/research/OML_ACS_G0_REAL_QWEN_SMOKE_20261021.json`
- `docs/research/OML_ACS_G1_SINGLE_MODEL_EVIDENCE_20261021.json`
- `docs/research/OML_ACS_G1_TWO_FAMILY_TECHNICAL_20261021.json`
- 실험 스크립트: `research/oml_acs/g1_smoke.py`, `research/oml_acs/g1_two_family_smoke.py`
- 공통 계산/어댑터/누수방지: `research/oml_acs/{acs_core,hf_adapter,manifest}.py`

## 앞으로 남은 G0 동결 항목

1. 원문의 HELM 15개 시나리오 × 20개 = 300개의 **정확한 샘플 ID, 텍스트, 순서** 또는 독립 고정한 300개 앵커 목록; 원문 재현과 방법 재구현을 구분.
2. 기존 V6에서 소비한 24개 평가 문서의 **권위 있는 해시 및 출처 영수증**. 확인 전 최종 평가 동결/재사용 금지.
3. 공식 3모델 파일럿의 모델 접근성: Llama-3.2-3B-Instruct는 gated/manual license, Phi-4-mini는 로컬 미캐시 및 모델별 hook/remote-code 검토 필요.
4. BoolQ/ARC-Challenge/CommonsenseQA의 실제 로컬 버전 고정 자료와 문서·의미 군집 격리; 별도 900/450/450/600/600 분할.
5. 표정·감성 대조 100쌍/50쌍/100쌍의 기준 데이터와 layer/probe 점수 규칙, 무작위/앵커교란/시스템 지시문 대조.
6. 3계열 leave-one-out 완수 및 실제 탐지 AUROC와 생성 steering 결과를 독립적으로 보고.

**중지/채택 원칙**: 앵커 정렬·검출·문장 변화를 정답 향상과 혼동하지 않는다. 프리레지스트레이션 결과가 없으므로 A/B/C/Q의 성공률이나 엔진 효율 개선을 주장하지 않는다. 기존 V6 비채택 결론, Draft PR #63, 실제 운영 릴리스는 변경하지 않았다.
