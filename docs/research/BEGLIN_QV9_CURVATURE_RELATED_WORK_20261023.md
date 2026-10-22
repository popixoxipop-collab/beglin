# BEGLIN Qv9 — 출력 곡률 기반 국소 정밀도 할당과 선행연구 비교

연구일 2026-10-23, 연구 브랜치 `research/qv9-ggn-robust-20261023`. 이는 문헌 비교 및 연구 정확도 경계 문서이며, Qv9 성능 우위를 선언하지 않습니다.

## 관련 선행연구 (현재 문헌)

| 작업 | 중요한 선행 개념 | Qv9과의 차이 / 경계 |
|---|---|---|
| APTQ, Guan et al., [arXiv:2402.14866](https://arxiv.org/abs/2402.14866) | Attention-aware mixed precision PTQ에 출력 곡률 및 Hessian trace 반영 | Qv9은 Qwen 원본 네 후보 출력의 상대 logit Jacobian과 실제 BEGLIN C Q4/n8 잔차 내적을 사용. 이는 APTQ 선행기술을 넘어 일반화 성공한 새 정리라는 뜻이 아님 |
| KronQ, Lee et al., [arXiv:2607.07964](https://arxiv.org/abs/2607.07964) | Kronecker-factored gradient/activation Hessian 및 mixed precision | Qv9의 조건부 4선택지 GGN과 KronQ의 Kronecker 근사는 수학적 대상이 다름; 어느 방식이 더 낫다는 주장 없음 |
| LFQ, Lee et al., [ICML 2026](https://proceedings.mlr.press/v306/lee26bi.html) | 최종 블록 전체 logit 정합, MSE 한계, 생성 품질 | Qv9은 실제 native C의 정밀도 선택 마스크와 4개 후보 조건부 분포에 한정됨. 전체 vocab logit 정합과 동등하지 않음 |
| G²PTQ, Liu et al., [arXiv:2609.31009](https://arxiv.org/abs/2609.31009) | 전역 지도와 1차/2차 정보 및 블록별 갱신 | Qv9은 frozen BF16 source 근사이고, 실제 양자화 과정에서 gradient/Hessian을 refresh하지 않음. 같은 방법으로 보지 않음 |

## Qv9이 직접 구현·검증한 범위

1. 실제 Apple Silicon BEGLIN 네이티브 Q4G64/n8 코드에서 얻은 투영 가중치 잔차에 원본 BF16 상대 정답 후보 logit Jacobian을 적용함.
2. 28/196개 8비트 승격, original CPU NEON 단위 커널 메모리·지연 및 단계별 DAG 상한을 동시에 강제함.
3. 원본 ARC train24만으로 결정하고, 또 다른 ARC diag24와 MMLU 심리학/컴퓨터과학 48문항을 독립된 SHA 잠금 데이터로 평가함.
4. 곡률 소스 목적함수보다 실제 BEGLIN 28계층 C 전체 어휘 정답 NLL 및 four-option accuracy를 과학적 판단의 우선순위로 둠.
5. 독립 감사, 로컬 1-swap 검증 및 음성 결과 보존을 제공함.

## 과학적 제약

Qv9의 `GGN`은 네 후보 A/B/C/D 분포에 조건부인 `Jᵀ (diag p − ppᵀ) J`만 추정합니다. 다른 어휘 확률과 신경망의 비선형 가중치-Hessian 두 번째 항은 빠지므로 정확한 전체 어휘 NLL Hessian이 아닙니다. 다른 로그 확률을 사용한 Qv8과의 비교 시 서로 다른 대리지표를 실제 NLL처럼 표시하지 마세요.

사전등록된 원본 source24의 상대 logit 이동 절댓값이 4보다 큰 샘플이 많으므로 국소 2차 테일러 영역 바깥이라는 직접적인 위험 신호가 있습니다. 기존 비판적 Qv5–Qv8 자료도 보존합니다.

선행연구와 BEGLIN의 연구 차별 가능성은 출력 Hessian 자체가 아니라 **원본 native Apple 비용·텐서별 비트 마스크·실제 엔진·엄격한 provenance를 하나의 계약으로 연결한 검증 체계**에 있습니다. 입력별 dynamic bit-switch 및 CPU/Metal 요청별 자동 배치는 Qv9에서 구현하지 않았으므로 고유 기능으로 주장하면 안 됩니다.

**판정:** 문헌 재현성·연구 분리 PASS, 범용 양자화 품질 / 계산량 우위 NO_VERDICT, 운영 승격 DENY.
