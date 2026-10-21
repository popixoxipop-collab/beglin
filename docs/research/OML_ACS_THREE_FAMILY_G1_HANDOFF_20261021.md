# BEGLIN OML–ACS — 3계열 BF16 실험 G0/G1 핸드오프 (2026-10-21)

**결론:** Qwen2.5-1.5B · Mistral-7B · Phi-4-mini의 300개 동일 앵커 투영, 실제 SST-2·CARER의 600개 학습/개발 문항 추출, 3계열 leave-one-out(LOO) 개발 실험을 **실기기에서 완료**했다. 그러나 **정식 논문 재현, 독립 최종 평가, A/B/C1/C2/Q 가설 채택은 모두 미완료**이다.

저장소: `popixoxipop-collab/beglin` / 독립 브랜치 `research/oml-acs-prereg-20261021`. `main`, 생산·서빙 바이너리, 이전 V6 비채택 결과, Tailnet Release Plane 운영 릴리스는 변경하지 않는다.

## 1. 3개 실모델 검증

| 모델 | 고정 revision | 모델 BF16 계층/차원 | 앵커 활성값 | 행동 데이터 활성값 |
|---|---|---|---:|---:|
| Qwen2.5-1.5B | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | layer 14 / 1536 | 300×1536 | 600×1536 |
| Mistral-7B-v0.3 | `c170c708c41dac9275d15a8fff4eca08d52bab71` | layer 16 / 4096 | 300×4096 | 600×4096 |
| Phi-4-mini-instruct | `cfbefacb99257ffa30c83adab238a50856ac3083` | layer 16 / 3072 | 300×3072 | 600×3072 |

모든 값은 모델별 선택 블록 **최종 토큰 잔차 활성값**을 BF16 가중치 추론 중 추출하고 float32 저장한 것이다. 문항의 순서·각 텍스트 SHA·레이블의 3모델 동일성이 별도 검증됐다. raw text, label, activation NPZ는 WSL ext4 소유자 전용 `0600`이며 GitHub에 올리지 않았다.

원문 앵커 source SHA-256: `47d5ddcb37b81212e3f43a34441662062b42a038d4cc606d9d7081c31131f83b`.

Phi checkpoint 신뢰성: 두 공식 LFS SHA-256
- 첫 번째 shard: `bc703090b63eda16f639fa4de7ac54635c23105ab1da2f6ec4d3403151d38ee6`
- 두 번째 shard: `7ff79b9d2d31076bac2663393451f6530f4fc8ca49b09002116c92c373dba983`

처음 분할 다운로드는 **Xet 해시와 원시 파일 LFS SHA의 차이를 구분하지 못해 거부**되었으나, 공식 저장소의 `lfs.oid`를 확인하고 분리 검증한 뒤 이미 전송한 83개 구간으로 복구했다. 원격 코드 `trust_remote_code=False`, 유료 API/운영 환경 변경 없음. 완료 작업: 파일 복구 `job_5f0dc129ed3633d1a8c9515eb8a68245`, Phi 300앵커 `job_4938c0df55cdf26f464c36457c5f7d00`, Phi 행동 문항 600개 `job_15b043430e7bead72d0cbc41be03d67b`. 모두 종료 코드 0.

## 2. 실제 행동 데이터 — 원문 분할 보존

| 데이터셋 | 원본 커밋 | 학습 출처 | 개발 출처 | 축 |
|---|---|---|---|---|
| `stanfordnlp/sst2` | `8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb` | 원래 train | 원래 validation | 긍정·부정 |
| `dair-ai/emotion` | `cab853a1dbdf4c42c2b3ef2173804746df8825fe` | 원래 train | 원래 validation | 기쁨·분노 |

각 축은 *training* 긍정/부정 100개씩, *development* 긍정/부정 50개씩이다. 실제 문항 **600개**이고 총 **300쌍**이다. 소스 파일 digest를 기록했고 정확 중복과 300개 앵커의 동일 문장 겹침은 0건이다. 의미상 바꿔 쓴 문장까지 배제한 것은 아니며 라이선스 재배포 검토가 끝나지 않았다. **원본 test, 새로운 final IID/OOD, 이전 V6 사용 평가문서 24개는 접근하거나 미사용으로 승격하지 않았다.**

실제 문항 private SHA: `b99fb5d7e8c909e6821f555ac590b36ff52393b9d4e7a01828ee936c615bd274`.

## 3. G1 실측 — 세 계열 순환 전이 (개발용만)

두 소스 모델에서 100쌍의 실제 훈련 레이블로 각각 native 축을 만든 뒤 300개 공통 앵커로 투영하고, 원문 식 (4)의 **단순 평균**을 취했다. 목표 모델 50쌍의 development 문항은 **평가에만 사용**했다. dev 레이블을 방향 부호 전환·가중치 학습에 사용하지 않았다.

| 타깃 | 소스 둘 | SST-2 AUROC | CARER AUROC |
|---|---|---:|---:|
| Phi | Qwen + Mistral | **0.8552** | **0.7800** |
| Qwen | Mistral + Phi | **0.7872** | **0.6664** |
| Mistral | Qwen + Phi | **0.8972** | **0.8176** |

각 대상·행동축마다 시드 42의 **199회 앵커 순서 교란**, **199회 무작위 공통축**, **999회 층화 부트스트랩**을 별도로 기록했다. 셔플 대조 경험적 p값은 Phi(0.005/0.040), Qwen(0.015/0.030), Mistral(0.020/0.025)이다. 여섯 번의 비교·복수 대조에 대한 사전등록 기반 다중검정·독립 검증을 수행한 결과로 간주하면 안 된다.

가장 중요한 **부정적 발견**: 단순 평균은 여섯 항목 모두에서 두 개의 단일 소스 중 개발 AUROC가 더 좋은 쪽을 능가하지 못했다. 그 비교 대상인 **“최고 단일 소스”는 목표 dev 정답을 본 뒤 사후 선택한 *비운영 oracle***이다. source-only 정책이 해당 모델을 미리 선택할 수 있다고 해석하지 않는다.

| 타깃 | SST-2: 평균 − 단일 최고 | CARER: 평균 − 단일 최고 |
|---|---:|---:|
| Phi | −0.0012 | −0.0240 |
| Qwen | −0.0144 | −0.0028 |
| Mistral | −0.0104 | −0.0212 |

따라서 **B의 기술 경로 및 개발용 전이 신호는 관찰**되었으나 B 가설의 정식 채택은 아직 불가하다. 균등 평균의 이득도 미입증이다.

## 4. 다음 작업: C1 source-only 신뢰도·오류 상관 기반 가중치

새 연구의 차별점인 **동적 가중치**는 다음과 같이 평가해야 한다.

1. 목표 모델의 dev 또는 final 정답을 쓰지 말고, *두 소스 모델의 훈련 자료에서 k-fold OOF* 방식으로 축 신뢰도 및 **오류 동시 발생 행렬**을 구축한다. 원자료 학습/검증 출처와 fold ID·샘플 해시를 고정한다.
2. 비교군: 균등 평균, 각 소스 단독, source-only 정확도/신뢰도 가중, 공분산·상관을 반영한 **simplex 제약 가중**. source label 접근이 없는 target 모델에 적응하면서도 **target-dev 최적 가중치 oracle를 학습 규칙에 섞지 않는다**.
3. 별도 개발 자료에서 C1 가중치의 AUROC/생성 결과를 비교하며, 모델마다 누가 더 좋은지 목표 정답을 이용한 사후 선택은 금지한다.
4. Llama-3.2-3B는 접근 승인(gated)이 필요해 **기존 계획과 달리 Mistral 7B를 파일럿 세 번째 계열로 사용**했다. 이는 사전등록 모델 구성에서의 명시적인 편차이므로 원논문 재현이라고 부르지 않는다.
5. V6 이전 24개 평가 문서 원본 식별/출처 결합, 독립 near-duplicate 검사, 라이선스 검토, 별도 final IID/OOD 수집·동결을 거친 다음에만 **정식 B/C1 판정**을 논한다.
6. C2와 Q의 8/4비트 실제 BEGLIN 엔진 통합 검증, 출력 모델 결합 A의 NLL 검증은 **아직 착수하지 않은 별도 실험**으로 기록한다.

## 5. 저장·재현 경로

- 공개 코드 (GitHub 독립 브랜치): `research/oml_acs/phi_pinned_range_download.py`, `extract_anchors_phi.py`, `extract_real_behavior.py`, `evaluate_real_behavior.py`, `evaluate_three_family_loo.py`, `acs_core.py`.
- 증거 JSON: `docs/research/OML_ACS_G1_THREE_FAMILY_REAL_DEV_EVIDENCE_20261021.json`; SST-2/CARER 두 모델 중간 결과: `OML_ACS_G1_REAL_SST2_CARER_QWEN_MISTRAL_EVIDENCE_20261021.json`.
- private local: `/home/alienware-r13/.local/share/beglin/oml_acs/20261021`, 원문/3모델 앵커/3모델 600행 활성값/3회 LOO 상세 결과 모두 **0600**.
- LOO 비공개 JSON SHA256: Phi `8b2d8db543f98e6cf1292659deaab8475552265c3f70b1cb0a763d7e3809b70a`, Qwen `07dd3e50701da691b9ad9319658f5285725d56ce154d299b622718b6cf24abf1`, Mistral `b4264406055683a202c264afc9ddd87f759db401a1c1c53fa550522b79a1cdc1`.

**상태:** 3계열 기술적 G1 개발실험 마무리 / 최종 A·B·C1·C2·Q 가설 판정 없음 / 운영 승격·메인 병합 없음.
