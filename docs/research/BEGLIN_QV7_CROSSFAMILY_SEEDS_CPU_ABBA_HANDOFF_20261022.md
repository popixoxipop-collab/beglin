# BEGLIN Qv7 연구 핸드오프 — 교차 모델 계열, 독립 보정 분할, CPU 처리량

2026-10-22 KST. 연구 브랜치: research/qv7-cross-family-seed-pareto-20261022. 운영 자동승격, main, 원본 HF 가중치 변경 없음.

## 1. 연구 결과 판정

- 이식성 PASS: Qwen2 계열 이외의 원본 TinyLlama-1.1B-Chat (Llama 아키텍처, 22개 계층, 154개 투영)에서 원본 BEGLIN C의 4비트 및 22개 n8 + 132개 Q4 NEON 추론을 실제 실행하고 전체 어휘 로짓 생성.
- 원본 BF16 Fisher PASS: 64개 ARC-Challenge 학습 문항, 154개 원본 BF16 투영 그래디언트와 실제 그룹64 C 양자화 오차를 결합. 64문항 홀수/짝수 순위 Spearman 0.969498; 이는 보정 점수 내부 안정성이지 모델 출력 품질의 증거가 아님.
- 새로운 TinyLlama source-only MILP PASS: 정확히 22개 8비트/132개 4비트, 최적성 간격 0. 원래 TinyLlama 원본 역할별 q4_1t/q4_6t/n8_repacked 네이티브 커널 비용을 실측한 뒤 최적화. 역할 분포 v=16, k=2, o=2, up=1, down=1.
- 품질 개선 FAIL/미입증: 새로운 OpenBookQA test 48문항 개발 비교에서 TinyLlama 원본 BF16 14/48 (NLL 4.259587), BEGLIN Q4 16/48 (7.095149), BEGLIN Fisher 혼합 15/48 (7.245960). Fisher-minus-Q4 NLL +0.150811, 5,000회 쌍별 bootstrap 95% 구간 [+0.016614,+0.285615]. 동일한 실측 C 환경에서 Fisher 혼합의 품질 우위 없음.
- 최초 48개 테스트 문항 중 1개는 원래 C Q4 커널 smoke에 앞서 사용됐음. 따라서 확증용 완전 미관측 holdout이 아니며 이 전체 평가는 탐색적 개발 검증. TinyLlama 원본 BF16 자체도 정답률이 낮아 능력의 바닥효과가 큼.
- 세 독립 분할 시드 PASS: Qwen 원본 BF16 128개 소스 방향성 Fisher 값에서 시드 17/41/73 각각 겹치지 않는 64문항 두 그룹을 구성해 6개 MILP를 실행. 순위 상관 0.445/0.461/0.614, 마스크 겹침 15/15/16개 (총 28개). 시드끼리는 같은 128개 원본을 반복 사용하므로 6개 완전 독립 데이터셋이라고 주장하지 않음.

## 2. 비용과 CPU 반복 실측

실제 TinyLlama 원본 BF16 7종 텐서의 BEGLIN 네이티브 C 커널 프로파일 기반 투영 코드/스케일 저장량: 전체 Q4 544.997 MB, Fisher 혼합 565.445 MB. 커널 비용 합산 예측 Q4 34.005 ms, Fisher 42.115 ms. 이는 모델 전체 토큰 지연이 아님.

실제 EOE Apple CPU에서 순서 Q4→혼합→혼합→Q4를 두 차례 반복한 8프로세스 ABBA 실험을 수행. 프로세스마다 warmup 후 64토큰 3회 측정, 정책당 768개 실제 기록:
- Q4: native best-of-three tok/s 프로세스 중앙값 31.0565, 토큰 forward+logits p50 30.8205 ms, p95 41.1535 ms, 최대 프로세스 RSS 중앙값 3420.078 MB.
- Fisher: 23.8785 tok/s, p50 40.5110 ms, p95 59.0730 ms, RSS 중앙값 3429.122 MB.
- 개별 전체 토큰 지연은 토큰별 forward+lm_head 연산으로 산출하며 argmax overhead 제외. throughput best-of-three는 별도 지표.
- EOE 동시 작업의 load average가 사전 정한 12를 넘었으므로 성능 판정 INCONCLUSIVE_LOAD_CONTENDED. 고부하 조건 실측을 독립 낮은 부하 서버의 serving p50/p95로 오해하거나 운영 승격에 사용하지 않음.

## 3. 실험 증거·한계

- 고정 모델: TinyLlama/TinyLlama-1.1B-Chat-v1.0, 공식 HF revision fe8a4ea1ffedaf415f4da2f062534de366a451e6. model.safetensors SHA256 6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933.
- source train: pinned ARC-Challenge 64개, test: 새 OpenBookQA test 48개, 사전 토큰 ID·정답 SHA 동결. 이전 Qv6 OpenBookQA validation 데이터셋 문항은 재사용하지 않음.
- 정책 SHA256 17d11a604cfcce68fb52971a5fd34cf57df7ec36ce5e870b4b5dcaa19c38bd4c.
- 원본 64개 BF16 역전파와 154개 quantizer 방향성 민감도는 인과 단일 텐서 최종 출력 교체 실험 또는 정확한 Hessian이 아닌 근사적 source-only 목적함수.
- C Q4 및 혼합 실행은 실제 원본 모델 가중치와 학습된 오프라인 bitmask를 사용했지만, 입력별 bit 전환·Metal GPU 자동 할당·다중 사용자 요청 처리를 구현하지는 않음.
- 독립 감사는 네이티브 원본 C 실행 로깅의 수치, 샘플 SHA, 모델/바이너리 SHA와 Fisher 점수를 별도 검산한 것이지 다른 엔진에서 매 토큰 로짓을 재생성한 것은 아님.
- raw 원본 모델, 개인 보정 토큰 배열, 학습 그래디언트 배열, 원본 평가 정답 및 개별 네이티브 로그는 GitHub에 업로드하지 않음. owner-only EOE scratch에 유지. 대용량 TinyLlama 원본은 D50 연구 캐시에 보관.

## 4. 승격 게이트 및 후속 연구

엔지니어링으로는 다른 Llama 계열에서 독립 요소 4비트/n8 실행 PASS. 그러나 품질 개선과 CPU 자원 Pareto에서 Q4 대비 Fisher 혼합 우위 없음. Qv6의 Qwen2 계열 부분적 이득을 다른 계열에 일반화할 수 없고, Fisher 지역 대리지표가 전체 답변 NLL을 직접 예측한다는 가설은 이번 개발 검증에서 지지되지 않음.

다음은 더 유능한 독립 Llama/SmolLM 계열에 대한 다른 텍스트 장르·긴 답변 평가, 독립 신규 calibration cohort와 복수 seed, 조용한 CPU 독립 부하 실측, 전력/Metal 및 자동 bit-switch 비용 검증. 기존 Qv6의 긍정 결과도 유지하며 새 실패 결과를 덮어쓰지 않는다.

현재 과학적 판정: CROSS_FAMILY_NATIVE_KERNEL_PASS / SOURCE_FISHER_MILP_PASS / SIX_SEED_SPLIT_POLICY_PASS / CROSS_FAMILY_QUALITY_SUPERIORITY_NOT_ESTABLISHED / CPU_SPEED_GATE_FAILCLOSED / RUNTIME_PROMOTION_DENIED.
