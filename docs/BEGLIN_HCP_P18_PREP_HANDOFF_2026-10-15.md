# BEGLIN P18 — 외부 480문항 실측 준비 인계

2026-10-15 KST | popixoxipop-collab/beglin
상태: P18_OFFLINE_PREP_QA_PASS_NOT_NATIVE
P17 모체 커밋: 9e01436670a40ed785effd03285a3aee8824fdb3
P18 연구 브랜치: research/hcp-p18-external-multimetric-20261015
정책: 운영 A 유지 / B 승격 금지 / P17 원시 자료 불변 / 운영 및 Release Plane 변경 금지

## 1. 직접 확인한 결과

- EOE runtime_identity / release_status: current = live = runtime = g6-0.9.0-operator.900350, converged=true, active_operation=null. 이 실행에서 릴리스 승격을 하지 않았음.
- XOX SSD free 20.03 GB, 예약 최소 20GB로 여유 0.03GB만 할당 가능. P18 작업은 XOX 내부 SSD 대신 EOE의 D50을 사용.
- D50 free 716.98GB, reserve를 제외한 allocatable 666.98GB.
- P17 원본 점수 재계산: 질문별 B−A NLL 평균 +0.027109, 표본 SD 0.234122. P17 단측 95% 상한 +0.143137 > +0.05로 여전히 BLOCKED. 같은 효과/분산 기준 필요 n≈284는 근삿값이지 승격 보증이 아님.
- 외부 공개 test Parquet 3종의 고정 revision 및 SHA 확인, 결정론적 SHA 우선순위와 seed 20261015로 각 160개, 총 480개 선정.
- 독립 원본-Parquet→정렬→문항-ID→정답 매핑 감사 **480/480 PASS**.
- 합성 점수를 이용한 P18 fail-closed 품질 게이트 **25/25 PASS**, exit code 0. 실제 모델 결과를 검증했다는 의미가 아님.
- P18 신규 BF16 teacher forward 0회, 네이티브 A/B 실측 0회, 생성 성능 실측 0회.

## 2. 데이터 소스

| 소스 | 전체 test | 선정 | 리비전 | Parquet SHA-256 | 라이선스 |
|---|---:|---:|---|---|---|
| MMLU (cais/mmlu) | 14,042 | 160 | c30699e8356da336a370243923dbaf21066bb9fe | 74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7 | MIT |
| ARC-Challenge (allenai/ai2_arc) | 1,172 | 160 | 210d026faf9955653af8916fad021475a3f00453 | 62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9 | CC BY-SA 4.0 |
| GSM8K (openai/gsm8k) | 1,319 | 160 | 740312add88f781978c0658806c59bc2815b9866 | ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59 | MIT |

출처: https://huggingface.co/datasets/cais/mmlu , https://huggingface.co/datasets/allenai/ai2_arc , https://huggingface.co/datasets/openai/gsm8k

공개 외부 데이터임은 확인되나, 원본 모델의 사전학습 데이터 오염 가능성은 배제할 수 없음. 원문 질문/정답 및 ARC 라이선스 적용 데이터는 GitHub에 게시하지 않음.

## 3. 파일·증거

D50 경로: /Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015

- source_manifest.json SHA: ca453fd6776f20c9ea4279e98bffb1ffd97e9f1963c1425bc874a46eaaf150a4
- candidate_bank_PRIVATE.json SHA: ea7cb77649fcdc285393234e019d56db0d7178fb7787cf23268a7c52c072bb5a (원문 포함, GitHub 비공개)
- candidate_selection_receipt.json SHA: 1e16f8d75487b7e8cc00e62fcda7511a3af4cc176ca31a9528eb046def76dc79
- candidate_independent_audit.json SHA: 06b337558c0c60935de9e8839edf17c913cf0b4b3aacbaef5dbf2147a0df30ee (480/480 PASS)
- P18_OFFLINE_PREP_QA_RECEIPT.json SHA: 22af19cbb444e9f8bc33349936ce8e82c397d93c1c2dec24d7645eabf2c3dd3f
- P18_SYNTHETIC_GATE_25_TESTS.log SHA: f732311ccfcd89ef33ac14ec3a9010ab653c1d65a927d96ba138b5fe53142a90
- p18_protocol_DRAFT.json SHA: b22ac5a313fc6ccb5b7e50c305281ba1302d805b850a8e4e1d13e133c491212e (**아직 FROZEN 아님**)
- hcp_p18_quality_gate.py SHA: dde5636c741699c567448735f888d4b8de12d612c33deb5b030bf1e6465efa9d
- test_hcp_p18_quality_gate.py SHA: fea746ef4e3b867cc745f9467c267d8f8fab1601dddacb7e5dfaafa12c8b1d85
- hcp_p18_download_sources.py SHA: 4557bb8b5bc9133b0e465ec13afe49b7bc73e316c3b8f5bae112a5b0331fa38d
- hcp_p18_sample_public.py SHA: 73451851f24a54280853989b7ea84ad86bdd60623aa9e6ea559405b27f23c5c0
- hcp_p18_independent_source_audit.py SHA: c5ce25c1b486acd0b06d78e21c22ce75f89df1d9c3e64c95c929296c07a84ccf
- hcp_p18_seal_offline_qa.py SHA: 690e5ba006a16a3b71cb347475b2eb1c6d0198e4db47bd57989ce98a140f8db3

## 4. 통계 계약 (초안, 아직 FROZEN 아님)

- 고정 480문항 = MMLU/ARC-Challenge/GSM8K 각 160개. 질문 단위, source별 층화.
- BF16 원본 / A incumbent / B challenger가 동일 프롬프트·골드·모델을 사용해야 함.
- gold token NLL: B−A 질문 평균의 one-sided upper ≤ +0.05 nats/token.
- 정답률: B−A question exact task accuracy의 one-sided lower ≥ −0.05.
- 지연시간: 같은 고정 decode workload, warm cache 정책별 3회; paired B/A geometric mean ratio one-sided upper ≤1.10. 전체 p95(B)/p95(A) ≤1.15 별도 안전선.
- 3개 신뢰구간은 10,000회 seed 20261015 질문별 층화 부트스트랩과 Bonferroni familywise alpha=0.05 적용.
- BF16 positive controls: MMLU >=0.30, ARC >=0.30, GSM8K >=0.10. 교사가 원천적으로 풀지 못하는 벤치마크로 승격 인증 불가.
- 길이 8/32/128, shots 0/3/5, 문맥 최대 4096은 아직 실기기 지원이 확인되지 않은 실험 목표.
- 증거 점수 파일은 원시 logits/실제 정답 생성/타이밍/ACK 감사를 선행해야 함. 현재 게이트 자체는 원시 배열 SHA 검증이나 독립 채점 실행을 대신하지 않음.
- 게이트 PASS라도 P18_RESEARCH_GATE_PASS_NOT_PRODUCTION이며 자동 운영 승격은 항상 false.

## 5. 정확한 재개 순서

1. 원본 DeepSeek-V2-Lite tokenizer identity 및 모델 index, P17 원본 SHA와 동일성 확인. P17처럼 AutoTokenizer의 로컬 fast 경로와 BPE prompt+answer prefix 안정성 검사.
2. 공개 dev 예시 (test와 분리) 별도 리비전/파일 SHA로 고정해 0/3/5-shot 프롬프트를 생성. 480개 모두 gold/prompt token ID, 해시, 길이 감사 후 cohort SHA 봉인. P4/P10/P15/P17와 정확 token hash 중복도 검사.
3. 네이티브 런타임 capability 감사: P17 증거는 P5 C CB128 / P9 MLX KV128. 4096 문맥 또는 128 생성 가능하다고 가정 금지. 필요한 KV/CB 확장 및 회귀가 먼저.
4. XOX internal SSD 20GB reserve를 침범하지 않도록 D50 저장 스트리밍/원격 checkpoint 접근 보증; 기존 P17 보존 자료 삭제 금지.
5. 최종 소스/정답/질문 SHA 봉인 후 **평가 시작 전에** 프로토콜 상태를 FROZEN_BEFORE_EVALUATION으로 전환하고 외부 예상 SHA를 따로 보존.
6. 480개 원본 BF16 전체 forward 및 A/B full-vocab teacher-forced logits·실제 생성·latency 3회; 소스 파일 SHA와 운영 PID/route/stage 불변 전후 비교.
7. P18 독립 원시 float 배열 감사 및 정답 채점기 재계산 성공 후에만 품질 게이트. 미충족이나 다른 release operation 발생 시 fail closed.
8. P17 정책 A 유지, B 운영 승격은 별도 인간 승인·production 계약 없이는 절대 실행하지 않음.

**최종 판정: P18의 공개 소스 480문항 선정, 독립 감사 및 게이트 합성 회귀 완료. 실제 BF16/A/B P18 품질 실측은 미진행.**
