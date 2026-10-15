# BEGLIN P18 — 실제 BF16 + A/B 원시 logits 3개 외부 평가셋 검증 인계
2026-10-15 KST · 연구 브랜치 research/hcp-p18-external-multimetric-20261015

## 1. 현재 판정

**P18_THREE_SOURCE_REAL_BF16_NATIVE_AB_GOLD_LOGITS_AUDITED_NOT_480_QUALITY**

- 실행 및 독립감사 완료: 원본 DeepSeek-V2-Lite BF16 27계층·원본 5,291텐서 **3문항 실제 전체 모델 forward**, MLX Metal 기반 정책 A/B 전체 102,400 vocab logits **10개 벡터** (3문항의 gold positions 1+1+3=5 × A/B).
- 미완료: 외부 480문항 전수 BF16·A/B, 673개 gold-position의 2× A/B 원시 logits 및 독립 감사, 8/32/128토큰 생성 정답률, 정밀도 전환 제외 고정 workload warm latency, 480문항 부트스트랩.
- **P17 운영 A 유지, B 승격 금지**. P18 연구 3문항 점수로 통계 비열등성 주장 금지.
- 본 작업에서 Operator 릴리스 배포/교체/활성화 없음. 확인 시점 current/live/runtime: g6-0.9.0-operator.900350, converged=true, active_operation=null.
- 원래 XOX 운영 stage binary SHA-256 8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9, route SHA c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47, 운영 PID 35481·35493 불변.

## 2. 실제 원본 모델의 질문별 관측

| 사전 봉인된 문항 | 프롬프트 / 정답 토큰 | Original BF16 gold NLL | 정책 A NLL | 정책 B NLL | B−A |
|---|---:|---:|---:|---:|---:|
| p18-mmlu-000 | 107 / 1 | 1.0178199 | 0.5151754 | 0.6301865 | +0.1150111 |
| p18-arc_challenge-000 | 57 / 1 | 0.3312650 | 1.2724723 | 1.3191959 | +0.0467236 |
| p18-gsm8k-000 | 59 / 3 | 3.8187934 | 3.7538625 | 3.6562524 | −0.0976101 |

질문별 B−A 차이를 동등 가중한 3문항 평균 **+0.0213749 nats/token**. 이 결과는 편의적 대표성 없는 *실기기 pilot*, 정식 사전등록 480문항 판정에 사용 금지. B가 MMLU/ARC에서 불리하고 GSM8K에서 유리함. 단일 토큰 정답률과 다문항 task accuracy는 다르므로 합산해서 표현하지 않음.

## 3. 권위 SHA와 증거

- **480문항 원본 토큰 은행 SHA:** c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df
- **원래 480 사전등록 프로토콜 SHA:** 77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c
- ARC/GSM 사전 선택한 2문항 입력 SHA: 7ffc9bd15eba15ff853c0d6dfd4e16ef204e6992b340a18a7e15ea5b0eb12979
- P18 673/673 외부 골드 접두사 native allowlist C source SHA: 82475145a4351b934cfccda6763112cdf25751e2ca470ac8417a16e6dc8fdb1c
- P18 exact MLX source SHA: a0122c8b2010573fb3b20b1c58ab4a3abb26b73602eddf92dd008dda7892f82b
- 원본 native A/B capture binary SHA: ed2af527c1291567b0e9e67a9940cb5f13b298d4a645d11742ff1cc34248bc6e
- P18 캡처 673 해시 독립 소스 감사: 9a419cf124c58ed3eff10e36567755c4469623837d1de1715d9e81b69c9f2a5f (10/10 보호 조건, P17 기존 접두사31개 제외).
- **MMLU BF16/A/B 실제 원시 logits 독립 감사:** 3182a319d3bc370aded568895be16fe91cc5bc16be47e887c8edcbfc81d8e055
- **ARC/GSM 실제 8개 원시 F32 벡터 native receipt:** d84ccb4a63bb52fa6819f2a50fc3df0dcc1227154d8473c81f9616adb38d37c5
- **3소스 × 진짜 BF16/A/B 독립 감사:** 2d9bd085bcfb28871ad9d8b42f82bb9347b02e0fcb9897a6ab6baf25ce57bd7e
- P17 original 5291 tensor value SHA manifest: 88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda. P18 MMLU/ARC/GSM 전부 동일 SHA 확인.
- ARC BF16 원본 raw logits SHA: ffb9d59f8b9739274498c2085d679ec5d2fb4137a4a7a482705f6761da83d490
- GSM8K BF16 원본 raw logits SHA: ba5423d1728802ea189af1ba0ef5ebceb64aa5c9f17c6dea96c924f1b5c7ead9
- 원본 GPU worker log ARC/GSM SHA: 3a2d5b61dbb4144a4adddcfbd3120e7625f2c6a0eea593bccec4266112edc6ce
- ARC/GSM 8 raw F32 files 각각 409,600바이트 D50로 SHA 재읽기 8/8 PASS. 파일별 SHA는 저장된 native 원본 receipt에 있으며, 원문 질문/토큰 배열·float32 원시 logits는 **GitHub에 올리지 말 것**.

## 4. 주소·인계 구조

EOE D50: /Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015

- private: p18_multishot_frozen_PRIVATE.json, p18_gate_cohort_PRIVATE.json, p18_ARC_GSM_two_pilot_PRIVATE.json, native1024_real_evidence/** (GPU 로그·교사 NPY·원시 logits), native_p18_capture_buildroot/qwen_infer.c (673 해시 포함).
- 공개 가능한 연구용 코드: hcp_p18_generate_native_capture.py, hcp_p18_native_observer_source_audit.py, hcp_p18_original_bf16_teacher_one.py, hcp_p18_multisource_original_teacher.py, hcp_p18_three_source_real_full_logit_audit.py, hcp_p18_one_real_triple_full_logit_audit.py, hcp_p18_ARC_GSM_8native_AB_logits.py.
- 집계 영수증: P18_NATIVE_OBSERVER_SOURCE_INDEPENDENT_AUDIT.json, P18_ONE_REAL_BF16_A_B_INDEPENDENT_LOGIT_AUDIT.json, P18_THREE_SOURCE_REAL_BF16_A_B_INDEPENDENT_AUDIT.json.
- XOX isolated research: /Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-capture1024-20261015; 기존 원본 P17 scratch 및 stage/route는 유지.

## 5. 다음에 반드시 구분할 차단 조건

1. 기존 P17 BF16 GoldWindowTeacher 입력 상한112토큰: P18 전체 문맥 최대963에 직접 쓸 수 없음. P17 원본 교사 코드 변경 없이 P18 연구용 확장·기준112 parity·963 boundary/RSS 감시가 필요.
2. XOX 내부 SSD 보호 20GiB, 가용 ~0.25GiB 수준이므로 480×원시 logits는 무조건 **D50 chunked SHA verified streaming**, XOX에 누적 불가. 480 question / 673 gold prefix × 2 × 409,600 = 551,321,600 bytes A/B 추가, BF16 원시 275,660,800 bytes 추가. 총 최소 약827MB 원시 logits, plus logs/manifests.
3. frozen 평가에서 task-accuracy 0/3/5-shot 및 8/32/128-generate-cap bins, BF16 positive control, 고정 workload latency 3 trials/response, 순서 교차, 독립 raw audit, 10,000 bootstrap, family-wise Bonferroni alpha를 모두 충족해야 품질 gate PASS 검토 가능.
4. 운영 B 승격은 품질 PASS여도 자동 금지; 현존 운영 정책 A 유지.

**현재: 진짜 첫 3 benchmark source 원본 logits 교차검증 PASS, but 480-question P18 noninferiority still BLOCKED.**
