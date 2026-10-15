# BEGLIN P18 — 독립 Task 정답률 채점 / 진짜 원시 logits 출처 검증
2026-10-15 KST · 연구 브랜치 research/hcp-p18-external-multimetric-20261015

## 상태

**P18_INDEPENDENT_TASK_AND_RAW_EVIDENCE_QA_PASS_NOT_480_QUALITY**

- 새로운 별도 전수 평가 인증은 아님. 기존 P18 480문항 테스트원본·673 gold positions과 pre-registered 마진은 변경하지 않음.
- 현재 수집된 실제 original BF16 전체 모델 forward: **4/480문항**, full-vocab gold-token logits 6/673 위치.
- native 실기기 A/B 페어 질문: **3/480문항**, gold-token 5/673 위치, A+B 실제 전체 vocab float32 벡터 **10/1,346개**.
- 실제 BF16 autoregressive full generation: 0/480. 정책 A/B 최종 과제 출력 완전 생성·독립 채점: **0/480**. Warm fixed-work matched latency three trials: **0/480**.
- 결론: P18 최종 연구 비열등성 (NLL/accuracy/latency) certification **BLOCKED**. 정책 B 운영 승격 불가, incumbent A 유지.

## 새 구현

1. `tools/hcp_p18_independent_task_scorer.py`: D50의 P17 원본 tokenizer.json과 원본 480 테스트셋/gold/gold IDs, P18 독립 토큰 감사/P17 parity SHA를 모두 확인. MMLU/ARC-Challenge는 출력 전체를 단일 선택지 문자로 요구하고, GSM8K는 정상적인 십진 숫자 전체 출력만 허용. 통화 단위·해설·복수답·숫자+문장·체인오브소트는 오답 처리. Signed/retrieved native terminal EOS 또는 정확히 cap에 도달한 종료 조건이 없는 1토큰 기술적 smoke는 절대 정답률로 인정하지 않음. **이 채점 함수는 source manifest의 진정성·native 생성 사실을 단독으로 인증하지 않는다.**
2. `tools/test_hcp_p18_independent_task_scorer.py`: positive + malformed/adversarial input 30개 케이스 PASS. 정책/프롬프트 토큰 SHA/생성 토큰 SHA/상한/답변 포맷·EOS·BOS/미완료 출력을 검사.
3. `tools/hcp_p18_real_raw_evidence_coverage.py`: 이미 복사·SHA 봉인한 original 27L 5291T BF16 NPYS 4개, 원본 GPU C worker witness 2+8건, A/B raw 409,600바이트 .f32 10개, source question/gold/prefix token SHA, logits 유한값, GOLD NLL 재계산, stage/route/PID before-after 불변성을 독립 확인. **부분 coverage는 부분 coverage로만 보고한다.**
4. `tools/test_hcp_p18_real_raw_evidence_coverage.py`: SHA 변조, 소스 누락, GPU 로그 변경, stage/PID 변조, nonfinite logits, 단위/길이 오류, 증거 공백 등 총 25개 PASS.
5. `tools/hcp_p18_seal_task_raw_qa.py`: 실제 test 30+25 재실행, source SHA 봉인 및 사전등록 `hcp_p18_quality_gate.py`가 불완전한 480문항 증거를 거부함을 증명.

## Source authority SHA

| 자료 | SHA-256 |
|---|---|
| Original frozen test/token bank 480 | `c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df` |
| Frozen scorer cohort 480 | `19e7080b7208b8c7dd6fd06a656bb400dd2a9f72ca5f5674a80f5cc9a654cf65` |
| Frozen statistical protocol | `77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c` |
| Original P17 tokenizer.json | `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669` |
| 3-source real BF16+native A/B raw audit | `2d9bd085bcfb28871ad9d8b42f82bb9347b02e0fcb9897a6ab6baf25ce57bd7e` |
| NEW real raw logits coverage | `02d440e8a73603b230a7c39f7d2ea65ae938414f616d25dbbd1c0f8034e47d10` |
| NEW 55/55 QA final receipt | `98801d5e675f115272af4fe63c60ba6b10bce230fb2d7b89eb018a5ce3670c5a` |
| NEW QA raw test log | `c5038388f44c67b79b326f8a08f527647e657dfb2366140c14848fe34004b611` |
| NEW task scorer source | `588a7b03f1f931e6a79a0358dc2bc85ef8cc6818e3235ef53a8113264bc6e28f` |
| NEW raw audit source | `239e6b476c9a163c68063283280a7e3b719bf55563c905eb7f44531f76f59c55` |

## 실제 관측 성능 — 3개 문항 *pilot* 전용

| 데이터 | BF16 gold NLL | A gold NLL | B gold NLL | B−A |
|---|---:|---:|---:|---:|
| MMLU 1문항 (1 gold token) | 1.017820 | 0.515175 | 0.630186 | +0.115011 |
| ARC-Challenge 1문항 (1 gold token) | 0.331265 | 1.272472 | 1.319196 | +0.046724 |
| GSM8K 1문항 (3 gold tokens) | 3.818793 | 3.753863 | 3.656252 | −0.097610 |

세 문항 질문별 단순 평균 B−A = +0.021375 nats. **대표성·확률적 비열등성 근거 아님.**
추가 BF16-only long pilot: p18-mmlu-018 203+1 입력, BF16 gold NLL 2.3724285, 27L+5291T PASS, A/B forced raw capture 미완료.

## 다음 blocker/구현

1. P18 capture binary의 `P16_INVALID_NATIVE_FIRST_LOGIT_CAPTURE_CONTEXT`는 첫 203토큰의 native raw capture를 차단. 기존 P17 보안 가드 우회/삭제 금지. **문제 라인의 preimage/semantic guard를 확인하여 P18 연구용 only, 원본 frozen 673 hash allowlist에 제한된 별도 인가 경로**를 구현하고 negative tests 및 original output ACK 검증 필요. 107토큰 capture는 작동.
2. XOX 내부 SSD free 약20.15GB, 20GB reserve 대비 여유 약0.15GB. 480문항 전체 원시 BF16+A+B f32 최소 826,982,400바이트를 XOX에 쌓을 수 없다. 질문 단위 D50 chunk streaming·source SHA/atomic receipts·readback 후 연구 전용 XOX 파일 정리, 비가역 삭제 가드 필요.
3. BF16 전체 480문항 및 max963 실제 원본 긴 문맥, A/B 원시 673×2 logits, 480 task complete generation BF16/A/B, 정밀도 전환 시간 제외 3회 warm fixed-work latency, 독립 감사 후 frozen 10,000× cluster bootstrap/Bonferroni gate 실행.
4. Gate PASS라 해도 정책 B 자동/운영 승격은 false 유지. 현 운영 Operator current/live/runtime = g6-0.9.0-operator.900350, converged=true 및 original stage+route·PID 불변 확인된 상태.

## 저장 장소 및 공유 경계

- EOE D50 격리: `/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015`
- 원문 문항·참조 답·Parquet·token IDs·GPU .f32/교사 NPY·원본 로그는 D50 private 보관. GitHub commit 금지.
- GitHub 연구 브랜치에 올릴 것: QA 코드/테스트, 집계 영수증, 위 핸드오프 문서만.
- 원본 P17 영수증/코드, XOX 운영 프로세스/라우트, Operator 릴리스는 변경하지 않음.

**판정: P18 independent task and real raw evidence verifier PASS, but 480-question quality gate BLOCKED.**
