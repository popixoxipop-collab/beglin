# BEGLIN P18 — 204-position original BF16 + native A/B full-vocab 연구 핸드오프

2026-10-15 KST · 저장소 popixoxipop-collab/beglin · research/hcp-p18-external-multimetric-20261015

## 1. 이번에 실제 완료한 작업

**P18_LONG203_27L_BF16_AND_NATIVE_AB_FULL_102400_LOGITS_AUDIT_PASS_NOT_480_QUALITY**

- 프롬프트 203토큰 + 정답 1토큰, 사전 고정된 p18-mmlu-018.
- P17 원본 GoldWindowTeacher 원본과 **단 하나의 입력 제한 112→1024 이외 완전히 일치**한 P18 연구용 파생 코드로 원본 DeepSeek-V2-Lite BF16 **27개 전체 계층 / 5,291 원본 텐서** 실제 완전 forward 성공.
- 동일 203토큰 프롬프트에 대해 정책 A 및 정책 B **102,400 float32 원본 Metal logits 각각 1개** 실제 캡처, 409600 bytes×2, SHA/argmax/NLL/원본 GPU 로그/정밀도 epoch/ACK 독립 대조 통과.
- 운영 stage SHA, route SHA, 운영 PID(35481,35493) 시험 전후 불변. 새 연구용 바이너리는 별도 build_long/에 존재하고 기존 P17/P18 실행 파일을 수정하지 않음.

| 원본 실제 모델 | gold NLL (nats/token) |
|---|---:|
| Original BF16 | 2.3724285 |
| Native qNg64 policy A | 2.2368270 |
| Native qNg64 policy B | 2.1508238 |
| B−A | −0.0860032 |

**한 문항에서는 B가 유리하지만, 이 점수는 480문항 전체 비열등성 인증에 사용할 수 없는 조기 pilot이다.**

## 2. 장애 원인과 수정

초기 203토큰 양자화 원시 캡처 실험이 P16_INVALID_NATIVE_FIRST_LOGIT_CAPTURE_CONTEXT로 중단됨. P18 원시 logits 기록 함수의 plen<=1023 제한은 이미 적용됐지만, 최초 logits hook 호출부에 P17 호환성 가드 rq_plen<=112가 남아 있었음. 원본 P18 C SHA 82475145a4351b934cfccda6763112cdf25751e2ca470ac8417a16e6dc8fdb1c에서 P18 새 연구용 복제본 하나만 변경:

- 기존: (rq_plen[r] < 20 || rq_plen[r] > 112)
- 새 구현: (rq_plen[r] < 20 || rq_plen[r] >= MOE_CBATCH_MAXPOS)
- MOE_CBATCH_MAXPOS=1024, MLX MLA_L0_MAXPOS=1024.
- 복제 C 두 파일의 차이는 이 **한 가지 조건 교체뿐**임을 독립 감사 스크립트로 확인.
- P17 원본, 이전에 봉인한 128위치 및 기존 P18 첫 캡처 검증 바이너리 불변.

## 3. 원본·수정 코드 및 영수증 SHA-256

- P18 외부 질문 480개 frozen full token bank: c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df
- P18 원래 frozen protocol: 77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c
- 이 장문 203 source frozen mini input: 374c74797344d68164ee84d3be8932ec86f7b2ffcea51b7cdbe4d6c4a72cde71
- P17 GoldWindowTeacher 원본 source: 40dd5550b7759ac2760a6c374e1ddf3af35e68af637ab8d2a4f75bca8bee01d3
- P18 원본 수식 그대로 입력 cap 1024 파생 teacher: c2a631ee7d82feef83efce59087b0a144f7561b9da26db920e1201b45b689820
- P18 extended teacher trial script: acce1081b4b19ebff0fa9986f1763ecc3e0a7f5f7f0c6b6d811c85c30de82388
- BF16 teacher real raw logits .npy: 2cc6e216e6c734bc041c8897de1a126a5562988dd93d1022b960eb2484a6a783
- BF16 27-layer one-item receipt: 73091f999dafc59d53ea6a8646e2543951bca68750129d9bb7b120cb24c3d924
- BF16 원본 5291 tensor values manifest: 88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda (P17, P18 모두 일치)
- P18 원본 캡처 제한 한 개만 수정한 새 C SHA: fa8a328bedc1e56e79938460e1d4a171f0fd363535fc610183b4022e41ce0bfa
- P18 long 연구용 새 바이너리 SHA: bf8a70ec5f0e449c3e2c35aeeccc3932dbafe0a376322570d0ebd0a0cae8f81d
- 정책 A raw F32 전체 logits SHA: bdebf665b3111db7c6482cd3682697a62e3b9785d1dd4f66571300343dcf9bfb
- 정책 B raw F32 전체 logits SHA: 5bf00419150101002257c713ce91eb4d44f901bec8634da81e76c84c9d13fb02
- 정책 A/B 원본 native receipt: 5e5f27925d3fc5eaa978dd91df41dcf957a56dfa0fd85c967b47e9a2e092d66d
- **BF16+A+B 독립 F32 재채점 및 C 단일 diff 감사 영수증: af761db9964bce9d4229f2b66d1ef6b6da7887baa2c77c4c006e83e8ecbc55d2**

## 4. 누적 실제 P18 연구 통과 범위

- 최초 외부 3개 벤치마크(각 1문항, 정답 5토큰) 원본 BF16 27-layer + native A/B 10개의 실제 원시 logits. 독립 감사: 2d9bd085bcfb28871ad9d8b42f82bb9347b02e0fcb9897a6ab6baf25ce57bd7e.
- 추가 긴 문맥 203토큰 MMLU 1문항, 정답 1토큰 원본 BF16 27-layer + native A/B 2개 원시 logits.
- 누적 **4문항**, 원본 BF16 forward 4회, gold 위치 6개, native A/B 전체 logits 12개. 이는 *학습/개발 기능성 확인* 수준이며 사전등록된 480문항 품질 인증이 아님.
- 정책 A 기본 운영 유지, B 자동승격 false, 릴리스 활성화 없음.

## 5. EOE D50·XOX 경로

EOE private D50:
- /Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015/native1024_real_evidence/P18_LONG203/
  - original_p18-mmlu-018_203BF16_logits.npy
  - original_p18-mmlu-018_5291_tensors.json
  - original_p18-mmlu-018_BF16_teacher_receipt.json
  - p18-long203-native-AB-receipt.json
  - p18-long203-native-AB-worker.log
  - f6a353d99df241609a5cc35838618433.f32 (A)
  - 056ed167aebe4a878931dd1152d9dc51.f32 (B)
- code source & audit parent root: hcp_p18_extended_original_bf16_teacher_core.py, hcp_p18_first_long203_original_teacher.py, hcp_p18_long203_native_AB_full_logits_v2.py, hcp_p18_long203_three_native_and_teacher_audit.py, P18_LONG203_REAL_BF16_AND_AB_INDEPENDENT_AUDIT.json
- P18 C source: native_p18_capture_long203_buildroot/qwen_infer.c

XOX private experimental root:
- /Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-capture1024-20261015/source_long/ 및 build_long/, p18_long203_gold_logits_worker_v2, capture/ (원시 logits 전용).
- 운영 서비스/원본 P17 로컬 코드, PID 및 sealed logits 변경 금지.

GitHub commit에는 **코드, 집계 영수증, 핸드오프만 올릴 것**. 전체 480문항 토큰 배열·원시 질문/정답 텍스트, BF16/Metal 원시 logits .npy/.f32, 원본 673개 token SHA가 박힌 생성 C 소스는 GitHub 업로드 금지.

## 6. 미완료/다음

1. P18 BF16 **최대 문맥 최대 963** 실제 original teacher 테스트 및 메모리 안전성 증명. 현재 BF16 실측204 위치까지 성공, native A/B 실측963까지 성공.
2. 480문항의 BF16 진짜 원본 forward와 A/B gold-prefix 673×2 전체 F32 캡처. XOX 내부 SSD free-after-reserve 약0.15GiB이므로 source raw 누적 금지. EOE D50으로 raw SHA 검증해 청크 스트리밍하고 중간 재시작/재개 영수증 구현.
3. 0/3/5-shot과 생성 cap 8/32/128의 진짜 exact-match task accuracy, BF16 positive control, A/B 정책 전환을 제외한 warm fixed-work latency 3개 trial씩 측정.
4. frozen 480 질문 단위 층화 bootstrap 10000회 + 3개 metric Bonferroni gate. 통과해도 생산 정책 B 자동승격 금지.

**결론: P18 1024 native kernel + original BF16 204 position 기능 연구 PASS. 통계 품질 인증 P18 아직 BLOCKED.**
