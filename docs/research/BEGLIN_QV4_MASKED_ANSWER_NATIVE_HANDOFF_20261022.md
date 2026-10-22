# BEGLIN Qv4 masked-answer native development handoff — 2026-10-22

Branch research/qv4-masked-answer-nll-20261022, based on commit 8c7417258a4c7037dad652ee27ffaa1c6b7823c0.

Verdict: REAL_NATIVE_5_MODE_PASS / DEVELOPMENT_SMOKE_ONLY / NO_GENERALIZATION / NO_PRODUCTION_PROMOTION.

## Motivation and method

Prior Qv3 evaluation compared full-prompt teacher-forced PPL over ten public development inputs and 187 prediction positions.
The values were F32 311.1430, Q4 643.4983, random balanced 506.7902,
v2_diag 664.2526, v3_cov 657.0903. Source-local weight/activation and
off-diagonal covariance proxies failed to dominate the strong random baseline.

We now separately measure exactly one teacher-forced next-token label likelihood
for each of eight authored two-choice public development problems. Correct
answer labels A and B each appear four times, and both candidate prefixes are
identical per problem. Input and mask SHA are frozen before Qv4 native scores.
The original native C ppl branch is preserved byte-for-byte in the isolated
patched source. Original model checkpoint and standard runtime are unchanged.

## Results
| Policy | Correct /8 | Answer-only correct token mean NLL lower | Log-prob margin higher |
|---|---:|---:|---:|
| f32 | 8/8 | 0.022690 | 9.204329 |
| q4 | 7/8 | 0.262787 | 7.389428 |
| role_random | 7/8 | 0.247457 | 8.521505 |
| v2_diag | 8/8 | 0.005342 | 8.167359 |
| v3_cov | 8/8 | 0.079681 | 7.946507 |

Old full-prompt PPL cannot be directly compared with new answer-only NLL.
Qv2 has the smallest answer-only NLL on this tiny development cohort, despite
worse previously observed whole-prompt PPL. Qv3 beats Q4 and the fixed random
baseline on answer-only NLL, while losing on previous full-prompt PPL.
This is a task-metric ranking reversal, NOT a quality generalization result.
Qv2 lower NLL than F32 is not evidence of general correctness gain or calibration.

## Evidence and audit
- Official source BF16 model revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
- Qv4 prelock SHA256: 984adbb205bc14e3ff76c1654a3143529fdd07a8ba017e445462cb3707c3677c
- All 16 native token arrays SHA-bound manifest: a6e6ab1e0b2054faabaf460fde280f19d9411ed26862c1cd1aa2b3118fa88f39
- Separate native-log/metadata audit SHA256: ae8523e7e83a28ecdd366ff33ac5b467b0faf505a0933a03544752e8823e6bac
- Public aggregate SHA256: 8a33fc28af887fd84401576453fa5a8d4cf6562ac93670b09ab432ac64c3678d
- Native C base/mixed binaries built successfully; five full 28-layer executions exit 0.
- Contract/negative suite 13/13 PASS.
- Two research source files: research/oml_acs/q_v4_masked_answer_native.py and
  research/oml_acs/q_v4_masked_answer_separate_audit.py.
- Qv4 original masked loss auditor and a separate differently implemented log
  parser recomputed loss totals and validated input/policy/binary identities.
- Private model log, token bytes and full model weights remain in owner-only
  lab_local; none are to be committed or uploaded.

## Limits and follow-up
Small easy 8-item binary development smoke; no calibrated probability,
multi-token answer, free-text generation, IID/OOD evaluation or statistical
significance claim. New task labels do not select a revised precision mask.
No per-tensor model-output causal intervention or previous V6 24 documents used.
EOE concurrent CPU load makes elapsed times unsuitable for speed comparisons.
Full parallel pthread remains OFF; current automatic quality/precision
promotion remains OFF. Main and immutable Tailnet operational release unchanged.

Next scientific step: freeze independent human-verifiable answer-only
multi-token datasets (e.g. BoolQ/ARC/CommonsenseQA) with source provenance,
disjoint train/dev/test splits and exact per-document exclusions; compare
paired NLL and task accuracy across these five fixed policies, stratified
by answer length/genre. Only when quality generalizes consider re-optimizing
a downstream-loss-aware sensitivity proxy; separately prove sustained runtime
and memory/cost Pareto constraints before manual promotion.

## 추가 독립 구현 검산 (Qv4 second auditor)

기존 native 평가기와 별개 구현한 q_v4_masked_answer_independent_audit.py는
16개 접미 답안 점수, 질문별 평균 손실/선택 정확도, 마스크 및
입력 SHA, 실제 바이너리 동일성과 학습/평가 격리를 독립 계산했다.
두 번째 출력은 이미 존재하므로 수정·재실행하지 않았고,
읽기 전용 재계산 결과 원본과 정확하게 일치했다.

- 비공개 두 번째 독립 감사 SHA-256: c34d00286791fa9e57e19bd6d0d583404d052ed05db507f043fa441d696c3029
- 추가 음성/원본 손상 검사 8/8 PASS, 전체 Qv4 테스트 21/21 PASS.
- 단, 이는 native C가 기록한 NLL과 바인딩을 별도 파서로 검증하는 수준이며,
  독립 구현 전체 모델 로짓 재생성으로 간주할 수 없다.

## 비교 해석 추가 제한 — 데이터와 채점 구간 동시 변화

Qv3의 10개 문장/187개 전체 토큰과 Qv4의 직접 작성한 8개 선택형 질문/16개 답안 후보는 **문항도 다르고 손실을 계산하는 토큰도 다르다**.
따라서 서로 다른 두 평가의 정책 순위가 달라진 사실만으로 **채점 구간의 변화가 순위 역전의 원인이라고 단정할 수 없다.**
이 가설을 검증하려면 **동일한 입력 배열과 동일한 정책/바이너리**에서 prompt-only, assistant-only NLL을 동시에 측정하고,
기존 V6 문서 제외와 사전 동결 평가를 유지해야 한다.
이번 Qv4는 독립적인 답변 전용 평가 경로가 작동한다는 실기기 증거이지, 기존 Qv3의 부정적 품질 판정을 대체하지 않는다.
