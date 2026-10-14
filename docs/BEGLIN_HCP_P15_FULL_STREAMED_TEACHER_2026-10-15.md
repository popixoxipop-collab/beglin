# Beglin HCP P15 — 전체 원본 BF16 교사 스트리밍 추론 완료

작성일: 2026-10-15 KST
연구 브랜치: research/hcp-p15-streamed-full-teacher-20261015
P14 기준 커밋: a73d84d45f036e120fd74229f0d93f82e7d36226
추론 전 소스·입력 사전등록 커밋: 1cd75c4668b700f3b1d6a2b008a64a02f2048366

## 1. 이번에 실제로 해결한 것

**P14까지 남아 있던 전체 BF16 교사 forward 실행을 P15에서 완료했다.** 원본 DeepSeek-V2-Lite의 27개 decoder layer와 5,291개 checkpoint tensor를 모두 읽고, 마지막 정규화와 LM head까지 계산해 입력별 다음 토큰에 대한 102,400차원 전체 vocabulary logits를 실제 생성했다.

부분 행렬 투영(P12), 양자화 패킹 SHA(P13), safetensors 헤더 분석(P14)으로 대신한 결과가 아니다. 실제 전체 transformer forward이며, 다만 짧은 두 입력의 마지막 위치에 대한 다음 토큰 분포를 측정한 범위다. 다중 토큰 문장 생성, 긴 문맥, perplexity, 실제 업무 정답률, 네이티브 양자화 정책 A/B 품질 비열등성은 이 결과로 인증하지 않는다.

메모리에 모델 전체를 상주시킨 것이 아니라 **원본 계층 하나를 로딩 → 실행 → 해제**한 뒤 다음 계층으로 진행했다. GPU나 운영 워커를 사용하지 않고 CPU 스레드 2개로 실행했다. 모델을 추가 양자화하거나 전문가 선택·attention을 근사하지 않았다.

## 2. 실제 실행 결과

| 실행 | 입력 길이 | 원본 가중치 | CPU 연산 | 실제 전체 forward | 최대 프로세스 RSS | 본문 추론 시간 |
|---|---:|---|---|---|---:|---:|
| prompt0_bf16 | 7토큰 | BF16 | BF16 | 27층·5,291텐서·102,400 logits | 2.48GiB | 78.04초 |
| prompt0_fp32 | 7토큰, 위와 동일 | 같은 BF16 값을 정확히 FP32로 확장 | FP32 | 27층·5,291텐서·102,400 logits | 2.79GiB | 43.31초 |
| prompt1_bf16 | 9토큰, 별도 입력 | BF16 | BF16 | 27층·5,291텐서·102,400 logits | 3.15GiB | 37.94초 |

모두 Tailnet 자원 예약 작업으로 실행해 **3/3 SUCCEEDED · exit_code=0**을 확인했다. 모든 logits는 finite이며, 총 307,200개의 logit 값을 저장했다. 한 번에 상주한 계층 파라미터 최대값은 BF16 1,169,695,744 bytes(약 1.0894GiB), FP32 2,339,391,488 bytes였다. 전체 프로세스 측정에는 로더·임시 버퍼·파이썬 런타임도 포함된다.

세 실행 모두 관측된 새로운 host swap 증가는 0 bytes. 6GiB 프로세스 RSS, 8GiB host available RAM, 20GB disk reserve, 추가 swap 1GiB 초과 시 자기 프로세스만 종료하는 watchdog을 사용했다. 반복적인 OS 캐시 상태와 입력·연산 모드가 달라 위 실행 시간을 가속 벤치마크로 해석하면 안 된다.

입력은 실제 업무 품질 벤치마크가 아닌 사전고정 engineering probe 두 개다. FROZEN_INPUTS.json에 텍스트와 토큰열·SHA를 기록하고 모델 실행 전에 GitHub에 봉인했다. 최상위 다음 토큰에는 공백이나 단어 일부가 포함될 수 있으며, 이것을 문제 정답률로 계산하지 않았다.

## 3. 수치적으로 무엇을 검증했는가

체크포인트에 동봉된 원본 modeling_deepseek.py에서 지정된 18개 torch 함수·클래스를 AST로 읽어 **본문 수정 없이** 재사용했다. 원본 attention의 YaRN mscale_all_dim, rotary permutation, router softmax/top-k, shared/routed experts와 residual을 그대로 유지했다. 최신 Transformers 모델로 자동 대체하거나 remote code를 다운로드해서 실행하지 않았다.

메모리에는 meta decoder layer를 만든 뒤 원본 이름·형상의 파라미터 전체를 strict assignment했다. 각 계층의 누락·추가 파라미터를 검사하고 원본 BF16 값을 연산 dtype으로 변환하기 전에 SHA를 계산했다. FP32 모드는 추가 학습 또는 재양자화가 아니라 원본 BF16 값의 정확한 확장이다.

원본 계산 클래스로 구성한 작은 3층 모델에서 상주 방식과 스트리밍 방식의 **전체 vocabulary logits를 같은 dtype 안에서 비트 단위로 대조**했다. BF16와 FP32 모두 완전히 일치했다. dense layer, MoE layer, 1토큰 입력, 반복 실행, 누락된 텐서·변조된 source·잘못된 dtype/형상 등을 포함해 **18/18 수치·로더 시험 PASS**.

EOE에서는 실제 저장된 .npy 전체를 다시 읽어 각 102,400개 값, shape/dtype, finite 여부, 통계·argmax, 27층 실행 순서, 5,291개 원본 텐서 해시 목록, 원본 모델 인덱스와 사전 입력 정합성을 독립 확인했다. 3개 출력 모두 PASS. 이 감사는 두 번째 전체 16B 상주 모델 실행을 대신한 것은 아니며, 작은 모델의 상주/스트리밍 동등성과 실제 전체 실행의 출처·배열 검증을 결합한 것이다.

추가로 NaN 삽입 후 해시 재계산, vocabulary 절단, 부분 계층의 전체 실행 가장, 가중치 누락, 거짓 품질 인증 등 **12/12 증거 변조 방어 시험 PASS**. 이번에 새로 실행한 시험은 18+12=30개다. P14의 과거 213개 시험을 이번에 전부 다시 실행했다고 주장하지 않는다.

## 4. 중요한 관측 — 연산 dtype도 교사 정의의 일부

동일한 7토큰 입력, 동일한 원본 BF16 가중치(5,291개 해시 전수 동일)에서도 BF16 연산과 FP32 연산의 최상위 다음 토큰이 달랐다.

- BF16 argmax: 공백 토큰
- FP32 argmax: ` fifty`
- 전체 logits 최대 절대 차이: 1.0220260620
- logit RMSE: 0.1474928103
- KL(BF16 분포 || FP32 분포): 0.0035638120
- Total variation distance: 0.0384056227

이는 이번 입력에서 관측한 **연산 정밀도에 따른 차이**다. A/B 양자화 정책의 정확도 차이나 어느 답변이 더 옳다는 증거가 아니다. 따라서 다음 단계의 teacher-vs-native 비교에서는 원본 가중치뿐 아니라 teacher arithmetic, 입력 토큰·위치, attention 구현, 평가 지표를 먼저 고정해야 한다.

## 5. 실행 및 SHA 계보

원본 모델 index SHA256:
`d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`
원본 config SHA256:
`f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d`
원본 modeling source SHA256:
`7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c`
사전 입력 SHA256:
`c85d16c113780d73692fd0bdb9b1afaacb40bda9a998876420f361899d3cbb8b`

| 결과 | 전체 logits .npy SHA256 | Tailnet job |
|---|---|---|
| prompt0_bf16 | 9022e1866f0cfd0713aafe45c2e5946770a0048112e8e8165e5986489314203e | job_26dbf6baaa130b2624b5be42fae10924 |
| prompt0_fp32 | 3598245173f95fa7f2a5611eae2be2e58bb5ac4aaecb997ea5df208dfee0e3fe | job_0cf9a156cd654eef9675389b23238f1c |
| prompt1_bf16 | 7261448911fe14927195d603eeb90bc86ace2dd98209eb3c800e834eaf49558d | job_86fad166ec68f897ade39f7ab1ee15c5 |

증거: `experiments/hcp_p15/evidence/native_xox_20261015/`.
실제 logits 배열, 실행 JSON, 원본 source tensor 해시 목록, SELFTEST 로그, 독립 감사 JSON, arithmetic 비교 JSON을 보존한다. 공개/공유한 것은 계산 결과와 해시·메타데이터이며 원본 모델 가중치 파일은 복사하지 않았다.

주요 코드:
- tools/hcp_p15_streamed_teacher.py — 원본 계산 그래프의 계층별 로딩·실행
- tools/hcp_p15_run_teacher.py — 입력 봉인, 자원 watchdog, full-forward 실행 영수증
- tools/test_hcp_p15_streamed_teacher.py — 18개 원본 상주/스트리밍 수치·경계 검사
- tools/hcp_p15_independent_audit.py — 실제 전체 logit 배열 및 출처 감사
- tools/test_hcp_p15_evidence.py — 12개 변조·거짓 완료 차단 시험

## 6. 운영 보존과 정확한 다음 관문

각 전체 forward 전후 원본 운영 바이너리 SHA, active-route SHA, 운영 PID 35481·35493을 비교해 변화 없음을 확인했다. 운영 main이나 정밀도 정책을 수정하지 않았으며 Release Plane 활성화도 하지 않았다. 실행 전 Current/Live/Runtime은 operator.900345로 수렴했다.

**이제 해결됨:** 전체 BF16 교사 추론을 실행할 메모리 경로와 실제 full-vocabulary reference logits 확보.

**아직 미완료:**
1. 동일한 토큰열·위치에서 이 교사와 실제 네이티브 AF/A/B logits의 직접 비교 및 차이 원인 분석.
2. 별도의 미사용 정답형 평가셋, 긴 입력/여러 토큰 생성, NLL·perplexity·품질 비열등성의 사전 정의와 검증.
3. production-grade teacher 재현성·캐시/자원/장기 부하 인증과 이후 명시적 운영 승격 심사.

따라서 이전의 `full_BF16_teacher_forward_executed=false` 상태는 P15 실측 범위에서 해소됐다. `native_A_B_quality_noninferiority_certified=false`와 `automatic_live_promotion=false`는 그대로 유지한다.

참고한 공식 API 문서:
- https://huggingface.co/docs/safetensors/en/index
- https://huggingface.co/docs/accelerate/main/en/concept_guides/big_model_inference
원본 수학 구현의 기준은 위 문서의 최신 모델 예제가 아니라 해시 고정된 로컬 체크포인트 source다.
