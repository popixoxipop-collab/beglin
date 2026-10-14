# Beglin HCP P15 — Fourth Complete Teacher Forward & Independent Audit Supplement

작성: 2026-10-15 KST · 연구용 격리 실측 · 운영 승격 금지

기준 저장소: `popixoxipop-collab/beglin`
브랜치: `research/hcp-p15-streamed-full-teacher-20261015`
원본 P15 세 실행 봉인 커밋: `110cefa0d44d09f1d07d5137705369920a652928`
소스·입력 사전등록 커밋: `1cd75c4668b700f3b1d6a2b008a64a02f2048366`

## 이번 추가 실행의 의미

원본 P15 결과는 BF16 계산 2건, FP32 계산 1건이었다. 이후 두 번째 고정 입력에 대해서도 FP32 계산으로 원본 BF16 체크포인트 전체를 **27계층·5,291텐서·102,400 vocabulary logits**까지 실행했다.

네 번째 요청은 XOX Tailnet job `job_589a5331544f1abe51bee61bcc1c2dab`의 `SUCCEEDED`, exit_code 0으로 확인되었다. 최대 실제 프로세스 RSS는 **2.717GiB**, 별도 스왑 증가 0, 전체 모델 forward 부분 시간 **48.33초**였다. 이미 검증된 세 원본 결과와 구분해 이 추가 결과를 별도 GitHub 증거 루트 `experiments/hcp_p15/evidence/supplement_prompt1_fp32_20261015`에 보관한다. 기존 세 실행의 MANIFEST.json 및 최종 핸드오프 문서·영수증은 변경하지 않았다.

## 실제 전체 BF16 가중치 교사 2×2 실행

| 고정 입력 | BF16 CPU forward | FP32 CPU forward |
|---|---|---|
| `The result of seven times eight is` (7토큰) | 78.04초, 2.48GiB RSS, Top-1 ` ` (ID 207) | 43.31초, 2.79GiB RSS, Top-1 ` fifty` (ID 15092) |
| `Complete the sentence: The capital of France is` (9토큰) | 37.94초, 3.15GiB RSS, Top-1 ` _____` (ID 51536) | **48.33초, 2.72GiB RSS, Top-1 ` _____` (ID 51536)** |

네 실행 모두 체크포인트 원본 저장 비트폭 BF16이며, FP32 경로는 원본 BF16 값을 정확히 FP32로 확장하여 계산했다. 두 경로의 연산 dtype을 혼용하거나 각각의 가중치를 다시 학습·양자화하지 않았다.

각 실행은 원본 27개 decoder layer 전체, 원본 tensor 5,291개 전체, 최종 normalization 및 LM head, 유한한 102,400개 float32-exported logits를 실제 생성했다. 모든 모델 파라미터의 **원본 BF16 바이트 SHA 목록이 네 실행 모두 일치**했다. 값 원본과 로그잇 `.npy`는 연구 증거 디렉터리 외부에 배포·수정하지 않았다.

### BF16 vs FP32 연산 차이 (동일한 원본 가중치)

| 지표 | 첫 번째 입력 | 두 번째 입력 |
|---|---:|---:|
| 전체 logits 상대 L2 차이 | 3.208% | 2.829% |
| 전체 logits RMSE | 0.14749 | 0.15879 |
| JS divergence (nats) | 0.0008865 | 0.0004389 |
| Top-5 토큰 집합 겹침 | 5/5 | 5/5 |
| Top-1 토큰 | 다름 | 일치 |

두 번째 고정 입력의 Top-1이 빈칸 형태로 나온 것은 원본 모델의 **다음 토큰 분포** 결과이지 해당 문장의 정답이 아니다. 위 수치는 정답률 또는 문제 해결 성능의 근거로 사용하지 않는다. 단 두 입력에서 계산 정밀도 차이를 관찰했다는 뜻이다.

## 독립 검증과 실패 차단

EOE에서 원본 `.npy` 전체를 직접 읽어 4개 실기기 결과 모두에 대해 SHA, shape, 102,400 finite float32 logits, argmax, logit 통계, 27계층 순서, 5,291개 텐서의 원본 BF16 SHA manifest, 원본 모델 인덱스와 사전 고정 토큰을 독립 확인했다. 원본 운영 바이너리·라우트·PID 불변 증거도 각 결과에서 확인됐다.

기존 18개 소형 상주형↔스트리밍 수치동등성 검증과 초기 12개 거짓 완료·증거 변조 테스트는 원본 핸드오프에 보존한다. 이번에는 진짜 XOX 결과를 복제한 격리 디렉터리에서 원본 logits 파일 위조, NaN 주입, source tensor 누락, 26/27계층 생략, 허위 full-model 상주형 대조·네이티브 정확도·승격 주장 등을 거부하는 **신규 28/28 실물 증거 변조 테스트**를 통과했다. 이 세 그룹(18+12+28)은 서로 다른 범위의 테스트이며 58/58개를 이번 한 번에 새로 전부 실행했다고 주장하지 않는다.

본 추가 증거의 `SUPPLEMENT_MANIFEST.json`에는 기존 원본 세 실측의 불변 MANIFEST SHA `4b2020d6b6955a1bd2c8174c051bf00ff005686f35ccdd38cf3b37080f68b0d5`와 추가 FP32 원본 logits SHA `74979f5a4a16e6df98e5ac9860b6166bf1afa8e150a37bccec8bd35ab45ad593`, 독립 감사 및 28개 신규 테스트 영수증을 기록했다.

## BF16 전체 교사 성과와 남은 차단 관문

**확인된 사실:** 전체 BF16 저장 가중치를 계층별로 실행하여 두 고정 입력의 BF16/FP32 full-vocabulary logits를 모두 얻었다. 과거 P12/P13의 부분 행렬 오라클과 달리 모든 Transformer 계층을 통과한 모델 전체 출력이다. 최고 관측 RSS 약 3.15GiB로 메모리 보호 상한 6GiB 안에서 완료했다.

**아직 미완료:** 네이티브 Beglin AF/A/B의 같은 입력·동일 시점 full-vocab logits 계측, 12개 이상 분리된 평가 과제, 여러 토큰 생성의 정답률 및 perplexity, 사전 정의된 품질 비열등성/신뢰구간. 전체 16B 원본 모델을 한 번에 상주시키는 방식과 직접 수치 대조한 것은 아니며, 작은 3층 모델에서만 스트리밍↔상주 동등성을 검증했다.

다음 직접 실행 관문은 별도 research-only 네이티브 worker에 **첫 생성 토큰 직전 102,400개 MLX GPU logits 수집**을 source SHA로 제한하여 추가하는 것이다. 원본 `qwen_infer.c`의 GPU persistent prefill 마지막 위치에서, 현재 토큰을 결정하는 `gpu_logits` 배열을 동일한 고정 prompt·정책 A/B별로 수집해야 한다. 이후 원본 BF16 교사의 두 calculation dtype을 구분해 Top-1/Top-5, JS divergence, KL, NLL 및 품질 비열등성의 출처와 판정 범위를 명시해야 한다.

운영 라우트 변경·main 병합·Tailnet 릴리스 승격은 실행하지 않았다. **P15 전체 교사 logits 수집: PASS. 양자화 정책 A/B 비열등성: BLOCKED.**
