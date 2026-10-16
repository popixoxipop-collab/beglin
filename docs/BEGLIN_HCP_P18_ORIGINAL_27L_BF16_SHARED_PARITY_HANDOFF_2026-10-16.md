# BEGLIN P18 — 원본 DeepSeek BF16 공유 가중치 실제 모델 실측 결과

작성: 2026-10-16 (KST) · 연구 격리 / EOE D50 · 운영 정책 A 유지

## 최종 판정

**P18_SHARED_27L_BF16_REAL_FULL_GOLD_VOCAB_BOTH_ORDERS_BYTE_EXACT_DEVONLY — PASS**

원본 DeepSeek-V2-Lite BF16 모델을 **실제 27계층, 5,291개 원본 BF16 텐서 전부** 읽어 검증하였다. 203토큰 *개발용* 입력만 사용했고, 봉인된 480문항 held-out 테스트 입력은 성능 최적화 또는 모델 조정에 사용하지 않았다.

### 실제 단독 계산·공유 계산 결과

| 실행 방식 | 원본 텐서 읽기 | 계산 시간 (초) | 최고 RSS | 새 스왑 증가 |
|---|---:|---:|---:|---:|
| 단독 101+3 | 5,291회 | 149.044618 | 2.138GiB | 0 |
| 단독 198+5 | 5,291회 | 210.379304 | 2.145GiB | 0 |
| **단독 합계** | **10,582회** | **359.423922** | — | — |
| 공유 짧은→긴 | **5,291회** | **250.639138** | 2.153GiB | 0 |
| 공유 긴→짧은 | **5,291회** | **296.184593** | 2.164GiB | 0 |

정방향 shared baseline speedup: **1.434030×**, 시간 절감 **30.266%**.  
역방향 shared baseline speedup: **1.213513×**, 시간 절감 **17.595%**.

**정확성**: 짧은 입력의 3개 gold positions × vocab 102,400 = 307,200개 float32 logits 및 긴 입력의 5개 gold positions × 102,400 = 512,000개를 원본 독립 단독 계산과 비교. 두 처리 순서 모두에서 byte-exact 일치 및 최대 절대 오차 **0.0**, 4회 비교 총 **1,638,400 float32 logits 비교 PASS**.

## 소스 및 증거 고정

- 원본 203 개발 입력 SHA-256: `b628ec3247cfc7a94785a93e539562a7a2bc7726f133c21162479749b8f0ac2c`.
- BF16 원본 모델 P17 teacher SHA-256: `40dd5550b7759ac2760a6c374e1ddf3af35e68af637ab8d2a4f75bca8bee01d3`.
- 원본 5,291 BF16 tensor value manifest SHA-256: `88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda`.
- 공유 gold window v2 코드 SHA-256: `c160e79ecb527ccfcc7c908e7fc64f67abf8938a4186c2f7033eac753c7cde2f`.
- 본 실기기 자격시험 원본 SHA-256: `5b50cd59f7b056177a5ef696e8603299c722081ae00f6d21ba634d15af2620a1`.
- 중단 복구 가능한 4단계 실기기 실행기 SHA-256: `cf5b20128bfa93d151d24eb1969397fa9e1def35025c13f295b392a953f6f7ff`.
- 외부 독립 F32 원시 logits 감사 코드 SHA-256: `fdeb3a4863ae7a6fee4d53b045633ea40f6554299e47dcc1bc4b5a6ab60ac1a7`.
- 원본 short 실행 영수증 SHA-256: `e093cd26a6587726486a2b52524e13f8d0afef70914ef129b5edcadf050afa10`.
- 원본 long 실행 영수증 SHA-256: `92a967ff7d99f1e88da0d5c276f4e3bffa61e66e802fe88bf6fce82a7036f07f`.
- shared short→long 영수증 SHA-256: `efb6486b4170edde162246aed75289769494fcb02bc228f1cdc98a744301f708`.
- shared long→short 영수증 SHA-256: `b9d2628ac38d6aaaf14b848e110dd80b34c9994ef448a76a81b93539b72ca54a`.
- 첫 연구용 내부 auditor 영수증 SHA-256: `0002852bec74dc9d1ac447403bd5af2bd4b6b501b3cd91c944f6c3a14f0ef0ef`.
- **별도 독립 auditor** SHA-256: `6c8ca8aaf80c09f5eb9176f6d50f4300784213f7b245f8a2a841be10000fd19c`.
- 원래 합성/정적 QA 45/45 PASS에 덧붙여 **실제 BF16 정방향·역방향 원시 logits 독립 검사**를 추가함.

EOE D50 원본 결과(비공개): `/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015/P18_DEVONLY_BF16_STAGED_PARITY_20261016/`

GitHub에는 코드와 *개발용 집계 영수증* 및 이 인계 문서만 올리며, 원문 토큰/원본 가중치/원시 BF16 .npy는 업로드 금지.

## 제한, 적용 판정 및 다음 작업

1. 이번 검증은 original BF16, **CPU 기반**, 2개 개발용 입력의 3/5 gold positions 실측이다. 480 frozen cohort, 다양한 모델/프롬프트 길이, native GPU A/B, task accuracy, live latency나 실서비스 처리량을 인증하지 않는다.
2. 203토큰 dev input에서 동치성은 입증되었으나 **최대 1024 문맥·다중 샘플 크기 일반화는 미검증**. 연구용 v2 코드와 원본 단독 경로 모두를 유지하고 대역별 dev-only 홀드백(예: 512·963/1024 boundary)에서 bit-exact·RSS/스왑 조건 확인 후 P18 BF16 전수 평가의 후보로 격리 승격 여부를 판단한다.
3. 단 1회씩 측정한 wall-time speedup이므로 CPU 부하/캐시 영향을 제거한 반복 warm/cold 실측과 변동성 표시는 이후 별도 필요. 속도 향상은 주로 가중치 재읽기를 줄이는 데서 나온 것이며 **디코더 연산량을 절반으로 줄인 것이 아니다**.
4. 모델 복제본 17개 원본/EOE SHA 읽기 전용 비교 PASS였으나 **공식 복사 인증 seal이 별도로 차단돼** 있고, EOE native A/B 전체 추론 및 최종 480문항 비열등성·정답률·고정 workload latency 검정은 여전히 미완료다.
5. 운영 변경 없음: Tailnet current=live `g6-0.9.0-operator.900353`, converged=true, active operation=null, XOX 원본 stage SHA `8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9`, active route SHA `c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47`, 둘 모두 최신 readback 불변. 운영 정책 A를 유지하고 정책 B 자동 승격 OFF.

**최종 결론**: 두 개발용 입력에 대한 원본 DeepSeek BF16 공유 계산의 논리적 정확성과 실제 속도 향상 PASS. *연구용 최적화 승인*, **480 전수평가/운영 적용은 미승인**.
