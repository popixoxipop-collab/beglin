# BEGLIN Qv2 — 실제 전체 CPU 비동기 실행, 동기화 실증 및 직렬 기본값 판정

**날짜:** 2026-10-22 · **연구 브랜치:** `research/oml-acs-prereg-20261021`  
**판정:** **실제 full-model 비동기 실행 기술 검증 PASS · 전체 실행 속도 우위 미증명 · 정밀도 정책 품질 FAIL · 운영 승격 NOT_READY / 기본 직렬 유지**.

## 1. 왜 추가 실험이 필요했는가

이전 v2는 Qwen2.5-1.5B의 28계층×7역할=196 텐서에 대해 원본 BF16 활성값 12개 보정 문장(461 위치)을 추출했고, 원본 BEGLIN Q4G64·qNg64 n8 인코더의 입력 가중 국소 복원 오차를 계산했다. Q/K/V 및 gate/up의 **독립 native Apple CPU stage**에서 병렬화 이득을 측정했으나, **전체 Qwen 디코더는 그때 직렬 실행**이었다. 실측 stage 시간으로 전체 토큰 속도를 단정할 수 없었다.

이번 실험은 독립된 Apple CPU BEGLIN 연구용 실행 파일에서 Q/K/V와 FFN gate/up **실제 pthread 비동기 실행을 만들어** 그 가정을 검증했다. 기존 28개의 8비트 승격 텐서 마스크와 168개의 4비트 텐서는 **일절 재선택하지 않았다**. 원본 Qwen BF16 체크포인트 `4ae77cb209f06199b8df1c94e21ff341332a3a89`에서 파생된 동일 원본 가중치, 동일 8개 공개 평가 입력(131개 다음 토큰 위치), 동일 HEAD/LN/편향/KV 경로만 사용했다.

**이것은 이전에 안전 검사로 차단됐던 텐서별 전체 모델 출력 교란 연구를 다시 호출한 것이 아니다.** 정밀도 할당 정책은 사전 동결됐다.

## 2. 전체 모델 비동기 구현

- `forward_token`에서 attention RMSNorm 출력 직후, **Q 투영은 메인**, **K/V 투영은 서로 독립된 백그라운드 워커**에서 시작한다. RoPE, KV 쓰기, attention 전 양쪽 워커에 **완료 barrier**를 적용한다.
- post-attention RMSNorm 출력 직후 **gate는 메인**, **up은 별도 백그라운드 워커**가 실행하며 SwiGLU/MLP down 전 완료 barrier를 적용한다.
- **백그라운드 worker 3개**, K/V/up마다 **독립 6-thread q4pool**을 보유하여 원래 BEGLIN의 공유 `q4pool.job` 및 `cv`에 대한 자료경합을 피한다. 백그라운드 N8 가중치는 기존 사전 재배열 코드용 native dispatcher를 호출한다. 기존 메인 스레드는 원래 `g_pool`로 Q/gate 실행을 유지한다. 각 워커는 완료 후 정지·join을 수행한다.
- 같은 연구 바이너리에서 `BEGLIN_QV2_FULL_ASYNC=0/1`로 직렬/전체 비동기를 구별한다. `QWEN_FUSED_DISPATCH=0`을 **명시적으로 요구**해 타 정밀도일 때 비호환 Q4 fused dispatch를 건드리지 않는다.
- 동작 범위는 **단일 토큰 `forward_token` CPU 경로**다. `forward_tokens`의 배치형 병렬화, Metal GPU, 실제 동시 사용자 스케줄러, 런타임 비트 전환을 지원한다고 주장하지 않는다.
- 별도 연구용 바이너리 복사본에서만 구현했고 기존 BEGLIN 원본 추론 및 운영 릴리스는 변경하지 않았다.

### 실제 독립 동기화·품질 검증

| 검증 | 결과 |
|---|---|
| Qwen 전체 28계층, Q/K/V·gate/up 실제 pthread 실행 | PASS |
| 첫 입력의 최종 151,936 logits와 기존 직렬 비교 | **바이트 단위 동일** |
| 3개 백그라운드 워커 Q/K/V·FFN barrier | PASS |
| dump 실행 barrier 작업 | QKV=280, FFN=280 |
| 단일 토큰 bench 작업 | QKV=1,596, FFN=1,596 |
| 131개 평가 위치 PPL 작업 | QKV=3,892, FFN=3,892 |
| 직렬과 전체 비동기 PPL | **모두 807.363** |
| 직렬 vs Q/K/V만 병렬 최종 logits | 동일 |
| 직렬 vs FFN만 병렬 최종 logits | 동일 |
| Q/K/V만 병렬 PPL(131개 위치) | **807.363** |

FFN만 병렬 경로는 **최종 logits 일치 및 단일 토큰 속도만 검증**했고, FFN-only PPL은 다시 실행하지 않았다. 검증했다고 주장하면 안 된다.

## 3. 실제 전체 토큰 속도 — 초기 결과와 독립 반복을 모두 보존

**단일 토큰 native BEGLIN `bench 1`**, 워밍업 후 3회 중 빠른 값을 기록한 매우 작은 연구 샘플이다. 지속 디코드/서빙 TPS나 통계적 신뢰구간이 아니다.

| 전체 모델 CPU 방식 | 초기 측정 | 동일 바이너리 순차 재측정 |
|---|---:|---:|
| 직렬 | 86.07ms | **83.12ms** |
| QKV만 병렬, FFN 직렬 | **82.12ms** | **99.27ms** |
| QKV 및 FFN 모두 병렬 | 90.14ms | 90.32ms |
| FFN만 병렬, QKV 직렬 | 105.82ms | 재측정 없음 |

**중요한 부정적 결과:** 초기 QKV만 병렬의 속도 이득(82.12 vs 86.07ms, 약 4.6%)은 **반복에서 재현되지 않았다**. 동일한 연구용 전체 바이너리 SHA `4cfc89d2ee4137745d7830a32b799beab20fe681d7c3c87581ce8cd67fffe718`에, 정밀도/토큰 입력 동일, 모드만 달리한 순차 측정에서 QKV 병렬 99.27ms, 직렬 83.12ms였다. 따라서 **QKV 병렬을 더 빠른 기본 경로로 자동 선정하지 않는다.**

초기 전체 비동기는 직렬 대비 4.7% 느렸고, 동일 바이너리 반복에서도 느렸다. FFN-only도 가장 느렸다. 원인 후보로 스레드풀 wake/join 및 3×6-스레드 작업자 과구독이 있으나 세부 기여는 개별 실험으로 분해하지 않았다.

**현재 직렬 기본값 유지가 가장 근거 있는 선택**이다. 자동 비동기 토글·롤아웃·운영 프로모션은 금지한다.

## 4. 정밀도 정책의 품질 검증은 별개이며 여전히 실패

현재 v2 정책은 196개 투영 중 8비트 28개, packed 4비트 168개이고, 8비트 역할별 개수는 `q=1, k=1, v=23, o=1, gate=0, up=2, down=0`. 8비트 n8는 BEGLIN 원본 qNg64 비트플레인에서 **동일 바이트 수의 연속 INT8 코드로 최초 로딩 때 1회 재배열**한 별도 in-memory kind이다.

원래 v2가 발표한 공개 합성 131개 평가 위치의 PPL(낮을수록 좋음)을 변경하거나 튜닝하지 않았다.

| 원본 BF16에서 FP32로 확장한 고정밀 | 역할 균등 무작위 혼합 | 전체 Q4 | Qv2 활성값+병렬 비용 혼합 |
|---:|---:|---:|---:|
| **348.3639** | **552.7020** | 727.9173 | **807.3630** |

Qv2는 전체 Q4보다 PPL·투영 가중치 저장량·관측 단일 토큰 속도에서도 열세여서, v2 최적화기의 수학적 최적성은 실제 모델 품질의 우위를 뜻하지 않는다. BF16 원본 모델 계산 자체가 아닌 BF16 가중치를 F32로 복원한 기준임을 유지한다. 원래 24개 V6 사용 문항 출처도 아직 인증되지 않았고, 본 시험은 IID/OOD 독립 최종 평가도 아니다.

## 5. 감사와 재현 가능한 증거

모든 파일은 연구 전용 GitHub 브랜치에 커밋했다. **원본 프롬프트 토큰 ID, 양자화 가중치, 전체 151,936개 로짓 및 민감한 파일은 EOE 비공개 디렉터리에만** 존재한다.

- 실제 연구용 forward_token 비동기 코드 생성: `research/oml_acs/q_v2_actual_cpu_async_build.py`, 원본 SHA `d8e149092c73d858dfb7a066f697f2b95e0668ffa747b9d66ddc4ee3617f447a`, 실험 연구 C SHA `ea6e8db6f01ac90505754e0883f2396e5903ff7a6a6c043d33c0558684f60018`.
- 같은 바이너리 직렬/전체 비동기 실제 실행: `research/oml_acs/q_v2_actual_cpu_async_eval.py`, 6회 dump/bench/PPL 별도 실행.
- QKV-only/FFN-only 연구 경로: `research/oml_acs/q_v2_hybrid_full_cpu_async_build.py`, `q_v2_hybrid_full_cpu_async_eval.py`.
- 동일 바이너리 통제 반복: `research/oml_acs/q_v2_hybrid_same_binary_timing.py`. 비공개 출력 SHA **`b13abf47bd2369a95a041741d65dfb9ee55e86a6a233da7742319165382564b2`**, 0600.
- 최초 전체 비동기 source-bound audit 영수증 SHA **`af142423b54012ca28bdcf3b17c8be6824df1b4e51abbca6eb6a70426314c391`**.
- 최종 fail-closed 감사 `research/oml_acs/q_v2_full_cpu_async_fallback_audit.py`, 비공개 결과 SHA **`cc5cb236d3bb9115fcb638f7e700db96a36a1b1ac3ab7d6a3ffc5df1caca8377`**. 원본 서명·동일 평가 입력·전체 로짓·실행 로그·실측을 독립 재계산한다.
- 회귀 `tests/research/test_q_v2_full_cpu_async_fallback.py` **9/9 PASS**. `--require-ready --verify-only`은 **종료 코드 2**, 판정 `NOT_READY_DEFAULT_SERIAL`.
- 비공개 격리 디렉터리: `beglin-q-apple-20261021/lab_local/q_v2_actual_full_cpu_async_20261022`, `lab_local/q_v2_hybrid_full_cpu_async_20261022`. 원본 양자화 보정·평가 데이터는 별도 보관.
- 새 공개 증거 `docs/research/BEGLIN_QV2_REAL_FULL_CPU_ASYNC_FAILCLOSED_20261022.json`(SHA 참조·집계만).

### 실행 확인 예

```bash
python3 beglin-q-apple-20261021/tests/research/test_q_v2_full_cpu_async_fallback.py
python3 beglin-q-apple-20261021/research/oml_acs/q_v2_full_cpu_async_fallback_audit.py --verify-only
# 다음 명령은 NOT_READY이므로 의도적으로 종료 코드 2를 반환
python3 beglin-q-apple-20261021/research/oml_acs/q_v2_full_cpu_async_fallback_audit.py --verify-only --require-ready
```

## 6. 후속 과제 및 승격 원칙

1. 스레드풀 개수를 실제 core/Q4pool contention 기반으로 제한하고, 같은 바이너리에서 **다수의 반복·워밍업·온도·스레드 사용률·p50/p95/연속 TPS**를 확보한다. 이 결과로 QKV-only의 속도 차이가 재현되는지 통계 검증해야 한다.
2. pthread 스레드 생존/데이터 경쟁에 대한 ThreadSanitizer·프로세스 재시작·실제 배치 M>1 및 일시적 CPU 압박 테스트를 별도 수행한다. `q4pool` 소유권, 재포장된 n8 데이터 형식 태그를 운영에서 계약화해야 한다.
3. 질적 우위 확보 전에는 v2 정책 및 병렬 스케줄을 배포하지 않는다. 현재 대각 활성값 대리점수는 전체 모델 NLL을 충분히 예측하지 못하며, 원래 전체 모델 단일 텐서 출력 민감도 실행 차단을 우회하지 않는다.
4. Apple Metal GPU/메모리, 동시 사용자 실행, 새로운 IID/OOD 및 원래 V6 24개 사용 행의 출처 대사는 **미완료**다.

**현재 운영 상태:** BEGLIN `main`·실사용 추론 프로세스·원본 모델·Tailnet immutable 릴리스 `g6-0.9.0-operator.900387`에 변경/승격 없음. 공학적 기술 검증 PASS와 실제 성능/품질 향상 FAIL을 혼동하지 않는다.
