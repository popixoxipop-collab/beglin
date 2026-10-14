# Beglin HCP P10 — XOX Runtime Qualification, Fresh-Input Parity, Restart, 2-Slot, Steady Tail

**작성: 2026-10-14 KST | 분류: 격리 연구 통과 / 운영 승격 BLOCKED**

## 1. 핵심 판정

P9의 prepared MLX qNg64 캐시 실기기 전환 가속(한정 범위 34.87%)을 유지한 채, P10에서 별도 실기기 안전성·출력·steady-state 시험을 수행했다.

- 신규 12분야·32~56토큰 입력(최대 12토큰 생성)에 대한 캐시 ON/OFF **48/48 네이티브 요청**, (입력, 정책)별 **24/24 출력 SHA 대응 일치**. 기존 P4/P6의 36개 입력과 SHA가 다름.
- 강제 종료는 **완료된 요청 뒤 연구 워커에만** 실시했다. 새 PID/시작 정책·epoch, 오래된 워커 요청 거부, B→A→B 재적용 9개 복구 요청을 검증. 정상 종료 후 같은 연구 워커 재시작도 PASS. 전원/트랜잭션 도중 충돌 실험은 수행하지 않음.
- **MLA 2-slot**: 2요청을 1 native batch로 처리하는 호출 **8/8 PASS**, 생성 시퀀스 **16/16** 원본 P6 1-slot 참조 SHA와 일치. 실제 두 요청 배치이지 서로 다른 워커의 동시 policy-write 증거가 아님.
- 캐시 정책 A를 유지한 별도 연구 워커로 4시간블록×12입력×ON/OFF, **96회 steady admission PASS**, p95 OFF 585.16ms, ON 583.08ms. ON−OFF 평균 −3.49ms, 4-block bootstrap 95% 구간 [−11.00,+4.47]ms로 0 포함. P9의 전체 steady p95 악화는 이번 반복에서 재현되지 않았지만 이득도 입증하지 못함.
- **BF16 teacher 정답/비열등성: 미실행·BLOCKED**. 원본 DeepSeek-V2-Lite BF16 4개 safetensors 31,413,626,576 bytes(29.256GiB), 관측 XOX 가용 메모리 23.575GB 및 swap 사용 21.84GB. EOE 가용 21.537GB, Alienware 가용 CUDA GPU 9.483GiB. 안정적 전체 BF16 적재를 보증할 자원 여유가 없어 운영·모델 서비스를 중단하거나 swap 초과 적재를 강행하지 않음.
- 기존 immutable serving stage, active route, PID, 운영 정밀도 정책과 Tailnet release plane을 변경하지 않음. 자동 승격 OFF.

**결론: P10 좁은 범위의 생성 동등성·재시작·2-slot 실증 완료. 전체 BF16 품질, 독립 외부 평가, 장기 동시성·트랜잭션 도중 복구·운영 승격은 미완료.**

## 2. 소스·사전등록·자원 계약

- GitHub 저장소: popixoxipop-collab/beglin
- 연구 브랜치: research/hcp-p10-runtime-qualification-20261014
- P9 베이스: c7d4d5fb46f5cb57abb1a1e7054a1294955a69d9
- 신규 문제 사전등록 source commit: b551237fb976e65447e6eb0e82b880cc57626caa
- 토큰화 SHA manifest 사전봉인 commit: 72a645e (native 실행 전 GitHub에 게시)
- 새 12분야 문제 원본 SHA: 7523f8e8ef5d2dea21391ad18ab575606b5804299386bf715a9647c815758083
- 새 token manifest SHA: 93f6f9f21983cb0c93e78b813b2972724aff45501b773943be2020574ada3b9f
- 원본 P4 동결 증거: c3b54b1ebe41d9a85ed94ddd0da87cde0de86ecf0af56d2e621733e439944f53
- 원본 P6 품질 증거: 2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104
- P5 C CB128 source SHA: bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612
- P9 opt-in packed cache MLX KV128 source SHA: ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f
- XOX 동등 재컴파일 연구용 GPU binary SHA: 103549b7f7f4a3d142be883013956b4f1437759c8be76bff31f2451956df37fd

P10도 P9와 **동일 SHA**의 MLX 네이티브 바이너리를 재컴파일했다. 캐시 OFF/ON은 동일 바이너리에 대한 연구 워커 환경 플래그 차이다. 기존 Tensor math, REBIND_SET terminal ACK, snapshot/rollback, MLX KV128, 14GiB RSS 상한과 20GB XOX SSD 보호 기준을 유지했다. 마이크로벤치 결과와 정밀도 정책 승인 계약을 변경하지 않았다.

신규 문제의 기본 질문·자체 예상 답안은 사전등록 GitHub 문서에 있다. 새 토큰 입력은 32~56토큰, 생성 budget 최대 12. 기존 36개 연구 input SHA와 겹치지 않는다. **이 자체 제작 12문항은 외부 독립 벤치마크나 실제 업무 정답률 표본을 대체하지 않는다.**

## 3. 실제 새 입력 출력 정합성

| 항목 | 관측 |
|---|---:|
| 신규 분야/문제 수 | 12 |
| cache OFF와 cache ON × 정책 A/B | 총 48 네이티브 admission |
| 동일 질문·정책 기준 출력 SHA 매칭 | **24/24** |
| 실제 생성 길이 | 3~12 토큰 |
| P9 prepared MLX cache ON 저장/적중 | 4 / 30 |
| cache OFF 저장/적중 | 0 / 0 |
| 캐시 소스 바이트 불일치·용량 거부 | 0 |
| 최대 RSS | 5.210 GiB |

자체 예상 답안과의 기초 정규화 문자열 포함 여부는 A 9/12, B 9/12. 이 수치는 **탐색적 관찰**이다. 문구 변형과 제너레이션 예산 영향이 있고, BF16 teacher·인간 전문가 외부 판정과의 교차검증이 없다. A/B 우열 및 품질 비열등성 게이트로 사용하지 않는다.

## 4. 실제 재시작·장애 복구

원본 모델 출력 SHA와 네이티브 epoch/ACK를 그대로 검증했다.

1. 캐시 ON 단일 GPU 워커에서 B→A→B 정상 적용 후, 연구용 워커의 정확한 PID만 SIGKILL 했다(요청 완료 후). 이미 죽은 워커로 요청 시 정상적으로 거부됨.
2. 같은 실험 워커 경로의 새 네이티브 PID로 재시작. 후보 정책 A의 합법적 startup ACK인 **PROMOTION_APPLIED** 및 새 정책 epoch(기존 5 → 새 2)를 확인. 재시작 후 B→A→B 3개 실제 추론 전부 원본 P6 SHA 일치.
3. 정상 종료 후 재시작, 다시 새 PID·정책 A·B→A→B 3개 출력 일치. 전체 성공 요청은 종료 전 3개까지 포함 **9개**.
4. 캐시가 재시작 후 새로 저장·적중함을 확인. 원래 워커와 PIDs 및 이미 인증된 stage는 재시작하지 않았다.

첫 실행은 P10 테스트의 잘못된 startup status 기대값(STARTUP_STATE)으로 **exit 1**이었으며, 이는 후보 A가 PROMOTION_APPLIED를 내는 기존 C 계약과 다르기 때문이었다. ACK 검증을 생략하거나 운영 코드를 고치지 않고 기대값을 소스 계약에 맞게 수정하여, 두 번째 별도 실기기 실행이 **exit 0 PASS**. 최초 실패를 소급하여 성공으로 바꾸지 않는다.

**이번 테스트는 active txn 도중 process crash, 전원 차단, durable command journal의 exactly-once 대사, 실제 다중 워커 정밀도 정책 동시 작성까지 포함하지 않는다.**

## 5. MLA 두 슬롯

QWEN_MOE_CB_SLOTS=2, QWEN_MOE_CB_REQS=2 인 단일 연구용 워커에서, 원본 P6 104~112토큰 입력 두 개를 하나의 batch로 처리. 캐시 OFF/ON 각각 실제 정책 A→B→A→B를 적용하여 총 **8배치·16개 응답**이 정상 종료됐다. 모든 생성 시퀀스가 단일 슬롯의 P6 reference와 일치했고 실제 native ACK·RSS 점검 통과.

| 지표 | 관측 |
|---|---:|
| 실제 batch admission | 8 |
| 시퀀스 | 16 |
| 원본 P6 한 슬롯 출력과 일치 | 16/16 |
| 최대 워커 RSS | 약 3.998 GiB |

이 시험은 2개 GPU 요청 슬롯을 한 워커에서 지원하는 좁은 범위를 보일 뿐, 임의 크기 동시사용자·병렬 worker·재시작 중 정책 변경·KV128 초과 문맥에 대한 보증이 아니다.

## 6. Steady p95 재현성 시험

P9의 steady p95 ON 621.81ms vs OFF 581.82ms 악화를 분리 조사했다. 4개 fresh temporal block마다 별도 워커를 만들고 OFF→ON/ON→OFF를 교차했다. 정책 A/B를 예열하고 **정책 A로 다시 고정한 다음**, 원본 P6 12개 입력을 각 캐시 조건에서 한 번씩, 총 96개 steady 요청으로 측정. 유의한 precision transition이 발생한 요청을 본 측정에 포함하지 않았다.

| 측정 | cache OFF | cache ON |
|---|---:|---:|
| n | 48 | 48 |
| Mean admission | 563.0421ms | 559.5549ms |
| p50 | 559.3744ms | 559.0380ms |
| p95 | 585.1640ms | 583.0779ms |
| 최대 worker RSS | 5.1256GiB | 5.1308GiB |
| 블록별 평균 | 562.79 / 558.73 / 571.98 / 558.66 | 556.84 / 556.84 / 557.94 / 566.60 |

**대응 B 아닌 캐시 ON−OFF 차이 −3.4871ms**, 4개 temporal-block bootstrap 10,000회/seed20261014 95% 구간 **[−11.0042,+4.4652]ms**. P9의 전체 steady-tail p95 악화가 이번 실험에서는 재현되지 않았다. 그러나 구간이 0을 포함하고 마지막 블록에서 ON이 다시 느렸으므로 안정 상태 성능의 개선/비열등성이 증명된 것은 아니다.

## 7. BF16 full teacher gate — resource blocked, not skipped silently

- XOX 원본 DeepSeek-V2-Lite BF16 weights: 4 shard 합계 **31,413,626,576 bytes = 29.256GiB**. Model index SHA pinned.
- 2026-10-14 14:23 KST 자원 재조회: XOX available RAM 23.575GB, swap used 21.84GB; EOE available RAM 21.537GB; Alienware free CUDA GPU memory 9.483GiB.
- 기존 서비스를 유지한 채 raw BF16 weight 전체+런타임 activations/attention을 수용할 검증된 안전 여유가 없다. whole teacher model load, inference, BF16-to-P9 logit measurement를 **실행하지 않았다**. 독립 task accuracy/비열등성 결과는 **NOT MEASURED / BLOCKED**이다.
- 가능한 다음 수단: 별도 40GiB 이상 실가용 메모리의 호스트를 마련하거나, 독립 검증된 out-of-core/streamed BF16 teacher 실행기를 메모리·시간 fail-closed 상한과 함께 준비. 데이터 오염 없이 새 holdout source/token SHA를 재사용한다.

## 8. 테스트와 증거 계보

- P10 독립 증거 감사지표·변조 차단 테스트 **10/10 PASS**
- 기존 P9 source/cache/fault 경계 **21/21 PASS**
- 기존 P8 비용/governor source 검증 **48/48 PASS**
- **총 79/79 Python 회귀 PASS** (각 그룹은 별도의 테스트, 과거 P6 시험과 합산하지 않음).
- 새 입력 실제 48 admission, 2-slot 8 batch admission, 재시작 총 9개 정상 생성, steady-tail 96 admissions 완료. BF16 모델 추론 0회.
- 실기기 실험 JSON 원본·EOE 재읽기 SHA 증거: experiments/hcp_p10/evidence/native_xox_20261014/new_holdout.json, restart.json, two_slots.json, steady_tail.json, audit.json, bf16_resource_gate.json.
- 독립 재계산 도구: tools/hcp_p10_evidence_audit.py. 생성 토큰 원본값·가중치·raw logit은 GitHub에 업로드하지 않음. 2-slot의 실제 생성 원본 토큰 값 대신 runner의 verified SHA 판정만 공개되므로, 별도 독립감사는 원본 2-slot 시퀀스까지 재구성한 것은 아니다.

대표 SHA:
- new_holdout.json b0d8c83c05caed8d9f7b2699c8fa68c4e89e69fd53d437fbc65691f89b6d8ec4
- restart.json 5f386dbcfb0dd6c372e484bbb5db121bd20d285dca859a4446ba04fca1df748c
- two_slots.json 1d88d6b8379b027a7c45f7a9add58984b882151de2b743d4993d7cc9aaa1ccb8
- steady_tail.json 193e44ab5f1b69c8a1b491d81292b323f77b3db10571e3235da13ecfa46e3f35
- P10 token holdout manifest 93f6f9f21983cb0c93e78b813b2972724aff45501b773943be2020574ada3b9f

## 9. Release and exact next-step restrictions

Operating stage SHA 8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9. Operating route SHA c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47. Original stage PIDs 35481·35493. No main change, no automatic promotion, no Tailnet operator activation by this research.

Remaining production blockers:
1. **BF16 teacher** exact quality/noninferiority on frozen new input and independent externally labeled corpus.
2. Crash **during** actual REBIND_SET and durable in-flight queue reconciliation, cold OS boot, multiple worker concurrent admissions and CAS behavior.
3. More temporal blocks across independent sessions; cache lifetime under hot reload, model hash/allocator pressure; multi-model precision paths.
4. Scoped immutable artifact/release certification, then separately guarded stage/verify and fresh current/live/runtime and expected-current CAS before any activation. Current research commit is not a production-certified release.

**Disposition: P10 testable bounded runtime paths PASS; overall PRODUCTION_READINESS=BLOCKED due unmet quality/long-term/recovery gates.**
