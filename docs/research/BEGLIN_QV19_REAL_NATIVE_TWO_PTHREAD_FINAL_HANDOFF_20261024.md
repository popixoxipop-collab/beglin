# BEGLIN Qv19 — 실제 Apple/Qwen28 단일 모델 프로세스 두 pthread 요청별 배타 잠금 실증 최종 핸드오프

**완료:** 2026-10-24 (KST)  
**Repo:** `popixoxipop-collab/beglin`, research branch `research/qv19-single-process-lease-20261023`  
**판정:** `REAL_ORIGINAL_QWEN28_APPLE_NATIVE_ONE_PROCESS_TWO_PTHREAD_MODEL_LEASE_PASS / GOLD_NLL_8_OF_8_EXACT / ED25519_DEV_ATTESTED / NEGATIVE_GATE_DENIED / NO_TRUE_PARALLEL_FORWARD / NO_PRODUCTION_PROMOTION`.

## 1. 최초 Qv19 장애 해소: 실제 Qwen28 C 통합

이전 Qv19 패키지는 standalone 1,200건·합성 C bridge 160건 및 sanitizer 회귀만 통과했으며 실제 원본 28계층 BEGLIN 엔진에는 연결되지 않은 `NOT_READY` 상태였다. 이번에는 **GitHub 연구 브랜치에서 사용자가 제공한 검증된 C 소스 세 파일을 동일 SHA로 등록한 뒤, EOE의 독립 연구용 clone에서 원본 C 엔진과 정식 컴파일**했다. 이것은 이전 중단 상태를 숨기는 것이 아니라 **새 실제 네이티브 증거로 갱신**한 것이다. 연결 경로의 공개 파일 SHA 검사와 원본 서명/커널 무결성을 통과했고 운영·모델 원본은 변경하지 않았다.

- 원본 Qv17 C 소스 SHA-256 `0c9d797c0066fdb1f8ca5972bdf7611ae75431050d0d55c0f0fdef0653f6bdaa`.
- 변경되지 않은 Qv17 `forward_token` → `final_logits` → gold-token NLL 점수 계산 부분 SHA-256 `10c84007027d15a9327c15ec687c2327e19390de65f03bf9d95aee4bd4c9fe72`.
- 사용자 검증 원본 잠금 구현 `qv19_request_lease.c` SHA `5788e0f4d2afbf592f93e7ca2bc6e1cbe0b3f5ecdfcaace3742b157842ad3b93`, 헤더 SHA `5fc29c9e25bf6148166f823eb1587c90ebe41d6cfe6d7a6e8e5f4d711398ec8d`, Qv19 브리지 SHA `aab974fa09b4ff65d2559aeaa1aff2bb9f03dcfb7743d51dbc11b41dd3363cd7`. 첨부 v2 패키지의 원본 바이트와 동일.
- 실제 분리 파생 C 소스 SHA `b3a2535c28cb276df66f0fc5ae677c0c0c4f996d34de3b3032907f3a5337fe32`.
- **실제 clang 컴파일/원본 23 Q4/n8 Apple 커널 링크 결과 Apple arm64 Mach-O** SHA `c333c5a95aca340e59352c7e0e716521b3f57028002a53e307a9f962a2c654c2`.
- 새로운 파생 바이너리는 원본 Qv17과 구분된 **새 로컬 개발용 Ed25519 detached envelope로 서명**했다. 서명된 envelope SHA `55a9f6573dcbd1cdbc1daf248f76de71af05e40fb4b3e4c2f192f855abaa7297`. 기존 Qv17 packed cache 서명과 원본 3GB BF16 SHA도 trusted runner에서 재검증했다. 로컬 자체 생성 개발키이므로 **운영용 인증서가 아니다**.
- 빌드·실제 실행 모든 원본은 EOE owner-only D50 `/Volumes/D50/TailnetCommander/work/beglin-qv19-native-20261023/`에 보존. 커널/checkpoint·GitHub main·운영 G6 release 변경 없음.

## 2. 실제 한 Qwen28 C 프로세스의 두 pthread 연구 시험

고정된 Qv16 ARC-Challenge validation 첫 8개 요청(Qv8 source와 random73 정밀도 선택 규칙, prefix 85토큰 기준)을 사용했고, Qv17 원본 서명 packed 83.5MiB 파일로 96MiB inactive cache를 준비했다. 새 바이너리 명령은 **명시적 opt-in `qv19_guarded_choice`**이며 기존 Qv17 native 경로를 덮어쓰지 않았다.

- 각각 원본 C에서 실제 토큰 추론, 공용 KV cache, 공용 activation scratch, 원본 full-vocabulary logits/Gold-token NLL 계산을 수행.
- 모델 라이프타임 mutex를 획득한 요청만 live tensor Q4↔n8 swap → `forward_token` 전체 → `final_logits` → gold-token NLL을 실행한 다음 잠금 해제. 두 pthread는 **서로 다른 요청**을 받아서 처리하지만 실제 forward는 직렬.
- **실제 작업 ID:** `job_f19cbf71e7fae4ddfc50c6a2ae60c7ba` — `SUCCEEDED / exit 0`.
- 실제 두 작업자가 **각 4문항**, 원본 Qv16 고정 정밀도 C의 입력 8문항·full-vocab 정답 NLL **8/8 정확히 동일**.
- **정답 8/8**, 원본 실측 평균 full-vocab gold token NLL **0.001969528362390539**.
- Qv19 model lease epoch **1~8 유일/순차**, 모든 요청 enter=exit epoch, peak 동시 활성 forward **1**, poison **0**.
- Cache miss **0**, drop **0**, inactivity cache peak **90.046875MiB**(96MiB cap 아래).
- Native 8문항 wall 약 **111.129초** (`Q4_THREADS=2, Q4_POOL_SPIN=0`). 이를 throughput 향상이라고 주장하지 않는다.
- 네이티브 원본 C 로그 별도 파서에서 경합 대기 **최대 21.233초, 중앙값 약 11.021초**, 실제 잠금 아래 추론 구간 중앙값 약 **11.021초**. 모델 상태를 안전하게 보호한 대신 요청이 기다리는 비용이 발생했다.
- Signed derived Qv19 최초 네이티브 실행 기록 SHA `9e6a237b18fcc6585cb5e826fc60306502500f8ab056d2e073d9cb3ffbf793e0`.
- **Qv19 독립 원본 C 로그/모델 BF16 전체/서명/23개 커널/에폭 재검산 PASS:** `INDEPENDENT_PUBLIC_AUDIT.json` SHA `21dd23465bad901339cdc02f2b6a5a8d4af3c41573bbdc3114c28adb8c4ee448`.

## 3. 실제 네이티브 안전 차단 음성시험

새 개발키 서명된 Qv19 바이너리를 검증한 다음, 실환경과 같은 원본 모델/서명된 cache를 로드하면서 **필수 `BEGLIN_QV19_ENABLE_EXCLUSIVE_MODEL_REQUESTS=1`을 일부러 설정하지 않은 상태**로 새 Qv19 명령을 호출했다.

- 원본 Qwen28 네이티브 C는 `FATAL: Qv19 guarded original C requires explicit original C research gate`로 **exit 2**.
- 실제 gold-token C 채점은 **0건**. 기본 경로에서 보호되지 않은 Qv19 모드를 실수로 켜는 걸 차단했다.
- 해당 테스트 래퍼 job `job_1cd13d5c5337f939ad7a0a98a8f4d2a4`는 의도된 네이티브 거부를 확인해 `SUCCEEDED / exit 0`. 실제 내부 C 거부 `exit 2`와 구분한다.
- 음성시험 공개 영수증 SHA `abf9dd0daea0cadbfce3777aca299783b04463527703a0f2a561f81d0145fb5b`.

## 4. 신뢰성과 실험 해석의 정확한 경계

**검증 완료:**
- 실제 Apple Silicon 28계층 Qwen 모델 **동일 프로세스**에서 pthread 요청 2개를 위한 보수적 배타 잠금 경로 기능.
- 요청 전체 라이프타임 동안 live Q4/n8 가중치/공용 KV cache/활성화 버퍼/최종 logits를 한 번에 하나만 사용한다는 원본 C mutex·epoch 증거.
- 두 pthread가 실제 실행됐고, 이전 Qv16 원본 C 정답 NLL 8/8 완전 재현.
- 개발용 Ed25519 바이너리 출처·기존 Qv17 packed signed asset·원본 BF16 전체 SHA/23개 Apple C 커널 바인딩.
- 필수 Qv19 opt-in 미설정 시 실제 원본 C에서 안전 차단.

**아직 검증되지 않은 범위:**
- 하나의 모델 인스턴스가 두 `forward_token`을 **동시에** 실행하는 진짜 병렬 추론, 요청별 KV snapshot 및 cancellation.
- 다른 `qv14_dynamic_choice`, `ppl` 등 원래 CLI 진입점까지 모델 lock을 반드시 경유하도록 통제하는 실제 제품 서빙.
- 더 많은 요청, 서로 다른 모델 가족과 고부하 장기간 soak, TSan 등 전체 native 코드의 데이터 레이스 부재 증명.
- Metal GPU의 커널 완료 barrier/unified memory precision version lease.
- 정확도를 개선하는 최적 동적 비트 할당 정책 및 별개 heldout 집합에서 통계적으로 의미 있는 정확도/latency/RSS Pareto 이득.
- 운영 공급망 서명/실제 cold boot/hardware fault recovery.

**판정:** Qv19의 기존 `실제 Qwen28 네이티브 통합 미완료` **차단은 해소**됐다. **CPU 원본 엔진 연구용 mutex 기반 요청 직렬화 실증 PASS**. 하지만 사용자 서비스를 위한 전 경로 동시성·GPU·운영 안정성 승격은 `NOT_READY`다.

## 5. 다음 연구 단계

1. 독립 경합 연속 실행(현재 8문항 첫 원본 실기기 확인), 요청 취소·외부 비활성 입력·원본 미수정 fail-closed 시험 확대.
2. 제품 동시 사용을 고려하면 **모든 원본 엔트리포인트**에서 모델 라이프타임 락 필수화 및 공용 KV/부작용/취소 계약 확정.
3. 병렬성을 실제 얻으려면 요청별 KV 및 activation state 분리, live WT precision-state pin/reader-lease, GPU queue completion fence 설계 후 동시 forward 검증.
4. 본 연구의 핵심 가치인 **정확도 개선 동적 할당**은 별도 calibration-only 품질/위험 비용 모델과 새 heldout test, full-vocab 및 4-choice conditional NLL·정답률·worst-tail·p95 latency·peak RSS로 완전히 별도 판정.

**운영 릴리스:** `g6-0.9.0-operator.900387`; 원본 모델·원본 Qv17 C·실제 G6 operator 파일/서비스 변경 없음.
