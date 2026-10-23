# BEGLIN Qv18 — 서명된 국소 정밀도 캐시의 독립 프로세스 동시 읽기 및 버전·신뢰 정책 실증

**실행일:** 2026-10-23 (KST)  
**연구 저장소:** `popixoxipop-collab/beglin`  
**연구 브랜치:** `research/qv18-concurrent-artifact-guards-20261023`  
**최종 상태:** `TWO_OVERLAPPING_ORIGINAL_QWEN28_NATIVE_PROCESSES_PASS / 16_OF_16_ORIGINAL_C_NLL_EXACT / NATIVE_FORMAT_VERSION_ROLE_FAIL_CLOSED / 10_OFFLINE_KEY_ROTATION_TESTS_PASS / NO_OPERATIONAL_KEY_ROTATION / NO_SAME_PROCESS_CONCURRENCY_PROOF / NO_LIVE_RELEASE_CHANGE`

## 1. Qv17 기반과 Qv18 목표

Qv17에서 실기기 BEGLIN 원본 Qwen2.5-1.5B-Instruct 28계층 Apple Silicon CPU/C 경로를 사용해 40개 대체 Q4/n8 정밀도 텐서의 **83.4856MiB** 패킹 자산을 보존·서명하고 다음 새 프로세스에서 로드했으며 48문항 NLL/정답을 원본 C와 bit exact로 검증했다. 단일 로더에 대해서만 검증한 것이다.

Qv18에서는 **원본 artifact 파일을 변경하지 않은 채, 서로 다른 원본 C 프로세스 두 개가 동일 서명 패킹 자산을 동시에 읽고 추론할 수 있는지**, **패킹 구조 위반이 예상 해시를 새로 계산한 공격 입력에서도 차단되는지**, **개발용 서명키 변경/모델·마스크·버전 바인딩을 정책 차원에서 차단할 수 있는지** 분리 시험했다.

## 2. 실제 두 개의 독립적인 원본 C 프로세스 동시 시험

`tools/qv18_two_process_signed_cache_stress.py`는 owner-only Qv17의 `alternate40_Qv8_random73_qng64_v1.pack` 파일 한 개를 사용하며 두 개의 실제 Apple 28계층 Qwen C 바이너리 프로세스를 겹쳐 실행한다. 각 프로세스는 **별도의 Python trusted bootstrap**에서 Qv17 로컬 개발용 Ed25519 서명/원본 BF16 3GB 모델 전체 SHA/가중치 파일·정밀도 마스크·Qv17 실행파일 SHA를 독립 검증한 다음 native packed Q4/n8 가중치를 로드한다.

- 대상 모델·자산은 Qv17의 원본과 동일, 바이너리 SHA-256 `16085378788b64b229f321b49c11540e286edffbff832db4e41fbf8d2a87643b`.
- 두 프로세스가 공유한 원본 packed artifact SHA-256: `49c409a79588b79dcefeba60e09b71459ac8df2d02ac0bda5e2cc1ab8076a86b`, 원래 파일의 mode `0600`.
- 입력은 이미 source-only 봉인됐던 Qv16 ARC validation 48문항 중 **처음 8문항**. 모든 프로세스에서 Qv8/random73 마스크를 교대로 선택한다. **이는 기능/경합 스트레스 시험이지 별개의 새로운 품질 검증 집합이 아니다.**
- CPU 동시 시험을 위해 각 프로세스 `Q4_THREADS=2`, `Q4_POOL_SPIN=0` 고정. 두 프로세스는 **완전히 별개 활성/비활성 가중치 모델 메모리를 소유**하며, 오직 서명된 읽기 전용 디스크 파일만 공유한다.

| 네이티브 측정 | 독립 프로세스 A | 독립 프로세스 B |
|---|---:|---:|
| 실제 C 8문항 | 8/8 PASS | 8/8 PASS |
| Qv16 원본 C와 문항별 NLL 정확히 일치 | **8/8** | **8/8** |
| 원본 정답 | 8/8 | 8/8 |
| Native C 자체 프로세스 wall | 104.693초 | 105.017초 |
| Native C packed asset read (SHA 포함) | 98.512ms | 208.919ms |
| Python trusted boot 전체 모델/서명 검증 | 1,857.416ms | 1,857.417ms |
| 각 C 프로세스 RSS peak | 4,722.58MiB | 4,722.73MiB |

- 실제 **동시 살아있는 C 프로세스 시간 겹침 104.692602667초** (서로 독립적으로 기록한 monotonic native 프로세스 시작/종료 구간으로 계산, native 두 개가 살아 있는 sampler 표본 481회).
- 두 프로세스와 자식 프로세스 등을 포함한 관측 최대 aggregate RSS **9,477.71875MiB**. 두 서버가 각자 모델 메모리를 보유하기 때문에 전체 RAM 비용은 단일 서버의 거의 두 배다.
- 원본 C 실제 로그의 각 q4/n8 정밀도 상태, 캐시 280 hits/프로세스, zero request miss/drop, C signature/pack SHA, 원본 Qv16 source-only 입력/정답 NLL 8/8을 독립 파서가 다시 검산했다.
- 시험 뒤 원래 packed artifact SHA가 같고, 원본 모델과 운영 릴리스는 변경되지 않았다.
- Q4 pool thread 수를 기존 6에서 2로 줄였음에도 **골드 토큰 NLL의 바이트 값은 Qv16 원본 C와 8/8씩 일치**했다.

**중요 범위 한계:** 이것은 **동일 서명 파일의 읽기 공유와 별도 프로세스의 동시 추론** 검증이지, **한 프로세스 안에서 동시 다중 요청이 같은 활성 WT/KV 상태를 안전하게 교체할 수 있다는 증명은 아니다**. 같은 프로세스의 무락 Q4↔n8 포인터 스왑을 병렬 요청에 바로 적용해서는 안 된다. 실제 서비스에서는 model-level lock/reader–writer barrier 또는 요청별 weight snapshot/version lease가 필요하며, KV·GPU 커널 동기화는 별도 검증해야 한다. 측정한 단일/동시 packed 읽기 시간은 OS page cache·CPU 부하가 달라 스토리지 cold-read p95를 의미하지 않는다.

## 3. 해시를 공격 입력에 맞춰 다시 계산해도 네이티브 포맷 검사 차단

`tools/qv18_native_version_role_negative.py`는 원본 signed artifact **83.5MiB 파일을 건드리지 않고**, 임시 owner-only 복사본에서
(1) 헤더 format version을 `1→2`로,
(2) 첫 텐서의 역할 번호를 `0→7`로 변경했다.

두 경우 모두 **변조된 복사본에 맞는 새로운 SHA-256을 계산하여 그 값을 C 로더에 expected digest로 직접 전달**했다. 즉 **해시 불일치 경로가 아니라 C의 역할/버전/shape 계약을 독립적으로 시험**했다.

| 실제 원본 C 음성 시험 | Native 종료 | 정답 scoring |
|---|---:|---:|
| 잘못된 패킹 포맷 버전 | exit 1 / fail-closed | **0건** |
| 잘못된 텐서 역할 ID | exit 1 / fail-closed | **0건** |

원본 C가 참조할 SHA가 변조본과 같더라도, 실제 패킹 스키마·40개 역할/계층/정밀도 계약이 맞지 않으면 실행하지 않았다. 임시 복사본은 시험 종료 후 삭제됐다. 원본 서명 자산 SHA 전후 변화 0. 이 시험은 Ed25519 신뢰 루트를 우회한 **직접 native 호출**이므로, **native C가 서명을 직접 검증한다는 주장은 절대 아니다**. 원래 Ed25519 검증은 신뢰된 Python bootstrap 경로가 수행한다.

실제 Tailnet 실증 작업:
- 2-process concurrent job: `job_b28240bf1dc17b23078623f1b4a7f90a` → **SUCCEEDED/exit 0**.
- Native format negative job: `job_4084f6f26b31b15ec8d90f9123986b27` → **SUCCEEDED/exit 0** (두 원본 C 실행은 의도적으로 각각 exit 1이며, 검증 래퍼가 이를 확인하고 정상 완료).

## 4. 로컬 개발키 회전·폐기·버전 롤백 정책: 오프라인 10항목

`tools/qv18_offline_trust_rotation_test_v2.py`는 실제 기존 Qv17의 서명 문서를 개발 공개키로 검증한 후, **두 번째 Ed25519 개인키를 메모리 내에서만 일시적으로 생성**한다. 두 번째 private key는 디스크·GitHub에 절대 저장하지 않았다.

- 새 키가 이전 신뢰 목록에서 거부됨.
- 신뢰 목록에 새 공개키가 등록되면 새 **서명된 envelope** 수락.
- 기존 v1 공개키가 새 신뢰 목록에서 철회 상태임을 확인.
- 서명은 유효하나 최소 허용 release sequence 미만인 이전 버전 거부.
- 올바르게 서명됐더라도 잘못된 원본 모델 digest·네이티브 실행파일 SHA·정밀도 마스크 SHA·패킹 파일 digest는 각각 거부.
- 서명된 sequence를 재서명 없이 고치는 변조와 다른 signer의 서명을 거부.

전체 **10/10 로컬 Ed25519/정책 음성·양성 확인 PASS**. 실제 Qv17 원본 dev 키/서명/패킹 파일은 이 시험에서 변경되지 않았다.

**한계:** 이건 **메모리 내 모의 신뢰 상태** 테스트다. 운영 서명키 교체를 실행하거나 기존 개발키를 실제 배포에서 폐기한 것이 아니고, tamper-resistant/monotonic anti-rollback 상태 저장을 구현한 것도 아니다. 개발키를 조직 생산용 배포 인증서로 해석할 수 없다.

## 5. 독립 검증과 공개 영수증

`tools/qv18_second_independent_audit.py`를 별도로 작성해 원본 두 프로세스 C 로그 16개, 서명된 원본 BF16 전체 3GB 해시, packed asset SHA/Ed25519 signature, 네이티브 역할/캐시 계측, 실제 프로세스 overlap, 포맷 음성 로그 2개, 모의 key lifecycle 항목 10개를 **기존 실행기의 집계 함수에 의존하지 않고** 다시 검산했다.

- 독립 감사 공개 영수증: `docs/research/BEGLIN_QV18_SECOND_INDEPENDENT_SIGNED_MULTIPROCESS_AUDIT_PUBLIC_20261023.json`, SHA-256 `c66c1c82e3053da2343ff21c5a0c91796bee2834d642470c7268ed88bdfdc0f2`, **PASS**.
- 첫 동시 네이티브 영수증 `BEGLIN_QV18_TWO_PROCESS_CONCURRENT_SIGNED_CACHE_PUBLIC_20261023.json`, SHA-256 `7f39e581a1fa492ea358af5603cea628194f8592e38b89dcc98b91107eb42605`.
- 실제 잘못된 포맷 C 거부 영수증 `BEGLIN_QV18_NATIVE_VERSION_ROLE_FAIL_CLOSED_PUBLIC_20261023.json`, SHA-256 `095a9701ff8a02b5268d7e3ea953d72df752031742fdbec173f008504a1c8720`.
- 오프라인 회전/롤백 모의 10항목 영수증 `BEGLIN_QV18_KEY_ROTATION_ANTIROLLBACK_V2_PUBLIC_20261023.json`, SHA-256 `8daa0f695dadb5964e9873f4f1cc17bceb6494d1c9864bfc4797508a5564c948`.

공개 파일에는 **원본 문항 내용, Qwen 토큰 배열, 문항별 NLL, private dev signing key, BF16 전체 가중치 파일, 83.5MiB packed data 본문을 절대 업로드하지 않았다**. 그 원본은 기존 EOE owner-only D50 저장공간에 그대로 남겨두었다.

## 6. 최종 판정과 다음 진입 게이트

**Qv18 PASS:**
1. 여러 독립 C 프로세스가 서명된 하나의 packed artifact를 읽기 전용으로 공유하고 원본 입력·정답 NLL을 동일하게 재현한다.
2. 원본 C 파서가 버전 및 역할/shape 계약 위반을 해시가 맞더라도 추론 전 차단한다.
3. 로컬 Ed25519 개발키 회전·폐기 및 서명된 모델·마스크·버전 바인딩 규칙을 순수 정책 수준에서 10건 검증했다.

**Qv18만으로 주장할 수 없는 것:** 실제 원격/물리 콜드부팅, OS page cache 없는 디스크 응답, **한 모델 프로세스 내부의 동시 요청/동시 포인터 교환 및 KV 캐시 독립성**, 운영 CA·서명키/키관리/키회전·rollback 방지, Metal GPU 메모리/커널 전환, 정답률을 올리는 독립 데이터 기반 최적 비트 할당 정책.

**후속 Qv19 우선순위:** (A) **같은 모델 프로세스의 요청별 precision-state 접근/쓰기 lock 및 inference lifetime snapshot**을 설계하고 모델 output 중 다른 요청이 WT를 바꾸지 못한다는 증거 확보; (B) Metal GPU의 같은 Q4/n8 packed 가중치 버전 리스/서명·residency 별도 실증; (C) calibration-only 목적함수와 사전등록 heldout에서 **loss/accuracy/p95-latency/RSS Pareto**를 검증해 동적 선택의 진짜 가치 판단. 아무 것도 결과 확인 이전에 완료라고 표기하지 않는다.

**운영 릴리스 상태:** immutable Tailnet G6 current/live/runtime `g6-0.9.0-operator.900387`, 기존 정상 수렴 상태. main/release/production 및 모델 checkpoint·원본 엔진 커널을 **변경하지 않았다**.

**한 문장 결론:** Qv18은 **동일 서명 캐시 파일의 동시 다중 프로세스 안전 읽기 및 실제 native C 포맷 오류의 fail-closed**를 검증했지만, 서비스용 같은 모델 인스턴스 공유 동시성 및 운영용 인증·정밀도 최적화를 입증한 것은 아니다.
