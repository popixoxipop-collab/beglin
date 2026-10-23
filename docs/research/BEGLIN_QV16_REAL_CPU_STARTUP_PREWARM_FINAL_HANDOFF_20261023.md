# BEGLIN Qv16 — 새 48문항의 명시적 시작 전 정밀도 캐시 준비: 실제 원본 C CPU 실증 및 독립 검산

**실험일:** 2026-10-23 (KST)  
**비공개 저장소:** `popixoxipop-collab/beglin`  
**연구 브랜치:** `research/qv16-cost-aware-prewarm-20261023`  
**판정:** `4/4_TAILNET_NATIVE_JOBS_SUCCEEDED / ORIGINAL_C_48_OF_48_STATIC_PARITY_PER_MODE / REAL_Q4_N8_SWITCH_AND_HARD_CAP_PASS / USER_REQUEST_COLD_SWAP_RELOCATED_TO_STARTUP / THROUGHPUT_QUALITY_ADVANTAGE_NOT_ESTABLISHED / NO_LIVE_PROMOTION`

## 1. 목적과 기존 결과 구분

Qv15는 Qwen2.5-1.5B-Instruct 28층 모델의 CPU C 원본 추론 엔진에서 실제 살아 있는 텐서의 Q4/n8 비트 폭을 요청 사이에서 변경하며, 최대 **96MiB**의 추가 비활성 packed/scales 캐시로 23회의 전환 비용을 **55.003초→2.424초**로 줄였다. 그러나 **처음에 대체 정밀도가 캐시에 없으면 텐서 40개를 약 2.4초 동안 다시 양자화**해야 했다.

Qv16에서는 **첫 사용자 요청 이전의 명시적 엔진 초기화 단계**에서 바뀌는 40개 텐서의 대체 packed/scales를 원본 BF16에서 생성한 뒤 초기 Qv8 활성 가중치로 돌아오게 한다. 사전준비된 각 텐서는 요청 사이에서는 기존 Qv15 live-WT cache buffer swap만 수행한다.

**비용을 숨기지 않는다:** 40개 양자화는 모델 프로세스 시작 중 약 2.412초가 소요됐으며 사라진 연산이 아니다. 반면 실제 요청 경로의 첫 장시간 전환은 2.432초→0.284ms가 됐다.

## 2. 원본과 코드 재현성

- Qv15 pinned C 부모 SHA256 `cf4a3bafa1e20887d97f44b66bcc29dccaef6b93027dbb9471f873e0ba219d7f`.
- Qv16 격리 연구 C 원본 SHA256 `b05c2e8aac488f9accba44c61073a72f2103be6284b37aa5de0a500185995a9f`.
- Qv16 실제 컴파일 Apple CPU 네이티브 실행파일 SHA256 `3fc79f4c31af4d70ba36df87f447134a2f6c79b980748c3f4c2048440ee9a44d`.
- 기존 Qv15 Q4/n8 Apple NEON 원본 오브젝트 **23개**를 변경하지 않고 링크했다. 소스 차이 **두 구간**, 시작 전 초기화 도우미 및 호출만 추가했다.
- `forward_token`부터 `final_logits`, 원본 C 정답 gold-token softmax NLL 계산 구간은 SHA256 `10c84007027d15a9327c15ec687c2327e19390de65f03bf9d95aee4bd4c9fe72`로 **완전히 동일**함을 확인했다.
- 캐시는 `BEGLIN_QV15_CACHE_MIB=96`을 고정하고 `BEGLIN_QV16_PREWARM=1`을 명시할 때만 활성화. 없거나 `0`이면 동일 바이너리의 기존 온디맨드 Qv15 방식이다. 다른 환경값이나 96MiB 외 사전준비에는 fail-closed.
- 사전준비는 대체 텐서의 packed/scales 메모리 한 벌만 캐싱하며, BF16 변환/임시 F32 scratch와 원래 active quantized model의 프로세스 RSS는 별도로 평가한다. 전체 양자화 모델 두 벌 상주 아님.
- 시작 후 active tensor `28 n8 + 168 Q4` 상태 복귀 검사 및 요청 처리 시 상태 검증. 변경되는 원본 40개 텐서의 개별 정밀도에 대해 재생성 후 초기값과 교체 원본을 캐시에 둔다.

## 3. 데이터: 새 ARC validation 48문항의 사전 봉인

실험 이전 Qv14의 ARC-Challenge validation **4+24=28 문항과 겹치지 않는** 새 48개를 SHA-정렬로 선택했다. 기존 Qv13의 ARC/OpenBookQA **test**와도 데이터 split이 다르다.

- `independent_arc_validation48/frozen_inputs_private.json` SHA256: `a95f678f656b8d04f74f0bb643542cdd12bde8e317c8d7fa013795c4c4b3c296`.
- 공개 선봉인 `BEGLIN_QV16_ARC_VALIDATION48_PREEXEC_PUBLIC_20261023.json` SHA256: `ac5223bdf25b6cf607cb3a589027d1a406737af7341e8f742231155579c3b26a`.
- 처음 28개 원본 질문 ID와 겹침 **0**; 새 질문 소스 ID **48개 모두 고유**, 짧은 24개·긴 24개, 입력 56~181 Qwen 토큰.
- 정밀도 선택은 오직 prefix 길이 **85토큰 미만 Qv8**, 이상 random73. 48개 배열은 교대로 Qv8/random73를 선택하도록 봉인됐다.
- **이번 Qv16은 캐시 시작 및 정밀도 스위치의 기능/지연 실험이다.** 학습된 bit allocation 정책의 정확도 향상 평가, 사전학습 미노출 증명, 범용 품질 평가가 아니다.

## 4. 실제 원본 Apple Silicon C 실행 결과

| 원본 C 실행 조건 | 실제 48 입력 중 정답 | mean gold-token NLL | 명시적 시작 전 준비 | 요청 첫 마스크 전환 | 요청 중 전환 47개 합계 | 요청 캐시 hits/misses | peak inactive cache | process RSS peak | 전체 wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qv16 동일 파생 바이너리, **prewarm=0** | 34/48 | 1.26280193 | 0ms | **2,432.433ms** | **2,445.796ms** | 1,840/40 | 90.0469MiB | 4,728.359MiB | 256.302초 |
| Qv16 동일 파생 바이너리, **prewarm=1** | 34/48 | 1.26280193 | **2,412.228ms** | **0.284ms** | **13.611ms** | **1,880/0** | 90.0469MiB | 4,727.734MiB | 356.944초 |

48문항 각각의 **정답 NLL과 A/B/C/D 정답 여부는 두 실행 사이에 정확히 일치**했다. 요청당 실제 마스크 전환 47회의 prepared-swap **median 0.285ms**, p95 약 **0.300ms**, max 0.305ms. 요청 중 변환 비용 감소 약 **2,432.185ms**. 시작 전 계산 **2,412.228ms**를 별도로 기록한다.

전 체계가 사용한 추가 비활성 캐시의 실제 peak는 **90.046875MiB**, 96MiB hard limit 미만. 기존 전체 active Qv8 static 실행의 peak RSS는 약 **4,641.36MiB**. Qv16은 약 86MiB 정도의 추가 peak RSS가 필요했다.

**wall 오해 방지:** startup 준비 1인 실행은 356.944초, 준비 0인 실행은 256.302초였다. 호스트 병렬 부하, 온도/스로틀, 입력별 배치, 측정 시점 등이 충분히 통제되지 않았으므로 **전체 wall이 개선됐다는 주장 금지**. 직접 원본 C에서 계측한 startup, request-switch 구간에 대해서만 정밀도 전환 지연의 이동 효과가 증명됐다.

## 5. 독립 원본 C 교차검산

실제 `Tailnet Commander EOE` 네이티브 작업 네 개 모두 `SUCCEEDED/exit 0`:

- `job_0335558b7c8be45ebce505f07b643c2c` : 원본 정적 Qv8, 48문항, 32/48 정답, mean NLL 1.235301.
- `job_ac562cca50d8ce30dd00900bd80cf2f7` : Qv16 prewarm=1, 48문항.
- `job_6ffe0e7256151f0fa6da459cbc18a120` : Qv16 prewarm=0, 48문항.
- `job_38dda1bb75b734b95c6a86533eeec53d` : 원본 정적 random73, **사전 규칙이 실제로 고른 긴 24문항만** 실행, 17/24 정답, mean NLL 1.161971. 이는 임의 평가성적 선택이 아니며 frozen profile 선택 인덱스 1,3,...47에 대응한다.

검산 도구 `tools/qv16_independent48_split_static_audit.py`는 아래를 검증했다.

1. prewarm0/1 **각 원본 C 로그의 전체 48개 NLL·예측과 영수증, 입력 ID·Gold label, SHA·네이티브 바이너리** 일치.
2. 원본 정적 Qv8에 배정된 **24개 짧은 문항**, 원본 정적 random73에 배정된 **24개 긴 문항** 모두 해당 Qv16 동적 결과의 full-vocab 골드 토큰 NLL **완전히 동일**. 결과적으로 **prewarm0와 prewarm1 각각 48/48 원본 C 정적 기준 수치 일치**.
3. 실제 네이티브 텐서의 정밀도 28n8+168Q4, 47회 mask 전환, 1,880회 실제 WT 타입 변경, 96MiB 캐시 상한·RSS 별도 실측.
4. prewarm1 요청 중 cold requantization **0**; prewarm0 첫 길이 ≥85토큰 요청 시 cold requantization **40**; prewarm2 이후는 두 조건 모두 warm.
5. 문항 원문·토큰 배열·개별 NLL은 사용자 전용 EOE D50에 보관하고, GitHub에는 **공개 집계만** 올린다.

**독립 감사 영수증:** `docs/research/BEGLIN_QV16_INDEPENDENT_48_NATIVE_STARTUP_PREWARM_PUBLIC_20261023.json` SHA256 `d11744f8ab7bba5c6d324c5fde13db7d1232b8bf3b91a33b0c3600361ad6fb92`.

## 6. 과학적 판정과 후속 계획

**실증된 기능:** 사전 봉인한 Qwen2.5-1.5B-Instruct 28계층 CPU C에서 프로세스를 유지하면서 Q4/n8 텐서 40개를 요청 사이에 변경하는 실제 동적 정밀도 정책은 이미 Qv14에서 입증됐다. Qv15에서 96MiB 제한 캐시로 warm 전환을 안정화했고, Qv16에서 40개 대체 텐서를 **시작 전 명시적으로 준비**함으로써 **첫 요청의 약 2.43초 전환을 수백 마이크로초 수준으로 감소**시켰다. 원본 C 수치와 두 방식은 새 48문항에서 완벽하게 일치했다.

**미입증 항목 및 경계:** 
- 시작 전 2.412초 변환비용은 제거된 것이 아니라 **요청 경로 바깥으로 이동했다**. 전체 처리량/TPS, p95 TTFT, 전체 경로의 순수 개선은 측정 부하를 표준화하기 전까지 미입증.
- 추가 캐시 약 90MiB, peak process RSS 약 4.73GiB는 **메모리 비용**. 같은 상주 RAM 예산에서 static Q4/GPU를 이겼다는 주장 금지.
- 현 routing은 prefix 길이의 임의 휴리스틱. 정답률 Qv16 34/48, static Qv8 32/48의 차이는 **사전등록된 품질 이득이나 통계적 유의성의 증거 아님**.
- Metal GPU 커널, 추가 모델 패밀리, 동시 클라이언트·KV cache 경쟁, 서비스 재시작 시 데이터/캐시 무결성과 장애복구, 사전 패킹물의 서명/버전 호환성은 미검증.
- model checkpoint, 원본 kernel, BEGLIN main, Tailnet immutable operator/current/live/runtime, 자동 정밀도 승격은 **일절 변경하지 않았다**.

**다음 연구(Qv17) 권장:** 캐시 미스 없이 startup 2.41초도 낮추려면 원본 BF16 revision 및 C kernel SHA에 바인딩한 **서명된 90MiB 대체 정밀도 사전 패킹 파일**을 연구용으로 별도 생성/재로드하는 경로부터 검증하고, 디스크 I/O/무결성/복원 비용을 정밀 계측한다. 별도 트랙에서는 32/64MiB cached slot의 비용가중 admission policy를 독립 calibration으로 정하고, 새 장문/OOD 입력에서 정답률·conditional choice NLL·fullvocab NLL·선택/지연/메모리 Pareto를 검증한다. **아직 구현·실행하지 않은 후속 목표를 완료로 표기하지 않는다.**

**최종 결론:** `Qv16 real CPU startup prewarm and exact numerical parity PASS`; **사용자 요청 첫 precision-mask 전환을 2,432ms에서 0.284ms로 이동**, 대신 **startup +2,412ms와 비활성 가중치 캐시 약 90MiB**를 부담한다. 신뢰성 개선 메커니즘이지 정확도 최적화나 전체 서빙 성능 우월성까지 증명한 것은 아니다.
