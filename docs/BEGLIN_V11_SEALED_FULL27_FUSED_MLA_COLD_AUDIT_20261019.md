# BEGLIN v11 — 사전양자화 실모델 27계층 + fused MLA KV, 연속 2토큰 및 콜드 스타트 3회

작성일: 2026-10-19 (Asia/Seoul)  
**판정: 연구 통합 PASS · 독립 감사 2종 PASS · 3회 독립 프로세스 PASS · 운영 승격 NOT_READY.**

이 문서는 EOE Apple Silicon 실기기의 원본 DeepSeek-V2-Lite BF16 체크포인트, 실제 BEGLIN 원본 C qNg64 n7/group64 압축기, MLX C++ Metal 가중치 관리, 소스에 정의된 MoE 라우터, 기존 GPU 상주 fused MLA KV 계산을 통합한 **격리 연구 검증**을 기록한다. 원본 가중치·프롬프트·logits 배열은 GitHub에 업로드하지 않았다.

## 1. 검증한 가중치 경계

| 원본/압축 항목 | 실측·판정 |
| --- | --- |
| 원본 safetensors 샤드 | 4개, 각 전체 SHA-256 검사 완료 |
| 원본 텐서 수 | 5,291개 |
| n7 양자화 대상 | **5,181개 / 5,181개**, 압축 파일 전부 생성 |
| BF16으로 유지하는 비대상 | 110개 (embedding, norm, router 등 기존 정책) |
| 실제 n7 bitplane/F32 scales 합계 | **14,328,299,520 B** |
| BF16 유지 대상 포함 모델 직렬화 용량 | **15,174,228,992 B**, 산술적 파일 페이로드 |
| 원본 BF16을 독립 C 재양자화한 전체 비교 | **5,181/5,181, 14,328,299,520 B 비트 단위 일치** |
| SHA 고정 봉인 매니페스트 | `c82361f7fbd73884b633bb595334805c0b7c0cf2f82ab8de26311ca78a0142b8` |
| 원본 샤드/압축기/내보내기 코드 권위 | `3b13bc1d98c5843f146afa428b6dfff45352f7def698d6a5a156d32d0fae136f` |

양자화 파일은 D50의 EOE 격리 연구 아티팩트로 저장되며 원본 BF16 체크포인트를 바꾸지 않았다. **모든 5,181개를 GPU에 동시에 올린 것은 아니다.** 디코더 단계에서 필요한 역할만 SHA 확인 후 메모리 매핑하고 제한된 GPU LRU에 등록한다. 모델 구조 및 원본 MoE 구성·고전적인 BF16 제외 텐서는 계속 필요하다.

## 2. 실모델 생성 — 기존 v11과 사전양자화 v11

| 검증 | 기존 v11, 8GiB CPU SHA memo / 캐시 + 별도 비캐시 참조 | 신규 v11, 봉인 prequant / fused MLA KV |
| --- | ---: | ---: |
| 실제 원본 DeepSeek 27계층 | 27/27 | **27/27** |
| 실제 첫 번째 선택 토큰 | 429 | **429** |
| 첫 번째 디코딩으로 선택한 토큰 | 4281 | **4281** |
| 두 번째 디코딩의 다음 예측 | 6230 | **6230** |
| prefill logits 상대 L2 vs 기존 독립 uniform7 | 0.014482878254655093 | **0.014482878254655093** |
| fused MLA GPU 호출 | 371 | **371** |
| 별도 GPU 상주 kv_b n7 가중치 | 53,084,160 B | **53,084,160 B** |
| 7토큰/27계층 BF16 latent KV | 217,728 B | **217,728 B** |
| 추론 중 원본 C 가중치 양자화 | 발생 (기존 8GiB SHA memo 사용) | **0회** |
| GPU 일반 qNg64 호출 | 8,289회 | **3,525회** |
| GPU 가중치 핸들 정리 | PASS | **PASS** |
| 전체 연구 실행 시간 | 464.119초 | **117.000초 (1회)** |
| 최대 프로세스 RSS | 9,133,473,792 B | **1,860,435,968 B (1회)** |

**중요한 비교 제한:** 이전 연구 스크립트는 실행마다 BF16 원본 구조·동적 압축, 캐시 디코딩과 **동시 비캐시 참조 재계산**, 8GiB CPU SHA memo를 포함한다. 새 캐시 통합 스크립트는 독립 비캐시 참조 실행을 별도 영수증으로 대조하고, 동일 프로세스에서는 해당 재계산을 반복하지 않는다. 따라서 이 표의 wall/RSS/GEMV 차이는 **동일 조건에서 측정한 속도 향상·메모리 절감 보증이 아니다.** 정상 요청 기반 TTFT/TPOT, 전체 통합 메모리, GPU allocator 피크도 별도 측정 전에는 주장할 수 없다.

신규 캐시 통합 1회에서는 실제 로드한 n7 소스 모듈 3,372개, 원본 C와 GPU를 수치 검사한 모듈 3,291개, 범용 GPU 역할 재사용 234회, LRU 교체 3,207회, FP32 원시 수치 확인 시 최대 상대 L2 **1.0239×10⁻⁷**을 기록했다. BF16 반올림 경계에서 인접한 1 ULP 변화 1회를 기록했으며 FP32 오차 및 BF16 인접값 계약을 함께 검증했다. 가중치 압축 바이트가 변조되어 발생한 오류는 아니다.

신규 실행은 실제 원본 MLA의 512차원 normalized latent와 64차원 RoPE key를 저장하며, 원래 16-head 확장 K/V를 영속 보존하지 않는다. 새 GPU-owned qNg64 레지스트리는 추가 C++ CPU 압축 버퍼를 유지하지 않지만 **원본 구조를 구성하기 위한 Python BF16 샤드 읽기는 아직 남아 있다.**

## 3. 독립 감사 영수증 (EOE 로컬에서 재읽기 SHA 확인)

| 감사 대상 | SHA-256 | 판정 |
| --- | --- | --- |
| 기존 v11 27계층 캐시·2토큰 | `8c36149cd9b062724efe604495b5f56d14aede59d9b86883ea93757cb8676745` | PASS 기존 기준 |
| 신규 27계층 prequant 비캐시·2토큰 | `ae29603fec2814c5164bd7598006117a7ba776cd154243e1a215cf5b30d8263e` | PASS |
| 신규 27계층 prequant **fused MLA cached**·2토큰 | **`165460894349ed00efbaa7ce96a276a6772d7e9bfa82c89e90c61bc244f21081`** | **PASS** |
| 첫 번째 독립 교차 검토 | `fc4596076ee42e9eb740e40630abc1685612b4d584e200d6b2a7d6fa5fc7625f` | PASS |
| 두 번째 독립 교차 검토 | `2b5c07e53c04c474d8589c417f062d8f762e18718e6a4267ef56d25c3352c02d` | PASS |

두 독립 감사 모두 실제 생성 토큰, 원본/정밀도/가중치 SHA, 27계층·3단계 forward, fused Metal 호출, 영속 MLA KV 217,728 B 및 CPU 재양자화 0회, GPU 핸들 정리를 점검했다. 독립 전체 logits 배열의 **비트 동일성**을 기존 v11과 직접 비교한 것은 아니며, 입력 토큰·출력 예측과 첫 단계의 수치 비교가 인증된 범위이다.

## 4. 실제 3개 독립 프로세스 재현성

최초 성공한 1회에 이어 새로운 OS 프로세스 2개를 순차 실행했다. 각각 27계층 prefill + 실제 선택 토큰 429·4281의 cached decode와 다음 예측 6230을 완료했다. 두 디코딩 단계의 **각 102,400개 vocabulary logits SHA-256이 세 프로세스 간 동일**했다.

| 프로세스 | 전체 연구 프로토콜 시간 | 최대 프로세스 RSS | 연구 영수증 SHA-256 |
| --- | ---: | ---: | --- |
| Cold 0 | 117.000초 | 1,860,435,968 B | `165460894349ed00efbaa7ce96a276a6772d7e9bfa82c89e90c61bc244f21081` |
| Cold 1 | 113.489초 | 1,862,565,888 B | `15c655ef1856b725bb9c255a9fe17cf1b87a23b25b8c5a704d1c1e63a2582b6b` |
| Cold 2 | 113.670초 | 1,915,699,200 B | `a7af3921e398382eb9686831b2e94c7649c175da3a408021196cd9564aa0d9e0` |

콜드 재시작 전체 감사: **`756b6d07de3c097e7aae0254d0bfebb6989d17bc4c76c20201196bb058bac814`** · `PASS_REAL_27L_SEALED_FUSED_MLA_INDEPENDENT_COLD_RESTARTS`.

세 프로세스 모두 원본 모델/압축 매니페스트 SHA 일치, 동적 C 재양자화 0회, GPU 가중치 해제 PASS다. 작은 3회 표본의 p95/p99은 제시하지 않는다. 113~117초는 새 연구 전체 실행 시간일 뿐 실제 서빙 모델의 1토큰 지연시간이 아니다.

## 5. 소스 및 재현 범위

GitHub PR #63 Draft의 연구 전용 코드:

- `research/n7_multi_projection_resident_v1/prequant_n7_sealed_v1.py` — 봉인 아티팩트 SHA + 읽기 전용 mmap
- `research/n7_multi_projection_resident_v1/full27_sealed_materializer_v1.py` — 원본 가중치 5,181개 일회성 직렬화
- `research/n7_multi_projection_resident_v1/prequant_full27_execution_adapter_v1.py` — 이전 C++ 기하학/형식 + 소스 권위 검사
- `research/n7_multi_projection_resident_v1/run_v11_sealed_full27_recompute_v1.py` — 실제 모델 27계층 비캐시 기준
- `research/n7_multi_projection_resident_v1/run_v11_sealed_full27_cached_mla_v1.py` — 원본 모델 fused MLA + 사전압축 GPU 가중치 캐시
- `research/n7_multi_projection_resident_v1/audit_v11_sealed_cached_mla_v1.py` 및 `audit_v11_sealed_cached_mla_independent_v1.py` — 독립 검토
- `research/n7_multi_projection_resident_v1/test_v11_sealed_cached_mla_cold_restarts_v1.py` — 3회 프로세스 재현성
- `.github/workflows/beglin-v11-prequant-sealed.yml` — 파일·스키마·부정 입력 회귀, CPU Python 3.11/3.13. 실모델 파일은 CI에 제공하지 않는다.

**원본 체크포인트, 샤드, 모델 토큰값 배열, 모델 은닉 상태와 프롬프트 문장은 절대 GitHub로 전송하지 않는다.** SHA/영수증과 범위 표만 저장했다.

## 6. 남은 생산 엔지니어링 승격 차단 조건

1. 소스 BF16 전체 가중치 재로드를 제거한 **prequant-only 서빙 로더**와 기존 엔진의 실행 계약/운영 바이너리 통합. 현재 원본 Python 구조 생성 및 BF16 소스 참조 읽기는 남아 있다.
2. 충분한 holdout 프롬프트·과제 평가와 NLL/KL/Top1/실제 품질 기준 검증. 본 결과는 1개 5토큰 프롬프트에서 2개 생성 토큰뿐이다.
3. 수십 분~수시간 연속 반복 및 다중 실제 요청 병렬성, GPU in-flight eviction, 메모리 압박·프로세스 중단·OS sleep/resume 실증. 3회 콜드 재시작만으로 장시간 안정성 보증 불가.
4. 동일 서빙 프로토콜에서 v10 fallback 대조 **TTFT/TPOT/TPS p50/p95/p99**, CPU RSS + MLX GPU allocator + 통합 메모리 비교. 이번 연구 벤치 시간으로 속도 우위 주장 금지.
5. 독립 원본 XOX 서빙 바이너리/소스 권위 확인 및 운영 승격 전 CAS/릴리스 인증. 운영 릴리스 자동 승격 비활성 유지.

**안전 결정:** PR #63 계속 Draft·미병합·미배포. v10 기존 경로 보존. EOE 운영 `g6-0.9.0-operator.900387`과 XOX 라이브 서빙은 연구 중 변경하지 않았다. **다음 목표: 생산 서빙 로더 분리 및 장시간 실요청 검증.**
