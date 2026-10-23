# BEGLIN Qv15 — 원본 C 28계층 실시간 국소 정밀도 전환 캐시: 32/64/96MiB 실제 CPU 성능 및 검산

**날짜:** 2026-10-23 KST  
**비공개 저장소/브랜치:** `popixoxipop-collab/beglin` / `research/qv15-bounded-precision-cache-20261023`  
**과학적 상태:** `REAL_NATIVE_CPU_Q4_N8_BOUNDED_CACHE_3/3_PASS / EXACT_STATIC_NUMERIC_PARITY_72/72_PASS / MECHANISTIC_SWITCH_SPEEDUP_PASS / NOVEL_DYNAMIC_QUALITY_ALLOCATOR_NOT_ESTABLISHED / METAL_NOT_TESTED / NO_LIVE_PROMOTION`

## 1. Qv14로부터 이어진 실제 병목

Qv14의 원본 Apple Silicon 28계층 BEGLIN C 엔진은 24개 ARC-Challenge validation 문항에서 **Qv8→random73→Qv8→random73**으로 번갈아 23번 실제 텐서 정밀도를 변경했다. 매 프로파일 전환에서 40개 WT 텐서의 Q4↔n8 비트 폭을 교체하며, 매번 원본 BF16→F32 전개→그룹64 재양자화를 수행했다.

- Qv14 원본 비캐시 24문항: **정밀도 변경 920개, 누적 전환 55.003343초**, 전체 wall 186.410426초, peak native process RSS 4,696.453125MiB, 17/24 정답, mean full-vocab gold-token NLL 1.590552635.
- 연구용 정밀도 선택은 오직 **prefix가 85토큰 미만일 때 Qv8, 그 외 random73**. 이는 정답률 최적화 학습 정책이 아니며, 모델 출력/정답을 조회해 정밀도를 정하지 않는다.
- 동등 역할 구성 28n8 + 168Q4, 활성 packed projection bytes 765,493,248. Qv15에서도 동일하게 유지.

## 2. Qv15 캐시: 구현 및 메모리 안전성

원본 Qv14 v3 C에서 연구용 버전만 파생해, 각 변화 대상 WT 텐서의 **비활성 Q4 또는 n8 packed buffer와 scale array 한 벌만** 보존한다. 요청이 해당 정밀도를 다시 요구하면 **그 WT 객체의 기존 packed/scales와 비활성 캐시 포인터를 맞바꾼다**. 전체 양자화 모델 2벌을 준비하거나 원본 C/NEON 커널을 대체하지 않는다. 캐시에 없는 정밀도는 원래의 BF16 기반 C 그룹64 양자화 경로로 처리한다.

- **하드 메모리 제한:** `BEGLIN_QV15_CACHE_MIB`는 연구용 `0|32|64|96`만 허용한다. cache cap은 *비활성 packed+scales 실제 바이트 합*이다. 텐서 전개용 임시 F32, 원본 BF16 mmap, 다른 모델 weight, 프로세스 RSS는 이 cap의 외부이므로 **각각 실제 RSS로 별도 관찰**한다.
- 안전한 순서: 역할별 **n8→Q4 하나를 처리한 직후 같은 역할 Q4→n8 하나를 처리**한다. 두 마스크의 **역할별 n8 cardinality가 동일한지 검증**하고 맞지 않으면 차단한다. 중간 active n8 수는 28을 초과하지 않는다. 비활성 캐시도 96MiB hard cap을 넘지 않는다.
- 모든 요청마다 실제 **28개 n8+168개 Q4** 역할 상태를 네이티브 C에서 확인한다. 로그에는 실제 cache hit/miss/drop, live inactive bytes/peak/cap 및 프로파일 변경 소요를 기록한다.
- 원본 Qv14 v3 C SHA-256 `07a69e0a67cac0c448d6c7f6ec23cc506d95cf9a3bfa905acbb1d41abbbd7702`.
- Qv15 최초 캐시 C SHA-256 `0b1a069384d042b1b1164eb6ca05499000f21219e5692773476021edfeaafd79`.
- 개선된 Qv15 paired C SHA-256 **`cf4a3bafa1e20887d97f44b66bcc29dccaef6b93027dbb9471f873e0ba219d7f`**, 컴파일한 원본 커널 23-오브젝트 Apple CPU 연구 바이너리 SHA-256 **`daba57fd5bd747d3e6923de05114f9d313c9dac4765d63b5fa2f84060f69af67`**.
- 원본 C의 `forward_token`, `final_logits`, 전체 vocab gold-token softmax NLL 본문 695바이트는 Qv14→Qv15 **바이트 동일**(SHA-256 `10c84007027d15a9327c15ec687c2327e19390de65f03bf9d95aee4bd4c9fe72`). 파생 커널·실제 수학 계산을 편법으로 변경하지 않았다.

## 3. 실패도 보존한 V1 실험

초기 Qv15 캐시는 동작했지만 Qv14의 기존 전역 **전부 내림→전부 올림** 순서를 그대로 사용해, 중간에 비활성 캐시가 96MiB에 가까워지면서 일부 slot이 버려졌다.

- 실제 원본 C 24문항 추론은 정상 계산, 정적 원본 C와 NLL/정답 24/24 일치.
- 관찰된 hit **859/920**, miss 61, drop 22, 누적 전환 9.320861초, hard cap 준수.
- 그러나 사전 목표인 880 hit/40 miss에 못 미쳐 **Python fail-closed 검증이 실패하였고, Tailnet job `job_34c89281c3a2b5fe315fc50011da9308`은 FAILED**. 이 실패 이력을 `BEGLIN_QV15_FIRST_GLOBAL_ORDER_CACHE_96MIB_FAILURE_PUBLIC_20261023.json`에 보존했다. 실패한 job을 PASS로 소급 수정하지 않았다.
- 해결: 단 하나의 C 루프 변경만 포함하는 **역할별 paired down/up v2**, 독립 바이너리와 새 job ID로 재시험.

## 4. 실제 Apple Silicon CPU — 동일한 고정 24문항 / 동등한 정확도

| 조건 | 전환 23회의 누적 switch-ms ↓ | 캐시 reuse hits / 920 | 실제 재양자화 misses | 캐시 drop | 비활성 cache peak | native wall 초(부하 혼입) | 프로세스 peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qv14 이전 비캐시(다른 C 바이너리) | **55,003.343** | 0 | 920 | 해당 없음 | 0MiB | 186.410 | 4,696.45MiB |
| Qv15 paired **32MiB** | **40,492.225** | 770 | 150 | 115 | 28.0078MiB | 234.728 | 4,680.70MiB |
| Qv15 paired **64MiB** | **20,412.679** | 831 | 89 | 51 | 63.2813MiB | 205.149 | 4,716.34MiB |
| Qv15 paired **96MiB** | **2,424.341** | **880** | **40** | **0** | **90.0469MiB** | 136.842 | 4,725.86MiB |

**핵심 검증:** Qv15 3개 설정 모두 기존 원본 static C가 해당 문항에 실제로 사용하는 Qv8/random73 마스크와 **gold-token full-vocab NLL 24/24 정확히 동일**하고, 최종 A/B/C/D 정답 여부도 완전히 같다. 합계 **72/72 원본 C 대응 NLL 검산 PASS**. 원본 Qv14 static, dynamic 엔진은 이번 과정에서 변경하지 않았다.

직접 계측한 누적 정밀도 변경 구간은 기존 대비 **32MiB 26.4%, 64MiB 62.9%, 96MiB 95.6%** 감소. 96MiB는 첫 전환 **2,417.962ms**, 나머지 22개 warm switch 중앙값 **0.284ms**, warm 누적 **6.246ms**로 나타났다.

96MiB 실험의 전체 wall은 Qv14 이전의 186.410초→136.842초로 **26.6% 단축**됐지만, 이는 동일한 시스템 부하에서 반복 측정한 통계적으로 안정적인 latency benchmark가 아니다. 32/64MiB 시점에 다른 OS/호스트 부하가 다르므로 wall 전체를 캐시 성능만의 인과관계로 주장하면 안 된다. 이에 비해 네이티브 스위치 자체의 내부 계측은 직접적이다.

**메모리 교환관계:** 96MiB의 실제 inactive cache 최대는 90.05MiB이고, 해당 실행의 peak process RSS는 4,725.86MiB다. 참고 정적 Qv8 원본 24문항의 RSS 약 4,638.91MiB보다 **86.95MiB 높다**. 모델 두 벌의 중복은 없지만, inactive per-tensor cache는 **추가 RAM 비용**이며 동일 메모리 예산의 무조건적인 승리 주장은 불가하다.

## 5. 과학적 판정과 다음 시험

**PASS**: Apple Silicon CPU에서 실제 동일 프로세스 요청별 WT Q4/n8 전환, 기존 원본 C 수치정확도 보존, 32/64/96MiB inactive 캐시 상한 준수, 96MiB에서 warm 전환 ~0.284ms 및 전체 전환 누적 95.6% 개선.

**NOT_YET_PROVEN**: (1) 현재 토큰 길이 휴리스틱이 독립 집합에서 정답률/conditional four-choice NLL을 향상시킨다는 주장, (2) 다른 모델 구조/계층 수에서 작동한다는 주장, (3) 멀티클라이언트 동시 실행에 대한 안전성 및 실제 대기시간 SLA, (4) Metal GPU 텐서 전환 및 unified residency, (5) 첫 캐시 미스 약 2.42초 자체의 제거, (6) 32/64MiB 캐시 정책이 최적이라는 주장.

**후속 개발 게이트:** 
1. **비용가중 캐시 정책:** 고정 슬롯 채택 순서를 넘어, 텐서별 실제 재양자화 latency/규모를 사용해 32/64MiB cache hit 효용(회피한 switch-ms)/MiB을 최적화하고 또 다른 독립 부하 순서에서 비교한다. 캐시 cap 및 실제 프로세스 RSS 모두 통제한다.
2. **초기 워밍 경로 분리:** 독립 스레드/백그라운드가 아니라 연구용 명시적 startup 초기화에서 대체 Q4/n8 포맷을 계산해 캐시를 미리 채우는 실험. Startup 가중치 변환 비용을 사용자 요청 지연에서 *옮긴 것*이지 계산을 없앤 것이 아님을 분리 기록한다. 상주 메모리는 96MiB cap을 초과하지 않아야 한다.
3. **새 품질/정책 검증:** Qv13의 test 256과 Qv14 validation4/24는 새로운 정책의 독립 성능 확증 집합으로 재사용하지 않는다. 별도 calibration train으로 품질·꼬리위험·선택지 조건부 NLL을 함께 추정하고, 사전등록한 동적 할당기로 새로운 OOD/heldout 데이터에서 static Q4/고정 V2/고정 Qv8/동등 바이트 random과 비교한다. 정확도 악화·메모리 또는 지연 제한 초과 시 fail closed.
4. **CPU/Metal 분리 판정:** 현재 Qv15는 원본 Apple C/NEON CPU 전용 연구 실험. GPU 실행·컨커런시·KV cache 안정성/배포는 아직 별도 미검증이며, 연구분기에서 실측 후 결정한다.
5. **운영 불변:** BF16 원본 체크포인트, 기존 Qv14 source/binary, Tailnet immutable 운영 릴리스, production serving 또는 BEGLIN main 자동승격 변경 없음.

## 6. 재현 도구 및 공개 영수증

- Qv15 독립 코드: `tools/qv15_build_bounded_cache.py`, `tools/qv15_build_paired_cache_v2.py`, `tools/qv15_run_paired_cache24.py`, `tools/qv15_independent_paired_cache24_audit.py`.
- 비공개 원본 logits/token/log: `/Volumes/D50/TailnetCommander/work/beglin-qv15-bounded-cache-20261023/native_24_runs_paired_v2/cap{32,64,96}/`.
- 모델·input SHA binding: Qv14 frozen validation24 private manifest SHA-256 `e836a4bb6ea86c43f5a9c28ecbbc235c4c9d7fd306dfc006511027e3bddf3dc6`.
- 전체 비용-성능/실측 원문 재검산 집계: `BEGLIN_QV15_PAIRED_32_64_96_MEMORY_LATENCY_FRONTIER_PUBLIC_20261023.json`, SHA-256 `7f9430f97dfcb219f39563784fad98cfeaf433cbaf50616e85b7ad757c0a57c8`.
- 독립 cap32 집계 SHA `042c4975ec8a490f005142555c684555df582ed7a6ebe0c915c736699b7a792d`.
- 독립 cap64 집계 SHA `e480f070ab25196a7549e37f11399027f4aefe312d738a3922bf76f13da3af51`.
- 독립 cap96 집계 SHA `01437d56097d341de2a118786ba6534b170490073e6c2ec5af86723e652cd81c`.
- 모든 영수증은 공개 집계만 저장하며 사용자 원문·토큰·문항별 개별 NLL은 비공개 D50에 보존.

**최종 한 줄:** **국소 동적 정밀도 전환 자체와 제한된 추가 RAM을 이용한 warm 전환 가속은 이제 실제 CPU에서 입증됐다. 품질 최적 할당, 동일 RAM Pareto 및 GPU 서비스 준비도는 여전히 별도 미입증이다.**
