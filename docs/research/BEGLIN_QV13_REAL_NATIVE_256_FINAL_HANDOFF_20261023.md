# BEGLIN Qv13 — 실제 Qwen28 네이티브 256문항 최종 검증 및 Qv14 동적 할당 게이트

**일시:** 2026-10-23 (KST)  
**연구 저장소:** `popixoxipop-collab/beglin`  
**연구 브랜치:** `research/qv13-byte-budget-20261023`  
**평가 상태:** `16/16 REAL_NATIVE_C_PASS / 2048/2048 GOLD-NLL ROWS INDEPENDENTLY REPARSED / Qv8 MEAN-NLL IMPROVEMENT VS FIVE EQUAL-BYTE RANDOMS EXPLORATORY / ACCURACY IMPROVEMENT NOT DEMONSTRATED / DYNAMIC SWITCHING NOT DEMONSTRATED / NO_LIVE_PROMOTION`

## 1. 검증된 사실과 데이터 출처

- 실제 EOE Apple Silicon CPU의 원본 BEGLIN C 엔진, Qwen2.5-1.5B-Instruct 28 layers, projection tensor 196개.
- 모델 HF 원본 revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Q4 원본 C 바이너리 SHA256 `408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c`, 혼합 28n8 원본 C SHA256 `f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221`.
- 원본 prelock SHA256 `5cb944375fa271c4528b4846f4b3c22cdf8179d4a38a2f5c17babbc03c4df92a`. 비공개 사전실행 입력/마스크 봉인 SHA256 `79826eb944ac2a4f87b637a430471f069a3789da37cdb12a9b7d1060d49db9b8`, 공개 봉인 SHA256 `9d045e1056e5ec8bc3a093a8a3d198277cbbc4083fb054a929b566cb2bb03c8c`.
- 평가 데이터는 ARC-Challenge **test 128문항**, OpenBookQA **test 128문항**. 처음 8+8=16개 파일럿은 평가 입력 전체 256문항의 부분집합이며 사전실행에 고정됐다. 파일럿 이후의 분석은 이 16개를 제외한 **ARC 120+OpenBookQA 120=240문항**을 따로 집계한다. Qv11/Qv12 RACE 데이터의 성능값을 Qv13 마스크 재선택에 사용하지 않았다.
- 실사용 정밀도 마스크 원본: Qv8 `9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922`. 28 n8 투영 역할은 Qv8과 seed 73,137,211,389,577의 무작위 대조군 5개가 **동일한 역할별 개수** 및 packed projection 비용을 갖는다. 무작위 대조군은 **계층 선택만 달리한 것**이며 역할 구성 최적화의 검증은 아니다.
- 저장 위치: EOE private original case/token/NLL/native logs `/Volumes/D50/TailnetCommander/work/beglin-qv13-20261023`. 원문 지문, 토큰, 정답 및 개별 문항 NLL을 GitHub에 올리지 않았다.

## 2. 원본 C 전체 평가 결과 (각 256문항)

정답률은 4개 선택지 A/B/C/D 중 최고 로짓 선택이다. NLL은 **전체 vocab(151,936개) softmax의 정답 single-letter next-token NLL**이며 **4선택지 조건부 정규화 NLL이 아니다.**

| 조건 | 원본 28계층 n8 역할 개수 | 정답/256 | 평균 gold-token NLL (↓) | packed projection bytes |
|---|---:|---:|---:|---:|
| 전체 Q4 | 0 | 159 | 2.010862 | 736,985,088 |
| Qv8 source 28n8 | 28 | 152 | 1.990176 | 765,493,248 |
| random 73 | 28 | 159 | 2.261043 | 765,493,248 |
| random 137 | 28 | 154 | 2.096505 | 765,493,248 |
| random 211 | 28 | 150 | 2.244450 | 765,493,248 |
| random 389 | 28 | 156 | 2.072044 | 765,493,248 |
| random 577 | 28 | 164 | 2.174107 | 765,493,248 |
| **기존 V2 28n8** | 28 | **165** | **1.605793** | 776,896,512 |

- 같은 packed byte 및 역할 개수에서 **Qv8은 무작위 5개 모두보다 평균 NLL이 낮았다**. 256문항에서 Qv8=1.990176, 무작위 5개 평균=2.169630, 차이 **+0.179454(무작위−Qv8)**. 문항 쌍 부트스트랩 10,000회 탐색적 95% 구간 **[+0.066662,+0.295792]**.
- Qv8 vs 전체 Q4 NLL 차이 = Q4−Qv8 **+0.020686**, 탐색적 95% 구간 **[−0.135305,+0.176653]**. **정답은 Qv8이 7개 적다**.
- V2는 Qv8보다 packed 가중치 비용 **11,403,264 bytes(약 10.875MiB)**가 더 들지만 평균 NLL이 0.384382 낮고 정답이 13개 더 많았다. 메모리 예산이 다르므로 V2와 Qv8을 equal-byte 승패로 취급하지 않는다.
- 5개 무작위 정책의 평균 정답 수는 156.6개. Qv8 정답 152개는 이에 못 미친다.
- native wall time은 모델 로드·주변 EOE 부하와 입력 길이를 포함한다. 안정적 TTFT, TPOT, p50/p95, 지속 TPS 증명으로 사용하지 않는다. **CPU 경로 결과이며 Metal GPU/런타임 정밀도 전환의 검증이 아니다.** 최대 프로세스 RSS는 대략 Q4 4,610~4,612MiB, Qv8 4,637~4,641MiB, V2 4,648~4,651MiB로 별도 관측했다.

## 3. 파일럿을 제외한 240문항: 탐색적 추가 검증

| 고정 정책 | 정답/240 | 평균 gold-token NLL (↓) |
|---|---:|---:|
| Q4 | 151 | 1.954247 |
| Qv8 | 144 | 1.936372 |
| 5개 동등 바이트 무작위 평균 | 148.8 | 2.094411 |
| V2 | 157 | 1.550577 |

- 동일 바이트의 무작위 5개 평균−Qv8 NLL 차이 **+0.158039**; ARC/OB 각각 120문항 내에서 재표집한 10,000회 stratified paired-bootstrap 95% 구간 **[+0.042709,+0.278358]**.
- 5개 개별 random 대비 source 이득의 탐색적 Holm-adjusted one-sided sign-flip p: random211 약 0.021, random73 약 0.039, random577 약 0.162, random137/389 각각 약 0.313. 이 부호 뒤집기 검정은 귀무가설 아래 pairwise sign exchangeability가 필요하며, **사전등록 그대로 재현된 확증 시험은 아니다**.
- Qv8은 Q4보다 정답 7개 적고, random577보다 12개 적고, V2보다 13개 적다. 문항별 정답률 개선을 보이지 않았다.

### 특히 주의할 분포적 함정: 평균과 문항 승률이 불일치

Qv8과 random73의 240문항 개별 골드 토큰 NLL 쌍 비교에서 **Qv8이 더 낮은 문항 89, random73이 더 낮은 문항 151**이었다. 그러나 Qv8의 평균 NLL은 더 낮다(Qv8 1.93637, random73 2.16489). 이점이 **모든 입력의 소규모 일관 개선이 아니라 일부 높은 NLL 오류의 완화**에 상당히 의존한다는 뜻이다. 
- paired difference `random73 NLL − Qv8 NLL` 평균 +0.22852, **중앙값 −0.00029**, 10% trimmed mean +0.15974.
- 반면 Qv8 vs V2의 같은 차이는 `V2−Qv8` 평균 −0.38579, 10% trimmed mean −0.23598. V2의 우위는 단순 극단값만의 결과로 보기 어렵다.
- 랜덤 정밀도 마스크가 gold-token NLL에서 실패하더라도 top-4 선택지 정답에는 영향을 적게 주는 경우가 있으므로, **NLL 향상은 정답률 및 실효 추론 성능 향상을 대신하지 않는다**.

## 4. 독립 검산과 영수증

- 장기 실행 Tailnet job ID: `job_8a24620f416ea11304645700c7f3c8c2`, **SUCCEEDED / exit 0**.
- 실제 분할 native: ARC128 8×128=1,024행, OpenBookQA128 8×128=1,024행, **총 2,048/2,048 원본 C 로그 행 재파싱 PASS**.
- 각 입력의 `wN.i32`, `wN.choices` SHA 및 정답 토큰과 C 출력 인덱스/정답률, 28n8+168Q4 또는 0n8+196Q4, 원본 바이너리, mask SHA, 16개 receipt+source log SHA, 원본 미변경을 독립 검산.
- 재현 코드:
  - `tools/qv13_native_256.py` (실제 C 평가)
  - `tools/qv13_full256_sequential.py` (결과 기반 선택을 허용하지 않는 고정 순서 실행)
  - `tools/qv13_native_log_crosscheck.py` (원본 로그 직접 독립 검산)
  - `tools/qv13_native_independent_audit.py` (256문항 영수증 집계)
  - `tools/qv13_equal_byte_holm.py` (5개 동등 바이트 대비 분석)
  - `tools/qv13_postpilot_heldout240_audit.py` (파일럿 제외 240문항 층화 집계)
  - `tools/qv13_postpilot_robustness_audit.py` (중앙값/trimmed/빈도·꼬리)
- 공개 집계 영수증: `BEGLIN_QV13_REAL_FULL256_8ARM_INDEPENDENT_PUBLIC_20261023.json` SHA `2c4e1e22b57fedec1189475fd19ba00c16582a8e29cc336b057d1dcd6d07f856`.
- 240문항 집계: `BEGLIN_QV13_POSTPILOT_240_STRATIFIED_PUBLIC_20261023.json` SHA `6a42ccbe8314d1e958e9b866128f017af96696591c4bf6b3ab6789e6675cc5d3`.
- 256문항 동일 비용 비교: `BEGLIN_QV13_EQUAL_BYTE_5RANDOM_HOLM_FULL256_PUBLIC_20261023.json` SHA `36c8bd8f1177d7db4cc9e751af1f6700dd19726c3299a5eeab5d52d958a3ecee`.
- 240문항 강건성: `BEGLIN_QV13_POSTPILOT_240_ROBUSTNESS_PUBLIC_20261023.json` SHA `fe4096b154c01ce5a8b4841ccd4e10339bdd7415b21f6d6e7f654baa9a3d88f9`.

## 5. 연구 판정과 범위 제한

**검증 완료:**
1. 동일 packed projection byte 및 역할 개수에서 고정 Qv8 층별 정밀도 배치가 **5개 무작위 배치 대비 평균 gold-token NLL**을 낮춘다(이번 ARC/OB 표본, 탐색적 관측).
2. 전체 28계층 네이티브 CPU 경로에서 정확히 28개의 n8 텐서와 168개의 Q4 텐서가 실행되는 것이 원본 C 로그로 검증됐다.
3. 사전고정된 원본 입력/정밀도/결과 로그의 해시 연결과 독립 재파싱을 통과했다.

**검증되지 않음:**
1. Qv8이 전체 Q4 또는 고정 V2보다 정답률이나 서비스 효율을 개선한다는 주장.
2. 매 요청/문맥에 따라 텐서를 자동으로 골라 n4↔n8을 **실제로 전환**하는 동적 정밀도 할당.
3. 총 기기 RAM, TPS, TTFT/TPOT, Metal GPU, 다른 모델 계열/더 긴 문맥에서의 우수성.
4. 모델 원본 사전학습에서 public benchmark가 노출되지 않았다는 보장.

**재현성 정정:** 원본 prelock의 잘못된 Qv8 해시 표시, 역할별 추첨 순서 및 문항 정렬 해시 설명이 실제 봉인 구현과 달랐음을 `BEGLIN_QV13_PREEXEC_PROVENANCE_ERRATA_20261023.md`에서 보존·기록했다. 원본 prelock 파일·실제 SHA-봉인된 정책/평가 자료는 이 문제를 덮기 위해 변경하지 않았다. 파일럿 결과 관측 뒤 정의한 240 분할 분석은 탐색적 성격이다.

## 6. 다음 Qv14 실증 게이트 (계획만; 아직 실행하지 않음)

1. **훈련/보정 원본 분리:** 아직 보지 않은 개발용 문항으로 per-tensor 출력민감도, 실제 packed-byte·NEON 지연 예측 및 정확도-꼬리위험 복합 손실을 보정한다. Qv13의 256개 test 결과를 하이퍼파라미터 탐색에 재사용하지 않는다. Qv13의 극단적 일부 NLL 성공만 새 정책에 맞춰 과적합하지 않는다.
2. **동일 자원 예산 3점 이상:** 정확히 계산한 projection packed bytes와 실행 시 실제 모델 residency/RSS 모두 통제. static Q4, static V2(예산 다름), training-source 고정 28n8, equal-budget random ≥5, 새로운 budget-constrained allocator를 비교한다. **최적 텐서 개수뿐 아니라 역할, 계층, 출력 위험의 결합 최적화**를 시험한다.
3. **진짜 요청별 동적 전환:** context-only 신호로 bit-allocation mask를 선택해 실제 네이티브 커널에 적용; 안전한 모델 로딩/재패킹/스위칭 시간·추가 메모리와 cold/warm latency, 모델 residency를 기록. 복수 사전패킹 가중치를 전부 RAM에 상주시켜 숨겨진 메모리 예산을 늘리는 방식을 금지한다. CPU와 Metal GPU는 별도 판정.
4. **새 확증 데이터와 품질 게이트:** 이전 Qv11~Qv13 평가와 독립된 문항/도메인에서 accuracy + **4-choice conditional NLL** + full-vocab NLL + worst-tail + calibration + paired CI/Holm을 함께 평가한다. 사전등록 exact SHA/코드·원문·정렬 방식을 일치시킨다. 정답률이나 지연이 나빠지면 동적 개선·운영 승격 **NOT_READY**.
5. **운영 경계:** BEGLIN 엔진 제품 main 병합, Tailnet immutable release, production 자동 정밀도 승격, 원본 BF16/커널 변경은 본 Qv13에서 **전혀 수행하지 않음**.

**한 줄 결론:** BEGLIN은 고정된 텐서별 혼합 정밀도 구조에서 같은 가중치 예산의 무작위 층 배치보다 낮은 평균 full-vocab 정답 NLL을 내는 실제 CPU 증거를 확보했지만, 그 성과가 **정답률 개선이나 요청별 동적 정밀도 선택의 우월성**이라는 증명은 아니다.
