# Beglin HCP P8 — XOX native follow-up, 2026-10-14

## Verdict

XOX MacBook Pro M1 Max에서 P5 CB128 단일 GPU 슬롯 연구 바이너리 두 종을 새로 컴파일했다. 하나는 원본 수식 그대로인 대조군, 하나는 가역적 9구간 관측 기능이 추가된 P8 버전이다. 두 개의 112토큰 이하 MLX 워커를 동시 사용하지 않고 순차적으로 실행했으며 운영 바이너리·라우트·PID는 변경하지 않았다.

**실제 12개 분야 전환 35건에서 캐시 적중 34건의 MLX 바인딩 경로가 entry-to-ACK 평균의 99.72%를 차지했다.** 이는 해당 함수의 5·6비트 재포장 및 할당 경로를 다음 최적화 후보로 좁히는 실측 증거다. 함수 안에서 할당/비트루프/MLX 배열 바인딩 각각의 비중은 아직 측정하지 않았다.

## P8 9-phase XOX observation

입력: P6와 같은 DeepSeek-V2-Lite 12분야, 106–112토큰, 생성 예산 4, GPU slot 1, 원본 tokenizer/checkpoint index/P4 prompt SHA와 P6 생성 시퀀스 SHA 비교. P8 계측 전후 **8/8 정책·입력의 정확한 생성 해시 일치**, 계측 확장 12분야에서 **35/35 native REBIND_SET 전환 구간과 ACK 시점 확인**.

| Source-timed segment | 평균 ms / 35회 |
|---|---:|
| Precheck | 0.0037 |
| GPU synchronize | 0.0996 |
| Binding snapshot | 0.0041 |
| qNg64 CPU cache lookup or build | 9.4379 |
| **MLX bind with allocation/repack path** | **329.5969** |
| Binding verification and loop residual | 0.0013 |
| Runtime epoch reset | 0.1139 |
| Metadata update | 0.0298 |
| Durable ACK | 0.6251 |
| **Entry through durable ACK** | **339.9122** |

**Cold/miss 1회:** qNg64 캐시 단계 328.340ms, MLX bind 312.360ms, total 641.447ms. **Warm/hit 34회:** qNg64 cache 0.0584ms, MLX bind 330.1039ms, entry-to-ACK 331.0435ms. MLX bind의 warm 총계 점유율 99.72%. 계측 기록 작성 시간은 구간 합계에서 제외했다.

### 계측 종료 예외 — 숨기지 않음

원본 9구간 실험에서 12개 분야의 결과 파일과 raw worker 정리는 끝났지만, 마지막 receipt 파일명 결합 표현의 PosixPath + str TypeError로 **실험 프로세스 exit code는 1**이었다. 자료를 재실행하지 않았다. 별도 독립 재검사 도구 hcp_p8_seal_existing.py가 이미 완성된 parity.json과 native-phases.json의 SHA, 35개 구간합, 12분야, 정확한 stage·route SHA와 live PID를 확인하여 research-receipt-profile.json을 새로 생성했다. 이 영수증은 postrun error를 명시하며 exit 0으로 소급하지 않는다. 파일명 버그를 고친 뒤 별도 성능 실험은 exit 0을 확인했다.

## Uninstrumented native five-arm comparison

같은 P5 C/MLX 수식과 단일 슬롯 바이너리에서 각 정책별 24개 요청, 합계 **120회 실제 추론**. 원본 P6 A/B 생성 SHA와 전수 대조, AF는 동일 입력의 2회 생성 해시 반복 일치.

| Research arm | 전환/24회 | 첫 요청 제외 23회 평균 ms | 첫 요청 제외 p95 ms |
|---|---:|---:|---:|
| Fixed original AF | 0 | 557.58 | 584.56 |
| Fixed A | 0 | 562.80 | 591.95 |
| Fixed B | 1 (first) | 559.95 | 587.02 |
| Alternate A/B every request | 23 | 884.33 | 1028.98 |
| Minimum hold 8 | 2 | 598.36 | 852.85 |

각 arm 첫 요청 초기화 비용은 약 3.85~5.81초였고 위 평균에서 제외했다. Source trial 순서는 seeded random order이지만 전략별 별도 워커 순차 실행, 12문항 재사용, 블록 2개뿐이다. 숫자는 기술통계이고 후보 간 통계적/인과적 성능 개선 증명이 아니다. 최소 유지 정책은 잦은 교대보다 부담이 작았으나 고정 AF/A/B보다 빠르다고 입증하지 못했다. B 품질 승격 권한 없음. Governor는 운영 워커에 연결하지 않았다.

## Identity and immutable boundaries

- Beglin repo: popixoxipop-collab/beglin
- Base research commit: 9d55fff527aae34bd2568b149661e52cc75a8504
- P5 plain C: bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612
- P8 observed C: 84ff47596d6912741b8dcd14c8bdb2028f09031c4c48b78272314a4374d88edf
- MLX KV128: daded3c0e4bf7cc2a86e0fab8c1f73322b5b0ffea0002d9b2d38857d2e9591ae
- Real P5 control GPU binary: ca376b07827471243d28cec3ca356fd71cdc2f8aa980f14a153602335dae644e
- Real P8 observer GPU binary: 8c234baa330c8f08925dd802574720a0c7419197e5dc006a6b0d92ebc0182200
- Original active-stage binary unchanged: 8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9
- Original active-route unchanged: c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47
- Native serving PIDs before/after: 35481 and 35493. No restart. No main merge or runtime release activation.
- XOX storage gate fixed by deleting only regenerable uv/Homebrew/IDE/build caches; source research data, original model weights and operational files preserved. resource_plan returned eligible=true, measured free around 23.2GB with 20GB reserve intact.

## Code/evidence files

New driver: tools/hcp_p8_native_trial.py; independent exit1 receipt recovery: tools/hcp_p8_seal_existing.py. Copies of exact XOX JSON and source-backed evidence hashes are kept in experiments/hcp_p8/evidence/native_xox_20261014. The manifest has SHA-256 for all six copied evidence JSONs. No raw prompt IDs, logit vectors, checkpoint bytes or generated token IDs are committed.

Prior P8 CPU tests: 48/48 PASS; real GPU full native build two variants PASS; observer/control parity 8/8; true switch traces 35; true performance requests 120/120, performance runner exit 0; original profiling runner exit 1 post-measurement, independent validated recovery PASS.

## Next research gate — not yet performed

1. Add separate isolated timing inside MLX bind for buffer allocation, bit-plane-to-native bit packing and array construction/registration. Verify instrumentation output parity and math-invariance before measuring.
2. Create a bounded prepared MLX packed-representation cache keyed by exact model/tensor/bitwidth/shape/source identity, with tensor lifetime and memory-residency checks. Never reuse stale pointers. Validate the original snapshot/rollback and durable ACK contracts fail closed. No production code updates.
3. Compare cache-on/off under identical DeepSeek prompt SHA and same 12 domains, then an independent unused holdout. Include cold memory cost, warm transition amortization, p50/p95 admission latency, RSS, exact AF/A/B output equivalence and full error recovery.
4. Require separate quality authorization, independent holdout, multi-slot/restart bounds and immutable release certification before any activation. Current B speed or general model quality improvement remains unproven.

**Final classification: actual XOX MLX binding bottleneck measured; no newly demonstrated production speedup or promotion.**
