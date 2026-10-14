# Beglin HCP P14 — 실제 이중 GPU 워커 병렬 추론·동료 충돌 생존, BF16 전체 교사 자원 게이트

작성일: 2026-10-14 (KST)
연구 상태: **격리 네이티브 다중 워커 연구 PASS / BF16 전체 교사·운영 승격 BLOCKED**

- 저장소: `popixoxipop-collab/beglin`
- 연구 브랜치: `research/hcp-p14-multiworker-crash-gate-20261014`
- 기준 P13 커밋: `dc4d2dbc999633e75ff5ff0a430642ceefc830b3`
- P14 실험 전 GitHub 사전등록 커밋: `93ac73bb6feefe4bf020b7882fbc0bf3d8d977e7`
- DeepSeek-V2-Lite / XOX Apple Silicon / 단일 GPU 장치의 **서로 다른 2개 네이티브 프로세스**; 각자 단일 GPU 요청 슬롯
- `main`, 운영 stage, active-route, Release Plane 및 실제 서빙 워커 미변경

## 결론

P13에서 P9 네이티브 가중치 패킹 SHA의 BF16 원본과의 4/4 완전 일치를 증명했다. P14에서는 그보다 한 단계 나아가 실제 **서로 다른 Python 제어자 2개와 각각의 실제 MLX GPU 워커 2개를 동시 운영**하고, 서로 반대되는 정밀도 A/B 전환 명령을 동시에 수행했다.

총 **24쌍 / 48개 실기기 네이티브 추론**이 성공했다. 독립 제어자 프로세스의 실제 admission 처리 시각 구간이 **24/24쌍 모두 겹쳤고**, 한 워커의 A/B 전환이 다른 워커의 정책과 섞이지 않았다. 모두 기존 P6 실기기 생성 SHA와 실제 native REBIND_SET epoch/txn ACK를 검사했다. 두 워커의 초기 최대 합산 RSS는 **11.71207 GiB**, 사전등록한 12GiB 상한 내였다. 정확도·가속이 일반 모델이나 새로운 데이터로 확장됨을 보증하지는 않는다.

이어서 정상 워커 하나를 그대로 가동한 상태에서, 동료 연구용 워커를 **native REBIND_SET의 변경 완료 후 내구성 ACK 직전**에 실제 SIGKILL했다. 다른 워커는 **같은 PID로 생존했고 6개 새 요청**을 원래 P6 생성 SHA와 일치하게 처리했다. 중단된 원래 명령은 완료/실패 어느 쪽으로도 간주하지 않고 **UNKNOWN**으로 분류하여 P11 fsync INTENT·VERIFIED로 격리하고 재제출하지 않았다. 충돌 워커는 새로운 PID와 정책 A에서 시작한 뒤 새 B→A→B 요청 **3개**를 성공적으로 처리했다.

이는 **실제 2개 MLX GPU 워커의 정책 분리, 살아남은 워커의 추론 지속, 장애 워커의 독립 격리·신규 요청 복구**를 입증한다. 같은 호스트의 단일 GPU에 대한 2워커 한정이며, 프로덕션 동시 다중 사용자, 실제 전원 손실, 임의 규모의 GPU 슬롯, 결과 불명 원래 요청의 exactly-once 처리 계약을 입증하지 않는다.

## 1. 소스·자원 계약

실험용 XOX 소스에 원본 P5 CB128를 적용하고, 정확한 P9 MLX 캐시 코드를 사용했다. P9/캐시 소스와 독립된 P11 before_ack 실패 주입 C는 별도 소스/바이너리로 컴파일했다. 양쪽 모두 과거 검증된 원본 SHA와 바이트 일치:

| 구분 | SHA-256 |
|---|---|
| P5 C CB128 | `bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612` |
| P9 MLX KV128 | `ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f` |
| P9 연구용 실제 바이너리 | `103549b7f7f4a3d142be883013956b4f1437759c8be76bff31f2451956df37fd` |
| P11 before_ack 실패 주입 C | `3d5a6d9f2f63d53e960c14a88dc4892ffa27cfd67b6d15258b6c930a7c17f2bd` |
| P11 실패 주입 연구 바이너리 | `a5fd3f590ad6bbddcb69e690ccf738b711b8f7ceb3319f99ee933123f5ac7e78` |
| 원본 DeepSeek BF16 인덱스 | `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf` |

다중 워커 실측 전 `resource_plan`이 XOX eligible=true를 반환했고, 연구용 두 워커 합산 12GiB RSS 상한, 단일 워커 7GiB 상한, 내부 SSD 여유 20GB+300MB를 명시적으로 시행했다. GPU 요청 슬롯은 1+1=2로 제한. 원본 운영 DeepSeek PID `35481`, `35493`와 원래 stage/route는 변경하지 않았다.

## 2. 실제 동시성 측정

- P6 원본에서 SHA 고정한 6개 분야 × 4회, 두 제어자는 각 요청에서 서로 반대되는 A/B 정책을 선택했다.
- 각 제어자는 **별도의 OS 프로세스**이고, 각자 `AdaptivePersistentRouteWorker`와 독립 `txn.cmd`, `applied_ack.json`, `queue` 경로를 사용했다.
- 총 24 대응쌍 / 48 native admissions. 실제 요청 구간 겹침 **24개**. 모든 생성 시퀀스 SHA·정책 해시·epoch·터미널 ACK 검사 통과.
- 원본 P6 6개 분야는 이미 사용했던 데이터다. 이 시험은 새 holdout의 답변 품질을 측정하지 않는다.
- 이 실험은 GPU 경합 환경이므로 P9의 34.87% 전환 가속을 재검증하는 latency benchmark로 사용하지 않는다.

## 3. 동료 워커 충돌·격리

정상 워커 PID를 유지한 채 동료 연구용 P11-before-ACK 바이너리를 시작했다. P11의 원래 fsync marker를 확인한 뒤 **동료 네이티브 PID만** 죽였고, 원래 트랜잭션에 완료 ACK가 없음을 확인했다. 명령 원본 SHA는 `6463a471f5bb77b60f6699483c139a2bf4cda654cabe7467d3f60adf43ebb9db`.

- 실제 충돌 지점: `before_ack`; marker SHA `b57807c4cef1d1e53f40d248476139f7c1aefe4870c52e70bed1dace640a5da5`.
- INTENT SHA `8e4b5e47a74420ab011deb74ac72518cb28c1ad481ef07a44e6ee29febac41b5`.
- VERIFIED SHA `02d779515be3d45dc29059270a13b1e5547ee22381ff81ae6778e8fa19426cc9`.
- UNKNOWN 원래 거래 재시도 0회. **새로운** B→A→B 3개 요청은 원본 P6 SHA·ACK 통과.
- 정상 워커 같은 PID·6개 추가 추론 성공. 강제 중단 당시 동료 워커가 동일 GPU에서 추가 요청을 보내도 정책이 섞이거나 서빙 PID가 재시작되는 오류 미관측.

**원래 충돌 중단 요청의 정확한 최종 적용 결과는 UNKNOWN이다.** P11과 마찬가지로 exactly-once를 주장하지 않는다.

## 4. 전체 BF16 교사 자원·아키텍처 검토

원본 DeepSeek-V2-Lite safetensors 4개 헤더를 값 로드 없이 읽어 모든 텐서의 모델 인덱스 매핑/shape/dtype/offset을 검증했다.

| 항목 | 실제 검사 |
|---|---:|
| 원본 체크포인트 파일 바이트 합계 | 31,413,626,576 bytes (29.26GiB) |
| 텐서 메타데이터 | **5,291개** |
| Transformer 레이어 | **27개** |
| 가장 큰 레이어 가중치 | **약 1.0894GiB** |
| 메타데이터 검사 프로세스 RSS | **약 23.56MiB** |
| 전체 BF16 모델 로드 또는 forward | **미실행** |

Transformers의 DeepSeek-V2 모델 모듈과 Accelerate가 설치돼 있음을 확인했다. 즉 BF16 가중치를 전부 상주시키지 않고 **레이어별로 적재·해제하는 streaming teacher**는 실험 설계상 가능성을 검토할 수 있다. 그러나 정확한 attention/MoE routing/layernorm/KV·tensor lifetime을 구현·검증하는 스트리밍 실행기는 존재를 확인하지 못했다. 현재 XOX 가용 RAM 약 23GiB, 기존 swap 약 21.84GiB와 운영 중인 모델 워커들을 고려해 전체 29.26GiB 교사를 강제 적재하지 않았다. BF16 full-vocab logits, teacher-score, 모델 품질 비열등성은 **NOT MEASURED / BLOCKED**다.

## 5. 검증·증거 원본 분리

- XOX에서 **요청별 원본 48개 P6 생성 SHA**, 24개 중첩 구간, 각 워커 별 native ACK와 PID, 중단된 동료 거래의 marker/command/intent/receipt를 읽은 별도 감사기를 실행하고 PASS. 원본 raw 실행 JSON SHA: `98aa76b19129a7634061595e98463267f506435069c3714db7f2b0a3ff7e9a71`.
- 원본 JSON은 XOX 영속 로컬 경로 `/Users/xox/mcp-sandbox/tailnet-commander/hcp-p14-native-forensic-20261014/`에 SHA·원본 바이너리와 함께 보존. 아카이브 manifest SHA `b425aa8ba5ce7a2939cc8c2ec508024f020e1f0c127eb2afdab1ec4c5c736976`.
- EOE/GitHub에는 원시 요청별 원본 JSON 대신 XOX 감사 요약 SHA `236475031894ea9ef1df87ff7c9a46a36f770fa89e0cc15373254927aedee470`와 원본 marker, command/intent/verified, BF16 헤더 검사, EOE 추가 감사 결과를 보존했다. **EOE는 전체 48개 raw 요청을 다시 재연산하지 않았다.** EOE는 원본 요약 SHA와 실제 격리 명령의 포렌식 일치·BF16 출처만 재검증했으며, 이 감사 범위의 차이를 유지한다.
- P14 신규 EOE 변조/무결성 방어 16개 포함, P8~P14 전체 Python 회귀 **213/213 PASS**.

## 6. 남은 강제 차단 관문

1. **실제 whole BF16 forward/teacher logits**: 별도 안전한 layer-by-layer BF16 실행기를 개발하고 (모델 고정, 메모리·시간 상한, BF16 source SHA/shape 전수 확인), 동일 입력/position/full-vocab을 가진 quantized A/B와 RMSE·KL·top-1·정답률을 비교해야 한다. 지금까지 P12·P13에서 얻은 BF16 국소 5/6비트 오차 및 바이트 일치는 full-model teacher를 대신하지 않는다.
2. **장기간·실제 운영형 다중 워커 안전성**: 동일 cache를 공유한 환경의 핫 리로드, 워커 3개 이상, 전원/호스트 cold boot, 커맨드 저널 쓰기 도중 중단, 요청 result idempotency·exactly-once/UNKNOWN 대사 수명주기를 별도 인증해야 한다.
3. **외부 미사용 장문 품질 및 회귀**: 새 홀드아웃의 응답 품질, 긴 문맥, 긴 생성, tail p95 안정성, RSS·swap·GPU starvation 및 정적 AF/A/B 대비 실제 서비스 지표 검증.
4. `main`이나 운영 stage에 이번 P14 코드를 병합하지 말고, 별도의 release certification/stage/verify 및 fresh Current·Live·Runtime, expected-current CAS 후 수동 승인만 검토. 현재 **automatic_live_promotion=false**.

**결론: 독립 2워커의 실제 동시 정밀도 전환 및 동료 충돌 생존·격리 복구 성공. 전체 BF16 모델의 품질 비열등성 및 운영형 exactly-once는 미해결.**
