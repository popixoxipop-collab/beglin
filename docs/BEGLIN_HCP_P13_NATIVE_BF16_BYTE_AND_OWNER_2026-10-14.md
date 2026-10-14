# Beglin HCP P13 — BF16 원본 대비 네이티브 qNg64 바이트 일치와 제어자 경합 검증

작성: 2026-10-14 (KST)
범위: 원본 DeepSeek-V2-Lite / XOX MLX 단일 GPU 워커 / 2개 텐서 / 5·6비트.
판정: **격리 연구 PASS / 전체 BF16 모델 품질·운영 승격 BLOCKED**

저장소: popixoxipop-collab/beglin
브랜치: research/hcp-p13-native-bf16-byte-parity-20261014
P12 기준 커밋: 6ae561d1b4bb3cb83d4ba4e43c6cea9b257f94e3
사전 원본 SHA 동결 커밋: 5418310d178f42d68dde09fa398414ab91fb38d4

## 1. 핵심 결론

P12에서는 원본 DeepSeek BF16 행렬 두 개에 실제 C qNg64(n5/n6)를 적용하고 국소 투영 오차를 측정했다. 그러나 P12의 C 출력 바이트가 실제 P9 런타임의 GPU 바인딩에 전달되는 packed bit-plane 가중치와 일치하는지는 직접 검증하지 않았었다.

P13에서 **P12 원본 BF16 참조 4개(2개 행렬×2비트폭)의 패킹 바이트 SHA-256와 float32 scale SHA-256를 실제 P9 계열 GPU 네이티브 런타임에 직접 삽입된 opt-in 해시 관측으로 검증했다. 네 개 모두 바이트·scale 완전 일치**한다. 비트패킹/스케일 출력이 실제 추론 런타임 경로와 동일하다는 결손 증거를 보완했다.

다만 BF16 전체 모델 forward, 전체 vocabulary logits, 독립 미사용 정답률, 실제 작업 비열등성은 이번에도 측정하지 않았다.

## 2. 실제 BF16 기반 패킹 참조와 네이티브 소스

P13 실험 시작 전에 P12에서 생성한 참조 결과 `p12_local_BF16_oracle.json`(SHA `104d8112a737515f7bd52ae187735940e5948afebea78601cb61a73e58484de4`)에서 4개 representation의 packed/scales SHA를 동결하고, GitHub 사전 커밋 `5418310d178f42d68dde09fa398414ab91fb38d4`로 올렸다.

원본 DeepSeek safetensors index SHA: `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`.

| 대상 | 비트폭 | 원본 BF16→C 패킹 바이트 | 원본 BF16→C scale 바이트 | XOX 실기기 SHA |
|---|---:|---:|---:|---|
| L3 shared_up_proj (2816×2048) | 5 | 3,604,480 | 360,448 | **패킹·scale 일치** |
| L3 shared_up_proj (2816×2048) | 6 | 4,325,376 | 360,448 | **패킹·scale 일치** |
| L26 shared_down_proj (2048×2816) | 5 | 3,604,480 | 360,448 | **패킹·scale 일치** |
| L26 shared_down_proj (2048×2816) | 6 | 4,325,376 | 360,448 | **패킹·scale 일치** |

참조: `experiments/hcp_p13/P13_REFERENCE_CODES.json` SHA `50d222534ab1324b990a2c4d83cc227e0ca2ba53c2037b500d688f742caeb078`.

실험용 C 변형은 기존 P5 CB128 원본 `bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612`에서 해시 출력 코드만 추가한 버전 `26e542118302a5a82f147ef1678ec0f6e4516b467d0a4166cc1bc7dce29fa8c6`. 해당 부분을 제거하면 원래 C 바이트로 역변환됨. P9 MLX KV128은 `ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f` 그대로이며, C와 MLX 수학·qNg64 양자화·ACK를 변경하지 않았다. 관측은 연구용 워커 환경변수 `BEGLIN_P13_PACKED_SHA=1`일 때만 활성화.

실기기 바이너리 SHA: `657acc942b39f85307f04857b2e8fdd86ebe776a8855af3928e7050c6ff9d065`. 해당 바이너리는 기존 운영 stage와 다른 연구용 전용 빌드다.

## 3. XOX 실제 추론

- P6와 동일한 DeepSeek-V2-Lite 기존 6개 분야 입력을 사용. 출처 SHA는 기존 P6 quality `2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104`.
- 1GPU 슬롯, 별도 워커에서 동일 바이너리로 해시 OFF 24회·해시 ON 24회, 한 실험당 **48개 실제 admission**. 입력마다 B→A→B→A 정책 전환과 실제 `REBIND_SET_APPLIED` txn·epoch·두 대상·generated sequence SHA를 확인.
- 첫 실험 48/48 PASS, Native qNg64 packed SHA 로그 48건 중 고유 4개 표현 정확히 일치. 결과 SHA `d9cd613a68be1e93101514ae9229c120e9b738909c3ada61ab7bb645d95a9de0`.
- 별도 재실행 48/48 PASS로 요청별 생성 SHA 기록을 보완했다. 독립 EOE 감사로 **48개 실제 admission 생성 SHA를 P6 원본과 전수 대조**, 4개 packed·scale SHA와 출처도 재계산 PASS. 결과 SHA `00f62f2d2799185fb1c63fd6efdb300f6bbd40e011837a30d4cd19618c26a7a5`, 독립 감사 SHA `d14ca95fc47eef94090bed1412040ea0b73d9e073d587f940afeb5d882bf7b0d`.
- 두 회차 실험 총 96개 요청 성공이며, 두 번째 회차를 첫 번째의 재분석으로 가장하지 않았다. 계측 비용이 지연시간에 포함되므로 **P13에서는 가속 수치를 새로 주장하지 않는다**.
- 실제 테스트 입력은 기존 P6 데이터를 재사용한다. 6개 분야 실험이 새로운 독립 holdout이나 긴 응답 품질의 외적 타당성을 보증하지 않는다.

## 4. 실제 제어자 잠금 및 오래된 ACK 위조 방어

P12의 연구용 재시작 전 `flock` 독점 게이트 위에 P13의 **네이티브 완료 ACK 의미 검증**을 추가했다.

- 기존 P12 실제 네이티브 완료 트랜잭션 영수증 두 건을 읽어, 이전 명령의 전후 정밀도 해시·epoch·두 레이어 역할·5/6비트 변경·진짜 ACK의 `txn_targets`를 다시 대조. 2/2 PASS.
- ACK 원문을 조작한 뒤 `INTENT`와 `VERIFIED`의 해시를 모두 일관되게 다시 계산하는 위조 시나리오에서도 원래 ACK의 epoch/정책/대상 의미가 틀리면 재시작을 거부하도록 강화했다. P13 신규 **12/12 변조·의미 경계 테스트 PASS**.
- EOE의 격리 디렉터리에서 **실제 Python 독립 프로세스 8개**가 동시에 잠금을 요청하게 했다. 기존 제어자가 소유할 때 8/8 거부. 해제 후 동시 경합에서 1개 획득, 7개 거부. 두 소유자의 독점 보유 구간 겹침 0건. 최종 순차 재진입도 정상.
- 해당 검증은 **제어자의 배타적 시작 권한** 검증이다. 동시 GPU 워커 여러 개가 외부 요청을 받는 운영 실증이나 OS 전원 차단 후 모든 요청의 exactly-once 보장은 포함하지 않는다.
- P11의 실제 처리 중단 후 상태 UNKNOWN인 두 거래는 여기에서도 완료로 바꾸거나 자동 재실행하지 않았다.

## 5. 회귀·증거 보존

- P13 신규 qNg64 네이티브 생성·해시 감사 변조 방어 **16/16 PASS**.
- P13 원본 네이티브 완료 저널 semantics/ACK 재계산 검증 **12/12 PASS**.
- P8 48/P9 21/P10 10/P11 46/P12 44까지 포함해 전체 **197/197 Python 회귀 PASS**.
- 실제 네이티브 SHA 출처, 원본 참조, ON/OFF 2차 실험, 제어자 경합·테스트 로그를 `experiments/hcp_p13/evidence/native_xox_20261014` 하위 SHA manifest에 보존.
- 소스: `tools/hcp_p13_pack_sha_patch.py`, `tools/hcp_p13_native_byte_parity.py`, `tools/hcp_p13_native_byte_parity_v2.py`, `tools/hcp_p13_independent_audit.py`, `tools/hcp_p13_strict_journal.py`, `tools/hcp_p13_controller_race.py`.

## 6. 운영 불변성과 남은 관문

운영 stage SHA: `8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9`.
운영 active-route SHA: `c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47`.
운영 워커 PID: `35481, 35493`를 실험 전후 확인했다. `main` 미변경, 릴리스 activation 없음, 자동 승격 OFF.

남은 차단 관문:
1. 모델 전체 BF16 teacher forward + same token alignment/full-vocab logits, 독립 품질 비열등성 검증. 원본 약 29.256GiB 가중치만으로도 현재 XOX 사용 가능 메모리를 초과하므로 기존 운영 워커를 중단하거나 swap을 과도하게 쓰는 방식으로 실행하지 않았다.
2. 여러 실제 GPU 워커가 정밀도 변경·재시작을 병행하는 상황에서의 transaction ownership, 원래 불명 요청의 대사와 exactly-once 처리, 호스트 cold boot·모델 hot reload.
3. 새 긴 문맥·긴 생성·외부 정답형 벤치마크에서 품질/latency/p95/memory 반례 확인.
4. 이 결과로 운영 바이너리를 직접 변경하거나 승격하지 않는다. 별도의 immutable release 인증 후 최신 Current/Live/Runtime과 CAS(expected_current)를 통한 승인 경로가 필요하다.

**P13 최종: 원본 BF16의 5/6비트 표현과 실제 P9 네이티브 출력의 바이트 완전 일치 확인. 배타적 재시작 권한 및 ACK 의미 위조 차단 범위를 확대. Full BF16 모델 품질/운영 승격은 BLOCKED.**
