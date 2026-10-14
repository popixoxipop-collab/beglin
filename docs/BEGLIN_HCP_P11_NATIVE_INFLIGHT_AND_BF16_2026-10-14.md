# Beglin HCP P11 — 실제 정밀도 전환 중단·격리 대사와 BF16 원본 부분 검증

작성일: 2026-10-14 (KST)
상태: 격리 XOX 연구 시험 PASS / 실제 운영 승격 BLOCKED
연구 브랜치: research/hcp-p11-inflight-reconcile-bf16-20261014
P10 기준 커밋: 645e2eba85680f33f8f2148ce574935c279b07c4
P11 사전등록 커밋: 6a3790590321b35e353f0b06978517e5407dcce9

## 1. 판정 요약

P10은 요청이 완료된 뒤의 워커 재시작을 검증했다. P11에서는 실제 DeepSeek-V2-Lite / Apple Silicon MLX 네이티브 정밀도 전환 중 두 지점에서 실험용 워커 PID만 강제 종료했다.

| 실제 중단 시점 | 중단 지점 확인 | 원래 명령 격리 | 새 워커 A→B→A→B 중 새 B→A→B | 판정 |
|---|---|---|---|---|
| first_bind | 첫 번째 텐서 bind 직후, 두 번째 이전 | durable INTENT/VERIFIED 및 SHA PASS | 새 B→A→B 3/3 PASS | 격리 복구 성공 |
| before_ack | 두 텐서 결합과 native epoch 변경 후, 내구성 ACK 직전 | durable INTENT/VERIFIED 및 SHA PASS | 새 B→A→B 3/3 PASS | 격리 복구 성공 |

**중단된 원래 요청 2건의 최종 상태는 UNKNOWN이다.** 완료 ACK가 없고, 두 번째 시험은 메모리 정책 변경 후 ACK를 쓰기 직전에 충돌했기 때문에 원래 변경이 영속적으로 완료되었다거나 원래 요청이 정확히 한 번 수행되었다고 주장할 수 없다.

두 중단 명령은 기존 런타임에 다시 제출하지 않았다. 실행 중인 워커의 종료를 입증한 후 fsync된 INTENT 장부를 작성하고, 원본 명령을 별도 quarantine으로 atomic rename, 디렉터리 fsync, VERIFIED 장부 발행 및 SHA 재검사를 수행했다. 안전 격리 후 **새로 제출한** 6개 B→A→B 요청의 생성 SHA가 기존 P6 원본 참조와 일치했다.

테스트 중 어떤 변경도 운영 바이너리·active-route·운영 워커에 반영하지 않았다. 운영형 자동 격리 대사 장치는 아직 설치하지 않았다.

## 2. 실기기 조건과 소스 정체성

- 대상: XOX MacBook Pro M1 Max, 단일 GPU 워커 / DeepSeek-V2-Lite.
- P5 C의 native CBATCH/KV128 연구 경계 및 기존 두 대상 정책은 보존.
- 기존 운영 stage 바이너리 SHA-256: 8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9.
- 기존 운영 active-route SHA-256: c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47.
- 기존 운영 stage PID: 35481, 35493.
- P5 CB128 C 원본 SHA: bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612.
- P9 prepared-cache MLX KV128 원본 SHA: ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f.
- P11 테스트용 2개 충돌창 C SHA: 3d5a6d9f2f63d53e960c14a88dc4892ffa27cfd67b6d15258b6c930a7c17f2bd.
- P11 실제 컴파일 실기기 전용 바이너리 SHA: a5fd3f590ad6bbddcb69e690ccf738b711b8f7ceb3319f99ee933123f5ac7e78.
- 변경된 C에 있는 충돌창 코드를 제거하면 P5 CB128 원본 C가 바이트 단위로 정확히 복원됨.
- 기존 P6 실제 DeepSeek 품질 SHA: 2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104.

연구용 충돌 바이너리는 운영 stage 경로가 아니라 XOX의 별도 .tailnet-tmp/hcp-p11-inflight-20261014-1816 경로에만 배치했고 단일 요청 슬롯을 사용했다. 네이티브 워커 종료는 해당 실험 Worker 객체의 정확한 PID에만 수행했다.

## 3. 실제 실패 관측과 수정 내역

- 최초 테스트 바이너리 컴파일 시 생성된 C 계측 코드의 줄바꿈 문자열 이스케이프 오류로 컴파일이 실패했다. source 생성기 수정 후 XOX에서 실제 GPU 빌드가 exit 0으로 통과했다. 연구용 충돌창 외 수식·기존 ACK·롤백 로직 변경 없음.
- 첫 first_bind 시도는 실행기에서 설정한 관측 대기 8초가 만료되어 마커를 확인하지 못했다. **이 시도는 실제 중단 지점 PASS로 계산하지 않았다.** 명령/ACK 증거를 별도 보존하고 원인을 프롬프트 처리 지연으로 분류했다. 다음 실행기에서 관측 대기를 45초로 연장했으며 이후 진짜 crash-marker, SIGKILL, quarantine, 신규 복구가 exit 0으로 성공했다.
- before_ack 시험도 별도 새 워커·별도 UUID 트랜잭션을 사용하여 exit 0으로 통과했다.
- 처음 실패한 C 빌드/마커 미관측 시도는 최종 성공과 분리된 원본 증거로 남겼다.

## 4. 장애 대사 증거 — 절대 혼동 금지

first_bind:
- 실제 실행 증거 first_bind.json SHA: 656db4280567702d2af7c0eab388bf9615935e538b51c5a8c091d95327a54fb4
- 격리된 원본 txn SHA: 648a07529a73203130e2f16513e63f824f95384416637e3167dd35d758c8c7ec
- fsync된 INTENT/VERIFIED 장부는 forensic/first_bind_INTENT.json, forensic/first_bind_VERIFIED.json에 보존.

before_ack:
- 실제 실행 증거 before_ack.json SHA: 39d272c4b94d0120d2b3a90c6aa65371017efbe6899e61571533bad3ee50c519
- 격리된 원본 txn SHA: 3e2322e05d7beded8f393faf793f831d691a3899bca64ee443ed7eef85e33564
- fsync된 INTENT/VERIFIED 장부는 forensic/before_ack_INTENT.json, forensic/before_ack_VERIFIED.json에 보존.

각 마커 SHA, 원본 명령 SHA, 사전 ACK 해시, 예상 epoch·정책 해시, 검증 로그를 XOX에서 확인한 뒤 EOE로 복사했다. 독립 Python 감사가 원본 명령·마커·fsync receipt 간 정체성, P6 생성 SHA, 서로 다른 두 txn ID와 인위적 중단 결과 UNKNOWN 판정을 다시 확인했다.

주의: 재시작 전 옛 ACK 전체 바이트를 별도 독립 증거로 복제하지는 않았다. 출처 해시는 XOX 실행기와 durable INTENT/VERIFIED에 기록되었으며, 감시기가 실제 영속 ACK 파일을 원본 상태로 다시 읽는 더 강한 인증은 추후 과제다. 따라서 P11을 production-wide exactly-once certification이라 부르지 않는다.

## 5. 원본 BF16 출처 검증 — 범위의 한계

XOX의 모델 인덱스 SHA: d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf.

원본 BF16 모델은 총 4개 safetensors shard, 31,413,626,576 bytes(29.256GiB)이며, 운영 워커를 유지한 XOX의 가용 RAM/기존 swap 상황 때문에 full teacher 모델의 정확한 로그잇 추론을 실행하지 않았다.

대신 XOX가 원본 파일을 읽기 전용으로 열어 다음 2개 텐서에서 각각 BF16 16×256 요소만 로딩했다.

- model.layers.3.mlp.shared_experts.up_proj.weight (원본 형상 2816×2048)
- model.layers.26.mlp.shared_experts.down_proj.weight (원본 형상 2048×2816)

총 16,384 bytes만 BF16 값으로 읽어 CPU FP32 16×256 타일·벡터 곱을 수행했고, 각 타일 및 결과 투영 SHA를 새 프로세스로 동일 원본 shard에서 다시 읽어 일치 확인했다.
- 첫 원본 조각 영수증 bf16_source_tiles.json SHA: 49af27431fe9e672ea5783a0bc0112f2b412fe655ea3f769467bbfad13983090
- 독립 재읽기 영수증 bf16_tile_reread.json SHA: 9d0efc226106bba46be862e2268755abd03aaf4c9802bef2a90861eed87d71d4
- 각 BF16 slice 프로세스 최대 RSS 약 204MiB / 203MiB.

이는 **원본 BF16 가중치 일부의 접근·CPU 연산·반복 SHA 검사**에 대한 증거일 뿐, **whole BF16 forward, BF16 참고 logits, 정답률, 모델 비열등성 인증은 전혀 아니다.**

## 6. 회귀·변조 차단

5개 독립 테스트 그룹 모두 PASS:

| 그룹 | 통과 |
|---|---:|
| P11 원본/충돌창/격리 코드 | 18/18 |
| P11 증거 변조·결손·거짓 성공/교사 주장 차단 | 16/16 |
| P10 신규 입력·재시작·2슬롯·p95 증거 | 10/10 |
| P9 MLX 캐시·실패 복구 소스 검사 | 21/21 |
| P8 정밀도 스케줄러·계측 회귀 | 48/48 |
| **기본 회귀 합계** | **113/113** |

외부에서 변조시킨 마커, 격리 명령, INTENT, VERIFIED, 사전 소스/바이너리 해시, 실제 생성 SHA, BF16 값/투영 재읽기 등을 거부하도록 확인했다. 원본 P6 질의 응답 생성 참조는 기존 2ad68622...에 고정한다.

## 7. 영구 산출물 / 이어서 수행할 작업

- GitHub: popixoxipop-collab/beglin
- 브랜치: research/hcp-p11-inflight-reconcile-bf16-20261014
- P11 문서: docs/BEGLIN_HCP_P11_NATIVE_INFLIGHT_AND_BF16_2026-10-14.md
- 실측 증거: experiments/hcp_p11/evidence/native_xox_20261014/
- 원본 2개 명령·마커·native crash JSON·INTENT/VERIFIED·BF16 2회 조각·독립 audit·113개 테스트 로그를 SHA manifest에 기록.
- 실행/검사 소스: tools/hcp_p11_inflight_patch.py, tools/hcp_p11_inflight_trial.py, tools/hcp_p11_quarantine.py, tools/hcp_p11_bf16_slice_probe.py, tools/hcp_p11_evidence_audit.py, tools/hcp_p11_forensic_audit.py, tools/test_hcp_p11_inflight.py, tools/test_hcp_p11_forensic.py.

**남은 강제 차단 관문**
1. 운영에도 적용 가능한 영속 요청/거래 저널과 부팅 전에 실행되는 분리된 대사 전용 절차, 실제 처리 도중 프로세스·호스트 재시작, ACK 원본 사전 보존 및 동시 워커 경합을 fail-closed 검증. P11의 quarantine 도구는 연구용 경로 한정이며 배포하지 않았다.
2. 메모리 안전성이 증명된 BF16 전체 DeepSeek 교사 추론 또는 별도의 충분한 전용 호스트에서 BF16 logits·미사용 데이터 정확도 및 A/B 품질 비열등성 검증.
3. 추가 긴 출력/더 긴 문맥, 다중 워커, 캐시 eviction/모델 재적재, 메모리 압력과 tail latency 위험성 실험.
4. 이 모든 관문 통과 후 immutable release certification, fresh Current/Live/Runtime, expected-current CAS로만 운영 승격 검토.

**최종 결론: P11 실제 crash windows 2/2 및 연구용 격리·새 요청 복구 성공. BF16 원본 slice 재읽기 PASS. 원래 충돌 요청 exactly-once, 전 모델 품질 및 운영 승격은 BLOCKED.**


### 추가 원본 증거 변조 회귀 (기본 manifest 봉인 이후)

기본 113개 결과 및 원래 SHA manifest는 수정하지 않고 보존했다. 기존 독립 감사 도구가 원래 crash 명령·마커·BF16 조각 등을 잘못 해석하지 않는지 확인하는 추가 12개 원본 증거 변조 테스트도 **12/12 PASS**했다.

- 추가 회귀: tools/test_hcp_p11_evidence_audit.py
- 로그: experiments/hcp_p11/evidence/native_xox_20261014/supplementary_original_evidence_tamper_12.log
- 추가 영수증: experiments/hcp_p11/evidence/native_xox_20261014/SUPPLEMENTAL_12_TESTS.json
- 기본 manifest SHA: bac8d375fe8c43b42223c9261f8dbcb1e726a282763a612a6ff38cee1535ded2
- 기본 113 + 추가 12 = **전체 실행 회귀 125/125 PASS** (중복 주제를 다루는 테스트가 일부 있으므로 125개 서로 다른 품질 기능 인증을 뜻하지 않음).

주의: 기본 manifest는 113건 실험 로그만 직접 열거하며, 12건은 별도 SHA 보충 영수증에서 증명한다. 연구 범위나 운영 승격 판정은 달라지지 않는다.
