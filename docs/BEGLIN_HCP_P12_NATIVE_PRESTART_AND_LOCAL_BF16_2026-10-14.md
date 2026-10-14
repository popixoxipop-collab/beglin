# Beglin P12 — 재시작 전 대사 저널 및 원본 BF16 국소 오차 실측

작성: 2026-10-14 (KST)
판정: **격리 실기기 연구 통과 / 전체 품질·운영 승격 BLOCKED**

기준 저장소: popixoxipop-collab/beglin
P11 기준 커밋: e2271d6fe758c6d09877130637dccb193427ea15
연구 브랜치: research/hcp-p12-prestart-journal-bf16-local-20261014

## 결론

P11은 네이티브 REBIND_SET 중 두 충돌 시점에서 실제 연구 워커를 종료해 결과 불명의 원래 명령을 SHA·fsync quarantine에 봉인하고, 새 요청으로 복구하는 시험이었다. P12에서는 **완료 ACK가 존재하는 다른 종류의 재시작 문제**를 다뤘다. 종료 전에 완료된 네이티브 명령이 디스크에 남았을 때, 새로운 워커 시작 전 정확한 명령·ACK·epoch·정밀도 정책을 검증하고 저널에 보관한 뒤에만 시작하도록 독립 연구용 게이트를 구현했다.

XOX 실제 P9와 동일 SHA의 네이티브 MLX 바이너리로 **완료된 전환 2건을 저널링**, 새로운 PID에서 정책 A를 다시 시작하고 새로운 B→A→B 요청을 실행했다. 생성 해시가 기존 P6와 정확히 일치했다. 기존 충돌 명령의 자동 재실행은 허용하지 않는다.

또한 전체 29.256GiB BF16 교사 모델을 적재하지 않고도 원본 BF16 공유 전문가 가중치 행렬 2개를 직접 읽어 **동일한 C 그룹64 양자화 알고리즘**으로 5·6비트 오차를 비교했다. 이는 가중치·국소 행렬투영 오차이며, 전체 LLM 답변 품질·BF16 logits 인증이 아니다.

## 1. 재시작 전 저널 게이트

구현: tools/hcp_p12_prestart_journal.py, tools/hcp_p12_real_native_prestart.py.

안전 계약:
- 실행 범위는 .tailnet-tmp/hcp-p11-inflight-p12-* 연구용 루트로 제한. 운영 디렉터리·main에 설치하지 않았다.
- 소스/모델/바이너리 SHA-256 pinning, 디렉터리 경계, symlink 거부.
- POSIX flock 비차단 독점 잠금을 워커 생존·입장·정지 동안 유지. 제2 제어자는 접근 거부.
- 이전 명령이 **REBIND_SET_APPLIED/REBIND_SET_FAILED**의 진짜 같은 txn_id·epoch·policy·변경대상 ACK를 보유한 경우에만 INTENT fsync → 명령 atomic rename → 원본 ACK 보존 → VERIFIED fsync 순서로 영속 기록.
- 마커·완료 ACK 없는 애매한 명령은 **UNKNOWN**으로 간주하고 워커 시작 자체 차단. P11에서 입증한 별도의 fsync quarantine 영수증이 있어야 미완료 원본 명령을 재제출하지 않은 채 새 시작 허용.
- 누락된 VERIFIED, 원본 명령이나 ACK 변조, 중복 txn, 다른 모델, 재시작 중간 저널 손상, 두 번째 제어자 경합을 fail-closed로 처리. 정확히 한 번 실행 보장 주장 금지.

XOX 실기기 결과:
- P9 원본과 동일 바이너리 SHA: 103549b7f7f4a3d142be883013956b4f1437759c8be76bff31f2451956df37fd.
- 첫 A→B 완료 후 연구용 워커를 종료, 이전 완료 ACK를 포함한 원본 명령을 재시작 전에 fsync 아카이브.
- 새 PID·초기 정책 A·새 startup ACK 확인, 이전 명령 자동 재실행 0회.
- 새 B→A→B 3요청 모두 기존 DeepSeek P6 시퀀스 SHA와 일치; 초기 B 요청을 포함해 총 4개 유효 생성. 두 번째 완료 명령도 최종 저널링해 **고유 완료 거래 2건** 봉인.
- 영수증: experiments/hcp_p12/evidence/native_xox_20261014/P12_native_prestart.json, 두 개 p12_journal/*/INTENT.json / VERIFIED.json / original.cmd / original_ack.json.

**한계:** P12는 known-terminal 재시작 경로의 진짜 네이티브 시험이며, P11의 UNKNOWN 원래 요청 exactly-once/자동 복구를 해결한 것이 아니다. 해당 UNKNOWN은 수동 검증·격리 후 새 요청만 허용하는 fail-closed 정책이다. P12 코드는 운영 서비스에 연결되지 않았다.

## 2. BF16 원본 가중치에 대한 국소 qNg64 정량 비교

XOX의 DeepSeek-V2-Lite 원본 BF16 safetensors index SHA: d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf.

기본 지표는 원본 BF16 → FP32 행렬을 참조로 삼고, 원본 소스에서 *실제로 사용되는* C gguf_quantize_qNg64(n=5 또는 6) 그룹64·오차피드백·비트플레인 포맷을 적용한 행렬의 **국소 투영** 값을 비교한 것이다. 각 행렬에 8개의 재현 가능한 입력 벡터를 사용했다. 패킹된 C 결과와 독립 numpy 비트플레인 디코딩은 첫 벡터 투영에서 최대 약 0.000003의 절대차 이내였다.

| 실제 BF16 가중치 (전체 텐서) | shape | 비트수 | BF16 대비 가중치 상대 L2 | 국소 투영 상대 L2 |
|---|---|---:|---:|---:|
| L3 shared_up_proj | 2816×2048 | 5 | 7.396% | **7.366%** |
| L3 shared_up_proj | 2816×2048 | 6 | 3.581% | **3.585%** |
| L26 shared_down_proj | 2048×2816 | 5 | 8.764% | **8.437%** |
| L26 shared_down_proj | 2048×2816 | 6 | 4.235% | **4.119%** |

- 원본 BF16 safetensors는 읽기 전용. 두 텐서를 하나씩 읽고 처리하여 총 모델 체크포인트를 적재하지 않았다.
- 원본 qNg64 C 구현에서 컴파일한 시험용 라이브러리 SHA: 1bb29f8c73abdf6afc157495b14efcd93d82f5cc61c6416b8359fc9e7259fe6e.
- 첫 BF16의 qNg64 C 결과는 실제 P9 동적 런타임의 packed weight와 바이트 동일하다고 아직 증명하지 않았다. 정확히 같은 양자화 함수라도 모델 초기화·source 해시·전환 경로가 일치하는지 별도 증거가 필요하다.
- BF16 국소 오라클 최고 메모리: **약 329.20 MiB**, 사전 상한 1.5GiB 미만.
- 증거: experiments/hcp_p12/evidence/native_xox_20261014/p12_local_BF16_oracle.json.

**반드시 구분:** 위 오차는 국소 가중치·projection에 관한 수치이며, 전체 모델 BF16 forward, 전체 vocab logits, perplexity, 전문과제 정답률, 정밀도 A/B 우열 또는 품질 비열등성을 입증하지 않는다.

## 3. 회귀, 감사, 무결성

- P12 독점 prestart 잠금/terminal·UNKNOWN·저널 손상: **24/24 PASS**.
- P12 실측 원본 BF16·네이티브 ACK/정책/생성 SHA 증거 변조 방어: **20/20 PASS**.
- 이전 P8~P11 연구용 회귀 그룹: P8 48, P9 21, P10 10, P11 46건. 전체 **169/169 PASS** (서로 다른 세대의 회귀를 모두 이번 브랜치에서 재실행).
- 독립 EOE 증거 감사 PASS: original P6 샘플 생성 SHA, 두 terminal ACK와 epoch 일치, 원본 qNg64 C SHA, BF16 로컬 수치·하드 상한·기록 무결성.
- 전체 원본 JSON·실측 preflight·fsynced command/ACK/journal·테스트 로그는 SHA-256 증거 MANIFEST.json으로 보존.
- 디버깅 이력: 최초 테스트 임시루트가 /var symlink를 거쳐 경로 가드에 거부됐으며, 테스트만 canonical resolved root로 고친 후 24/24 PASS. 독립 감사 실행기 파이썬 개행 구문 오류를 수정한 뒤 감사 PASS. 원본 운영 소스·ACK 계약 수정은 없음.

## 4. 운영 보존 및 다음 관문

원본 운영 stage SHA-256: 8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9
원본 active-route SHA-256: c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47
기존 운영 워커 PID: 35481, 35493 — 실험 전후 유지 확인.

남은 강제 차단:
1. 모델 전체 BF16 독립 교사의 full-vocab logits/새 미사용 과제 정답률에 대한 품질 비열등성. 현재 원본 모델 가중치만 29.256GiB이고 기존 워커를 유지할 때 가용 RAM/스왑 위험이 있어 **전체 교사 시험은 미실행**.
2. 트랜잭션 도중 프로세스·OS 중단에서 요청 결과 exactly-once 계약과 영속 transaction reconciliation; 임의 다중 워커 경합·hot reload. P12 연구용 게이트는 운영 서비스에 미설치.
3. 정적 AF/A/B와 반복 국소 전략의 공통 품질·성능·메모리·꼬리 지연시간 인증, 새로운 긴 문맥/다중 사용자.
4. 위 인증 후 독립 immutable release build/stage/verify, 최신 Current/Live/Runtime 확인과 expected-current CAS로만 활성화. 자동 운영 승격 금지.

**최종 분류: P12 국소 BF16 오라클 및 안전한 완료 명령 저널·재시작 검증 완료. full-model BF16 teacher 및 운영 승격 BLOCKED.**
