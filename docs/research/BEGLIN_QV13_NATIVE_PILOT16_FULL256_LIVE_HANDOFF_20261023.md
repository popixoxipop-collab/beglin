# BEGLIN Qv13: 원본 Apple C 28n8 동일 바이트 파일럿 및 256문항 실행 인계

**작성:** 2026-10-23 KST  
**연구 브랜치:** `research/qv13-byte-budget-20261023` (`popixoxipop-collab/beglin`)  
**판정:** `REAL_NATIVE_8_ARM_16_PILOT_PASS / INDEPENDENT_128_LOG_ROW_REPARSE_PASS / FULL256_JOB_RUNNING / NO_POLICY_GENERALIZATION_CLAIM / NO_LIVE_PROMOTION`

## 1. 배경과 사전실행 고정

Qv12에서는 Qwen2.5-1.5B 28계층, 총 196개 projection 텐서의 0/1/2개 n8 승격 7조건×16문항을 실제 원본 C로 측정했다. Qv13은 더 많은 28n8 정책에서 **동일한 packed-weight byte 예산과 동일 투영 역할별 정밀도 승격 개수**를 맞춰 평가한다.

- 원본 BF16 모델 revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- 원본 Q4 C 바이너리 SHA: `408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c`.
- 원본 혼합 28n8 C 바이너리 SHA: `f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221`.
- Qv13 원본 prelock SHA: `5cb944375fa271c4528b4846f4b3c22cdf8179d4a38a2f5c17babbc03c4df92a`.
- 실제 소스 정책 Qv8 마스크 SHA: `9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922`.
- 256문항 사전실행 공개 봉인 SHA: `9d045e1056e5ec8bc3a093a8a3d198277cbbc4083fb054a929b566cb2bb03c8c`.
- 전체 비공개 봉인 SHA: `79826eb944ac2a4f87b637a430471f069a3789da37cdb12a9b7d1060d49db9b8`.

RACE 원문 전체·기존 208토큰 이하 조건을 적용하면 middle/test의 적합 지문은 55개, high/test는 2개여서 128+128 RACE 설계가 불가능했다. 모델 결과 확인 **이전**에 ARC-Challenge test 128건과 OpenBookQA test 128건으로 전환하고 각각의 문항 ID·프롬프트/정답 토큰 SHA를 봉인했다. 파일럿은 사전에 정한 ARC 앞 8건+OpenBookQA 앞 8건이다. 공개 자료는 aggregate만 포함하고 owner-only 원문·토큰·개별 NLL은 D50에 둔다.

**중요한 사전등록 정정:** 기존 prelock의 Qv8 소스 SHA 필드, 역할별 시드 샘플링 순서, 문항 정렬 해시 설명은 실제 C 평가 전에 생성된 SHA-봉인 마스크/입력의 구현 설명과 일치하지 않는다. 원문은 변경하지 않고 `BEGLIN_QV13_PREEXEC_PROVENANCE_ERRATA_20261023.md`에 별도 기록했다. 따라서 동일한 prelock 문구의 정확한 절차 재현을 주장하지 않는다.

## 2. 실제 28계층 BEGLIN C 파일럿: 8조건×16문항

정답은 A/B/C/D 중 argmax, NLL은 **전체 어휘 softmax의 정답 single-letter next-token NLL**이다. 이번 결과는 모두 **동일한 16개 문항**에 대한 네이티브 실제 추론 결과다.

| 원본 C 정밀도 마스크 | n8 수 | 정답 | mean gold NLL ↓ | projection packed bytes | peak process RSS MiB | wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| all Q4 | 0 | 8/16 | 2.860083 | 736,985,088 | 4,609.58 | 173.108 |
| Qv8 source | 28 | 8/16 | 2.797238 | 765,493,248 | 4,637.16 | 188.301 |
| random seed73 | 28 | 8/16 | 3.703373 | 765,493,248 | 4,637.17 | 141.371 |
| random seed137 | 28 | 7/16 | 3.111264 | 765,493,248 | 4,637.16 | 108.611 |
| random seed211 | 28 | 8/16 | 3.204203 | 765,493,248 | 4,637.59 | 105.950 |
| random seed389 | 28 | 8/16 | 3.082859 | 765,493,248 | 4,637.14 | 88.746 |
| random seed577 | 28 | 8/16 | 3.387852 | 765,493,248 | 4,637.19 | 93.485 |
| Original V2 (별도 비용) | 28 | 8/16 | 2.434044 | 776,896,512 | 4,649.44 | 110.874 |

Qv8과 다섯 무작위 마스크는 **동일 projection packed bytes 765,493,248, 추가 비용 28,508,160 bytes, 동일 역할별 승격 수**다. Original V2는 11,403,264 bytes 더 크므로 **같은 바이트 예산 비교군이 아니다.**

5개 무작위 정책 평균 NLL **3.297910** vs Qv8 **2.797238**, NLL 평균 차이(무작위−Qv8) **+0.500672**. 같은 16문항에 대한 10,000회 쌍별 부트스트랩 구간 **[+0.093459, +0.924229]**. 다만 **5개 개별 비교의 탐색적 sign-flip 검정에 Holm 보정 적용 시 0.05를 통과한 것은 0/5**다(최소 보정 p 약 0.102). 정확도 변화가 거의 없고, 16건 파일럿이므로 **Qv8이 범용적으로 우수하다고 결론 내리지 않는다.**

wall 시간은 모델 로딩·CPU 환경 부하가 포함된 총 소요이고 지속 토큰/s·per-token p50/p95의 실측이 아니다. peak RSS는 process RSS 스냅샷이며 packed-weight 장부와 구분한다.

## 3. 이중 독립 검산 결과

- 원본 영수증의 소스 마스크, 원본 Q4/28n8 네이티브 바이너리, 입력 manifest와 실제 개별 문항 수 검증 PASS.
- **원본 C 로그 8조건×16 = 128개 줄 직접 재파싱 PASS**: 토큰 바이트 SHA, 정답 인덱스, prefix 길이, 28n8+168Q4 등록, 개별 NLL, 집계 합계/정확도 검증.
- GitHub에 등록한 코드: `tools/qv13_native_256.py`, `tools/qv13_native_independent_audit.py`, `tools/qv13_native_log_crosscheck.py`, `tools/qv13_equal_byte_holm.py`.
- 공개 영수증: `BEGLIN_QV13_PILOT16_ORIGINAL_NATIVE_8ARM_INDEPENDENT_PUBLIC_20261023.json`, `BEGLIN_QV13_PILOT16_REAL_C_LOG_REPARSE_PUBLIC_20261023.json`, `BEGLIN_QV13_PILOT16_EQUAL_BYTE_HOLM_EXPLORATORY_PUBLIC_20261023.json`.

## 4. full 256 실제 네이티브 실행 상태 (본 문서 작성 시점)

- **작업 ID:** `job_8a24620f416ea11304645700c7f3c8c2`
- **호스트:** `eoe` (Apple Silicon native CPU); Tailnet 자원 예약 `res_1151a029f3a14925833c84899ac71609`; 저장소 풀 `d50_work`.
- **현재 확인 상태:** `RUNNING`, 순차 고정 작업 16개(ARC128/OB128 각각 Q4, Qv8, 무작위 5개, Original V2). 현 시점에는 전체 256 결과가 **미완료**다.
- **실행 코드:** `tools/qv13_full256_sequential.py`; 작업 시작 시 16/16 파일럿 독립 로그 감사를 통과하도록 선행 게이트 설정.
- 원본 로그·개별 문항 NLL·토큰 파일은 `/Volumes/D50/TailnetCommander/work/beglin-qv13-20261023`에 owner-only로 저장.
- 진행 로그: `/Volumes/D50/TailnetCommander/work/beglin-qv13-20261023/qv13_full256_sequential_progress.jsonl`; 이미 원본 영수증이 검산된 조건만 건너뛰고 불완전 로그/오류가 있으면 중단한다.
- job 상태 조회/조사는 `Tailnet Commander EOE.job_status(host=eoe, job_id=...)`, `job_output` 및 `qv13_full256_sequential.py check`를 사용. **중간 성공을 최종 256문항 결과로 발표하지 않는다.**

## 5. 최종 연구·운영 경계

- 이번 Qv13은 **고정된 정밀도 마스크들의 실제 C 비교**다. 입력별 실시간 정밀도 전환 알고리즘이나 Metal 커널·스위칭 오버헤드는 아직 실증하지 않았다.
- 공개 test가 모델 사전학습에 전혀 노출되지 않았다는 보장은 없다. 파일럿 16개에서 이미 중간 결과를 확인했고 원래 prelock 설명과 구현에 차이가 있으므로, 전체 256 데이터도 **보존형 독립 추가 관측**이지 사전등록 문구와 완전히 일치하는 pristine confirmatory trial로 취급하지 않는다.
- **main 병합 없음, 원본 커널·BF16 checkpoint 변경 없음, Tailnet 운영 릴리스 수정 없음, 자동 precision promotion 없음.**
