# BEGLIN P18 — 공식 모델 복사 authority gate 수리 계약 (운영 비접촉)

작성: 2026-10-16 KST
범위: DeepSeek-V2-Lite original BF16 (4 shards/12 registered relative files) + native quantized MoE (5 files), XOX → EOE/D50
현재 판정: **BLOCKED / no official copy-source authority issued / policy A 유지**
기준 증거: `BEGLIN_P18_MASTER_HANDOFF_2026-10-16.md` SHA `366231fd95e86152f1526906828c477021634485bca5ee1b4f5f1f288aebb736`

## 1. 새 독립 코드 정적 점검 결과

EOE D50의 `hcp_p18_eoe_480_readonly_admission.py` SHA-256은 `c80cb18c5f2b2ed0b3fa25b4dde0ac5bda9bdd30cfd740191e3c157fbe58f33f`.

- 37행: `receipts[name]=p.is_file() and not p.is_symlink()`
- 38행: `has_both=(not copied and all(receipts.values()))`
- 54행: `has_both`가 참이면 `P18_EOE_MODEL_COPIES_READY_FOR_SAFE_ITEM_RESOURCE_CHECK` 상태로 표시.

따라서 현 진입 코드는 모델 영수증 2개의 **존재 여부만 검사**하며 내용·인증 주체·승인된 원본 모델 SHA·전송 성공 증거의 결속을 평가하지 않는다. 파일 생성이나 인증 작업을 실제 실행해 우회한 것이 아니라 원본 연구용 코드를 읽어 점검한 정적 발견이다. 이 상태로 영수증 존재만으로 native A/B나 480문항 승인 신호를 사용하면 안 된다.

EOE D50의 `hcp_p18_eoe_d50_model_copy_seal.py` SHA-256은 `06a8dfcdf3f5b48935179a52f22d9fd405f9ab0b85fe93acfa7615bce7c95ffd`. `--inspect`는 누락 0/`P18_D50_COPY_READY_TO_SEAL`을 반환하나, **공식** `EOE_P18_BF16_MODEL_COPY_RECEIPT.json` 및 `EOE_P18_MOE_BASE_COPY_RECEIPT.json`은 미발급이다. 앞선 세션의 `--seal` 쓰기 실행은 도구 보안 검사에서 차단됐으며 재실행이나 권한 우회는 하지 않는다.

기존 원본/사본 **17/17 SHA read-only 일치**와 전송 작업 2개의 종료 코드 0은 ‘파일 복사 무결성’의 근거일 뿐, 서버 소유의 모델-source authority 승인과는 구별해야 한다.

## 2. 수리할 서버 소유 검증 계약

수정은 릴리스 등록 범위가 고정된 `beglin/p18-model-copy-authority` 같은 **서버 소유 전용 capability**로 구현한다. 현재 런타임에 이런 capability가 있다고 가정하지 않는다.

1. **등록된 출처만 채택**: 고정 XOX source 및 EOE D50 target, 기대 파일 상대경로 17개, 원본 BF16 4 shard 크기 및 모델 index/config/P17 class SHA, 정해진 transfer operation identity를 서버 정책에서 고정. 호출자 임의 경로나 임의 SHA/명령어를 authority 근거로 받지 않는다.
2. **완료 증거**: 실제 등록된 BF16 및 MOE 복사 operation의 terminal `EXITED 0`를 서버 저장 기록과 매칭하고, 현재 canonical source/target 17개 파일의 SHA-256를 전부 독립 확인. 실행 기록 없는 유사 JSON이나 사용자 생성 텍스트는 불허.
3. **발행 권위**: 검증 완료 시 서버가 canonical 입력/출력/model identity/release_id/issue operation/source revision을 결속한 immutable attestation을 원자적으로 등록하고, 서버 소유 서명 또는 동일 수준의 불변 authoritative registry를 이용해 발행자 및 무결성을 검증할 수 있게 한다. 일반 파일시스템의 단순 SHA/영수증 존재는 발행 권위가 아니다.
4. **평가 진입**: 원래 `has_both`/파일 존재 확인 대신 등록된 authority API/검증 함수로 두 복사 attestation을 각각 검증. 전체 original BF16, quantized MOE가 하나의 일관된 P18 model identity와 고정 소스 권위를 공유할 때만 native A/B 연구용 입장을 허용. 발행자 검증 실패·누락·모델 버전 불일치·작업 미완료·재사용·파일 변조·레이스는 모두 `BLOCKED`.
5. **별도 통계/운영 게이트**: 이 복사 source authority PASS는 모델 사용 준비 증거일 뿐 480문항 NLL/과제 정확도/latency/10,000 bootstrap/Bonferroni 품질 PASS가 아니다. P17 incumbent A, `automatic_live_promotion=false`, XOX serving PID/route 불변을 계속 강제한다.

## 3. 구현 전·후 QA (모두 격리 테스트)

다음 부정 시험은 독립 temp fixture / mock registry에서만 수행하며 진짜 `EOE_P18_*_RECEIPT.json` 경로에 가짜 데이터를 생성하지 않는다.

- receipt absent, 이름만 있는 0-byte 파일, 존재하지만 잘못된 JSON, 필드만 유사한 자체 발급 JSON.
- 서명/registry 검증 실패, 다른 release ID, 다른 model/tokenizer/index SHA, 원본↔사본 SHA mismatch, MoE/BF16 pairing mismatch.
- 전송 종료 코드가 0이 아니거나 해당 job이 등록되지 않음, 권위 없는 transfer operation id, symlink/partial shard, 이전 authority replay.
- 동시 issuance 경쟁, append-only/CAS 충돌, 실패 후 중간 출력을 PASS로 오인하는 문제.
- 긍정 fixture는 **테스트 전용 issuer**로만 구축; 실제 운영 authority와 혼동 금지. 완료 시 모델 17/17과 원래 frozen cohort의 source pin을 별도로 확인.
- 원본 `hcp_p18_eoe_480_readonly_admission.py` 변경 전 테스트 리포트에서 기존 취약 조건이 검출되는지 확인하고, 신규 승인 검증 경로로 모든 음성 시험 fail-closed, 정상 등록된 서버 소유 증거만 PASS인지 검증.

## 4. 릴리스 / 승인 순서

1. BEGLIN 연구 브랜치 또는 Tailnet 격리 development session에서 계약·테스트 구현, 원본 모델/개인 token/raw 480 dataset 업로드 금지.
2. current/live/runtime **fresh** 상태 확인, 최신 current immutable release 베이스로 재베이스, full QA 및 release_stage→release_verify 증거 확인.
3. 릴리스 전환이 별도 승인되는 경우에만 최신 `expected_current` CAS로 activate하고 runtime 수렴 증거를 남긴다. 병렬 release를 덮어쓰지 않는다.
4. 공식 authority/연구용 EOE native A/B 시험은 등록된 제한된 endpoint만 사용. 권위가 없으면 본 문서와 무관하게 BLOCKED.

**현 결론**: 지금은 정식 source seal/production 변경 없이 963토큰 개발용 BF16 parity 검증만 독립 수행 중이다. 공식 source authority 차단을 다른 실행 권한으로 우회하지 않는다.
