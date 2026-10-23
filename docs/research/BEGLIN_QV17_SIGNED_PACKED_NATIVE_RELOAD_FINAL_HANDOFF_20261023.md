# BEGLIN Qv17 — 서명 검증된 대체 정밀도 디스크 캐시의 실제 C 재로드 및 독립 감사

**연구일:** 2026-10-23 KST  
**연구 저장소:** `popixoxipop-collab/beglin`  
**연구 분기:** `research/qv17-signed-packed-cache-20261023`  
**검증 판정:** `ACTUAL_ORIGINAL_28L_QWEN_C_SIGNED_DEV_PACKED_RELOAD_PASS / 48_OF_48_ORIGINAL_STATIC_NLL_EXACT / ED25519_AND_NATIVE_SHA256_FAIL_CLOSED / PRODUCTION_RELEASE_NOT_READY`.

## 1. 해결 대상: 매번 startup 시 40개 대체 텐서를 재양자화하는 Qv16 병목

Qv16은 Qwen2.5-1.5B-Instruct 28계층의 실제 C/NEON CPU 엔진에서 요청 사이 Q4↔n8 40개 텐서를 실제 변경한다. 96MiB 제한의 inactive packed/scales 캐시를 startup에 준비하면, 최초 요청의 약 **2,432.433ms** 전환을 **0.284ms**로 줄일 수 있었지만, startup BF16→F32→Q4/n8 양자화 **2,412.228ms**가 여전히 들었다.

**Qv17은 대체 정밀도 40개의 packed+scales 데이터를 owner-only D50 파일 하나로 영속화하고, 새 프로세스에서 서명/원본 무결성을 검사한 후 이를 실제 live cache 슬롯에 올리는 구현이다.** 전체 BF16/양자화 모델을 복제하지 않고, Qv14/15와 동일한 선택 규칙(85토큰 미만 Qv8, 이상 random73)을 사용한다. 품질 최적 동적 할당 정책의 효과를 평가한 실험이 **아니다**.

## 2. 원본 네이티브 C 경계와 자산 형식

- 부모 Qv16 고정 C 소스 SHA-256: `b05c2e8aac488f9accba44c61073a72f2103be6284b37aa5de0a500185995a9f`.
- 분리 Qv17 C 소스 SHA-256: **`0c9d797c0066fdb1f8ca5972bdf7611ae75431050d0d55c0f0fdef0653f6bdaa`**.
- 실제 Apple CPU 네이티브 Qv17 바이너리 SHA-256: **`16085378788b64b229f321b49c11540e286edffbff832db4e41fbf8d2a87643b`**.
- 원래 BEGLIN Q4 및 QNg64 n8 NEON 오브젝트 **23개를 원본 그대로 링크**했다. `forward_token`, `final_logits`, gold-token NLL 식 계산 코드 SHA-256 `10c84007027d15a9327c15ec687c2327e19390de65f03bf9d95aee4bd4c9fe72`는 부모 Qv16과 바이트 동일했다.
- Packed artifact schema `BEGQ17A1` v1: little-endian Apple arm64 전용, 24바이트 헤더 + 40×32바이트 역할/계층/precision/shape 바인딩 + 실제 40개 packed/scales 연속 구간. 파일 **87,541,016 bytes = 83.4856MiB**, 실제 텐서 payload **87,539,712 bytes = 83.484375MiB**.
- 부가 inactive cache 예산은 여전히 **96MiB**, 최종 실제 추론 중 cache peak **90.046875MiB**. 변환 중 새 Q4/n8 모델 전체의 다른 버전을 상주시키지 않는다.
- C 로더는 소스에서 `CommonCrypto SHA256`으로 전체 파일 해시를 **같은 열린 FD에서 계산한 다음 rewind 후 읽고**, 서명 검증된 호출자가 넘긴 64자리 SHA와 일치하는지 검사한다. 40개 인덱스·역할·계층·Q4/n8 선택·out/in shape·packed/scales 바이트·전체 길이·trailing bytes·cache cap을 전수 검사한다.
- C 로더의 SHA 결과 검사는 독립적인 안전 장치지만 Ed25519는 **Python trusted launcher에서 검증**한다. C 바이너리를 직접 불러서 올바른 파일 SHA를 임의 지정하는 호출자까지 막는 네이티브 서명검증 구조는 **아직 구현되지 않았다**.

## 3. 개발용 로컬 Ed25519 서명 체계와 원본 체크포인트 검증

- Ed25519 개인키는 **EOE D50의 소유자 전용 디렉터리(0700), 파일(0600)**에만 보관했다. GitHub나 최종 문서에 키 바이트는 절대 업로드하지 않았다.
- 공개키는 `docs/research/BEGLIN_QV17_ED25519_DEV_TRUST_ROOT_PUBLIC_20261023.txt`, 파일 SHA-256 `5f535c54985ac18257b50c4bd7969ec4ce57aecb766def485c68c8b0a4ad4847`. Python launcher는 그 파일 자체의 SHA-256을 코드로 고정하고, 서명 문서의 Ed25519 서명을 검사한다.
- Pack artifact SHA-256: `49c409a79588b79dcefeba60e09b71459ac8df2d02ac0bda5e2cc1ab8076a86b`.
- Signed manifest SHA-256: `491deb61dbc30fad08b4a5d819b270b64167d5551b422823ae0b431f4a5c6824`.
- Detached signature SHA-256: `2c346634543b222890a99e0536fcca7ec6488b09a7cac05aa5562fe31c0b168c`.
- Original Qwen2.5-1.5B BF16 source model file **전체 3,087,467,067 bytes**의 SHA-256 `073400b311d049b6379b8f3f7ad5a0258268031dd009b99389cfd4030144850a`.
- Source config SHA-256 `86afa7d8ef2e622fb9a917c75008198e791eac225937c687c0910fc6e46f377e`; source Qv16 frozen 48 inputs SHA-256 `a95f678f656b8d04f74f0bb643542cdd12bde8e317c8d7fa013795c4c4b3c296`; Qv8/random73 original mask SHA를 서명 문서에 함께 묶었다.
- 별도 연구용 `qv17_local_dev_signing.py`는 manifest·binary·model full hash·mask·input 바인딩을 확인한 후, signature를 만들고 새 실행 시 반드시 검증한다. **이 키는 로컬 개발용 자체 생성 키**이지 운영용 공개 인증기관/릴리스 인증서가 아니므로 외부 공급망 인증 완료라고 주장하지 않는다.

## 4. 실제 Apple Silicon 네이티브 생성 및 새 프로세스 로드

Owner-only D50 경로: `/Volumes/D50/TailnetCommander/work/beglin-qv17-signed-packed-cache-20261023/`.

**실제 생성 과정**:
- 원본 C에서 startup BF16 양자화로 40개 대체 packed/scales를 재구성 후 해당 메모리를 `BEGQ17A1` artifact 파일로 실기기 기록. 4개 원본 source-only 잠금 ARC validation 문항의 동적 C 점수 원본 reference와 일치. 생성 자산 SHA를 독립 기록.
- Native 파일 쓰기 **39.626ms**, payload 83.4844MiB, full file 83.4856MiB.
- Native create Tailnet job `job_a4de33178e707609dfb31f1e334df0ce`: **SUCCEEDED / exit 0**.
- Local Ed25519 detached manifest signed, 서명 생성 중 원본 BF16 전체 SHA 해시 **1,541.045ms** (직접 계측, 파일 캐시 조건).

**실제 새 프로세스 로드 (원본 모델 재실행)**:
- 전체 패킹 파일의 Ed25519 detached signature, 공개키 trust-root, artifact SHA, model BF16 file의 실제 전체 3GB SHA, frozen policy/instruction SHA, derived C/binary SHA를 모두 Python bootstrap에서 검증. 직접 계측 **1,488.393ms**.
- 다음으로 새 원본 C 모델 프로세스에서 native SHA256 체크 **42.956ms**, SHA 포함 디스크 packed cache 총 로딩 **54.557ms**. file source는 OS page cache 영향이 있을 수 있으며, 물리적으로 캐시가 완전히 비워진 완전한 cold boot는 측정하지 않았다.
- 48개 Qv16 사전봉인 ARC-Challenge validation 문항 평가 성공: **34/48 정답, mean full-vocab gold-token NLL 1.2628019266720496**. 47회 요청별 precision-mask 전환, **1,880회 원래 WT 정밀도 변화에 대해 cache hit 1,880, requant miss 0**. 전환 누적 **13.625ms**.
- Native per-case NLL과 정답은 기존 Qv16 prewarm1/원본 정적 Qv8 24문항과 원본 정적 random73 24문항에 **각각 48/48 완전히 동일**. 독립 감사 프로그램이 원본 48 C 로그를 다시 파싱해 확인했다.
- Qv17 reload Tailnet job `job_fe9a46f93023893f67c67885f7a1ad98`: **SUCCEEDED / exit 0**. Native process peak RSS **4,725.344MiB**. 총 process wall **267.967초**지만 CPU/OS 부하/모델 초기화 시간 통제가 부족해 Qv16 356.944초와 직접 비교한 전체 처리량 개선 주장 금지.
- 원본 Qv16 warm-only startup 양자화 비용 **2,412.228ms** 대신 Qv17은 native reload 약 54.6ms + 엄격 전체 서명/BF16 출처 검증 약 1,488ms를 부담했다. 즉 별도의 두 구간을 합산하면 약 **1,543ms**이고 약 869ms가 적었지만, 다른 실행 시점과 cache 상태의 측정이므로 장기적인 콜드부팅 가속 수치로 확정하지 않는다. SHA 검증을 안 한 빠른 경로의 결과만으로 무결성을 보장한다고 주장하지 않는다.

## 5. 공격/오류 fail-closed 시험

**Python 서명/파일 오염 음성시험 3개 PASS**:
1. 정확한 Ed25519 서명 + 원래 문서 수락.
2. 서명 문서의 **한 바이트 변경** → 원래 detached 서명 검증 실패.
3. 다른 Ed25519 공개키로 검증 → 원본 서명 거부.
4. 원래 83.5MiB 파일의 임시 복사본에서 **한 비트 변경** → signed manifest의 artifact SHA와 불일치하여 거부.

구현상 음성 차단 항목은 3가지(문서 변조/키 교체/pack 변조)이며 원본 artifact·실제 개인키·원본 모델은 변경하지 않았다. 원래 pack SHA 전후 일치, 임시 파일 정리 PASS.

**네이티브 C 원본 실행 별도 음성시험 PASS**:
- Python bootstrap을 의도적으로 우회해 C 실행 파일에 **잘못된 expected SHA-256 `000...000`**을 전달하고 원래 서명된 packed file 자체는 수정하지 않았다.
- 네이티브 로더는 `FATAL: Qv17 packed cache pinned packed SHA256 mismatch`로 **C 종료코드 1 / 실제 gold-token 추론 0건**. 원본 asset digest 전후 변경 없음.
- 테스트 래퍼는 거부가 확인되었기에 성공 상태로 보고하되, 실제 네이티브 **거부**와 정상 종료를 혼동하지 않는다.
- Native negative job `job_935b1102361d7a9ec940e7e35d16f3cb`: 테스트 SUCCEEDED; 내부 의도적 C exit 1.
- 음성시험은 로컬 dev 키/런처 코드까지 장악한 공격자, 신뢰 루트 자체 교체나 외부 인증기관 침해까지 증명하지 않는다.

## 6. 공개 검증·재현 코드와 사전 봉인 근거

- `tools/qv17_build_native_persisted_cache.py` — Qv16 pinned C에서 바이트 동일한 원본 C 추론·채점식과 23개 기본 오브젝트를 사용한 격리 연구 바이너리 빌드.
- `tools/qv17_local_dev_signing.py` — 로컬 Ed25519 trust root 고정, detach sign/verify, 원본 BF16 3GB 실제 전체 SHA.
- `tools/qv17_real_native_signed_reload.py` — SHA-봉인된 48 Qv16 독립 데이터, CREATE 4/RELOAD 48 original native C 실행 및 유효한 signed-pack 검증.
- `tools/qv17_independent_signed_reload_audit.py` — 별도 구현으로 packed binary 포맷/디지털 서명/full BF16 hash 검사, Qv16 원본 정적 C 48개와 NLL bit exact 교차 검산.
- `tools/qv17_local_signed_tamper_gates.py` — 실제 임시 복사본 변조/Ed25519 음성시험.
- `tools/qv17_native_wrong_sha_fail_closed.py` — 실 C SHA 검사 fail-closed 음성시험.
- `docs/research/BEGLIN_QV17_INDEPENDENT_REAL_NATIVE_SIGNED_RELOAD48_PUBLIC_20261023.json` — SHA **`e835e1e12043f0b03dbc52dfca5e25998a60bb6a142c47c234bc8c06985a3baa`**.
- `docs/research/BEGLIN_QV17_OFFLINE_ED25519_TAMPER_TEST_PUBLIC_20261023.json` — SHA **`d2caf490f0511870c7a99884a5c518a40107a3a603c28c658e379343a3edfcfa`**.
- `docs/research/BEGLIN_QV17_NATIVE_WRONG_SHA_FAIL_CLOSED_PUBLIC_20261023.json` — SHA **`59e155d9b467b1148a393d084a62a90127cba1e415345f32c10b21f98a6fd018`**.
- `docs/research/BEGLIN_QV17_RESEARCH_DEV_ED25519_SIGNED_MANIFEST_PUBLIC_20261023.json` — SHA **`75a9d5e85337a1e1415b7df8a98e3ec6979ce2f3c76738f06de483b683d3960c`**.
- 사용자 전용 D50에 비공개 실제 packed artifact(83.5MiB), raw native 로그, 원본 연구 데이터, detached signature, 개발용 Ed25519 private key가 있다. **패킹 파일 본문과 private signing key는 GitHub에 업로드하지 않았다.**

## 7. 연구/운영 판정

**Qv17 기능 완료:** 실제 Apple CPU 원본 28계층 Qwen C 엔진에서 40개 대체 양자화 텐서의 영속화·로컬 개발용 Ed25519 문서 서명·원본 BF16/model SHA에 바인딩된 파일 무결성 검사·다음 새 프로세스 로드·48건 정확한 원본 수치 재현·변조 및 native wrong SHA 거부를 검증했다.

**아직 미검증:** 컴퓨터 물리적 콜드부팅과 OS page-cache 없는 NVMe/D50 디스크 최초 I/O, 라이브 수백 개 동시 요청의 포인터 스왑/KV-cache 안전성, 배포용 공급망 키·권한·키 회전·서명 갱신/롤백, Metal GPU residency 및 인퍼런스 경로, 다른 모델 계열, 전체 end-to-end TTFT/TPS. 단순 prefix 길이 휴리스틱이 모델 품질을 향상시킨다는 과학적 결과도 아직 없다. 이것은 **성능 최적화용 인프라의 기능 검증이지 유효성 기반 동적 텐서 정책의 우월성 증명은 아님**.

**다음 과제:** 실제 모델/가중치 revision·커널 ABI에 따라 안전하게 재사용할 수 있는 signed artifact cache store, 잘못된 서명·버전/서빙 프로토콜의 fail-closed, 제한된 RAM 조건의 prompt utility-aware precision selector를 독립 calibration/test로 평가, GPU 및 다중 요청 모드 분리 실증.

**운영 불변:** BEGLIN 원본 checkpoint/커널, main merge, Tailnet immutable G6 operator release, production auto-promotion 모두 미변경.

**한 문장 결론:** Qv17의 **서명·무결성 검증 가능한 국소 정밀도 디스크 캐시가 Apple CPU에서 실제로 저장·재로드되고, Qv16 원본 C 출력과 bit exact하며 변조를 거부한다**. 다만 로컬 개발용 서명과 CPU 캐시 실증을 운영 인증이나 정확도 우위로 확대하지 않는다.
