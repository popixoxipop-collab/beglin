# BEGLIN P18 — 963-token original BF16 shared-weight boundary 실기기 통합 인계

검증일: 2026-10-16 KST (최종 확인 19:58 전후)
범위: 원본 DeepSeek-V2-Lite BF16 27계층·5,291텐서, 개발용 합성 963토큰 입력과 기존 203토큰 입력
판정: **963 original BF16 baseline PASS, shared 963+203 full F32 logits bit-exact PASS; source authority/480 gate BLOCKED**
운영 영향: **없음**; P17 incumbent 정책 A 유지; `automatic_live_promotion=false`

## 1. 963-token 원본 BF16 실기기 검증 성과 (신규)

- 실기기: EOE Mac Studio CPU, D50 원본 BF16 checkpoint 복사본, original 27 decoder layers/5,291 tensor stream.
- 개발 데이터: SHA-pinned 개발용 203-token prompt를 결정적으로 반복한 963-token synthetic prefix, prompt 955 + pseudo-gold 8. 별도로 원래 203 tokens, prompt 198 + pseudo-gold5를 공유 계산에 사용했다. 두 데이터는 모두 `development_only_not_480_test`의 확장이다. **원본 frozen 480 test set 미사용**.
- 구현 스크립트는 원본 샤드/모델 분기 외에 운영 serving나 정밀도 정책 A/B를 변경하지 않았다. 해당 후보 스크립트 SHA-256: `46ec82d51462c8a93d6ddd8d9862becbbd89652fdceee18d82bbbe4355d33b1d`.
- 원본 963 단독 패스: Tailnet process `proc_9afe0284ea04f6534832584d7050f5ec`, terminal `EXITED 0`, gold 8×102400 F32 logits, wall `355.8469163752161` 초, peak RSS `2,461,073,408` bytes (~2.292 GiB), new swap 0. Immutable baseline receipt SHA-256: `1f69b7cef045daf788219eb1f06b0cc01f80931ef1ee9d43ea1921d389c93407`.
- 원본 963+203 공유 패스: process `proc_d8d411d0171169df891b394e2134e310`, terminal `EXITED 0`, 2개 시퀀스 gold 8+5, 전체 `1,331,200` F32 logits, 원본 TensorStore 5,291 tensor 단일 스트림, wall `396.25575720798224` 초, peak RSS `2,466,578,432` bytes (~2.297 GiB), new swap 0. Immutable shared receipt SHA-256: `736bdcca78c5503785cc292fe6272300e10609ec18c1382161d0114e9fb5059b`.
- 별도 감사 단계 `--context 963 --phase audit`: terminal exit 0, `P18_DEVONLY_BOUNDARY_F32_BYTE_EXACT_PASS`, 원본 단독 기준선 대비 전체 `1,331,200/1,331,200` float32 값 byte-exact, max abs diff=0. Audit receipt SHA-256: `cb0264ae60089e38b647e216e3fc4a26fa39b655f4ad39f6177c12ce0cbfd6da`.
- 별도 OS `shasum -a 256` 원시 NPY 재검증:
  - 963 standalone gold8 raw NPY SHA = 963 shared gold8 raw NPY SHA = `56eab8b1120727cbc6e981155462c432249835cd8b72685617945fda077698d3`.
  - 기존 203 standalone gold5 raw NPY SHA = 963 shared 203 gold5 raw NPY SHA = `c77844de38d0294281f435b6d3bc660db653a344fa29c6b3ca57eec8cf53b0ac`.
- 단계와 OS 원시 NPY hash 비교는 각각 실제 수행됐다. 다만 감사 단계는 후보 실험 스크립트 자체의 별도 phase이며, 이를 제3자 독립 구현에 의한 재현으로 과장하지 않는다.
- 963 standalone + 이미 존재하는 203 standalone의 시간을 합한 값과 shared wall을 비교하면 단일 횟수에서 공유 경로의 효율이 관측되지만, cache 상태·입력 순서·CPU load를 통제한 3회 이상 비교가 아니므로 정식 속도 개선 인증·통계적 유의성이라고 주장하지 않는다.

## 2. 전 단계 누적 경계 확인

- 203 dev-only original BF16 27L/5291: 기존 short101+gold3/long198+gold5 공유 양방향 F32 로그잇 일치; 구간 합계 1,638,400 F32 비교. 이후 동일 dev-only 공유 반복 1회에서 819,200 F32 일치 및 독립 read-only 재감사 PASS.
- 512 synthetic dev-only: baseline 512 gold8 vs shared 512 gold8+203 gold5 = 전체 1,331,200 F32 byte-exact PASS; receipt `0004580c8759a097191be4d00cc5137ef7b0b81b9288d01f7f635bdb3895e901`.
- **신규 963 synthetic dev-only**: baseline 963 gold8 vs shared 963 gold8+203 gold5 = 전체 1,331,200 F32 byte-exact PASS; receipt `cb0264ae60089e38b647e216e3fc4a26fa39b655f4ad39f6177c12ce0cbfd6da`.
- 정확성 실증 경계는 203·512·963 synthetic/dev-only; 아직 모든 실제 자연어 입력과 1024 최대 길이, 다양한 pseudo-gold 1/3/5/8 조합, cold/warm 성능 반복을 일반화하지 않는다.

## 3. 현재 운영 / release / 평가 분리

- 최신 Tailnet EOE `release_status`: `current=live=g6-0.9.0-operator.900355`, `converged=true`, `active_operation=null`. `runtime_identity`도 동일 release, PID 42429.
- XOX 실제 serving binary SHA `8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9`, active route SHA `c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47`, PIDs 35481/35493를 963 실행 전후 읽기 전용 확인; 변경 없음.
- EOE `resource_plan`은 각 phase마다 `d50_work` EOE 단일 CPU 워커 `eligible=true`. 기존 결과 덮어쓰기 없음. 임계치 초과/신규 스왑 없음.
- 원본 XOX↔EOE/D50 BF16·MoE 17/17 파일 SHA 일치는 이전 단계 증거로 유지. 그러나 공식 두 건의 `EOE_P18_BF16_MODEL_COPY_RECEIPT.json` 및 `EOE_P18_MOE_BASE_COPY_RECEIPT.json` **server-owned authority seal은 계속 미발급**.
- Frozen 480 품질 조사 범위는 이번 세션 종료 후에도 original BF16 **10/480 unique**, BF16+native A/B paired **6/480**. 963 실험은 **dev-only**이므로 480 진척에 추가하지 않는다.
- 정확도/complete generation/3회 warm workload latency/10,000-bootstrap+Bonferroni 비열등성은 **미인증/BLOCKED**. 정책 B 운영 승격·promotion OFF 유지.

## 4. 모델 복사 authority 진입 경로 보안 발견 (추가 수리 문서)

- EOE `hcp_p18_eoe_480_readonly_admission.py` SHA `c80cb18c5f2b2ed0b3fa25b4dde0ac5bda9bdd30cfd740191e3c157fbe58f33f`.
- 현재 파일 존재 조건만으로 copy receipt readiness를 계산하는 코드(37–38, 54행) 확인; 실제 issuer/source digest/복사 operation authority 검증 불충분.
- 기존 복사 `--inspect`는 ready-to-seal, incomplete=0이나 공식 issuance는 여전히 BLOCKED. 제한을 우회해 seal을 생성하지 않았다.
- 상세 repair contract: EOE `/Users/eoe/mcp-sandbox/tailnet-commander/BEGLIN_P18_SOURCE_AUTHORITY_GATE_REPAIR_2026-10-16.md`.
  문서 SHA-256: `4487b53bdc686e1e2c3c9f4ae38f5ba7e6a824e7e877ed653e7e4c854cfdf4ae`. GitHub `docs/BEGLIN_P18_SOURCE_AUTHORITY_GATE_REPAIR_2026-10-16.md` (커밋 `9dc8bc8d94aae558479cafa23c14208211af5f9c`).
- 서버 소유 검증·발급자 인증·원자성·매칭 source SHA/transfer operation id 결속, admission-side fail closed, synthetic 부정 테스트가 선행돼야 함. 단순 파일 수작업·영수증 위조·권한 우회는 허용하지 않는다.

## 5. 개발 산출물 및 안전 재개

Private D50 root: `/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015`

Append-only:
- `P18_DEVONLY_BF16_BOUNDARY_963_20261016/baseline/receipt.json`
- `P18_DEVONLY_BF16_BOUNDARY_963_20261016/shared/receipt.json`
- `P18_DEVONLY_BF16_BOUNDARY_963_20261016/INDEPENDENT_F32_PARITY_AUDIT.json`
- 위 각 단계 `original_devonly_source_*_float32.npy` 원시 자료는 D50 private 유지. GitHub에는 원문 token IDs, 480 frozen corpus, 모델 weights, raw F32/NPY, secret 등록 금지.

다음 순서는:
1. 203/512/963 dev-only 경계 통과를 baseline으로 1024 최대 길이 및 pseudo-gold 1/3/5/8 순서·반복·multi-prompt exact parity/OS 원시 SHA 비교 확장.
2. Independent source-authority server-owned 발급/registry integration 코드 및 **격리 QA** (fake receipt/잘못된 source SHA/잘못된 operation/replay/CAS/서명 검증 실패 등), 기존 480 admission 존재-only 판단 수리.
3. 인증 발급이 실제 PASS일 때만 EOE native A/B 원본 모델 203 dev-only 실기기 비교, 다음 frozen 480 BF16/A/B 완전 평가지표 진행. 운영 B 승격은 별도 명시적 승인.
4. Tailnet runtime/release 승격이 필요할 경우 기존 `current/live_release` 최신 재확인→재베이스→stage/verify→`expected_current` CAS 승인 전환. 이번 단계는 release를 변경하지 않았다.

**요약: 원본 BF16 공유 가중치 963 경계 실증 PASS, 전체 task quality/source authority BLOCKED, 운영 경로와 정책 A 불변.**
