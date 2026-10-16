# BEGLIN P18 — 연구 전용 작업 재개 및 신규 검증 영수증

검증일: 2026-10-16 KST (최종 재검증 약 19:08)
범위: P17 정책 A 유지, P18 원본 DeepSeek-V2-Lite BF16 연구용 가중치 공유 재현성, 소스 인증 차단 상태
원본 인계: `/Users/eoe/mcp-sandbox/tailnet-commander/BEGLIN_P18_MASTER_HANDOFF_2026-10-16.md`
원본 인계 SHA-256: `366231fd95e86152f1526906828c477021634485bca5ee1b4f5f1f288aebb736`
원본 GitHub blob SHA: `c59c4fd9127f3e5452f2e685786e77f6b466cf2f`

## 1. 신규 실기기 연구 성과 (이번 세션)

- EOE/D50 단일 CPU 연구 프로세스 `proc_b1838dabb0f149e2dfdeb808f5d6443e`: 종료 코드 **0**.
- 신규 반복 프로그램 SHA-256: `bd02c3e753dfcd413db9ec68d182bf5146e1ef7ff9d98de027dc6505bc64cb15`.
- 연구 데이터: 기존 불변 **개발용 203토큰** 소스 SHA-256 `b628ec3247cfc7a94785a93e539562a7a2bc7726f133c21162479749b8f0ac2c`; frozen 480 테스트 문항 **미사용**.
- 원본 DeepSeek-V2-Lite BF16 27계층 / 5,291개 텐서 스트림 검증. 공유 short→long, pseudo-gold 위치 3+5, 전체 vocab 102,400개의 F32 logits 총 **819,200개**를 기존 불변 단독 reference와 비교.
- 반복 연구 결과: **819,200 / 819,200 byte-exact 일치**, 실측 wall **176.8586354162544초**, peak RSS **2,333,818,880 bytes** (약 2.174 GiB), 신규 swap **0 bytes**. 이는 별도의 제어된 cold/warm 벤치마크가 아니므로 반복 속도 개선의 유의성이나 production 성능을 인증하지 않는다.
- 신규 worker 영수증 SHA-256: `40eea69cf09834de2c8cf9acc6ecc449ed106bad21445ea5d32b3f1da49e6656`.
- 신규 독립 감사 코드 SHA-256: `0acb9fb04c374a99dd5e7ce0a52e2b2dd38c6695edc74ebe351e5208b40b91f2`.
- 독립 감사 결과 **PASS**: 원본 증거·worker 소스 SHA 검증, 신규 F32 NPY SHA 재검증, 불변 단독 control과 819,200개 원시 logits bit-exact. 감사 영수증 SHA-256: `b3cbdc4daf3c47935d72ef87e4694200b7e13bc929b649393ceebddb643c2d9d`.
- 연구용 민감·원시 데이터 위치: `/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015/P18_DEVONLY_BF16_REPEAT_20261016_RUN01/`. GitHub에는 NPY·토큰 ID·모델 파일을 커밋하지 않는다.

## 2. 재개 시 안전 검사 / 회귀

- EOE Tailnet Release Plane: `current=live=runtime=g6-0.9.0-operator.900355`, `converged=true`, `active_operation=null`; runtime PID `42429`.
- XOX 기존 실제 서빙 binary SHA-256: `8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9` 재확인.
- XOX 현재 `active-route.json` SHA-256: `c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47` 재확인.
- XOX 실제 서빙 프로세스 `35481`, `35493` 생존 확인. 바이너리·라우트·PID를 변경하지 않았다.
- EOE `resource_plan` 단일 CPU 연구 작업 및 D50 `d50_work` eligible=true 확인 후 1개 작업만 시작. 이전 단계 영수증·원본 파일은 덮어쓰지 않았다.
- 재실행한 변경/악성 증거 원시 QA 25/25 PASS, synthetic quality gate QA 25/25 PASS. task-scorer 30개는 이번 세션에서 도구 보안 검사로 재실행이 차단되어 **이번 신규 PASS로 계산하지 않음**; 과거 30/30 PASS와 명확히 구분.
- 생산·라이브 라우트·릴리스·인증 정책 변경 없음. `automatic_live_promotion=false`, incumbent policy **A** 유지.

## 3. 공식 P18 품질 인증 차단 상태 (완료로 표시 금지)

- 기존 SHA read-only 모델 이관 **17/17 PASS** 및 BF16/MoE 원본 복사 작업 종료 코드 0은 이미 확인됐다. `hcp_p18_eoe_d50_model_copy_seal.py --inspect`는 이번 세션에서도 `P18_D50_COPY_READY_TO_SEAL` / 누락 0 반환.
- 그러나 **공식** `EOE_P18_BF16_MODEL_COPY_RECEIPT.json` 및 `EOE_P18_MOE_BASE_COPY_RECEIPT.json`은 여전히 없음. 보호된 원래 인증 경로의 도구 보안 차단을 우회하거나 위조된 seal로 대체하지 않았다.
- `hcp_p18_eoe_480_readonly_admission.py`는 여전히 `P18_EOE_MODELS_PENDING_ORIGINAL_SHA_COPY`. BF16 골드 로그잇 검증 **중복 제외 10/480**, BF16+A/B 페어 **6/480**. 이번 개발용 반복은 frozen 480 문항 카운트에 더하지 않는다.
- EOE 완전한 native A/B 원본 모델 인증·480문항 BF16/A/B NLL·8/32/128 cap 완전 생성 task accuracy·3회 warm latency·10,000-bootstrap/Bonferroni 비열등성 gate **모두 미인증**.
- 기존 P17 정책 B 비열등성 FAIL 및 incumbent 정책 A 유지. B 자동 승격/운영 canary는 별도 권위 승인 전 금지.

## 4. 다음 실행 순서

1. **공식 source-authority 계약 복구:** 원래 서버 소유·승인 경로에서 BF16와 MoE의 원본 파일 SHA, 완료된 복사 프로세스 exit=0, immutable receipt, 권한 검증을 충족한다. 이미 있는 17/17 read-only 증거를 임의로 official receipt에 승격시키지 않는다.
2. **개발용 독립성 확대:** frozen 480과 분리하여 512·963/1024 문맥 및 pseudo-gold 1/3/5/8 경계에서 standalone↔shared full-vocab logits 비교. D50 append-only, CPU/RAM/swap guard, 반복 cold/warm 분리.
3. **반복 성능 분포:** I/O와 decoder 시간을 분리하고 순서·메모리 캐시 영향과 실측 분포 확인. 이전 1.434×/1.214× 및 이번 wall 값을 성능 보장으로 오용하지 않는다.
4. 공식 source seal PASS 후에만 EOE native 원본 A/B 개발용 203토큰 full-vocab parity/ACK → frozen 480 gold NLL/완전 생성/지연 → 사전등록 통계 검정. **운영 정책 승격은 여전히 별도 승인**.

판정: **이번 세션 dev-only 원본 공유 BF16 재현성 추가 검증 PASS, 품질 인증 및 native EOE A/B source authority BLOCKED**.
