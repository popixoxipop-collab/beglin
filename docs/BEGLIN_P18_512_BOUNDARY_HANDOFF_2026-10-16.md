# BEGLIN P18 — 512-token original BF16 shared-weight boundary handoff
작성일: 2026-10-16 KST (마지막 확인 약 19:35)
분류: research-only / development-only / production untouched / no live promotion
상위 마스터: /Users/eoe/mcp-sandbox/tailnet-commander/BEGLIN_P18_MASTER_HANDOFF_2026-10-16.md
마스터 SHA-256: 366231fd95e86152f1526906828c477021634485bca5ee1b4f5f1f288aebb736
선행 재개 문서 SHA-256: 1dfb53c770d962c0861d16d5684bbb4c518e9fe01e0ed733abbfaead472c930f

## 1. 신규 검증 PASS — 512 / 203 dev-only 전체 F32 비트 동등성

- 사용 모델: 원본 DeepSeek-V2-Lite BF16, 27 decoder layers, 정확히 5,291 원본 tensor 읽기/패스.
- 입력: 원래 SHA-pinned 203-token DEVELOPMENT-ONLY 데이터만 이용하여 토큰들을 결정적으로 반복한 합성 512-token(504 prompt + pseudo-gold8) 및 원래 203-token(198 prompt + pseudo-gold5). 원본 480개 held-out 테스트 문항 미사용.
- 검증 방식: 512 gold8 단독 original forward, 이어서 512 gold8 + 203 gold5 공유 원본 weight-stream forward. 별도 audit phase는 immutable receipts 및 NPY 재읽기/whole-vocab F32 bit 비교.
- 512 original standalone: EXIT 0; 전체 819,200 F32 logits; 원본 5291 tensor; wall 244.54509079176933s; peak RSS 2,367,406,080 bytes (약 2.205 GiB); new swap 0. Process proc_381fd693bcb2046c00fe0ecc414857ae.
- 512 + 203 shared: EXIT 0; 819,200 + 512,000 = **1,331,200 F32 logits**; 두 입력 모두 standalone reference에 대해 **byte-exact**, 최대 절대 차 0; 원본 5,291텐서를 한 번 스트림; wall 281.31926729111s; peak RSS 2,375,270,400 bytes (약 2.212 GiB); new swap 0. Process proc_1a8a044233eca650b280d93f9dbb3350.
- 공유 512 gold8 raw NPY SHA는 standalone와 동일: 109f04fa99d1d47459221e54a7da453e7ea4405cb3c8c51dd689c8ce8044267b. 203 gold5 공유 raw SHA: c77844de38d0294281f435b6d3bc660db653a344fa29c6b3ca57eec8cf53b0ac.
- 독립 audit phase exit 0 / P18_DEVONLY_BOUNDARY_F32_BYTE_EXACT_PASS; 비교한 F32 값 1,331,200개; 품질 인증=false, 자동 운영 승격=false.
- raw evidence는 EOE D50 연구용 비공개 디렉터리에만 남겼고 GitHub에 원시 logits/NPY/토큰 ID/가중치/테스트셋을 올리지 않았다.

## 2. 저장된 원본 파일과 고정 SHA

Private D50 root:
  /Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015

이번 후보 스크립트:
  hcp_p18_devonly_long_boundary_20261016.py
  SHA-256 = 46ec82d51462c8a93d6ddd8d9862becbbd89652fdceee18d82bbbe4355d33b1d

기존 original staged source unmodified SHA:
  cf5b20128bfa93d151d24eb1969397fa9e1def35025c13f295b392a953f6f7ff

새 D50 append-only 512 scope:
  P18_DEVONLY_BF16_BOUNDARY_512_20261016/
  baseline/receipt.json SHA = 7eeb413bc83f70da72e862801bd2efccceaec0733794851fed074d0877d72883
  shared/receipt.json SHA   = 5a1a7724b38a14d192525351d0684ea02b985c30d7278c7da5c6b6abc9ecbeb5
  INDEPENDENT_F32_PARITY_AUDIT.json SHA = 0004580c8759a097191be4d00cc5137ef7b0b81b9288d01f7f635bdb3895e901

작업 중 한 번 발생한 안전한 오류:
- 초기에 신규 후보 스크립트의 original staged source 고정 SHA를 한 글자 잘못 적어 첫 launch가 fail-closed (exit 1). 실제 original source SHA drift가 아니었고 모델 계산 전에 중단됐다.
- 자신의 신규 후보 스크립트만 expected_current SHA(CAS)로 정확한 SHA에 고쳐 재실행. 원본 소스/모델/기존 영수증 및 live는 변경하지 않음.
- 별도의 외부 감사 소스 신규 쓰기는 보안 검사에서 차단돼 미생성. 따라서 이번 PASS는 기존과 동일한 신규 스크립트의 별도 audit phase가 원시 파일과 이전 단독 기준선을 다시 비교한 결과이며, 별도 독립 구현 2중 감사까지 완료한 것으로 주장하지 않음.

## 3. 현재 운영·평가 상태와 권한

- Fresh Tailnet release_status/runtime_identity 확인: current=live=runtime=g6-0.9.0-operator.900355, converged=true, active_operation=null, PID 42429.
- 512 baseline 및 shared 실행 직전 XOX original serving binary/active-route SHA, 기존 serving PID(35481, 35493), EOE resource_plan eligible=true 검증 통과.
- 512 연구 작업 전후 현재 Tailnet 릴리스 불변. 이 스크립트는 serving route/정책 A/B/production을 호출하거나 변경하지 않음.
- 공식 XOX→EOE model-copy seal 2종은 아직 미발급. 17/17 SHA 복사 검증과 정식 authority seal은 별도 계약; 우회 또는 수동 receipt 위조 금지.
- frozen original BF16 raw verified unique 10/480; native A/B paired 6/480. 본 512 합성 dev-only 증거는 이 카운트를 증가시키지 않음.
- BF16/A/B 전체 480 NLL, 8/32/128 완전 생성, 3회 warm latency, 10,000-bootstrap Bonferroni 비열등성 인증 BLOCKED. incumbent policy A 유지, automatic_live_promotion=false.
- 512 단독과 공유의 시간을 하나의 통제된 cold/warm 페어로 측정한 것은 아니므로 속도 우월성·통계적 비열등성 주장 금지.

## 4. 963토큰 경계 실행 차단 및 안전 재개

- 963 개발용 합성 입력에 대한 1회 CPU resource_plan은 eligible=true였다.
- 그러나 963 baseline 시작 직전 필수로 요청한 XOX serving binary SHA 읽기 도구가 OpenAI 보안 검사에서 차단됨. fail-closed 판단으로 963 baseline/shared를 **실행하지 않았다**. 다른 경로나 관리자 명령으로 우회하지 않았다.
- 다음 세션은 정당하게 허용된 읽기 전용 경로에서 release_status + runtime_identity current/live/runtime 수렴, XOX serving binary/route SHA/PIDs, EOE resource_plan 적합을 먼저 확인. 모두 PASS일 때만 개발용 original 963 baseline (8 pseudo-gold) → 963 + 기존 203 shared (8+5 pseudo-gold) → append-only audit 단계로 진행.
- 불변 모델 source SHA/원래 203 dev source SHA 검증 유지, D50 각 phase 기존 결과 재실행·덮어쓰기 금지, 모니터링 메모리·swap·storage guard 유지.
- 본 실행용 연구 스크립트는 사용 가능하도록 남겨뒀지만 963 신규 작업/예약/자동화는 생성하지 않았다.

## 5. 다음 우선 순위

1. 963 dev-only 장문 boundary의 original standalone ↔ shared whole F32 bitwise equality를 합법적 preflight 통과 후 증명.
2. 공식 model-copy source authority contract를 server-owned 승인 경로로 복구. 영수증 파일 존재만으로 신뢰하는 gate는 강화 검토하되, seal 금지·우회 금지 유지.
3. 개발용 long/repeated cold/warm latency/I/O 분해 및 상이한 원본 dev 시퀀스에서 독립 재현.
4. AUTHORIZED source seal PASS 후에만 EOE native 원본 A/B 203-token ACK/epoch/route parity 및 frozen 480 평가.
5. B 운영 자동 승격 여전히 금지, 운영 승인 별도.

최종 판정: **P18 dev-only original BF16 512-context full-logit bit-exact feasibility PASS, 963 unrun on safety preflight, official source authority & frozen 480 quality remain BLOCKED.**
