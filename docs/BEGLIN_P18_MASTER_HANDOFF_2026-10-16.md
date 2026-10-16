# BEGLIN / vDSP 국소 양자화 엔진 — P17·P18 통합 핸드오프
작성일: 2026-10-16 KST
최종 실증 시점: 2026-10-16 18:36 KST 전후
범위: P17 기준선 → P18 native 1024 위치 → 480문항 외부 평가 → 원본 BF16 공유 가중치 연구 → EOE/D50 이관
작업 규칙: 연구 전용, production 비접촉, 기본 정책 A 유지, challenger B의 자동 운영 승격 금지

---

## 0. 핵심 결론 / 다음 세션은 여기부터

현재 상태를 혼동하면 안 된다.

- 원본 DeepSeek-V2-Lite BF16 27계층·5,291텐서 모델, P18 1,024위치 연구용 native Metal 추론, P9의 실제 정밀도 정책 A/B 교체·ACK는 실기기에서 동작한다.
- P18의 엄격한 원시 logits 증거 집계는 **원본 BF16 10/480문항(중복 제외), BF16+A/B 동시 페어 6/480문항**이다. 페어 6문항에서 금지되지 않은 연구용 정답 접두사 8/673개, native 전체 float32 vocab logits 16/1,346개를 실제 확보·감사했다. 이 숫자들은 task accuracy나 full-cohort quality PASS를 의미하지 않는다.
- 원본 BF16 가중치/양자화 MoE 구성의 XOX→EOE/D50 **17개 파일 전체 읽기 전용 SHA 비교 PASS**. 공식 모델 복사 authority seal 2개는 안전 검사에 차단돼 미발급이다. 임의 대체 인증·우회 금지.
- **새로운 핵심 성과 (2026-10-16)**: 실제 원본 DeepSeek BF16 27계층·5,291텐서에서 공유 가중치 2문항 처리의 개발용 3/5 gold-token logits가 단독 계산과 **짧은→긴, 긴→짧은 양쪽 순서 모두 1,638,400개 float32 값 비트 단위 일치**. 한 번 실측한 공유 속도는 단독 두 번 합산 대비 1.434×(정순), 1.214×(역순). 연구 후보로 기술적 정확성 실증 PASS이나 480문항 평가·운영 승격 허가는 아니다.
- 공식 P18 480문항 전체 BF16/A/B NLL, 8/32/128 cap 완전 생성 정답률, 3회 warm fixed-work latency, 10,000회 bootstrap 및 Bonferroni 연구 게이트는 **미완료/BLOCKED**.
- 가장 중요한 운영 원칙: **P17/P18 native incumbent 정책 A를 그대로 유지**, B의 automatic_live_promotion=false. 독립 성능 판정과 수동 승격 권한을 구분한다.

## 1. 현재 운영·호스트 상태 (신선한 도구 읽기)

Tailnet Commander EOE:
- release_status: current=live=g6-0.9.0-operator.900355, converged=true, active_operation=null.
- runtime_identity: release_id=g6-0.9.0-operator.900355; EOE PID 42429. 이 릴리스는 이번 인계 작성 작업에서 변경한 것이 아니다.
- XOX 실제 serving binary SHA-256: 8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9.
- XOX active-route.json SHA-256: c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47.
- XOX serving PIDs: 35481, 35493 (2026-10-16 18:36 KST SSH 읽기 전용 재검증).
- EOE D50 연구용 스토리지 예약 후 여유 약 626GB. XOX free disk 약 30.04GB는 관측값으로 과거 19.1GB보다 회복됐으나 새 장기 작업의 배정 가능 용량은 항상 fresh resource_plan으로 다시 판정해야 한다.
- EOE CPU load1은 최신 관측 약 109이며 작업 시점마다 변동이 컸다. 무조건적인 무한 평가·병렬 GPU/CPU 작업 실행 금지. resource_plan 및 원본 런타임 identity를 먼저 확인.

운영 원본 코드/라우트/릴리스는 본 인계 작업에서 수정하거나 승격하지 않았다.

## 2. 프로젝트 목적과 엔진 지원 범위

목적: 하나의 LLM에 대해 layer/attention/KV/expert 등 국소 요소의 정밀도를 독립적으로 지정하고, 해당 정밀도 정책을 추론 중 원자적으로 바꾸며, 신뢰도·정답률·속도·메모리를 계량적으로 비교한다. vDSP/Beglin은 Apple Silicon MLX Metal 경로를 중심으로 연구한다.

검증된 부분:
- P9 precision policy A/B 재바인딩 실증: 2개 대상, REBIND_SET_APPLIED ACK, epoch 증가, source/route/PID 불변 테스트.
- P18 1,024 위치 연구용 candidate C/MLX source와 native 실기기 107/203/962 토큰 프롬프트 시험; 962 입력 + 1 생성의 최대 963 위치 GPU 동작 증거.
- BF16 P17 단독 teacher와 P18 원본 BF16 27계층 계산의 EOE/XOX 교차 검증.

아직 일반적으로 입증되지 않은 부분:
- 모든 요소/모든 모델 아키텍처에 대한 임의 국소 양자화.
- CPU와 GPU의 모든 정밀도 옵션 및 동일한 품질·성능을 보편적으로 제공한다는 주장.
- B가 A보다 비열등하거나 우월하다는 통계적 판정, production 승격.
위 세 항목은 연구 목표이지 현재의 출하 인증이 아니다.

## 3. P17 기준선 및 P18 평가 계약

이전 P17 A/B 평가의 비열등성 차단 기록: B-A gold NLL 약 +0.027109, bootstrap upper bound 약 +0.143137, 사전 기준 +0.05를 초과해 B 승격 실패. 이 값은 P17 인계 기록이며 이번에 새로 재계산한 값이 아니다. Incumbent=A.

P18 원본 외부 평가 권위:
- 480문항 원본 BPE/프롬프트/정답 SHA-256: c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df.
- MMLU 160, ARC-Challenge 160, GSM8K 160; shot 0/3/5 구성.
- Gold positions 673개, 정책 A/B 전체 원시 102,400 float32 logits 요구량 1,346개; prompt+gold/decode를 합친 최대 연구 위치 963.
- 사전등록 통계 프로토콜 SHA-256: 77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c.
- 480문항 scorer cohort SHA-256: 19e7080b7208b8c7dd6fd06a656bb400dd2a9f72ca5f5674a80f5cc9a654cf65.
- 원본 tokenizer.json SHA-256: 41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669.
- Native P18 C는 frozen gold prefix SHA 673개만 full-vocab F32 캡처할 수 있도록 allowlist, opt-in env 및 exclusive/O_NOFOLLOW 파일 생성 안전 조건 적용.
- Quality gate는 독립 BF16 positive control, A/B 원시 logits의 NLL 재계산, 각 데이터셋의 완전 task output, 고정 workload 반복 latency, 문제 단위 통계 검정/Bonferroni를 종합해야 한다. 기술적 1토큰 smoke나 단독 BF16 gold NLL은 완전 생성 정답률로 인정되지 않는다.
- 영구적인 운영 승격은 연구 quality gate와 별개의 수동 승인 절차를 필요로 한다.

## 4. XOX native P18 실제 GPU 실증

1024-position 연구용 candidate:
- Separate research-only native source/binary; original serving stage untouched.
- 107토큰과 203토큰 input의 실제 Apple MLX GPU 1-token smoke PASS; 203 dev-only A/B 전환 PASS.
- 962-token dev-only input + 1-token decode에서 policy A/B 실제 full-vocab finite GPU logits와 native ACK 확인. RSS 약 6.48/6.50GiB, 단일 실행 wall 약 19.04/17.34초(비교용 warm latency 증거 아님).
- 1024-position C lazy KV arena ASAN/UBSAN 테스트 458건 PASS. 이는 native 전체 480 품질 검증과 구분한다.

원본 BF16와 native A/B의 실제 gold NLL을 독립 확인한 6개 문항(아래 숫자는 실측 일부 자리 반올림, 모두 대표 표본이 아님):

| frozen item | BF16 gold NLL | A gold NLL | B gold NLL | B-A |
|---|---:|---:|---:|---:|
| p18-mmlu-000 | 1.017820 | 0.515175 | 0.630186 | +0.115011 |
| p18-arc_challenge-000 | 0.331265 | 1.272472 | 1.319196 | +0.046724 |
| p18-gsm8k-000 (gold 3개 평균) | 3.818793 | 3.753863 | 3.656252 | -0.097610 |
| p18-mmlu-018 (장문 203) | 2.372429 | 2.2368 | 2.1508 | 약 -0.0860 |
| p18-mmlu-001 | 0.951138 | 2.468491 | 2.657912 | +0.189421 |
| p18-mmlu-002 | 4.299583 | 3.130000 | 2.965443 | -0.164557 |

- 엄격하게 중복 제거한 A/B paired coverage: 6/480문항, gold 8/673 positions, native A+B raw full-vocab F32 16/1346 vectors.
- P18 실측 3데이터셋 독립 감사 JSON SHA: 2d9bd085bcfb28871ad9d8b42f82bb9347b02e0fcb9897a6ab6baf25ce57bd7e.
- 203-long 원본 BF16+A/B 독립 감사 SHA: af761db9964bce9d4229f2b66d1ef6b6da7887baa2c77c4c006e83e8ecbc55d2.
- 6문항 집계 JSON SHA: dae63937830b7e382bb1e59c145df97fe597f2efa3aee98e578d404435b4d0b3.
- 결과 변동 방향이 문항별로 다르며 6문항은 의도적으로 작게 선택된 실기기 pilot이다. 이 평균을 사용해 사전등록 비열등성 PASS/FAIL을 판정하지 않는다.

## 5. EOE/D50 연구 환경 이관 및 원본 BF16 계산

EOE D50의 checkpoint 복사:
- XOX 원본 DeepSeek-V2-Lite BF16 4 shards, 총 31,413,626,576바이트; quantized MoE 본체/보조 파일; 총 17개 원본/복사본 SHA-256 읽기 전용 대조 17/17 PASS.
- 실제 BF16 네 샤드에서 원본 P17 값 SHA 목록의 결정론적 텐서 48/48 대조 PASS.
- Original BF16 checkpoint index SHA-256: d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf.
- Original P17 TensorStore 5,291개 tensor value manifest SHA-256: 88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda.
- Quantized MoE main binary SHA-256: 9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4.
- 17/17 읽기 전용 교차검증 집계 JSON SHA: 284cb4e58db6739fffaea0481b3562b38225b1c170681c4f83a6b5aec3512ab5.
- **중요한 제한:** 별도로 요구되는 EOE_P18_BF16_MODEL_COPY_RECEIPT.json 및 EOE_P18_MOE_BASE_COPY_RECEIPT.json 공식 authority seal은 2026-10-16 최신 경로 검사에서도 없음. 공식 인증 도구 호출은 이전 안전 검사에서 차단됐다. 읽기 전용 17/17 바이트 일치와 공식 모델-source authority seal은 다른 계약이며 하나를 다른 하나로 위조·대체하지 말 것.

EOE 원본 BF16 PyTorch 계산:
- EOE에서는 원본 P17 계산과 비교해 입력 길이 검사 112→1024만 확장한 연구용 원본 클래스 사용.
- XOX vs EOE에서 동일 p18-mmlu-000의 102,400 full float32 logits **원시 NPY SHA까지 일치, 최대 절대 차 0.0**. SHA: 7025aed3a8c23fdc024f9a2f4e4a9c4944129f09d1bc3c32633c57c21a1249b6.
- EOE/XOX BF16 전체 로그잇 비트 동일 감사 SHA: 8cc0b3dfbeccff083297e66a17e4ef2eda654b6cde2b2ac7738fb863b9690d56.
- EOE original BF16 완료 디렉터리: unit_000, unit_003, unit_004, unit_005, unit_006; 최신 EOE 목록 재읽기로 확인.
- 새로운 BF16-only 4개 중 p18-mmlu-003 gold NLL 0.0541907802, -004 0.0277405525, -005 1.5891014715, -006 6.1624655836.
- 이전 A/B paired 6개와 EOE-only 4개를 중복 제거한 **BF16 unique verified coverage = 10/480**. EOE unit_000은 XOX 기존 000의 교차 재현이라 신규 1문항으로 중복 가산하지 않는다.
- 추가 BF16 기술적 장문 시도(p18-mmlu-088, 931+1 positions)는 독립 통합 coverage에 포함하지 않는 별도 technical pilot.
- EOE에서 완전한 native A/B full-model forward는 **아직 미인증**. 원본 BF16 forward가 PASS여도 A/B source authority 미발급은 우회하지 않는다.

## 6. 새 2026-10-16 핵심 성과: 원본 27계층 BF16 공유 가중치

목표: 연속된 두 BF16 평가 입력이 같은 원본 5,291 tensor를 두 번 읽는 비용을 줄인다. 개별 문항의 unpadded causal mask, position IDs와 original decoder 계산은 동일하게 유지한다. 원본 수학, RoPE, MoE, expert routing, 정답위치 scoring 의미를 바꾸지 않는다.

초기 설계 검증:
- v1은 한 위치 logits만 반환했으므로 multi-gold GSM8K scoring 불충분, v2에서 prompt_length와 gold_len (1~8) 전체를 명시.
- 합성·정적 CPU 회귀 45/45 PASS. 이것만으로 DeepSeek 원본 호환성을 주장하지 않았었다.

이번 실제 BF16 qualified 개발용 연구 실측:
- source role: development_only_not_480_test, 기존 MMLU dev 기반 203토큰. 입력 SHA: b628ec3247cfc7a94785a93e539562a7a2bc7726f133c21162479749b8f0ac2c.
- short: prompt101+gold3, long: prompt198+gold5 (모두 pseudo-gold 개발용). 480 external held-out frozen 문항을 튜닝에 사용하지 않음.
- 각 독립 단독 original forward가 27 decoder layers / 5,291 original BF16 tensors ×2 로드. shared는 두 행을 각 층에서 처리하면서 **계층 텐서 로딩 총 5,291회/1회 shared pass**.
- EOE D50에서 4회 실제 실행, 각 원본 TensorStore load+value SHA 검증 PASS:

| source forward | original tensor reads | wall seconds | peak RSS GiB | new swap |
|---|---:|---:|---:|---:|
| standalone short(101+3) | 5,291 | 149.044618 | 2.138 | 0 |
| standalone long(198+5) | 5,291 | 210.379304 | 2.145 | 0 |
| standalone 합계 | 10,582 | 359.423922 | — | — |
| shared short→long | 5,291 | 250.639138 | 2.153 | 0 |
| shared long→short | 5,291 | 296.184593 | 2.164 | 0 |

- Shared short→long 실측 speedup 1.4340배 / 단독 합산 대비 시간 절감 30.266%.
- Shared long→short speedup 1.2135배 / 시간 절감 17.595%.
- short 3×102,400 F32, long 5×102,400 F32, 두 처리 순서 각각 비교해 **1,638,400개 F32 logits 비교 모두 byte-exact / 최대 abs diff=0**.
- EOE D50 실기기 결과 폴더: P18_DEVONLY_BF16_STAGED_PARITY_20261016 (standalone_short, standalone_long, shared_short_long, shared_long_short, 두 개 집계 JSON).
- original per-run model-source 5,291 tensor SHA 고정, 비트 일치 검증 추가 외부 독립 auditor PASS.
- original shared-v2 source SHA: c160e79ecb527ccfcc7c908e7fc64f67abf8938a4186c2f7033eac753c7cde2f.
- 4단계 staged runner SHA: cf5b20128bfa93d151d24eb1969397a9e1def35025c13f295b392a953f6f7ff.
- Original stage summarizer receipt SHA: 0002852bec74dc9d1ac447403bd5af2bd4b6b501b3cd91c944f6c3a14f0ef0ef.
- **독립 real F32 auditor JSON SHA**: 6c8ca8aaf80c09f5eb9176f6d50f4300784213f7b245f8a2a841be10000fd19c.
- GitHub research commit: 7cbbf73aae0880bc5aec1a8a292742816535d6f2.
- **판정**: 이 두 개 개발 입력에서 실제 원본 27L BF16 정확성/가중치 재읽기 최적화의 feasibility PASS. 단 한 번의 wall-time 결과이므로 CPU 부하, warm/cold variance, 더 긴 512/963 문맥, 여러 길이·다양한 유형 일반화는 미실증이다. GPU native A/B 속도나 480문항 전체 성능 개선을 의미하지 않는다.
- 공유 v2 경로를 기존 480 공식 BF16 참고값의 권위 경로로 **자동 교체 금지**. 반복 dev-only boundary·개별 baseline과 bit parity가 더 축적된 후 연구용 후보로 제한 적용 여부 판단.

## 7. 독립 평가기 / QA 증거

- 480 source/원본 tokenizer 정확성에 묶은 task scorer: MMLU/ARC 단일 문자 정확 일치, GSM8K 전체 출력 정규화된 십진 숫자 비교. 정상 EOS/정확한 cap 종료·생성 토큰 SHA가 필요.
- Source/worker evidence checker: 원본 BF16 full-vocab NPY, native A/B 409,600바이트 F32, C real GPU witness/ACK/token SHA, 모델 stage before-after, float64 logsumexp NLL 재계산.
- Synthetic and adversarial QA 총 55/55 PASS (scorer 30 + raw evidence 25).
- QA receipt SHA: 98801d5e675f115272af4fe63c60ba6b10bce230fb2d7b89eb018a5ce3670c5a.
- EOE BF16 read-only raw coverage verifier: hcp_p18_eoe_bf16_independent_raw_coverage.py, source SHA efabf2efb862b766c8a817b7068d81251cfcac394fe91270887e242fbadd171c. 마지막 유효 결과는 EOE BF16 5개 재실측, 전체 중복 제거 BF16 10, old A/B paired 6; 새 평가 run은 없었다.

## 8. 작업 디렉터리, 증거·GitHub 경계

EOE D50 private canonical root:
  /Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015

주요 하위 경로:
- p18_multishot_frozen_PRIVATE.json (480 test token bank; **외부 공개·GitHub 업로드 금지**).
- p18_devonly_203_hardware_input_PRIVATE.json (개발용 고정 203-token source; GitHub raw 금지).
- teacher_checkpoint_source_copy/ (31.4GB 원본 BF16 복사, 4 shards).
- native_moe_base_source_copy/ (MoE base 가중치).
- eoe_p18_bf16_480_runs/unit_NNN/ (EOE original BF16 raw NPY/receipts).
- native1024_real_evidence/ 및 480_stream/staging/ (XOX 실제 F32/BF16 복사·QA; private).
- P18_DEVONLY_BF16_STAGED_PARITY_20261016/ (새 원본 real shared dev-only QA).
- P18_DEVONLY_BF16_STAGED_PARITY_20261016/P18_INDEPENDENT_STAGED_REAL_DEVONLY_BF16_AUDIT.json (독립 auditor).
- P18_EOE_READ_ONLY_17_SOURCE_FILE_SHA_CROSSCHECK_NOT_SEAL_20261016.json.
- P18_EOE_XOX_FULL_102400_BF16_LOGITS_BIT_PARITY_20261016.json.

EOE 문서 canonical:
  /Users/eoe/mcp-sandbox/tailnet-commander/BEGLIN_P18_MASTER_HANDOFF_2026-10-16.md

GitHub private:
- 저장소: popixoxipop-collab/beglin
- 연구 브랜치: research/hcp-p18-external-multimetric-20261015
- 2026-10-16 현재 원격 참조 HEAD 읽기: 7cbbf73aae0880bc5aec1a8a292742816535d6f2
- 최신 원본 real-shared 핸드오프: docs/BEGLIN_HCP_P18_ORIGINAL_27L_BF16_SHARED_PARITY_HANDOFF_2026-10-16.md
- 이전 통합 480 raw QA 인계: docs/BEGLIN_HCP_P18_TASK_RAW_AUDIT_HANDOFF_2026-10-15.md
- 모델 이관 및 새 BF16 인계: docs/BEGLIN_HCP_P18_EOE_MODELS_SHA_BF16_2REAL_HANDOFF_2026-10-16.md

GitHub에는 **코드·안전하게 요약한 hash receipt·handoff 문서만** 허용. 테스트 원문, raw token IDs, 모델 가중치, 원시 F32/NPY, worker private 로그/시크릿은 GitHub 커밋 금지. EOE D50 보존본에 SHA/사본 위치 기록 유지.

## 9. 남은 차단 조건과 우선 순위

### P0 — 보호 및 권위 수렴 (절대 생략 금지)
1. 매 세션 fresh Tailnet release_status + runtime_identity, XOX active stage/route SHA/PID, EOE resource_inventory 및 resource_plan. release current/live/runtime 불일치, active_operation 존재, serving identity 변경 시 실행 중단.
2. 모델 소스 17/17 읽기 전용 hash 비교와 **공식 EOE model-copy authority seal 계약을 구분**. 이전 --seal 안전 검사 차단을 도구 우회나 인증영수증 위조로 해소하지 말고 허용된 원래 서버 소유 검증·승인 경로를 확보한다.
3. EOE CPU load 폭증(실측 load1 약 109) 시 원본 27L 작업 병렬 시작 금지. XOX SSD 보호, D50 예약, 스왑/메모리 여유 감시. 이미 완료된 per-stage 공유 연구 실행을 재실행하거나 덮어쓰지 않는다.

### P1 — 공유 BF16 generalization
4. 연구용 **dev-only** 512·963/1024 경계, gold1/3/5/8, sequence order interleaving에 대해 원본 단독 vs 공유 full 102,400 F32 logits의 bit-exact parity, 원본 텐서 SHA, 적재 횟수, RSS/swap을 재검증. 기존 frozen 480 held-out test를 최적화·튜닝에 사용하지 않는다.
5. 같은 입력을 여러 cold/warm 반복해 실제 속도 단축률 분포와 CPU load 영향, I/O 대기 vs 디코더 실행 시간을 분리한다. 빠르지 않거나 일치하지 않으면 공유 후보 FAIL 유지하고 P17/P18 단독 teacher를 기준선으로 남긴다.
6. 효율을 확인하더라도 별도 reviewer의 **공식 평가용 공유 BF16 경로 승인** 전까지 480 품질 gate input 변경 금지.

### P2 — P18 EOE native A/B 권위 검증
7. 공식 모델-copy authority seal이 PASS일 때만 EOE full native A/B original 1024 binary를 연구용으로 시험. 먼저 기존 203토큰 DEV 입력에서 XOX A/B 동일 full logits 및 policy ACK/epoch/route 불변 parity; 그 후 frozen 480 실험에 진입.
8. 정확한 source-bound SHA 673 allowlist, policy B 2-target transaction, finite logits, native C witness/RSS·GPU, before/after PID를 집계. 정책 A incumbent·promotion OFF 유지.

### P3 — 480 외부 평가 및 판정
9. 전체 480문항(원본 0/3/5-shot) BF16 교사 원시 logits, 정책 A/B 673 gold prefixes 각 102400 logits, 독립 SHA/NLL 재계산을 **질문별 append-only D50 영수증**으로 완주한다.
10. 8/32/128 cap에서 BF16/A/B 완전 생성의 실제 termination/정답률을 원본 frozen scorer로 측정. Gold prefix 1-token smoke를 task accuracy로 잘못 쓰지 않는다.
11. 정책 전환 제외 고정 workload latency 3회/문항, warm/cold, RSS/스왑/오류와 source provenance를 함께 확인한다.
12. frozen cohort 단위 10,000 bootstrap, family-wise Bonferroni, 기존 +0.05 margin 등 사전등록 계약으로 비열등성 판정한다. PASS하더라도 실제 운영 B 자동 승격 금지, 별도 승인·rollback·shadow/canary 후 단계적으로 판단한다.

## 10. 재개 시 실행 예시 (읽기 전용 / 실패 시 중단)

EOE에 연결하여 root를 다음과 같이 정한다:
  R=/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015

읽기 전용 근거:
  python3.11 "$R/hcp_p18_eoe_bf16_independent_raw_coverage.py"
  python3.11 "$R/hcp_p18_eoe_480_readonly_admission.py"
  python3.11 "$R/hcp_p18_eoe_d50_model_copy_seal.py" --inspect
  shasum -a 256 "$R/P18_DEVONLY_BF16_STAGED_PARITY_20261016/P18_DEVONLY_STAGED_PARITY_FINAL_RECEIPT.json" "$R/P18_DEVONLY_BF16_STAGED_PARITY_20261016/P18_INDEPENDENT_STAGED_REAL_DEVONLY_BF16_AUDIT.json"

최초 상태 점검:
  Tailnet Commander EOE release_status, runtime_identity, resource_inventory(eoe,xox), resource_plan(적합 리소스 요청).
  XOX: serving binary/active-route SHA와 실제 stage PIDs fresh read-only.
  GitHub: 연구 브랜치 HEAD와 handoff 최신 SHA 재읽기.

주의:
- 본 문서의 read-only 명령 외 실제 계산/정밀도 변경은 resource_plan eligible=true 및 모든 권위 계약 PASS 후 수행.
- 기존 P18_DEVONLY_BF16_STAGED_PARITY_20261016 네 stage의 원시 값·receipt는 불변 보존. 같은 경로 재실행·덮어쓰기 금지.
- 공식 --seal 차단을 삭제, 특권 변경, 수동 proof 위조, 다른 플러그인 우회로 해결하지 말 것.
- 여기 적힌 이전 완료 성과와 최신 도구 조회 결과를 구분하며, 미실행 항목을 완료로 기재하지 않는다.

---

## 11. 최종 판정 (2026-10-16)

검증 PASS:
- P18 원본 독립 양자화 후보의 MLX Metal native 단일/장문 GPU 실증.
- XOX 원본 BF16 + A/B 원시 gold NLL 6문항 / native 16개 F32 vector.
- EOE original BF16 source / XOX full 102400 logits byte-exact 복제 재현.
- XOX↔EOE original BF16+MoE source 17/17 SHA read-only parity.
- EOE independent BF16 raw evidence 누적 10/480 unique.
- Shared BF16 연구용 원본 DeepSeek 27L/5291T 실모델 short/long 양방향 정확 일치 1,638,400 F32 요소 / 실제 1.434×·1.214× 단일 실측 speedup.
- 평가 채점기·원시 증거 55/55 QA 및 공유 v2 합성·정적 45/45 QA.

미인증/BLOCKED:
- EOE native A/B full-model source authority seal 및 실기기 EOE 페어 추론.
- Shared BF16 범용 context 성능·반복 실기기 CI 및 480 official teacher 전면 대체.
- P18 480문항 전체 NLL/accuracy/latency/통계 비열등성.
- 정책 B의 운영 승격.

**운영 결론: 정책 A 유지, 자동 라이브 승격 OFF, P18 연구 자료와 장기 영수증은 append-only 보존.**
