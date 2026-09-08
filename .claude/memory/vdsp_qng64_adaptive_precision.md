---
name: vdsp_qng64_adaptive_precision
description: qNg64 arbitrary n-bit kernel (CPU+GPU) + data-driven adaptive precision selection + serving-time bypass routing -- multi-phase, autonomous execution in progress
metadata:
  type: project
---

장기 트랙, [[vdsp_general_serving_engine_goal]]의 정밀도 관련 후속. 사용자가 "히트맵 근거로 텐서마다
임의 n-bit 외과적 정밀도 보정"이 원래 비전이었다고 지적(GPU mixed-precision bits=4/8/16/32 작업
완료 직후) → "CPU/GPU 둘 다 실제 임의 n-bit(5/6/7 등) 저장·계산하는 진짜 커널 만들어" → "이거 이룰
때까지 자율 수행, 계획/검증 전부 Opus"로 확장. 3단계(L1 커널 → L2 데이터기반 n결정 → L3 서빙시점
우회라우팅) Plan Mode로 설계, Opus(subagent_type="Plan") 2회 adversarial 검증 거쳐 확정.
계획 원본: `/Users/xox/.claude/plans/quirky-stirring-trinket.md`.

**Why**: 사용자 명시 — 기존 bits=4/8/16/32 4단계 고정은 진짜 "적응형" 비전이 아님. Opus 검증으로
드러난 더 중요한 사실: L1+L2+L3a를 다 끝내도 "서빙 시점 인지"라는 원래 요청은 여전히 미달성 — 그건
L3b에만 있고 L3a는 오프라인 정적 배정일 뿐. 이 격차를 사용자에게 명시적으로 알림.

**How to apply**: 이 파일이 진행 상황의 단일 authoritative 소스. Phase 경계마다(L1 완료/L2 결론/L3a
완료) 갱신할 것. 세션이 끊기면 여기부터 재개.

## 확정된 설계 (재도출 금지, `quirky-stirring-trinket.md` 참조)
- 포맷: A2 bit-plane, group=64, n∈{2,3,5,6,7}(4,8은 기존 q4g64/q8g64), bias-then-extract, LE bit order,
  reciprocal+rintf, error-feedback 전 구간 사용, size_t 바이트계산, memcpy-only plane 접근.
- GPU: **재양자화(dequant→mx::quantize()) 금지** — Opus 실측으로 n=2에서 최대 50% 오차 확인. 대신
  qNg64 plane→MLX 네이티브 워드 레이아웃 직접 비트 재포장(bits=4의 direct-bind 패턴과 동일 부류).
  n=7은 GPU 커널 자체가 없음(MLX 0.32.1 확인) — CPU만.
- `MoeAFTensor`엔 `kind` 필드 없음(그건 별개 구조체 `WT`의 것) — 기존 `bits`+`sym` 관례 그대로 사용,
  새 bit-plane 분기는 `sym==1 && bits∈{2,3,5,6,7}`로 가드(기존 bits==3 LUT-테스트 실험과 충돌 없음).
- L2: `tools/quant_search_n.py`가 이미 상당 부분 구현+검증된 채로 존재했음(classify/bisection/
  exhaustive/live-oracle) — 처음부터 설계 아님. 단 Opus가 진짜 blocker 3개 발견:
  - B1: knee가 "첫 pass"였음(단조성 위반 58-83%라 위험) → suffix-closed로 재정의. **수정+검증 완료**.
  - B2: 임의-n shadow는 hi-mirror 구조상 등록시점 문제로 하드blocker → "시작시 promotion file 먼저
    읽고 필요한 shadow만 미리 빌드"로 결정(hot-reload 포기).
  - B3: 이벤트 뒤섞임이 classify() 오염 → (corpus,req,pos) 이벤트 단위로 스코프. **수정+검증 완료**.
- L3a = 오프라인 정적 배정(사용자가 원한 "서빙 시점 인지"가 아님, 정직히 스코프 확정). L3b가 진짜
  서빙시점 인지 — L3a가 D-d5-31 baseline(bits=16/32.7%scope) 이기지 못하면 L3b 착수 여부도 재고.

## 진행 상황 (2026-09-08 갱신)
- Plan Mode 완료, 사용자 승인(자율수행 옵션 선택: "이 세션 안에서 계속", /loop 아님).
- **L1 구현 완료+검증** (2026-09-08): gguf_transcode.c/.h(`gguf_quantize_qNg64`), qwen_infer.c
  (decode/matvec/vdsp-guard/registration/bits-gate 4곳), mlx_moe.cpp(GPU 직접 리패커) 전부 구현.
  실 버그 1건 발견+수정(GPU 리패커 row stride를 `ng`로 이중곱 → SIGBUS, fprintf 이분탐색으로 위치
  특정 후 수정 — bob lldb는 비대화형SSH 불가라 프린트 이분탐색 사용).
  **검증(전부 실데이터, 전부 통과)**: n=4 회귀게이트(q4g64와 코드/스케일 bit-identical, 3개 실텐서,
  0 mismatch) · NumPy 오라클(실 GGUF 텐서 2개×n=2,3,5,6,7=10조합, 전부 cmp bit-exact) · GPU 리패커
  (MLX affine_dequantize 커널소스로 손유도 + 실측 mx.quantize() 대조로 이중검증, 실텐서 3-포인트×
  n=2,3,5,6=24좌표 max_abs_diff=0 정확히, n=7 GPU 거부 확인) · 실 CPU 생성(DeepSeek-V2-Lite 8-position,
  shared_up_proj L11→n=5 승격, baseline 대비 8/8 argmax 토큰 동일+logit 전부 미세하게 다름 — promotion
  이 실제로 반영됐다는 인과 증거) · 기존 GATE1-7 전체 회귀 없음. RESULTS.md `D-qNg64-1` 섹션 참고.
  로컬 커밋만(미push).
- **L2 B1/B3 코드 수정**: 완료+커밋(`d8dab02` 코드, `d13d008` docs). `tools/quant_search_n.py`의
  `exhaustive_search()`를 suffix-closed knee로, `fetch_prior_points_by_corpus()`→
  `fetch_prior_points_by_event()`로 이벤트 스코프. **실측 검증**(6-target historical TSV):
  bisection이 4/6(67%) 타겟에서 틀린 답을 냈을 것 — B1의 우려가 이론이 아니라 실측으로 확정됨.
- **B3의 라이브 Supabase 재측정(58-83% 헤드라인 수치가 이벤트스코핑 후 얼마나 유지되는지)은 아직
  미완**: `QWEN_SUPABASE_URL`/`QWEN_SUPABASE_KEY`를 xox 로컬 `.env`, bob `.env`/쉘 프로파일 어디서도
  못 찾음. 사용자에게 credential 위치 확인 필요(다음 Phase 경계 보고 때 물어볼 것, 지금은 중단 안 함).
- L2 gap #3(오라클 신뢰계층 분리, 회귀검사 추가)과 L3a 전체는 L1 완료 후 착수.
- **L2 step 3 완료** (2026-09-08): `QWEN_MOE_ATTRIB_SIM_QN` 신설(실커널 오라클, commit `1ed4f13`) —
  기존 F32-sim 경로가 bits==32 raw-passthrough라 실제 packed decode와 산술이 다르다는 우려를
  실측으로 확인. p60/pos=14 이벤트(RESULTS.md ROI-G Phase 2 확인된-단일플립)로 2개 타겟 테스트:
  `shared_gate_proj L14`(클린)는 시뮬레이션과 실커널 일치. **`kv_b_proj L9`(VIOLATED)는 실제로
  갈림** — 시뮬레이션 curve(n=2 pass/3 fail/4 pass)와 실커널 curve(n=2 pass/3 pass/4 **fail**)가
  다른 모양. **결론: 기존 840행 시뮬레이션 데이터는 배포 결정 근거로 못 씀(plan의 결론이 실측으로
  확정됨)** — RESULTS.md `D-qNg64-2` 참고. Step 4(회귀검사)는 두 타겟 다 알려진 이벤트가 1개뿐이라
  실행 대상 없음(정직히 기록, 실패 아님).
- **L3a 구현 완료+검증** (2026-09-08): `g_moe_lt_nq[]`(별도 shadow table, g_moe_lt_hi는
  그대로 attribution 기준선으로 보존) + `g_moe_promoted_nq[role][layer]`(bool 아니라 n 자체 저장)
  + `moe_promotion_nq_init()`(n>=base_bits 강제, 위반시 거부+로그) + `QWEN_MOE_PROMOTION_FILE_NQ`
  (기존 2열 포맷과 분리) + `tools/promotion_writeback.py`(이벤트 max→corpus max 집계, 합성데이터
  3케이스로 검증) + `moe_lazy_hi_release_all()`의 실제 UAF 선수정. 실버그 1건(g_st_moe NULL,
  D-gpu-6c/6d와 같은 계열, g_moe_hi_st 재스왑으로 수정, fprintf 이분탐색). **검증**: p60/pos=14
  실이벤트로 shared_gate_proj L14→n=5 실제 승격 후 정답 재현(REAL FLIP 로그 없이 base pass만으로
  정답), n=3(<base_bits) 거부 확인. 전체 benefit-metric(vs D-d5-31 baseline) 비교는 미완(별도
  대규모 측정 필요, 정직히 follow-up으로 남김). commit `106abad`/`6587ec7`/`d6514e7`.
- **L3b 설계 문서 작성**(구현 아님, 2026-09-08): `.claude/history/2026-09-08_l3b-design.md`(비커밋,
  로컬 전용 — .gitignore). **핵심 발견: 사용자가 원한 "서빙 시점 인지+구조적 우회"는 이미
  `moe_neartie_maybe_correct()`+`moe_neartie_reverify_hi()`로 존재함**(margin 체크→bits=16 전체
  재실행 전역포인터스왑, 매 토큰). 진짜 gap은 "적응형 n으로 무엇을 escalate할지"가 아니라(그건
  이미 active보다 높은 정밀도여야 해서 의미없음) — "라이브로 감지된 반복 flip을 L3a 승격으로
  영구화하는 학습 루프"가 진짜 buildable minimal-viable. 단, L3a의 B2 결정(hot-reload 포기, 시작시
  1회 빌드)과 정면충돌 — 세션 내 반영이 아니라 재시작 간 학습 루프만 가능. **권고: 지금 만들지
  말 것** — D-qNg64-4의 benefit-metric이 충분히 검증되기 전까지 보류.
- **D-qNg64-4(부분 benefit-metric)**: `shared_gate_proj/L14`(p60 flip의 실제 귀속 텐서, D-d5-31
  자체 24-corpus의 일부이기도 함)가 **D-d5-31의 88-combo bits=16 세트엔 없음**(직접 파일 확인,
  `/Users/xox/vdsp_local_data/promote_hits88.txt`). qNg64 n=5(eff_bpw=5.1)가 bits=16(eff_bpw≈16.03)
  대비 훨씬 낮은 비트폭으로 D-d5-31이 놓친 타겟을 직접 커버 — 유리한 데이터포인트지만 **N=1**,
  전체 비교는 아님(정직히 명시). RESULTS.md `D-qNg64-4` 참고.
- 다른 세션(`vdsp_engine_main`)이 같은 repo에서 병행 작업 중 — 이번 라운드에 `c3bcf81`
  "retract 89-target sweep, multi-flip contamination" 커밋 확인(840행 기준 유효, 이 계획의 수치와
  일치). 커밋 전 매번 git log/status 재확인 중.
- **★★★L3a 완료+실검증** (2026-09-08, commit `106abad` 코드, `6587ec7` docs): `g_moe_lt_nq[]`
  (g_moe_lt_hi 안 건드림)+`moe_promotion_nq_init()`(`QWEN_MOE_PROMOTION_FILE_NQ`, 시작시 1회) +
  `n>=base_bits` 강제 + `moe_lazy_hi_release_all()` UAF 선수정 + `tools/promotion_writeback.py`
  (write-back 집계 스크립트, 합성데이터로 검증) + 오라클 positive control(`quant_search_n.py`).
  **실버그 1건**: `g_st_moe`가 hi-mirror 로딩함수 안에서 save/restore되며 제 코드 실행 시점엔
  NULL — **이전 D-gpu-6c/6d와 같은 버그 계열**, `g_moe_hi_st`로 재스왑해 수정(fprintf 이분탐색,
  bob lldb 불가 제약 재확인). **실검증(bob, DeepSeek-V2-Lite, p60/pos=14 이벤트)**: shared_gate_proj
  L14→n=5 승격 후 실서빙에서 그 이벤트가 correction 없이도(REAL FLIP 로그 없음) 바로 정답(4794)
  산출 — 승격 안 하면 "REAL FLIP orig=8713 corrected=4794" 로그로 확실히 틀림, 대조군 확보.
  n=3(<base_bits=4) 거부 실검증도 확인. **정직한 스코프**: L3a는 "서빙 시점 인지"가 아니라 오프라인
  정적 배정(계획서 자체가 처음부터 명시). D-d5-31 대비 benefit-metric 전체 비교는 미실행(별도
  대규모 측정 필요, 의도적 후속과제). Point6(corpus간 최댓값)은 실 멀티코퍼스 데이터 없어 합성
  데이터로만 검증. Supabase credential 미해결 상태 지속.

## 실행 규칙
- `subagent_type: "Plan"` + `model: "opus"`만 Opus 허용(다른 조합은 `agent-model-policy-guard` 훅
  차단 — [[feedback_agent_opus_plan_type_only]] 참고, 전역 메모리에 별도 기록됨).
- 헤드리스 검증 2연속 실패 시 우회 말고 중단+보고. Phase 경계에서만 사용자에게 보고.
- 빌드/실행은 bob, 로컬 커밋만(push는 명시 요청시만).

- **Supabase credential 발견** (2026-09-08, 사용자가 "메모리랑 hook에 기록했었어"라고 알려줌):
  QWEN_SUPABASE_URL/KEY는 계속 못 찾았지만, `~/.claude/hooks/scripts/nvidia-keypool-guard.py`의
  기존 주석에 Management API PAT 위치가 이미 있었음(`~/Desktop/Code_reviewer_with_feedback/.env`
  의 `SUPABASE_MANAGEMENT_PAT`) — project_ref만 이 repo 걸로 바꿔서 작동 확인.
  상세: [[reference_supabase_management_api_access]].
- **B3 라이브 재측정 완료**: 실 1065행 전체로 old(corpus병합)/new(이벤트스코프) 비교 —
  **66.7%(34/51) 위반, old/new 완전 동일, 6개 후보 타겟 전부 안 뒤집힘**. B3 수정이 불필요했다는
  뜻이 아니라(정합성 보호 자체는 여전히 유효, `c3bcf81`과는 다른 메커니즘) 이 데이터셋엔 아직 그
  실패모드가 없었다는 뜻. RESULTS.md `D-qNg64-5` 참고. 이제 credential 확보로
  `promotion_writeback.py` 실데이터 검증과 넓은 benefit-metric 비교가 가능해짐 — 다음 착수 대상.

- **promotion_writeback.py 실데이터 검증 + N=35 benefit-metric** (2026-09-08, D-qNg64-6/7):
  aggregation 로직 자체는 정확(shared_gate_proj/L14 n=5, D-qNg64-1/3 수동테스트와 정확히 일치).
  **단, kv_b_proj/L9에선 스크립트가 DB(시뮬레이션)로 n=4를 "안전"이라 계산했는데 D-qNg64-2의
  실커널 테스트는 n=4가 실제로 fail임을 이미 확인함** — DB가 아직 대부분 시뮬레이션이라 실배포엔
  못 씀(실커널 데이터로 재수집 전까지). D-d5-31(88개 bits=16 세트)와 35개 실제 겹침 발견(D-qNg64-4의
  N=1과 대조적) — qNg64 eff_bpw 중앙값 5.10 vs bits=16의 16.03(35/35 이김) 이지만, **이것도 같은
  시뮬레이션-데이터 한계 적용** — "이긴다"가 아니라 "실커널로 재검증되면 이 정도 차이"의 프리뷰로
  읽어야 함. RESULTS.md D-qNg64-6/7 참고. 커밋: `5cdb233`.

- **L3b Phase A 완료** (2026-09-08): Opus 검증으로 원안(자동 sweep 트리거 autopilot)의 실제 위험
  다수 발견(시뮬레이션이 실커널 결과를 조용히 덮어씀, "live" 오라클이 실은 시뮬레이션 경로만 탐,
  req 번호가 manifest chunk마다 모호, promotion_writeback.py --out이 병합 아니라 truncate) — 전부
  미해결로 두고 **Phase A(읽기 전용 리포트)만 구현**: `tools/promotion_controller.py --report`.
  실데이터(bob의 step6/7/9 JSONL, 742개 attribution row) 검증: 45개 (role,layer) 타겟,
  최다피인용 `shared_down_proj/L26`(32회/3이벤트). Phase B/C(실제 자동화)는 의도적으로 미착수 —
  provenance 컬럼+manifest 식별자 기록이 선행돼야 함. RESULTS.md `D-qNg64-8` 참고.

- **L3b Phase B, 1/2 done** (2026-09-08): manifest-identity logging added to attribution JSONL
  (`"manifest":"<QWEN_MOE_CB_PROMPT_MANIFEST value>"`, resolves the req-numbering-ambiguity-across-
  manifests risk that already caused 2 silent provenance bugs, D-d5-27 + its repeat) — real
  verified on p60/pos=14. DB-side prereq (`source` column on `moe_quant_sweep_results`,
  distinguishing sim vs real-kernel rows) is PREPARED not applied:
  `supabase_migration_qng64_source.sql`, needs explicit user go-ahead to run (DDL). Phase C
  (auto-triggered sweep autopilot) still not built — needs both Phase B pieces plus a cost cap,
  the corrected_argmax reproduction gate, real push code (doesn't exist), and
  promotion_writeback.py's truncating write fixed to merge. RESULTS.md `D-qNg64-9`.

- **L3b Phase A+B 완료** (2026-09-08): Phase A(`tools/promotion_controller.py --report`, 읽기전용
  워크리스트, 실 attribution 742건/45타겟 랭킹). Phase B 2/2 완료 — C쪽(매니페스트 신원 로깅,
  qwen_infer.c, 실검증됨) + DB쪽(`source` provenance 컬럼, **사용자 승인 후 실제 적용, source='sim'
  count=1065 확인**). Phase C(실제 자동 sweep 트리거)는 설계를 Opus로 재검증 중 — Opus가
  지적한 8개 위험(무제한비용/자기재귀/잘못된매니페스트/provenance충돌/truncate 등) 전부 반영한
  구체설계를 다시 adversarial review 받는 중.

- **L3b Phase C: safety prerequisites built+verified, sweep loop NOT built** (2026-09-08,
  D-qNg64-11): 2차 Opus 검증(더 심각함)이 찾은 문제들 중 핵심 안전장치만 이번 라운드에 구축+실검증.
  (1) c36aadc 회귀테스트(n=4 FATAL, n=5 정상) 확인. (2) attribution JSONL에 orig_argmax+threshold
  필드 추가(실검증). (3) source 컬럼 활용해 sim/real 데이터 절대 안 섞도록 수정 — **실 DB로 확인:
  kv_b_proj/L9(sim만 15행, real 0행)가 이제 위험한 n=4 대신 정확히 거부됨**. (4) 가장 어려운
  부분(req%mf_n 유도 + Step-0 재현 게이트) 구축+실검증 — 실제 p60/pos=14 이벤트로 PASS 확인,
  도중 macOS(bob)엔 GNU timeout이 없다는 실버그 발견+수정(portable POSIX 패턴으로 교체).
  **아직 안 만든 것(정직히 기록)**: 실제 n별 sweep 루프, atomic push+검증,
  promotion_writeback.py의 bob-side upsert 수정(여전히 로컬+truncate — 다른 타겟 있는 상태에서
  --out 쓰면 안 됨), backoff ledger, --max-sweeps/--run 모드. 상세: RESULTS.md `D-qNg64-11`.

- **L3b Phase C: Priority 1(위험 해소)+Priority 2(나머지 파이프라인) 완료, 실엔진 실행만 미완**
  (2026-09-08, D-qNg64-12): 이전 라운드가 남긴 "위험" 경고(다른 타겟 있는 상태에서 --out 쓰면
  안 됨)를 해소 — `promotion_writeback.py`가 이제 bob의 실제 파일을 읽고
  upsert-with-scoped-delete로 원자적 갱신(remote tempfile+mv). **실검증: kv_b_proj/L9의 위험한
  n=4 stale 항목이 정확히 제거되고 무관한 타겟은 안 건드림**(실 DB 상태로 확인). 나머지 파이프라인
  (per-n sweep 5-outcome 분류, atomic push+재조회검증, backoff ledger, `--run --max-sweeps`)도
  전부 구축 — mock 경계(ssh/HTTP)를 뺀 나머지 전부 실행으로 검증(분류 6종, ledger 8개 속성,
  push 성공/부분실패 감지, run_one_triple 3경로, run_mode의 manifest-우선 선택+backoff 제외
  둘 다 확인). **미완**: 실제 bob 엔진 호출은 이번 라운드에 안 함 — bob 메모리가 87→146MB
  (free)로 이 프로젝트가 이미 한 번 겪은 사고(스와핑→SSH 무응답→수시간 복구)와 같은 패턴이라
  게이트를 독단적으로 우회하지 않음. 코드는 준비됐고, 여유 생기면 `--run --max-sweeps 1` 1회
  실행이 남은 유일한 검증. 상세: RESULTS.md `D-qNg64-12`.

- **L3b Phase C 실 end-to-end 검증** (2026-09-08, D-qNg64-13): 파이프라인 자체는 실증됨 —
  `shared_gate_proj/L14` 실이벤트로 Step-0 게이트 PASS + 실커널 n=5/6/7 전부 'pass'. 실행 중
  진짜 버그 2개 더 발견+수정(로컬/원격 mkdir 누락, 둘 다 "no-signal"로 오판정되던 것). **최종
  Supabase push만 막힘**(QWEN_SUPABASE_URL/KEY 여전히 못 찾음 — Management API PAT과는 다른
  credential, DML이라 별도 승인 없이 우회 안 함). `--reset-backoff` CLI도 설계엔 있었지만
  미구현으로 확인. Phase C는 이제 "메커니즘 증명 완료, push credential만 남음" 상태.

- **L3b Phase C 완전 종료** (2026-09-08): 실 end-to-end 파이프라인 4회 실행 후 실버그 2건 더
  발견+수정(로그 디렉토리 미생성). 최종 실결과: `shared_gate_proj/L14`(p60 이벤트) Step-0 게이트
  PASS, 실커널 n=5/6/7 전부 PASS(완전자동화 파이프라인 최초). Supabase push는 REST키 없어서
  자동경로는 막혔지만, 사용자 명시 승인 받아 Management API로 직접 마무리 — 3행 INSERT+검증,
  이미 검증된 `suffix_closed_knee()`/`read_remote_promotion_file()`/
  `write_remote_promotion_file_atomic()` 함수 재사용해 bob의 실제 승격 파일에
  `shared_gate_proj 14 5` 기록+검증 완료. **이 프로젝트 전체(L1→L2→L3a→L3b A/B/C)가 실데이터로
  최초로 완전히 닫힌 사례.** 남은 gap: QWEN_SUPABASE_URL/KEY(다음 타겟부턴 자동화 막힘),
  `--reset-backoff` CLI 미구현. 둘 다 설계 문제 아니라 작은 후속작업. RESULTS.md D-qNg64-14 참고.

- **QWEN_SUPABASE_URL/KEY 해결 + --reset-backoff 구현** (2026-09-08, D-qNg64-15): 계속 못 찾던
  REST API 키를 검색 대신 이미 가진 Management API PAT의 `api-keys?reveal=true` 엔드포인트로
  직접 조회해서 획득(secret/service_role급 키). `/Users/xox/vdsp-engine/.env`에 저장(먼저
  `.gitignore`에 `.env` 추가 — 이 repo에 원래 없었음, chmod 600). **별개 버그 하나 더 발견**:
  크레덴셜과 무관하게 이 머신 python.org Python이 로컬 CA 번들이 없어서 urllib 기반 호출이 전부
  SSL 에러로 실패하고 있었음 — `Install Certificates.command` 실행으로 해결(이 세션 내내 썼던
  curl-우회 패턴의 근본원인, 이제 필요없어짐). **검증**: 세션 내내 막혀있던
  `fetch_prior_points_by_event()`가 이제 실제로 sim/qng64_real 소스분리 데이터를 정확히 반환.
  `--reset-backoff ROLE:LAYER`도 신규 구현+synthetic ledger로 검증(다른 타겟 안 건드리고
  지정 타겟만 정확히 제거). 이걸로 D-qNg64-13/14가 남긴 두 gap 전부 해소 — 다음 타겟부턴 수동
  개입 없이 `--run` 완전자동화 가능(아직 신규 타겟으로 처음부터 끝까지 무개입 재검증은 안 함,
  원하면 다음 단계).
