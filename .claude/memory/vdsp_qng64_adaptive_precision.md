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
- 다른 세션(`vdsp_engine_main`)이 같은 repo에서 병행 작업 중 — 이번 라운드에 `c3bcf81`
  "retract 89-target sweep, multi-flip contamination" 커밋 확인(840행 기준 유효, 이 계획의 수치와
  일치). 커밋 전 매번 git log/status 재확인 중.

## 실행 규칙
- `subagent_type: "Plan"` + `model: "opus"`만 Opus 허용(다른 조합은 `agent-model-policy-guard` 훅
  차단 — [[feedback_agent_opus_plan_type_only]] 참고, 전역 메모리에 별도 기록됨).
- 헤드리스 검증 2연속 실패 시 우회 말고 중단+보고. Phase 경계에서만 사용자에게 보고.
- 빌드/실행은 bob, 로컬 커밋만(push는 명시 요청시만).
