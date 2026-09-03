# ROI-G Phase 2 + corpus provenance + WikiText-103 재수집 + cross-corpus 검증 (2026-09-02~03)

## 배경
ROI-G Phase 1(2026-09-02, builtin corpus 6 target)이 "attention-family 4/4 위반,
shared-FFN 0/2 위반"이라는 role-family 패턴을 찾았으나 단일 corpus 기반이었음.
사용자가 "이게 데이터 특이적인지 판단 가능해져야 한다"고 명시적으로 요청 →
corpus provenance를 스키마 레벨에서 강제하고, 두 번째 corpus(WikiText-103)를
확보해 실제로 비교하는 게 이 라운드 전체의 목표.

## D-quant-supabase-2/3: Supabase corpus-aware 확장
- `moe_role_precision_state` PK를 `(model,role,layer)` → `(model,corpus,role,layer)`로
  재마이그레이션(D-quant-supabase-1의 model-aware 확장과 같은 패턴). `increment_role_precision()`
  RPC가 **UPDATE-only**(no upsert)라서, seed 안 된 corpus/role 조합에 hit이 들어오면
  에러 없이 조용히 씹힌다는 게 핵심 위험 — seed 먼저, push는 반드시 그 다음.
- `tools/seed_precision_map.py` 신설(하드코딩 SQL 블록을 대체) — `(model,corpus)`
  파라미터화된 재사용 스크립트.
- **5번째 role family 발견+추가(D-quant-supabase-3)**: 사용자가 SLAM MAP 아티팩트를
  보고 "이게 엔진이 실제로 조절 가능한 전부를 보여주는 거냐"고 질문 → qwen_infer.c에
  `MOE_ST_EXPERT_ROLES`(routed experts, expert_gate/up/down_proj, layer 1-26, layer당
  E=64개 전문가 전체를 한 번에 promote)가 실제로 존재하는데 seed script가 빠뜨리고
  있었음을 발견. 269행(191+78)으로 확장, 실 Supabase 반영+검증 완료.
- SLAM MAP 아티팩트(`https://claude.ai/code/artifact/48d0b1f2-a752-4ced-a133-104e7ef93750`)에
  ROUTED EXPERTS 5번째 패밀리 추가 + 축 라벨 버그(짝수 layer만 표시되던 것) 수정.

## Phase 7/8: WikiText-103 재수집 — 실제 버그 2개 + 비용 폭증
- **Bug 1**: `QWEN_MOE_ATTRIB_MAX_POS=8`이 prompt_len=9(pos 0-8)인 corpus에서 생성토큰
  (pos≥9) attribution을 전부 스킵 — `D-d5-14`와 동일 버그 클래스의 재발. 첫 시도 chunk 00:
  REAL FLIP 7건 중 6건 스킵.
- **Bug 2**: `QWEN_MOE_NEARTIE_LOG=1`을 안 켜면 `moe_neartie_events_init()`이 아예 안 불려서
  `QWEN_MOE_NEARTIE_EVENTS_LOG` 경로를 줘도 JSONL 파일 자체가 안 생김(별개의 gate).
- 두 버그 다 고쳐서(`MAX_POS=19`, `NEARTIE_LOG=1`) 재실행 → **비용이 예상보다 훨씬 큼**:
  chunk 00(60req, 13개 REAL FLIP 전부 attribute)에 **6h45m48s** 실측(이벤트당 평균 31분,
  405 combo × pos 최대19 × MAX_EVENTS 무제한). 4청크 전체면 24~30시간 추정.
- 사용자 결정: chunk00(완료)+chunk01(6/7 진행)에서 조기 중단. 703 raw row(204 event+499
  attribution) push+독립검증(SELECT로 count 일치 확인). ROUTED 0 hit — routed-expert
  정밀도가 near-tie에 안 영향을 준다는 기존 결론(ROADMAP D-roadmap-2 Track A/B)의 세 번째
  독립 재확인.

## Step 6: cross-corpus n-monotonicity 실측 (fork로 실행, 60개 실측 테스트)
- 겹치는 hit target 85개 중 attention 1개(`kv_b_proj` L8)+shared-FFN 1개
  (`shared_down_proj` L26) 선정, WikiText-2-fullext(req=32,pos=8)/WikiText-103
  (req=17,pos=9) 각각 n=2..16 실 sweep(`tools/quant_sim_n.py` override +
  `QWEN_MOE_ATTRIB_SIM_ROLE/_LAYER/_PATH`, single-request manifest로 격리).
- **재현성 검증에서 진짜 버그 하나 잡음**: WikiText-2-fullext 원본 런이 4개 manifest
  chunk를 한 프로세스에서 연속 처리해서 req 번호가 **chunk마다 0부터 재시작** — "req=32"가
  파일 전체에서 모호했음(실제로는 chunk_ac의 local req=32 = 전역 `p152.i32`).
- **1차 결과(target 2개)**: builtin-corpus의 role-family 패턴(attention 위반/shared-FFN 안전)이
  재현 안 됨. WikiText-2에서는 두 타겟 다 clean(knee 4/2), WikiText-103에서는 **두 타겟
  다 동일한 pass@3→fail@4→recover@5 모양으로 위반**. role family보다 corpus/flip 자체가
  더 강한 요인일 가능성 제기 — 단 target 2개 표본, 확정 아님(2026-09-03 표본 확장으로 반증).
- **2차 결과(target 2개 추가, 2026-09-03)**: `q_proj`L1(attention)+`dense_down_proj`L0(DENSE FFN,
  처음 테스트) 추가 → **정반대 방향** — WikiText-2에서 두 타겟 다 위반(knee2/fail@4,
  knee3/fail@6), WikiText-103에서 두 타겟 다 clean. 4개 타겟 종합: 2개는 "WT2 clean/WT103
  위반", 2개는 "WT2 위반/WT103 clean" — **corpus-level 패턴이 살아남지 못함**. 정직한 결론:
  위반 여부는 role family도 corpus도 아니라 **개별 target/flip 특이적**으로 보임. 1차의
  "corpus가 강한 요인" 결론 자체가 표본 2개짜리 성급한 판단이었음 — 데이터를 더 모아서
  스스로 반증한 사례(round1/round2 둘 다 데이터 부족으로 인한 오판, 재측정으로 교정).
- WikiText-103 재현성 검증에서 추가 버그 발견: 격리 시 correction eligibility margin이
  threshold(0.1) 바로 근처에 있으면 배치 구성에 따라 넘나들 수 있음(생성 토큰 자체는
  동일해도) — req=10/pos=11, req=40/pos=9 둘 다 재현 실패 후 req=46/pos=9에서 성공.
  single-request 격리가 항상 안전한 재현 방법은 아님, (req,pos)별로 재확인 필요.
- 총 120행(4 target × 2 corpus × 15 n) `moe_quant_sweep_results`에 push+독립 SELECT 검증 완료.

## 교훈
- **UPDATE-only RPC + 새 corpus/role은 항상 seed 먼저** — 안 하면 에러 없이 hit이 증발.
- **격리 실행 전 req/pos provenance 재확인 필수** — 멀티청크 실행은 인덱스가 파일 단위로
  안 이어질 수 있음, "req=N"이라는 라벨만 보고 재현 시도하면 조용히 틀린 요청을 테스트하게 됨.
- **attribution 비용은 pos×combo수에 매우 민감** — MAX_POS를 "정확하게" 고치는 것과
  "비용이 감당 가능한 것"은 별개 문제, 고치기 전에 반드시 예상 비용을 어림잡을 것
  (`QWEN_MOE_ATTRIB_MAX_EVENTS`로 캡 안 걸면 무제한 이벤트가 전부 풀 attribution 돔).

## 남은 것
- WikiText-103 나머지 chunk(01 후반+02+03)는 미실행 — 필요하면 비용 인지 하에 재개.
- graphify --update가 세션 토큰 한도로 9개 chunk 중 5개(4/5/6/8) 미완료.
- target/flip-specific 결론이 맞다면, 다음 단계는 "무엇이 그 flip을 개별적으로 만드는가"
  (margin 크기, 특정 role의 파라미터 분포 등) — 아직 가설조차 없음, 완전히 열린 질문.
