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
- **완주(2026-09-04~05)**: 나머지 chunk01 잔여14건+chunk02+chunk03 재개해 18시간 만에 전부
  완료(exit=0, FATAL 없음). 최종 1846행(500 event+1346 attribution) push+독립검증 완료.
  ROUTED/GLOBAL 여전히 0 hit(4번째 독립 재확인, 이번엔 전체 코퍼스 기준). **주의**: 이걸로
  event pool만 커진 것이지 n-monotonicity 데이터가 자동으로 느는 건 아님 — margin↔위반
  상관관계를 진짜 통계적으로 풀려면 이 큰 pool에서 target을 더 뽑아 Step 6와 같은 arbitrary-n
  sweep을 추가로 돌려야 함(비용 있음, 아직 미시도).

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

## margin_before 상관관계 탐색 (2026-09-04, 기존 데이터만 사용, 새 bob 컴퓨팅 없음)
- 8개 target×corpus 조합이 실제로는 3개의 서로 다른 flip event로 압축됨(A: WikiText-2
  req32/pos8 margin=0.000513, B: WikiText-103 req17/pos9 margin=0.097809, C: WikiText-103
  req46/pos9 margin=0.003763).
- **margin 크기와 위반율 사이 깔끔한 관계 없음**: margin이 threshold(0.1)에 제일 가까운(가장
  큰) event B가 100% 위반, 제일 작은(가장 깊은 near-tie) event A는 target별로 갈림(mixed),
  중간 크기인 event C는 100% clean — 단조적이지 않음.
- **event A 내부 관찰**: 같은 event인데도 round1 쌍(kv_b_proj/shared_down_proj)은 clean,
  round2 쌍(q_proj/dense_down_proj)은 위반 — event 단위 속성도 아니고 target×event 개별
  상호작용임을 시사. n=3 event로는 결론 못 냄, 진짜 풀려면 더 많은 서로 다른 real flip
  event가 필요(비용 큰 새 attribution 필요, 미시도).

## Step 6 round 3 (2026-09-05, WikiText-103 전체 완주 후): margin 8개 event로 확장 — 여전히 신호 없음, 그러나 더 날카로운 가설
- WikiText-103 500개 이벤트 pool에서 margin 0.002~0.100(threshold 전체 범위) 걸친 새 event 7개
  선정 → fork 실행. **2/7은 재현성 검증에서 탈락**(margin=0.002는 격리 시 flip 자체가 안 남,
  margin=0.038은 격리 시 완전히 다른 event로 바뀜 — round2에서 이미 확인된 격리-불안정성
  재확인).
- **진짜 방법론적 함정 하나 더 발견+수정**: `shared_down_proj`@L26의 n=2,4에서 SIM override가
  그 role 자체를 훼손할 만큼 낮은 정밀도라 "corrected" 기준값 자체가 오염됨(839/298, 진짜
  기준은 1222 — 재현성검증+n=3/5-16 전부 1222로 일치해서 확인). fork가 스스로 잡아서 두
  행 수동 재분류 후 push. **앞으로 이 방식으로 sweep할 때 항상 corrected가 n 전체에서
  불변인지 확인 필요** — role/layer hit line만 grep하면 이 오염을 놓친다.
- **5개 성공 결과**(margin 오름차순): kv_a_proj_with_mqa@L0(0.004)=clean knee5,
  q_proj@L3(0.021)=위반 knee4/fail@6, shared_down_proj@L26(0.044)=위반 knee3/fail@2,4,
  kv_a_proj_with_mqa@L1(0.080)=위반 knee4/fail@5, q_proj@L0(0.100)=위반 knee2/fail@4.
- **8개 event 종합 — 여전히 margin과 위반 사이 단조 관계 없음**(낮은/중간/높은 margin 전부
  위반과 clean 둘 다 나옴). 8개면 여전히 작은 표본이지만 margin 전체 구간을 다 커버했는데도
  패턴이 안 나온다는 게 정직한 결론.
- **더 날카로운 새 가설(독립검증됨)**: `shared_down_proj`@L26이 완전히 다른 두 real event
  (round1: corrected=1, round3: corrected=1222 — 서로 다른 corpus position, 다른 정답 토큰,
  중복 push 아님, SELECT로 확인)에서 **완전히 동일한 위반 모양**(n=2,4에서만 fail)과
  **byte-identical한 rel_l2**를 보임. rel_l2가 텐서 자체의 함수라 같은 건 당연하지만, pass/fail
  패턴까지 일치하는 건 우연이 아닐 수 있음 — **"target/flip-specific"보다 "target-tensor 고유
  quantization curve"가 더 정확한 설명일 가능성**. 텐서 하나만 2회 확인된 거라 확정은 아니고,
  다음에 이 질문을 다시 판다면 여러 텐서를 각각 2개 이상 event로 테스트해서 이 가설을
  검증하는 게 자연스러운 다음 단계.
- 스키마 주의사항: 격리 후 req는 항상 0으로 재시작되므로, 원본 pos가 같은 두 개의 다른
  real event는 `(req,pos)`만으로 `moe_quant_sweep_results`에서 구별 불가(위 shared_down_proj
  사례처럼 30행이 15+15로 겹쳐 보임 — unique 제약이 없어서 버그는 아니지만 향후 조회 시 주의).
- 75행 push+독립검증. Step6 전체 누적 195행(12개 target×event 조합), 전체 테이블(builtin
  corpus 90행 포함) 285행 — 전부 SELECT로 확인.

## Step 6 round 4 (2026-09-06): "target-tensor 고유 curve" 가설 — 2번째 텐서로 확인됨
- `kv_b_proj`@L8(round1 WT103 req17/pos9=위반 knee3 fail@4)에 대해 WT103 내 다른 real event
  (req35/pos12, corrected=438) 하나 더 찾아서 sweep. **rel_l2/pass 패턴이 pos=9와 완전
  byte-identical**(독립 SELECT 확인, 30행: 15+15).
- **2/2 텐서가 이제 가설을 지지함**(shared_down_proj@L26에 이어). rel_l2 동일은 당연하지만
  pass/fail 패턴까지 일치하는 건 우연이 아닐 가능성 — 아직 반증 시도는 안 해봄(다른 텐서 
  선택 편향 가능성 존재), 확정은 아니지만 두 번 연속 생존.
- moe_quant_sweep_results 총 300행, SELECT로 확인.
- graphify --update는 실제로 9개 chunk 전부 이미 완료돼있었음(제가 세션 중간에 몰랐던
  부분 — dedup guard가 재추출 시도를 막아서 발견) → 나머지 파이프라인(merge/cluster/
  label/HTML) 직접 완주. `graphify-out/graph.html`+`GRAPH_REPORT.md` 생성됨(1249 노드,
  3239 엣지, 90 커뮤니티, 토큰절감 60.2배).

## Step 6 round 5 (2026-09-06, 같은 세션 후속): "target-tensor curve" 가설 반증 + 위반율 83%로 상승
- WikiText-103 재개(chunk01_tail+02+03, 53 real flip event 확보) 완료 데이터에서 12개 신규
  (event,target) 쌍을 골라 n=2..16 스윕(180회 엔진실행, 전부 exit=0). **10/12(83%) 위반** —
  기존 표본(2/3, Step6 절반 정도)보다 훨씬 높음. role family/hit count/corpus 위치 전부 예측력
  없음, 기존 결론("target×event 고유 상호작용") 유지·강화.
- **★★"target-tensor curve"(round3/4, 2/2 텐서에서 지지됐던 가설) 반증**: `shared_down_proj`
  L26을 세 번째로 3개 이벤트(t06/t07/t08)에서 테스트했는데, override 파일이 (role,layer)로만
  결정돼 **세 이벤트가 물리적으로 완전히 동일한 override**를 쓰는데도 pass/fail 모양이 3가지로
  갈림(t06=t07: n=4만 실패, t08: n=3만 실패 — round1/round3의 n=2,4 실패와도 다름). override가
  구조적으로 동일 보장되므로 이건 통계적 우연이 아니라 **텐서만의 속성이 아니라는 확정적 증거** —
  진짜 결정 요인은 텐서 양자화 오차와 각 이벤트 고유의 로컬 margin/context 간 상호작용. (round3/4가
  본 t06=t07류 "완전일치"도 여전히 실재 — 텐서 요인이 무관하다는 게 아니라 그것만으론 부족하다는 것.)
- 초기 검증 중 자체 실수 발견+정정: `corrected=` 값을 grep -tail1로 뽑았다가 같은 로그 안의
  무관한 다른 position near-tie를 잘못 집어 "값이 오염된 것처럼" 보였음 — (req,pos) 정확히
  필터링해 재검증, 진짜 결과는 클린(round3가 겪은 override-corrupts-ground-truth류 문제 아님).
- 상세: RESULTS.md "Step 6 round 5" 섹션. 원자료: bob `/tmp/mono_sweep/sweep_results.tsv`(180행)
  + 개별 로그 180개. **push 완료**(2026-09-06, 커밋 `babbc1c`) — moe_quant_sweep_results 총
  480행(builtin 90 + wikitext-2 60 + wikitext-103 330), 독립 REST count로 확인됨. 이 메모리
  파일의 예전 "로컬 전용/미push" 기록은 그 이후 stale이 됨(2026-09-07 재확인 시 발견 —
  compaction으로 컨텍스트 유실된 뒤 같은 세션이 자기 자신의 이전 작업을 몰랐던 사례).
- graphify 파이프라인 완주됨(HTML+리포트 생성 완료, 1249 노드/3239 엣지/90 커뮤니티).

## Step 6 종결 + quant_search_n.py 프로덕션 배포 (2026-09-07, 같은 세션 재개 후)
- round5가 이미 "target-tensor curve" 가설을 same-override/different-outcome으로 확정적으로
  반증했음을 compaction 이후 재발견(위 항목). 병행으로 진행한 cross-corpus 각도(5개 target,
  WT2 vs WT103 100% 불일치)는 round5와 같은 결론의 보완적 재확인 — 새 bob 컴퓨팅 불필요,
  기존 DB 데이터만으로 확인. RESULTS.md에 짧은 addendum으로 정리(중복 재도출 부분은 축약).
- `tools/quant_search_n.py`에 **live 오라클 백엔드 구현+실제 bob 라운드트립 검증 완료**:
  `fetch_prior_points_by_corpus()`(Supabase 실측 조회) + `make_live_oracle()`(SSH로 실제
  엔진 1회 실행, stdout의 `hit req=.. pos=.. role=.. layer=..` 라인 파싱). 검증 과정에서
  실버그 2개 발견+수정: (1) 원격 커맨드에 `cd /Users/bob/vdsp_m4_bench` 누락 — 없어도 에러
  안 나고 조용히 전부 fail로 보임(weights_moe/arch_config_moe.txt 상대경로 로드 실패가
  원인). (2) WikiText-103 `/tmp/d4_wikitext103_short_manifest/`가 또 비어있어서(★bob /tmp
  주기적 정리, D-gpu-4/5 섹션과 같은 날 별도 재현) WikiText-2 타겟(q_proj@L1, req=0/pos=8)으로
  전환 — 이번엔 `manifest_wt2_req32.txt`(chunk_aa, 틀림) vs `manifest_wt2_req32_ac.txt`
  (chunk_ac, 맞음)를 착각해 D-d5-27이 이미 문서화한 "req32 두 번 틀린 추측" 함정을 독립적으로
  재현. 최종적으로 기존 known-good 값(fail@n=4만, 나머지 pass)과 정확히 일치하는 실측 재현
  성공 — 라이브 오라클 배선 검증 완료.
- **실전 배포 함의**: 현재 Supabase에 실측 데이터가 있는 모든 (model,role,layer)가
  `classify()`에서 `exhaustive_required`로 분류됨(2개 코퍼스 모두 clean인 타겟이 아직 하나도
  없음 — round5의 83% 위반율과 정합) → 지금은 항상 전수스캔만 선택. bisection 경로는 배선은
  됐지만 아직 한 번도 실제 트리거된 적 없음 — 버그 아니라 현재 데이터의 정직한 반영.
- 이 push는 안 함(기존 event 재검증용이라 중복행 방지 목적으로 의도적으로 skip).
- RESULTS.md "Step 6 closure addendum" + "ROI-G Phase 2: quant_search_n.py live oracle" 섹션.

## (a) live 모드 프로덕션 배포 -- 24개 미검증 조합, 단일 event, 58% 위반 (2026-09-07)
- WikiText-103 `/tmp` 프롬프트 파일이 **또** 사라짐(이번 세션 3번째 재현 — D-gpu-4/5, live
  오라클 검증, 이번 건). round5의 `/tmp/mono_sweep/`(하루 전 생성)도 이미 사라짐. bob `/tmp`는
  며칠 못 버팀이 이제 확정적 패턴 — WikiText-103을 또 쓰려면 `/Users/bob/...`류 영구경로로
  재배치할 것(할 일로 남김, 이번엔 안 함). WikiText-2는 `/Users/bob/d4_wikitext2_short_manifest/`
  라 안전 — 이번 라운드는 전부 WikiText-2로 진행.
- WikiText-2 200개 프롬프트 중 196개가 한 번도 안 건드려짐. 새 후보 탐색(HI_COMBOS 제한 없는
  discovery pass)으로 p60/pos=14에서 real flip 발견 → **24개 서로 다른 role/layer가 전부
  개별적으로 correction 재현**(전 role family 포함). discovery 15개 큐잉했다가 p60 하나만으로
  round5 전체(12개)보다 많은 타겟을 확보해 나머지 취소 — 비용 통제(discovery 1건 ~20분).
- 24개 타겟 × n=2..16 스윕(360회, 전부 exit=0) → **14/24(58%) 위반**, 어떤 role family도
  면제 안 됨(kv_a_proj_with_mqa 6/8, kv_b_proj 4/6, o_proj 1/4, q_proj 1/1, shared_up 1/2,
  shared_down 1/2). round5(같은 텐서·다른 event)의 정반대 축(같은 event·다른 텐서)에서 같은
  결론 재확인 — "단일 요인 예측력 없음"이 양방향에서 성립.
- 360행 push+독립검증(총 840행, 480+360 정확히 일치). rel_l2 파싱 버그(단일자리 n의 공백
  정렬로 필드 밀림) 발견+수정 후 push.
- RESULTS.md "ROI-G Phase 2: live-mode production deployment" 섹션.

## (b)+(c) 병행 완료 (2026-09-07)
- **(b) bisection 최초 트리거**: 840행 전체에 classify() 돌려서 2개 코퍼스 모두 clean인
  타겟 발견 — `kv_a_proj_with_mqa`/L11(n=2~16 전부 pass, WT103 round5 t02 + WT2 이번 배치
  둘 다). `quant_search_n.py --live` 실제 재실행 → `mode=bisection` 처음 선택됨, **15번
  대신 2번**만에 knee=2 확정(두 코퍼스 exhaustive 결과와 정확히 일치). push 안 함(기존
  이벤트 재검증). → classify() 설계가 실전에서 진짜로 작동함을 증명.
- **(c) WikiText-103 영구경로 재배치**: macstudio에 원본 Phase7 프롬프트 40개(p200~239,
  9/2)가 `~/d4_wikitext103_short_manifest/`에 생존해있던 걸 이번에 처음 발견(이전 탐색이
  놓쳤던 것 — 그래도 40개론 부족해서 어차피 재생성 필요했음). 새로 200개 토큰화(real
  DeepSeek tokenizer, `datasets` validation split, non-streaming) → macstudio→bob 직접
  전송(`bob-lan`, 로컬 경유 안 함) → `/Users/bob/d4_wikitext103_short_manifest/`(영구경로,
  `/tmp` 아님). **버그 발견+수정**: manifest.txt가 macstudio 자기 경로(`/Users/eoe/...`)를
  그대로 담고 있어서 bob에서 그대로 쓰면 전부 FATAL 났을 것 — sed로 `/Users/bob/...`로
  치환 후 실제 엔진 1회 실행으로 검증(정상 로드+10토큰 생성, FATAL 없음).
- 이번 라운드에서 새 WT103 200개 프롬프트로 flip 헌팅은 안 함(범위 밖, 의도적 보류).
- RESULTS.md "ROI-G Phase 2: bisection triggers..." 섹션.

## (e) WT103 flip 헌팅 — 89타겟 스윕했으나 방법론 오염으로 전량 retract (2026-09-08)
- p10 discovery에서 91 hit(거의 전 role family) 발견 → 재현성 체크 생략(이번이 처음, 매번
  하던 걸 건너뜀)하고 바로 89타겟×15n(1335회) 스윕 진행(사용자 확인 후 전체 규모로).
  도중 디스크풀 크래시(75/89 완료, override 파일 누적 ~30GB) → 정리+재개로 89개 완주.
- **그런데 결과가 이상함**: 77/89가 어떤 n에서도 안 pass — discovery가 "이 텐서 승격만으로
  correction 재현됨"이라 확인한 텐서들인데도. 원인 추적: **p10 프롬프트에 real flip이
  2개**(pos=8 razor-thin margin=0.003420, pos=11 89-hit) 있었음. SIM override는 전체
  forward pass에 적용되므로, pos=11용 텐서를 바꿔도 pos=8(더 앞선 위치)의 correction 결과가
  같이 흔들리고(margin은 동일 텍스트인데 n=5는 corrected=3000, n=10/16은 corrected=5226,
  n=2는 flip 자체가 안 일어남), 그 결과 pos=8이 실제 생성한 토큰이 달라지면서 pos=11이
  보는 컨텍스트 자체가 n마다 달라짐 — "같은 위치"가 더 이상 같은 결정을 재는 게 아니게 됨.
  기존 round3의 "override가 ground truth를 오염시킴" 트랩과 같은 계열이지만 원인이 다름
  (이번은 override 심각도가 아니라 margin이 threshold에 극도로 가까운 것, 0.0034).
- **검증**: 기존에 신뢰하던 p60 24타겟 배치(단일 flip 확인됨)는 동일 점검에서 corrected
  값이 n 전체에서 완전히 안정적 — round4/이전 결론("14/24 위반")은 그대로 유효.
- **조치**: 1335행 전부 Supabase에서 DELETE(정확한 tested_at 타임스탬프로 타겟팅, 총계
  2175→840 정확히 복귀 확인). 실제 bob 비용(~6.3시간+디스크풀 처리)은 못 건짐 — 그대로
  기록(이 프로젝트 관행: 부정적 결과도 숨기지 않고 기록).
- **교훈**: (a) discovery가 "깨끗해 보여도" 재현성 체크를 절대 생략하지 말 것, (b) 스윕
  규모 정하기 전에 `grep -c "REAL FLIP"`로 flip 개수부터 확인 — 2개 이상이면 표준 단일
  타겟 스윕 방법론이 통째로 안 맞음(다른 설계 필요, 이번엔 시도 안 함).
- RESULTS.md "ROI-G Phase 2: a real methodological trap..." 섹션.

## (f) 단일-flip 규율 적용해서 flip 헌팅 재시도 — 성공 (2026-09-08)
- p10 retraction의 교훈 2가지를 실제로 적용: (1) 스윕 규모 정하기 전 `grep -c "REAL FLIP"`로
  flip 개수 확인, 1개만 통과, (2) 스윕 전 재현성 체크 필수 선행(생략 안 함).
- 19개 신규 WT103 프롬프트 discovery: 8개는 flip 0, 1개(p95)는 46분+ 진행 없이 멈춤(이상
  케이스로 kill, 원인 미조사·향후 과제로 남김), p105에서 단일 flip(pos=13,
  margin_before=0.099848 — 이번 세션 전체에서 가장 threshold에 가까운 razor-thin 마진)
  2개 hit, p155에서 단일 flip(pos=12) 13개 hit.
- **재현성 체크 둘 다 통과**(p105: margin/corrected 정확히 재현. p155: margin_before=0.048599,
  corrected=438 — round1/round4의 kv_b_proj@L8과 완전히 동일한 값! 새로 재생성한 코퍼스가
  같은 실제 이벤트를 다른 인덱스로 우연히 재발견한 것으로 보임, 실제로 kv_b_proj/L8 재스윕
  결과가 기존 {2,4} fail 패턴과 정확히 일치 — 의도적으로 중복 push함, 테이블이 허용하고
  기존에도 같은 종류 중복이 있었음).
- **결과**: p105 2/2 clean(전부 monotonic), p155 8/13(62%) 위반. 합계 8/15(53%) —
  기존 라운드들의 범위(13~83%)와 정합, "단일 요인 예측력 없음" 결론 재확인.
- 30행+195행 push+독립검증(840→870→1065, 전부 정확히 일치). 이번엔 방법론 자체가 설계대로
  작동함(flip개수체크→재현성체크→스윕→디스크정리 전부 지켜짐) — 직전 retract 라운드와 대조.
- RESULTS.md "ROI-G Phase 2: redone flip hunt..." 섹션.

## p95 이상현상 조사 (2026-09-08) — 원인 미확정이나 심각한 버그는 배제
- p95 재현 후 `/usr/bin/sample`(macOS 프로파일러, PATH상 `sample`은 무관한 Python 패키지라
  풀경로 필요)로 콜스택 2회 샘플링(5분+ 간격).
- **메모리 누수 가설 기각**: RSS가 오히려 감소(4,757,872→4,613,456 KB). 누수라면 단조증가해야 함.
- **완전한 데드락 가설 약화(완전 배제는 아님)**: CPU 450~550%로 실작업 중, 콜스택은 정상
  forward-pass 경로(`moe_neartie_maybe_correct→moe_forward_token→moe_attention/
  moe_matvec_af_row`), 대기 부분도 정상 스레드풀 동기화(`moe_scalar_pool_go_and_wait`)이지
  진짜 락 데드락 시그니처 아님. 다만 두 샘플이 완전 동일한 모양이라 "정상적으로 여러 콤보를
  순회 중"과 "같은 콤보에서 영원히 멈춤"을 콜스택만으로 구분 불가 — 콤보별 진행 로그가 없음
  (hit만 로깅, no-hit은 무음).
- **가장 유력(미확정)**: 이 특정 near-tie 이벤트의 forward-pass replay 자체가 다른
  이벤트(p60/p10/p105/p155)보다 콤보당 본질적으로 더 비쌈(컨텍스트별 MoE 전문가 라우팅
  차이 등) — 근거와 정합하지만 콤보별 진행 계측(재빌드 필요) 없이는 확증 불가.
  **다음 세션 열린 과제로 유지**: 콤보별 progress 로그 추가하거나 dtrace로 확인.
- RESULTS.md "p95's '46+ minute hang' investigated..." 섹션.

## p95 최종 해결 (2026-09-09) — 멈춘 적 없었음, 그리고 다른 세션에 폐 끼침
- `qwen_infer.c`에 `QWEN_MOE_ATTRIB_PROGRESS=1` 계측 추가(콤보별 fprintf, 기본 꺼짐,
  D-p95-1) → 빌드 위해 bob의 stale `gguf_transcode.c/.h`(8/26, qNg64 함수 없음)를
  로컬 최신본(9/8)으로 동기화해야 했음(git HEAD 어떤 빌드 시도든 이 문제 있었을 것,
  나만의 문제 아님). p105로 회귀검증(정확히 일치) 후 p95 재실행.
- **최종 결과**: tested=85/~216이 11분에 정상 진행 확인(데드락 완전 기각) → 20분(wall_ms=
  1200099)에 정상 종료, **0/191 hit** — 이 근접결정은 단일텐서로 설명되는 조합이
  전혀 없는 것뿐이었음(정상 실행, hit 로그가 없어서 조용했을 뿐). "46분+"이었던 최초
  관측은 아래 항목의 리소스 경쟁 때문에 부풀려졌을 가능성 높음(재현 불가, 판단만).
- **★★다른 세션에 실제 지장**: RESULTS.md의 D-qNg64-16/17을 나중에 읽고 발견 — 내
  `qwen_quantsim4_bin`이 같은 저장소의 다른 세션(qNg64 실커널 벤치마크 파이프라인)을
  **2번** 막았음(그쪽이 매너 있게 내 프로세스는 안 건드리고 자기 것만 죽이고 후퇴).
  지금 확인해서 내 프로세스 전부 정리됨(load 2.14, swap 819M/2048M로 정상 복귀) —
  RESULTS.md에 사과 겸 기록 남김.
- RESULTS.md "ROI-G Phase 2: p95's hang, definitively resolved..." 섹션. `qwen_infer.c`
  변경사항 커밋 완료(instrumentation, 기존 동작 무변화).

## GPU에서 실제 DeepSeek monotonicity 검증 (2026-09-09) — 메커니즘 확인 + 새 축 발견
- 다른 세션(qNg64 GPU Phase 1, session_01WiJ5s4mjwhFyhTMNmyvgL8)이 GQA 게이트 2개에만
  `moe_promotion_nq_init_gpu()`를 연결해뒀는데, DeepSeek는 MLA라 아예 테스트 불가능한 상태였음
  발견 → `run_moe_gpu_generate_gate()`(V5k, 이미 이 세션 D-gpu-4/5에서 실제 DeepSeek 생성
  검증된 게이트)에 같은 호출 한 줄 추가. `D-p95-1`과 같은 방식으로 isolated patch 커밋(다른
  세션 working tree 안 건드림).
- p155(이미 CPU에서 완전 특성화된 real event) 재사용, GPU에서 실제 네이티브 qNg64 승격
  테스트: **kv_a_proj_with_mqa/L0(CPU clean, knee=4)는 GPU n=2/5에서 CPU와 완전 일치**(fail/
  pass). **kv_a_proj_with_mqa/L3(CPU 위반: pass,fail,pass,pass @ n=2,3,5,6)는 GPU에서
  fail,fail,pass,fail — n=2와 n=6에서 CPU와 불일치**.
- **새로운 발견**: clean 타겟은 CPU 시뮬레이션(RTN)과 GPU 네이티브(bit-plane qNg64)가
  일치하지만, 위반(borderline) 타겟은 **양자화 알고리즘 자체가 다르면 다른 n에서 위반**
  — 텐서/이벤트/코퍼스에 이은 4번째 "단일 요인 예측 불가" 축. 표본 작음(1 clean + 1 violated,
  총 6회 실측)이라 이 크기에서의 실측 신호로만 기록.
- BOB_LOAD_OK=1 바이패스는 사용자가 직접 승인(bob 데스크톱 사용 중 메모리 압박 상황 설명 후).
- RESULTS.md "D-qNg64-gpu-2" 섹션.

## 남은 것 (2026-09-09 기준, GPU 검증 포함 전부 완료 후)
- ROI-G Phase 2 전체 계획 완전 종결. p95 미스터리 완전 해결. GPU에서도 첫 실측 완료(clean
  일치, violated 불일치 — 4번째 예측불가 축 확인). Supabase 1065행(CPU, 신뢰 가능) + GPU
  실측 6건(RESULTS.md에만 기록, Supabase 미push).
- 남은 것: "Phase 3"는 사용자가 언급했으나 이 메모리 파일/계획 문서 어디에도 정의 안 돼있음
  — 사용자에게 범위 확인 필요. GPU monotonicity 표본 확장(현재 2타겟뿐)도 원하면 가능.
