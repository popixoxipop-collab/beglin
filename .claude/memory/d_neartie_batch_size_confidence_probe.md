---
name: d_neartie_batch_size_confidence_probe
description: MA-LIO(SNU RPM Lab)식 point-wise uncertainty 아이디어를 근접타이(near-tie) 토큰 신뢰도에 적용 시도 — D-neartie-batch-1(관찰적, confound)/2(paired replay, 정제) 두 라운드, 최종 n.s.
metadata:
  type: project
---

**배경**: 대학원 준비 중 SNU 김아영 RPM Lab 논문(MA-LIO, RA-L 2023 — 다중 LiDAR간 point-wise uncertainty propagation)을 읽다가, "배치 병렬처리(B=n) 시 계산경로에 따라 토큰별 출력 신뢰도가 달라질 수 있는가"라는 질문이 나와 이 엔진(beglin/vdsp-engine)의 실제 `moe_neartie_events` Supabase 인프라(578건 기존 데이터)로 검증 시도.

## D-neartie-batch-1 (관찰적, confound 있음 — 최종 결론 아님)

- `qwen_infer.c:moe_neartie_maybe_log()`에 `batch_size` 파라미터 추가(cbatch_step의 실제 활성 슬롯 카운트 `A`를 그대로 로깅) — 호출 2곳(`:7713`,`:7733`) 모두 수정, JSONL/Supabase 스키마(`moe_neartie_events.batch_size`)에 반영
- 별개로 발견한 실제 버그: `d4_supabase_push.py`가 attribution row에 margin을 안 담아서 `moe_role_precision_state.min_margin_observed`가 269/269행 전부 null이었음(폐루프 승급 트리거가 처음부터 죽어있었음) — event row 스트리밍하며 (model,corpus,req,pos) 키로 margin 매칭해 수정. 알려진 한계: req/pos가 manifest마다 재시작되므로 한 파일에 여러 manifest 섞이면 오조인 가능(문서화만, 미해결)
- 로컬(xox, M1 Max) 실측: DeepSeek-V2-Lite, 24-request manifest, threshold=0.5, 진짜 근접타이 38건(mean margin 0.295)
- **관찰적 상관**: Pearson r=-0.132, t=-0.80(df=36), **n.s.** — batch_size 2~18 범위, 구간별 평균도 비단조
- ★이 결과 자체가 방법론적으로 결함 있음을 즉시 인지: 서로 다른 (req,pos) 토큰들이 각자 우연한 시점의 batch_size로 관측된 것뿐이라 "토큰 고유의 난이도"와 "배치효과"가 confound됨 — GICP paired 설계(같은 세션 앞부분)와 정반대의 설계

## D-neartie-batch-2 (paired replay, 최종 결론)

- 기존에 이미 있던 `moe_attrib_replay_one()`(role/layer ablation replay 메커니즘)을 재활용 — `layer=-1`(baseline probe, 텐서 스왑 없음)이면 **같은 (req,pos) 토큰을 단일스트림(A=1 등가)으로 처음부터 재실행**하는 정확히 필요했던 도구였음
- `MoeAttribRole` enum이 C에서 forward-declare 불가능해 얇은 wrapper(`moe_attrib_replay_baseline`) 추가로 우회, `moe_neartie_maybe_log()`에 4개 파라미터(af/t_embed/t_lmhead/w_finalnorm — 호출부에 이미 스코프 안에 있던 `af_blob` 등) 추가 threading, 근접타이 로그 시점에 즉시 replay 호출해 `replay_margin_b1` 필드로 원본 margin과 나란히 로깅
- 같은 24-request 런 재실행(리플레이 오버헤드로 wall_ms 1,018,791→1,479,456, ~45%↑): 동일 38건
- **Paired t-test**: mean_diff=0.00000021, max|diff|=2e-6, 20/38건 완전 0 일치, t=0.984(df=37) → **n.s.**
- **최종 결론**: 배치 구성이 부동소수점 계산 순서를 실제로 바꾸긴 함(비associativity 실증, 18/38건이 정확히 0이 아님) — 하지만 그 크기가 float32 반올림 잡음 수준(~2e-6)이라 이 엔진·이 정밀도(int4/int8 양자화 MoE)에서 실전 토큰 선택에 영향을 줄 정도는 아님. 관찰적 결과(D1)와 정제된 결과(D2) 둘 다 "없다"로 일치하지만, D2가 confound 제거된 진짜 검증.

## 라이브 데이터

- Supabase(`project_ref btdjbfgqzglucifcnuoc`) `moe_neartie_events`에 `batch_size`(integer)+`replay_margin_b1`(double precision) 컬럼 신규 추가(DDL, Management API PAT로 실행 — PAT 위치는 [[reference_supabase_management_api_access]] 참조)
- 실측 38건 전부 `corpus=replay-paired-probe-24req`로 push 완료, count 검증(0-37/38) 완료

## 교훈

- **로컬 파일시스템 검색은 좁게 하면 놓친다**: `ls ~ | grep`으로 가중치를 못 찾고 "여기 라이브 아니다"로 잘못 판단할 뻔함 — 사용자가 직접 bob 3개 사본 vs 로컬 리포를 diff/git-log로 대조해 "로컬이 라이브"임을 실측으로 뒤집어줌. 검색범위 좁음≠부재.
- **GitHub 코드검색 API가 실제로 존재하는 파일(`d4_supabase_push.py`, `supabase_schema_d_roadmap4.sql`)을 0건으로 반환**하는 인덱싱 지연을 이 세션에서 직접 겪음 — 이후 git tree API로 전환.
- **관찰적 상관 vs paired 설계의 차이가 실제 발견에서도 재현됨**(GICP 작업과 동일 패턴): 처음 낸 결론(D1)이 방법론적 결함 때문에 신뢰 못할 결론이었고, 같은 세션 안에서 스스로 재설계해 정제된 결론(D2)에 도달.
