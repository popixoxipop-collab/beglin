---
name: vdsp_sme2_paper_codex_review_final
description: "vdsp SME2 arXiv 논문의 동료검증 이력 — Codex 10라운드 종료 후에도 독립 웹 LLM 1회 리뷰가 §5.1 핵심수치모순(3~5배)을 새로 잡아냄, 실측 재검증으로 수정 완료"
metadata: 
  node_type: memory
  type: project
  originSessionId: e6c100cc-beb0-426d-8425-0959ae41d7af
  modified: 2026-08-19T20:20:36.436Z
---

`~/Desktop/vdsp_SME2_paper/paper.tex` (ARM KleidiAI SME2 통합 논문)는 Codex
동료검증 **Round 1~10을 전부 완료**하고 매 라운드 확인된 결함을 전부 수정·재컴파일한
상태로, **Round 10이 이 논문의 마지막 완결 검증 라운드**다.

**Why:** Round 11 준비 중 두 rater 모두 Codex 계정 단위 사용량 한도(reset 08-20
23:44 추정)로 실패. 사용자에게 "리셋 후 자동 재시도 / 대기 후 직접 요청 / 여기서
종료" 3가지 중 선택을 물었고, **"여기서 리뷰 사이클 종료"**를 선택함(2026-08-20).
Round 10 시점 결과: avg 77.875(A 77.45/B 78.3, 경계 규칙 적용 시 MAJOR_REVISION —
75 임계값 ±3 이내라 보수적으로 낮은 등급 채택). 10라운드에 걸쳐 확인·수정된 결함은
측정 재검증(M4/M1max cross-engine, 배치 클램핑 버그, powermetrics 열스로틀링 배제),
문서 정합성(반복된 "부분수정" 패턴 6회 발견·수정, [[feedback_paper_orchestra_sibling_consistency_sweep]]),
과장 표현 제거(byte-identical 범위, "correct explanation" 등) 등 다수.

**How to apply:** 이 논문에 대해 "다음 라운드", "Round 11 계속" 같은 요청이 오면,
이는 새로 시작하는 것이지 이전 세션에서 자동 진행 중이던 게 아님 — Round 11은
한 번도 실제로 완주된 적 없다(쿼터 에러로 2회 모두 즉시 실패).

**★ 중요 업데이트(2026-08-20, Phase 31)**: "Round 10까지 충분히 검증됨"은 틀렸다.
사용자가 codex-review와 동일 6-axis 프롬프트를 웹 LLM(Codex 아닌 별도 모델)에 직접
붙여넣어 받은 Round 12 리뷰(avg 69.7, MAJOR_REVISION)가, 10라운드 Codex-rescue
리뷰가 전부 놓친 진짜 신규 결함을 잡아냄: §5.1의 배치=16 크로스오버 설명이 인용한
프로파일링 수치("+1.1~1.7%, both KV-cache modes")가 같은 조건의 실측 처리량(+9.0%)과
3~5배 모순 — bob에서 QWEN_PROF=1 fp32-KV를 실제로 재측정해 확인·수정. 덤으로 이
세션 10라운드 + 자체 사전감사도 놓친 stale `+2%`(Abstract·Appendix 각 1건, +1.4%로
안 고쳐진 채 생존) 2건도 추가 발견·수정.
**교훈**: 같은 채점 도구/프롬프트 계열(Codex-rescue)로 아무리 반복해도 그 계열
전체가 공유하는 맹점은 못 잡는다 — 완전히 이질적인 리뷰어(다른 모델·다른 실행
환경)를 섞는 것 자체가 유효한 방어선. 앞으로 "이 논문 충분히 검증됐다"는 이
메모리조차 다시 뒤집힐 수 있는 잠정적 스냅샷으로 취급할 것.

**★★ Phase 32(같은 날 후속)**: Round 12 수정을 바로 다음 웹 리뷰(Round 13, avg
71.05)에 다시 붙였더니 **내 직전 수정 자체가 새 모순을 만든 걸 잡아냄**: §5.1에
fp32 실측 원시값을 새로 삽입했는데, 그 옆에 이미 있던(내가 안 건드린) "+7.7%
under fp32 KV" 문구가 새로 삽입한 원시값으로 재계산하면 안 맞음(+5.3%가 맞음).
같은 라운드에서 "+1.4%/+2.6~9.0%, both 5-repetition means"라는 내 표현도
부정확함 확인(+2.6%는 5회가 아니라 안정화된 3회 평균) — 둘 다 수정.
**추가 교훈**: "이 fix가 건드리는 문자열"만 grep하는 걸로는 부족하다 — 새 숫자를
문단에 *추가*하면, 그 문단에 이미 있던 다른(안 건드린) 숫자와의 산술 정합성도
깨질 수 있다. 이 하위 패턴을 paper-orchestra SKILL.md Sibling Consistency Sweep에
6번 항목("Inverse direction")으로 추가 반영함.
