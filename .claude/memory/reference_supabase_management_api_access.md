---
name: reference_supabase_management_api_access
description: how to run real read queries against this project's Supabase data (moe_neartie_events/moe_role_precision_state/moe_quant_sweep_results) when QWEN_SUPABASE_URL/KEY aren't set
metadata:
  type: reference
---

이 repo의 Supabase 데이터(project_ref `btdjbfgqzglucifcnuoc`, "beglin")에 라이브로 접근하는 두 방법:

1. **REST API**(`d4_supabase_push.py`/`tools/quant_search_n.py --live` 등 기존 스크립트가 씀):
   `QWEN_SUPABASE_URL`/`QWEN_SUPABASE_KEY` env var 필요 — 이번에 확인한 결과 xox 로컬 `.env`, bob
   `.env`/쉘프로파일, repo 로컬 `.env` 어디에도 없음(2026-09-08 3회 확인).
2. **Management API**(RLS 완전 우회, 어떤 SQL이든 직접 실행 가능 — DDL/DML은 사용자 확인 없이 절대
   실행 금지, SELECT만): `~/.claude/hooks/scripts/nvidia-keypool-guard.py`의 주석(2026-07-16, D1)에
   이미 상세히 기록돼 있었음 — `~/Desktop/Code_reviewer_with_feedback/.env`의
   `SUPABASE_MANAGEMENT_PAT`를 `Authorization: Bearer $PAT`로
   `https://api.supabase.com/v1/projects/{ref}/database/query`(POST, body
   `{"query":"..."}`)에 사용. **2026-09-08에 vdsp-engine의 project_ref
   (`btdjbfgqzglucifcnuoc`)로 직접 테스트해 작동 확인**(`moe_quant_sweep_results` count=1065).
   그 훅 자체는 다른 프로젝트(`oziaeqcvrkrqkhwrybfj`, Code_reviewer_with_feedback 자체 프로젝트)를
   예시로 들지만, 이 PAT는 계정 레벨이라 project_ref만 바꾸면 이 repo에도 그대로 씀.

**교훈**: 이 세션에서 QWEN_SUPABASE_URL/KEY를 못 찾아 세 번(D-qNg64-plan-1/2/3) 라이브 재측정을
막힌 채로 넘겼는데, 실제로는 처음부터 hook 주석에 답이 있었음 — 사용자가 "메모리랑 hook에 기록했었어"
라고 알려줘서 발견. **앞으로 이 repo에서 Supabase 라이브 쿼리가 필요하면 REST API 키를 더 찾지 말고
바로 이 Management API 경로를 쓸 것.**

안전 원칙(훅 주석 원문 그대로): PAT 값 자체를 절대 echo/print/log에 남기지 않기(bash 변수에 담아
subshell 안에서만 사용, `unset` 하기, 또는 Python이면 값을 변수에 담되 출력하지 않기). curl 사용 시
Python urllib보다 SSL 인증서 문제가 적어 더 안정적(이 machine의 Python.org 배포판이 로컬 CA 번들을
못 찾는 이슈가 실측됨, curl은 시스템 인증서 스토어를 써서 문제 없음).
