---
name: vdsp-m42-complex-quant-no-go
description: "vdsp 브랜치 m42-complex-w2의 'M42'(Fairy2i식 복소 widely-linear W2 양자화) 결론=NO-GO, 미머지 방치. ★M-번호 충돌 주의 — main의 M42(qwen_score.c/qwen_spec.c 구조적 일반화, 2026-08-06 완료·머지)와 이름만 같고 완전히 다른 작업."
metadata: 
  node_type: memory
  type: project
  originSessionId: e4ea267a-bca6-4005-9365-ff309438c071
  modified: 2026-08-05T23:38:01.005Z
---

**★ M-번호 충돌 (2026-08-06 발견)**: 이 문서가 다루는 "M42"는 `m42-complex-w2` **브랜치**(main에서 갈라짐, 미머지) 위에서 2026-07-26에 진행된 복소 양자화 실험이다. 그런데 2026-08-06 세션에서 **`main` 브랜치에 별도로 "M42"라는 이름의 완전히 다른 커밋**(`dbda9ac`+`0a2e517`, qwen_score.c/qwen_spec.c 구조적 일반화 — [[vdsp_m42_structural_gen_qwen_score_spec]] 참조)이 생겨 같은 프로젝트 안에 "M42"가 두 개 존재하게 됐다. 이 문서의 M42(복소 양자화·NO-GO)는 **여전히 `m42-complex-w2` 브랜치에만 있고 main에는 없다** — main의 M42는 이것과 무관. 다음에 이어서 작업할 때는 반드시 "어느 M42"인지 브랜치까지 명시해서 헷갈리지 말 것.

repo popixoxipop-collab/vdsp. `~/vdsp_local/HANDOFF_speed_optimization.md`는 M41(구조 일반화, commit `dabd3d6`)까지만 기록하고 그 파일 자체가 "M27-M41 done" 선언 후 멈춰 있음 — 하지만 같은 세션 뒷부분(2026-07-26 09:27~21:47)에 **HANDOFF엔 없는 M42 아크**가 별도 브랜치 `m42-complex-w2`에서 진행되어 최종 결론까지 났다. 2026-08-06 기준 그 이후 커밋 없음(11일 방치), main은 여전히 `6dded59`(M41 followup)에 있고 M42 4커밋(`323292f`~`901f8bf`)은 미머지.

**M42 내용**: Fairy2i(arXiv:2512.02901) 기반 "복소 widely-linear W2 양자화"가 Qwen2.5-1.5B 프로젝션에 통할지 C0-C2 feasibility만 보는 스코프(QAT/NEON 커널은 M43로 명시적 이연). 결정 로그: `M42_DECISIONS.md` (D1-D14).
- **C1**: lossless widely-linear 복소 변환기 — 완전 검증(byte-exact, 29개 hidden state 게이트). D8에서 1개 레이어 rel_l2 스팁 발견 → fablize 3가설 조사로 "attention-softmax 수치 카오스, 코드결함 아님" 확정(다른 프롬프트로 스파이크 위치 이동 확인).
- **C2**: 순수 PhaseQuant(4-codeword 위상양자화, T단 recursive)가 **naive RTN q4에도 2x~70x 열위**(7개 프로젝션 타입 전부), PhaseQuant가 비트 수 적은데도 짐.
- **D12**: GPTQ식 오차보정을 복소로 일반화하다 진짜 버그 발견+수정 — Hermitian H에서 `Hinv[col,:]`(행)이 아니라 `conj(Hinv[col,:])`(켤레-열)이어야 함. 3가설 조사로 확정, 수정 후 GPTQ식이 naive를 이기는 걸로 반전(30-40% 개선).
- **C2b (최종, D14)**: 버그 수정된 GPTQ식 PhaseQuant도 **실제 배포된 GPTQ q4 기준 대비 평균 ~5.15배 열위**(최선 config도 3-4.3배, 최악은 6-8배) — calibration으로 gap의 일부(5-19%)만 회복, Fairy2i 논문 자체도 이 gap을 30B-token QAT로만 닫았던 패턴과 일치.
- **결론(D14, 명시적)**: "M43(QAT+NEON 커널) 현재 Track A 설계로는 NO-GO". 근거: 1.5B 모델은 이 프로젝트 자체 실측(int3 collapse)으로 이미 공격적 양자화에 취약한데, Fairy2i 성공사례는 7B(redundancy 훨씬 많음). 30B-token QAT는 비싸고 이 증거로는 정당화 안 됨.
- **재개 조건(D14 EXIT, 3가지 중 하나)**: (1) C2b를 production 규모 calibration(16384 tokens)으로 재실행해 gap이 실제로 안 줄어드는지 확인, (2) Track B(저랭크 residual, M42에서 스코프만 하고 안 돌림)를 같은 harness로 공정 비교, (3) 수천 토큰짜리 짧은 QAT로 회복 궤적 기울기만 먼저 확인(30B 전체 커밋 전).
- 남는 자산: C1 변환기(재사용 가능, lossless 검증됨) + C2/C2b harness(`m42/*.py`) — 폐기 아님, "M43 직행" 경로만 닫힘.

**Why**: 사용자가 나중에 "vdsp 어디까지 했지" 물으면 HANDOFF 파일만 읽어서는 M42를 놓친다 — 파일 자체가 갱신 안 됐고 브랜치도 안 머지됨. git branch/log 대조 없이 HANDOFF만 믿으면 최근 11일간의 실제 결론(NO-GO)을 놓치게 됨.

**How to apply**: vdsp 관련 질문 나오면 HANDOFF 파일 읽는 것에 더해 반드시 `git log --oneline --all`/`git branch -a`로 main 대비 다른 브랜치 진행 여부 확인할 것 — 이 프로젝트는 실제로 그런 미반영 브랜치가 존재했던 전례. [[vdsp-complex-pitfalls]] (복소 커널 numerics 일반 함정, 이 M42도 그 계열 연장선) 참조.
