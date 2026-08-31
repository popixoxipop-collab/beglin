---
name: vdsp-sme2-lazy-repack-serve-sigill
description: "beglin/vdsp 엔진: QWEN_SME2_LAZY_REPACK=1 + serve(배치) 모드 조합이 SIGILL(exit 132) — GGUF 작업과 무관한 기존 코드에서 재현된 미해결 버그"
metadata:
  type: project
  originSessionId: e6c100cc-beb0-426d-8425-0959ae41d7af
  modified: 2026-08-25T08:58:26.855Z
---

`beglin` 엔진(`/Users/xox/vdsp-engine`)에서 **`QWEN_SME2_LAZY_REPACK=1`을
`serve`(M20 배치 디코드) 모드와 같이 켜면 SIGILL(exit 132)** — 2026-08-25
GGUF 범용로더 Phase 2 작업 중 발견. **완전히 기존/미수정 코드**(legacy
`QWEN_INT4_BIN` 경로, `weights_prod_llama31_8b`)에서도 동일하게 재현됨 —
GGUF/transcode 작업이 원인이 아니라 이전부터 있던, 한 번도 테스트 안 된
조합의 버그.

**재현**:
```
QWEN_INT4_BIN=<...> QWEN_SME2_LAZY_REPACK=1 QWEN_BATCH=16 \
  ./qwen_infer serve 8
```
→ `[engine] SME2 lazy: repacked model.layers.0.self_attn.q_proj.weight
on first use (...)` 로그 한 줄 찍고 바로 SIGILL.

**왜 지금까지 안 걸렸는지**: `QWEN_SME2_LAZY_REPACK`은 기본 OFF(unset)라
지금까지 실배포/실측은 전부 eager repack(load 시점에 전부 repack 완료 후
`serve` 진입)만 썼음 — lazy 모드는 이 프로젝트 자체 문서에도 "memory-
tradeoff prototype"이라고 적혀 있어 원래 실험적 취급. `greedy`/`bench`
모드는 M=1이라 `kai_route()`의 `M >= kai_sme2_min_m()` 조건에 걸려
`kai_repack_one_lazy()`가 아예 호출 안 됨 — 그래서 lazy 모드 자체는
그동안 문제없이 보여왔음(진짜론 그 경로가 한 번도 SME2를 태운 적이
없었을 뿐).

**가설(미검증)**: `serve`는 `q4pool`의 여러 워커 스레드가 동시에 도는
컨텍스트인데, `kai_repack_one_lazy()`는 그 스레드 동시성을 전혀 고려한
동기화가 없어 보임(단일스레드 load-time repack만 상정하고 짠 코드) —
경쟁상태로 SME2 명령어 실행 컨텍스트가 깨지는 것으로 추정되나 실증은
안 함.

**현재 대응**: GGUF 로더 경로는 lazy를 기본 on으로 켰지만(`g_sme2_lazy=1`,
`QWEN_SME2_LAZY_REPACK` 미설정시), GGUF 경로는 `greedy`/`bench`만
지원하고 `serve`는 `g_int4` 게이트로 막혀 있어(`FATAL: serve requires
int4 mode`) 이 버그를 실제로 밟지 않음 — GGUF를 `serve`에 열어주는
확장은 시도했다가 이 버그를 만나 **되돌림**(별도 커밋 없이 세션 내
revert, `qwen_infer.c` 최종본엔 흔적 없음).

**해야 할 것(미착수)**: 근본원인 규명(스레드 동기화 이슈인지 확인),
수정 후 `serve`/GGUF 확장 재시도. 우선순위는 낮음 — 현재 shipped
경로(eager repack) 무관.

연관: [[vdsp_general_serving_engine_goal]]
