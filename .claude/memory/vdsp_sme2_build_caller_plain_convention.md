---
name: vdsp-sme2-build-caller-plain-convention
description: vdsp qwen_infer.c/SME2 빌드 시 호출자 파일은 항상 plain(-march 없이) 컴파일해야 한다는 확립된 컨벤션 + M4는 SVE를 SME2 스트리밍모드 안에서만 지원한다는 근본원리(2026-08-25 lldb로 확정), whole-TU 코드젠 부작용으로 다른 함수까지 터질 수 있다는 교훈
metadata:
  type: reference
  originSessionId: e6c100cc-beb0-426d-8425-0959ae41d7af
  modified: 2026-08-25T00:37:31.157Z
---

## 확립된 빌드 컨벤션 (Phase 63/MoE-3c에서 최초 발견, 2026-08-25 lldb로 근본원리 확정)

vdsp(`bob:~/vdsp_m4_bench/qwen_infer.c`)의 SME2 관련 빌드에서:
- **호출자 파일(`qwen_infer.c`)은 항상 plain 컴파일** — `clang -O3 -w -c qwen_infer.c -o qwen_infer.o`, `-march` 플래그 절대 넣지 않음.
- **벤더 커널 `.c`/`.S` 파일(`sme2_kai.c`, `kai_*.c/.S`)만 SME2 관련 `-march` 플래그를 받음**.
- **근본원리(2026-08-25 lldb 백트레이스로 확정)**: Apple M4는 SVE를 별도 항상켜짐 유닛이 아니라 **SME2 스트리밍 모드(PSTATE.SM=1) 안에서만** 제공한다. `main()`을 포함해 일반 함수는 스트리밍 모드가 아닌 상태로 실행되므로, 그 함수를 `-march=...+sme2`로 컴파일해 컴파일러가 SVE 명령어(예: 스택할당의 `addvl`, 벡터화된 루프의 `index z1.d` 등)를 아무 데나 쓰게 하면 그 즉시 illegal instruction(SIGILL)이 난다. 오직 손으로 짠 커널 `.S` 어셈블리 파일들만 `smstart`/`smstop`으로 스트리밍모드를 직접 관리하며 SME/SVE 명령어를 안전하게 쓴다.
- 이 컨벤션을 어기면(호출자까지 SME 플래그로 컴파일) SIGILL(exit 132) 재현.

## ★★ 핵심 교훈: whole-TU 코드젠 변화가 "검증된" 다른 함수까지 깨뜨릴 수 있다 (2026-08-25)

f16p-LHS SME2 커널 실험(Step3 엔진배선) 도중 겪은 실제 사례:

1. `sme2_kai.c`에 새 함수(`kai_sme2_repack_q4g64_f16lhs`)를 추가하면서 그 새 함수의 루프가 자동벡터화로 SVE 명령어를 방출 → 이건 격리 마이크로벤치마크로 미리 잡아서 pragma(`#pragma clang loop vectorize(disable) interleave(disable)`)로 수정.
2. 엔진(`qwen_infer.c`)에 배선하니 SIGILL — 처음엔 "호출자까지 -march 플래그를 받았다"는 컨벤션 위반이 원인이었음(1차 크래시, `main()`의 `addvl`). 이건 기존 컨벤션 재적용으로 해결.
3. **호출자를 plain으로 고쳐도 여전히 SIGILL** — 이번엔 **제가 전혀 건드리지 않은, 몇 주째 검증돼있던 기존 함수** `kai_sme2_repack_q4g64()`(int8 경로)의 nibble순열 루프에서 같은 클래스(자동벡터화→SVE)의 버그가 새로 노출됨. 원인: **1번에서 새 함수를 같은 파일에 추가한 것만으로, `-O2/-O3`의 whole-translation-unit 최적화가 그 파일 안의 다른(무관해 보이는) 함수의 코드젠 결정까지 바꿔버렸다.** 같은 pragma를 이 루프에도 적용해서야 완전히 해결됨.

**일반화된 교훈**: 같은 파일(TU)에 새 코드를 추가하는 것은 "그 새 코드가 하는 일"만 바꾸는 게 아니라, 최적화 레벨에 따라 **그 파일 안의 다른, 이미 검증된 함수의 실제 생성 코드까지 바꿀 수 있다** — 특히 자동벡터화처럼 컴파일러가 전역적으로 판단하는 최적화에서. "이 함수는 안 건드렸으니 그대로일 것"이라는 가정은 안전하지 않다. 회귀 테스트를 "내가 수정한 부분"에만 좁혀서 돌리면 이런 부작용을 놓친다 — 실제로 이번엔 전체 회귀(Gate0/1)까지 통과한 뒤에도 이 버그가 남아있었고, 오직 실제 런타임 실행(CBATCH 경로)에서만 드러났다.

## ★★ 원인불명 blocker에 부딪혔을 때: 대화형 lldb 백트레이스가 결정적

같은 증상(SIGILL exit132, "nthreads=64" 직후 크래시)의 원인을 30개 이상의 격리테스트(코드내용/오브젝트파일/최적화레벨/-march/스레드수/메모리/코드사이닝/재부팅/웹서치+XNU문서조사)로 몇 시간을 못 찾았다. **비대화형 SSH로는 lldb 디버깅 권한(macOS TCC)을 못 받아서** 스택 트레이스 없이 stderr 출력이 끊기는 지점으로만 크래시 위치를 추정하는 블라인드 이분탐색만 계속했던 게 근본 원인이었다.

사용자가 bob 앞에서 직접(Terminal.app) `lldb ./binary` → `run` → `bt`를 실행하자 정확한 크래시 함수/명령어가 즉시 나왔고, 그 자리에서 원인을 이해하고 고칠 수 있었다.

**적용**: SIGILL/SIGSEGV류의 원인불명 크래시를 비대화형 SSH 환경에서 여러 번(2회 이상) 격리테스트로도 못 찾으면, 더 파지 말고 바로 사용자에게 **대화형 세션에서 lldb `bt`를 요청**할 것 — 이게 블라인드 이분탐색보다 압도적으로 빠르다.

## 다음 참고

관련: [[vdsp_general_serving_engine_goal]]
