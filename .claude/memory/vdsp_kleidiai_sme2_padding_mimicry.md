---
name: vdsp-kleidiai-sme2-padding-mimicry
description: "vdsp × ARM KleidiAI SME2 프로덕션 통합 — ★★★★★★★★ 완료+SHIPPED: Phase 0~5 전부 완료, matmul_t/matmul_sdot에 실제 dispatch 연결, macstudio(M1 Max, SME2없음)는 4중 게이트로 완전 무변화 검증, bob(M4)에서 QWEN_SME2 기본값을 하드웨어감지로 승격(무설정시 자동 ON). 실측 end-to-end: Qwen serve B=64 +37%(재현검증됨, B클수록 이득증가), Llama B=16 +26%, 지는 케이스 없음. 후속: cbatch M22 mixed스케줄러 기본값 승격(QWEN_CB_MIXED, macstudio에도 적용되는 비-SME2 변경, TBT지연 165ms→80ms). M1 Max vs M4 범용성능은 M1 Max가 우세(디코드+8%,B=8서빙+20%) — M4는 SME2 발동구간에서만 역전. 근본원인(틀린 pack함수 페어링)은 이전 라운드에서 규명 완료."
metadata: 
  node_type: memory
  type: project
  originSessionId: e6c100cc-beb0-426d-8425-0959ae41d7af
  modified: 2026-08-16T15:13:52.019Z
---

## ★★★★★★★★★ (08-17) 후속: M1 Max vs M4 실측 + SME2 적용구간 확장 3트랙

- **M1 Max vs M4 범용 성능**: 오염 안 된 상태로 직접 측정 — 디코드(M=1) M1 Max
  +8%, 배치서빙(B=8,NEON only) M1 Max +20%. **M4가 이기는 유일한 구간은 SME2가
  실제 켜지는 대배치뿐**(M4 자체 연산력이 아니라 SME2 하드웨어 덕). 처음 관측된
  "M4가 B=8도 이김"은 macstudio의 다른 세션 오염 잔재였음(재측정하면 뒤집힘) —
  이 세션에서 반복된 "동시 세션 측정오염" 패턴 재확인.
- **SME2 적용구간 확장 3트랙 병렬조사**(fork×3): 트랙1(배치상한↑)은 "이득이
  사라진다"는 fork 주장을 **직접 재현검증 실패로 기각**(조용한 상태 재측정하면
  B=16/32/64에서 +2.6%→+14.0%→+37.5%로 오히려 커짐, fork 데이터는 연속 4회
  무거운 연산 직후 열/주파수 이상으로 판단) — ★subagent 반전주장도 그대로 안
  믿고 재현검증한 사례. 트랙2(qwen_spec.c 확장)는 구조적으로 가능하나 이 엔진
  자체튜닝(SPEC_K=3)이 M을 1~4로 묶어 SME2 문턱(16) 못 넘음 → 진행 안 함.
  트랙3(cbatch 활용률)이 유일하게 실제 적용: 이미 구현+문서화(D1-D6)돼있던
  M22 mixed스케줄러(`QWEN_CB_MIXED`)를 단순히 안 켜서 낮은 활용률로 보였던
  것 — 직접 재검증(4요청 bit-identical + TBT 165.7→79.7ms + stall 3→0) 후
  기본값 1로 승격(커밋 macstudio `3537ce6`+bob `f7c6c38`). **QWEN_SME2과 달리
  하드웨어게이트 없어 macstudio에도 즉시 적용되는 변경** — 더 신중히 검증(golden
  무변화+명시적=0으로 구식동작 재현 확인) 후 커밋.

## ★★★★★★★★ (08-16, 최최신) 프로덕션 통합 Phase 0~5 완료 + 기본값 승격 — SHIPPED

아래의 모든 "★★★★★★ 근본적 재정정"~"8차 라운드" 내용은 **원인 규명 단계**(bob의
프로토타입 하네스 `kai_test.c`로 커널 정확성만 검증)의 기록이고, 그 이후 **완전히 별도의
후속 세션**에서 이 결론을 실제 vdsp 프로덕션 엔진(`macstudio:/Volumes/D50/vdsp`)에
통합하는 작업까지 전부 완료됐다. 아래가 현재 최종 상태.

**아키텍처**: 런타임 감지 옵션 경로 — `sysctlbyname("hw.optional.arm.FEAT_SME2"+"SME_I8I32")`
하드웨어 체크(`kai_sme2_available()`) 통과 시에만 `load_int4()`에서 K_Q4G64 텐서를
KleidiAI RHS 포맷으로 1회 리패킹, `matmul_t`/`matmul_sdot`(배치 서빙 경로, `serve`/`cbatch`/
`spec`)에 `kai_route()` 5조건 게이트로 실제 dispatch. `matvec_t`(M=1 디코드)는 의도적으로
미변경. 빌드는 `build_qwen_infer.sh`로 단일TU→멀티파일 전환, SME2 전용 플래그
(`-march=armv9-a+sve2+sme2`)는 `.S` 2개+`kai_matmul_clamp_*.c`(자체 `#error` 가드 보유,
실제 시도로 발견)에만 격리 — `qwen_infer.c` 자체 코드젠은 매 Phase마다 otool로 SVE/SME
명령어 카운트=0 확인.

**안전장치(4중, macstudio(M1 Max, SME2 없음) 완전 무변화 보장)**: env `QWEN_SME2`/
`QWEN_SME2_REPACK`(미설정 시 하드웨어감지값이 기본값 — SME2 기기서 자동ON, 비-SME2
기기서 자동OFF, 명시적 0/1로 항상 오버라이드 가능) × 하드웨어체크 × 포맷(K_Q4G64) ×
per-tensor 리팩성공여부. `QWEN_SME2=1`+`QWEN_SME2_REPACK=1` 강제해도 macstudio에서
SIGILL/출력변화 전혀 없음(하드웨어게이트가 실질 방어선) 매 Phase 재검증.

**정확성 검증**: 실제 프로덕션 가중치(Qwen2.5-1.5B+Llama-3.1-8B) 8개 텐서로 oracle대조+
byte단위 독립재구성대조+의도적손상3종 negative control+guard-page 오버런테스트 전부 PASS
(rel_rms~0.5%, 이 세션 전체가 확립한 정상 양자화노이즈 대역). bias-add 경로(q/k/v_proj)도
별도 검증(거의 기계정밀도). 실제 엔진 dispatch에서 토큰스트림이 SME2 off/on 간 완전 동일
(양쪽 모델, 여러 배치크기).

**실측 성능(bob M4, 실제 serve 모드, end-to-end)**:
| 모델 | B | NEON | SME2 | 배율 |
|---|---|---|---|---|
| Qwen2.5-1.5B | 16(fp32 KV, 상한) | ~199 tok/s | ~203 tok/s | +2% |
| Qwen2.5-1.5B | 64(KV int4) | 160.0 | 219.4 | **+37%** |
| Llama-3.1-8B | 16(KV int4) | 35.9 | 45.4 | **+26%** |
| Llama-3.1-8B | 64 | 36.1 | 미확보(bob 16GB 한계, 아래) | — |

지는 구성 없음, 배치 클수록 이득 커짐. Llama B=64+SME2는 6.6GB blob+7.25GB KV int4+
3.71GB 리팩버퍼≈17.5GB > bob 물리메모리16GB로 memory pressure 발생, 크래시 아니고 진행
중이었으나 시간상 kill — **더 큰 RAM SME2 기기 확보 시 재측정 대상**(사용자 명시 보류
지시, `RESULTS_SME2_INTEGRATION.md`에 open item으로 기록됨).

**커밋(둘 다 로컬만, push 없음)**: macstudio `18c2674`(Phase2,vendor)→`b5b43f9`
(Phase3,repack)→`e6e4813`(Phase4,dispatch)→`7954a61`(기본값승격+
`results/RESULTS_SME2_INTEGRATION.md`). bob `67e768f`+`a68a51e`+`5189750`(미러+검증하네스).
Phase1(`9c0d48b`, 빌드분리)은 더 앞서 별도 커밋.

**남은 항목**: Llama B=64 재측정(RAM 확보 시), Task#10(llama31_8b fp32 golden 재캡처,
Phase0에서 다른세션 메모리경합으로 미완), ARM GitLab 이슈 2건(is_clamp_valid=제출가능,
fmlalb건=철회됨) — **전부 사용자 명시 승인 전엔 제출 안 함**(CLAUDE.md §18).

상세 전체 경위(정찰+커밋별 diff요약+검증숫자 전부): `~/Desktop/HISTORY/2026-08-16_vdsp-kleidiai-sme2-padding-mimicry.md`
전체 문서, `results/RESULTS_SME2_INTEGRATION.md`(macstudio, 프로젝트 자체 최종 문서).

---

## ★★★★★★ (08-16, 원인규명 단계 기록) 이 프로젝트 전체를 관통하는 근본적 재정정 — "PASS" 전부 거짓이었음

**결론부터: down_proj "76.4% 빠름", q/k/gate padding "41~57% 개선+bit-exact", 이 세션에서
KleidiAI 관련해 "검증됐다"고 보고한 모든 정확성 주장이 전부 거짓이었다.** 원인은 KleidiAI
커널이 아니라 **내 테스트 하네스(`kai_test.c`)의 비교 로직 자체의 버그**:

```c
if (rel > worst_rel) { worst_rel = rel; worst_i = i; }
if (a > worst_abs) worst_abs = a;
```

IEEE754에서 `NaN > 무엇이든`은 항상 false다. y_kai가 NaN으로 가득 차 있어도 이 `>` 비교는
절대 true가 안 되므로 worst_rel/worst_abs가 초기값 0에서 전혀 움직이지 않고,
`0 <= 5e-3` → **무조건 PASS로 오보고**된다. LLDB로 실제 메모리를 직접 읽어 확인: "PASS"라고
출력된 실행에서 y_kai의 1024개 원소 **전부**가 NaN이었다.

**비교 로직을 고치고(명시적 `isnan()` 체크 추가, D28) 원래 배터리(down_proj 대표shape
nb=64~128 전부 포함)를 재실행하니 예외 없이 전부 FAIL.** nb=128(가장 신뢰할 수 있다고
믿었던, "down_proj에서는 확실히 이긴다"의 근거였던 shape)조차 100% NaN. **즉 이 세션
(그리고 아마 이전 세션들까지) 전체에서 KleidiAI SME2 커널이 올바른 결과를 낸 적이 단
한 번도 검증되지 않았다** — 오히려 항상 NaN을 냈는데 하네스 버그가 이걸 숨겨온 것.

## 재확정 필요한 것 (전부 미검증 상태로 되돌림)
- ~~KleidiAI 큰 K(down_proj류) 정확성~~ → 미검증, 아마도 실패
- ~~zero-padding mimicry 정확성(q/k/gate)~~ → 미검증, 아마도 실패
- ~~작은 K 이분탐색 결과(nb=64 실패/nb=224 성공)~~ → "성공"으로 분류된 것들도 재검증 필요
- is_clamp_valid 32/64비트 폭 불일치 버그 자체는 **여전히 실재하는 진짜 버그**(D27, 아래
  참고) — 다만 이게 NaN의 원인인지, 아니면 원래도 다른 이유로 항상 NaN이었는지는 재검증 필요

## D27: is_clamp_valid 버그 (실재, 패치 완료 — 위 재정정과 별개로 여전히 유효)
`kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa.c`의 `KernelArgs.is_clamp_valid`가
`uint32_t`(4바이트)인데 어셈블리 커널은 `ldr x5,[x0,#0x28]`(8바이트)+`cbz x5`로 읽음 → 컴파일러
정렬 패딩(0x2C~0x30)이 초기화 안 된 스택 쓰레기값 → 매 호출 랜덤하게 clamp 경로 진입/스킵
결정. `size_t`로 타입 변경해 패딩 자체를 제거하는 패치 적용 완료, LLDB로 재확인: x5가 이제
결정적으로 0x0. **그러나 이 패치 후에도 결과는 여전히 NaN** — 그래서 "두 번째 버그를 찾자"고
더 팠다가 위의 하네스 버그(D28)를 발견하게 됨.

## D29: 재검증 진행 상황 (더 깊이 팠으나 정확한 지점 미확정)

D28(하네스 NaN버그) 수정 후 `down_proj` 대표shape(nb=128)부터 재검증 → 전부 FAIL 재확인.
가장 단순한 손검증 가능 케이스(out=4, in=64 단일블록, 전부 code=1/scale=1.0/act=1.0,
기대값 64.0)로 축소해 LLDB로 명령어 단위 추적:

- `vdsp_ref`(참조 구현)는 정확히 64.0 산출 — 참조 구현 자체는 무죄.
- KleidiAI 커널은 이 극도로 단순한 케이스에서도 NaN.
- 추적 중 `fmlalb`(FP16 전용 widening multiply-accumulate) 명령이 bf16으로 packing된
  weight scale을 처리하는 지점에서 z8이 NaN이 되는 걸 발견 — `FMLALB`는 FP16 전용,
  `BFMLALB`가 BF16 전용인 별개 명령인데 어셈블리는 `fmlalb`를 씀. RHS packing 소스
  (`kai_rhs_pack_nxk_..._s1s0.c`)를 확인하니 scale은 `memcpy`로 bf16 그대로 보존(변환 없음)
  — 즉 진짜로 bf16 데이터를 fp16 전용 명령으로 처리하는 셈.
- **그런데 이 가설은 일반화 안 됨**: 이 hand-verifiable 테스트는 activation을 전부 동일값
  (1.0)으로 넣은 극단적 edge case였고, 그 결과 activation quant code가 전부 127(0x7f)이 되어
  z19가 FP16으로 봤을 때 우연히 NaN 범위에 빠진 것. **랜덤 데이터 케이스(nb=1)로 같은 지점을
  재확인하니 z19/z8/z28 전부 유한하고 정상적인 값**(fmlalb 단계는 무죄). z24(fmul 이후, 16개
  ZA_LOOP 반복 중 첫 4개)도 전부 유한값 — NaN이 정확히 어느 반복/어느 단계에서 나오는지는
  아직 미확정.

**결론**: is_clamp_valid(D27)와 하네스 NaN버그(D28) 둘 다 실재하는 진짜 버그였고 수정
완료했지만, 여전히 미해결인 "커널이 첫 호출부터 항상 wrong/NaN을 낸다"는 현상의 정확한
발생 지점은 이번 세션에서 못 찾음. fp16/bf16 명령어 불일치는 특정 edge case에서만 관찰됨
(일반 원인 아님). 남은 16개 ZA_LOOP 반복 전체 + 최종 저장까지 명령어 단위로 훑는 작업이
남아있음 — 상당한 추가 시간 필요.

## D30: 16개 ZA_LOOP 반복 전체 추적 → 두 개의 별개 버그로 구체화

Python LLDB 스크립트로 nb=1(랜덤 데이터, out=64 in=64 M=16)의 16개 ZA_LOOP 반복(각 반복 =
M-row 1개, `dst_stride_row`=256바이트 증가로 확인) 전체를 자동 추적:

**버그 A: 마지막 M-row(iteration 16/16)만 100% NaN**, 나머지 15개 row는 NaN은 아님. out=64가
정확히 4의 배수라 padding이 필요 없는 shape인데도 발생 — 경계/off-by-one 버그로 추정.

**버그 B: NaN이 아닌 15개 row 전부 크기가 약 100만배 틀림.** y_ref는 -0.78~1.2 수준인데 y_kai는
5.9e+06 등. 원인 추적: `fmul z24.s, z8.s, z28.s`에서 z28(raw INT32 누적값, scvtf 후)은
~49364로 합리적(64개 int4×int8 곱의 합으로 타당). **z8(합쳐진 dequant scale, `fmlalb
z8.s,z19.h,z0.h`로 계산)이 ~120으로 나오는데, 원래 0.0001 수준이어야 함.** z0(weight scale)은
독립적으로 bf16 디코딩하면 ~0.015로 정상. z19(activation scale 관련, LHS packing에서 옴)는
FP16으로 해석하면 ~109 — z8(~120)이 z19(~109)와 비슷한 크기라는 게 수상함(단순 z19×z0였다면
~1.6이어야 함).

**가설(미확정)**: `FMLALB`/`FMLALT`는 ARM ISA상 FP16 전용 widening multiply-accumulate이고
`BFMLALB`/`BFMLALT`가 별개의 BF16 전용 명령인데, 어셈블리는 `fmlalb`(FP16)를 쓴다. RHS
packing 소스는 scale을 memcpy로 bf16 그대로 보존(변환 없음, D29에서 확인)하므로, 커널이
BF16 비트패턴을 FP16으로 잘못 해석해 스케일이 크게 틀어지는 것으로 보임 — 다만 정확한 배율
불일치(약 100만배)를 완전히 재현하는 수식적 증명까지는 못 함(fmlalb가 z19/z0의 특정
half-lane을 조합하는 정확한 방식까지는 ARM ISA 세부문서 없이 확정 어려움).

## D32: Codex 브레인스토밍 — 중요 정정 (단순 fmlalb→bfmlalb 치환은 오답)

Codex가 공개 KleidiAI 소스를 직접 읽고 확인: **RHS(weight) scale은 BF16이 맞지만, LHS
(activation) scale은 실제로 FP16**(`kai_cast_f16_f32()`로 저장)이다. 즉 커널은 FP16(LHS)과
BF16(RHS)을 **의도적으로 혼합**해서 쓰는 설계이고, 이 둘을 합치는 `fmlalb`가 FP16 전용이라
RHS(BF16) 쪽만 오독된다. → **"fmlalb를 bfmlalb로 단순 치환"은 틀린 해법**(그러면 이번엔
LHS의 진짜 FP16을 BF16으로 오독하게 됨). 올바른 방향은 양쪽 스케일을 FP32로 명시 변환 후
FP32 곱셈으로 결합(기존 packed ABI는 유지).

Codex가 제시한 "지금 당장 할 만한 것 Top 3":
1. NaN-safe 중앙 비교기(`vdsp_compare_f32` 류) + 독살 테스트 + 전체 `rg` 감사(1–2일) — 과거/
   미래의 거짓 PASS 광범위 차단.
2. KernelArgs류 asm-ABI 구조체 단일 스키마 생성 + `_Static_assert`(2–4일) — 폭 불일치
   버그 클래스 자체를 컴파일타임에 구조적으로 제거.
3. SME2 커널 검증 전까지 격리 유지 + ARM GitLab에 이슈 2개(폭불일치/스케일오독 별도) 제출
   + FP32 scale-convert 프로토타입 — 단순 BFMLALB 치환은 배제.

ARM 공식 기여 경로는 GitHub(read-only 미러) 아닌 **Arm GitLab**. 버그1(폭불일치)은 재현
결정적이라 바로 PR 가능, 버그2(스케일오독)는 먼저 이슈로 메인테이너와 방향 합의 필요.

전체 브레인스토밍 로그: `/Users/xox/.claude/plugins/data/codex-openai-codex/state/xox-3fac7160d2f521a6/jobs/task-msv36y8o-k1ey9j.log`

## D31: ARM 공식 ISA 레퍼런스로 버그 B 100% 확정

ARM 공식 SVE 명령어 인덱스(stanford.edu 미러, arm developer 원본과 동일 계보) 및
FMLALB 개별 명령어 페이지로 확정:
- **`FMLALB (vectors, FP16 to FP32)`**: Source 1/2가 **Half-precision(FP16) 전용**,
  연산은 `FPMulAddH()`(half-precision 전용 헬퍼) 사용. "No BF16 variant is specified."
- **`BFMLALB (vectors)`**: "BFloat16 multiply-add to single-precision" — **완전히 별개의
  인코딩을 가진 다른 명령**.

RHS packing 소스(D30에서 확인)는 scale을 `memcpy`로 BF16 비트 그대로 보존하는데, 커널
어셈블리는 이걸 처리할 때 `fmlalb`/`fmlalt`(FP16 전용)를 쓴다. **이건 ARM KleidiAI 벤더
소스 자체의 확정된 진짜 버그**: RHS packing이 명시적으로 `scale_dt==kai_dt_bf16`을 요구
(assert)하면서, 그걸 소비하는 커널이 FP16 전용 widening 명령으로 처리 — dequant scale이
완전히 잘못된 크기로 계산됨(관측: ~100만배 차이, 특정 edge case에서는 NaN까지 발생).

**리포트 가치 있음**: is_clamp_valid(D27) + 이 FMLALB/BFMLALB 불일치(D31) 둘 다 ARM-software/
kleidiai에 리포트할 가치가 있는 실재 버그. 단 ARM 소유 repo라 CLAUDE.md §18에 따라 사용자
명시 확인 후에만 실제 이슈/PR 제출.

## Phase 0/1 완료 (08-16, Opus 에이전트 우선순위 계획 기반)

D28(하네스 NaN버그)+D32(Codex 브레인스토밍) 이후, Opus 에이전트가 우선순위 실행계획을
작성 → 사용자 승인 순서대로 진행:

- **Phase 0 완료**: bob(`~/vdsp_m4_bench`, 버전관리 전무했던 데이터손실 리스크)
  전체 백업(`~/vdsp_m4_bench_backup_20260816_101437.tar.gz`) + `git init` + 최초 커밋
  `a375267` + ARM 업스트림 클린 baseline 클론(`kleidiai_upstream/`, D27 패치 외 diff 없음
  확인).
- **Phase 1 완료**: D28과 동일한 NaN-swallow 버그 클래스(`if(e>cur)cur=e;` / Python
  `max(cur,x)` — NaN과의 `>` 비교는 항상 false라 값이 조용히 갱신 안 됨)가 KleidiAI
  하네스뿐 아니라 **vdsp 메인 프로덕션 저장소**(`macstudio:/Volumes/D50/vdsp`)의 실제
  테스트/벤치 정확성 게이트에도 광범위하게 존재함을 확인. 저장소 전체(C/H/S 45개+Python
  63개, paper_width/paper_engine 워크스페이스 제외) grep 스윕 → 분류 A(진짜 거짓-PASS
  버그) 11개 파일/사이트 전부 확정+수정+무회귀검증+poison-test검증(NaN 주입 시
  PASS→FAIL 또는 값 자체가 "nan"으로 노출 확인)+배포. `vdsp-compare-nan-audit` 브랜치
  커밋 `d1501e3`(로컬 커밋만, main에 머지 안 됨, push 안 함).
  - 수정 파일: test_attn_neon.c, g256_bench.c, compare_parity.py, test_vdsp_nn_ops.c,
    train_vdsp.c, gradcheck_fftw.c, gradcheck_main.c, spectral_trunk_infer.c,
    spectral_trunk_infer_v2.c, verify_spectral_trunk_head_lib.py,
    verify_spectral_trunk_lib.py.
  - 분류 B(게이트 자체 안전, 리포트값만 오해소지, 미수정 유지): parity_g256sf.c.
  - **주의**: 이 저장소의 `llm_engine/eval/gptq_quantize_fold_g64.py`가 이 audit과
    무관한 M52(joint alpha×percdamp grid search) 작업으로 uncommitted 상태로 남아있음
    — 다른/동시 세션 작업으로 추정, 손대지 말 것.
  - 상세 근거/poison-test 로그: `~/Desktop/HISTORY/2026-08-16_vdsp-kleidiai-sme2-padding-mimicry.md`

## Phase 2/3 완료 (08-16, 같은 세션 계속)

- **Phase 2** (bob 커밋 `67076a7`): `KernelArgs`에 `_Static_assert` 15개(오프셋) +
  `sizeof` 방어 14개 추가. **중요 설계 교훈**: offset-only assert는 D27류 회귀를 못 잡음
  (8바이트 정렬 padding 때문에 뒤 필드 offset이 안 변함 — negative control로 실측 확인),
  `sizeof(((KernelArgs*)0)->field)==8` 방어가 반드시 추가로 필요. 양성/음성 대조 둘 다
  실제 컴파일+전체 통합 빌드(`kai_test.c`)로 검증.
- **Phase 3** (bob 커밋 `5e1eb8c`+`d4321ad`): D31 FP32 scale-convert 프로토타입 작성+
  bob(SME2 실기기)에서 실행 확인(오차 72배→0배로 해소). ARM GitLab 이슈 초안 2건 작성
  (이슈1=D27 패치첨부, 이슈2=D31 discussion-first) — **DRAFT ONLY, 미제출**, 실제 제출은
  사용자 명시 승인 필수(CLAUDE.md §18).

**Opus 우선순위 계획(Phase 0/1/2/3) 전체 완료.**

## 실제 재검증 결과 (08-16, 같은 세션) — macstudio main 머지 + bob 실측

macstudio `vdsp-compare-nan-audit`→`main` fast-forward 머지 완료(`d1501e3`, push 안 함).

bob에서 is_clamp_valid+fmlalb 두 버그를 실제로 고치고 재검증 → **★★★★★ 신규 버그 #3 발견**:
`kai_matmul_clamp_f32_..._sme_mopa.c`의 `lhs_scales` 포인터 계산식(`lhs_packed_stride -
mr*num_blocks*multiplier`)이 실제 레이아웃(scale이 앞쪽 32바이트, code가 뒤 1024바이트)과
정반대로 계산돼 코드 바이트 영역을 가리킴 — LLDB로 ground truth(`kai_lhs_quant_pack...c:95`
직접 breakpoint) 대 실제 커널이 읽는 위치를 대조해 확정. ARM 업스트림과 diff 동일(벤더
소스 자체 buggy이거나 pack/matmul 페어링 오선택, 원인 미분류).

**실측 개선(nb=1, out=64 in=64 M=16 랜덤데이터)**:
baseline(NaN 64개, worst-rel=5.68e10) → lhs_scales 포인터 수정만(NaN **0개**, worst-rel=5.91e5,
10^5배 개선) → +RHS scale을 진짜 FP16 비트로 저장(asm 안 건드리고 D31 가설 우회검증,
worst-rel=2.32e3, 추가 250배 개선).

**최종 결론: 부분적으로 성공, 완전하지는 않음.** 두 알려진 버그+신규 버그#3까지 다 고치면
NaN 완전 해소+오차 7자릿수 개선이지만, 여전히 요소별 2.99~58.79배(부호도 뒤섞임, 고정배율
아님) 잔차 존재 — **네 번째 미확정 원인**(SME2 zip1/zip2 인터리빙 또는 fmlalb/fmlalt
b/t분할 레인 구조 자체일 가능성) 존재. 실제 asm 수정은 리스크 판단상 이번 세션엔 시도 안 함
(모든 실험은 임시 `/tmp` 파일로만, 세션 종료 시 정리 완료 — 커밋된 파일은 변경 없음).

상세 실측 로그/build recipe: `~/Desktop/HISTORY/2026-08-16_vdsp-kleidiai-sme2-padding-mimicry.md`
"2026-08-16 세션 재개 2차" 섹션.

## "네 번째 원인" 계속 추적 (같은 세션 3차)

가설1(니블 offset-by-8 vs natural 2's-complement 불일치) **기각**: `kai_rhs_pack_nxk_...c:241`의
`dst_q ^ 0x8888`가 정확히 두 컨벤션을 상호변환하는 연산이라 기본(미수정) 니블 경로가 이미
정확함을 수식으로 확정(natural 2's-complement 강제 실험은 오히려 이미 맞던 걸 깨뜨렸음, 원복).

**신규 발견(버그 #4, 원인 미확정)**: 행마다 다른 weight code(-8..7)+동일 wscale 컨트롤드
테스트에서 `y_kai[r]`(r=0..7)가 `y_kai[r+8]`(전혀 다른 code인데도)와 **정확히 동일** — 커널
출력이 `r mod 8`에만 의존하는 8행 주기 앨리어싱. D31(스케일 타입)과는 확실히 별개(고유
wscale+균일code 테스트에선 이 앨리어싱이 안 보였음 — code가 안 변하니 숨어있었을 뿐).
의심지점: `za0h.b[w12,0..3]`(raw INT8 dot 누적값 추출)가 `z8/z9/z10/z11`(zip1/zip2로 만든
스케일 그룹, 이론상 행0-15/16-31/32-47/48-63)과 같은 행 그룹핑을 따르는지 불확실 — SME MOPA
ZA타일 2차원 주소지정 완전검증에는 추가 깊은 LLDB 추적 필요, 이번 세션엔 명령어 수준까지
못 짚음.

**최종 버그 목록(4개)**: (1)is_clamp_valid폭불일치(D27,패치됨) (2)fmlalb BF16오독(D31,미패치)
(3)lhs_scales포인터공식오류(미패치) (4)8열주기앨리어싱(신규,원인미확정,미패치).

## 4번째 원인 추가 추적 (같은 세션 4차) — ISA 레벨 한계 도달

ARM 공식 SMOPA/MOVA ISA로 확정: `ZA[row][col]+=sum_k(Zn[4row+k]*Zm[4col+k])`, dim=16(M4
SVL=512bit). `za0h.b[w12,0..3]`은 tile 4개를 각각 읽는 게 아니라 **같은 물리스토리지의
byte-row w12+0..3을 읽는 것**(byte_row=4R+tile_num) — 추출 로직 자체는 정상 확인.

이전 라운드 "행8이 먼저 나옴" 관측은 **XOR-0x8888 역변환 안 거치고 읽은 내 계산 실수**로
정정 — 재확인 결과 행0~15 패킹 순서 완전 정상.

**신규**: `register read za`로 SMOPA 누적 직후(추출 이전) raw 바이트 직접 파싱 →
za0.s row=0의 16개 int32(col0-15)가 **이미 8열 주기로 정확히 반복**(추출 단계 이전에
버그 존재 확정, SMOPA 누적 자체 문제로 좁힘). 유력 가설(미확정): SMOPA가 col당 4바이트
소비하는데 RHS 패킹은 row당 2바이트만 쓰는 구조적 그룹핑 불일치 — 정확한 k축/row축 교차
지점은 블랙박스 리버싱 한계로 미확정.

**CLAUDE.md 운영모드 "2회+ 막히면 정직하게 한계 보고" 기준 도달로 이번 라운드 consolidat.**
계속하려면 ARM 공식 SME programmer's guide 원문 또는 이 커널 원저자 자료 필요할 가능성.

## ★★★★★★★ 6차 라운드 — 진짜 근본원인: pack/kernel 조합 자체가 틀렸었음 (전면 재해석)

`sme2_mopa` 변형 조사 중 헤더 doc comment에서 결정적 단서 발견: 이 프로젝트가 처음부터
써온 커널(`..._sme_mopa`)의 헤더가 명시적으로 `kai_lhs_quant_pack_qsi8d32p_f32_neon` +
`kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon`을 의존성으로 선언 — 그런데 이
프로젝트는 지금까지 **완전히 다른 함수**(`kai_lhs_quant_pack_qsi8d32p_f32`no-neon +
`kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0`)를 써왔음(비슷한 이름으로 오선택, investigation
훨씬 이전 복구불가 세션에서). ARM 공식 테스트 스위트
(`test/tests/matmul_clamp_f32_qsi8d32p_qsi4c32p_test.cpp`의 `UKernelVariants` 테이블)로
100% 확정.

올바른 pack 함수 소스 확인: **scale은 처음부터 끝까지 진짜 FP16**(BF16 아님, D31 전제
자체가 틀렸었음), **1/16 사전보정**(lsl/and 니블디코드용, 내가 추측했던 바로 그것),
**K+0/K+4 페어링**(K+0/K+16 아님, SMOPA 4-way가 실제 기대하는 그룹핑), LHS scale은
"packed값들 뒤에 저장"(wrapper 원래 공식과 정확히 일치, 5차라운드 "수정"은 불필요했음).

ARM 공식 테스트 fixture 그대로 참고해 새 테스트(`kai_test_correct.c`, bob 커밋
`844da1f`) 작성 — **원본 wrapper**(D27만 유지, 다른 핵 전부 제거)+**올바른 `_neon` pack
함수 2개**로 재구성 → **시드 4개 전부 NaN 0개, worst_abs/typical 2.09~2.62%**(정상
int4/int8 양자화 노이즈, 버그 아님). worst_rel 큰 값은 전부 ref≈0 지점 아티팩트로 확인.

**전면 재해석**: D27(is_clamp_valid)만 진짜 독립 버그(pack선택과 무관한 순수 C 구조체
문제, 패치 유지, GitLab이슈#1 유효). D31(fmlalb/BF16), 5차의 lhs_scales포인터,
4차의 SMOPA크로스블록앨리어싱 — **전부 버그 아니었음, 틀린 pack함수를 써서 생긴
파생 증상**. GitLab이슈#2는 철회(파일에 철회사유 명시, bob 커밋 `3a4c8b5`).

독립 리서치 agent가 ARM 공식 SME Programmer's Guide(109246)로 교차검증: "byte-row =
4×tile_row+tile_number" 메커니즘을 ARM 자신의 Figure 2-11/2-12에서 직접 확인 — 4차
라운드의 크로스블록 관찰과 정합적으로 설명됨(추가 확인용, 실제 근본원인 규명은 ARM
테스트 스위트로 이미 완료).

**최종 결론**: "두 버그(is_clamp_valid+fmlalb)를 고치면 통과하는가?" — **애초에 커널
버그는 D27 하나뿐이었다. 그것만 고치고(+올바른 pack 함수 사용) 재검증하면 통과한다.**
4~5차 라운드의 깊은 SME2 명령어 추적이 안 풀렸던 이유: 애초에 없는 버그(pack 불일치가
만든 신기루)를 쫓고 있었기 때문.

## 7차 라운드 — 후속: 큰 shape 검증 + kai_test.c 본체 개편 (같은 세션)

**nb=128(down_proj대표, out=64/in=8192/128블록) 3시드 검증**: NaN 0개,
worst_abs/typical 1.17~1.50%(소shape보다 오히려 좋음, 정상). bob 커밋 `09885e1`
(`kai_test_correct2.c`, CLI인자화+멀티블록 참조계산 버그도 같이 수정).

**kai_test.c(원본 하네스, 7개 test_* 함수/996줄) 본체 개편**: 이 investigation 전체가
의존한 핵심 두 경로(`test_shape`=nb=N모드, `test_hand_verifiable`=handv모드)를 올바른
`_neon` pack함수쌍으로 이식. bob 커밋 `a3bf2d0`. **검증: nb=1,2,4,8,16 개별 PASS,
handv(D29를괴롭혔던균일활성화edge case) PASS, 기본실행(15개shape 전체스윕,
nb=66~128) ALL SHAPES PASS**(worst-abs/typical 0.13~0.20%, 원래엄격한0.5%임계값도
여유통과). 나머지 5개 함수(padding-mimicry 별도연구트랙, in_real/in_padded/persistent
buffer 메커니즘)는 미이식 — 원래 연구질문 자체가 잘못된 커널호출의 인공물이었을 가능성
있어 기계적 포팅보다 전제 재검토가 먼저 필요.

**최종 결론**: 진짜 커널버그는 D27 하나뿐. 나머지(D31,lhs_scales,SMOPA앨리어싱포함)는
전부 잘못 고른 pack함수의 파생증상. bob 커밋: a375267→67076a7→5e1eb8c→d4321ad→
844da1f(★근본원인)→3a4c8b5→09885e1→a3bf2d0.

## 8차 라운드 — padding-mimicry 트랙 전제 재검토: 전제 자체가 거짓이었음 확정 (같은 세션)

원전제(D19, 이 프로젝트보다 더 이전 "선행조사"): "q/k/v/gate/up 실제 K=4096에서 SME2
small-K dead zone에 걸려 실패" → 패딩으로 크게 보이게 속이자는 아이디어.

**직접 재검증**: 이식된 kai_test.c로 nb=1,2,4,8,16,32,48,63,**64**,65(K=64~4160) 전체
스윕 → **예외 없이 전부 PASS**(K=4096 포함, worst-abs/typical 0.16~0.5%, 정상 노이즈).
**"small-K dead zone"은 실재 SME2 하드웨어/커널 특성이 아니라, 틀린 pack함수의
correctness버그가 K와 무관하게 모든 shape에서 발생한 걸 랜덤데이터 corruption패턴 우연으로
크기-의존 경계처럼 오인한 것.**

**실무 함의**: padding-mimicry 관련 미이식 5개 함수(test_shape_padded 등, in_real/
in_padded/persistent buffer 메커니즘)는 **포팅 불필요** — 풀려던 문제 자체가 없었음.
q/k/v/gate/up은 그냥 test_shape로 실제 K 그대로(패딩 없이) 호출하면 이미 정상. 코드는
과거기록으로 남기되 "이 방법 써야 한다"는 결론은 폐기.

**남은 작업**: GitLab이슈#1(D27)만 사용자승인시 제출가능. padding-mimicry 인프라는
불필요한 복잡도로 확정, 추가 작업 없음.

## 5차 라운드 — 결정적 반증: 크로스-블록 오염 확정, 수렴 안 됨

가설(SMOPA col이 2행씩 묶는다) 검증하려 행16-31에 "역순이지만 서로 다른" code 배정 →
행0(code=-8)/행16(code=+7, 완전 다름)의 kai가 거의 동일 → 단순 행-쌍 가설도 기각(진짜
버그로 재확정).

**결정적**: 64행 전부 4개 다른 순열로 채운 테스트(za3)에서 **행0-15의 kai는 완전 불변,
행16-31의 kai는 자기 자신 code가 안 바뀌었는데도 행32-63 데이터를 바꾸자 값이 바뀜**
(예: r=16이 za2에서 1941.115, za3에서 675.540 — code는 둘 다 7로 동일). → 크로스-블록
데이터 오염(단순 오배선 아니라 실제 합산/누적 오염) 확정. 블록0(row0-15)은 오염 안 되고
블록1(row16-31)만 오염되는 비대칭성까지 있음.

**4라운드 연속 "가설→실측반증→더 복잡한 그림" 패턴 반복 — 블랙박스 LLDB 리버싱으로는
수렴 안 됨 확정.** 버그 존재/일부특성(NaN,크로스블록오염,SMOPA누적단계원인)은 견고하게
실증됐으나 정확한 명령어 수준 원인은 ARM 공식 문서/원저자자료 없이는 이 세션 범위 밖.

## 다음 세션 시작점 (KleidiAI SME2 정확성 재검증 — Phase 0/1과 별개 트랙)
1. **모든 정확성 재검증은 D28(하네스 NaN버그) 수정 이후 버전의 `kai_test.c`로만 할 것.**
   로컬: `/private/tmp/claude-501/-Users-xox/e6c100cc-beb0-426d-8425-0959ae41d7af/scratchpad/q4gemv_m4/kleidiai/kai_test.c`
   (세션 종료 시 스크래치 디렉터리 유실 가능성 있음 — 필요시 원격 bob:`~/vdsp_m4_bench/kleidiai/`에서 복구)
2. down_proj 대표 shape(nb=128 등)부터 D28-fixed 하네스로 재검증 — 이게 여전히 실패한다면
   is_clamp_valid 패치와 무관한 **더 근본적인 문제**(빌드 환경? 커널 자체의 실제 결함? 잘못된
   API 사용?)를 의미하므로, KleidiAI 사용법을 처음부터 재검토해야 함.
3. is_clamp_valid 패치(D27)는 유지 — 진짜 버그이므로 되돌리지 말 것.
4. ARM KleidiAI 업스트림에 D27은 리포트할 가치 있음(내 소유 repo 아님, CLAUDE.md §18에 따라
   사용자 명시 확인 후에만 실제 제출). D28(하네스 버그)은 내 로컬 테스트 코드 문제라 리포트
   대상 아님.

상세 경위: `~/Desktop/HISTORY/2026-08-16_vdsp-kleidiai-sme2-padding-mimicry.md`

## bob은 은퇴 안 함 (스킬 설명 문구 정정)
`~/.claude/skills/studio/SKILL.md`가 "bob은 2026-08-13 완전 은퇴"라고 적어놨지만 이는
**Finance repo 맥락 한정**(Finance 상시가동 워크로드가 Mac Studio로 이전됐다는 뜻)이다.
bob(M4 mini, SME2 보유 실기기)은 실제로 살아있고 SSH 접속 가능하며, Mac Studio엔 SME2가
없어서 이 vdsp/SME2 작업은 반드시 bob에서 해야 한다. vdsp/SME2/M4 하드웨어 관련 작업 시
"bob 은퇴" 메모리를 그대로 믿지 말고 `ssh bob "echo alive"`로 직접 재확인할 것.

## LLDB 사용법 (이 세션에서 확립, 재사용 가능)
- 비대화형 SSH에서 "cannot get permission to debug processes" 에러 → sudo 문제, `codesign -f
  -s - --entitlements <get-task-allow.plist> ./binary`로 ad-hoc 서명하면 sudo 없이 해결.
- 주소 기반 breakpoint(`breakpoint set -a 0x...`)는 프로세스가 **launch되기 전**에 걸면
  안정적으로 안 걸림(이 플랫폼 한정 관찰) — 반드시 이름 기반 breakpoint로 최소 한 번 정지시킨
  뒤에 주소 기반 breakpoint를 추가할 것. Python 스크립트에서도 동일 — `LaunchSimple` 전에
  offset 기반 주소를 계산하면 안 됨.
- SME2 레지스터(z0-z31, za, svcr, p0-p15)는 `smstart` 실행 이후 지점이면 `register read`로
  읽힘(이전 세션 가정과 달리 LLDB가 실제로 지원함).
- 주소는 프로세스 실행마다 ASLR로 달라짐(heap 특히) — **절대 한 lldb 세션에서 캡처한 주소를
  다른 세션에 하드코딩해서 쓰지 말 것**(이 실수로 "memory read failed" 반복 경험, "call1도
  NaN"이라는 오판으로 이어질 뻔함). 같은 세션 내에서 `expression -- unsigned long $var = $reg`로
  값을 convenience variable에 저장해 재사용할 것.
