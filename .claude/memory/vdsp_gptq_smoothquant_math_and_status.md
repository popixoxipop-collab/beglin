---
name: vdsp-gptq-smoothquant-math-and-status
description: vdsp repo의 GPTQ+SmoothQuant 파이프라인 수학 원리 + M48(Llama golden gate 수립+엔진실측) 완료 현황, 엔진/Python ppl 미해결 격차 포함. repo popixoxipop-collab/vdsp
metadata:
  node_type: memory
  type: reference
  originSessionId: 9c8a5d58-13cb-48b9-b36e-c65239ff2688
  modified: 2026-08-16T12:38:08.674Z
---

## SmoothQuant (Xiao et al. ICML 2023) — 핵심 항등식

이 repo는 W=[out,in] 컨벤션(HF Linear 그대로, `export_weights.py` D2/`gptq_quantize.py:44` docstring으로 확인). `Y=XWᵀ`에서 **`Ŵ = W·diag(s)`**(입력채널축=마지막축 브로드캐스트 곱 — `phase0b_smoothquant_scope.py:394` `orig_W[...] * si["qkv"][None,:]`로 실증), 대응해서 `X̂=X·diag(s)⁻¹`. fp32 정밀도에서 완전 동일값. 활성화(X)는 아웃라이어 채널 때문에 양자화 어렵고 가중치(W)는 여유 있음 → `s_j = max(|X_j|)^α / max(|W_j|)^(1-α)`로 "어려움"을 활성화에서 가중치로 옮김. α=0.4는 Qwen 실측 튜닝값(원논문 기본은 0.5), `S_CLAMP=(1e-3,1e3)`으로 극단값 클램프.

## GPTQ (Frantar et al. 2022) — 핵심 아이디어

목표 `min ||WX-ŴX||²`(출력공간 재구성오차, RTN처럼 가중치를 독립취급 안 함). Hessian `H=xᵀx`(`x`는 `[N,in]`=토큰×입력채널로 캘리브레이션 데이터에서 실측 — "입력채널 Gram matrix" 불변량)를 열(column) 단위로 순회하며 i번째 열 양자화 오차를 `H⁻¹`로 아직 안 한 열들에 보정전파(Cholesky, `percdamp=0.01`로 수치안정화). RTN보다 훨씬 정확.

## 이 repo의 결합 (`llm_engine/eval/gptq_quantize_fold_g64.py` 등)

1. WikiText-103 **train**(val과 분리, ppl gate 누수방지), `N_CALIB=32`×`CALIB_LEN=512`
2. 실모델 forward hook(4곳: qkv_in/gateup_in/o_in/down_in)에서 H와 채널별 max(|X|) 동시수집
3. s 계산 → **폴딩**(다음층 가중치에 미리곱하고 앞단에서 상쇄, 런타임비용 0)
4. 스무딩된 가중치에 GPTQ(group-64, percdamp=0.01)
5. `EXPECT_ARM2C_PPL`(Qwen 하드코딩 기댓값)로 재현성 게이트
6. M40에서 배포승격: ppl RTN단독14.87 → **11.5759**(fp32 10.65 대비 +8.8%)

## ★★★ 구현 상태 (2026-08-13 코드 직접 대조로 재확인·정정됨)

- SmoothQuant+GPTQ 수학·코드는 Qwen2.5-1.5B에 **완전 검증+실배포** 중.
- **★정정(당초 이 메모리가 틀렸었음): GQA 풀링은 이미 완전히 일반화되어 있다.** `gptq_quantize_fold_g64.py:231-234`와 `phase0b_smoothquant_scope.py:229-233` 둘 다 `cfg.num_attention_heads`/`cfg.num_key_value_heads`를 모델의 실제 `config.json`에서 런타임에 읽고 `N_REP=NUM_HEADS//NUM_KV`로 계산, o_proj 풀링(`.view(NUM_KV,N_REP,HEAD_DIM)`)도 동적 reshape — 12/2/6 같은 리터럴 하드코딩 아님. `o_proj`를 건드리는 5개 파일(`phase0b_smoothquant_scope.py`/`gptq_quantize_fold_g64.py`/`gptq_quantize_g256.py`/`phase0c_gateup_diag.py`/`phase0b_verify.py`) 전부 동일 패턴. 단 `phase0b_smoothquant_scope.py:234-236`에 로그용(non-blocking) `(12,2)` sanity-check 문자열이 하나 있음 — 실행 분기 아님, 무해.
- q/k/v·gate/up 공유입력축 스케일 문제도 이미 해결됨(D2: 공유 프로젝션들의 W-side 통계를 max-pool로 합쳐 공유 스케일 하나만 생성 — 엔진의 활성화양자화 경로가 공유 스케일 하나만 쓰는 것과 정합).
- **진짜 하드코딩되어 재측정 필요한 것**: `SNAP`(Qwen HF캐시경로, 7개 중 6개 파일 — `phase0b_verify.py`는 `phase0b_smoothquant_scope`에서 import), `ALPHA=0.4`(4개 site 전부 동일값), `EXPECT_FP32_PPL`/`EXPECT_ARM2C_PPL`(Qwen 전용 golden gate 상수), `percdamp=0.01`(GPTQ 표준 기본값이라 그대로 써도 될 가능성 높지만 확인 필요). **`VDSP_BASE` override 자체는 이미 있음**(`gptq_quantize_fold_g64.py` D2: "VBASE via $VDSP_BASE" — 당초 "override 전혀 없음"이라 썼던 것도 부정확했음, `SNAP`만 하드코딩이고 출력경로는 이미 override 가능).
- **★새 리스크(당초 못 봤던 것) — Hessian 메모리**: D4 "모든 레이어 Hessian을 두 arm에 걸쳐 계속 보유(~15GB peak, Qwen D=1536/IM=8960 기준)". Llama D=4096/IM=14336 — projection별 Hessian이 (4096/1536)²≈7.1×~(14336/8960)²≈2.56× 커짐 → 64GB Mac Studio(+`brain.selfplay` 상주)에서 OOM 위험. M48 착수 시 스모크런(N_CALIB=2)으로 실측 먼저.
- **스코프**: `phase0_superblock_scope.py`/`phase0c_gateup_diag.py`/`phase0b_verify.py`는 연구/진단 전용(자체 EXIT: "one-off diagnostic") — Llama 포팅 불필요. `gptq_quantize_g256.py`는 자기 자신을 **"Phase 2 WP1"**이라 부르는 별개 트랙(group-256 대체 포맷, M39/M40에서 fold-g64에 밀려 배포 안 됨) — **레포 내부 "Phase 2"와 우리의 Llama 작업을 혼동하지 말 것, 우리 쪽은 M48로 부름**. 실질적으로 건드릴 파일은 `gptq_quantize_fold_g64.py` 하나.
- `gptq_quantize_fold_g64.py`는 "이미 배포된 g64 블롭"에서 안 바뀐 텐서를 바이트카피하는 구조(`G64_BLOB`/`G64_LAYOUT` 소스 필요) — M47에서 만든 Llama RTN-g64 블롭이 같은 포맷이라 이 용도로 그대로 쓸 수 있을 가능성 높음(M48 첫 단계에서 확인).
- 논문(`paper_engine/workspace/final/paper.tex`, `paper/`가 아니라 이쪽이 실제 LLM엔진 논문)은 GPTQ단독 ppl=**12.10**까지만 기술, SmoothQuant는 Related Work 인용만(L46/L82/L194) — 현재 배포된 11.5759는 **논문 이후 진전된, 논문에 없는 결과**.

## M48 진행 상황 (2026-08-13, Hessian 재구조화+alpha 스윕 완료)

- **Hessian 메모리 리스크 해결됨(측정 기반)**: 실제로 2건의 OOM/스왑쓰레싱 사고를 거쳐 확정. (1) fp32 모델 32GB+`orig_W` 전체클론 ~26GiB(이전 미카운트) 베이스라인만 ~58GB → `VDSP_DTYPE=bf16` 로딩으로 해결(Hessian 수집훅은 내부에서 이미 fp32 캐스팅하므로 정밀도 영향 없음). (2) bf16 수정 직후 곧바로 재발 — `armA_Q`/`armB_Q`가 그룹 전체에 걸쳐 dequant fp32 텐서를 계속 누적(arm당 ~28GiB) → int8 codes+scales 저장(`to_codes`/`dequant_codes`)으로 해결(~14GiB). `VDSP_LAYER_GROUP`(레이어그룹 단위로 Hessian 즉시 del)도 병행 도입. 이 교훈으로 신규 hook `~/.claude/hooks/scripts/memory-accumulation-guard.py` 작성(레이어/타겟 전체 순회+텐서 누적 패턴을 크기추정 주석 없이 짜면 차단).
- **GQA 32:8 실행 검증 완료**: 코드 대조로 이미 일반화돼 있다고 판단했던 것이 실행으로도 확인됨 — Llama config로 shape 에러 없이 완주.
- **S1b 스무딩-only equivalence 게이트 신설**: 폴딩 직후(양자화 전) 로짓을 pristine 모델과 비교, dtype별 tolerance(fp32 1e-3/bf16 5e-2) — alpha 0.2/0.4/0.6/0.8 전부 PASS(rel-L2 0.017~0.020).
- **N-way alpha 스윕 코드 일반화**: 기존 하드코딩 2-arm(max/gate) 메커니즘을 `CANDIDATES` dict 기반 N-way로 확장, 매 리팩토링마다 Qwen 회귀(정확히 `11.5759` 재현)로 검증하며 진행.
- **★alpha 스윕 실측 결과(N_CALIB=32 풀스케일, percdamp=0.01 고정, fp32 baseline=8.3408)**:
  | alpha | ppl | vs fp32 |
  |---|---|---|
  | 0.2 | 10.0893 | +20.96% |
  | **0.4** | **8.9445** | **+7.24%** ★최선 |
  | 0.6 | 9.0971 | +9.07% |
  | 0.8 | 9.8496 | +18.09% |

  Qwen에서 튜닝된 α=0.4가 Llama에서도 그대로 최적 — M47 RTN-only(+17.70%) 대비 열화폭을 절반 이하로 줄임. (2개 모델만으론 "α=0.4가 보편적"이라 일반화하기엔 이름, 데이터 포인트 부족 — 과확대해석 주의.)
- **layout q4 set != target set FATAL 해결**: Qwen 도너는 lm_head를 `q8g64`(byte-copy 전용)로, Llama M47 RTN 도너는 lm_head를 `q4g64`로 저장(M46 export/quantize 경로가 균일 처리) — GPTQ 타겟(`is_target()`)엔 원래도 lm_head가 없으므로, q4g64인데 QCODES에 없는 항목은 "미변경, 도너에서 byte-copy"로 처리하도록 수정(정확 일치 대신 subset 체크로 완화 + q4g64 분기에 byte-copy 폴백 추가). Qwen 스모크 재검증(ppl 13.107976589598518/13.091076165312062, 완전 무변화)으로 확인.
- **★bf16 emit 버그(percdamp=0.01 최종런 도중 실제로 겪음)**: S6 검증 코드가 `sd_t.detach().numpy().astype(np.float32)` 순서라 bf16 텐서에서 `.numpy()` 자체가 `TypeError: Got unsupported ScalarType BFloat16`로 즉시 크래시(astype은 numpy() 성공 이후에나 실행되므로 너무 늦음) — `.float()`를 `.numpy()` **앞**으로 이동해서 해결(`folded_f32[name].float().numpy()`, `sd_t.detach().float().numpy()`). Qwen 회귀가 VDSP_DTYPE 기본값(fp32)만 썼던 탓에 이 경로는 못 잡았음 — **교훈: dtype-분기 코드는 그 dtype으로 직접 스모크해야 함, 다른 dtype 회귀로는 검증 안 됨.** 크래시 시점이 emit 마지막 파일쓰기 단계였던 덕에 핵심 계산(arm B ppl=8.944502088361716, alpha 스윕과 완전 일치)은 이미 확보된 상태로 재현성만 재확인하면 됐음.
- **★★★ 2026-08-14 task #13 완료** (엔진 실측 ppl 검증 + golden gate 수립 + 문서화, commit `8ebd743` push완료): golden gate 상수(`EXPECT_FP32_PPL`=8.340758394928915, `EXPECT_ARM2C_PPL`=8.865384820378445=armA, `_IS_LLAMA31_8B_SNAP` 패턴)를 `gptq_quantize_fold_g64.py`에 반영. `eval_ppl_llama.py`를 `VDSP_INT4_BIN`/`VDSP_INT4_LAYOUT`/`VDSP_RESULTS_TAG`로 파라미터화.
- **★★★★★ 2026-08-15 M49 완료 — 근본원인 찾고 수정+재측정 완료(commit `b9f149c` push됨)**: 원인=`quantize_int4.py`가 tied embedding만 제외하고 `lm_head.weight`는 제외 안 해서, Llama의 untied lm_head가 일반루프에서 q4g64로 한번, 전용 int8블록에서 q8g64로 또 한번 — 레이아웃에 중복 라인 생성. `qwen_infer.c`의 `wt()`는 이름 첫매치 반환이라 항상 q4g64(조악한 쪽)를 실제 서빙하면서 로그엔 "lm_head=int8"이라 찍음(존재여부만 확인, kind는 확인 안 함 — 둘 다 직접 코드 대조로 확인). **이 버그는 M47 도너 레이아웃에도 있어서 M47 자체 숫자도 오염돼 있었음.**
  - 수정: (1) `quantize_int4.py` — lm_head를 tied embedding처럼 일반루프에서 제외 (2) `qwen_infer.c` — `load_int4/load_fp32`에 중복이름 FATAL 추가 + int8-head 플래그를 존재여부 아닌 `kind==K_Q8G64`로 명시 확인
  - 검증: 새 바이너리 Qwen에서 회귀없음(argmax동일) 확인 + 구버그 레이아웃에 새 FATAL이 정확히 걸리는 것 확인 후 프로덕션 교체. M47 RTN 도너는 실제 재생성(빠름, ~3분), M48은 GPTQ 본체 연산 무관이라 레이아웃 텍스트만 교정(수 시간짜리 Hessian 재실행 불필요, 재측정으로 등가성 확인)
  - **결과**: M47 engine RTN-int4 9.8169(+17.70%)→**9.4881(+13.76%)**. M48 engine foldg64-int4 9.2576(+11.00%)→**8.9527(+7.34%)**, engine-vs-Python 격차 +3.50%→**+0.09%**(Qwen M38 정상범위 안으로 완전히 닫힘)
  - golden gate 상수(EXPECT_FP32_PPL/EXPECT_ARM2C_PPL)는 Python 쪽 수치라 이 버그와 무관, 변경 불필요
  - 조사는 fork에 위임(investigation-protocol 준수: 가설8개+isolation C테스트로 커널자체는 결백 확인) 후 핵심 주장(중복라인 존재, wt() 첫매치, 4곳 콜사이트) 전부 직접 코드 대조로 재검증 완료
- **작업 중 발견**: 백그라운드 실행 직후 **다른 로컬 세션이 동일 작업(engine ppl 측정)을 이미 3분 먼저 진행 중**이었음을 뒤늦게 발견 — Mac Studio 메모리가 여유 100MB까지 몰려 OOM 직전까지 감 → 본인 중복 프로세스를 즉시 kill해서 해소. [[feedback_check_other_sessions_before_heavy_compute]]가 실제로 재현된 사례로 추가 기록할 가치 있음(사전 확인 시점과 실제 launch 시점 사이 수분의 gap에서 발생 가능).
- `RESULTS_LLAMA31_8B.md`에 M48 섹션 작성 완료(엔진/Python 격차를 헤드라인으로 명시, 얼버무리지 않음).
- **★★★★★ 2026-08-15 M50 완료(percdamp 튜닝, commit `2ba67bb` push됨)**: M48이 미해결로 남긴 percdamp(GPTQ default 0.01, 미검증) 스윕 완료. `VDSP_PERCDAMP_SWEEP` 메커니즘 신규 추가(alpha 스윕과 달리 Hessian도 공유 가능해서 더 저렴 — Cholesky damping에만 영향). {0.005,0.01,0.02,0.05,0.1} 스윕 결과 **percdamp=0.05가 명확한 우승**(ppl 8.7184 vs 기존 0.01의 8.9445, 다른 후보들과 뚜렷한 격차). 최종 emission 재실행 → armA=8.770210(신규 golden gate D15, 구D13 대체) / armB=8.718396(emit값과 정확히 일치) / 291텐서(M49 dedup 자동유지 확인). **엔진 재측정 8.7198(+4.55%), Python과 격차 겨우 +0.016%**(M49 자체 결과보다도 타이트 — M49 수정이 새 블롭에도 견고함을 재확인).
  - **최종 개선 체인(전부 엔진실측+M49버그프리)**: M47 RTN +13.76% → M48(percdamp=0.01) +7.34%(RTN갭46.6%해소) → **M50(percdamp=0.05) +4.55%(RTN갭66.9%해소)**
  - **작업 중 실제 사고 회피**: 무거운 작업 중간에 `qwen_infer.c`/`qwen_spec.c`에 커밋 안 된 변경사항 발견(다른 세션이 M4칩 Q4_THREADS 자동감지 작업 중, 나중에 `M51`로 스스로 커밋함) — `git status`로 미리 감지해서 `git add -A` 대신 항상 내 파일만 명시적으로 add해서 충돌 회피. 같은 시점에 그 세션이 이미 끝난 M49를 처음부터 다시 조사 중이었던 것도 목격(중복작업, 새 hook `studio-gitlog-guard`가 정확히 이 상황을 경고)
  - **교훈**: "--smoke"가 항상 저렴하지 않음 — percdamp 스윕 비용은 GPTQ Cholesky 분해(O(n³), N_CALIB과 무관)가 지배적이라 N_CALIB=2로 줄여도 스모크 테스트가 ~98분 걸림(alpha 스윕은 반대로 calibration forward pass가 지배적이라 스모크가 진짜 빠름) — 스윕 종류마다 비용 구조가 다르다는 걸 먼저 분석해야 함
- **★★★★★ 2026-08-16 M52 완료(alpha×percdamp 조인트 그리드, commit `1111d7b` push됨)**: 4×5=20조합 전체 그리드 실측 완료. **결과: alpha=0.4/percdamp=0.05(M50이 이미 찾아 배포한 값)가 20개 중 전역 최적** — 재emission 불필요, M50 블롭 그대로 유효. 2위 alpha=0.2/percdamp=0.005(8.7394)는 근소하게 못 미침. alpha별 percdamp 반응 패턴이 완전히 다름(0.2=단조감소로 폭발적악화, 0.8=단조개선, 0.4/0.6=중간에 최적점) — 순차튜닝이 우연히 전역최적을 찾은 것으로 확인.
  - **★★★★★ 실제 사고: 첫 시도(20개 한 프로세스) OOM으로 조용히 죽음** — Python 트레이스백 없이 SIGKILL(resource_tracker 경고만). 원인: Q_STORE가 후보 전부(각~7GiB)를 동시보유하는 구조라 20개=~140GiB로 64GB초과. **계산시간 스케일링(20×조합당시간)은 사전계산했으면서 메모리 스케일링(20×조합당메모리)은 똑같은 산수를 완전히 빠뜨림.**
  - **재발방지 훅 신규 작성+테스트+등록**: `~/.claude/hooks/scripts/sweep-grid-memory-guard.py`(PreToolUse:Bash, settings.json 등록) — 서로 다른 콤마-split 스윕리스트 2개가 중첩루프로 결합되는데 곱셈 크기추정 주석이 없으면 차단. 실제 버그코드/수정된코드/단일축스윕 3케이스로 검증 후 적용.
  - **복구**: 20개 한방 대신 alpha별로 나눠 4번(M50재사용으로 실제론 3번) 순차실행 — 이미 검증된 5후보=~35GiB 안전구조 재사용. alpha=0.2/0.6/0.8 각 percdamp=0.01에서 M48 원래 alpha스윕 수치와 정확히 일치(재현성 재확인).
  - **작업 중 또 다른 세션의 병행 커밋 발견**(`d1501e3 fix(vdsp-nan-audit)`) — `git add -A` 대신 항상 내 파일만 명시 add하는 습관이 이번에도 충돌 회피
- **2026-08-16 논문 반영 완료(commit `b89ae44` push됨)**: `paper_engine/workspace/final/paper.tex`("Nothing but NEON" Qwen2.5-1.5B 엔진 논문)에 M47~M52 전체를 새 섹션 "Generalizing to a Second, Larger Model: Llama-3.1-8B-Instruct"로 추가. paper-orchestra 스킬은 D50 심볼릭링크가 로컬에서 깨져 있고(Mac Studio SSH로만 실제 파일 접근 가능) 원래 새 논문 전체작성용(outline+plotting+lit-review 풀파이프라인)이라 "완성 논문에 섹션 추가"엔 안 맞아서, 사용자 확인 후 직접 섹션 작성으로 진행(paper-orchestra-guard 훅의 반응도 CGLM/물리논문 전용 "velocity dependence declaration" 리마인더라 이 논문 도메인과 무관— 참고만 하고 논문 자체 스타일([V0]/[V1]/[V2] 태그, booktabs 표)만 그대로 모방). Llama 3 인용(dubey2024llama3) 1건 추가. **로컬 pdflatex+bibtex 3-pass 풀 컴파일 검증 완료**(undefined reference 0개, 25페이지, PDF재생성) — Mac Studio엔 LaTeX 툴체인 자체가 없어서 로컬에서 실측 컴파일. 커밋 전 사용자 확인 받음.
- 세부 경과: [[2026-08-13_vdsp-m47-m48-llama-gptq-smoothquant]] (session history)

## 관련 결과
- [[vdsp_m46_python_pipeline_and_d50_incident]] - M47: RTN단독 결과 +17.70%(fp32대비, HF fp32=8.3404, engine fp32=8.3412 패리티1.00010) — M48이 개선을 노리는 baseline. Qwen RTN단독(+39.6%)보다는 낫지만 Qwen GPTQ+스무딩 배포수준(+8.8%)에는 못 미침.
- M48 실행 계획: `/Users/xox/.claude/plans/lively-greeting-sprout.md` (승인됨, 실행 중)
