---
name: vdsp-m46-python-pipeline-and-d50-incident
description: vdsp M46(Python export/quantize/layout 파이프라인 일반화) + M47(Llama-3.1-8B 실체크포인트 RTN ppl 측정, 블로커5 Phase1) 완료. 세션 중 D50 실제 데이터 덮어쓰기 사고와 복구 포함. repo popixoxipop-collab/vdsp
metadata: 
  node_type: memory
  type: project
  originSessionId: 6cb2cc15-42f7-402b-9e6d-4bf8864d1488
  modified: 2026-08-13T08:18:25.562Z
---

## M47 완료 (2026-08-13, commit `939ad73` on `main`, origin push 확인됨 — `git ls-remote`로 대조)

블로커(5) Phase 1: eoe Mac Studio에 SSH로 실제 `meta-llama/Llama-3.1-8B-Instruct`(Instruct, 사용자 확인) 다운로드 → M46 파이프라인(export/layout/quantize GROUP=64 RTN/arch_config) 실행 → `qwen_infer dump`로 fp32/int4 양쪽 실디스패치 검증(GROUP=4 fast path, untied lm_head→int8, llama3 RoPE 전부 정상) → WikiText-103 val ppl 3종 측정.

**결과**: HF fp32(정답) 8.3404 / engine fp32 8.3412(**비율 1.00010, 사실상 완벽한 패리티** — 블로커1~4가 실제 모델에서도 수치적으로 정확함을 첫 실증) / engine RTN-int4 9.8169(**+17.70% vs fp32**). Qwen2.5-1.5B의 RTN단독 결과(+39.6%)보다 훨씬 낫지만(큰 모델이 RTN에 더 강건 — 일반론과 일치), Qwen의 GPTQ+SmoothQuant 배포본(+8.8%)에는 못 미침.

**사용자 결정(2026-08-13)**: +17.70%로 **여기서 멈춤**. Phase 2(GPTQ+SmoothQuant를 Llama용으로 일반화 — GQA 32:8 풀링 재검증+Mac Studio 캘리브레이션 시간 미지수라 꽤 큰 작업)는 **시작 안 함**. 블로커(5)는 "구조적으로 답할 수 없음"에서 "실측했고 RTN으로 충분하다고 판단, GPTQ는 보류"로 상태 전환.

**세션 중 발견한 별도 버그+수정**: `eval_ppl_llama.py`가 unbuffered여도 내부 `subprocess.run(capture_output=True)`로 qwen_infer를 감싸서 47분간 진행률 블랙박스(자식 프로세스 stdout이 부모 종료까지 버퍼링). `Popen`+라인스트리밍으로 수정+재실행. **일반화**: `~/.claude/hooks/scripts/experiment-progress-guard.py`에 체크 E 추가(스크립트 내부 subprocess capture 패턴 탐지) — 전역 hook이라 이 repo 밖에서도 적용됨.

**격리 원칙**: 모든 Llama 아티팩트는 `/Volumes/D50/vdsp/llm_engine_llama31_8b/`(Qwen 프로덕션 `llm_engine/weights/`와 완전 분리, `.gitignore` 추가)에 격리 — M46 D50 덮어쓰기 사고 재발 구조적 방지.

산출물: `llm_engine/eval/eval_ppl_llama.py`(신규), `results/RESULTS_LLAMA31_8B.md`(신규, M47 전체 상세), `results/RESULTS_QWEN_VDSP.md`(포인터 +9줄), `.gitignore`(+2줄). `.c`/`.h` 변경 0.

## M46 완료 (2026-08-12, commit `05c729d` on branch `feature/vdsp-m46-export-pipeline`, main@`8256575` 기준). **★2026-08-13 main에 fast-forward push 완료** — GitHub origin/main = `05c729d` 확인됨(`git ls-remote` 직접 대조).

M41-M45가 C엔진(`qwen_infer.c`)을 구조적으로 일반화(런타임 sidecar)했고 M44/M45가 GROUP=4 NEON 커널을 실제로 검증완료. M46은 그 나머지 블로커(4) — HF 체크포인트에서 C엔진이 읽는 아티팩트를 만드는 *Python* 툴링이 여전히 Qwen 하드코딩 — 을 닫음. **스코프: Python 전용, `.c`/`.h` 변경 0. `qwen_infer.c`는 이미 검증된 오라클로만 사용, 수정 안 함.**

**대상 정확히 4개 파일** (export/quantize/layout 관련 11개 스크립트 중, 나머지 7개는 GPTQ/g256/fold-g64 계열 bit-exact ppl-anchor 하드게이트 보유 = 블로커(5) 영역, 이번 스코프 아님):
- `export_weights.py`: sharded-safetensors(`model.safetensors.index.json`) 지원 — `contextlib.ExitStack`로 샤드핸들 동시오픈, single-file 모드는 동일 코드경로의 자명한 1-샤드 케이스로 환원(같은 mmap파일 한번더 여는 것만 다름, 의도적 disclosure). `VDSP_BASE`+신규 위치인자 `OUT_BIN`/`OUT_MANIFEST`.
- `make_layout.py`: `VDSP_BASE`+위치인자 `W`/`R`. `prompt_ids.npy` 이제 optional(없으면 크래시 대신 graceful skip).
- `quantize_int4.py`: `VDSP_BASE`만(신규 위치인자 없음, argv[0]이 이미 GROUP).
- `emit_arch_config.py`: 신규 `check_layout_matches_config()` 가드 — LAYOUT의 q_proj in-dim과 CONFIG_SOURCE의 hidden_size 불일치시 stderr 경고(FATAL 절대 아님, 이 스크립트엔 FATAL 전례 없음).

## 검증 결과 요약 (전체 상세: `results/RESULTS_QWEN_VDSP.md` M46섹션)

1. D50 실데이터 대비 zero-regression: 텍스트 산출물(layout.txt/layout_int4.txt/arch_config.txt) 전부 byte-identical. int4 바이너리 블롭만 D50의 7월 파일과 한 바이트(933500985)부터 다른데, **pristine main@`82222d7`(D50 파일을 실제로 만든 그 커밋) 컨트롤런이 정확히 같은 바이트에서 정확히 같은 방식으로 다름** → M46 회귀 아니라 순수 환경드리프트(오늘 numpy 2.2.6/torch 2.12.0 vs 7월 당시 설치본, 코드결정론은 별도로 확인함: 내 코드 두 번 독립실행이 byte-identical).
2. single-file 브랜치: 코드동치성 리뷰+실제 스모크테스트(정확한 값 라운드트립)+FileNotFoundError negative case 확인.
3. 신규가드 positive/negative 둘 다 확인, make_layout skip-path 확인.
4. **헤드라인**: M44/M45가 이미 검증한 GROUP=4 shape(`NH=8,NKV=2,D=1024,IM=64,VOCAB=256,NL=2`)를 실제 HF 포맷(bf16, 2-shard, untied lm_head, 실제 Llama-3.1 rope_scaling)으로 손수 구성해서 **이 프로젝트 역사상 최초로 Python 툴링 전체 파이프라인을 통과시켜 실제 GROUP=4 NEON 디스패치**(`qwen_infer dump` → exit 0, "GROUP=4 family" 확인, 256/256 finite logits)까지 검증.
5. 권장 stretch leg 3/4 클린(greedy/ppl/rope override), int4 leg는 진짜 사전존재 한계 발견 후 미수정(스코프 밖): `q4gemv.h`의 `q4pool`이 `IM`을 항상 최대 in-차원으로 가정(실제 Qwen/Llama는 항상 IM>D라 문제없지만 이 합성모델은 M44/45 attention-전용 shape을 그대로 재사용해 IM=64<D=1024).

## ★★★★★ 세션 중 사고: D50 실제 프로덕션 파일 덮어씀 (완전 복구됨)

Verify 4 준비 중 `export_weights.py`를 합성 스냅샷에 대해 실행하면서 **`VDSP_BASE` 지정을 빠뜨림** → 하드코딩된 기본경로(`/Volumes/D50/vdsp/llm_engine/weights/`)로 폴백 → 실제 `qwen15b_fp32.bin`(6.17GB, 진짜 Qwen2.5-1.5B 가중치)과 `manifest.json`이 25MB 합성 테스트 blob으로 덮어써짐. 로컬/Time Machine 백업 없음, WD_BLACK(2026-07-30 아카이브 기록 위치)에도 없음. **복구**: 공개모델이라 원본 HF 리비전(`mlx-community/Qwen2.5-1.5B-Instruct-bf16`@`4ae77cb2...`, 스크립트 자체의 하드코딩된 fallback과 동일 커밋해시)을 재다운로드 후 **수정 안 한 pristine main의 export_weights.py**로 재생성. **3중 독립검증**(단순 "크기 맞음"에 안 멈춤): ①복원된 manifest.json에서 재도출한 layout.txt가 D50의 원본 layout.txt와 byte-identical, ②파일크기 6174857216 정확히 일치, ③가장 결정적 — 복원파일을 재양자화하면 위 검증1에서 이미 확인한 환경드리프트 fingerprint와 **정확히 같은 바이트**에서만 다름(실제 가중치 값에 의존하는 결과라 조금이라도 잘못 복원됐으면 다른 위치에서 갈렸을 것).

**일반화 가능한 교훈**: [[feedback_explicit_output_path_near_real_data]] 참조.

## 남은 작업 / 미결정
- ~~`feature/vdsp-m46-export-pipeline` 브랜치는 아직 main에 merge도 origin push도 안 함~~ → **2026-08-13 완료**(`git push origin feature/vdsp-m46-export-pipeline:main`, FF, 로컬 `main` ref도 동기화). 브랜치 자체는 삭제 안 하고 유지(이 repo의 기존 관행 — `feature/vdsp-lowbit-kv-g4`도 병합 후 그대로 남아있는 패턴을 그대로 따름).
- ~~블로커(5) "per-model quantization recalibration"는 여전히 미착수~~ → **2026-08-13 M47로 RTN 실측 완료**(위 섹션 참조, +17.70%). → **같은 날 M48로 이어서 착수**: GPTQ+SmoothQuant를 Llama용으로 일반화. ★GQA 풀링은 재검증만 필요, 재작성 불필요로 정정됨 — 상세는 [[vdsp_gptq_smoothquant_math_and_status]] 참조. 진행상황은 그 메모리에서 추적.
- **작업 위치**: 블로커(5) 이후 작업은 로컬(M1 Max)이 아니라 **SSH로 eoe Mac Studio에서**(`/studio` 스킬 또는 `~/.claude/skills/studio/studio.sh`, Tailscale 경유) — D50 볼륨이 거기 물려있음. Mac Studio는 ML 스택 처음부터 없었음(torch/transformers/datasets 없음, system python 3.9.6, brew 없음) → `.venv_llama31_8b/`(격리 venv, torch 2.8.0 MPS) 세팅 완료, 재사용 가능. ★eoe Mac Studio에는 `brain.selfplay`/`alpha_predictor_7b` 상시가동 프로세스가 CPU 코어 점유 중 — 무거운 작업 전 `ps aux` 확인 필수([[feedback_studio_resident_cpu_jobs]] 참조).
- 로컬 repo(`/Users/xox/vdsp_local`)와 Mac Studio 측(`/Volumes/D50/vdsp`) mirror가 다를 수 있으니 **commit hash로만 신뢰**, 로컬 요약을 그대로 최신이라 가정 금지.
- 새로 발견했지만 `.c` 파일이라 스코프 밖이라 안 고친 것 2건: `q4pool` max_in=IM 가정, untied-lm_head 체크포인트의 `layout_int4.txt` 중복엔트리(`lm_head.weight`가 q4g64+q8g64 둘 다로 방출되고 `wt()` 선형탐색이 먼저 오는 쪽만 사용).
