---
name: vdsp-hiddenstate-signal-and-file-map
description: MoE 근접-동점(near-tie) 조사에서 검증된 hidden-state-magnitude 신호의 정확한 계산 레시피 + 이 트랙 전체에서 누적된 macstudio/bob/repo 파일 위치 지도
metadata:
  type: reference
  node_type: memory
  originSessionId: e6c100cc-beb0-426d-8425-0959ae41d7af
  modified: 2026-09-01T12:10:00.000Z
---

MoE 라우터 근접-동점(near-tie) 취약 레이어를 찾을 때 **cleanlab 대신
이미 검증된 hidden-state-magnitude 신호를 재사용**하기 위한 정확한
계산 방법과, 이 조사 전체에서 흩어진 파일들의 위치. [[prism_cleanlab_moe_insight]]
[[vdsp_general_serving_engine_goal]] 참고.

## D1: cleanlab 대신 hidden-state-magnitude 신호를 채택

**WHY**: 사용자 본인이 이 프로젝트에서 cleanlab식 지표(라우팅빈도
기반 중요도)를 실제 어블레이션과 직접 대조했다가 두 독립 모델에서
패배시킨 전례가 있음(OLMoE spearman 0.235 vs 0.141, gpt-oss-20b
0.001 vs -0.113로 부호역전 — `prism_cleanlab_moe_insight.md`).
반면 hidden-state-magnitude는 OLMoE에서 r=0.90(variance)/0.93(norm)
으로 강한 상관을 실측 확인했고, 이미 계산된 산출물이 존재함.

**COST**: 상관관계일 뿐 인과증명이 아님. 그리고 **DeepSeek-V2-Lite
포팅에서 이 메커니즘의 예측이 빗나감**(2026-09-01 확인) — OLMoE는
얕은 레이어가 위험(작은 hidden state → tight margin이라는 이 신호의
논리와 정합), 그런데 DeepSeek는 깊은 레이어가 위험 — 만약 DeepSeek
에도 이 hidden-state 프로파일러를 그대로 이식했다면 아마 얕은
레이어가 안전하다고 잘못 예측했을 것(직접 검증은 아직 안 함, 이건
추정). 즉 이 신호는 **아키텍처마다 재검증 없이 못 믿는다**.

**EXIT**: 새 아키텍처에 적용하려면 (1) 그 아키텍처의 MoE 블록
`__call__`(또는 게이트 전용 모듈의 `__call__`)을 후킹해 라우터
직전 hidden state를 가로채고, (2) 이미 측정된 margin_median과
Pearson r을 재계산해서, (3) r이 OLMoE 수준(|r|>0.8)으로 재현되는지
확인 후에만 신뢰할 것. 재현 안 되면(DeepSeek 가능성) 이 신호를
버리고 직접 어블레이션으로 돌아갈 것.

## 계산 레시피 (실제 스크립트 `moe_hiddenstate_diff_profiler.py` 기준)

1. `mlx_lm.utils.load()`로 모델 로드.
2. `block_to_layer = {id(layer.mlp): layer_idx for layer in model.model.layers}`
   — 각 MoE 블록(OLMoE 기준 `SparseMoeBlock`)의 파이썬 객체 id를
   레이어 인덱스에 매핑.
3. `BlockCls.__call__`을 몽키패치: 원본 호출 전에 입력 `x`(shape
   `(B,L,D)`, 이게 곧 라우터 게이트에 들어가는 hidden state)를
   가로채 `x_flat = x.reshape(-1, D)`로 펼치고 `mx.eval` 후
   `np.array(x_flat, copy=True)`로 numpy 복사, `layer_vecs[li]`에
   누적.
4. 코퍼스 전체(이 세션에선 두 30프롬프트 코퍼스 합쳐 60개)를
   forward — margin 측정용 forward를 새로 돌 필요 없이 이 패치
   자체가 margin profiler와 **완전히 동일한 후킹 지점**을 재사용.
5. 레이어별로 `V = concat(layer_vecs[li])` (shape `(N_tokens, D)`)에
   대해:
   - `var_total = V.var(axis=0).mean()` — 토큰 축 분산을 차원별로
     구한 뒤 차원 평균 (스칼라).
   - `norm_mean = mean(||v||_2 for v in V)` — 토큰별 L2 노름의 평균.
   - `cos_to_mean = mean(unit(v) · unit(mean(V)))` — 레이어 평균
     방향과의 코사인 유사도 평균 (참고용 — 이건 **반증된** "표현
     미분화" 가설을 위한 것, r=0.0125로 사실상 무상관 확인됨. 계산은
     같이 남겨두되 결론에 쓰지 말 것).
6. 별도로 이미 측정해둔 레이어별 margin median(두 코퍼스 JSON에서
   읽음, 재계산 안 함)과 Pearson r 상관:
   `pearson(var_vals, margin_median_vals)`,
   `pearson(norm_vals, margin_median_vals)`.
7. **OLMoE 실측 결과**: `r(var_total, margin_median) = 0.900`,
   `r(norm_mean, margin_median) = 0.932`, `r(cos_to_mean,
   margin_median) = 0.0125`(반증).
8. **해석 방향**: r이 양수라는 건 "hidden state가 클수록(=norm/var
   가 클수록) margin이 느슨(안전)"이라는 뜻 — 라우터 로짓이
   `gate_weight @ hidden_state`라 hidden state가 작으면 로짓 스프레드
   자체가 작아 softmax 후 근접값이 되는 메커니즘. 얕은 레이어는
   hidden state가 아직 덜 누적돼 작으므로 자동으로 위험해진다는
   설명 — **단, 이 설명은 OLMoE에서만 검증됨, DeepSeek에선 안 맞음
   (D1의 COST 참고)**.

## 이식 체크리스트 (다른 모델에 적용할 때)

- MoE 게이트 후킹 지점부터 확인: OLMoE는 `mlp.gate`가 맨 `nn.Linear`
  라 `SparseMoeBlock.__call__`을 패치하면 됐지만, DeepSeek-V2-Lite는
  게이트가 전용 `MoEGate` 모듈이라 `MoEGate.__call__`을 직접 패치
  해야 함(`deepseek_router_margin_profiler.py`가 이미 이 케이스의
  선례).
- margin JSON이 먼저 있어야 함 — 없으면 margin profiler류부터 실행.
- 상관계수가 낮거나(|r|<0.5) 부호가 반전되면 이 신호를 믿지 말고
  직접 레이어별 어블레이션으로 돌아갈 것(DeepSeek이 바로 이 케이스일
  가능성 — 아직 실제로 이식해서 확인은 안 함, 추정만 있음).

## 파일 지도 — macstudio (`eoe@macstudio`, `~/.claude/skills/studio/studio.sh`로 접근)

스크립트(전부 `/Users/eoe/` 바로 밑):
- `moe_router_margin_profiler.py` / `_v2.py` — OLMoE margin 최초측정
  + 교차코퍼스 재현검증용, 각 30프롬프트.
- `moe_hiddenstate_diff_profiler.py` — 이 메모리의 계산 레시피 원본.
- `moe_precision_sweep.py` / `moe_precision_sweep_output.py` — Track B
  비트폭{4,8,16,32} 스윕, router-flip / output-flip 각각.
- `moe_precision_sweep_extra_layers.py` — 6레이어 확장 스윕(fork작성).
- `moe_local_vs_upstream.py` — local-vs-upstream 원인귀속(fork작성).
- `moe_wikitext_neartie_scan.py` — WikiText-103 대규모 재현빈도 스캔.
- `deepseek_router_margin_profiler.py` — DeepSeek-V2-Lite 포팅판.
- `mlx_deepseek_to_q4g64af.py` — DeepSeek C엔진 export용(구버전 작업).

산출물 JSON:
- `/Users/eoe/vdsp_olmoe_full_weights/moe_router_margin.json`,
  `moe_router_margin_v2.json`, `moe_hiddenstate_diff.json`,
  `moe_precision_sweep*.json`, `moe_local_vs_upstream.json`,
  `moe_wikitext_neartie.json` — OLMoE 관련 전부 이 디렉토리 하나에
  모여 있음.
- `/Users/eoe/deepseek_router_margin.json` — **주의: DeepSeek 산출물은
  `vdsp_olmoe_full_weights/` 밑이 아니라 홈 바로 밑**(경로 패턴이
  다름).

가상환경: `mlx_venv`(`source ~/mlx_venv/bin/activate`) — DeepSeek
다운로드 로그에서 확인된 실제 사용 venv, 다른 venv(`moe_st_venv` 등)
와 혼동 주의.

## 파일 지도 — bob (`ssh bob`, C엔진 실측 전용)

- `/Users/bob/olmoe_1b7b_hf` — OLMoE HF 체크포인트
  (`QWEN_MOE_SAFETENSORS=` 이 경로 + `model.safetensors.index.json`).
- `/tmp/qwen_f16tier_bin` — 16비트 티어 코드가 반영된 dense-only
  링크 바이너리(Track A/블랭킷승격 검증에 사용한 바로 그 바이너리).
- `/tmp/role_bits_16_all.txt` — 블랭킷 승격 config 4줄
  (`q_proj -1 16` / `k_proj -1 16` / `v_proj -1 16` / `o_proj -1 16`,
  `-1`은 전 레이어 wildcard).
- `/Users/bob/vdsp_m4_bench/moe3a_mlx_ref_logits.bin` — **주의: 이건
  13-position 코퍼스용 참조파일, 8-position 기본 코퍼스 스팟체크와
  안 맞음**(블랭킷승격 최종검증에서 이 불일치 때문에 방향성 검증
  미결로 남음 — RESULTS.md Synthesis 섹션 참고).

## repo 내 문서 위치 (`/Users/xox/vdsp-engine/`)

`RESULTS.md` 섹션 제목으로 검색(전부 이 파일 하나에 순서대로 있음):
"Router near-tie (margin) statistics profiler" →
"cross-corpus replication" → "Root-cause probe: why are OLMoE's
early layers (1-4) structurally tighter?" → "D-roadmap-2 Track A" →
"D-roadmap-2 Track B" → "(a) Output-token causal validation" →
"(b) Expert/FFN precision sweep" → "D-roadmap-2 Track B extension" →
"D-roadmap-3 residual-accumulation test" → "WikiText-103-scale
near-tie recurrence scan" → "Synthesis: blanket all-layer attention
promotion" → "DeepSeek-V2-Lite port of the router-margin profiler".

`ROADMAP.md`: `D-roadmap-2`(정밀도 캘리브레이션), `D-roadmap-3`
(런타임 승격, 설계만·미구현).

`qwen_infer.c` 16비트 티어 관련(라인번호는 이후 편집으로 약간
밀렸을 수 있음, 함수명으로 검색 권장):
`st_register_moe_f16_as_af()`, `moe_decode_af()`의 `bits0==16` 분기,
`moe_matvec_af_row()`의 `bits==16` 분기,
`moe_matvec_af_row_vdsp()`의 `bits==16` FATAL 가드(SME2 미지원),
`st_register_moe_role()`의 `bits==16` 분기, `moe_load_role_bits()`/
`QWEN_MOE_EXPERT_BITS` 파서의 `bits!=16` FATAL 확장,
`st_register_moe_experts_mixed_as()`의 `b==16` row_pbytes 분기.
**아직 커밋 안 됨** — 사용자 명시요청 대기.

메인 메모리: [[vdsp_general_serving_engine_goal]] (가장 자세한
서사, ★★★★★★ 마킹된 종합 엔트리들). 이 세션 전체 타임라인:
`.claude/history/2026-08-31_v5c-fused-throughput-optimization.md`.
