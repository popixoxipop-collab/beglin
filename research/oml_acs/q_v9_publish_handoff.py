#!/usr/bin/env python3
"""Publish source-verified Qv9 GGN scientific handoff after all 5 native arms."""
import hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[2]
D=R/"docs/research";C=R/"configs/research"
LOCK=C/"BEGLIN_QV9_GGN_SOURCE_PRELOCK_20261023.json"
INPUT=D/"BEGLIN_QV9_INPUT_PREMODEL_FREEZE_20261023.json"
TRAIN=D/"BEGLIN_QV9_TRAIN24_CONDITIONAL_GGN_PUBLIC_20261023.json"
DIAG=D/"BEGLIN_QV9_DIAGNOSTIC24_CONDITIONAL_GGN_PUBLIC_20261023.json"
POL=D/"BEGLIN_QV9_ROBUST_CONDITIONAL_GGN_SOURCE_POLICY_20261023.json"
SOUR=D/"BEGLIN_QV9_INDEPENDENT_OUTPUT_GGN_TRAIN_DIAGNOSTIC_20261023.json"
REF=D/"BEGLIN_QV9_ORIGINAL_BF16_MMLU48_PUBLIC_20261023.json"
NAT=D/"BEGLIN_QV9_MMLU48_NATIVE_FIVE_ARM_INDEPENDENT_PUBLIC_20261023.json"
MASK=C/"BEGLIN_QV9_GGN_28N8_FROZEN_MASK_20261023.txt"
OUT=D/"BEGLIN_QV9_OUTPUT_GGN_ROBUST_MMLU48_HANDOFF_20261023.md"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if OUT.exists():raise FileExistsError("no overwrite Qv9 final scientific handoff")
 for p in (LOCK,INPUT,TRAIN,DIAG,POL,SOUR,REF,NAT,MASK):
  if not p.is_file():raise RuntimeError("missing original Qv9 output-GGN/fullmodel audit "+str(p))
 obj=[json.loads(p.read_text()) for p in (LOCK,INPUT,TRAIN,DIAG,POL,SOUR,REF,NAT)]
 protocol,inputs,cal,diagnostic,policy,proxy,reference,native=obj
 if native["status"]!="PASS_SOURCE_AND_ORIGINAL_MMLU48_FIVE_NATIVE_POLICY_AUDIT" or \
    policy["mask_sha256"]!=sha(MASK) or not proxy["only_source24_train_selects_mask"] or \
    native["generalization_established"] or native["auto_promotion"]:
  raise RuntimeError("not native source-only audited research")
 summary=native["native_summary"]
 if set(summary)!={"q4","qv6_64","qv6_128","qv8_joint96","qv9_ggn24"}:raise RuntimeError("not 5 real fullmodel native comparisons")
 rows=[]
 for name,title in (("q4","전체 Q4"),("qv6_64","기존 Fisher64"),("qv6_128","기존 Fisher128"),
                    ("qv8_joint96","기존 signed96"),("qv9_ggn24","Qv9 출력 GGN24")):
  x=summary[name]
  rows.append(f"| {title} | {x['correct']}/48 | {100*x['correct']/48:.1f}% | {x['mean_fullvocab_correct_nexttoken_NLL']:.6f} |")
 comparison=[]
 for mode,obs in native["paired_comparisons"].items():
  ci=obs["paired_bootstrap95_percent"]
  comparison.append(f"| {mode} | {obs['GGN_minus_baseline_mean_fullvocab_NLL']:+.6f} | "
       f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | {obs['accuracy_difference']:+d} | "
       f"{obs['McNemar_exact_p']:.4f} |")
 proxies=[]
 for key in ("q4","fisher64","fisher128","signed96","ggn24"):
  x=proxy["source_proxy_vs_diagnostic_metrics"][key]
  proxies.append(f"| {key} | {x['train']['robust_GGN']:.6f} | {x['diagnostic']['robust_GGN']:.6f} |")
 bestname=min(summary,key=lambda k:summary[k]["mean_fullvocab_correct_nexttoken_NLL"])
 bq4=summary["q4"];new=summary["qv9_ggn24"]
 verdict=("이번 48개 개발 문항에서 Qv9은 Q4보다 정답 NLL이 낮지만 독립 모델군 우위는 증명하지 못함"
   if new["mean_fullvocab_correct_nexttoken_NLL"]<bq4["mean_fullvocab_correct_nexttoken_NLL"]
   else "이번 48개 개발 문항에서 Qv9은 전체 Q4의 정답 NLL을 이기지 못함")
 txt=f"""# BEGLIN Qv9 — 4선택지 출력 Gauss–Newton 곡률, 보정 과적합 진단, MMLU48 실기기 검증

**연구일:** 2026-10-23 · **분리 브랜치:** \`research/qv9-ggn-robust-20261023\` · main·운영 릴리스 변경/자동 승격 **없음**.

## A. 앞선 Qv8 음성 근거와 목적함수 수정

기존 Qv8은 전체 모델 96개 보정 정답에 관한 signed first-order 손실변화 제곱을 원본 소스에서 크게 낮췄지만 HellaSwag 48개 전체 어휘 정답 NLL에서 Qv6 Fisher64를 이기지 못했다. Qv7 TinyLlama 등 교차 모델에서는 최종 품질 우위가 오히려 부정됐다. Qv9은 이를 일반화하지 않고 원래의 BEGLIN Qwen2.5-1.5B 모델의 **조건부 네 선택지 출력 확률**에 관한 Gauss–Newton 곡률을 측정한다.

Qwen 원본 BF16 학습 소스에서 한 개의 4선택지 프롬프트당 BF16 forward 1회와 상대 logit(각 A−D,B−D,C−D) 역전파 **3회**를 실행해 실제 native C Q4G64/n8 오차와 내적했다. 이 연산은 원본 BF16 모델만 읽고, 단일 텐서를 실제 C 전체 모델 출력에 삽입하거나 모델 가중치를 수정하지 않는다.

\(u_{{s,k}}(x)=\sum_{{i=1}}^{{196}} \nabla_{{W_i}}(z_{{s,k}}-z_{{s,D}})^\top \delta W_i(x)\), 선택지 \(k∈\{{A,B,C}}\), \(u_{{s,D}}=0\).

\(G_{{s}}(x)=\frac12[\sum_k p_{{s,k}} u_{{s,k}}^2-(\sum_k p_{{s,k}}u_{{s,k}})^2]\), \(\ p=\mathrm{{softmax}}(z_A,z_B,z_C,z_D)\).

이 식은 4선택지 조건부 출력 분포의 **실제 원본 로짓 Jacobian에 기반한 GGN PSD 대리값**이다. 전체 어휘 152K 로짓에 대한 정확한 Hessian이나 실제 native 모델 NLL이 아니다. \(\delta W\)가 큰 경우 선형 출력 근사가 무너질 수 있고, 조건부 네 선택지 분포와 전체 어휘 NLL은 다르다.

사전 고정 목적함수는 학습 24개 홀짝 12문항 중 **악화된 반쪽의 평균 GGN**에 0.65, 서로 독립 텐서의 isolated GGN 합계에 0.35를 배정한다. 이는 무제한 tensor 간 상쇄에 지나치게 의존하는 선택을 제한하려는 고정 규칙이다. 계수는 진단24나 평가48 결과에 맞춰 조정하지 않았다.

## B. 원본 사전등록과 분리된 데이터

- 원본 모델 HF revision: \`989aa7980e4cf806f80c7fef2b1adb7bc71aa306\`.
- source: ARC-Challenge train 원본 SHA 순서 **128번째 이후 새 48개** = 학습24/진단24. 기존 Qv6 첫128과 완전 동일한 정렬·토큰 출처를 먼저 검증.
- 새 비교: \`cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe\` test, psychology 24 + computer science 24, 총48개. 선택 당시 모델 성능은 사용하지 않았고 전체 문항·정답·토큰 SHA를 고정.
- MMLU public test는 기존 모델의 사전학습에 등장했을 수도 있으므로 원칙적으로 **확증용 독립 최종 holdout이 아닌 외부 개발 평가**다.
- source24 original BF16 72 상대 출력 역전파; diagnostic24 원본 BF16 72 추가 역전파.
- 사전등록 SHA: \`{sha(LOCK)}\`, 원본 입력 잠금 집계 SHA: \`{sha(INPUT)}\`.

## C. 학습 지표와 별도 원본 진단 24 — 과적합 탐지

| 정책 | source train24 robust conditional GGN ↓ | frozen diagnostic24 ↓ |
|---|---:|---:|
{chr(10).join(proxies)}

학습 최소치는 Qv9이지만 독립 진단의 최솟값은 **{min(proxy['source_proxy_vs_diagnostic_metrics'],key=lambda k:proxy['source_proxy_vs_diagnostic_metrics'][k]['diagnostic']['robust_GGN'])}**. 이 자체로 조건부 곡률 근사의 근거와 소스 과적합 가능성이 확인됐고, 학습 결과를 실제 성능으로 잘못 표시하지 않았다.

Qv9 selected policy는 **정확히 28 n8 / 168 Q4**를 유지하고, 원래의 Apple NEON 실제 커널·메모리 예측 비용을 재활용한 엄격한 실행 가능한 1-swap 국소 탐색이다. **전역 binary GGN 최적화의 수학적 증거가 없다.**

- 마스크 SHA: \`{policy['mask_sha256']}\`.
- 투영 가중치 저장량: **{policy['final_Apple_native_source_costs']['memory_bytes']:,}B** (전체 Q4 736,985,088B).
- 원래 비용 기반 순차 모델 토큰 지연 예측 {policy['final_Apple_native_source_costs']['serial_estimate_ms']:.4f}ms, 독립 단계 비용 {policy['final_Apple_native_source_costs']['isolated_async_stage_ms']:.4f}ms. **실측 지속 TPS나 p95가 아니다.**

## D. 실제 BEGLIN Qwen 28계층 48×5 native CPU 네이티브 평가

동일한 48 MMLU 프롬프트, 원본 BF16 Qwen 모델 가중치, 동일한 Q4 원본 native C 바이너리 하나와 나머지 네 혼합 정책에 공통된 n8 원본 C 바이너리 하나, 동일한 정답 토큰. 정확도는 네 후보 A/B/C/D argmax, NLL은 **전체 어휘에 대한 정답 토큰 softmax 확률**이다.

독립 원본 Transformers BF16 / Apple MPS: **{reference['correct']}/48 정답, 평균 정답 NLL {reference['mean_correct_answer_full_vocab_NLL']:.6f}**. 기존 C의 widened-F32 Q4 또는 n8과 동일 수치 실행이라는 뜻이 아니다.

| 정책 | 맞힌 문항 | 정답률 | 전체 어휘 정답 한 글자 NLL ↓ |
|---|---:|---:|---:|
{chr(10).join(rows)}

이 데이터셋에서 가장 낮은 네이티브 평균 NLL: **{bestname}**. Qv9 − Q4 평균 NLL **{new['mean_fullvocab_correct_nexttoken_NLL']-bq4['mean_fullvocab_correct_nexttoken_NLL']:+.6f}**. 판정: {verdict}.

### 쌍별 탐색적 불확실성

각 문항에 대한 Qv9−기준 정책 NLL을 5,000회 동일 문항 재표집했다. 음수면 Qv9이 낮은 NLL, 양수면 기준이 낫다. 정확도는 정확 McNemar p. **여러 정책 비교의 다중 검정 보정 없이 범용 우위를 주장하지 않는다.**

| 비교 기준 | 평균 NLL 차이 (Qv9−기준) | 95% bootstrap 구간 | 정확도 변화 | McNemar p |
|---|---:|---:|---:|---:|
{chr(10).join(comparison)}

## E. 재현 증거와 안전 제한

- 원본 BF16 conditional choice Jacobian24·진단24 독립 감사 \`{sha(SOUR)}\`.
- source-only 출력 곡률 정책 \`{sha(POL)}\`.
- 원본 full Qwen native MMLU48 5조건 독립 로그/토큰/바이너리 감사 \`{sha(NAT)}\`.
- BF16 original external arithmetic reference \`{sha(REF)}\`.
- owner-only \`lab_local/\`에 원본 model forward/Jacobian/dot, 후보 토큰, 전체 질문, 5×48 네이티브 로그 원문을 보관. GitHub에는 코드·정책·공개 aggregate만 업로드.
- 요청별 동적 n-bit 변경, Metal GPU·CPU 자동 분배, 근거 없는 CPU 속도 개선, main 병합, immutable Tailnet 운영 release 변경 및 live 자동 승격 **미구현/차단**.
- 본 실험의 주장은 output choice curvature로 국한한다. 선행 mixed precision 양자화에서 Hessian과 logits alignment를 다룬 APTQ (arXiv 2402.14866), LFQ (ICML 2026), KronQ (arXiv 2607.07964), G²PTQ (arXiv 2609.31009) 등은 관련 선행기술이며, Qv9의 아이디어가 최초라는 주장은 하지 않는다.

## F. 연구 다음 게이트

1. 소스24·진단24·MMLU48에서의 차이를 분석해 **GGN surrogate-to-native error calibration**을 수치적으로 검증. 단, 동일 평가48에 맞춰 목적함수를 수정한 뒤 승리했다고 주장하지 않는다.
2. 원본 logits top-k+other 분포를 보존하는 보다 정확한 출력 곡률 근사로 개선한 뒤, 별도 모델 계열·새 데이터셋을 미리 봉인해 평가할 것.
3. 원본 Apple CPU 저부하 ABBA p50/p95/tps/RSS/전력을 외부 잡 부하 기준 통과에서 재평가. 승격은 그 이후.
4. 모든 과거 Qv5–Qv8 긍정·부정 결과와 source SHA를 보존하고 후행 발견을 덮어쓰지 않는다.

**최종 판정:** OUTPUT_CHOICE_GGN_MEASUREMENT_PASS / SOURCE_ONLY_MASK_PASS / INDEPENDENT_DIAGNOSTIC_PASS / NATIVE_MMLU_5ARM_PASS / GENERALIZATION_NOT_PROVEN / NO_LIVE_AUTO_PROMOTION.
"""
 data=txt.encode()
 with OUT.open("xb") as f:f.write(data);f.flush()
 print("QV9_FULL_RESEARCH_HANDOFF_SEALED",sha(OUT),"bytes",len(data),"best_MMLU_NLL",bestname)
if __name__=="__main__":main()
