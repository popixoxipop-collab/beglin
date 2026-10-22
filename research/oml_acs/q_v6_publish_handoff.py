#!/usr/bin/env python3
"""Publish research-only Qv6 24/64/128 source + four native-policy handoff."""
import hashlib,json,os,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
D=ROOT/"docs/research"
S=D/"BEGLIN_QV6_FISHER_SCALING_196_SOURCE_AGGREGATE_20261022.json"
P=D/"BEGLIN_QV6_NESTED_64_128_SOURCE_POLICY_20261022.json"
A=D/"BEGLIN_QV6_CROSSCORPUS_NATIVE_INDEPENDENT_PUBLIC_20261022.json"
B=D/"BEGLIN_QV6_ORIGINAL_BF16_OPENBOOKQA48_REFERENCE_20261022.json"
OUT=D/"BEGLIN_QV6_FISHER_SAMPLE_SCALING_CROSSCORPUS_HANDOFF_20261022.md"
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    if OUT.exists():raise FileExistsError("Qv6 handoff already sealed; never override")
    src,plan,rep,bf16=[json.loads(p.read_text()) for p in (S,P,A,B)]
    if rep["status"]!="PASS_4_POLICY_48_REAL_APPLE_C_NATIVE_EXECUTIONS_AND_BF16_SOURCE_AUDIT" or \
       plan["status"]!="FROZEN_64_128_POLICIES_BEFORE_NEW_OPENBOOKQA_VALIDATION" or \
       rep["generalization_quality_superiority_proven"] or rep["automated_live_promotion"] or \
       not plan["evaluation_48_OpenBookQA_logits_not_opened"]:
        raise RuntimeError("not independently audited source-only Qv6 quality experiment")
    m=rep["native_summary"]
    if set(m)!={"q4","v5_fisher_24","v6_fisher_64","v6_fisher_128"} or any(not (0<=v["correct"]<=48) for v in m.values()):
        raise RuntimeError("not all four actual Qwen policies evaluated")
    baseline=m["q4"]["correct_answer_NLL"]
    rows=[]
    rows.append(f"| 원본 공식 BF16 (Torch MPS) | {bf16['correct']}/48 | {bf16['original_BF16_correct_oneletter_NLL']:.6f} | Original model arithmetic, not BEGLIN C |")
    for key,title in (("q4","전체 Q4G64"),("v5_fisher_24","Fisher 소스 24"),("v6_fisher_64","Fisher 소스 64"),("v6_fisher_128","Fisher 소스 128")):
        d=m[key]
        changed=("0.00%" if key=="q4" else f"{100*(d['correct_answer_NLL']/baseline-1):+.2f}%")
        rows.append(f"| {title} | {d['correct']}/48 | {d['correct_answer_NLL']:.6f} | {changed} |")
    rho=src["nested_split_stats"]
    overlaps=plan["constrained_selected_overlap"]
    rhos=" | ".join(f"{n}: {rho[str(n)]['odd_even_sample_Fisher_rank_spearman']:.3f}" for n in (24,64,128))
    diffs=rep["paired_q4_comparisons"]
    diagnostics=[]
    for key in ("v5_fisher_24","v6_fisher_64","v6_fisher_128"):
        v=diffs[key]
        lo,hi=v["paired_bootstrap_95_percentile"]
        diagnostics.append(f"| {key} | {v['candidate_minus_Q4_NLL_mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {v['candidate_minus_Q4_correct_count']:+d} |")
    report=f"""# BEGLIN Qv6 — Original BF16 Fisher 보정 표본 24→64→128 및 교차 데이터셋 검증

**날짜:** 2026-10-22 · **연구 브랜치:** \`research/qv6-fisher-sample-scaling-20261022\` · **main·운영 자동승격 변경 없음**.

## 연구 질문과 분리된 판정

Qv5에서는 24개 ARC-Challenge 학습 문제에 대해 공식 Qwen2.5-1.5B-Instruct BF16 모델을 역전파하고, 196개 투영의 원본 BEGLIN C Q4G64 및 qNg64-n8 오차 방향과 내적해 Fisher 유사 대리지표를 만들었다. 40개 같은 ARC 검증 문제에서는 전체 Q4의 0.9808 NLL을 이기지 못했다.

이번 Qv6의 **A 실험**은 원본 BF16 그래디언트 민감도 추정의 **보정 표본 수**를 24→64→128로 늘릴 때 텐서 순위·실제 할당이 안정되는지를 판정한다. **B 실험**은 정책을 원본 자료만으로 SHA 동결한 후, 별도 \`OpenBookQA\` 검증 48개를 실제 BEGLIN 네이티브 C 엔진에서 평가해 **답변 정확도와 정답 글자 NLL**을 비교한다. 두 결과는 독립 판정이다.

- 공식 모델 BF16 소스: \`Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306\`.
- 원본 소스 자료: \`allenai/ai2_arc\` revision \`210d026faf9955653af8916fad021475a3f00453\` 의 train 128건, 첫 24건 Qv5 자료 **토큰 배열/문항 SHA 완전 동일**.
- 새 검증 자료: \`allenai/openbookqa\` revision \`388097ea7776314e93a529163e0fea805b8a6454\` validation 48건; 학습·이전 개발 문항과 동일 텍스트 없음. **둘 다 과학 중심이라 엄격한 장르 OOD로 주장하지 않음.**
- 보정 표본과 평가 입력은 모두 새 그래디언트·네이티브 정책 실행 전에 잠금. Qv5 기존 24건 원본 per-tensor gradient²를 변경하지 않고 재사용하고, **104개의 새 source BF16 backward 실행**.
- 기존 Qv1~5의 부정적·미입증 결과와 V6 소진 24개 불명확 ID는 보존하고 열지 않음.

## A. 그래디언트 Fisher 순위 안정성

하드웨어 원본 C 양자화 코드의 **196/196 코드 SHA와 BF16 원본 가중치 SHA를 재확인**한 뒤 24/64/128 문항별 경험적 방향성 Fisher Q4→n8 이득을 계산했다. 이는 원본 모델에서 \`E[(grad·(Q_b(W)-W))²]\`의 합산 대리지표이며 Hessian, 실제 전체 모델 NLL이나 인과 단일 텐서 교체 평가가 아니다. BF16 MPS에서 도함수를 산출하고 부동소수 오차가 존재한다.

| 보정 문항 | 홀수/짝수 텐서 민감도 순위 Spearman ρ |
|---|---:|
| 24 | {rho['24']['odd_even_sample_Fisher_rank_spearman']:.6f} |
| 64 | {rho['64']['odd_even_sample_Fisher_rank_spearman']:.6f} |
| 128 | {rho['128']['odd_even_sample_Fisher_rank_spearman']:.6f} |

| 보정 규모 비교 | 196개 순위 Spearman | MILP로 선택한 8비트 텐서 28개 중 공통 |
|---|---:|---:|
| 24 vs 64 | {overlaps['24_64']['source_gradient_rank_spearman']:.6f} | {overlaps['24_64']['policy_28_overlap']}/28 |
| 24 vs 128 | {overlaps['24_128']['source_gradient_rank_spearman']:.6f} | {overlaps['24_128']['policy_28_overlap']}/28 |
| 64 vs 128 | {overlaps['64_128']['source_gradient_rank_spearman']:.6f} | {overlaps['64_128']['policy_28_overlap']}/28 |

이 실측에서는 규모 증가에 따라 분할 안정성이 개선되지만, **24→128 정책의 선택 겹침은 제한적**이다. 128이 최적 표본 규모임을 증명하지는 않는다. 충분한 독립 calibration 부트스트랩 또는 다른 원본 모델 재현은 여전히 필요하다.

## B. 원본 Apple Silicon CPU 커널·메모리 비용 구속 HiGHS 정책

동일한 원본 BEGLIN 역할별 Q4G64 및 재배열 n8 NEON 커널의 지연/저장 비용을 사용했다. 소스 지표 목적함수만 바꾸어 64/128 각각 **196개 중 정확히 28개 8비트, 168개 Q4**, 투영 바이트 ≤830,570,496B, 순차 단일 토큰 예측 ≤100ms, 가상 QKV/Gate-Up stage 합계 ≤50ms를 만족하는 정수 MILP 최적해를 고정했다. **최적성 간격 0**은 정의된 대리 목적함수에 관한 최적 증거이지 실제 NLL 최적 증거가 아니다.

- 64개 마스크: \`{plan['selected_policies']['64']['mask_sha256']}\`, 가중치 {plan['selected_policies']['64']['estimated_native_projection_bytes']:,}B, 순차 예측 {plan['selected_policies']['64']['predicted_single_token_serial_ms']:.3f}ms.
- 128개 마스크: \`{plan['selected_policies']['128']['mask_sha256']}\`, 가중치 {plan['selected_policies']['128']['estimated_native_projection_bytes']:,}B, 순차 예측 {plan['selected_policies']['128']['predicted_single_token_serial_ms']:.3f}ms.
- 원래 24개 Qv5 마스크: \`{plan['original_Qv5_24_mask_sha256']}\`.

이 수치는 **정적 비용 예측**이며 실제 장시간 생성 처리량, 활성 MPS GPU 메모리, 요청별 실시간 정밀도 변경을 실증한 수치가 아니다.

## C. 실제 모델 48개 문항 × 4개 정밀도 정책

동일하게 SHA로 잠긴 48개 OpenBookQA 프롬프트, 동일한 Qwen BF16 모델 가중치와 tokenizer, 원래 Qv5의 **독립 감사된 28계층 BEGLIN 네이티브 C 바이너리** 두 종류를 재사용했다. 평가마다 원본 모델을 새로 로드해 네 가지 정밀도 정책의 정답 후보 A/B/C/D 로짓을 평가했다. 점수는 *정답 단일 글자*의 teacher-forced NLL이고 정답률은 A/B/C/D 후보 간 argmax이다.

| 정책 | 정답 | 정답 글자 평균 NLL ↓ | 전체 Q4 대비 NLL 변화 |
|---|---:|---:|---:|
{chr(10).join(rows)}

**공정성/해석 주의:** 원본 BF16 Torch MPS와 네이티브 BEGLIN C Q4/n8 실행은 서로 다른 연산 경로다. 네 가지 C 혼합 정책은 같은 네이티브 바이너리 및 입력을 사용했고, C Q4만 별도 타입 경로 바이너리다. 기록된 wall time은 이 엔진의 지속 serving p50/p95가 아니다.

### 쌍별 불확실성 확인

OpenBookQA 48개 **같은 문항 기준** 후보 − Q4 NLL 차이를 계산했다. 5,000회 결정적 재표집의 95% 백분위 구간이며, 여러 정책·모델군·장르에 대한 검정력이나 다중비교 보정은 아니다. 음수는 후보 정책의 NLL이 낮다는 뜻이다.

| 후보 | 후보−Q4 평균 NLL | 95% 부트스트랩 구간 | 정답 개수 변화 |
|---|---:|---:|---:|
{chr(10).join(diagnostics)}

**이 48개 작은 교차 데이터셋에서의 긍정 결과는 최종 일반화·배포 채택과 분리해야 한다.** 추가적인 미관측 질문·과제·새 모델 가중치/계열과 장시간 처리량을 검증해야 한다.

## D. 증거·독립 감사

- source 사전등록 SHA: \`{plan['Qv6_prelock_SHA256']}\`
- 보정 집계 SHA: \`{sha(S.read_bytes())}\`
- 64/128 소스 정책 전체 SHA: \`{sha(P.read_bytes())}\`
- 이번 4×48 네이티브 원본 로그·입력/바이너리 독립 감사 SHA: \`{rep['original_private_audit_sha256']}\`
- 독립 감사 공개 집계 SHA: \`{sha(A.read_bytes())}\`
- BF16 독립 Torch MPS 원본 증거 SHA: \`{sha(B.read_bytes())}\`
- 명시적 소스 provenance, 이전 24개 reuse 확인, per-item 추가 104개 원본 BF16 backward SHA, 24/64/128 마스크, 실측 C 4조건의 전체 192개 로그 점수와 정답을 독립 재계산했다.

**원본 tokenizer ID 배열·BF16 gradient 196×128, 텐서 가중치, 개별 평가 로짓/출력은 GitHub에 업로드하지 않는다.** \`lab_local/qv6_fisher_scaling_20261022\` 아래 원본은 사적인 \`0700/0600\`로 유지한다.

## E. 제한 및 다음 게이트

1. 완전한 OOD는 아니다(서로 다른 자료셋이지만 과학 문제 모두 포함). 복수 모델 계열/다른 장르로 확대해야 한다.
2. 보정 128건·교차 평가 48건에서 최적 bit을 주장할 수 없다. 구간은 작은 표본의 개발 결과다.
3. 새 Qv6는 **요청별 온라인 bit 전환**이나 자동 Metal GPU 할당을 구현한 것이 아니다.
4. 다음 연구는 독립 source calibration seed 반복, Q4/n8 외 n-bit 지원 포맷별 실측, 같은 입력에서 ABBA steady TPS/CPU RSS/에너지, 새 계열의 장르·길이별 정확도 강건성.
5. 원래 V6 소진 24문항의 provenance 문제, 비동기 fullmodel 0% 속도 이득, Qv5 개발 결과는 그대로 보존.
6. \`main\`, 원본 HF 모델, Tailnet Commander immutable 운영 릴리스, 자동 live promotion은 변경하지 않음.

**현재 판정: SOURCE_RANK_SCALING_PASS / EXACT_CPU_MILP_PASS / NATIVE_4x48_PASS / INDEPENDENT_AUDIT_PASS / FULL_GENERALIZATION_NO_VERDICT / OPERATIONAL_PROMOTION_DENIED.**
"""
    report=report.replace(chr(92)+chr(96),chr(96)).replace('네 가지 C 혼합 정책','세 가지 C 혼합 정책').replace('BF16 gradient 196×128','BF16 방향성 점수 196×128')
    b=report.encode()
    with OUT.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    print("QV6_HANDOFF_SEALED",sha(b),len(b),flush=True)
if __name__=="__main__":main()
