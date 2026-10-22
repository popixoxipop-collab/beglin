#!/usr/bin/env python3
"""Qv12 final original BEGLIN C two-tensor NLL interaction research handoff.

Publishes only aggregate source/audit provenance from 7 native C arms.
Refuses missing SHA-backed source or model reference and never promotes
bit policy, model, operating release or GitHub main.
"""
import hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[2]
D=R/"docs/research";C=R/"configs/research"
PRE=C/"BEGLIN_QV12_TWO_N8_INTERACTION_PRELOCK_20261023.json"
FREEZE=D/"BEGLIN_QV12_RACE16_NEW_INPUT_FREEZE_PUBLIC_20261023.json"
BUILD=D/"BEGLIN_QV12_REAL_NATIVE_2N8_C_SOURCE_PROVENANCE_PUBLIC_20261023.json"
COST=D/"BEGLIN_QV12_ONE_TWO_N8_PACKED_WEIGHT_COST_PUBLIC_20261023.json"
REF=D/"BEGLIN_QV12_ORIGINAL_BF16_NEW_RACE16_REFERENCE_PUBLIC_20261023.json"
AUDIT=D/"BEGLIN_QV12_REAL_TWO_N8_NATIVE_INTERACTION_INDEPENDENT_PUBLIC_20261023.json"
OUT=D/"BEGLIN_QV12_TWO_TENSOR_NONADDITIVE_NATIVE_RACE16_HANDOFF_20261023.md"
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 if OUT.exists():raise FileExistsError("final Qv12 handoff already published; cannot overwrite")
 for p in (PRE,FREEZE,BUILD,COST,REF,AUDIT):
  if not p.is_file():raise RuntimeError("original source/model/7arm native QA artifact absent: "+str(p))
 cfg,freeze,build,cost,reference,audit=[json.loads(p.read_text()) for p in (PRE,FREEZE,BUILD,COST,REF,AUDIT)]
 if audit["status"]!="PASS_7X16_ORIGINAL_BEGLIN_C_TRUE_SINGLE_AND_PAIR_TWO_N8_INTERACTION" or \
    audit["automatic_precision_live_promotion"] or \
    audit["new_RACEtest16_private_sha256"]!=freeze["RACE_new_test16_private_SHA256"] or \
    build["new_Qv12_2N8_C_binary_sha256"]!=audit["authentic_original_C_0N8_1N8_2N8_binary_SHA256"]["pair"]:
  raise RuntimeError("not same source-frozen original C binary/RACE7arm evidence")
 if reference["status"]!="REFERENCE_ORIGINAL_BF16_QWEN_NEW_DISJOINT_RACE16_MODEL_COMPLETE" or \
    reference["frozen_new_RACEtest16_input_SHA256"]!=audit["new_RACEtest16_private_sha256"]:
  raise RuntimeError("original BF16 reference not same new RACE16 source")
 arms=audit["native_summary"];pairs=audit["native_pairwise_interaction_exact_finite_differences"]
 if set(arms)!=set(cfg["planned_arms"]) or len(pairs)!=3:
  raise RuntimeError("all 7 preregistered true 0/1/2 n8 C paths not executed")
 baseline=arms["q4"]["mean_fullvocab_correct_token_NLL"]
 modelrows=[]
 for name in cfg["planned_arms"]:
  x=arms[name]
  mem=cost["native_one_two_bit_arm_costs"][name]["projected_packed_projection_weight_bytes"]
  modelrows.append(f"| {name} | {x['n8_count']} / {x['q4_count']} | {x['correct']}/16 | {x['mean_fullvocab_correct_token_NLL']:.6f} | {baseline-x['mean_fullvocab_correct_token_NLL']:+.6f} | {mem:,} |")
 pairrows=[]
 for name in cfg["planned_arms"][4:]:
  x=pairs[name];ci=x["interaction_excess_NLL_paired_bootstrap95"]
  pairrows.append(f"| {name} | {x['naive_sum_singleton_NLL_recovery']:+.6f} | {x['actual_doubleton_NLL_recovery']:+.6f} | {x['interaction_excess_NLL_mean']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {'excludes zero' if x['ci_excludes_zero'] else 'includes zero'} |")
 doc=f"""# BEGLIN Qv12 — 실제 두 텐서 동시 8비트 승격의 비가산성 검증

**날짜:** 2026-10-23 KST
**분리 연구 브랜치:** research/qv12-two-n8-interaction-20261023
**최종 범위:** REAL_NATIVE_C_SINGLE_AND_DOUBLE_N8 / INDEPENDENT_AUDIT / NO_GENERALIZATION_CLAIM / NO_LIVE_PROMOTION.

## A. 동기와 사전 실험 계약

Qv11에서 개별 텐서를 Q4에서 8비트로 바꾸는 실질적인 전체 Qwen28 BEGLIN C 개입을 수행했다. 그러나 28개 텐서를 동시에 승격시키는 혼합 할당이 개별 개선을 선형적으로 더할 수 있는지는 조사되지 않았다. Qv12는 이를 확인하기 위해 **진짜 두 텐서의 동시 개입**을 수행한 후, 둘의 단독 효과 합과 비교한다. 이것은 source Fisher/Gauss-Newton 등 대리점수가 아닌 동일 원본 네이티브 전체 모델의 실측 정답 토큰 NLL 차이이다.

원본 모델 HF revision: {cfg['model_HF_rev']}. 모델/토크나이저는 고정하고, **기존 Qv11 개발 데이터의 16문항에서 역할별 단독 개선이 가장 컸던** down_proj:10, v_proj:20, k_proj:0을 평가 전에 선정했다. 선별에 기존 결과를 사용했으므로 초기 3텐서 후보 선정은 outcome-driven이며, 새 RACE 평가 뒤에 후보를 바꾸지 않았다.

새 평가: ehovy/race@{cfg['held_revision']}, middle/test, 서로 다른 원문 지문 16개, 이전 Qv11 RACE16에 쓰인 지문을 모두 제외했다. 질문, 원문 지문, 4지 보기 무자름, 161~208 Qwen ChatML tokens. 전 일곱 모델에서 동일한 A/B/C/D one-token labels, source prompt SHA와 정답 SHA 사용. 그러나 16개 공개 test 문항이 언어모델의 사전학습에서 전혀 노출되지 않았다는 보장은 없다.

사전등록 SHA: {sha(PRE.read_bytes())}.
새 RACE16 private input SHA: {freeze['RACE_new_test16_private_SHA256']}.

## B. 0개, 1개, 2개 n8 원본 C 실행

실험은 같은 Qwen2.5-1.5B 원본 BF16 사전양자화 가중치를 BEGLIN CPU 실기기 C로 로드했다.

- 기준 모델: 196개 Q4, n8 0개.
- 단독 모델 3개: 역할별 1개 n8, 나머지 195개 Q4.
- 두 텐서 조합 3개: 2개 n8, 나머지 194개 Q4.
- 원본 Q4·1n8 연구 바이너리와, 새 2n8 연구용 복제본은 original C source SHA에 묶었다. 2n8 복제본은 total_n8 cardinality assert를 28→2로 바꾸었고, 진단 문구 2개만 고쳤다. 원래 Apple NEON n8/Q4 kernel, 모델 계층 및 weight는 수정하지 않았다.

원본 Q4 binary SHA: {build['original_Qv5_allQ4_C_binary_sha256']}.
원본 1n8 binary SHA: {build['original_Qv11_1N8_C_binary_sha256']}.
신규 2n8 binary SHA: {build['new_Qv12_2N8_C_binary_sha256']}.

아래 투영 저장 크기는 진짜 소스 그룹별 Q4/n8 packed weights + scales를 통한 원가 장부 추정이다. 실제 RSS, GPU 사용량, 지속 실행 tok/s, 온도/전력은 실측한 항목이 아니다. 특히 down_proj n8은 k_proj/v_proj보다 추가 저장 비용이 크다. **2n8은 정식 28n8 정책과 예산이 맞지 않는다.**

## C. 진짜 전체 모델 7조건 ×16문항 결과

정답률은 A/B/C/D 후보 argmax, NLL은 전체 어휘로 정규화된 정답 single-letter next-token negative log likelihood다.
비교용 원본 BF16 PyTorch/MPS 참고 점수: **{reference['correct']}/16**, 평균 NLL **{reference['mean_fullvocab_correct_answer_NLL']:.6f}** (네이티브 widened-F32 경로와 별도 산술).

| 실제 native 정밀도 정책 | n8 / Q4 | 정답 수 | mean NLL ↓ | Q4 대비 NLL 개선 ↑ | projection packed bytes |
|---|---:|---:|---:|---:|---:|
{chr(10).join(modelrows)}

단독 개선량은 **전혀 다른 이번 RACE16 문항에서 새로 측정**했고 과거 Qv11의 단독 NLL 절대값을 조합 효과 산술에 재사용하지 않았다.

## D. 실제 단독-동시 정밀도 상호작용

각 동일한 원본 평가 지문 s에 대해 아래 실제 네이티브 NLL 값을 사용한다.

    I_ij(s) = L_pair_ij(s) − L_single_i(s) − L_single_j(s) + L_Q4(s)

**I<0**이면 두 텐서 승격의 NLL 개선이 단독 이득 합보다 큰 초가산적(synergistic) 효과.
**I>0**이면 두 텐서 승격의 개선이 단독 이득 합보다 작은 부분가산적 효과.
**I=0**이면 가산적 효과.

16개 원문 지문을 단위로 5,000회 재표집(seed 20261023)한 쌍별 탐색적 부트스트랩 구간을 계산했다.

| 선택한 두 텐서 | 단독 NLL 이득 합 예상 ↑ | 실제 2n8 NLL 이득 ↑ | 초과 손실 I_ij (음수=시너지) | paired 95% CI | CI 판정 |
|---|---:|---:|---:|---|---|
{chr(10).join(pairrows)}

관측한 각 텐서 조합의 유한한 비가산적 평균 차이가 수치상 0이 아닌 쌍은 **{audit['observed_nonzero_pair_interaction_count']}/3**이고, 부트스트랩 95% 구간이 0을 제외한 쌍은 **{audit['number_pair_ci_excludes_zero']}/3**이다. 표본 16개·쌍 3개, 다중비교 보정 없음. 이것만으로 일반적인 196텐서 결합 효과나 범용 동적 할당기 승격을 주장하지 않는다.

## E. 독립 감사, 보안 및 향후 실험

원본 실제 Qwen C 7조건 ×16개 정답 NLL 총 **112개**를 독립 코드로 다시 파싱하고, C 바이너리/새 2n8 C 원본 소스 바이트와 exactly cardinality-only source diff, SHA-locked 평가 원문/프롬프트/one-token 정답, native 기록의 실제 Q4/n8 역할 수를 확인했다. 원본 개별 모델 입력/손실/가중치/로그는 untracked owner-only lab_local에 남겨두며 GitHub에 올리지 않는다.

**공개 독립 감사 SHA:** {sha(AUDIT.read_bytes())}.
**원본 C 빌드 계보 공개 SHA:** {sha(BUILD.read_bytes())}.
**원본 평가 입력 사전 봉인 SHA:** {sha(FREEZE.read_bytes())}.

남은 핵심 게이트는 (1) 다른 모델 계열의 진짜 2n8 개입 재현, (2) 원래 예산인 28n8 정책에서 순차 추가 이득/상호작용, (3) 더 큰 신규 평가 문항에 대한 bootstrap 신뢰도, (4) 저부하 Apple CPU p50/p95, RSS, Metal GPU 별도 검증이다. 이번 새 RACE16에 맞춰 후행 마스크를 조정하거나 같은 데이터에서 새 알고리즘 성능 우위를 주장해서는 안 된다.

**운영:** main 병합 없음, immutable Tailnet release 미변경, 원본 모델 및 커널 미변경, 자동 정밀도 라우팅/승격 금지.

**최종 판정:** REAL_2N8_INTERACTION_EXECUTED / INDEPENDENT_112_NATIVE_AUDIT_PASS / FULL_28N8_POLICY_NOT_PROVEN / NO_LIVE_PROMOTION.
"""
 blob=doc.encode()
 with OUT.open("xb") as f:f.write(blob);f.flush()
 print("QV12_REAL_SINGLE_DOUBLE_7ARM_RESEARCH_HANDOFF_SEALED",json.dumps({
   "public_sha":sha(blob),"native_audit_sha":sha(AUDIT.read_bytes()),
   "pair_ci_excludes_zero":audit["number_pair_ci_excludes_zero"],"seven_rows":len(modelrows),
   "no_promotion":True},sort_keys=True))
if __name__=="__main__":main()
