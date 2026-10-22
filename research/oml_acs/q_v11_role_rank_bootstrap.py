#!/usr/bin/env python3
"""Independent post-native 16-article paired bootstrap for six 1-N8 interventions.

Reads original sealed C run receipts already independently audited, preserves
the source prelock and never refits the Fisher score or reruns any model.
"""
import hashlib,json,math,os,pathlib
import numpy as np
from scipy.stats import spearmanr
R=pathlib.Path(__file__).resolve().parents[2]
L=R/"lab_local/qv11_single_original_C_20261023"
D=R/"docs/research"
P=R/"configs/research/BEGLIN_QV11_REAL_ONE_N8_INTERVENTION_PRELOCK_20261023.json"
PRIVATE=L/"independent_actual_real_Qwen28_1tensor_C_probe_audit_private.json"
PUBLIC=D/"BEGLIN_QV11_REAL_ONE_TENSOR_NATIVE_CAUSAL_AUDIT_PUBLIC_20261023.json"
OUT=D/"BEGLIN_QV11_ROLE_RANK_PAIRED_BOOTSTRAP_PUBLIC_20261023.json"
ARMS=("down_proj_2","down_proj_10","k_proj_0","k_proj_1","v_proj_20","v_proj_27")
def sha(b):return hashlib.sha256(b).hexdigest()
def secure(f):
 if not f.is_file() or f.is_symlink() or f.stat().st_mode&0o777!=0o600:raise RuntimeError("private audited native source unsealed")
 return f.read_bytes()
def main():
 if OUT.exists():raise FileExistsError("role rank experiment already summarized")
 original=secure(PRIVATE);d=json.loads(original);pub=json.loads(PUBLIC.read_text());plan=json.loads(P.read_text())
 if pub["independent_private_original_native_audit_SHA256"]!=sha(original) or \
    d["status"]!="REAL_SINGLE_TENSOR_1N8_VS_ALLQ4_16RACE_7ARMS_INDEPENDENT_AUDIT_PASS":
  raise RuntimeError("original seven-arm audit invalid")
 truth=d["raw_actual_native_seven_arm_scores"]
 base=np.asarray(truth["q4"]["per_item_correct_token_NLL"],dtype=float)
 gains=np.asarray([d["measured_per_tensor_fullmodel_actual_NLL_effect"][a]["source_BF16_Fisher128_proposed_gain"] for a in ARMS],dtype=float)
 improvements=np.column_stack([base-np.asarray(truth[a]["per_item_correct_token_NLL"]) for a in ARMS])
 if improvements.shape!=(16,6):raise RuntimeError("original 16x6 native outcomes missing")
 rho=float(spearmanr(gains,improvements.mean(axis=0)).statistic)
 if abs(rho-d["source_scores_vs_native_RACE_improvements_Spearman_6points"])>1e-12:
  raise RuntimeError("Spearman not recomputed from sealed genuine native trials")
 rng=np.random.default_rng(20261023)
 draws=rng.integers(0,16,size=(5000,16))
 comparisons={}
 for role in ("down_proj","k_proj","v_proj"):
  high=plan["role_extremes"][role]["high_layer"]
  low=plan["role_extremes"][role]["low_layer"]
  x=f"{role}_{high}";y=f"{role}_{low}"
  diff=np.asarray(truth[x]["per_item_correct_token_NLL"])-np.asarray(truth[y]["per_item_correct_token_NLL"])
  CI=np.percentile(diff[draws].mean(axis=1),[2.5,97.5])
  comparisons[role]={"high_source_layer":high,"low_source_layer":low,
    "mean_high_minus_low_native_NLL":float(diff.mean()),
    "paired_bootstrap_95CI":[float(v) for v in CI],
    "high_n8_better_per_case_count":int((diff<0).sum()),
    "low_n8_better_per_case_count":int((diff>0).sum()),
    "rank_direction_correct":bool(diff.mean()<0)}
 samples=np.asarray([spearmanr(gains,improvements[x].mean(axis=0)).statistic for x in draws],dtype=float)
 samples=samples[np.isfinite(samples)]
 if len(samples)<4900:raise RuntimeError("rank bootstrap mostly invalid")
 ci=np.percentile(samples,[2.5,97.5])
 result={"schema":"BEGLIN-Qv11-actual-original-C-one-tensor-Fisher-vs-downstream-role-rank/1",
  "status":"PAIRWISE_TRUE_C_16ARTICLE_6INTERVENTION_BOOTSTRAP_COMPLETE",
  "source_private_authentic_C_independent_audit_SHA256":sha(original),
  "source_public_independent_C_audit_SHA256":sha(PUBLIC.read_bytes()),
  "source_prelock_SHA256":sha(P.read_bytes()),
  "paired_source_bootstrap_n":5000,"bootstrap_seed":20261023,
  "six_source_Fisher_vs_actual_C_NLL_gain_Spearman":rho,
  "six_probe_rank_Spearman_question_bootstrap_95CI":[float(x) for x in ci],
  "within_role_high_vs_low_all_paired":comparisons,
  "pair_direction_concordance":int(sum(x["rank_direction_correct"] for x in comparisons.values())),
  "pair_direction_total":3,
  "no_multiple_comparisons_significance_claim":True,
  "only_six_source_preselected_extremes":True,
  "sample_size_original_test_articles":16,
  "evaluation_RACE_test_not_unseen_by_model_pretraining":True,
  "source_quant_singleton_not_runtime_28n8_budget_matched":True,
  "source_Fisher_global_ranking_not_established":True,
  "promotion_denied":True}
 data=(json.dumps(result,sort_keys=True,indent=2)+"\n").encode()
 with OUT.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 print("QV11_TRUE_SINGLETON_NATIVE_ROLE_RANK_BOOTSTRAP_PASS",
  json.dumps({"rho":rho,"rho_CI":[float(x) for x in ci],
  "withinrole":{k:v["mean_high_minus_low_native_NLL"] for k,v in comparisons.items()},
  "aligned":result["pair_direction_concordance"],
  "public_sha":sha(data)},sort_keys=True))
if __name__=="__main__":main()
