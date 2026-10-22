#!/usr/bin/env python3
"""Independent SHA and actual full-model RACE 16 × (Q4+6 singleton n8) audit.

Auditor does not import Qv11 source capture, binary builder, native generator or
source rank selector. It checks exact original C binary and C source provenance,
authentic seven native 28-layer process transcripts, identical label/prompt
arrays, each one-n8 model's 195 Q4 roles, per-case teacher forced full-vocab
correct-letter NLL, paired ΔNLL and source Fisher128-to-native rank alignment.
Only 16 public questions; estimates are exploratory and no success gate grants
live promotion.
"""
from __future__ import annotations
import hashlib,itertools,json,math,os,pathlib,re,struct
import numpy as np
from scipy.stats import spearmanr
ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv11_single_original_C_20261023"
FREEZE=LAB/"RACE_middle_test16_distinct_article_locked.json"
RUN=LAB/"native_7arms_RACEtest16"
PLAN=ROOT/"configs/research/BEGLIN_QV11_REAL_ONE_N8_INTERVENTION_PRELOCK_20261023.json"
SRC=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv6-fisher-20261022/docs/research/BEGLIN_QV6_FISHER_SCALING_196_SOURCE_AGGREGATE_20261022.json")
ORIG=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_base")
PATCHED=LAB/"counterfactual_C_1n8/beglin_qv11_singleton_1n8"
BUILD=LAB/"counterfactual_C_1n8/build_evidence.json"
OUT=LAB/"independent_actual_real_Qwen28_1tensor_C_probe_audit_private.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV11_REAL_ONE_TENSOR_NATIVE_CAUSAL_AUDIT_PUBLIC_20261023.json"
BASE_SHA="408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c"
SINGLE_SHA="a054e68789fd6bc20a4bd815da5108b63361b14227b1fc4ab9e68a3b7b9a2711"
FREEZE_SHA="0de242b608837bd29ed72db99f1e910626753e2b27ea6ada9a24bdf864c96499"
PRE_SHA="2cd480dc4c72b5383014a1325d350e51a0b47b1335ec0013a6d069773b546a39"
ARMS=("q4","down_proj_2","down_proj_10","k_proj_0","k_proj_1","v_proj_20","v_proj_27")
ROLE=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
POINT=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
SUM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
def sha(b):return hashlib.sha256(b).hexdigest()
def sealed(p):
 if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:
  raise RuntimeError("private original C experiment evidence missing or insecure: "+str(p))
 return p.read_bytes()
def parse(arm,source,manifest):
 folder=RUN/arm
 actual=sealed(folder/"true_28layer_RACE16_original_C_process.log")
 receipt=json.loads(sealed(folder/"original_singleton_1n8_RACE16_native_result.json"))
 if receipt["status"]!="ACTUAL_SINGLE_TENSOR_N8_28LAYER_NATIVE_QWEN_16_RACE_CASES_PASS" or \
    receipt["arm"]!=arm or receipt["original_native_CPU_process_log_SHA256"]!=sha(actual) or \
    receipt["original_7arm_before_forward_SHA256"]!=sha(sealed(RUN/"Qv11_true_one_tensor_native_all_7arms_prelock.json")):
  raise RuntimeError("native one-tensor research result claimed without original C log SHA")
 expected=BASE_SHA if arm=="q4" else SINGLE_SHA
 binary=ORIG if arm=="q4" else PATCHED
 if sha(binary.read_bytes())!=expected or receipt["original_singleton_C_binary_sha256"]!=expected:
  raise RuntimeError("source authentic C native code changed after intervention")
 r=ROLE.findall(actual.decode())
 if len(r)!=1 or tuple(map(int,r[0]))!=((0,0,196) if arm=="q4" else (0,1,195)):
  raise RuntimeError("C output did not execute correct 196 tensor precision topology")
 if arm!="q4":
  role,layer=arm.rsplit("_",1)
  required=f"{role} {layer}\n".encode()
  if sealed(folder/"native_single_1n8_mask.txt")!=required or sha(required)!=manifest["actual_source_role_mask_SHA256"][arm]:
   raise RuntimeError("changed singleton tensor identity")
 rows=POINT.findall(actual.decode())
 if len(rows)!=16:raise RuntimeError("not 16 original C native RACE test item forward passes")
 losses=[];rights=[]
 for i,(idx,nll,expected_answer,predicted,n) in enumerate(rows):
  q=source[i];f=manifest["same_question_SHA_and_candidate_SHA_across_7arms"][i]
  raw=struct.pack("<"+"i"*len(q["prefix_ids"]),*q["prefix_ids"])
  answers=(" ".join(str(q["suffix_ids"][letter]) for letter in "ABCD")+" "+str("ABCD".index(q["correct"]))+"\n").encode()
  if int(idx)!=i or int(n)!=len(q["prefix_ids"]) or int(expected_answer)!="ABCD".index(q["correct"]) or not 0<=int(predicted)<4 or \
     sealed(folder/f"eval/ppl_work/w{i}.i32")!=raw or sealed(folder/f"eval/ppl_work/w{i}.choices")!=answers or \
     sha(raw)!=f["raw_prompt_sha"] or sha(answers)!=f["answer_option_sha"] or \
     f["article_id_sha"]!=q["id_sha256"]:
    raise RuntimeError("real native source model original frozen prompt/answer token mismatch")
  loss=float(nll)
  if not math.isfinite(loss) or loss<0:raise RuntimeError("negative/nan native loss")
  losses.append(loss);rights.append(int(expected_answer==predicted))
 summary=SUM.findall(actual.decode())
 if len(summary)!=1:raise RuntimeError("source native C score header/tail missing")
 total,num,correct,mean,ppl=map(float,summary[0])
 if num!=16 or int(correct)!=sum(rights) or abs(total-sum(losses))>1e-7 or \
    abs(mean-total/16)>1e-7 or abs(math.log(ppl)-mean)>1e-7:
  raise RuntimeError("real native 1N8 C score aggregation inconsistent")
 if receipt["correct"]!=sum(rights) or abs(receipt["correct_answer_fullvocab_mean_NLL"]-mean)>1e-10:
  raise RuntimeError("native scorer receipt differs from independent original C")
 return {"actual_mean_correct_token_NLL":float(mean),"accuracy":int(correct)/16,"correct":int(correct),
     "raw_process_SHA256":sha(actual),"per_item_correct_token_NLL":losses,
     "per_item_choice_right":rights}
def main():
 if OUT.exists() or PUBLIC.exists():raise FileExistsError("Qv11 independent 7-arm audit already sealed")
 if sha(PLAN.read_bytes())!=PRE_SHA or sha(sealed(FREEZE))!=FREEZE_SHA:
  raise RuntimeError("Qv11 source prereg or evaluation cohort SHA changed")
 samples=json.loads(sealed(FREEZE))["real_question_source"]
 if len(samples)!=16 or len({d["article_original_id"] for d in samples})!=16:
  raise RuntimeError("RACE test 16 non-independent article IDs")
 build=json.loads(sealed(BUILD))
 if build["new_binary_sha256"]!=SINGLE_SHA or build["original_Qv5_C_sha256"]!="375b3dfb02ac2c0fc90f3f500718578fe62d38cdcac541099fb896797470dbeb":
  raise RuntimeError("true native one-N8 card C source not authenticated")
 manifest=json.loads(sealed(RUN/"Qv11_true_one_tensor_native_all_7arms_prelock.json"))
 if manifest["status"]!="BEFORE_ANY_C_FORWARD_SEVEN_ARMS_WITH_TRUE_SINGLE_TENSOR_INTERVENTIONS_FROZEN" or \
    manifest["Qv11_only_1N8_native_binary_sha256"]!=SINGLE_SHA or len(manifest["same_question_SHA_and_candidate_SHA_across_7arms"])!=16:
  raise RuntimeError("data frozen only after native C execution")
 truth={arm:parse(arm,samples,manifest) for arm in ARMS}
 cfg=json.loads(PLAN.read_text())
 source=json.loads(SRC.read_text())["quality_upgrade_by_role_layer_per_size"]["128"]
 rng=np.random.default_rng(20261023)
 boot=rng.integers(0,16,size=(5000,16))
 ref=np.asarray(truth["q4"]["per_item_correct_token_NLL"])
 measured={};gains=[];out=[]
 for arm in ARMS[1:]:
  role,idx=arm.rsplit("_",1)
  id=int(idx)
  gain=source[role][id]
  diff=np.asarray(truth[arm]["per_item_correct_token_NLL"])-ref
  b=diff[boot].mean(axis=1)
  entry={
    "role":role,"layer":id,"source_BF16_Fisher128_proposed_gain":float(gain),
    "actual_one_tensor_N8_minus_fullQ4_mean_fullvocab_NLL":float(diff.mean()),
    "actual_N8_improvement_fullQ4_minus_N8":float(-diff.mean()),
    "paired_bootstrap95_delta_NLL":[float(x) for x in np.percentile(b,[2.5,97.5])],
    "actual_change_correct_count":truth[arm]["correct"]-truth["q4"]["correct"],
    "number_RACE_passages_singleton_NLL_better":int(np.sum(diff<0)),
    "native_one_N8_binary_SHA256":SINGLE_SHA,
    "actual_singleton_mask_SHA256":manifest["actual_source_role_mask_SHA256"][arm]
  }
  measured[arm]=entry;gains.append(float(gain));out.append(float(-diff.mean()))
 corr=float(spearmanr(gains,out).statistic)
 if not math.isfinite(corr):raise RuntimeError("source BF16 Fisher vs real C downstream Spearman invalid")
 pairs={}
 for role in ("down_proj","k_proj","v_proj"):
  high=cfg["role_extremes"][role]["high_layer"]
  low=cfg["role_extremes"][role]["low_layer"]
  h=measured[f"{role}_{high}"];l=measured[f"{role}_{low}"]
  pairs[role]={"source_gain_high":h["source_BF16_Fisher128_proposed_gain"],
   "source_gain_low":l["source_BF16_Fisher128_proposed_gain"],
   "actual_source_high_NLL_recovery":h["actual_N8_improvement_fullQ4_minus_N8"],
   "actual_source_low_NLL_recovery":l["actual_N8_improvement_fullQ4_minus_N8"],
   "source_predicts_high_benefits_more":True,
   "actual_high_n8_more_helpful_on_RACE16":h["actual_N8_improvement_fullQ4_minus_N8"]>l["actual_N8_improvement_fullQ4_minus_N8"]}
 successes=sum(p["actual_high_n8_more_helpful_on_RACE16"] for p in pairs.values())
 report={"schema":"BEGLIN-Qv11-independent-six-1N8-source-sensitivity-vs-real-C-Qwen28-native/1",
   "status":"REAL_SINGLE_TENSOR_1N8_VS_ALLQ4_16RACE_7ARMS_INDEPENDENT_AUDIT_PASS",
   "prelock_SHA256":PRE_SHA,"original_RACE_test16_SHA256":FREEZE_SHA,
   "original_beG_C_model_checkpoint_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
   "source_Qv6_128_Fisher_public_SHA256":sha(SRC.read_bytes()),
   "real_native_original_BF16_weight_C_Q4_binary_sha256":BASE_SHA,
   "isolated_true_one_tensor_quant_N8_binary_SHA256":SINGLE_SHA,
   "raw_actual_native_seven_arm_scores":truth,
   "measured_per_tensor_fullmodel_actual_NLL_effect":measured,
   "within_role_high_vs_low_source_proxy_recovery_alignment":pairs,
   "within_role_predicted_rank_concordance_count":successes,
   "source_scores_vs_native_RACE_improvements_Spearman_6points":corr,
   "fullQ4_baseline_correct":truth["q4"]["correct"],
   "fullQ4_baseline_mean_fullvocab_NLL":truth["q4"]["actual_mean_correct_token_NLL"],
   "model_single_tensor_native_intervention_proven":True,
   "only_one_actual_quantized_projection_tensor_changed_each_arm":True,
   "single_tier_role_memory_differs_between_roles_not_a_matched_28N8_budget":True,
   "one_test_cohort16_and_six_prior_source_extreme_roles_small_exploratory":True,
   "no_generalized_precision_quality_advantage_established":True,
   "no_production_or_main_model_mutation":True,
   "automated_live_precision_promotion":False}
 os.umask(0o077)
 b=(json.dumps(report,sort_keys=True,indent=2)+"\n").encode()
 with OUT.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
 OUT.chmod(0o600)
 pub={k:v for k,v in report.items() if k!="raw_actual_native_seven_arm_scores"}
 pub["independent_private_original_native_audit_SHA256"]=sha(b)
 doc=(json.dumps(pub,sort_keys=True,indent=2)+"\n").encode()
 with PUBLIC.open("xb") as f:f.write(doc);f.flush();os.fsync(f.fileno())
 print("QV11_SIX_ACTUAL_SINGLE_TENSOR_NATIVE_C_CAUSAL_RESULTS_INDEPENDENT_AUDIT_PASS",
  json.dumps({"baseline_NLL":report["fullQ4_baseline_mean_fullvocab_NLL"],
   "source_to_real_Spearman":corr,"within_role_high_low_concordance":successes,
   "mean_delta_NLL":{k:round(v["actual_one_tensor_N8_minus_fullQ4_mean_fullvocab_NLL"],7) for k,v in measured.items()},
   "private_SHA":sha(b),"public_SHA":sha(doc)},sort_keys=True),flush=True)
if __name__=="__main__":main()
