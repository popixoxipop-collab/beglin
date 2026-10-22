#!/usr/bin/env python3
"""Independent full BEGLIN C Q4 / 1N8 / 2N8 RACE-test16 interaction audit.

Separate from Qv12 model generator/encoder and previous observed Qv11 mask
selection. Rechecks original source C code transformations, three executable
SHAs, 7×16 actual CPU Qwen28-window scores, exact source candidates/labels,
2n8 masks, and finite-difference pair interaction using article-level paired
bootstrap. Small public research sample; automatic live promotion forbidden.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,struct
import numpy as np
from scipy.stats import binomtest
ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv12_twotensor_20261023"
OLD=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv11-trueintervention-20261023")
Q5=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
PRE=ROOT/"configs/research/BEGLIN_QV12_TWO_N8_INTERACTION_PRELOCK_20261023.json"
SOURCE=LAB/"new_RACE_middle_test16_disjoint_Qv11_frozen.json"
PRIOR=OLD/"lab_local/qv11_single_original_C_20261023/RACE_middle_test16_distinct_article_locked.json"
ORIGIN_C=Q5/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/qv5_mixed.c"
NEW_C=LAB/"native_2n8_original_C/qv12_original_qv5_mixed_only2n8.c"
BUILD=LAB/"native_2n8_original_C/build_original_native_two_n8_receipt.json"
RUN=LAB/"native_Qv12_7arm_new_RACEtest16"
OUTPUT=LAB/"independent_Qv12_real_7arm_two_n8_interaction_private.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV12_REAL_TWO_N8_NATIVE_INTERACTION_INDEPENDENT_PUBLIC_20261023.json"
BINARY={
 "q4":Q5/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_base",
 "single":OLD/"lab_local/qv11_single_original_C_20261023/counterfactual_C_1n8/beglin_qv11_singleton_1n8",
 "pair":LAB/"native_2n8_original_C/beglin_qv12_two_tensor_n8"}
SHA={
 "q4":"408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c",
 "single":"a054e68789fd6bc20a4bd815da5108b63361b14227b1fc4ab9e68a3b7b9a2711",
 "pair":"d40fd6d42e08b0bc9d5dd3c7f8e0a4f5bcaabb2d9de7310d89f033e02500e961"}
PRE_SHA="a565bdec341d675ce52a37132b06355760c2fef96b8e4329c17508a237e4374f"
RACE_SHA="0650f791937f0d924706db6462b7f5e370fc58c1819b1228e91402d4937ca63e"
MODES=("q4","down_proj_10","v_proj_20","k_proj_0","down_proj_10__v_proj_20","down_proj_10__k_proj_0","v_proj_20__k_proj_0")
ROLE=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
TOTAL=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
def sha(b:bytes):return hashlib.sha256(b).hexdigest()
def sealed(p):
 if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
  raise RuntimeError("private native original BEGLIN result missing or unsafe "+str(p))
 return p.read_bytes()
def check_binary_source():
 build=json.loads(sealed(BUILD))
 if build["prelock_sha256"]!=PRE_SHA or \
    build["status"]!="ORIGINAL_QV5_28LAYER_2N8_194Q4_REAL_C_BINARY_COMPILED":
  raise RuntimeError("not original singleton/pair BEGLIN source")
 previous=ORIGIN_C.read_bytes()
 if sha(previous)!=build["original_Qv5_source_C_sha256"] or \
    sha(NEW_C.read_bytes())!=build["new_Qv12_two_N8_source_C_sha256"]:
  raise RuntimeError("native original C source SHA after kernel modification")
 expected=previous.decode()
 for old,new in (("if(total_n8!=28)","if(total_n8!=2)"),
                 ("expected exactly 28 n8 tensor roles","expected exactly 2 n8 tensor roles"),
                 ("validated mixed 28/196 n8 tensor role-layer mask","validated mixed 2/196 n8 tensor role-layer mask")):
  if expected.count(old)!=1:raise RuntimeError("not original audited C source patch anchor")
  expected=expected.replace(old,new,1)
 if expected.encode()!=NEW_C.read_bytes():
  raise RuntimeError("new native BEGLIN 2n8 changed additional C kernel semantics")
 for tag,p in BINARY.items():
  if not p.is_file() or sha(p.read_bytes())!=SHA[tag]:
   raise RuntimeError("real CPU 28-layer C Q4/1n8/2n8 binary drift "+tag)
 if build["new_Qv12_two_N8_binary_sha256"]!=SHA["pair"] or \
    build["original_Qv11_one_N8_binary_sha256"]!=SHA["single"] or \
    build["original_Qv5_full_Q4_binary_sha256"]!=SHA["q4"]:
  raise RuntimeError("three actual C native binaries not bound")
def parse_arm(mode,source,manifest):
 group="q4" if mode=="q4" else ("pair" if "__" in mode else "single")
 folder=RUN/mode
 log=sealed(folder/"actual_original_Qwen28_C_pair_single_RACEtest16.log")
 record=json.loads(sealed(folder/"original_C_RACE16_arm_receipt.json"))
 if record["status"]!="ACTUAL_QWEN28_ORIGINAL_C_RACE16_PAIRED_PRECISION_EXECUTION_PASS" or \
    record["mode"]!=mode or record["real_original_C_binary_sha256"]!=SHA[group] or \
    record["actual_original_C_process_log_sha256"]!=sha(log) or \
    record["original_RACE16_preexec_manifest_sha256"]!=sha(sealed(RUN/"seven_arm_original_inputs_and_masks_preexec_sha_manifest.json")):
  raise RuntimeError("original C real fullmodel log SHA/receipt mismatch "+mode)
 target={"q4":(0,0,196),"single":(0,1,195),"pair":(0,2,194)}[group]
 role=ROLE.findall(log.decode())
 if len(role)!=1 or tuple(map(int,role[0]))!=target or \
    record["actual_n8_role_count"]!=target[1] or record["actual_q4_role_count"]!=target[2]:
  raise RuntimeError("actual 28-layer role tensor precision not 0/1/2 n8 "+mode)
 if group!="q4":
  expected="".join((t.rsplit("_",1)[0]+" "+t.rsplit("_",1)[1]+"\n") for t in mode.split("__")).encode()
  mask=sealed(folder/"native_n8_role_mask.txt")
  if mask!=expected or sha(mask)!=manifest["true_native_role_masks_SHA256"][mode] or \
      record["specific_original_role_mask_sha256"]!=sha(mask):
   raise RuntimeError("single or pair source projection mask changed "+mode)
 rows=ROW.findall(log.decode())
 if len(rows)!=16:raise RuntimeError("16 genuine original C windows absent "+mode)
 nll=[];acc=[]
 for ix,(index,value,correct,pred,length) in enumerate(rows):
  q=source[ix];meta=manifest["input_cases_and_answers_SHA256"][ix]
  prompt=struct.pack("<"+"i"*len(q["prefix_ids"]),*q["prefix_ids"])
  correct_option=(" ".join(str(q["suffix_ids"][k]) for k in "ABCD")+" "+str("ABCD".index(q["correct"]))+"\n").encode()
  if int(index)!=ix or int(correct)!="ABCD".index(q["correct"]) or not 0<=int(pred)<4 or \
      int(length)!=len(q["prefix_ids"]) or \
      sealed(folder/f"eval/ppl_work/w{ix}.i32")!=prompt or \
      sealed(folder/f"eval/ppl_work/w{ix}.choices")!=correct_option or \
      sha(prompt)!=meta["prompt_sha256"] or sha(correct_option)!=meta["answer_sha256"] or \
      meta["full_RACE_untruncated_original_source_prompt_SHA256"]!=q["original_full_prompt_sha256"]:
   raise RuntimeError("original complete RACE article prompt/answer or binary token SHA changed "+mode)
  f=float(value)
  if not math.isfinite(f) or f<0:raise RuntimeError("nonfinite actual original BEGLIN C NLL "+mode)
  nll.append(f);acc.append(int(correct==pred))
 totals=TOTAL.findall(log.decode())
 if len(totals)!=1:raise RuntimeError("missing original C sum of 16 next-answer losses")
 total,n,right,mean,ppl=map(float,totals[0])
 if n!=16 or right!=sum(acc) or abs(total-sum(nll))>1e-7 or \
    abs(mean-total/16)>1e-7 or abs(math.log(ppl)-mean)>1e-7 or \
    record["correct_count"]!=sum(acc) or \
    abs(record["correct_label_mean_fullvocab_NLL"]-mean)>1e-10:
  raise RuntimeError("independent BEGLIN C 16 RACE windows incorrect arithmetic")
 if not np.array_equal(np.asarray(record["per_case_correct_next_token_NLL"]),np.asarray(nll)) or \
    not np.array_equal(np.asarray(record["per_case_top4_accuracy_correct"]),np.asarray(acc)):
  raise RuntimeError("reported per-case original C scores inconsistent with transcript")
 return {"n":16,"correct":int(right),"mean_fullvocab_correct_token_NLL":float(mean),
   "per_case_correct_fullvocab_NLL":nll,"per_case_choice_correct":acc,
   "log_sha256":sha(log),"n8_count":target[1],"q4_count":target[2]}
def main():
 if OUTPUT.exists() or PUBLIC.exists():raise FileExistsError("native source audit append only")
 if sha(PRE.read_bytes())!=PRE_SHA or sha(sealed(SOURCE))!=RACE_SHA:
  raise RuntimeError("Qv12 RACE test16 or previous-source-based intervention prelock changed")
 old=json.loads(sealed(PRIOR))["real_question_source"]
 cohort=json.loads(sealed(SOURCE))["full_frozen_questions"]
 if len(old)!=16 or len(cohort)!=16 or \
   set(x["article_original_id"] for x in old)&set(x["article_original_id"] for x in cohort) or \
   len(set(x["article_original_id"] for x in cohort))!=16:
  raise RuntimeError("new RACE16 articles overlap previously used Qv11 RACE")
 check_binary_source()
 manifest=json.loads(sealed(RUN/"seven_arm_original_inputs_and_masks_preexec_sha_manifest.json"))
 if manifest["status"]!="ALL_SEVEN_ORIGINAL_QWEN_NATIVE_PAIR_AND_SINGLE_ARMS_FROZEN_BEFORE_ANY_FORWARD" or \
    manifest["prelock_sha256"]!=PRE_SHA or tuple(manifest["original_pairs_and_singletons"])!=MODES or \
    manifest["new_RACEtest16_source_sha256"]!=RACE_SHA:
  raise RuntimeError("seven true C native mask/input manifest not preregistered")
 truth={mode:parse_arm(mode,cohort,manifest) for mode in MODES}
 q4=np.asarray(truth["q4"]["per_case_correct_fullvocab_NLL"],dtype=np.float64)
 if q4.shape!=(16,):raise RuntimeError("baseline real native Qwen C NLL missing")
 seed=20261023;rng=np.random.default_rng(seed)
 draws=rng.integers(0,16,size=(5000,16))
 singles={}
 for mode in MODES[1:4]:
  nll=np.asarray(truth[mode]["per_case_correct_fullvocab_NLL"],dtype=np.float64)
  gains=q4-nll
  dist=gains[draws].mean(axis=1)
  singles[mode]={"mean_NLL_gain_allQ4_minus_singleN8":float(gains.mean()),
   "gain_95CI":[float(x) for x in np.percentile(dist,[2.5,97.5])],
   "correct_change":truth[mode]["correct"]-truth["q4"]["correct"],
   "improves_per_article_count":int((gains>0).sum())}
 pairs={}
 for mode in MODES[4:]:
  a,b=mode.split("__")
  i=np.asarray(truth[a]["per_case_correct_fullvocab_NLL"],dtype=np.float64)
  j=np.asarray(truth[b]["per_case_correct_fullvocab_NLL"],dtype=np.float64)
  both=np.asarray(truth[mode]["per_case_correct_fullvocab_NLL"],dtype=np.float64)
  extra=both-i-j+q4
  expected=q4-i+q4-j
  achieved=q4-both
  if not np.allclose(achieved,expected-extra,atol=1e-12,rtol=0):
   raise RuntimeError("native interaction algebra not reproducible")
  boot=extra[draws].mean(axis=1)
  ci=np.percentile(boot,[2.5,97.5])
  npositive=int((extra>0).sum());nnegative=int((extra<0).sum())
  pairs[mode]={
    "singleton_components":[a,b],
    "native_pair_correct":truth[mode]["correct"],
    "naive_sum_singleton_NLL_recovery":float(expected.mean()),
    "actual_doubleton_NLL_recovery":float(achieved.mean()),
    "interaction_excess_NLL_mean":float(extra.mean()),
    "interaction_excess_NLL_paired_bootstrap95":[float(x) for x in ci],
    "interaction_sign":"superadditive_improvement" if extra.mean()<0 else "subadditive_improvement",
    "per_article_superadditive_count":nnegative,
    "per_article_subadditive_count":npositive,
    "exact_nonzero_sign_test_p":float(binomtest(min(nnegative,npositive),nnegative+npositive,p=.5).pvalue) if nnegative+npositive else 1.,
    "nonadditivity_nonzero_numerically":bool(abs(float(extra.mean()))>1e-9),
    "ci_excludes_zero":bool(ci[0]>0 or ci[1]<0)
  }
 info={
   "schema":"BEGLIN-Qv12-real-native-Qwen28-twoN8-interactions-7arm-RACEtest16-independently-audited/1",
   "status":"PASS_7X16_ORIGINAL_BEGLIN_C_TRUE_SINGLE_AND_PAIR_TWO_N8_INTERACTION",
   "prelock_sha256":PRE_SHA,"new_RACEtest16_private_sha256":RACE_SHA,
   "authentic_original_C_0N8_1N8_2N8_binary_SHA256":SHA,
   "prior_Qv11_articleIDs_disjoint":True,
   "preselected_three_roles_from_previous_RACE16_only":True,
   "new_RACEtest16_not_used_to_select_tensors":True,
   "native_7arm_results":truth,
   "native_singleton_gains_on_new_RACE":singles,
   "native_pairwise_interaction_exact_finite_differences":pairs,
   "seed":seed,"paired_article_bootstrap_resamples":5000,
   "observed_nonzero_pair_interaction_count":int(sum(x["nonadditivity_nonzero_numerically"] for x in pairs.values())),
   "number_pair_ci_excludes_zero":int(sum(x["ci_excludes_zero"] for x in pairs.values())),
   "sample_size_16_articles_no_multi_test_adjustment":True,
   "exact_output_is_real_endtask_NLL_not_GGN_proxy":True,
   "only_pairs_two_n8_vs_singletons_one_n8_not_official_28n8_budget":True,
   "low_load_latency_and_Genuine_Metal_speed_not_evaluated":True,
   "same_model_multiple_existing_Qv11_Qv12_public_test_not_generalization":True,
   "source_original_model_and_C_NEON_kernels_not_changed":True,
   "automatic_precision_live_promotion":False}
 os.umask(0o077)
 data=(json.dumps(info,sort_keys=True,indent=2)+"\n").encode()
 with OUTPUT.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 OUTPUT.chmod(0o600)
 pub={k:v for k,v in info.items() if k!="native_7arm_results"}
 pub["native_summary"]={mode:{k:v for k,v in scores.items() if not k.startswith("per_case")} for mode,scores in truth.items()}
 pub["actual_private_independent_full_receipt_sha256"]=sha(data)
 b=(json.dumps(pub,sort_keys=True,indent=2)+"\n").encode()
 with PUBLIC.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
 print("QV12_INDEPENDENT_112_TRUE_BEGLIN_NATIVE_PAIR_INTERACTIONS_PASS",json.dumps({
   "correct":{k:v["correct"] for k,v in truth.items()},
   "nll":{k:round(v["mean_fullvocab_correct_token_NLL"],6) for k,v in truth.items()},
   "interactions":{k:{"delta":round(v["interaction_excess_NLL_mean"],6),
        "CI":v["interaction_excess_NLL_paired_bootstrap95"]} for k,v in pairs.items()},
   "privateSHA":sha(data),"publicSHA":sha(b)},sort_keys=True),flush=True)
if __name__=="__main__":main()
