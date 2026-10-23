#!/usr/bin/env python3
"""Qv20 SECOND independent no-peek heldout audit: full native C case receipts.
Do NOT run until calibration policy frozen and BOTH static controls plus actual
prompt-only Qv19-guarded native dynamic execution finished per heldout dataset.
Use stratified paired resampling, exact McNemar, and two-comparator Holm correction.
No threshold or feature/model selection uses heldout outcomes.
No raw case rows or tokens are published.
"""
import hashlib,json,math,pathlib
import numpy as np
from scipy.stats import binomtest
BASE=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv20-prompt-aware-20261024")
SPLITS=("test_arc_validation","test_ob_validation")
ARMS=("static_source","static_random")
INPUT_SHA={"test_arc_validation":"b0a27e22f0c5769f49c3dc95d2200feef9f06747349abef55bd82c29d13e2119",
"test_ob_validation":"2f80539f7eca065e8dd5effcc1c320e2fbcf74f7e6af0e1e2816a8fe72aeb204"}
PRE_SHA="2aae8540fef6bce9c15e4c71becd873f62d69da556056583a41bae810ec5da1a"
COUNT_PER_SPLIT=24
BOOT_N=30000
RNG_SEED=20261024
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(262144),b""):h.update(b)
 return h.hexdigest()
def must(ok,reason):
 if not ok:raise RuntimeError("QV20_INDEPENDENT_AUDIT_FAIL_CLOSED_"+reason)
def data(path):return json.loads(path.read_text())
def finite(x):return isinstance(x,(int,float)) and math.isfinite(x) and x>=0
def report():
 must(sha(BASE/"Qv20_PREEXEC_PUBLIC.json")==PRE_SHA,"preexecution lock")
 policy=BASE/"policy/calibration_policy_private.json"
 must(policy.is_file(),"policy calibrated and frozen")
 p=data(policy)
 must(p["calibration_input_SHA256"]=="6bedf97854f140939b248001a491b6f7ae38a078b502a8af9a85d3a47cc99f28","only original train32 calibration")
 must(p["source_mask_sha256"]=="9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922" and
 p["random_mask_sha256"]=="5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde","equal byte masks")
 loss={k:[] for k in ("static_source","static_random","dynamic")}
 correct={k:[] for k in loss}
 bydomain={}
 usage={"static_source":0,"static_random":0}
 cpu_stats={}
 for split in SPLITS:
  must(sha(BASE/split/"frozen_inputs_private.json")==INPUT_SHA[split],"source frozen split "+split)
  original={name:data(BASE/"native_original_C_static"/split/name/"PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json") for name in ARMS}
  dyn=data(BASE/"native_original_C_dynamic"/split/"PRIVATE_PER_CASE.json")
  must(dyn["status"]=="ACTUAL_ORIGINAL_QWEN28_APPLE_C_TWO_MASK_SOURCE_ROUTER_PASS" and
       dyn["frozen_calibration_only_policy_SHA256"]==sha(policy),"actual source router verified "+split)
  n=COUNT_PER_SPLIT
  for arm,d in original.items():
   must(d["n"]==n and d["source_frozen_private_SHA256"]==INPUT_SHA[split],
      "matched original C source data "+split+arm)
   must(d["original_Qwen28_C_executable_sha256"]=="f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221","reference original Qwen C identity")
   must(len(d["original_full_vocab_gold_nexttoken_NLL"])==n and len(d["original_choice_correct_each"])==n,
      "C static source rows")
   must(sha(BASE/"native_original_C_static"/split/arm/"native_original_C_Qwen28.log")==d["real_native_original_C_log_SHA256"],"original private log sha")
  must(len(dyn["original_gold_full_vocab_NLL"])==len(dyn["correct_each"])==
   len(dyn["selected_profile_per_case"])==n,"Qv20 dynamic C 24")
  must(sha(BASE/"native_original_C_dynamic"/split/"actual_Qwen28_Qv20_C.log")==dyn["native_C_log_SHA256"],"actual Qv20 native C log bound")
  must(dyn["cache_miss"]==dyn["cache_drops"]==0 and
       dyn["real_request_epochs"]==n and dyn["max_concurrent_forward"]==1 and not dyn["poisoned"],"Qv19 model lease safety")
  indices=[]
  for i in range(n):
   chosen="static_source" if dyn["selected_profile_per_case"][i]==0 else "static_random"
   target=original[chosen]
   must(dyn["original_gold_full_vocab_NLL"][i]==target["original_full_vocab_gold_nexttoken_NLL"][i],
        "dynamic fullvocab gold loss does not match exact chosen original C native static mask")
   must(dyn["correct_each"][i]==target["original_choice_correct_each"][i],"chosen native A/B/C/D winner differs")
   must(all(finite(z["original_full_vocab_gold_nexttoken_NLL"][i]) for z in original.values()),"invalid static loss")
   must(finite(dyn["original_gold_full_vocab_NLL"][i]),"invalid dynamic loss")
   indices.append(dyn["selected_profile_per_case"][i])
   usage[chosen]+=1
   for a in ARMS:
    loss[a].append(float(original[a]["original_full_vocab_gold_nexttoken_NLL"][i]))
    correct[a].append(int(original[a]["original_choice_correct_each"][i]))
   loss["dynamic"].append(float(dyn["original_gold_full_vocab_NLL"][i]))
   correct["dynamic"].append(int(dyn["correct_each"][i]))
  bydomain[split]={
    "n":n,"Qv8_selected":indices.count(0),"random73_selected":indices.count(1),
    "means":{k:float(np.mean([loss[k][j] for j in range(len(loss[k])-n,len(loss[k]))])) for k in loss},
    "correct":{k:int(np.sum(correct[k][-n:])) for k in correct},
    "original_static_numerical_parity_all_cases":True,
    "native_dynamic_signed_alt_cache_peak_MiB":dyn["cache_peak_MiB"],
    "native_dynamic_wall_s_uncontrolled":dyn["native_wall_s"]
  }
  cpu_stats[split]={k:original[k]["actual_native_peak_RSS_MiB"] for k in ARMS}
 mean={k:float(np.mean(loss[k])) for k in loss}
 acc={k:int(sum(correct[k])) for k in correct}
 comparisons={}
 pvalues={}
 for comp in ARMS:
  delta=np.asarray(loss[comp])-np.asarray(loss["dynamic"])
  splitdelta=(delta[:24],delta[24:])
  rng=np.random.default_rng(RNG_SEED+(0 if comp=="static_source" else 1))
  bs0=splitdelta[0][rng.integers(0,24,size=(BOOT_N,24))].mean(axis=1)
  bs1=splitdelta[1][rng.integers(0,24,size=(BOOT_N,24))].mean(axis=1)
  ci=np.quantile((bs0+bs1)/2,[.025,.975])
  # Sign swap randomization p for positive mean gap. Exploratory, not PREREG
  signs=rng.choice([-1.0,1.0],size=(BOOT_N,len(delta)))
  test=np.mean(signs*delta,axis=1)
  perm_p=(1+int(np.sum(test>=float(np.mean(delta)))))/(BOOT_N+1)
  pvalues[comp]=float(perm_p)
  a=np.asarray(correct["dynamic"]);b=np.asarray(correct[comp])
  win=int(np.sum((a==1)&(b==0)));lose=int(np.sum((a==0)&(b==1)))
  mc=float(binomtest(min(win,lose),win+lose,.5).pvalue) if win+lose else 1.0
  comparisons[comp]={
  "paired_mean_NLL_advantage_comparator_minus_dynamic":float(np.mean(delta)),
  "paired_prompt_stratified_bootstrap95_CI":ci.tolist(),
  "signed_randomization_one_sided_exploratory_p":float(perm_p),
  "native_prompt_rows_with_dynamic_lower_gold_NLL":int(np.sum(delta>0)),
  "native_prompt_rows_with_dynamic_higher_gold_NLL":int(np.sum(delta<0)),
  "paired_median_NLL_advantage":float(np.median(delta)),
  "dynamic_only_correct":win,"static_only_correct":lose,
  "exact_McNemar_two_sided_p":mc}
 # 2-hypothesis Holm adjustment for one-sided exploratory sign randomization
 ranked=sorted(pvalues,key=pvalues.get);prev=0.0
 for i,k in enumerate(ranked):
  prev=max(prev,(len(ranked)-i)*pvalues[k])
  comparisons[k]["Holm_adjusted_exploratory_one_sided_p"]=min(1.0,prev)
 status="EXPLORATORY_QUALITY_SUCCESS_IF_FULLY_PREDECLARED_AND_ALL_GATES_MET"
 passquality=all(comparisons[k]["paired_prompt_stratified_bootstrap95_CI"][0]>0 and
  comparisons[k]["Holm_adjusted_exploratory_one_sided_p"]<.05 for k in ARMS)
 result={"schema":"BEGLIN-Qv20-independent-Qwen28-real-CPU-prompt-only-two-mask-heldout-stratified/1",
 "status":"HELDOUT_48_NATIVE_REAL_C_NUMERICAL_PARITY_PASS__QUALITY_GAIN_"+("EXPLORATORY_SUPPORTED" if passquality else "NOT_ESTABLISHED"),
 "n_train":32,"n_heldout":48,
 "domains":bydomain,
 "policy_dynamic_admitted_on_calibration_only":p["dynamic_policy_admitted_from_calibration"],
 "actual_Qv8_vs_random73_profile_usage":{"Qv8":usage["static_source"],"random73":usage["static_random"]},
 "same_role_and_projection_packed_weight_bytes":765493248,
 "candidate_original_C_runtime_full_vocab_gold_loss_mean":mean,
 "candidate_real_C_choice_correct_of48":acc,
 "paired_comparison":comparisons,
 "original_static_source_and_random_Qwen28_C_exact_parity_for_actual_selected_mask":48,
 "original_calibration_POLICY_sha256":sha(policy),
 "original_Qv20_preexecution_source_SHA256":PRE_SHA,
 "source_calibration_TRAIN_only_and_two_validation_splits_completely_disjoint":True,
 "new_eval_label_free_at_policy_fit":True,
 "prompt_only_ridge_not_tensorwise_general_allocation":True,
 "no_arbitrary_N8_mask_mixture_claim":True,
 "error_bars_and_p_values_exploratory_given_48_samples":True,
 "true_model_packed_projection_bytes_equal_but_total_RAM_NOT_equal_due_to_dynamic_inactive_cache":True,
 "native_RSS_static_reference_MiB_by_domain":cpu_stats,
 "four_choice_conditional_gold_token_NLL_not_measured":True,
 "no_Metal_GPU_or_new_model_family_evidence":True,
 "no_Qwen_checkpoint_kernel_or_operator_main_change":True}
 p=BASE/"Qv20_HELDOUT_INDEPENDENT_PUBLIC.json"
 if p.exists():raise FileExistsError("never overwrite independent 48 audit")
 b=(json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
 with p.open("xb") as f:f.write(b)
 print("QV20_FULL_NATIVE_48_HELDOUT_INDEPENDENT",json.dumps({"status":result["status"],
 "accuracy":acc,"mean_NLL":{k:round(v,6) for k,v in mean.items()},
 "router_counts":result["actual_Qv8_vs_random73_profile_usage"],
 "NLL_gain_claim_supported":passquality,
 "receipt_SHA256":hashlib.sha256(b).hexdigest()},sort_keys=True),flush=True)
if __name__=="__main__":report()
